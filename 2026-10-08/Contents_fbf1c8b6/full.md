# B<sub>eyon</sub>d N<sub>om</sub>i<sub>na</sub>l E<sub>qu</sub>ilib<sub>r</sub>i<sub>a:</sub> Ri<sub>s</sub>k<sub>-</sub>A<sub>verse</sub> M<sub>u</sub>lti<sub>-</sub>P<sub>opu</sub>l<sub>a</sub>ti<sub>on</sub> M<sub>ea</sub>n<sub>-</sub>Fi<sub>e</sub>ld G<sub>a</sub>m<sub>es</sub>

Bhavini Jeloka , Siddhartha Ganguly , and Panagiotis Tsiotras

(BJ and SG contributed equally)

Abstract. Recent advances in mean-field games and its multi-population variants enable large-scale heterogeneous multi-agent systems to be modeled through representative agents and their associated mean-field distributions. However, existing approaches do not explicitly account for uncertainty in the behavior of other populations. To this end, we introduce a new paradigm: risk-averse multi-population mean-field games, where each population optimizes a worst-case expected reward over dynamically feasible ambiguity sets of mean-field flows of a subset of the other populations. Employing an occupation-measure formulation along with tools from set-valued analysis, we establish, under mild assumptions, several theoretical properties of the multi-population game, including the geometric properties of the ambiguity sets and the existence of a novel risk-averse multi-population mean-field equilibrium. Further, we derive contractivity results of the fixed-point operator under entropy regularization and show that it can be utilized to learn the equilibrium. Finally, we propose a risk-averse fictitious-play scheme and show that exploitability decays to zero, despite the additional nonlinearity introduced by the worst-case objective. We report several numerical experiments to illustrate convergence and risk-averse behavior.

## Contents

1. Introduction 2   
2. Preliminaries: MP-MFGs 3   
3. Risk-Averse MP-MFGs 6   
4. Entropy Regularization 8   
5. Fictitious Play 10   
6. Numerical Experiments 11   
7. Conclusion 16   
References 16   
Appendix 20   
Appendix A. Technical TL;DR and a roadmap 20   
Appendix B. Reachable set 20   
Appendix C. Ambiguity set 22   
Appendix D. Proof of Theorem 1 23   
Appendix E. Proof of Theorem 2 28   
Appendix F. Proofs of Theorems 3 and 4 32   
Appendix G. Fictitious Play and Proof of Theorem 5 44   
Appendix H. Numerical Experiments: Additional Details 47

## 1<sub>.</sub> Intr<sub>o</sub>d<sub>uc</sub>ti<sub>o</sub>n

Mean-field theory provides a principled framework to address the curse of dimensionality in multi-agent decision making by approximating interactions in the infinitepopulation limit [HMC06, LL07]. This perspective has led to mean-field games (MFGs) for non-cooperative settings [HMC06, CK21], mean-field control (MFC) for cooperative settings [SBR23, SC19], multi-population mean-field games (MP-MFGs) [BHL18, PPE<sup>+</sup>21] and mean-field team games (MFTGs) [GAT24, JGT25], with the latter two capturing heterogeneous and mixed cooperative–competitive interactions, respectively.

The presence of multiple sub-populations with potentially conflicting objectives imposes additional complexity in MP-MFG and MFTG problems. Despite the recent progress in computing equilibria for MP-MFGs and MFTGs [PPE<sup>+</sup>21, GAT24], robustness in these settings remains relatively underexplored. In practice, populations face uncertainty arising from environmental noise, unforeseen events, and ambiguity about the behavior and distributions of other populations. Such uncertainty motivates risk-averse decision-making, in which agents trade of expected performance against exposure to unfavorable outcomes. To address this gap, we introduce a framework for risk-averse multi-population mean-field games (RA-MP-MFGs) and study the associated solution concepts along with their theoretical guarantees.

A motivating example. The problem of stock trading is inherently subject to the curse of multi-agency, as aggregate market trends and price discovery emerge from the collective decisions of a vast population of interacting agents. To motivate the discussion, consider a stylized market with algorithmic (AI) traders and human investors trading a single asset. While both populations seek to maximize expected returns, human behavior is afected by cognitive biases such as trend-following, herding, and panic selling, as well as risk aversion toward stochastic price fluctuations. Since AI traders cannot directly observe or predict the realized human population distribution, which is generally infeasible to observe or compute, they must account for deviations from its nominal trajectory, motivating strategic risk aversion to uncertainty in human behavior.

Our central idea is to replace each risk-averse population’s nominal mean-field trajectory with a dynamically feasible ambiguity set of plausible trajectories and optimize the worst-case expected reward over this set. This yields a coupled robust optimization problem in which each population accounts for uncertainty in the behavior of the others, leading to a new risk-averse equilibrium concept.

Contributions. Our primary advancements are the following:

(Cont-a) We introduce a framework for RA-MP-MFGs and study the associated solution concepts and their theoretical guarantees. Specifically, using tools from set-valued theory and variational analysis, we establish the existence of a risk-averse multi-population mean-field equilibrium through an occupationmeasure formulation and fixed-point arguments;

(Cont-b) we show that entropy-regularized RA-MP-MFGs approximate the original equilibrium under suitable conditions via a novel occupation-measuredependent norm; and

(Cont-c) we establish convergence of fictitious play under mild assumptions on the reward function, despite the nonlinear worst-case objective. Numerical experiments, including a real-world, data-driven stock-trading example, demonstrate algorithmic convergence and characterize the resulting riskaverse policies under distributional uncertainty.

We keep our technical results at an informal level in the main body of the article; detailed technical statements along with their proofs can be found in Appendix $_ { \mathrm { B - 5 . } }$ 1 Moreover, additional numerical details are in Appendix H.

Related Work. Multi-population MFGs extend the standard MFG framework to heterogeneous populations with distinct dynamics, rewards, and policies, and have been studied under both decentralized open-loop policies $[ \mathbf { P } \mathbf { P } \mathbf { E } ^ { + } 2 \mathbf { 1 } , \mathbf { L } \mathbf { P } \mathbf { G } ^ { + } 2 2 ]$ and feedback information structures [SSAL24, SKM23, SPTH22]. Existing formulations primarily optimize expected rewards and generally do not explicitly account for uncertainty or adversarial deviations in the population trajectories generated by other populations. Zero-sum multi-team formulations [GAT24, JGT25] capture strategic interactions but rely on additional structural assumptions, such as localstate-independent rewards or mean-field-dependent policies. In contrast, we consider decentralized policies that remain robust to deviations of other populations from their nominal trajectories. This perspective is natural in decentralized settings, where each population has no control over the dynamics or the actions of other populations.

Risk-sensitive and robust MFGs have also been studied in continuous- and discrete-time settings [MB14, TZB13, Tem15, PPL<sup>+</sup>20, SBMR20, CJ26, BLP21, YWSM25], but existing approaches typically model environmental or model uncertainty [LZZZ26], uncertainty over initial distributions [JGT26], entropic uncertainty [DL26] or common noise afecting the population collectively [CDL16, CSS25]. Our framework, instead, focuses on the multi-population regime, where we analyze strategic uncertainty arising from deviations in the mean-field trajectories of other populations while retaining the underlying population dynamics.

## 2<sub>.</sub> P<sub>re</sub>li<sub>m</sub>i<sub>nar</sub>i<sub>es:</sub> MP<sub>-</sub>MFG<sub>s</sub>

Notation. Let ℕ be the set of of natural numbers and for any $\boldsymbol { d } \in \mathbb { N } , \mathbb { R } ^ { d }$ is the �-dimensional Euclidean space which is equipped with the standard norm $\mathbb { R } ^ { d } \ni x \mapsto$ $\| x \|$ and inner product $\langle \cdot , \cdot \rangle$ . For every fixed $N \in \mathbb { N }$ , we write $[ N ] : = \{ 1 , \dots , N \}$ . For a set � and an element �, we denote the indicator of � by $\mathbb { 1 } _ { A } ( e )$ , where

$$
\mathbb { 1 } _ { A } ( e ) : = { \left\{ \begin{array} { l l } { 1 , } & { e \in A , } \\ { 0 , } & { e \notin A . } \end{array} \right. }
$$

In particular, $ \mathbb { 1 } _ { \{ a \} } ( b ) = 1 { \mathrm { i f } } a = b$ , and $\mathbb { 1 } _ { \{ a \} } ( b ) = 0$ otherwise. Let $E$ be a finite and nonempty set. We denote the set of probability measures on � by

$$
{ \mathcal { P } } ( E ) : = { \Big \{ } \mu : E \to [ 0 , 1 ] \Big | \sum _ { e \in E } \mu ( e ) = 1 { \Big \} } .
$$

Since � is finite, we identify every probability measure $\mu \in { \mathcal { P } } ( E )$ with its probabilitymass vector $( \mu ( e ) ) _ { e \in E }$ . For $v \in \mathbb { R } ^ { E }$ , we denote its ℓ<sup>1</sup>-norm by $\begin{array} { r } { \| v \| _ { 1 } : = \sum _ { e \in E } | v ( e ) | } \end{array}$ For two measures $\mu$ and � on the same measurable space, we write $\mu \ll \nu$ if $\mu$ is absolutely continuous with respect to �, i.e., if $\nu ( A ) = 0$ implies $\mu ( A ) = 0$ for all Borel subsets �; see [Øks03, Fol99] for more details.

Setting and MP-MFGs. We briefly review MP-MFGs and subsequently establish our risk-averse framework for heterogeneous large population multi-team games. We consider a finite-horizon, discrete-time, general-sum stochastic game involving $M \in \mathbb { N }$ large populations, with each population representing a distinct group of agents, thereby capturing macroscopic heterogeneity. We have the following problem data:

(D-a) We use $j \in [ M ]$ to index the populations. The state and action spaces of the game are finite, nonempty sets X and U, respectively, both equipped with their discrete �-algebras.

(D-b) For each population $j \in [ M ]$ , the state of the representative agent at time index $t ~ = ~ 0 , 1 , \ldots , T$ , will be denoted by $x _ { t } ^ { j }$ , with initial state distributed according to $X _ { 0 } ^ { j } \sim \mu _ { 0 } ^ { j } \in \mathcal { P } ( \mathsf { X } )$ . Henceforth, we use the term “agent” to mean a representative agent.

(D-c) At each time $t \in \{ 0 , \ldots , T - 1 \}$ , the agent observes its current state $X _ { t } ^ { j } = x _ { t } ^ { j } \in \mathsf { X }$ and selects an action $U _ { t } ^ { j } = u _ { t } ^ { j } \in \mathsf { U }$ according to a policy $\pi _ { t } ^ { j } : \mathsf { X } \times \mathsf { U } \to [ 0 , 1 ]$ Here, $\pi _ { t } ^ { j } ( \cdot | x ) \in \mathcal { P } ( \cup )$ for every $x \in \mathsf { X } .$ . We let $\Pi _ { t }$ denote set of randomized Markov decision rules at time �; to wit

$$
\Pi _ { t } : = { \Bigl \{ } \pi _ { t } : { \mathsf { X } } \times { \mathsf { U } } \to [ 0 , 1 ] \Big |  \sum _ { u \in { \mathsf { U } } } \pi _ { t } ( u | x ) = 1 { \mathrm { ~ f o r ~ e v e r y ~ } } x \in { \mathsf { X } } { \Bigr \} } .
$$

An element of this set assigns an action distribution $\pi _ { t } ( \cdot | x ) \in { \mathcal { P } } ( \cup )$ to each state $x \in \mathsf { X } .$ . Thus $\Pi _ { t }$ can be identified with $\Pi _ { x \in { \mathsf { X } } } { \mathcal { P } } ( \cup )$

(D-d) Let $\begin{array} { r } { \Pi ^ { j } : = \Pi _ { t = 0 } ^ { T - 1 } \Pi _ { t } } \end{array}$ be the set of policies for agent $j .$ , and let $\begin{array} { r } { \Pi : = \Pi _ { j = 1 } ^ { M } \Pi ^ { j } } \end{array}$ be the set of policies of all representative agents. Using the action $u _ { t } ^ { j }$ selected according to the policy $\pi _ { t } ^ { j }$ , agent � then transitions to $X _ { t + 1 } ^ { j }$ according to the controlled transition kernel $f _ { t } ^ { j } ( \cdot \ | \ x _ { t } ^ { j } , u _ { t } ^ { j } ) \ \in \ \mathcal { P } ( \mathsf { X } )$ . As in many classical MFG formulations [LL07, PPE<sup>+</sup>21], we assume throughout that the transition kernels are independent of the multi-population distribution.

Mean-field. Let $\mu _ { t } ^ { j } \in \mathcal { P } ( \mathsf { X } )$ denote the distribution of population $j$ at time � and let $\mu _ { t } : = ( \mu _ { 1 } ^ { 1 } , \ldots , \dot { \mu } _ { t } ^ { N } ) \in \mathcal { P } ( \mathsf { X } ) ^ { M }$ denote the collection of the distributions of all populations at time �. We represent the mean-field flow of population $j$ from $t = \{ 0 , \ldots , T \}$ by $\mu ^ { j } : = ( \mu _ { t } ^ { j } ) _ { t = 0 } ^ { T } \in \mathsf { M } ^ { j } \subseteq \mathcal { P } ( \mathsf { X } ) ^ { T + 1 }$ where $\mathsf { M } ^ { j }$ is the set of all meanfield flows of population $j .$ Further, $\mu : = ( \mu ^ { 1 } , \dots , \mu ^ { M } )$ denotes the collection of the mean-field flows of all populations. Let $\begin{array} { r } { \mathsf { M } : = \prod _ { j = 1 } ^ { M } \mathsf { M } ^ { j } } \end{array}$ denote the set of all such flows. For a multi-population distribution $\mu _ { t }$ , the collection of distributions of all populations other than � is $\mu _ { t } ^ { - j } : = \big ( \mu _ { t } ^ { 1 } , \dots , \mu _ { t } ^ { j - 1 } , \mu _ { t } ^ { j + 1 } , \dots , \mu _ { t } ^ { M } \big )$ . Hence, $\mu _ { t } = ( \mu _ { t } ^ { j } , \mu _ { t } ^ { - j } )$ .

Since all agents in population � follow the same policy $\pi ^ { j } \ ( \mathrm { i . e . }$ ., the policy of the representative agent), given a fixed initial distribution $\mu _ { 0 } ^ { j } \in \mathcal { P } ( \mathsf { X } )$ , the evolution of the distribution of the �-th population is deterministic and can be computed recursively via

$$
\mu _ { t + 1 } ^ { j } ( x ^ { \prime } ) = \sum _ { x \in \mathsf { X } } \sum _ { u \in \mathsf { U } } f _ { t } ^ { j } ( x ^ { \prime } \mid x , u ) \pi _ { t } ^ { j } ( u \mid x ) \mu _ { t } ^ { j } ( x )\tag{1}
$$

for all $x \in \mathsf X$ and $t \in \{ 0 , \ldots , T - 1 \}$ . We sometime use by $\mu ^ { + }$ instead of $\mu _ { t + 1 } ^ { j } ( x ^ { \prime } )$ to denote the next state. The mean-field dynamics (1) can also be expressed in a compact matrix form as $\mu _ { t + 1 } ^ { j } = \mu _ { t } ^ { j } F _ { t } ^ { j } ( \pi _ { t } ^ { j } )$ .

For each $j ~ \in ~ [ M ]$ , let $\mathcal { B } _ { \mathrm { p r o p } } ^ { j } : \Pi ^ { j } \ \to \ \mathsf { M } ^ { j }$ denote the propagation operator associated with (1) and the fixed initial distribution $\mu _ { 0 } ^ { j } .$ . Accordingly, $\mathcal { B } _ { \mathrm { p r o p } } : \Pi  \mathsf { M }$ is defined by $\mathcal { B } _ { \mathrm { p r o p } } ( \pi ) : = \bigl ( \mathcal { B } _ { \mathrm { p r o p } } ^ { 1 } ( \pi ^ { 1 } ) , \dots , \mathcal { B } _ { \mathrm { p r o p } } ^ { M } ( \pi ^ { \check { M } } ) \bigr )$ where $\pi : = ( \pi ^ { 1 } , \dots , \pi ^ { M } )$ is the joint policy profile, and propagates all distributions. For a fixed mean-field distribution $\mu _ { t } ^ { j }$ , varying the local policy $\pi _ { t } ^ { j }$ induces a set of distributions that can be reached at the next time step. This motivates the following definition.

Definition 1 (Reachable set). $F i x \ j \in [ M ] , t \in \{ 0 , . . . , T - 1 \}$ , and $\mu _ { t } ^ { j } \in \mathcal { P } ( \mathsf { X } )$ . The one-step reachable set of population � from $\mu _ { t } ^ { j }$ is defined by $\mathcal { R } _ { t } ^ { j } ( \mu _ { t } ^ { j } ) : =$ $\{ \mu ^ { + } \in \mathcal { P } ( \mathsf { X } ) \mid \exists \pi _ { t } ^ { j } \in \Pi _ { t }$ such that $\mu ^ { + } = \mu _ { t } ^ { j } F _ { t } ^ { j } ( \pi _ { t } ^ { j } ) \}$

By definition, $\mathcal { R } _ { t } ^ { j } \left( \mu _ { t } ^ { j } \right)$ is a polytope contained in $\mathcal { P } ( \mathsf { X } )$ ; see Appendix B for a proof. For a multi-population distribution $\mu _ { t } : = ( \mu _ { t } ^ { 1 } , \ldots , \mu _ { t } ^ { M } ) \in \mathcal { P } ( \mathsf { X } ) ^ { M }$ , we denote the vector of distributions of all populations other than � by

$$
\boldsymbol { \mu } _ { t } ^ { - j } : = ( \mu _ { t } ^ { 1 } , \dots , \mu _ { t } ^ { j - 1 } , \mu _ { t } ^ { j + 1 } , \dots , \mu _ { t } ^ { M } ) .
$$

For each $j \in [ M ]$ and $t \in \{ 0 , . . . , T - 1 \}$ , we further define

$$
\mathcal { R } _ { t } \big ( \mu _ { t } ^ { - j } \big ) : = \prod _ { k \neq j } \mathcal { R } _ { t } ^ { k } \big ( \mu _ { t } ^ { k } \big ) ,
$$

which is the joint one-step reachable set of all populations other than $j .$

Reward. The reward received by the representative agent of population $j$ at time � is given by $r _ { t } ^ { j } : \mathsf { X } \times \mathcal { P } ( \mathsf { X } ) ^ { M } \to \mathbb { R }$ . The reward depends on the distribution flows of all populations, and hence this is the main mechanism by which the decisions (and hence the resulting flows) of the populations are coupled.

Assumption 1. For each $j \in [ M ]$ , there exists $L _ { r } ^ { j } > 0$ such that $| r _ { t } ^ { j } ( x , \nu ) - r _ { t } ^ { j } ( x , \nu ^ { \prime } ) | \leq$ $\boldsymbol { L } _ { r } ^ { j } \mathrm { d } _ { \mathrm { T V } } \left( \nu , \nu ^ { \prime } \right)$ for all $t \in \{ 0 , \ldots , T \} , x \in \mathsf { X } ,$ , and $\nu , \nu ^ { \prime } \in \mathcal { P } ( \mathsf { X } ) ^ { M }$

Lipschitz continuity of the rewards is standard in the MFG literature [HMC06, GGWX21, CHFK24].

Occupation measures provide a natural representation of policies and meanfield flows, with a linear structure that facilitates the characterization of dynamically feasible ambiguity sets and the analysis of the game [YYP26, HZ25].

Occupation measure. For each $j \in [ M ]$ , let ${ \sf O } ^ { j } \subset { \sf M } ^ { j } \times \left( \mathbb { R } _ { + } ^ { \sf X \times U } \right) ^ { T }$ denote the set of occupation measures $o ^ { j } : = ( \xi ^ { j } , \eta ^ { j } )$ , where $\xi ^ { j } : = \{ \xi _ { t } ^ { j } \} _ { t = 0 } ^ { T }$ and $\eta ^ { j } : = \{ \eta _ { t } ^ { j } \} _ { t = 0 } ^ { T - 1 }$ satisfying

(2)

$$
\xi _ { 0 } ^ { j } = \mu _ { 0 } ^ { j } , \quad \sum _ { u \in \mathsf { U } } \eta _ { t } ^ { j } ( x , u ) = \xi _ { t } ^ { j } ( x )\tag{3}
$$

$$
\xi _ { t + 1 } ^ { j } ( y ) = \sum _ { x \in \mathsf { X } } \sum _ { u \in \mathsf { U } } \eta _ { t } ^ { j } ( x , u ) f _ { t } ^ { j } ( y | x , u )
$$

for all $x , y \in \mathsf { X }$ and $t = 0 , \ldots , T - 1$ . Here, $\boldsymbol { \xi } _ { t } ^ { j }$ and $\eta _ { t } ^ { j }$ represent, respectively, the state distribution and the joint state–action distribution of the representative agent of population � at time �. Under a Markov policy $\pi ^ { j }$ , we have $\xi _ { t } ^ { j } ( x ) = \mathbb { P } ^ { \pi ^ { j } } ( X _ { t } ^ { j } = x )$ for $t ~ = ~ 0 , \ldots , T$ , and $\eta _ { t } ^ { j } ( x , u ) \ = \ \mathbb { P } ^ { \pi ^ { j } } ( X _ { t } ^ { j } \ = \ x , U _ { t } ^ { j } \ = \ u ) \ = \ \xi _ { t } ^ { j } ( x ) \pi _ { t } ^ { j } ( u \ | \ x )$ for $t ~ = ~ 0 , \ldots , T - 1$ , where $\mathbb { P } ^ { \mu ^ { j } }$ denotes the probability law induced by the initial distribution $\mu _ { 0 } ^ { j } .$ , the transition kernels $f _ { t } ^ { j }$ , and the policy $\pi ^ { j }$ . We denote the joint occupation-measure space by $\begin{array} { r } { \mathsf { O } : = \prod _ { j = 1 } ^ { M } \mathsf { O } ^ { j } } \end{array}$ . For $o : = ( o ^ { 1 } , \dots , o ^ { M } ) \in { \sf O }$ , we denote the corresponding multi-population state-flow by $\xi ( o ) : = ( \xi ^ { 1 } , \dots , \xi ^ { M } )$

Objective. Given a multi-population flow $\mu \in { \sf M }$ , the cumulative performance of the representative agent of population $j$ under policy $\pi ^ { j }$ is

$$
\mathcal { J } _ { j } \left( \pi ^ { j } , \mu \right) = \mathsf { E } ^ { \pi ^ { j } } \left[ \sum _ { t = 0 } ^ { T } r _ { t } ^ { j } ( x _ { t } ^ { j } , \mu _ { t } ) \right] .\tag{4}
$$

We use the correspondence $\mathcal { B } _ { \mathrm { o p t } } : \mathsf { M }  \Pi$ to denote the multi-population bestresponse correspondence, defined by<sup>2</sup>

$$
\mathcal { B } _ { \mathrm { o p t } } ( \mu ) : = \prod _ { j = 1 } ^ { M } \operatorname { a r g m a x } J ^ { j } ( \pi ^ { j } , \mu ) .
$$

Using the operators ${ \mathcal { B } } _ { \mathrm { p r o p } }$ and ${ \mathcal { B } } _ { \mathrm { o p t } }$ , we define the multi-population mean-field equilibrium (MFE).

Definition 2 (MP-MFE [HMC06]). The MP-MFE is a consistentpair $( \pi ^ { \star } , \mu ^ { \star } ) \in$ $\Pi \times \mathsf { M }$ such that

$$
\begin{array} { r } { \pi ^ { \star } \in \mathcal { B } _ { \mathrm { o p t } } ( \mu ^ { \star } ) \quad a n d \quad \mu ^ { \star } = \mathcal { B } _ { \mathrm { p r o p } } ( \pi ^ { \star } ) . } \end{array}
$$

Under Assumption 1, the existence of such consistent pair can be established through Kakutani’s fixed point argument [CK21]. Lastly, we define exploitability in order to characterize a policy relative to the optimal, reward maximizing policy.

Definition 3 (Exploitability [PPE<sup>+</sup>21]). For a policy profile $\pi \in \Pi ,$ let $\mu ^ { \pi } : =$ ${ \mathcal { B } } _ { \mathrm { p r o p } } ( \pi )$ denote the induced multi-population mean-field flow. The exploitability $o f$ � measures the distance to an equilibrium and is defined a

$$
\phi ( \pi ) : = \sum _ { j = 1 } ^ { M } \phi ^ { j } ( \pi ) , \quad \phi ^ { j } ( \pi ) : = \operatorname* { m a x } _ { \pi ^ { \prime } { } ^ { j } \in \Pi ^ { j } } \mathcal { J } _ { j } ( \pi ^ { \prime j } , \mu ^ { \pi } ) - \mathcal { J } _ { j } ( \pi ^ { j } , \mu ^ { \pi } )
$$

for each $j \in [ M ]$ . A policy profile $\pi \in \Pi$ is a MP-MFE if and only if $\phi ( \pi ) = 0 .$

Although $\mathcal { J } ^ { j } ( \pi ^ { j } , \mu )$ depends on all population flows, population/agent � controls only its own policy and therefore it must rely on nominal beliefs about the other team population flows. Deviations from these beliefs create strategic uncertainty, motivating policies that are robust to unfavorable mean-field trajectories of the other populations. This motivates the incorporation of risk aversion with respect to such uncertainties in population distributions and adapting the current objective to increase robustness.

## 3<sub>.</sub> Ri<sub>s</sub>k<sub>-</sub>A<sub>verse</sub> MP<sub>-</sub>MFG<sub>s</sub>

In this section, we formulate the Risk-Averse Multi-Population Mean-Field Game (RA-MP-MFG) that adapts the standard MP-MFG objective [PPE<sup>+</sup>21] to account for uncertainties in population distributions. In particular, we analyze the worst-case return that population $j$ can obtain under distributional uncertainty with respect to all other population types. To this end, we formulate constraints imposed by other populations. Fix $\varepsilon \in [ 0 , 1 ]$ , and define the dynamically feasible ambiguity set $\mathcal { W } _ { j } ( \mu , \varepsilon )$ by

$$
\begin{array} { r } {  { \mathcal W } _ { j } ( \mu , \varepsilon ) : = \left\{ \{ \hat { \mu } _ { t } ^ { k } \} _ { t = 0 } ^ { T } , \forall k \in [ M ] , k \neq j \left| \begin{array} { l } { \hat { \mu } _ { 0 } ^ { k } = \mu _ { 0 } ^ { k } , \hat { \mu } _ { t + 1 } ^ { k } \in  { \mathcal R } _ { t } ^ { k } ( \hat { \mu } _ { t } ^ { k } ) , } \\ {  { \operatorname { d } } _ { \mathrm { T V } } \big ( \mu _ { t } ^ { k } , \hat { \mu } _ { t } ^ { k } \big ) \le \varepsilon } \end{array} \right. \right\} . } \end{array}\tag{5}
$$

which contains all reachable flows that are not “far away” from their nominal values and $\hat { \mu } _ { t + 1 } ^ { k } \in \mathcal { R } _ { t } ^ { k } ( \hat { \mu } _ { t } ^ { k } ) , k \neq j$ ensures that the trajectories $\{ \hat { \mu } _ { t } ^ { k } \} _ { t = 0 } ^ { T }$ are dynamically feasible.

Remark 1. The set $\mathcal { W } _ { j } ( \mu , \varepsilon )$ in (5) assumes that population � is risk-averse to every population $k \neq j .$ This can be generalized to the setting where � is riskaverse only to a subset $\mathcal { K } _ { j } \subseteq [ M ] \backslash j$ of populations, by defining the ambiguity set only over the corresponding mean-field trajectories. Under Assumption 1, the existence and entropy-approximation results extend directly to this generalized setting, while the contraction andfictitious-play guarantees can be extended by strengthening the contraction condition to account for the Lipschitz dependence on the remaining nominal populations. The multi-population chasing example in Section 6 ahead empirically demonstrates this extension.

Proposition 1 (Informal). Fix $j \in [ M ] , \mu \in \mathsf { M } ,$ and let $\varepsilon \in [ 0 , 1 ]$ Then, $\mathcal { W } _ { j } ( { \boldsymbol { \mu } } , { \boldsymbol { \varepsilon } } )$ is a nonempty compact polytope.

Remark 2 (Policy–occupation-measure equivalence). The occupation-measure representation in $( 2 ) { - } ( 3 )$ is equivalent to the Markov-policy representation [GAXJ24]. Indeed, let $\pi ^ { j } \in \Pi ^ { j }$ , let $\xi ^ { j } = \mathcal { B } _ { \mathrm { p r o p } } ^ { j } ( \pi ^ { j } )$ be the induced state-flow, and define

$$
\begin{array} { r } { \eta _ { t } ^ { j } ( x , u ) : = \xi _ { t } ^ { j } ( x ) \pi _ { t } ^ { j } ( u | x ) \quad f o r x \in \mathsf X , u \in \mathsf U , t = 0 , \ldots , T - 1 . } \end{array}\tag{6}
$$

Then, $o ^ { j } = ( \xi ^ { j } , \eta ^ { j } ) \in { \sf O } ^ { j }$ . Conversely, every $o ^ { j } = ( \xi ^ { j } , \eta ^ { j } ) \in { \sf O } ^ { j }$ induces a Markov policy by

$$
\pi _ { t } ^ { j } ( u | x ) : = \frac { \eta _ { t } ^ { j } ( x , u ) } { \xi _ { t } ^ { j } ( x ) } \quad i f \xi _ { t } ^ { j } ( x ) > 0 ,\tag{7}
$$

and by choosing $\pi _ { t } ^ { j } ( \cdot | x )$ arbitrarily in ${ \mathcal { P } } ( \cup )$ whenever $\xi _ { t } ^ { j } ( x ) = 0 .$ . Indeed, $i f \xi _ { t } ^ { j } ( x ) =$ $0 ,$ then (2) and the nonnegativity of $\eta _ { t } ^ { j }$ imply that $\eta _ { t } ^ { j } ( x , u ) = 0 .$ for every $u \in \mathsf { U } .$ . The policy constructed in $( 7 )$ generates the state-flow $\xi ^ { j }$ and the state-action occupation flow $\eta ^ { j }$ . Consequently, $\xi ^ { j } \in \mathsf { M } ^ { j }$ for every $o ^ { j } \in \mathsf { O } ^ { j }$ , and hence $\xi ( o ) \in { \mathbb { M } }$ for every $o \in { \mathsf { O } } .$

From the point of view of population $j ,$ we are interested in evaluating the performance of a policy $\pi ^ { j }$ under the risk-averse payof defined below.

Definition 4 (Risk-averse payof). Fix $\mu \in { \sf M }$ and $\varepsilon \in [ 0 , 1 ]$ . For each $j \in [ M ]$ and $\pi ^ { j } \in \Pi ^ { j }$ , the risk-averse payof of population $j$ is defined by

$$
\bar { \mathcal { J } } _ { j } ( \pi ^ { j } ; \mu ) : = \operatorname* { i n f } _ { \widehat { \mu } ^ { - j } \in \mathcal { W } _ { j } ( \mu , \varepsilon ) } \mathsf { E } ^ { \pi ^ { j } } \left[ \sum _ { t = 0 } ^ { T } r _ { t } ^ { j } ( x _ { t } ^ { j } , \mu _ { t } ^ { j } , \widehat { \mu _ { t } } ^ { - j } ) \right] ,\tag{8}
$$

where $x _ { 0 } ^ { j } \sim \mu _ { 0 } ^ { j } , u _ { t } ^ { j } \sim \pi _ { t } ^ { j } ( \cdot | x _ { t } ^ { j } ) ,$ , and $x _ { t + 1 } ^ { j } \sim f _ { t } ^ { j } ( \cdot | x _ { t } ^ { j } , u _ { t } ^ { j } )$

We further show in Appendix D that (8) can be reformulated in terms of the occupation measure as

$$
\Phi _ { j } ( \tilde { \sigma } ^ { j } ; o ) : = \operatorname* { i n f } _ { \substack { \widehat { \mu } ^ { - j } \in \mathcal { W } _ { j } ( \xi ( o ) , \varepsilon ) } } \sum _ { t , x } \tilde { \xi } _ { t } ^ { j } ( x ) r _ { t } ^ { j } \big ( x , \xi _ { t } ^ { j } , \widehat { \mu _ { t } } ^ { - j } \big )\tag{9}
$$

where $\tilde { o } ^ { j } : = ( \tilde { \xi } ^ { j } , \tilde { \eta } ^ { j } ) \in 0 ^ { j }$

Given a nominal flow $\mu ,$ population � chooses $\pi ^ { j }$ to maximize $\bar { \mathcal { J } } _ { j } ( \pi ^ { j } ; \mu )$ against dynamically feasible adversarial deviations $\hat { \mu } _ { t } ~ = ~ ( \mu _ { t } ^ { j } , \hat { \mu } _ { t } ^ { - j } )$ for all $t \in \{ 0 , \ldots , T \}$ of the other populations’ mean-field trajectories. Define $\mathcal { B } _ { \mathrm { r i s k - o p t } } : \mathsf { M } \to \Pi$ as

the risk-averse best-response correspondence, i.e., for each $\mu \in \mathsf { M } , ( \pi ^ { 1 } , \ldots , \pi ^ { M } ) \in$ ${ \mathcal { B } } _ { \mathrm { r i s k - o p t } } ( \mu )$ , where, for every $j \in [ M ]$ ，

$$
\pi ^ { j } \in \underset { \pi ^ { j } \in \Pi ^ { j } } { \arg \operatorname* { m a x } } \bar { \mathcal { J } } _ { j } ( \pi ^ { j } ; \mu ) .
$$

Definition 5 (Risk-averse exploitibility). For any $\varepsilon \in [ 0 , 1 ]$ and � ∈ Π, let $\mu ^ { \pi } =$ ${ \mathcal { B } } _ { \mathrm { p r o p } } ( \pi )$ The risk-averse exploitability is defined as $\begin{array} { r } { \phi _ { \mathsf { R A } } ( \pi ) = \sum _ { j = 1 } ^ { M } \phi _ { \mathsf { R A } } ^ { j } ( \pi ) } \end{array}$ where $\begin{array} { r } { \phi _ { \mathsf { R A } } ^ { j } ( \pi ) = \operatorname* { s u p } _ { \tilde { \pi } ^ { j } \in \Pi ^ { j } } \bar { \mathcal { J } } _ { j } ( \tilde { \pi } ^ { j } ; \mu ^ { \pi } ) - \bar { \mathcal { J } } _ { j } ( \pi ^ { j } ; \mu ^ { \pi } ) , \forall j \in [ M ] } \end{array}$

The quantity $\phi _ { \mathsf { R A } } ^ { j } ( \pi )$ is the largest improvement population � can obtain by unilaterally replacing its current policy � with a risk-averse best response, while $\mu ^ { \pi }$ is held fixed. In particular, $\phi _ { \mathsf { R A } } ^ { j } ( \pi ) \ge 0$ , and it vanishes precisely when $\pi ^ { j }$ is a risk-averse best response to $\mu ^ { \pi }$

Definition 6 (Risk-averse equlibrium). Fix $\varepsilon \in [ 0 , 1 ]$ and let $\pi ^ { \star } \in \Pi$ and $\mu ^ { \star } \in { \mathsf { M } } .$ A pair $( \pi ^ { \star } , \mu ^ { \star } )$ $\Pi \times \mathsf { M }$ is a Risk-Averse Multi-Population Mean-Field Equilibrium (RA-MP-MFE) if the following two conditions hold: i) $\mu ^ { \star } =$ $\mathcal { B } _ { \mathrm { p r o p } } ( \pi ^ { \star } )$ for all $j \in [ M ]$ ; and ii) $\pi ^ { \star } \in \mathcal { B } _ { \mathrm { r i s k - o p t } } ( \mu ^ { \star } )$

Thus, �<sub>RA</sub>(�) = 0 if $( \pi , \mu ^ { \pi } )$ is an ${ \mathrm { R A - M P - M F E } } .$

Theorem 1 (Existence of equilibria (informal)). Fix $\mu _ { 0 } ^ { j } \in \mathcal { P } ( \mathsf { X } )$ for all $j \in$ [�]. Then, under Assumption 1, for every fixed $\varepsilon \in \mathsf { \Gamma } [ 0 , 1 ]$ there exists a RA-MP-MFE $( \pi ^ { \star } , \mu ^ { \star } ) \in \Pi \times \mathsf { M } .$

A proof of Theorem 1 is given in Appendix D and it relies on techniques from variational analysis and fixed-point theory [AB06]. However, note that the operator $\mathcal { B } _ { \mathrm { p r o p } } \circ \mathcal { B } _ { \mathrm { r i s k - o p t } } : \mathsf { M } \to \mathsf { M }$ is generally not contractive (this is usual, see [CK21]); a statement and proof is in Appendix D. This naturally brings us to the next section.

## 4<sub>.</sub> E<sub>n</sub>t<sub>ropy</sub> R<sub>egu</sub>l<sub>ar</sub>i<sub>za</sub>ti<sub>on</sub>

Non-contractivity implies that the risk-averse best-response dynamics may not support standard fixed-point iteration schemes. This motivates the introduction of an entropy-regularized variant of the risk-averse equilibrium. This concept models boundedly rational decision-making [MPS25] through an entropy reward and also provides a smoother surrogate. We first define the entropy-regularized risk-averse payof.

Definition 7 (Entropy-regularized payof). Fix $\mu \in { \mathsf { M } } .$ For each population $j \in \left[ M \right]$ , policy $\pi ^ { j } \in \Pi ^ { j } , \ \varepsilon \in [ 0 , 1 ]$ , and entropy-regularization parameter $\beta > 0 ,$ , the entropy-regularized risk-averse payof of population $j , \bar { \mathcal { F } } _ { j , \beta } ( \pi ^ { j } ; \mu )$ , is defined by

$$
\bar { \mathcal { J } } _ { j , \beta } ( \pi ^ { j } ; \mu ) : = \operatorname* { i n f } _ { \widehat { \mu } ^ { - j } \in \mathcal { M } _ { j } ( \mu , \varepsilon ) } \mathsf { E } ^ { \pi ^ { j } } \left[ \sum _ { t = 0 } ^ { T } r _ { t } ^ { j } \big ( x _ { t } ^ { j } , \mu _ { t } ^ { j } , \widehat { \mu } _ { t } ^ { - j } \big ) - \beta \sum _ { t = 0 } ^ { T - 1 } \log \pi _ { t } ^ { j } ( u _ { t } ^ { j } | x _ { t } ^ { j } ) \right] ,
$$

where $\boldsymbol { x } _ { 0 } ^ { j } \sim \mu _ { 0 } ^ { j } , \boldsymbol { u } _ { t } ^ { j } \sim \pi _ { t } ^ { j } ( \cdot | \boldsymbol { x } _ { t } ^ { j } ) , a n d \boldsymbol { x } _ { t + 1 } ^ { j } \sim f _ { t } ^ { j } ( \cdot | \boldsymbol { x } _ { t } ^ { j } , \boldsymbol { u } _ { t } ^ { j } ) .$

The rationale behind introducing entropy regularization is that it gives a tractable perturbed equilibrium notion, which we call the entropy-regularized RA-MP-MFE and define below.<sup>3</sup>

Definition 8 (Entropy-regularized RA-MP-MFE). A pair $( \pi _ { \beta } ^ { \star } , \mu _ { \beta } ^ { \star } ) \in \Pi \times \mathsf { M } ,$ with $\pi _ { \beta } ^ { \star } : = ( \pi _ { \beta } ^ { 1 , \star } , \ldots , \pi _ { \beta } ^ { M , \star } ) \in \Pi$ and $\mu _ { \beta } ^ { \star } : = ( \mu _ { \beta } ^ { 1 , \star } , \ldots , \mu _ { \beta } ^ { M , \star } ) \in \mathsf { M } ,$ is an entropy-regularized RA-MP-MFE with $\varepsilon \in \dot { [ 0 , 1 ] }$ and $\beta > 0 \ i j$ the following two conditions hold:

(8-a) The mean-field flows are consistent with the corresponding population policies, i.e.,

$$
\mu _ { \beta } ^ { j , \star } = \mathcal { B } _ { \mathrm { p r o p } } ^ { j } ( \pi _ { \beta } ^ { j , \star } ) \quad f o r \ j \in [ M ] .\tag{10}
$$

(8-b) Each population is entropy-regularized risk-averse optimal, $i . e . ,$

$$
\bar { \xi } _ { j , \beta } ( \pi _ { \beta } ^ { j , \star } ; \mu _ { \beta } ^ { \star } ) \ge \bar { \mathcal { J } } _ { j , \beta } ( \pi ^ { j } ; \mu _ { \beta } ^ { \star } ) \quad f o r a l l \pi ^ { j } \in \Pi ^ { j } , \ j \in [ M ] ,\tag{11}
$$

equivalently,

$$
\pi _ { \beta } ^ { j , \star } \in \underset { \pi ^ { j } \in \Pi ^ { j } } { \arg \operatorname* { m a x } } \bar { \mathcal { J } } _ { j , \beta } ( \pi ^ { j } ; \mu _ { \beta } ^ { \star } ) .\tag{12}
$$

Theorem 2 (Existence of Ent. Reg. equilibria (informal)). Let Assumption 1 be satisfied. Then, for every fixed $\varepsilon \in [ 0 , 1 ]$ and $\beta > 0$ , there exists an entropyregularized RA-MP-MFE $( \pi _ { \beta } ^ { \star } , \mu _ { \beta } ^ { \star } ) \in \Pi \times \mathsf { M }$

A proof of Theorem 2 is given in the Appendix E.

Similar to $\mathcal { B } _ { \mathrm { r i s k - o p t } }$ , we define $\mathcal { B } _ { \mathrm { r i s k - o p t } , \beta } : \mathsf { M } \to \Pi$ as the entropy-regularized risk-averse best-response correspondence, $\mathrm { i . e . }$ , for each $\mu \in \mathsf { M } , ( \pi ^ { 1 } , \ldots , \pi ^ { M } ) \in$ $\mathcal { B } _ { \mathrm { r i s k - o p t } , \beta } ( \mu )$ , where, for every $j \in [ M ] , \pi ^ { j } \in$ arg max $\ L _ { \pi ^ { j } \in \Pi ^ { j } } \bar { \mathcal { F } } _ { j , \beta } ( \pi ^ { j } ; \mu )$ . Further, we define the composite operator $\Gamma _ { \beta } : = \mathcal { B } _ { \mathrm { p r o p } } \circ \mathcal { B } _ { \mathrm { r i s k - o p t } , \beta } : \mathsf { M } \to \mathsf { M }$

4.1. Steps towards contractivity. We are now ready to prove several key results on the contractivity of best response mappings. We provide two separate contractivity results depending on the value of $\varepsilon .$

◦ $\varepsilon = 1 ;$ in this case, the ambiguity set is fixed. Under a particular structure on the reward, contractivity follows. This result is reported in Theorem 3 with the detailed proof given in Appendix F.

◦ $ { \varepsilon } \in ( 0 , 1 )$ ; for this case, and under additional assumptions on the reward, contractivity follows if we further regularize the inner inf problem (with respect to the adversarial opponent mean-field flows) in (8). This result is reported in Theorem 4 with a detailed proof given in Appendix F..

Assumption 2. For every $j \in [ M ]$ and $t \in \{ 0 , \ldots , T \}$ , assume that the reward admits the decomposition

$$
\begin{array} { r } { \left( x , \mu ^ { j } , \nu ^ { - j } \right) \mapsto r _ { t } ^ { j } ( x , \mu ^ { j } , \nu ^ { - j } ) : = a _ { t } ^ { j } ( x , \nu ^ { - j } ) + b _ { t } ^ { j } ( x , \mu ^ { j } ) , } \end{array}\tag{13}
$$

where su $\mathsf { p } _ { t , x , \nu ^ { - j } } \big | a _ { t } ^ { j } ( x , \nu ^ { - j } ) \big | <$ +∞. Moreover, there exists $\ell _ { j , t } \geq 0$ such that $\| b _ { t } ^ { j } ( \cdot , \mu ^ { j } ) -$ $b _ { t } ^ { j } ( \cdot , \bar { \mu } ^ { j } ) \| _ { \infty } \leq \ell _ { j , t } \| \mu ^ { j } - \bar { \mu } ^ { j } \| _ { 1 }$ for all �<sup>�</sup> , $\bar { \mu } ^ { j } \in \mathcal { P } ( \mathsf { X } )$ with $\begin{array} { r } { L _ { j } : = \sum _ { t = 1 } ^ { T } \ell _ { j , t } . } \end{array}$

We define the occupation norm under which we show contraction. Let $o ^ { j } \ =$ $( \xi ^ { j } , \eta ^ { j } ) , { \bar { o } } ^ { j } = ( { \bar { \xi } } ^ { j } , { \bar { \eta } } ^ { j } ) , { \hat { u } } _ { t } ^ { j } : = \| \xi _ { t } ^ { j } - { \bar { \xi } } _ { t } ^ { j } \| _ { 1 }$ , and $v _ { t } ^ { j } : = \| \eta _ { t } ^ { j } - \bar { \eta } _ { t } ^ { j } \| _ { 1 }$ <sub>1</sub>, then

$$
\Vert o ^ { j } - \bar { o } ^ { j } \Vert _ { \mathrm { o c c } } : = \operatorname* { m a x } \left\{ \operatorname* { m a x } _ { 0 \leq t \leq T } u _ { t } ^ { j } , \operatorname* { m a x } _ { 0 \leq t < T } v _ { t } ^ { j } \right\}\tag{14}
$$

and $\begin{array} { r } { \| o - \bar { o } \| _ { \mathrm { o c c , \infty } } : = \operatorname* { m a x } _ { j \in [ M ] } \| o ^ { j } - \bar { o } ^ { j } \| _ { \mathrm { o c c } } } \end{array}$ . Further, for $\beta > 0$ , we also define

$$
\Psi _ { j , \beta } ( o ^ { j } ) : = - \beta \sum _ { t = 0 } ^ { T - 1 } \sum _ { x \in \mathsf { X } } \sum _ { u \in \mathsf { U } } \eta _ { t } ^ { j } ( x , u ) \log \frac { \eta _ { t } ^ { j } ( x , u ) } { \xi _ { t } ^ { j } ( x ) } .
$$

and $\Phi _ { j , \beta } ( \tilde { o } ^ { j } ; o ) = \Phi _ { j } ( \tilde { o } ^ { j } ; o ) + \Psi _ { j , \beta } ( \tilde { o } ^ { j } )$ , where $\Phi _ { j } ( \tilde { o } ^ { j } ; o )$ is defined in (9).

Theorem 3 (Contractivity, $\varepsilon = 1$ (informal)). Let Assumption 2 hold. Fix the parameter $\beta > 0 .$ . Define $\mathsf { B r } _ { \beta } ( o ) : = \left( \mathsf { B r } _ { 1 , \beta } ( o ) , \hdots , \mathsf { B r } _ { M , \beta } ( o ) \right)$ where $\mathsf { B r } _ { j , \beta } ( o ) : =$ <sup>ar</sup><sub>in</sub> $\mathrm { m a x } _ { \widetilde { o } ^ { j } \in \mathsf { O } ^ { j } } \Phi _ { j , \beta } ( \widetilde { o } ^ { j } ; o )$ $\begin{array} { r } { I f \beta > \operatorname* { m a x } _ { j \in [ M ] } L _ { j } , } \end{array}$ , then $\mathsf { B r } _ { \beta }$ is a contraction on O $\| \cdot \| _ { \mathrm { o c c , \infty } } .$ Consequently, the recursion $\mu ( n + 1 ) = \Gamma _ { \beta } ( \mu ( n ) )$ converges to the unique entropy-regularized $R A { - } M P { - } M F E$

We now furnish our final technical result, a contraction for $ { \varepsilon } \in \left( 0 , 1 \right)$ . We first need the following assumption.

Assumption 3. Consider the decomposition (13). We assume that there exist $c _ { t } ^ { j } :$ $\Chi \to \mathbb { R }$ and $A _ { t } ^ { j , k } : \mathsf { X } \times \mathsf { X } \to \mathbb { R } , k \neq j ,$ , such thatfor all $x \in \mathsf X$ and $\nu ^ { - j } \in \mathcal { P } ( \mathsf { X } ) ^ { M - 1 }$

$$
a _ { t } ^ { j } ( x , \nu ^ { - j } ) = c _ { t } ^ { j } ( x ) + \sum _ { k \not = j } \sum _ { y \in { \mathsf { X } } } A _ { t } ^ { j , k } ( x , y ) \nu ^ { k } ( y ) .\tag{15}
$$

We also define the quantities which will be used later, in particular, $\ell _ { j , t } ^ { a } : =$ $\begin{array} { r } { \sum _ { k \neq j } \operatorname* { m a x } _ { x \in \mathsf { X } } \| A _ { t } ^ { j , k } ( x , \cdot ) \| _ { \infty } } \end{array}$ and $\begin{array} { r } { L _ { j } ^ { a } : = \sum _ { t = 1 } ^ { T } \ell _ { j , t } ^ { a } } \end{array}$

Theorem 4 (Contractivity, $ { \varepsilon } \in \left( 0 , 1 \right)$ (informal)). Suppose that Assumptions 2 and 3 hold. For $\widehat { \mu } ^ { - j } = \{ \widehat { \mu } _ { t } ^ { k } \} _ { k \neq j , t = 0 } ^ { T } ,$ let $\begin{array} { r } { \left\| \widehat { \mu } ^ { - j } \right\| _ { 2 } ^ { 2 } : = \sum _ { k \neq j } \sum _ { t = 0 } ^ { T } \sum _ { x \in \mathsf { X } } \left| \widehat { \mu } _ { t } ^ { k } ( x ) \right| ^ { 2 } } \end{array}$ define the inner-regularized risk-averse payof as

$$
\Phi _ { j , \beta } ^ { \sigma } ( \widetilde { \sigma } ^ { j } ; o ) : = \operatorname* { m i n } _ { \substack { \widehat { \mu } ^ { - j } \in \mathcal { W } _ { j } ( \xi ( o ) , \varepsilon ) } } [ \sum _ { t = 0 } ^ { T } \sum _ { x \in \mathsf { X } } \widetilde { \xi } _ { t } ^ { j } ( x ) \cdot\tag{16}
$$

$$
r _ { t } ^ { j } \left( x , \xi _ { t } ^ { j } , \widehat { \mu } _ { t } ^ { - j } \right) + \frac { \sigma } { 2 } \left\| \widehat { \mu } ^ { - j } \right\| _ { 2 } ^ { 2 } \Bigg ] + \Psi _ { j , \beta } ( \widetilde { \sigma } ^ { j } ) .
$$

Define $\mathsf { B r } _ { j , \beta } ^ { \sigma } ( o ) : = \arg \operatorname* { m a x } _ { \widetilde { o } ^ { j } \in \Theta ^ { j } } \Phi _ { j , \beta } ^ { \sigma } ( \widetilde { o } ^ { j } ; o )$ , and the concatenated qunatity $\mathsf { B r } _ { \beta } ^ { \sigma } ( o ) : = \left( \mathsf { B r } _ { 1 , \beta } ^ { \sigma } ( o ) , \ldots , \mathsf { B r } _ { M , \beta } ^ { \sigma } ( o ) \right)$ . There exist constants $\Lambda _ { j , \varepsilon } \geq 0$ such that, for � > ma $\mathsf { \bar { \rho } } _ { j \in [ M ] } ( L _ { j } + 2 L _ { j } ^ { a } \Lambda _ { j , \varepsilon } )$ B is a contraction on O in $\| \cdot \| _ { \mathrm { o c c , \infty } } ,$ uniformly in $\sigma > 0 .$ As $\sigma \to 0 ^ { + }$ , its fixed point converges to the unique fixed point of $\mathsf { B r } _ { \beta }$ Moreover, the recursion $\mu ( n + 1 ) = \Gamma _ { \beta } ( \mu ( n ) )$ converges to the unique entropy-regularized $R A { - } M P { - } M F E$

## 5<sub>.</sub> Fi<sub>c</sub>titi<sub>ous</sub> Pl<sub>ay</sub>

Fictitious play (FP) has been a long-standing algorithm used to compute equilibria for mean-field games with and without entropy regularization [Bro51, $\mathrm { \bf P } { \bf P } { \bf L } ^ { + } 2 { \bf 0 } ;$ EFCK25, Ber07]. In this work, we present an extended version of the algorithm to multiple populations under risk aversion, thereby utiziling it as an efective algorithm for computing robust policies. In particular, for each population $j \in [ M ]$ , each time step $t = 0 , \ldots , T$ , and iteration index ℓ, the algorithm iteratively computes a best response $\bar { \pi } _ { t } ^ { j , \ell }$ to the population distribution induced by the average of the best responses from previous iterations. Specifically, for each population $j \in [ M ]$ , each time step $t = 0 , \ldots , T$ , and iteration index ℓ, we have [PPL<sup>+</sup>20]:

$$
\bar { \mu } _ { t } ^ { j , \ell } = \frac { \ell - 1 } { \ell } \bar { \mu } _ { t } ^ { j , \ell - 1 } + \frac { 1 } { \ell } \mu _ { t } ^ { \pi ^ { j , \ell } } .\tag{17}
$$

The policy generating this average distribution is [PPL<sup>+</sup>20]:

$$
\bar { \pi } _ { t } ^ { j , \ell } ( u | x ) = \frac { \sum _ { i = 0 } ^ { \ell } \mu _ { t } ^ { \pi ^ { j , i } } ( x ) \pi _ { t } ^ { j , i } ( u | x ) } { \sum _ { i = 0 } ^ { \ell } \mu _ { t } ^ { \pi ^ { j , i } } ( x ) } ,\tag{18}
$$

where $\bar { \pi } _ { t } ^ { j , \ell } ( u | x )$ is the average of the best responses of population � from iteration 0 upto iteration ℓ. More details are given in Appendix G.

Similar to [EFCK25] we present the proof of the algorithm for continuous time iterations and the implemented discrete time version can be seen as a numerical approximation. The continuous time analysis enables us to deploy standard tools from calculus and control theory, in particular, continuous-time Lyapunov functions. We denote the fictitious play time by �, so as to explicitly diferentiate it from the MP-MFG time horizon. We make the following standard assumptions on the reward structure defined in (13) [PPL<sup>+</sup>20, PPE<sup>+</sup>21, EFCK25]:

Assumption 4. Let $\mathcal { V } \subset \mathbb { R } ^ { | \mathsf { X } | }$ be an open set containing P(X). Under Assumption 2, for every $j \in [ M ] , t \in \{ 0 , . . . , T \}$ , and $x \in \mathsf { X } , b _ { t } ^ { j } ( x , \cdot )$ is continuously diferentiable on Vand $\nu \mapsto b _ { t } ^ { j } ( \cdot , \nu )$ is monotone, i.e., $\begin{array} { r } { \sum _ { x \in \mathsf { X } } ( \nu ( x ) - \nu ^ { \prime } ( x ) ) \big ( b _ { t } ^ { j } ( x , \nu ) - b _ { t } ^ { j } ( x , \nu ^ { \prime } ) \le 0 } \end{array}$ for all $\nu , \nu ^ { \prime } \in \mathcal { P } ( \mathsf { X } )$

Remark 3. Observe that Assumption 4 imposes a monotonicity condition on the own-population component $b _ { t } ^ { j }$ ofthe rewardfunction and does not restrict the reward component depending on the opponent-dependent populations $a _ { t } ^ { j } .$ . It is therefore weaker than requiring monotonicity of the full multi-population reward with respect to the entire mean-field profile, which is the case in existing literature $[ P P L ^ { + } 2 0 .$ $P P E ^ { + } 2 1$ , EFCK25].

Recall the definition of risk-averse exploitibility in Definition 5. Then, the following result holds.

Theorem 5 (Convergence of FP (informal)). Let the RA-MP-MFG satisfy Assumptions 1 and 4, and let $\varepsilon \ = \ 1$ in (5). Then, the exploitability of the continuous-time fictitious play process is a strong Lyapunov function of the system, i.e., ${ \begin{array} { l } { { \frac { \mathrm { d } } { \mathrm { d } s } } \phi _ { \mathsf { R A } } ^ { j } ( { \bar { \pi } } ^ { s } ) \ \leq \ - { \frac { 1 } { s } } \phi _ { \mathsf { R A } } ^ { j } ( { \bar { \pi } } ^ { s } ) \ \forall \ s \ \geq \ 1 } \end{array} }$ and $j ~ \in ~ [ M ]$ . Consequently, $\phi _ { \mathsf { R A } } ( \bar { \pi } ^ { s } ) = \odot ( 1 / s )$

## 6<sub>.</sub> N<sub>u</sub>m<sub>e</sub>ri<sub>ca</sub>l E<sub>xpe</sub>rim<sub>e</sub>nt<sub>s</sub>

In this section, we investigate the efects of risk aversion in MP-MFGs and demonstrate the eficacy of our proposed entropy-regularized risk-averse fixed-point iteration (RA-FPI) and risk-averse fictitious play (RA-FP) for computing equilibrium solutions across a range of settings. For all problems, we report the exploitability of the RA-MP-MFE (Definition 5) and its entropy-regularized counterpart, where $\bar { \mathcal { J } } _ { j } ( \pi ^ { j } ; \mu ^ { \pi } )$ in Definition 5 is replaced by $\bar { \mathcal { J } } _ { j , \beta } ( \pi ^ { j } ; \mu ^ { \pi } )$ .

To the best of our knowledge, there are no existing baselines that can be directly applied to the class of RA-MP-MFG problems considered here. In the absence of standard baselines, we instead investigate the sensitivity of diferent MP-MFG equilibrium notions to deviations in the mean-field trajectories. Specifically, we consider the standard multi-population MFE (MP-MFE) [PPE<sup>+</sup>21], its entropy-regularized counterpart (ER-MP-MFE), our proposed RA-MP-MFE (Definition 6) computed using RA-FP, and its entropy-regularized counterpart (Definition 8) computed using RA-FPI, hereafter referred to a ER-RA-MP-MFE, respectively. For each equilibrium policy, we generate � perturbations of the nominal equilibrium mean-field distributions. For each perturbed mean-field flow, we evaluate the expected cumulative reward of the representative agent under the corresponding equilibrium policy. We then rank the � perturbed trajectories according to their expected cumulative rewards and report the mean diference between the nominal performance and the average performance over the worst 10% of perturbations

<table><tr><td></td><td></td><td>MP- MFE</td><td>ER-MP- MFE</td><td>RA-MP- MFE</td><td>ER-RA-MP- MFE</td></tr><tr><td>Grid</td><td>Nominal</td><td>7.67</td><td>7.60</td><td>0.35</td><td>0.37</td></tr><tr><td>Navigation</td><td>∆EQ</td><td>7.65</td><td>7.58</td><td>0.29</td><td>0.35</td></tr><tr><td>3 Population</td><td>Nominal</td><td>12.13</td><td>11.73</td><td>12.05</td><td>11.77</td></tr><tr><td>Chasing</td><td> $\Delta _ { \mathrm { E Q } }$ </td><td>1.16</td><td>1.91</td><td>1.10</td><td>2.0</td></tr><tr><td>4 Population Nominal</td><td></td><td>11.91</td><td>11.59</td><td>11.78</td><td>11.58</td></tr><tr><td>Chasing</td><td> $\Delta _ { \mathrm { E Q } }$ </td><td>1.26</td><td>2.0</td><td>1.13</td><td>2.06</td></tr><tr><td>Stock</td><td>Nominal</td><td>136.65</td><td>136.77</td><td>130.56</td><td>130.49</td></tr><tr><td>Trading</td><td> $\Delta _ { \mathrm { E Q } }$ </td><td>25.21</td><td>25.51</td><td>22.09</td><td>22.16</td></tr></table>

Table 1. Expected rewards under nominal mean-field flow $( \mu _ { t } ^ { j , \mathrm { E Q } } , \mu _ { t } ^ { - j , \mathrm { E Q } } )$ and sensitivity of corresponding equilibrium policies as defined in (19).

$$
\begin{array} { r l } & { \displaystyle \Delta _ { \mathrm { E Q } } : = \mathbb { E } ^ { \pi ^ { j , \mathrm { E Q } } } \left[ \sum _ { t = 0 } ^ { T } r _ { t } ^ { j } \left( { x } _ { t } ^ { j } , { \mu } _ { t } ^ { j , \mathrm { E Q } } , { \mu } _ { t } ^ { - j , \mathrm { E Q } } \right) \right] - } \\ & { \displaystyle \frac { 1 } { | \mathcal { I } _ { 1 0 } \circ \mathcal { I } _ { \mathrm { o } } | } \sum _ { i \in \mathcal { I } _ { 1 0 } \circ \mathcal { I } _ { 0 } } \mathbb { E } ^ { \pi ^ { j , \mathrm { E Q } } } \left[ \sum _ { t = 0 } ^ { T } r _ { t } ^ { j } \left( { x } _ { t } ^ { j } , { \mu } _ { t } ^ { j , \mathrm { E Q } } , { \mu } _ { t } ^ { - j , ( i ) } \right) \right] , } \end{array}\tag{19}
$$

where $\mathcal { I } _ { 1 0 \% }$ denotes the set of indices corresponding to the 10% of perturbations with the lowest expected cumulative rewards, and ${ \mathrm { E Q ~ } } \in \ \left\{ { \tt { M P - M F E , E R - M P - M F E } } , \right. \qquad $ Thus, $\Delta _ { \mathrm { E Q } }$ provides an empirical measure of the sensitivity of each equilibrium policy to unfavorable mean-field trajectory deviations. Additionally, we perform extensive ablation studies to examine the efects of the key parameters introduced in our formulation, namely $\varepsilon , \beta ,$ and $\sigma .$ . Further details are given in Appendix H.

In the case where $\varepsilon = 1$ , the ambiguity set (5) becomes independent of the nominal trajectory, allowing the inner minimization over the uncertain population distributions in (8) to be reformulated using dual variables $V _ { t } \in \mathbb { R } ^ { | \times | }$ for all $t = 0 , \ldots , T .$ . A detailed derivation of this dual reformulation is provided in Appendix H.

Grid N<sub>av</sub>i<sub>ga</sub>ti<sub>o</sub>n $( \varepsilon = 1 )$ . We consider an RA-MP-MFG with two populations traversing a 5×5 grid. Population 2 aims to reach the bottom-right corner (Figure 1(c)) and receives a per-time-step reward proportional to the fraction of the population at the target. Population 1, at a given state $x _ { t } ^ { 1 } \in \mathsf { X } .$ , receives $r _ { t } ^ { 1 } ( x _ { t } ^ { 1 } , \mu _ { t } ^ { 1 } , \mu _ { t } ^ { 2 } ) = \dot { \mu _ { t } ^ { 2 } } ( x _ { t } ^ { 1 } )$ , thereby incentivizing it to align its distribution with that of Population 2. We assume that Population 1 is risk-averse to deviations in Population $2 \mathrm { { ' } s }$ trajectory. Table 1 shows that MP-MFE and ER-MP-MFE achieve higher nominal cumulative rewards but exhibit a substantial degradation in performance under deviations from the equilibrium flows, indicating greater sensitivity to trajectory perturbations. In contrast, RA-MP-MFE and ER-RA-MP-MFE achieve lower nominal rewards while exhibiting greater robustness to deviations, highlighting the tradeof between reward maximization and robustness.

As shown in Figure 1(d)-(f), risk aversion leads Population 1 from a nearly deterministic trajectory to a more spatially diversified strategy. When $\varepsilon = 1$ , the ambiguity set contains all dynamically feasible trajectories of Population $^ { 2 , }$ whose behavior is independent of Population 1. The resulting worst-case objective therefore favors policies that distribute probability mass across multiple states, providing greater robustness to distributional deviations. Increasing $\beta$ strengthens entropy regularization and, consequently, improves the reliability of RA-FPI, as illustrated in Figure 1(a). However, increasing $\beta$ also encourages greater stochasticity in the decision-making policies, leading to more uniform population trajectories. Figure 1(b) further shows that the FP exploitability decays at a rate consistent with Theorem 5.

(d)

![](images/28bde17aba292a8749121dd482be15b35c05c9521a94828626bc5b6a81861cdd.jpg)

![](images/a0e4833b0c8a3eb8e913bf7b470e43ed2edcaf00dea4d7741443b7b35a43e5ca.jpg)

![](images/9522c5637e1a7d377767914f3d5ea37baf255545dbfe35c23ea41bd155de4f8a.jpg)  
Figure 1. (a) RA-FPI; (b) RA-FP; (c)-(f) population distributions of Navigation game.

M<sub>u</sub>lti<sub>-</sub>P<sub>opu</sub>l<sub>a</sub>ti<sub>on</sub> Ch<sub>as</sub>i<sub>ng</sub> $( \varepsilon = 1 )$ . The multi-population chasing game introduced in [PPE<sup>+</sup>21] is a standard MP-MFG setting in which � populations move over a 2D grid, with rewards of the form: $r ^ { j } ( x , \mu ^ { 1 } , \ldots , \mu ^ { M } ) : = - \log ( \mu ^ { j } ( x ) ) +$ $\textstyle \sum _ { i \neq j } \mu ^ { i } ( x ) { \bar { r } } ^ { j , i } ( x )$ , for all $x \in \mathsf X$ . The dependence of each population’s reward on the mean-field distributions of all populations results in a coupled problem. We consider two variants of this problem on a $5 \times 5$ grid: (i) a 3 population chasing problem in which Population 1 is risk-averse to deviations in Population 2’s trajectory, and (ii) a 4 population chasing problem in which Population 1 is risk-averse to deviations in the trajectories of Populations 2 and 3, while all other interactions remain risk-neutral. Once again, Table 1 shows that RA-MP-MFE may achieve lower nominal rewards but its average worst-case reward degrades less under deviations. We attribute the higher sensitivity of ER-RA-MP-MFE to the larger value of $\beta ( = 1 )$ , which, although necessary to ensure convergence, can induce increasingly uniform behavior and lead to greater deviations from the true objective function.

Figure 2 similarly shows that entropy regularization promotes convergence toward the ER-RA-MP-MFE as $\beta$ increases, while FP exhibits slower but steady convergence toward the RA-MP-MFE in both variants. In the 4 population case, the risk-neutral equilibrium is characterized by symmetric, spatially dispersed trajectories (Figure 2(e)). Risk aversion qualitatively changes this behavior, with the key diferences between the risk-neutral and risk-averse settings highlighted by the red box. Under risk aversion, Population 1 breaks this symmetry and adopts a safer trajectory with respect to Populations 2 and 3 by reducing the mass pursuing Population 2, thereby reducing its vulnerability to feasible deviations in Population 2’s trajectory.

St<sub>oc</sub>k M<sub>a</sub>rk<sub>e</sub>t Tr<sub>a</sub>din<sub>g</sub> $( \varepsilon = 1 )$ . Consider the stock market example presented in Section 1. We construct a simplified, multi-population model of the the trading scenario introduced as the price impact problem in [Car20, CL18] between AI (Population 1) and human (Population 2) populations by discretizing the inventory [WFT<sup>+</sup>26] and assuming endogenous price changes that are independent of the mean-field.

The discrete inventory positions are $\mathsf { X } : = \{ - K , \hdots , 0 , \hdots , K \}$ for some integer $K > 0$ representing the holdings of a population. The actions available to each population are SELL, HOLD, and BUY, with inventory transitions evolving deterministically. The change in the stock price at time � is denoted by $\Delta p _ { t }$ and follows $\Delta p _ { t } : = \gamma _ { t } + \theta _ { t } z _ { t }$ , where $\gamma _ { t }$ represents the price drift, $\theta _ { t }$ represents the volatility and $z \sim$ Discrete $( - 1 , 0 , 1 )$ , where $\mathsf { E } [ z _ { t } ] ~ = ~ 0$ and $\mathrm { v a r } ( z _ { t } ) ~ = ~ 1$ . The rewards for the AI population follows $r _ { t } ^ { 1 } ( x _ { t } ^ { 1 } , \mu _ { t } ^ { 1 } , \mu _ { t } ^ { 2 } ) : = \gamma _ { t } x _ { t } ^ { 1 } + c _ { 0 } x _ { t } ^ { 1 } \cdot \left. q , \mu _ { t } ^ { 2 } \right. - c _ { 1 } ( x _ { t } ^ { 1 } ) ^ { 2 }$ , where $\boldsymbol { q } ^ { \intercal } : = [ - K , \dots , 0 , \dots K ]$ and $c _ { 0 } , c _ { 1 } > 0$ . The term $c _ { 0 } x _ { t } ^ { 1 } \cdot \left. q , \mu _ { t } ^ { 2 } \right.$ represents the price pressure created by aggregate human inventory shifts and $c _ { 1 } ( x _ { t } ) ^ { 2 }$ penalizes large AI inventory positions due to volatility and holding costs. The rewards for the human population is given by $r _ { t } ^ { 2 } ( x _ { t } ^ { 2 } , \mu _ { t } ^ { 1 } , \mu _ { t } ^ { 2 } ) : = \gamma _ { t } x _ { t } ^ { 2 } - w _ { 0 } ( \bar { \theta } _ { t } x _ { t } ) ^ { 2 } + w _ { 1 } x _ { t } ^ { 2 } \cdot \left. q , \mu _ { t } ^ { 2 } \right. - w _ { 2 } \mathbb { 1 } _ { \gamma _ { t } < - \chi } ( x _ { t } ^ { 2 } + \hat { K } )$ where $- w _ { 0 } ( \theta _ { t } x _ { t } ) ^ { 2 } , w _ { 1 } x _ { t } ^ { 2 } \cdot \left. q , \mu _ { t } ^ { 2 } \right.$ and $w _ { 2 } \mathbb { 1 } _ { \gamma _ { t } < - \chi } ( x _ { t } ^ { 2 } + K )$ represent the penalty due to volatility in the price, herding behavior and penalty due to panic selling when the stock price drift falls below $- \chi$

![](images/28b27b48279ff1311ecf199e147d968d9dbb7dfb0a9486df946f6de93cc7a35e.jpg)

4 Population Chasing  
![](images/ef2065f3ce3fdc5347a855f43656be5bc6cfa417b63bd427731a5516d016aa17.jpg)

![](images/4573ec74d4da4b7410a06026030b6f0d536aa1aa8c941a8287aa68edc95d4f1e.jpg)

![](images/e87b33b47848bf2a15a590d413e7f7274ad2fe00327e0bddfcb1defb80f63ac6.jpg)

(e)  
![](images/f90f1eab00f7517bc29dd837d7f82f617d4fced042d63d16d7001954de036047.jpg)  
Figure 2. (a)-(b) RA-FPI; (c)-(d) RA-FP; (e) population distributions of 4 Population Chasing.

In order to bridge the gap between mean-field theory and real-world applications, we calibrate the price model to mimic the dynamics of Apple Inc. (AAPL) over September 2025, using data retrieved through yfinance [Aro19] and analyze the efect of risk-aversion of the AI population to the deviations in the human population’s meanfield trajectory. One can verify that the AI reward does not satisfy the monotonocity condition in Assumption 4. We therefore use a small entropy-regularization parameter, $\beta = 1 0 ^ { - 6 }$ , and compute the ER-RA-MP-MFE as an approximation to the RA-MP-MFE, as supported by Proposition 3 in Appendix E. Table 1 shows that the risk-averse policy exhibits the lowest sensitivity to trajectory deviations, while Figure 3 illustrates the resulting exploitability for diferent values of $\beta .$

Lin<sub>ea</sub>r N<sub>av</sub>i<sub>ga</sub>ti<sub>o</sub>n $( \varepsilon \in ( 0 , 1 ) )$ . We consider an MP-MFG with two populations traversing a three-state environment, where agents can move LEFT, RIGHT, or remain STATIONARY. Similar to the Grid Navigation problem, Population 1, at a given state $x _ { t } ^ { 1 } \in \mathsf { X } ,$ , receives $r _ { t } ^ { 1 } ( x _ { t } ^ { 1 } , \mu _ { t } ^ { 1 } , \mu _ { t } ^ { 2 } ) : = \ : \breve { \mu } _ { t } ^ { 2 } ( x _ { t } ^ { 1 } )$ , which satisfies Assumption 3. We assume that Population 1 is risk-averse to deviations in Population $2 \mathrm { { } s }$ trajectory. For $\varepsilon \in ( 0 , 1 ) , \mathcal { W } _ { j } ( \mu , \varepsilon )$ depends explicitly on the nominal mean-field trajectory. Consequently, the inner minimization is coupled with the nominal trajectory and the dual formulation no longer yields a tractable problem or a computational speedup. We therefore solve the resulting max-min problem directly using the iterative

![](images/db247846a80cad563348dac667031e1fb69e861bcf957497bcf6f626bd224d23.jpg)

Figure 3. Exploitability for Stock Market Trading.  
![](images/25238f11c1ba06a3381e5a7e96e8f78a524c369afb2359d4188d4c918c37b4c4.jpg)

![](images/5471e03e98215357da2cfe150a0679f970a438023204439df1ea984797288a01.jpg)

![](images/339052ced8f07b53d51a5163a6e103d4aca9f24fba069a20434c3909be499f77.jpg)  
Figure 4. (a) Varying � under $\sigma = 1 ;$ (b) Varying � under $\varepsilon = 0 . 1$ (c) Cumulative rewards.

MINIMAX-APPA algorithm proposed in [LJJ20], which is designed to handle such coupled minimax optimization problems. Figure 4(a) shows a reduction in exploitability for fixed values of $\sigma$ and $\beta = 1$ , consistent with the guarantee of Theorem 4. We also observe that the risk-averse payof decreases as � increases (Figure 4(c)), as expected since the set of dynamically feasible perturbations expands with �. Finally, an ablation over diferent values of $\sigma ,$ with fixed $\varepsilon = 0 . 1$ and $\beta = 1$ (Figure 4(b)), shows that reducing the value of $\sigma$ makes the inner minimization problem more dificult to solve due to weaker convexity.

While MINIMAX-APPA performs well in low-dimensional settings, its iterative nature can lead to numerical errors as the state-action space grows. Indeed, directly solving the resulting max-min problem can become challenging due to the complexity of alternating optimization and the scalability of iterative minimax methods. Developing eficient and numerically viable approaches for larger state-action spaces remains an important direction for future work.

## 7<sub>.</sub> C<sub>o</sub>n<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n

We introduced a framework for risk-averse multi-population mean-field games, in which each population solves a coupled robust optimization problem against dynamically feasible deviations of the other populations. This coupling gave rise to a novel risk-averse multi-population mean-field equilibrium. Using tools from variational analysis, we established the existence of an RA-MP-MFE and leveraged entropy regularization to obtain contractive approximate RA-MP-MFEs. Through Lyapunov analysis, we further showed that an extended fictitious-play dynamics converges to an RA-MP-MFE under appropriate monotonicity conditions. Numerical experiments demonstrated that RA-MP-MFEs are less sensitive to deviations in the nominal meanfield trajectories compared to traditional MP-MFEs, highlighting the robustness induced by our formulation. An important direction for future work is to extend the framework to model-free settings with function approximation.

## R<sub>e</sub>f<sub>erences</sub>

[AB06] C. D. Aliprantis and K. C. Border, Infinite Dimensional Analysis: A Hitchhiker’s Guide, third ed., Springer, Berlin, 2006, doi: https://doi.org/10.1007/3-540-29587-9.

[ApS19] MOSEK ApS, The mosek optimization software., 2019.

[Aro19] Ran Aroussi, yfinance: Pythonic access to market data, 2019, https://github.com/ ranaroussi/yfinance.

[BBM15] F. Borrelli, A. Bemporad, and M. Morari, Predictive Control for Linear and Hybrid Systems, Cambridge Univ. Press, Cambridge, England, UK, 2015.

[Bec17] A. Beck, First-Order Methods in Optimization, MOS-SIAM Series on Optimization, vol. 25, Society for Industrial and Applied Mathematics (SIAM), Philadelphia, PA; Mathematical Optimization Society, Philadelphia, PA, 2017, doi: https://doi.org/10.1137/ 1.9781611974997.ch1.

[Ber07] U. Berger, Brown’s original fictitious play, Journal of Economic Theory 135 (2007), no. 1, 572–578, doi: https://doi.org/10.1016/j.jet.2005.12.010.

[BHL18] A. Bensoussan, T. Huang, and M. Laurière, Mean field control and mean field game models with several populations, 2018, doi: https://doi.org/10.48550/arXiv.1810. 00783.

[BLP21] J. F. Bonnans, P. Lavigne, and L. Pfeifer, Discrete-time meanfield games with risk-averse agents, ESAIM: Control, Optimisation and Calculus of Variations 27 (2021), 44, doi: https://doi.org/10.1051/cocv/2021044.

[BMDP02] A. Bemporad, M. Morari, V. Dua, and E. N. Pistikopoulos, The explicit linear quadratic regulatorfor constrained systems, Automatica 38 (2002), no. 1, 3–20, doi: https://doi. org/10.1016/S0005-1098(01)00174-1.

[Bro51] G. W. Brown, Iterative solution of games by fictitious play, Activity Analysis of Productions and Allocation 13 (1951), no. 1, 374.

[BV04] S. Boyd and L. Vandenberghe, Convex Optimization, Cambridge University Press, Cambridge, 2004, doi: https://doi.org/10.1017/CBO9780511804441.

[Can23] C. L. Canonne, A short note on an inequality between KL and TV, 2023, doi: https: //doi.org/10.48550/arXiv.2202.07198.

[Car20] Rene Carmona, Applications of mean field games in financial engineering and economic theory, 2020, doi: https://doi.org/10.48550/arXiv.2012.05237.

[CDL16] R. Carmona, F. Delarue, and D. Lacker, Mean field games with common noise, The Annals of Probability 44 (2016), no. 6, 3740–3803, doi: https://doi.org/10.1214/ 15-AOP1060.

[CH19] P. E. Caines and M. Huang, Graphon mean field games and the GMFG equations: �- Nash equilibria, 2019 IEEE 58th Conference on Decision and Control (CDC), 2019, doi: https://doi.org/10.1109/CDC40024.2019.9029871, pp. 286–292.

[CHFK24] K. Cui, S. Hauck, C. Fabian, and H. Koeppl, Learning decentralized partially observable mean field control for artificial collective behavior, 2024, doi: https://doi.org/10. 48550/arXiv.2307.06175.

[CJ26] Z. Cheng and S. Jaimungal, Risk-averse mean field games: exploitability and nonasymptotic analysis, Stochastic Processes and their Applications (2026), 105022, doi: https://doi.org/10.1016/j.spa.2026.105022.

[CK21] K. Cui and H. Koeppl, Approximately solving mean field games via entropy-regularized deep reinforcement learning, International Conference on Artificial Intelligence and Statistics, PMLR, 2021, doi: https://doi.org/10.48550/arXiv.2102.01585, pp. 1909– 1917.

[CL18] P. Cardaliaguet and C.-A. Lehalle, Mean field game of controls and an application to trade crowding, Mathematics and Financial Economics 12 (2018), no. 3, 335–363, doi: https://doi.org/10.1007/s11579-017-0206-z.

[CSS25] P. Cardaliaguet, B. Seeger, and P. Souganidis, Mean field games with common noise and degenerate idiosyncratic noise, The Annals of Applied Probability 35 (2025), no. 3, 1531– 1569, doi: https://doi.org/10.1214/25-AAP2148.

[Dan12] J. M. Danskin, The Theory ofMax-Min and its Application to Weapons Allocation Problems, Springer Science & Business Media, 2012, doi: https://doi.org/10.1007/ 978-3-642-46092-0.

[DL26] F. Delarue and P. Lavigne, Robust mean-field games under entropy-based uncertainty, Mathematical Control and Related Fields (2026), doi: https://doi.org/10.3934/ mcrf.2026031.

[DR14] A. L. Dontchev and T. R. Rockafellar, Implicit Functions and Solution Mappings, 2nd ed., Springer Series in Operations Research and Financial Engineering, Springer, New York, 2014, doi: https://doi.org/10.1007/978-1-4939-1037-3.

[EFCK25] Y. Eich, C. Fabian, K. Cui, and H. Koeppl, Bounded rationality equilibrium learning in mean field games, Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, 2025, doi: https://doi.org/10.1609/aaai.v39i13.33508, pp. 13796–13804.

[Fol99] G. B. Folland, Real Analysis: Modern Techniques and their Applications, John Wiley & Sons, 1999.

[GAT24] Y. Guan, M. Afshari, and P. Tsiotras, Zero-sum games between mean-field teams: Reachability-based analysis under mean-field sharing, Proceedings of the AAAI Conference on Artificial Intelligence 38 (2024), no. 9, 9731–9739, doi: https://doi.org/ 10.1609/aaai.v38i9.28831.

[GAXJ24] Y. Gao, A. Abate, L. Xie, and K. H. Johansson, Distributional reachability for Markov decision processes: Theory and applications, IEEE Transactions on Automatic Control 69 (2024), no. 7, 4598–4613, doi: https://doi.org/10.1109/TAC.2023.3341282.

[GC26] P. J. Goulart and Y. Chen, Clarabel: An interior-point solver for conic programs with quadratic objectives, Mathematical Programming Computation (2026), 1–83, doi: https: //doi.org/10.1007/s12532-026-00320-7.

[GGWX21] H. Gu, X. Guo, X. Wei, and R. Xu, Mean-field controls with Q-learning for cooperative MARL: convergence and complexity analysis, SIAM Journal on Mathematics of Data Science 3 (2021), no. 4, 1168–1196, doi: https://doi.org/10.1137/20M1360700.

[GMU25] L. Guth, D. Maldague, and J. Urschel, Estimating the matrix � → � norm, SIAM Journal on Matrix Analysis and Applications 46 (2025), no. 3, 2080–2092, doi: https://doi. org/10.1137/24M1647035.

[HMC06] M. Huang, R. P. Malhamé, and P. E. Caines, Large population stochastic dynamic games: closed-loop McKean-Vlasov systems and the Nash certainty equivalence principle, Communications in Information & Systems 6 (2006), no. 3, 221 – 252, doi: https://doi.org/10.4310/cis.2006.v6.n3.a5.

[HZ25] A. Hu and J. Zhang, MF-OML: Online mean-field reinforcement learning with occupation measures for large population games, Applied Mathematics & Optimization 92 (2025), no. 3, 52, doi: https://doi.org/10.1007/s00245-025-10328-5.

[JGT25] B. Jeloka, Y. Guan, and P. Tsiotras, Learning large-scale competitive team behaviors with mean-field interactions and online opponent modeling, 2025, doi: https://doi.org/ 10.48550/arXiv.2504.21164.

[JGT26] , Robust mean-field games with risk aversion and bounded rationality, 2026, doi: https://doi.org/10.48550/arXiv.2602.13353.

[LJJ20] Tianyi Lin, Chi Jin, and Michael I. Jordan, Near-optimal algorithmsfor minimax optimization, Proceedings of Thirty Third Conference on Learning Theory (Jacob Abernethy and Shivani Agarwal, eds.), Proceedings of Machine Learning Research, vol. 125, PMLR, 09– 12 Jul 2020, URL: https://proceedings.mlr.press/v125/lin20a.html, pp. 2738– 2779.

[LL07] J.-M. Lasry and P.-L. Lions, Meanfield games, Japanese Journal of Mathematics 2 (2007), no. 1, 229–260, doi: https://doi.org/10.1007/s11537-007-0657-8.

[LLL<sup>+</sup>19] Marc Lanctot, Edward Lockhart, Jean-Baptiste Lespiau, Vinicius Zambaldi, Satyaki Upad hyay, Julien Pérolat, Sriram Srinivasan, Finbarr Timbers, Karl Tuyls, Shayegan Omidshafiei, Daniel Hennes, Dustin Morrill, Paul Muller, Timo Ewalds, Ryan Faulkner, János Kramár, Bart De Vylder, Brennan Saeta, James Bradbury, David Ding, Sebastian Borgeaud,

Matthew Lai, Julian Schrittwieser, Thomas Anthony, Edward Hughes, Ivo Danihelka, and Jonah Ryan-Davis, OpenSpiel: A framework for reinforcement learning in games, CoRR (2019), doi: https://doi.org/10.48550/arXiv.1908.09453.

[LPG<sup>+</sup>22] M. Laurière, S. Perrin, S. Girgin, P. Muller, A. Jain, T. Cabannes, G. Piliouras, J. Pérolat, R. Élie, O. Pietquin, and M. Geist, Scalable deep reinforcement learning algorithms for meanfield games, 2022, doi: https://doi.org/10.48550/arXiv.2203.11973.

[LPP<sup>+</sup>24] M. Laurière, S. Perrin, J. Pérolat, S. Girgin, P. Muller, R. Élie, O. Pietquin, and M. Geist, Learning in mean field games: A survey, 2024, doi: https://doi.org/10.48550/ arXiv.2205.12944.

[LZZZ26] Z. Liang, Z. Zhou, Y. Zhuang, and B. Zou, Mean-field games under model uncertainty, 2026, doi: https://doi.org/10.48550/arXiv.2601.12226.

[MB14] J. Moon and T. Başar, Linear-quadratic risk-sensitive mean field games, 53rd IEEE Conference on Decision and Control, 2014, doi: https://doi.org/10.1109/CDC.2014. 7039801, pp. 2691–2696.

[MPS25] E. Mazumdar, K. Panaganti, and L. Shi, Tractable multi-agent reinforcement learning through behavioral economics, The Thirteenth International Conference on Learning Representations, 2025, URL: https://openreview.net/forum?id=stUKwWBuBm.

[NJG17] G. Neu, A. Jonsson, and V. Gómez, A unified view ofentropy-regularized Markov decision processes, 2017, doi: https://doi.org/10.48550/arXiv.1705.07798.

[NSŠ23] A. Neufeld, J. Sester, and M. Šikić, Markov decision processes under model uncertainty, Mathematical Finance 33 (2023), no. 3, 618–665, doi: https://doi.org/10.1111/ mafi.12381.

[Øks03] B. Øksendal, Stochastic Diferential Equations: An Introduction with Applications, sixth ed., Universitext, Springer-Verlag, Berlin, 2003, doi: https://doi.org/10.1007/ 978-3-642-14394-6.

[PB14] N. Parikh and S. Boyd, Proximal Algorithms, Foundations and Trends in optimization 1 (2014), no. 3, 127–239, doi: https://doi.org/10.1561/2400000003.

[PPE<sup>+</sup>21] J. Perolat, S. Perrin, R. Elie, M. Laurière, G. Piliouras, M. Geist, K. Tuyls, and O. Pietquin, Scaling up mean field games with online mirror descent, 2021, doi: https://doi.org/ 10.48550/arXiv.2103.00623.

[PPL<sup>+</sup>20] S. Perrin, J. Perolat, M. Laurière, M. Geist, R. Elie, and O. Pietquin, Fictitious play for mean field games: Continuous time analysis and applications, Advances in neural information processing systems 33 (2020), 13199–13213, URL: https://proceedings.neurips.cc/paper\_files/paper/ 2020/file/995ca733e3657ff9f5f3c823d73371e1-Paper.pdf.

[RF88] H. L. Royden and P. M. Fitzpatrick, Real Analysis, fourth ed., vol. 32, Macmillan New York, 1988.

[Roc97] T. R. Rockafellar, Convex Analysis, Princeton Landmarks in Mathematics, Princeton University Press, Princeton, NJ, 1997, doi: https://doi.org/10.1515/9781400873173.

[Rud76] W. Rudin, Principles of Mathematical Analysis, third ed., International Series in Pure and Applied Mathematics, McGraw-Hill Book Co., New York-Auckland-Düsseldorf, 1976. MR 385023

[SBMR20] N. Saldi, T. Başar, and Maxim M. Raginsky, Approximate Markov-Nash equilibria for discrete-time risk-sensitive mean-field games, Mathematics of Operations Research 45 (2020), no. 4, 1596–1620, doi: https://doi.org/10.1287/moor.2019.1044.

[SBR23] N. Saldi, T. Başar, and M. Raginsky, Partially observed discrete-time risk-sensitive mean field games, Dynamic Games and Applications 13 (2023), no. 3, 929–960, doi: https: //doi.org/10.1007/s13235-022-00453-z.

[SC19] N. Sen and P. E. Caines, Mean field games with partial observation, SIAM Journal on Control and Optimization 57 (2019), no. 3, 2064–2091, doi: https://doi.org/10. 1137/17M1140133.

[SKM23] J. Subramanian, A. Kumar, and A. Mahajan, Mean-field games among teams, 2023, doi: https://doi.org/10.48550/arXiv.2310.12282.

[SPTH22] S. G. Subramanian, P. Poupart, M. E. Taylor, and N. Hegde, Multi type mean field reinforcement learning, 2022, doi: https://doi.org/10.48550/arXiv.2002.02513.

[SSAL24] K. Shao, J. Shen, C. An, and M. Laurière, Reinforcement learningforfinite space mean-field type games, 2024, doi: https://doi.org/10.48550/arXiv.2409.18152.

[Tem15] H. Tembine, Risk-sensitive mean-field-type games with �<sup>�</sup>-norm drifts, Automatica 59 (2015), 224–237, doi: https://doi.org/10.1016/j.automatica.2015.06.036.

[TZB13] H. Tembine, Q. Zhu, and T. Başar, Risk-sensitive mean-field games, IEEE Transactions on Automatic Control 59 (2013), no. 4, 835–850, doi: https://doi.org/10.1109/TAC. 2013.2289711.

[WFT<sup>+</sup>26] Clarisse Wibault, Johannes Forkel, Sebastian Rene Towers, Tiphaine Wibault, Juan Agustin Duque, George Whittle, Andreas Schaab, Yucheng Yang, Chiyuan Wang, Michael A Osborne, Benjamin Moll, and Jakob Nicolaus Foerster, Recurrent structuralpolicy gradient forpartially observable meanfield games, Forty-third International Conference on Machine Learning, 2026, URL: https://openreview.net/forum?id=VkZQThGNgI.

[YWSM25] Yucheng Yang, Chiyuan Wang, Andreas Schaab, and Benjamin Moll, Structural reinforcement learning for heterogeneous agent macroeconomics, 2025, doi: https: //doi.org/10.48550/arXiv.2512.18892.

[YYP26] D. Yu, S. You, and C. Pei, Occupation-measure mean-field control: Optimization over measures and Frank-Wolfe methods, 2026, doi: https://doi.org/10.48550/arXiv. 2603.16094.

[Zei86] E. Zeidler, Nonlinear Functional Analysis and its Applications. I: Fixed-Point Theorems, Springer-Verlag, New York, 1986, doi: https://doi.org/10.1007/ 978-1-4612-4838-5.

[Zei95] , Applied Functional Analysis: Applications to Mathematical Physics, Applied Mathematical Sciences, vol. 108, Springer-Verlag, New York, 1995, doi: https://doi. org/10.1007/978-1-4612-0815-0.

![](images/cc3ddf23fb8431f8bcd476da8bd5db3855bc86f8acb90faaa8edc73477417a20.jpg)  
Figure 5. Our primary theoretical developments.

## A<sub>pp</sub>endix

## A<sub>ppe</sub>ndi<sub>x</sub> A<sub>.</sub> T<sub>ec</sub>hni<sub>ca</sub>l TL<sub>;</sub>DR <sub>a</sub>nd <sub>a</sub> r<sub>oa</sub>dm<sub>ap</sub>

We summarize our primary theoretical contributions in brief; see also Figure 5 for a bird’s-eye view.

(Res-a) Main theorems: Theorem 1 establishes existence of the RA-MP-MFG; Theorems 2–4 provide existence and contractivity results (for � = 1 and � ∈ (0, 1)) for the entropy-regularized RA-MP-MFG. Without any entropy regularization, for the RA-MP-MFG, Theorem 5 analyzes continuous-time fictitious play and provides a certificate of convergence. Proofs are in Appendices B–H.

(Res-b) Auxiliary and standalone results: Lemma 1, 2, and Proposition 1 are technical results needed to prove the Theorems mentioned in (Res-a). Proposition 2 and 3 concerns non-contractivity of the RA-MP-MFE and approximation of MFEs. These are standalone results.

## A<sub>ppe</sub>ndi<sub>x</sub> B<sub>.</sub> R<sub>eac</sub>h<sub>a</sub>bl<sub>e</sub> <sub>se</sub>t

Let us recall the definition of the reachable set (this is same as that of Definition 1 in Section 2).

Definition 1 (Reachable set). Fix $j \in [ M ] , t \in \{ 0 , \ldots , T - 1 \}$ , and $\mu _ { t } ^ { j } \in \mathcal { P } ( \mathsf { X } )$ The one-step reachable set ofpopulation � from $\mu _ { t } ^ { j }$ is defined by

$$
\begin{array} { r } { \mathcal { R } _ { t } ^ { j } \big ( \mu _ { t } ^ { j } \big ) : = \Big \{ \mu ^ { + } \in \mathcal { P } ( \mathsf { X } ) \Big | \exists \pi _ { t } ^ { j } \in \Pi _ { t } s u c h t h a t \mu ^ { + } = \mu _ { t } ^ { j } F _ { t } ^ { j } \big ( \pi _ { t } ^ { j } \big ) \Big \} . } \end{array}\tag{20}
$$

Lemma 1 records some structural properties of $\mathcal { R } _ { t } ^ { j } \left( \mu _ { t } ^ { j } \right)$ , which will be utilized later in our analysis.

Lemma 1 (Structure of $\mathscr { R } _ { t } ^ { j } ( \mu _ { t } ^ { j } ) )$ . Fix $j \in [ M ] , t \in \{ 0 , \ldots , T - 1 \}$ , and $\mu _ { t } ^ { j } \in$ ${ \mathcal { P } } ( \mathsf { X } )$ . Then, the following assertions hold:

```latex
(1-a) The reachable set $\mathcal { R } _ { t } ^ { j } \left( \mu _ { t } ^ { j } \right)$ defined in (20) is nonempty, compact, convex,
and polyhedral. In particular, $\mathscr { R } _ { t } ^ { j } \left( \mu _ { t } ^ { j } \right)$ is a polytope contained in ${ \mathcal { P } } ( \mathsf { X } )$
(1-b) $\mathcal { R } _ { t } ^ { j } \left( \mu _ { t } ^ { j } \right)$ admits the representation
(21) $\mathcal R _ { t } ^ { j } \big ( \mu _ { t } ^ { j } \big ) = \{ \mu ^ { + } \begin{array} { l } { \displaystyle \big | \exists \eta : \mathsf X \times \mathsf { U } \to \mathbb { R } _ { + } s u c h \ t h a t \ \sum _ { u \in \mathsf { U } } \eta ( x , u ) = \mu _ { t } ^ { j } ( x ) } \\ { \displaystyle f o r \ a l l \ x \in \mathsf { X } , \ \mu ^ { + } ( y ) = \sum _ { x \in \mathsf { X } } \sum _ { u \in \mathsf { U } } \eta ( x , u ) f _ { t } ^ { j } ( y | x , u ) \} , } \\ { \displaystyle f o r \ a l l \ y \in \mathsf { X } } \end{array} $
where $\mu ^ { + } \in \mathcal { P } ( \mathsf { X } )$
```

Proof. Fix $j \in [ M ] , t \in \{ 0 , \ldots , T - 1 \}$ , and $\mu _ { t } ^ { j } \in \mathcal { P } ( \mathsf { X } )$ . Recall that $\Pi _ { t }$ is the set of randomized Markov decision rules at time �

$$
\Pi _ { t } : = { \Bigl \{ } \pi _ { t } : { \mathsf { X } } \times { \mathsf { U } } \to [ 0 , 1 ] \Big |  \sum _ { u \in { \mathsf { U } } } \pi _ { t } ( u | x ) = 1 { \mathrm { ~ f o r ~ e v e r y ~ } } x \in { \mathsf { X } } { \Bigr \} } .
$$

An element of this set assigns an action distribution $\pi _ { t } ( \cdot | x ) \in { \mathcal { P } } ( \cup )$ to each state $x \in \mathsf { X }$ . Thus $\Pi _ { t }$ can be identified with $\textstyle \prod _ { x \in { \mathsf { X } } } { \mathcal { P } } ( \cup )$ Since X and U are finite and nonempty, $\Pi _ { t }$ is nonempty. For $\pi _ { t } ^ { j } \in \Pi _ { t }$ , denote the next mean-field state by $\mu ^ { + } : = \mu _ { t } ^ { j } F _ { t } ^ { j } ( \pi _ { t } ^ { j } )$ , so that

$$
\textsf { X } \ni y \mapsto \mu ^ { + } ( y ) = \sum _ { x \in \mathsf { X } } \sum _ { u \in \mathsf { U } } \mu _ { t } ^ { j } ( x ) \pi _ { t } ^ { j } ( u | x ) f _ { t } ^ { j } ( y | x , u ) .
$$

It follows that $\mu ^ { + } ( y ) \geq 0$ for all $y \in \mathsf X$ , and

$$
\sum _ { y \in \mathbb { X } } \mu ^ { + } ( y ) = \sum _ { x \in \mathbb { X } } \sum _ { u \in \mathbb { U } } \mu _ { t } ^ { j } ( x ) \pi _ { t } ^ { j } ( u | x ) \sum _ { y \in \mathbb { X } } f _ { t } ^ { j } ( y | x , u ) = \sum _ { x \in \mathbb { X } } \mu _ { t } ^ { j } ( x ) \sum _ { u \in \mathbb { U } } \pi _ { t } ^ { j } ( u | x ) = 1 .
$$

Hence, $\mu ^ { + } \in \mathcal { P } ( \mathsf { X } )$ , and thus $\mathscr { R } _ { t } ^ { j } \left( \mu _ { t } ^ { j } \right) \neq \emptyset$

Finiteness of X and U implies that $\Pi _ { t }$ is a compact and convex polytope. The map $\Pi _ { t } \ni \pi _ { t } ^ { j } \mapsto \mu _ { t } ^ { j } F _ { t } ^ { j } ( \pi _ { t } ^ { j } ) \in \mathcal { P } ( \mathsf { X } )$ is continuous and afine. Therefore, its image $\mathcal { R } _ { t } ^ { j } \left( \mu _ { t } ^ { j } \right)$ is compact [Rud76, Ch. 3] and convex [BV04, Sec. 2].

We now show that $\mathcal { R } _ { t } ^ { j } ( \mu _ { t } ^ { j } )$ has the representation (21) and it admits a polyhedral structure. Let $\mu ^ { + } \in \mathcal { R } _ { t } ^ { j } \left( \mu _ { t } ^ { j } \right)$ . The following holds:

◦ By definition, there exists $\pi _ { t } ^ { j } \in \Pi _ { t }$ such that $\mu ^ { + } = \mu _ { t } ^ { j } F _ { t } ^ { j } ( \pi _ { t } ^ { j } )$

◦ Define $\eta : \mathsf { X } \times \mathsf { U } \to \mathbb { R } _ { + } \mathsf { b y } \left( x , u \right) \mapsto \eta ( x , u ) : = \mu _ { t } ^ { j } ( x ) \pi _ { t } ^ { j } ( u | x )$ . Then, $\begin{array} { r } { \sum _ { u \in \mathsf { U } } \eta ( x , u ) = } \end{array}$ $\begin{array} { r } { \mu _ { t } ^ { j } ( x ) \sum _ { u \in \mathsf { U } } \pi _ { t } ^ { j } ( u | x ) = \mu _ { t } ^ { j } ( x ) } \end{array}$ for all $x \in \mathsf { X } ,$ , and $\begin{array} { r } { \mu ^ { + } ( y ) = \sum _ { x \in \mathsf { X } } \sum _ { u \in \mathsf { U } } \eta ( x , u ) f _ { t } ^ { j } ( y | x , u ) } \end{array}$ for all $y \in \mathsf { X }$

◦ Conversely, suppose that $\eta ( \cdot )$ satisfies $\begin{array} { r } { \sum _ { u \in \mathsf { U } } \eta ( x , u ) = \mu _ { t } ^ { j } ( x ) } \end{array}$ for all $x \in \mathsf { X }$ . If $\mu _ { t } ^ { j } ( x ) > 0 .$ , define $\pi _ { t } ^ { j } ( u | x ) : = \eta ( x , u ) / \mu _ { t } ^ { j } ( x )$ for $u \in \mathsf { U } .$ . If $\mu _ { t } ^ { j } ( x ) = 0$ , then $\begin{array} { r } { \sum _ { u \in \mathsf { U } } \eta ( x , u ) = 0 , } \end{array}$ , and since $\eta ( x , u ) \geq 0 \quad$ , we have $\eta ( x , u ) = 0 $ for all $u \in \mathsf { U } ;$ in this case choose $\pi _ { t } ^ { j } ( \cdot | x )$ arbitrarily in ${ \mathcal { P } } ( \cup )$ . Thus, $\pi _ { t } ^ { j } \in \Pi _ { t }$ . Moreover, for every

� ∈ X,

$$
\mu ^ { + } ( y ) = \sum _ { x \in \mathsf { X } } \sum _ { u \in \mathsf { U } } \eta ( x , u ) f _ { t } ^ { j } ( y | x , u ) = \sum _ { x \in \mathsf { X } } \sum _ { u \in \mathsf { U } } \mu _ { t } ^ { j } ( x ) \pi _ { t } ^ { j } ( u | x ) f _ { t } ^ { j } ( y | x , u ) ,
$$

and hence $\mu ^ { + } = \mu _ { t } ^ { j } F _ { t } ^ { j } ( \pi _ { t } ^ { j } )$

Therefore, $\mu ^ { + } \in \mathcal { R } _ { t } ^ { j } \left( \mu _ { t } ^ { j } \right)$ , proving (21). Finally, note that, the feasible set for � in (21) is described by finitely many linear equalities and inequalities. It is also bounded, since

$$
0 \leq \eta ( x , u ) \leq \sum _ { u ^ { \prime } \in \mathsf { U } } \eta ( x , u ^ { \prime } ) = \mu _ { t } ^ { j } ( x ) \leq 1 .
$$

Hence, it is a polytope. The map $\begin{array} { r } { \eta \mapsto \sum _ { x \in \mathsf { X } } \sum _ { u \in \mathsf { U } } \eta ( x , u ) f _ { t } ^ { j } ( y | x , u ) } \end{array}$ is linear. Therefore, $\mathcal { R } _ { t } ^ { j } \left( \mu _ { t } ^ { j } \right)$ is a linear image of a polytope, and hence is itself a polytope. Our proof is complete. □

## A<sub>pp</sub>endix C<sub>.</sub> Ambi<sub>gu</sub>it<sub>y</sub> set

Recall the definition of the ambiguity set from Section 3

$$
\mathcal { W } _ { j } ( \mu , \varepsilon ) : = \{ \begin{array} { c } { \{ \hat { \mu } _ { t } ^ { k } \} _ { t = 0 } ^ { T } , } \\ { \forall k \in [ M ] , k \neq j } \end{array} | \hat { \mu } _ { 0 } ^ { k } = \mu _ { 0 } ^ { k } , \hat { \mu } _ { t + 1 } ^ { k } \in \mathcal { R } _ { t } ^ { k } ( \hat { \mu } _ { t } ^ { k } ) , \} .\tag{22}
$$

We now restate and provide a detailed proof of Proposition 1 from Section 3.

Proposition 1 (Structure of the ambiguity set). Fix $j \in [ M ]$ , a nominal flow ${ \boldsymbol \mu } : = \{ { \boldsymbol \mu } _ { t } \} _ { t = 0 } ^ { T }$ with $\mu _ { t } : = ( \mu _ { t } ^ { 1 } , \ldots , \mu _ { t } ^ { M } ) \bar { \in \mathcal P } ( \mathsf X ) ^ { M }$ , and let $\varepsilon \in [ 0 , 1 ]$ . Then:

$( 1 - a ) \ \mathcal { W } _ { j } ( \mu , \varepsilon )$ is a compact polyhedral subset of $\begin{array} { r } { \prod _ { k \neq j } \prod _ { t = 0 } ^ { T } \mathcal { P } ( \mathsf { X } ) } \end{array}$ .

If, in addition, for every $k \neq j$ , the nominal flow $\mu ^ { k } = \{ \mu _ { t } ^ { k } \} _ { t = 0 } ^ { T }$ is dynamically feasible, i.e., there exists a policy sequence $\{ \pi _ { t } ^ { k } \} _ { t = 0 } ^ { T - 1 }$ such that $\mu _ { t + 1 } ^ { k } = \mu _ { t } ^ { k } F _ { t } ^ { k } ( \pi _ { t } ^ { k } )$ for $t = 0 , \ldots , T - 1$ , then,

(1-b) $\mathcal { W } _ { j } ( \mu , \varepsilon ) \neq \emptyset .$ . In that case, $\mathcal { W } _ { j } ( \mu , \varepsilon )$ is a nonempty compact polytope.

Proof. We first show that $\mathcal { W } _ { j } ( \mu , \varepsilon )$ is compact and admits a polyhedral nature. By Lemma 1, for each $k \neq j$ and $t \in \{ 0 , \ldots , T - 1 \}$ , the reachability condition $\hat { \mu } _ { t + 1 } ^ { k } \in$ $\mathcal { R } _ { t } ^ { k } ( \hat { \mu } _ { t } ^ { k } )$ is equivalent to the existence of the mapping $\eta _ { t } ^ { k } : \mathsf { X } \times \mathsf { U } \to \mathbb { R } _ { + }$ such that $\begin{array} { r } { \sum _ { u \in \mathsf { U } } \eta _ { t } ^ { k } ( x , u ) = \hat { \mu } _ { t } ^ { k } ( x ) } \end{array}$ for all $x \in \mathsf X$ and $\begin{array} { r } { \hat { \mu } _ { t + 1 } ^ { k } ( y ) = \sum _ { x \in { \cal { X } } } \sum _ { u \in \mathsf { U } } \eta _ { t } ^ { k } ( x , u ) f _ { t } ^ { k } ( y | x , u ) } \end{array}$ for all $y \in \mathsf { X }$ . To show compactness of $\mathcal { W } _ { j } ( \dot { \mu } , \dot { \varepsilon } )$ , we first construct a larger set which we will show is compact. Then, we obtain $\mathcal { W } _ { j } ( \mu , \varepsilon )$ as a continuous projection.

To this end, let $\widehat { \mathcal { W } _ { j } } ( \mu , \varepsilon )$ be the set of all variables $( \hat { \mu } ^ { - j } , \eta ^ { - j } , s ^ { - j } )$ satisfying the following finite family of constraints

$$
{ \hat { \mu } } _ { t } ^ { k } ( x ) \geq 0 , \quad \sum _ { x \in { \cal X } } { \hat { \mu } } _ { t } ^ { k } ( x ) = 1 , \quad k \not = j , t = 0 , \ldots , T \mathrm { f o r } x \in { \sf X } , { \hat { \mu } } _ { 0 } ^ { k } = { \mu } _ { 0 } ^ { k } \mathrm { f o r } k \not = j ,
$$

$$
\eta _ { t } ^ { k } ( x , u ) \geq 0 , \quad \sum _ { u \in \mathfrak { U } } \eta _ { t } ^ { k } ( x , u ) = \hat { \mu } _ { t } ^ { k } ( x ) \mathrm { ~ f o r ~ } k \neq j , t = 0 , \ldots , T - 1 , x \in \mathsf { X } , \mathrm { ~ a n d ~ } u \in \mathsf { U } ,
$$

$$
{ \hat { \mu } } _ { t + 1 } ^ { k } ( y ) = \sum _ { x \in \mathsf { X } } \sum _ { u \in \mathsf { U } } \eta _ { t } ^ { k } ( x , u ) f _ { t } ^ { k } ( y | x , u ) { \mathrm { ~ f o r ~ } } k \neq j , \ t = 0 , \ldots , T - 1 , \ \mathrm { a n d } \ y \in \mathsf { X } ,
$$

$$
s _ { t } ^ { k } ( x ) \geq 0 , \ s _ { t } ^ { k } ( x ) \geq \hat { \mu } _ { t } ^ { k } ( x ) - \mu _ { t } ^ { k } ( x ) , \ s _ { t } ^ { k } ( x ) \geq \mu _ { t } ^ { k } ( x ) - \hat { \mu } _ { t } ^ { k } ( x ) , \sum _ { x \in X } \frac { s _ { t } ^ { k } ( x ) } { 2 } \leq \varepsilon ,
$$

for $k \neq j , t = 0 , \ldots , T$ , and $x \in \mathsf { X } .$ . The auxiliary variables $x \mapsto s _ { t } ^ { k } ( x )$ encode the total-variation constraints. Indeed, if d<sub>TV</sub> $\left( \mu _ { t } ^ { k } , \hat { \mu } _ { t } ^ { k } \right) \leq \varepsilon .$ , one may choose $x \mapsto s _ { t } ^ { k } ( x ) : =$ $\left| \mu _ { t } ^ { k } ( x ) - \hat { \mu } _ { t } ^ { k } ( x ) \right|$ . Conversely, if such $s _ { t } ^ { k } ( x )$ exist, then $\left| \mu _ { t } ^ { k } ( x ) - \hat { \mu } _ { t } ^ { k } ( x ) \right| \leq s _ { t } ^ { \dot { k } } ( x )$ for every $x \in \mathsf { X } ,$ and therefore

$$
\mathrm { d } _ { \mathrm { T V } } \big ( \mu _ { t } ^ { k } , \hat { \mu } _ { t } ^ { k } \big ) = \frac { 1 } { 2 } \sum _ { x \in \mathsf { X } } \big | \mu _ { t } ^ { k } ( x ) - \hat { \mu } _ { t } ^ { k } ( x ) \big | \leq \frac { 1 } { 2 } \sum _ { x \in \mathsf { X } } s _ { t } ^ { k } ( x ) \leq \varepsilon .
$$

All constraints defining $\widehat { \mathcal { W } _ { j } } ( \mu , \varepsilon )$ are linear equalities or linear inequalities in the finitedimensional variables $( \hat { \mu } ^ { - j } , \eta ^ { - j } , s ^ { - j } )$ . Therefore $\widehat { \mathcal { W } _ { j } } ( \mu , \varepsilon )$ is a polyhedron. Moreover, $\widehat { \mathcal { W } _ { j } } ( \mu , \varepsilon )$ is bounded. Indeed, $\begin{array} { r } { 0 \le \hat { \mu } _ { t } ^ { k } ( x ) \le 1 , 0 \le \eta _ { t } ^ { k } ( x , u ) \le \sum _ { u ^ { \prime } \in \cup } \eta _ { t } ^ { k } ( x , u ^ { \prime } ) = } \end{array}$ $\hat { \mu } _ { t } ^ { k } ( x ) \leq 1$ , and $0 \leq s _ { t } ^ { k } ( x ) \leq 2 \varepsilon$ . Hence, $\widehat { \mathcal { W } _ { j } } ( \mu , \varepsilon )$ is a bounded polyhedron, and therefore compact. If it is nonempty, it is a polytope.

By Lemma 1 and the auxiliary-variable representation of the total-variation constraints, $\mathcal { W } _ { j } ( \mu , \varepsilon )$ is the projection of $\widehat { \mathcal { W } _ { j } } ( \mu , \varepsilon )$ onto the variables $( \hat { \mu } ^ { - j } , \eta ^ { - j } , s ^ { - j } )$ ↦→ $\hat { \mu } ^ { - j }$ . Since projection is a linear (and thus continuous) map, the projection of a compact set is compact. Moreover, the projection of a polyhedron is polyhedral. Therefore, $\mathcal { W } _ { j } ( \mu , \varepsilon )$ is a compact polyhedral subset of $\begin{array} { r } { \prod _ { k \neq j } \prod _ { t = 0 } ^ { T } \mathcal { P } ( \mathsf { X } ) } \end{array}$

We now prove the non-emptiness of $\mathcal { W } _ { j } ( \mu , \varepsilon )$ . Suppose that, for every $k \neq j ,$ there exists a policy sequence $\{ \bar { \pi _ { t } ^ { k } } \} _ { t = 0 } ^ { T - 1 }$ such that $\mu _ { t + 1 } ^ { k } = \mu _ { t } ^ { k } F _ { t } ^ { k } ( \pi _ { t } ^ { k } )$ for $t = 0 , \ldots , T { - } 1$ Define $\hat { \mu } _ { t } ^ { k } : = \mu _ { t } ^ { k }$ for $k \neq j$ and $t = 0 , \ldots , T$ . Then $\hat { \mu } _ { 0 } ^ { k } = \mu _ { 0 } ^ { k }$ , and $\hat { \mu } _ { t + 1 } ^ { k } = \mu _ { t + 1 } ^ { k } =$ $\mu _ { t } ^ { k } F _ { t } ^ { k } ( { \boldsymbol \pi } _ { t } ^ { k } ) = \hat { \mu } _ { t } ^ { k } F _ { t } ^ { k } ( { \boldsymbol \pi } _ { t } ^ { k } )$ , so $\hat { \mu } _ { t + 1 } ^ { k } \in \mathcal { R } _ { t } ^ { k } ( \hat { \mu } _ { t } ^ { k } )$ . Finally,

$$
\mathrm { d } _ { \mathrm { T V } } \big ( \mu _ { t } ^ { k } , \hat { \mu } _ { t } ^ { k } \big ) = \mathrm { d } _ { \mathrm { T V } } \big ( \mu _ { t } ^ { k } , \mu _ { t } ^ { k } \big ) = 0 \leq \varepsilon .
$$

Therefore $\hat { \mu } ^ { - j } = \mu ^ { - j }$ belongs to $\mathcal { W } _ { j } ( \mu , \varepsilon )$ , and hence $\mathcal { W } _ { j } ( \mu , \varepsilon ) \neq \emptyset$ . Since $\mathcal { W } _ { j } ( \mu , \varepsilon )$ has already been shown to be compact and polyhedral, it follows in the nonempty case that $\mathcal { W } _ { j } ( \mu , \varepsilon )$ is a nonempty compact polytope. □

## A<sub>ppe</sub>ndi<sub>x</sub> D<sub>.</sub> Pr<sub>oo</sub>f <sub>o</sub>f Th<sub>eo</sub>r<sub>e</sub>m 1

In this section we provide detailed proof of Theorem 1; we restate it for convinience.

Theorem 1 (Existence of RA-MP-MFE). Recall the problem datafrom Section 2 and let $\mu _ { 0 } ^ { j } \in \mathcal { P } ( \mathsf { X } )$ be fixed for all $j \in [ M ]$ Let Assumption 1 hold. Then, for every fixed $\varepsilon \in [ 0 , 1 ]$ , there exists a RA-MP-MFE $( \pi ^ { \star } , \mu ^ { \star } ) \in \Pi \times \mathsf { M }$ . More precisely, there exists $\pi ^ { \star }$ such that, $\mu ^ { \star } : = \mathcal { B } _ { \mathrm { p r o p } } ( \pi ^ { \star } )$ , for every $j \in [ M ]$ , and

$$
\pi ^ { j , \star } \in \underset { \pi ^ { j } \in \Pi ^ { j } } { \arg \operatorname* { m a x } } \ \underset { \widehat { \mu } ^ { - j } \in \mathcal { W } _ { j } ( \mu ^ { \star } , \varepsilon ) } { \operatorname* { i n f } } \ \mathsf { E } ^ { \pi ^ { j } } \left[ \sum _ { t = 0 } ^ { T } r _ { t } ^ { j } \big ( x _ { t } ^ { j } , \mu _ { t } ^ { j , \star } , \widehat { \mu _ { t } } ^ { - j } \big ) \right] .\tag{23}
$$

Remark 4 (Proof roadmap for Theorem 1). The main ingredient ofourproofconsists of techniques from set-valued and variational analysis [AB06, Zei86, DR14]. Our proof for Theorem 1 roughly proceeds via the following steps (see also Fig. 6): (a) First, for each population, we introduce the finite-horizon occupation-measure set and use the equivalence between Markov policies and occupation measures to reformulate the risk-averse best-response problem on a nonempty compact convex set. (b) The main technical step is to establish the required regularity of the riskaverse payof. To this end, we show that the ambiguity-set correspondence has closed graph and is both upper and lower hemicontinuous. Its continuity, together with the continuity of the reward functions, allows us to apply Berge’s maximum principle [AB06, Chapter 17, Theorem 17.31] and conclude that the occupationmeasure payof is jointly continuous. We also show that the payof is concave in the candidate occupation measure. These properties imply that the resulting occupationmeasure best-response correspondence has nonempty, compact, and convex values and is upper hemicontinuous. (c) Kakutani’s fixed-point theorem [AB06, Chapter 17, Corollary 17.55] therefore yields afixed occupation-measure profile. (d) Finally, we reconstruct Markov policies from this fixed point and verify that the resulting policy and mean-field pair satisfies both the risk-averse optimality and mean-field consistency conditions.

![](images/eb00f719e5dc26af29de12cf1aef9ea98fe140359fea62be725365caaef56511.jpg)  
Figure 6. A roadmap of the proof steps for Theorem 1.

Proof of Theorem 1. We write the proof in several steps as outlined in Remark 4.

Step 1 (Occupation measures). For each population $j \in [ M ]$ , let ${ \mathsf { O } } ^ { j }$ denote the set of occupation measures $o ^ { j } : = ( \xi ^ { j } , \eta ^ { j } )$ , where $\xi ^ { j } : = \{ \xi _ { t } ^ { j } \} _ { t = 0 } ^ { T }$ and $\eta ^ { j } : = \{ \eta _ { t } ^ { j } \} _ { t = 0 } ^ { T - 1 }$ satisfying the properties $\left( 2 \right) - \left( 3 \right) . ^ { 4 }$ By the finite-dimensional linear representation used in Lemma 1, each ${ \mathsf { O } } ^ { j }$ is a nonempty compact convex polytope. Hence $\begin{array} { r } { \mathsf { O } : = \prod _ { i = 1 } ^ { M } \mathsf { O } ^ { j } } \end{array}$ is a nonempty compact convex subset of a finite-dimensional Euclidean space [BV04, Section 2.3].

Step 2 (Structural properties of sets). We recall the standard equivalence between Markov policies and occupation measures [GAXJ24]. Every policy $\pi ^ { j } \in \Pi ^ { j }$ induces $o ^ { j } = ( \xi ^ { j } , \eta ^ { j } ) \in { \sf O } ^ { j }$ by $\eta _ { t } ^ { j } ( x , u ) = \xi _ { t } ^ { j } ( x ) \pi _ { t } ^ { j } ( u | x )$ , where $\xi ^ { j }$ is the state-flow generated by $\pi ^ { j }$ . Conversely, every $o ^ { j } = ( \xi ^ { j } , \eta ^ { j } ) \in { \sf O } ^ { j }$ induces a Markov policy by

$$
\mathsf X \ni x \mapsto \pi _ { t } ^ { j } ( u | x ) : = \frac { \eta _ { t } ^ { j } ( x , u ) } { \xi _ { t } ^ { j } ( x ) } \quad \mathrm { i f ~ } \xi _ { t } ^ { j } ( x ) > 0 ,\tag{24}
$$

and by choosing $\pi _ { t } ^ { j } ( \cdot | x )$ arbitrarily in ${ \mathcal { P } } ( \cup ) { \mathrm { ~ i f ~ } } \xi _ { t } ^ { j } ( x ) = 0$ . This policy generates the same occupation measure and the same state-flow [GAXJ24].

For $o : = ( o ^ { 1 } , \dots , o ^ { M } ) \in { \mathsf { O } } .$ , denote the corresponding multi-population flow by $\xi ( o ) : = ( \xi ^ { 1 } , \dots , \xi ^ { M } )$ . For each $j \in [ M ]$ , define the payof of a candidate occupation measure $\tilde { o } ^ { j } : = ( \tilde { \xi } ^ { j } , \tilde { \eta } ^ { j } ) \in 0 ^ { j }$ against the nominal occupation profile � by

$$
( 2 5 ) \enspace \mathsf { O } ^ { j } \times \mathsf { O } \ni \big ( \widetilde { \boldsymbol { \sigma } } ^ { j } ; \boldsymbol { o } \big ) \mapsto \Phi _ { j } ( \widetilde { \boldsymbol { \sigma } } ^ { j } ; \boldsymbol { o } ) : = \operatorname* { i n f } _ { \widetilde { \mu } ^ { - j } \in \mathcal { W } _ { j } ( \xi ( \boldsymbol { o } ) , \varepsilon ) } \sum _ { t = 0 } ^ { T } \sum _ { \boldsymbol { x } \in \mathsf { X } } \widetilde { \xi } _ { t } ^ { j } ( \boldsymbol { x } ) r _ { t } ^ { j } ( \boldsymbol { x } , \xi _ { t } ^ { j } , \widetilde { \mu _ { t } } ^ { - j } ) .
$$

Since $\xi ( o )$ is dynamically feasible by construction, Proposition 1 implies that $\mathcal { W } _ { j } ( \xi ( o ) , \varepsilon )$ is nonempty and compact.

Step 3 (Regularity of $\Phi _ { j } : { \mathsf { O } } ^ { j } \times { \mathsf { O } } \to \mathbb { R } )$ . We need two properties of the function $\Phi _ { j } ( \cdot )$

(1) the joint continuity of $\begin{array} { r } { \mathsf { O } ^ { j } \times \mathsf { O } \ni \left( \tilde { \sigma } ^ { j } ; o \right) \mapsto \Phi _ { j } ( \tilde { \sigma } ^ { j } ; o ) \in \mathbb { R } } \end{array}$ ; and

(2) concavity of ${ \sf O } ^ { j } \ni \tilde { \partial } ^ { j } \mapsto \Phi _ { j } ( \tilde { o } ^ { j } ; o ) \in$ ℝ for each fixed $o \in \mathsf { O }$

We need the following ingredients to prove them.

Ste<sub>p</sub> 4 $( \mu \mapsto \mathcal { W } _ { j } ( \mu , \varepsilon )$ has closed graph [Zei95, Chapter 5, Problem 5.3]). Let $\mathsf { F } ^ { - j }$ denote the set of dynamically feasible opponent flows with the prescribed initial distributions, i.e.,

$$
\mathsf { F } ^ { - j } : = \left\{ \boldsymbol { \nu } ^ { - j } = \{ \nu _ { t } ^ { k } \} _ { t = 0 } ^ { T } \Bigg | \nu _ { 0 } ^ { k } = \mu _ { 0 } ^ { k } , \ \nu _ { t + 1 } ^ { k } \in R _ { t } ^ { k } ( \nu _ { t } ^ { k } ) \mathrm { ~ f o r ~ a l l ~ } k \neq j , \ t = 0 , \ldots , T - 1 \right\} .\tag{26}
$$

By Proposition $1 , \mathsf { F } ^ { - j }$ is a compact convex polytope. Moreover, we have the representation

$$
\mathbb { W } _ { j } ( \mu , \varepsilon ) = \left\{ \nu ^ { - j } \in \mathsf { F } ^ { - j } \biggm | \mathrm { d } _ { \mathrm { T V } } \big ( \nu _ { t } ^ { k } , \mu _ { t } ^ { k } \big ) \le \varepsilon \mathrm { ~ f o r ~ a l l ~ } k \neq j , t = 0 , \ldots , T \right\} .\tag{27}
$$

We first show that $\mu \mapsto { \mathcal { W } } _ { j } ( \mu , \varepsilon )$ , restricted to dynamically feasible nominal flows, has closed graph. The graph [AB06, Chapter 2, Section 2.14] of the set-valued map $\mathcal { W } _ { j }$ is

$$
\operatorname { G r } ( { \mathcal { W } } _ { j } ) : = { \big \{ } ( \mu , \nu ^ { - j } ) ~ { \big | } ~ \nu ^ { - j } \in { \mathcal { W } } _ { j } ( \mu , \varepsilon ) { \big \} } .
$$

Since $\mathcal { W } _ { j }$ has a closed graph it follows that, whenever

$$
( \mu _ { n } ) _ { n \in \mathbb { N } } \to \mu , ( \nu _ { n } ^ { - j } ) _ { n \in \mathbb { N } } \to \nu ^ { - j } , ( \nu _ { n } ^ { - j } ) _ { n \in \mathbb { N } } \in \mathcal { W } _ { j } ( \mu _ { n } , \varepsilon ) , \mathrm { ~ w e ~ h a v e ~ } \nu ^ { - j } \in \mathcal { W } _ { j } ( \mu , \varepsilon ) .
$$

To this end, let $( \mu _ { n } ) _ { n \in \mathbb { N } } \to \mu$ be a convergent sequence of feasible nominal flows, and let $( \nu _ { n } ^ { - j } ) _ { n \in \mathbb { N } } \to \nu ^ { - j }$ with $\nu _ { n } ^ { - j } \in \mathcal { W } _ { j } ( \mu _ { n } , \varepsilon )$ for every $n \in \mathbb N$ . By (27), for every �, we have $\nu _ { n } ^ { - j } \in \mathsf { F } ^ { - j }$ , and since $\mathsf { F } ^ { - j }$ is closed, $\nu ^ { - j } \in \mathsf { F } ^ { - j }$ (closed sets contains all of its limit points [Rud76, Chapter 2]). We also have for every $k \neq j$ and every $t \in \{ 0 , \ldots , T \} , { \mathrm { ~ d } } _ { \mathrm { T V } } ( \nu _ { n , t } ^ { k } , \mu _ { n , t } ^ { k } ) \leq \varepsilon$ . Since the TV distance is continuous over the finite-dimensional simplex, passing to the limit, gives us

$$
\begin{array} { r } { \mathbf { d } _ { \mathrm { T V } } \big ( \nu _ { t } ^ { k } , \mu _ { t } ^ { k } \big ) \leq \varepsilon \quad \mathrm { f o r ~ a l l } \ k \neq j \ \mathrm { a n d } \ t \in \{ 0 , . . . , T \} . } \end{array}
$$

Thus, $\nu ^ { - j } \in \mathcal { W } _ { j } ( \mu , \varepsilon )$ and the graph $\mu \mapsto { \mathcal { W } } _ { j } ( \mu , \varepsilon )$ is closed.

Step 5 (Upper hemicontinuity). We next prove upper hemicontinuity of the correspondence $\mu \mapsto { \mathcal { W } } _ { j } ( \mu , \varepsilon )$ at an arbitrary dynamically feasible nominal flow $\mu .$ We use the open-set definition of upper hemicontinuity [AB06, Chapter 17, Section 17.2, Definition 17.2], together with the sequential characterization for compact-valued correspondences [AB06, Chapter 17, Theorem $1 7 . 1 6 ] . ^ { 5 }$ In the present finite-dimensional setting, it is enough to show that whenever $( \mu _ { n } ) _ { n \in \mathbb { N } } \to \mu$ and $\nu _ { n } ^ { - j } \in \mathcal { W } _ { j } ( \mu _ { n } , \varepsilon )$ for all $n \in \mathbb { N } .$ , the sequence $( \nu _ { n } ^ { - j } ) _ { n \in \mathbb { N } }$ has a subsequence converging to some element of $\mathcal { W } _ { j } ( \mu , \varepsilon )$

Fix an open set � containing $\mathcal { W } _ { j } ( \mu , \varepsilon )$ . Suppose, by contradiction, that $\mu \mapsto$ $\mathcal { W } _ { j } ( { \boldsymbol { \mu } } , { \boldsymbol { \varepsilon } } )$ is not upper hemicontinuous at $\mu$ . Then, there exist a sequence $( \mu _ { n } ) _ { n \in \mathbb { N } } \to \mu$ of dynamically feasible nominal flows and elements $\nu _ { n } ^ { - j } \in \mathcal { W } _ { j } ( \mu _ { n } , \varepsilon )$ such that

$$
\mathbb { N } \ni n \mapsto \nu _ { n } ^ { - j } \notin O .
$$

Since $\mathcal { W } _ { j } ( \mu _ { n } , \varepsilon ) \subseteq \mathsf { F } ^ { - j }$ for every �, and since $\mathsf { F } ^ { - j }$ is compact, there exists a subsequence, still denoted by $( \nu _ { n } ^ { - j } ) _ { n \in \mathbb { N } }$ , and some $\nu ^ { - j } \in \mathsf { F } ^ { - j }$ , such that $( \nu _ { n } ^ { - j } ) _ { n \in \mathbb { N } } \to \nu ^ { - j }$

By the closed-graph property proved in Step 4, we see that $\nu ^ { - j } \in \mathcal { W } _ { j } ( \mu , \varepsilon )$ . Since ${ \mathcal { W } } _ { j } ( \mu , \varepsilon ) \subset O$ and � is open, we have $\nu ^ { - j } \in O$ , and hence $\nu _ { n } ^ { - j } \in O$ for all suficiently large �. This contradicts the fact that $\mathbb { N } \ni n \mapsto \nu _ { n } ^ { - j } \notin O$ . Therefore, $\mu \mapsto { \mathcal { W } } _ { j } ( \mu , \varepsilon )$ is upper hemicontinuous at $\mu .$

Step 6 (Lower hemicontinuity). We now prove lower hemicontinuity of the correspondence $\mu \mapsto { \mathcal { W } } _ { j } ( \mu , \varepsilon )$ [AB06, Chapter 17, Section 17.2, Definition 17.2]. By the sequential characterization of lower hemicontinuity [AB06, Theorem 17.19], it sufices to show that whenever $( \mu _ { n } ) _ { n \in \mathbb { N } } \to \mu$ and $\nu ^ { - j } \in \mathcal { W } _ { j } ( \mu , \varepsilon )$ , there exist a subsequence $\left( \mu _ { n _ { \ell } } \right) _ { \ell \in \mathbb { N } }$ and points $\nu _ { n _ { \ell } } ^ { - j } \in \mathcal { W } _ { j } ( \mu _ { n _ { \ell } } , \varepsilon )$ such that $\left( \nu _ { n _ { \ell } } ^ { - j } \right) _ { \ell \in \mathbb { N } } \to \nu ^ { - j }$ . In fact, we prove the stronger statement by constructing such a sequence along the whole sequence $( \mu _ { n } ) _ { n \in \mathbb { N } }$ . Let $( \mu _ { n } ) _ { n \in \mathbb { N } } \to \mu$ be dynamically feasible nominal flows, and fix $\nu ^ { - j } \in \mathcal { W } _ { j } ( \mu , \varepsilon )$ . We construct $\nu _ { n } ^ { - j } \in \mathcal { W } _ { j } ( \mu _ { n } , \varepsilon )$ such that $( \nu _ { n } ^ { - j } ) _ { n \in \mathbb { N } } \to \nu ^ { - j }$ . If $\varepsilon = 0 ,$ , then $d _ { \mathrm { T V } } ( \nu _ { t } ^ { k } , \mu _ { t } ^ { k } ) = 0$ for all $k \neq j$ and $t = 0 , \ldots , T$ , so $\begin{array} { r } { \nu ^ { - j } = \mu ^ { - j } . } \end{array}$ Since each $( \mu _ { n } ) _ { n \in \mathbb { N } }$ is dynamically feasible, $\mu _ { n } ^ { - j } \in \mathsf { F } ^ { - j }$ . Thus, choosing $\nu _ { n } ^ { - j } : = \mu _ { n } ^ { - j }$ gives $\nu _ { n } ^ { - j } \in \mathcal { W } _ { j } ( \mu _ { n } , 0 )$ and $( \nu _ { n } ^ { - j } ) _ { n \in \mathbb { N } } \to \nu ^ { - j }$

Now suppose $\varepsilon > 0$ . Define

$$
\mathbb { N } \ni n \longmapsto \Delta _ { n } : = \operatorname* { m a x } _ { k \neq j , 0 \leq t \leq T } d _ { \mathrm { T V } } ( \nu _ { t } ^ { k } , \mu _ { n , t } ^ { k } ) .
$$

Since $( \mu _ { n } ) _ { n \in \mathbb { N } } \to \mu$ and $\nu ^ { - j } \in \mathcal { W } _ { j } ( \mu , \varepsilon )$ , we have lim $\begin{array} { r } { \operatorname* { s u p } _ { n  + \infty } \Delta _ { n } \leq \varepsilon . } \end{array}$ . If $\Delta _ { n } \leq \varepsilon .$ set $\lambda _ { n } : = 0$ . If $\Delta _ { n } > \varepsilon$ , set $\lambda _ { n } : = 1 - \varepsilon / \Delta _ { n }$ . Then, $0 \leq \lambda _ { n } < 1$ and $\lambda _ { n } \to 0$ for all $n \in \mathbb { N } .$ . We now employ an interpolation argument which in spirit is analogous to the convex-mixture construction in [NS<sup>Š</sup>23, Proposition 3.1]. To this end, define the sequence

$$
\mathbb { N } \ni n \longmapsto \nu _ { n } ^ { - j } : = ( 1 - \lambda _ { n } ) \nu ^ { - j } + \lambda _ { n } \mu _ { n } ^ { - j } .
$$

Because $\mathsf { F } ^ { - j }$ is convex and both $\nu ^ { - j }$ and $\mu _ { n } ^ { - j }$ belong to $\mathsf { F } ^ { - j }$ , we have $\nu _ { n } ^ { - j } \in \mathsf { F } ^ { - j }$ Moreover, for every $k \neq j$ and $t = 0 , \ldots , T$

$$
\begin{array} { r l } & { d _ { \mathrm { T V } } ( \nu _ { n , t } ^ { k } , \mu _ { n , t } ^ { k } ) = d _ { \mathrm { T V } } \big ( ( 1 - \lambda _ { n } ) \nu _ { t } ^ { k } + \lambda _ { n } \mu _ { n , t } ^ { k } , \mu _ { n , t } ^ { k } \big ) } \\ & { \qquad = ( 1 - \lambda _ { n } ) d _ { \mathrm { T V } } ( \nu _ { t } ^ { k } , \mu _ { n , t } ^ { k } ) \leq ( 1 - \lambda _ { n } ) \Delta _ { n } \leq \varepsilon . } \end{array}
$$

Hence, $\nu _ { n } ^ { - j } \in \mathcal { W } _ { j } ( \mu _ { n } , \varepsilon )$ . Moreover, we know that, $\nu _ { n } ^ { - j } - \nu ^ { - j } = \lambda _ { n } ( \mu _ { n } ^ { - j } - \nu ^ { - j } )$ . Since $\mu _ { n }  \mu .$ , the sequence $\{ \mu _ { n } ^ { - j } \} _ { n \in \mathbb { N } }$ is bounded. Therefore, because $\lambda _ { n } \to 0$ , we get $\nu _ { n } ^ { - j }  \nu ^ { - j }$ . Thus, for every sequence $( \mu _ { n } ) _ { n \in \mathbb { N } } \to \mu$ and every $\nu ^ { - j } \in \mathcal { W } _ { j } ( \mu , \varepsilon )$ , we have constructed $\nu _ { n } ^ { - j } \in \mathcal { W } _ { j } ( \mu _ { n } , \varepsilon )$ such that $( \nu _ { n } ^ { - j } ) _ { n \in \mathbb { N } } \to \nu ^ { - j }$ . Hence, $\mu \mapsto { \mathcal { W } } _ { j } ( \mu , \varepsilon )$ is lower hemicontinuous.

Step 7 (Continuity of $\Phi _ { j } ( \cdot ) )$ . Combining upper hemicontinuity and lower hemicontinuity, the correspondence $\mu \mapsto { \mathcal { W } } _ { j } ( \mu , \varepsilon )$ is continuous [AB06, Chapter 17, Section 17.2, Definition 17.2]. Moreover, by Proposition $1 , \mathcal { W } _ { j } ( \mu , \varepsilon )$ is nonempty and compact for every dynamically feasible nominal flow $\mu .$

We now have the required ingredients; let us return to the regularity of $\Phi _ { j } ( \cdot )$ , as asserted in Step 3. Recall the definitions of $\tilde { o } ^ { j }$ and � and define

$$
\left( \widetilde { \sigma } ^ { j } , o , \widetilde { \mu } ^ { j } \right) \mapsto \mathcal { G } _ { j } \big ( \widetilde { \sigma } ^ { j } , o , \widetilde { \mu } ^ { - j } \big ) : = \sum _ { t = 0 } ^ { T } \sum _ { x \in \mathsf { X } } \widetilde { \xi } _ { t } ^ { j } ( x ) r _ { t } ^ { j } ( x , \xi _ { t } ^ { j } , \widetilde { \mu _ { t } } ^ { - j } ) .\tag{28}
$$

Since the reward functions are Lipschitz continuous in the mean-field argument and the horizon and state space are finite, $( \widetilde { \sigma } ^ { j } , o , \widehat { \mu } ^ { - j } ) \mapsto \mathcal { G } _ { j } \big ( \widetilde { \sigma } ^ { j } , o , \widehat { \mu } ^ { - j } \big )$ is continuous. Moreover,

$$
\Phi _ { j } ( \tilde { \sigma } ^ { j } ; o ) = \operatorname* { m i n } _ { \substack { \hat { \mu } ^ { - j } \in \mathcal { W } _ { j } ( \xi ( o ) , \varepsilon ) } } \mathcal { G } _ { j } ( \tilde { \sigma } ^ { j } , o , \widehat { \mu } ^ { - j } ) ,
$$

where the ‘infimum’ is $\mathbf { a } \ { \cdot } _ { \mathrm { m i n } } ^ { \prime }$ because $\mathcal { W } _ { j } ( \xi ( o ) , \varepsilon )$ is compact and $\mathcal { G } _ { j } ( \cdot )$ is continuous. Since $o \mapsto \xi ( o )$ is continuous and $\mu \mapsto { \mathcal { W } } _ { j } ( \mu , \varepsilon )$ is a nonempty compact-valued continuous correspondence, Berge’s maximum theorem implies that $\Phi _ { j } ( \cdot )$ is continuous on ${ \cal O } ^ { j } \times { \cal O } ;$ see [AB06, Chapter 17].

Step 8 (Existence via the Kakutani fixed-point theorem). We first show that, for a fixed $o \in { \mathsf { O } } .$ , the mapping $\tilde { o } ^ { j } \mapsto \Phi _ { i } ( \tilde { o } ^ { j } ; o )$ is concave. Indeed, for every fixed $\widehat { \mu } ^ { - j } \in \mathcal { W } _ { j } ( \xi ( o ) , \varepsilon )$ , the map $\tilde { \sigma } ^ { j } \mapsto \mathring { \mathcal { G } } \big ( \tilde { \sigma } ^ { j } , o , \widehat { \mu } ^ { - j } \big )$ is afine in $\tilde { \xi } ^ { j }$ . Thus, $\Phi _ { j } ( \cdot ; o )$ being the pointwise infimum of afine mappings, is concave $[ \mathbf { R o c 9 7 } ]$

Define the best response correspondence $\mathsf { B r } _ { j } : \mathsf { O } \Rightarrow \mathsf { O } ^ { j }$ by

$$
\begin{array} { r } { \mathsf { B r } _ { j } ( o ) : = \underset { \tilde { o } ^ { j } \in \mathsf { O } ^ { j } } { \arg \operatorname* { m a x } } \Phi _ { j } ( \tilde { o } ^ { j } ; o ) . } \end{array}
$$

Since ${ \mathsf { O } } ^ { j }$ is compact and $\Phi _ { j } ( \cdot )$ is continuous, Weierstrass’ theorem implies that $\mathsf { B r } _ { j } ( o )$ is nonempty and compact for every $o \in \mathsf { O }$ . Since $\Phi _ { j } ( \cdot ; o )$ is concave and ${ \cal O } ^ { j }$ is convex, the set $\mathsf { B r } _ { j } ( o )$ is convex. Moreover, since the feasible set ${ \mathsf { O } } ^ { j }$ is fixed and compact and the objective $\Phi _ { j } ( \cdot )$ is continuous, Berge’s maximum theorem implies that $\mathsf { B r } _ { j }$ is upper hemicontinuous; see [AB06, Chapter 17].

Define the product correspondence ${ \mathsf { B r } } : { \mathsf { O } } \Rightarrow { \mathsf { O } }$ by $\mathsf { B r } ( o ) : = \Pi _ { i = 1 } ^ { M } \mathsf { B r } _ { j } ( o )$ . Because � is finite, Br has nonempty, compact, convex values and is upper hemicontinuous. Since O is a nonempty compact convex subset of a finite-dimensional Euclidean space, Kakutani’s fixed-point theorem [AB06, Chapter 17, Corollary 17.55] implies that there exists $o ^ { \star } \in \mathsf { O }$ such that $o ^ { \star } \in \mathsf { B R } ( o ^ { \star } )$

Step 9 (Consistency and optimality). Finally, we verify the conditions (i) and (ii) in Definition 6:

◦ Let $o ^ { j , \star } : = ( \xi ^ { j , \star } , \eta ^ { j , \star } )$ , and define $\mu ^ { \star } : = ( \xi ^ { 1 , \star } , \ldots , \xi ^ { M , \star } )$ . Following (24), construct $\pi ^ { j , \star } \in \Pi ^ { j }$ from $o ^ { j , \star }$ by setting $\pi _ { t } ^ { j , \star } ( u | x ) : = \eta _ { t } ^ { j , \star } ( x , u ) / \xi _ { t } ^ { j , \star } ( x )$ whenever $\xi _ { t } ^ { j , \star } ( x ) > 0$ , and choosing $\pi _ { t } ^ { j , \star } ( \cdot | x )$ arbitrarily in ${ \mathcal { P } } ( \cup )$ whenever $\xi _ { t } ^ { j , \star } ( x ) = 0$ By construction, this policy generates the state-flow $\xi ^ { j , \star }$ . Hence, we have $\mu ^ { j , \star } =$ $\mathcal { B } _ { \mathrm { p r o p } } ^ { j } ( \pi ^ { j , \star } )$ for all $j \in [ M ]$ , and thus the condition in Definition 6 holds.

◦ Fix $j \in [ M ]$ and take any $\pi ^ { j } \in \Pi ^ { j }$ . Let $\tilde { o } ^ { j } \in \bigcirc ^ { j }$ be the occupation measure induced by $\pi ^ { j }$ . Since $o ^ { \star } \in \mathsf { B r } ( o ^ { \star } )$ , we have $o ^ { j , \star } \in \mathsf { B r } _ { j } ( o ^ { \star } )$ . Therefore $\Phi _ { j } ( o ^ { j , \star } ; o ^ { \star } ) \ge$ $\Phi _ { j } ( \tilde { o } ^ { j } ; o ^ { \star } )$ . By the equivalence between policies and occupation measures, this inequality implies

$$
\bar { \mathcal { J } } _ { j } ( \pi ^ { j , \star } ; \mu ^ { \star } ) \geq \bar { \mathcal { J } } _ { j } ( \pi ^ { j } ; \mu ^ { \star } ) .
$$

Since $j \in [ M ]$ and $\pi ^ { j } \in \Pi ^ { j }$ were arbitrary, the risk-averse optimality condition holds for every population.

Therefore $( \pi ^ { \star } , \mu ^ { \star } )$ is a risk-averse multi-population mean-field equilibrium. Our assertion of existence stands established. □

Non-contractivity: As we noted in Section 3: contractivity, as in many instances in the mean-field literature [CK21, LPP<sup>+</sup>24, CH19], in general does not follow. The following result is similar in spirit to [CK21, Theorem 2].

Proposition 2 (Non-contractivity of the fixed point operator). Let $\begin{array} { r l } { \mathscr { B } _ { \mathrm { r i s k - o p t } } : } & { { } \leq \frac { 2 } { \pi } \mathopen { } \mathclose \bgroup \left( \tau \aftergroup \egroup \right) } \end{array}$ M → Π be a single-valued selector of the risk-averse best-response correspondence, $i . e . , f o r$ each $\mu \in { \sf M }$ , let $\mathcal { B } _ { \mathrm { r i s k - o p t } } ( \mu ) = ( \pi ^ { 1 } , \dots , \pi ^ { M } ) \in \Pi$ , where, for every $j \in [ M ]$

$$
\pi ^ { j } \in \underset { \pi ^ { j } \in \Pi ^ { j } } { \arg \operatorname* { m a x } } \ \underset { \widehat { \mu } ^ { - j } \in \mathcal { W } _ { j } ( \mu , \varepsilon ) } { \operatorname* { i n f } } \ E ^ { \pi ^ { j } } \left[ \sum _ { t = 0 } ^ { T } r _ { t } ^ { j } \big ( x _ { t } ^ { j } , \mu _ { t } ^ { j } , \widehat { \mu _ { t } } ^ { - j } \big ) \right] .\tag{29}
$$

Define the mean-field fixed-point map $\Gamma : { \mathsf { M } } \to { \mathsf { M } } b .$ y

$$
\mathsf { M } \ni \mu \mapsto \Gamma ( \mu ) : = \mathcal { B } _ { \mathrm { p r o p } } \big ( \mathcal { B } _ { \mathrm { r i s k - o p t } } ( \mu ) \big ) .\tag{30}
$$

Suppose that the image Γ(M) is finite. Then, either Γ is constant, or Γ is not Lipschitz continuous and hence it cannot be a contraction.

Remark 5. Proposition 2 plays an auxiliary role in our analysis: it illustrates why contractivity of the unregularized mean-field map cannot generally be assumed and motivates the use of entropy regularization. Here, the single-valuedness of $\mathcal { B } _ { \mathrm { r i s k } }$ −opt is understood as fixing a selector of the risk-averse best-response correspondence, for example through deterministic tie-breaking.

Proof of Proposition 2. The proof follows the finite-image non-contractivity argument of [CK21, Theorem 2], applied to the mean-field map Γ defined in (30). If Γ is constant there is nothing to show. Let Γ be non-constant. Then, there exist $\mu , \mu ^ { \prime } \in \mathsf { M }$ with $\mu \neq \mu ^ { \prime }$ such that $\Gamma ( \boldsymbol \mu ) \neq \Gamma ( \boldsymbol \mu ^ { \prime } )$ . Since, by assumption, $\Gamma ( \boldsymbol { \mathsf { M } } )$ is finite, the quantity

$$
\delta _ { \Gamma } : = \operatorname* { m i n } _ { \nu , \nu ^ { \prime } \in \mathbb { M } } \mathbb { d } _ { \Gamma \vee } \big ( \Gamma ( \nu ) , \Gamma ( \nu ^ { \prime } ) \big ) ,
$$

is well-defined and $\delta _ { \Gamma } ~ > ~ 0$ by definition. Hence, for any $\nu , \nu ^ { \prime } \in \mathsf { M }$ such that $\Gamma ( \nu ) \neq \Gamma ( \nu ^ { \prime } )$ , we have $\mathrm { d _ { T V } } \big ( \Gamma ( \nu ) , \Gamma ( \nu ^ { \prime } ) \big ) \geq \delta _ { \Gamma }$

To show that Γ cannot be Lipschitz continuous, assume, by contradiction, that Γ is Lipschitz with a Lipschitz constant $L _ { c } > 0$ . Since M is convex, for every $N \in$ ℕ and every $i \in \{ 0 , \ldots , N \}$ , the combination $\begin{array} { r } { \mu ^ { i } : = \frac { i } { N } \mu + \frac { N - i } { N } \mu ^ { \prime } } \end{array}$ belongs to M. Moreover, $\mu _ { 0 } = \mu ^ { \prime }$ and $\mu ^ { N } = \mu$ , and $\mathrm { d _ { T V } } \big ( \mu ^ { i } , \mu ^ { i + 1 } \big ) = \mathrm { d _ { T V } } \big ( \mu , \mu ^ { \prime } \big ) / N$ for all $i \in \{ 0 , \ldots , N - 1 \}$ Choose � suficiently large so that

$$
L _ { c } \mathrm { d } _ { \mathrm { T V } } \big ( \mu ^ { i } , \mu ^ { i + 1 } \big ) = L _ { c } \frac { \mathrm { d } _ { \mathrm { T V } } \big ( \mu , \mu ^ { \prime } \big ) } { N } < \delta _ { \Gamma }
$$

for all $i \in \{ 0 , \ldots , N - 1 \}$ . By the triangle inequality,

$$
\mathrm { d } _ { \mathrm { T V } } \big ( \Gamma ( \mu ) , \Gamma ( \mu ^ { \prime } ) \big ) = \mathrm { d } _ { \mathrm { T V } } \big ( \Gamma ( \mu ^ { N } ) , \Gamma ( \mu _ { 0 } ) \big ) \leq \sum _ { i = 0 } ^ { N - 1 } \mathrm { d } _ { \mathrm { T V } } \big ( \Gamma ( \mu ^ { i } ) , \Gamma ( \mu ^ { i + 1 } ) \big ) .
$$

Since $\Gamma ( \boldsymbol { \mu } ) \neq \Gamma ( \boldsymbol { \mu } ^ { \prime } )$ , there exists some $i \in \{ 0 , \ldots , N - 1 \}$ such that $\Gamma ( \mu ^ { i } ) \neq \Gamma ( \mu ^ { i + 1 } )$ Otherwise, all terms along the chain would have identical images under $\Gamma ,$ which would imply $\Gamma ( \boldsymbol { \mu } ) = \Gamma ( \boldsymbol { \mu } ^ { \prime } )$ , a contradiction.

For the index �, the preceding estimate gives $\mathrm { d _ { T V } } \big ( \Gamma ( \mu ^ { i } ) , \Gamma ( \mu ^ { i + 1 } ) \big ) \geq \delta _ { \Gamma }$ . On the other hand, since Γ is Lipschitz with constant $L _ { c } ,$ , we have

$$
\mathrm { d _ { T V } } \left( \Gamma ( \mu ^ { i } ) , \Gamma ( \mu ^ { i + 1 } ) \right) \leq L _ { c } \mathrm { d _ { T V } } \left( \mu ^ { i } , \mu ^ { i + 1 } \right) < \delta _ { \Gamma } ,
$$

a contradiction. Thus, Γ cannot be Lipschitz continuous and consequently, Γ cannot be a contraction. □

## A<sub>ppe</sub>ndi<sub>x</sub> E<sub>.</sub> Pr<sub>oo</sub>f <sub>o</sub>f Th<sub>eo</sub>r<sub>e</sub>m 2

We now restate and provide a proof of Theorem 2 from Section 4.

Theorem 2 (Existence of entropy-regularized RA-MP-MFE). Recall the problem data from Section 2 and assume that $\mu _ { 0 } ^ { j } \in \mathscr { P } ( \mathsf { X } )$ are fixed for all $j \in [ M ]$ . Let Assumption 1 be in force; then, for every fixed $\varepsilon \in [ 0 , 1 ]$ and $\beta > 0 ,$ there exists an entropy-regularized RA-MP-MFE $( \pi _ { \beta } ^ { \star } , \mu _ { \beta } ^ { \star } ) \in \Pi \times \mathsf { M }$ in the sense of Definition 8. More precisely, there exists $\pi _ { \beta } ^ { \star }$ such that, with $\mu _ { \beta } ^ { \star } : = \mathcal { B } _ { \mathrm { p r o p } } ( \pi _ { \beta } ^ { \star } ) , f o r$ every $j \in [ M ]$

$$
\pi _ { \beta } ^ { j , \star } \in \underset { \pi ^ { j } \in \Pi ^ { j } } { \arg \operatorname* { m a x } } \underset { \widehat { \mu } ^ { - j } \in \mathcal { W } _ { j } ( \mu _ { \beta } ^ { \star } , \varepsilon ) } { \operatorname* { i n f } } \in \pi ^ { j } \left[ \sum _ { t = 0 } ^ { T } r _ { t } ^ { j } \big ( x _ { t } ^ { j } , \mu _ { t , \beta } ^ { j , \star } , \widehat { \mu _ { t } } ^ { - j } \big ) - \beta \sum _ { t = 0 } ^ { T - 1 } \log \pi _ { t } ^ { j } ( u _ { t } ^ { j } | x _ { t } ^ { j } ) \right] ,\tag{31}
$$

and in addition, i.e., $\mu _ { \beta } ^ { j , \star } = \mathcal { B } _ { \mathrm { p r o p } } ^ { j } ( \pi _ { \beta } ^ { j , \star } )$ for all $j \in [ M ]$

Proof. Our proof is a modification of the arguments presented in Theorem 1. We only verify the additional regularity introduced by the entropy regularization step, $\mathrm { i . e . , }$ , continuity and concavity.

To this end, recall $\mathsf { O } ^ { j } , \mathsf { O }$ , and $\xi ( o )$ from the definition of occupation measures and properties (2)–(3). For $\tilde { o } ^ { j } : = ( \tilde { \xi } ^ { j } , \tilde { \eta } ^ { j } ) \in 0 ^ { j }$ , define

$$
\tilde { \sigma } ^ { j } \mapsto \Psi _ { j , \beta } ( \tilde { \sigma } ^ { j } ) : = - \beta \sum _ { t = 0 } ^ { T - 1 } \sum _ { x \in \mathsf { X } } \sum _ { u \in \mathsf { U } } \tilde { \eta } _ { t } ^ { j } ( x , u ) \log \frac { \tilde { \eta } _ { t } ^ { j } ( x , u ) } { \tilde { \xi } _ { t } ^ { j } ( x ) } .\tag{32}
$$

Note that, when $\tilde { \xi } _ { t } ^ { j } ( x ) = 0 .$ , the occupation constraints imply $\tilde { \eta } _ { t } ^ { j } ( x , u ) = 0$ for every $u \in \mathsf { U } .$ , so the corresponding summand is defined to be zero. Also, define the entropyregularized robust occupation-measure payof by

$$
( \widetilde { \sigma } ^ { j } , o ) \mapsto \Phi _ { j , \beta } ( \widetilde { \sigma } ^ { j } ; o ) : = \operatorname* { i n f } _ { \widehat { \mu } ^ { - j } \in \mathcal { W } _ { j } ( \xi ( o ) , \varepsilon ) } \left[ \sum _ { t = 0 } ^ { T } \sum _ { x \in \mathbb { X } } \widetilde { \xi } _ { t } ^ { j } ( x ) r _ { t } ^ { j } ( x , \xi _ { t } ^ { j } , \widehat { \mu _ { t } } ^ { - j } ) + \Psi _ { j , \beta } ( \widetilde { o } ^ { j } ) \right] .\tag{33}
$$

Since $\Psi _ { j , \beta } ( \cdot )$ in (32) does not depend on $\widehat { \mu } ^ { - j }$ , we can write (33) as

$$
\Phi _ { j , \beta } ( \tilde { o } ^ { j } ; o ) = \Phi _ { j } ( \tilde { o } ^ { j } ; o ) + \Psi _ { j , \beta } ( \tilde { o } ^ { j } ) ,\tag{34}
$$

where $\Phi _ { j } ( \cdot )$ is the unregularized robust occupation-measure payof defined in (25).

In Theorem 1, we have established that the map $( \tilde { o } ^ { j } , o ) \mapsto \Phi _ { j } ( \tilde { o } ^ { j } ; o )$ is continuous on ${ \cal O } ^ { j } \times { \cal O } ,$ and, for fixed $o \in \mathsf { O }$ , the map $\tilde { o } ^ { j } \mapsto \Phi _ { j } ( \tilde { o } ^ { j } ; o )$ is concave. We now show that the same continuity and concavity properties hold for $\Psi _ { j , \beta } ( \cdot )$

C<sub>o</sub>ntin<sub>u</sub>it<sub>y o</sub>f $\Psi _ { j , \beta } ( \cdot )$ on $O ^ { j } { \mathrm { : } }$ : Indeed, on the region where $\tilde { \xi } _ { t } ^ { j } ( x ) > 0$ , this follows directly from continuity of the function $( a , b ) \mapsto a \log ( a / b )$ . We need to check the regularity when $\tilde { \xi } _ { t } ^ { j } ( x ) \to 0$ . Whenever $\tilde { \xi } _ { t } ^ { j } ( x ) > 0$ , define $\tilde { \pi } _ { t } ^ { j } ( u | x ) : = \tilde { \eta } _ { t } ^ { j } ( x , u ) / \tilde { \xi } _ { t } ^ { j } ( x )$ Then,

$$
0 \leq - \sum _ { u \in \mathsf { U } } \tilde { \eta } _ { t } ^ { j } ( x , u ) \log \frac { \tilde { \eta } _ { t } ^ { j } ( x , u ) } { \tilde { \xi } _ { t } ^ { j } ( x ) } = \tilde { \xi } _ { t } ^ { j } ( x ) \left( - \sum _ { u \in \mathsf { U } } \tilde { \pi } _ { t } ^ { j } ( u | x ) \log \tilde { \pi } _ { t } ^ { j } ( u | x ) \right) \leq \tilde { \xi } _ { t } ^ { j } ( x ) \log | \mathsf { U } | ,
$$

and thus, whenever $\tilde { \xi } _ { t } ^ { j } ( x )  0$ , the entropy contribution converges to zero. Hence, $\Psi _ { j , \beta } ( \cdot )$ is continuous on ${ \cal O } ^ { j }$

C<sub>o</sub>n<sub>cav</sub>it<sub>y o</sub>f $\Psi _ { j , \beta } ( \cdot )$ on $\bigcirc ^ { j } { \mathsf { \colon } }$ For each fixed � and $x ,$ define

$$
\mathfrak { D } _ { \mathsf { U } } : = \bigg \{ ( \xi , \zeta ) \in \mathbb { R } _ { + } \times \mathbb { R } _ { + } ^ { | \mathsf { U } | } \bigg | \sum _ { u \in \mathsf { U } } \zeta ( u ) = \xi \bigg \} .
$$

Then, the map $\begin{array} { r } { \mathfrak { D } _ { \mathsf { U } } \ni \left( \xi , \zeta \right) \mapsto - \sum _ { u \in \mathsf { U } } \zeta ( u ) \log \frac { \zeta ( u ) } { \xi } \ \in \ \mathbb { R } } \end{array}$ is concave in the joint   
variable $( \xi , \zeta )$ . Therefore $\Psi _ { j , \beta } ( \cdot )$ , being a finite sum of concave functions, is concave $O ^ { j } .$ 6   
on

Thus, it follows from (34) that $\Phi _ { j , \beta } ( \cdot )$ is continuous on ${ \cal O } ^ { j } \times { \cal O }$ , and that, for every fixed $o \in \mathsf { O }$ , the map $\tilde { o } ^ { j } \mapsto \Phi _ { j , \beta } ( \tilde { o } ^ { j } ; o )$ is concave. In a similar manner as before, define the entropy-regularized best-response correspondence

$$
o \mapsto \mathsf { B r } _ { j , \beta } ( o ) : = \underset { \tilde { o } ^ { j } \in \mathsf { O } ^ { j } } { \arg \operatorname* { m a x } } \Phi _ { j , \beta } ( \tilde { o } ^ { j } ; o ) .\tag{35}
$$

Since ${ \mathsf { O } } ^ { j }$ is compact and $\Phi _ { j , \beta } ( \cdot )$ is continuous, Weierstrass’ theorem implies that $\mathsf { B r } _ { j , \beta } ( o )$ is nonempty and compact. Since $\Phi _ { j , \beta } ( \cdot ; o )$ is concave and ${ \mathsf { O } } ^ { j }$ is convex, $\mathsf { B r } _ { j , \beta } ( o )$ is convex. Moreover, since the feasible set is fixed and compact and $\Phi _ { j , \beta } ( \cdot )$ is jointly continuous, Berge’s maximum principle [AB06, Chapter 17, Theorem 17.31] implies that $\mathsf { B r } _ { j , \beta } ( \cdot )$ is upper hemicontinuous.

Then, the correspondence ${ \mathsf { B r } } _ { \beta } : 0  0 .$ , defined by $\mathsf { B r } _ { \beta } ( o ) : = \Pi _ { i = 1 } ^ { M } \mathsf { B r } _ { j , \beta } ( o )$ has nonempty, compact, convex values and is upper hemicontinuous. Since O is nonempty, compact, and convex, Kakutani’s fixed-point theorem [AB06, Chapter 17, Corollary 17.55] implies that there exists $o ^ { \star } \in \mathsf { O }$ such that $o ^ { \star } \in \mathsf { B r } _ { \beta } ( o ^ { \star } )$ 7

Finally, reconstruct Markov policies from the fixed point $o ^ { \star }$ as in the proof of Theorem 1. The policy–occupation-measure equivalence gives the consistency condition $\mu ^ { j , \star } = \mathcal { B } _ { \mathrm { p r o p } } ^ { j } ( \pi ^ { j , \star } )$ , while $o ^ { \star } \in \mathsf { B r } _ { \beta } ( o ^ { \star } )$ gives the entropy-regularized risk-averse optimality inequality in (31). Our proof is complete. □

Remark 6. In fact the best response map is a singleton and the optimizer is unique.   
Thisfollowsfrom Lemma 2 ahead.

An approximation result: Our next (additional) result concerns the approximation of the RA-MFE under entropy regularization.

Proposition 3 (Approximation and vanishing-regularization limit). Suppose that the assumptions of Theorem 2 hold. $F i x \varepsilon \in [ 0 , 1 ]$ , and for any � $: = ( \pi ^ { 1 } , \dots , \pi ^ { M } )$ Π, let $\mu ^ { \pi } : = \mathcal { B } _ { \mathrm { p r o p } } ( \pi )$ , and recall the definition of (population-wise) risk-averse exploitability by

$$
\phi _ { \mathsf { R A } } ^ { j } ( \pi ) : = \operatorname* { s u p } _ { \tilde { \pi } ^ { j } \in \Pi ^ { j } } \bar { \mathcal { J } } _ { j } ( \tilde { \pi } ^ { j } ; \mu ^ { \pi } ) - \bar { \mathcal { J } } _ { j } ( \pi ^ { j } ; \mu ^ { \pi } ) \quad f o r j \in [ M ] ,\tag{36}
$$

and the total risk-averse exploitability by

$$
\phi _ { \mathsf { R A } } ( \pi ) : = \sum _ { j = 1 } ^ { M } \phi _ { \mathsf { R A } } ^ { j } ( \pi ) .\tag{37}
$$

Then, thefollowing assertions hold:

(3-a) For every $\beta > 0 ,$ let $( \pi _ { \beta } ^ { \star } , \mu _ { \beta } ^ { \star } ) \in \Pi \times \mathsf { M }$ be an entropy-regularized RA-MP-MFE in the sense of Definition 8. Then, for every $j \in [ M ]$ , we have the bound

$$
\phi _ { \mathsf { R A } } ^ { j } ( \pi _ { \beta } ^ { \star } ) \leq \beta T \log | \mathsf { U } | ,\tag{38}
$$

and consequently, �<sub>RA</sub> $( \pi _ { \beta } ^ { \star } ) \leq M \beta T$ log |U|. In particular, every entropyregularized RA-MP-MFE is a (population-wise) $\beta T$ log |U|-approximate RA-MP-MFE ofthe unregularized problem.

(3-b) Let $\{ \beta _ { n } \} _ { n \in \mathbb { N } } \subset ( 0 , + \infty )$ be such that $\beta _ { n } \to 0$ as $n  + \infty .$ For each $n \in \mathbb N ,$ , let $( \pi _ { \beta _ { n } } ^ { \star } , \mu _ { \beta _ { n } } ^ { \star } )$ ∈ Π × M be an entropy-regularized RA-MP-MFE, and let $o _ { \beta _ { n } } ^ { \star } \in \mathsf { O }$ be an occupation-measure representation induced by $\pi _ { \beta _ { n } } ^ { \star } .$ If $o ^ { \star } \in \dot { \mathsf { O } }$ is an accumulation point of the sequence $\{ o _ { \beta _ { n } } ^ { \star } \} _ { n \in \mathbb { N } }$ then $o ^ { \star }$ induces a policy $\pi ^ { \star } \in$ Π and a mean-field flow $\mu ^ { \star } : = \xi ( o ^ { \star } )$ such that $( \pi ^ { \star } , \mu ^ { \star } )$ is a RA-MP-MFE in the sense of Definition 6.

Remark 7. Theorem 3 highlights a natural trade-offor the choice ofthe parameter � used for regularization. As � tends to 0, the (entropy-regularized) equilibrium becomes an increasingly accurate approximation of the original risk-averse equilibrium, with the approximation error vanishing at a rate $\Theta ( \beta )$ . On the other hand, stronger entropy regularization improves the stability of the best-response mapping and, as shown below, facilitates contractivity. Thus, smaller values of � favor fidelity to the unregularized game, whereas larger values favor contractive behavior and computational convergence. This is common phenomenon in entropy regularized mean-field games [CK21]. This is also visible in the multi-population chasing example presented in Section 6.

Proof of Proposition 3. We start with the proof of the first assertion, i.e., (3-a). To this end, fix $\beta > 0$ , and let $( \pi _ { \beta } ^ { \star } , \mu _ { \beta } ^ { \star } ) \in \Pi \times \mathsf { M }$ be an entropy-regularized RA-MP-MFE. For each $j \in [ M ]$ and $\pi ^ { j } \in \Pi ^ { j }$ , define the entropy

$$
\pi ^ { j } \mapsto \mathsf { H } _ { j } ( \pi ^ { j } ) : = \mathsf { E } ^ { \pi ^ { j } } \left[ - \sum _ { t = 0 } ^ { T - 1 } \log \pi _ { t } ^ { j } ( u _ { t } ^ { j } | x _ { t } ^ { j } ) \right] .
$$

Since the Shannon entropy of a probability vector on U is bounded between 0 and log |U|, it follows that

$$
0 \leq \mathsf { H } _ { j } ( \pi ^ { j } ) \leq T \log | \mathsf { U } | \quad \mathrm { f o r e v e r y } \pi ^ { j } \in \Pi ^ { j } .\tag{39}
$$

By Definition 8, for every $\tilde { \pi } ^ { j } \in \Pi ^ { j }$ , we have $\bar { \mathcal { J } } _ { j , \beta } ( \pi _ { \beta } ^ { j , \star } ; \mu _ { \beta } ^ { \star } ) \ge \bar { \mathcal { J } } _ { j , \beta } ( \tilde { \pi } ^ { j } ; \mu _ { \beta } ^ { \star } )$ . Since the entropy term does not depend on $\widehat { \mu } ^ { - j }$ , we have $\bar { \mathcal { J } } _ { j , \beta } ( \pi ^ { j } ; \mu ) = \bar { \mathcal { J } } _ { j } ( \pi ^ { j } ; \mu ) \dot { + } \beta { \sf H } _ { j } ( \pi ^ { j } )$ Therefore, for every $\tilde { \pi } ^ { j } \in \Pi ^ { j }$ , we have the bound

$$
\bar { \mathcal { J } } _ { j } ( \pi _ { \beta } ^ { j , \star } ; \mu _ { \beta } ^ { \star } ) + \beta \mathsf { H } _ { j } ( \pi _ { \beta } ^ { j , \star } ) \geq \bar { \mathcal { J } } _ { j } ( \tilde { \pi } ^ { j } ; \mu _ { \beta } ^ { \star } ) + \beta \mathsf { H } _ { j } ( \tilde { \pi } ^ { j } ) .
$$

Rearranging and using (39), we obtain

$$
\bar { \xi } _ { j } ( \tilde { \pi } ^ { j } ; \mu _ { \beta } ^ { \star } ) - \bar { \mathcal { J } } _ { j } ( \pi _ { \beta } ^ { j , \star } ; \mu _ { \beta } ^ { \star } ) \leq \beta \left[ \mathsf { H } _ { j } ( \pi _ { \beta } ^ { j , \star } ) - \mathsf { H } _ { j } ( \tilde { \pi } ^ { j } ) \right] \leq \beta T \log | \mathsf { U } | .
$$

Taking the supremum over $\tilde { \pi } ^ { j } \in \Pi ^ { j }$ , and using $\mu _ { \beta } ^ { \star } = \mathcal { B } _ { \mathrm { p r o p } } ( \pi _ { \beta } ^ { \star } )$ , yields $\phi _ { \mathsf { R A } } ^ { j } ( \pi _ { \beta } ^ { \star } ) \leq$ $\beta T$ log |U|. This proves (38), and summing over $j \in [ M ]$ gives �<sub>RA</sub> $( \pi _ { \beta } ^ { \star } ) \leq M \beta T$ log |U|; as desired.

We now prove (3-b). Let $o ^ { \star } \in \mathsf { O }$ be an accumulation point of $\{ o _ { \beta _ { n } } ^ { \star } \} _ { n \in \mathbb { N } }$ . Passing to a subsequence if necessary (and for convinience, not relabeling) suppose that $o _ { \beta _ { n } } ^ { \star } $ $o ^ { \star }$ as $n  + \infty$ . We write $o _ { \beta _ { n } } ^ { \star } : = ( o _ { \beta _ { n } } ^ { 1 , \star } , \ldots , o _ { \beta _ { n } } ^ { M , \star } )$ and $o ^ { \star } : = ( o ^ { 1 , \star } , \ldots , o ^ { M , \star } )$ Fixing $j \in [ M ]$ and $\tilde { o } ^ { j } \in \mathsf { O } ^ { j }$ and noting the fact that $o _ { \beta _ { n } } ^ { \star }$ is an entropy-regularized fixed point in occupation measures, we have $\Phi _ { j , \beta _ { n } } ( o _ { \beta _ { n } } ^ { j , \star } ; o _ { \beta _ { n } } ^ { \star } ) \ \geq \ \Phi _ { j , \beta _ { n } } ( \tilde { o } ^ { j } ; o _ { \beta _ { n } } ^ { \star } )$ Using the fact that $\Phi _ { j , \beta _ { n } } = \Phi _ { j } + \Psi _ { j , \beta _ { n } }$ , we have

$$
\begin{array} { r l } & { \Phi _ { j } ( \tilde { o } ^ { j } ; o _ { \beta _ { n } } ^ { \star } ) - \Phi _ { j } ( o _ { \beta _ { n } } ^ { j , \star } ; o _ { \beta _ { n } } ^ { \star } ) \le \Psi _ { j , \beta _ { n } } ( o _ { \beta _ { n } } ^ { j , \star } ) - \Psi _ { j , \beta _ { n } } ( \tilde { o } ^ { j } ) . } \end{array}
$$

By the entropy bound in occupation-measure form, $0 \leq \Psi _ { j , \beta _ { n } } \mathopen { } \mathclose \bgroup \left( o ^ { j } \aftergroup \egroup \right) \leq \beta _ { n } T$ log |U| for every $o ^ { j } \in \bigcirc ^ { j }$ . Hence,

$$
\begin{array} { r } { \Phi _ { j } ( \tilde { o } ^ { j } ; o _ { \beta _ { n } } ^ { \star } ) - \Phi _ { j } ( o _ { \beta _ { n } } ^ { j , \star } ; o _ { \beta _ { n } } ^ { \star } ) \le \beta _ { n } T \log | \mathsf { U } | . } \end{array}\tag{40}
$$

Letting $n $ +∞ in (40), the right-hand side converges to zero. Moreover, by the proof of Theorem 1, the map $( \tilde { o } ^ { j } , o ) \mapsto \Phi _ { i } ( \tilde { o } ^ { j } ; o )$ is continuous on ${ \cal O } ^ { j } \times { \cal O }$ . Therefore, $\Phi _ { j } ( \tilde { o } ^ { j } ; o ^ { \star } ) - \Phi _ { j } ( o ^ { j , \star } ; o ^ { \star } ) \le 0$ . Since $\tilde { o } ^ { j } \in \bigcirc ^ { j }$ was arbitrary, it follows that

$$
o ^ { j , \star } \in \underset { \tilde { o } ^ { j } \in \mathsf { O } ^ { j } } { \arg \operatorname* { m a x } } \Phi _ { j } \big ( \tilde { o } ^ { j } ; o ^ { \star } \big ) .
$$

Since $j \in [ M ]$ was arbitrary, $o ^ { \star }$ satisfies the unregularized risk-averse optimality condition for every population.

Finally, since $o ^ { \star } \in \mathsf { O }$ , the policy–occupation-measure equivalence used in the proof of Theorem 1 gives a policy $\pi ^ { \star } \in \Pi$ inducing $o ^ { \star }$ , and the corresponding MFflow is $\mu ^ { \star } : = \xi ( o ^ { \star } )$ . The occupation constraints give $\mu ^ { j , \star } = \mathcal { B } _ { \mathrm { p r o p } } ^ { j } ( \pi ^ { j , \star } )$ for every $j \in [ M ]$ , while the optimality condition above gives the risk-averse best-response condition. Hence $( \pi ^ { \star } , \mu ^ { \star } )$ is a RA-MP-MFE in the sense of Definition 6. Our proof is complete. □

## A<sub>ppe</sub>ndi<sub>x</sub> F<sub>.</sub> Pr<sub>oo</sub>f<sub>s</sub> <sub>o</sub>f Th<sub>eo</sub>r<sub>e</sub>m<sub>s</sub> 3 <sub>a</sub>nd 4

In Theorem 2 we showed that the mapping $\Psi _ { j , \beta } ( \cdot )$ is concave. We start by showing a stronger result, namely, that $\Psi _ { j , \beta } ( \cdot )$ is strongly concave; this is one of the main ingredients for showing contractivity in the sequel.

Lemma 2 (Strong concavity of $\Phi _ { j , \beta } ( \cdot , o ) )$ . Fix $j \in [ M ] , T \in \mathbb { N } ,$ and $\beta > 0$ . For   
$o ^ { j } = ( \xi ^ { j } , \eta ^ { j } )$ and $\bar { o } ^ { j } = ( \bar { \xi } ^ { j } , \bar { \eta } ^ { j } )$ in ${ \mathsf { O } } ^ { j }$ , recall the norm   
∥�<sup>�</sup> − �¯<sup>�</sup> ∥occ := maxn max ∥�<sup>�</sup><sub>�</sub> − �¯<sup>�</sup><sub>�</sub> ∥1, max ∥�<sup>�</sup><sub>�</sub> − �¯<sup>�</sup><sub>�</sub> ∥1o. 0≤�≤� 0≤�<�   
Consider the mapping $R _ { j } : { \mathsf { O } } ^ { j } \to \mathbb { R }$ given by   
(41) O<sup>�</sup> ∋ �<sup>�</sup> ↦→ �<sub>�</sub> (�<sup>�</sup> ) := <sup>�</sup>∑︁<sup>−1</sup> ∑︁ ∑︁ �<sup>�</sup><sub>�</sub> (�, �) log <sup>��� (�,</sup> <sup>�)</sup> ∈ ℝ.   
�=0 �∈X �∈U �<sup>�</sup><sub>�</sub> (�)   
Then $R _ { j }$ is 1-strongly convex on ${ \mathsf { O } } ^ { j }$ with respect to ∥ · ∥<sub>occ</sub>, i.e., for every   
$o ^ { j } , \bar { o } ^ { j } \in \mathsf { O } ^ { j }$ and $\lambda \in [ 0 , 1 ]$   
(42)   
�<sub>�</sub>  �� <sup>�</sup> + (1 − �)�¯<sup>�</sup>  ≤ ��<sub>�</sub> (�<sup>�</sup> ) + (1 − �)�<sub>�</sub> (�¯<sup>�</sup> ) − <sup>�(1</sup> <sup>−</sup> <sup>�)</sup> <sub>∥�</sub>� <sub>−</sub> <sub>�¯</sub>� <sub>∥</sub>2<sub>occ.</sub>   
2   
Consequently, we have the following assertions:   
(2-a) The mapping $\Psi _ { j , \beta } ( \cdot ) ~ = ~ - \beta R _ { j } ( \cdot )$ in (32) is �-strongly concave on ${ \mathsf { O } } ^ { j }$   
with respect $t o \parallel \cdot \parallel _ { \mathrm { o c c } }$ Moreover, for every fixed $o \in { \mathsf { O } } ,$ the map $\tilde { o } ^ { j } \mapsto$   
$\Phi _ { j , \beta } ( \tilde { o } ^ { j } ; o )$ in (34) is �-strongly concave on ${ \mathsf { O } } ^ { j }$   
(2-b) The best response $\mathsf { B r } _ { j , \beta } ( o )$ defined in (35) is a singleton for every $o \in { \mathsf { O } } .$

Proof. The proof is inspired by a nice argument presented in [NJG17] relating Bregman and Kullback–Leibler divergences along with some standard facts from convex optimization [Bec17].

Fix $j \in [ M ]$ . We will simplify notation and denote $\Theta = 0 ^ { j }$ . Also, we suppress the population subcript � by writing $o = ( \xi , \eta )$ , and $\bar { o } : = ( \bar { \xi } , \bar { \eta } )$ , and $R : = R _ { i }$ . We first establish strong convexity of $o \mapsto R ( o )$ on the relative interior ${ \dot { \operatorname { r i } } } ( \odot )$ (the boundary case will be handled later). To this end, take $o = ( \xi , \eta )$ and $\bar { o } = ( \bar { \xi } , \bar { \eta } )$ in $\operatorname { r i } ( \mathbb { O } )$ . For each fixed � and �, define $\begin{array} { r } { ( \xi , \zeta ) \mapsto g _ { 0 } ( \xi , \zeta ) : = \sum _ { u \in \mathsf { U } } \zeta ( u ) \log \frac { \zeta ( u ) } { \xi } } \end{array}$ . For $\xi > 0$ and $\zeta ( u ) > 0$ , we have that

$$
\frac { \partial g _ { 0 } } { \partial \zeta ( u ) } = \log \frac { \zeta ( u ) } { \xi } + 1 \quad \mathrm { a n d } \quad \frac { \partial g _ { 0 } } { \partial \xi } = - \frac { \sum _ { u \in \mathsf { U } } \zeta ( u ) } { \xi } = - 1 ,
$$

where in the last equality we used the fact that $\begin{array} { r } { \sum _ { u \in \mathsf { U } } \zeta ( u ) = \xi } \end{array}$ (which is the occupation constraint). We consider the Bregman divergence of �, which is given by

$$
( o , \bar { o } ) \mapsto \mathsf { D } _ { R } ^ { \mathsf { B r e g } } ( o , \bar { o } ) : = R ( o ) - R ( \bar { o } ) - \langle \nabla R ( \bar { o } ) , o - \bar { o } \rangle .
$$

Substituting the preceding derivatives gives

$$
\begin{array} { r l } & { \mathsf { D } _ { R } ^ { \mathsf { B r e g } } ( o , \bar { o } ) = R ( o ) - R ( \bar { o } ) + \displaystyle \sum _ { t = 0 } ^ { T - 1 } \sum _ { x \in \mathsf { X } } \left( \xi _ { t } ( x ) - \bar { \xi } _ { t } ( x ) \right) - } \\ & { \qquad \displaystyle \sum _ { t = 0 } ^ { T - 1 } \sum _ { x \in \mathsf { X } } \sum _ { u \in \mathsf { U } } \left( \log \frac { \bar { \eta } _ { t } ( x , u ) } { \bar { \xi } _ { t } ( x ) } + 1 \right) \left( \eta _ { t } ( x , u ) - \bar { \eta } _ { t } ( x , u ) \right) . } \end{array}
$$

Using the occupation constraints $\begin{array} { r } { \sum _ { u \in \mathsf { U } } \big ( \eta _ { t } ( x , u ) - \bar { \eta } _ { t } ( x , u ) \big ) \ = \ \xi _ { t } ( x ) - \bar { \xi } _ { t } ( x ) } \end{array}$ , and simplifying, we get

$$
\begin{array} { l } { { \displaystyle \mathsf { D } _ { R } ^ { \mathsf { B r e g } } ( o , \bar { o } ) = R ( o ) - R ( \bar { o } ) - \sum _ { t = 0 } ^ { T - 1 } \sum _ { x \in \mathsf { X } } \sum _ { u \in \mathsf { U } } \log \frac { \bar { \eta } _ { t } ( x , u ) } { \bar { \xi } _ { t } ( x ) } \big ( \eta _ { t } ( x , u ) - \bar { \eta } _ { t } ( x , u ) \big ) } } \\ { { \displaystyle \qquad = \sum _ { t = 0 } ^ { T - 1 } \sum _ { x \in \mathsf { X } } \sum _ { u \in \mathsf { U } } \eta _ { t } ( x , u ) \log \frac { \eta _ { t } ( x , u ) / \xi _ { t } ( x ) } { \bar { \eta } _ { t } ( x , u ) / \bar { \xi } _ { t } ( x ) } . } } \end{array}
$$

Let � and �¯ be the Markov policies induced by � and ${ \bar { o } } ,$ respectively, so that

$$
\pi _ { t } ( u | x ) = \frac { \eta _ { t } ( x , u ) } { \xi _ { t } ( x ) } \quad \mathrm { a n d } \quad \bar { \pi } _ { t } ( u | x ) = \frac { \bar { \eta } _ { t } ( x , u ) } { \bar { \xi } _ { t } ( x ) } .
$$

Since $\eta _ { t } ( x , u ) = \xi _ { t } ( x ) \pi _ { t } ( u | x )$ , we obtain

$$
\mathsf { D } _ { R } ^ { \mathsf { B r e g } } ( o , \bar { o } ) = \sum _ { t = 0 } ^ { T - 1 } \sum _ { x \in \mathsf { X } } \xi _ { t } ( x ) \mathsf { D } _ { \mathsf { K L } } \left( \pi _ { t } ( \cdot | x ) \parallel \bar { \pi } _ { t } ( \cdot | x ) \right) .\tag{43}
$$

We now express the right-hand side of (43) as a path-space relative entropy. Let $\Omega _ { \mathrm { p a t h } } : = \mathsf X ^ { \hat { T } + 1 } \times \mathsf U ^ { T }$ be the path space of finite trajectories, and we write a typical element of $\Omega _ { \mathrm { { p a t h } } }$ as $\omega = ( x _ { 0 } , u _ { 0 } , \ldots , u _ { T - 1 } , x _ { T } )$ . For all $\omega \in \Omega _ { \mathrm { p a t h } }$ , define the path laws induced by � and $\bar { o }$ by

$$
P _ { o } ( \omega ) : = \mu _ { 0 } ^ { j } ( x _ { 0 } ) \prod _ { t = 0 } ^ { T - 1 } \pi _ { t } ( u _ { t } | x _ { t } ) f _ { t } ^ { j } ( x _ { t + 1 } | x _ { t } , u _ { t } )
$$

and

$$
P _ { \bar { o } } ( \omega ) : = \mu _ { 0 } ^ { j } ( x _ { 0 } ) \prod _ { t = 0 } ^ { T - 1 } \bar { \pi } _ { t } ( u _ { t } | x _ { t } ) f _ { t } ^ { j } ( x _ { t + 1 } | x _ { t } , u _ { t } ) .
$$

By the policy–occupation-measure equivalence (see Step 2 in the proof of Theorem 2) and the flow-conservation constraints, the $x _ { t } .$ -marginals of $P _ { o }$ and $P _ { \bar { o } }$ are $\xi _ { t }$ and $\bar { \xi } _ { t }$ , respectively, while their $\left( { { x } _ { t } } , { { u } _ { t } } \right)$ -marginals are $\eta _ { t }$ and $\bar { \eta _ { t } }$ , respectively. Indeed, $P _ { o } ( x _ { t } = x , u _ { t } = u ) = \xi _ { t } ( x ) \pi _ { t } ( u | x ) = \eta _ { t } ( x , u )$ , and

$$
P _ { o } ( x _ { t + 1 } = y ) = \sum _ { x \in \mathsf { X } } \sum _ { u \in \mathsf { U } } \eta _ { t } ( x , u ) f _ { t } ^ { j } ( y | x , u ) = \xi _ { t + 1 } ( y ) ,
$$

with the analogous identities holding under $P _ { \bar { o } }$

Since $o , \bar { o } \in \mathrm { r i } ( \mathbb { O } )$ , the occupation measures � and $\bar { \eta }$ have the same support. Indeed, for each coordinate $( t , x , u )$ , either $\eta _ { t } ( x , u ) = 0$ for every $o \in \mathbb { O } , \operatorname { o r } \eta _ { t } ( x , u ) > 0$ for every $o \in { \mathrm { r i } } ( \mathbb { O } )$ . Let us verify that $P _ { o } \ll P _ { \bar { o } } . ^ { 8 }$ Let $\omega : = ( x _ { 0 } , u _ { 0 } , \ldots , u _ { T - 1 } , x _ { T } ) \in$ $\Omega _ { \mathrm { { p a t h } } }$ be such that $P _ { o } ( \omega ) > 0$ Then, for every $t \in \{ 0 , \ldots , T - 1 \} , \eta _ { t } ( x _ { t } , u _ { t } ) \ =$ $\bar { P _ { o } } ( X _ { t } = x _ { t } , U _ { t } = u _ { t } ) > 0$ . Since $\eta$ and $\bar { \eta }$ have the same support, it follows that $\bar { \eta } _ { t } ( x _ { t } , u _ { t } ) > 0$ . Consequently,

$$
\bar { \xi } _ { t } ( x _ { t } ) = \sum _ { u \in \mathsf { U } } \bar { \eta } _ { t } ( x _ { t } , u ) > 0 \quad \mathrm { a n d } \quad \bar { \pi } _ { t } ( u _ { t } \mid x _ { t } ) = \frac { \bar { \eta } _ { t } ( x _ { t } , u _ { t } ) } { \bar { \xi } _ { t } ( x _ { t } ) } > 0 .
$$

Also, $P _ { o } ( \omega ) > 0$ implies $\mu _ { 0 } ^ { j } ( x _ { 0 } ) > 0$ and $f _ { t } ^ { j } ( x _ { t + 1 } \mid x _ { t } , u _ { t } ) > 0$ for every �. Since $P _ { o }$ and $P _ { \bar { o } }$ have the same initial distribution and transition kernels, every factor defining $P _ { \bar { o } } ( \omega )$ is therefore strictly positive. Hence, $P _ { o } ( \omega ) > 0$ implies that $P _ { \bar { o } } ( \omega ) > 0$ , and thus $P _ { o } \ll P _ { \bar { o } }$ . Therefore, for every path in the support of $P _ { o }$

$$
\frac { P _ { o } ( x _ { 0 } , u _ { 0 } , \dots , u _ { T - 1 } , x _ { T } ) } { P _ { \bar { o } } ( x _ { 0 } , u _ { 0 } , \dots , u _ { T - 1 } , x _ { T } ) } = \prod _ { t = 0 } ^ { T - 1 } \frac { \pi _ { t } \left( u _ { t } | x _ { t } \right) } { \bar { \pi } _ { t } \left( u _ { t } | x _ { t } \right) } .
$$

It follows that

$$
\begin{array} { l } { \displaystyle \mathsf { D } _ { \mathsf { K L } } ( P _ { o } \| P _ { \bar { o } } ) = \mathsf { E } ^ { P _ { o } } \left[ \log \frac { P _ { o } } { P _ { \bar { o } } } \right] = \sum _ { t = 0 } ^ { T - 1 } \mathsf { E } ^ { P _ { o } } \left[ \log \frac { \pi _ { t } ( U _ { t } | X _ { t } ) } { \bar { \pi } _ { t } ( U _ { t } | X _ { t } ) } \right] } \\ { \displaystyle = \sum _ { t = 0 } ^ { T - 1 } \sum _ { x \in \mathsf { X } } \sum _ { u \in \mathsf { U } } \eta _ { t } ( x , u ) \log \frac { \pi _ { t } ( u | x ) } { \bar { \pi } _ { t } ( u | x ) } } \\ { \displaystyle = \sum _ { t = 0 } ^ { T - 1 } \sum _ { x \in \mathsf { X } } \xi _ { t } ( x ) \mathsf { D } _ { \mathsf { K L } } \big ( \pi _ { t } ( \cdot | x ) \| \bar { \pi } _ { t } ( \cdot | x ) \big ) . } \end{array}
$$

Comparing the preceding identity with (43) yields

$$
\mathsf { D } _ { R } ^ { \mathsf { B r e g } } ( o , \bar { o } ) = \mathsf { D } _ { \mathsf { K L } } ( P _ { o } \| P _ { \bar { o } } ) .\tag{44}
$$

Every $\xi _ { t }$ and $\eta _ { t }$ is a marginal of $P _ { o }$ , and every $\bar { \xi } _ { t }$ and $\bar { \eta _ { t } }$ is the corresponding marginal of $P _ { \bar { o } }$ . Since marginalization cannot increase the $\ell _ { 1 }$ -distance, for every $t = 0 , \ldots , T .$

$$
\begin{array} { r l } & { \| \xi _ { t } - \bar { \xi } _ { t } \| _ { 1 } = \displaystyle \sum _ { x \in \mathsf { X } } \left. \displaystyle \sum _ { \omega : X _ { t } ( \omega ) = x } \big ( P _ { o } ( \omega ) - P _ { \bar { o } } ( \omega ) \big ) \right. } \\ & { \qquad \le \displaystyle \sum _ { \omega \in \Omega _ { \mathrm { p a t h } } } \vert P _ { o } ( \omega ) - P _ { \bar { o } } ( \omega ) \vert = \| P _ { o } - P _ { \bar { o } } \| _ { 1 } . } \end{array}
$$

Similarly, for every $t = 0 , \ldots , T - 1 , \| \eta _ { t } - \bar { \eta } _ { t } \| _ { 1 } \leq \| P _ { o } - P _ { \bar { o } } \| _ { 1 }$ . Therefore,

$$
\| o - \bar { o } \| _ { \mathrm { o c c } } \leq \| P _ { o } - P _ { \bar { o } } \| _ { 1 } .\tag{45}
$$

By Pinsker’s inequality [Can23, Section 1, Lemma 2] and (44)–(45),

$$
\mathsf { D } _ { R } ^ { \mathsf { B r e g } } ( o , \bar { o } ) = \mathsf { D } _ { \mathsf { K L } } ( P _ { o } \| P _ { \bar { o } } ) \geq \frac { 1 } { 2 } \| P _ { o } - P _ { \bar { o } } \| _ { 1 } ^ { 2 } \geq \frac { 1 } { 2 } \| o - \bar { o } \| _ { \mathrm { o c } } ^ { 2 } .\tag{46}
$$

Thus, for every $o , \bar { o } \in \mathop { \Gamma } \mathrm { i } ( \mathbb { O } )$ ,

$$
R ( o ) \geq R ( \bar { o } ) + \langle \nabla R ( \bar { o } ) , o - \bar { o } \rangle + \frac { 1 } { 2 } \| o - \bar { o } \| _ { \mathrm { o c c } } ^ { 2 } .\tag{47}
$$

From the equivalence between the first-order and inequality characterizations ofstrong convexity [Bec17, Theorem 5.24], inequality (47) implies that, for every $\lambda \in [ 0 , 1 ]$ and every $o , \bar { o } \in \mathop { \Gamma } \mathrm { i } ( \mathbb { O } )$ ,

$$
R \big ( \lambda o + ( 1 - \lambda ) \bar { o } \big ) \leq \lambda R ( o ) + ( 1 - \lambda ) R ( \bar { o } ) - \frac { \lambda ( 1 - \lambda ) } { 2 } \| o - \bar { o } \| _ { \mathrm { o c } } ^ { 2 } .\tag{48}
$$

We now extend the result to the boundary of O. Choose $o ^ { \circ } \in \operatorname { r i } ( \mathbb { O } )$ . For arbitrary $o , \bar { o } \in \mathbb { O }$ and $\delta \in [ 0 , 1 ]$ , define

$$
o _ { \delta } : = ( 1 - \delta ) o + \delta o ^ { \circ } \quad \mathrm { a n d } \quad \bar { o } _ { \delta } : = ( 1 - \delta ) \bar { o } + \delta o ^ { \circ } .
$$

Then, $o _ { \delta } , \bar { o } _ { \delta } \in \mathrm { r i } ( \mathbb { O } )$ , and hence

$$
R \big ( \lambda o _ { \delta } + ( 1 - \lambda ) \bar { \sigma } _ { \delta } \big ) \leq \lambda R ( o _ { \delta } ) + ( 1 - \lambda ) R ( \bar { o } _ { \delta } ) - \frac { \lambda ( 1 - \lambda ) } { 2 } \| o _ { \delta } - \bar { o } _ { \delta } \| _ { \mathrm { o c } } ^ { 2 } .
$$

Since $o _ { \delta } - \bar { o } _ { \delta } = ( 1 - \delta ) ( o - \bar { o } )$ , we have $\| o _ { \delta } - \bar { \sigma } _ { \delta } \| _ { \mathrm { o c c } } = ( 1 - \delta ) \| o - \bar { \sigma } \| _ { \mathrm { o c c } }$ . Letting $\delta  0 ^ { + }$ and using the continuity of $R ,$ we obtain

$$
R \big ( \lambda o + ( 1 - \lambda ) \bar { o } \big ) \leq \lambda R ( o ) + ( 1 - \lambda ) R ( \bar { o } ) - \frac { \lambda ( 1 - \lambda ) } { 2 } \| o - \bar { o } \| _ { \mathrm { o c } } ^ { 2 } .
$$

Therefore, $R _ { j }$ is 1-strongly convex on ${ \mathsf { O } } ^ { j }$ with respect to $\| \cdot \| _ { \mathrm { o c c } }$ . Moreover, $\Psi _ { j , \beta } =$ $- \beta R _ { j } .$ , and thus, it follows that $\Psi _ { j , \beta }$ is $\beta \mathrm { \cdot }$ -strongly concave on ${ \mathsf { O } } ^ { j }$ . We also know that, for every fixed $o \in \bigcirc$ , the mapping $\tilde { o } ^ { j } \mapsto \Phi _ { i } ( \tilde { o } ^ { j } ; o )$ is concave, and $\Phi _ { j , \beta } ( \tilde { o } ^ { j } ; o ) =$ $\Phi _ { j } ( \tilde { o } ^ { j } ; o ) + \Psi _ { j , \beta } ( \tilde { o } ^ { j } )$ . Hence, $\Phi _ { j , \beta } ( \cdot ; o )$ is �-strongly concave on ${ \mathsf { O } } ^ { j }$

Finally, compactness of ${ \mathsf { O } } ^ { j }$ and continuity of $\Phi _ { j , \beta } ( \cdot ; o )$ imply that $\mathsf { B r } _ { j , \beta } ( o )$ is nonempty. Suppose that two distinct points $o _ { 1 } ^ { j } , o _ { 2 } ^ { j } \in \bar { \mathsf { O } } ^ { j }$ are both maximizers, with common maximum value �. Strong concavity with $\begin{array} { r } { \lambda = \frac { 1 } { 2 } } \end{array}$ gives

$$
\Phi _ { j , \beta } \left( \frac { \boldsymbol { o } _ { 1 } ^ { j } + \boldsymbol { o } _ { 2 } ^ { j } } { 2 } ; \boldsymbol { o } \right) \geq V + \frac { \beta } { 8 } \| \boldsymbol { o } _ { 1 } ^ { j } - \boldsymbol { o } _ { 2 } ^ { j } \| _ { \mathrm { o c c } } ^ { 2 } > V ,
$$

which contradicts maximality. Therefore, $\mathsf { B r } _ { j , \beta } ( o )$ is a singleton for every $o \in \mathsf { O }$ Our proof is complete. □

We restate Theorem 3 in detail and provide a detailed proof.

Theorem 3 (Contractivity of BR mapping when $\varepsilon = 1 )$ . Let $\varepsilon = 1$ , and fix $\beta > 0 .$ Suppose that Assumption 2 holds and let $q _ { \beta } : = \operatorname* { m a x } _ { j \in [ M ] } L _ { j } / \beta$ . Recall that, for $o \in { \mathsf { O } } ,$ , we have

$$
\mathsf { B r } _ { j , \beta } ( o ) : = \underset { \widetilde { o } ^ { j } \in \mathsf { O } ^ { j } } { \arg \operatorname* { m a x } } \Phi _ { j , \beta } ( \widetilde { o } ^ { j } ; o ) ,
$$

and define $\mathsf { B r } _ { \beta } ( o ) : = \left( \mathsf { B r } _ { 1 , \beta } ( o ) , \hdots , \mathsf { B r } _ { M , \beta } ( o ) \right)$ . Then ,

(3-a) For every $o , \bar { o } \in \mathrm { O }$ and $j \in [ M ]$ , the following holds

$$
\left\| \mathsf { B r } _ { j , \beta } ( o ) - \mathsf { B r } _ { j , \beta } ( \bar { o } ) \right\| _ { \mathrm { o c c } } \leq \frac { L _ { j } } { \beta } \left\| o ^ { j } - \bar { o } ^ { j } \right\| _ { \mathrm { o c c } } .
$$

(3-b) Consequently, with $\begin{array} { r } { \| o - \bar { o } \| _ { \mathrm { o c c , \infty } } : = \operatorname* { m a x } _ { j \in [ M ] } \| o ^ { j } - \bar { o } ^ { j } \| _ { \mathrm { o c c } } , } \end{array}$ we have the estimate $\begin{array} { r } { \left\| \mathsf { B r } _ { \beta } ( o ) - \mathsf { B r } _ { \beta } ( \bar { o } ) \right\| _ { \mathrm { o c c . } , \infty } \le q _ { \beta } \| o - \bar { o } \| _ { \mathrm { o c c . } , \infty } } \end{array}$ Therefore, if $\beta >$ $\operatorname* { m a x } _ { j \in [ M ] } L _ { j }$ , then $\mathsf { B r } _ { \beta }$ is a contraction on O. Consequently, for every feasible initial flow $\mu _ { 0 } { \mathrm { : } }$ , the iteration $\mu ( n + 1 ) = \Gamma _ { \beta } ( \mu ( n ) )$ converges to the unique equilibrium mean-field flow $\mu _ { \beta } ^ { \star }$ in Definition 8.

Proof. Fix $j \in [ M ]$ and $o , \bar { o } \in \mathrm { O }$ and let us write $o ^ { j } : = ( \xi ^ { j } , \eta ^ { j } )$ and $\bar { o } ^ { j } : = 1$ $( \bar { \xi } ^ { j } , \bar { \eta } ^ { j } )$ . From Lemma 2, the mapping $\tilde { o } ^ { j } \mapsto \Phi _ { j , \beta } ( \tilde { o } ^ { j } ; o )$ is �-strongly concave on ${ \mathsf { O } } ^ { j }$ with respect to $\| \cdot \| _ { \mathrm { o c c } }$ , and $\mathsf { B r } _ { j , \beta } ( o )$ is a singleton for every $o \in \mathsf { O }$ . Thus, we may identify $\mathsf { B r } _ { j , \beta } ( o )$ with its unique element and define

$$
z ^ { j } : = \mathsf { B r } _ { j , \beta } ( o ) \quad \mathrm { a n d } \quad \bar { z } ^ { j } : = \mathsf { B r } _ { j , \beta } ( \bar { o } ) ,
$$

and $z ^ { j } : = ( n ^ { j } , \zeta ^ { j } )$ and $\bar { z } ^ { j } : = ( \bar { n } ^ { j } , \bar { \zeta } ^ { j } ) . ^ { 9 }$

Recall the expression of the ambiguity set in (27). Since $\operatorname { d } _ { \mathrm { T V } } \big ( \nu , \mu \big ) \leq 1$ for every $\nu , \mu \in \mathcal { P } ( \mathsf { X } )$ , we see that

$$
\mathbb { W } _ { j } ( \xi ( o ) , 1 ) = \mathsf { F } ^ { - j } \quad \mathrm { f o r e v e r y } o \in \mathsf { O } .\tag{49}
$$

In particular, the ambiguity set is independent of the nominal occupation profile. Now, let $\tilde { o } ^ { j } : = ( \tilde { \xi } ^ { j } , \tilde { \eta } ^ { j } ) \in 0 ^ { j }$ . By (49) and the structure of the reward $r _ { t } ^ { j } ( \cdot )$ , we have

$$
\begin{array} { r l } & { \Phi _ { j } ( \tilde { \sigma } ^ { j } ; o ) = \underset { \hat { \mu } ^ { - j } \in \mathbb { F } ^ { - j } } { \operatorname* { i n f } } \displaystyle \sum _ { t = 0 } ^ { T } \sum _ { \boldsymbol { x } \in \mathbb { X } } \xi _ { t } ^ { j } ( \boldsymbol { x } ) \left[ a _ { t } ^ { j } ( \boldsymbol { x } , \widehat { \mu _ { t } } ^ { - j } ) + b _ { t } ^ { j } ( \boldsymbol { x } , \xi _ { t } ^ { j } ) \right] } \\ & { \qquad = \underset { \hat { \mu } ^ { - j } \in \mathbb { F } ^ { - j } } { \operatorname* { i n f } } \displaystyle \sum _ { t = 0 } ^ { T } \sum _ { \boldsymbol { x } \in \mathbb { X } } \xi _ { t } ^ { j } ( \boldsymbol { x } ) a _ { t } ^ { j } ( \boldsymbol { x } , \widehat { \mu _ { t } } ^ { - j } ) + \displaystyle \sum _ { t = 0 } ^ { T } \sum _ { \boldsymbol { x } \in \mathbb { X } } \xi _ { t } ^ { j } ( \boldsymbol { x } ) b _ { t } ^ { j } ( \boldsymbol { x } , \xi _ { t } ^ { j } ) . } \end{array}
$$

Suppose that, in the above expression,

$$
A _ { j } ( \widetilde { \boldsymbol { \sigma } } ^ { j } ) : = \operatorname* { i n f } _ { \widehat { \boldsymbol { \mu } } ^ { - j } \in \mathbb { F } ^ { - j } } \sum _ { t = 0 } ^ { T } \sum _ { \boldsymbol { x } \in \mathbb { X } } \widetilde { \xi } _ { t } ^ { j } ( \boldsymbol { x } ) a _ { t } ^ { j } ( \boldsymbol { x } , \widehat { \mu _ { t } } ^ { - j } ) , \mathrm { ~ a n d ~ } C _ { j } ( \widetilde { \boldsymbol { \sigma } } ^ { j } ; \boldsymbol { \xi } ^ { j } ) : = \sum _ { t = 0 } ^ { T } \sum _ { \boldsymbol { x } \in \mathbb { X } } \widetilde { \xi } _ { t } ^ { j } ( \boldsymbol { x } ) b _ { t } ^ { j } ( \boldsymbol { x } , \boldsymbol { \xi } _ { t } ^ { j } ) .
$$

The mapping $A _ { j } ( \cdot )$ is independent of the nominal profile $^ { o , }$ while $C _ { j } ( \tilde { o } ^ { j } ; \xi ^ { j } )$ depends on � only through the nominal state-flow $\xi ^ { j }$ . Consequently, we can write

$$
\Phi _ { j } ( \tilde { o } ^ { j } ; o ) = A _ { j } ( \tilde { o } ^ { j } ) + C _ { j } ( \tilde { o } ^ { j } ; \xi ^ { j } ) .\tag{50}
$$

We first record a consequence of the strong concavity established in Lemma 2. Let $v ^ { j } \in \mathsf { O } ^ { j }$ be arbitrary and let $\lambda \in ( 0 , 1 )$ . Since ${ \mathsf { O } } ^ { j }$ is convex, $( 1 - \lambda ) z ^ { j } + \lambda v ^ { j } \in { \mathsf { O } } ^ { j }$ By the �-strong concavity of $\Phi _ { j , \beta } ( \cdot ; o )$

$$
\Phi _ { j , \beta } \big ( ( 1 - \lambda ) z ^ { j } + \lambda v ^ { j } ; o \big ) \geq ( 1 - \lambda ) \Phi _ { j , \beta } ( z ^ { j } ; o ) + \lambda \Phi _ { j , \beta } ( v ^ { j } ; o ) + \frac { \beta \lambda ( 1 - \lambda ) } { 2 } \| z ^ { j } - v ^ { j } \| _ { \mathrm { o c } } ^ { 2 } .
$$

Since $z ^ { j }$ maximizes $\Phi _ { j , \beta } ( \cdot ; o )$ over ${ \mathsf { O } } ^ { j }$ , we also have $\Phi _ { j , \beta } \big ( z ^ { j } ; o \big ) \geq \Phi _ { j , \beta } \big ( ( 1 - \lambda ) z ^ { j } +$ $\lambda v ^ { j } ; o )$ . Combining the preceding inequalities and dividing by $\lambda > 0$ and letting $\lambda \to 0 ^ { + }$ , it follows that

$$
\Phi _ { j , \beta } ( z ^ { j } ; o ) - \Phi _ { j , \beta } ( v ^ { j } ; o ) \geq \frac { \beta } { 2 } \| z ^ { j } - v ^ { j } \| _ { \mathrm { o c c } } ^ { 2 } .\tag{51}
$$

Taking $v ^ { j } = \bar { z } ^ { j }$ in (51) gives $\begin{array} { r } { \Phi _ { j , \beta } ( z ^ { j } ; o ) - \Phi _ { j , \beta } ( \bar { z } ^ { j } ; o ) \geq \frac { \beta } { 2 } \| z ^ { j } - \bar { z } ^ { j } \| _ { \mathrm { o c c } } ^ { 2 } . } \end{array}$ .. Applying the same argument to $\Phi _ { j , \beta } ( \cdot ; \bar { o } )$ , whose maximizer is $\bar { z } ^ { j }$ , gives $\Phi _ { j , \beta } ( \bar { z } ^ { j } ; \bar { o } ) { - } \Phi _ { j , \beta } ( z ^ { j } ; \bar { o } ) \geq$ $\begin{array} { r l } {  { \frac { \beta } { 2 } \| z ^ { j } - \bar { z } ^ { j } \| _ { \mathrm { o c c } } ^ { 2 } } } \end{array}$ . Adding the preceding two quantities, we obtain

$$
\begin{array} { r } { \beta \| z ^ { j } - \bar { z } ^ { j } \| _ { \mathrm { o c c } } ^ { 2 } \leq \Phi _ { j , \beta } ( z ^ { j } ; o ) - \Phi _ { j , \beta } ( \bar { z } ^ { j } ; o ) + \Phi _ { j , \beta } ( \bar { z } ^ { j } ; \bar { o } ) - \Phi _ { j , \beta } ( z ^ { j } ; \bar { o } ) . } \end{array}\tag{52}
$$

Using (34), the right-hand side of (52) is equal to

$$
\begin{array} { r l } & { \Phi _ { j } ( z ^ { j } ; o ) - \Phi _ { j } ( \bar { z } ^ { j } ; o ) + \Phi _ { j } ( \bar { z } ^ { j } ; \bar { o } ) - \Phi _ { j } ( z ^ { j } ; \bar { o } ) } \\ & { + \Psi _ { j , \beta } ( z ^ { j } ) - \Psi _ { j , \beta } ( \bar { z } ^ { j } ) + \Psi _ { j , \beta } ( \bar { z } ^ { j } ) - \Psi _ { j , \beta } ( z ^ { j } ) } \\ & { = \Phi _ { j } ( z ^ { j } ; o ) - \Phi _ { j } ( \bar { z } ^ { j } ; o ) + \Phi _ { j } ( \bar { z } ^ { j } ; \bar { o } ) - \Phi _ { j } ( z ^ { j } ; \bar { o } ) . } \end{array}
$$

Using the decomposition (50), basic algebra yields

$$
\begin{array} { r l } & { \Phi _ { j } ( z ^ { j } ; o ) - \Phi _ { j } ( \bar { z } ^ { j } ; o ) + \Phi _ { j } ( \bar { z } ^ { j } ; \bar { o } ) - \Phi _ { j } ( z ^ { j } ; \bar { o } ) = C _ { j } ( z ^ { j } ; \xi ^ { j } ) } \\ & { - C _ { j } ( \bar { z } ^ { j } ; \xi ^ { j } ) - C _ { j } ( z ^ { j } ; \bar { \xi } ^ { j } ) + C _ { j } ( \bar { z } ^ { j } ; \bar { \xi } ^ { j } ) . } \end{array}
$$

Therefore, (52) implies that

$$
\begin{array} { r } { \beta \| z ^ { j } - \bar { z } ^ { j } \| _ { \mathrm { o c c } } ^ { 2 } \leq C _ { j } ( z ^ { j } ; \xi ^ { j } ) - C _ { j } ( \bar { z } ^ { j } ; \xi ^ { j } ) - C _ { j } ( z ^ { j } ; \bar { \xi } ^ { j } ) + C _ { j } ( \bar { z } ^ { j } ; \bar { \xi } ^ { j } ) . } \end{array}\tag{53}
$$

By the definition of $C _ { j } ( \cdot ; \cdot )$ , the right-hand side of (53) is equal to

$$
\sum _ { t = 0 } ^ { T } \sum _ { x \in { \cal X } } \big ( n _ { t } ^ { j } ( x ) - \bar { n } _ { t } ^ { j } ( x ) \big ) \big ( b _ { t } ^ { j } ( x , \xi _ { t } ^ { j } ) - b _ { t } ^ { j } ( x , \bar { \xi } _ { t } ^ { j } ) \big ) .
$$

Consequently,

$$
\begin{array} { r l } & { \displaystyle \beta \| z ^ { j } - \bar { z } ^ { j } \| _ { \mathsf { o c } } ^ { 2 } \leq \left| \displaystyle \sum _ { t = 0 } ^ { T } \sum _ { x \in \mathsf { X } } \bigl ( n _ { t } ^ { j } ( x ) - \bar { n } _ { t } ^ { j } ( x ) \bigr ) \bigl ( b _ { t } ^ { j } ( x , \xi _ { t } ^ { j } ) - b _ { t } ^ { j } ( x , \bar { \xi } _ { t } ^ { j } ) \bigr ) \right| } \\ & { \qquad \leq \displaystyle \sum _ { t = 0 } ^ { T } \| n _ { t } ^ { j } - \bar { n } _ { t } ^ { j } \| _ { 1 } \left\| b _ { t } ^ { j } ( \cdot , \xi _ { t } ^ { j } ) - b _ { t } ^ { j } ( \cdot , \bar { \xi } _ { t } ^ { j } ) \right\| _ { \infty } } \\ & { \qquad \leq \displaystyle \sum _ { t = 0 } ^ { T } \ell _ { j , t } \| n _ { t } ^ { j } - \bar { n } _ { t } ^ { j } \| _ { 1 } \| \xi _ { t } ^ { j } - \bar { \xi } _ { t } ^ { j } \| _ { 1 } . } \end{array}
$$

Since every occupation measure in ${ \mathsf { O } } ^ { j }$ has the prescribed initial distribution $\mu _ { 0 } ^ { j } .$ we have $\xi _ { 0 } ^ { j } = \bar { \xi } _ { 0 } ^ { j } = n _ { 0 } ^ { j } = \bar { n } _ { 0 } ^ { j } = \mu _ { 0 } ^ { j }$ . Hence, the term corresponding to $t = 0$ vanishes, and

therefore

$$
\begin{array} { r l } & { \beta \| z ^ { j } - \bar { z } ^ { j } \| _ { \mathrm { o c c } } ^ { 2 } \leq \displaystyle \sum _ { t = 1 } ^ { T } \ell _ { j , t } \| n _ { t } ^ { j } - \bar { n } _ { t } ^ { j } \| _ { 1 } \| \xi _ { t } ^ { j } - \bar { \xi } _ { t } ^ { j } \| _ { 1 } \leq \| z ^ { j } - \bar { z } ^ { j } \| _ { \mathrm { o c c } } \left( \displaystyle \sum _ { t = 1 } ^ { T } \ell _ { j , t } \right) \displaystyle \operatorname* { m a x } _ { 1 \leq t \leq T } \| \xi _ { t } ^ { j } - \bar { \xi } _ { t } ^ { j } \| _ { 1 } } \\ & { \qquad = L _ { j } \| z ^ { j } - \bar { z } ^ { j } \| _ { \mathrm { o c c } } \displaystyle \operatorname* { m a x } _ { 1 \leq t \leq T } \| \xi _ { t } ^ { j } - \bar { \xi } _ { t } ^ { j } \| _ { 1 } . } \end{array}
$$

If $\overset { \cdot } { z ^ { j } } = \hat { z } ^ { j }$ , then the desired estimate holds trivially. Otherwise, dividing the preceding inequality by $\beta \| z ^ { j } - { \bar { z } } ^ { j } \| _ { \mathrm { o c c } }$ gives us the estimate

$$
\left\| \mathsf { B r } _ { j , \beta } ( o ) - \mathsf { B r } _ { j , \beta } ( \bar { o } ) \right\| _ { \mathrm { o c c } } \leq \frac { L _ { j } } { \beta } \operatorname* { m a x } _ { 1 \leq t \leq T } \| \xi _ { t } ^ { j } - \bar { \xi } _ { t } ^ { j } \| _ { 1 } \leq \frac { L _ { j } } { \beta } \| o ^ { j } - \bar { o } ^ { j } \| _ { \mathrm { o c c } } ,\tag{54}
$$

where the last inequality follows from the fact that $\| \cdot \| _ { \mathrm { o c c } } .$ , max $\bar { \mathsf { \varepsilon } } _ { 1 \leq t \leq T } \| \boldsymbol { \xi } _ { t } ^ { j } - \bar { \boldsymbol { \xi } } _ { t } ^ { j } \| _ { 1 } \leq$ $\| o ^ { j } - { \bar { o } } ^ { j } \| _ { \mathrm { o c c } }$ . Finally, taking the maximum over $j \in [ M ]$ in (54), we obtain

$$
\begin{array} { r l } & { \left\| \mathsf { B r } _ { \beta } ( o ) - \mathsf { B r } _ { \beta } ( \bar { o } ) \right\| _ { \mathrm { o c c } , \infty } = \displaystyle \operatorname* { m a x } _ { j \in [ M ] } \left\| \mathsf { B r } _ { j , \beta } ( o ) - \mathsf { B r } _ { j , \beta } ( \bar { o } ) \right\| _ { \mathrm { o c c } } } \\ & { \qquad \le \displaystyle \operatorname* { m a x } _ { j \in [ M ] } \frac { L _ { j } } { \beta } \| o ^ { j } - \bar { o } ^ { j } \| _ { \mathrm { o c c } } = q _ { \beta } \| o - \bar { o } \| _ { \mathrm { o c c } , \infty } . } \end{array}
$$

If $\beta > \operatorname* { m a x } _ { j \in [ M ] } L _ { j }$ , then $q _ { \beta } < 1$ . Therefore, $\mathsf { B r } _ { \beta } : 0 \to 0$ is a contraction with respect to $\| \cdot \| _ { \mathrm { o c c , \infty } } .$

Since O is nonempty and compact and $q _ { \beta } < 1$ , Banach’s fixed-point theorem [AB06] gives a unique fixed point $O _ { \beta } ^ { \star } ,$ to which all iterates of $\mathsf { B r } _ { \beta }$ converge. By Lemma $^ { 2 , }$ the best-response occupation measure is unique. Together with the policy– occupation-measure equivalence, this makes $\Gamma _ { \beta }$ single-valued and thus

$$
\xi ( \mathsf { B r } _ { \beta } ( o ) ) = \Gamma _ { \beta } ( \xi ( o ) ) \quad \mathrm { f o r \ a l l \ } o \in { \mathsf { O } } .
$$

Hence, by continuity of $\xi ,$ , the mean-field iteration converges from every feasible initial flow to $\mu _ { \beta } ^ { \star } : = \xi ( o _ { \beta } ^ { \star } )$ . The same equivalence shows that $o _ { \beta } ^ { \star }$ yields an equilibrium in Definition 8 and that every such equilibrium induces a fixed point of $\mathsf { B r } _ { \beta }$ . Uniqueness of $o _ { \beta } ^ { \star }$ therefore gives uniqueness of the equilibrium mean-field flow. □

Let us recall Assumptions 2 and 3 along with several quantities defined therein. We defined $\begin{array} { r } { \ell _ { j , t } ^ { a } : = \sum _ { k \neq j } \operatorname* { m a x } _ { x \in \mathsf { X } } \| A _ { t } ^ { j , k } ( x , \cdot ) \| _ { \infty } } \end{array}$ and $\begin{array} { r } { L _ { j } ^ { a } : = \sum _ { t = 1 } ^ { T } \ell _ { j , t } ^ { a } } \end{array}$ . Further, recall $\mathsf { O } ^ { j } , \mathsf { O }$ , and $\xi ( o )$ from the definition of occupation measures and properties (2)–(3). For $\overline { { { o } } } ^ { j } : = ( \overline { { { \xi } } } ^ { j } , \overline { { { \eta } } } ^ { j } ) \in { \sf O } ^ { j }$ and $\beta > 0$ , define

$$
\overline { { o } } ^ { j } \mapsto \Psi _ { j , \beta } ( \overline { { o } } ^ { j } ) : = - \beta \sum _ { t = 0 } ^ { T - 1 } \sum _ { x \in \mathsf { X } } \sum _ { u \in \mathsf { U } } \overline { { \eta } } _ { t } ^ { j } ( x , u ) \log \frac { \overline { { \eta } } _ { t } ^ { j } ( x , u ) } { \overline { { \xi } } _ { t } ^ { j } ( x ) } .
$$

We now restate Theorem 4 and then record a detailed proof.

Theorem 4 (Contractivity of BR mapping when $\varepsilon \in ( 0 , 1 ) )$ . Fix $\varepsilon \in \mathsf { \Gamma } ( 0 , 1 )$ and $\beta > 0 ,$ and suppose that Assumptions 2 and 3 hold. For $\sigma > 0$ and $\widehat { \mu } ^ { - j } \ = \ \{ \widehat { \mu } _ { t } ^ { k } \} _ { k \neq j , t = 0 } ^ { T } ,$ let $\begin{array} { r } { \| \widehat { \mu } ^ { - j } \| _ { 2 } ^ { 2 } : = \hat { \sum } _ { k \neq j } \sum _ { t = 0 } ^ { T } \sum _ { x \in \mathsf { X } } | \widehat { \mu } _ { t } ^ { k } ( x ) | ^ { 2 } } \end{array}$ and define the inner-regularized payof

$$
\begin{array} { l } { \Phi _ { j , \beta } ^ { \sigma } ( \widetilde { \sigma } ^ { j } ; o ) : = \underset { \widehat { \mu } ^ { - j } \in \mathcal { W } _ { j } ( \xi ( o ) , \varepsilon ) } { \operatorname* { m i n } } \left[ \displaystyle \sum _ { t = 0 } ^ { T } \displaystyle \sum _ { x \in \mathsf { X } } \widetilde { \xi } _ { t } ^ { j } ( x ) r _ { t } ^ { j } \big ( x , \xi _ { t } ^ { j } , \widehat { \mu _ { t } } ^ { - j } \big ) + \frac { \sigma } { 2 } \| \widehat { \mu } ^ { - j } \| _ { 2 } ^ { 2 } \right] } \\ { ) \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad + \Psi _ { j , \beta } ( \widetilde { \sigma } ^ { j } ) . } \end{array}\tag{55}
$$

Define the best response mappings $\begin{array} { r } { \mathsf { B r } _ { j , \beta } ^ { \sigma } ( o ) : = \arg \operatorname* { m a x } _ { \widetilde { o } ^ { j } \in \Theta ^ { j } } \Phi _ { j , \beta } ^ { \sigma } ( \widetilde { o } ^ { j } ; o ) } \end{array}$ and $\mathsf { B r } _ { \beta } ^ { \sigma } ( o ) : = ( \mathsf { B r } _ { 1 , \beta } ^ { \sigma } ( o ) , \hdots , \mathsf { B r } _ { M , \beta } ^ { \sigma } ( o ) )$ . Then, for every $j \in [ M ]$ , there exists a

constant $\Lambda _ { j , \varepsilon } \geq 0 \mathrm { { } } _ { \mathrm { { } } }$ , independent of �, such that, for every $\sigma > 0$ and $o , \bar { o } \in \mathsf { O } ,$

$$
\left\| \mathsf { B r } _ { j , \beta } ^ { \sigma } ( o ) - \mathsf { B r } _ { j , \beta } ^ { \sigma } ( \bar { o } ) \right\| _ { \mathrm { o c c } } \leq \frac { L _ { j } + 2 L _ { j } ^ { a } \Lambda _ { j , \varepsilon } } { \beta } \| o - \bar { o } \| _ { \mathrm { o c c } , \infty } .\tag{56}
$$

Consequently, defining $\begin{array} { r } { q _ { \beta , \varepsilon } : = \operatorname* { m a x } _ { j \in [ M ] } ( L _ { j } + 2 L _ { j } ^ { a } \Lambda _ { j , \varepsilon } ) / \beta , } \end{array}$ , we have

$$
\begin{array} { r } { \left\| \mathsf { B r } _ { \beta } ^ { \sigma } ( o ) - \mathsf { B r } _ { \beta } ^ { \sigma } ( \bar { o } ) \right\| _ { \mathrm { o c c } , \infty } \leq q _ { \beta , \varepsilon } \| o - \bar { o } \| _ { \mathrm { o c c } , \infty } . } \end{array}
$$

If $\beta > \mathrm { m a x } _ { j \in [ M ] } ( L _ { j } + 2 L _ { i } ^ { a } \Lambda _ { j , \varepsilon } )$ , then $\mathsf { B r } _ { \beta }$ and $\mathsf { B r } _ { \beta } ^ { \sigma }$ , for every $\sigma > 0$ are contractions on O in $\| \cdot \| _ { \mathrm { o c c , \infty } }$ with the common factor $q _ { \beta , \varepsilon } < 1$ Their unique fixed points $o _ { \beta } ^ { \star }$ and $o _ { \beta , \sigma } ^ { \star } ,$ , respectively, satisfy

$$
\operatorname* { l i m } _ { \sigma \to 0 ^ { + } } \| o _ { \beta , \sigma } ^ { \star } - o _ { \beta } ^ { \star } \| _ { \mathrm { o c c , \infty } } = 0 .
$$

Moreover, for every $\mu _ { 0 } ,$ the iteration $\mu ( n + 1 ) = \Gamma _ { \beta } ( \mu ( n ) )$ converges to $\mu _ { \beta } ^ { \star } : =$ $\xi ( o _ { \beta } ^ { \star } )$ , the unique equilibrium mean-field flow in Definition 8.

Proof. Fix $j \in [ M ]$ . We first establish the regularity of the inner minimizer; our proof relies on techniques from variational analysis [DR14] and polyhedral partition [BBM15]. To this end, we first establish uniqueness. Throughout the proof, $\Phi _ { j , \beta } ^ { 0 } : =$ $\Phi _ { j , \beta }$ and $\mathsf { B r } _ { j , \beta } ^ { 0 } : = \mathsf { B r } _ { j , \beta }$ , with the same fixed $ { \varepsilon } \in ( 0 , 1 )$

Step 1 (Establishing uniqueness). By Assumption 3, for every $\widetilde { \xi } ^ { j } = \{ \widetilde { \xi } _ { t } ^ { j } \} _ { t = 0 } ^ { T } .$ , define the vector $c _ { j } ( \widetilde { \xi ^ { j } } )$ by

$$
\left[ c _ { j } ( \widetilde { \xi } ^ { j } ) \right] _ { k , t , y } : = \sum _ { x \in \mathbb { X } } \widetilde { \xi } _ { t } ^ { j } ( x ) A _ { t } ^ { j , k } ( x , y ) \quad \mathrm { f o r } \ k \not = j , t = 0 , \ldots , T , \ \mathrm { a n d } \ y \in \mathbb { X } .
$$

Note that, we view $c _ { j } ( \widetilde { \xi ^ { j } } )$ and $\widehat { \mu } ^ { - j }$ as vectors indexed by $( k , t , y )$ with $k \neq j$ so that $\begin{array} { r } { \big \langle c _ { j } ( \widetilde { \xi } ^ { j } ) , \widehat { \mu } ^ { - j } \big \rangle = \sum _ { t = 0 } ^ { T } \sum _ { k \neq j } \sum _ { y \in \mathsf { X } } [ c _ { j } ( \widetilde { \xi } ^ { j } ) ] _ { k , t , y } \widehat { \mu } _ { t } ^ { k } ( y ) } \end{array}$ . Then, by Assumptions 2 and 3, the terms in the inner objective of (55) that depend on $\widehat { \mu } ^ { - j }$ are given by $\begin{array} { r } { \left. c _ { j } ( \widetilde { \xi } ^ { j } ) , \widehat { \mu } ^ { - j } \right. + \frac { \sigma } { 2 } \left\| \widehat { \mu } ^ { - j } \right\| _ { 2 } ^ { 2 } } \end{array}$ . Consequently, the inner minimizers of (55) coincide with the minimizers of

$$
\operatorname* { m i n } _ { \substack { \widehat { \mu } ^ { - j } \in \mathcal { W } _ { j } ( \xi ( o ) , \varepsilon ) } } \left[ \left. c _ { j } ( \widetilde { \xi } ^ { j } ) , \widehat { \mu } ^ { - j } \right. + \frac { \sigma } { 2 } \left. \widehat { \mu } ^ { - j } \right. _ { 2 } ^ { 2 } \right] .\tag{57}
$$

The ambiguity set $\mathcal { W } _ { j } ( \xi ( o ) , \varepsilon )$ is nonempty, compact, and convex (Proposition 1). Since $\sigma > 0$ , the objective in (57) is �-strongly convex in $\widehat { \mu } ^ { - j }$ . Therefore, (57) admits a unique minimizer. We will denote it by $\begin{array} { r } { \bar { \mu } _ { \sigma } ^ { - j , \star } ( \widetilde { \sigma } ^ { j } ; o ) \dot { = } \{ \widehat { \mu } _ { \sigma , t } ^ { - j , \star } ( \widetilde { \sigma } ^ { j } ; o ) \} _ { t = 0 } ^ { T } , } \end{array}$ , where $\widehat { \mu } _ { \sigma , t } ^ { - j , \star } ( \widetilde { \sigma } ^ { j } ; o )$ denotes the tuple of opponents’ state distributions at time �.

Step 2 (Establishing structure and regularity). We next establish a Lipschitz estimate for the inner minimizer with a constant independent of $\sigma$ . By Proposition 1, the occupation constraints and the auxiliary-variable representation of the total-variation constraints are jointly linear in the nominal and adversarial flows. Since a linear projection of a polyhedron is again a polyhedron [Roc97, Theorem 19.3], there exist fixed matrices $E _ { j } , F _ { j } , C _ { j } , G _ { j }$ and fixed vectors $e _ { j } , d _ { j }$ , depending only on $j$ and $\varepsilon ,$ such that, for every $\mu \in \xi ( \mathsf { O } )$

$$
\begin{array} { r } { \mathcal { W } _ { j } ( \mu , \varepsilon ) = \big \{ \nu \big \vert E _ { j } \nu + F _ { j } \mu = e _ { j } , \quad C _ { j } \nu + G _ { j } \mu \leq d _ { j } \big \} . } \end{array}\tag{58}
$$

Here � denotes the vector obtained by stacking the opponent flows.

Recall that $\widehat { \mu } _ { \sigma } ^ { - j , \star } ( \widetilde { \sigma } ^ { j } ; o )$ is the unique minimizer of (57) (established in the previous step). For an arbitrary vector � of the same dimension as �, define the projection

operator [PB14, Chapter 1]

$$
P _ { j } ( v , \mu ) : = \operatorname * { a r g m i n } _ { \nu \in \mathcal { W } _ { j } ( \mu , \varepsilon ) } \frac { 1 } { 2 } \left\| \nu - v \right\| _ { 2 } ^ { 2 } .\tag{59}
$$

First, note that, since $\mathcal { W } _ { j } ( \mu , \varepsilon )$ is nonempty, closed, and convex, $P _ { j } ( v , \mu )$ is well defined and single-valued. The primary idea of this step is to first, interpret the minimizer $\widehat { \mu } _ { \sigma } ^ { - j , \star } ( \widetilde { \sigma } ^ { j } ; o )$ as a projection along the lines of [PB14, Chapter 6] and then, following the active-set analysis<sup>10</sup> of [BMDP02, Section 4], to establish a piecewiseafine representation of $P _ { j }$ and to show that $P _ { j } ( v , \cdot )$ is Lipschitz on �(O), with a Lipschitz constant independent of �.

To this end, let $\mathcal { R } _ { j }$ denote the row-index set of $C _ { j } .$ . For every $I \subseteq { \mathcal { R } } _ { j }$ , let $C _ { j , I }$ and $G _ { j , I }$ denote the submatrices of $C _ { j }$ and $G _ { j }$ , respectively, formed by the rows indexed by �, let $d _ { j , \ l }$ <sub>�</sub> denote the corresponding subvector of $d _ { j }$ , and let $I ^ { c } : = \mathcal { R } _ { j } \backslash I .$ . Define

$$
H _ { I } : = \left[ { E _ { j } \atop C _ { j , I } } \right] , \quad R _ { I } : = \left[ { F _ { j } \atop G _ { j , I } } \right] , \quad h _ { I } : = \left[ { e _ { j } \atop d _ { j , I } } \right] .\tag{60}
$$

Fix � and $\mu ,$ and write $\nu = P _ { j } ( v , \mu )$ . By the variational-inequality characterization of Euclidean projection [DR14, Theorem 2A.7] together with the normal-cone representation for a polyhedron [DR14, Theorem 2E.3], there exist a subset $I \subseteq { \mathcal { R } } _ { j }$ an equality multiplier $\zeta ,$ and a multiplier $\lambda _ { I } \geq 0$ such that

(61a)

$$
E _ { j } { \nu } + F _ { j } { \mu } = e _ { j } ,\tag{61b}
$$

$$
C _ { j , I ^ { \gamma } } + G _ { j , I } \mu = d _ { j , I } ,\tag{61c}
$$

$$
C _ { j , I ^ { c } } \nu + G _ { j , I ^ { c } } \mu \leq d _ { j , I ^ { c } } ,\tag{61d}
$$

$$
\nu - v + E _ { j } ^ { \top } \zeta + C _ { j , I } ^ { \top } \lambda _ { I } = 0 , \qquad \lambda _ { I } \geq 0 .
$$

Conversely, if $\nu , \zeta , \lambda _ { I }$ satisfy (61) for some $I \subseteq { \mathcal { R } } _ { j }$ , then $\nu = P _ { j } ( v , \mu )$ . Equations (61a)–(61b) give

$$
H _ { I } \nu = h _ { I } - R _ { I } \mu .\tag{62}
$$

Moreover, (61d) implies $\nu  v \in$ range $( H _ { I } ^ { \top } )$ , which together with (62) uniquely determine the Euclidean projection of � onto the afine set $\{ \eta \ | \ H _ { I } \eta = h _ { I } - R _ { I } \mu \}$ Therefore [PB14, Chapter 6],

$$
P _ { j } ( v , \mu ) = v + H _ { I } ^ { \dagger } \big ( h _ { I } - R _ { I } \mu - H _ { I } v \big ) ,\tag{63}
$$

where $H _ { I } ^ { \dagger }$ denotes the Moore–Penrose pseudoinverse. For each $I \subseteq { \mathcal { R } } _ { j }$ , define

$$
\mathfrak { Q } _ { I } : = \big \{ ( v , \mu , \nu , \zeta , \lambda _ { I } ) \big | \mu \in \xi ( \mathsf { O } ) \mathrm { ~ a n d ~ } ( 6 1 \mathrm { a } ) - ( 6 1 \mathrm { d } ) \mathrm { ~ h o l d s } \big \}
$$

Since O is a polytope and $\xi$ is linear, $\xi ( 0 )$ is a polytope Hence $\mathbb { Q } _ { I }$ is a polyhedron. Let

$$
\mathcal { P } _ { I } : = \big \{ ( v , \mu ) \big | \mathrm { t h e r e ~ e x i s t } \ : \nu , \zeta , \lambda _ { I } \ : \mathrm { w i t h } \ : ( v , \mu , \nu , \zeta , \lambda _ { I } ) \in \mathbb { Q } _ { I } \big \} .
$$

Note that, the set $\mathcal { P } _ { I }$ is the coordinate projection of the polyhedron $\mathbb { Q } _ { I } .$ , and is therefore itself a polyhedron [Roc97, Theorem 19.3]. For every $( v , \mu ) \in \mathbb { R } ^ { \dim ( \nu ) } \times \xi ( 0 )$ , the projection $P _ { j } ( v , \mu )$ exists and satisfies (61) for some $I \subseteq { \mathcal { R } } _ { j }$ and suitable multipliers. By the definitions of $\mathbb { Q } _ { I }$ and $\mathcal { P } _ { I }$ , this implies $( v , \mu ) \in \mathcal { P } _ { I }$ . Hence, these finitely many sets cover the entire parameter domain; see also [BMDP02, Section 4.2, Theorem 4]. Moreover, on each ${ \mathcal P } _ { I }$ , the projection mapping is given by the afine formula (63). Notice that, for fixed $v ,$ the expression in (63) is afine in $\mu ,$ with coeficient matrix $- H _ { I } ^ { \dagger } R _ { I }$ . Let $\| A \| _ { 2 \to 2 } : = \operatorname* { s u p } _ { \| z \| _ { 2 } = 1 } \| A z \| _ { 2 }$ denotes the operator norm induced by the Euclidean norms on the domain and codomain; see [GMU25]. Consequently, we set

$$
\kappa _ { j , \varepsilon } : = \operatorname* { m a x } _ { I \subseteq \mathcal { R } _ { j } } \left\| H _ { I } ^ { \dagger } R _ { I } \right\| _ { 2 \to 2 } .\tag{64}
$$

Since $\mathcal { R } _ { j }$ is finite, $K _ { j , \varepsilon } < + \infty$ . Fix � and $\mu , \bar { \mu } \in \xi ( \mathsf { O } )$ , and consider the combination $\mu _ { s } : = ( 1 - s ) \mu + s \bar { \mu }$ for all $s \in [ 0 , 1 ]$ . Because $\xi ( 0 )$ is convex, $\mu _ { s } \in \xi ( \mathsf { O } )$ for every $s \in [ 0 , 1 ]$ . For fixed $v ,$ each intersection $\left\{ s \in [ 0 , 1 ] \ \left| \ ( v , \mu _ { s } ) \in \mathcal { P } _ { I } \right. \right\}$ is an interval (possibly empty) because $\mathcal { P } _ { I }$ is convex. Since only finitely many sets $\mathcal { P } _ { I }$ occur and they cover the segment, there exists a finite partition $0 = s _ { 0 } < s _ { 1 } < \cdot \cdot \cdot < s _ { N } = 1$ such that, on every interval $[ s _ { \ell - 1 } , s _ { \ell } ]$ , one afine representation (63) is valid. Now notice that $P _ { j } ( v , \mu _ { s _ { \ell } } ) - P _ { j } ( v , \mu _ { s _ { \ell - 1 } } ) = - H _ { I \ell } ^ { \dagger } R _ { I _ { \ell } } ( \mu _ { s _ { \ell } } - \mu _ { s _ { \ell - 1 } } )$ ; taking norm and expanding it further, we obtain the estimate

$$
\begin{array} { r } { \left\| P _ { j } ( v , \mu _ { s _ { \ell } } ) - P _ { j } ( v , \mu _ { s _ { \ell - 1 } } ) \right\| _ { 2 } \le \left\| H _ { I _ { \ell } } ^ { \dagger } R _ { I _ { \ell } } \right\| _ { 2 \to 2 } \left\| \mu _ { s _ { \ell } } - \mu _ { s _ { \ell - 1 } } \right\| _ { 2 } \le \kappa _ { j , \varepsilon } \left\| \mu _ { s _ { \ell } } - \mu _ { s _ { \ell - 1 } } \right\| _ { 2 } . } \end{array}
$$

Summing over $\ell = 1 , \ldots , N$ and using $\begin{array} { r } { \sum _ { \ell = 1 } ^ { N } \left\| \mu _ { s _ { \ell } } - \mu _ { s _ { \ell - 1 } } \right\| _ { 2 } = \left\| \mu - \bar { \mu } \right\| _ { 2 } } \end{array}$ , we obtain

$$
\left\| P _ { j } ( v , \mu ) - P _ { j } ( v , \bar { \mu } ) \right\| _ { 2 } \leq \kappa _ { j , \varepsilon } \left\| \mu - \bar { \mu } \right\| _ { 2 } .\tag{65}
$$

We now return to the QP (57) and employ the standard completion of square, which yields

$$
\left. c _ { j } ( \widetilde { \xi } ^ { j } ) , \widetilde { \mu } ^ { - j } \right. + \frac { \sigma } { 2 } \left\| \widehat { \mu } ^ { - j } \right\| _ { 2 } ^ { 2 } = \frac { \sigma } { 2 } \left\| \widehat { \mu } ^ { - j } + \frac { c _ { j } ( \widetilde { \xi } ^ { j } ) } { \sigma } \right\| _ { 2 } ^ { 2 } - \frac { 1 } { 2 \sigma } \left\| c _ { j } ( \widetilde { \xi } ^ { j } ) \right\| _ { 2 } ^ { 2 } .
$$

Therefore, we now have the characterization

$$
\widehat { \mu } _ { \sigma } ^ { - j , \star } ( \widetilde { \sigma } ^ { j } ; o ) = P _ { j } \left( - \frac { c _ { j } ( \widetilde { \xi } ^ { j } ) } { \sigma } , \xi ( o ) \right) .\tag{66}
$$

For fixed $\widetilde { o } ^ { j }$ , the first argument of $P _ { j }$ in (66) is the same when � is replaced by ${ \bar { o } } .$ Hence (65) yields

$$
\begin{array} { r } { \left\| \widehat { \mu } _ { \sigma } ^ { - j , \star } ( \widetilde { \sigma } ^ { j } ; o ) - \widehat { \mu } _ { \sigma } ^ { - j , \star } ( \widetilde { \sigma } ^ { j } ; \bar { o } ) \right\| _ { 2 } \leq \kappa _ { j , \varepsilon } \| \xi ( o ) - \xi ( \bar { o } ) \| _ { 2 } . } \end{array}\tag{67}
$$

The constant in (67) is therefore independent of $\sigma .$ . Finally, from definition of occupation measure, we have

$$
\| \xi ( o ) - \xi ( \bar { o } ) \| _ { 2 } \leq \sqrt { M ( T + 1 ) } \| o - \bar { o } \| _ { \mathrm { o c c , \infty } } .
$$

Moreover, for every $k \neq j$ and $t = 0 , \ldots , T$

$$
\begin{array} { r l } & { \mathrm { d } _ { \mathrm { T V } } \big ( \widehat { \mu } _ { \sigma , t } ^ { k , \star } ( \widetilde { \sigma } ^ { j } ; o ) , \widehat { \mu } _ { \sigma , t } ^ { k , \star } ( \widetilde { \sigma } ^ { j } ; \bar { o } ) \big ) = \displaystyle \frac { 1 } { 2 } \left\| \widehat { \mu } _ { \sigma , t } ^ { k , \star } ( \widetilde { \sigma } ^ { j } ; o ) - \widehat { \mu } _ { \sigma , t } ^ { k , \star } ( \widetilde { \sigma } ^ { j } ; \bar { o } ) \right\| _ { 1 } } \\ & { \qquad \leq \displaystyle \frac { \sqrt { | { \mathsf X } | } } { 2 } \left\| \widehat { \mu } _ { \sigma } ^ { - j , \star } ( \widetilde { \sigma } ^ { j } ; o ) - \widehat { \mu } _ { \sigma } ^ { - j , \star } ( \widetilde { \sigma } ^ { j } ; \bar { o } ) \right\| _ { 2 } . } \end{array}
$$

Thus, with

$$
\Lambda _ { j , \varepsilon } : = \frac 1 2 \sqrt { | \mathsf { X } | M ( T + 1 ) } \kappa _ { j , \varepsilon } ,\tag{68}
$$

we obtain, for every $\sigma > 0 , \widetilde { o } ^ { j } \in { \cal 0 } ^ { j }$ , and $o , \bar { o } \in \mathrm { O }$

$$
\operatorname* { m a x } _ { k \neq j } \operatorname* { m a x } _ { 0 \leq t \leq T } \mathrm { d } _ { \mathrm { T V } } \big ( \widehat { \mu } _ { \sigma , t } ^ { k , \star } ( \widetilde { \sigma } ^ { j } ; o ) , \widehat { \mu } _ { \sigma , t } ^ { k , \star } ( \widetilde { \sigma } ^ { j } ; \bar { o } ) \big ) \leq \Lambda _ { j , \varepsilon } \| o - \bar { o } \| _ { \mathrm { o c c } , \infty } .\tag{69}
$$

Step 3 (Lipschitz estimate and contractivity of Br). We now derive the Lipschitz estimate for the best-response mapping. Define the non-entropic component of the

inner-regularized payof by

$$
H _ { j , \sigma } ( \widetilde { \xi } ^ { j } ; o ) : = \operatorname* { m i n } _ { \substack { \hat { \mu } ^ { - j } \in \mathcal { W } _ { j } ( \xi ( o ) , \varepsilon ) } } \left[ \sum _ { \ell = 0 } ^ { T } \sum _ { x \in \mathsf { X } } \widetilde { \xi } _ { t } ^ { j } ( x ) r _ { t } ^ { j } \big ( x , \xi _ { t } ^ { j } , \widetilde { \mu _ { t } } ^ { - j } \big ) + \frac { \sigma } { 2 } \big \Vert \widehat { \mu } ^ { - j } \big \Vert _ { 2 } ^ { 2 } \right] ,\tag{70}
$$

so that $\Phi _ { j , \beta } ^ { \sigma } ( \widetilde { o } ^ { j } ; o ) = H _ { j , \sigma } ( \widetilde { \xi } ^ { j } ; o ) + \Psi _ { j , \beta } ( \widetilde { o } ^ { j } )$ . For fixed $\tilde { \xi } ^ { j }$ and $^ { o , }$ the objective in (70) is the sum of an afine function of $\widehat { \mu } ^ { - j }$ and the �-strongly convex quadratic term. Since the feasible set is nonempty, compact, and convex, the minimizer exists and is unique. For fixed $^ { o , }$ compactness of $\mathcal { W } _ { j } ( \xi ( o ) , \varepsilon )$ , joint continuity of the objective in $( \widetilde \xi ^ { j } , \widehat \mu ^ { - j } )$ , and uniqueness of the minimizer in (70) imply that this minimizer depends continuously on $\widetilde { \xi ^ { j } }$ . Therefore, Danskin’s theorem [Dan12] gives continuous diferentiability and yields

$$
\begin{array} { r } { \nabla _ { \widetilde { \xi } _ { t } ^ { j } } H _ { j , \sigma } ( \widetilde { \xi } ^ { j } ; o ) = a _ { t } ^ { j } \left( \cdot , \widetilde { \mu } _ { \sigma , t } ^ { - j , \star } ( \widetilde { o } ^ { j } ; o ) \right) + b _ { t } ^ { j } ( \cdot , \xi _ { t } ^ { j } ) \mathrm { ~ f o r ~ a l l ~ } t = 0 , \dots , T . } \end{array}\tag{71}
$$

Fix $o , \bar { o } \in \mathsf { O } .$ , and define $\Delta _ { j } ( \widetilde { o } ^ { j } ) : = \Phi _ { j , \beta } ^ { \sigma } ( \widetilde { o } ^ { j } ; \bar { o } ) - \Phi _ { j , \beta } ^ { \sigma } ( \widetilde { o } ^ { j } ; o )$ . The entropy terms cancel in $\Delta _ { j }$ , and hence, by (71),

$$
\nabla _ { \widetilde { \xi } _ { t } ^ { j } } \Delta _ { j } ( \widetilde { \sigma } ^ { j } ) = a _ { t } ^ { j } \left( \cdot , \widetilde { \mu _ { \sigma , t } } ^ { j , \star } ( \widetilde { \sigma } ^ { j } ; \bar { \sigma } ) \right) - a _ { t } ^ { j } \left( \cdot , \widehat { \mu _ { \sigma , t } } ^ { - j , \star } ( \widetilde { \sigma } ^ { j } ; \sigma ) \right) + b _ { t } ^ { j } ( \cdot , \bar { \xi } _ { t } ^ { j } ) - b _ { t } ^ { j } ( \cdot , \xi _ { t } ^ { j } ) .\tag{72}
$$

It follows from Assumption 3 and the definition of $\ell _ { j , t } ^ { a }$ that

$$
\left\| a _ { t } ^ { j } ( \cdot , \nu ^ { - j } ) - a _ { t } ^ { j } ( \cdot , \bar { \nu } ^ { - j } ) \right\| _ { \infty } \leq \ell _ { j , t } ^ { a } \operatorname* { m a x } _ { k \neq j } \left\| \nu ^ { k } - \bar { \nu } ^ { k } \right\| _ { 1 } ,\tag{73}
$$

for every $\nu ^ { - j } , \bar { \nu } ^ { - j } \in \mathcal { P } ( \mathsf { X } ) ^ { M - 1 }$ . Combining (69), (73), and the identity $\| { \boldsymbol { \mu } } - { \bar { \boldsymbol { \mu } } } \| _ { 1 } =$ $2 \mathrm { d } _ { \mathrm { T V } } ( \mu , \bar { \mu } )$ , we obtain

$$
\begin{array} { r } { \left\| a _ { t } ^ { j } \left( \cdot , \widehat { \mu } _ { \sigma , t } ^ { - j , \star } ( \widetilde { \sigma } ^ { j } ; \bar { \sigma } ) \right) - a _ { t } ^ { j } \left( \cdot , \widehat { \mu } _ { \sigma , t } ^ { - j , \star } ( \widetilde { \sigma } ^ { j } ; o ) \right) \right\| _ { \infty } \leq 2 \ell _ { j , t } ^ { a } \Lambda _ { j , \varepsilon } \| o - \bar { \sigma } \| _ { \mathsf { o c c } , \infty } . } \end{array}\tag{74}
$$

Moreover, Assumption 2 gives

$$
\begin{array} { r } { \left\| b _ { t } ^ { j } ( \cdot , \bar { \xi } _ { t } ^ { j } ) - b _ { t } ^ { j } ( \cdot , \xi _ { t } ^ { j } ) \right\| _ { \infty } \leq \ell _ { j , t } \left\| \bar { \xi } _ { t } ^ { j } - \xi _ { t } ^ { j } \right\| _ { 1 } \leq \ell _ { j , t } \left\| o - \bar { o } \right\| _ { \mathrm { o c c } , \infty } . } \end{array}\tag{75}
$$

Every occupation measure in O has the prescribed initial distributions, and every element of the ambiguity set has the same prescribed opponent initial distributions. Therefore,

$$
\begin{array} { r } { \xi _ { 0 } ^ { j } = \bar { \xi } _ { 0 } ^ { j } = \mu _ { 0 } ^ { j } , \mathrm { ~ a n d ~ } \widehat { \mu } _ { \sigma , 0 } ^ { k , \star } ( \widetilde { \sigma } ^ { j } ; o ) = \widehat { \mu } _ { \sigma , 0 } ^ { k , \star } ( \widetilde { \sigma } ^ { j } ; \bar { o } ) = \mu _ { 0 } ^ { k } \mathrm { ~ f o r ~ a l l ~ } k \neq j , } \end{array}
$$

and hence $\nabla _ { \widetilde { \xi } _ { 0 } ^ { j } } \Delta _ { j } ( \widetilde { o } ^ { j } ) = 0$ . It follows from (72)–(75) that, for $t = 1 , \ldots , T .$

$$
\begin{array} { r } { \left\| \nabla _ { \widetilde { \xi } _ { t } ^ { j } } \Delta _ { j } ( \widetilde { \sigma } ^ { j } ) \right\| _ { \infty } \le \left( \ell _ { j , t } + 2 \ell _ { j , t } ^ { a } \Lambda _ { j , \varepsilon } \right) \| o - \bar { \sigma } \| _ { \mathrm { o c c } , \infty } . } \end{array}\tag{76}
$$

Let $\widetilde { o } ^ { j } , \widetilde { o } ^ { j } \in \mathbb { O } ^ { j }$ , and let $\widetilde { \sigma } _ { s } ^ { j } : = ( 1 - s ) \widetilde { \sigma } ^ { j } + s \widetilde { \underline { { \sigma } } } ^ { j }$ for $s \in [ 0 , 1 ]$ . Since ${ \mathsf { O } } ^ { j }$ is convex, $\widetilde { \boldsymbol { o } } _ { s } ^ { j } \in \mathsf { O } ^ { j }$ for every $s \in [ 0 , 1 ]$ . Using the fundamental theorem of calculus, Hölder’s inequality, and (76), we have

$$
\begin{array} { r l } {  { | \Delta _ { j } ( \widetilde { \boldsymbol { \sigma } } ^ { j } ) - \Delta _ { j } ( \underline { { \widetilde { \boldsymbol { \sigma } } } } ^ { j } ) | \leq \sum _ { t = 1 } ^ { T } \| \widetilde { \xi } _ { t } ^ { j } - \widetilde { \underline { { \xi } } } _ { t } ^ { j } \| _ { 1 } \operatorname* { s u p } _ { s \in [ 0 , 1 ] } \| \nabla _ { \widetilde { \xi } _ { t } ^ { j } } \Delta _ { j } ( \widetilde { \boldsymbol { \sigma } } _ { s } ^ { j } ) \| _ { \infty } } } \\ & { \leq ( \sum _ { t = 1 } ^ { T } \ell _ { j , t } + 2 \Lambda _ { j , \varepsilon } \sum _ { t = 1 } ^ { T } \ell _ { j , t } ^ { a } ) \| o - \bar { \boldsymbol { \sigma } } \| _ { \mathrm { o c c } , \infty } \| \widetilde { \boldsymbol { \sigma } } ^ { j } - \underline { { \widetilde { \boldsymbol { \sigma } } } } ^ { j } \| _ { \mathrm { o c c } } } \\ & { = ( L _ { j } + 2 L _ { j } ^ { a } \Lambda _ { j , \varepsilon } ) \| o - \bar { \boldsymbol { \sigma } } \| _ { \mathrm { o c c } , \infty } \| \widetilde { \boldsymbol { \sigma } } ^ { j } - \widetilde { \underline { { \sigma } } } ^ { j } \| _ { \mathrm { o c c } } . } \end{array}
$$

Thus, the Lipschitz constant is

$$
\begin{array} { r } { \mathrm { L i p } ( \Delta _ { j } ) \leq \left( L _ { j } + 2 L _ { j } ^ { a } \Lambda _ { j , \varepsilon } \right) \left\| o - \bar { o } \right\| _ { \mathrm { o c c } , \infty } . } \end{array}\tag{77}
$$

For fixed $o \in \Theta$ and $\widehat { \mu } ^ { - j } \in \mathcal { W } _ { j } ( \xi ( o ) , \varepsilon )$ , the mapping

$$
\widetilde { \sigma } ^ { j } \longmapsto \sum _ { t = 0 } ^ { T } \sum _ { x \in \mathsf { X } } \widetilde { \xi } _ { t } ^ { j } ( x ) r _ { t } ^ { j } \big ( x , \xi _ { t } ^ { j } , \widehat { \mu } _ { t } ^ { - j } \big ) + \frac { \sigma } { 2 } \big \| \widehat { \mu } ^ { - j } \big \| _ { 2 } ^ { 2 }
$$

is afine. Therefore, $H _ { j , \sigma } ( \cdot ; o )$ , being the pointwise infimum of afine functions, is concave on ${ \cal O } ^ { j }$ . By Lemma 2, $\Psi _ { j , \beta }$ is �-strongly concave on ${ \mathsf { O } } ^ { j }$ with respect to ∥·∥<sub>occ</sub>. Hence, $\Phi _ { j , \beta } ^ { \sigma } ( \cdot ; o )$ is �-strongly concave on ${ \mathsf { O } } ^ { j }$ . Since ${ \mathsf { O } } ^ { j }$ is compact and $\Phi _ { j , \beta } ^ { \sigma } ( \cdot ; o )$ is continuous, $\mathsf { B r } _ { j , \beta } ^ { \sigma } ( o )$ is nonempty, and strong concavity implies that it is a singleton.

Let $z ^ { j } : = \mathsf { B r } _ { j , \beta } ^ { \sigma } ( o )$ , and $\bar { z } ^ { j } : = \mathsf { B r } _ { j , \beta } ^ { \sigma } ( \bar { o } )$ . The quadratic-growth consequence of strong concavity at a maximizer gives

$$
\Phi _ { j , \beta } ^ { \sigma } ( z ^ { j } ; o ) - \Phi _ { j , \beta } ^ { \sigma } ( \bar { z } ^ { j } ; o ) \geq \frac { \beta } { 2 } \left. z ^ { j } - \bar { z } ^ { j } \right. _ { \mathrm { o c c } } ^ { 2 } ,
$$

and

$$
\Phi _ { j , \beta } ^ { \sigma } ( \bar { z } ^ { j } ; \bar { \sigma } ) - \Phi _ { j , \beta } ^ { \sigma } ( z ^ { j } ; \bar { \sigma } ) \geq \frac { \beta } { 2 } \left. z ^ { j } - \bar { z } ^ { j } \right. _ { \mathrm { o c c } } ^ { 2 } .
$$

Adding the preceding inequalities and recalling the definition of $\Delta _ { j }$ , we obtain

$$
\begin{array} { r l } & { \beta \left\| z ^ { j } - \bar { z } ^ { j } \right\| _ { \mathrm { o c c } } ^ { 2 } \leq \Phi _ { j , \beta } ^ { \sigma } ( z ^ { j } ; o ) - \Phi _ { j , \beta } ^ { \sigma } ( \bar { z } ^ { j } ; o ) + \Phi _ { j , \beta } ^ { \sigma } ( \bar { z } ^ { j } ; \bar { o } ) - \Phi _ { j , \beta } ^ { \sigma } ( z ^ { j } ; \bar { o } ) } \\ & { \qquad = \Delta _ { j } ( \bar { z } ^ { j } ) - \Delta _ { j } ( z ^ { j } ) \leq \left( L _ { j } + 2 L _ { j } ^ { a } \Lambda _ { j , \varepsilon } \right) \| o - \bar { o } \| _ { \mathsf { o c c } , \infty } \left\| z ^ { j } - \bar { z } ^ { j } \right\| _ { \mathsf { o c c } } , } \end{array}
$$

where the last inequality follows from (77). $\operatorname { I f } { \boldsymbol { \ z } } ^ { j } = { \boldsymbol { \bar { z } } } ^ { j }$ , then (56) holds trivially. Otherwise, dividing the preceding inequality by $\beta \left\| z ^ { j } - { \bar { z } } ^ { j } \right\| _ { \mathrm { o c c } }$ yields

$$
\left\| \mathsf { B r } _ { j , \beta } ^ { \sigma } ( o ) - \mathsf { B r } _ { j , \beta } ^ { \sigma } ( \bar { o } ) \right\| _ { \mathrm { o c c } } \leq \frac { L _ { j } + 2 L _ { j } ^ { a } \Lambda _ { j , \varepsilon } } { \beta } \left\| o - \bar { o } \right\| _ { \mathrm { o c c } , \infty } ,
$$

which proves (56). Finally, defining $\begin{array} { r } { q _ { \beta , \varepsilon } : = \operatorname* { m a x } _ { j \in [ M ] } ( L _ { j } + 2 L _ { i } ^ { a } \Lambda _ { j , \varepsilon } ) / \beta } \end{array}$ and taking the maximum over $j \in [ M ]$ , we obtain

(78)

$$
\left. \mathsf { B r } _ { \beta } ^ { \sigma } ( o ) - \mathsf { B r } _ { \beta } ^ { \sigma } ( \bar { o } ) \right. _ { \mathrm { o c c } , \infty } = \displaystyle \operatorname* { m a x } _ { j \in [ M ] } \left. \mathsf { B r } _ { j , \beta } ^ { \sigma } ( o ) - \mathsf { B r } _ { j , \beta } ^ { \sigma } ( \bar { o } ) \right. _ { \mathrm { o c c } }\tag{79}
$$

$$
\leq \operatorname* { m a x } _ { j \in [ M ] } \frac { L _ { j } + 2 L _ { j } ^ { a } \Lambda _ { j , \varepsilon } } { \beta } \| o - \bar { o } \| _ { \mathsf { o c c } , \infty } = q _ { \beta , \varepsilon } \| o - \bar { o } \| _ { \mathsf { o c c } , \infty } .
$$

If $\begin{array} { r } { \beta > \operatorname* { m a x } _ { j \in [ M ] } \left( L _ { j } + 2 L _ { j } ^ { a } \Lambda _ { j , \varepsilon } \right) } \end{array}$ , then $q _ { \beta , \varepsilon } < 1$ , and hence $\mathsf { B r } _ { \beta } ^ { \sigma } : 0 \to 0$ is a contraction with respect to $\| \cdot \| _ { \mathrm { o c c , \infty } }$ , with the same contraction factor for every $\sigma > 0$ Step 4 (Passing to $\sigma = 0 )$ . Let $D : = ( M - 1 ) ( T + 1 )$ . Since every opponent state marginal is a probability vector, $\left\| \widehat { \mu } ^ { - j } \right\| _ { 2 } ^ { 2 } \leq D$ . Consequently, for every $j , o \in \mathsf { O }$ 9 $\widetilde { o } ^ { j } \in \bigcirc ^ { j }$ , and $\sigma > 0$

$$
0 \leq \Phi _ { j , \beta } ^ { \sigma } ( \widetilde { o } ^ { j } ; o ) - \Phi _ { j , \beta } ( \widetilde { o } ^ { j } ; o ) \leq \frac { \sigma D } { 2 } .\tag{80}
$$

Set $z _ { \sigma } ^ { j } : = \mathsf { B r } _ { j , \beta } ^ { \sigma } ( o )$ and $z _ { 0 } ^ { j } : = \mathsf { B r } _ { j , \beta } ( o )$ . By Lemma 2, the original best response is unique. Strong concavity of the original payof, optimality of $z _ { \sigma } ^ { j }$ for the perturbed payof, and (80) imply

$$
\frac { \beta } { 2 } \left\| z _ { \sigma } ^ { j } - z _ { 0 } ^ { j } \right\| _ { \mathrm { o c c } } ^ { 2 } \leq \Phi _ { j , \beta } ( z _ { 0 } ^ { j } ; o ) - \Phi _ { j , \beta } ( z _ { \sigma } ^ { j } ; o ) \leq \Phi _ { j , \beta } ^ { \sigma } ( z _ { \sigma } ^ { j } ; o ) - \Phi _ { j , \beta } ( z _ { \sigma } ^ { j } ; o ) \leq \frac { \sigma D } { 2 } .
$$

Thus,

$$
\operatorname* { s u p } _ { \sigma \in \mathbf { O } } \| \mathsf { B r } _ { \beta } ^ { \sigma } ( o ) - \mathsf { B r } _ { \beta } ( o ) \| _ { \mathsf { o c c } , \infty } \leq \delta _ { \sigma } : = \sqrt { \frac { \sigma D } { \beta } } \longrightarrow 0 \mathrm { ~ a s ~ } \sigma  0 ^ { + } .\tag{81}
$$

Combining this with (78) gives

$$
\begin{array} { r } { \left\| \mathsf { B r } _ { \beta } ( o ) - \mathsf { B r } _ { \beta } ( \bar { o } ) \right\| _ { \mathsf { o c c } , \infty } \leq 2 \delta _ { \sigma } + q _ { \beta , \varepsilon } \left\| o - \bar { o } \right\| _ { \mathsf { o c c } , \infty } . } \end{array}
$$

Letting $\sigma \to 0 ^ { + }$ shows that Br<sub>�</sub> satisfies the same Lipschitz estimate.

Step 5 (Fixed points and the equilibrium mean-field flow). Suppose now that $\beta > \mathrm { m a x } _ { j \in [ M ] } ( L _ { j } + 2 L _ { i } ^ { a } \Lambda _ { j , \varepsilon } )$ , so $q _ { \beta , \varepsilon } < 1$ . The set O is nonempty and compact, hence complete under $\| \cdot \| _ { \mathrm { o c c } , \infty } .$ Banach’s fixed-point theorem gives unique fixed points $o _ { \beta } ^ { \sigma , \star }$ and $o _ { \beta } ^ { \star }$ of $\mathsf { B r } _ { \beta } ^ { \sigma }$ and $\mathsf { B r } _ { \beta }$ , respectively. Moreover, (81) and the contraction estimate yield

$$
\begin{array} { r l } & { \left\| \sigma _ { \beta } ^ { \sigma , \star } - \sigma _ { \beta } ^ { \star } \right\| _ { \mathrm { o c c } , \infty } = \left\| \mathsf { B r } _ { \beta } ^ { \sigma } \big ( \sigma _ { \beta } ^ { \sigma , \star } \big ) - \mathsf { B r } _ { \beta } \big ( \sigma _ { \beta } ^ { \star } \big ) \right\| } \\ & { \qquad \leq \left\| \mathsf { B r } _ { \beta } ^ { \sigma } \big ( \sigma _ { \beta } ^ { \sigma , \star } \big ) - \mathsf { B r } _ { \beta } \big ( \sigma _ { \beta } ^ { \sigma , \star } \big ) \right\| + \left\| \mathsf { B r } _ { \beta } \big ( \sigma _ { \beta } ^ { \sigma , \star } \big ) - \mathsf { B r } _ { \beta } \big ( \sigma _ { \beta } ^ { \star } \big ) \right\| } \\ & { \qquad \leq \delta _ { \sigma } + q _ { \beta , \varepsilon } \left\| \sigma _ { \beta } ^ { \sigma , \star } - \sigma _ { \beta } ^ { \star } \right\| _ { \mathrm { o c c } , \infty } , } \end{array}
$$

and therefore

$$
\| o _ { \beta } ^ { \sigma , \star } - o _ { \beta } ^ { \star } \| _ { \mathrm { o c c } , \infty } \leq \frac { 1 } { 1 - q _ { \beta , \varepsilon } } \sqrt { \frac { \sigma D } { \beta } } \longrightarrow 0 \mathrm { ~ a s ~ } \sigma  0 ^ { + } .
$$

The corresponding mean-field flows converge as well, since $\xi$ is linear and continuous.

By the policy–occupation-measure equivalence and the uniqueness of optimal occupations from Lemma 2, the composite operator $\Gamma _ { \beta }$ is single-valued on feasible flows and satisfies

$$
\xi { \big ( } \mathsf { B r } _ { \beta } ( o ) { \big ) } = \Gamma _ { \beta } { \big ( } \xi ( o ) { \big ) } { \mathrm { ~ f o r ~ e v e r y ~ } } o \in \mathsf { O } .
$$

For any feasible initial flow $\mu _ { 0 }$ , choose $o ^ { 0 } \in \mathsf { O }$ with $\xi ( o ^ { 0 } ) = \mu _ { 0 }$ , and set $o ^ { n + 1 } \ =$ $\mathsf { B r } _ { \beta } ( o ^ { n } )$ . Banach’s theorem gives $o ^ { n } \to o _ { \beta } ^ { \star } ;$ the preceding identity then shows that $\mu ^ { n } : = \xi ( o ^ { n } )$ satisfies $\mu ^ { n + 1 } = \Gamma _ { \beta } ( \mu ^ { n } )$ and converges to $\mu _ { \beta } ^ { \star } : = \xi ( o _ { \beta } ^ { \star } )$

Reconstructing a policy from $o _ { \beta } ^ { \star } ,$ , feasibility gives mean-field consistency and the fixed-point condition gives entropy-regularized risk-averse optimality. Thus the reconstructed policy and $\mu _ { \beta } ^ { \star }$ satisfy Definition 8. Conversely, every equilibrium in Definition 8 induces a fixed point of $\mathsf { B r } _ { \beta }$ . Uniqueness of $o _ { \beta } ^ { \star }$ therefore gives uniqueness of the equilibrium occupation profile and mean-field flow. □

Remark 8 (Interpretation ofthe parameters and sensitivity constants). Theparameters �, �, and $\sigma$ serve diferent purposes. Recall the expression of the ambiguity set $\mathcal { W } _ { j } ( \mu , \varepsilon )$ in (5). The ambiguity radius � specifies the admissible deviations of the opponents’ flows. For a fixed policy and nominal flow, increasing � enlarges the ambiguity set and can only decrease the worst-case reward. Thus, � controls the conservatism of the decision criterion. The regularization parameter � in Definition 7, weights the entropy reward, encouraging randomized policies and providing the strong concavity used in the contraction analysis. As emphasized before entropy regularization models bounded rationality in mean-field games [MPS25].

The constants $L _ { j }$ and $L _ { j } ^ { a }$ in Assumptions 2 and 3 respectively, quantify the sensitivity ofpopulation �’s reward to its own population distribution and the opponents’ distributions, respectively. The constant $\Lambda _ { j , \varepsilon }$ bounds the sensitivity ofthe inner minimizingflow to changes in the nominal profile, with the candidate occupation measure held fixed. It depends on the geometry of the dynamically feasible ambiguity sets and is independent of �. These are problem-dependent bounds. Under its stated assumptions, Theorem 4 certifies contraction when

$$
q _ { \beta , \varepsilon } : = \frac { \operatorname* { m a x } _ { j \in [ M ] } ( L _ { j } + 2 L _ { j } ^ { a } \Lambda _ { j , \varepsilon } ) } { \beta } < 1 .
$$

Increasing � therefore improves the certified contraction factor, but increases the entropy perturbation of the original game. The displayed condition is suficient and may be conservative. When $\varepsilon = 1$ , the ambiguity set is independent of the nominal opponents’ flows, and Theorem 3 gives the simpler factor max<sub>�</sub> $L _ { j } / \beta .$

The parameter � in (16) makes the inner problem strongly convex, ensuring a unique minimizing flow and enabling diferentiation of its optimal value. Positive $\sigma$ also perturbs the payof; taking � small reduces this perturbation. The contraction estimate is uniform in $\sigma > 0$ and remains valid at $\sigma = 0 ;$ the theorem therefore imposes no requirement to increase $\beta$ when � increases.

## A<sub>ppe</sub>ndi<sub>x</sub> G<sub>.</sub> Fi<sub>c</sub>titi<sub>ous</sub> Pl<sub>ay</sub> <sub>a</sub>nd Pr<sub>oo</sub>f <sub>o</sub>f Th<sub>eo</sub>r<sub>e</sub>m 5

We provide some additional details about the risk-averse fictitious play algorithm introduced in Section 5. Recall from Section 5 that the FP updates iteratively compute a best response to the population distribution induced by the average of the best responses from previous iterations. Specifically, for each population $j \in [ M ]$ , each time step $t = 0 , \ldots , T$ , and iteration index $\ell \geq 1$ , we have [PPL<sup>+</sup>20]:

$$
\bar { \mu } _ { t } ^ { j , \ell } = \frac { \ell - 1 } { \ell } \bar { \mu } _ { t } ^ { j , \ell - 1 } + \frac { 1 } { \ell } \mu _ { t } ^ { \pi ^ { j , \ell } } .\tag{82}
$$

The policy generating this average distribution ${ \mathrm { i s } } ,$ for $t ~ = ~ 0 , \ldots , T - 1$ , given by [PPL<sup>+</sup>20]:

$$
\bar { \pi } _ { t } ^ { j , \ell } ( u | x ) = \frac { \sum _ { i = 1 } ^ { \ell } \mu _ { t } ^ { \pi ^ { j , i } } ( x ) \pi _ { t } ^ { j , i } ( u | x ) } { \sum _ { i = 1 } ^ { \ell } \mu _ { t } ^ { \pi ^ { j , i } } ( x ) } ,\tag{83}
$$

where $\bar { \pi } _ { t } ^ { j , \ell } ( u | x )$ is the weighted average of the best responses of population � from iteration 1 up to iteration ℓ. The ratio is defined whenever the denominator is positive; otherwise, $\bar { \pi } _ { t } ^ { j , \ell } ( \cdot | x )$ may be chosen arbitrarily.

Fix an arbitrary population $j \in [ M ]$ . At the starting point, $0 \leq s < 1$ , let us fix an initial policy for population � as ${ \bar { \pi } ^ { j , s < 1 } } = \left\{ \bar { \pi } _ { t } ^ { j , s < 1 } \right\} _ { t } = \left\{ \pi _ { t } ^ { j , s < 1 } \right\}$ with induced distribution $\bar { \mu } ^ { j , s < 1 } = \mu ^ { j , s < 1 } = \mu ^ { \pi ^ { j , s < 1 } } = \left\{ \mu _ { t } ^ { \pi ^ { j , s < 1 } } \right\}$ [PPL<sup>+</sup>20]. Moving forward, we drop the superscript � for brevity since the analysis remains consistent across all populations $j ~ \in ~ [ M ]$ . The continuous-time fictitious play process is defined for all $s \geq 1$ and $t \in \{ 0 , \ldots , T \}$ . For each fictitious play time $s \geq 1$ , let $\pi ^ { \mathsf { B R } , { \boldsymbol { s } } }$ denote the $j ^ { \mathrm { t h } }$ component of a measurably selected risk-averse best response, i.e., $\pi ^ { \mathsf { B R } , s } \in \mathcal { B } _ { \mathrm { r i s k - o p t } } ^ { j } ( \bar { \mu } ^ { s } )$ , where $\bar { \mu } ^ { s }$ in this argument denotes the joint population flow. Let $\mu ^ { \mathsf { B R } , s } = \mathcal { B } _ { \mathsf { p r o p } } ^ { j } ( \pi ^ { \mathsf { B R } , s } )$ denote the corresponding induced mean-field flow. The averaged flow satisfies

$$
\frac { \mathrm { d } } { \mathrm { d } s } \bar { \mu } _ { t } ^ { s } ( x ) = \frac { 1 } { s } \left( \mu _ { t } ^ { \mathsf { B R } , s } ( x ) - \bar { \mu } _ { t } ^ { s } ( x ) \right) \mathrm { ~ e q u i v a l e n t l y , ~ } \bar { \mu } _ { t } ^ { s } ( x ) = \frac { 1 } { s } \int _ { 0 } ^ { s } \mu _ { t } ^ { \mathsf { B R } , z } ( x ) \mathrm { d } z ,\tag{84}
$$

where the diferential identity holds for almost every $s \geq 1$ , while the integral identity holds for every $s \geq 1$ . Here $\mu _ { t } ^ { \mathsf { B R } , s }$ is induced by a best response policy $\{ \pi _ { t } ^ { \mathsf { B R } , s } \}$ � computed against the joint averaged population flow. From $[ \mathbf { P } \mathbf { P } \mathbf { L } ^ { + } 2 \mathbf { 0 } ]$ , it also follows that

$$
\mathrm { f o r } \ t = 0 , \dots , T - 1 , \ \bar { \mu } _ { t } ^ { s } ( x ) \frac { \mathrm { d } } { \mathrm { d } s } \bar { \pi } _ { t } ^ { s } ( u | x ) = \frac { 1 } { s } \mu _ { t } ^ { \mathsf { B R } , s } ( x ) \left[ \pi _ { t } ^ { \mathsf { B R } , s } ( u | x ) - \bar { \pi } _ { t } ^ { s } ( u | x ) \right]\tag{85}
$$

$$
\mathrm { e q u i v a l e n t l y } , \bar { \pi } _ { t } ^ { s } ( u | x ) \int _ { 0 } ^ { s } \mu _ { t } ^ { \mathsf { B R } , z } ( x ) \mathrm { d } z = \int _ { 0 } ^ { s } \mu _ { t } ^ { \mathsf { B R } , z } ( x ) \pi _ { t } ^ { \mathsf { B R } , z } ( u | x ) \mathrm { d } z .
$$

For $0 \leq z < 1 , \pi ^ { \mathsf { B R } , z }$ and $\mu ^ { \tt B R , z }$ are taken to be the fixed initial policy and its induced flow, respectively.

We now restate and provide a detailed proof of Theorem 5. Recall the expression of the risk-averse exploitability in (36) and (37).

Theorem 5. Let the RA-MP-MFG satisfy Assumptions 1 and $^ { 4 , }$ and let $\varepsilon = 1$ in (27). Consider the continuous-time fictitious play process $\{ \bar { \pi } ^ { s } , \bar { \mu } ^ { s } \} _ { s \geq 1 }$ defined in (84) and (85). Then, for every $j \in [ M ]$ the map $s \mapsto \phi _ { \mathsf { R A } } ^ { J } ( \bar { \pi } ^ { s } )$ is locally absolutely continuous and satisfies

$$
\frac { \mathrm { d } } { \mathrm { d } s } \phi _ { \mathsf { R A } } ^ { j } ( \bar { \pi } ^ { s } ) \leq - \frac { 1 } { s } \phi _ { \mathsf { R A } } ^ { j } ( \bar { \pi } ^ { s } ) \quad f o r a l m o s t e \nu e r y s \geq 1 .
$$

Consequently, the exploitability is a strong Lyapunov function along these trajectories and $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } s } \phi _ { \mathsf { R A } } ( \bar { \pi } ^ { s } ) \leq - \frac { 1 } { s } \phi _ { \mathsf { R A } } ( \bar { \pi } ^ { s } ) } \end{array}$ for almost every $s \geq 1$ and $\phi _ { \mathsf { R A } } ( \bar { \pi } ^ { s } ) =$ $\Theta ( 1 / s )$

Proof. Fix $j ~ \in ~ [ M ]$ . Write $\pi ^ { s } : = \bar { \pi } ^ { j , s }$ and $\mu ^ { s } ~ : = ~ \bar { \mu } ^ { j , s }$ , and suppress the population index in $\pi ^ { \mathsf { B R } , s } , \mu ^ { \mathsf { B R } , s } , r _ { t } , a _ { t }$ , and $b _ { t }$ . The symbols $\bar { \pi } ^ { s }$ and $\bar { \mu } ^ { s }$ continue to denote the joint profiles. From (84), we have

$$
\frac { \mathrm { d } } { \mathrm { d } s } \mu _ { t } ^ { s } ( x ) = \frac { 1 } { s } \left[ \mu _ { t } ^ { \mathsf { B R } , s } ( x ) - \mu _ { t } ^ { s } ( x ) \right] \quad \mathrm { ~ f o r ~ a l m o s t \ e v e r y ~ } s \ge 1 .\tag{86}
$$

Since $\varepsilon \ = \ 1$ , the ambiguity set is independent of the nominal flow, and hence $\mathcal { W } _ { j } ( \bar { \mu } ^ { s } , 1 ) = \mathcal { W } _ { j } ( \bar { \mu } ^ { 1 } , 1 )$ for every $s \geq 1$ . For each $s \geq 1$ , let $\pi ^ { \mathsf { B R } , s } \in \mathcal { B } _ { \mathrm { r i s k - o p t } } ^ { j } ( \bar { \mu } ^ { s } )$ be the selected risk-averse best response, with induced flow $\mu ^ { \mathsf { B R } , s } = \mathcal { B } _ { \mathrm { p r o p } } ^ { j } ( \pi ^ { \mathsf { B R } , s } )$ , and choose

$$
\nu ^ { \mathsf { B R } , s } \in \mathop { \arg \operatorname* { m i n } } _ { \widehat { \mu } ^ { - j } \in \mathcal { W } _ { j } ( \bar { \mu } ^ { 1 } , 1 ) } \sum _ { t = 0 } ^ { T } \sum _ { x \in \mathsf { X } } \mu _ { t } ^ { \mathsf { B R } , s } ( x ) r _ { t } \left( x , \mu _ { t } ^ { s } , \widehat { \mu _ { t } ^ { - j } } \right) .
$$

Similarly, choose a worst-case trajectory for the current policy

$$
\nu ^ { s } \in \underset { \widehat { \mu } ^ { - j } \in \mathcal { W } _ { j } ( \bar { \mu } ^ { 1 } , 1 ) } { \arg \operatorname* { m i n } } \sum _ { t = 0 } ^ { T } \sum _ { x \in \mathsf { X } } \mu _ { t } ^ { s } ( x ) r _ { t } \big ( x , \mu _ { t } ^ { s } , \widehat { \mu } _ { t } ^ { - j } \big ) .
$$

These optimizers exist by compactness of the feasible sets and continuity under Assumption 1. Consequently,

$$
\begin{array} { l } { \phi _ { \mathsf { R A } } ^ { j } ( \bar { \pi } ^ { s } ) = \displaystyle \operatorname* { m a x } _ { \tilde { \pi } \in \Pi ^ { j } } \bar { \mathcal { J } } _ { j } ( \tilde { \pi } ; \bar { \mu } ^ { s } ) - \bar { \mathcal { J } } _ { j } ( \pi ^ { s } ; \bar { \mu } ^ { s } ) } \\ { \displaystyle \quad = \sum _ { t = 0 } ^ { T } \displaystyle \sum _ { x \in \mathsf { X } } \Big [ \mu _ { t } ^ { \mathsf { B R } , s } ( x ) r _ { t } \big ( x , \mu _ { t } ^ { s } , \nu _ { t } ^ { \mathsf { B R } , s } \big ) - \mu _ { t } ^ { s } ( x ) r _ { t } \big ( x , \mu _ { t } ^ { s } , \nu _ { t } ^ { s } \big ) \Big ] . } \end{array}
$$

We first justify the diferentiation of these optimized values. We record them via the following steps:

(Dif-a) The map $z \mapsto \mu _ { t } ^ { \mathsf { B R } , z }$ is measurable and bounded, since each $\mu _ { t } ^ { \mathsf { B R } , z }$ is a probability distribution. Therefore, its indefinite integral is locally absolutely continuous [Fol99, Chapter 3, Theorem 3.35]. Multiplication by the continuously diferentiable function $s \mapsto 1 / s$ preserves absolute continuity on compact intervals in $[ 1 , + \infty )$ . Thus, the integral representation in (84) implies that $\mu ^ { s }$ is locally absolutely continuous.

(Dif-b) Moreover, (86) gives

$$
\left\| \frac { \mathrm { d } } { \mathrm { d } s } \mu _ { t } ^ { s } \right\| _ { 1 } = \frac { 1 } { s } \left\| \mu _ { t } ^ { \mathsf { B R } , s } - \mu _ { t } ^ { s } \right\| _ { 1 } \leq \frac { 2 } { s } \quad \mathrm { f o r ~ a l m o s t ~ e v e r y ~ } s \geq 1 .
$$

Using the fundamental theorem of calculus for absolutely continuous functions [Fol99, Chapter 3], we consequently obtain, for $s \geq r \geq s _ { 0 } \geq 1$

$$
\left\| \mu _ { t } ^ { s } - \mu _ { t } ^ { r } \right\| _ { 1 } \leq \int _ { r } ^ { s } \left\| { \frac { \mathrm { d } } { \mathrm { d } z } } \mu _ { t } ^ { z } \right\| _ { 1 } \mathrm { d } z \leq { \frac { 2 } { s _ { 0 } } } | s - r | .\tag{87}
$$

Hence, $\mu ^ { s }$ is locally Lipschitz.

(Dif-c) Next, let $R _ { j } < + \infty$ be a uniform bound on $| r _ { t } ( \cdot ) |$ , which exists by Assumption 1 and compactness, and recall that $L _ { r } ^ { j }$ was the Lipschitz constant in Assumption 1. Since $\varepsilon = 1$ , the ambiguity set is fixed, and the nominal population flow enters the payof only through $\mu ^ { s }$ . For a fixed candidate policy ${ \tilde { \pi } } ,$ , its induced flow is independent of the nominal flow. The reward Lipschitz bound therefore applies uniformly over all candidate policies and feasible opponent flows. Taking an infimum (or supremum) over a fixed set preserves such a uniform bound, and consequently,

$$
\begin{array} { l } { \displaystyle \left| \underset { \bar { \pi } \in \Pi ^ { j } } { \operatorname* { m a x } } { \bar { \mathcal { J } } } _ { j } ( \tilde { \pi } ; \bar { \mu } ^ { s } ) - \underset { \bar { \pi } \in \Pi ^ { j } } { \operatorname* { m a x } } { \bar { \mathcal { J } } } _ { j } ( \tilde { \pi } ; \bar { \mu } ^ { r } ) \right| \leq L _ { r } ^ { j } \displaystyle \sum _ { t = 0 } ^ { T } \left\| \mu _ { t } ^ { s } - \mu _ { t } ^ { r } \right\| _ { 1 } , } \\ { \displaystyle \left| \bar { \mathcal { J } } _ { j } ( \pi ^ { s } ; \bar { \mu } ^ { s } ) - \bar { \mathcal { J } } _ { j } ( \pi ^ { r } ; \bar { \mu } ^ { r } ) \right| \leq ( R _ { j } + L _ { r } ^ { j } ) \displaystyle \sum _ { t = 0 } ^ { T } \left\| \mu _ { t } ^ { s } - \mu _ { t } ^ { r } \right\| _ { 1 } . } \end{array}\tag{88}
$$

For the second inequality, we use $\mu ^ { s } = \mathcal { B } _ { \mathrm { p r o p } } ^ { j } ( \pi ^ { s } )$ and compare the trajectory objectives at a fixed feasible opponent flow. Changing the state probabilities contributes at most $\begin{array} { r } { R _ { j } \sum _ { t = 0 } ^ { T } \| \dot { \mu _ { t } ^ { s } } - \mu _ { t } ^ { r } \| _ { 1 } } \end{array}$ , while changing the reward arguments contributes at most $\begin{array} { r } { L _ { r } ^ { j } \sum _ { t = 0 } ^ { T } \| \mu _ { t } ^ { s } - \mu _ { t } ^ { r } \| _ { 1 } } \end{array}$

Combining (87) and (88) shows that both optimized values are locally Lipschitz in �. Their diference, $\phi _ { \mathsf { R A } } ^ { j } ( \bar { \pi } ^ { s } )$ , is also locally Lipschitz by the triangle inequality. Thus, both optimized values and their diference are locally absolutely continuous, since every Lipschitz function on a compact interval is absolutely continuous [RF88, Section 6.4, Proposition 7].

Fix a time $s > 1$ at which (86) holds and the flow and both optimized values are diferentiable. Keeping $\pi ^ { \mathsf { B R } , s }$ fixed and using $r _ { t } ( x , \mu , \nu ) = a _ { t } ( x , \nu ) + b _ { t } ( x , \mu )$ , we obtain, for suficiently small ℎ of either sign,

$$
\begin{array} { l } { \displaystyle \operatorname* { m a x } _ { \tilde { \pi } \in \Pi ^ { j } } \bar { \mathcal { J } } _ { j } ( \tilde { \pi } ; \bar { \mu } ^ { s + h } ) - \operatorname* { m a x } _ { \tilde { \pi } \in \Pi ^ { j } } \bar { \mathcal { J } } _ { j } ( \tilde { \pi } ; \bar { \mu } ^ { s } ) \geq \bar { \mathcal { J } } _ { j } ( \pi ^ { \mathsf { B R } , s } ; \bar { \mu } ^ { s + h } ) - \bar { \mathcal { J } } _ { j } ( \pi ^ { \mathsf { B R } , s } ; \bar { \mu } ^ { s } ) } \\ { \displaystyle \qquad = \sum _ { t = 0 } ^ { T } \sum _ { x \in \mathsf { X } } \mu _ { t } ^ { \mathsf { B R } , s } ( x ) \left[ b _ { t } ( x , \mu _ { t } ^ { s + h } ) - b _ { t } ( x , \mu _ { t } ^ { s } ) \right] . } \end{array}
$$

Similarly, keeping $\nu ^ { s }$ fixed gives

$$
\begin{array} { l } { \displaystyle \bar { \mathcal { J } } _ { j } ( \pi ^ { s + h } ; \bar { \mu } ^ { s + h } ) - \bar { \mathcal { J } } _ { j } ( \pi ^ { s } ; \bar { \mu } ^ { s } ) } \\ { \displaystyle \qquad \le \sum _ { t = 0 } ^ { T } \sum _ { x \in \mathsf X } \left[ \mu _ { t } ^ { s + h } ( x ) r _ { t } ( x , \mu _ { t } ^ { s + h } , \nu _ { t } ^ { s } ) - \mu _ { t } ^ { s } ( x ) r _ { t } ( x , \mu _ { t } ^ { s } , \nu _ { t } ^ { s } ) \right] . } \end{array}
$$

Dividing these inequalities by ℎ and taking limits from both sides justifies differentiating the optimized values while holding the selected optimizers fixed. Using

$\nabla _ { \mu } r _ { t } ( x , \mu , \nu ) = \nabla _ { \mu } b _ { t } ( x , \mu )$ and (86), we therefore obtain, for almost every $s \geq 1$

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } } { \mathrm { d } t } \phi _ { t } ^ { \lambda } ( u ^ { ( T ) } ) = \displaystyle \sum _ { i = 1 } ^ { T } \sum _ { j = 1 } ^ { N } | R _ { i } ^ { \mathrm { R } _ { i } ^ { \mathrm { R } _ { j } } , \mathrm { R } _ { j } ^ { \mathrm { R } _ { j } } } u ^ { \mathrm { R } _ { j } } | ^ { 2 } } \\ { \displaystyle - \theta _ { i } ^ { \lambda } ( \Delta t ) \{ \nabla _ { i j } r _ { i } ( \Delta , \mu ^ { ( T ) } ) , \frac { \mathrm { d } } { \mathrm { d } t } \theta _ { i } ^ { \lambda } \} - r _ { j } ( \mathcal { A } , \mu ^ { ( T ) } , \nu _ { i } ^ { ( T ) } ) \frac { \mathrm { d } } { \mathrm { d } t } \theta _ { i } ^ { \lambda } / ( \Delta ) } \\ { = \displaystyle \sum _ { i = 1 } ^ { T } \sum _ { j = 1 } ^ { N } \{ \nabla _ { i j } h _ { i } ( \nu _ { i } , \mu ^ { ( T ) } ) , \frac { \mathrm { d } } { \mathrm { d } t } \theta _ { i } ^ { \lambda } \} \frac { \mathrm { d } } { \mathrm { d } t } \mu _ { i } ^ { \lambda } / ( \Delta ) } \\ { \displaystyle - \frac { 1 } { \mathrm { d } } \sum _ { j = 1 } ^ { T } \sum _ { i = 1 } ^ { N } r _ { i } ( \mathcal { A } , \mu ^ { ( T ) } , \nu _ { i } ^ { ( T ) } ) | R _ { i } ^ { \mathrm { R } _ { i } ^ { \mathrm { R } _ { j } } , \mathrm { R } _ { j } } - \mu _ { i } ^ { \lambda } ( \mathcal { A } ) | } \\  \displaystyle \frac { \mathrm { d } } { \mathrm { d } t } - \frac { 1 } { \mathrm { d } } \sum _ { j = 1 } ^ { N } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \{ \nabla _ { j } h _ { i } ( \Delta , \mu ^ { ( T ) } ) , \frac { \mathrm { d } } \end{array}
$$

In (†), we employed the expression for the exploitability and our reward decomposition. The second term in (†) is nonpositive by Assumption 4; we refer to $[ \mathbf { P } \mathbf { P } \mathbf { L } ^ { + } 2 \mathbf { 0 } ]$ for more details. The third term in (†) is also nonpositive by definition of the $\mathrm { \dot { \Omega } _ { i n f } } ,$ operator. Consequently, $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } s } \phi _ { \mathsf { R A } } ^ { j } ( \bar { \pi } ^ { s } ) \leq - \frac { 1 } { s } \phi _ { \mathsf { R A } } ^ { j } ( \bar { \pi } ^ { s } ) } \end{array}$ for almost every $s \geq 1$ . By local absolute continuity, this implies $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } s } \left\lceil s \phi _ { \mathsf { R A } } ^ { j } ( \bar { \pi } ^ { s } ) \right\rceil \leq 0 } \end{array}$ a.e., and summing over $j \in [ M ]$ we obtain $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } s } \phi _ { \mathsf { R A } } ( \bar { \pi } ^ { s } ) \leq - \frac { 1 } { s } \phi _ { \mathsf { R A } } ( \bar { \pi } ^ { s } ) } \end{array}$ a.e. This proves our claim. □

Remark 9. Observe that Assumption 4 imposes a monotonicity condition on the own-population component $b _ { t } ^ { j }$ ofthe rewardfunction and does not restrict the reward component depending on the opponent-dependent populations $a _ { t } ^ { j } .$ It is therefore weaker (compared to the existing literature) than requiring monotonicity of the full multi-population reward with respect to the entire mean-field profile.

## A<sub>ppe</sub>ndi<sub>x</sub> H<sub>.</sub> N<sub>u</sub>m<sub>e</sub>ri<sub>ca</sub>l E<sub>xpe</sub>rim<sub>e</sub>nt<sub>s:</sub> Additi<sub>o</sub>n<sub>a</sub>l D<sub>e</sub>t<sub>a</sub>il<sub>s</sub>

In this section, we provide additional details about the numerical examples presented in the main text (Section 6). In particular, we discuss in detail the dual formulation of the risk-averse objective that simplifies computation by enabling the use of of-the-shelf optimization solvers and the hyperparameters used for the experiments. Note that all computations are performed on the CPU, in particular, macOS 14.1. Further, the pseudocodes of the proposed risk-averse entropy-regularized fixed point iteration and risk-averse fictitious play algorithms are presented in Algorithms 1 and 2 respectively. Note that we use Polyak averaging for the policy update in Algorithm 1 to ensure smooth and stable updates $[ \mathbf { L P P ^ { + } } 2 4 ]$

H.1. Grid Navigation. In this example, both populations traverse a $5 { \times } 5$ grid with an action space corresponding to {LEFT, RIGHT, UP, DOWN, STAY} and $T = 1 2$ The populations transition deterministically.

Population 1, at a given state $x _ { t } ^ { 1 } \in \mathsf { X } ,$ receives $r _ { t } ^ { 1 } ( x _ { t } ^ { 1 } , \mu _ { t } ^ { 1 } , \mu _ { t } ^ { 2 } ) : = \mu _ { t } ^ { 2 } ( x _ { t } ^ { 1 } )$ . We assume that Population 1 is risk-averse to deviations in Population 2’s trajectory. The reward $r _ { t } ^ { 1 } ( \cdot )$ is linear in $\mu _ { t } ^ { 2 }$ and $\varepsilon = 1 , \mathrm { i . e . }$ ., the ambiguity set is independent of the nominal trajectory. We seek to find a policy that maximizes the risk-averse payof (Definition 4) for a fixed joint mean-field flow $\mu ,$ i.e., arg max $\pi ^ { 1 } { \in } \Pi ^ { 1 } \bar { \mathcal { J } } _ { 1 } ( \pi ^ { 1 } ; \mu )$ . The

Algorithm 1: Fixed Point Iteration for RA-MP-MFGs   
Input: MP-MFG instance, entropy-regularization parameter $\beta ,$ damping   
factor �   
Output: Policy profile �   
1 Initialize $\pi ^ { \ell = 0 } = \{ \pi _ { t } ^ { j , \ell = 0 } \} _ { t \in \{ 0 , \ldots , T - 1 \} , j \in [ M ] }$   
2 f<sub>or</sub> $\ell = 0 , 1 , \ldots$ do   
3 $\mu ^ { \ell } = \mathcal { B } _ { \mathrm { p r o p } } ( \pi ^ { \ell } )$   
4 $\pi ^ { \ell + 1 } = \mathcal { \bar { B } } _ { \mathrm { r i s k - o p t } , \beta } ( \mu ^ { \ell } )$   
5 $\pi ^ { \ell + 1 } = \tau \pi ^ { \prime } + ( 1 - \tau ) \pi ^ { \ell }$   
6 end   
7 return �

Algorithm 2: Fictitious Play for RA-MP-MFGs   
Input: MP-MFG instance   
Output: Policy profile �¯   
1 Initialize $\pi ^ { \ell = 0 } = \{ \pi _ { t } ^ { j , \ell = 0 } \} _ { t \in \{ 0 , \dots , T - 1 \} , j \in [ M ] }$   
2 $\bar { \pi } ^ { \ell = 0 } = \pi ^ { \ell = 0 }$   
3 $\bar { \mu } ^ { \ell = 0 } = \mathcal { B } _ { \mathrm { p r o p } } ( \pi ^ { \ell } )$   
4 f<sub>or</sub> $\ell = 0 , 1 , \ldots$ do   
5 $\pi ^ { \ell + 1 } = \mathcal { B } _ { \mathrm { r i s k - o p t } } ( \bar { \mu } ^ { \ell } )$   
6 $\mu ^ { \ell + 1 } = \mathcal { B } _ { \mathrm { p r o p } } ( \bar { \pi } ^ { \ell + 1 } )$   
7 $\bar { \pi } ^ { \ell + 1 } = \arg ( \bar { \pi ^ { 0 } } , \ldots , \pi ^ { \ell + 1 } )$ using (83)   
8 $\bar { \mu } ^ { \ell + 1 } = \arg ( \mu ^ { 0 } , \ldots , \mu ^ { \ell + 1 } )$ using (82)   
9 end   
10 return �¯

ensuing optimization problem is

$$
\underset { \pi ^ { 1 } \in \Pi ^ { 1 } } { \arg \operatorname* { m a x } } \bar { \mathcal { J } } _ { 1 } ( \pi ^ { 1 } ; \mu ) = \underset { \pi ^ { 1 } \in \Pi ^ { 1 } } { \arg \operatorname* { m a x } } \ \underset { \hat { \mu } ^ { 2 } \in \mathcal { W } _ { 1 } ( \mu , 1 ) } { \operatorname* { i n f } } \in { \boldsymbol { \pi } } ^ { 1 } \left[ \sum _ { t = 0 } ^ { T } \hat { \mu } _ { t } ^ { 2 } ( x _ { t } ^ { 1 } ) \right]\tag{89}
$$

By exploiting duality together with the occupation-measure representation, (89) can be reformulated using the dual variables $V : = \{ V _ { t } \} _ { t = 0 } ^ { T }$ such that $V _ { t } \in \mathbb { R } ^ { | \times | }$ , and consequently we have

```html
max ∑︁ �<sup>2</sup><sub>0</sub> (�)�<sub>0</sub> (�)
<sub>�</sub><sup>1</sup>≥0, � �
Í<sub>� �</sub><sup>1</sup> (�, �) = <sub>�</sub><sup>1</sup> (�), for all �,
Í<sub>� �</sub><sup>1</sup> (<sub>�</sub>, �) = Í<sub>� � �</sub><sup>1</sup><sub>�</sub> (�, �) �<sup>1</sup><sub>�</sub> (<sub>�</sub> | �, �), for all �, �,
s. t.
�<sub>�</sub> (�) ≤ Í<sub>� �</sub><sup>1</sup> (�, �), for all �,
(Vt(x) ≤ Σu η1 (x, u) + Σy f2(y | x, u)Vt+1(y), for all x, u,t.
```

We use MOSEK [ApS19] to solve (90). The policy-occupation-measure equivalence allows us to recover the optimal policy from $\eta ^ { 1 , * }$ , the maximizer of (90). The optimal policy for the risk-neutral Population 2 can be computed using existing algorithms for MP-MFGs, and in our work we employ online mirror descent (OMD) [PPE<sup>+</sup>21]. In our example, for Population 2, the OMD parameter $\alpha = 0 . 0 5$ Under entropy regularization, (90) can be modified to include the regularization (under

parameter $\beta )$ as

$$
\begin{array} { r l } { \underset { \eta ^ { 1 } \geq 0 , V } { \operatorname* { m a x } } } & { ~ \displaystyle \sum _ { x } \mu _ { 0 } ^ { 2 } ( x ) V _ { 0 } ( x ) - \beta \displaystyle \sum _ { t = 0 } ^ { T - 1 } \displaystyle \sum _ { x , u } \eta _ { t } ^ { 1 } ( x , u ) \log \left( \frac { \eta _ { t } ^ { 1 } ( x , u ) } { \sum _ { u ^ { \prime } } \eta _ { t } ^ { 1 } ( x , u ^ { \prime } ) } \right) } \\ & { ~ \displaystyle \left( \sum _ { u } \eta _ { 0 } ^ { 1 } ( x , u ) = \mu _ { 0 } ^ { 1 } ( x ) , \right. } \\ { \mathrm { s . ~ t . ~ } } & { ~ \left. _ { V _ { T } } \left( x \right) \leq \sum _ { u } \eta _ { T } ^ { 1 } ( x , u ) , \right. } \\ & { ~ \left. \left[ V _ { T } ( x ) \leq \sum _ { u } \eta _ { T } ^ { 1 } ( x , u ) , \right. \right. } \\ & { \left. \left. V _ { t } ( x ) \leq \sum _ { u } \eta _ { t } ^ { 1 } ( x , u ) + \sum _ { y } f _ { t } ^ { 2 } ( y \mid x , u ) V _ { t + 1 } ( y ) , \right. \right. \left. \mathrm { ~ f o r ~ a l l ~ } x , u , t . \right. } \end{array}\tag{90}
$$

As before, we employ MOSEK [ApS19] to solve (90) and we can recover the optimal policy from $\eta ^ { 1 , * }$ . For Population 2, we utilize Boltzamnn Iteration [CK21]. Table 1 computes the ER-RA-MFE with $\beta = 0 . 1$ and a damping factor $\tau = 0 . 2$ (see Algorithm 1) for both populations while Figure 1 presents an ablation over the values of $\beta .$

H.2. Multi-Population Chasing. In both variants of multi-population chasing $( M = 3$ and 4) populations traverse a $. 5 \times 5$ grid and $T = 5$ , similar to the Navigation example. The population dynamics follow “noisy" transition dynamics similar to the original examples presented in [PPE<sup>+</sup>21, LLL<sup>+</sup>19]. We use CLARABEL [GC26] as the solver for the optimization problems associated with the multi-population chasing problem.

H.2.1. 3 Population Chasing. In this example, the fully expanded reward obtained by Population 1 is given by $r _ { t } ^ { 1 } ( x _ { t } ^ { 1 } , \mu _ { t } ^ { 1 } , \mu _ { t } ^ { 2 } , \hat { \mu _ { t } ^ { 3 } } ) : = - \log ( \dot { \mu _ { t } ^ { 1 } } ( \hat { x ) } ) + \mu _ { t } ^ { 2 } ( x ) + \mu _ { t } ^ { 3 } ( x )$ which is again, linear in $\mu _ { t } ^ { 2 }$ . The corresponding dual formulation using the dual variables $V _ { t } \in \mathbb { R } ^ { | \times | }$ for all $t = 0 , \ldots , T$ as follows

$$
\begin{array} { r l } { \underset { \eta ^ { 1 } \geq 0 , V } { \operatorname* { m a x } } } & { \displaystyle \sum _ { x } \mu _ { 0 } ^ { 2 } ( x ) V _ { 0 } ( x ) + \sum _ { t , x } \xi _ { t } ^ { 1 } ( x ) \Bigl ( \mu _ { t } ^ { 3 } ( x ) - \log ( \mu _ { t } ^ { 1 } ( x ) ) \Bigr ) } \\ & { \mathrm { ~ } } \\ & { \displaystyle \left( \begin{array} { l l } { \sum _ { u } \eta _ { 0 } ^ { 1 } ( x , u ) = \mu _ { 0 } ^ { 1 } ( x ) , } & { \mathrm { f o r ~ a l l ~ } x , } \\ { \sum _ { u } \eta _ { t } ^ { 1 } ( x , u ) = \xi _ { t } ^ { 1 } ( x ) , } & { \mathrm { f o r ~ a l l ~ } x , t , } \\ { \sum _ { u } \eta _ { t + 1 } ^ { 1 } ( y , u ) = \sum _ { x , u } \eta _ { t } ^ { 1 } ( x , u ) f _ { t } ^ { 1 } ( y \mid x , u ) , } & { \mathrm { f o r ~ a l l ~ } y , t , } \\ { V _ { T } ( x ) \leq \sum _ { u } \eta _ { \mathrm { T } } ^ { 1 } ( x , u ) , } & { \mathrm { f o r ~ a l l ~ } x , } \\ { V _ { t } ( x ) \leq \sum _ { u } \eta _ { \mathrm { t } } ^ { 1 } ( x , u ) + \sum _ { y } f _ { t } ^ { 2 } ( y \mid x , u ) V _ { t + 1 } ( y ) , } & { \mathrm { f o r ~ a l l ~ } x , u , t . } \end{array} \right. } \end{array}\tag{91}
$$

The optimal policy for the risk-neutral Populations 2 and $3 \ ( \mathrm { i . e . }$ , maximize expected reward instead of risk-averse payof) are computed using OMD with $\alpha = 0 . 0 1$ To obtain the entropy-regularized problem, we append (91) with the entropy term similar to (90). We report the ER-RA-MFE value in Table 1 under $\beta = 1$ and a damping factor $\tau = 0 . 2$

H.2.2. 4 Population Chasing. In this example, the fully expanded reward obtained by Population 1 is given by $r _ { t } ^ { 1 } ( x _ { t } ^ { 1 } , \mu _ { t } ^ { 1 } , \mu _ { t } ^ { 2 } , \mu _ { t } ^ { 3 } , \mu _ { t } ^ { 4 } ) : = { \dot { - \log } } ( \mu _ { t } ^ { 1 } ( x ) ) + \mu _ { t } ^ { 2 } ( x ) +$ $\textstyle { \frac { 1 } { 2 } } \mu _ { t } ^ { 3 } ( x ) - \mu _ { t } ^ { 4 } { \bar { ( x ) } }$ . Population 1 is now risk-averse to Populations 2 and 3. The reward $r _ { t } ^ { 1 }$ is linear in $\mu _ { t } ^ { 2 }$ and $\mu _ { t } ^ { 3 }$ and independent of cross-coupling terms like $\mu _ { t } ^ { 2 } ( x ) \mu _ { t } ^ { 3 } ( x )$ The corresponding dual formulation using the dual variables $V ^ { 2 } = \{ V _ { t } ^ { 2 } \} \stackrel { \cdot } { \in } \mathbb { R } ^ { | \times | }$ and $V ^ { 3 } = \{ V _ { t } ^ { 3 } \} \hat { \mathbf { \Psi } } \in \mathbb { R } ^ { | \times | }$ for all $t = 0 , \ldots , T$ is as follows

$$
\begin{array} { r l r l } { \displaystyle \operatorname* { m a x } _ { \eta ^ { 1 } \geq 0 , V } \quad } & { \displaystyle \sum _ { \upsilon } \mu _ { 0 } ^ { 2 } ( x ) V _ { 0 } ^ { 2 } ( x ) + \mu _ { 0 } ^ { 3 } ( x ) V _ { 0 } ^ { 3 } ( x ) + \sum _ { t , x } \xi _ { t } ^ { 1 } ( x ) \Big ( \mu _ { t } ^ { 4 } ( x ) - \log ( \mu _ { t } ^ { 1 } ( x ) ) \Big ) } \\ & { } & & { \displaystyle \sum _ { \upsilon = \eta } \eta _ { 0 } ^ { 1 } ( x , \upsilon ) = \mu _ { 0 } ^ { 1 } ( x ) , } & & { \mathrm { f o r ~ a l l } x , } \\ & { } & & { \displaystyle \sum _ { \upsilon } \eta _ { t } ^ { 1 } ( x , \upsilon ) = \xi _ { t } ^ { 1 } ( x ) , } & & { \mathrm { f o r ~ a l l } x , } \\ & { } & & { \displaystyle \sum _ { \upsilon = \eta } \eta _ { t + 1 } ^ { 1 } ( y , \ u ) = \sum _ { x , u } \eta _ { t } ^ { 1 } ( x , u ) f _ { \tau } ^ { 1 } ( y \mid x , u ) , } & & { \mathrm { f o r ~ a l l } y , t , } \\ { \mathrm { s . t . } \quad } & { } & & { \displaystyle \xi _ { t } ^ { T } ( x ) \leq \sum _ { u } \eta _ { t } ^ { 1 } ( x , u ) , } & & { \mathrm { f o r ~ a l l } x , } \\ & { } & & { \displaystyle \sum _ { \upsilon = \eta } \sum _ { u } \eta _ { t } ^ { 1 } ( x , u ) + \sum _ { y } \int _ { t } ^ { 2 } ( y \mid x , u ) V _ { \tau + 1 } ^ { 2 } ( y ) , } & & { \mathrm { f o r ~ a l l } x , u , } \\ & { } & & { \displaystyle \operatorname* { F r } _ { 1 } ^ { 3 } ( x ) \leq \sum _ { u = 1 } \frac { 1 } { 2 } \eta _ { t } ^ { 1 } ( x , u ) , } & & { \mathrm { f o r ~ a l l } x , u , } \\ & { } & &  \displaystyle \operatorname* { F r } _  2  \end{array}\tag{92}
$$

The optimal policy for the risk-neutral populations are computed using OMD with $\alpha = 0 . 0 1$ . To obtain the entropy-regularized problem, we append (92) with the entropy term similar to (90). We report the ER-RA-MFE value in Table 1 under $\beta = 1$ and $\tau = 0 . 2$

H.3. Stock Market Trading. The discrete inventory positions are specified as: $\textsf { X } : = \{ - K , \ldots , 0 , \ldots , K \}$ for some integer $K > 0$ representing the holdings of a population. The actions available to each population are $\mathsf { U } : = \{ - 1 , 0 , 1 \}$ representing SELL, HOLD and BUY respectively. The inventory transitions deterministically, i.e., $x _ { t + 1 } = \mathrm { c l i p } _ { ( - K , K ) } ( x _ { t } + a _ { t } )$ . The change in the stock price at time � is denoted by $\Delta p _ { t }$ and follows $\Delta p _ { t } = \gamma _ { t } + \theta _ { t } z _ { t }$ , where $\gamma _ { t }$ represents the price drift, $\theta _ { t }$ represents the volatility and $\boldsymbol z \sim \mathrm { D i s c r e t e } ( - 1 , 0 , 1 )$ , where $\mathsf { E } [ z _ { t } ] = 0$ and var $( z _ { t } ) \ = \ 1$ . The rewards for the AI population follows $r _ { t } ^ { 1 } ( x _ { t } ^ { 1 } , \mu _ { t } ^ { 1 } , \mu _ { t } ^ { 2 } ) = \bar { \gamma } _ { t } x _ { t } ^ { 1 } + c _ { 0 } x _ { t } ^ { 1 } \cdot \left. q , \mu _ { t } ^ { 2 } \right. - c _ { 1 } ( x _ { t } ^ { 1 } ) ^ { 2 }$ where $q ^ { \intercal } = [ - K , \dots , 0 , \dots K ]$ and $c _ { 0 } , c _ { 1 } > 0$ . The term $c _ { 0 } x _ { t } ^ { 1 } \cdot \left. q , \mu _ { t } ^ { 2 } \right.$ represents the price pressure created by aggregate human inventory shifts and $c _ { 1 } ( x _ { t } ) ^ { 2 }$ penalizes large AI inventory positions due to volatility and holding costs. The rewards for the human population is given by $r _ { t } ^ { 2 } ( x _ { t } ^ { 2 } , \mu _ { t } ^ { 1 } , \mu _ { t } ^ { 2 } ) = \gamma _ { t } x _ { t } ^ { 2 } - \overset { \smile } { w _ { 0 } } ( \theta _ { t } x _ { t } ) ^ { 2 } + w _ { 1 } x _ { t } ^ { 2 } \cdot \left. q , \mu _ { t } ^ { 2 } \right. -$ $w _ { 2 } \mathbb { 1 } _ { \gamma _ { t } < - \chi } ( x _ { t } ^ { 2 } + K )$ , where $- w _ { 0 } ( \theta _ { t } x _ { t } ) ^ { 2 } , w _ { 1 } x _ { t } ^ { 2 } \cdot \left. q , \mu _ { t } ^ { 2 } \right.$ and $w _ { 2 } \mathbb { 1 } _ { \gamma _ { t } < - \chi } ( x _ { t } ^ { 2 } + K )$ represent the penalty due to volatility in the price, herding behavior and penalty due to panic selling when the stock price drift falls below $- \chi$

We calibrate the price model (in particular, � and �) to mimic the dynamics of Apple Inc. (AAPL) over September 2025 $( T = 2 0 )$ , using data retrieved through yfinance [Aro19] and analyze the efect of risk-aversion of the AI population to the deviations in the human population’s mean-field trajectory. We assume $K = 5 ,$ , which implies $| \mathsf { X } | = 1 1$ . Other parameters are given as follows:

$$
\begin{array} { c c c c c c c c c c c c c c c c c c c c c c } { { c _ { 0 } = 0 . 1 } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } &   \end{array}
$$

The reward $r _ { t } ^ { 1 }$ linear in $\mu _ { t } ^ { 2 }$ . However, it does not satisfy Assumption 4, and consequently the convergence guarantee for risk-averse fictitious play does not apply. Instead, we employ entropy-regularized fixed-point iteration to obtain a close approximation of the RA-MFE, using a “small” regularization parameter $\beta = 1 0 ^ { - 6 }$ and $\tau = 0 . 2$ . The corresponding dual formulation using the dual variables $\dot { V } _ { t } \in \mathbb { R } ^ { | \mathsf { X } | }$

for all $t = 0 , \ldots , T$ as follows

$$
\begin{array} { r l } { \underset { \eta ^ { 1 } \geq 0 , V } { \operatorname* { m a x } } } & { J _ { \mathrm { s m } } } \\ & { \left. \begin{array} { l l l } { \sum _ { u } \eta _ { t } ^ { 1 } ( \boldsymbol { x } , u ) = \xi _ { t } ^ { 1 } ( \boldsymbol { x } ) , } & & { \mathrm { f o r ~ a l l ~ } \boldsymbol { x } , t , } \\ { \sum _ { u } \eta _ { t + 1 } ^ { 1 } ( y , u ) = \sum _ { \boldsymbol { x } , u } \eta _ { t } ^ { 1 } ( \boldsymbol { x } , u ) f _ { t } ^ { 1 } ( y \mid \boldsymbol { x } , u ) , } & & { \mathrm { f o r ~ a l l ~ } y , t , } \end{array} \right. } \\ { \mathrm { s . ~ t . ~ } \quad } & { \left. \begin{array} { l l l } { V _ { T } ( \boldsymbol { x } ) \leq c _ { 0 } x \left. \boldsymbol { q } , \xi _ { T } ^ { 1 } \right. , } & & { \mathrm { f o r ~ a l l ~ } \boldsymbol { x } , } \\ { V _ { t } ( \boldsymbol { x } ) \leq c _ { 0 } x \left. \boldsymbol { q } , \xi _ { t } ^ { 1 } \right. + \sum _ { y } f _ { t } ^ { 2 } ( y \mid \boldsymbol { x } , u ) V _ { t + 1 } ( y ) , } & & { \mathrm { f o r ~ a l l ~ } \boldsymbol { x } , t , } \end{array} \right. } \end{array}\tag{93}
$$

where the objective function is

$$
\begin{array} { l } { { \displaystyle { \cal J } _ { \mathrm { s m } } : = \sum _ { x } \mu _ { 0 } ^ { 2 } ( x ) V _ { 0 } ( x ) + \sum _ { t } \gamma _ { t } \left. q , \xi _ { t } ^ { 1 } \right. - \sum _ { t , x } c _ { 1 } \xi _ { t } ^ { 1 } ( x ) ( x ) ^ { 2 } } } \\ { { - \beta \sum _ { t = 0 } ^ { T - 1 } \sum _ { x , u } \eta _ { t } ^ { 1 } ( x , u ) \log \left( \frac { \eta _ { t } ^ { 1 } ( x , u ) } { \sum _ { u ^ { \prime } } \eta _ { t } ^ { 1 } ( x , u ^ { \prime } ) } \right) } } \end{array}
$$

As before, we use MOSEK [ApS19] to solve (93).

H.4. Linear Navigation. We consider an MP-MFG with two populations traversing a three-state environment, where agents can move LEFT, RIGHT, or remain STATIONARY and $T = 5 .$ . The dynamics of the game follow Figure 1(a) in [GAXJ24]. Similar to the Grid Navigation problem, Population 1, at a given state $x _ { t } ^ { 1 } \in \mathsf { X } .$ , receives $r _ { t } ^ { 1 } ( x _ { t } ^ { 1 } , \mu _ { t } ^ { 1 } , \mu _ { t } ^ { 2 } ) = \mu _ { t } ^ { 2 } ( x _ { t } ^ { 1 } )$ , which satisfies Assumption 3. We assume that Population 1 is risk-averse to deviations in Population $2 \mathrm { { } s }$ trajectory. As mentioned in the main text, for $\varepsilon \in ( 0 , 1 ) , \mathcal { W } _ { j } ( \mu , \varepsilon )$ depends explicitly on the nominal mean-field trajectory. Consequently, the inner minimization is coupled with the nominal trajectory and the dual formulation no longer provides a computational advantage. We therefore solve the resulting max-min problem directly using the iterative MINIMAX-APPA algorithm proposed in [LJJ20], which is designed to handle such coupled minimax optimization problems. In particular, we our aim to compute a solution to

$$
\begin{array} { r } { \underset { \pi ^ { 1 } \in \Pi ^ { 1 } } { \arg \operatorname* { m a x } } \bar { \mathcal { J } } _ { 1 } ( \pi ^ { 1 } ; \mu ) = \underset { \pi ^ { 1 } \in \Pi ^ { 1 } } { \arg \operatorname* { m a x } } \underset { \hat { \mu } ^ { 2 } \in \mathcal { W } _ { 1 } ( \mu , \varepsilon ) } { \operatorname* { i n f } } \in \pi ^ { 1 } \left[ \underset { t = 0 } { \overset { T } { \sum } } \hat { \mu } _ { t } ^ { 2 } ( x _ { t } ^ { 1 } ) \right] } \\ { - \beta \mathsf { E } ^ { \pi ^ { 1 } } \left[ \underset { t = 0 } { \overset { T } { \sum } } \log \pi _ { t } ^ { 1 } ( u _ { t } ^ { 1 } | x _ { t } ^ { 1 } ) \right] + \frac { \sigma } { 2 } \| \hat { \mu } _ { t } ^ { 2 } \| _ { 2 } ^ { 2 } } \end{array}\tag{94}
$$

Equivalently, we can reformulate (94) using the policy-occupation-measure equivalence (16) and obtain,

$$
\begin{array} { l } { \displaystyle \underset { \tilde { \sigma } ^ { 1 } = ( \tilde { \xi } ^ { 1 } , \tilde { \eta } ^ { 1 } ) \in \Theta ^ { 1 } } { \mathrm { a r g } } \Phi _ { 1 , \beta } ^ { \sigma } ( \widetilde { \sigma } ^ { 1 } ; \sigma ) = \underset { \tilde { \sigma } ^ { 1 } \in \Theta ^ { 1 } } { \mathrm { a r g } } \displaystyle \operatorname* { m a x } _ { \hat { \mu } ^ { 2 } \in \mathcal { W } _ { 1 } ( \mu , \varepsilon ) } \left[ \sum _ { t = 0 } ^ { T } \sum _ { x \in \mathsf { X } } \widetilde { \xi } _ { t } ^ { 1 } ( x ) \cdot \hat { \mu } _ { t } ^ { 2 } ( x ) + \frac { \sigma } { 2 } \left\| \hat { \mu } ^ { 2 } \right\| _ { 2 } ^ { 2 } \right] } \\ { \displaystyle - \beta \sum _ { t = 0 } ^ { T - 1 } \sum _ { x \in \mathsf { X } } \sum _ { u \in \mathsf { U } } \widetilde { \eta } _ { t } ^ { 1 } ( x , u ) \log \frac { \widetilde { \eta } _ { t } ^ { 1 } ( x , u ) } { \widetilde { \xi } _ { t } ^ { 1 } ( x ) } . } \end{array}
$$

We apply this algorithm for $\varepsilon = 0 . 1 , \beta = 1$ and $\sigma = 1$ . We perform 10 outer iterations and 5 inner iterations for the minimizing problem $( \mathrm { i } . \mathrm { e } . , ( 1 6 ) )$ . Additionally, for the outer iteration, we scale the original learning rate for the minimization problem provided in [LJJ20] by 0.01 for empirically stable updates. For the maximizing variable (Population 1’s occupation measure), we perform 100 iteration of Nesterov’s accelerated gradient descent and similarly scale the learning rate by 0.05 for stability.

H.5. Generating Adversarial Mean-Field Flows. In all our examples with $\varepsilon =$ 1, we generate deviated mean-field flows by uniformly sampling policies from Π and propagating each sampled policy $\pi ^ { k }$ where $k \neq j$ to obtain the corresponding mean-field flow over the entire simplex. For $ { \varepsilon } \in ( 0 , 1 )$ , we instead construct a random mean-field trajectory and project it onto $\mathcal { W } _ { j } ( \mu , \varepsilon )$ . Recall from Proposition 1 that the ambiguity set is a polytope. Consequently, the projection is well-defined and maps the sampled trajectory to a dynamically feasible flow in $\mathcal { W } _ { j } ( \mu , \varepsilon )$ . Since the projection is onto the boundary of the ambiguity set, the resulting flow lies at the maximum prescribed deviation from the nominal flow, i.e., ${ \bf { d } } _ { \mathrm { T V } } \big ( \mu _ { t } ^ { k } , \hat { \mu } _ { t } ^ { k } \big ) = \varepsilon$ for all (t). To also sample trajectories satisfying $\mathrm { d } _ { \mathrm { T V } } \big ( \mu _ { t } ^ { k } , \hat { \mu } _ { t } ^ { k } \big ) < \varepsilon .$ , we randomly choose a smaller radius $\varepsilon ^ { \prime } \in ( 0 , \varepsilon )$ and project the sampled trajectory onto the corresponding constraint set with $\mathrm { d } _ { \mathrm { T V } } \big ( \mu _ { t } ^ { k } , \hat { \mu } _ { t } ^ { k } \big ) \leq \varepsilon ^ { \prime }$