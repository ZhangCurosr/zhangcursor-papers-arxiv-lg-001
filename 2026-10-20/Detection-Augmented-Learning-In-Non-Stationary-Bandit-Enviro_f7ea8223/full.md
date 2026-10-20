# Detection Augmented Learning In Non-Stationary Bandit Environments

Argyrios Gerogiannis, Yu-Han Huang, Subhonmesh Bose,

and Venugopal V. Veeravalli

## Abstract

We study the problem of non-stationary bandits without knowledge of the environmental changes. We propose Detection Augmented Learning (DAL) a black-box paradigm that comprises three modular components: a bandit algorithm, a change detector, and a forced exploration schedule. This modular design enables its applicability across a broad spectrum of bandit problems, from classical to highly structured settings. First, we provide theoretical insights into DAL’s strong empirical performance: a single regret bound shows that DAL transfers the regret of its stationary input to the piecewise stationary setting, attaining or improving the best known bounds in every setting considered. We further identify conditions under which DAL performs well in drifting environments. Then, we showcase DAL’s practical performance and versatility through extensive experimentation in which DAL consistently surpasses all state-of-the-art methods, both prior-free and prior-based, across diverse non-stationary scenarios in nine bandit settings. Finally, we investigate the effectiveness of the forced exploration and show that a bandit-driven strategy improves performance and shortens detection delay.

## Index Terms

Bandits, non-stationary learning, black-box algorithms, sequential change detection.

## I. INTRODUCTION

Bandit models underpin a wide range of engineering systems, from recommendation to realtime bidding [2, 3]. Many variants of bandits have emerged since the work of [4], including parametric bandits (PBs) [5–7], non-parametric bandits [8, 9], contextual bandits (CBs) [10, 11], combinatorial bandits [12, 13], and bandits with graph feedback [14]. In the general bandit problem, in each round, an agent receives a context $C _ { t }$ randomly sampled from a set $\mathcal { C }$ according to $\mathcal { P } _ { t }$ , and chooses action $A _ { t } = \pi _ { t } ( C _ { t } )$ from a compact action set $\mathcal { A } \subseteq \mathbb { R } ^ { d }$ using a causal policy $\pi _ { t }$ . We compare the agent with a policy set Π—a set of mappings from $\mathcal { C }$ to ${ \mathcal { A } } .$ The agent receives reward $X _ { t } = f _ { t } ( C _ { t } , A _ { t } ) + \varepsilon _ { t }$ , where $f _ { t } : \mathcal { C } \times \mathcal { A }  \mathbb { R }$ is the mean reward function and $\varepsilon _ { t }$ is conditionally zero-mean sub-Gaussian noise. Depending on the feedback model, the agent may also observe individual item outcomes or the rewards of other actions, likewise with conditionally sub-Gaussian noise. The goal is to minimize the dynamic regret, using the policy $\pi _ { t } \colon$

$$
R _ { T } : = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \operatorname* { m a x } _ { \pi \in \Pi } f _ { t } ( C _ { t } , \pi ( C _ { t } ) ) - f _ { t } ( C _ { t } , A _ { t } ) \right] .\tag{1}
$$

CBs follow the general formulation above, where the context $C _ { t }$ is independently sampled from $\mathcal { P } _ { t }$ and $| { \cal A } |$ is finite. In the other settings, the context is fixed across time and, with slight abuse of notation, we write $f _ { t } ( C _ { t } , A _ { t } ) = f _ { t } ( A _ { t } )$ . For parametric bandits, $f _ { t } ( A _ { t } ) = \mu ( \langle \theta _ { t } , A _ { t } \rangle )$ , where $\theta _ { t } \in \mathbb { R } ^ { d }$ is a bounded unknown parameter, and $\mu : \mathbb { R } \to \mathbb { R }$ is injective. These include linear bandits (LBs), with $\mu$ as identity, generalized linear bandits (GLBs), and self-concordant bandits (SCBs), where $\mu$ is self-concordant and the noise variance may depend on the mean [15].

For non-parametric bandits, we consider kernelized bandits (KBs) and Holder bandits (HBs).¨ In KBs $f _ { t } \in H _ { k }$ , a reproducing kernel Hilbert space (RKHS) induced by a continuous positive semi-definite kernel $k : \mathcal { A } \times \mathcal { A }  \mathbb { R }$ with $k ( a , a ) \leq 1$ and $\| f _ { t } \| _ { H _ { k } } \le B$ and we define the maximum information gain $\gamma _ { T }$ to be the worst-case mutual information between $f$ and $T$ noisy evaluations [8]. In HBs, $\mathcal { A } = [ 0 , 1 ] ^ { d }$ and $| f _ { t } ( a ) - f _ { t } ( a ^ { \prime } ) | \leq L \| a - a ^ { \prime } \| _ { \infty } ^ { \beta }$ with $0 < \beta \leq 1$ , where $L > 0$ is the smoothness constant, and $\beta$ is the Holder exponent;¨ $\beta = 1$ recovers Lipschitz rewards. The instance-dependent complexity is described by a common near-optimality packing exponent $d _ { z } \geq 0 ~ [ 9 ]$ . It controls the largest number of pairwise ε-separated, $\mathcal { O } ( \varepsilon )$ -optimal actions under the metric $\rho _ { \beta } ( a , a ^ { \prime } ) = L \| a - a ^ { \prime } \| _ { \infty } ^ { \beta }$ : this number is $\mathcal { O } ( \varepsilon ^ { - d _ { z } } )$ , with common constants across stationary segments. The choice $d _ { z } = d / \beta$ is always admissible for the Holder class, while a¨ smaller exponent captures favorable geometry near the optimum.

In combinatorial bandits, actions form a feasible family A of subsets of at most K of d items, so, with slight abuse in notation, $d$ is the number of items, K is the maximum action size, and $| { \cal A } |$ counts feasible subsets. Their mean reward is $\begin{array} { r } { f _ { t } ( a ) = \sum _ { j \in a } \theta _ { t , j } } \end{array}$ , with $\theta _ { t } \in [ 0 , 1 ] ^ { d }$ . We consider combinatorial semi-bandits (CSBs), where every selected item’s outcome is observed, and combinatorial full-bandits (CFBs), where only the sum of these outcomes is observed. Both have linear mean rewards on the incidence vectors $x _ { a } = ( \mathbb { 1 } \{ j \in a \} ) _ { j = 1 } ^ { d }$ , but differ in the feedback available for learning and detection. For CFBs, let r = dim span $\lbrace x _ { a } : a \in \mathcal { A } \rbrace \leq d$ denote the incidence-span dimension and we consider conditionally independent item noises with uniformly bounded sub-Gaussian variance, so the observed sum has a common variance $\sigma _ { \Sigma } ^ { 2 } = \mathcal { O } ( K )$

In graph-feedback bandits (GFBs), $A = [ d ]$ consists of d arms and $f _ { t } ( a ) = \theta _ { t , a }$ , with $\theta _ { t } \in [ 0 , 1 ] ^ { d }$ choosing a reveals its reward and those of its neighbors in a known undirected graph G [14]. $\alpha ( G )$ denotes the independence number, the largest number of pairwise nonadjacent vertices, and $\chi _ { \mathrm { c l } } ( G )$ denotes the clique-partition number, the minimum number of cliques needed to partition the vertices. These quantities satisfy $\alpha ( G ) \leq \chi _ { \mathrm { c l } } ( G )$ and describe the graph dependence of the regret bounds [16, 17]. Thus, the same decision problem encompasses finite and continuous action spaces, as well as feedback that reveals one reward or several outcomes in a round. In all bandit settings, we consider bounded or sub-Gaussian observations, including logistic rewards.

The lion’s share of the literature on bandits assumes stationarity—i.e., fixed $f _ { t } , \theta _ { t } , \mathcal { P } _ { t } -$ but this rarely holds in practice due to evolving conditions [18–21]. Non-stationary (NS) settings are often categorized into two types—gradual drifts and abrupt changes. In the drifting model, $f _ { t }$ and $\mathcal { P } _ { t }$ evolve slowly under a variation budget constraint [22]. In contrast, piecewise stationary (PS) models assume abrupt shifts at unknown change-points, with a total of $N _ { T }$ changes:

$$
1 = : \nu _ { 0 } < \nu _ { 1 } < \cdot \cdot \cdot < \nu _ { N _ { T } } < \nu _ { N _ { T } + 1 } : = T + 1 ,
$$

with $f _ { t } = f _ { t ^ { \prime } }$ and $\mathcal { P } _ { t } = \mathcal { P } _ { t ^ { \prime } }$ for $t , t ^ { \prime } \in \{ \nu _ { k } , \ldots , \nu _ { k + 1 } - 1 \}$ . The feedback law is likewise fixed within each segment. Conditional on the segment laws and boundaries, the context and potential feedback are drawn independently across rounds. Item marginals do not depend on which action reveals them; outcomes of different items in the same round may be dependent.

NS bandit algorithms are typically either adaptive, adjusting continuously, or restarting, choosing to unlearn and kickstart the learning process at certain times. They may also be prior-based (assuming knowledge of the non-stationarity) or prior-free. Prior-based adaptive methods (discounting/sliding window) weigh recent observations more heavily and have been developed for LBs [23, 24], GLBs [25, 26], SCBs [15, 27], KBs [28, 29], CSBs [30], and GFBs [17]. Prior-based restarting approaches use budgeted restarts and have been studied for LBs and GLBs [31], KBs [29], HBs [32], and GFBs [17]. Detection-based restarting methods are prior-free for LBs and KBs [33], CBs [34], HBs [35], CSBs [30, 36], and GFBs [17]. The most closely related work is [37], which addresses PS multi-armed bandits and introduces techniques that we build upon in establishing our theory.

Among prior-free methods, the black-box paradigm is particularly appealing: it equips a stationary bandit algorithm with non-stationarity handling capabilities without redesigning its learning rule. MASTER [22] is a prominent method of this class, detecting non-stationarity by checking for violations of stationary regret bounds. However, its prescribed thresholds prevent detection at practical horizons [38], while a recently identified error in its regret analysis invalidates the published proof of its theoretical guarantees [39]. To this end, we consider MASTER empirically but not theoretically. More broadly, methods tailored to individual settings do not by themselves explain how to combine a preferred stationary learner with a practical detector across different action and feedback models. We address this problem with a theoretically grounded, practical black-box framework and comprehensive synthetic and real-world evaluation.

Contributions. We present Detection Augmented Learning (DAL), a practical prior-free, black-box detection-based paradigm for general NS bandits. The design is motivated by the need to adapt without prior knowledge of non-stationarity and by a black-box reduction that simplifies non-stationary algorithm design to specifying when to restart a stationary learner. Our method is simple—combining a change detector, a forced exploration schedule, and a stationary bandit algorithm—modular, and easy to implement. Its design applies to PBs, KBs, CBs, HBs, CSBs and CFBs, and GFBs. Theoretically, a single regret theorem transfers the stationary input’s guarantee to the PS setting under detectable, sufficiently separated changes. The resulting bounds match the best existing bounds in PS-LB and PS-CB rates and improve all other settings, under the assumptions specified in Table I. For drifting regimes, we identify conditions for good performance and validate them empirically. Empirically, extensive synthetic and real-world evaluations show consistent gains over both prior-free and prior-based baselines across nine bandit settings. Finally, we introduce a bandit-driven forced-exploration rule for DAL that learns which actions to probe, and show empirically that it can improve performance compared to standard round-robin probing.

## II. THE DAL FRAMEWORK

The DAL framework is a black-box characterized by a modular structure of three components: a non-stationarity detector, a forced exploration scheme, and a bandit algorithm. We provide highlevel ideas of the structure of our approach and formally present our framework in Algorithm 1.

Algorithm 1 Detection Augmented Learning (DAL)   
Input: stationary algorithm B, detector D, covering set $A _ { \mathrm { e } } ,$ , context set C, frequencies $( \alpha _ { k } ) _ { k = 1 } ^ { T }$ , horizon T   
Initialize: empty monitored histories; $\tau  0 , k  1$   
1: for $t = 1 , \dots , T$ do   
2: Observe context $C _ { t }$ and set $i  ( ( t - \tau - 1 )$ mod $\lceil N _ { \mathrm { e } } / \alpha _ { k } \rceil ) + 1$   
3: if $i \leq N _ { \mathrm { e } }$ then play $a ^ { ( i ) }$ and add the feedback to the corresponding monitored histories   
4: if D declares a change on any updated history then   
5: Reset B and clear all monitored histories and set $\tau \gets t ,$ k $\gets k + 1$   
6: else Run $\boldsymbol { B }$ for one local step, using its own feedback

Non-Stationarity Detector. To identify changes in the environment, DAL uses a generalpurpose detector D for monitoring shifts in the distribution of judiciously chosen observation sequences obtained through forced exploration. Rather than testing the stationary learner’s regret bound, it tests for changes in the means of these observations. We adopt a detector aligned with [37, 40], grounded in the well-established theory of quickest change detection [41]. With scalar feedback, DAL detects changes in the mean reward associated with a context–action pair, keeping histories $\mathcal { H } _ { ( c , a ) }$ for $( c , a ) \in \mathcal { C } \times \mathcal { A } _ { \mathrm { e } }$ . With semi-bandit or graph feedback, it instead keeps a history ${ \mathcal { H } } _ { j }$ for each item or arm, and updates every history revealed by a forced action. Thus, one action may contribute to several detectors. The detector can be replaced without changing the learner, provided that it satisfies the property in Section III-A.

Forced Exploration. In stationary settings, effective algorithms quickly concentrate on optimal actions for each context, rarely exploring suboptimal actions. In NS environments, however, this behavior may lead to missed changes on these rarely sampled actions, and thus, forced exploration is essential for reliable detection. When the action space is large or infinite, exploring all actions becomes infeasible. Therefore, DAL only does extra exploration on a finite covering set, $\mathcal { A } _ { \mathrm { e } } = \{ a ^ { ( i ) } : i \in [ N _ { \mathrm { e } } ] \} \subseteq \mathcal { A }$ , in which $a ^ { ( i ) }$ denotes the ith action in $A _ { \mathrm { e } }$ . The set is designed such that the mean of at least one monitored history changes whenever a detectable change occurs. Its construction uses both the reward model and the available feedback; for instance, $\lceil d / K \rceil$ feasible actions suffice to observe all d items in top-K selection. In particular, after the (k − 1)th restart, DAL is forced to play each action in $A _ { \mathrm { e } }$ once for $N _ { \mathrm { e } }$ steps, followed by the bandit algorithm for the next $\lceil N _ { \mathrm { e } } / \alpha _ { k } \rceil - N _ { \mathrm { e } }$ steps, repeatedly, until the kth restart. Here, $\alpha _ { k } \in ( 0 , 1 )$ is the exploration frequency, striking a balance between detection delay and regret

Algorithm 2 Generalized Likelihood Ratio Test Algorithm 3 Generalized Shiryaev–Roberts Test   
1: Input: History ${ \mathcal { H } } = \{ X _ { 1 } , \ldots , X _ { n } \}$ , δ , δ , KL 1: Input: History ${ \mathcal { H } } = \{ X _ { 1 } , \ldots , X _ { n } \}$ , δ<sub>F</sub>, $ { \delta _ { \mathrm { D } } }$ , KL   
divergence $\operatorname { k l } ( \cdot , \cdot )$ divergence $\mathrm { k l } ( \cdot , \cdot ) , \mathrm { G S R } _ { k } \gets 0$   
2: for k = 1 to n − 1 do 2: for k = 1 to n − 1 do   
3: Compute empirical means ${ \hat { \mu } } _ { 1 : k } , { \hat { \mu } } _ { k + 1 : n } , { \hat { \mu } } _ { 1 : n }$ 3: Compute empirical means ${ \hat { \mu } } _ { 1 : k } , { \hat { \mu } } _ { k + 1 : n } , { \hat { \mu } } _ { 1 : n }$   
4: $\mathrm { G L R } _ { k } \gets k \operatorname { k l } ( \hat { \mu } _ { 1 : k } , \hat { \mu } _ { 1 : n } )$ 4: Compute $\mathrm { G L R } _ { k }$ according to Alg. 2   
5: $+ \left( n - k \right) \mathrm { k l } ( \hat { \mu } _ { k + 1 : n } , \hat { \mu } _ { 1 : n } )$ 5: $\mathrm { G S R } _ { k } \gets \mathrm { G S R } _ { k } + \exp ( \mathrm { G L R } _ { k } )$   
6: if $\mathrm { G L R } _ { k } \geq \beta _ { \mathrm { G L R } } ( n , \delta _ { \mathrm { F } } )$ then 6: if $\mathrm { G S R } _ { k } \geq \beta _ { \mathrm { G S R } } ( n , \delta _ { \mathrm { F } } )$ then   
7: return detection 7: return detection

from extra exploration.

Bandit Algorithm. With a detector D and forced exploration, DAL augments a (stationary) bandit algorithm B: it resets B entirely whenever D detects a change in any monitored history, and runs B with periodic forced exploration otherwise. A key advantage of DAL is its ability to translate strong stationary performance into robust performance under NS conditions. The learner uses its own local clock and observations, so that pausing it for exploration does not alter its stationary interface. Each restart initializes a fresh copy. Any stationary algorithm satisfying the regret requirement of Theorem III.9 can be used; retaining an order-optimal rate additionally requires that the stationary input be order-optimal and that exploration not dominate its regret.

## III. THEORETICAL INSIGHTS

## A. On Effective Detection

When selecting a non-stationarity detector, accuracy and efficiency are essential for ensuring optimal regret growth. Any detector aiming to identify distribution shifts inherently requires a certain number of samples, both before and after the change. This sample complexity should scale appropriately to avoid negatively impacting the total regret.

The stopping time $\tau$ of a change detector D denotes the time-step at which a change is identified, measured in the detector’s own observations. Let $\mathbb { P } _ { \nu }$ denote the probability law with change-point at $\nu ,$ and $\mathbb { P } _ { \infty }$ the law with no change-point. The latency $\ell _ { D }$ is the length of time post-change within which a change is declared with probability at least $1 - \delta _ { \mathrm { D } }$ , i.e.,

$$
\ell _ { D } : = \operatorname* { i n f } \left\{ t \in [ T ] : \mathbb { P } _ { \nu } ( \tau \geq \nu + t ) \leq \delta _ { \mathrm { D } } , \quad \forall \nu \in [ m _ { \mathcal { D } } + 1 , T - t ] \right\} ,
$$

where $m _ { D }$ is the required pre-change window. A good detector seeks to minimize $\ell _ { D }$ while ensuring low false-alarm probability over horizon $T ,$ , namely $\mathbb { P } _ { \infty } ( \tau \leq T ) \leq \delta _ { \mathrm { F } }$ . We require the following property.

Property III.1. The detector satisfies $\mathbb { P } _ { \infty } ( \tau \leq T ) \leq \delta _ { \mathrm { F } }$ , and its pre-change window and latency satisfy $\ell _ { \mathcal { D } } + m _ { \mathcal { D } } = \mathcal { O } \big ( \log T + \log ( 1 / \delta _ { \mathrm { F } } ) + \log ( 1 / \delta _ { \mathrm { D } } ) \big )$ . We refer to any detector that satisfies this property as a Horizon-Logarithmic Detector (HLD).

In [42] it was shown that the Generalized Likelihood Ratio (GLR) and Generalized Shiryaev-Roberts (GSR) tests are HLDs for independent sub-Gaussian observations with fixed variance [42]. The two HLDs are given in Algorithms 2 and 3 and $\operatorname { k l } ( x ; y )$ is KL-divergence between two Gaussian or Bernoulli distributions, depending on the rewards. To select which samples should be fed into the HLD, one needs to properly select the covering set $A _ { \mathrm { e } } ,$ so that it contains actions whose feedback captures changes in the reward function. However, changes cannot be arbitrarily small if detection is to have logarithmic sample complexity with uniform constants. Hence, $A _ { \mathrm { e } }$ should be designed such that whenever a change occurs, at least one monitored history exhibits an appreciable mean-shift. For scalar-reward histories, define

$$
\Delta _ { \mathrm { c } } : = \operatorname* { m i n } _ { k \in [ N _ { T } ] } \operatorname* { m a x } _ { ( c , a ) \in \mathcal { C } \times \mathcal { A } _ { \mathrm { e } } } | f _ { \nu _ { k } } ( c , a ) - f _ { \nu _ { k } - 1 } ( c , a ) | .\tag{2}
$$

In CSBs and GFBs, the maximum is instead over the monitored item or arm means, giving $\begin{array} { r } { \Delta _ { \mathrm { c } } = \operatorname* { m i n } _ { k \in [ N _ { T } ] } \| \theta _ { \nu _ { k } } - \theta _ { \nu _ { k } - 1 } \| _ { \infty } } \end{array}$ . Then, $\Delta _ { \mathrm { c } }$ captures how well the monitored observations can discern between successive reward functions. Its value crucially affects the performance of the HLD, as their pre- and post-change sample complexity grows with the noise-to-squared-shift ratio [37, 42]. The more discernible the changes are, the easier the detection becomes. Since forced exploration incurs regret, $A _ { \mathrm { e } }$ should also be small. We therefore use the structure of the reward function and the available feedback to construct informative covering sets. The following propositions provide such choices; their proofs are given in the Appendix.

Proposition III.2. In NS-PBs, $A _ { \mathrm { e } }$ can be any maximal linearly independent subset of A. The same holds in NS-CFBs on the incidence vectors, with $N _ { \mathrm { e } } = r \leq d ,$ the dimension of their span.

Proposition III.3. In NS-KBs, assume $\mathcal { A } \subseteq [ 0 , R ] ^ { d }$ , that the permitted RKHS functions are $B L _ { u } .$ Lipschitz in the Euclidean norm, and that there is a $\tilde { a } \in \mathcal A$ such that inf $_ { f \neq f ^ { \prime } } | f ( \tilde { a } ) - f ^ { \prime } ( \tilde { a } ) | > L _ { T } ,$ for some $L _ { T } > 0$ , with the infimum over successive changes. Let $\delta _ { T } = L _ { T } / ( 2 B L _ { u } )$ and let $\nu _ { T } \subseteq { \cal A }$ be the centers of a δ<sub>T</sub>-cover. Then $A _ { \mathrm { e } }$ can be taken as $\gamma _ { T } .$ . Such a cover can be chosen with $| \mathcal { V } _ { T } | \leq \lceil \sqrt { d } R / \delta _ { T } \rceil ^ { d } ; f o r \mathcal { A } = [ 0 , R ] ^ { d }$ , a regular cell-center grid gives $| \mathcal { V } _ { T } | \leq \lceil \sqrt { d } R / ( 2 \delta _ { T } ) \rceil ^ { d } .$

Proposition III.4. In $N S { \cdot } H B s ,$ assume that every change satisfies $\| \boldsymbol { f } - \boldsymbol { f } ^ { \prime } \| _ { \infty } \geq L _ { T }$ for some $L _ { T } > 0$ . Let $\delta _ { T } = ( L _ { T } / ( 4 L ) ) ^ { 1 / \beta }$ and let $\nu _ { T }$ be the centers of a δ -cover of $[ 0 , 1 ] ^ { d } i n \parallel \cdot \parallel _ { \infty }$ . Then $A _ { \mathrm { e } }$ can be taken as $\nu _ { T }$ , with $\Delta _ { \mathrm { c } } \geq L _ { T } / 2 .$ . The regular grid $\{ 0 , 1 / M , \ldots , 1 \} ^ { d }$ , with $M = \lceil 1 / ( 2 \delta _ { T } ) \rceil$ gives such a cover with $( M + 1 ) ^ { d }$ points.

Proposition III.5. In NS-CSBs and NS-GFBs, $A _ { \mathrm { e } }$ can be any family of feasible actions whose items or observation neighborhoods, respectively, cover all d items or arms. Every forced cycle then observes each monitored history at least once (items that belong to no feasible action can be discarded). If every reward-function change has sup-norm at least $L _ { T }$ , then $\Delta _ { \mathrm { c } } \ge L _ { T } / K$ in CSBs and $\Delta _ { \mathrm { c } } \geq L _ { T }$ in GFBs. A CSB cover always exists with $N _ { \mathrm { e } } \leq d ;$ top-K selection admits $N _ { \mathrm { e } } = \lceil d / K \rceil$ , but this smaller size need not hold for a general feasible family. In GFBs, a greedy maximal independent set is a valid dominating set, with $N _ { \mathrm { e } } \leq \alpha ( G )$

Remark III.6. In NS-CBs, $i f f _ { t }$ and A satisfy the preceding structural assumptions simultaneously for all contexts, we can set $A _ { \mathrm { e } }$ similarly. Otherwise, set $A _ { \mathrm { e } } = A ,$ , as A is finite.

The constructions in Propositions III.2–III.5 and Remark III.6 are one-time preprocessing steps performed before DAL is run, and do not introduce per-round overhead. In PBs and CFBs, the cost scales with the number of inspected candidate actions and the corresponding linear-independence checks, rather than with the horizon $T .$ . For continuous actions, the resolution of a fixed cover determines which changes are detectable. The corresponding propositions can be read as conditions on the environments covered by that choice, setting the resolution using the unknown true minimum change would instead require a prior. Spanning or covering alone does not ensure a uniform statistical margin, so the regret analysis retains the requirement $\Delta _ { \mathrm { c } } > 0$

Prior-Free Tuning. With the HLD considered here, DAL does not require $N _ { T }$ , the changepoints $\nu _ { k }$ , the minimum change size, or the segment lengths as inputs. Its exploration frequency uses the observed restart index $k ,$ and the HLD (GLR/GSR) thresholds use confidence levels and a known noise bound, not the unknown pre- or post-change means. In continuous settings, $A _ { \mathrm { e } }$ is fixed independently of the future changes. The quantities $L _ { T } , \Delta _ { \mathrm { c } } , m _ { \mathcal { D } }$ and $\ell _ { D }$ describe the detectable environments and the separation needed for the guarantee; their values are not supplied to DAL. Thus DAL is prior-free with respect to non-stationarity, while its guarantee remains conditional on detectable, sufficiently separated changes.

## B. On Order-Optimality in Piecewise Stationary Environments

In the PS setting, restarting a stationary learner at the true change-points gives a natural benchmark. DAL aims to attain the same regret order without knowing their number or locations. As in the detection-based analysis of [37], the intervals between change-points must be long enough to acquire the samples needed for detection. For brevity, we first define the relevant quantities and then state the assumption. For the contextual guarantee, assume a common finite support C, with $\mathcal { P } _ { t } ( c ) > 0$ for all $c \in { \mathcal { C } }$ and $t \in [ T ]$

Definition III.7. For the settings with a fixed context, i.e., PS-PBs, PS-KBs, PS-HBs, PS-CSBs, PS-CFBs and PS-GFBs, let $\begin{array} { r } { m _ { k } : = \left\lceil \frac { N _ { \mathrm { e } } } { \alpha _ { k } } \right\rceil m _ { \mathcal { D } } , \ell _ { k } : = \left\lceil \frac { N _ { \mathrm { e } } } { \alpha _ { k } } \right\rceil \ell _ { \mathcal { D } } } \end{array}$ . For PS-CBs, let $m _ { k } : =$ $\begin{array} { r l r } { \left\lceil \frac { N _ { 6 } } { \alpha _ { k } } \right\rceil \left\lceil \frac { m _ { \mathcal { D } } } { s } + \frac { \log T } { 4 s ^ { 2 } } + \sqrt { \frac { m _ { \mathcal { D } } \log T } { 2 s ^ { 3 } } + \frac { ( \log T ) ^ { 2 } } { 1 6 s ^ { 4 } } } \right\rceil , \ell _ { k } } & { : = } & { \left\lceil \frac { N _ { 6 } } { \alpha _ { k } } \right\rceil \left\lceil \frac { \ell _ { \mathcal { D } } } { s } + \frac { \log T } { 4 s ^ { 2 } } + \sqrt { \frac { \ell _ { \mathcal { D } } \log T } { 2 s ^ { 3 } } + \frac { ( \log T ) ^ { 2 } } { 1 6 s ^ { 4 } } } \right\rceil } \end{array}$ , for $k \in [ \dot { N } _ { T } ]$ , with $\begin{array} { r } { s : = \operatorname* { m i n } _ { \substack { c \in \mathcal { C } , t \in [ T ] : \mathcal { P } _ { t } ( c ) > 0 } } \dot { \mathcal { P } } _ { t } ( c ) } \end{array}$

Assumption III.8. For $N _ { T } \geq 1$ , assume $\nu _ { 1 } > m _ { 1 }$ and $\nu _ { k } - \nu _ { k - 1 } \geq \ell _ { k - 1 } + m _ { k }$ for $k \in \{ 2 , \ldots , N _ { T } \}$

In the fixed-context settings, DAL performs round-robin forced exploration every $\lceil N _ { \mathrm { e } } / \alpha _ { k } \rceil$ rounds, so each history receives at least one observation per cycle. In PS-CBs, a context– action pair is observed only in expectation, at least once every $\lceil N _ { \mathrm { e } } / \alpha _ { k } \rceil / s$ rounds. The larger windows in Definition III.7 ensure enough pre- and post-change samples with high probability. Assumption III.8 is used for the guarantee, but is not imposed in the experiments.

Based on Algorithm 1, DAL can incorporate any stationary bandit algorithm. Since different algorithms yield different regret guarantees, its regret depends on that of its input. To characterize DAL’s performance under piecewise stationarity, we employ the methodology of [37], incorporating the regret analysis of the stationary bandit algorithm and that of the change detector. The full proof is given in Appendix A-E.

Theorem III.9. For the PS setting, consider DAL with an HLD satisfying Property III.1, a stationary bandit algorithm B with the same concave, increasing expected-regret upper bound $R _ { B } ( n )$ for every local prefix length $n \leq T$ in every segment, and a covering set $A _ { \mathrm { e } }$ with $\Delta _ { \mathrm { c } } > 0$ whenever $N _ { T } \geq 1$ . If Assumption III.8 holds, set $\begin{array} { r } { \alpha _ { k } = \frac { \sqrt { k | \mathcal { C } | N _ { \mathrm { e } } } } { 2 \sqrt { T } \log ^ { 2 } T } , \delta _ { \mathrm { F } } = \delta _ { \mathrm { D } } = T ^ { - \eta } , \eta > 1 } \end{array}$ , then, DAL’s regret satisfies $\begin{array} { r } { R _ { T } = \widetilde { \mathcal { O } } \Big ( \big ( N _ { T } + 1 \big ) R _ { \mathcal { B } } \Big ( \frac { T } { N _ { T } + 1 } \Big ) + \bar { \Delta } \sqrt { | \mathcal { C } | N _ { \mathrm { e } } ( N _ { T } + 1 ) T } \Big ) } \end{array}$ , where $\bar { \Delta }$ is the maximum mean-reward gap. For CSBs and $G F B s ,$ , set $\delta _ { \mathrm { F } } = T ^ { - \ i }$ <sup>η</sup>/d instead.

TABLE I  
REGRET COMPARISONS FOR PS BANDITS, UNDER ASSUMPTION III.8. “•” DENOTES PRIOR-BASED TUNING. DAL ATTAINS THE BEST REGRET OR IMPROVES THE BEST EXISTING BOUNDS (BLUE = IMPROVEMENT).
<table><tr><td rowspan=1 colspan=1>Setting</td><td rowspan=1 colspan=1>Input Regret</td><td rowspan=1 colspan=1>Best Existing Regret</td><td rowspan=1 colspan=1>DAL Regret</td></tr><tr><td rowspan=1 colspan=1>PS-LBs</td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( d \sqrt { T } )$ [43]</td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( d \sqrt { N _ { T } T } )$ [33]</td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( d \sqrt { N _ { T } T } )$ </td></tr><tr><td rowspan=1 colspan=1>PS-GLBs</td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( d \sqrt { T } ) \ [ 7 ]$ </td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( d ^ { 2 / 3 } N _ { T } ^ { 1 / 3 } T ^ { 2 / 3 } ) ^ { \bullet }$ [27]</td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( d \sqrt { N _ { T } T } )$ </td></tr><tr><td rowspan=1 colspan=1>PS-SCBs</td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( d \sqrt { T } )$ [44]</td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( d ^ { 2 / 3 } N _ { T } ^ { 1 / 3 } T ^ { 2 / 3 } ) ^ { \bullet } \ [ 2 7 ]$ </td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( d \sqrt { N _ { T } T } )$ </td></tr><tr><td rowspan=1 colspan=1>PS-KBs</td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( \sqrt { \gamma _ { T } T } )$ [45]</td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( \sqrt { d \gamma _ { T } N _ { T } T } )$ [33]</td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( \sqrt { \gamma _ { T } N _ { T } T } )$ </td></tr><tr><td rowspan=1 colspan=1>PS-CBs</td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( \sqrt { | \mathcal { A } | T \log { | \Pi } | } )$ [46]</td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( \sqrt { | \mathcal { A } | N _ { T } T \log { | \Pi } | } )$ [34]</td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( \sqrt { | \mathcal { A } | N _ { T } T \log { | \Pi } | } )$ </td></tr><tr><td rowspan=1 colspan=1>PS-HBs (General $d , \beta )$  $\mathrm { P S  – H B s ~ } ( \mathrm { c o m m o n ~ } d _ { z } )$  $\mathrm { P S \mathrm { - } H B s ~ } ( d = 1 ~ \mathrm { o r } ~ \beta = 1 )$ </td><td rowspan=1 colspan=1> $\tilde { \mathcal O } ( T ^ { \frac { d + \beta } { d + 2 \beta } } )$ [9] $\tilde { \mathcal { O } } ( T ^ { \frac { d _ { z } + 1 } { d _ { z } + 2 } } )$ [9] $\tilde { \mathcal { O } } ( T ^ { \frac { d + \beta } { d + 2 \beta } } )$ [9]</td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( N _ { T } ^ { \frac { \beta } { d + 3 \beta } } T ^ { \frac { d + 2 \beta } { d + 3 \beta } } ) ^ { \bullet } \ [ 3 2 ]$  $\begin{array} { r } { { \tilde { \mathcal { O } } } ( N _ { T } ^ { \frac { 1 } { d _ { z } + 3 } } T ^ { \frac { d _ { z } + 2 } { d _ { z } + 3 } } ) ^ { \bullet } } \end{array}$ [32] $\tilde { \mathcal { O } } ( N _ { T } ^ { \frac { \beta } { d + 2 \beta } } T ^ { \frac { d + \beta } { d + 2 \beta } } )$ [35]</td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( N _ { T } ^ { \frac { \beta } { d + 2 \beta } } T ^ { \frac { d + \beta } { d + 2 \beta } } )$  $\tilde { \mathcal { O } } ( N _ { T } ^ { \frac { 1 } { d _ { z } + 2 } } T ^ { \frac { d _ { z } + 1 } { d _ { z } + 2 } } )$  $\tilde { \mathcal { O } } ( N _ { T } ^ { \frac { \beta } { d + 2 \beta } } T ^ { \frac { d + \beta } { d + 2 \beta } } )$ </td></tr><tr><td rowspan=1 colspan=1>PS-CSBs (General) $\mathrm { P S - C S B s \ ( T o p \mathrm { - } } K )$ </td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( K \sqrt { d T } )$ [12] $\tilde { \mathcal { O } } ( \sqrt { K d T } )$ [13]</td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( K \sqrt { d N _ { T } T } )$ [30] $\tilde { \mathcal { O } } ( K \sqrt { d N _ { T } T } )$ [30]</td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( K \sqrt { d N _ { T } T } )$  $\tilde { \mathcal { O } } ( \sqrt { K d N _ { T } T } )$ </td></tr><tr><td rowspan=1 colspan=1>PS-CFBs</td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( \sqrt { K d r T } ) ~ [ 4 3 ]$ </td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( \sqrt { K d r N _ { T } T \log | \boldsymbol { A } | } )$ [33]</td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( \sqrt { K d r N _ { T } T } )$ </td></tr><tr><td rowspan=1 colspan=1>PS-GFBs</td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( \sqrt { \alpha ( G ) T } ) \ [ 1 6 ]$ </td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( \sqrt { \chi _ { \mathrm { c l } } ( G ) N _ { T } T } ) ~ [ 1 7 ]$ </td><td rowspan=1 colspan=1> $\tilde { \mathcal { O } } ( \sqrt { \alpha ( G ) N _ { T } T } )$ </td></tr></table>

Using Theorem III.9, Propositions III.2–III.5 and Remark III.6, we present the regret of DAL.

Corollary III.10. Assume the conditions of Theorem III.9. In PBs, KBs, CBs, CSBs, CFBs and GFBs, suppose $R _ { B } ( n ) = \widetilde { \mathcal { O } } ( C _ { B } \sqrt { n } )$ and select the corresponding covering set with $\bar { \Delta } \sqrt { | \mathcal { C } | N _ { \mathrm { e } } } =$ $\mathcal { O } ( C _ { B } )$ . Then DAL attains $R _ { T } = \widetilde { \mathcal { O } } \Big ( C _ { B } \sqrt { ( N _ { T } + 1 ) T } \Big )$ . In HBs, a fixed cover and an input with $R _ { B } ( n ) = \widetilde { \mathcal { O } } ( n ^ { ( d _ { z } + 1 ) / ( d _ { z } + 2 ) } )$ give $R _ { T } = \widetilde { \mathcal { O } } \big ( ( N _ { T } + 1 ) ^ { 1 / ( d _ { z } + 2 ) } T ^ { ( d _ { z } + 1 ) / ( d _ { z } + 2 ) } \big )$

That is, if the stationary algorithm has order-optimal regret and its learning term dominates exploration, DAL retains its rate under the stated detection conditions. Corollary III.10 yields regret bounds across all considered settings while allowing flexible choices of the stationary algorithm. Choosing the stationary algorithms in Section IV, which achieve optimal stationary regret, leads to the bounds summarized in Table I.

## C. On Drifting Environments

While our results in the previous section focused on PS environments, naturally we ask whether DAL performs well in drifting environments. As a first step to study this, we perform an experiment with LBs, GLBs and SCBs, the three parametric settings covered by WeightUCB. Specifically, the parameter $\theta _ { t }$ at each time-step evolves according to a projected random walk, $\theta _ { t + 1 } : = \operatorname { P r o j } _ { \{ \theta : \| \theta \| \leq S \} } ( \theta _ { t } + \zeta _ { t + 1 } )$ , where $\zeta _ { t + 1 } \in \mathbb { R } ^ { d }$ is chosen uniformly over a δ-ball. We compare the cumulative dynamic regret up to time $T$ of DAL with GLR and the corresponding stationary learner, and WeightUCB, over a range of $\delta \mathbf { \dot { s } }$ in Figure 1. WeightUCB receives the true variation of each run. The LB experiment uses $d = 5$ and $\varepsilon _ { t } \sim \mathcal { N } ( 0 , 0 . 1 )$ ; the remaining parameters, and those of GLBs and SCBs, are as in Section IV. We use 30 runs per point.

We now shed light on our hypothesis behind the observations from Figure 1. Write $h _ { t + 1 } : =$ $\theta _ { t + 1 } - \theta _ { t }$ for the actual increment after projection. Since projection is nonexpansive and $\theta _ { t }$ lies in the parameter ball, $\| h _ { t + 1 } \| \leq \delta$ . In LBs, for playing an action $a \in { \mathcal { A } }$ at time $t + 1$ , we get the random reward $X _ { t + 1 } = \langle \theta _ { t } , a \rangle + \langle h _ { t + 1 } , a \rangle + \varepsilon _ { t + 1 }$ . If $\theta _ { t }$ does not change, then the reward from playing action a would have been $X _ { t + 1 } ^ { \prime } = \langle \theta _ { t } , a \rangle + \varepsilon _ { t + 1 } ^ { \prime }$ , where $\boldsymbol \varepsilon _ { t + 1 } ^ { \prime }$ is another realization of the noise. When $\delta$ is small relative to the observation noise, the small mean-shift $\left. h _ { t + 1 } , a \right.$ may be difficult to distinguish from noise over a short window. The same reasoning applies to GLBs and SCBs: if the link has Lipschitz constant $L _ { \mu }$ on the bounded parameter domain, the per-round mean shift is at most $L _ { \mu } \| a \| \delta$

For practical purposes, a slowly drifting environment may therefore resemble a stationary one over the learner’s recent observations. As a result, one expects an algorithm tailored to handle piecewise stationarity to perform reasonably well for slowly drifting environments, restarting when the accumulated mean shift becomes appreciable. Conversely, if δ is large, the accumulated variation can invalidate the stationary learner’s estimates before a reliable restart is possible. This is a statistical explanation for the observed behavior, not an application of the PS theorem to drift: the effective variation need not be zero-mean, and the detector’s piecewise-stationary latency guarantee does not by itself cover an arbitrary random walk.

The experiments support this explanation across the three reward models. DAL performs better than WeightUCB for smaller values of $\delta ,$ but the conclusion reverses upon increasing $\delta$ in LBs and SCBs. At the smallest tested radii, WeightUCB incurs 7.3 times DAL’s regret in GLBs and 3.4 times in SCBs. In SCBs, WeightUCB overtakes DAL beyond $\delta \approx 1$ , with parameter radius $S = 3 .$ . In GLBs, it approaches DAL but remains 3–4% above it up to $\delta = 1$ . Thus, small per-round variation favors DAL in these experiments, whereas a weighted method that tracks drift can overtake it as the variation grows.

![](images/0f74e056521a5d6fc7c30468a66f290a28015b5fb3294ca492e8ebe11d27d83d.jpg)

![](images/13e7a6d656c48bc3477ecb236ffe94655c715c2087415b542f0a6c9ab147c627.jpg)

![](images/d967e7c3aa3f30a97c019c0d88b76084284a4a990380996a062ee9fb656e08a0.jpg)  
Fig. 1. Final dynamic regret vs. radius δ of the drift, mean ± one standard deviation over 30 runs: DAL vs. WeightUCB; the range of δ differs between panels. DAL performs best at small drift radii; WeightUCB overtakes DAL beyond $\delta \approx 0 . 1$ in LBs and $\delta \approx 1$ in SCBs.

## IV. PRACTICAL PERFORMANCE

## A. Experimental Baselines

We evaluate the relevant prior-free and prior-based methods referenced in Section I, highlighting the strongest state-of-the-art algorithms applicable to PS and drifting settings across both synthetic and real-world benchmarks. These include ADA-OPKB [33] for LBs and KBs, ADA-ILTCB+ [34] for CBs, ZoomingTS-R [32] and MDBE [35] for HBs, GLR-CUCB [36], CUCB-SW and ADA-LCMAB [30] for CSBs, and UCB-NEW, SEASIDE and ASG [17] for GFBs. In CFBs, the incidence-vector representation permits applying RestartUCB [31], SW-LinUCB [23], D-LinUCB [24], and MASTER [22]. We also compare against the prior-based discounted approaches WeightUCB [27] in parametric models and WGP-UCB [28] in KBs, together with the applicable sliding-window and budget-restart methods.

Unless otherwise stated, all methods are tuned according to their original works. To maintain readability, we group algorithms by paradigm (discounted, sliding-window, budget-restart) while keeping distinct methods separate when they differ meaningfully. In real-world experiments, we focus on the strongest methods from the corresponding synthetic comparisons.

## B. Practical Tuning of DAL

Detector and Bandit Selection. Across all settings, DAL uses the Generalized Likelihood Ratio (GLR) and the Generalized Shiryaev–Roberts (GSR) tests [42] as the detector D. We use the practical thresholds $\beta _ { \mathrm { G L R } } ( n , \delta _ { \mathrm { F } } ) = \log ( n ^ { 3 / 2 } / \delta _ { \mathrm { F } } ) , \beta _ { \mathrm { G S R } } ( n , \delta _ { \mathrm { F } } ) = n ^ { 5 / 2 } / \delta _ { \mathrm { F } }$ , with $\delta _ { \mathrm { F } } = T ^ { - 1 / 2 }$ following [37, 40]. Concretely, Gaussian GLR and GSR pair with LinUCB [43] in LBs, GLM-UCB [7] in GLBs, REDS [45] in KBs, HOO [9] in HBs, CUCB [12] in CSBs, LinUCB on incidence vectors in CFBs, and UCB-N [14] in GFBs. Bernoulli GLR and GSR pair with OFUGLB [44] in SCBs and SquareCB [46] in CBs. Unless stated otherwise, we use the exploration schedule in Theorem III.9, $\alpha _ { k } = \sqrt { k | \mathcal { C } | N _ { \mathrm { e } } } / ( 2 \sqrt { T } \log ^ { 2 } T )$

Construction of $A _ { \mathrm { e } } .$ For PBs and CFBs, we follow Proposition III.2: we greedily select linearly independent action vectors until they span all feasible action vectors, using incidence vectors in CFBs. In CFBs, merely covering all items is insufficient because item changes may cancel in the observed sums; spanning the incidence vectors prevents a reward-function change from leaving every monitored mean unchanged. For continuous KBs and HBs, we use cover centers as in Propositions III.3 and III.4; in HBs, a regular grid ensures that every action lies near a monitored point. The finite KB action sets in our experiments are monitored in full. In CBs, we take $A _ { \mathrm { { e } } } = A .$ , as noted in Remark III.6. For CSBs and GFBs, we follow Proposition III.5. In CSBs, we repeatedly add a feasible action containing an uncovered item until every active item is covered; this suffices because item outcomes are observed separately. In GFBs, we construct a greedy maximal independent set by repeatedly selecting an uncovered vertex and marking it and its neighbors as covered until every vertex is covered. The resulting set is dominating, so every arm is observed during each completed forced cycle. These are one-time preprocessing steps performed independently of the unknown changes.

Practical Variant. In the finite-action experiments, we also monitor the observations obtained by B, using reward histories for actions or histories for individual items and neighboring arms. This often detects changes before a forced cycle is completed. Theorem III.9 analyzes the forced-only histories of Algorithm 1; the all-observation implementation is the practical variant which improves performance. This does not affect the theoretical properties of the algorithm, and we employ this variation for our experiments.

## C. Synthetic Experiments

We evaluate DAL in nine bandit settings under piecewise stationarity and drifting nonstationarity. The two HB instances and the two CSB action families give eleven benchmarks in total. These experiments vary the reward model, action space, available feedback, and frequency of change, while retaining the same detector-and-restart design.

Common Parameters. We use $T = 5 0 { , } 0 0 0$ and average results over 15 independent runs. In PBs, KBs and CBs, the action set contains 100 actions of dimension 10, generated as described below, and the contexts in CBs also have dimension 10. The HB, combinatorial and graph benchmarks have structured action sets (grids, combinatorial families and a graph), which we describe together with their benchmarks below.

PBs. Actions are sampled from $\mathcal { N } ( 0 , I )$ and rescaled to an L-ball. We use $S = L = 1$ for LBs and GLBs and $( L , S ) = ( 1 , 3 )$ for SCBs. At each initialization or change, entries of $\theta _ { t }$ are drawn i.i.d. $\operatorname { U n i f } [ - 1 , 1 ]$ and then rescaled to satisfy $\| \theta _ { t } \| \leq S$ . The GLB and SCB link is $\mu ( x ) = ( 1 + e ^ { - x } ) ^ { - 1 }$ . Noise in LBs and GLBs is $\varepsilon _ { t } \sim \mathcal { N } ( 0 , 0 . 0 1 )$ per round, and the SCB rewards are sampled as Bernoulli $\left( \mu ( \left. \theta _ { t } , A _ { t } \right. ) \right)$

KBs. We use the squared-exponential kernel with length scale 0.2 and noise $\mathcal { N } ( 0 , 0 . 0 1 )$ Following [28], reward functions are generated as finite kernel expansions, $\begin{array} { r } { f ( { a } ) = \sum _ { i = 1 } ^ { 2 0 0 } \alpha _ { i } k ( { a } , { x } _ { i } ) } \end{array}$ with $\alpha _ { i } \sim \mathrm { U n i f } [ - 1 , 1 ]$ . The covering set is the full finite action set.

CBs. The contexts $C _ { t } \in \mathbb { R } ^ { d _ { c } }$ with $d _ { c } = 1 0$ are randomly drawn each round from a fixed pool of 1000 normalized vectors. At every initialization or change, at least one of $f _ { t }$ or $\mathcal { P } _ { t }$ changes. For $a \in { \mathcal { A } } .$ , define the [0, 1]-clipped mean reward $f _ { t } ( C _ { t } , a ) = \Big [ b _ { a } + z ^ { ( \sigma ) } \sigma ( u _ { a } ^ { \top } C _ { t } ) +$ $\begin{array} { r } { z ^ { ( s ) } \sin ( v _ { a } ^ { \top } C _ { t } ) + z ^ { ( x ) } C _ { t , 2 } C _ { t , 3 } \Big ] _ { [ 0 , 1 ] } , } \end{array}$ where $u _ { a } , v _ { a } \sim \mathcal { N } ( 0 , I ) , b _ { a } \sim \mathrm { U n i f } [ 0 . 3 , 0 . 7 ]$ , and $z ^ { ( \sigma ) } , z ^ { ( s ) } , z ^ { ( x ) }$ are drawn uniformly from [0.25, 0.45], [0.15, 0.35] and [0.10, 0.25], respectively. Rewards are sampled as Bernoulli $\left( f _ { t } ( C _ { t } , A _ { t } ) \right)$ . Since there is no arm-related structure, we set $A _ { \mathrm { e } } ~ = ~ A$ (Remark III.6).

HBs. Two benchmarks use $( d , \beta ) \in \{ ( 2 , 1 ) , ( 1 , 1 / 2 ) \}$ and $f _ { t } ( a ) = 0 . 9 - 0 . 6 5 \| a - \theta _ { t } \| _ { \infty } ^ { \beta }$ . Under abrupt changes, $\theta _ { t }$ is redrawn uniformly from $[ 0 . 1 , 0 . 9 ] ^ { d }$ . Under drift, it moves linearly from (0.22, 0.24) to (0.78, 0.76), using the first coordinate for $d = 1$ , with a common offset drawn from $\operatorname { U n i f } [ - 0 . 0 1 5 , 0 . 0 1 5 ] ^ { d }$ . HOO operates on 121 grid points in one dimension or a $2 1 \times 2 1$ grid in two dimensions. The detection cover has nine regularly spaced probes. Observation noise is $\mathcal { N } ( 0 , 0 . 1 ^ { 2 } )$ .

CSBs. In the combinatorial and graph benchmarks, the item or arm means are permutations of d equally spaced values in [0.08, 0.84]. Every abrupt change draws a new permutation; drift interpolates from the initial permutation to its cyclic rotation by $\lfloor d / 3 \rfloor$ . Top-K selection uses $d = 1 2 , K = 3$ and four covering actions. Shortest paths use a layered graph with three nodes per layer, 15 edges and nine paths of length three. Each path includes a distinct middle edge, so all nine paths are necessary to cover the edges. Action optimization is exact by enumeration. Since all paths have equal length, maximizing reward is equivalent to minimizing edge costs $1 - X _ { t , j }$ . Semi-bandit item noises have variance $0 . 1 ^ { 2 }$

CFBs. The full-bandit benchmark has $d = 8 , K = 3$ and all 56 feasible triples, with item means generated as above. Its eight covering actions are {012, 013, 014, 015, 016, 017, 023, 123} on items $\{ 0 , \ldots , 7 \}$ ; their incidence vectors span $\mathbb { R } ^ { 8 }$ . Independent item noises have variance $0 . 1 ^ { 2 } / 3$ , so the observed sum has variance $0 . 1 ^ { 2 }$

GFBs. The feedback graph is a self-observing 12-cycle, with means and drift as above. Each action reveals its own observation and its two neighbors. The cover $\{ 0 , 3 , 6 , 9 \}$ is a maximal independent set and is minimal as a dominating set: one action observes three vertices. Here $\alpha ( G ) = 6$ , and a partition into six edges gives $\chi _ { \mathrm { c l } } ( G ) = 6$ . Noise has variance $0 . 1 ^ { 2 }$

Piecewise Stationarity. We adopt the geometric change-point model [38], and sample the intervals between the change-points according to a geometric distribution with parameter $\rho = T ^ { - \xi }$ for $\xi \in \ \{ 0 . 4 , 0 . 6 , 0 . 8 \}$ . The expected number of changes is approximately 660, 76 and 8.7, respectively. Parameters are redrawn independently at each change, according to the construction of each benchmark. We do not impose any restriction on the lengths of the intervals between change-points in our experiments.

Drifting Non-Stationarity. We drift linearly over T rounds from an initial value to a final value, where the end-points are chosen as in the beginning of the section. Specifically, $\theta _ { t } =$ $( 1 - t / T ) \theta _ { \mathrm { i n i t } } + ( t / T ) \theta _ { \mathrm { f i n a l } }$ in parametric, combinatorial and graph benchmarks; in HBs the same rule moves the peak. In KBs, we interpolate between $f _ { \mathrm { i n i t } }$ and $f _ { \mathrm { f i n a l } }$ , and in CBs between the coefficients defining the two reward functions.

Experimental Results. Figure 2 presents the dynamic regret across all settings and regimes. DAL follows its stationary learner between forced rounds and restarts learning when a change detector flags a mean shift in the monitored observations. Unlike budgeted restarting, it need not discard observations simply because a prescribed block has ended. Both GLR and GSR give strong performance across the reward and feedback models, including under drift. The gains are largest for long stationary segmentsm, while with frequent changes, more segments are missed and the advantage narrows, but persists. Regarding drifting non-stationarity, DAL performs well despite the absence of stationary segments. None of these experiments enforces the separation condition of the theory.

![](images/01a0b11642900f7362dd8c6251d587c1b5c2ec34016567d927ae8a81473ec1e2.jpg)  
Fig. 2. Dynamic regret in all synthetic benchmarks (lower is better), averaged over 15 runs. Rows correspond to the bandit settings and action families; the four columns use geometric change-points with $\xi = 0 . 4 , 0 . 6 , 0 . 8$ and drift, respectively.

## D. Real-World Experiments

To validate DAL’s practical effectiveness, we conduct experiments on nine real-world datasets spanning microarchitecture, financial time series, clinical-trial records, click logs, and sensor measurements. The constructions follow exactly [1]. Each comparison is averaged over 15 independent runs.

Micro-Prefetcher Selection. We evaluate on an NS bandit benchmark constructed from the microarchitecture prefetcher dataset of [47]. The environment has K = 11 prefetcher configurations (actions) with rewards given by normalized IPC in [0, 1] over a horizon $T = 2 6 2 2 4$ . Following [47], we also evaluate D-UCB [48] in its native form, while all other methods are modeled as NS-SCBs. Performance is measured by cumulative reward.

Stock Market Benchmarks. We consider two NS-KB benchmarks following [28]: one based on their original Quandl dataset and one constructed from stocks retrieved via Yahoo Finance.<sup>1</sup> In the Yahoo-based benchmark, we retain assets with sufficient history $( T = 2 0 0 0 )$ and select the 50 most volatile as actions. Rewards are derived from daily closing prices, the empirical price covariance is used as the kernel, and we add i.i.d. $\mathcal { N } ( 0 , 0 . 0 1 )$ noise to increase difficulty. Evaluation is by dynamic regret.

COVID-NMA Clinical Benchmark. We construct an NS-SCB benchmark from the open COVID-NMA database [49], modeling treatment effectiveness over time using a union endpoint on reported outcomes. Treatments are mapped into K = 13 classes and month-level counts are expanded and concatenated into $T \approx 7 . 4 \times 1 0 ^ { 4 }$ . Performance is evaluated by cumulative reward.

Click Log Benchmarks. We evaluate on the Yahoo! R6A click log dataset<sup>2</sup>, yielding an NS-SCB problem with K = 64 actions and horizon $T = 5 0 0 0 0 ;$ the metric is dynamic regret. In addition, we construct a fixed-arm replay benchmark from the Yahoo! R6B click logs as an NS-CB with 51 actions and $T = 5 0 0 0 0$ , following the logged-bandit evaluation approach of [50]; the metric is replay cumulative reward.

Live Traffic Benchmark. We build an NS-GLB benchmark from the Criteo live traffic dataset [51], following [24] but introducing changes via a geometric change-point model and using a horizon of $T = 5 0 0 0 0$ . Evaluation is based on dynamic regret.

![](images/e4dc6e457d4cf4dde108284e243f19c142d2da3fb5e9adfef19ea047fc371e1d.jpg)  
Fig. 3. Real-world results, averaged over 15 independent runs. Top: dynamic regret (lower is better). Bottom: cumulative reward (higher is better).

Sensor Correlation Benchmark. We use the Bioliq sensor dataset of [52] to construct an NS-SCB environment with K = 190 actions over a week-long horizon. Rewards are binary indicators derived from temporal correlation statistics following [52]. Performance is measured by cumulative reward.

Ad Recommendation Benchmark. We evaluate on the Zozo ad recommendation environment from [52] but retaining all K = 80 ads as actions. Rewards are binary click indicators and evaluation is by cumulative reward.

Experimental Results. Based on Figure 3, DAL consistently performs well across real-world benchmarks, in both dynamic regret and cumulative reward with both GLR and GSR, giving the lowest or closely overlapping leading curves. We attribute this strong performance to the robustness DAL demonstrates in the synthetic settings, which capture a range of challenging NS scenarios. These findings underscore DAL’s practical effectiveness. In what follows, we provide theoretical insights into its performance.

## V. BANDIT-DRIVEN FORCED EXPLORATION

The previous sections examined DAL’s stationary learner and detector. We now turn to its third component, forced exploration. Algorithm 1 cycles through $A _ { \mathrm { e } }$ in a fixed order. This ensures that every probe is observed, but does not account for the evidence collected so far: after a change, some probes may reveal a much larger mean shift than others. We therefore ask whether choosing where to probe can improve detection and regret without changing the stationary learner.

The Probe Selector. We use a rule inspired by the controlled-sensing method of [53], which treats observation streams as arms and change evidence as a reward. Their known-model sensing rule has a first-order asymptotic guarantee; its unknown-mean GLR version is studied empirically. We investigate the latter idea inside DAL. Let $G _ { j }$ and $N _ { j }$ be the GLR statistic and observation count of probe $j$ since the last restart. Group the forced slots into blocks of $W = \lceil w N _ { \mathrm { e } } \rceil$ slots, with $w \ge 1$ . At the start of every block, play every probe once in a fixed order. In each remaining slot, select the largest index

Algorithm 4 Bandit-driven probe selection at a forced slot   
Input: covering set ${ \mathcal A } _ { \mathrm { e } } ,$ block length $\overline { { W = \lceil w N _ { \mathrm { e } } \rceil } }$ , bonus $a > 0$   
Initialize once: $u  0 ; G _ { j } , N _ { j } , n _ { j } , s _ { j }  0$ for all probes   
1: $u \gets u + 1$   
2: if $u \leq N _ { \mathrm { e } }$ then select probe $j = u$   
3: else select $j \in \arg \operatorname* { m a x } _ { i } G _ { i } / N _ { i } + s _ { i } \sqrt { a \log u / n _ { i } }$   
4: Play $a ^ { ( j ) }$ , update its statistics, and apply DAL’s detector   
5: if DAL restarts then $G _ { j } , N _ { j } \gets 0$ for all probes   
6: if $u = W$ or DAL restarts then $u  0 ; n _ { j }  0 , s _ { j }  0$ for all probes

$$
\frac { G _ { j } } { N _ { j } } + s _ { j } \sqrt { \frac { a \log u } { n _ { j } } } ,\tag{3}
$$

where $u$ is the slot’s position in the block, $n _ { j }$ is the number of observations of probe $j$ in that block, and $s _ { j }$ is the root mean square of its GLR increments in the block. The first term favors evidence of a change; the bonus encourages sampling probes whose evidence is still variable. We set $G _ { j } = 0$ before two observations are available, and clear $G _ { j } , N _ { j }$ at every DAL restart.

Only the choice of action in a forced slot changes; the stationary learner, exploration frequency and restart rule remain the same. Every probe is still sampled in each block. This preserves coverage, but not necessarily the realized exploration regret, since the chosen actions and restart times can differ. The experiments below evaluate the direct selector in Algorithm 4.

Experimental Protocol. We compare the two probe rules in DAL (GLR) on the PS-LBs of Section IV-C, with $\xi = 0 . 8$ , horizons $T \in \{ 5 \times 1 0 ^ { 4 } , 2 \times 1 0 ^ { 5 } , 5 \times 1 0 ^ { 5 } , 1 0 ^ { 6 } \}$ . Detectors use only probe observations. We use 15 evaluation runs for $T \le 2 \times 1 0 ^ { 5 }$ and 10 otherwise. Regarding the the bandit-based probe selector we set $a = 0 . 0 1 , w = 1 . 5 .$

Experimental Results. At $T = 5 0 { , } 0 0 0 .$ , the main comparison gives nearly equal regret, 3055 for round-robin and 3083 for the bandit rule. At longer horizons, the benefit becomes clearer: regret decreases from 8176 to 7775 at 200,000, from 13,975 to 12,251 at 500,000, and from 20,989 to 18,320 at $1 0 ^ { 6 }$ , gains of approximately 5%, 12% and 13%. Figures 4 and 5 show the trajectories and horizon comparisons. With the smaller budget, the gains at the two longest horizons are approximately 16% and 9%. Longer segments allow more repeated probing, making the allocation of observations more useful.

![](images/671a6d2302aa25988dde1d1a9c448b17b8290980998d2e581695a25ae74f5c04.jpg)

![](images/0aaaf22941e6478bb46073ab57e7e5f7a79d8a1074bf0c987cfc3ad7bd286abb.jpg)

![](images/3eb3a93e748a8976de33af6c069d78544b929fcb809fb030eaa83ecdba63f6d3.jpg)

![](images/0dd49e0b37723e7ddf05561917adcdcaf5572eeb2f64765db4238908b2941d9c.jpg)

Fig. 4. Round-robin versus bandit-driven forced exploration in DAL (GLR), with $\xi = 0 . 8 .$ . Means and one standard error over 15 runs for $T \le 2 \times 1 0 ^ { 5 }$ and 10 otherwise.  
![](images/d4287af3d97d872b1f03924605dc709f9de257a72ca175e51566d2b5d9b8d66a.jpg)

![](images/a364f50b32daa5224aff5b00807fc18f3642dcfcc1d6404cec5c510f251eef46.jpg)  
(a) Final dynamic regret  
(b) Mean detection delay  
Fig. 5. Round-robin and bandit-driven forced exploration across horizons. Left: final regret with one-standard-error bars. Right: mean delay over detected changes.

The detection ablation shows how probe allocation changes both the delay and the number of detected changes. Mean delay decreases from 1624 to 1030 rounds at $T = 5 0 { , } 0 0 0 .$ , while detected changes increase from 5.1 to 6.3 out of 9.3. At $T = 1 0 ^ { 6 }$ , delay decreases from 7482 to 3288, and detected changes increase from 13.1 to 13.9 out of 15.7. At the intermediate horizons $T = 2 0 0 { , } 0 0 0$ and 500,000, the corresponding detected counts increase from 9.5 to 10.1 and from 11.2 to 12.6, respectively. Thus, the bandit rule detects approximately 6–24% more changes at this exploration scale across the tested horizons.

## VI. SUMMARY AND OUTLOOK

We introduced DAL, a practical, prior-free black-box framework for general NS bandits. Its plug-and-play design integrates seamlessly with a wide range of stationary bandit algorithms and different detectors, using forced exploration to monitor informative observations. The same design applies to parametric, kernelized, contextual and Holder bandits, combinatorial bandits¨ with semi-bandit or full-bandit feedback, and graph-feedback bandits.

On the theoretical side, we showed that DAL inherits and adapts the regret guarantees of its stationary input algorithm under detectable, sufficiently separated changes. A single regret bound retains the general stationary rate and makes the cost of exploration explicit. Appropriate covering sets yield the square-root rates in parametric and finite models, the benefits of item and graph feedback, and the instance-dependent rates of HBs. Regarding drifting non-stationarity, we hypothesized key conditions under which DAL excels—an insight further validated through experiments with different parametric reward models.

Through extensive experiments in both PS and drifting settings, spanning synthetic and realworld benchmarks, DAL consistently performs well against both prior-free and prior-based baselines. Its leading performance in real-world scenarios highlights its value as a practical and effective solution. Finally, our study of forced exploration shows that learning where to probe can shorten detection delay and improve regret when detection depends on repeated probing.

While DAL advances both theory and practice, it opens new directions. First, a direct regret analysis in drifting environments would deepen our understanding of how accumulated variation affects detection and learning. Second, the current regret bounds rely on separation and detectableshift conditions; weakening them would broaden the theoretical guarantees. The practical use of the learner’s observations also motivates guarantees for adaptive monitoring without a separate prescheduled history. Deepening the study of piecewise stationarity may be the key to tackling these broader challenges and DAL can serve as a solid foundation towards that goal.

## REFERENCES

[1] A. Gerogiannis, Y.-H. Huang, S. Bose, and V. Veeravalli, “DAL: A practical prior-free black-box framework for piecewise stationary bandits,” in Proceedings of the 43rd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, T. Zhang, M. Dudik, M. Jaggi, A. Agarwal, S. Li, D. Schuurmans, J. Zhu,

F. Berkenkamp, H. Dong, and A. Bietti, Eds., vol. 306. PMLR, 06–11 Jul 2026, pp. 34 611–34 638. [Online]. Available: https://proceedings.mlr.press/v306/gerogiannis26a.html

[2] L. Li, W. Chu, J. Langford, and R. E. Schapire, “A contextual-bandit approach to personalized news article recommendation,” in Proceedings of the 19th International Conference on World Wide Web, ser. WWW ’10. New York, NY, USA: Association for Computing Machinery, 2010, p. 661–670. [Online]. Available: https://doi.org/10.1145/1772690.1772758

[3] A. Flajolet and P. Jaillet, “Real-Time Bidding with Side Information,” in Advances in Neural Information Processing Systems, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds., vol. 30. Curran Associates, Inc., 2017. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2017/file/ 0bed45bd5774ffddc95ffe500024f628-Paper.pdf

[4] H. Robbins, “Some aspects of the sequential design of experiments,” Bulletin of the American Mathematical Society, vol. 58, no. 5, pp. 527 – 535, 1952.

[5] P. Auer, “Using confidence bounds for exploitation-exploration trade-offs,” Journal of Machine Learning Research, vol. 3, no. Nov, pp. 397–422, 2002.

[6] L. Faury, M. Abeille, C. Calauzenes, and O. Fercoq, “Improved Optimistic Algorithms for Logistic Bandits,” in Proceedings of the 37th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, H. D. III and A. Singh, Eds., vol. 119. PMLR, 13–18 Jul 2020, pp. 3052–3060. [Online]. Available: https://proceedings.mlr.press/v119/faury20a.html

[7] S. Filippi, O. Cappe, A. Garivier, and C. Szepesvari, “Parametric Bandits: The Generalized ´ Linear Case,” in Advances in Neural Information Processing Systems, J. Lafferty, C. Williams, J. Shawe-Taylor, R. Zemel, and A. Culotta, Eds., vol. 23. Curran Associates, Inc., 2010. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2010/file/ c2626d850c80ea07e7511bbae4c76f4b-Paper.pdf

[8] N. Srinivas, A. Krause, S. Kakade, and M. Seeger, “Gaussian process optimization in the bandit setting: No regret and experimental design,” in Proceedings of the 27th International Conference on Machine Learning (ICML-10), J. Furnkranz and¨ T. Joachims, Eds. Haifa, Israel: Omnipress, June 2010, pp. 1015–1022. [Online]. Available: http://www.icml2010.org/papers/422.pdf

[9] S. Bubeck, R. Munos, G. Stoltz, and C. Szepesvari, “X-armed bandits,” ´ J. Mach. Learn. Res., vol. 12, no. null, p. 1655–1695, Jul. 2011.

[10] M. Woodroofe, “A one-armed bandit problem with a concomitant variable,” Journal of the American Statistical Association, vol. 74, no. 368, pp. 799–806, 1979. [Online]. Available: https://www.tandfonline.com/doi/abs/10.1080/01621459.1979.10481033

[11] J. Langford and T. Zhang, “The epoch-greedy algorithm for multi-armed bandits with side information,” in Advances in Neural Information Processing Systems, J. Platt, D. Koller, Y. Singer, and S. Roweis, Eds., vol. 20. Curran Associates, Inc., 2007. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2007/file/ 4b04a686b0ad13dce35fa99fa4161c65-Paper.pdf

[12] W. Chen, Y. Wang, and Y. Yuan, “Combinatorial multi-armed bandit: General framework and applications,” in Proceedings of the 30th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, S. Dasgupta and D. McAllester, Eds., vol. 28, no. 1. Atlanta, Georgia, USA: PMLR, 17–19 Jun 2013, pp. 151–159. [Online]. Available: https://proceedings.mlr.press/v28/chen13a.html

[13] B. Kveton, Z. Wen, A. Ashkan, and C. Szepesvari, “Tight Regret Bounds for Stochastic Combinatorial Semi-Bandits,” in Proceedings of the Eighteenth International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, G. Lebanon and S. V. N. Vishwanathan, Eds., vol. 38. San Diego, California, USA: PMLR, 09–12 May 2015, pp. 535–543. [Online]. Available: https://proceedings.mlr.press/v38/kveton15.html

[14] S. Caron, B. Kveton, M. Lelarge, and S. Bhagat, “Leveraging side observations in stochastic bandits,” in Proceedings of the 28th Conference on Uncertainty in Artificial Intelligence, ser. Proceedings of Machine Learning Research, N. de Freitas and K. Murphy, Eds., vol. R10. PMLR, 14–18 Aug 2012, pp. 140–149, reissued by PMLR on 04 October 2026. [Online]. Available: https://proceedings.mlr.press/r10/caron12a.html

[15] Y. Russac, L. Faury, O. Cappe, and A. Garivier, “ Self-Concordant Analysis of Generalized´ Linear Bandits with Forgetting ,” in Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, A. Banerjee and K. Fukumizu, Eds., vol. 130. PMLR, 13–15 Apr 2021, pp. 658–666. [Online]. Available: https://proceedings.mlr.press/v130/russac21a.html

[16] T. Lykouris, E. Tardos, and D. Wali, “Feedback graph regret bounds for thompson sampling and ucb,” in Proceedings of the 31st International Conference on Algorithmic Learning Theory, ser. Proceedings of Machine Learning Research, A. Kontorovich and

G. Neu, Eds., vol. 117. PMLR, 08 Feb–11 Feb 2020, pp. 592–614. [Online]. Available: https://proceedings.mlr.press/v117/lykouris20a.html

[17] S. Lu, Y. Hu, and L. Zhang, “Stochastic bandits with graph feedback in non-stationary environments,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 10, 2021, pp. 8758–8766.

[18] S. Agrawal and R. Jia, “Learning in Structured MDPs with Convex Cost Functions: Improved Regret Bounds for Inventory Management,” in Proceedings of the 2019 ACM Conference on Economics and Computation, 2019.

[19] H. Cai, K. Ren, W. Zhang, K. Malialis, J. Wang, Y. Yu, and D. Guo, “Real-Time Bidding by Reinforcement Learning in Display Advertising,” in Proceedings of the Tenth ACM International Conference on Web Search and Data Mining (WSDM), 2017.

[20] C. Chen, H. Wei, N. Xu, G. Zheng, M. Yang, Y. Xiong, K. Xu, and Zhenhui, “Toward A Thousand Lights: Decentralized Deep Reinforcement Learning for Large-Scale Traffic Signal Control,” in AAAI Conference on Artificial Intelligence, 2020.

[21] J. Lu, C. Yang, X. Gao, L. Wang, C. Li, and G. Chen, “Reinforcement Learning with Sequential Information Clustering in Real-Time Bidding,” in Proceedings of the 28th ACM International Conference on Information and Knowledge Management, 2019.

[22] C.-Y. Wei and H. Luo, “Non-stationary Reinforcement Learning without Prior Knowledge: an Optimal Black-box Approach,” in Proceedings of Thirty Fourth Conference on Learning Theory, ser. Proceedings of Machine Learning Research, M. Belkin and S. Kpotufe, Eds., vol. 134. PMLR, 15–19 Aug 2021, pp. 4300–4354. [Online]. Available: https://proceedings.mlr.press/v134/wei21b.html

[23] W. C. Cheung, D. Simchi-Levi, and R. Zhu, “Learning to Optimize under Non-Stationarity,” in Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, K. Chaudhuri and M. Sugiyama, Eds., vol. 89. PMLR, 16–18 Apr 2019, pp. 1079–1087. [Online]. Available: https://proceedings.mlr.press/v89/cheung19b.html

[24] Y. Russac, C. Vernade, and O. Cappe, “Weighted Linear Bandits for Non-Stationary´ Environments,” in Advances in Neural Information Processing Systems, H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alche-Buc, E. Fox, and R. Garnett, Eds., vol. 32.´ Curran Associates, Inc., 2019. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2019/file/263fc48aae39f219b4c71d9d4bb4aed2-Paper.pdf

[25] L. Faury, Y. Russac, M. Abeille, and C. Calauzenes, “Regret Bounds for Generalized Linear\` Bandits under Parameter Drift,” 2021. [Online]. Available: https://arxiv.org/abs/2103.05750

[26] Y. Russac, O. Cappe, and A. Garivier, “Algorithms for Non-Stationary Generalized Linear´ Bandits,” 2020. [Online]. Available: https://arxiv.org/abs/2003.10113

[27] J. Wang, P. Zhao, and Z.-H. Zhou, “Revisiting Weighted Strategy for Non-stationary Parametric Bandits,” in Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, F. Ruiz, J. Dy, and J.-W. van de Meent, Eds., vol. 206. PMLR, 25–27 Apr 2023, pp. 7913–7942. [Online]. Available: https://proceedings.mlr.press/v206/wang23k.html

[28] Y. Deng, X. Zhou, B. Kim, A. Tewari, A. Gupta, and N. Shroff, “ Weighted Gaussian Process Bandits for Non-stationary Environments ,” in Proceedings of The 25th International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, G. Camps-Valls, F. J. R. Ruiz, and I. Valera, Eds., vol. 151. PMLR, 28–30 Mar 2022, pp. 6909–6932. [Online]. Available: https://proceedings.mlr.press/v151/deng22b.html

[29] X. Zhou and N. Shroff, “No-Regret Algorithms for Time-Varying Bayesian Optimization,” in 2021 55th Annual Conference on Information Sciences and Systems (CISS), 2021, pp. 1–6.

[30] W. Chen, L. Wang, H. Zhao, and K. Zheng, “Combinatorial semi-bandit in the non-stationary environment,” in Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence, ser. Proceedings of Machine Learning Research, C. de Campos and M. H. Maathuis, Eds., vol. 161. PMLR, 27–30 Jul 2021, pp. 865–875. [Online]. Available: https://proceedings.mlr.press/v161/chen21a.html

[31] P. Zhao, L. Zhang, Y. Jiang, and Z.-H. Zhou, “A Simple Approach for Non-stationary Linear Bandits,” in Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, S. Chiappa and R. Calandra, Eds., vol. 108. PMLR, 26–28 Aug 2020, pp. 746–755. [Online]. Available: https://proceedings.mlr.press/v108/zhao20a.html

[32] Y. Kang, C.-J. Hsieh, and T. Lee, “Online continuous hyperparameter optimization for generalized linear contextual bandits,” Transactions on Machine Learning Research, 2024. [Online]. Available: https://openreview.net/forum?id=lQE2AcbYge

[33] K. Hong, Y. Li, and A. Tewari, “An Optimization-based Algorithm for Non-stationary Kernel Bandits without Prior Knowledge,” in Proceedings of The 26th International

Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, F. Ruiz, J. Dy, and J.-W. van de Meent, Eds., vol. 206. PMLR, 25–27 Apr 2023, pp. 3048–3085. [Online]. Available: https://proceedings.mlr.press/v206/hong23b.html

[34] Y. Chen, C.-W. Lee, H. Luo, and C.-Y. Wei, “A new algorithm for non-stationary contextual bandits: Efficient, optimal and parameter-free,” in Proceedings of the Thirty-Second Conference on Learning Theory, ser. Proceedings of Machine Learning Research, A. Beygelzimer and D. Hsu, Eds., vol. 99. PMLR, 25–28 Jun 2019, pp. 696–726. [Online]. Available: https://proceedings.mlr.press/v99/chen19b.html

[35] N. Nguyen, S. Gaucher, and C. Vernade, “Non-stationary lipschitz bandits,” in Advances in Neural Information Processing Systems, D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, Eds., vol. 38, Main Conference. Curran Associates, Inc., 2025, pp. 15 129–15 180. [Online]. Available: https://proceedings.neurips. cc/paper files/paper/2025/file/1626be0ab7f3d7b3c639fbfd5951bc40-Paper-Conference.pdf

[36] H. Zhou, L. Wang, L. Varshney, and E.-P. Lim, “A Near-Optimal Change-Detection Based Algorithm for Piecewise-Stationary Combinatorial Semi-Bandits,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 04, pp. 6933–6940, Apr. 2020. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/6176

[37] Y.-H. Huang, A. Gerogiannis, S. Bose, and V. V. Veeravalli, “Detection augmented bandit procedures for piecewise stationary mabs: A modular approach,” IEEE Transactions on Information Theory, vol. 72, no. 9, pp. 7031–7048, 2026.

[38] A. Gerogiannis, Y.-H. Huang, and V. Veeravalli, “Is Prior-Free Black-Box Non-Stationary Reinforcement Learning Feasible?” in Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, Y. Li, S. Mandt, S. Agrawal, and E. Khan, Eds., vol. 258. PMLR, 03–05 May 2025, pp. 2692–2700. [Online]. Available: https://proceedings.mlr.press/v258/gerogiannis25a.html

[39] A. Gerogiannis, Y.-H. Huang, and V. V. Veeravalli, “Darling: Detection augmented reinforcement learning with non-stationary guarantees,” 2026. [Online]. Available: https://arxiv.org/abs/2604.16684

[40] L. Besson, E. Kaufmann, O.-A. Maillard, and J. Seznec, “Efficient Change-Point Detection for Tackling Piecewise-Stationary Bandits,” Journal of Machine Learning Research, vol. 23, no. 77, pp. 1–40, 2022.

[41] V. V. Veeravalli and T. Banerjee, “Chapter 6 - quickest change detection,” in

Academic Press Library in Signal Processing: Volume 3, ser. Academic Press Library in Signal Processing, A. M. Zoubir, M. Viberg, R. Chellappa, and S. Theodoridis, Eds. Elsevier, 2014, vol. 3, pp. 209–255. [Online]. Available: https://www.sciencedirect.com/science/article/pii/B9780124115972000060

[42] Y.-H. Huang and V. V. Veeravalli, “Sequential Change Detection for Learning in Piecewise Stationary Bandit Environments,” 2025. [Online]. Available: https://arxiv.org/abs/2501.10974

[43] Y. Abbasi-yadkori, D. Pal, and C. Szepesv´ ari, “Improved Algorithms for Linear Stochastic´ Bandits,” in Advances in Neural Information Processing Systems, J. Shawe-Taylor, R. Zemel, P. Bartlett, F. Pereira, and K. Weinberger, Eds., vol. 24. Curran Associates, Inc., 2011. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2011/file/ e1d5be1c7f2f456670de3d53c7b54f4a-Paper.pdf

[44] J. Lee, S.-Y. Yun, and K.-S. Jun, “A Unified Confidence Sequence for Generalized Linear Models, with Applications to Bandits,” in Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, Eds., vol. 37. Curran Associates, Inc., 2024, pp. 124 640–124 685.

[45] S. Salgia, S. Vakili, and Q. Zhao, “Random Exploration in Bayesian Optimization: Order-Optimal Regret and Computational Efficiency,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, Eds., vol. 235. PMLR, 21–27 Jul 2024, pp. 43 112–43 141. [Online]. Available: https://proceedings.mlr.press/v235/salgia24a.html

[46] D. Foster and A. Rakhlin, “Beyond UCB: Optimal and efficient contextual bandits with regression oracles,” in Proceedings of the 37th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, H. D. III and A. Singh, Eds., vol. 119. PMLR, 13–18 Jul 2020, pp. 3199–3210. [Online]. Available: https://proceedings.mlr.press/v119/foster20a.html

[47] G. Gerogiannis and J. Torrellas, “Micro-armed bandit: Lightweight & reusable reinforcement learning for microarchitecture decision-making,” in Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture, ser. MICRO ’23. New York, NY, USA: Association for Computing Machinery, 2023, p. 698–713. [Online]. Available: https://doi.org/10.1145/3613424.3623780

[48] L. Kocsis and C. Szepesvari, “Discounted ucb,” in ´ 2nd PASCAL Challenges Workshop, vol. 2,

2006, pp. 51–134. [Online]. Available: https://www.lri.fr/<sup>∼</sup>sebag/Slides/Venice/Kocsis.pdf

[49] I. Boutron, A. Chaimani, L. Ghosn, T. Evrenoglou, G. Ferrand, A. Jarde, H. Bonnet, C. Riveros, R. Assi, C. Grana, N. Henschke, D. Devane, J. J. Meerpohl, G. Rada,˜ G. Grasselli, A. Hrobjartsson, D. Tovey, and P. Ravaud, “Dataset-covid-nma initiative- ´ living systematic review on covid-19 treatments and vaccines,” Mar 2025. [Online]. Available: https://doi.org/10.5281/zenodo.14965887

[50] L. Li, W. Chu, J. Langford, and X. Wang, “Unbiased offline evaluation of contextualbandit-based news article recommendation algorithms,” in Proceedings of the Fourth ACM International Conference on Web Search and Data Mining, ser. WSDM ’11. New York, NY, USA: Association for Computing Machinery, 2011, p. 297–306. [Online]. Available: https://doi.org/10.1145/1935826.1935878

[51] E. Diemert, J. Meynet, P. Galland, and D. Lefortier, “Attribution Modeling Increases Efficiency of Bidding in Display Advertising,” in Proceedings of the ADKDD’17, ser. ADKDD’17. New York, NY, USA: Association for Computing Machinery, 2017. [Online]. Available: https://doi.org/10.1145/3124749.3124752

[52] J. Komiyama, E. Fouche, and J. Honda, “Finite-time Analysis of Globally Nonstationary´ Multi-Armed Bandits,” Journal of Machine Learning Research, vol. 25, no. 112, pp. 1–56, 2024. [Online]. Available: http://jmlr.org/papers/v25/21-0916.html

[53] Y.-H. Huang, A. Gerogiannis, S. Bose, and V. V. Veeravalli, “Learning where to look: Ucb-driven controlled sensing for quickest change detection,” 2026. [Online]. Available: https://arxiv.org/abs/2603.28563

## APPENDIX A THEORETICAL RESULTS

## A. Proof of Proposition III.2

In NS-PBs and NS-CFBs, the mean reward at time t is given by $f _ { t } ( a ) = \mu ( \langle \theta _ { t } , a \rangle )$ for all $a \in A ,$ where $\mu$ is injective; in CFBs, $\mu$ is the identity and actions are incidence vectors. Hence, equality of the mean rewards at an action is equivalent to equality of the corresponding inner products. This concerns the reward means, not inversion of noisy observations.

Suppose that $\mathcal { A } _ { \mathrm { { e } } } \subseteq \mathcal { A }$ is a maximal linearly independent subset of ${ \mathcal { A } } .$ Every action is a linear combination of its elements. If the inner products of two parameters agree on ${ \mathcal A } _ { \mathrm { e } } ,$ they therefore agree on every action in ${ \mathcal { A } } ,$ and injectivity of $\mu$ implies that the two reward functions coincide. Consequently, any change in the reward function changes the mean reward of at least one action in $A _ { \mathrm { e } }$ . Thus, $A _ { \mathrm { e } }$ can be taken to be any maximal linearly independent subset of ${ \mathcal { A } } ,$ with $N _ { \mathrm { e } } = \dim \operatorname { s p a n } ( { \cal A } ) \leq d ;$ this dimension is denoted by r in CFBs. The parameter itself need not be identifiable outside the action span.

## B. Proof of Proposition III.3

In this subsection, we establish the construction of $A _ { \mathrm { e } }$ in the NS-KB setting. Recall that $\nu _ { T }$ consists of the centers of a $\delta _ { T } \mathrm { - c o v e r }$ of ${ \mathcal { A } } ,$ with $\delta _ { T } = L _ { T } / ( 2 B L _ { u } )$ . Let $[ a ] _ { \mathrm { e } }$ denote the action in $\nu _ { T }$ closest to $a \in { \mathcal { A } }$ . Then, we can leverage the assumed Lipschitz property of the permitted RKHS functions to obtain the following upper bound:

$$
| f ( a ) - f ( [ a ] _ { \mathrm { e } } ) | \leq B L _ { u } \| a - [ a ] _ { \mathrm { e } } \| _ { 2 } \leq B L _ { u } \delta _ { T } .\tag{4}
$$

For $\tilde { a }$ in the proposition and successive reward functions $f , f ^ { \prime }$ , the triangle inequality yields

$$
\begin{array} { r l r } & { } & { | f ( [ \widetilde a ] _ { \mathrm { e } } ) - f ^ { \prime } ( [ \widetilde a ] _ { \mathrm { e } } ) | \ge | f ( \widetilde a ) - f ^ { \prime } ( \widetilde a ) | - | f ( \widetilde a ) - f ( [ \widetilde a ] _ { \mathrm { e } } ) | - | f ^ { \prime } ( \widetilde a ) - f ^ { \prime } ( [ \widetilde a ] _ { \mathrm { e } } ) | } \\ & { } & { \ge | f ( \widetilde a ) - f ^ { \prime } ( \widetilde a ) | - 2 B L _ { u } \delta _ { T } = | f ( \widetilde a ) - f ^ { \prime } ( \widetilde a ) | - L _ { T } > 0 . } \end{array}
$$

This indicates that the value of the reward function at $[ \tilde { a } ] _ { \mathrm { e } }$ must change by a nonzero amount. Thus, one can use observations from this action to detect the change. Taking the infimum over successive changes and using the strict inequality in the proposition gives a positive monitored margin. For the covering number on a general subset of $[ 0 , R ] ^ { d }$ , partition the box into cells of side at most $\delta _ { T } / \sqrt { d }$ and choose one action from each nonempty cell. There are at most $\lceil \sqrt { d } R / \delta _ { T } \rceil ^ { d }$ cells. On the full box, cell centers with side length at most $2 \delta _ { T } / \sqrt { d }$ give the sharper bound.

## C. Proof of Proposition III.4

Since $f - f ^ { \prime }$ is continuous on $[ 0 , 1 ] ^ { d }$ , choose a˜ with $| f ( \tilde { a } ) - f ^ { \prime } ( \tilde { a } ) | = \| f - f ^ { \prime } \| _ { \infty } \geq L _ { T }$ Let $[ \tilde { a } ] _ { \mathrm { e } }$ be a nearest cover center. Each permitted reward function satisfies $| f ( \tilde { a } ) - f ( [ \tilde { a } ] _ { \mathrm { e } } ) | \leq$ $L \| \tilde { { \boldsymbol { a } } } - [ \tilde { { \boldsymbol { a } } } ] _ { \mathrm { e } } \| _ { \infty } ^ { \beta } \leq L \delta _ { T } ^ { \beta }$ . Thus,

$$
| f ( [ { \tilde { a } } ] _ { \mathrm { e } } ) - f ^ { \prime } ( [ { \tilde { a } } ] _ { \mathrm { e } } ) | \geq | f ( { \tilde { a } } ) - f ^ { \prime } ( { \tilde { a } } ) | - 2 L \delta _ { T } ^ { \beta } \geq L _ { T } - 2 L \delta _ { T } ^ { \beta } > 0 .
$$

Taking the minimum over changes proves the margin bound. If $L _ { T } \ge 4 L \delta _ { T } ^ { \beta }$ , the right-hand side is at least $L _ { T } / 2$ . The witnessing point may differ between changes. The stated grid has radius $1 / ( 2 M ) \leq \delta _ { T } \mathrm { ~ i n ~ } \| \cdot \| _ { \infty }$ and contains $( M + 1 ) ^ { d }$ points. Its construction uses $\delta _ { T }$ , not $L _ { T }$ .

## D. Proof of Proposition III.5

For each item or arm $j ,$ fix a covering action that reveals its observation. Every cycle through $A _ { \mathrm { e } }$ therefore adds at least one observation to its history, whose mean is $\theta _ { t , j }$ . Hence the largest shift among the histories is $\| \theta - \theta ^ { \prime } \| _ { \infty }$ . In CSBs, $\begin{array} { r } { | f ( a ) - f ^ { \prime } ( a ) | = \left| \sum _ { j \in a } ( \theta _ { j } - \theta _ { j } ^ { \prime } ) \right| \leq K \| \theta - \theta ^ { \prime } \| _ { \infty } . } \end{array}$ so $\| \boldsymbol { f } - \boldsymbol { f } ^ { \prime } \| _ { \infty } \geq L _ { T }$ implies a monitored shift of at least $L _ { T } / K$ . Choosing one feasible revealing action for every active item, and removing duplicates, gives a CSB cover of at most d actions. In top-K selection, divide the items into groups of at most K and complete the last group with already covered items if exactly K items are required. This gives $\lceil d / K \rceil$ actions. In $\mathbf { G F B s } .$ $\| f - f ^ { \prime } \| _ { \infty } = \| \theta - \theta ^ { \prime } \| _ { \infty }$ and a maximal independent set is dominating: otherwise, an undominated vertex could be added. Its size is at most $\alpha ( G )$ , and $\alpha ( G ) \leq \chi _ { \mathrm { c l } } ( G )$ since an independent set contains at most one vertex from each clique of a clique partition.

## E. Proof of Theorem III.9

First, we partition the regret into two cases. If no false alarm occurs and all changes are detected within a short delay, we can separate the regret into three components: the regret due to forced exploration, the regret during the short detection (restart) delay after changes, and the regret incurred by the stationary bandit algorithm after the change is detected. If not, we use a crude linear bound to bound the regret and show that the probability of false alarm and that of late detection are low, which ensures that the regret due to detection failure is small [37].

We use the same decomposition for all the feedback models. For indexing the monitored histories within this proof, write

$$
\mathcal { I } = \left\{ \begin{array} { l l } { \mathcal { C } \times [ N _ { \mathrm { e } } ] , } & { \mathrm { f o r ~ s c a l a r - r e w a r d ~ h i s t o r i e s } , } \\ { [ d ] , } & { \mathrm { f o r ~ i t e m ~ o r ~ a r m ~ h i s t o r i e s ~ i n ~ C S B s ~ a n d ~ G F B s } . } \end{array} \right.
$$

For $j = \left( c , i \right)$ , let $\mathcal { H } _ { j } = \mathcal { H } _ { ( c , a ^ { ( i ) } ) }$ and $\mu _ { t , j } = f _ { t } ( c , a ^ { ( i ) } )$ ; for an item or arm $j ,$ let $\mu _ { t , j } = \theta _ { t , j }$ . The number $| \mathcal { I } |$ determines the false-alarm union bound, whereas $N _ { \mathrm { e } }$ remains the number of forced actions per cycle. Fixed contexts correspond to $| { \mathcal { C } } | = 1$

It suffices to consider $| { \mathcal { C } } | N _ { \mathrm { e } } ( N _ { T } + 1 ) \leq T$ , since otherwise the claimed bound already exceeds $\bar { \Delta } T$ . In this range clipping is inactive and the schedule satisfies $\alpha _ { k } < 1$ for $k \leq N _ { T } + 1$ . Consider a PS environment satisfying Assumption III.8. Let $\tau _ { k }$ be the kth detection point, with $\tau _ { 0 } = 0 \colon$

$$
\tau _ { k } : = \operatorname* { i n f } \left\{ t > \tau _ { k - 1 } : { \mathcal { D } } ( { \mathcal { H } } _ { j } ) = \operatorname* { d e t e c t i o n } { \mathrm { ~ f o r ~ s o m e ~ } } j \in { \mathcal { I } } \right\}\tag{5}
$$

Recall that $\nu _ { 0 } = 1$ and $\nu _ { N _ { T } + 1 } = T + 1$ . Define

$$
\mathcal G _ { k } : = \{ \forall l \in [ k - 1 ] , \ \tau _ { l } \in \{ \nu _ { l } , \dots , \nu _ { l } + \ell _ { l } - 1 \} \} \cap \{ \tau _ { k } \geq \nu _ { k } \} ,\tag{6}
$$

$$
\mathcal { E } _ { k } : = \{ \forall l \in [ k - 1 ] , \ \tau _ { l } \in \{ \nu _ { l } , \dots , \nu _ { l } + \ell _ { l } - 1 \} \} ,\tag{7}
$$

for $k \in [ N _ { T } + 1 ]$ , with ${ \mathcal { E } } _ { 1 }$ the universal space. Thus, $\mathcal { G } _ { k }$ means that the previous changes were detected within their latency and there is no false alarm before $\nu _ { k }$ . A detection at $\nu _ { k }$ is timely. First suppose every detection window ends before the horizon; the short final-segment case is handled below. Decomposing regret according to $\mathcal { G } _ { k }$ gives

$$
\begin{array} { r l } & { R _ { T } = \displaystyle \sum _ { k = 1 } ^ { N _ { T } + 1 } \mathbb { E } \left[ \displaystyle \sum _ { t = \nu _ { k - 1 } } ^ { \nu _ { k } - 1 } \left( \operatorname* { m a x } _ { \pi \in \Pi } f _ { t } ( C _ { t } , \pi ( C _ { t } ) ) - f _ { t } ( C _ { t } , A _ { t } ) \right) \right] } \\ & { \quad \le \displaystyle \sum _ { k = 1 } ^ { N _ { T } + 1 } \bar { \Delta } ( \nu _ { k } - \nu _ { k - 1 } ) \mathbb { P } ( \mathcal { G } _ { k } ^ { c } ) + \displaystyle \sum _ { k = 1 } ^ { N _ { T } + 1 } \mathbb { E } \left[ \mathbb { 1 } \{ \mathcal { G } _ { k } \} \sum _ { t = \nu _ { k - 1 } } ^ { \nu _ { k } - 1 } \left( \operatorname* { m a x } _ { \pi \in \Pi } f _ { t } ( C _ { t } , \pi ( C _ { t } ) ) - f _ { t } ( C _ { t } , A _ { t } ) \right) \right] , } \end{array}\tag{8}
$$

where $\bar { \Delta }$ is the largest mean-reward gap over contexts, actions and time. Decomposing the bad event according to the first unsuccessful detection yields

$$
\begin{array} { l } { \displaystyle \mathbb { P } ( \mathcal { G } _ { k } ^ { c } ) = \sum _ { l = 1 } ^ { k - 1 } \mathbb { P } \big ( \mathcal { E } _ { l } , \tau _ { l } \notin \{ \nu _ { l } , \dots , \nu _ { l } + \ell _ { l } - 1 \} \big ) + \mathbb { P } ( \mathcal { E } _ { k } , \tau _ { k } < \nu _ { k } ) } \\ { \displaystyle \qquad \leq \sum _ { l = 1 } ^ { k } \underbrace { \mathbb { P } ( \tau _ { l } < \nu _ { l } \mid \mathcal { E } _ { l } ) } _ { \displaystyle \Phi _ { 1 } } + \sum _ { l = 1 } ^ { k - 1 } \underbrace { \mathbb { P } ( \tau _ { l } \geq \nu _ { l } + \ell _ { l } \mid \mathcal { E } _ { l } ) } _ { \displaystyle \Phi _ { 2 } } . } \end{array}\tag{9}
$$

We bound $\Phi _ { 1 }$ and $\Phi _ { 2 }$ separately. In the conditional estimates below, first condition on the history through $\tau _ { l - 1 }$ , and then average over histories in $\mathcal { E } _ { l }$ . This fixes the phase of the forced schedule. Counts after that restart refer to its continuation without further resets, as in the definitions below. • Upper-bounding $\Phi _ { 1 }$ . Recall that $\mathcal { A } _ { \mathrm { e } } = \{ a ^ { ( i ) } : i \in [ N _ { \mathrm { e } } ] \}$ . Continue the forced schedule after $\tau _ { l - 1 }$ without further resets. Let $B _ { j } ( t )$ indicate whether its feedback at time t is added to history $j .$ For a scalar-reward history $j = \left( c , i \right)$

$$
B _ { ( c , i ) } ( t ) = 1 \left\{ C _ { t } = c , ( t - \tau _ { l - 1 } - 1 ) \mathrm { m o d } \left\lceil \frac { N _ { \mathrm { e } } } { \alpha _ { l } } \right\rceil = i - 1 \right\} .
$$

For an item or arm history, $B _ { j } ( t ) = 1$ when the prescribed forced action reveals that item or arm, and is zero on the learner’s rounds. Define the uth observation time of history $j$ after $\tau _ { l - 1 }$ by

$$
t _ { j , u } ^ { \prime } : = \operatorname* { i n f } \{ t > t _ { j , u - 1 } ^ { \prime } : B _ { j } ( t ) = 1 \} , \qquad t _ { j , 0 } ^ { \prime } = \tau _ { l - 1 } ,\tag{10}
$$

and the number of observations added to it by time t by

$$
n _ { j } ( t ) : = \sum _ { v = \tau _ { l - 1 } + 1 } ^ { t } B _ { j } ( v ) .\tag{11}
$$

Let $\tau _ { j }$ be its detector’s stopping time in its own sample count,

$$
\tau _ { j } : = \operatorname* { i n f } \{ u \in { \mathbb { N } } : \mathcal { D } ( \mathcal { H } _ { j } ) = \operatorname* { d e t e c t i o n } \mathrm { ~ a t ~ t i m e ~ } t _ { j , u } ^ { \prime } \} .\tag{12}
$$

Each detector is continued even if another history triggers first; statistical independence between histories is not required. Before $\nu _ { l }$ , the monitored mean is $\mu _ { \nu _ { l - 1 } , j }$ . A union bound and the detector’s false-alarm guarantee give

$$
\mathbb { P } ( \tau _ { l } < \nu _ { l } \mid \mathcal { E } _ { l } ) \le \sum _ { j \in \mathcal { I } } \mathbb { P } \big ( \tau _ { j } \le n _ { j } ( \nu _ { l } - 1 ) \mid \mathcal { E } _ { l } \big ) \le \sum _ { j \in \mathcal { I } } \mathbb { P } _ { \infty } \big ( \tau _ { j } \le T \mid \mathcal { E } _ { l } \big ) \le | \mathcal { I } | \delta _ { \mathrm { F } } .\tag{13}
$$

The stationary continuation agrees with the actual law up to $\nu _ { l } - 1$ . Conditional on the restart history and the future contexts, the observations of each fixed history are independent with a constant mean until that boundary. For item and arm histories, the assumption that the marginal law does not depend on which action reveals the outcome ensures the same property. Hence the false-alarm bound applies uniformly. Dependence between different items observed in one round does not affect this argument.

• Upper-bounding $\Phi _ { 2 } .$ . Let $j ^ { * }$ be a history with the largest monitored mean shift at $\nu _ { l } \mathbf { : }$

$$
j ^ { * } \in \mathop { \arg \operatorname* { m a x } } _ { j \in \mathcal { I } } | \mu _ { \nu _ { l } , j } - \mu _ { \nu _ { l } - 1 , j } | .\tag{14}
$$

Its shift is at least $\Delta _ { \mathrm { c } }$ . Define the sufficient-sample events

$$
\begin{array} { r } { \mathcal { M } _ { l } : = \{ n _ { j ^ { * } } ( \nu _ { l } - 1 ) - n _ { j ^ { * } } ( \pi _ { l - 1 } ) \geq m _ { \mathcal { D } } \} , \mathcal { L } _ { l } : = \{ n _ { j ^ { * } } ( \nu _ { l } + \ell _ { l } - 1 ) - n _ { j ^ { * } } ( \nu _ { l } - 1 ) \geq \ell _ { \mathcal { D } } \} . } \end{array}\tag{15}
$$

A late detection requires either insufficient observations or failure of the detector despite enough observations. Consequently,

$$
\mathbb { P } ( \pi _ { l } \geq \nu _ { l } + \ell _ { l } \mid \mathcal { E } _ { l } ) \leq \mathbb { P } ( \mathcal { M } _ { l } ^ { c } \mid \mathcal { E } _ { l } ) + \mathbb { P } ( \mathcal { L } _ { l } ^ { c } \mid \mathcal { E } _ { l } ) + \mathbb { P } ( \tau _ { l } \geq \nu _ { l } + \ell _ { l } \mid \mathcal { E } _ { l } \cap \mathcal { M } _ { l } \cap \mathcal { L } _ { l } ) .\tag{16}
$$

For each covering action $a ^ { ( i ) }$ , every interval after the restart contains the following number of forced opportunities:

$$
\sum _ { t = v + 1 } ^ { u } \mathbb { 1 } \left\{ ( t - \tau _ { l - 1 } - 1 ) \mathrm { m o d } \left\lceil \frac { N _ { \mathrm { e } } } { \alpha _ { l } } \right\rceil = i - 1 \right\} \geq \left\lfloor \frac { u - v } { \lceil N _ { \mathrm { e } } / \alpha _ { l } \rceil } \right\rfloor .\tag{17}
$$

Equality holds when $u - v$ is divisible by $\lceil N _ { \mathrm { e } } / \alpha _ { l } \rceil$ . For each item or arm history, fix one covering action that reveals it, as in Proposition III.5. Its observations satisfy the same lower

bound, and other covering actions can only add more samples. On $\mathcal { E } _ { l }$ , Assumption III.8 implies $\tau _ { l - 1 } \leq \nu _ { l } - m _ { l } - 1$ , including $l = 1$ since $\tau _ { 0 } = 0$ and $\nu _ { 1 } > m _ { 1 }$ . Thus the last $m _ { l }$ pre-change rounds lie after the restart. In CBs, write $j ^ { * } = \left( c ^ { * } , i ^ { * } \right)$ . Conditioning on the restart history gives

$$
\mathbb { E } \left[ n _ { j ^ { * } } ( \nu _ { l } - 1 ) - n _ { j ^ { * } } ( \nu _ { l } - m _ { l } - 1 ) \ | \ \mathcal { E } _ { l } , \tau _ { l - 1 } \right]
$$

$$
= \sum _ { t = { \nu _ { l } } - m _ { l } } ^ { \nu _ { l } - 1 } \mathcal { P } _ { t } ( c ^ { * } ) \mathbb { 1 } \left\{ ( t - \tau _ { l - 1 } - 1 ) \bmod \left\lceil \frac { N _ { \mathrm { e } } } { \alpha _ { l } } \right\rceil = i ^ { * } - 1 \right\} \geq \frac { s m _ { l } } { \lceil N _ { \mathrm { e } } / \alpha _ { l } \rceil } ,\tag{18}
$$

$$
\mathbb { E } \big [ n _ { j ^ { * } } ( \nu _ { l } + \ell _ { l } - 1 ) - n _ { j ^ { * } } ( \nu _ { l } - 1 ) \mid \mathcal { E } _ { l } , \tau _ { l - 1 } \big ] \ge \frac { s \ell _ { l } } { \lceil N _ { \mathrm { e } } / \alpha _ { l } \rceil } .\tag{19}
$$

The two windows contain exactly $m _ { l } / \lceil N _ { \mathrm { e } } / \alpha _ { l } \rceil$ and $\ell _ { l } / \lceil N _ { \mathrm { e } } / \alpha _ { l } \rceil$ opportunities. Applying Hoeffding’s inequality to the independent context indicators in these windows and then averaging over $\tau _ { l - 1 }$ yields

$$
\mathbb { P } ( \mathcal { M } _ { l } ^ { c } \mid \mathcal { E } _ { l } ) \leq \exp \left[ - \frac { 2 \left( s m _ { l } / \lceil N _ { \mathrm { e } } / \alpha _ { l } \rceil - m _ { \mathcal { D } } \right) ^ { 2 } } { m _ { l } / \lceil N _ { \mathrm { e } } / \alpha _ { l } \rceil } \right] \leq T ^ { - 1 } ,\tag{20}
$$

$$
\mathbb { P } ( \mathcal { L } _ { l } ^ { c } \mid \mathcal { E } _ { l } ) \le \exp \left[ - \frac { 2 \left( s \ell _ { l } / \left[ N _ { \mathrm { e } } / \alpha _ { l } \right] - \ell _ { D } \right) ^ { 2 } } { \ell _ { l } / \left[ N _ { \mathrm { e } } / \alpha _ { l } \right] } \right] \le T ^ { - 1 } .\tag{21}
$$

The last inequalities are precisely the quadratic inequalities solved by the two ceilings in Definition III.7. The pre-change bound uses only the last $m _ { l }$ rounds, rather than replacing the denominator for the whole pre-change history by that of a shorter window. For fixed contexts, including item and arm histories, both sufficient-sample events hold deterministically and their failure probabilities are zero.

For the third term of (16), first fix the restart history and context sequence. The change index of the witnessing history is then fixed. On $\mathcal { M } _ { l } \cap \mathcal { L } _ { l }$ , it has at least $m _ { D }$ pre-change samples and $\ell _ { D }$ post-change samples. Therefore,

$$
\begin{array} { r l } & { \mathbb { P } ( \tau _ { l } \geq \nu _ { l } + \ell _ { l } \mid \mathcal { E } _ { l } \cap \mathcal { M } _ { l } \cap \mathcal { L } _ { l } ) } \\ & { \quad \leq \mathbb { P } \big ( \tau _ { j ^ { * } } > n _ { j ^ { * } } ( \nu _ { l } + \ell _ { l } - 1 ) \mid \mathcal { E } _ { l } \cap \mathcal { M } _ { l } \cap \mathcal { L } _ { l } \big ) } \\ & { \quad \leq \mathbb { P } \big ( \tau _ { j ^ { * } } > n _ { j ^ { * } } ( \nu _ { l } - 1 ) + \ell _ { \mathcal { D } } \mid \mathcal { E } _ { l } \cap \mathcal { M } _ { l } \cap \mathcal { L } _ { l } \big ) \leq \delta _ { \mathrm { D } } . } \end{array}\tag{22}
$$

The final step is the latency guarantee, applied conditionally on the fixed sample times and then averaged. Substituting (20)–(22) into (16) gives

$$
\mathbb { P } ( \tau _ { l } \geq \nu _ { l } + \ell _ { l } \mid \mathcal { E } _ { l } ) \leq 2 T ^ { - 1 } + \delta _ { \mathrm { D } } .\tag{23}
$$

Together with (13), this gives

$$
\mathbb { P } ( \mathcal { G } _ { k } ^ { c } ) \le | \mathcal { I } | k \delta _ { \mathrm { F } } + ( k - 1 ) ( 2 T ^ { - 1 } + \delta _ { \mathrm { D } } ) .\tag{24}
$$

This bounds the first term in (8). For the second term, define $\bar { \alpha } : = \operatorname* { m a x } _ { k \in [ N _ { T } + 1 ] } \alpha _ { k }$ and $\ell _ { 0 } = 0$ Separating the detection delay, forced exploration and stationary learner gives

$$
\begin{array} { r l } & { \mathbb { E } \Bigg [ \mathbf { 1 } \{ \mathcal { G } _ { k } \} \sum _ { \ell = \nu _ { k - 1 } } ^ { \nu _ { k } - 1 } \big ( \operatorname* { m a x } _ { \tau } f _ { \ell } ( C _ { t } , \pi ( C _ { t } ) ) - f _ { \ell } ( C _ { t } , A _ { \ell } ) \big ) \Bigg ] } \\ & { \quad \leq \bar { \Delta } \ell _ { k - 1 } + \bar { \Delta } N _ { \mathrm { e } } \Bigg [ \frac { \nu _ { k } - \nu _ { k - 1 } } { \lceil N _ { \mathrm { e } } / \alpha _ { k } \rceil } \Bigg ] } \\ & { \quad \quad + \mathbb { E } \Bigg [ \mathbb { 1 } \{ \mathcal { G } _ { k } \} \sum _ { \ell = \nu _ { k - 1 } + 1 \atop ( t - \pi _ { k - 1 } - 1 ) \mathrm { ~ m a x } \in \mathbb { N } _ { \mathrm { e } } } \big ( \operatorname* { m a x } _ { \tau \in \mathbb { T } } f _ { \ell } ( C _ { t } , \pi ( C _ { t } ) ) - f _ { \ell } ( C _ { t } , A _ { \ell } ) \big ) \Bigg ] } \\ & { \quad \quad \leq \bar { \Delta } \ell _ { k - 1 } + \bar { \Delta } \lceil \tilde { \alpha } ( \nu _ { k } - \nu _ { k - 1 } ) + N _ { \mathrm { e } } \rceil + R _ { \mathrm { g } } \{ \nu _ { k } - \nu _ { k - 1 } \} . } \end{array}\tag{25}
$$

To justify the stationary term, condition on the forced observations, including their contexts, which determine the resets and $\mathcal { G } _ { k }$ . The analyzed DAL uses no learner observations for these decisions. With independent feedback across rounds and fresh learner initialization, its non-forced rounds after $\tau _ { k - 1 }$ therefore form a fresh stationary run of $\boldsymbol { B }$ of length at most $\nu _ { k } - \nu _ { k - 1 }$ . The expected bound and monotonicity of $R _ { B }$ apply to this run. The detector and learner observations are separate, as specified in Algorithm 1. Plugging (24) and (25) into (8) yields

$$
\begin{array} { r l } {  { R _ { T } \leq \sum _ { k = 1 } ^ { N _ { T } + 1 } \bar { \Delta } \big ( \nu _ { k } - \nu _ { k - 1 } \big ) \big [ | \mathcal { I } | k \delta _ { \mathrm { F } } + ( k - 1 ) ( 2 T ^ { - 1 } + \delta _ { \mathrm { D } } ) \big ] } } \\ & { \qquad \times \sum _ { k = 1 } ^ { N _ { T } + 1 } \big \{ \bar { \Delta } \ell _ { k - 1 } + \bar { \Delta } \big [ \bar { \alpha } ( \nu _ { k } - \nu _ { k - 1 } ) + N _ { \mathrm { e } } \big ] + R _ { \delta } \big ( \nu _ { k } - \nu _ { k - 1 } \big ) \big \} } \\ & { \qquad \times \sum _ { k = 1 } ^ { N _ { T } } \big \{ ( N _ { T } + 1 ) \delta _ { \mathrm { F } } + 2 \bar { \Delta } N _ { T } + \bar { \Delta } T N _ { T } \delta _ { \mathrm { D } } + \bar { \Delta } \sum _ { k = 1 } ^ { N _ { T } } \ell _ { k } } \\ & { \qquad \quad + \bar { \Delta } \big [ \bar { \alpha } T + ( N _ { T } + 1 ) N _ { \mathrm { e } } \big ] + ( N _ { T } + 1 ) R _ { B } \bigg ( \frac { T } { N _ { T } + 1 } \bigg ) . } \end{array}\tag{26}
$$

The last step uses Jensen’s inequality. If the last detection window extends to or past $T .$ , apply the same decomposition before the final change and charge the remaining rounds to $\bar { \Delta } \ell _ { N _ { T } }$ . This gives the same bound without a terminal-segment separation condition.

Finally substitute the theorem’s parameters. The exploration contribution satisfies

$$
\bar { \alpha } T = \frac { \sqrt { | \mathcal { C } | N _ { \mathrm { e } } ( N _ { T } + 1 ) T } } { 2 \log ^ { 2 } T } , \qquad ( N _ { T } + 1 ) N _ { \mathrm { e } } \leq \sqrt { | \mathcal { C } | N _ { \mathrm { e } } ( N _ { T } + 1 ) T } .
$$

Moreover,

$$
\left\lceil { \frac { N _ { \mathrm { e } } } { \alpha _ { k } } } \right\rceil \leq 1 + 2 \log ^ { 2 } T \sqrt { \frac { N _ { \mathrm { e } } T } { k | \mathcal { C } | } } , \qquad \sum _ { k = 1 } ^ { N _ { T } } k ^ { - 1 / 2 } \leq 2 \sqrt { N _ { T } } .
$$

In fixed-context settings this implies

$$
\sum _ { k = 1 } ^ { N _ { T } } \ell _ { k } \leq \ell _ { \mathcal { D } } \left[ N _ { T } + 4 \log ^ { 2 } T \sqrt { \frac { N _ { \mathrm { e } } N _ { T } T } { | \mathcal { C } | } } \right] .
$$

For CBs, replace $\ell _ { D }$ by the ceiling factor in Definition III.7, which is $\mathcal { O } ( ( \ell _ { D } + \log T ) / s ^ { 2 } )$ Property III.1 gives $\ell _ { \mathcal { D } } = \mathcal { O } ( \log T )$ . Thus the delay and exploration terms have the claimed order for fixed s. The failure terms are also dominated. For scalar-reward histories, $| \mathcal { I } | = | \mathcal { C } | N _ { \mathrm { e } }$ and $\delta _ { \mathrm { F } } = T ^ { - \eta }$ . Using $| { \mathcal { C } } | N _ { \mathrm { e } } ( N _ { T } + 1 ) \leq T$ and $\eta > 1$ gives

$$
\begin{array} { r } { T | \mathcal { I } | ( N _ { T } + 1 ) \delta _ { \mathrm { F } } \leq \sqrt { | \mathcal { C } | N _ { \mathrm { e } } ( N _ { T } + 1 ) T } . } \end{array}
$$

For CSBs and GFBs, $| { \mathcal { I } } | = d$ and the chosen per-history level $\delta _ { \mathrm { F } } = T ^ { - \eta } / d$ instead gives

$$
T | \mathcal { I } | ( N _ { T } + 1 ) \delta _ { \mathrm { F } } = ( N _ { T } + 1 ) T ^ { 1 - \eta } \leq \sqrt { ( N _ { T } + 1 ) T } .
$$

The sample requirement acquires an additional log d through $\log ( 1 / \delta _ { \mathrm { F } } )$ . Setting $ { \delta _ { \mathrm { D } } } = T ^ { - \eta }$ gives

$$
\begin{array} { r } { T N _ { T } \delta _ { \mathrm { D } } + 2 N _ { T } \leq 3 \sqrt { N _ { T } T } . } \end{array}
$$

Together with (26), these inequalities complete the proof.

## F. Proof of Corollary III.10

By Theorem III.9, it suffices to bound the exploration term for each choice of $A _ { \mathrm { e } }$ . The detector constants and context-recurrence parameter s are fixed, as in the theorem.

In PS-PBs, the inputs in Table I have $C _ { B } = \widetilde { \mathcal { O } } ( d )$ , and $N _ { \mathrm { e } } \leq d$ by Proposition III.2. Thus the $\sqrt { d ( N _ { T } + 1 ) T }$ exploration term is absorbed by the stationary learning term. In PS-KBs, select a cover satisfying Proposition III.3 with $N _ { \mathrm { e } } = \mathcal { O } ( C _ { B } ^ { 2 } )$ . Then both terms are $\widetilde { \mathcal { O } } ( C _ { B } \sqrt { ( N _ { T } + 1 ) T } )$ When a finite action set is smaller than the prescribed cover, it may instead be monitored in full. In PS-CBs, $N _ { \mathrm { e } } \leq | { \mathcal { A } } |$ . For the universal policy class, $| \Pi | = | { \mathcal { A } } | ^ { | { \mathcal { C } } | }$ , so $| \mathcal { C } | | \mathcal { A } | = \mathcal { O } ( | \mathcal { A } | \log | \Pi | )$ when $| { \mathcal { A } } | \geq 2$ . This absorbs the exploration term into the finite-policy input’s rate; the singleaction case has zero regret. More generally, the covering condition in the corollary gives the same conclusion.

In PS-HBs, HOO with a compatible hierarchical partition has the stated stationary rate under the common, attained packing bound described in Section I [9, proof of Theorem 6]. The cover of Proposition III.4 has fixed size. Since $N _ { T } + 1 \leq T$ and $d _ { z } \geq 0$

$$
\begin{array} { r } { \sqrt { ( N _ { T } + 1 ) T } \le ( N _ { T } + 1 ) ^ { 1 / ( d _ { z } + 2 ) } T ^ { ( d _ { z } + 1 ) / ( d _ { z } + 2 ) } . } \end{array}
$$

The exploration term is therefore dominated by the HOO input’s restarted rate. Taking $d _ { z } = d / \beta$ gives the worst-case Holder exponent.¨

In PS-CSBs, $\bar { \Delta } \leq K$ and $C _ { B } = \sqrt { d K }$ for the input in Table I. Hence $N _ { \mathrm { e } } = \mathcal { O } ( d / K )$ gives $\bar { \Delta } \sqrt { N _ { \mathrm { e } } } = \mathcal { O } ( \sqrt { d K } )$ . For a general feasible family, the bound is instead

$$
R _ { T } = \widetilde { \mathcal { O } } \Big ( ( \sqrt { d K } + K \sqrt { N _ { \mathrm { e } } } ) \sqrt { ( N _ { T } + 1 ) T } \Big ) .
$$

In PS-CFBs, $N _ { \mathrm { e } } = r$ and $\bar { \Delta } \leq K$ . On the incidence span, $\| x _ { a } \| _ { 2 } \leq \sqrt { K }$ and the parameter norm is at most ${ \sqrt { d } } .$ . With ridge parameter K and confidence-failure probability $T ^ { - 2 }$ , the OFUL confidence and elliptical-potential bounds [43] give

$$
R _ { B } ( n ) = \widetilde { \mathcal { O } } \Big ( ( \sqrt { K d } + \sigma _ { \Sigma } \sqrt { r } ) \sqrt { r n } \Big ) .
$$

Under the table’s aggregate-noise condition this has $C _ { B } = \sqrt { K d r }$ , and $K \sqrt { r } \le \sqrt { K d r }$ since $K \leq d .$ . In PS-GFBs, a maximal independent set gives $N _ { \mathrm { e } } \leq \alpha ( G )$ and UCB-N has $C _ { B } = \sqrt { \alpha ( G ) }$ These choices likewise absorb the exploration term, completing the proof.