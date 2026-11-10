# APO2CS: Advantage-Aware Preference Optimization for Two-Level Constructive Search in Combinatorial Optimization

Kaixi Yang   
National University of Defense   
Technology   
Changsha, China   
kxyang1998@nudt.edu.cn   
Linbo Qiao<sup>✉</sup>   
National University of Defense   
Technology   
Changsha, China   
qiao.linbo@nudt.edu.cn   
Yanqi Shi   
National University of Defense   
Technology   
Changsha, China   
yqshi@nudt.edu.cn   
Wei Shang   
National University of Defense   
Technology   
Changsha, China   
shangwei24@nudt.edu.cn   
Dongsheng Li   
National University of Defense   
Technology   
Changsha, China   
dsli@nudt.edu.cn

## Abstract

Preference optimization has emerged as a promising alternative to reward-based training for neural combinatorial optimization. However, existing methods typically rely on binary pairwise labels, which discard the magnitude of solution-quality diferences and provide supervision only at the complete-solution level. We propose Advantage-aware Preference Optimization for Two-level Constructive Search (APO2CS), a preference-based training objective for constructive neural solvers that integrates both macro-level and micro-level supervision. At the macro level, APO2CS enriches pairwise preferences with an advantage term derived from objective value diferences. At the micro level, it compares partial trajectories, thereby providing denser supervision throughout the sequential construction process. The resulting objective can be used either to train a policy from scratch or to fine-tune a pretrained one. Experimental results show that APO2CS improves the expressiveness and search capability of preference learning for combinatorial optimization, while demonstrating strong efectiveness and adaptability across multiple combinatorial optimization tasks.

## CCS Concepts

• Computing methodologies → Artificial intelligence; Machine learning; Reinforcement learning.

## Keywords

Combinatorial Optimization, Learning to Optimize, Neural Combinatorial Optimization, Preference Optimization, Reinforcement Learning

ACM Reference Format:   
Kaixi Yang, Linbo Qiao, Wei Shang, Yanqi Shi, and Dongsheng Li. 2018. APO2CS: Advantage-Aware Preference Optimization for Two-Level Constructive Search in Combinatorial Optimization. In Proceedings ofMake sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX). ACM, New York, NY, USA, 10 pages. https: //doi.org/XXXXXXX.XXXXXXX

## 1 Introduction

Combinatorial optimization problems (COPs) [19] arise extensively in practical applications. In general, their goal is to determine a solution that minimizes or maximizes an objective function over a finite but typically exponentially large solution space, subject to a set of constraints. Classic examples of COPs include the Traveling Salesman Problem (TSP)[12], the Vehicle Routing Problem (VRP)[24], the Job Shop Scheduling Problem (JSSP)[31], and the Knapsack Problem (KP)[22]. Beyond their fundamental theoretical significance, these problems are also central to numerous real-world applications, such as logistics and distribution [28], communication network design [11], chip routing [21], resource allocation [13], and production scheduling [9].

In contrast to continuous optimization problems, COPs are typically characterized by highly non-convex and non-smooth solution spaces, posing significant challenges to gradient-based optimization methods. Furthermore, many classical COPs are NP-hard, making it generally impractical to obtain global optimal solutions within an acceptable computational time. Therefore, from the perspective of real-world applications, the eficient acquisition of high-quality approximate solutions is of paramount importance.

With the advancement of deep learning [16], the paradigm for solving COPs has progressively shifted from traditional exact and heuristic algorithms toward learning-based approximate approaches. Conventional supervised learning [7] typically relies on precise labels or objective values as training signals. However, in COPs, such supervision is often expensive to obtain and susceptible to numerical noise, making it dificult for models to efectively capture the structural diferences in solution quality. In contrast, preference optimization (PO) [8][17][18] replaces absolute objective supervision with relative comparisons between solutions, making it a more suitable paradigm when high-quality explicit supervision is dificult to acquire. Despite this advantage, most existing PO methods rely on discretized binary preference labels. While this design improves training stability, it inevitably compresses the underlying optimization information and limits the model’s ability to capture the magnitude of advantage between solutions. Moreover, these methods are often highly sensitive to the quality of the candidate solution set, with their learning dynamics largely constrained by the sample distribution within the candidate pool, efectively reducing the task to posterior re-ranking over a fixed set of candidates. This limitation, in turn, restricts the broader generalizability of PO.

To address these limitations, we propose a novel PO algorithm, APO2CS, that integrates advantage-aware and two-level constructive search. Our contributions are:

• We introduce an advantage-aware preference objective that incorporates pairwise objective diferences into preferencebased training for constructive neural solvers.

• We propose a two-level supervision scheme that combines macro-level and micro-level preference learning, enabling global ranking and local trajectory refinement to be opti mized jointly.

• We show that the resulting objective can be applied in both training-from-scratch and fine-tuning settings, and we empirically study its behavior on benchmarks.

## 2 Related Work

Learning-based Combinatorial Optimization. Recent years have witnessed rapid progress in learning-based combinatorial optimization, where neural models are trained to directly construct or improve solutions. Early work, such as Pointer Networks [26], demonstrated that sequence-to-sequence architectures with attention [25] can efectively model combinatorial output spaces and solve problems such as the TSP. Building on this line of research, [4] introduced reinforcement learning (RL) [23] to directly optimize solution quality, showing that neural policies can learn heuristic solvers without handcrafted rules. Subsequent studies further improved model architectures and training strategies. For instance, [14] proposed an attention-based autoregressive solver that has become a representative framework for neural routing optimization. Hybrid approaches such as NeuroLKH [30] further demonstrated that learned modules can be efectively integrated with strong traditional heuristics, leading to substantial improvements in both solution quality and computational eficiency. However, existing learning-based methods still rely heavily on explicit labels, scalar rewards, or handcrafted solver integration, which limits their robustness and applicability when high-quality supervision is costly or dificult to design. More recently, inspired by the success of preference-based alignment methods such as Direct Preference Optimization (DPO) [20], an emerging line of research has begun to replace explicit scalar supervision with relative preference signals between candidate solutions, opening a new avenue for PO in combinatorial optimization.

Preference Optimization. Preference Optimization ofers an alternative supervision paradigm by replacing absolute objective values with relative comparisons between candidate solutions. Classical studies on ranking from pairwise preferences, such as [2] and [1], established the fundamental formulation of learning from pairwise preference labels and highlighted the efectiveness of relative supervision for ranking tasks. These studies laid the theoretical foundation for applying PO to combinatorial optimization. Building on this foundation, recent methods such as POCO [18] and BOPO [17] construct preference pairs from objective values and employ preference-based objectives to improve exploration and optimization eficiency. More recently, UCPO [8] extended this line of research to constrained combinatorial optimization by incorporating constraint satisfaction directly into the preference objective, thereby providing a plug-and-play framework for handling complex constraints without requiring architectural modifications. Nevertheless, most existing PO methods rely on binary preference labels, which capture only relative ordering while failing to reflect the magnitude of superiority between solutions.

Constructive Search. Another important line of research focuses on constructive search, where solutions are generated sequentially through a series of decisions. Early neural solvers, as well as later attention-based routing models, adopt an autoregressive construction process that enables the model to incrementally build solutions rather than merely score complete candidates after generation. In a similar spirit, POMO [15] improves constructive RL by exploiting multiple optima and encouraging diverse rollout trajectories, thereby enhancing both training stability and solution quality. Beyond purely constructive generation, NeuRewriter [6] demonstrates that search can be combined with learned local rewriting policies, enabling iterative refinement of intermediate solutions during optimization. Despite their efectiveness, existing constructive search methods typically optimize generation through rewards or heuristics alone, without explicitly incorporating fine-grained preference signals into both local trajectory construction and global solution evaluation.

Our work bridges these gaps by introducing finer-grained preference comparison and a two-level solution construction strategy within the PO framework. Specifically, APO2CS explicitly models both preference direction and advantage magnitude, while incorporating a two-level constructive search mechanism that injects preference signals into both local trajectory optimization and global solution comparison.

## 3 Motivation

Most existing preference optimization methods for combinatorial optimization rely on binary pairwise preference labels. Given a COP instance $x \in \chi$ , the objective is to solve

$$
\operatorname* { m i n } _ { s \in S ( x ) } f ( x , s ) ,\tag{1}
$$

where $S ( x )$ is the feasible solution space defined by problemspecific constraints, and $f ( x , s )$ denotes the objective value. A policy network $\pi _ { \theta } ( \cdot \mid x )$ generates a set of candidate solutions:

$$
S _ { x } = \{ s _ { i } \} _ { i = 1 } ^ { N } \sim \pi _ { \theta } ( \cdot | x ) .\tag{2}
$$

For each ordered pair $( s _ { i } , s _ { j } )$ , a binary preference label is typically defined as

$$
y _ { i j } = 1 \{ f ( x , s _ { i } ) < f ( x , s _ { j } ) \} ,\tag{3}
$$

where $y _ { i j } = 1$ indicates that $s _ { i }$ is preferred over $s _ { j } .$ . While simple and stable, such binary labels compress the original optimization information: they indicate which solution is better but ignore how much better it is. Consequently, solution pairs with very diferent quality gaps may receive the same label, weakening the supervision signal and limiting the model’s ability to capture advantage magnitude. This is particularly restrictive in combinatorial optimization, where the magnitude of improvement often reflects meaningful optimization progress. Therefore, preference supervision should move beyond binary comparisons and incorporate both preference direction and advantage magnitude. Another limitation is that existing preference signals are usually applied only after candidate solutions have been generated. The model first constructs a candidate pool and then learns preferences by comparing complete solutions, making preference optimization more like post-hoc re-ranking than search guidance. As a result, performance depends heavily on the quality and diversity of the sampled candidates, while preference information cannot directly influence the construction process. These limitations suggest that preference supervision should be incorporated at multiple levels of constructive search. Local pref erences over partial trajectories can guide intermediate decisions toward more promising construction paths, while global preferences over complete solutions ensure consistency with the final optimization objective. Since multi-level preference learning may introduce additional training complexity and computational cost, a flexible training mechanism is also needed to support both training from scratch and eficient adaptation from pretrained checkpoints.

Motivated by these observations, we propose an advantageaware preference optimization method with two-level constructive search, enabling finer-grained and more search-aware solution optimization.

## 4 Methodology

In this section, we present the Advantage-Aware Preference Optimization for Two-Level Constructive Search (APO2CS) framework, as illustrated in Figure 1. APO2CS trains a constructive policy using two complementary preference objectives. At the macro level, it compares complete solutions sampled from the current policy and encourages higher likelihood for solutions with better final objective values, thereby maintaining consistency with the global optimization objective. At the micro level, it decomposes sampled trajectories into local segments and compares partial constructions, providing denser supervision for long sequential rollouts while better respecting local construction constraints. Both objectives adopt advantage-aware pairwise supervision, where pairwise objective diferences determine the strength of the preference signal. The final training objective is formulated as a weighted combination of the macro- and micro-level preference losses.

We first introduce the proposed advantage-aware preference optimization, which extends conventional binary preference learning with fine-grained supervision. We then detail the two-level constructive search mechanism that incorporates macro-level and micro-level preferences. APO2CS can be trained either from scratch in an end-to-end manner or eficiently adapted from pretrained checkpoints via lightweight fine-tuning; the details of these two training strategies are provided in Appendix A.1.

## 4.1 Advantage-Aware Preference Optimization

Preference optimization methods [29] commonly adopt the Bradley– Terry model [5] to characterize pairwise preferences between candidate solutions. We first introduce this standard formulation as background before presenting its task-specific instantiation for constructive combinatorial optimization. Specifically, the preference probability that solution �<sub>�</sub> is preferred over solution $s _ { j }$ is defined as

$$
p _ { \theta } ( s _ { i } \succ s _ { j } \mid x ) = \sigma \big ( \alpha \big [ \log \pi _ { \theta } ( s _ { i } \mid x ) - \log \pi _ { \theta } ( s _ { j } \mid x ) \big ] \big ) ,\tag{4}
$$

where $s _ { i }$ and $s _ { j }$ are two candidate solutions sampled for instance $x ,$ $\sigma ( \cdot )$ denotes the sigmoid function, and �[log $\pi _ { \theta } ( s _ { i } \mid x ) - \log \pi _ { \theta } ( s _ { j } \mid$ �)] represents the scaled log-likelihood diference between the two solutions.

Maximizing the likelihood over all preference pairs is equivalent to minimizing the following objective:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { P O } } ( \pi _ { \theta } ) = - \mathbb { E } _ { x \sim \boldsymbol { X } } \Bigg [ \frac { 1 } { | S _ { x } | ^ { 2 } } \sum _ { i \neq j } y _ { i j } } \\ & { \qquad \cdot \log \sigma \big ( \alpha \left[ \log \pi _ { \theta } ( s _ { i } \mid x ) - \log \pi _ { \theta } ( s _ { j } \mid x ) \right] \big ) \Bigg ] . } \end{array}\tag{5}
$$

However, the standard objective only captures the direction of preference through the binary label $( y _ { i j } ) _ { ; }$ , while ignoring the magnitude of the quality diference between the two solutions. To preserve such fine-grained information, we define the set of ordered preference pairs for instance � as

$$
\begin{array} { r } { \mathcal { P } _ { x } ^ { + } = \left\{ ( i , j ) ~ | ~ i \neq j , ~ f ( x , s _ { i } ) < f ( x , s _ { j } ) \right\} . } \end{array}\tag{6}
$$

where a smaller objective value indicates a better solution. Therefore, for every $( i , j ) \in \mathcal { S } _ { x } ^ { + }$ , solution $s _ { i }$ is preferred over solution $s _ { j }$ . For each ordered preference pair, we define the corresponding pairwise quality diference as

$$
\gamma _ { i j } = r ( x , s _ { i } ) - r ( x , s _ { j } ) , \qquad ( i , j ) \in \mathcal { P } x ^ { + } ,\tag{7}
$$

where $r ( x , s )$ denotes the quality score of solution � for instance $x ,$ defined such that a larger value indicates a better solution. Under this definition, $\gamma i j > 0$ for every $( i , j ) \in \mathcal { S } _ { x } ^ { + }$

To make the pairwise diferences comparable across instances, we normalize $\gamma _ { i j }$ within each candidate-solution set:

$$
\hat { \gamma } i j = \frac { \gamma i j } { \underset { ( p , q ) \in { \mathscr P x } ^ { + } } { \operatorname* { m a x } } \gamma _ { \mathscr P q } + \varepsilon } ,\tag{8}
$$

where � is a small positive constant introduced for numerical stability. Based on the normalized pairwise diference, we construct the pair-specific coeficient

$$
\lambda _ { i j } = 1 + \hat { \gamma } _ { i j } .\tag{9}
$$

We further incorporate the normalized pairwise diference into the Bradley–Terry logit, resulting in the enhanced preference score

$$
\tilde { \mathcal { P } } _ { \theta } ( s _ { i } \gets s _ { j } \mid \boldsymbol { x } ) = \sigma \left( \alpha \left[ \log \pi _ { \theta } ( s _ { i } \mid \boldsymbol { x } ) - \log \pi _ { \theta } ( s _ { j } \mid \boldsymbol { x } ) \right] + \beta \hat { \gamma } _ { i j } \right) ,\tag{10}
$$

where $\beta$ is a scaling factor that controls the influence of the pairwise quality diference.

![](images/beb5e2637be81096bf765ef9eb5ee8eaf44580afe89bff5a0d04c689a817c657.jpg)  
Figure 1: Overview of the APO2CS framework. Given a COP instance �, the constructive policy �<sub>�</sub> generates candidate solutions and log-probabilities, from which advantage-aware macro-level preferences over complete solutions and micro-level preferences over trajectory segments are constructed. These two preference losses are integrated into the APO2CS objective to update $\pi _ { \theta }$ via back-propagation.

To incorporate the pairwise quality diference into the preference learning objective, we equivalently absorb the coeficient $\lambda _ { i j }$ into the preference score as an exponent:

$$
\bar { p } _ { \theta } ( s _ { i } \succ s _ { j } \mid x ) = \left[ \tilde { p } _ { \theta } ( s _ { i } \succ s _ { j } \mid x ) \right] ^ { \lambda _ { i j } } .\tag{11}
$$

Here, $\bar { p } _ { \theta } ( s _ { i } \succ s _ { j } \mid x )$ denotes a gap-weighted pairwise likelihood score, which adjusts the contribution ofeach preference pair accord ing to its relative quality diference. Note that it serves as a weighted likelihood term rather than a separately normalized Bradley–Terry probability.

The resulting advantage-aware preference optimization objective is formulated as

$$
\mathcal { L } _ { \mathrm { A P O } } ( \pi _ { \theta } ) = - \mathbb { E } _ { x \sim \mathcal { X } } \left[ \frac { 1 } { | \mathcal { P } x ^ { + } | } \sum _ { ( i , j ) \in \mathcal { P } x ^ { + } } \log \bar { p } _ { \theta } ( s _ { i } \succ s _ { j } \mid x ) \right] .\tag{12}
$$

## 4.2 Theoretical Interpretation

We show that the proposed advantage-aware preference objective implicitly induces a gap-modulated pairwise policy update.

Proposition 1. Let $\pi _ { \theta }$ denote a stochastic policy, and let ${ \mathcal { P } } _ { x } ^ { + }$ denote the set of ordered preference pairs. We define the advantage-aware

preference objective as

$$
\mathcal { L } _ { \mathrm { A P O } } ( \boldsymbol { \theta } ) = - \mathbb { E } _ { \boldsymbol { x } \sim \boldsymbol { X } } \left[ \frac { 1 } { | \mathcal { P } _ { \boldsymbol { x } } ^ { + } | } \sum _ { ( i , j ) \in \mathcal { P } _ { \boldsymbol { x } } ^ { + } } \log \bar { p } _ { \boldsymbol { \theta } } ( s _ { i } \succ s _ { j } \mid \boldsymbol { x } ) \right] ,\tag{13}
$$

where

$$
\bar { p } _ { \theta } ( s _ { i } \succ s _ { j } \mid x ) = \left[ \sigma \left( \alpha \left[ \log \pi _ { \theta } ( s _ { i } \mid x ) - \log \pi _ { \theta } ( s _ { j } \mid x ) \right] + \beta \hat { \gamma } _ { i j } \right) \right] _ { ( 1 4 ) } ^ { \lambda _ { i j } } ,
$$

$\hat { \gamma } _ { i j }$ is the normalized pairwise quality gap, and $\lambda _ { i j } = 1 + \hat { \gamma } _ { i j }$

Assuming that �ˆ<sub>��</sub> and $\lambda _ { i j }$ are independent of �, its gradient is

$$
\begin{array} { r l }   { \nabla _ { \theta } \mathcal { L } _ { \mathrm { A P O } } ( \theta ) = - \mathbb { E } _ { x \sim X } \Bigg [ \frac { 1 } { | \mathscr { P } _ { x } ^ { + } | } \sum _ { ( i , j ) \in \mathscr { P } _ { x } ^ { + } } w _ { i j } ( \hat { \gamma } _ { i j } ) } \\ & { \quad \quad \cdot ( \nabla _ { \theta } \log \pi _ { \theta } ( s _ { i } \mid x ) - \nabla _ { \theta } \log \pi _ { \theta } ( s _ { j } \mid x ) ) \Bigg ] , } \end{array}\tag{15}
$$

where

$$
w _ { i j } \big ( \hat { \gamma } _ { i j } \big ) = \alpha \lambda _ { i j } \left[ 1 - \sigma ( z _ { i j } ) \right] ,\tag{16}
$$

and

$$
z _ { i j } = \alpha \left[ \log \pi _ { \boldsymbol { \theta } } ( s _ { i } \mid \boldsymbol { x } ) - \log \pi _ { \boldsymbol { \theta } } ( s _ { j } \mid \boldsymbol { x } ) \right] + \beta \hat { \gamma } _ { i j } .\tag{17}
$$

Thus, the gradient weight is jointly determined by the normalized quality gap and the current pairwise preference residual. In

the small-gap regime, the change in $w _ { i j }$ relative to the binarypreference baseline is approximately linear in $\hat { \gamma } _ { i j }$

Proof sketch. Since

$$
\log \bar { p } _ { \theta } ( s _ { i } \succ s _ { j } \mid x ) = \lambda _ { i j } \log \sigma ( z _ { i j } ) ,\tag{18}
$$

and

$$
\nabla _ { \theta } \log \sigma ( z _ { i j } ) = \left[ 1 - \sigma ( z _ { i j } ) \right] \nabla _ { \theta } z _ { i j } ,\tag{19}
$$

we have

$$
\nabla _ { \theta } z _ { i j } = \alpha \Big ( \nabla _ { \theta } \log \pi _ { \theta } ( s _ { i } \mid x ) - \nabla _ { \theta } \log \pi _ { \theta } ( s _ { j } \mid x ) \Big ) .\tag{20}
$$

Substituting these expressions into the gradient of $\mathcal { L } _ { \mathrm { A P O } }$ yields the stated result. Moreover, a first-order Taylor expansion around $\hat { \gamma } _ { i j } = 0$ gives

$$
w _ { i j } \big ( \hat { \gamma } _ { i j } \big ) = w _ { i j } \big ( 0 \big ) + O \big ( \hat { \gamma } _ { i j } \big ) ,\tag{21}
$$

showing that the first-order variation of the gradient weight is linear in the normalized quality gap.

Insight. Binary preference optimization learns which solution is better, but ignores how much better. APO2CS incorporates pairwise quality gaps directly into the objective, yielding gradients modulated by both the improvement magnitude and the current preference residual. This converts preference learning from a pure ranking signal into a gap-modulated pairwise optimization principle. Consequently, APO2CS can be viewed as a pairwise policy update that incorporates solution-quality diferences without relying on the conventional advantage function used in reinforcement learning.

## 4.3 Two-Level Constructive Search

Although the advantage-aware preference objective improves the ranking consistency among complete solutions, NCO is inherently a sequential decision-making process, where the final solution quality is determined by a series of local choices. Supervising only the complete solution may therefore lead to insuficient guidance for intermediate construction steps. To address this issue, we further pro pose a Two-Level Constructive Search, which extends advantageaware preference optimization from the solution level to the partialconstruction level. The key idea is to jointly optimize: (i) global preference over complete solutions and (ii) local preference over partial trajectories, so that the policy is encouraged to produce not only better final solutions but also better intermediate decisions during the constructive process.

In NCO model, a candidate solution $s _ { i } = ( a _ { i } ^ { 1 } , a _ { i } ^ { 2 } , \ldots , a _ { i } ^ { T } )$ is generated autoregressively by the constructive policy, where � is the construction horizon and $a _ { i } ^ { t }$ denotes the action taken at step �. Its likelihood can be factorized as

$$
\pi _ { \theta } ( s _ { i } \mid x ) = \prod _ { t = 1 } ^ { T } \pi _ { \theta } ( a _ { i } ^ { t } \mid x , a _ { i } ^ { < t } ) ,\tag{22}
$$

and the corresponding log-likelihood is

$$
\log \pi _ { \theta } ( s _ { i } \mid x ) = \sum _ { t = 1 } ^ { T } \log \pi _ { \theta } ( a _ { i } ^ { t } \mid x , a _ { i } ^ { < t } ) .\tag{23}
$$

Macro-level constructive search. At the macro level, we compare complete solutions exactly as in advantage-aware preference optimization. For clarity, we denote the scaling factor in the global Bradley-Terry model by $\alpha _ { \mathrm { m a c r o } }$ . Then the macro-level preference probability is defined as

$$
\begin{array} { r } { \tilde { \mathcal { P } } _ { \theta } ^ { \mathrm { m a c r o } } ( s _ { i } \times s _ { j } \mid x ) = \sigma ( \alpha _ { \mathrm { m a c r o } } } \\ { \cdot \left[ \log \pi _ { \theta } ( s _ { i } \mid x ) - \log \pi _ { \theta } ( s _ { j } \mid x ) \right] + \beta \hat { \gamma } _ { i j } ) . } \end{array}\tag{24}
$$

Accordingly, the macro-level loss is written as

$$
\mathcal { L } _ { \mathrm { m a c r o } } ( \pi _ { \theta } ) = - \mathbb { E } _ { x \sim \mathcal { X } } \left[ \frac { 1 } { | \mathcal { P } x ^ { + } | } \sum _ { ( i , j ) \in \mathcal { P } x ^ { + } } \log \bar { p } _ { \theta } ^ { \mathrm { m a c r o } } ( s _ { i } \succ s _ { j } | x ) \right] .\tag{25}
$$

This term provides global supervision by encouraging the policy to assign higher probability to better complete solutions.

Micro-level constructive search. To provide finer-grained supervision during the construction process, we further divide each solution trajectory into a sequence of local segments. Let � denote a predefined segment length, and let $\tau _ { i } ^ { ( u ) }$ be the �-th segment extracted from solution $s _ { i } ,$ where $u = 1 , \ldots , U$ and � is the number of extracted segments. The likelihood of a segment is defined as the product of step-wise action probabilities within the corresponding interval:

$$
\pi _ { \theta } ( \tau _ { i } ^ { ( u ) } \mid x ) = \prod _ { t \in \bar { J } _ { u } } \pi _ { \theta } ( a _ { i } ^ { t } \mid x , a _ { i } ^ { < t } ) ,\tag{26}
$$

where $\mathcal { T } _ { u }$ denotes the set of time indices belonging to the �-th segment. Its log-likelihood is thus

$$
\log \pi _ { \theta } ( \tau _ { i } ^ { ( u ) } \mid x ) = \sum _ { t \in { \cal I } _ { u } } \log \pi _ { \theta } ( a _ { i } ^ { t } \mid x , a _ { i } ^ { < t } ) .\tag{27}
$$

For each segment, we compute a local segment cost �(�, $\tau _ { i } ^ { ( u ) } )$ ), which evaluates the partial construction quality within that segment. Since lower cost indicates a better partial trajectory, the local preference label is defined as

$$
y _ { i j } ^ { ( u ) } = 1 \left\{ c ( x , \tau _ { i } ^ { ( u ) } ) < c ( x , \tau _ { j } ^ { ( u ) } ) \right\} .\tag{28}
$$

Correspondingly, the local quality gap is defined as

$$
\begin{array} { r } { \gamma _ { i j } ^ { ( u ) } = r ^ { ( u ) } ( x , \tau _ { i } ^ { ( u ) } ) - r ^ { ( u ) } ( x , \tau _ { j } ^ { ( u ) } ) , } \end{array}\tag{29}
$$

where $r ^ { ( u ) } ( x , \tau ) = - c ( x , \tau )$ denotes the segment-quality score. Similar to the global case, we introduce a normalized local weight

$$
\hat { \gamma } _ { i j } ^ { ( u ) } = 1 + \frac { \gamma _ { i j } ^ { ( u ) } } { \displaystyle \operatorname* { m a x } _ { ( p , q ) \in \mathcal { P } _ { x , u } ^ { + } } \gamma _ { p q } ^ { ( u ) } + \varepsilon } ,\tag{30}
$$

where $\mathcal { P } _ { x , u } ^ { + } = \Big \{ ( i , j ) | i \neq j , r ^ { ( u ) } ( x , \tau _ { i } ^ { ( u ) } ) > r ^ { ( u ) } ( x , \tau _ { j } ^ { ( u ) } ) \Big \}$ denotes the ordered local preference-pair set. Based on these local quantities, we define the micro-level preference probability by an advantageenhanced Bradley-Terry model:

$$
\begin{array} { r l } & { \widetilde {  { \boldsymbol { p } } } _ { \theta } ^ { \mathrm { m i c r o } } \big ( \tau _ { i } ^ { ( u ) } \succ \tau _ { j } ^ { ( u ) } \mid x \big ) } \\ & { \qquad = \sigma \Big ( \alpha _ { \mathrm { m i c r o } } \big [ \log \pi _ { \theta } ( \tau _ { i } ^ { ( u ) } \mid x ) - \log \pi _ { \theta } ( \tau _ { j } ^ { ( u ) } \mid x ) \big ] + \beta \widehat { \gamma } _ { i j } ^ { ( u ) } \Big ) . } \end{array}\tag{31}
$$

where $\alpha _ { \mathrm { m i c r o } }$ controls the sensitivity to segment-level likelihood diferences. The micro-level loss is then defined as

$$
\begin{array} { c } { { \displaystyle { \mathcal { L } } _ { \mathrm { m i c r o } } ( \pi _ { \theta } ) = - { \mathbb { E } } _ { x \sim \boldsymbol { X } } \left[ \frac { 1 } { U } \sum _ { u = 1 } ^ { U } \frac { 1 } { | { \mathcal { P } } _ { x , u } ^ { + } | } \sum _ { ( i , j ) \in { \mathcal { P } } _ { x , u } ^ { + } } \right. } } \\ { { \left. \log \bar { p } _ { \theta } ^ { \mathrm { m i c r o } } \left( \tau _ { i } ^ { ( u ) } \succ \tau _ { j } ^ { ( u ) } \mid x \right) \right] . } } \end{array}\tag{32}
$$

This objective explicitly regularizes the local construction behavior by encouraging the policy to assign higher probability to better partial trajectories.

Joint optimization. Finally, the two levels are integrated into a unified training objective:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { A P O 2 C S } } ( \pi _ { \theta } ) = ( 1 - \eta ) \mathcal { L } _ { \mathrm { m a c r o } } ( \pi _ { \theta } ) + \eta \mathcal { L } _ { \mathrm { m i c r o } } ( \pi _ { \theta } ) , } \end{array}\tag{33}
$$

where $\eta \in \left[ 0 , 1 \right]$ controls the trade-of between global solution ranking and local trajectory refinement. When $\eta = 0 ;$ , the objective reduces to purely global preference optimization; when $\eta > 0 ,$ , the policy is additionally guided by fine-grained local preferences.

The two objectives provide complementary guidance. The macrolevel objective preserves the global search direction by comparing complete solutions according to their final quality, whereas the micro-level objective alleviates long-horizon credit assignment by introducing preference signals into intermediate construction steps. Their combination allows the policy to learn both which complete solutions are preferable and which local decisions contribute to constructing them.

## 5 Experiments

In this section, we present the main experimental results to evaluate the efectiveness of the proposed APO2CS algorithm for COPs. Specifically, we aim to answer the following research questions:

• RQ1. How does APO2CS perform in comparison with existing algorithms on standard benchmarks, such as the Traveling Salesman Problem (TSP), the Capacitated Vehicle Routing Problem (CVRP) and the knapsack problem (KP)?

• RQ2. How efective is the proposed two-level preference optimization mechanism, compared with conventional preference optimization methods, for constructive search?

• RQ3. How flexible and eficient is the proposed training paradigm under diferent initialization settings, including cold-start and warm-start training?

RQ1 is addressed through the main results presented in Section 5.2, where APO2CS is compared with representative baselines. RQ2 and RQ3 are investigated in Sections 5.3 and 5.4, respectively, through comprehensive ablation studies and startup strategy analyses.

## 5.1 Experimental Setup

Benchmark. APO2CS is a reinforcement learning based neural solver designed for COPs. To evaluate its efectiveness, we consider three fundamental COP benchmarks as our testbeds: the TSP, the CVRP, and the KP. In the main experiments, for TSP and CVRP, we follow Kool et al. [14] and generate problem instances by uniformly sampling node coordinates. For KP, we follow the setup by Bello et al. [4]. For each routing problem, we evaluate three standard scales with 20, 50, and 100 nodes. For KP, we evaluate three problem scales with 50, 100, and 200 items. In TSP, the objective is to identify a minimum-length Hamiltonian tour over all nodes in the graph. In CVRP, the objective is to design a set of routes that start from and return to the depot, while satisfying vehicle capacity constraints and customer demand requirements, so as to minimize the total travel distance. For both routing problems, the reward is defined as the Euclidean length of the resulting route. In KP, each instance consists ofa set ofitems with associated weights and values, and the objective is to select a subset of items that maximizes the total value while satisfying the knapsack capacity constraint. Accordingly, the reward for KP is defined as the total value of the selected items. All experiments are performed on a server with NVIDIA RTX A6000 GPUs (48 GB) and an Intel(R) Xeon(R) Gold 6230 CPU @ 2.10 GHz.

Baselines. We assess the optimality gap using strong problemspecific solvers. For TSP and CVRP, we use Concorde [3] and LKH3 [10], respectively, as widely adopted heuristic solvers. For KP, we compute the optimal solution using dynamic programming and report the optimality gap with respect to the DP results. In addition, we include a greedy heuristic baseline for KP, which selects items according to a hand-crafted value-to-weight criterion. We further compare APO2CS against reinforcement-learning-based neural solvers trained with variants of REINFORCE, including Pointer Net [4], AM [14], and POMO [15]. AM estimates the advantage function based on the solution generated at the previous step, while POMO refines this estimate by averaging the quality ofmultiple sampled solutions from the same instance and using them as baselines. For KP, we also compare with MTL-COP [27], a multi-task learning based neural solver designed for combinatorial optimization problems. We additionally include comparisons with preference-optimizationbased neural solvers, namely BOPO [17] and POCO [18], both of which are built upon the POMO framework. BOPO enhances sample eficiency through three specifically designed components, whereas POCO improves the trade-of between exploitation and exploration by introducing a reparameterized entropy-regularized objective. Unless otherwise stated, all hyperparameter settings, network architectures, and optimizer configurations are kept consistent with those in the original papers.

Evaluation. Inference follows a unified protocol: 1) Each validation set consists of exactly 10,000 instances. 2) We apply 8-fold augmentation (denoted as x8 aug.) to each instance. 3) The metrics include the average objective value of the solutions on the validation set, the average optimality gap, and the total time to solve these instances.

## 5.2 Main Results (RQ1)

The performance of APO2CS on TSP, CVRP, and KP is reported in Table 1 and Table 2. Overall, APO2CS achieves the best or near-best results among neural baselines across problem scales, demonstrating strong optimization capability in constructive search. Results on KP further confirm its flexibility in handling diferent types of combinatorial optimization problems.

Without augmentation, APO2CS attains the lowest optimality gaps among neural solvers on all TSP and CVRP benchmark settings. Its advantage is particularly clear on larger instances, such as

TSP-100 and CVRP-100, where it outperforms both RL-based baselines and existing preference-optimization-based methods. With x8 augmentation, APO2CS further improves performance on most routing settings and remains the strongest neural method overall. On TSP, the augmented version achieves an almost zero gap and stays very close to LKH3. On CVRP, augmentation brings additional gains on small and medium instances, while yielding only marginal improvements on larger ones. Moreover, APO2CS ofers an attractive trade-of between solution quality and eficiency. Although strong heuristic solvers such as Concorde and LKH3 remain important references, they require substantially higher runtime, whereas APO2CS achieves near-heuristic solution quality with much lower inference cost.

To further evaluate its flexibility beyond routing problems, we conduct additional experiments on KP. We adapt APO2CS by representing each item with its weight and value, replacing the �- and �-coordinates used for TSP nodes. During decoding, the generated item sequence is inserted into the knapsack sequentially until the capacity is reached, after which decoding terminates. As shown in Table 2, APO2CS achieves the best or tied-best performance among neural baselines on KP50, KP100, and KP200. Compared with greedy heuristics, Pointer Net, AM, POMO, and MTL-COP, it consistently obtains smaller optimality gaps relative to the optimal DP solutions, reaching only 0.02%, 0.01%, and 0.01% on KP50, KP100, and KP200, respectively. These results indicate that APO2CS can efectively adapt to non-routing COPs, where the objective is subset selection rather than route construction.

In summary, Table 1 and Table 2 show that APO2CS consistently outperforms existing neural baselines on TSP and CVRP and generalizes well to KP. These results highlight its efectiveness and flexibility across diferent problem sizes and structures, while maintaining competitive performance against strong problem-specific solvers.

## 5.3 Ablation Studies (RQ2)

We conduct systematic ablation studies to evaluate the contributions of advantage-aware preference optimization and the two-level constructive search mechanism. Specifically, we compare four variants: POMO\*, which augments POMO with conventional binarylabel preference optimization; APO2CS\_macro, which retains only the macro-level preference loss; APO2CS\_micro, which retains only the micro-level preference loss; and the full APO2CS model.

Table 3 reports the ablation results on TSP with problem sizes N=50 and N=100. The comparison ofAPO2CS\_macro, APO2CS\_micro, and APO2CS clearly shows that all components of APO2CS are necessary. Using only the micro-level preference loss leads to severe performance degradation, yielding gaps of 23.57% and 29.50%, respectively. This indicates that local trajectory-level supervision alone is insuficient to ensure high global solution quality, as microlevel preference primarily captures short-horizon construction quality and may bias the policy toward locally favorable yet globally suboptimal decisions. In contrast, using only the macro-level preference loss already achieves much stronger performance, with gaps of 0.33% and 0.36%, confirming the importance of global solution-level preference learning. Nevertheless, the full APO2CS further reduces the gap to 0.14% and 0.10%, respectively, demonstrating that microlevel preference provides useful complementary guidance when combined with macro-level supervision. Specifically, the macrolevel aligns optimization with final solution quality, whereas the micro-level supplies denser local supervision during constructive search.

Moreover, the comparison between POMO\* and APO2CS\_macro highlights the advantage of the proposed advantage-aware preference optimization over conventional binary preference optimization. Although both variants rely solely on global solution-level preference supervision, APO2CS\_macro consistently outperforms POMO\*, reducing the gap from 0.51% to 0.33% on N=50 and from 0.59% to 0.36% on N=100. This improvement suggests that incorporating advantage magnitude into the preference signal provides more informative supervision than binary labels alone. Rather than learning only which solution is better, the model can additionally exploit how much better one solution is than another, leading to stronger preference discrimination and more efective optimization.

## 5.4 Analysis of Startup Strategy (RQ3)

We evaluate the efect ofdiferent initialization strategies in APO2CS by comparing cold-start training from random initialization with warm-start training from a pretrained POMO-500 checkpoint. The results shown in Figure 2 demonstrate that APO2CS already outperforms POMO under the cold-start setting, indicating the efectiveness of the proposed method itself. However, cold-start training begins from a relatively poor policy and therefore requires more optimization steps to reach stable performance. By contrast, APO2CS with warm-start consistently achieves lower objective values across TSP-N20, TSP-N50, and TSP-N100. Since warm-start uses the pretrained POMO-500 model as initialization, it avoids the ineficient exploration stage in early training and starts from a much stronger policy. More importantly, when compared with continued POMO training from the same POMO-500 checkpoint, APO2CS still obtains better results, showing that the gain is not only due to improved initialization, but also to the efectiveness of the proposed APO2CS optimization mechanism. In addition, warm-start reaches better performance with fewer additional training steps, demonstrating higher training eficiency and better utilization of the available training budget. Overall, these results verify the standalone efectiveness of APO2CS under cold-start training and further confirm that warm-start serves as a complementary strategy for accelerating convergence and improving final solution quality.

## 6 Conclusion

In this paper, we proposed APO2CS, an advantage-aware preference optimization framework for two-level constructive search in combinatorial optimization. Unlike conventional methods based on binary preference labels, APO2CS captures both preference di rection and advantage magnitude, allowing the model to better distinguish solution-quality diferences. It further combines macrolevel preferences over complete solutions with micro-level preferences over partial construction trajectories, enabling preference signals to guide both global optimization and local decision making during generation. APO2CS also supports both cold-start training and warm-start fine-tuning, making it flexible under diferent computational budgets. Experiments on TSP and CVRP show that APO2CS consistently achieves strong performance across problem scales, outperforming existing reinforcement-learning-based and preference-optimization-based neural solvers while maintaining competitive inference eficiency. In addition, we further evaluate APO2CS on KP to examine its flexibility beyond routing problems. The results on KP demonstrate that APO2CS achieves highly competitive performance against both heuristic and neural baselines, indicating its potential applicability to diverse combinatorial opti mization structures beyond routing.

Table 1: Experiment results on TSP and CVRP. Gap is evaluated on 10k instances, and Time is the sum of them.
<table><tr><td rowspan="3">Method</td><td colspan="8">TSP</td><td colspan="8"></td></tr><tr><td colspan="2">N=20</td><td colspan="2"></td><td colspan="2">N=50</td><td colspan="2">N=100</td><td colspan="2">N=20</td><td colspan="2"></td><td colspan="2">N=50</td><td colspan="2">N=100</td></tr><tr><td>Obj.</td><td>Gap%</td><td>Time</td><td>Obj.</td><td>Gap%</td><td>Time</td><td>Obj.</td><td>Gap%</td><td>Time</td><td>Obj. Gap%</td><td>Time</td><td>Obj.</td><td>Gap%</td><td>Time</td><td>Obj.</td><td>Gap%</td><td>Time</td></tr><tr><td>Concorde</td><td>3.833</td><td></td><td>5m</td><td>5.690</td><td></td><td>13m</td><td>7.761</td><td>1h</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LKH3</td><td>3.832</td><td>0.00</td><td>42s</td><td>5.693</td><td>0.00</td><td>6m</td><td>7.762</td><td>0.00</td><td>25m</td><td>6.120 0.00</td><td>2h</td><td>10.409</td><td>0.00</td><td>7h</td><td>15.687</td><td>0.00</td><td>12h</td></tr><tr><td>AM</td><td>3.846</td><td>0.37</td><td>1s</td><td>5.779</td><td>1.51</td><td>1s</td><td>8.039</td><td>3.57</td><td>2s</td><td>6.401 4.59</td><td>1s</td><td>10.934</td><td>5.04</td><td>1s</td><td>16.781</td><td>6.97</td><td>3s</td></tr><tr><td>POMO</td><td>3.835</td><td>0.08</td><td>1s</td><td>5.700</td><td>0.12</td><td>2s</td><td>7.795</td><td>0.43</td><td>11s</td><td>6.175 0.90</td><td>1s</td><td>10.496</td><td>0.84</td><td>4s</td><td>15.832</td><td>0.92</td><td>19s</td></tr><tr><td>BOPO</td><td>3.839</td><td>0.08</td><td>3s</td><td>5.702</td><td>0.12</td><td>6s</td><td>7.787</td><td>0.32</td><td>12s</td><td>6.181 1.00</td><td>1s</td><td>10.502</td><td>0.89</td><td>5s</td><td>15.843</td><td>0.99</td><td>20s</td></tr><tr><td>POCO</td><td>3.841</td><td>0.23</td><td>3s</td><td>5.701</td><td>0.14</td><td>6s</td><td>7.775</td><td>0.17</td><td>18s 6.175</td><td>0.90</td><td>1s</td><td>10.496</td><td>0.84</td><td>6s</td><td>15.794</td><td>0.68</td><td>21s</td></tr><tr><td>APO2CS</td><td>3.832</td><td>0.00</td><td>2s</td><td>5.699</td><td>0.11</td><td>5s</td><td>7.770</td><td>0.10</td><td>13s 6.159</td><td>0.64</td><td>1s</td><td>10.425</td><td>0.15</td><td>4s</td><td>15.725</td><td>0.24</td><td>18s</td></tr><tr><td>POMO (x8 aug.)</td><td>3.834</td><td>0.05</td><td>3s</td><td>5.695</td><td>0.04</td><td>16s</td><td>7.772</td><td>0.13 1m</td><td>6.142</td><td>0.36</td><td>5s</td><td>10.426</td><td>0.16</td><td>26s</td><td>15.731</td><td>0.28</td><td>2m</td></tr><tr><td>BOPO (x8 aug.)</td><td>3.835</td><td>0.08</td><td>3s</td><td>5.697</td><td>0.07</td><td>6s</td><td>7.779</td><td>0.22</td><td>18s 6.143</td><td>0.38</td><td>4s</td><td>10.429</td><td>0.19</td><td>19s</td><td>15.737</td><td>0.32</td><td>29s</td></tr><tr><td>POCO (x8 aug.)</td><td>3.839</td><td>0.18</td><td>6s</td><td>5.694</td><td>0.02</td><td>17s</td><td>7.774</td><td>0.15</td><td>1m 6.139</td><td>0.31</td><td>5s</td><td>10.418</td><td>0.09</td><td>21s</td><td>15.731</td><td>0.28</td><td>2m</td></tr><tr><td>APO2CS (x8 aug.)</td><td>3.832</td><td>0.00</td><td>6s</td><td>5.693</td><td>0.00</td><td>17s</td><td>7.763</td><td>0.01</td><td>1m</td><td>6.132 0.20</td><td>5s</td><td>10.417</td><td>0.08</td><td>20s</td><td>15.726</td><td>0.25</td><td>2m</td></tr></table>

Table 2: Experiment results on KP. Gap is evaluated on 10k instances.
<table><tr><td rowspan="2">Method</td><td colspan="2">KP50</td><td colspan="2">KP100</td><td colspan="2">KP200</td></tr><tr><td>Score</td><td>Gap%</td><td>Score</td><td>Gap%</td><td>Score</td><td>Gap%</td></tr><tr><td>Optimal / DP</td><td>20.128</td><td>0.00</td><td>40.458</td><td>0.00</td><td>57.602</td><td>0.00</td></tr><tr><td>Greedy Heuristics</td><td>19.907</td><td>1.10</td><td>40.205</td><td>0.63</td><td>57.247</td><td>0.62</td></tr><tr><td>Pointer Net</td><td>19.914</td><td>1.06</td><td>40.218</td><td>0.59</td><td>57.262</td><td>0.59</td></tr><tr><td>AM</td><td>19.961</td><td>0.83</td><td>40.250</td><td>0.51</td><td>57.278</td><td>0.56</td></tr><tr><td>POMO</td><td>20.120</td><td>0.04</td><td>40.452</td><td>0.01</td><td>57.594</td><td>0.01</td></tr><tr><td>MTL-COP</td><td>20.122</td><td>0.03</td><td>40.452</td><td>0.01</td><td>57.599</td><td>0.01</td></tr><tr><td>APO2CS</td><td>20.123</td><td>0.02</td><td>40.453</td><td>0.01</td><td>57.599</td><td>0.01</td></tr></table>

![](images/96ac5d9e143f1fdfe46b56df4cb192ec4bd61072cafbc370497af098d2ea3908.jpg)  
(a) N=20

![](images/0d3395dae429c3f1f3bd297945194ad7e6cce5c017cdea1f11d2e5b9322155b6.jpg)  
(b) N=50

![](images/e66b1ad6774bd73b9f58c7ecca8440adacdc7b5db3e3ccfb3f2318ea4db9bfee.jpg)  
(c) N=100  
Figure 2: Performance comparison of POMO and APO2CS on TSP instances with diferent sizes. "Random" denotes training from scratch, while "POMO-500" denotes warm-start initialization from a pretrained POMO model after 500 epochs.

Despite these encouraging results, APO2CS still has a limitation: its micro-level preference pairs rely on predefined trajectory segmentation, which may not always match the most informative decision boundaries across diferent problem domains. Future work will explore adaptive mechanisms for constructing local preference signals and further investigate the theoretical properties of advantage-aware preference optimization in constructive search. We also plan to extend APO2CS to broader constrained and heterogeneous combinatorial optimization problems, particularly those arising in parallel and distributed training.

Table 3: Ablation experimental results on TSP. Gap is evaluated on 10k instances, and Time is the sum of them.
<table><tr><td rowspan="2">Method</td><td colspan="3">N=50</td><td colspan="3">N=100</td></tr><tr><td>Obj.</td><td>Gap%</td><td>Time</td><td>Obj.</td><td>Gap%</td><td>Time</td></tr><tr><td>POMO*</td><td>5.722</td><td>0.51</td><td>6s</td><td>7.808</td><td>0.59</td><td>18s</td></tr><tr><td>APO2CS_macro</td><td>5.712</td><td>0.33</td><td>5s</td><td>7.79</td><td>0.36</td><td>12s</td></tr><tr><td>APO2CS_micro</td><td>7.035</td><td>23.57</td><td>6s</td><td>10.052</td><td>29.50</td><td>14s</td></tr><tr><td>APO2CS</td><td>5.701</td><td>0.14</td><td>5s</td><td>7.77</td><td>0.10</td><td>13s</td></tr></table>

## Ethics and Privacy Statement

This work focuses on developing learning-based optimization algorithms for combinatorial optimization problems. It does not involve human subjects, personal data collection, or sensitive information. The proposed method is intended for algorithmic decision-making and does not introduce additional privacy risks beyond those associated with existing optimization approaches.

## References

[1] Nir Ailon. 2012. An Active Learning Algorithm for Ranking from Pairwise Preferences with an Almost Optimal Query Complexity. Journal of Machine Learning Research 13, 5 (2012), 137–164.

[2] Nir Ailon and Mehryar Mohri. 2010. Preference-based learning to rank. Machine Learning 80, 2 (2010), 189–211.

[3] David L Applegate, Robert E Bixby, Vašek Chvátal, and William J Cook. 2011. The traveling salesman problem: a computational study. Princeton university press, Princeton, NJ.

[4] Irwan Bello, Hieu Pham, Quoc V. Le, Mohammad Norouzi, and Samy Ben gio. 2017. Neural Combinatorial Optimization with Reinforcement Learning. arXiv:1611.09940 [cs.AI] https://arxiv.org/abs/1611.09940

[5] Ralph Allan Bradley and Milton E Terry. 1952. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika 39, 3/4 (1952), 324–345.

[6] Xinyun Chen and Yuandong Tian. 2019. Learning to perform local rewriting for combinatorial optimization. Advances in neural information processing systems 32 (2019), 6278–6289.

[7] Pádraig Cunningham, Matthieu Cord, and Sarah Jane Delany. 2008. Supervised learning. In Machine learning techniques for multimedia: case studies on organization and retrieval. Springer, Berlin, Heidelberg, 21–49.

[8] Zhanhong Fang, Debing Wang, Jinbiao Chen, Jiahai Wang, and Zizhen Zhang. 2025. UCPO: A Universal Constrained Combinatorial Optimization Method via Preference Optimization.

[9] Stephen C Graves. 1981. A review of production scheduling. Operations research 29, 4 (1981), 646–675.

[10] Keld Helsgaun. 2017. An extension of the Lin-Kernighan-Helsgaun TSP solver for constrained traveling salesman and vehicle routing problems. Roskilde: Roskilde University 12 (2017), 966–980.

[11] Halid Hrasnica, Abdelfatteh Haidine, and RalfLehnert. 2005. Broadband powerline communications: network design. John Wiley & Sons, Chichester, UK

[12] Michael Jünger, Gerhard Reinelt, and Giovanni Rinaldi. 1995. The traveling salesman problem. Handbooks in operations research and management science 7 (1995), 225–330.

[13] Naoki Katoh and Toshihide Ibaraki. 1998. Resource Allocation Problems. In Handbook of Combinatorial Optimization. Springer, Boston, MA, 905–1006.

[14] Wouter Kool, Herke Van Hoof, and Max Welling. 2019. Attention, Learn to Solve Routing Problems! arXiv:1803.08475 [cs.LG]

[15] Yeong-Dae Kwon, Jinho Choo, Byoungjip Kim, Iljoo Yoon, Youngjune Gwon, and Seungjai Min. 2020. Pomo: Policy optimization with multiple optima for reinforcement learning. Advances in neural information processing systems 33 (2020), 21188–21198.

[16] Yann LeCun, Yoshua Bengio, and Geofrey Hinton. 2015. Deep learning. nature 521, 7553 (2015), 436–444.

[17] Zijun Liao, Jinbiao Chen, Debing Wang, Zizhen Zhang, and Jiahai Wang. 2025. BOPO: Neural Combinatorial Optimization via Best-Anchored and Objective Guided Preference Optimization. arXiv:2503.07580 [cs.LG]

[18] Mingjun Pan, Guanquan Lin, You-Wei Luo, Bin Zhu, Zhien Dai, Lijun Sun, and Chun Yuan. 2025. Preference Optimization for Combinatorial Optimization Problems. arXiv:2505.08735 [cs.LG]

[19] Christos H. Papadimitriou and Kenneth Steiglitz. 1998. Combinatorial Optimization: Algorithms and Complexity. Courier Corporation, Mineola, NY.

[20] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems 36 (2023), 53728–53741.

[21] Ville Rantala, Teijo Lehtonen, Juha Plosila, et al. 2006. Network on Chip Routing Algorithms. Turku Centre for Computer Science, Turku, Finland.

[22] Harvey M Salkin and Cornelis A De Kluyver. 1975. The knapsack problem: a survey. Naval Research Logistics Quarterly 22, 1 (1975), 127–144.

[23] Ashish Kumar Shakya, Gopinatha Pillai, and Sohom Chakrabarty. 2023. Reinforcement learning algorithms: A brief survey. Expert Systems with Applications 231 (2023), 120495.

[24] Paolo Toth and Daniele Vigo. 2002. The Vehicle Routing Problem. SIAM, Philadelphia, PA.

[25] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings ofthe 31st International Conference on Neural Information Processing Systems (Long Beach, California, USA) (NIPS’17). Curran Associates Inc., Red Hook, NY, USA, 6000–6010.

[26] Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. 2015. Pointer networks. In Proceedings ofthe 29th International Conference on Neural Information Processing Systems - Volume 2 (Montreal, Canada) (NIPS’15). MIT Press, Cambridge, MA, USA, 2692–2700.

[27] Chenguang Wang, Zhang-Hua Fu, Pinyan Lu, and Tianshu Yu. 2025. Eficient Training of Multi-task Neural Solver for Combinatorial Optimization. arXiv:2305.06361 [cs.LG] https://arxiv.org/abs/2305.0636

[28] C. Donald J. Waters. 2003. Global Logistics and Distribution Planning: Strategies for Management. Kogan Page Publishers, London, UK.

[29] Christian Wirth, Riad Akrour, Gerhard Neumann, and Johannes Fürnkranz. 2017. A survey ofpreference-based reinforcement learning methods. Journal ofMachine Learning Research 18, 136 (2017), 1–46.

[30] Liang Xin, Wen Song, Zhiguang Cao, and Jie Zhang. 2021. Neurolkh: Combining deep learning model with lin-kernighan-helsgaun heuristic for solving the traveling salesman problem. Advances in Neural Information Processing Systems 34 (2021), 7472–7483.

[31] Hegen Xiong, Shuangyuan Shi, Danni Ren, and Jinjin Hu. 2022. A survey of job shop scheduling problem: The types and models. Computers & Operations Research 142 (2022), 105731.

Table 4: General training hyperparameters for APO2CS.
<table><tr><td>Category</td><td>Parameter</td><td>Value / Setting</td></tr><tr><td>Problem setting</td><td>Problem sizes</td><td>20, 50, 100 for TSP and CVRP 50, 100, 200 for KP</td></tr><tr><td rowspan="6">Model architecture</td><td>Embedding dimension</td><td>128</td></tr><tr><td>Encoder layers</td><td>6</td></tr><tr><td>Decoder layers</td><td>1</td></tr><tr><td>QKV dimension</td><td>16</td></tr><tr><td>Attention heads</td><td>8</td></tr><tr><td>Feed-forward hidden dimension</td><td>512</td></tr><tr><td rowspan="6"></td><td>Multi-start hidden dimension</td><td>16</td></tr><tr><td>Optimizer</td><td> $\mathrm { A d a m }$ </td></tr><tr><td>Learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Epochs Training episodes</td><td>2000</td></tr><tr><td>Batch size</td><td>50000 64</td></tr><tr><td rowspan="4"></td><td>Optimizer step interval</td><td>1</td></tr><tr><td>Data augmentation</td><td>Disabled</td></tr><tr><td></td><td></td></tr><tr><td>Loss type</td><td>dual_po_loss</td></tr><tr><td rowspan="2">Preference optimization</td><td>η</td><td>0.5</td></tr><tr><td>Segment length</td><td>5</td></tr></table>

Table 5: Assets, Licenses and Their Usage
<table><tr><td>Type</td><td>Asset</td><td>License</td><td>Usage</td></tr><tr><td rowspan="9">Code</td><td>Concorde(David et al. 2011)</td><td>Available for academic use</td><td>Evaluation</td></tr><tr><td>Pointer Net(Bello et al. 2016)</td><td>MIT License</td><td>Evaluation</td></tr><tr><td>LKH3(Helsgaun 2017)</td><td>Available for academic use</td><td>Evaluation</td></tr><tr><td>AM(Kool et al. 2019)</td><td>MIT License</td><td>Evaluation</td></tr><tr><td>POMO(Kwon et al. 2020)</td><td>MIT License</td><td>Evaluation</td></tr><tr><td>BOPO(Liao et al. 2025)</td><td>MIT License</td><td>Evaluation</td></tr><tr><td>POCO(Pan et al. 2025)</td><td>MIT License</td><td>Evaluation</td></tr><tr><td>MTL-COP(Wang et al. 2025)</td><td>MIT License</td><td>Evaluation</td></tr><tr><td></td><td></td><td></td></tr></table>

## Algorithm 1 APO2CS Training Protocol

Require: Epochs �, dataset D, policy �<sub>�</sub>, mode � ∈ {cold, warm},   
optional checkpoint $\theta _ { 0 } ,$ segment level �, candidate number �,   
learning rate �.   
1: if $m = \mathrm { w }$ arm then   
2: Initialize $\theta  \theta _ { 0 }$   
3: else   
4: Initialize $\theta$ randomly   
5: end if   
6: for epoch � = 1 to � do   
7: for each mini-batch $\mathcal { B } \subset \mathcal { D }$ do   
8: for each � ∈ B do   
9: Sample candidates $S _ { x } = \{ \tau _ { i } \} _ { i = 1 } ^ { N } \sim \pi _ { \theta } ( \cdot \mid x )$   
10: Extract level-� segments $S _ { x } ^ { ( u ) }$   
11: Construct macro-level pairs $\mathcal { P } _ { x }$ from $S _ { x }$   
12: Construct micro-level pairs $\mathcal { P } _ { x } ^ { ( u ) }$ from $S _ { x } ^ { ( u ) }$   
13: end for   
14: Compute $\scriptstyle { \mathcal { L } } _ { \mathrm { m a c r o } }$ and $\scriptstyle { \mathcal { L } } _ { \mathrm { m i c r o } }$   
15: L<sub>APO2CS</sub> ← (1 − <sub>�</sub>)L<sub>macro</sub> + <sub>�</sub>L<sub>micro</sub>   
16: Update $\theta \gets \theta - \alpha \nabla _ { \theta } \mathcal { L }$ APO2CS   
17: end for   
18: end for

## A Appendix

## A.1 End-to-End Training and Lightweight Fine-Tuning

To make APO2CS more practical under diferent computational budgets, we introduce a flexible training strategy that supports both cold-start and warm-start initialization. APO2CS can be trained either from scratch or from a pretrained checkpoint, reducing training cost and computational overhead while maintaining the benefits of advantage-aware two-level preference optimization.

In the cold-start setting, the policy network is randomly initialized and trained end-to-end on the target combinatorial optimization task. For each mini-batch, the model samples candidate solutions, builds macro- and micro-level preference pairs, computes the APO2CS loss, and updates the parameters via backpropagation. This setting requires no external initialization and enables the model to learn both solution construction and preference-guided optimization directly from data. When a pretrained constructive policy is available, APO2CS can be initialized from the checkpoint and further fine-tuned with the same preference-based objective. The pretrained model ofers a strong starting point for solution generation, while macro- and micro-level preference supervision further improves its search behavior. Compared with training from scratch, warm-start fine-tuning is more eficient and requires fewer optimization steps. The overall training process is summarized in Algorithm 1.

## A.2 Train Details

Table 4 summarizes the general training hyperparameters used for APO2CS. We evaluate APO2CS on three problem families, including TSP, CVRP, and KP. For TSP and CVRP, the problem sizes are set to 20, 50, and 100, while for KP, the problem sizes are set to 50, 100, and 200.

For the model architecture, we use a unified configuration across tasks. The model adopts a 128-dimensional embedding space, 6 encoder layers, 1 decoder layer, 8 attention heads, 16-dimensional QKV projections, and a 512-dimensional feed-forward hidden layer. The multi-start hidden dimension is set to 16.

For optimization, we use the Adam optimizer with a learning rate of $3 \times 1 0 ^ { - 4 }$ and a weight decay of $1 \times 1 0 ^ { - 6 }$ . The model is trained for 2000 epochs with 50000 training episodes and a batch size of 64. The optimizer is updated every step, and data augmentation is disabled during training.

For preference optimization, we use dual\_po\_loss as the training objective. The balancing coeficient � is set to 0.5, and the segment length for local preference modeling is fixed to 5 across all experiments.

## A.3 Licenses for existing assets

The assets used in this work are listed in Table 5 and are all opensource for academic research. We will release our source code under the MIT License.

Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009