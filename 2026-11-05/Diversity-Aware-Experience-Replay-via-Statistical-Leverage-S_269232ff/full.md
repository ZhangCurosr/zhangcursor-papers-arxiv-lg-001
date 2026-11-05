# Diversity-Aware Experience Replay via Statistical Leverage Scores

Shaifalee Saxena<sup>1,2</sup> and Alexander Scheinker<sup>1</sup>

Abstract— Off-policy deep reinforcement learning trains on random samples from a replay buffer of past experience. In physical robot settings, that buffer is dominated by routine: thousands of near-identical transitions from normal operation, while the rare events that define the edges of the task are rarely sampled. We propose scoring every stored transition by its statistical leverage, which measures how unusual it is relative to the buffer’s own data, computed from the stored numbers alone, with no labels, rewards, or value network. We draw each minibatch from a mixture of this score and uniform sampling. First, the natural way to build the score along the directions where the data varies most turns out to be blind when actions are stored as one-hot indicators, because indicator columns always win the variance ranking; we instead select the feature columns with the heaviest tails (largest kurtosis) where small far-away populations are revealed. Second, the importance weights that prioritized replay uses to keep gradients unbiased cancel the method exactly: with full correction, training matches uniform sampling, so we drop the correction and deliberately optimize the rebalanced objective. Third, when the rare situations of a log are relational, visible only in how a stretch of behavior relates to its goal, not in any single stored coordinate—per-transition scores are blind; applying the same tail rule to short trajectory windows described relative to their goal restores the signal, and on a three-joint arm that must push an object around an obstacle it raises the rare-task success from 0.49 to 0.69 against a labeled-oracle ceiling of 0.74.

## I. INTRODUCTION

Deep reinforcement learning (RL) [1] has become a practical tool for synthesizing feedback controllers directly from interaction data. The modern off-policy template descends from the Deep Q-Network (DQN) [2], which revived the classical idea of experience replay [3]: transitions $( s , a , r , s ^ { \prime } )$ are stored in a large ring buffer, and mini-batches drawn from the buffer decorrelate consecutive updates and allow each interaction to be reused many times. Deep Deterministic Policy Gradient (DDPG) [4], followed by Twin Delayed DDPG (TD3) [5] and Soft Actor-Critic (SAC) [6] carried this template to continuous control, where they remain the workhorse algorithms of robot learning [7]. Sample efficiency has been improved mainly by extracting more gradient signal from the same stored data: ensembling and high update-to-data ratios [8], normalization and network scaling [9], and learned state–action representations with prioritized replay [10]. These advances support real-robot manipulation pipelines that train in minutes to hours [11], [12] and autonomous control of large scientific instruments, including the tuning of particle accelerators [13], [14]. Despite this progress, uniform sampling remains the default replay rule in standard off-policy RL.

Uniform sampling implicitly assumes that every stored transition is equally worth revisiting, and this assumption degrades precisely as learning succeeds. A partially converged policy fills the buffer with thousands of near-identical transitions, such as the gripper hovering over the object in almost the same pose, or the beam sitting at a near-optimal setpoint with very low beam loss. At the same time, rare transitions that delineate the boundaries of the task (unusual contacts, failed grasps, off-nominal beam excursions) are sampled with vanishing frequency and eventually overwritten. In this work, we make the buffer’s own geometry the sampling signal. We view the buffer as a tall data matrix, one standardized transition per row, and score each transition by its statistical leverage [15]: the squared, whitened length of its projection onto a chosen score subspace, equivalently a (truncated) Mahalanobis distance from the buffer’s own empirical distribution. Mini-batches are drawn from a mixture placing weight β on the normalized leverage scores and $1 - \beta$ on the uniform distribution. Because $N \gg d ,$ the scores can be refreshed efficiently with buffer-wide statistics and a vectorized pass over the stored transitions.

Our contributions are: (1) a leverage-mixture replay rule using excess-kurtosis column selection to avoid variancesubspace blindness under one-hot actions, and omitting importance-sampling correction, which we show cancels the rebalancing exactly; (2) exact, numerically verified samplingshare laws on Gaussian-mixture buffers, including a rarityindependent amplification ceiling and a multi-cluster mass law; (3) offline Q-learning experiments against exact $Q ^ { * }$ demonstrating improved learning in an underrepresented region; and (4) a window-level extension for relational rare behaviors, where goal-relative trajectory scoring on a threejoint arm trained with TD3+BC improves over transitionlevel replay and approaches a labeled oracle.

## II. RELATED WORK

Non-uniform replay is not a new idea, and its adoption is strikingly uneven. Prioritized Experience Replay (PER) [16], which samples transitions in proportion to their temporaldifference (TD) error, is a standard ingredient of value-based and distributed agents. Yet in the DDPG/TD3/SAC lineage that dominates continuous control, the de facto standard remains plain uniform sampling: the original algorithms use it [4], [5], [6], and it is the default in many implementations.

![](images/18c8bac764bbd5c2d928d8517bcae1aaf97a65211aa62f0f315fed52a4c9d06b.jpg)  
Fig. 1. Overview of the proposed kurtosis-based leverage-mixture replay framework.

![](images/5b76a639c242f464473b674d529bc4b58544068bcd4ab02a632bd61c38b2dbf2.jpg)  
Fig. 2. Two-room gridworld. (a) The logged buffer at $\varepsilon = 0 . 0 5$ : routine operation covers the main room (≈ 275 samples/cell) while the corridor and annex region B, dashed, receive ≈ 12. (b) Ground-truth $Q ^ { * }$ with the optimal escape policy: the annex’s value structure follows its own geometry and cannot be inferred from region-A data.

Where modern robot-learning systems do bias the buffer, they use simple fixed composition rules rather than pertransition priorities: symmetric 50/50 sampling of online and offline data in RLPD and SERL [11], [12].

Empirically, PER’s gains transfer inconsistently from Atari to continuous control, where studies find it often matches or underperforms uniform sampling [17], [18]. Theoretically, prioritized sampling combined with its importance-sampling correction is largely equivalent to uniform sampling under a reshaped loss, so much of its effect can be replicated (or cancelled) by loss design [17]; priorities are computed only when a transition was last visited and therefore go stale; the scheme introduces sensitive hyperparameters; and TDerror-based scores systematically chase noise and outliers in stochastic environments [16], [18], [19]. Broader studies of replay find that buffer capacity, replay ratio, and retention policy often matter as much as the sampling rule itself [20], [21], [22]. When outlier sensitivity is repaired, as in the loss-adjusted prioritization used in TD7 [17], [10], prioritization becomes beneficial again in continuous control, suggesting that the obstacle is not non-uniform sampling per se but the fragility of error-based scores. This motivates a complementary signal that does not depend on the current critic at all.

A smaller body of work targets buffer coverage directly. Experience-selection studies show that which transitions are retained and replayed can matter as much as how many [22]; determinantal point processes (DPPs) have been used to select diverse trajectories and goals within hindsight replay [23] and to prioritize diverse transitions in high-dimensional environments [24]. The recurring obstacle in this line is computational: DPP kernels over N stored transitions are expensive, and much of the engineering effort in these works goes into approximating them. Our proposal can be read as the cheap linear-algebra limit of this idea: rank-k leverage scores are exactly the single-point marginals of a projection DPP, computable for the entire buffer in $O ( N d k )$ per refresh and maintainable incrementally at $O ( d ^ { 2 } )$ per transition. To the best of our knowledge, leverage-score sampling has not previously been studied as a replay-sampling rule for RL.

Recent robot learning has increasingly shifted toward large-scale imitation and vision-language-action models [25]. These trends reduce the role of replay in some settings, but do not eliminate the regime considered here: when physical interaction is expensive, sample-efficient off-policy learning remains important [11], [12]. Recent work such as FastTD3 further shows that replay-based off-policy learning can remain competitive even at large scale and transfer to real hardware [26], while emerging VLA systems are themselves beginning to incorporate reinforcement-learning post-training from autonomously collected experience [27]. Thus, the replay buffer is not disappearing; the question is how stored experience should be selected and reused.

## III. LEVERAGE-MIXTURE REPLAY

## A. Buffer geometry and leverage scores

Each stored transition contributes a feature row $x _ { i } \in \mathbb { R } ^ { d } ;$ a buffer of size N is the matrix $X \in \mathbb { R } ^ { N \times d }$ . Which columns enter $x _ { i }$ is itself a design decision that validation forced us to make explicit (Sec. III-B); by default $x _ { i }$ contains the continuous state features. Columns are standardized,

$$
\bar { x } _ { i j } = \frac { x _ { i j } - \mu _ { j } } { c _ { j } + \epsilon _ { 0 } } , \qquad \epsilon _ { 0 } \approx 1 0 ^ { - 8 } ,\tag{1}
$$

with $\mu _ { j } , c _ { j }$ the buffer mean and standard deviation of column $j .$ The thin SVD ${ \bar { X } } = U \Sigma V ^ { \top }$ gives principal directions $v _ { j }$ , PCA scores ${ \cal T } \ = \ \bar { X } V .$ , and covariance eigenvalues $\bar { \lambda _ { j } } = \sigma _ { j } ^ { 2 } / ( N - 1 )$ of $\begin{array} { r } { C = \frac { 1 } { N - 1 } \bar { X } ^ { \top } \bar { X } } \end{array}$ . For a score subspace spanned by an index set K of directions, the leverage score

of transition i is

$$
\ell _ { i } = \frac { 1 } { N - 1 } \sum _ { j \in \mathcal { K } } \frac { \left( \bar { x } _ { i } ^ { \top } v _ { j } \right) ^ { 2 } } { \lambda _ { j } } , \qquad 0 \leq \ell _ { i } \leq 1 , \quad \sum _ { i = 1 } ^ { N } \ell _ { i } = | K | .\tag{2}
$$

With ${ \mathcal { K } } = \{ 1 , \ldots , d \}$ and invertible C this is exactly the squared Mahalanobis distance of $x _ { i }$ from the buffer’s own mean and covariance, divided by $N - 1$

$$
\begin{array} { r } { \ell _ { i } = \frac { 1 } { N - 1 } \bar { x } _ { i } ^ { \top } C ^ { - 1 } \bar { x } _ { i } = \frac { 1 } { N - 1 } ( x _ { i } - \mu ) ^ { \top } S ^ { - 1 } ( x _ { i } - \mu ) , } \end{array}\tag{3}
$$

where $S$ is the raw-coordinate covariance; the standardization cancels. Equivalently, up to constants $\ell _ { i }$ is the negative logdensity of transition i under a Gaussian fit to the buffer: high leverage means precisely “improbable under the buffer.” Leverage scores are also the single-point inclusion marginals of the projection DPP with kernel $K = U \kappa U _ { \kappa } ^ { \top }$ , so this is the cheap, independent-sampling limit of DPP-style diverse batch selection. Because $N \gg d ,$ , we never form the $N \times$ N side: one $O ( N d ^ { 2 } )$ pass (or $O ( d ^ { 2 } )$ incremental moments) plus an $O ( d ^ { 3 } )$ eigendecomposition, then all N scores in one matrix multiply.

## B. Which columns to score, and how to pick the subspace

Version 1 of this method scored $[ s | a ]$ rows and chose $\kappa$ as the top-k principal directions with the smallest k capturing 90% of variance. Validation shows both choices fail.

Categorical blindness. Suppose the action enters as a $K \cdot$ level one-hot block. Then a takes K equally likely values encoded as indicators ${ \bf 1 } _ { j }$ . Each standardized column has unit variance, and for $\bar { j } \ \ne \ k , \ \mathrm { C o v } ( { \bf 1 } _ { j } , { \bf 1 } _ { k } ) \ = \ - 1 / K ^ { 2 }$ gives correlation $- \frac { 1 } { K - 1 }$ . The block correlation matrix is therefore $\begin{array} { r } { R = \frac { K } { K - 1 } \bar { I } - \frac { 1 } { K - 1 } J , } \end{array}$ , with eigenvalue 0 on the all-ones vector and $K / ( K - 1 )$ with multiplicity $K - 1$ on its complement. Since every standardized continuous column has variance exactly 1, the one-hot contrasts always head the ranking $( 4 / 3$ for $K { = } 4 , \ 3 / 2$ for $K { = } 3 )$ . Raremode separation is therefore hidden inside that unit budget. Variance ranking therefore always places the action contrasts first: in our gridworld (K=4) the measured $\phi ( s , a )$ spectrum is (1.334, 1.333, 1.332, 1.014, 0.986, 0).

Low-variance separating directions. If the rare mode separates along a direction whose within-mode variance is small, truncated leverage is blind until k reaches that direction’s variance rank r and diluted beyond it (Sec. IV); the useful subspace is the smallest one containing the separating directions, which variance ordering does not target.

The amended rule. We select columns, not eigendirections, by a tail statistic: the excess kurtosis $\kappa _ { j } = \mathbb { E } [ \bar { x } _ { \cdot j } ^ { \bar { 4 } } ] - 3$ of each standardized continuous column, keeping the top-m columns $( m { = } 1$ by default) and computing (2) within them. For $m { = } 1$ the selected direction is the coordinate axis $e _ { j ^ { \ast } }$ , where $j ^ { * } =$ arg $\operatorname* { m a x } _ { j } \left\{ \kappa _ { j } \right\}$ , rather than an eigenvector, so the projection in (2) is the standardized entry itself, $\bar { x } _ { i } ^ { \top } e _ { j ^ { \ast } } = \bar { x } _ { i j ^ { \ast } }$ ∗, whose normalizing variance $e _ { i ^ { * } } ^ { \top } C e _ { j ^ { * } } = 1$ by standardization (up to the immaterial $N / ( N - 1 )$ convention factor), giving

$$
\ell _ { i } = \frac { \bar { x } _ { i j ^ { * } } ^ { 2 } } { N - 1 } \propto \bar { x } _ { i j ^ { * } } ^ { 2 } .\tag{4}
$$

No eigendecomposition is required. The normalization $\textstyle \sum _ { i = 1 } ^ { N } { \bar { \ell } } _ { i } = | K | { \bar { \mathbf { \alpha } } } = m$ of (2) is preserved, and because the mixture (19) uses only the normalized scores $\ell _ { i } / \sum _ { k } \ell _ { k }$ , the prefactor is immaterial: a transition’s score is simply the squared standardized value of its heaviest-tailed feature.

A column carrying a rare mass fraction ε at standardized distance z gains ≈ $\varepsilon z ^ { 4 }$ of kurtosis, while uniform or Gaussian background columns are platykurtic; the statistic is the classical projection-pursuit signal for interesting directions [28]. Selecting fixed axes avoids the eigenvector degeneracy that per-column standardization induces (uncorrelated standardized columns give C ≈ I, so principal directions rotate arbitrarily across seeds), and is immune to the onehot pathology. The selection is not free of failure modes of its own—eigen-direction moments can be captured by skewed categorical contrasts, which is why we restrict to continuous columns—and full state-block Mahalanobis (K = all state directions) is the selection-free fallback, in which case (2) is evaluated over all state directions and collapses to the Mahalanobis form (3); this variant is selection-free and never blind, but pays the dilution cost of (10) with $k = d _ { s }$ Reward is excluded from scoring: including r re-introduces outcome-chasing, the very failure of error-based priorities the geometric signal is meant to complement; categorical action encodings are excluded whenever the logged action marginal is region-independent, since they then contribute only dilution and the pathology above.

## IV. ANALYTIC GAUSSIAN MIXTURE STUDY

We first study the sampling rule analytically in an idealized scenario where the standardized buffer X<sup>¯</sup> of (1) is a mixture of two Gaussian clusters: a common cluster A holding a fraction $1 - \varepsilon$ of the transitions and a rare cluster B holding the remaining fraction ε. The two clusters differ along a single direction, described by a unit vector u and their means are a distance $\Delta$ from each other giving the mixture $( 1 - \varepsilon ) \mathcal { N } ( \mu _ { A } , s _ { A } ^ { 2 } ) + \varepsilon \mathcal { N } ( \mu _ { B } , s _ { B } ^ { 2 } )$ . By projecting each transition onto it, $t _ { i } ~ = ~ u ^ { \top } \bar { x } _ { i }$ , we get one number per transition, its coordinate along u. Along u the common cluster scatters around its center with standard deviation $s _ { A }$ and the rare cluster around its own center with standard deviation $s _ { B }$ . The buffer is centered, so the weighted average of the two centers is zero: the common center sits at $- \varepsilon \Delta$ and the rare center at $( 1 - \varepsilon ) \Delta$ . The total variance of $t _ { i }$ is the average within-cluster variance $\bar { s } ^ { 2 } = ( 1 - \varepsilon ) s _ { A } ^ { 2 } + \varepsilon s _ { B } ^ { 2 }$ plus the extra variance created by the two centers being apart,

$$
\lambda = \bar { s } ^ { 2 } + ( 1 - \varepsilon ) ( \varepsilon \Delta ) ^ { 2 } + \varepsilon \big ( ( 1 - \varepsilon ) \Delta \big ) ^ { 2 } = \bar { s } ^ { 2 } + \varepsilon ( 1 - \varepsilon ) \Delta ^ { 2 } .\tag{5}
$$

With $\kappa = \{ u \}$ , the leverage score (2) is $\ell _ { i } = t _ { i } ^ { 2 } / [ ( N - 1 ) \lambda ]$ the squared coordinate along u, normalized so that the scores sum to one over the buffer. Under pure leverage sampling $( \beta = 1$ in (19)) the probability of drawing a rare transition is therefore the total score of cluster B divided by the total score of the buffer. The average of $t _ { i } ^ { 2 }$ inside a cluster is its variance plus its squared center, so the average score of a rare point and of a common point, relative to the buffer-wide

![](images/6e91e918896c92005a41a547e8c54f19fbefbacda5889de7f2194e29c3513f23.jpg)

![](images/744f90e10789f090d68a0f288f5cb445eed0dd6ebc5865ac0440424d7bebcdd5.jpg)

![](images/5a4dc807add0e79a42b53a508f4bb6c54d355539197f3bae4acb66f33c4cb191.jpg)

Fig. 3. Sampling-share laws on two-cluster Gaussian buffers $( N = 2 \times 1 0 ^ { 5 } , d = 2 0 , s _ { A } = s _ { B } = s =$ 1): theory (lines) vs. simulation (markers). (a) Rare share under pure-leverage sampling; dashed: uniform sampling (ϵ), dotted: full-inversion limit $( 1 - \epsilon )$ . (b) Rare batch fraction versus mixing weight $\beta$ at $\epsilon = 0 . 0 2$ , for mixture draws (squares) and stratified batches (triangles). (c) Amplification over uniform sampling at $\beta = 1 / 2 ;$ dotted: theoretical ceiling $1 + \Delta ^ { 2 } / ( 2 s ^ { 2 } )$ , dashed: naive $\dot { 1 } / ( 2 \epsilon )$ ).  
![](images/972effaf94a7336ca68fdd567230ecb95ebcb776afb175a50f4ccbbc38257e39.jpg)

![](images/35a453cb9e0003fb0598bfe58b25ea9a7d747672ece796e00aa034ddc08c911f.jpg)  
Fig. 4. Effect of rare cluster width $( s _ { A } = 1 )$ . (a) Rare share of pureleverage draws at $\Delta { = } 5$ for $s _ { B } / s _ { A }$ from 0.1 to 3, divided by the equalwidth value: lines (7). (b) Fraction of rare points whose score falls below the 99th percentile of the common cluster’s scores $_ { ( \varepsilon = 0 . 0 1 ) }$

average score of 1, are

$$
m _ { B } = \frac { s _ { B } ^ { 2 } + ( 1 - \varepsilon ) ^ { 2 } \Delta ^ { 2 } } { \lambda } , \qquad m _ { A } = \frac { s _ { A } ^ { 2 } + \varepsilon ^ { 2 } \Delta ^ { 2 } } { \lambda } ,\tag{6}
$$

with $( 1 - \varepsilon ) m _ { A } + \varepsilon m _ { B } = 1$ . Cluster B holds a fraction $\varepsilon$ of the points, each worth $m _ { B }$ average points, so its share of pure-leverage draws is $\varepsilon m _ { B } / [ \varepsilon m _ { B } + ( 1 - \varepsilon ) m _ { A } ] = \varepsilon m _ { B } \colon$

$$
\mathrm { s h a r e } _ { B } = \frac { \varepsilon \big ( s _ { B } ^ { 2 } + ( 1 - \varepsilon ) ^ { 2 } \Delta ^ { 2 } \big ) } { ( 1 - \varepsilon ) s _ { A } ^ { 2 } + \varepsilon s _ { B } ^ { 2 } + \varepsilon ( 1 - \varepsilon ) \Delta ^ { 2 } } ,\tag{7}
$$

verified to 0.003 absolute error across $\varepsilon ~ \in ~ [ 0 . 0 0 2 , 0 . 3 ] .$ $\Delta \in \{ 5 , 1 0 , 2 0 \}$ for equal widths (Fig. 3a) and to 0.006 for $s _ { B } / s _ { A } \in [ 0 . 1 , 3 ]$ (Fig. 4a). Everything in (7) is governed by one dimensionless number, $\varepsilon ( 1 - \varepsilon ) \Delta ^ { 2 } / \bar { s } ^ { 2 } ;$ : the variance “bump” the rare cluster adds along u, relative to the withincluster variance. If $\Delta = 0$ the clusters are indistinguishable and share $_ { B } = \varepsilon ,$ , i.e. uniform sampling. If the bump is large, $\varepsilon \Delta ^ { 2 } \gg \bar { s } ^ { 2 }$ , then share $\operatorname { \varepsilon } _ { B }  1 - \varepsilon \colon$ the rare cluster receives the share the common cluster used to have, a full inversion of the buffer’s composition; for a 1% event this needs a separation of well over ten standard deviations. If the bump is small, $\varepsilon \Delta ^ { 2 } \ll s _ { A } ^ { 2 }$ , then share ${ \bf \Phi } : _ { B } \approx \varepsilon ( \Delta ^ { 2 } + s _ { B } ^ { 2 } ) / s _ { A } ^ { 2 }$

The rare cluster’s own width plays only a minor role, as seen in (6): a rare point’s score is its squared distance from the buffer mean, which is $\Delta ^ { 2 }$ from the offset plus $s _ { B } ^ { 2 }$ from the cluster’s internal scatter, measured against a total variance that is set by the common cluster’s width $s _ { A }$ (and by the bump). Therefore $s _ { B }$ only enters additively next to $\Delta ^ { \bar { 2 } }$ . At $\Delta \ : = \ : 5$ and $\varepsilon ~ = ~ 0 . 0 1$ a tight rare cluster $( s _ { B } \leq 0 . 3 s _ { A } )$ receives 3% less share than an equal-width one and a diffuse cluster $( s _ { B } = 3 s _ { A } )$ 24% more; at $\Delta = 1 0$ all widths are within 4% of each other (Fig. 4a). A tight rare cluster ${ \mathrm { i s } } ,$ if anything, found more cleanly: all of its points sit at the same distance from the mean, so none of them is scored like a routine transition, whereas for $s _ { B } = 3 s _ { A }$ at $\Delta = 5$ one fifth of the rare points score below the 99th percentile of the common cluster (Fig. 4b). Leverage looks for distance from the bulk, not for local density, so a compact rare mode cannot hide from it the way it can from a density- or nearest-neighbor-based novelty score. The column-selection rule behaves the same way: in these sweeps the separating column has the largest excess kurtosis of all 20 columns for every $\varepsilon \ \leq \ 0 . 1$ and every width tested, so the rule of Sec. III-B finds it without labels; between $\varepsilon = 0 . 1$ and 0.3 the column turns from heavy-tailed into bimodal and its excess kurtosis becomes negative (in the tight, far limit the column is a two-point distribution whose excess kurtosis crosses zero at $\varepsilon \approx 0 . 2 1$ ; wider rare clusters cross later), so the rule stops selecting it, which is consistent with its purpose as a raremode detector. Under the mixture rule (19) a fraction $1 - \beta$ of each batch is drawn uniformly and $\beta$ by leverage, so the expected rare fraction of a batch is

$$
p _ { B } ( \beta ) = ( 1 - \beta ) \varepsilon + \beta \mathrm { s h a r e } _ { B } ,\tag{8}
$$

and $p _ { B } ( \beta ) / \varepsilon$ is rare cluster amplification over uniform.

Three consequences of this analysis inform our method. (i) Amplification is separation-limited, not rarity-limited.

Making the event rarer at fixed $\Delta$ gives share<sub>B</sub> ${ \bf \nabla } / \varepsilon \to ( \Delta ^ { 2 } +$ $s _ { B } ^ { 2 } ) / s _ { A } ^ { \overline { { { 2 } } } } , \ s o \ p _ { B } ( \beta ) / \varepsilon  1 + \beta \big [ ( \bar { \Delta ^ { 2 } } + s _ { B } ^ { 2 } ) / s _ { A } ^ { 2 } - 1 \big ] ~$ , which for equal widths is $1 + \beta \Delta ^ { 2 } / s ^ { 2 } ;$ at $\begin{array} { r } { \beta = \frac { 1 } { 2 } } \end{array}$ the amplification saturates at $1 + \Delta ^ { 2 } / 2 s ^ { 2 }$ , i.e. 13.5×, 51×, and 201× for $\Delta = 5 , 1 0 , 2 0 .$ , not at the naive $1 / 2 \varepsilon$ that would put half of every batch in the rare cluster $( 2 5 0 \times \mathrm { ~ a t ~ } \varepsilon = 0 . 0 0 2 )$ . The ceiling is reached to within $1 0 \%$ once $\varepsilon \Delta ^ { 2 } / s _ { A } ^ { 2 } \lesssim 0 . 1 \mathrm { : }$ at $\varepsilon = 0 . 0 0 2$ the simulated amplifications are 12.9×, 42.8×, and $1 1 1 \times ,$ so the $\Delta { = } 5$ curve is at its ceiling while the $\Delta { = } 2 0$ curve is still far below it (it would need $\varepsilon \lesssim 3 \times 1 0 ^ { - 4 } )$ In practice, a rare mode that is not geometrically distinct from routine operation will not be amplified much no matter how rare or how compact it is, and the boost of one that is distinct can be predicted from $\Delta / s _ { A } . \ ( i i )$ Batch composition is exactly linear in β, (8), and a stratified batch made of $( 1 - \beta ) B$ uniform draws and $\beta B$ leverage draws has the same expected composition as B draws from the mixture (Fig. 3b, agreement within 0.002); an implementation can therefore keep its uniform sampler and simply append a leverage-sampled half-batch. (iii) Multi-cluster mass law. With q clusters of fractions $\pi _ { 1 } , \ldots , \pi _ { q }$ whose centers are far apart compared with their spreads, the centered centers span $\mathrm { ~ a ~ } ( q { - } 1 )$ )-dimensional subspace; taking $\kappa$ to be that subspace, every point of cluster c sits essentially at its center, and (2) gives it the score $1 / ( N \pi _ { c } ) - 1 / N$ (the $N \pi _ { c }$ points of a cluster share one location’s pull on the fit, minus a centering correction), the total leverage cluster c mass is

$$
\begin{array} { r } { M _ { c } = 1 - \pi _ { c } , \qquad \sum _ { c } M _ { c } = q - 1 = | { \cal K } | , } \end{array}\tag{9}
$$

and pure-leverage shares are $( 1 - \pi _ { c } ) / ( q - 1 )$ : a cluster’s share depends only on its size, with an amplification $( 1 -$ ${ \pi } _ { c } ) / [ ( q - 1 ) { \pi } _ { c } ]$ that exceeds one for small clusters and falls below one for large ones. For four clusters of fractions $0 . 7 0 / 0 . 2 0 / 0 . 0 8 / 0 . 0 2$ , each minority cluster displaced by $\Delta$ along its own feature column, the 2% cluster is amplified 14.5× and the 70% cluster suppressed to 0.19× at $\Delta = 2 0$ approaching the limits 16.3× and 0.14× (16.0× and 0.15× at $\Delta = 5 0 ;$ Fig. 5a); the law requires $\Delta ^ { 2 } \gg 1 / \pi _ { c }$ for every cluster. The three highest-kurtosis columns are exactly the three displaced columns, so the selection rule recovers this subspace with no labels anywhere.

Real buffers vary along many directions with very different variances, and the rare cluster is displaced along only one of them, the separating direction, whose total variance is $\lambda = \bar { s } ^ { 2 } + \varepsilon ( 1 - \varepsilon ) \Delta ^ { 2 }$ from (5): its within-cluster variance plus the small bump. Let r be its variance rank, its position when all directions of the buffer are sorted from largest to smallest variance. If K is the top-k principle directions, then for $k \ < \ r$ the separating direction is not in the subspace at all and leverage is blind, shar $\begin{array} { r } { \mathrm { ~  ~ \psi ~ } _ { B } ^ { \mathrm { ~ } } = \varepsilon ; } \end{array}$ for $k \geq r$ it is included together with $k - 1$ background directions that carry no information about the clusters, and since every transition, rare or common, receives on average one unit of score along each such direction, the share is diluted to

$$
\operatorname { s h a r e } _ { B } ( k ) = { \frac { \varepsilon \left( k - 1 + m _ { B } \right) } { \varepsilon \left( k - 1 + m _ { B } \right) + ( 1 - \varepsilon ) ( k - 1 + m _ { A } ) } } ,\tag{10}
$$

with $m _ { B } , m _ { A }$ from (6); this equals (7) at $k = 1$ and decays back toward ε as k grows. In the example of Fig. 5b $( r = 8 )$ top-k leverage amplifies the rare cluster 1.0× for $k < 8 ,$ $5 . 0 \times$ at $k = 8 ,$ , and 2.6× with all $d = 2 0$ directions (the full Mahalanobis score), against 32.8× for the separating direction alone. For the boundary between the two regimes, the separating direction enters the top k when the bump lifts its variance above the k-th largest background variance $\lambda _ { ( k ) } ^ { \mathrm { w } } ,$

$$
\varepsilon ( 1 - \varepsilon ) \Delta ^ { 2 } \ > \ \lambda _ { ( k ) } ^ { \mathrm { w } } / \bar { s } ^ { 2 } - 1 ,\tag{11}
$$

which reproduces the blind/visible boundary of the simulated $( \varepsilon , \Delta )$ phase diagram in 97% of its cells (Fig. 5c). Ridge leverage, $\begin{array} { r } { \ell _ { i } ^ { \gamma } = \frac { \bar { 1 } } { N - 1 } \bar { x } _ { i } ^ { \top } ( C + \gamma I ) ^ { - 1 } \bar { x } _ { i } } \end{array}$ , keeps every direction and shrinks the low-variance ones, so it is never blind, but it is diluted by the effective dimension $\begin{array} { r } { d _ { \mathrm { e f f } } ( \gamma ) = \sum _ { i } \lambda _ { j } / ( \lambda _ { j } + } \end{array}$ $\gamma )$ that it retains; its optimum is interior (too small a γ dilutes over all d directions, too large $\mathrm { ~ a ~ } \gamma$ shrinks away a low-variance separating direction), and in the same example the best $\gamma$ recovers only 2.9×. The design goal is thus the smallest score subspace containing the separating directions. Variance ranking does not target it, because it orders directions by how much the data spreads, not by whether a small, far-away group is present; a column carrying such a group has heavy tails regardless of its variance, which is the motivation for tail-based column selection of Sec. III-B.

## V. EXPERIMENTS

## A. Simple 2D Grid World

As a proof-of-principal experiment we consider a Qlearning Deep RL agent trying to find the goal in a simple 2D grid world which is a $2 4 \times 1 2$ lattice $( x \in \{ 0 , \ldots , 2 3 \}$ $y \in \{ 0 , \ldots , 1 1 \} )$ , with a main room consisting of 144 cells located at x $\mathbf { \sigma } \in \{ 0 , 1 1 \} , y \in \{ 0 , 1 1 \}$ , a narrow corridor whose cells are $x \in \{ 1 2 , \ldots , 1 5 \} , y = 5 ,$ , and an annex whose cells are $x \in \{ 1 6 , \ldots , 2 3 \} , y \in \{ 2 , \ldots , 9 \}$ . The goal in this case is always located at position (1, 5) in the main room. The goal is in the main room, escaping the annex requires values propagated along the chain: annex → corridor → room, this structure is invisible when focused on only main-room data. We model this as a Markov decision process (MDP) tuple $( S , { \mathcal { A } } , P , r , \gamma )$ , a state space, action space, transition probabilities $P ( \mathbf { s } ^ { \prime } | \mathbf { s } , a )$ , a reward function, and a discount factor. Here the state is the cell we occupy, $\mathrm { ~ \bf ~ s ~ } = \mathrm { ~ \bf ~ ( } x , y ) ;$ the actions are ${ \mathcal { A } } = ( { \mathrm { u p } }$ , down, left, right); the transition rule is “move one cell in the chosen direction, except with probability 0.1 the action slips to a uniformly random one, and bumping into a wall leaves you in place”; the reward is $r ~ = ~ - 1$ for every step; the episode ends at the goal $( 1 , 5 )$ ; and $\gamma = 0 . 9 9$ . Maximizing the return is equivalent to minimizing the number of steps required to reach the goal. Starting at $t = 0$ and taking $T$ steps to reach the goal gives

$$
G = \sum _ { t = 0 } ^ { T - 1 } \gamma ^ { t } ( - 1 ) = - \frac { 1 - \gamma ^ { T } } { 1 - \gamma } .\tag{12}
$$

![](images/c620535f61d04c8d412707f14816fe80fca2e23f9c68d8a9684df59c0d29cb5b.jpg)

![](images/416b1096484f1957fb445b112cb30e6f12d1f6ecfd0ae77553f65d44c7c22e99.jpg)

![](images/3ed4b245ca87aaf33404b6cd6982cc2e7c6ef9b1bcf95d61fa84d35f4dbfa977.jpg)  
Fig. 5. Leverage behavior in higher-dimensional buffers. (a) Multi-cluster leverage mass $M _ { c }$ versus the law $1 - \pi _ { c }$ in (9). (b) Rare share under top-k PCA leverage when the separating direction has variance rank $r = 8 ;$ lines show (10) and leverage along the separating direction alone. (c) Rare-cluster amplification under top-4 PCA leverage over $( \epsilon , \Delta )$ ; white curve shows the predicted boundary (11)

For a policy $\pi ,$ if action a is taken now and $\pi$ is followed afterwards, the expected return is

$$
Q ^ { \pi } ( \mathbf { s } , a ) = \mathbb { E } \left[ G _ { t } | \mathbf { s } _ { t } = \mathbf { s } , a _ { t } = a , \mathrm { \ t h e n \ f o l l o w \ } \pi \right] ,\tag{13}
$$

and the optimal action-value function is $\begin{array} { r l } { Q ^ { * } ( \mathbf { s } , a ) } & { { } = } \end{array}$ $\operatorname* { m a x } _ { \pi } Q ^ { \pi } ( \mathbf { s } , a )$ , which is the best expected return achievable from $( \mathbf { s } , a )$ , whose state version is $V ^ { * } ( \mathbf { s } ) = \operatorname* { m a x } _ { a } Q ^ { * } ( \mathbf { s } , a )$ According to the Bellman optimality equation, $Q ^ { * }$ is the optimal solution of

$$
\begin{array} { r c l } { Q ^ { * } ( \mathbf { s } , a ) } & { = } & { \mathbb { E } _ { \mathbf { s } ^ { \prime } \sim P ( \cdot \vert \mathbf { s } , a ) } \left[ r + \gamma \underset { a ^ { \prime } } { \operatorname* { m a x } } Q ^ { * } ( \mathbf { s } ^ { \prime } , a ^ { \prime } ) \right] } \\ & { = } & { \displaystyle - 1 + \gamma \sum _ { a ^ { \prime \prime } } p ( a ^ { \prime \prime } \vert a ) V ^ { * } ( \mathrm { n e x t } ( \mathbf { s } , a ^ { \prime \prime } ) ) , } \end{array}\tag{14}
$$

(15)

where $V ^ { * } ( \mathrm { g o a l } ) = 0$ and we have simplified the equation because in our case $r \equiv - 1$ , and the randomness comes from the slip so $a ^ { \prime \prime }$ represents the action actually taken. We can use this rule iteratively to find $Q ^ { * }$ exactly for this simple finite-state grid world. The setup is shown in Figure 2 with the image on the far right colored by $Q ^ { * }$ for each cell. For our Q-learning setup, we assume we do not have access to $P$ and we use a fixed replay buffer of transitions $( \mathbf { s } , a , r , \mathbf { s } ^ { \prime } )$ , where Q-learning replaces the expectation in the Bellman equation with a single sampled transition, each giving a temporaldifference (TD) target

$$
y = r + \gamma ( 1 - \mathrm { d o n e } ) \operatorname* { m a x } _ { a ^ { \prime } } Q ( \mathbf { s } ^ { \prime } , a ^ { \prime } ) ,\tag{16}
$$

and we train a neural network model to learn $Q _ { \theta } ( \mathbf { s } , a )$ by minimizing the training cost function $( Q _ { \theta } ( \mathbf { s } , a ) - y ) ^ { 2 }$ over mini-batches from the buffer. As $Q _ { \theta } ( \mathbf { s } , a )$ is trained, for each state $\textbf { s } = ~ ( x , y )$ it outputs four numbers which are its estimates of the expected discounted return for each action taken from that cell. TThe agent then chooses actions greedily according to

$$
\pi ( \mathbf { s } ) = \operatorname * { a r g m a x } _ { a \in \mathcal { A } } Q _ { \theta } ( \mathbf { s } , a ) .\tag{17}
$$

We perform an experiment in which we fill a buffer with $N = 4 0$ , 000 transitions $( \mathbf { s } , a , r , \mathbf { s } ^ { \prime } )$ , with a fraction $\epsilon = 0 . 0 5$ of them coming from the rare corridor + annex region and $1 - \epsilon = 0 . 9 5$ from the main room. We use the approach described above, standardizing every column by subtracting its mean and dividing by its standard deviation, and consider the excess kurtosis of each standardized column

$$
\bar { x } _ { i j } = \frac { x _ { i j } - \mu _ { j } } { c _ { j } } , \quad \kappa _ { j } = \frac { 1 } { N } \sum _ { i } \bar { x } _ { i j } ^ { 4 } - 3 .\tag{18}
$$

In this simple example, we score only the state coordinates because $r \ \equiv \ - 1$ , and get excess kurtosis values of 1.87 for the x−column and −1.19 for the y-column. We then compare various buffer sampling approaches for training.

• Uniform: $p _ { i } = 1 / N$ to each entry in the buffer.

• β-leverage:

$$
p _ { i } = ( 1 - \beta ) \frac { 1 } { N } + \beta \frac { \ell _ { i } } { \sum _ { j } \ell _ { j } } , \quad \beta \in [ 0 , 1 ] . \quad \ell _ { j } = z _ { i j ^ { * } } ^ { 2 }\tag{19}
$$

Column of standardized x values chosen by kurtosis.

• Tail-flat: This approach uses the score to only find the top-decile tail T and then samples uniformly inside of there, taking T as the top-decile leverage tail, giving

$$
p _ { i } = ( 1 - \beta ) \frac { 1 } { N } + \beta \frac { \mathbf { 1 } [ i \in T ] } { | T | } .\tag{20}
$$

• Oracle: The oracle knows rare/common locations uniformly samples half of its values from each.

We use a total batch size of $B \ = \ 2 5 6$ and our network $Q _ { \theta } : \mathbb { R } ^ { 2 }  \mathbb { R } ^ { 4 }$ is a simple dense network with 2 hidden layers of width 24 each and ReLU activation functions. Once we start training the network, we check its performance on the common room region (A) and on the rare corridor + annex region (B) after every 500 training (network weight gradient update) steps. For each check, we start in every single one of the 68 cells in region B as well as in every single one of the 143 cells in region A, apply the greedy policy of Equation (17) for a maximum of 100 steps, and record the total fraction of starts which reach the goal within those 100 steps. We also calculate the root mean squared error of $Q _ { \theta }$ in each region $\mathcal { R } \in \{ { \bf A } , { \bf B } \}$ , relative to the ground truth $Q ^ { * }$ according to

TABLE I  
SUCCESS RATE ON TASK B AND Q-VALUE RMSE ON TASKS B AND A UNDER EACH SAMPLING CONDITION (MEAN ± STD.) OVER 20 SEEDS.
<table><tr><td></td><td colspan="2">Q-RMSE</td></tr><tr><td>Condition</td><td>Final succ. B B</td><td>A</td></tr><tr><td>Uniform</td><td> $0 . 8 8 2 \pm 0 . 0 1 8$  3.33</td><td>0.31</td></tr><tr><td>Leverage  $( \beta = 0 . 5 )$ </td><td> $0 . 9 4 1 \pm 0 . 0 1 0$  2.42</td><td>0.36</td></tr><tr><td>Leverage  $( \beta = 0 . 7 5 )$ </td><td> $0 . 9 2 8 \pm 0 . 0 2 1$  1.43</td><td>0.38</td></tr><tr><td>Tail-flat  $( \beta = 0 . 5 )$ </td><td> $0 . 9 7 8 \pm 0 . 0 0 7$  1.63</td><td>0.57</td></tr><tr><td>Tail-flat  $( \beta = 0 . 7 5 )$ </td><td> $\mathbf { 0 . 9 8 9 \pm 0 . 0 0 6 }$  1.58</td><td>0.87</td></tr><tr><td>PER (α = 0.6)</td><td> $0 . 8 8 8 \pm 0 . 0 2 4$  2.71</td><td>0.26</td></tr><tr><td>Oracle 50/50</td><td> $0 . 9 8 7 \pm 0 . 0 0 9$  1.06</td><td>0.72</td></tr></table>

![](images/e79fab6979f7f07ae4cf3738b1e8eabdbc709b8c49758ce0b82fe3501810d978.jpg)  
Fig. 6. Success rate for each method for regions A and B as a function of training steps of $Q _ { \theta } ( \mathbf { s } , a )$ are shown for the various methods. All methods quickly succeed in the common region A (opaque curves), and their performance separates in the rare/difficult region B with our proposed methods all outperforming PER and uniform sampling.

$$
\mathrm { R M S E } _ { \mathcal { R } } = \sqrt { \frac { 1 } { 4 | \mathcal { R } | } \sum _ { \mathbf { s } \in \mathcal { R } } \sum _ { a } \big [ Q _ { \theta } ( \mathbf { s } , a ) - Q ^ { * } ( \mathbf { s } , a ) \big ] ^ { 2 } } .\tag{21}
$$

We repeat this for 20 different random seeds, for $\beta = 0 . 5$ and $\beta = 0 . 7 5$ . The success rate vs training step results are shown in Figure 6 and are summarized in Table I.

## B. Robot Arm & Window Tail-Flat

The gridworld’s rare region B is the set of large x, so the kurtosis rule found it in a single column. The second experiment is built so that this is not the case. A threelink planar arm (Fig. 7) pushes a disk from a start $o _ { 0 }$ to a goal g past a fixed obstacle centered at C; the log is generated by a scripted controller with exploration noise, with 950 straight episodes, in which the line of sight from $o _ { 0 }$ to g clears the obstacle, and 35 orbit episodes, in which $g$ lies behind it and the object must be pushed around; the learner is TD3+BC [29], trained offline on this fixed log. The object start $o _ { 0 }$ and goal g are drawn from the same annulus around C, so every object and goal coordinate has the same marginal; what makes an orbit transition rare is where the goal sits relative to the object and the obstacle, and that this relation persists for tens of steps while the object travels around. Per-transition scoring of the state columns does not see this, so the rule of Sec. III-B falls back on a joint-angle tail. Extreme arm configurations arereached in both families, and the row-level leverage and tail-flat samplers raise the orbit share of a batch only from 0.069 to 0.10–0.13, leaving orbit success at the uniform level. This is the failure mode anticipated in Sec. III-B, where the choice of which columns enter $x _ { i }$ was flagged as decisive.

TABLE II  
TD3+BC FOR 20 SEEDS & 10K GRADIENT STEPS. ORBIT PERFORMANCE IS AVERAGED OVER THE LAST THREE EVALUATIONS.
<table><tr><td>Condition</td><td>Orbit (last-3)</td><td>Straight</td><td>Batch orbit share</td></tr><tr><td>Uniform</td><td>0.49</td><td>1.00</td><td>0.069</td></tr><tr><td>Window tail-flat (b = 0.5)</td><td>0.69</td><td>1.00</td><td>0.220</td></tr><tr><td>Leverage  $( \beta = 0 . 5 )$ </td><td>0.51</td><td>1.00</td><td>0.110</td></tr><tr><td>Leverage (β = 0.75)</td><td>0.44</td><td>1.00</td><td>0.131</td></tr><tr><td>Tail-flat (β = 0.5)</td><td>0.45</td><td>1.00</td><td>0.098</td></tr><tr><td>Tail-flat (β = 0.75)</td><td>0.38</td><td>0.99</td><td>0.113</td></tr><tr><td>PER (α = 0.6)</td><td>0.51</td><td>0.99</td><td>0.084</td></tr><tr><td>Oracle 50/50</td><td>0.74</td><td>1.00</td><td>0.500</td></tr></table>

![](images/7a1577c94737b14a04cfd5323a0477b28141aea006676711f486299d893363ff.jpg)  
Fig. 7. The dashed circles show the annulus from which the object start o<sub>0</sub> and goal $g$ are drawn. (a) For straight tasks the line of sight $o _ { 0 }  g$ clears the obstacle. (b) For orbit tasks the goal lies behind the obstacle and the object must be pushed around it. Solid curves are logged object paths (scripted controller with exploration noise), dotted curves the tool-tip path.

Window tail-flat. Each logged episode is cut into consecutive windows w of $L _ { w } = 2 4$ transitions, and a window is described relative to its own goal,

$$
z _ { w } = \left[ g _ { w } , ~ g _ { w } - \bar { o } _ { w } , ~ \left\| g _ { w } - \bar { o } _ { w } \right\| \right] \in \mathbb { R } ^ { 5 } ,\tag{22}
$$

where $\begin{array} { r } { \bar { o } _ { w } = \frac { 1 } { | w | } \sum _ { i \in w } o _ { i } } \end{array}$ is the mean object position over the window and $g _ { w }$ its (episode-constant) goal: where the goal is, the offset from where the object currently is to it, and how far it still has to ${ \bf g o . }$ . The descriptors are standardized as in (1) and scored by the selection-free Mahalanobis form (3) over all $d _ { w } = 5$ descriptor directions,

RARE task: orbit success (dotted: scripted ceiling)  
![](images/31fda4761d4e87006163620aa4835b40d12f5d6cf6c835eaa3eac4df76db1a3f.jpg)  
Fig. 8. Comparison of the various approaches for the 3-joint robot arm.

$$
\ell _ { w } = \bar { z } _ { w } ^ { \top } R _ { z } ^ { - 1 } \bar { z } _ { w } ,\tag{23}
$$

with $R _ { z }$ the descriptor correlation matrix; with $\approx \textit { N } / L _ { w }$ windows and $d _ { w } = 5$ this is one $5 \times 5$ inverse. The topdecile tail $T _ { w } = \{ w : \ell _ { w } \geq Q _ { 0 . 9 } ( \ell ) \}$ is then replayed with the tail-flat randomly chosen as in $\left( \mathrm { V } { \cdot } \mathrm { A } \right)$ applied to windows with $p = 0 . 5$ a window uniformly and a transition uniformly inside it; $p = 0 . 5$ , a uniform transition

$$
p _ { i } = ( 1 - \beta ) \frac { 1 } { N } + \beta \frac { 1 [ w ( i ) \in T _ { w } ] } { \left| T _ { w } \right| \left| w ( i ) \right| } , \qquad \beta = 0 . 5 ,\tag{24}
$$

where $w ( i )$ is the window containing transition i. As in (8) the orbit share of a batch is linear in $\beta , ( 1 - \beta ) \varepsilon + \beta \pi _ { T }$ with $\pi _ { T }$ the orbit fraction of the tail’s transitions; here π ≈ 0.37, more than five times the base rate $\varepsilon = 0 . 0 6 9$ , so at $\beta = 0 . 5$ the batch share rises to 0.22 with no labels, rewards, or critic involved. Window tail-flat is thus the rule of Sec. III with two of its free choices made differently. The unit scored is a window rather than a transition, and the feature row is a goal-relative summary of that window rather than raw state columns. The score (3), the tail, and the mixture (V-A) are unchanged. Table II and Figure 8 summarize the results.

## VI. CONCLUSION

We introduced a diversity-aware experience replay framework that uses statistical leverage to preferentially revisit underrepresented experience without relying on rewards, labels, or the current value function. Analysis of Gaussianmixture buffers characterized when leverage amplifies rare data, showing that amplification is governed by geometric separation rather than rarity alone. The analysis also revealed that variance-ranked subspaces can miss rare modes, motivating kurtosis-based feature selection. In the gridworld, the resulting leverage and tail-based replay rules improved learning in the underrepresented region compared with uniform replay and PER. The robot-arm experiment further showed that the representation used for scoring is critical: when rare behavior could not be distinguished from individual state coordinates, transition-level scoring provided little improvement, whereas scoring short goal-relative trajectory windows increased rare-task success from 0.49 to 0.69, compared with 0.74 for a labeled oracle. These results suggest that statistical leverage provides a simple, critic-independent mechanism for rebalancing imbalanced replay buffers.

## REFERENCES

[1] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. Cambridge, MA: MIT Press, 2018.

[2] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski et al., “Human-level control through deep reinforcement learning,” nature, vol. 518, no. 7540, pp. 529–533, 2015.

[3] L.-J. Lin, “Self-improving reactive agents based on reinforcement learning, planning and teaching,” Machine Learning, vol. 8, no. 3– 4, pp. 293–321, 1992.

[4] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra, “Continuous control with deep reinforcement learning,” arXiv preprint arXiv:1509.02971, 2015.

[5] S. Fujimoto, H. van Hoof, and D. Meger, “Addressing function approximation error in actor-critic methods,” in Proceedings of the 35th International Conference on Machine Learning (ICML), ser. Proceedings of Machine Learning Research, vol. 80, 2018.

[6] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor,” in Proceedings of the 35th International Conference on Machine Learning (ICML), ser. Proceedings of Machine Learning Research, vol. 80, 2018, pp. 1861–1870.

[7] J. Ibarz, J. Tan, C. Finn, M. Kalakrishnan, P. Pastor, and S. Levine, “How to train your robot with deep reinforcement learning: Lessons we have learned,” The International Journal of Robotics Research, vol. 40, no. 4–5, pp. 698–721, 2021.

[8] X. Chen, C. Wang, Z. Zhou, and K. Ross, “Randomized ensembled double Q-learning: Learning fast without a model,” in International Conference on Learning Representations (ICLR), 2021.

[9] A. Bhatt, D. Palenicek, B. Belousov, M. Argus, A. Amiranashvili, T. Brox, and J. Peters, “CrossQ: Batch normalization in deep reinforcement learning for greater sample efficiency and simplicity,” in International Conference on Learning Representations (ICLR), 2024.

[10] S. Fujimoto, W.-D. Chang, E. J. Smith, S. S. Gu, D. Precup, and D. Meger, “For SALE: State-action representation learning for deep reinforcement learning,” in Advances in Neural Information Processing Systems, vol. 36, 2023.

[11] P. J. Ball, L. Smith, I. Kostrikov, and S. Levine, “Efficient online reinforcement learning with offline data,” in Proceedings of the 40th International Conference on Machine Learning (ICML), ser. Proceedings of Machine Learning Research, vol. 202, 2023.

[12] J. Luo, Z. Hu, C. Xu, Y. L. Tan, J. Berg, A. Sharma, S. Schaal, C. Finn, A. Gupta, and S. Levine, “Serl: A software suite for sampleefficient robotic reinforcement learning,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 16 961–16 969.

[13] S. Saxena, A. Williams, R. Fierro, and A. Scheinker, “Improved robustness of deep reinforcement learning for control of time-varying systems by bounded extremum seeking,” in 2026 American Control Conference (ACC). IEEE, 2026, pp. 1165–1172.

[14] S. Saxena and A. Scheinker, “Mahalanobis-guided latent ood detection for hybrid es-drl control in time-varying systems,” arXiv preprint arXiv:2606.11474, 2026.

[15] P. Drineas, M. Magdon-Ismail, M. W. Mahoney, and D. P. Woodruff, “Fast approximation of matrix coherence and statistical leverage,” Journal of Machine Learning Research, vol. 13, pp. 3475–3506, 2012.

[16] T. Schaul, J. Quan, I. Antonoglou, and D. Silver, “Prioritized experience replay,” in International Conference on Learning Representations (ICLR), 2016, arXiv:1511.05952.

[17] S. Fujimoto, D. Meger, and D. Precup, “An equivalence between loss functions and non-uniform sampling in experience replay,” in Advances in Neural Information Processing Systems, vol. 33, 2020.

[18] T. Lahire, M. Geist, and E. Rachelson, “Large batch experience replay,” in Proceedings of the 39th International Conference on Machine Learning (ICML), ser. Proceedings of Machine Learning Research, vol. 162, 2022, pp. 11 790–11 813.

[19] Y. Pan, J. Mei, A.-m. Farahmand, M. White, H. Yao, M. Rohani, and J. Luo, “Understanding and mitigating the limitations of prioritized experience replay,” in Proceedings of the 38th Conference on Uncertainty in Artificial Intelligence (UAI), ser. Proceedings of Machine Learning Research, vol. 180, 2022, pp. 1561–1571.

[20] W. Fedus, P. Ramachandran, R. Agarwal, Y. Bengio, H. Larochelle, M. Rowland, and W. Dabney, “Revisiting fundamentals of experience replay,” in Proceedings of the 37th International Conference on Machine Learning (ICML), ser. Proceedings of Machine Learning Research, vol. 119, 2020.

[21] S. Zhang and R. S. Sutton, “A deeper look at experience replay,” arXiv preprint arXiv:1712.01275, 2017.

[22] T. de Bruin, J. Kober, K. Tuyls, and R. Babuska, “Experience selection ˇ in deep reinforcement learning for control,” Journal of Machine Learning Research, vol. 19, no. 9, pp. 1–56, 2018.

[23] T. Dai, H. Liu, K. Arulkumaran, G. Ren, and A. A. Bharath, “Diversity-based trajectory and goal selection with hindsight experience replay,” in Pacific rim international conference on artificial intelligence. Springer, 2021, pp. 32–45.

[24] K. Zhao, Y. Wang, Y. Chen, Y. Li, L. H. U, and X. Niu, “Efficient diversity-based experience replay for deep reinforcement learning,” in Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI), 2025, pp. 7083–7091.

[25] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter et al., “π<sub>0</sub>: A visionlanguage-action flow model for general robot control,” arXiv preprint arXiv:2410.24164, 2024.

[26] Y. Seo, C. Sferrazza, H. Geng, M. Nauman, Z.-H. Yin, and P. Abbeel, “FastTD3: Simple, fast, and capable reinforcement learning for humanoid control,” arXiv preprint arXiv:2505.22642, 2025.

[27] Y. Guo, J. Zhang, X. Chen, X. Ji, Y.-J. Wang, Y. Hu, and J. Chen, “Improving vision-language-action model with online reinforcement learning,” in 2025 IEEE International Conference on Robotics and Automation (ICRA), 2025, pp. 15 665–15 672.

[28] P. J. Huber, “Projection pursuit,” The Annals of Statistics, vol. 13, no. 2, pp. 435–475, 1985.

[29] S. Fujimoto and S. S. Gu, “A minimalist approach to offline reinforcement learning,” in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 20 132–20 145.