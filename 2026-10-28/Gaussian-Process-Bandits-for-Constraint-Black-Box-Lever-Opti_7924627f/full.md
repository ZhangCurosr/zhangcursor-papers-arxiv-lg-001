# Gaussian Process Bandits for Constraint Black Box Lever Optimization

Bram van den Akker bram.vandenakker@booking.com Booking.com Amsterdam, The Netherlands

Hidde Fokkema\* hidde.fokkema@gmail.com University of Amsterdam Amsterdam, The Netherlands

Anastasiia Kornilova anastasiia.kornilova@booking.com Booking.com Amsterdam, The Netherlands

## Abstract

Optimizing discrete allocation weights across competing segments of a shared budget is a fundamental challenge in resource-constrained sequential decision-making environments. In practice, this optimization is often hindered by delayed, noisy, and aggregate feedback, alongside strict operational constraints on how frequently weights can be adjusted. Because of the resulting combinatorial action space and extreme data sparsity, classical multi-armed bandit algorithms that evaluate arms independently are rendered highly inefficient. To overcome these limitations, we propose a segmentationagnostic approach utilizing Gaussian Process Upper Confidence Bound (GP-UCB) algorithms. By placing a Gaussian Process prior on the reward function, our method effectively shares information across neighboring configurations, drastically reducing the necessary exploration horizon. In this paper, we bridge the gap between theoretical bandit research and production-scale deployment. We first introduce a high-fidelity synthetic simulator to evaluate various kernel configurations under stationary, drifting, and reactive environments, identifying a composite Matérn + Linear kernel as the most robust choice for overcoming local noise and global shifts. Finally, we validate our findings through a large-scale Randomized Controlled Trial (RCT) in a live production environment. Our empirical results demonstrate that the GP-UCB approach delivers a statistically significant efficiency lift compared to a standard ε- greedy baseline, offering a highly practical solution for real-world black-box lever optimization.

## 1 Introduction

Consider a decision-maker who must periodically set discrete allocation weights across competing segments of a shared budget, then wait days or weeks for a single noisy aggregate outcome before being allowed to adjust the weights again. With even a handful of segments and modest discretization, the combinatorial action space can explode to thousands of configurations, yet we observe only tens of aggregate outcomes over the entire optimization horizon. This extreme mismatch between action-space size and data availability renders classical bandit algorithms, which require repeated per-arm sampling, fundamentally unsuitable.

This problem structure arises across diverse operational domains: in marketing mix optimization, firms allocate spend across channels and observe delayed conversion signals [18]; in automated bidding for digital advertising, platforms expose segment-level multipliers that steer budget towards audience subgroups [1, 7]; and in adaptive clinical trials, limited enrollment slots are distributed across treatment arms based on interim outcomes [23];

Despite the diversity of these applications, they share a common structure: segments draw from a unified budget, so an incorrect weight in one segment directly degrades the performance of others. The optimization problem is therefore coupled and combinatorial. Furthermore, practical constraints restrict how rapidly weights may change and limit feedback to aggregate, periodic observations and thus creating a high-noise, low-frequency decision environment.

While Multi-Armed Bandits [16] (MAB) and Reinforcement Learning [22] (RL) are common approaches for sequential decision-making, they face significant practical challenges in such settings. Decisionmakers operate under strict data constraints, receiving only aggregated, delayed feedback. Furthermore, operational policies require adjustments to be gradual and infrequent (typically on a weekly or bi-weekly cadence). To navigate this environment, we propose a segmentation-agnostic method based on Gaussian Process Multi-Armed Bandits [14, 20].

In this paper, we demonstrate the practical application of Gaussian Process Upper Confidence Bound (GP-UCB) [20] algorithms to optimize budget allocation under the strict constraints of real-world operating environments. We bridge the gap between theoretical bandit research and production-scale deployment by providing a simulation-to-deployment pipeline. Our key contributions are as follows:

• Environment Simulation: We develop a high-fidelity synthetic simulator that models the key complexities of blackbox allocation systems, including observation noise, reward drift, and reactive (adversarial) dynamics. This environment serves as a benchmark for testing agent robustness before live deployment.

• Offline-to-Online Evaluation Framework: We propose a methodology for kernel selection and hyperparameter optimization using synthetic regret curves.

• Live Production Validation (RCT): We report results from a large-scale Randomized Controlled Trial (RCT) in a live production environment. Our empirical results confirm the model's efficacy, delivering a statistically significant performance lift in the primary efficiency metric relative to the standard ε-greedy baseline.

## 2 Related Work

Gaussian Process bandits. The GP-UCP algorithm, introduced by Srinivas et al. [20], extends the classical UCB algorithm for finite arms [2] to continuous domains by leveraging the mean and covariance function of the Gaussian Process posterior. The GP model captures dependencies between arms that are close in the action space, and its posterior variance provides calibrated uncertainty estimates, together yielding a principled exploration-exploitation trade-off over the continuous action set. Following its introduction, the GP-UCB algorithm was further improved and extended to improve the theoretical regret bounds, incorporate contextual information and dynamic environments [3, 6, 13]. GP-based optimization algorithms have recently been applied to online marketing problems such as bid optimization [5] and budget allocation [11].

Bandit convex optimization. Bandit optimization over continuous rewards has a rich history of theoretical research going back to the work of Flaxman, Kalai and McMahan [8]. In particular, when assuming that the reward function is convex, there is an extensive stream of work on gradually improving the theoretical upper bound and computational complexity, [4, 9, 10, 21]. A comprehensive overview of these results can be found in [15]. When the number of data samples becomes large, recent algorithms for bandit convex optimization would be computationally appealing over GP-UCP. We explored the usage of Convex Bandits in our preliminary study. However, we were unable to find evidence that our reward landscapes are convex and that we did not have sufficient samples to obtain an edge in the computation. Moreover, we did not observe competitive results, and therefore we do not further discuss Bandit convex optimization in our work.

## 3 Methodology

Let $\lambda \in [ 0 , 1 ] ^ { d }$ represent the vector of allocation weights for d distinct segments. To simplify our search space, we constrain these weights to discretized values within a fixed range (e.g., [0.10, 1.0]) using fixed increments (e.g., 0.05). Due to operational limitations, these weights can only be adjusted periodically (e.g., once per week), after which we observe a single aggregated reward metric. Optionally, we can subject the weights to a step-size limit $\Delta _ { \mathrm { m a x } }$

Our objective is to maximize the efficiency of budget utilization across targeted segments. At each time step t, we select an action $A _ { t } \in [ 0 . 1 0 , 1 . 0 ] ^ { d }$ and observe a noisy reward $Y _ { t : \cdot }$ formalized as:

$$
Y _ { t } = f ( A _ { t } ) + \varepsilon _ { t }\tag{1}
$$

where f represents the true, unobserved reward landscape and $\varepsilon _ { t }$ denotes observational noise.

The main challenge of this setup is the extreme data sparsity. We may receive only a single aggregated reward observation (summed across all segments) per week, making it impossible to attribute performance to individual segment weights. Furthermore, even when the continuous treatments are simplified to discrete actions in 0.05 increments, this creates a vast combinatorial action space. For example, an action space between 0.10 and 1.0 with 0.05 increments, and just 3 segments results into 6, 859 unique actions, taking 131 years to try each action at a once-per-week cadence.

This implies that if we treat each weight configuration (arm) as strictly independent, the problem becomes unsolvable within a realistic time horizon. Standard multi-armed bandit (MAB) algorithms, such as Upper Confidence Bound (UCB) and Thompson Sampling (TS), rely heavily on repeated sampling to build confidence bounds for individual arms. Consequently, UCB and TS algorithms are forced into a state of near-perpetual exploration, performing substantially worse in terms of cumulative regret than simple e-greedy strategies.

To overcome the limitations of independent arm evaluation, we must exploit the underlying structure of the reward function. It is reasonable to assume that the reward landscape is continuous and that similar multiplier allocations yield similar returns.

## 3.1 Gaussian Process Bandits

One method of exploiting this structural dependency is by using Gaussian Process (GP) Bandits. In this setup, we place a Gaussian Process prior on the reward function, $f \sim { \mathcal { G P } } ( \mu , k )$ , where $\mu$ is the mean function and k is the covariance kernel. This allows the algorithm to infer the expected reward of unexplored multiplier configurations by leveraging the observed rewards of neighboring actions, effectively sharing context across the domain. The most common approach for this is the GP Upper Confidence Bound (GP-UCB) [20] algorithm. At each step, the GP-UCB algorithm balances exploitation and exploration by selecting the arm that maximizes the upper confidence bound of the predictive posterior distribution:

$$
A _ { t } = \underset { a \in [ 0 . 1 0 , 1 . 0 ] ^ { d } } { \arg \operatorname* { m a x } } \mu _ { t - 1 } ( a ) + \beta _ { t } \sigma _ { t - 1 } ( a ) .\tag{2}
$$

Here, $\mu _ { t - 1 }$ and $\sigma _ { t - 1 }$ are the posterior mean and standard deviation of the posterior of all data up to previous round. The parameter $\beta _ { t }$ is an exploration parameter, which ensures that enough exploration occurs even after many steps.

3.1.1 Kernel Selection. The efficiency of the GP-UCB algorithm is highly dependent on the choice of the kernel function $k ,$ which encodes our prior assumptions about the smoothness and shape of f [12]. We initially consider several independent kernel choices:

• Radial Basis Function (RBF) [19]: Utilized for smooth, infinitely differentiable reward approximations.

• Matérn Kernel [17]: Evaluated with varying ν parameters to accommodate rougher, less predictable underlying reward landscapes.

• Linear Kernel: Employed independently to model straightforward, monotonic trends across the reward space.

Additionally, the mechanics of the platform suggest inherent monotonicity; increasing the budget allocation for a segment generally reduces its marginal ROI due to audience saturation and diminishing returns from reaching less responsive user segments. To explicitly enforce this structural assumption while maintaining the expressive power of non-linear models, we evaluate composite kernels by adding a linear component to the base kernels (i.e., kRBF $+ k _ { \mathrm { L i n e a r } }$ and $k _ { \mathrm { M a t } \acute { \mathrm { e r n } } } + k _ { \mathrm { L i n e a r } } )$ . This additive structure effectively regularizes the predicted landscape and improves algorithmic convergence. The algorithm's behavior is further tuned via the length scale parameter, which dictates spatial correlation.

## 4 Experimental Setup

We evaluate the proposed GP-UCB framework in a simulator that aims to capture the key dynamics of black-box budget allocation optimization. The simulator lets us (i) control the reward landscape, (ii) introduce non-stationarity and adversarial effects, and (iii) measure regret against an oracle. This level of counterfactual analysis is impossible with real-world A/B tests alone.

## 4.1 Simulated Allocation Environment

Proportional allocation. We model a decision-maker that distributes a fixed periodic budget X across d segments. Each segment i is assigned an allocation weight $\lambda _ { i }$ , and the system allocates spend proportionally:

$$
X _ { i } \ = \ X \cdot \ { \frac { \lambda _ { i } } { \sum _ { k = 1 } ^ { d } \lambda _ { k } } } .\tag{3}
$$

This represents a common allocation mechanism where weights steer resources towards segments with higher expected returns.

Reward landscape. For each segment i we construct a smooth, non-monotone reward curve $r _ { i } ( \lambda _ { i } )$ using a sum of Gaussian inflection points. Concretely, let $\{ ( x _ { j } , s _ { j } , w _ { j } ) \} _ { j = \mathrm { : } } ^ { J _ { i } }$ be the set of inflectionpoint parameters (location, steepness, width) for segment i. We first form the derivative

$$
r _ { i } ^ { \prime } ( \lambda ) \ = \ \sum _ { j = 1 } ^ { J _ { i } } - s _ { j } \ \exp \Bigl ( - \frac { ( \lambda - x _ { j } ) ^ { 2 } } { 2 w _ { j } ^ { 2 } } \Bigr ) ,\tag{4}
$$

then obtain $r _ { i }$ by numerical integration (cumulative trapezoidal rule) and min-max scaling to a target range $[ y _ { \mathrm { m i n } } , y _ { \mathrm { m a x } } ]$ . The number of inflection points per segment is sampled uniformly from $\{ 5 , \ldots , 9 \}$ steepness values are drawn from $\mathcal { U } ( 0 , 1 )$ and widths from $\mathcal { U } ( 0 , 0 . 1 )$

![](images/400195e255a2f1022ec8dfe9782a6f908fcbafb2f8a311683cdbd9bb3126dfab.jpg)

![](images/e312e73a779ed1676ad78570d8c37f14d2844d98963127f1b4144b4ff3a07de3.jpg)

![](images/18da1ec7aa7288df7ce90470cbbcc945cf9ca36a0ffdc955ef994029f8d313f2.jpg)

![](images/78a2d8395e90a008e4381af76befb98d0e60e34484689fb5d908b42668dbef97.jpg)  
Figure 1: Reward landscapes are constructed using gaussian inflection points as shown (a) and (b). These inflection points represent the derivatives of the reward landscape and result into a non-linear function, which could be either monotonically increasing/decreasing (c) or cause pockets of incrementality where additional allocation results in incremental rewards.

This construction produces landscapes with a variety of slopes, including occasional regions of diminishing returns, reflecting the empirical observation that black-box allocation systems can leave exploitable pockets of incrementality as shown in Figure 1.

In Figure 2, we highlight how multiple segments collectively form a ground truth reward landscape with multiple local optima. Even if there are no pockets of incrementality in the individual segments, the reallocation of budget can result into increased efficiency of the overall return on investment.

Action space and constraints. Multipliers are discretised to $\lambda _ { i } \in$ $\{ 0 . 1 0 , 0 . 1 5 , \ldots , 1 . 0 0 \}$ (19 values per segment). Two operational constraints restrict the agent at each step:

(1) Step-size limit. $\Delta _ { \mathrm { m a x } } = 0 . 1 0$ (at most two discrete increments per segment per step), preventing drastic week-toweek changes that could destabilise system performance.

(2) Anchor constraint. At least one segment must remain at $\lambda _ { i } = 1 . 0 .$ This prevents configurations that are scalar multiples of each other from appearing as distinct arms (e.g., [0.1, 0.1, 0.1] and [0.2, 0.2, 0.2] yield identical allocations under Equation 3).

These constraints mirror real-world operational policies and reduce the effective action space considerably.

Noise. Observed rewards are corrupted by additive Gaussian noise, $\tilde { r } _ { i } = r _ { i } + \varepsilon$ with $\varepsilon \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ . The noise scale σ is sampled uniformly from $[ 0 . 2 , 0 . 5 ]$ per experiment, simulating the inherent measurement uncertainty in weekly outcome metrics.

## 4.2 Environment Variants

A key question in practice is whether a kernel that performs well under stationary conditions retains its advantage when the reward landscape shifts over time, either organically or adversarially. We therefore evaluate four environment types of increasing difficulty.

Base (noiseless). The reward landscape is fixed and observations are noise-free $( \sigma = 0 ) .$ This idealised setting isolates the pure benefit of GP-based exploration over a stationary, deterministic reward surface and serves as a best-case reference.

Stationary with noise (NOISE). The reward landscape is fixed for the entire horizon; the only source of uncertainty is observation noise. This serves as the base case.

Non-stationary (DRIFT). Each inflection point undergoes a random walk: every Tshift steps, its parameters are perturbed as

$$
x _ { j } \gets x _ { j } + \eta _ { x } , \qquad s _ { j } \gets s _ { j } + \eta _ { s } , \qquad w _ { j } \gets w _ { j } + \eta _ { w } ,\tag{5}
$$

where $\eta . \sim N ( 0 , \sigma _ { \mathrm { s h i f t } } ^ { 2 } )$ . We set $T _ { \mathrm { s h i f t } } = 1 0$ steps and $\sigma _ { \mathrm { s h i f t } } = 0 . 1$ yielding gradual landscape drift that invalidates earlier observations.

Reactive / adversarial (REACTIVE). In practice, allocation environments are rarely fully stationary: competing advertisers may react to observed market shifts, and platform dynamics can create feedback loops that partially penalize predictable policies. Rather than modelling a fully adversarial game (which would require minimax or game-theoretic methods), we introduce a mildly reactive environment as a robustness stress-test. The goal is not to find the optimal policy under adversarial conditions, but to assess whether a kernel's advantage degrades gracefully when the environment partially counteracts the agent's strategy. At each step the environment:

(1) pre-samples K=1,000 candidate adversarial landscapes;

(2) predicts the agent's next action $\hat { \lambda } _ { t + 1 }$ (using the agent's most recent choice as a proxy);

(3) selects the candidate whose worst-performing point is closest to $\hat { \lambda } _ { t + 1 } ;$

(4) blends the selected adversarial landscape with the base landscape via a mixture parameter α:

$$
r _ { i } ^ { ( \mathrm { e f f } ) } ( \lambda ) ~ = ~ ( 1 - \alpha ) r _ { i } ^ { ( \mathrm { b a s e } ) } ( \lambda ) ~ + ~ \alpha r _ { i } ^ { ( \mathrm { a d v } ) } ( \lambda ) ,\tag{6}
$$

The mixture parameter $\alpha = 0 . 2$ ensures that 80% of the base landscape is preserved, modelling a setting where competitive reactions (e.g. shifting competitor pricing or competing agents) create headwinds but do not fully dictate the reward. This represents a realistic middle ground: environments are rarely purely stationary, but also rarely fully adversarial.

## 4.3 Algorithms

GP-UCB variants. We instantiate the GP-UCB algorithm (Section 3) with five kernel configurations:

• RBF: squared-exponential kernel with length scale $\ell = 1 . 5 ;$

• Matérn : Matérn kernel with $\ell = 1 . 2 , \nu = 1 . 5 ;$

• Linear: dot-product kernel $( \sigma _ { 0 } = 0 ) ;$

• RBF + Linear : sum of RBF (l=1.5) and dot-product $( \sigma _ { 0 } = 1 ) ;$

• Matérn + Linear : sum of Matérn $( \ell { = } 1 . 2 , \nu { = } 1 . 5 )$ and dotproduct $\scriptstyle ( \sigma _ { 0 } = 1 )$

Composite (sum) kernels allow the GP to capture both smooth local structure and a global linear trend simultaneously. All variants use $\delta = 0 . 1$ in the exploration parameter $\beta _ { t } = 1 + \sqrt { 2 ( \gamma _ { t } + 1 + \ln ( 1 / \delta ) ) }$ with $\begin{array} { r } { \gamma _ { t } = \frac { 1 } { 2 } ( 1 + \ln t ) ^ { 2 } } \end{array}$ as an upper bound on the maximum information gain.

ε-greedy baseline. We compare against an ε-greedy policy with a decreasing exploration rate:

$$
\varepsilon _ { t } \ = \ \operatorname* { m i n } \bigr ( 1 , \ t ^ { - 1 / 3 } \left( \ln t \right) ^ { 1 / 3 } \bigr ) ,\tag{7}
$$

which ensures sub-linear regret in multi-armed bandit settings. At each step the agent either explores uniformly at random or exploits the arm with the highest observed mean reward.

Reference baselines. Two additional baselines bracket performance:

• Constant: holds $\lambda _ { t } = \left( 1 , 1 , \ldots , 1 \right)$ throughout, representing a decision-maker that does not optimize weights at all.

• Optimal: an oracle that knows the true reward landscape and follows the shortest path to the global optimum under the step-size constraint (stationary setting) or greedily tracks the per-step optimum (non-stationary and reactive settings).

Why UCB and Thompson Sampling are omitted. Classical multiarmed bandit algorithms such as UCB and Thompson Sampling treat each arm independently. In our setting, the combinatorial action space contains $\vert \mathcal { A } \vert = 6 , 8 5 9$ feasible arms (after applying the anchor constraint), yet the decision horizon is only $T = 5 2$ weeks. Standard bandit methods would therefore spend the entire horizon exploring and thus incur near-maximal regret. GP-UCB circumvents this by generalising across arms via the kernel, requiring far fewer observations to identify high-reward regions.

## 4.4 Evaluation Methodology

Regret. We measure performance via cumulative regret:

$$
R ( T ) ~ = ~ \sum _ { t = 1 } ^ { T } \bigl [ r ^ { t } ( \lambda _ { t } ^ { * } ) - r ^ { t } ( \lambda _ { t } ) \bigr ] ,\tag{8}
$$

where $\lambda _ { t } ^ { * }$ is the (possibly time-varying) optimal arm of the rewards and $r ^ { t } ( \cdot )$ denotes the expected (noise-free) reward at round t. To highlight relative improvement over a naïve strategy, we also report difference regret: $R _ { \mathrm { m o d e l } } ( T ) - R _ { \mathrm { C o n s t a n t } } ( T )$ , which is negative when a model outperforms the Constant baseline.

Protocol. Each configuration is repeated over N = 100 independent experiments, each with a freshly sampled reward landscape and noise realisation. The horizon is $T = 5 2$ steps (one year of weekly decisions) with $d \ : = \ : 3$ segments. We report means and standard errors $( { \mathrm { s t d } } / { \sqrt { N } } )$ across experiments. The focus on nonstationarity with noise, adversarial, and reactive environments allows us to assess the generalisability of each kernel across fundamentally different forms of distributional shift. To the best of our knowledge, this comparison has not been systematically explored in the Gaussian process bandit optimization literature.

## 5 Results

## 5.1 Offline Synthetic Results

We compare all models from Section 4.3 across the four environments of Section 4.2, aggregating N=100 experiments per configuration (T=52 steps each). Shaded bands denote ±1 SEM. Significance is assessed via one-sided paired t-tests against ε-greedy. Figure 3 shows cumulative regret curves; Table 1 summarises final regret and the first step at which each GP-UCB variant significantly outperforms ε-greedy.

Pure kernels are brittle. In the noiseless base environment, pure Matérn achieves the lowest regret of all models (10.6, a 65% reduction vs. ε-greedy; significant from step 4), followed by RBF (19.1, 37%). However, this advantage vanishes entirely once any form of environment complexity is introduced. Under observation noise, drift, and adversarial reactions alike, pure RBF and Matérn regrets are statistically indistinguishable from ε-greedy. However, under mild adversarial pressure, pure kernels are significantly worse than ε-greedy $( \mathrm { p } < 0 . 0 0 1 )$ , as the reactive environment exploits the $\mathrm { G P } ^ { \prime } { \bf s }$ tendency to concentrate predictions in a narrow region. Composite kernels, by contrast, maintain their advantage, confirming their robustness beyond stationary conditions.

Composite kernels are robust. Augmenting either kernel with a linear component restores a significant advantage across all four environments. The linear term captures a coarse global trend in the weight space, anchoring exploration even when local GP estimates are degraded by noise or distributional shift. Both RBF + Linear and Matérn + Linear achieve 15-20% regret reductions over ε-greedy in every challenging setting (all $p < 1 0 ^ { - 3 } \mathrm { a t } T = 5 2 )$

Matérn + Linear is the most consistent performer. Table 1 shows that Matérn + Linear achieves significance earliest across environments (steps 4, 24, 19, 23) and delivers the largest or near-largest improvement in every setting. RBF + Linear, by contrast, requires up to step 38 under adversarial conditions and loses significance in the base case by $T { = } 5 2 .$ The finite smoothness of the Matérn component (v=1.5) avoids the oversmoothing of RBF, while the linear component provides a stable global trend estimate that regularizes predictions under noisy observations. Together, the composite kernel balances local adaptability with global structure. Based on these results, we select Matérn + Linear as the candidate kernel for the real-world experiment in Section 5.2.

![](images/f3718b26a54f8d68b9908b8a5c722f4d0b3053df79a86d94b5a927744713dcef.jpg)  
Figure 2: When segments share a budget (proportional allocation), the combined reward surface can have local optima even if individual segment curves are monotonic, because shifting weight between segments creates interaction effects. The top row shows a set of segment without pockets of incrementality, and the bottom row shows landscapes with multiple segments. We have marked the optimal points for each segment and their combined landscape with a star. Our goal is to find these points.

Table 1: Final cumulative regret at T=52 (mean ± SEM) and first step with significant improvement over ε-greedy (p < 0.05, one-sided paired t-test). “—": not significant; “↓": significantly worse.
<table><tr><td></td><td>BASE</td><td>NOISE</td><td>DRIFT</td><td>REACT.</td></tr><tr><td>Constant</td><td>39.8±2.0</td><td>42.7±2.4</td><td>51.1±3.1</td><td>42.3±2.0</td></tr><tr><td rowspan="2">Optimal ε-greedy</td><td>6.8±0.4</td><td>7.4±0.5</td><td>7.8±0.7</td><td>9.9±0.7</td></tr><tr><td>30.2±1.8</td><td>40.4±2.2</td><td>50.3±3.1</td><td>38.8±1.9</td></tr><tr><td>RBF</td><td>19.1±1.6</td><td>42.0±2.3</td><td>50.2±3.1</td><td>41.9±2.0</td></tr><tr><td rowspan="2">Matérn</td><td>step 4</td><td></td><td></td><td>↓</td></tr><tr><td>10.6±0.8</td><td>41.3±2.3</td><td>50.8±3.0</td><td>41.5±2.0</td></tr><tr><td rowspan="2">RBF+Lin.</td><td>step 4</td><td></td><td></td><td>↓</td></tr><tr><td>29.0±0.9</td><td>33.6±1.6</td><td>40.4±1.9</td><td>32.9±1.1</td></tr><tr><td rowspan="2">Mat.+Lin.</td><td>step 4*</td><td>step 31</td><td>step 20</td><td>step 38</td></tr><tr><td>18.1±1.0</td><td>32.3±1.2</td><td>41.7±1.7</td><td>30.9±0.9</td></tr><tr><td></td><td>step 4</td><td>step 24</td><td>step 19</td><td>step 23</td></tr></table>

\*Significant at step 4 but not at T=52 (p = 0.24).

## 5.2 Real world results

Guided by the synthetic benchmarks in Section 5.1, we deployed the Matérn + Linear GP-UCB model into a live production environment to validate its performance under true non-stationary conditions. Operational environments are characterized by heterogeneous trends and latent seasonality across individual segments; thus, a robust kernel is essential for stable exploration.

Experimental Design. We implemented a Randomized Controlled Trial (RCT) with two variants:

• Control: Decisions governed by the ε-greedy multi-armed bandit (MAB) strategy described in Section 4.3.

• Treatment: Decisions governed by the GP-UCB model utilizing the Matérn + Linear kernel.

The trial duration was set to T=6 weeks. This window was determined by a priori power calculations to ensure sufficient sensitivity to detect a meaningful performance lift, while also aligning with the synthetic results in Section 5.1. To improve discriminative power, models were initialized using parameters optimized from about 52 weeks of historical operational data. The newly collected data points were not shared among control and treatment.

Performance Lift. The deployment results aligned with the synthetic findings¹, demonstrating GP-UCB's ability to navigate complex reward landscapes more efficiently than the ε-greedy baseline. Our primary efficiency metric showed a substantial and statistically significant improvement. The treatment group exhibited a positive performance lift across all tested segments at the 90% confidence level. Notably, the treatment group exhibited faster adaptation to short-term environmental fluctuations, likely facilitated by the Matérn component's ability to model local fluctuations without the rigid global smoothness of an RBF kernel.

Conclusion. The real-world results validate the robustness of the composite Matérn + Linear kernel. By anchoring local GP estimates with a global linear trend, the model maintained high performance despite the noise and drift inherent in live allocation environments.

![](images/c2a62fdfd5c89298d8e6c55ffdbd9c61ec004f53fcaeb8e58b51e560c8a1daa5.jpg)  
(a) Base (noiseless)

![](images/838419d417c6dcea44fe208c01309dfd063e7bd63c97496d554d19e76b9c7a10.jpg)  
(b) Stationary with noise

![](images/fcede470cbe8dab972cfc561ff76a80fcac0c81709c6ba9399064699a4096c40.jpg)  
(c) Non-stationary (drift)

![](images/bd8fd01428b5afe7d1978ac6ab8fe812c7cacf44dc590345a4d1abde8ac4ba57.jpg)  
(d) Reactive / adversarial  
Figure 3: Cumulative regret over T=52 weeks. The black diagonal represents the regret when the reward is consistently zero

This confirms that the selection criteria established in Section 5.1 translate effectively to improved operational outcomes.

## 6 Conclusion & future work

This study addressed the challenge of optimizing allocation performance in highly non-stationary landscapes by bridging the gap between controlled simulation and live production. By developing a robust simulation environment that replicates the noise, drift, and adversarial dynamics of real-world allocation environments, we identified the Matérn + Linear kernel as a superior alternative to traditional exploration strategies. Our simulation-first approach provided the necessary statistical confidence to deploy the model in a live Randomized Controlled Trial (RCT), where the GP-UCB framework delivered a statistically significant efficiency gain over the standard ε-greedy baseline.

Future Work. Building on the current deployment's success, our simulation results reveal the potential for additional significant performance improvements. Preliminary results within the simulator indicate that explicit modeling of temporal non-stationarity and the inclusion of contextual features (e.g., segment characteristics, environment-specific signals) can further reduce regret. However, the transition from a Multi-Armed Bandit (MAB) to a Contextual Bandit framework increases the parameter space exponentially, necessitating further research in the following areas:

• High-Dimensional Hyperparameter Optimization: We intend to leverage our simulator to conduct large-scale sensitivity analyses, identifying which contextual variables provide the highest information gain relative to the increased computational and statistical complexity.

• Dynamic Contextual Modeling: Future iterations will explore the use of spatio-temporal kernels designed to adapt to rapid distribution shifts, converting the promising results observed in our "Drift" and "Reactive" simulations into a scalable, real-world contextual experiment.

• Automated Kernel Selection: Investigating automated methods to adaptively weight the linear and non-linear components of the kernel as the environment's volatility changes over time.

## References

[1] Gagan Aggarwal, Ashwinkumar Badanidiyuru, and Aranyak Mehta. 2019. Autobidding with Constraints. In Proceedings of the 15th International Conference on Web and Internet Economics (WINE). 17–30.

[2] Peter Auer, Nicolò Cesa-Bianchi, and Paul Fischer. 2002. Finite-time Analysis of the Multiarmed Bandit Problem. Machine Learning 47, 2–3 (2002), 235–256.

[3] Ilija Bogunovic, Jonathan Scarlett, and Volkan Cevher. 2016. Time-Varying Gaussian Process Bandit Optimization. In International Conference on Artificial Intelligence and Statistics (Proceedings of Machine Learning Research), Arthur Gretton and Christian C. Robert (Eds.). PMLR.

[4] Sébastien Bubeck, Ronen Eldan, and Yin Tat Lee. 2021. Kernel-based methods for bandit convex optimization. Journal of the ACM (JACM) (2021).

[5] Matteo Castiglioni, Alessandro Nuara, Giulia Romano, Giorgio Spadaro, Francesco Trovò, and Nicola Gatti. 2025. Safe Online Bid Optimization with Return on Investment and Budget Constraints. In ACM SIGKDD Conference on Knowledge Discovery and Data Mining. Association for Computing Machinery.

[6] Sayak Ray Chowdhury and Aditya Gopalan. 2017. On Kernelized Multi-armed Bandits. In Proceedings of the 34th International Conference on Machine Learning (ICML). 844–853.

[7] Yuan Deng, Jieming Mao, Vahab Mirrokni, and Song Zuo. 2021. Towards Efficient Auctions in an Auto-bidding World. In Proceedings of the Web Conference (WWW). 3965-3973.

[8] Abraham D. Flaxman, Adam Tauman Kalai, and H. Brendan McMahan. 2005. Online convex optimization in the bandit setting: gradient descent without a gradient. In ACM-SIAM Symposium on Discrete Algorithms. Society for Industrial and Applied Mathematics.

[9] Hidde Fokkema, Dirk van der Hoeven, Tor Lattimore, and Jack J Mayo. 2024. Online Newton method for bandit convex optimisation. arXiv preprint

arXiv:2406.06506 (2024).

[10] Elad Hazan and Kfir Levy. 2014. Bandit convex optimization: Towards tight bounds. In Advances in Neural Information Processing Systems.

[11] Mingjie Hu, Jie Xu, Chun-Hung Chen, and Jian-Qiang Hu. 2025. Optimal computation budget allocation with Gaussian process regression. European Fournal of Operational Research (2025).

[12] Motonobu Kanagawa, Philipp Hennig, Dino Sejdinovic, and Bharath K Sriperumbudur. 2018. Gaussian processes and kernel methods: A review on connections and equivalences. arXiv preprint arXiv:1807.02582 (2018).

[13] Andreas Krause and Cheng Ong. 2011. Contextual Gaussian Process Bandit Optimization. In Advances in Neural Information Processing Systems.

[14] Andreas Krause and Cheng Soon Ong. 2011. Contextual Gaussian Process Bandit Optimization. In Advances in Neural Information Processing Systems 24 (NeurIPS). 2447-2455.

[15] Tor Lattimore. 2026. Bandit Convex Optimisation. Cambridge University Press.

[16] Tor Lattimore and Csaba Szepesvári. 2020. Bandit Algorithms. Cambridge University Press.

[17] Bertil Matérn. 1960. Spatial Variation. Meddelanden från Statens Skogsforskningsinstitut. Second edition: Springer Lecture Notes in Statistics, 1986.

[18] Prasad A. Naik and Kalyan Raman. 2003. Understanding the Impact of Synergy in Multimedia Communications. Journal of Marketing Research 40, 4 (2003), 375-388.

[19] Carl Edward Rasmussen and Christopher K. I. Williams. 2006. Gaussian Processes for Machine Learning. MIT Press, Cambridge, MA.

[20] Niranjan Srinivas, Andreas Krause, Sham M. Kakade, and Matthias Seeger. 2010. Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design. In Proceedings of the 27th International Conference on Machine Learning (ICML). 1015–1022.

[21] Arun Suggala, Y Jennifer Sun, Praneeth Netrapalli, and Elad Hazan. 2024. Second Order Methods for Bandit Optimization and Control. In Proceedings of Thirty Seventh Conference on Learning Theory (Proceedings of Machine Learning Research). PMLR.

[22] Richard S Sutton, Andrew G Barto, et al. 1998. Reinforcement learning: An introduction. Vol. 1. MIT press Cambridge.

[23] Sofía S. Villar, Jack Bowden, and James Wason. 2015. Multi-armed Bandit Models for the Optimal Design of Clinical Trials: Benefits and Challenges. Statist. Sci. 30, 2 (2015), 199–215.