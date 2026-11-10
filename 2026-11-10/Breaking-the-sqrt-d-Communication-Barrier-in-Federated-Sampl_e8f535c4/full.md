# Breaking the $\sqrt { d }$ Communication Barrier in Federated Sampling with Adaptive Hamiltonian Monte Carlo

Jiajun Liang Independent Researcher liangjiajun0529@gmail.com

Guang Lin Purdue University guanglin@purdue.edu

Linxuan Wang Purdue University wang6559@purdue.edu

Qifan Song Purdue University qfsong@purdue.edu

## Abstract

This work proposes the Adaptive Federated Hamiltonian Monte Carlo (AFHMC) for Bayesian federated learning (FL) tasks. The prior work on HMC applications for FL achieves $O ( \sqrt { d } / \epsilon )$ communication cost under log-convex distributions, already demonstrating its advantage over the Langevin dynamic-based counterpart. Based on a refined analysis, this paper further reveals the benefit of HMC for the FL regime by incorporating a control mechanism for local node shifts. We prove that AFHMC at most requires $O ( \{ d / \epsilon ^ { 2 } \} ^ { 1 / 3 } )$ communication cost up to a logarithmic term, which is significantly better than the existing results. Moreover, the proposed method adapts to client heterogeneity. Under low-heterogeneity settings (i.e., the target sampling precision $\epsilon ^ { 2 }$ is higher than the heterogeneity level), the communication rate of AFHMC can be as good as $O ( \log ( d / \epsilon ^ { 2 } ) )$ ), resembling the communication costs for FL optimization tools. To the best of our knowledge, this provides the first logarithmic communication complexity result for federated sampling, even though it requires low heterogeneity. Numerically, we also justify AFHMC’s advantage under the low-heterogeneity regime. Our code is available at https://github.com/laaaarrywang/Adaptive-Federated-HMC.

## 1 Introduction

Federated learning (FL) provides a collaborative training paradigm in which multiple clients jointly learn a global model without sharing their raw data, thereby preserving data privacy while leveraging distributed data resources (McMahan et al., 2017; Bonawitz et al., 2019; Li et al., 2020a). A typical FL algorithm involves iterative rounds of local training followed by aggregating updates via node-tocenter communications. When FL is applied to real-time systems, e.g., low-earth orbit (LEO) satellite constellations (Chen et al., 2022), it is critical that the FL algorithm design aligns well with the system constraints, such as limited connection bandwidth/windows and high communication latency. Therefore, a successful FL framework must be communication-efficient and tolerate low aggregation frequency. Despite its empirical success in large-scale applications, most existing FL algorithms (e.g., Konecnˇ y et al., 2015, 2016; Li et al., 2020b; Wang et al., 2020), are designed within a frequentist\` framework and focus on point estimation, which makes it difficult to quantify uncertainty in model predictions and parameters. Bayesian federated learning (BFL) addresses this limitation by inferring the full posterior, thus enabling principled uncertainty quantification and robust decision-making under data heterogeneity and limited local samples. However, scalable Bayesian FL inference remains challenging due to multiple factors: (i) high-dimensional parameter spaces, (ii) heterogeneous and potentially non-IID client data, and (iii) strict communication and privacy constraints.

Hamiltonian Monte Carlo (HMC, Duane et al. (1987); Neal (2012)) is a MCMC method for sampling from complex posteriors, which exploits gradient information and an auxiliary momentum variable to construct proposals that can efficiently explore high-dimensional distributions. Compared with first-order methods such as stochastic gradient Langevin dynamics (SGLD, Welling & Teh (2011); Sato & Nakagawa (2014)), HMC and its stochastic variants often achieve superior mixing and lower asymptotic variance but its second-order nature also makes theoretical analysis and distributed implementation more involved. Extending HMC to the federated setting is nontrivial, because naive combinations of stochastic HMC with FedAvg-style aggregation (McMahan et al., 2017) can easily suffer from bias accumulation, instability of global dynamics, and inefficient communication.

To address these challenges, recent work proposes the Federated Averaging stochastic Hamiltonian Monte Carlo (FA-HMC, Liang et al. (2025)) algorithm, which integrates stochastic HMC updates on local clients with periodic aggregation of both positions and momenta at a central server. Liang et al. (2025) showed that FA-HMC requires $O ( \sqrt { d } / \epsilon )$ communication cost (equivalently, $O ( \sqrt { d } / \epsilon )$ updates occur in the center node) to achieve ϵ Wasserstein sampling error. In contrast, under non-distributed settings, HMC achieves a tight $O ( \{ d / \epsilon ^ { 2 } \} ^ { 1 / 4 } )$ gradient computation cost (Mangoubi & Vishnoi, 2018). It naturally leads to the question: can we further close the gap between the communication cost in FL paradigm and the computation cost in the non-FL paradigm, in terms of the dimension dependency?

This work introduces an adaptive variant, Adaptive Federated Hamiltonian Monte Carlo (AFHMC), which incorporates surrogate leapfrog steps and an additional global shift vector to better align local and global dynamics. We conduct refine analysis which explicitly characterizes the discrepancy between surrogate and exact leapfrog dynamics, and establish non-asymptotic convergence guarantees under standard smoothness and strong convexity assumptions. AFHMC can achieve improved dependence on the number of clients, communication rounds, and stochastic gradient noise level, and our experiments demonstrate competitive or superior performance compared with existing Bayesian federated baselines.

## 2 Related Works and Our Contributions

Federated optimization Early studies, such as FedAvg (McMahan et al., 2017), established convergence guarantees for smooth and strongly convex objectives, typically achieving sublinear rates in the presence of data heterogeneity. To address this challenge, subsequent work introduced variance reduction and control variate techniques, such as SCAFFOLD (Karimireddy et al., 2020) and its variants, which improve convergence by mitigating client drift and reducing bias induced by heterogeneous local updates. In particular, Mitra et al. (2021) provided the first result establishing linear convergence rates in terms of communication complexity using control variates, while Mishchenko et al. (2022) further advanced this line of research by proposing randomized algorithms that achieve constant communication complexity under suitable conditions.

Bayesian federated learning. Bayesian extensions of federated learning aim to characterize posterior uncertainty over model parameters or predictions while respecting client-level privacy and communication constraints. Existing approaches include variational Bayes formulations (e.g., Zhang et al., 2022; Guo et al., 2023), hierarchical Bayesian models for heterogeneous clients (e.g., Vedadi et al., 2024; Kim & Hospedales, 2025), and Bayesian variants of federated averaging where each client maintains local posterior approximations that are periodically aggregated (e.g., Al-Shedivat et al., 2021; Chen & Chao, 2021; Liu et al., 2024). These methods often rely on factorized approximations or conjugate priors to keep the communication and computation overhead manageable.Our work instead targets exact (or asymptotically exact) MCMC-based Bayesian inference in federated environments, and develops HMC-based algorithms that can operate under realistic communication budgets.

HMC and stochastic variants. HMC is an efficient sampler for complex high-dimensional Bayesian inference, with rigorous guarantees on convergence and mixing under smoothness and curvature conditions of the target distribution (Durmus et al., 2020; Mangoubi & Smith, 2017, 2021). To scale HMC to large datasets, stochastic gradient HMC (SGHMC Chen et al. (2014)) utilizes mini-batch estimates, typically introducing extra friction or correction terms to control the additional stochastic noise. Subsequent works have investigated the convergence behavior of SGHMC (Zou & Gu, 2021), variance-reduction techniques (Zou et al., 2019), low-precision sampling techniques (Wang et al., 2024) and etc. These developments, however, are primarily designed for centralized settings, and their direct application to federated learning is hindered by client heterogeneity and limited access to the full dataset.

Adaptive and geometry-aware HMC. A line of research has explored adaptive HMC schemes that adjust hyperparameters such as step sizes, mass matrices, and integrator structures using information gathered during sampling. Examples include Hoffman & Gelman (2014); Hoffman et al. (2021); Hirt et al. (2021); Fernández-Pendás et al. (2016); Girolami & Calderhead (2011), all aiming to enhance stability and sampling efficiency in complex geometries. More recently, adaptive tuning strategies have been proposed to automatically select splitting integrators and hyperparameter ranges for (generalized) HMC, reducing reliance on heuristic tuning in practice (Akhmatskaya et al., 2026; Nagar et al., 2024). Our AFHMC algorithm is conceptually related to these works in that it uses a surrogate Hamiltonian flow informed by global information, but it is tailored to the federated setting, where the adaptation must be communication-efficient and robust to client heterogeneity.

Beyond existing results, the main contributions of this paper are summarized as follows:

1. We propose AFHMC, an adaptive HMC algorithm for FL tasks based on surrogate leapfrog updates. The new algorithm introduces a global drift control variate to address the heterogeneity among local gradient update trajectories.

2. We introduce a principled framework to analyze divergence errors induced by approximate Hamiltonian dynamics.

3. Under strong convexity, we derive non-asymptotic error bounds that quantify the interplay between the surrogate leapfrog integrator and data heterogeneity. In particular, we prove that AFHMC achieves $O ( \{ d / \epsilon ^ { 2 } \} ^ { 1 / 3 } )$ communication cost under extreme heterogeneity, and $O ( \log ( d / \epsilon ^ { 2 } ) )$ communication cost under low heterogeneity. To the best of our knowledge, both rates of communication complexity have never been achieved by prior federated sampling algorithms. Not only do we obtain a better communication rate, but we also don’t rely on stringent Hessian smoothness conditions that are commonly required to achieve an advanced rate (e.g., Liang et al., 2025; Mangoubi & Vishnoi, 2018; Plassier et al., 2023). Please refer to Table 1 for a comprehensive comparison against prior theoretical works. Note that if we indeed enforce an extra Hessian smoothness condition, we could technically obtain even better results.

4. We empirically demonstrate that the proposed method achieves improved computational efficiency and stability in distributed and approximate-gradient settings.

## 3 Motivation and Algorithm

The goal of a standard sampling problem is to generate approximate samples from the target posterior $\bar { \pi ( \theta ) ( \propto \exp ( - f ( \theta ) ) }$ ) where f is the potential function. In the standard leapfrog HMC, as illustrated in Figure 1(left), the parameter θ is evolved using a randomly sampled Gaussian momentum for K leapfrog steps (defined later in (2)), after which the momentum is resampled before starting a new sequence of leapfrog updates. In Mangoubi & Vishnoi (2018), it is shown that performing ${ \bar { O } } ( \log d )$ such leapfrog trajectories, each consisting of $O ( \{ d / \epsilon ^ { 2 } \} ^ { 1 / 4 } )$ steps, suffices to ensure convergence of the task distribution. Therefore, the total gradient computational cost of the overall procedure is dominated by the number of leapfrog steps, leading to a total complexity of order $O ( d ^ { 1 / 4 } )$ ) up to logarithmic factors.

Let $[ N ] = \{ 1 , \dots , N \}$ index the clients. In federated settings, each client c has a local potential $f ^ { ( c ) } ( \theta )$ with weight $w _ { c } \geq 0$ and thus global potential $\begin{array} { r } { f ( \theta ) = \sum _ { c = 1 } ^ { N } w _ { c } f ^ { ( c ) } ( \theta ) } \end{array}$ , where $\textstyle \sum _ { c = 1 } ^ { N } w _ { c } = 1$ To achieve logarithmic communication complexity in federated sampling, the above theoretical insight motivates us to integrate communication steps into the alternating leapfrog procedure, expecting that the order of FL communication cost matches $O ( \log d )$ order of leapfrog trajectories in centralized HMC. To be concrete, as illustrated in Figure 1 (right), we aim that each local node performs K leapfrog steps followed by a communication round (e.g., averaging the local parameters across nodes), before proceeding to the next leapfrog phase. This corresponds to the FA-HMC algorithm (Liang et al., 2025) with $\bar { T } = 1$ . Unfortunately, Liang et al. (2025) establishes a lower bound showing that, under high federated heterogeneity, the communication cost of such an HMC scheme is at least on the order of $O ( \{ d / \epsilon ^ { 2 } \} ^ { 1 / 2 } )$ . Thus, additional non-trivial FL training designs are necessary to reach a better rate.

![](images/e6a2c4dc54e1690f81552399fe4f0d4fb647f4761a23464caba2d5e17559ce7c.jpg)  
Figure 1: An illustration of the centralized HMC algorithm (left panel) and the federated HMC algorithm (right panel).

In any FL optimization tasks, a key challenge to achieve efficient communication is to control the local shift caused by heterogeneity across local nodes. Local gradient updates move the local parameter toward the location optimum, resulting in a global bias after aggregating local solutions. A breakthrough was introduced by Karimireddy et al. (2020), which introduced a control shift technique that reduces the communication cost of FedAvg (McMahan et al., 2017) from $O ( d / \epsilon ^ { 2 } )$ to $O ( \log ( d / \epsilon ^ { 2 } ) )$ . In the FL sampling regime, FA-LD (Deng et al., 2024), a Langevin-counterpart of FedAvg also incurs a communication cost of $O ( d / \epsilon ^ { 2 } )$ . Plassier et al. (2023) further improved the sampling algorithm by adapting the control shift technique of (Karimireddy et al., 2020).

Inspired by existing works on local shift control, we propose an adaptive HMC federated update rule by introducing a drift variate. Given an energy function F, the HMC dynamics follow the Hamiltonian equations:

$$
d \theta ( t ) = p ( t ) d t , \qquad d p ( t ) = - \nabla _ { \theta } F ( \theta ( t ) ) d t .\tag{1}
$$

The above continuous-time dynamic cannot be solved analytically. Commonly, one uses the leapfrog integrator to discretize the dynamics over a step size η by

$$
\theta _ { k + 1 } = \theta _ { k } + \eta p _ { k } - \frac { \eta ^ { 2 } } { 2 } \nabla F ( \theta _ { k } ) , \qquad p _ { k + 1 } = p _ { k } - \frac { \eta } { 2 } \nabla F ( \theta _ { k } ) - \frac { \eta } { 2 } \nabla F ( \theta _ { k + 1 } ) .\tag{2}
$$

Under the FL setting, update algorithm (2) is running on local nodes, and $F$ represents a local energy function which can differ substantially from the global energy function f due to node heterogeneity. To account for this potential gap between local $\scriptstyle { \overline { { F } } }$ and global f, we inject a bias correction term to the gradient and replace all $\check { \nabla } \dot { F } ( \cdot )$ terms in (2) with $\begin{array} { r } { \dot { \nabla F } ( \cdot ) - \nabla F ( \dot { \theta _ { 0 } } ) + \nabla f ( \theta _ { 0 } ) } \end{array}$ (see, Algorithm 1). This bias correction term $\nabla f ( \dot { \theta } _ { 0 } ) - \nabla F ( \theta _ { 0 } )$ addresses the discrepancy between local and global losses, and the drift variate $\nabla f (  { \theta _ { 0 } } )$ needs to be broadcast to all local nodes. This novel local update rule is named the surrogate leapfrog update.

The AFHMC algorithm consists of the following iterative steps: 1. After synchronizing parameter $\theta _ { t }$ across all nodes, each node c shares the local $\nabla f ^ { ( c ) } ( \theta _ { t } )$ to obtain a global drift variate $\nabla f ( \theta _ { t } ) . 2$ All local nodes run $K$ surrogate leapfrog steps, with the shared global drift variate, initialized from the same position and momentum. 3. All nodes share and synchronize the local parameter $\theta _ { t + 1 }$ , after K leapfrog steps. The algorithm can be found in Algorithm 2. In contrast to FA-HMC (Liang et al., 2025), which runs multiple local HMC steps where each HMC step consists of K leapfrog updates, the proposed AFHMC only runs one single local HMC step (i.e., K updates of surrogate leapfrog, see Figure 1). As discussed previously, this design is to ensure minimal communication cost, because every local HMC step will need a fresh new drift control variate, which requires communication.

The AFHMC algorithm can be easily implemented with stochastic gradients for acceleration purposes, similarly to Liang et al. (2025), but our analysis in this paper focuses on the use of the exact gradient.

Algorithm 1 Surrogate leapfrog approximation $\widetilde { h } _ { \mathrm { L F } } ( \cdot )$   
Input: energy function $F ( \cdot )$ , initial position $\theta _ { 0 } .$ drift vector $\nabla f ( \theta _ { 0 } )$ , momentum $p _ { 0 }$ , stepsize $\eta .$   
leapfrog step $\overset { \cdot } { K }$   
for $k = 0$ to $K - 1$ do   
$\begin{array} { r } { \theta _ { k + 1 } = \theta _ { k } + \eta p _ { k } - \frac { \eta ^ { 2 } } { 2 } ( \nabla F ( \theta _ { k } ) - \nabla F ( \theta _ { 0 } ) + \nabla f ( \theta _ { 0 } ) ) } \end{array}$   
$\begin{array} { r } { p _ { k + 1 } = p _ { k } - \frac { \hat { \eta } } { 2 } \nabla F ( \dot { \theta _ { k } } ) - \frac { \eta } { 2 } \nabla \dot { F } ( \theta _ { k + 1 } ) + \eta ( \nabla F ( \dot { \theta } _ { 0 } ) - \nabla f ( \theta _ { 0 } ) ) } \end{array}$   
end for   
Output: $\widetilde { h } _ { \mathrm { L F } } ( F , \theta _ { 0 } , \nabla f ( \theta _ { 0 } ) , p _ { 0 } , \eta , K ) = \theta _ { K }$

```latex
Algorithm 2 Adaptive federated hamiltonian monte carlo (AFHMC) algorithm
Input: initial position $\theta _ { 0 } ,$ stepsize $\eta ;$ leapfrog update step $K$
Each local node receives the initial parameter $\theta _ { 0 }$ from the center
for $t = 0$ to $t _ { \mathrm { m a x } }$ do
1st broadcast: The c-th local node sends $\nabla f ^ { ( c ) } ( \theta _ { t } )$ to the center, for all $c = 1 , \ldots , N ;$
1st communicate: The center node samples Gaussian momentum $p ^ { t } ,$ , and sends $p _ { t }$ and drift
vector $\begin{array} { r } { \nabla f ( \theta _ { t } ) : = \sum _ { c = 1 } ^ { N } w _ { c } \nabla f ^ { ( c ) } ( \theta _ { t } ) } \end{array}$ to all local nodes;
for $c = 1 , 2 \ldots , N$ do
Initialize $\theta _ { t , 0 } ^ { ( c ) } = \theta _ { t } , p _ { t , 0 } ^ { ( c ) } = p _ { t } ;$
update $\begin{array} { r } { \theta _ { t , K } ^ { ( c ) } = \widetilde { h } _ { \mathrm { L F } } ( f ^ { ( c ) } , \theta _ { t , 0 } ^ { ( c ) } , \nabla f ( \theta _ { t } ) , p _ { t , 0 } ^ { ( c ) } , \eta , K ) ; } \end{array}$
end for
2nd broadcast: The c-th local node sends $\theta _ { t , K } ^ { ( c ) }$ to the center for all $c = 1 , \ldots , N$
2nd communicate: The center node sends back $\begin{array} { r } { \theta _ { t + 1 } : = \sum _ { c = 1 } ^ { N } w _ { c } \theta _ { t , K } ^ { ( c ) } } \end{array}$ to all local nodes;
end for
Output: $\{ \theta _ { t } : t = 0 , \ldots , t _ { \operatorname* { m a x } } \}$
```

## 4 Convergence Analysis

Let $z _ { i } ^ { c } , 1 \le i \le n _ { c } , 1 \le c \le N$ be the local data at the c-th node, $\theta \in \mathbb { R } ^ { d }$ and $\ell ( \theta ; z _ { i } ^ { c } )$ be the negative log-likelihood (or loss) function. Define $\begin{array} { r } { w _ { c } = n _ { c } / \sum _ { c = 1 } ^ { N } n _ { c } , } \end{array}$ , and $\textstyle f ^ { ( c ) } ( \theta ) : = \sum _ { i = 1 } ^ { n _ { c } } \ell ( \theta ; z _ { i } ^ { c } ) / w _ { c }$ is the local loss function of accessible to the c-th local node. The goal is to simulate the global target distribution $\pi ( \theta ) \propto \exp ( - f ( \theta ) )$ , where $\begin{array} { r } { f ( \theta ) = \sum _ { c = 1 } ^ { N } w _ { c } f ^ { ( c ) } ( \theta ) , w _ { c } \geq 0 } \end{array}$ with $\textstyle \sum _ { c } w _ { c } = 1$

This section will investigate the theoretical properties of AFHMC algorithm. Before stating our main results, we specific the model assumption as follows:

Assumption 4.1 (m-Strongly Convex). For each $c = 1 , 2 , \ldots , N , f ^ { ( c ) }$ is m-strongly convex for some $\begin{array} { r } { \mu > 0 , \mathbf { i . e . } , \forall x , y \in \mathbb { R } ^ { d } f ^ { ( c ) } ( y ) \geq f ^ { ( c ) } ( x ) + \langle \nabla f ^ { ( c ) } ( x ) , y - x \rangle + \frac { m } { 2 } \| y - x \| _ { 2 } ^ { 2 } . } \end{array}$

Assumption 4.2 (M-Smoothness). For each $c = 1 , 2 , \ldots , N , f ^ { ( c ) }$ is M-smooth for some $M > 0$ $\mathrm { i . e . , } \forall x , \bar { y } \in \mathbb { R } ^ { d }$ , we have $\| \nabla f ^ { ( c ) } ( y ) - \nabla f ^ { ( c ) } ( x ) \| \leq M \| x - y \|$

The assumptions are fairly standard in the theoretical literature of gradient-based MCMC algorithms (e.g., Dalalyan, 2017; Mangoubi & Vishnoi, 2018; Dalalyan & Karagulyan, 2019; Erdogdu & Hosseinzadeh, 2021). Unlike prior advanced results (e.g., Dalalyan & Karagulyan, 2019; Liang et al., 2025; Plassier et al., 2023; Mangoubi & Vishnoi, 2018), we don’t require any second-order smoothness (e.g., Hessian) of the energy functions.

Our first result provides a non-asymptotic bound, studying how the gap between $\theta _ { t }$ and $\theta _ { t } ^ { \pi }$ contracts, where $\theta _ { t } ^ { \pi }$ is a coupled stationary continuous HMC process satisfying $\theta _ { t } ^ { \pi } \sim \pi$ for all $t \geq 0$ (refer to Appendix Section A.1 for formal definitions).

Theorem 4.3 (Convergence). Under Assumptions $4 . l { - } 4 . 2 ,$ , there exist sufficient large constants $C _ { 0 } , C _ { 1 } , C _ { 2 }$ such that $\{ \bar { \theta } _ { t } \}$ <sub>t</sub> satisfies that, for $\bar { K } \eta \leq \sqrt { m } / ( C _ { 0 } M ) \leq \sqrt { m } / ( 2 \bar { M } )$ ,

$$
{  { \mathbb E } } \| \theta _ { t + 1 } - \theta _ { t + 1 } ^ { \pi } \| ^ { 2 } \le \left( 1 - \frac { m ( K \eta ) ^ { 2 } } { 1 6 } \right) {  { \mathbb E } } \| \theta _ { t } - \theta _ { t } ^ { \pi } \| ^ { 2 } + C _ { 1 } K ^ { 2 } \eta ^ { 4 } \frac { M ^ { 2 } } m d + C _ { 2 } ( K \eta ) ^ { 8 } \frac { M ^ { 2 } } m H
$$

where $\begin{array} { r } { H : = \operatorname* { m a x } _ { i \leq K } \sum _ { c = 1 } ^ { N } w _ { c } \mathbb { E } \| \frac { 1 } { i \eta } ( \nabla f ^ { ( c ) } ( \bar { \mathcal { H } } _ { \eta } ^ { i } ( \theta ^ { \pi } ) ) - \nabla f ^ { ( c ) } ( \theta ^ { \pi } ) + \nabla f ( \theta ^ { \pi } ) - \nabla f ( \bar { \mathcal { H } } _ { \eta } ^ { i } ( \theta ^ { \pi } ) ) \| ^ { 2 } } \end{array}$ is a metric quantifying the federated heterogeneity.

The $\bar { \mathcal { H } } _ { \eta } ^ { i } ( \theta )$ (formally defined in Appendix Section A.1) denotes the output by running iη-length ideal continuous HMC dynamic (i.e., (4) with $F$ being replaced by $f )$ initialized from $\theta .$

Remark 4.4. H is a metric only related to the target distribution and local distribution. It characterizes the client-level gradient heterogeneity under the target distribution. H is a theoretical quantity usually unknown in practice. But in terms of its dependency on the dimension, due to the nature of the squared $L _ { 2 }$ norm, one can prove that $H = O ( { \bar { d } } )$ . The proof of this bound and examples of calculating the exact H are given in Appendix Section C.

Sketch of proof We decompose $\lVert \theta _ { t + 1 } - \theta _ { t + 1 } ^ { \pi } \rVert ^ { 2 }$ into two terms: $\| \bar { \mathcal { H } } _ { \eta } ^ { K } ( \theta _ { t } ) - \theta _ { t + 1 } ^ { \pi } \| ^ { 2 }$ and $\lVert \theta _ { t + 1 } -$ $\bar { \mathcal { H } } _ { \eta } ^ { K } ( \theta _ { t } ) \lVert ^ { 2 }$ . The first term is subject to the classic contraction theorem of ideal HMC dynamics. The second term can be characterized by (i) the divergence between $\theta _ { t , k } ^ { ( c ) }$ and $\textstyle \sum _ { c = 1 } ^ { N } w _ { c } \theta _ { t , k } ^ { ( c ) }$ , and (ii) the divergence between ideal HMC dynamics $\bar { \mathcal { H } } _ { \eta } ^ { i } ( \theta )$ and the discretized HMC dynamics ${ \mathcal { H } } _ { n } ^ { i } ( \theta )$ . Thanks to the local shift control mechanism, we establish a new technical bound (Lemma B.6)) to study the surrogate error and give E $\begin{array} { r } { \lvert \lvert \theta _ { t , k } ^ { ( c ) } - \sum _ { c = 1 } ^ { N } w _ { c } \theta _ { t , k } \rvert \rvert ^ { 2 } \approx O ( ( K \eta ) ^ { 6 } H ) } \end{array}$ ) , and bound the divergence caused by discretizations by $O ( \eta ^ { 4 } d )$ (via Lemma B.10).

Based on the recursive contraction result in Theorem 4.3, one can easily derive the convergence rate for $\mathbb { E } \Vert \theta _ { t + 1 } - \theta _ { t + 1 } ^ { \pi } \Vert ^ { 2 }$ . Let $\pi _ { t }$ be the marginal distribution of $\theta _ { t }$ and $\mathcal { W } _ { 2 } ( \mu , \nu )$ denote the Wassertein-2 distance between probability measures $\mu$ and $\nu .$ . Then, by definition, $\mathcal { W } _ { 2 } ^ { 2 } ( \pi _ { t } , \pi ) =$ in $\dot { \mathbf { \zeta } } _ { \theta \sim \pi } \mathbb { E } \lVert \theta _ { t } - \theta \rVert ^ { 2 } \leq \mathbb { E } \lVert \theta _ { t } - \theta _ { t } ^ { \dot { \pi } } \rVert ^ { 2 }$ , leading to

$$
\mathcal { W } _ { 2 } ^ { 2 } ( \pi _ { t } , \pi ) \le \left( 1 - \frac { m ( K \eta ) ^ { 2 } } { 1 6 } \right) ^ { t } \mathcal { W } _ { 2 } ^ { 2 } ( \pi _ { 0 } , \pi ) + C _ { 1 } ^ { \prime } \eta ^ { 2 } \frac { M ^ { 2 } d } { m ^ { 2 } } + C _ { 2 } ^ { \prime } K ^ { 6 } \eta ^ { 6 } \frac { M ^ { 2 } H } { m ^ { 2 } } .
$$

This leads to our next theorem, which studies the communication cost rate for AFHMC.

Theorem 4.5. Under Assumptions $4 . l - 4 . 2$ and $i f H \gg M ^ { 4 } \epsilon ^ { 2 } / m > 0$ , then there exists some small absolute constant C such that ifwe choose

$$
K \eta = C \left( \frac { \epsilon ^ { 2 } m ^ { 2 } } { H M ^ { 2 } } \right) ^ { 1 / 6 } \ll C \frac { \sqrt { m } } { M } ~ a n d ~ \eta = C \left( \frac { \epsilon ^ { 2 } m ^ { 2 } } { M ^ { 2 } d } \right) ^ { 1 / 2 } ,
$$

$\mathcal { W } _ { 2 } ( \pi _ { t _ { \epsilon } } , \pi ) \leq \epsilon$ can be attained with communication times

$$
t _ { \epsilon } \asymp \Big ( \frac { H M ^ { 2 } } { \epsilon ^ { 2 } m ^ { 5 } } \Big ) ^ { 1 / 3 } \log ( \mathcal { W } _ { 2 } ^ { 2 } ( \pi _ { 0 } , \pi ) / \epsilon ^ { 2 } ) .
$$

The corresponding local iteration number is

$$
K \asymp \frac { M ^ { 2 / 3 } d ^ { 1 / 2 } } { \epsilon ^ { 2 / 3 } m ^ { 2 / 3 } H ^ { 1 / 6 } } .
$$

Moreover, $i f H \ll M ^ { 4 } \epsilon ^ { 2 } / m ( i . e .$ , heterogeneity across local nodes is very small), it only requires

$$
t _ { \epsilon } \asymp \log ( \mathcal { W } _ { 2 } ^ { 2 } ( \pi _ { 0 } , \pi ) / \epsilon ^ { 2 } ) ,
$$

achieving a linear rate with respect to the communication cost.

For a heterogeneous situation, the communication cost $t _ { \epsilon }$ implicitly depends on the model dimension via the quantity H. As discussed in Remark 4.4, H can scale linearly with $d ,$ leading to $t _ { \epsilon } \ \asymp$ $( d / \epsilon ^ { 2 } ) ^ { 1 / \bar { 3 } } \log ( \dot { d } / \epsilon ^ { 2 } )$ . For a homogeneous situation where $H \approx 0$ , the communication cost reduces to ${ \dot { O ( \log ( d / \epsilon ^ { 2 } ) ) } }$ . To the best of our knowledge, both rates have never been achieved by prior Bayesian FL methods in the literature. Moreover, our analysis do not require extra stringent Hessian smoothness conditions that are commonly required to achieve an advanced rate $\mathrm { e . g . }$ , Assumption A3 in Plassier et al. (2023), Assumption 3.3 in Liang et al. (2025). We compare all these results in Table 1.

This theorem can also be interpreted from a different perspective. AFHMC can have two training phases: in the first phase (fast convergence), the sampling error linearly decays until $\mathcal { W } _ { 2 } ^ { 2 } ( \pi _ { t } , \pi ) \sim$ $m H / M ^ { 4 }$ ; in the second phase (slow convergence), it needs $O ( ( d / \epsilon ^ { 2 } ) ^ { 1 / 3 } \log ( d / \epsilon ^ { 2 } ) )$ ) communications to reach ${ \mathcal W } _ { 2 } ^ { 2 } ( \pi _ { t } , \pi ) = \epsilon ^ { \hat { 2 } } = o ( m H / M ^ { 4 } )$ . Note that these two phases do require different tuning of $K$ and $\eta ,$ as shown by the proof of Theorem 4.5.

<table><tr><td rowspan=1 colspan=1>Algorithm</td><td rowspan=1 colspan=1>Condition beyondAssumptions 4.1-4.2</td><td rowspan=1 colspan=1>Rate of communication</td><td rowspan=1 colspan=1>Rate of gradients(up to logarithmic terms)</td></tr><tr><td rowspan=1 colspan=1>Centralized HMCMangoubi &amp; Vishnoi (2018)</td><td rowspan=1 colspan=1>Hessian Lipschitz under infinitynorm (Assum. 1 in their paper)</td><td rowspan=1 colspan=1>none</td><td rowspan=1 colspan=1> $( d / \epsilon ^ { 2 } ) ^ { 1 / 4 }$ </td></tr><tr><td rowspan=1 colspan=1>VR-FALDPlassier et al. (2023)</td><td rowspan=1 colspan=1>Hessian Lipschitz under L2norm (Assum. HX1 in their paper)</td><td rowspan=1 colspan=1> $( d / \epsilon ^ { 2 } ) ^ { 1 / 2 }$ </td><td rowspan=1 colspan=1> $( d / \epsilon ^ { 2 } ) ^ { 1 / 2 }$ </td></tr><tr><td rowspan=1 colspan=1>FA-HMCLiang et al. (2025)</td><td rowspan=1 colspan=1>Hessian Lipschitz under L2norm (Assum. 3 in their paper)</td><td rowspan=1 colspan=1> $( d / \epsilon ^ { 2 } ) ^ { 1 / 2 }$ </td><td rowspan=1 colspan=1> $( d / \epsilon ^ { 2 } ) ^ { 1 / 2 }$ </td></tr><tr><td rowspan=1 colspan=1>AFHMC (this paper)</td><td rowspan=1 colspan=1>none</td><td rowspan=1 colspan=1> $\log ( d / \epsilon ^ { 2 } ) \sim ( d / \epsilon ^ { 2 } ) ^ { 1 / 3 }$ </td><td rowspan=1 colspan=1> $( d / \epsilon ^ { 2 } ) ^ { 1 / 2 } \sim ( d / \epsilon ^ { 2 } ) ^ { 2 / 3 }$ </td></tr></table>

Table 1: Comparison of Bayesian theoretical results: requires assumptions, rate of communication and rate of gradient computation to achieve ϵ Wasserstein convergence

Remark 4.6. Our main theory is derived based on using the exact gradient updates. While it is of great interest to investigate the theoretical behavior when stochastic gradient is used instead, it poses great challenges to obtain a sharp rate that surpasses the Langevin counterpart. Such a result doesn’t exist so far, even for the centralized HMC algorithm. With a lack of theoretical guarantee, we still believe in the advantage of AFHMC under noisy gradient, which is empirically justified by our numerical studies.

## 5 Numerical Evidences

In this section, we present our numerical results, including both simulation studies and real data applications, to support our theoretical claims. We compare the empirical performance of AFHMC, FA-HMC and FA-LD, and demonstrate the superiority of the proposed AFHMC algorithm. Throughout this section, a communication round refers to one outer iteration of the federated algorithm. Each AFHMC round consists of two server exchanges (the gradient aggregation for the drift correction and the parameter aggregation after the surrogate trajectory; see the two broadcasts in Algorithm 2), whereas each FA-HMC or FA-LD round consists of one. This constant factor does not change the communication complexity rates. For an empirical comparison at an equal number of server exchanges, we compare AFHMC at round R with FA-HMC at round 2R.

Simulated Bayesian logistic regression. Following Liang et al. (2025); Mangoubi & Vishnoi (2018), we consider Bayesian logistic regression with a standard Gaussian prior on $\theta \in \mathbb { R } ^ { d }$ . Given feature vectors $x _ { i } \in \mathbb { R } ^ { d }$ and binary labels $y _ { i } \in \{ - 1 , + 1 \}$ , the posterior is

$$
\pi ( \theta ) \propto \exp \left( - \frac { 1 } { 2 } \| \theta \| ^ { 2 } \right) \prod _ { i = 1 } ^ { n } \sigma ( y _ { i } x _ { i } ^ { \top } \theta ) ,
$$

where $\sigma ( \cdot )$ denotes the sigmoid function. The $n = 1 0 0 0$ observations are split evenly across $N = 2 0$ clients. We evaluate the samplers in two settings: (1) a high-dimensional case with $d = 1 0 0 0$ and exact local gradients; (2) a lower-dimensional case with $d = 1 0$ with inaccurate gradients. Specifically, we add $\mathcal { N } ( \bar { 0 } , 1 0 0 I _ { d } )$ noise to the exact gradients, simulating the stochastic mini-batching.

The algorithms (FA-LD, FA-HMC, and AFHMC) are implemented with the following tuning parameter settings. FA-LD uses $T = 1 0$ local Langevin steps (each Langevin step is equivalent to $K = 1$ leapfrog step). FA-HMC uses $T = 1 0$ local HMC steps, and each HMC step consists of $K _ { F }$ classic leapfrog steps. We sweep $K _ { \mathrm { F } }$ over a grid centered around the trajectory-length rule $K _ { \mathrm { F } } \eta \approx \pi / 3$ (Mangoubi & Vishnoi, 2018). AFHMC uses $K _ { \mathrm { A } }$ local surrogate leapfrog steps, and we consider two candidates of $K _ { \mathrm { A } } \colon ( \mathrm { i } ) \ K _ { \mathrm { A } } = K _ { \mathrm { F } }$ such that AFHMC and FA-HMC match the HMC dynamic trajectory length, and (ii) $K _ { \mathrm { A } } = 1 0 K _ { \mathrm { F } }$ such that AFHMC and FA-HMC match the same number of leapfrog steps between consecutive rounds of communication. For different $\eta ,$ the $K _ { \mathrm { A } }$ and $K _ { \mathrm { F } }$ values that deliver the best performance are chosen. For each (algorithm, $\eta , K )$ configuration, we run $M = 2 0$ independent chains with a total leapfrog budget of $2 \times 1 0 ^ { 6 }$ and discard the first half of each chain as burn-in. We measure performance using the marginal error

$$
\mathrm { M E } \ = \ { \frac { 1 } { d } } \sum _ { i = 1 } ^ { d } { \mathcal W } _ { 1 } \big ( \hat { \mu } _ { i } , \mu _ { i } ^ { \mathrm { M H M C } } \big ) ,\tag{3}
$$

![](images/be5d8b64f467ae07d2bc330c4a17c815f71ec340864efe7cc30d91bb935025ec.jpg)

![](images/3c0348c622d3b67dcc3636a7cf420786e17af03baf528ce4e8e8736642cb9eba.jpg)  
Figure 2: Simulated Bayesian logistic regression. Left: $d = 1 0 0 0$ with exact gradients. Right: $d = 1 0$ with coordinate-wise N (0, 100) gradient noise. Each panel reports marginal error relative to a centralized MHMC reference chain under a fixed total leapfrog budget. AFHMC is more stable across stepsizes and improves the small-stepsize behavior of FA-HMC and FA-LD.

where $\hat { \mu } _ { i }$ is the empirical marginal of the i-th coordinate produced by the candidate sampler, $\mu _ { i } ^ { \mathrm { M H M C } }$ is the surrogate true posterior margins, approximated by a centralized Metropolis-adjusted HMC reference chain with $1 0 ^ { 7 }$ steps, and $\mathcal { W } _ { 1 }$ denotes the Wasserstein-1 distance. For each (algorithm, η) pair, we report the best-performing K from the candidate grid; the full grid is provided in Appendix D.1.

Figure 2 reports the resulting marginal error curves for the exact-gradient and noisy-gradient settings. In the exact-gradient high-dimensional setting, AFHMC maintains low marginal error over a broader range of small stepsizes: for example, at $\eta = \bar { 5 } \times 1 0 ^ { - 4 }$ , AFHMC attains $\mathrm { M E } = 2 . 3 \times 1 0 ^ { - 2 }$ , compared with $5 . 3 \times 1 0 ^ { - 2 }$ for FA-HMC and $3 . 9 \times 1 0 ^ { - 1 }$ for FA-LD. In the noisy-gradient setting, AFHMC achieves the best overall marginal error in the sweep, $\mathrm { M E } = 2 . 7 \times 1 0 ^ { - 3 }$ , and remains competitive with the best tuned FA-HMC and FA-LD configurations at the other stepsizes.

Dimension scaling on heterogeneous Gaussian targets. Following the heterogeneous Gaussian experiment of Liang et al. (2025), we study how the communication cost scales with dimension on a controlled federated Gaussian benchmark. There are $N = 1 0$ clients. For half of the clients the local target is $\mathcal { N } ( 2 0 \mathbf { 1 } _ { d } , I _ { d } )$ , and for the remaining clients it is $\mathcal { N } ( \mathbf { 1 } _ { d } , 4 I _ { d } )$ . Averaging the local quadratic potentials gives the global target $\mathcal { N } ( 1 6 . 2 \bar { \mathbf { 1 } _ { d } } , 1 . 6 I _ { d } )$ . We vary $d \in \{ 2 , 5 0 , 1 0 0 , 1 5 0 , \ldots , 5 0 0 \}$ and measure convergence by the squared 2-Wasserstein distance to this global Gaussian target. Note that the empirical 2-Wasserstein distance is expensive to compute for high-dimensional data, therefore, we approximate the posterior distribution by fitting a multivariate Gaussian and compute the $2 \AA ^ { - }$ Wasserstein distance between the fitted Gaussian distribution and the target distribution via the closed-form formula.

Suggested by the theoretical choice provided by Theorem 4.5, for AFHMC, the leapfrog count and stepsize are scaled with dimension as

$$
K ( d ) = \operatorname* { m a x } \left\{ 1 , \left\lfloor 5 0 ( d / 1 0 0 ) ^ { 1 / 3 } \right\rceil \right\} , \qquad \eta ( d ) = 0 . 0 0 6 ( d / 1 0 0 ) ^ { - 1 / 2 } ,
$$

so that $K ( 1 0 0 ) = 5 0 $ and $\eta ( 1 0 0 ) = 0 . 0 0 6$ . Thus, as d increases, AFHMC uses longer trajectories but smaller leapfrog steps. We run AFHMC until $\mathcal { W } _ { 2 } ^ { 2 } < 0 . 1$ for 20 consecutive checks and record the first communication round at which the threshold is crossed. The 2-Wassertein distance calculation is based on $M ( d ) = 1 0 0 d ( d - 1 )$ independent parallel chains (note that we need to estimate a d-dimensional covariance matrix).

As a matched-budget baseline, we run FA-HMC with the same $\eta ( d )$ and the same number of communication rounds as AFHMC. FA-HMC uses $T = 1 0$ local HMC update, and the leapfrog step count within each HMC update is chosen as

$$
K _ { \mathrm { F } } ( d ) = \operatorname* { m a x } \left\{ 1 , \lfloor K ( d ) / 1 0 \rceil \right\} ,
$$

so that the total number of leapfrog updates per communication round is matched to AFHMC up to rounding. To check that the gap below is not caused by giving FA-HMC the AFHMC stepsize, we additionally run FA-HMC for $2 R ( d )$ rounds, i.e., the same number of server exchanges as AFHMC at round $R ( d )$ , under two FA-HMC-specific schedules: the dimension-scaling setting of Liang et al. (2025), $K _ { \mathrm { F } } = 5 , T = 1 0$ and $\eta ( d ) = 0 . 0 2 d ^ { - 1 / 4 } ,$ ; and the stepsize suggested by Theorem 4.1 of Liang et al. (2025), $\eta ( d ) = 0 . 0 0 6 ( d / 1 0 0 ) ^ { - 7 / 1 2 }$ , with $T = 1 0$ and $K _ { \mathrm { F } } ( d )$ as above (see Appendix D.2).

![](images/1841089784e8d41ac366e400d01f2102a267d58649e9c0bda3893c6d420bd982.jpg)

![](images/8857392bffc885ea3c6c7c7e70fc89f7b4539155dcf312266d0ef33a83a1da0c.jpg)

![](images/eb63ec28923ef61b1a67150a7affbfffe95da60aaaf506b04726c3be8b86e491.jpg)  
Figure 3: Dimension scaling on Gaussian targets. Left: communication rounds required by AFHMC to reach $\mathcal { W } _ { 2 } ^ { 2 } < 0 . 1$ on the heterogeneous target as dimension increases. Middle: the same quantity on the low-heterogeneity target (mean ± sd over three seeds), with the fitted logarithmic law. Right: FA-HMC ${ \cal W } _ { 2 } ^ { 2 }$ on the heterogeneous target under the original setting $( R ( d )$ rounds) and under two FA-HMC-specific stepsize schedules at the equal server-exchange budget $2 R ( d )$ ; the horizontal line is the AFHMC stopping threshold $\mathcal { W } _ { 2 } ^ { 2 } = 0 . 1$

Figure 3 summarizes the dimension-scaling experiment. The left panel shows that the number of communication rounds required by AFHMC increases from 42 at $d = 2$ to 430 at $d = 5 0 0$ . As shown by the figure, the empirical behavior closely resembles the theoretical trend illustrated by the dotted black curve. When fitting the plot by regression $C ( d / \epsilon ^ { 2 } ) ^ { \alpha } \log ( d / \epsilon ^ { 2 } )$ with $\epsilon ^ { 2 } = 0 . { \dot { 1 } }$ (excluding the outlier case $d = 2 )$ , we have $\hat { \alpha } = 0 . 2 8 2$ . This log-corrected scaling is close to the predicted $O ( ( d / \epsilon ^ { 2 } ) ^ { 1 / 3 } \log ( d / \epsilon ^ { 2 } ) )$ communication complexity, especially on the larger-dimensional regime. The right panel reports the ${ \mathcal W } _ { 2 } ^ { 2 }$ error of FA-HMC when FA-HMC spends the same dimensiondependent budget used by AFHMC, together with the two FA-HMC-specific schedules at the equal server-exchange budget. The horizontal “adaptive threshold” line marks $\mathcal { W } _ { 2 } ^ { 2 } = 0 . 1$ , the stopping accuracy used for AFHMC in the left panel. Under the matched budget, FA-HMC remains orders of magnitude above this threshold: its final $\mathcal { W } _ { 2 } ^ { 2 }$ ranges from $3 . 2 8 \times \mathrm { \breve { 1 } 0 ^ { 2 } }$ at $d = 2$ to $3 . 8 1 \times 1 0 ^ { 4 }$ at $d = 5 0 0$ , with empirical growth speed $O ( d ^ { 0 . 8 \mathsf { \tilde { 4 } 3 } } )$ . The FA-HMC-specific schedules do not close the gap even with twice as many rounds: at $d = 5 0 0$ , they reach $\mathcal { W } _ { 2 } ^ { 2 } = 1 . 1 9 \times 1 0 ^ { 4 }$ and $1 . 9 8 \times 1 0 ^ { 4 }$ after 860 rounds, whereas AFHMC crosses 0.1 at round 430, and their errors grow as $O ( d ^ { 1 . 2 3 } )$ and $O ( d ^ { 0 . 9 7 } )$ , i.e., roughly linearly in $d ,$ as expected from a heterogeneity bias. This significant performance gap between AFHMC and FA-HMC indicates that the FA-HMC dynamics retain a persistent heterogeneity bias, whereas AFHMC reaches a higher level of accuracy and efficiency.

Low-heterogeneity regime. To examine the $O ( \log ( d / \epsilon ^ { 2 } ) )$ communication complexity predicted by Theorem 4.5 when the heterogeneity is small, we keep $N = 1 0$ clients but estimate every local Gaussian from $1 0 ^ { 5 }$ samples drawn from the same distribution $\mathcal { N } ( 1 6 . 2 1 _ { d } , 1 . 6 I _ { d } )$ , so that the heterogeneity only comes from sampling error; the pooled samples define the target. We use $K ( d ) = \lfloor 5 0 ( d / 1 0 0 ) ^ { 1 / 2 } \rceil$ and $\eta ( d ) = 0 . 0 0 6 ( d / 1 0 0 ) ^ { - 1 / 2 }$ , and repeat each setting with three random seeds. AFHMC reaches $\mathcal { W } _ { 2 } ^ { 2 } < 0 . 1$ for every dimension and seed (Figure $^ { 3 , }$ middle), and the firstcrossing round is well described by $R ( d ) = 1 0 4 . 9 + 1 6 . 8 \log ( d / \epsilon ^ { 2 } )$ with $R ^ { 2 } = 0 . 9 9 6$ . As d grows from 50 to 500, $R ( d )$ only increases from 212 to 248 (a factor of $1 . 1 7 )$ , compared with an increase from 162 to 430 (a factor of 2.65) on the heterogeneous target. This provides direct numerical evidence for the logarithmic communication complexity in the low-heterogeneity regime (details in Appendix D.2).

Logistic regression for FMNIST. We next evaluate the proposed method on a Bayesian multinomial logistic regression task using the Fashion-MNIST (FMNIST) dataset (Xiao et $\mathrm { a l . , } 2 0 1 7 )$ which consists of 60,000 training images and 10,000 test images from 10 clothing categories. The dataset is released by Zalando Research under the MIT license. Each image is a $2 8 \times 2 8$ grayscale image, flattened into a 784-dimensional feature vector. To model client heterogeneity, we partition the training data across $N = 1 0$ clients using a label-skewed Dirichlet split with concentration parameter $\alpha = 0 . 1$ , see Appendix D.3 for details.

We compare FA-LD, FA-HMC, and AFHMC under the same heterogeneous data split. For FA-LD we use the $K = 1$ case, while for FA-HMC and AFHMC we evaluate $K \in \{ 1 0 , \mathsf { \bar { s } } 0 , 1 0 0 \}$ . Each method is run for 3600 server exchanges, i.e., 3600 rounds for FA-LD and FA-HMC and 1800 rounds for AFHMC. Between two synchronizations, each client performs 50 leapfrog steps in FA-LD and FA-HMC and K leapfrog steps in AFHMC. For each reported metric, we select the stepsize separately for every method and K value.

Figure 4 reports the test accuracy against the number of server exchanges. AFHMC with $K \in \{ 1 0 , 1 0 0 \}$ overtakes all FA-HMC variants after about 1000 exchanges, stays ahead thereafter, and attains the highest final accuracy (0.842, versus 0.841 for the best FA-HMC at the same number of exchanges and 0.836 at the same number of rounds). With $K = 1 0$ , AFHMC achieves this using about ten times fewer gradient evaluations per client than FA-HMC. This result shows that the drift correction in AFHMC continues to be useful beyond the Gaussian experiments and transfers to a non-Gaussian Bayesian logistic regression posterior with real image data. Additional predictive-quality metrics, including Brier score and negative log-likelihood, are reported in Appendix D.3.

![](images/93fd6df943cbbb9171b04a310223bedbd33f3419f87cf76edb61ff9488160062.jpg)  
Figure 4: Test accuracy for Bayesian logistic regression on heterogeneous FMNIST against the number of server exchanges (one per round for FA-LD and FA-HMC, two per round for AFHMC).

## 6 Conclusions and Discussions

In this work, we propose a novel Bayesian Federated sampling algorithm based on the

Hamiltonian dynamics. We incorporate a global drift control variate into the leapfrog integrator to alleviate the local drifting caused by the data heterogeneity. This surrogate leapfrog update rule leads to a significant advancement of the theoretical behavior, especially for the communication cost with respect to the system’s high dimensionality. Our analysis shows that to achieve an ϵ sampling accuracy, AFHMC only requires $O ( ( d / \epsilon ^ { 2 } ) ^ { 1 / \bar { 3 } } )$ communication rounds. Compared with the $O ( ( d / \epsilon ^ { 2 } ) ^ { 1 / 2 } )$ rate achieved by Langevin-based federated sampler (Plassier et al., 2023; Deng et al., 2024), our result reaffirms the superiority of Hamiltonian dynamics over Langevin dynamics for the Federated sampling task, promoting the communities to utilize HMC-based Federated learning methods for complicated high-dimensional tasks. The advantage claimed by our theory is justified by empirical evidence. In addition, AFHMC can achieve ${ \cal O } ( \log ( d / \epsilon ^ { 2 } ) )$ rate for communication complexity under low heterogeneity, matching the rate of communication cost for the Federated optimization realm; our low-heterogeneity Gaussian experiment in Section 5 confirms this logarithmic scaling numerically.

Limitations. The scope of our theoretical investigation is limited to the log-convex target distribution and doesn’t consider the stochastic gradient. The theoretical analysis for Hamiltonian dynamics or HMC is highly non-trivial. In the literature, not limited to FL, no result achieves an advanced dimension-dependent rate for HMC under nonconvexity or stochastic gradients. Our contribution is mainly theoretical, hence our numerical experiments are on a small scale, similarly to prior theoretical works, serving for the purpose of validating our theorems.

## Acknowledgments and Disclosure of Funding

Guang Lin gratefully acknowledges the support from DARPA MATHBAC program under grant number: HR0011-26-3-E152, the National Science Foundation (DMS-2533878, DMS-2053746, DMS-2134209, ECCS-2328241, CBET-2347401 and OAC-2311848), and U.S. Department of Energy (DOE) Office of Science Advanced Scientific Computing Research program under the "Uncertainty Quantification for Multifidelity Operator Learning (MOLUcQ)" project (Project No. 81739), DE-SC0023161, the SciDAC LEADS Institute, and DOE–Fusion Energy Science, under grant number: DE-SC0024583.

## References

Akhmatskaya, E., Nagar, L., Carrillo, J. A., Gavira Balmacz, L., Inouzhe, H., Parga Pazos, M., and Rodríguez Álvarez, M. X. Adaptive tuning of hamiltonian monte carlo methods. Applied Mathematical Modelling, 157:116892, 2026. ISSN 0307-904X. doi: https://doi.org/10.1016/j.apm.2026.116892. URL https://www.sciencedirect.com/science/article/pii/S0307904X26001538.

Al-Shedivat, M., Gillenwater, J., Xing, E., and Rostamizadeh, A. Federated learning via posterior averaging: A new perspective and practical algorithms. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=GFsU8a0sGB.

Bonawitz, K., Eichner, H., Grieskamp, W., Huba, D., Ingerman, A., Ivanov, V., Kiddon, C., Konecnˇ y,\` J., Mazzocchi, S., and McMahan, B. Towards Federated Learning at Scale: System design. Proceedings of Machine Learning and Systems, 1:374–388, 2019.

Chen, H., Xiao, M., and Pang, Z. Satellite-based computing networks with federated learning. IEEE Wireless Communications, 29(1):78–84, 2022.

Chen, H.-Y. and Chao, W.-L. Fed{be}: Making bayesian model ensemble applicable to federated learning. In International Conference on Learning Representations, 2021. URL https:// openreview.net/forum?id=dgtpE6gKjHn.

Chen, T., Fox, E., and Guestrin, C. Stochastic Gradient Hamiltonian Monte Carlo. In International Conference on Machine Learning (ICML), pp. 1683–1691, 2014.

Dalalyan, A. S. Theoretical Guarantees for Approximate Sampling from Smooth and Log-concave Densities. Journal ofthe Royal Statistical Society: Series B, 79(3):651–676, 2017.

Dalalyan, A. S. and Karagulyan, A. User-friendly Guarantees for the Langevin Monte Carlo with Inaccurate Gradient. Stochastic Processes and their Applications, 129(12):5278–5311, 2019.

Deng, W., Zhang, Q., Ma, Y.-A., Song, Z., and Lin, G. On convergence of federated averaging langevin dynamics. In Proceedings of the Fortieth Conference on Uncertainty in Artificial Intelligence, UAI ’24, 2024.

Duane, S., Kennedy, A., Pendleton, B., and Roweth. Hybrid Monte Carlo. Physics Letters B, 195(2): 216–222, 1987.

Durmus, A. and Moulines, E. High-dimensional Bayesian Inference via The Unadjusted Langevin Algorithm. Bernoulli, 25(4A):2854–2882, 2019.

Durmus, A., Éric Moulines, and Saksman, E. Irreducibility and Geometric Ergodicity of Hamiltonian Monte Carlo. Annals ofStatistics, 48(6):3545–3564, 2020.

Erdogdu, M. A. and Hosseinzadeh, R. On the Convergence of Langevin Monte Carlo: the Interplay between Tail Growth and Smoothness. In Proc. of Conference on Learning Theory (COLT), pp. 1776–1822, 2021.

Fernández-Pendás, M., Akhmatskaya, E., and Sanz-Serna, J. M. Adaptive multi-stage integrators for optimal energy conservation in molecular simulations. Journal of Computational Physics, 327: 434–449, 2016.

Girolami, M. and Calderhead, B. Riemann Manifold Langevin and Hamiltonian Monte Carlo Methods. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 73(2): 123–214, 2011.

Guo, H., Greengard, P., Wang, H., Gelman, A., Kim, Y., and Xing, E. Federated learning as variational inference: A scalable expectation propagation approach. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=dZrQR7OR11.

Hirt, M., Titsias, M., and Dellaportas, P. Entropy-based adaptive hamiltonian monte carlo. Advances in neural information processing systems, 34:28482–28495, 2021.

Hoffman, M., Radul, A., and Sountsov, P. An adaptive-mcmc scheme for setting trajectory lengths in hamiltonian monte carlo. In International Conference on Artificial Intelligence and Statistics, pp. 3907–3915. PMLR, 2021.

Hoffman, M. D. and Gelman, A. The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, pp. 1593–1623, 2014.

Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S., Stich, S., and Suresh, A. T. Scaffold: Stochastic Controlled Averaging for Federated Learning. In International Conference on Machine Learning (ICML), pp. 5132–5143, 2020.

Kim, M. and Hospedales, T. Fedhb: Hierarchical bayesian federated learning. Journal ofMachine Learning Research, 26(272):1–50, 2025.

Konecnˇ y, J., McMahan, B., and Ramage, D. Federated optimization: Distributed optimization \` Beyond the Datacenter. arXiv preprint arXiv:1511.03575, 2015.

Konecnˇ y, J., McMahan, H. B., Yu, F. X., Richtarik, P., Suresh, A. T., and Bacon, D. Federated\` Learning: Strategies for Improving Communication Efficiency. In NIPS Workshop on Private Multi-Party Machine Learning, 2016. URL https://arxiv.org/abs/1610.05492.

Li, T., Sahu, A. K., Talwalkar, A., and Smith, V. Federated Learning: Challenges, Methods, and Future Directions. IEEE Signal Processing Magazine, 37(3):50–60, 2020a.

Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., and Smithy, V. Federated Optimization in Heterogeneous Networks. In Proceedings of the 3rd MLSys Conference, 2020b.

Liang, J., Zhang, Q., Deng, W., Song, Q., and Lin, G. Bayesian federated learning with hamiltonian monte carlo: Algorithm and theory. Journal ofComputational and Graphical Statistics, 34(2): 509–518, 2025.

Liu, L., Jiang, X., Zheng, F., Chen, H., Qi, G.-J., Huang, H., and Shao, L. A bayesian federated learning framework with online laplace approximation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(1):1–16, 2024. doi: 10.1109/TPAMI.2023.3322743.

Mangoubi, O. and Smith, A. Rapid mixing of hamiltonian monte carlo on strongly log-concave distributions. arXiv preprint arXiv:1708.07114, 2017.

Mangoubi, O. and Smith, A. Mixing of hamiltonian monte carlo on strongly log-concave distributions: Continuous dynamics. The Annals of Applied Probability, 31(5):2019–2045, 2021.

Mangoubi, O. and Vishnoi, N. Dimensionally tight bounds for second-order hamiltonian monte carlo. In Bengio, S., Wallach, H., Larochelle, H., Grauman, K., Cesa-Bianchi, N., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/paper\_files/paper/2018/ file/e07bceab69529b0f0b43625953fbf2a0-Paper.pdf.

McMahan, B., Moore, E., Ramage, D., Hampson, S., and y Arcas, B. A. Communication-efficient Learning of Deep Networks from Decentralized Data. In Proc. ofthe International Conference on Artificial Intelligence and Statistics (AISTATS), pp. 1273–1282, 2017.

Mishchenko, K., Malinovsky, G., Stich, S., and Richtarik, P. ProxSkip: Yes! Local gradient steps provably lead to communication acceleration! Finally! In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S. (eds.), Proceedings ofthe 39th International Conference on Machine Learning, volume 162 of Proceedings ofMachine Learning Research, pp. 15750–15769. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/mishchenko22b. html.

Mitra, A., Jaafar, R., Pappas, G. J., and Hassani, H. Linear convergence in federated learning: Tackling client heterogeneity and sparse gradients. Advances in Neural Information Processing Systems, 34:14606–14619, 2021.

Nagar, L., Fernández-Pendás, M., Sanz-Serna, J. M., and Akhmatskaya, E. Adaptive multi-stage integration schemes for hamiltonian monte carlo. Journal ofComputational Physics, 502:112800, 2024.

Neal, R. M. MCMC using Hamiltonian dynamics. In Handbook ofMarkov Chain Monte Carlo, volume 54, pp. 113–162. Chapman and Hall/CRC, 2012.

Plassier, V., Moulines, E., and Durmus, A. Federated averaging langevin dynamics: Toward a unified theory and new algorithms. In International Conference on Artificial Intelligence and Statistics, pp. 5299–5356. PMLR, 2023.

Sato, I. and Nakagawa, H. Approximation analysis of stochastic gradient langevin dynamics by using fokker-planck equation and ito process. In International Conference on Machine Learning, pp. 982–990. PMLR, 2014.

Vedadi, E., Dillon, J. V., Mansfield, P. A., Singhal, K., Afkanpour, A., and Morningstar, W. R. Federated variational inference: Towards improved personalization and generalization. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum? id=6OmRkUHgs5.

Wang, H., Yurochkin, M., Sun, Y., Papailiopoulos, D., and Khazaeni, Y. Federated Learning with Matched Averaging. In International Conference on Learning Representations (ICLR), 2020.

Wang, Z., Chen, Y., Song, Q., and Zhang, R. Enhancing low-precision sampling via stochastic gradient hamiltonian monte carlo. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=uSLNzzuiDJ.

Welling, M. and Teh, Y. W. Bayesian Learning via Stochastic Gradient Langevin Dynamics. In International Conference on Machine Learning (ICML), 2011.

Xiao, H., Rasul, K., and Vollgraf, R. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.

Zhang, X., Li, Y., Li, W., Guo, K., and Shao, Y. Personalized federated learning via variational bayesian inference. In International Conference on Machine Learning, pp. 26293–26310. PMLR, 2022.

Zongchen Chen, S. S. V. Optimal Convergence Rate of Hamiltonian Monte Carlo for Strongly Logconcave Distributions. In Approximation, Randomization, and Combinatorial Optimization, 2019.

Zou, D. and Gu, Q. On the Convergence of Hamiltonian Monte Carlo with Stochastic Gradients. In International Conference on Machine Learning (ICML), pp. 13012–13022, 2021.

Zou, D., Xu, P., and Gu, Q. Stochastic Gradient Hamiltonian Monte Carlo Methods with Recursive Variance Reduction. Advances in Neural Information Processing Systems (NeurIPS), 32, 2019.

## Appendix

## A Algorithm, assumption and notations

## A.1 notations and definitions

The ideal HMC dynamic for the global loss f is defined as:

$$
d \theta ( t ) = p ( t ) d t , \qquad d p ( t ) = - \nabla _ { \theta } f ( \theta ( t ) ) d t .\tag{4}
$$

According to Algorithm 2, we define $\theta _ { t , j } ^ { ( c ) }$ and $p _ { t , j } ^ { ( c ) }$ to be the local position variable and local momentum variable at the j-th surrogate leapfrog step of the c-th local node within the t-th global iteration. In particular, $p _ { t , 0 } ^ { ( c ) } = p _ { t }$ is a freshly sampled Gaussian variable at iteration t shared by all local nodes; for $j > 0 , p _ { t , j } ^ { ( c ) }$ is generated following the local surrogate leapfrog. Also, we have $\begin{array} { r } { \theta _ { t } : = \sum _ { c = 1 } ^ { N } w _ { c } \theta _ { t - 1 , K } ^ { ( c ) } } \end{array}$ aggregated from local nodes and $\theta _ { t , 0 } ^ { ( c ) } : = \theta _ { t }$ broadcast from the center node.

Define $\begin{array} { r } { \theta _ { t , k } : = \sum _ { c = 1 } ^ { N } w _ { c } \theta _ { t , j } ^ { ( c ) } } \end{array}$ as the corresponding (implicit) global position variable, and $p _ { t , k } =$ $\textstyle \sum _ { c = 1 } ^ { N } w _ { c } p _ { t , j } ^ { ( c ) }$ is the corresponding (implicit) global momentum variable.

Define $\mathcal { H } _ { \eta } ( \theta ) : = \theta + \eta p - ( \eta ^ { 2 } / 2 ) \nabla f ( \theta )$ as the classical discrete leapfrog update from position θ associated with momentum p and stepsize η and $\begin{array} { r } { \bar { \mathcal { H } } _ { \eta } ( \theta ) : = \theta + \int _ { 0 } ^ { \eta } p ( t ) d t } \end{array}$ as the classical continuous leapfrog update from position θ associated with momentum $p ( t )$ and length η, following the HMC dynamic (4). Similarly, define $\mathcal { H } _ { \eta } ( p ) : = p - \eta / 2 ( \nabla f ( \theta ) + \nabla \hat { f } ( \theta + \eta p - \hat { \eta ^ { 2 } } / 2 \hat { \nabla } f ( \theta ) ) )$ as a classical discrete leapfrog update from p associated with position θ and stepsize η and $\begin{array} { r } { \bar { \mathcal { H } } _ { \eta } ( p ) : = p - \int _ { 0 } ^ { \eta } \theta ( t ) } \end{array}$ dt as a classical continuous leapfrog update from p associated with position $\theta ( t )$ and length η following the HMC dynamic (4).

Based on these, we use the shorthands $\mathcal { H } _ { \eta } ^ { 1 } ( \theta ) : = \mathcal { H } _ { \eta } ( \theta ) , \mathcal { H } _ { \eta } ^ { i + 1 } ( \theta ) : = \mathcal { H } _ { \eta } ( \mathcal { H } _ { \eta } ^ { i } ( \theta ) ) = \mathcal { H } _ { \eta } ^ { i } ( \theta ) +$ $\eta \mathcal { H } _ { \eta } ^ { i } ( p ) \ : - \ : \eta ^ { 2 } / 2 \nabla f ( \mathcal { H } _ { \eta } ^ { i } ( \theta ) )$ , and $\begin{array} { r l r } { \mathcal { H } _ { \eta } ^ { 1 } ( p ) } & { : = } & { \mathcal { H } _ { \eta } ( p ) , \ \mathcal { H } _ { \eta } ^ { i + 1 } ( p ) \ : = \ \mathcal { H } _ { \eta } ( \mathcal { H } _ { \eta } ^ { i } ( \dot { p } ) ) \ = \ \mathcal { H } _ { \eta } ^ { i } \dot { ( p ) } \ - } \end{array}$ $\eta ^ { 2 } / \dot { 2 } ( \nabla f ( \mathcal { H } _ { \eta } ^ { i } ( \theta ) ) + \nabla \dot { f } ( \mathcal { H } _ { \eta } ^ { i } ( \theta ) + \eta \dot { \mathcal { H } } _ { \eta } ^ { i } ( p ) - \eta ^ { 2 } / 2 \nabla f ( \mathcal { H } _ { \eta } ^ { i } ( \theta ) ) )$ . We also use the shorthands $\begin{array} { r } { \bar { \mathcal { H } } _ { \eta } ^ { i } ( \theta ) = \theta + \int _ { 0 } ^ { i \eta } p ( s ) d s } \end{array}$ and $\begin{array} { r } { \bar { \mathcal { H } } _ { \eta } ^ { i } ( p ) : = p - \int _ { 0 } ^ { i \eta } \theta ( s ) d s } \end{array}$

In addition, we define a coupled continuous HMC process as follows. θ<sup>π</sup> follows the target distribution $\propto \exp ( - f ) . \ \theta _ { t , 0 } ^ { \pi } : = \theta _ { t } ^ { \pi } , \theta _ { t , k + 1 } ^ { \pi } : = \bar { \mathcal { H } } _ { \eta } ( \theta _ { t , k } ^ { \pi } )$ , and $\theta _ { t + 1 } ^ { \pi } : = \theta _ { t , K } ^ { \pi } = \bar { \mathcal { H } } _ { K \eta } ( \theta _ { t , 0 } ^ { \pi } )$ . For this process, the associated initial momentum is the same as the discrete HMC process, i.e., $p _ { t , 0 } ^ { \pi } = p _ { t } ^ { \pi } = p _ { t }$

We also denote $\theta ^ { * } = \operatorname * { a r g m i n } _ { \theta } f ( \theta )$

## B Proofs

## B.1 main proof

Theorem 4.5. Under Assumptions 4.1-4.2 and if $H \gg M ^ { 4 } \epsilon ^ { 2 } / m > 0$ , then there exists some small absolute constant C such that ifwe choose

$$
K \eta = C \left( \frac { \epsilon ^ { 2 } m ^ { 2 } } { H M ^ { 2 } } \right) ^ { 1 / 6 } \ll C \frac { \sqrt { m } } { M } ~ a n d ~ \eta = C \left( \frac { \epsilon ^ { 2 } m ^ { 2 } } { M ^ { 2 } d } \right) ^ { 1 / 2 } ,
$$

$\mathcal { W } _ { 2 } ( \pi _ { t _ { \epsilon } } , \pi ) \leq \epsilon$ can be attained with communication times

$$
t _ { \epsilon } \asymp \Big ( \frac { H M ^ { 2 } } { \epsilon ^ { 2 } m ^ { 5 } } \Big ) ^ { 1 / 3 } \log ( \mathcal { W } _ { 2 } ^ { 2 } ( \pi _ { 0 } , \pi ) / \epsilon ^ { 2 } ) .
$$

The corresponding local iteration number is

$$
K \asymp \frac { M ^ { 2 / 3 } d ^ { 1 / 2 } } { \epsilon ^ { 2 / 3 } m ^ { 2 / 3 } H ^ { 1 / 6 } } .
$$

Moreover, $i f H \ll M ^ { 4 } \epsilon ^ { 2 } / m ( i . e .$ , heterogeneity across local nodes is very small), it only requires

$$
t _ { \epsilon } \asymp \log ( \mathcal { W } _ { 2 } ^ { 2 } ( \pi _ { 0 } , \pi ) / \epsilon ^ { 2 } ) ,
$$

achieving a linear rate with respect to the communication cost.

Proof. Let us define

$$
K ^ { 2 } \eta ^ { 2 } = C ( { \frac { \epsilon ^ { 2 } m ^ { 2 } } { H M ^ { 2 } } } ) ^ { 1 / 3 } , \qquad \eta ^ { 2 } = C { \frac { \epsilon ^ { 2 } m ^ { 2 } } { M ^ { 2 } d } }\tag{5}
$$

for some sufficiently small C, we have

$$
\frac { K ^ { 2 } \eta ^ { 4 } \frac { M ^ { 2 } } { m } d } { m K ^ { 2 } \eta ^ { 2 } } = C \epsilon ^ { 2 } \leq \frac { 1 } { 3 } \epsilon ^ { 2 } , \qquad \frac { ( K \eta ) ^ { 8 } \frac { M ^ { 2 } } { m } H } { m K ^ { 2 } \eta ^ { 2 } } = C ^ { 3 } \epsilon ^ { 2 } \leq \frac { 1 } { 3 } \epsilon ^ { 2 } .\tag{6}
$$

One can check that if

$$
t \ge \frac { \log ( \epsilon ^ { 2 } / ( 3 \mathbb { E } \| \theta _ { 0 } - \theta _ { 0 } ^ { \pi } \| ^ { 2 } ) ) } { \log ( 1 - C m ( \frac { \epsilon ^ { 2 } m ^ { 2 } } { H M ^ { 2 } } ) ^ { 1 / 3 } / 1 6 ) } \asymp \frac { 1 } { C } ( \frac { H M ^ { 2 } } { \epsilon ^ { 2 } m ^ { 5 } } ) ^ { 1 / 3 } ( \log ( 3 \mathbb { E } \| \theta _ { 0 } - \theta _ { 0 } ^ { \pi } \| ^ { 2 } / \epsilon ^ { 2 } ) ) ,
$$

we have

$$
( 1 - m ( K \eta ) ^ { 2 } / 1 6 ) ^ { t } \mathbb { E } \| \theta _ { t } - \theta _ { t } ^ { \pi } \| ^ { 2 } \leq \frac { 1 } { 3 } \epsilon ^ { 2 } .\tag{7}
$$

Combining (6)-(7) and by Theorem 4.3, we can see that, for enough small C and set (5), we can have

$$
\begin{array} { r } { \mathbb { E } \lVert \theta _ { t } - \theta _ { t } ^ { \pi } \rVert ^ { 2 } \leq \epsilon ^ { 2 } . } \end{array}
$$

This justifies the required communication round

$$
t _ { \epsilon } \ge \frac { 1 } { C } ( \frac { H M ^ { 2 } } { \epsilon ^ { 2 } m ^ { 5 } } ) ^ { 1 / 3 } \log ( 3 \mathbb { E } \| \theta _ { 0 } - \theta _ { 0 } ^ { \pi } \| ^ { 2 } / \epsilon ^ { 2 } )
$$

On the other hand, if $H \ll M ^ { 4 } \epsilon ^ { 2 } / m$ (i.e., the heterogeneity cross local nodes is small), by choosing enough small C such that with $\bar { K ^ { 2 } \eta ^ { 2 } } = \bar { C } m / M ^ { 2 }$ and $\eta ^ { 2 } \stackrel { \bullet } { = } C \epsilon ^ { 2 } m ^ { 2 } / ( M ^ { 2 } d )$ , we have

$$
\frac { K ^ { 2 } \eta ^ { 4 } \frac { M ^ { 2 } } { m } d } { m K ^ { 2 } \eta ^ { 2 } } \leq C ^ { 2 } \epsilon ^ { 2 } \leq \frac { 1 } { 3 } \epsilon ^ { 2 } , \qquad \frac { ( K \eta ) ^ { 8 } \frac { M ^ { 2 } } { m } H } { m K ^ { 2 } \eta ^ { 2 } } = C ^ { 3 } \frac { m } { M ^ { 4 } } H \ll \epsilon ^ { 2 } \leq \frac { 1 } { 3 } \epsilon ^ { 2 } .\tag{8}
$$

One can check that if

$$
t \ge \frac { \log ( \epsilon ^ { 2 } / ( 3 \mathbb { E } \| \theta _ { 0 } - \theta _ { 0 } ^ { \pi } \| ^ { 2 } ) ) } { \log ( 1 - C m ^ { 2 } / ( 1 6 M ^ { 2 } ) ) } ,
$$

we have

$$
( 1 - \frac { C m ^ { 2 } } { M ^ { 2 } } ) ^ { t } \mathbb { E } \| \theta _ { t } - \theta _ { t } ^ { \pi } \| ^ { 2 } \leq \frac { 1 } { 3 } \epsilon ^ { 2 } .\tag{9}
$$

Combining (8)-(9) and by Theorem 4.3, we obtain that E $\lVert \theta _ { t } - \theta _ { t } ^ { \pi } \rVert ^ { 2 } \leq \epsilon ^ { 2 }$ . This justified the required communication round to be

$$
t _ { \epsilon } \geq C ^ { \prime } \log ( 3 \mathbb { E } \| \theta _ { 0 } - \theta _ { 0 } ^ { \pi } \| ^ { 2 } / \epsilon ^ { 2 } ) .
$$

Finally, we can choose $\theta _ { 0 } ^ { \pi }$ such that $\mathcal { W } ^ { 2 } ( \pi _ { t } , \pi ) = \mathbb { E } \lVert \theta _ { 0 } - \theta _ { 0 } ^ { \pi } \rVert ^ { 2 }$ , which concludes the proof. □

Theorem 4.3 (Convergence). Under Assumptions 4.1-4.2, there exist sufficient large constants $C _ { 0 } , C _ { 1 } , C _ { 2 }$ such that $\{ \bar { \theta } _ { t } \}$ <sub>t</sub> satisfies that, for $\bar { K } \eta \leq \sqrt { m } / ( C _ { 0 } M ) \leq \sqrt { m } / ( 2 \bar { M } )$

$$
{  { \mathbb E } } \| \theta _ { t + 1 } - \theta _ { t + 1 } ^ { \pi } \| ^ { 2 } \le \left( 1 - \frac { m ( K \eta ) ^ { 2 } } { 1 6 } \right) {  { \mathbb E } } \| \theta _ { t } - \theta _ { t } ^ { \pi } \| ^ { 2 } + C _ { 1 } K ^ { 2 } \eta ^ { 4 } \frac { M ^ { 2 } } m d + C _ { 2 } ( K \eta ) ^ { 8 } \frac { M ^ { 2 } } m H
$$

where $\begin{array} { r } { H : = \operatorname* { m a x } _ { i \leq K } \sum _ { c = 1 } ^ { N } w _ { c } \mathbb { E } \| \frac { 1 } { i \eta } ( \nabla f ^ { ( c ) } ( \bar { \mathcal { H } } _ { \eta } ^ { i } ( \theta ^ { \pi } ) ) - \nabla f ^ { ( c ) } ( \theta ^ { \pi } ) + \nabla f ( \theta ^ { \pi } ) - \nabla f ( \bar { \mathcal { H } } _ { \eta } ^ { i } ( \theta ^ { \pi } ) ) \| ^ { 2 } } \end{array}$ is a metric quantifying the federated heterogeneity.

Proof. We introduce an intermediate variable $\bar { \mathcal { H } } _ { n } ^ { K } ( \theta _ { t } )$ being the continuously updated version of $\theta _ { t , K }$ from $\theta _ { t }$ . For $c _ { 1 } , c _ { 2 } > 0$ with $c _ { 1 } + c _ { 2 } = 1$ , by Lemma B.12, we can separate the errors into two parts

$$
\| \theta _ { t , K } - \theta _ { t , K } ^ { \pi } \| ^ { 2 } \leq \frac { 1 } { c _ { 1 } } \underbrace { \| \theta _ { t , K } - \bar { \mathcal { H } } _ { \eta } ^ { K } ( \theta _ { t } ) \| ^ { 2 } } _ { \mathrm { d i s c r e t e ~ d i v e r g e n c e ~ e r r o r } } + \frac { 1 } { c _ { 2 } } \underbrace { \| \bar { \mathcal { H } } _ { \eta } ^ { K } ( \theta _ { t } ) - \theta _ { t , K } ^ { \pi } \| ^ { 2 } } _ { \mathrm { i d e a l ~ c o n v e r g e n c e ~ e r r o r } }\tag{10}
$$

The convergence of the ideal process has been well studied in the literature Zongchen Chen (2019). Indeed, recall the definitions that that $\bar { \mathcal { H } } _ { n } ^ { K } ( \theta _ { t } )$ and $\theta _ { t , K } ^ { \pi }$ share the same initial momentum $p _ { t }$ for each t, by Lemma B.1, it can be bounded by a contraction factor of the distance between their initial positions

$$
\| \bar { \mathcal { H } } _ { \eta } ^ { K } ( \theta _ { t } ) - \theta _ { t , K } ^ { \pi } \| ^ { 2 } \leq ( 1 - \frac { m ( K \eta ) ^ { 2 } } { 4 } ) \| \theta _ { t } - \theta _ { t } ^ { \pi } \| ^ { 2 } , \qquad \mathrm { f o r } K \eta \leq 1 / ( 2 \sqrt { M } ) .\tag{11}
$$

We choose $c _ { 1 } , c _ { 2 }$ such that $\textstyle \frac { 1 - m ( K \eta ) ^ { 2 } / 4 } { c _ { 2 } } = 1 - m ( K \eta ) ^ { 2 } / 8$ , and it follows

$$
c _ { 2 } = \frac { 1 - m ( K \eta ) ^ { 2 } / 4 } { 1 - m ( K \eta ) ^ { 2 } / 8 } , \qquad c _ { 1 } = \frac { m ( K \eta ) ^ { 2 } / 8 } { 1 - m ( K \eta ) ^ { 2 } / 8 } \geq \frac { m ( K \eta ) ^ { 2 } } { 8 }\tag{12}
$$

This choice ensures that, after amplification by $1 / c _ { 1 }$ , the contraction factor in (10) is still strictly smaller than 1.

In the following, to finish the proof, it suffice to show that the divergence error

$$
\mathbb { E } \Vert \theta _ { t , K } - \bar { \mathcal { H } } _ { \eta } ^ { K } ( \theta _ { t } ) \Vert ^ { 2 } \leq \frac { m ( K \eta ) ^ { 2 } } { 1 6 } \Vert \theta _ { t } - \theta _ { t } ^ { \pi } \Vert ^ { 2 } + C _ { 1 } K ^ { 2 } \eta ^ { 4 } \frac { M ^ { 2 } } { m } d + C _ { 2 } ( K \eta ) ^ { 8 } \frac { M ^ { 2 } } { m } H ,
$$

or some constants $C _ { 1 } , C _ { 2 } > 0$ . To study the discrete divergence error, we separate the error within the leapfrog-step level. Since by definitions $\begin{array} { r } { \theta _ { t , K } = \sum _ { c = 1 } ^ { N } \theta _ { t , K } ^ { ( c ) } } \end{array}$ and $\theta _ { t } = \theta _ { t , 0 }$ we can write

$$
\theta _ { t , K } - \bar { \mathcal { H } } _ { \eta } ^ { K } ( \theta _ { t } ) = \theta _ { t , K } - \bar { \mathcal { H } } _ { \eta } ^ { K } ( \theta _ { t , 0 } ) = \sum _ { k = 1 } ^ { K } \bar { \mathcal { H } } _ { \eta } ^ { K - k } ( \theta _ { t , k } ) - \bar { \mathcal { H } } _ { \eta } ^ { K - k + 1 } ( \theta _ { t , k - 1 } ) .
$$

By convexity of $\| \cdot \| ^ { 2 }$ and Lemma B.12,

$$
\begin{array} { r l } {  { \Vert \theta _ { t , K } - \bar { \mathcal { H } } _ { \eta } ^ { K } ( \theta _ { t } ) \Vert ^ { 2 } = \bigr \Vert \displaystyle \sum _ { k = 1 } ^ { K } \bar { \mathcal { H } } _ { \eta } ^ { K - k } ( \theta _ { t , k } ) - \bar { \mathcal { H } } _ { \eta } ^ { K - k + 1 } ( \theta _ { t , k - 1 } ) \Vert ^ { 2 } } \quad } & { } \\ & { \leq K \displaystyle \sum _ { k = 1 } ^ { K } \Vert \bar { \mathcal { H } } _ { \eta } ^ { K - k } ( \theta _ { t , k } ) - \bar { \mathcal { H } } _ { \eta } ^ { K - k + 1 } ( \theta _ { t , k - 1 } ) \Vert ^ { 2 } } \end{array}
$$

We emphasis that $\bar { \mathcal { H } } _ { \eta } ^ { K - k } ( \theta _ { t , k } )$ starts with position $\theta _ { t , k }$ and momentum $p _ { t , k }$ while $\bar { \mathcal { H } } _ { \eta } ^ { K - k + 1 } ( \theta _ { t , k - 1 } )$ can be viewed as a $( K - k ) r$ -length process starting with position $\bar { \mathcal { H } } _ { \eta } ( \theta _ { t , k - 1 } )$ and momentum $\bar { \mathcal { H } } _ { \eta } ( p _ { t , k - 1 } )$ . Lemma B.1 yields

$$
\lVert \bar { \mathcal { H } } _ { \eta } ^ { K - k } ( \theta _ { t , k } ) - \bar { \mathcal { H } } _ { \eta } ^ { K - k + 1 } ( \theta _ { t , k - 1 } ) \rVert ^ { 2 } \leq 2 \lVert \theta _ { t , k } - \bar { \mathcal { H } } _ { \eta } ( \theta _ { t , k - 1 } ) \rVert ^ { 2 } + 2 ( K - k ) ^ { 2 } \eta ^ { 2 } \lVert p _ { t , k } - \bar { \mathcal { H } } _ { \eta } ( p _ { t , k - 1 } ) \rVert ^ { 2 } .
$$

It follows that, we can bound the divergence error as follow

$$
\left\| \theta _ { t , K } - \bar { \mathcal { H } } _ { \eta } ^ { K } ( \theta _ { t } ) \right\| ^ { 2 } \leq 2 K \sum _ { k = 1 } ^ { K } \Bigl ( \underbrace { \| \theta _ { t , k } - \bar { \mathcal { H } } _ { \eta } ( \theta _ { t , k - 1 } ) \| ^ { 2 } } _ { \mathrm { e r r o r ~ i n d u c e d b y i n a c c u r a t e ~ p o s i i o n s } } + \left( K - k \right) ^ { 2 } \eta ^ { 2 } \underbrace { \| p _ { t , k } - \bar { \mathcal { H } } _ { \eta } ( p _ { t , k - 1 } ) \| ^ { 2 } } _ { \mathrm { e r r o r ~ i n d u c e d b y ~ i n a c c u r a t e ~ m o m e n t u m s } } \Bigr ) .\tag{13}
$$

The divergence errors can be further separated into systematic leapfrog approximation error and surrogate-induced error. To see this, by definitions

$$
\begin{array} { r l } & { \theta _ { t , k } - \bar { \mathcal { H } } _ { \eta } ( \theta _ { t , k - 1 } ) = \underbrace { \mathcal { H } _ { \eta } ( \theta _ { t , k - 1 } ) - \bar { \mathcal { H } } _ { \eta } ( \theta _ { t , k - 1 } ) } _ { \mathrm { s y s t e m a i t i c l e a p t r o g e } } - \frac { \eta ^ { 2 } } { 2 } \sum _ { c = 1 } ^ { N } w _ { c } ( \underbrace { \nabla f ^ { ( c ) } ( \theta _ { t , k - 1 } ^ { ( c ) } ) - \nabla f ^ { ( c ) } ( \theta _ { t , k - 1 } ) } _ { \mathrm { s u r r o g a t e i n d u c e d \ : g a p } } ) } \\ & { p _ { t , k } - \bar { \mathcal { H } } _ { \eta } ( p _ { t , k - 1 } ) = \underbrace { \mathcal { H } _ { \eta } ( p _ { t , k - 1 } ) - \bar { \mathcal { H } } _ { \eta } ( p _ { t , k - 1 } ) } _ { \mathrm { s y s t e m a i t i c l e a p f r o g \ : g a p } } } \\ & { \quad \quad \quad - \frac { \eta } { 2 } \sum _ { c = 1 } ^ { N } w _ { c } ( \underbrace { \nabla f ^ { ( c ) } ( \theta _ { t , k - 1 } ^ { ( c ) } ) - \nabla f ^ { ( c ) } ( \theta _ { t , k - 1 } ) + \nabla f ^ { ( c ) } ( \theta _ { t , k } ^ { ( c ) } ) - \nabla f ^ { ( c ) } ( \theta _ { t , k } ) } _ { \mathrm { s u r o g a t i n d u c e d \ : g a p } } ) . } \end{array}
$$

Applying Lemma B.12, we can have

$$
\begin{array} { r l } & { \displaystyle \| \theta _ { t , k } - \bar { \mathcal { H } } _ { \eta } ( \theta _ { t , k - 1 } ) \| ^ { 2 } \leq 2 \| \mathcal { H } _ { \eta } ( \theta _ { t , k - 1 } ) - \bar { \mathcal { H } } _ { \eta } ( \theta _ { t , k - 1 } ) \| ^ { 2 } + 2 \frac { \eta ^ { 4 } } { 4 } \displaystyle \sum _ { c = 1 } ^ { N } w _ { c } \| \nabla f ^ { ( c ) } ( \theta _ { t , k - 1 } ^ { ( c ) } ) - \nabla f ^ { ( c ) } ( \theta _ { t , k - 1 } ) \| ^ { 2 } } \\ & { \displaystyle \| p _ { t , k } - \bar { \mathcal { H } } _ { \eta } ( p _ { t , k - 1 } ) \| ^ { 2 } \leq 2 \| \mathcal { H } _ { \eta } ( p _ { t , k - 1 } ) - \bar { \mathcal { H } } _ { \eta } ( p _ { t , k - 1 } ) \| ^ { 2 } + 2 \frac { \eta ^ { 2 } } { 4 } \displaystyle \sum _ { c = 1 } ^ { N } w _ { c } \displaystyle \sum _ { j = k - 1 } ^ { k } 2 \| \nabla f ^ { ( c ) } ( \theta _ { t , j } ^ { ( c ) } ) - \nabla f ^ { ( c ) } ( \theta _ { t , j } ) \| ^ { 2 } . } \end{array}
$$

Inserting these beck to (13), we have

$$
\begin{array} { r l } & { \displaystyle \| \theta _ { t , K } - \widetilde { \mathcal { H } } _ { \eta } ^ { K } ( \theta _ { t , s } ) \| ^ { 2 } } \\ & { \leq 2 K \displaystyle \sum _ { k = 1 } ^ { K } \Big ( 2 \| \mathcal { H } _ { \eta } ( \theta _ { t , k - 1 } ) - \mathcal { \bar { H } } _ { \eta } ( \theta _ { t , k - 1 } ) \| ^ { 2 } + 2 ( K - k ) ^ { 2 } \eta ^ { 2 } \| \mathcal { H } _ { \eta } ( \rho _ { t , k - 1 } ) - \bar { \mathcal { H } } _ { \eta } ( \rho _ { t , k - 1 } ) \| ^ { 2 } } \\ & { \quad \quad \quad + \displaystyle \frac { \eta ^ { 4 } } { 2 } \displaystyle \sum _ { \epsilon = 1 } ^ { K } w _ { c } ( | \nabla f ^ { ( c ) } ( \theta _ { t , k - 1 } ^ { ( c ) } ) - \nabla f ^ { ( c ) } ( \theta _ { t , k - 1 } ) | ^ { 2 } + ( K - k ) ^ { 2 } \displaystyle \sum _ { \epsilon = k - 1 } ^ { k } \| \nabla f ^ { ( c ) } ( \theta _ { t , k } ^ { ( c ) } ) - \nabla f ^ { ( c ) } ( \theta _ { t , k } ) \| ^ { 2 } ) \Big ) } \\ & { \leq 5 K \displaystyle \sum _ { k = 1 } ^ { K - 1 } \Big ( \| \mathcal { H } _ { \eta } ( \theta _ { t , k } ) - \bar { \mathcal { H } } _ { \eta } ( \theta _ { t , k } ) \| ^ { 2 } + ( K - k ) ^ { 2 } \eta ^ { 2 } \| \mathcal { H } _ { \eta } ( \rho _ { t , k } ) - \bar { \mathcal { H } } _ { \eta } ( \rho _ { t , k } ) \| ^ { 2 } } \\ & { \quad \quad \quad + ( K \eta ) ^ { 2 } \eta ^ { 2 } \displaystyle \sum _ { \epsilon = 1 } ^ { K } | \nabla f ^ { ( c ) } ( \theta _ { t , k } ^ { ( c ) } ) - \nabla f ^ { ( c ) } ( \theta _ { t , k } ) \| ^ { 2 } \Big ) , } \end{array}\tag{14}
$$

where we have applied $2 \eta ^ { 4 } ( 1 / 2 + 2 ( K - k ) ^ { 2 } ) \leq 5 K ^ { 2 } \eta ^ { 4 }$ for $k < K$ in the last inequality. By M-smoothness assumption,

$$
\sum _ { c = 1 } ^ { N } w _ { c } \mathbb { E } \| \nabla f ^ { ( c ) } ( \theta _ { t , k } ^ { ( c ) } ) - \nabla f ^ { ( c ) } ( \theta _ { t , k } ) \| ^ { 2 } \leq \sum _ { c = 1 } ^ { N } w _ { c } M ^ { 2 } \mathbb { E } \| \theta _ { t , k } ^ { ( c ) } - \theta _ { t , k } \| ^ { 2 }
$$

and further by Lemma B.6 that $\begin{array} { r } { \sum _ { c = 1 } ^ { N } w _ { c } \mathbb { E } \| \theta _ { t , k } ^ { ( c ) } - \theta _ { t , k } \| ^ { 2 } \leq k ^ { 3 } \eta ^ { 4 } \sum _ { i = 1 } ^ { k - 1 } ( 1 6 M ^ { 2 } \mathbb { E } \| \theta _ { t , i } ^ { \pi } - \theta _ { t , i } \| ^ { 2 } + } \end{array}$ $1 6 M ^ { 2 } \mathbb { E } \| \theta _ { t } ^ { \pi } - \theta _ { t } \| ^ { 2 } + 4 ( k \eta ) ^ { 2 } H )$ , we have

$$
\sum _ { c = 1 } ^ { N } w _ { c } \mathbb { E } \| \nabla f ^ { ( c ) } ( \theta _ { t , k } ^ { ( c ) } ) - \nabla f ^ { ( c ) } ( \theta _ { t , k } ) \| ^ { 2 } \leq k ^ { 3 } \eta ^ { 4 } M ^ { 2 } \sum _ { i = 1 } ^ { k - 1 } ( 1 6 M ^ { 2 } \mathbb { E } \| \theta _ { t , i } ^ { \pi } - \theta _ { t , i } \| ^ { 2 } + 1 6 M ^ { 2 } \mathbb { E } \| \theta _ { t } ^ { \pi } - \theta _ { t } \| ^ { 2 } + 4 ( k \eta ) ^ { 2 } H ) .\tag{15}
$$

Here we recall that $\begin{array} { r } { H : = \operatorname* { m a x } _ { i \leq K } \sum _ { c = 1 } ^ { N } w _ { c } \mathbb { E } \| \frac { 1 } { i \eta } ( \nabla f ^ { ( c ) } ( \bar { \mathcal { H } } _ { \eta } ^ { i } ( { \theta } ^ { \pi } ) ) - \nabla f ^ { ( c ) } ( { \theta } ^ { \pi } ) + \nabla f ( { \theta } ^ { \pi } ) - \frac { 1 } { i \eta } ( \nabla f ^ { ( c ) } ( \bar { \mathcal { H } } _ { \eta } ^ { i } ( { \theta } ^ { \pi } ) ) ) ) \| _ { L ^ { 2 } } , } \end{array}$ $\nabla f ( \bar { \mathcal { H } } _ { \eta } ^ { i } ( \theta ^ { \pi } ) ) ) \lvert \rvert ^ { 2 }$ is a quantity measuring the federated heterogeneity.

As for the systematic leapfrog approximation errors, by Lemma B.3 and Lemma B.4, we have

$$
\begin{array} { l } { \displaystyle \| \mathcal { H } _ { \eta } ( \theta _ { t , k } ) - \bar { \mathcal { H } } _ { \eta } ( \theta _ { t , k } ) \| ^ { 2 } \leq \eta ^ { 6 } \Big ( \frac { M ^ { 2 } } { 1 0 } \| p _ { t , k } \| ^ { 2 } + \frac { M ^ { 4 } \eta ^ { 2 } } { 6 0 } \| \theta _ { t , k } - \theta ^ { * } \| ^ { 2 } \Big ) } \\ { \displaystyle \| \mathcal { H } _ { \eta } ( p _ { t , k } ) - \bar { \mathcal { H } } _ { \eta } ( p _ { t , k } ) \| ^ { 2 } \leq \eta ^ { 4 } \Big ( \frac { 5 M ^ { 2 } } { 2 } \| p _ { t , k } \| ^ { 2 } + \frac { 2 \eta ^ { 2 } M ^ { 4 } } { 3 } \| \theta _ { t , k } - \theta ^ { * } \| ^ { 2 } \Big ) } \end{array}\tag{16}
$$

Inserting (15)-(16) back to (14) and applying Lemma B.10 on $_ { p _ { t , k } }$ and $\theta _ { t , k }$ , we obtain

$$
\begin{array} { l } { \displaystyle \mathbb { E } \| \theta _ { t , K } - \widetilde { \mathcal { H } } _ { \eta } ^ { K } ( \theta _ { t } ) \| ^ { 2 } \leq 5 K \displaystyle \sum _ { k = 0 } ^ { K - 1 } \Big ( ( \frac { \eta ^ { \beta } M ^ { 2 } } { 1 0 } + \frac { ( K \eta ) ^ { 2 } \eta ^ { 4 } \xi M ^ { 2 } } { 2 } ) \mathbb { E } \| \eta _ { t , k } \| ^ { 2 } + ( \frac { \gamma ^ { \beta } M ^ { 4 } } { 6 0 } + \frac { ( K \eta ) ^ { 2 } 2 \eta ^ { 6 } M ^ { 4 } } { 3 } ) \mathbb { E } \| \theta _ { t , k } - \theta ^ { * } \| ^ { 2 } } \\ { \displaystyle \qquad + K ^ { 5 } \eta ^ { 8 } M ^ { 2 } \displaystyle \sum _ { \ell = 1 } ^ { k - 1 } ( 1 6 M ^ { 2 } \mathbb { E } \| \theta _ { t , i } ^ { * } - \theta _ { t , \ell } \| ^ { 2 } + 1 6 M ^ { 2 } \mathbb { E } \| \theta _ { t } ^ { * } - \theta _ { t } \| ^ { 2 } + 4 ( k \eta ) ^ { 2 } H \Big ) } \\ { \displaystyle \leq 5 K \displaystyle \sum _ { k = 0 } ^ { K - 1 } \Big ( ( \frac { \eta ^ { \beta } M ^ { 2 } } { 1 0 } + \frac { ( K \eta ) ^ { 2 } \eta ^ { 4 } \mathbb { E } M ^ { 2 } } { 2 } ) ( 5 \mathbb { E } \| \eta _ { t } \| ^ { 2 } + 5 M ^ { 2 } ( k \eta ) ^ { 2 } \mathbb { E } \| \theta _ { t } - \theta ^ { * } \| ^ { 2 } ) } \\ { \displaystyle \qquad + ( \frac { \gamma ^ { \beta } M ^ { 4 } } { 6 0 } + \frac { ( K \eta ) ^ { 2 } 2 \eta ^ { 6 } M ^ { 4 } } { 3 } ) ( 4 ( k \eta ) ^ { 2 } \mathbb { E } \| \rho _ { t } \| ^ { 2 } + 3 \mathbb { E } \| \theta _ { t } - \theta ^ { * } \| ^ { 2 } ) } \\  \displaystyle \qquad + K ^ { 5 } \eta ^ { 8 } M  \end{array}
$$

Rearranging the coefficients, it can be seen that

$$
\begin{array} { r l } { \mathbb { E } \| \theta _ { t , K } - \widetilde { \mathcal { H } } _ { \mathcal { T } } ^ { K } ( \theta _ { t } ) \| ^ { 2 } \leq \biggl ( \displaystyle \frac { 5 } { 2 } K ^ { 2 } \eta ^ { 9 } M ^ { 2 } + \frac { 1 2 5 } { 2 } K ^ { 4 } \eta ^ { 6 } M ^ { 2 } + \frac { K ^ { 4 } \eta ^ { 1 0 } M ^ { 4 } } { 3 } + \frac { 4 0 K ^ { 6 } \eta ^ { 1 0 } M ^ { 4 } } { 3 } \biggr ) \mathbb { E } \| \rho _ { t } \| ^ { 2 } } & { } \\ & { \quad + \biggl ( \frac { \tilde { 5 } } { 2 } K ^ { 4 } \eta ^ { 8 } M ^ { 4 } + \frac { 1 2 5 } { 2 } K ^ { 6 } \eta ^ { 8 } M ^ { 4 } + \frac { K ^ { 2 } \eta ^ { 5 } M ^ { 4 } } { 4 } + 8 0 K ^ { 4 } \eta ^ { 8 } M ^ { 2 } \biggr ) \mathbb { E } \| \theta _ { t } - \theta ^ { \star } \| ^ { 2 } } \\ & { \qquad \kappa - 1 } \\ & { = 8 0 K ^ { 6 } \eta ^ { 8 } M ^ { 4 } \displaystyle \sum _ { i = 1 } ^ { k - 1 } \mathbb { E } \| \theta _ { t , i } ^ { \star } - \theta _ { t , i } \| ^ { 2 } } \\ & { \quad + 8 0 ( K \eta ) ^ { 8 } M ^ { 4 } \mathbb { E } \| \theta _ { t } ^ { 6 } - \theta _ { t } \| ^ { 2 } } \\ & { \quad + 2 0 ( K \eta ) ^ { 1 0 } M ^ { 2 } M ^ { 2 } H } \\ & { \leq C K ^ { 4 } \eta ^ { 6 } M ^ { 2 } + C K ^ { 6 } \eta ^ { 8 } M ^ { 4 } \mathbb { E } \| \theta _ { t } - \theta ^ { \star } \| ^ { 2 } + C K ^ { 7 } \eta ^ { 8 } M ^ { 4 } \displaystyle \sum _ { i = 1 } ^ { K - 2 } \mathbb { E } \| \theta _ { t , i } ^ { \star } - \theta _ { t , i } \| ^ { 2 } } \\ &  \quad + C ( K \eta ) ^  \end{array}
$$

for enough large C and $K \eta \le 1 / ( 2 \sqrt { M } )$

Combining this with E $\lVert \theta _ { t } - \theta ^ { * } \rVert ^ { 2 } \leq 2 \mathbb { E } \lVert \theta _ { t } - \theta _ { t } ^ { \pi } \rVert ^ { 2 } + 2 \mathbb { E } \lVert \theta _ { t } ^ { \pi } - \theta ^ { * } \rVert ^ { 2 }$ and Lemma B.2 that $\mathbb { E } \| \theta _ { t } ^ { \pi } - \theta ^ { * } \| ^ { 2 } \leq$ $d / m$ , we get

$$
\begin{array} { r } { \mathbb { E } \| \theta _ { t , K } - \bar { \mathcal { H } } _ { \eta } ^ { K } ( \theta _ { t } ) \| ^ { 2 } \leq C ^ { \prime } K ^ { 4 } \eta ^ { 6 } M ^ { 2 } d + C ^ { \prime } K ^ { 7 } \eta ^ { 8 } M ^ { 4 } \displaystyle \sum _ { i = 1 } ^ { K - 2 } \mathbb { E } \| \theta _ { t , i } ^ { \pi } - \theta _ { t , i } \| ^ { 2 } } \\ { + C ^ { \prime } ( K \eta ) ^ { 8 } M ^ { 4 } \mathbb { E } \| \theta _ { t } ^ { \pi } - \theta _ { t } \| ^ { 2 } + C ^ { \prime } ( K \eta ) ^ { 1 0 } M ^ { 2 } H } \end{array}\tag{17}
$$

for enough large $C ^ { \prime }$

Finally, combining it with (10)-(12), it can be seen that for an enough large $C ^ { \prime \prime }$

$$
\begin{array} { r l } & { \mathbb { E } \| \theta _ { t , K } - \theta _ { t , K } ^ { \pi } \| ^ { 2 } \leq ( 1 - \frac { m ( K \eta ) ^ { 2 } } { 8 } + \frac { C ^ { \prime \prime } ( K \eta ) ^ { 6 } M ^ { 4 } } { m } ) \mathbb { E } \| \theta _ { t } - \theta _ { t } ^ { \pi } \| ^ { 2 } + C ^ { \prime \prime } K ^ { 2 } \eta ^ { 4 } \frac { M ^ { 2 } } { m } d } \\ & { \quad \quad \quad \quad \quad \quad \quad + C ^ { \prime \prime } K ^ { 5 } \eta ^ { 6 } \frac { M ^ { 4 } } { m } \displaystyle \sum _ { i = 1 } ^ { K - 2 } \mathbb { E } \| \theta _ { t , i } ^ { \pi } - \theta _ { t , i } \| ^ { 2 } + C ^ { \prime \prime } ( K \eta ) ^ { 8 } \displaystyle \frac { M ^ { 2 } } { m } H . } \end{array}\tag{18}
$$

To prove our final claim of this theorem, we use induction to tackle $\begin{array} { r } { \sum _ { i = 1 } ^ { K - 2 } \mathbb { E } \| \theta _ { t , i } ^ { \pi } - \theta _ { t , i } \| ^ { 2 } } \end{array}$ . In specific, we are going to show that there exist constants $C _ { 0 } , C _ { 1 } , C _ { 2 }$ (to be determined later) such that for $K \eta \leq \sqrt { m } / ( C _ { 0 } M ) \leq 1 / ( 2 \sqrt { M } )$

$$
\mathbb { E } \| \theta _ { t , K } - \theta _ { t , K } ^ { \pi } \| ^ { 2 } \leq ( 1 - \frac { m ( K \eta ) ^ { 2 } } { 1 6 } ) \mathbb { E } \| \theta _ { t } - \theta _ { t } ^ { \pi } \| ^ { 2 } + C _ { 1 } K ^ { 2 } \eta ^ { 4 } \frac { M ^ { 2 } } { m } d + C _ { 2 } ( K \eta ) ^ { 8 } \frac { M ^ { 2 } } { m } H .\tag{19}
$$

Firstly, (19) holds for $K = 0 , 1 , 2$ by (18) for a $C _ { 0 }$ with $1 6 C _ { 0 } ^ { 4 } > C ^ { \prime \prime }$

Next, assume that (19) holds for all $K \le \widetilde K - 1$ , we are going to show that (19) also holds for $K = { \widetilde K }$

Applying our induction assumption on (18), we have

$$
\begin{array} { r l } & { \quad \mathbb { E } \| \theta _ { t , \widetilde { K } } - \bar { \mathcal { H } } _ { \eta } ^ { \widetilde { K } } ( \theta _ { t } ) \| ^ { 2 } } \\ & { \le ( 1 - \displaystyle \frac { m ( K \eta ) ^ { 2 } } { 8 } + \frac { C ^ { \prime \prime } ( K \eta ) ^ { 6 } M ^ { 4 } } { m } ) \mathbb { E } \| \theta _ { t } - \theta _ { t } ^ { \pi } \| ^ { 2 } + C ^ { \prime \prime } K ^ { 2 } \eta ^ { 4 } \displaystyle \frac { M ^ { 2 } } { m } d + C ^ { \prime \prime } ( K \eta ) ^ { 8 } \displaystyle \frac { M ^ { 2 } } { m } H } \\ & { \quad + C ^ { \prime \prime } K ^ { 5 } \eta ^ { 6 } \displaystyle \frac { M ^ { 4 } } { m } \displaystyle \sum _ { i = 1 } ^ { K - 2 } \Big ( ( 1 - \displaystyle \frac { m ( i \eta ) ^ { 2 } } { 1 6 } ) \mathbb { E } \| \theta _ { t } - \theta _ { t } ^ { \pi } \| ^ { 2 } + C _ { 1 } i ^ { 2 } \eta ^ { 4 } \displaystyle \frac { M ^ { 2 } } { m } d + C _ { 2 } ( i \eta ) ^ { 8 } \displaystyle \frac { M ^ { 2 } } { m } H \Big ) } \\ & { \le ( 1 - \displaystyle \frac { m ( K \eta ) ^ { 2 } } { 8 } + \displaystyle \frac { C ^ { \prime \prime } ( K \eta ) ^ { 6 } M ^ { 4 } } { m } + C ^ { \prime \prime } ( K \eta ) ^ { 6 } \displaystyle \frac { M ^ { 4 } } { m } ) \mathbb { E } \| \theta _ { t } - \theta _ { t } ^ { \pi } \| ^ { 2 } } \\ &  \quad + ( C ^ { \prime \prime } + C ^ { \prime \prime } K ^ { 6 } \eta ^ { 6 } \displaystyle \frac { M ^ { 4 } } { m } C _ { 1 } ) K ^ { 2 } \eta ^ { 4 } \displaystyle \frac { M ^ { 2 } } { m } d + ( C ^ { \prime \prime } + C ^ { \prime \prime } K ^ { 6 } \eta ^ { 6 } C _ { 2 } ) ( K \eta ) ^ { 8 } \displaystyle \frac  M ^  2 \end{array}
$$

To finish the induction, we need to satisfy that for $K \eta \leq \sqrt { m } / ( C _ { 0 } M )$ ，

$$
\frac { m ( K \eta ) ^ { 2 } } { 8 } - \frac { 2 C ^ { \prime \prime } ( K \eta ) ^ { 6 } M ^ { 4 } } { m } \geq \frac { m ( K \eta ) ^ { 2 } } { 1 6 } , ( C ^ { \prime \prime } + C ^ { \prime \prime } K ^ { 6 } \eta ^ { 6 } \frac { M ^ { 4 } } { m } C _ { 1 } ) \leq C _ { 1 } , ( C ^ { \prime \prime } + C ^ { \prime \prime } K ^ { 6 } \eta ^ { 6 } C _ { 2 } ) \leq C _ { 2 } .
$$

which holds trivally for an enough large constant $C _ { 0 }$

Noting that $M \geq m$ and that $\theta _ { t + 1 } = \theta _ { t , K }$ and $\theta _ { t + 1 } ^ { \pi } = \theta _ { t , K } ^ { \pi }$ , this finishes the proof.

## B.2 ideal process

Lemma B.1 (Lemma D.3 in Liang et al. (2025)). Under Assumption $_ { 4 . 2 }$ consider two HMC systems $\theta ( t )$ and $\bar { \theta } ( t )$ starting with the same initial positions and different initial momentum $p ( 0 )$ and $\widetilde { p } ( 0 )$ respectively. We have

$$
\| \theta ( t ) - \bar { \theta } ( t ) \| _ { 2 } \leq \frac { 1 } { \sqrt { M } } \sinh ( \sqrt { M } t ) \| p ( 0 ) - \widetilde { p } ( 0 ) \| _ { 2 } ,
$$

Further $i f$ we have Assumption $4 . l ,$ consider two HMC systems $\theta ( t )$ and $ { \widetilde { \theta } } ( t )$ starting with initial positions $\theta ( 0 )$ and $\widetilde { \theta } ( 0 )$ , and initial momentum $p ( 0 )$ and $\widetilde { p } ( 0 )$ respectively, then for $\begin{array} { r } { 0 \leq t \leq \frac { 1 } { 2 \sqrt { M } } } \end{array}$ we have

$$
\| \theta ( t ) - \widetilde { \theta } ( t ) \| _ { 2 } \le ( 1 - \frac { m } { 4 } t ^ { 2 } ) \| \theta ( 0 ) - \widetilde { \theta } ( 0 ) \| _ { 2 } + t \| p ( 0 ) - \widetilde { p } ( 0 ) \| _ { 2 } .
$$

Lemma B.2 (Proposition 1 in Durmus & Moulines (2019)). Let $x ^ { * } \in \mathbb { R } ^ { d }$ be the global minimizer of a loss function $f \bar { ( \cdot ) }$ satisfying Assumption $4 . I$ and $x ^ { \pi }$ be a random variable following distribution ∝ $e ^ { - f ( x ) }$ , then

$$
\mathbb { E } \| x ^ { * } - x ^ { \pi } \| _ { 2 } ^ { 2 } \leq \frac { d } { m } .
$$

## B.3 discrete error control

Lemma B.3 (Approximation error on position of classical HMC). Suppose p is the associated initial momentum of position q. Under Assumption 4.2,

$$
\| \mathcal { H } _ { \eta } ( { q } ) - \bar { \mathcal { H } } _ { \eta } ( { q } ) \| ^ { 2 } \leq \eta ^ { 6 } \Big ( \frac { M ^ { 2 } } { 1 0 } \| p \| ^ { 2 } + \frac { M ^ { 4 } \eta ^ { 2 } } { 6 0 } \| q - \theta ^ { * } \| ^ { 2 } \Big ) .
$$

Proof. By definitions,

$$
\mathcal { H } _ { \eta } ( q ) - \bar { \mathcal { H } } _ { \eta } ( q ) = q + \eta p - \frac { \eta ^ { 2 } } { 2 } \nabla f ( q ) - ( q + \int _ { 0 } ^ { \eta } p ( s ) d s ) , \qquad \mathrm { w i t h } \qquad p ( s ) = p - \int _ { 0 } ^ { s } \nabla f ( q ( u ) ) d u
$$

Inserting $\begin{array} { r } { p ( s ) = p - \int _ { 0 } ^ { s } \nabla f ( q ( u ) ) \ d u } \end{array}$ )du and $\begin{array} { r } { \frac { \eta ^ { 2 } } { 2 } \nabla f ( q ) = \int _ { 0 } ^ { \eta } \int _ { 0 } ^ { s } \nabla f ( q ) d u d s } \end{array}$ , we get

$$
\mathcal { H } _ { \eta } ( { q } ) - \bar { \mathcal { H } } _ { \eta } ( { q } ) = \int _ { 0 } ^ { \eta } \int _ { 0 } ^ { s } \nabla f ( { q } ( u ) ) - \nabla f ( { q } ) d u d s
$$

By Jensen inequality (i.e., Lemma B.12 with $\lambda \equiv 1 )$ and M-smoothness assumption of $\nabla f ,$

$$
\| \mathcal { H } _ { \eta } ( q ) - \bar { \mathcal { H } } _ { \eta } ( q ) \| ^ { 2 } \leq \frac { \eta ^ { 2 } } { 2 } \int _ { 0 } ^ { \eta } \int _ { 0 } ^ { s } M ^ { 2 } \| q ( u ) - q \| ^ { 2 } d u d s .\tag{20}
$$

In the following, we are going to apply Lemma B.1 to bound $\| q ( u ) - q \| ^ { 2 }$ . First, by Cauchy–Schwarz inequality and Lemma B.12

$$
\begin{array} { r l r } {  { \| q ( u ) - q \| ^ { 2 } = \| u p - \int _ { 0 } ^ { u } \int _ { 0 } ^ { s } \nabla f ( q ( t ) ) d t d s \| ^ { 2 } \leq 2 \| u p \| ^ { 2 } + 2 \| \int _ { 0 } ^ { u } \int _ { 0 } ^ { s } \nabla f ( q ( t ) ) d t d s \| ^ { 2 } } } \\ & { } & { \leq 2 \| u p \| ^ { 2 } + u ^ { 2 } \int _ { 0 } ^ { u } \int _ { 0 } ^ { s } \| \nabla f ( q ( t ) ) \| ^ { 2 } d t d s } \end{array}
$$

Inserting result from Lemma B.5 that $\| \nabla f ( q ( t ) ) \| ^ { 2 } \leq 2 M ^ { 2 } ( \| q - \theta ^ { * } \| ^ { 2 } + t ^ { 2 } \| p \| ^ { 2 } ) ^ { 1 }$ into the last inequality, we have

$$
\| q ( u ) - q \| ^ { 2 } \leq ( 2 u ^ { 2 } + \frac { M ^ { 2 } u ^ { 6 } } { 6 } ) \| p \| ^ { 2 } + M ^ { 2 } u ^ { 4 } \| q - \theta ^ { * } \| ^ { 2 }
$$

Since $u \leq \eta \leq 1 / ( 2 \sqrt { M } )$ , we also have

$$
\lVert q ( u ) - q \rVert ^ { 2 } \leq \frac { 2 4 } { 1 0 } u ^ { 2 } \lVert p \rVert ^ { 2 } + M ^ { 2 } u ^ { 4 } \lVert q - { \theta } ^ { * } \rVert ^ { 2 }
$$

and inserting back to (20) yields the claim.

Lemma B.4 (Approximation error on momentum of classical HMC). Suppose $q$ is the associated initial position ofmomentum $p .$ Under Assumption 4.2 and $\eta \leq 1 / ( 2 \sqrt { M } )$ ,

$$
\| \mathcal { H } _ { \eta } ( p ) - \bar { \mathcal { H } } _ { \eta } ( p ) \| ^ { 2 } \leq \frac { 5 } { 2 } \eta ^ { 4 } M ^ { 2 } \| p \| ^ { 2 } + \frac { 2 } { 3 } \eta ^ { 6 } M ^ { 4 } \| q - \theta ^ { * } \| ^ { 2 }
$$

Proof. By definitions,

$$
\mathcal { H } _ { \eta } ( p ) - \bar { \mathcal { H } } _ { \eta } ( p ) = p - \frac { \eta } { 2 } ( \nabla f ( q ) + \nabla f ( \mathcal { H } _ { \eta } ( q ) ) ) - ( p - \int _ { 0 } ^ { \eta } \nabla f ( q ( s ) ) d s )
$$

Inserting $\begin{array} { r } { \nabla f ( q ( s ) ) = \nabla f ( q ) + \int _ { 0 } ^ { s } \nabla ^ { 2 } f ( q ( u ) ) p ( u ) d u } \end{array}$ into above, we get

$$
\mathcal { H } _ { \eta } ( p ) - \bar { \mathcal { H } } _ { \eta } ( p ) = \frac { \eta } { 2 } ( \nabla f ( q ) - \nabla f ( \mathcal { H } _ { \eta } ( q ) ) ) + \int _ { 0 } ^ { \eta } \int _ { 0 } ^ { s } \nabla ^ { 2 } f ( q ( u ) ) p ( u ) d u d s
$$

Applying Lemma B.12, we can get

$$
\begin{array} { r l r } {  { \Vert \mathcal { H } _ { \eta } ( p ) - \bar { \mathcal { H } } _ { \eta } ( p ) \Vert ^ { 2 } \leq 2 \Vert \frac { \eta } { 2 } ( \nabla f ( q ) - \nabla f ( \mathcal { H } _ { \eta } ( q ) ) ) \Vert ^ { 2 } + 2 \Vert \int _ { 0 } ^ { \eta } \int _ { 0 } ^ { s } \nabla ^ { 2 } f ( q ( u ) ) p ( u ) d u d s \Vert ^ { 2 } } } \\ & { } & { \leq 2 \Vert \frac { \eta } { 2 } ( \nabla f ( q ) - \nabla f ( \mathcal { H } _ { \eta } ( q ) ) ) \Vert ^ { 2 } + 2 \frac { \eta ^ { 2 } } { 2 } \int _ { 0 } ^ { \eta } \int _ { 0 } ^ { s } \Vert \nabla ^ { 2 } f ( q ( u ) ) p ( u ) \Vert ^ { 2 } d u d s } \\ & { } & { = \frac { \eta ^ { 2 } } { 2 } \Vert \nabla f ( q ) - \nabla f ( \mathcal { H } _ { \eta } ( q ) ) \Vert ^ { 2 } + \eta ^ { 2 } \int _ { 0 } ^ { \eta } \int _ { 0 } ^ { s } \Vert \nabla ^ { 2 } f ( q ( u ) ) p ( u ) \Vert ^ { 2 } d u d s . } \end{array}
$$

Since f satisfies M-smoothness assumption, we further have

$$
\| \mathcal { H } _ { \eta } ( p ) - \bar { \mathcal { H } } _ { \eta } ( p ) \| ^ { 2 } \leq \frac { \eta ^ { 2 } M ^ { 2 } } { 2 } \| q - \mathcal { H } _ { \eta } ( q ) \| ^ { 2 } + \eta ^ { 2 } M ^ { 2 } \int _ { 0 } ^ { \eta } \int _ { 0 } ^ { s } \| p ( u ) \| ^ { 2 } d u d s\tag{21}
$$

To bound the RHS, setting $F = f$ in Lemma B.8 gives

$$
\begin{array} { r } { \| q - \mathcal { H } _ { \eta } ( q ) \| ^ { 2 } \leq 2 \eta ^ { 2 } \| p \| ^ { 2 } + \eta ^ { 4 } M ^ { 2 } \| q - \theta ^ { * } \| ^ { 2 } } \end{array}
$$

and at the same time,

$$
\begin{array} { l } { \displaystyle \| p ( u ) \| ^ { 2 } = \| p - \displaystyle \int _ { 0 } ^ { u } \nabla f ( q ( s ) ) d s \| ^ { 2 } \leq 2 \| p \| ^ { 2 } + 2 \| \displaystyle \int _ { 0 } ^ { u } \nabla f ( q ( s ) ) d s \| ^ { 2 } } \\ { \displaystyle \leq 2 \| p \| ^ { 2 } + 2 u \displaystyle \int _ { 0 } ^ { u } \| \nabla f ( q ( s ) ) \| ^ { 2 } d s \leq 2 \| p \| ^ { 2 } + 2 u \displaystyle \int _ { 0 } ^ { u } 2 M ^ { 2 } ( \| q - \theta ^ { * } \| ^ { 2 } + s ^ { 2 } \| p \| ^ { 2 } ) d s } \\ { \displaystyle = ( 2 + \frac { u ^ { 4 } M ^ { 2 } } { 3 } ) \| p \| ^ { 2 } + 4 u ^ { 2 } M ^ { 2 } \| q - \theta ^ { * } \| ^ { 2 } \leq 3 \| p \| ^ { 2 } + 4 u ^ { 2 } M ^ { 2 } \| q - \theta ^ { * } \| ^ { 2 } , } \end{array}
$$

where the first inequality is by Cauthy-Schwarz inequality, the second inequality is by Lemma B.12, the third inequality is by Lemma B.5, and the last inequality is by $u \leq \eta \leq 1 / 2 \sqrt { M }$

Inserting these back to (21) yields

$$
\| \mathcal { H } _ { \eta } ( p ) - \bar { \mathcal { H } } _ { \eta } ( p ) \| ^ { 2 } \leq \frac { 5 } { 2 } \eta ^ { 4 } M ^ { 2 } \| p \| ^ { 2 } + \frac { 2 } { 3 } \eta ^ { 6 } M ^ { 4 } \| q - \theta ^ { * } \| ^ { 2 } .
$$

This finishes the proof.

## B.4 Lemmas

Lemma B.5. Suppose p is the associated initial momentum of position q. Under Assumption 4.2

$$
\| \nabla f ( q ) \| ^ { 2 } \leq M ^ { 2 } \| q - \theta ^ { * } \| ^ { 2 } , \qquad \| \nabla f ( q ( t ) ) \| ^ { 2 } \leq 2 M ^ { 2 } ( \| q - \theta ^ { * } \| ^ { 2 } + t ^ { 2 } \| p \| ^ { 2 } )
$$

Proof. Noting that $\| \nabla f ( q ( t ) ) \| ^ { 2 } = \| \nabla f ( q ( t ) ) - \nabla f ( \theta ^ { * } ) \| ^ { 2 } \leq M ^ { 2 } \| q ( t ) - \theta ^ { * } \| ^ { 2 }$ , and by Lemma B.1

$$
\| q ( t ) - \theta ^ { * } \| \leq \| q - \theta ^ { * } \| + t \| p - 0 \| .
$$

Noting that f satisfies M-smoothness assumption

$$
\| \nabla f ( q ( t ) ) \| ^ { 2 } \leq M ^ { 2 } \| q ( t ) - \theta ^ { * } \| ^ { 2 } \leq M ^ { 2 } ( \| q - \theta ^ { * } \| + t \| p \| ) ^ { 2 } .
$$

We finish the proof by applying Cauchy–Schwarz inequality on the RHS.

Lemma B.6. For any $q , p , q ^ { \pi } \in \mathbb { R } ^ { d }$ , let $\begin{array} { r } { q _ { k } ^ { ( c ) } = \widetilde { h } _ { \mathrm { L F } } ( f ^ { ( c ) } , q , \nabla f ( q ) , p _ { 0 } , \eta , k ) , q _ { k } = \sum _ { c = 1 } ^ { N } w _ { c } q _ { k } ^ { ( c ) } } \end{array}$ and $q _ { k } ^ { \pi } = \bar { \mathcal { H } } _ { \eta } ^ { k } ( q ^ { \pi } )$ . Under the smoothness Assumption 4.2, we have

$$
\sum _ { c = 1 } ^ { N } w _ { c } \mathbb { E } \| q _ { k } ^ { ( c ) } - q _ { k } \| ^ { 2 } \leq 2 k ^ { 3 } \eta ^ { 4 } \sum _ { i = 1 } ^ { k - 1 } ( 1 6 M ^ { 2 } \mathbb { E } \| q _ { i } ^ { \pi } - q _ { i } \| ^ { 2 } + 1 6 M ^ { 2 } \mathbb { E } \| q ^ { \pi } - q \| ^ { 2 } + 4 ( k \eta ) ^ { 2 } H )
$$

where $\begin{array} { r } { H : = \operatorname* { m a x } _ { i \leq K } \sum _ { c = 1 } ^ { N } w _ { c } \mathbb { E } \| \frac { 1 } { i \eta } ( \nabla f ^ { ( c ) } ( q _ { i } ^ { \pi } ) - \nabla f ^ { ( c ) } ( q ^ { \pi } ) + \nabla f ( q ^ { \pi } ) - \nabla f ( q _ { i } ^ { \pi } ) ) \| ^ { 2 } } \end{array}$ is a quantity measuring thefederated heterogeneity.

Proof. To prove this lemma, we use mathematical induction to remove the discrete heterogeneous term $\begin{array} { r } { \sum _ { c = 1 } ^ { N } w _ { c } \mathbb { E } \| q _ { k } ^ { ( c ) } - q _ { k } \| ^ { 2 } } \end{array}$ . In specific, we are going to use induction to prove that there are constants $C _ { 1 } , C _ { 2 }$ (to be determined later) such that

$$
\sum _ { c = 1 } ^ { N } w _ { c } \mathbb { E } \| q _ { j } ^ { ( c ) } - q _ { j } \| ^ { 2 } \leq ( j - 1 ) ^ { 3 } \eta ^ { 4 } \sum _ { k = 1 } ^ { j - 1 } ( C _ { 1 } M ^ { 2 } \mathbb { E } \| q _ { k } ^ { \pi } - q _ { k } \| ^ { 2 } + C _ { 1 } M ^ { 2 } \mathbb { E } \| q ^ { \pi } - q \| ^ { 2 } + C _ { 2 } ( k \eta ) ^ { 2 } H ) .\tag{22}
$$

First, for $j = 1$ , noting that $\begin{array} { r } { q _ { 1 } ^ { ( c ) } = q + p - \frac { \eta } { 2 } ( \nabla f ^ { ( c ) } ( q ) - \nabla f ^ { ( c ) } ( q ) + \nabla f ( q ) ) } \end{array}$ is constant for all $c ,$ thus

$$
\sum _ { c = 1 } ^ { N } w _ { c } \mathbb { E } \| q _ { j } ^ { ( c ) } - q _ { j } \| ^ { 2 } = 0 , \qquad { \mathrm { w h e n ~ } } j = 1 .
$$

Now, assume that (22) holds for $j \leq k$ . Next, we check it also holds for $k + 1$ based on the assumption. By Lemma B.7, we have

$$
\begin{array} { r l } & { \underset { c = 1 } { \overset { N } { \sum } } w _ { c } \mathbb { E } \| q _ { k + 1 } ^ { ( c ) } - q _ { k + 1 } \| ^ { 2 } } \\ & { \leq 2 ( k + 1 ) ^ { 3 } \eta ^ { 4 } \underset { i = 1 } { \overset { k } { \sum } } \Big ( 4 M ^ { 2 } \underset { c = 1 } { \overset { N } { \sum } } w _ { c } \mathbb { E } \| q _ { i } ^ { ( c ) } - q _ { i } \| ^ { 2 } + 4 M ^ { 2 } \mathbb { E } \| q _ { i } ^ { \pi } - q _ { i } \| ^ { 2 } + 4 M ^ { 2 } \mathbb { E } \| q ^ { \pi } - q \| ^ { 2 } + ( i \eta ) ^ { 2 } H \Big ) } \\ & { \leq 2 ( k + 1 ) ^ { 3 } \eta ^ { 4 } \underset { i = 1 } { \overset { k } { \sum } } \Big ( 4 M ^ { 2 } t ^ { 3 } \eta ^ { 4 } \underset { j = 1 } { \overset { i - 1 } { \sum } } ( C _ { 1 } M ^ { 2 } \mathbb { E } \| q _ { j } ^ { \pi } - q _ { j } \| ^ { 2 } + C _ { 1 } M ^ { 2 } \mathbb { E } \| q ^ { \pi } - q \| ^ { 2 } + C _ { 2 } ( j \eta ) ^ { 2 } H ) \Big ) } \\ & { \quad + 2 ( k + 1 ) ^ { 3 } \eta ^ { 4 } \underset { i = 1 } { \overset { k } { \sum } } \Big ( 4 M ^ { 2 } \mathbb { E } \| q _ { i } ^ { \pi } - q _ { s } \| ^ { 2 } + 4 M ^ { 2 } \mathbb { E } \| q ^ { \pi } - q \| ^ { 2 } + ( i \eta ) ^ { 2 } H \Big ) } \end{array}
$$

where the second inequality is by our induction assumption for $i \leq k$

Noting that for any sequence $\{ a _ { j } \}$

$$
\sum _ { i = 1 } ^ { k } i ^ { 3 } \sum _ { j = 1 } ^ { i - 1 } a _ { j } = \sum _ { j = 1 } ^ { k - 1 } \sum _ { i = j + 1 } ^ { k } i ^ { 3 } a _ { j } \leq \sum _ { j = 1 } ^ { k - 1 } k ^ { 4 } a _ { j }
$$

Therefore, we further have

$$
\begin{array} { r l r } {  { \sum _ { c = 1 } ^ { N } w _ { c } \mathbb { E } \| q _ { k + 1 } ^ { ( c ) } - q _ { k + 1 } \| ^ { 2 } \le ( k + 1 ) ^ { 3 } \eta ^ { 4 } \sum _ { j = 1 } ^ { k } \Bigl ( ( 8 M ^ { 4 } C _ { 1 } k ^ { 4 } \eta ^ { 4 } + 8 M ^ { 2 } ) \mathbb { E } \| q _ { j } ^ { \pi } - q _ { j } \| ^ { 2 } } } \\ & { } & { \quad \quad \quad \quad \quad \quad \quad \quad \quad + ( 8 M ^ { 4 } C _ { 1 } k ^ { 4 } \eta ^ { 4 } + 8 M ^ { 2 } ) \mathbb { E } \| q ^ { \pi } - q \| ^ { 2 } + ( 8 M ^ { 2 } C _ { 2 } k ^ { 4 } \eta ^ { 4 } + 2 ) H \Bigr ) } \end{array}
$$

By choosing $C _ { 1 } , C _ { 2 }$ such that

$$
8 M ^ { 4 } C _ { 1 } k ^ { 4 } \eta ^ { 4 } + 8 M ^ { 2 } \le C _ { 1 } M ^ { 2 } , \qquad 8 M ^ { 2 } C _ { 2 } k ^ { 4 } \eta ^ { 4 } + 2 \le C _ { 2 } ,
$$

we can finish the induction. Recall that $K \eta$ is set to be $\leq 1 / ( 2 \sqrt { M } )$ . One choice is that $C _ { 1 } =$ $1 6 , C _ { 2 } = 4$ . This finishes the proof. □

Lemma B.7. Under the same notations and conditions ofthat ofLemma B.6,

$$
\sum _ { c = 1 } ^ { N } w _ { c } \mathbb { E } \| q _ { j } ^ { ( c ) } - q _ { j } \| ^ { 2 } \leq 2 j ^ { 3 } \eta ^ { 4 } \sum _ { k = 1 } ^ { j - 1 } \Bigl ( 4 M ^ { 2 } \sum _ { c = 1 } ^ { N } w _ { c } \mathbb { E } \| q _ { k } ^ { ( c ) } - q _ { k } \| ^ { 2 } + 4 M ^ { 2 } \mathbb { E } \| q _ { k } ^ { \pi } - q _ { k } \| ^ { 2 } + 4 M ^ { 2 } \mathbb { E } \| q ^ { \pi } - q \| ^ { 2 } + ( k \eta ) ^ { 2 } H \Bigr )
$$

Proof. By definitions,

$$
\begin{array} { l } { \displaystyle q _ { j } ^ { ( c ) } - q _ { j } = q + j \eta p _ { 0 } - \frac { ( j \eta ) ^ { 2 } } { 2 } \nabla f ( q ) - \eta ^ { 2 } \sum _ { k = 1 } ^ { j - 1 } \sum _ { i = 1 } ^ { k } ( \nabla f ^ { ( c ) } ( q _ { i } ^ { ( c ) } ) - \nabla f ^ { ( c ) } ( q ) ) } \\ { \displaystyle - \sum _ { c = 1 } ^ { N } w _ { c } \Big ( q + j \eta p _ { 0 } - \frac { ( j \eta ) ^ { 2 } } { 2 } \nabla f ( q ) - \eta ^ { 2 } \sum _ { k = 1 } ^ { j - 1 } \sum _ { i = 1 } ^ { k } ( \nabla f ^ { ( c ) } ( q _ { i } ^ { ( c ) } ) - \nabla f ^ { ( c ) } ( q ) ) \Big ) } \\ { \displaystyle = \eta ^ { 2 } \sum _ { k = 1 } ^ { j - 1 } \sum _ { i = 1 } ^ { k } \Big ( \sum _ { \widetilde { c } = 1 } ^ { N } w _ { \widetilde { c } } ( \nabla f ^ { ( \overline { { c } } ) } ( q _ { i } ^ { ( \overline { { c } } ) } ) - \nabla f ^ { ( \overline { { c } } ) } ( q ) ) - \nabla f ^ { ( c ) } ( q _ { i } ^ { ( c ) } ) + \nabla f ^ { ( c ) } ( q ) \Big ) . } \end{array}
$$

By Lemma B.12, we have

$$
\begin{array} { r } { \| q _ { j } ^ { ( c ) } - q _ { j } \| ^ { 2 } \leq \displaystyle \frac { j ( j - 1 ) } { 2 } \sum _ { k = 1 } ^ { j - 1 } \displaystyle \sum _ { i = 1 } ^ { k } \| \eta ^ { 2 } \sum _ { \widetilde { c } = 1 } ^ { N } w _ { \widetilde { c } } ( \nabla f ^ { ( \widetilde { c } ) } ( q _ { i } ^ { ( \widetilde { c } ) } ) - \nabla f ^ { ( \widetilde { c } ) } ( q ) ) - \nabla f ^ { ( c ) } ( q _ { i } ^ { ( c ) } ) + \nabla f ^ { ( c ) } ( q ) \| ^ { 2 } } \\ { \displaystyle \qquad = \frac { j ( j - 1 ) } { 2 } \sum _ { i = 1 } ^ { j - 1 } \displaystyle \sum _ { k = i } ^ { j - 1 } \eta ^ { 4 } \| \sum _ { \widetilde { c } = 1 } ^ { N } w _ { \widetilde { c } } ( \nabla f ^ { ( \widetilde { c } ) } ( q _ { i } ^ { ( \widetilde { c } ) } ) - \nabla f ^ { ( \widetilde { c } ) } ( q ) ) - \nabla f ^ { ( c ) } ( q _ { i } ^ { ( c ) } ) + \nabla f ^ { ( c ) } ( q ) \| ^ { 2 } } \end{array}
$$

It follows that

$$
\| q _ { j } ^ { ( c ) } - q _ { j } \| ^ { 2 } \leq \frac { j ^ { 3 } \eta ^ { 4 } } { 2 } \sum _ { i = 1 } ^ { j - 1 } \| \sum _ { \tilde { c } = 1 } ^ { N } w _ { \tilde { c } } ( \nabla f ^ { ( \tilde { c } ) } ( q _ { i } ^ { ( \tilde { c } ) } ) - \nabla f ^ { ( \tilde { c } ) } ( q ) ) - \nabla f ^ { ( c ) } ( q _ { i } ^ { ( c ) } ) + \nabla f ^ { ( c ) } ( q ) \| ^ { 2 } .\tag{23}
$$

The error $\begin{array} { r } { \sum _ { \tilde { c } = 1 } ^ { N } w _ { \tilde { c } } ( \nabla f ^ { ( \tilde { c } ) } ( q _ { i } ^ { ( \tilde { c } ) } ) - \nabla f ^ { ( \tilde { c } ) } ( q ) ) - \nabla f ^ { ( c ) } ( q _ { i } ^ { ( c ) } ) + \nabla f ^ { ( c ) } ( q ) ) } \end{array}$ can be split into several part as follow

$$
\begin{array} { r l } & { \underset { \mathrm { e } = 1 } { \overset { N } { \sum } } u _ { \alpha } ( \nabla f ^ { ( i ) } ( \mathfrak { g } _ { i } ^ { ( i ) } ) - \nabla f ^ { ( i ) } ( \mathfrak { g } ) ) - \nabla f ^ { ( i ) } ( \mathfrak { g } _ { i } ^ { ( i ) } ) + \nabla f ^ { ( i ) } ( \mathfrak { g } ) } \\ & { = \underset { \mathrm { e } = 1 } { \overset { N } { \sum } } u _ { \alpha } ( \nabla f ^ { ( i ) } ( \mathfrak { g } _ { i } ^ { ( i ) } ) - \nabla f ^ { ( i ) } ( \mathfrak { g } _ { i } ) ) - \nabla f ^ { ( i ) } ( \mathfrak { g } _ { i } ^ { ( i ) } ) + \nabla f ^ { ( i ) } ( \mathfrak { g } _ { i } ) } \\ & { \overset { \mathrm { ( E q . ~ ) } } { = } 1 } \\ & { \underset { \mathrm { d a s t ~ c e ~ h e t e r w a g e a t i t y } } { \overset { N } { \sum } } ( \underbrace { q _ { i } \mathfrak { g } _ { i } ^ { ( i ) } } _ { \mathrm { e } \times \mathrm { e } \times \mathrm { e } \times \mathrm { e x p e r w a g e a t i t y } } ) } \\ & { + \underset { \mathrm { V o r ~ o p . ~ e x p e r w a g e a t i t y } } { \overset { N } { \sum } } ( \underbrace { q _ { i } \mathfrak { g } _ { i } ^ { ( i ) } } _ { \mathrm { e } \times \mathrm { e } \times \mathrm { e x p e r w a g e a t i t y } } - \nabla f ( \mathfrak { g } _ { i } ^ { ( i ) } ) } \\ & { + \underset { \mathrm { V o r ~ o p . ~ e x p e r w a g e ~ e x p e r w a g e } } { \overset { N } { \sum } } ( \mathfrak { g } ^ { ( i ) } ) - \nabla f ^ { ( i ) } ( \mathfrak { g } ^ { ( i ) } ) } \\ & { + \underset { \mathrm { V o r ~ o p . ~ e x p e r w a g e ~ e x p e r w a g e } } { \overset { N } { \sum } } ( \mathfrak { g } ^ { ( i ) } ) + \nabla f ^ { ( i ) } ( \mathfrak { g } ^ { ( i ) } ) } \\ &  + \underset  \mathrm  V o r ~ o p  \end{array}
$$

Again, by Lemma B.12,

$$
\| \sum _ { \tilde { c } = 1 } ^ { N } w _ { \tilde { c } } ( \nabla f ^ { ( \tilde { c } ) } ( q _ { i } ^ { ( \tilde { c } ) } ) - \nabla f ^ { ( \tilde { c } ) } ( q ) ) - \nabla f ^ { ( c ) } ( q _ { i } ^ { ( c ) } ) + \nabla f ^ { ( c ) } ( q ) \| ^ { 2 } \leq 4 ( A ^ { ( c ) } + B ^ { ( c ) } + C ^ { ( c ) } + D ^ { ( c ) } )\tag{24}
$$

where

$$
\begin{array} { r l } & { \displaystyle \boldsymbol { A } ^ { ( c ) } : \| \sum _ { \widetilde { c } = 1 } ^ { N } w _ { \widetilde { c } } ( \nabla f ^ { ( \widetilde { c } ) } ( q _ { i } ^ { ( \widetilde { c } ) } ) - \nabla f ^ { ( \widetilde { c } ) } ( q _ { i } ) ) - \nabla f ^ { ( \widetilde { c } ) } ( q _ { i } ^ { ( c ) } ) + \nabla f ^ { ( c ) } ( q _ { i } ) \| ^ { 2 } , } \\ & { \displaystyle \boldsymbol { B } ^ { ( c ) } : \| \nabla f ^ { ( c ) } ( q _ { i } ^ { \pi } ) - \nabla f ^ { ( c ) } ( q _ { i } ) + \nabla f ( q _ { i } ) - \nabla f ( q ^ { \pi } ) \| ^ { 2 } } \\ & { \displaystyle \boldsymbol { C } ^ { ( c ) } : \| \nabla f ^ { ( c ) } ( q ) - \nabla f ^ { ( c ) } ( q ^ { \pi } ) + \nabla f ( q ^ { \pi } ) - \nabla f ( q ) \| ^ { 2 } , } \\ & { \boldsymbol { D } ^ { ( c ) } : \| \nabla f ^ { ( c ) } ( q ^ { \pi } ) - \nabla f ^ { ( c ) } ( q _ { i } ^ { \pi } ) + \nabla f ( q _ { i } ^ { \pi } ) - \nabla f ( q ^ { \pi } ) \| ^ { 2 } . } \end{array}
$$

Consider $A ^ { ( c ) }$ first, observe that

$$
\begin{array} { c } { { \displaystyle \sum _ { \widetilde { c } = 1 } ^ { N } w _ { \widetilde { c } } ( \nabla f ^ { ( \widetilde { c } ) } ( q _ { i } ^ { ( \widetilde { c } ) } ) - \nabla f ^ { ( \widetilde { c } ) } ( q _ { i } ) ) - \nabla f ^ { ( \widetilde { c } ) } ( q _ { i } ^ { ( c ) } ) + \nabla f ^ { ( c ) } ( q _ { i } ) } } \\ { { = \displaystyle \sum _ { \widetilde { c } = 1 } ^ { N } \displaystyle \frac { w _ { \widetilde { c } } - \mathbb { I } ( \widetilde { c } = c ) } { 2 ( 1 - w _ { c } ) } \cdot 2 ( 1 - w _ { c } ) \Big ( \nabla f ^ { ( \widetilde { c } ) } ( q _ { k } ^ { ( c ) } ) - \nabla f ^ { ( \widetilde { c } ) } ( q _ { k } ) \Big ) } } \end{array}
$$

and that $\begin{array} { r } { \sum _ { \widetilde { c } = 1 } ^ { N } \frac { | w _ { \widetilde { c } } - \mathbb { I } ( \widetilde { c } = c ) | } { 2 ( 1 - w _ { c } ) } = 1 } \end{array}$ , by Lemma B.12

$$
\begin{array} { l } { \displaystyle \sum _ { c = 1 } ^ { N } w _ { c } A ^ { ( c ) } \leq \displaystyle \sum _ { c = 1 } ^ { N } w _ { c } \displaystyle \sum _ { \tilde { c } = 1 } ^ { N } | w _ { c } - \mathbb { I } ( \tilde { c } = c ) | 2 ( 1 - w _ { c } ) \| \nabla f ^ { ( \tilde { c } ) } ( q _ { k } ^ { ( c ) } ) - \nabla f ^ { ( \tilde { c } ) } ( q _ { k } ) \| ^ { 2 } } \\ { \displaystyle \qquad = 2 \sum _ { \tilde { c } = 1 } ^ { N } w _ { \tilde { c } } ( \displaystyle \sum _ { \zeta \neq \tilde { c } } ^ { N } w _ { c } ( 1 - w _ { c } ) + ( 1 - w _ { \tilde { c } } ) ^ { 2 } ) \| \nabla f ^ { ( \tilde { c } ) } ( q _ { k } ^ { ( c ) } ) - \nabla f ^ { ( \tilde { c } ) } ( q _ { k } ) \| ^ { 2 } } \\ { \displaystyle \qquad \leq 4 \sum _ { \tilde { c } = 1 } ^ { N } w _ { \tilde { c } } \| \nabla f ^ { ( \tilde { c } ) } ( q _ { k } ^ { ( c ) } ) - \nabla f ^ { ( \tilde { c } ) } ( q _ { k } ) \| ^ { 2 } } \end{array}
$$

Further, by smoothness assumption,

$$
\sum _ { c = 1 } ^ { N } w _ { c } A ^ { ( c ) } \leq 4 \sum _ { c = 1 } ^ { N } w _ { c } M ^ { 2 } \| q _ { k } ^ { ( c ) } - q _ { k } \| ^ { 2 }\tag{25}
$$

Similarly, for $B ^ { ( c ) }$ and $C ^ { ( c ) }$ , we have

$$
\sum _ { c = 1 } ^ { N } w _ { c } B ^ { ( c ) } \leq 4 \sum _ { c = 1 } ^ { N } w _ { c } M ^ { 2 } \| q _ { k } ^ { \pi } - q _ { k } \| ^ { 2 } = 4 M ^ { 2 } \| q _ { k } ^ { \pi } - q _ { k } \| ^ { 2 }\tag{26}
$$

$$
\sum _ { c = 1 } ^ { N } w _ { c } C ^ { ( c ) } \leq 4 \sum _ { c = 1 } ^ { N } w _ { c } M ^ { 2 } \Vert q ^ { \pi } - q \Vert ^ { 2 } = 4 M ^ { 2 } \Vert q ^ { \pi } - q \Vert ^ { 2 }
$$

For $D ^ { ( c ) }$ , by definitions, we directly have

$$
\sum _ { c = 1 } ^ { N } w _ { c } \mathbb { E } D ^ { ( c ) } \leq ( k \eta ) ^ { 2 } H\tag{27}
$$

Combining (23)-(27), we have

$$
\sum _ { c = 1 } ^ { N } w _ { c } \mathbb { E } \| q _ { j } ^ { ( c ) } - q _ { j } \| ^ { 2 } \leq 2 j ^ { 3 } \eta ^ { 4 } \sum _ { k = 1 } ^ { j - 1 } \Bigl ( 4 M ^ { 2 } \sum _ { c = 1 } ^ { N } w _ { c } \mathbb { E } \| q _ { k } ^ { ( c ) } - q _ { k } \| ^ { 2 } + 4 M ^ { 2 } \mathbb { E } \| q _ { k } ^ { \pi } - q _ { k } \| ^ { 2 } + 4 M ^ { 2 } \mathbb { E } \| q ^ { \pi } - q \| ^ { 2 } + ( k \eta ) ^ { 2 } H \Bigr )
$$

This finishes the proof.

□

Lemma B.8 (0-th order Approximation Bound). Let $q _ { 0 } , p _ { 0 } \in \mathbb { R } ^ { d }$ be two vectors. For any $F ( \cdot )$ $\mathbb { R } ^ { d } \mapsto \mathbb { R }$ satisfying smoothness Assumption 4.2, then $\boldsymbol { q } _ { K } = \dot { h } _ { \mathrm { L F } } ( \boldsymbol { F } , \boldsymbol { q } _ { 0 } , \nabla f ( \boldsymbol { q } _ { 0 } ) , p _ { 0 } , \eta , k )$ satisfies

$$
\| q _ { k } - q _ { 0 } \| ^ { 2 } \leq 2 ( k \eta ) ^ { 2 } \| p _ { 0 } \| ^ { 2 } + ( k \eta ) ^ { 4 } M ^ { 2 } \| q _ { 0 } - \theta ^ { * } \| ^ { 2 }\tag{28}
$$

$$
\begin{array} { r } { \| \nabla F ( q _ { k } ) - \nabla F ( q _ { 0 } ) \| ^ { 2 } \leq 2 M ^ { 2 } ( k \eta ) ^ { 2 } \| p _ { 0 } \| ^ { 2 } + ( k \eta ) ^ { 4 } M ^ { 4 } \| q _ { 0 } - \theta ^ { * } \| ^ { 2 } } \end{array}\tag{29}
$$

Note that we actually prove 30, which is an interesting result in the sense that the k-th leapfrog step can be bounded by the shift vector $\nabla f ( q _ { 0 } )$ rather than $\nabla F ( q _ { 0 } )$ .

Proof. We are going to prove

$$
\begin{array} { r } { \| q _ { k } - q _ { 0 } \| ^ { 2 } \leq 2 ( k \eta ) ^ { 2 } \| p _ { 0 } \| ^ { 2 } + ( k \eta ) ^ { 4 } \| \nabla f ( q _ { 0 } ) \| ^ { 2 } . } \end{array}\tag{30}
$$

We prove claim (30) by induction. Then the claim (28) and (29) follows directly from (30) and M-smoothness Assumption 4.2 applied with $\nabla f ( \theta ^ { * } ) = 0$

First, according to the definitions and Algorithm 1, we can write the full expression of $q _ { k }$ as follow

$$
\begin{array} { r } { q _ { k } = q _ { 0 } + k \eta p _ { 0 } - \frac { ( k \eta ) ^ { 2 } } { 2 } \nabla f ( q _ { 0 } ) - \eta ^ { 2 } { \displaystyle \sum _ { j = 1 } ^ { k - 1 } } ( k - j ) ( \nabla F ( q _ { j } ) - \nabla F ( q _ { 0 } ) ) } \end{array}
$$

which is proved by Lemma B.9.

For claim (30) with $k = 1$ , by Cauchy–Schwarz inequality, we have that

$$
\| q _ { 1 } - q _ { 0 } \| ^ { 2 } = \| \eta p _ { 0 } - \frac { \eta ^ { 2 } } { 2 } \nabla f ( q _ { 0 } ) \| ^ { 2 } \leq 2 \| \eta p _ { 0 } \| ^ { 2 } + 2 \| \frac { \eta ^ { 2 } } { 2 } \nabla f ( q _ { 0 } ) \| ^ { 2 }
$$

Next, we assume that the claim (30) holds for $j = 1 , 2 , \dots , k - 1$ . By direct calculations, for any $c _ { 1 } , c _ { 2 } , c _ { 3 } > 0$ such that $\textstyle \sum _ { i } c _ { i } = 1$ we have

$$
\begin{array} { l } { { \displaystyle \| q _ { k } - q _ { 0 } \| ^ { 2 } } } \\ { { \displaystyle \le \frac { 1 } { c _ { 1 } } ( k \eta ) ^ { 2 } \| p _ { 0 } \| ^ { 2 } + \frac { ( k \eta ) ^ { 4 } } { 4 c _ { 2 } } \| \nabla f ( q _ { 0 } ) \| ^ { 2 } + \displaystyle \sum _ { j = 1 } ^ { k - 1 } \frac { ( k - j ) \eta ^ { 4 } k ^ { 3 } } { 6 j c _ { 3 } } \| \nabla F ( q _ { j } ) - \nabla F ( q _ { 0 } ) \| ^ { 2 } } } \\ { { \displaystyle \le \frac { 1 } { c _ { 1 } } ( k \eta ) ^ { 2 } \| p _ { 0 } \| ^ { 2 } + \frac { ( k \eta ) ^ { 4 } } { 4 c _ { 2 } } \| \nabla f ( q _ { 0 } ) \| ^ { 2 } + \displaystyle \sum _ { j = 1 } ^ { k - 1 } \frac { ( k - j ) \eta ^ { 4 } k ^ { 3 } M ^ { 2 } } { 6 j c _ { 3 } } \| q _ { j } - q _ { 0 } \| ^ { 2 } } } \\ { { \displaystyle \le \frac { 1 } { c _ { 1 } } ( k \eta ) ^ { 2 } \| p _ { 0 } \| ^ { 2 } + \frac { ( k \eta ) ^ { 4 } } { 4 c _ { 2 } } \| \nabla f ( q _ { 0 } ) \| ^ { 2 } + \displaystyle \sum _ { j = 1 } ^ { k - 1 } \frac { ( k - j ) \eta ^ { 4 } k ^ { 3 } M ^ { 2 } } { 6 j c _ { 3 } } \cdot ( 2 ( j \eta ) ^ { 2 } \| p _ { 0 } \| ^ { 2 } + ( j \eta ) ^ { 4 } \| \nabla f ( q _ { 0 } ) \| ^ { 2 } ) } } \\   \displaystyle \le ( \frac { 1 } { c _ { 1 } } + \frac { M ^ { 2 } } { 6 c _ { 3 } } \frac { 2 ( k \eta ) ^ { 4 } } { 6 } ) ( k \eta ) ^ { 2 } \| p _ { 0 } \| ^ { 2 } + ( \frac { 1 } { 4 c _ { 2 } } + \frac { M ^ { 2 } } { 6 c _ { 3 } } \frac  ( k \eta \end{array}
$$

where the first inequality is due to Lemma B.12 and noting that $\begin{array} { r } { c _ { 1 } + c _ { 2 } + \frac { \sum _ { j = 1 } ^ { k - 1 } ( k - j ) j } { ( k ^ { 3 } - k ) / 6 } c _ { 3 } = 1 ; } \end{array}$ the second inequality is due to smoothness Assumption 4.2; the third inequality holds by the induction assumption; the fourth inequality is by $\begin{array} { r } { \sum _ { j = 1 } ^ { k - \bar { 1 } } ( k - j ) j = \frac { k ^ { 3 } - k } { 6 } \leq \frac { k ^ { 3 } } { 6 } } \end{array}$ and $\begin{array} { r } { \sum _ { j = 1 } ^ { k - 1 } ( k - j ) j ^ { 3 } = } \end{array}$ $\begin{array} { r } { \frac { k ^ { 5 } } { 2 0 } - \frac { k ^ { 3 } } { 1 2 } + \frac { 1 } { 2 0 } k \leq \frac { k ^ { 5 } } { 2 0 } } \end{array}$ . The completion of the induction follows by step size assumption $K \eta \le 1 / \sqrt { M }$ and setting $\left( c _ { 1 } , c _ { 2 } , c _ { 3 } \right) = \left( \frac { 3 } { 5 } , \frac { 1 } { 3 } , \frac { 1 } { 1 5 } \right)$ in the last inequality. Therefore, we conclude the proof of (30). □

Lemma B.9. Fo $\begin{array} { r } { \cdot q _ { k } = \widetilde { h } _ { \mathrm { L F } } ( F , q _ { 0 } , \nabla f ( q _ { 0 } ) , p _ { 0 } , \eta , k ) a n d p _ { k } = p _ { k - 1 } - \eta ( \nabla F ( q _ { k - 1 } ) + \nabla F ( q _ { k } ) ) / 2 + \eta ( \nabla F ( q _ { k - 1 } ) + q _ { k } ) , } \end{array}$ $\eta ( \nabla F ( q _ { 0 } ) - \nabla f ( \bar { q } _ { 0 } ) )$ , we can rewrite as

$$
q _ { k } = q _ { 0 } + k \eta p _ { 0 } - \frac { ( k \eta ) ^ { 2 } } { 2 } \nabla f ( q _ { 0 } ) - \eta ^ { 2 } \sum _ { i = 1 } ^ { k - 1 } ( k - i ) ( \nabla F ( q _ { j } ) - \nabla F ( q _ { 0 } ) )\tag{31}
$$

$$
p _ { k } = p _ { 0 } - k \eta \nabla f ( q _ { 0 } ) - \eta \sum _ { j = 1 } ^ { k - 1 } ( \nabla F ( q _ { j } ) - \nabla F ( q _ { 0 } ) ) - \frac { \eta } { 2 } ( \nabla F ( q _ { k } ) - \nabla F ( q _ { 0 } ) ) .\tag{32}
$$

Proof. For simplicity of notations, we use $\widetilde f ( \cdot ) : = F ( \cdot ) - F ( q _ { 0 } ) + f ( q _ { 0 } )$ . According to Algorithm 1,

$$
q _ { k } = q _ { k - 1 } + \eta p _ { k - 1 } - \frac { \eta ^ { 2 } } { 2 } \nabla \widetilde { f } ( q _ { k - 1 } ) , \qquad p _ { k } = p _ { k - 1 } - \eta ( \nabla \widetilde { f } ( q _ { k - 1 } ) + \nabla \widetilde { f } ( q _ { k } ) ) / 2\tag{33}
$$

Summing from $k = 1 , \dots , k$ on both side of the second equation of (33), we can get

$$
p _ { k } = p _ { 0 } - \frac { \eta } { 2 } \nabla \widetilde { f } ( q _ { 0 } ) - \eta \sum _ { j = 1 } ^ { k - 1 } \nabla \widetilde { f } ( q _ { j } ) - \frac { \eta } { 2 } \nabla \widetilde { f } ( q _ { k } ) .
$$

This proves (32). Inserting this into the first equality of (33), we can get

$$
q _ { k } = q _ { k - 1 } + p _ { 0 } - \frac { \eta ^ { 2 } } { 2 } \nabla \widetilde { f } ( q _ { 0 } ) - \eta ^ { 2 } \sum _ { j = 1 } ^ { k - 1 } \nabla \widetilde { f } ( q _ { j } )
$$

Summing from $k = 1 , \ldots , k$ on both side yields

$$
q _ { k } = q _ { 0 } + k \eta p _ { 0 } - \frac { ( k \eta ) ^ { 2 } } { 2 } \nabla f ( q _ { 0 } ) - \eta ^ { 2 } \sum _ { i = 1 } ^ { k - 1 } ( k - i ) ( \nabla F ( q _ { j } ) - \nabla F ( q _ { 0 } ) )
$$

which proves (31).

Lemma B.10. Under the same notations of Theorem 4.3 and Assumption 4.2, for $K \eta \leq 1 / ( 2 \sqrt { M } )$ we have

$$
\begin{array} { r l } & { \| p _ { t , k } \| ^ { 2 } \leq 5 \| p _ { t } \| ^ { 2 } + 5 M ^ { 2 } ( k \eta ) ^ { 2 } \| \theta _ { t } - \theta ^ { * } \| ^ { 2 } } \\ & { \| \theta _ { t , k } - \theta ^ { * } \| ^ { 2 } \leq 4 ( k \eta ) ^ { 2 } \| p _ { t } \| ^ { 2 } + 3 \| \theta _ { t } - \theta ^ { * } \| ^ { 2 } . } \end{array}
$$

Proof. Recall that $\begin{array} { r } { p _ { t , k } : = \sum _ { c = 1 } ^ { N } w _ { c } p _ { t , k } ^ { ( c ) } } \end{array}$ , and note that by Lemma B.11 that $\| p _ { t , k } ^ { ( c ) } \| ^ { 2 } \leq 5 \| p _ { t } \| ^ { 2 } +$ $5 M ^ { 2 } ( k \eta ) ^ { 2 } \Vert \theta _ { t } - \theta ^ { \ast } \Vert ^ { 2 }$ , we have

$$
\| p _ { t , k } \| ^ { 2 } \leq \sum _ { c = 1 } ^ { N } w _ { c } \| p _ { t , k } ^ { ( c ) } \| ^ { 2 } \leq 5 \| p _ { t } \| ^ { 2 } + 5 M ^ { 2 } ( k \eta ) ^ { 2 } \| \theta _ { t } - \theta ^ { * } \| ^ { 2 } .
$$

This proves the first claim.

For the second claim, first, by convexity of $\| \cdot \| ^ { 2 }$ and Lemma B.12

$$
\| \theta _ { t , k } - \theta ^ { * } \| ^ { 2 } \leq 2 \| \theta _ { t , k } - \theta _ { t } \| ^ { 2 } + 2 \| \theta _ { t } - \theta ^ { * } \| ^ { 2 } \leq \sum _ { c = 1 } ^ { N } w _ { c } 2 \| \theta _ { t , k } ^ { ( c ) } - \theta _ { t } \| ^ { 2 } + 2 \| \theta _ { t } - \theta ^ { * } \| ^ { 2 } .
$$

At the same time, for $\| \theta _ { t , k } ^ { ( c ) } - \theta _ { t } \| ^ { 2 }$ , Lemma B.8 states that $\| \theta _ { t , k } ^ { ( c ) } - \theta _ { t } \| ^ { 2 } \ \leq \ 2 ( k \eta ) ^ { 2 } \| p _ { t } \| ^ { 2 } \ +$ $( k \eta ) ^ { 4 } M ^ { 2 } \lVert \theta _ { t } - \theta ^ { * } \rVert ^ { 2 }$ . And so

$$
\begin{array} { r } { \| \theta _ { t , k } - \theta ^ { * } \| ^ { 2 } \leq 4 ( k \eta ) ^ { 2 } \| p _ { t } \| ^ { 2 } + ( 2 + 2 ( k \eta ) ^ { 4 } M ^ { 2 } ) \| \theta _ { t } - \theta ^ { * } \| ^ { 2 } \leq 4 ( k \eta ) ^ { 2 } \| p _ { t } \| ^ { 2 } + 3 \| \theta _ { t } - \theta ^ { * } \| ^ { 2 } } \end{array}
$$

where the last inequality is due to $k \eta \leq 1 / ( 2 \sqrt { M } )$ . This finishes the proof.

Lemma B.11. Suppose $q _ { k } \ = \ \widetilde { h } _ { \mathrm { L F } } ( F , q _ { 0 } , \nabla f ( q _ { 0 } ) , p _ { 0 } , \eta , k )$ and $p _ { k } \ = \ p _ { k - 1 } - \eta ( \nabla F ( q _ { k - 1 } ) \ +$ $\nabla F ( q _ { k } ) ) / 2 + \eta ( \nabla F ( q _ { 0 } ) - \nabla f ( q _ { 0 } ) )$ . Under Assumption 4.2, for $K \eta \le 1 / ( 2 \sqrt { M } )$ , we have

$$
\begin{array} { r } { \| p _ { j } \| ^ { 2 } \leq 5 \| p _ { 0 } \| ^ { 2 } + 5 M ^ { 2 } ( j \eta ) ^ { 2 } \| q _ { 0 } - \theta ^ { * } \| ^ { 2 } . } \end{array}
$$

Proof. By definitions and Algorithm 1, we have

$$
p _ { k } = p _ { 0 } - k \eta \nabla f ( q _ { 0 } ) - \eta \sum _ { j = 1 } ^ { k - 1 } ( \nabla F ( q _ { j } ) - \nabla F ( q _ { 0 } ) ) - \frac { \eta } { 2 } ( \nabla F ( q _ { k } ) - \nabla F ( q _ { 0 } ) )
$$

By Cauchy–Schwarz inequality and M-smoothness assumption,

$$
\begin{array} { r l } & { \displaystyle | | p _ { k } | | ^ { 2 } \le 4 \| p _ { 0 } \| ^ { 2 } + 4 \| k \eta \nabla f ( q _ { 0 } ) \| ^ { 2 } + 4 \| \eta \displaystyle \sum _ { j = 1 } ^ { k - 1 } ( \nabla F ( q _ { j } ) - \nabla F ( q _ { 0 } ) ) \| ^ { 2 } + 4 \| \frac { \eta } { 2 } ( \nabla F ( q _ { k } ) - \nabla F ( q _ { 0 } ) ) \| ^ { 2 } } \\ & { \qquad \le 4 \| p _ { 0 } \| ^ { 2 } + 4 \| k \eta \nabla f ( q _ { 0 } ) \| ^ { 2 } + 4 ( k - 1 ) \displaystyle \sum _ { j = 1 } ^ { k - 1 } \eta ^ { 2 } \| \nabla F ( q _ { j } ) - \nabla F ( q _ { 0 } ) \| ^ { 2 } + 4 \| \frac { \eta } { 2 } ( \nabla F ( q _ { k } ) - \nabla F ( q _ { 0 } ) ) \| ^ { 2 } } \\ & { \displaystyle \le 4 \| p _ { 0 } \| ^ { 2 } + 4 ( k \eta ) ^ { 2 } \| \nabla f ( q _ { 0 } ) \| ^ { 2 } + 4 ( k - 1 ) \displaystyle \sum _ { j = 1 } ^ { k - 1 } \eta ^ { 2 } M ^ { 2 } \| q _ { j } - q _ { 0 } \| ^ { 2 } + \eta ^ { 2 } M ^ { 2 } \| q _ { k } - q _ { 0 } \| ^ { 2 } } \\ & { \displaystyle \le 4 \| p _ { 0 } \| ^ { 2 } + 4 ( k \eta ) ^ { 2 } \| \nabla f ( q _ { 0 } ) \| ^ { 2 } + 4 ( k \eta ) ^ { 2 } M ^ { 2 } \operatorname* { m a x } _ { \eta \le t } \| q _ { j } - q _ { 0 } \| ^ { 2 } } \end{array}
$$

where the first two inequalities are by Lemma B.12 and the third inequality is by M-smoothness assumption.

Note that Lemma B.8 gives that $\Vert q _ { j } - q _ { 0 } \Vert ^ { 2 } \leq 2 ( j \eta ) ^ { 2 } \Vert p _ { 0 } \Vert ^ { 2 } + ( j \eta ) ^ { 4 } M ^ { 2 } \Vert q _ { 0 } - \theta ^ { \ast } \Vert ^ { 2 }$ . Inserting this into the above inequality yields

$$
\| p _ { k } \| ^ { 2 } \leq 4 ( 1 + 2 ( k \eta ) ^ { 4 } M ^ { 2 } ) \| p _ { 0 } \| ^ { 2 } + 4 ( k \eta ) ^ { 2 } ( 1 + M ^ { 2 } ( k \eta ) ^ { 4 } ) M ^ { 2 } \| q _ { 0 } - \theta ^ { \ast } \| ^ { 2 } .
$$

The second claim follows by noting that kη $\leq 1 / ( 2 \sqrt { M } )$

Lemma B.12 (Jensen’s inequality). For any vectors $\{ x _ { i } \} _ { i = \cdot } ^ { n }$ and positive constants $\{ \lambda _ { i } \} _ { i = 1 } ^ { n }$ with $\sum \lambda _ { i } = 1$ , due to the convexity $o \dot { f } \parallel \cdot \parallel ^ { 2 }$ , we have

$$
\| \sum _ { i } \lambda _ { i } x _ { i } \| ^ { 2 } \leq \sum _ { i } \lambda _ { i } \| x _ { i } \| ^ { 2 } ;
$$

$$
\| \sum _ { i } x _ { i } \| ^ { 2 } \leq \sum _ { i } \lambda _ { i } \| x _ { i } / \lambda _ { i } \| ^ { 2 } = \sum _ { i } { \frac { 1 } { \lambda _ { i } } } \| x _ { i } \| ^ { 2 } .
$$

For any vector-valuefunction $x ( t )$ and non-negative integrablefunction $\lambda ( t )$

$$
\begin{array} { r l } & { \| \displaystyle \int \lambda ( t ) x ( t ) d t \| ^ { 2 } \leq \int \lambda ( t ) d t \int \lambda ( t ) \| x ( t ) \| ^ { 2 } d t . } \\ & { \| \displaystyle \int \lambda ( t ) x ( t ) d t \| ^ { 2 } \leq \int \lambda ( t ) \| x ( t ) \| ^ { 2 } d t , \quad i f \int \lambda ( t ) d t = 1 . } \end{array}
$$

## C Study on heterogeneity metric H

## C.1 Bound on H

Recall that $\begin{array} { r l r } { H } & { { } : = } & { \operatorname* { m a x } _ { i \leq K } \sum _ { c = 1 } ^ { N } w _ { c } \mathbb { E } \| \frac { 1 } { i \eta } ( \nabla f ^ { ( c ) } ( \bar { \mathcal { H } } _ { \eta } ^ { i } ( \theta ^ { \pi } ) ) ~ - ~ \nabla f ^ { ( c ) } ( \theta ^ { \pi } ) ~ + ~ \nabla f ( \theta ^ { \pi } ) ~ - ~ \nabla f ^ { ( c ) } ( \theta ^ { \pi } ) ) ~ \xi = 0 , } \end{array}$ $\nabla f ( \bar { \mathcal { H } } _ { \eta } ^ { i } ( \theta ^ { \pi } ) ) ) \lvert \rvert ^ { 2 }$ By the M-Lipschitz condition,

$$
H \leq 4 \operatorname* { m a x } _ { 1 \leq i \leq K } \sum _ { c = 1 } ^ { N } w _ { c } \mathbb { E } \left\| \frac { M } { i \eta } \left( \bar { \mathcal { H } } _ { \eta } ^ { i } ( \theta ^ { \pi } ) - \theta ^ { \pi } \right) \right\| ^ { 2 } .
$$

Moreover,

$$
\bar { \mathcal { H } } _ { \eta } ^ { i } ( \theta ^ { \pi } ) - \theta ^ { \pi } = \int _ { 0 } ^ { i \eta } p ( t ) d t .
$$

where $p ( t )$ has a stationary marginal multivariate normal distribution. Overall, we can obtain that

$$
H \leq C M ^ { 2 } d
$$

for some absolute constant C.

## C.2 Solutions of H on Gaussian examples

Give normal clients with same covariance structure $( \mathbf { e } . \mathbf { g } . , N ( 1 , I )$ and $N ( - 1 , I )$ , and the global is $N ( 0 , I ) )$ ), then

$$
\nabla f ^ { ( c ) } ( \theta _ { 2 } ) - \nabla f ^ { ( c ) } ( \theta _ { 1 } ) + \nabla f ( \theta _ { 1 } ) - \nabla f ( \theta _ { 2 } ) \equiv 0 ,
$$

meaning that $H \equiv 0$ always.

For non-equal covariance case, $\mathrm { e . g . , } N ( 0 , 3 I / 4 )$ and $N ( 0 , 3 I / 2 )$ , and the global is $N ( 0 , I )$ , the HMC trajectory for the global has a closed form, i.e., $\theta _ { t } = \theta _ { 0 } \cos ( t ) + p _ { 0 } \sin ( t )$ elementwisely. The calculation of H eventually reduces to

$$
d \times \operatorname* { s u p } _ { t } \mathbb { E } _ { \theta , p \sim N ( 0 , I ^ { 2 } ) } \left| \frac { \theta ( 1 - \cos ( t ) ) - p \sin ( t ) } { t } \right| ^ { 2 } \Big / 9 = \frac { 2 d } { 9 } \times \operatorname* { s u p } _ { t } \frac { 1 - \cos ( t ) } { t ^ { 2 } } = \frac { d } { 9 } .
$$

## D Experimental Details

## D.1 Additional Details for Figure 2

Table 2 reports the stepsize and leapfrog grids used for the simulated Bayesian logistic regression experiment in Figure 2. The same grid is used for both panels. FA-LD uses $T = 1 0$ and $K = 1$ for every stepsize. FA-HMC uses $\check { T } = 1 0$ and the K values listed in the table. AFHMC uses the candidate leapfrog counts $K _ { \mathrm { A } } \in K _ { \mathrm { H M C } } ( \eta ) \cup 1 0 K _ { \mathrm { H M C } } ( \eta )$ . For each (algorithm, η) pair, Figure 2 reports the best-performing K from this grid.

Table 2: Candidate grid for the simulated Bayesian logistic regression experiment. Here $\mathcal { K } _ { \mathrm { H M C } } ( \eta )$ denotes the FA-HMC leapfrog grid at stepsize η; the AFHMC grid is $\mathcal { K } _ { \mathrm { H M C } } ( \eta ) \cup 1 0 \mathcal { K } _ { \mathrm { H M C } } ( \eta )$
<table><tr><td>Stepsize η</td><td> $\mathrm { F A - L D } \left( T , K \right)$ </td><td> $\mathrm { F A - H M C } \left( T , \mathcal { K } _ { \mathrm { H M C } } \right)$ </td><td>AFHMC KA</td></tr><tr><td> $5 \times 1 0 ^ { - 4 }$ </td><td>(10, {1})</td><td>(10, {500, 2000})</td><td>{500, 2000, 5000, 20000}</td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td>(10, {1})</td><td>(10, {300, 1000})</td><td>{300, 1000, 3000, 10000}</td></tr><tr><td> $2 \times 1 0 ^ { - 3 }$ </td><td>(10, {1})</td><td>(10, {150, 500})</td><td>{150, 500, 1500, 5000}</td></tr><tr><td> $5 \times 1 0 ^ { - 3 }$ </td><td>(10, {1})</td><td>(10, {50, 200})</td><td>{50, 200, 500, 2000}</td></tr><tr><td> $1 0 ^ { - 2 }$ </td><td>(10, {1})</td><td>(10, {30, 100})</td><td>{30, 100, 300, 1000}</td></tr><tr><td> $2 \times 1 0 ^ { - 2 }$ </td><td>(10, {1})</td><td>(10, {10, 50})</td><td>{10, 50, 100, 500}</td></tr><tr><td> $5 \times 1 0 ^ { - 2 }$ </td><td>(10, {1})</td><td>(10, {2, 5, 10, 20})</td><td>{2, 5, 10, 20, 50, 100, 200}</td></tr><tr><td> $1 0 ^ { - 1 }$ </td><td>(10, {1})</td><td>(10, {1, 2, 5, 10})</td><td>{1, 2, 5, 10, 20, 50, 100}</td></tr><tr><td> $2 \times 1 0 ^ { - 1 }$ </td><td>(10, {1})</td><td>(10, {1, 2, 5})</td><td>{1, 2, 5, 10, 20, 50}</td></tr><tr><td> $5 \times 1 0 ^ { - 1 }$ </td><td>(10, {1})</td><td>(10, {1, 2, 5})</td><td>{1, 2, 5, 10, 20, 50}</td></tr><tr><td>1</td><td>(10, {1})</td><td>(10, {1, 2})</td><td>{1, 2, 10, 20}</td></tr></table>

Table 3: First communication round at which AFHMC reaches $\mathcal { W } _ { 2 } ^ { 2 } < 0 . 1$ on the low-heterogeneity Gaussian target, with $K ( d ) = \lfloor 5 0 ( d / 1 0 0 ) ^ { 1 / 2 } \rceil$ and $\eta ( d ) = 0 . 0 0 6 ( d / 1 0 0 ) ^ { - 1 / 2 }$
<table><tr><td> $d$ </td><td> $K ( d )$ </td><td> $\eta ( d )$ </td><td>Rounds (seeds 0, 1, 2)</td><td>Mean ± sd</td></tr><tr><td>2</td><td>7</td><td>0.04243</td><td>161, 153, 149</td><td> $1 5 4 . 3 3 \pm 6 . 1 1$ </td></tr><tr><td>50</td><td>35</td><td>0.008485</td><td>213, 212, 212</td><td> $2 1 2 . 3 3 \pm 0 . 5 8$ </td></tr><tr><td>100</td><td>50</td><td>0.006</td><td>220, 220, 220</td><td> $2 2 0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>150</td><td>61</td><td>0.004899</td><td>229, 229, 229</td><td> $2 2 9 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>200</td><td>71</td><td>0.004243</td><td>231,231, 230</td><td> $2 3 0 . 6 7 \pm 0 . 5 8$ </td></tr><tr><td>250</td><td>79</td><td>0.003795</td><td>237,237,237</td><td> $2 3 7 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>300</td><td>87</td><td>0.003464</td><td>238, 238, 237</td><td> $2 3 7 . 6 7 \pm 0 . 5 8$ </td></tr><tr><td>350</td><td>94</td><td>0.003207</td><td>240, 240, 240</td><td> $2 4 0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>400</td><td>100</td><td>0.003</td><td>245, 245, 245</td><td> $2 4 5 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>450</td><td>106</td><td>0.002828</td><td>247,248, 247</td><td> $2 4 7 . 3 3 \pm 0 . 5 8$ </td></tr><tr><td>500</td><td>112</td><td>0.002683</td><td>248, 248, 248</td><td> $2 4 8 . 0 0 \pm 0 . 0 0$ </td></tr></table>

## D.2 Additional Details for the Gaussian Dimension-Scaling Experiments

Low-heterogeneity target. For each dimension d and each of the $N = 1 0$ clients, we draw $1 0 ^ { 5 }$ samples from $\mathcal { N } ( 1 6 . 2 \mathbf { 1 } _ { d } , 1 . 6 I _ { d } )$ and set the local target of client c to $\mathcal { N } ( \hat { \mu } _ { c } , \hat { \sigma } _ { c } ^ { 2 } I _ { d } )$ , where $\hat { \mu } _ { c }$ is the sample mean and $\hat { \sigma } _ { c } ^ { 2 }$ is the coordinate-averaged sample variance. The global target is fitted in the same way to the samples pooled over all clients, and ${ \cal W } _ { 2 } ^ { 2 }$ is computed as in the heterogeneous experiment. The heterogeneity therefore only comes from sampling error: across all dimensions and seeds, the root-mean-square deviation of the client means from the global mean is between $3 . 6 \times 1 0 ^ { - 3 }$ and $4 . 5 \times 1 0 ^ { - 3 }$ per coordinate, compared with 11.1 in the heterogeneous experiment, where the client means are 20 and 1 and the global mean is 16.2. AFHMC runs one trajectory of $K ( d )$ surrogate leapfrog steps per round and uses the stopping rule of Section $5 ( \mathcal { W } _ { 2 } ^ { 2 } < \mathrm { 0 . 1 }$ for 20 consecutive checks, at most 1000 rounds). The random seed controls both the client samples and the HMC noise. Table 3 reports the first-crossing rounds for every seed.

FA-HMC baselines at an equal number of server exchanges. Theorem 4.1 of Liang et al. (2025) shows that, under strong convexity, Hessian smoothness and bounded stochastic-gradient variance, FA-HMC attains $\mathcal { W } _ { 2 } \le \epsilon$ when

$$
\eta ^ { 2 } = C \operatorname* { m i n } \left\{ \frac { 1 } { K ^ { 2 } L } , \frac { \epsilon } { K ^ { 2 } \sqrt { d } T } , \frac { \epsilon ^ { 2 } } { K ^ { 2 } d T ^ { 2 } ( 1 - \rho ) N } , \frac { \epsilon ^ { 2 } } { K d \sum _ { c = 1 } ^ { N } w _ { c } ^ { 2 } \sigma _ { g } ^ { 2 } } \right\} .
$$

In our Gaussian experiment all clients share the momentum, so $\rho = 1$ , and the gradients are exact, so $\sigma _ { g } ^ { 2 } = 0 ;$ ; the last two terms therefore impose no constraint. The remaining dimension-dependent term gives $\eta \propto ( K \sqrt { T } d ^ { 1 / 4 } ) ^ { - 1 }$ . With $T = 1 0$ and $K _ { \mathrm { F } } ( d ) = \mathrm { m a x } \{ 1 , \lfloor K ( d ) / 1 0 \rceil \} \propto d ^ { 1 / 3 }$ , this yields $\eta ( d ) \propto d ^ { - 7 / 1 2 }$ , which we anchor at d = 100 as $\eta ( d ) = 0 . 0 0 6 ( d / 1 0 0 ) ^ { - 7 / 1 2 }$ . The second schedule is the dimension-scaling setting of Liang et al. (2025), $K _ { \mathrm { F } } = 5 , T = 1 0$ and $\eta ( d ) = 0 . 0 2 d ^ { - 1 / 4 }$

Table 4: FA-HMC ${ \mathcal W } _ { 2 } ^ { 2 }$ on the heterogeneous Gaussian target. $R ( d )$ is the AFHMC first-crossing round for $\mathcal { W } _ { 2 } ^ { 2 } \ < \ 0 . 1$ The original setting uses the AFHMC stepsize for $R ( d )$ rounds; the two FA-HMC-specific schedules are run for $2 R ( d )$ rounds. All settings use $T = { \mathrm { i 0 } } ;$ ; the original and Theorem-4.1 settings use $K _ { \mathrm { F } } ( d )$ , and the second setting uses $K _ { \mathrm { F } } = 5 .$
<table><tr><td colspan="4"></td><td colspan="2">Original (R(d))</td><td colspan="2"> $K _ { \mathrm { F } } { = } 5 \ ( 2 R ( d ) )$ </td><td colspan="2">Thm. 4.1  $( 2 R ( d ) )$ </td></tr><tr><td>d</td><td> $R ( d )$ </td><td> $2 R ( d )$ </td><td> $K _ { \mathrm { F } } ( d )$ </td><td>η</td><td>W2</td><td>η</td><td>W2</td><td>η</td><td>W2</td></tr><tr><td>2</td><td>42</td><td>84</td><td>1</td><td>0.04243</td><td>327.9</td><td>0.01682</td><td>13.0</td><td>0.05878</td><td>85.7</td></tr><tr><td>50</td><td>162</td><td>324</td><td>4</td><td>0.008485</td><td>4092.3</td><td>0.007521</td><td>752.1</td><td>0.00899</td><td>960.2</td></tr><tr><td>100</td><td>220</td><td>440</td><td>5</td><td>0.006</td><td>7621.3</td><td>0.006325</td><td>1682.8</td><td>0.006</td><td>2213.9</td></tr><tr><td>150</td><td>263</td><td>526</td><td>6</td><td>0.004899</td><td>9523.9</td><td>0.005715</td><td>2694.0</td><td>0.004736</td><td>2774.3</td></tr><tr><td>200</td><td>294</td><td>588</td><td>6</td><td>0.004243</td><td>15968.2</td><td>0.005318</td><td>3911.7</td><td>0.004005</td><td>6298.0</td></tr><tr><td>250</td><td>320</td><td>640</td><td>7</td><td>0.003795</td><td>16013.5</td><td>0.00503</td><td>5236.5</td><td>0.003516</td><td>5827.7</td></tr><tr><td>300</td><td>347</td><td>694</td><td>7</td><td>0.003464</td><td>22009.6</td><td>0.004806</td><td>6444.4</td><td>0.003161</td><td>9424.0</td></tr><tr><td>350</td><td>368</td><td>736</td><td>8</td><td>0.003207</td><td>20227.2</td><td>0.004624</td><td>7867.1</td><td>0.002889</td><td>7877.5</td></tr><tr><td>400</td><td>393</td><td>786</td><td>8</td><td>0.003</td><td>25525.1</td><td>0.004472</td><td>9014.2</td><td>0.002673</td><td>11122.8</td></tr><tr><td>450</td><td>403</td><td>806</td><td>8</td><td>0.002828</td><td>32540.4</td><td>0.004342</td><td>11000.1</td><td>0.002495</td><td>15877.5</td></tr><tr><td>500</td><td>430</td><td>860</td><td>8</td><td>0.002683</td><td>38053.9</td><td>0.004229</td><td>11872.9</td><td>0.002346</td><td>19755.7</td></tr></table>

![](images/7a7048c2581a1d334628203edbf1342dde9e29c9bf70fd5d07bd6ad58ccab710.jpg)  
Figure 5: Additional predictive-quality metrics for Bayesian logistic regression on heterogeneous FMNIST against the number of server exchanges (one per round for FA-LD and FA-HMC, two per round for AFHMC). Left: Brier score. Right: negative log-likelihood. The data are partitioned across $N = 1 0$ clients using a label-skewed Dirichlet split with concentration parameter $\alpha = 0 . 1$

Both schedules are run for $2 R ( d )$ rounds, where $R ( d )$ is the AFHMC first-crossing round, so that FA-HMC uses the same number of server exchanges as AFHMC. Table 4 reports the resulting errors together with the original matched-budget setting. These schedules are theory-guided choices rather than an exhaustive search over $( \eta , K , T )$ , and each setting is run once.

## D.3 Additional Details for MNIST Logistic Regression

To create heterogeneous federated learning scenario, the FMNIST training set is partitioned across $N = 1 0$ clients using a label-skewed Dirichlet split. Specifically, for each class $c ,$ we sample a client-proportion vector

$$
p _ { c } \sim \mathrm { D i r i c h l e t } ( \alpha { \bf 1 } _ { N } ) , \qquad \alpha = 0 . 1 ,
$$

and distribute the examples of class c across clients according to $p _ { c }$ . This produces heterogeneous clients with different label distributions.

Each configuration is repeated over five random seeds. The reported values in Table 5 are the final mean and standard error (SE) across seeds at 3600 server exchanges, i.e., at round 3600 for FA-LD and FA-HMC and at round 1800 for AFHMC. Since the stepsize is tuned separately for each metric, the selected stepsize may differ between Accuracy, Brier score, and NLL for the same method and K. Table 6 reports the accuracy at intermediate budgets. At 3600 server exchanges, AFHMC with $K \in \{ 1 0 , 1 0 0 \}$ attains the highest accuracy and the lowest ${ \mathrm { N L L } } ,$ and ties with FA-HMC $( K = 1 0 0 )$ in Brier score; FA-HMC with $K = 5 0$ has lower Brier score and NLL at intermediate budgets.

Table 5: Final performance on heterogeneous FMNIST at 3600 server exchanges. Entries are mean ± standard error over five random seeds.
<table><tr><td>Metric</td><td>Method</td><td>K</td><td>Rounds</td><td>η</td><td> $\mathbf { M e a n } \pm \mathbf { S E }$ </td></tr><tr><td>Accuracy</td><td>FA-LD FA-HMC FA-HMC</td><td>1 10 50</td><td>3600 3600</td><td> $1 \times 1 0 ^ { - 4 }$   $2 \times 1 0 ^ { - 5 }$ </td><td> $0 . 8 0 7 5 \pm 0 . 0 0 3 1$   $0 . 8 2 7 4 \pm 0 . 0 0 3 6$ </td></tr><tr><td></td><td>FA-HMC</td><td>100</td><td>3600 3600</td><td> $5 \times 1 0 ^ { - 6 }$   $2 \times 1 0 ^ { - 6 }$ </td><td> $0 . 8 3 8 4 \pm 0 . 0 0 0 7$   $0 . 8 4 0 8 \pm 0 . 0 0 0 3$ </td></tr><tr><td></td><td>AFHMC</td><td>10</td><td>1800</td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $0 . 8 4 2 2 \pm 0 . 0 0 0 5$ </td></tr><tr><td></td><td>AFHMC</td><td>50</td><td>1800</td><td> $5 \times 1 0 ^ { - 6 }$ </td><td> $0 . 8 4 0 2 \pm 0 . 0 0 0 5$ </td></tr><tr><td></td><td>AFHMC</td><td>100</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>1800</td><td> $2 \times 1 0 ^ { - 6 }$ </td><td> $0 . 8 4 2 2 \pm 0 . 0 0 0 4$ </td></tr><tr><td>Brier</td><td>FA-LD</td><td>1</td><td>3600</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $0 . 2 7 3 9 \pm 0 . 0 0 4 8$ </td></tr><tr><td></td><td>FA-HMC</td><td>10</td><td>3600</td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $0 . 2 4 4 7 \pm 0 . 0 0 4 7$ </td></tr><tr><td></td><td>FA-HMC</td><td>50</td><td>3600</td><td> $5 \times 1 0 ^ { - 6 }$ </td><td></td></tr><tr><td></td><td>FA-HMC</td><td>100</td><td></td><td></td><td> $0 . 2 3 0 0 \pm 0 . 0 0 1 5$ </td></tr><tr><td></td><td></td><td></td><td>3600</td><td> $2 \times 1 0 ^ { - 6 }$ </td><td> $0 . 2 2 9 0 \pm 0 . 0 0 0 3$ </td></tr><tr><td></td><td>AFHMC</td><td>10</td><td>1800</td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $0 . 2 2 9 0 \pm 0 . 0 0 0 4$ </td></tr><tr><td></td><td>AFHMC AFHMC</td><td>50 100</td><td>1800 1800</td><td> $5 \times 1 0 ^ { - 6 }$   $2 \times 1 0 ^ { - 6 }$ </td><td> $0 . 2 3 7 6 \pm 0 . 0 0 0 8$ </td></tr><tr><td>NLL</td><td></td><td></td><td></td><td></td><td> $0 . 2 2 9 0 \pm 0 . 0 0 0 4$ </td></tr><tr><td></td><td>FA-LD</td><td>1</td><td>3600</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $5 4 4 5 . 1 8 \pm 1 3 6 . 2 6$ </td></tr><tr><td></td><td>FA-HMC</td><td>10</td><td>3600</td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $4 8 7 6 . 1 8 \pm 1 2 2 . 3 5$ </td></tr><tr><td></td><td>FA-HMC</td><td>50</td><td>3600</td><td> $5 \times 1 0 ^ { - 6 }$ </td><td> $4 5 4 4 . 4 3 \pm 3 4 . 7 2$ </td></tr><tr><td></td><td>FA-HMC</td><td>100</td><td>3600</td><td> $2 \times 1 0 ^ { - 6 }$ </td><td> $4 5 4 1 . 7 4 \pm 4 . 9 9$ </td></tr><tr><td></td><td>AFHMC</td><td>10</td><td>1800</td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $4 5 3 2 . 8 9 \pm 7 . 6 6$ </td></tr><tr><td></td><td>AFHMC</td><td>50</td><td>1800</td><td> $5 \times 1 0 ^ { - 6 }$ </td><td> $4 6 8 0 . 3 4 \pm 1 5 . 8 9$ </td></tr><tr><td></td><td>AFHMC</td><td>100</td><td>1800</td><td> $2 \times 1 0 ^ { - 6 }$ </td><td> $4 5 3 1 . 5 8 \pm 7 . 6 1$ </td></tr></table>

Table 6: Test accuracy on heterogeneous FMNIST at an equal number of server exchanges (AFHMC at round $x / 2 ,$ FA-LD and FA-HMC at round x). Entries are means over five random seeds; stepsizes as in Table 5.
<table><tr><td>Method</td><td>K</td><td>η</td><td> $x = 2 0 0 0$ </td><td>2400</td><td>2800</td><td>3200</td><td>3600</td></tr><tr><td>FA-LD</td><td>1</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.8089</td><td>0.8085</td><td>0.8080</td><td>0.8076</td><td>0.8075</td></tr><tr><td>FA-HMC</td><td>10</td><td> $2 \times 1 0 ^ { - 5 }$ </td><td>0.8270</td><td>0.8272</td><td>0.8273</td><td>0.8277</td><td>0.8274</td></tr><tr><td>FA-HMC</td><td>50</td><td> $5 \times 1 0 ^ { - 6 }$ </td><td>0.8362</td><td>0.8369</td><td>0.8376</td><td>0.8381</td><td>0.8384</td></tr><tr><td>FA-HMC</td><td>100</td><td> $2 \times 1 0 ^ { - 6 }$ </td><td>0.8369</td><td>0.8381</td><td>0.8391</td><td>0.8401</td><td>0.8408</td></tr><tr><td>AFHMC</td><td>10</td><td> $2 \times 1 0 ^ { - 5 }$ </td><td>0.8389</td><td>0.8403</td><td>0.8413</td><td>0.8419</td><td>0.8422</td></tr><tr><td>AFHMC</td><td>50</td><td> $5 \times 1 0 ^ { - 6 }$ </td><td>0.8373</td><td>0.8386</td><td>0.8391</td><td>0.8399</td><td>0.8402</td></tr><tr><td>AFHMC</td><td>100</td><td> $2 \times 1 0 ^ { - 6 }$ </td><td>0.8392</td><td>0.8404</td><td>0.8410</td><td>0.8416</td><td>0.8422</td></tr></table>

Computational cost. Between two synchronizations, each client performs 50 leapfrog steps in FA-LD and FA-HMC, i.e., 100 mini-batch gradient evaluations per round, and K leapfrog steps plus one gradient evaluation for the drift correction in AFHMC, i.e., $2 K + 1$ evaluations per round. At 3600 server exchanges, AFHMC with $K = 1 0$ therefore uses $1 8 0 0 \times 2 1 = 3 7 { \cdot } 8 0 0$ gradient evaluations per client, compared with $3 6 0 0 \times 1 0 0 = 3 6 0 , 0 0 0$ for FA-HMC.