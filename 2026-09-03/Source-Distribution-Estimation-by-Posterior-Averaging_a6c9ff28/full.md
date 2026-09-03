# Source Distribution Estimation by Posterior Averaging

## Trung-Dung Hoang

Department of Digital Medicine, University of Bern, Switzerland Department of Diabetes, Endocrinology, Nutritional Medicine and Metabolism UDEM, Inselspital, Bern University Hospital, University of Bern, Switzerland Graduate School for Cellular and Biomedical Sciences (GCB), University of Bern, Switzerland Diabetes Center Berne, Switzerland trung.hoang@unibe.ch

Lisa M. Koch Department of Digital Medicine, University of Bern, Switzerland Department of Diabetes, Endocrinology, Nutritional Medicine and Metabolism UDEM, Inselspital, Bern University Hospital, University of Bern, Switzerland Diabetes Center Berne, Switzerland lisa.koch@unibe.ch

## Abstract

Simulation-based science often requires a distribution over simulator parameters whose push-forward reproduces a set of real observations: this is the source distribution estimation (SDE) problem. Existing methods fit the source against a likelihood surrogate trained once from a fixed proposal prior. Their objective is therefore stated only in terms of the surrogate instead of the true simulator, which may fail for inaccurate areas in parameter space where the surrogate was never trained. We instead solve SDE by expectation maximization: an E-step trains an amortized posterior on fresh simulations from the current source estimate, and an M-step refits the source to the average of that posterior over the observed data. We give two parameterizations, (1) separate source and posterior flows and (2) a single shared conditional flow. We evaluate our method on three benchmark tasks under both broad and misspecified initial priors. Both improve on existing fixed surrogate approaches and on iterated variants of each, most clearly on Lotka–Volterra, where no baseline falls below 0.96 data-space C2ST while our methods reach 0.64-0.68 in three of four initial-prior settings.

## 1 Introduction

Researchers often encode their knowledge of a system as a simulator in many domains such as astrophysics, epidemiology, or neuroscience: given parameters, one can generate synthetic observations and study how the system works internally [Cranmer et al., 2020]. However, what is typically available is not the distribution of parameters but a distribution of real observations. Therefore, to run the simulator consistently with reality, one must infer a distribution over parameters that is compatible with the data. This identification problem is known as Source Distribution Estimation (SDE) [Vandegar et al., 2021, Vetter et al., 2024]. Formally, the simulator is represented as $p ( x \mid \theta )$ where, given parameter θ, we can sample $x \sim p ( x \mid \theta )$ via simulation, but the density $p ( x \mid \theta )$ itself is intractable. Given a dataset $\mathcal { D } = \{ \bar { x } _ { 1 } , \dots , \bar { x } _ { n } \}$ of real observations with empirical distribution $p _ { o } ( x )$ , we need to find the source distribution $q ( \theta )$ such that $\begin{array} { r } { q ( x ) = \int p ( x \mid \bar { \theta } ) q ( \theta ) } \end{array}$ dθ matches $p _ { o } ( x )$ . Throughout this paper we assume the simulator $p ( x \mid \theta )$ is a black box: we can draw forward samples $x \sim p ( x \mid \theta )$ for any θ, but we have no access to the simulator’s implementation or gradients $\nabla _ { \theta } p ( x \mid \theta )$ . This is the common setting for licensed, proprietary scientific simulators [Yao et al., 2026], and for simulators with discrete or combinatorial internal stochasticity [Warne et al., 2019].

Existing methods [Vandegar et al., 2021, Vetter et al., 2024] often build a differentiable surrogate of the simulator by sampling θ from a distribution $\pi ( \theta )$ (we call this initial prior) and running the simulator forward to generate training data. The prior is then optimized so that the resulting marginal matches $p _ { o } ( x )$ under the surrogate. However, when the surrogate is inaccurate in regions explored by the learned source, agreement under the surrogate need not imply agreement under the true simulator. Here, we instead propose an iterative approach Posterior Averaging (PA) where, at each round, we learn the posterior from data simulated under the current source estimate and construct the next source as the posterior averaged over the observed data. We show that this procedure implements an Expectation-Maximization scheme [Dempster et al., 1977] whose induced-data KL decreases monotonically at every iteration in the idealized setting. If the iteration converges, the resulting source necessarily satisfies a self-consistency condition with respect to the true likelihood: the source equals the average of the true posteriors it induces over real data. Since multiple source distributions can induce the same marginal, $q ( \theta )$ is generally not identifiable from $p _ { o } ( x )$ . We therefore evaluate methods by how well the recovered $q ( \theta )$ reproduces the push-forwarded observation distribution, rather than by recovery of a single "true" source.

## 2 Related Work

Under our non-differentiable black-box setting, the relevant implementations of existing SDE methods rely on a fixed learned simulator surrogate rather than direct simulator differentiation.

Neural Empirical Bayes (NEB) NEB [Vandegar et al., 2021] trains a normalizing flow $\hat { p } ( x \mid$ θ) as a likelihood surrogate. Training data is generated by fixing a broad prior $\pi ( \theta )$ , sampling $\theta \sim \pi ( \theta )$ , and simulating $x \sim p ( x \mid \theta ) . \ q _ { \phi } ( \theta )$ is then optimized to maximize $\begin{array} { r } { \sum _ { x \sim p _ { o } } \log q _ { \phi } ( \bar { x } ) \stackrel { - } { = } } \end{array}$ $\begin{array} { r } { \sum _ { x \sim p _ { o } } \log \int \hat { p } ( x \mid \theta ) q _ { \phi } ( \theta ) d \theta } \end{array}$

Sample-based Maximum Entropy SDE (Sourcerer) Sourcerer [Vetter et al., 2024] resolves the non-uniqueness of SDE by selecting the maximum-entropy feasible source distribution: max<sub>ϕ</sub> H(q<sub>ϕ</sub>) s.t. $D ( q _ { \phi } , p _ { o } ) = 0$ , where $D$ is a distance between distributions. Sourcerer’s primary sample-based approach differentiates through the simulator directly; under our non-differentiable black-box assumption, only its surrogate-likelihood variant applies, so Sourcerer in this paper refers to $\textstyle q _ { \phi } ( x ) = \int \hat { p } ( \bar { x } \mid \theta ) q _ { \phi } ( \dot { \theta } )$ dθ, with ${ \hat { p } } ( x \mid \theta )$ trained identically to NEB’s surrogate.

## 3 Method

Suppose some π solves SDE exactly, $\begin{array} { r } { \int \pi ( \theta ) p ( x \mid \theta ) d \theta = p _ { o } ( x ) } \end{array}$ for all x. Its true Bayes posterior is $\pi ( \theta ~ \mid x ) ~ = ~ \pi ( \theta ) p ( x ~ \mid ~ \theta ) / p _ { o } ( x )$ , and averaging over real data gives $\textstyle \int p _ { o } ( x ) \pi ( \theta \mid x ) d x \ =$ $\textstyle \int \pi ( \theta ) p ( x \mid \theta ) d x = \pi ( \theta )$ . Therefore, while Vetter et al. [2024] claims that the average posterior does not always converge to a source distribution, self-consistency $\textstyle { \pi ( \theta ) = \int p _ { o } ( x ) \pi ( \theta \mid x ) }$ dx is indeed necessary for exact recovery. Since the objective of fixed-surrogate methods is stated purely in terms of afrozen surrogate, any self-consistency they achieve is with respect to the surrogate, and only transfers to the true simulator if the surrogate is accurate over the fitted source.

In our method, we alternate between training a posterior under the current source (E-step) and refitting the new source to the average posterior over real data (M-step), targeting self-consistency directly, and prove this alternation is monotone every round.

## 3.1 General algorithm

Algorithm 1 describes our method. At E-step, given an initial prior or the current source $\pi _ { t } ( \theta )$ , we fit a posterior $p _ { t } ( \theta \mid x )$ via a Neural Posterior Estimator (NPE) [Papamakarios and Murray, 2016, Greenberg et al., 2019]

$$
\mathcal { L } _ { \mathrm { N P E } } ( \pi _ { t } ) = - \mathbb { E } _ { \theta \sim \pi _ { t } , \ x \sim p ( x \mid \theta ) } \left[ \log p _ { t } ( \theta \mid x ) \right] .
$$

At M-step, given the posterior $p _ { t } ( \theta \mid x )$ , the next source targets the average posterior over real data,

$$
\pi _ { t + 1 } ( \theta ) = \int p _ { o } ( x ) p _ { t } ( \theta \mid x ) d x \approx \hat { \pi } _ { t + 1 } ( \theta ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } p _ { t } ( \theta \mid x _ { i } ) , \qquad x _ { i } \in \mathcal { D } ,
$$

fit by $\mathcal { L } _ { \mathrm { s o u r c e } } = - \mathbb { E } _ { \theta \sim \hat { \pi } _ { t + 1 } } [ \log \pi ( \theta ) ]$ over pooled posterior samples. Next, we describe two variants of our method: PA with separate flows and PA-Shared with a shared flow.

```perl
Algorithm 1 General procedure
Require: Simulator $p ( x \mid \theta )$ , data ${ \mathcal { D } } ,$ , initial prior $\pi _ { 0 } .$ , rounds $T$
1: repeat
2: E-step: simulate $\theta _ { j } \sim \pi _ { t } , x _ { j } \sim p ( x \mid \theta _ { j } )$ ; fit $p _ { t } ( \theta \mid x ) $ arg min $\mathcal { L } _ { \mathrm { N P E } } ( \pi _ { t } )$
3: M-step: for each $\bar { x _ { i } } \in \mathcal { D }$ , draw $\theta _ { s } \sim p _ { t } ( \theta \mid x _ { i } ) ;$ fit $\pi _ { t + 1 }  \arg$ min L<sub>source</sub>
4: $t \gets t + 1$
5: until $t = T$ or converged
```

## 3.2 Variants

PA (Algorithm 2) represents the source via an unconditional flow $\pi _ { \phi } ( \theta )$ , and the posterior via a conditional flow $p _ { \psi } ( { \bar { \theta } } \mid x )$ . In practice, both source and posterior can be represented within a single conditional flow $f _ { \phi } ( z , c )$ , where $c = e _ { \emptyset }$ (a learnable null embedding) gives $\pi _ { \phi } ( \theta ) = f _ { \phi } ( z , e _ { \mathcal { D } } )$ and $c = h _ { \xi } ( x )$ gives $p _ { \phi } ( \theta \mid x ) = f _ { \phi } ( z , h _ { \xi } ( x ) )$ . We refer to this implementation as PA-Shared ( (Algorithm 3). Context dropout during training $( c \to e _ { \emptyset }$ with probability $p _ { \emptyset }$ , as in classifier-free guidance) lets one objective train both modes, so posterior training informs the prior’s density estimate directly. Since $\phi$ is shared, fitting the posterior mode also moves the source it represents. However, the context dropout during training can reduce this drift. This also requires drawing all M-step samples right after the E-step and before any M-step update.

## 3.3 Convergence

Write $\textstyle \pi _ { t } ( x ) : = \int p ( x \mid \theta ) \pi _ { t } ( \theta )$ dθ for the marginal induced by $\pi _ { t }$ through the simulator likelihood.

Proposition 1 (Monotone x-marginal KL) Under (A1) full-support $\pi _ { t }$ and (A2) exact E-/M-steps $( p _ { t } ( \theta \mid x )$ the exact posterior under $\pi _ { t } ,$ , and $\begin{array} { r } { \pi _ { t + 1 } ( \theta ) = \int p _ { o } ( x ) p _ { t } ( \theta \mid x ) } \end{array}$ )dx exactly), every round satisfies

$$
D _ { \mathrm { K L } } \big ( p _ { o } ( x ) \| \pi _ { t + 1 } ( x ) \big ) \ \leq \ D _ { \mathrm { K L } } \big ( p _ { o } ( x ) \| \pi _ { t } ( x ) \big ) ,
$$

with equality iff $\cdot _ { \pi _ { t + 1 } } = \pi _ { t }$ a.e., provided the displayed quantities are finite. Full derivation in Appendix A

Remark 1 (Initialization) Monotonicity is informative only when $D _ { \mathrm { K L } } ( p _ { o } \Vert \pi _ { 0 } ) < \infty ;$ in particular, ifthe initial induced marginal $\pi _ { 0 } ( x )$ assigns zero density to a region ofpositive $p _ { o }$ -mass, the initial KL is infinite and the bound is vacuous throughout.

## 3.4 Why Iteration Is Not Guaranteed to Help Fixed-Surrogate Baselines

Our update targets self-consistency directly; the baselines target it only indirectly. Under (A2), our M-step is the self-consistency operator $\begin{array} { r } { \pi _ { t + 1 } ( \theta ) = \int p _ { o } ( x ) \overline { { p } } _ { t } ( \theta \mid x ) \overline { { d x } } } \end{array}$ , with $p _ { t } ( \theta \mid x )$ the Bayes posterior induced by $\pi _ { t }$ and the true likelihood $p ( x \mid { \bar { \theta } } ) ;$ at a fixed point, $\pi _ { t + 1 } = \pi _ { t }$ reduces directly to the necessary condition above. In practice $p _ { t } ( \theta \mid x )$ is approximated by an NPE model $\hat { p } _ { t } ( \theta \mid x )$ , giving $\begin{array} { r } { \hat { \pi } _ { t + 1 } ( \dot { \theta } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \hat { p } _ { t } ( \theta \mid x _ { i } ) } \end{array}$ , an approximation to, not an exact instance of, the self-consistency operator.

NEB and Sourcerer approximate a different object. Surrogate methods approximate $p ( x \mid \theta )$ and then optimize an SDE objective under that approximation; our method instead approximates the Bayes posterior induced by the current source and true simulator samples, and therefore approximates the self-consistency operator directly. Self-consistency is thus not what the baselines’ update evaluates but a byproduct, depending on the surrogate’s accuracy and coverage. One could iterate this procedure: retrain the surrogate under the previously fitted source, then re-optimize the source, and repeat. But each round’s objective is still stated purely in terms of that round’s frozen surrogate rather than the self-consistency operator itself.

Table 1: $\pi _ { 0 }$ specifications for all four variants, alongside the true prior $\pi ^ { \star }$
<table><tr><td></td><td>Two Moons</td><td>SLCP</td><td>LV</td></tr><tr><td> $\pi ^ { \star } .$  true</td><td> $\mathrm { U n i f } ( [ - 1 , 1 ] ^ { 2 } )$ </td><td> $\operatorname { U n i f } ( [ - 3 , 3 ] ^ { 5 } )$ </td><td> $\alpha , \gamma \mathrm { { \sim L o g N ( - 0 . 1 2 5 , 0 . 5 ) , } } \beta , \delta \mathrm { { \sim L o g N ( - 3 , 0 . 5 ) } }$ </td></tr><tr><td>gaussian</td><td> $\mathcal { N } ( 0 , 3 ^ { 2 } I )$ </td><td> $\mathcal { N } ( 0 , 9 ^ { 2 } I )$ </td><td> $\mathcal { N } ( ( 1 . 0 , 0 . 0 5 6 , 1 . 0 , 0 . 0 5 6 ) , \mathrm { d i a g } ( 1 . 6 5 , 0 . 0 9 , 1 . 6 5 , 0 . 0 9 ) ^ { 2 } )$ </td></tr><tr><td>widebox</td><td> $[ - \mathrm { i } 0 , 1 0 ] ^ { 2 }$ </td><td> $[ - \mathrm { i } 0 , 1 0 ] ^ { 5 }$ </td><td> $[ 0 , 1 0 ] \times [ 0 , \dot { 1 } ] \times [ 0 , \dot { 1 } 0 ] \times [ 0 , 1 ]$ </td></tr><tr><td>shift_half</td><td> $[ - 0 . 5 , 1 . \dot { 5 } ] ^ { 2 }$ </td><td> $[ - 1 . 5 , 4 . { \dot { 5 } } ] ^ { 5 }$ </td><td> $[ 0 . 6 , 2 . 2 ] \times [ 0 . 0 3 3 , 0 . 1 2 3 ] \times [ 0 . 6 , \dot { 2 } . 2 ] \times [ 0 . 0 3 3 , 0 . 1 2 3 ]$ </td></tr><tr><td>shift_full</td><td> $[ 0 , 2 ] ^ { 2 }$ </td><td> $[ 0 , 6 ] ^ { 5 }$ </td><td> $[ 1 . 4 , 3 . 0 ] \times [ 0 . 0 8 , 0 . 1 7 ] \times [ 1 . 4 , 3 . 0 ] \times [ 0 . 0 8 , 0 . 1 7 ]$ </td></tr></table>

Table 2: Data-space C2ST under broad prior gaussian and widebox $\pi _ { 0 }$
<table><tr><td></td><td colspan="3">gaussian</td><td colspan="3">widebox</td></tr><tr><td>Method</td><td>Two Moons</td><td>SLCP</td><td>LV</td><td>Two Moons</td><td>SLCP</td><td>LV</td></tr><tr><td>NEB</td><td> $0 . 5 1 4 \pm 0 . 0 1 3$ </td><td> $0 . 8 9 4 \pm 0 . 0 3 7$ </td><td> $0 . 9 9 3 \pm 0 . 0 0 4$ </td><td> $0 . 5 8 8 \pm 0 . 0 3 4$ </td><td> $0 . 6 5 0 \pm 0 . 0 4 1$ </td><td> $0 . 9 6 4 \pm 0 . 0 0 7$ </td></tr><tr><td>Sourcerer</td><td> $0 . 5 8 8 \pm 0 . 0 1 3$ </td><td> $0 . 8 4 4 \pm 0 . 0 3 5$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 6 0 1 \pm 0 . 0 1 3$ </td><td> $0 . 8 5 5 \pm 0 . 0 2 2$ </td><td> $0 . 9 9 9 \pm 0 . 0 0 1$ </td></tr><tr><td>NEB-it</td><td> $\mathbf { 0 . 5 1 3 \pm 0 . 0 0 2 }$ </td><td> $0 . 6 4 6 \pm 0 . 0 0 6$ </td><td> $0 . 9 8 7 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 4 9 8 \pm 0 . 0 1 0 }$ </td><td> $0 . 6 1 1 \pm 0 . 0 1 6$ </td><td> $0 . 9 9 2 \pm 0 . 0 0 4$ </td></tr><tr><td>Sourcerer-it</td><td> $0 . 5 7 2 \pm 0 . 0 0 7$ </td><td>0.934 ± 0.013</td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 5 9 1 \pm 0 . 0 0 9$ </td><td> $0 . 9 2 2 \pm 0 . 0 1 0$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>PA</td><td> $0 . 5 1 5 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 5 7 7 \pm 0 . 0 1 1 }$ </td><td> $0 . 9 1 4 \pm 0 . 0 0 8$ </td><td> $0 . 5 2 4 \pm 0 . 0 1 4$ </td><td> $0 . 5 4 7 \pm 0 . 0 1 5$ </td><td> $0 . 8 9 2 \pm 0 . 0 1 7$ </td></tr><tr><td>PA-Shared</td><td> $0 . 5 2 1 \pm 0 . 0 2 4$ </td><td> $0 . 5 8 4 \pm 0 . 0 2 2$ </td><td> $\mathbf { 0 . 6 5 5 \pm 0 . 0 2 7 }$ </td><td> $0 . 5 5 1 \pm 0 . 0 4 3$ </td><td> $\mathbf { 0 . 5 3 0 \pm 0 . 0 0 7 }$ </td><td> $\mathbf { 0 . 6 7 5 \pm 0 . 0 3 4 }$ </td></tr></table>

## 4 Experiments

## 4.1 Setup

Tasks and methods. We evaluate on three tasks : Two Moons with $\theta \in \mathbb { R } ^ { 2 }$ , SLCP with $\theta \in \mathbb { R } ^ { 5 }$ and Lotka–Volterra, LV, with $\theta \in \mathbb { R } ^ { 4 }$ . We use their true priors as an exact source of each task to generate the observations. Exact task definitions are in Appendix C. Every method sees $n _ { \mathrm { o b s } } = 1 0 { , } 0 0 0$ observed datasets per task.

We compare PA and PA-Shared against baselines NEB and Sourcerer, and their iterated variants NEBit and Sourcerer-it. There, in each round we re-simulate from the current source estimate, refit the surrogate from scratch, and refit the source, using the original authors’ unmodified implementations. PA, PA-Shared, NEB-it, and Sourcerer-it run 8 rounds at 4,000 fresh simulator calls per round; NEB and Sourcerer fit one frozen surrogate on the same total budget of 32,000 simulations. We report data-space classifier two-sample test [Lopez-Paz and Oquab, 2017] (C2ST, closer to 0.5 is better, meaning two distributions are indistinguishable), mean ± std over 3 seeds. The architecture and optimization details of PA and PA-Shared are in Appendix D.

Initial priors. We first study broad-support priors $\pi _ { 0 }$ centred on or containing $\pi ^ { \star } \colon$ gaussian, an unbounded diagonal Gaussian, and widebox, a uniform box with large margin (Section 4.2). Next, we examine misspecified priors with shifted support (Section 4.3). For this, we slide a box of $\pi ^  \star \} \mathbf { s }$ own width right so it only partially overlaps $\pi ^ { \star }$ , excluding a region of positive $\pi ^ { \star }$ -mass: shift\_half, a quarter-width offset, and shift\_full, a half-width offset that leaves the box near one-sided. Table 1 gives exact specifications alongside $\pi ^ { \star }$

## 4.2 Broad initial prior

Table 2 reports C2ST under the two broad-support variants. Our methods ranked best across nearly all task-prior combinations, with the exception of Two Moons, where NEB-it marginally outperformed our method: 0.513 vs. 0.515 under Gaussian, 0.498 vs. 0.524 under wide-box. NEB-it improved substantially over one-shot NEB in every cell, for example on SLCP dropping from 0.894 to 0.646 under gaussian and from 0.650 to 0.611 under widebox. Meanwhile Sourcerer-it was worse than one-shot Sourcerer on SLCP under both priors, rising from 0.844 to 0.934 and from 0.855 to 0.922, respectively. LV was the hardest task where most methods failed. PA-Shared clearly performed best at 0.655 and 0.675 for gaussian and widebox, respectively. PA performed worse (0.914 and 0.892), but still outperformed all baselines on both priors (> 0.964).

Table 3: Data-space C2ST under mild shift\_half and near one-sided shift\_full misspecification.
<table><tr><td></td><td colspan="3">shift_half</td><td colspan="3">shift_full</td></tr><tr><td>Method</td><td>Two Moons</td><td>SLCP</td><td>LV</td><td>Two Moons</td><td>SLCP</td><td>LV</td></tr><tr><td>NEB</td><td> $0 . 5 1 3 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 5 4 5 \pm 0 . 0 2 6 }$ </td><td> $0 . 9 7 9 \pm 0 . 0 0 8$ </td><td> $0 . 5 2 6 \pm 0 . 0 0 8$ </td><td> $0 . 5 9 2 \pm 0 . 0 1 4$ </td><td> $0 . 9 8 7 \pm 0 . 0 0 1$ </td></tr><tr><td>Sourcerer</td><td> $0 . 5 2 5 \pm 0 . 0 1 3$ </td><td> $0 . 5 8 4 \pm 0 . 0 2 1$ </td><td> $0 . 9 9 0 \pm 0 . 0 0 2$ </td><td> $0 . 5 3 6 \pm 0 . 0 1 3$ </td><td> $0 . 7 1 2 \pm 0 . 0 3 2$ </td><td> $0 . 9 9 3 \pm 0 . 0 0 1$ </td></tr><tr><td>NEB-it</td><td> $\mathbf { 0 . 4 9 7 \pm 0 . 0 0 7 }$ </td><td> $0 . 5 8 6 \pm 0 . 0 0 8$ </td><td> $0 . 9 8 6 \pm 0 . 0 0 9$ </td><td> $\mathbf { 0 . 5 0 2 \pm 0 . 0 0 6 }$ </td><td> $0 . 6 1 2 \pm 0 . 0 0 9$ </td><td> $0 . 9 9 2 \pm 0 . 0 0 7$ </td></tr><tr><td>Sourcerer-it</td><td> $0 . 5 2 8 \pm 0 . 0 1 1$ </td><td> $0 . 9 1 3 \pm 0 . 0 0 7$ </td><td> $0 . 9 9 5 \pm 0 . 0 0 4$ </td><td> $0 . 5 2 8 \pm 0 . 0 1 1$ </td><td> $0 . 9 1 5 \pm 0 . 0 1 4$ </td><td> $0 . 9 9 8 \pm 0 . 0 0 1$ </td></tr><tr><td>PA</td><td> $0 . 5 2 1 \pm 0 . 0 1 2$ </td><td> $0 . 5 8 9 \pm 0 . 0 0 7$ </td><td> $0 . 6 8 3 \pm 0 . 0 1 7$ </td><td> $0 . 5 1 6 \pm 0 . 0 0 9$ </td><td> $0 . 7 1 3 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 9 0 3 \pm 0 . 0 3 8 }$ </td></tr><tr><td>PA-Shared</td><td> $0 . 5 2 9 \pm 0 . 0 0 4$ </td><td> $0 . 5 6 1 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 6 4 3 \pm 0 . 0 1 0 }$ </td><td> $0 . 5 2 4 \pm 0 . 0 1 5$ </td><td> $\mathbf { 0 . 5 8 6 \pm 0 . 0 2 1 }$ </td><td> $0 . 9 8 4 \pm 0 . 0 0 5$ </td></tr></table>

![](images/452b62e1f54a0d18e23a3106c16ec1b762c470458b8a79d14ecaa0ae9a8b686d.jpg)  
NEB Sourcerer NEB-it Sourcerer-it PA (ours) PA-Shared (ours)  
Figure 1: Data-space C2ST by round for the four iterative methods: NEB-it, Sourcerer-it, PA, and PA-Shared. One-shot baselines (flat by construction) are included for reference.

## 4.3 Misspecified initial prior

Table 3 reports C2ST under the two misspecified prior settings. On Two Moons, NEB-it again performed marginally best, 0.497 vs. 0.521 for PA under shift\_half and 0.502 vs. 0.516 under shift\_full. On the other two tasks, iteration was unreliable for both baselines, more so than under the broad priors: NEB-it now lost to one-shot NEB on SLCP in both conditions, rising from 0.545 to 0.586 under shift\_half and from 0.592 to 0.612 under shift\_full. Sourcerer-it again performed substantially worse than one-shot Sourcerer on SLCP, rising from 0.584 to 0.913 under shift\_half and from 0.712 to 0.915 under shift\_full. In LV, under shift\_half, both our methods clearly outperformed the baselines: PA at 0.683 and PA-Shared at 0.643. Under shift\_full, PA reached 0.903 while PA-Shared failed at 0.984, similarly to the baselines.

## 4.4 Convergence across rounds

Figure 1 plots per-round C2ST for the four iterative methods. NEB-it and both our methods generally descended across rounds. In contrast, Sourcerer-it improved for Two Moons, stayed constant for LV, but deteriorated over rounds for the SLCP task.

## 5 Discussion and Conclusion

We started from the observation that existing SDE methods state their objective purely in terms of a surrogate likelihood trained once under a fixed proposal prior, so agreement under the surrogate need not imply agreement under the true simulator. Posterior averaging sidesteps this by never forming a surrogate likelihood at all: each round trains a posterior on fresh draws from the current source and refits the source to that posterior averaged over the observed data, which we prove decreases the induced-data KL monotonically under idealized conditions. Empirically, it consistently outperforms fixed-surrogate baselines and their iterated variants, most clearly on Lotka–Volterra, where iteration alone does not close the surrogate gap.

Limitations. Because multiple sources induce the same marginal, our evaluation speaks to reproducing $p _ { o }$ and not to recovering $\pi ^ { \star } \colon$ a source scoring well in data space may still place mass where a domain scientist would not. Proposition 1 assumes exact E- and M-steps, whereas both are neural approximations in practice, and Remark 1 makes it vacuous whenever $D _ { \mathrm { K L } } ( p _ { o } \Vert \pi _ { 0 } ) = \infty$ Our black-box assumption excludes Sourcerer’s primary sample-based variant, which differentiates through the simulator directly; so we compare only against its surrogate-likelihood form. Finally, we evaluate on three benchmark simulators with synthetic $p _ { o } ,$ a single simulation budget, n = 10,000 observations, and three seeds; behaviour on real observations, at smaller $n ,$ and on higher-dimensional simulators remains open.

Concurrent work: Simulation-Based Empirical Bayes (SBEB) Concurrently with this work, Shen et al. [2026] propose a closely related method with an equivalent convergence guarantee, matching our own monotonicity result (Proposition 1). While we make no priority claim relative to this work, we introduce some differences: SBEB is framed as simulation-based empirical Bayes and represents the fitted population prior implicitly through posterior samples, whereas our SDE formulation additionally fits an explicit queryable source density and studies its relation to existing SDE baselines, both fixed-surrogate and iterative (Section 3.4).

## Acknowledgments.

This project was supported by the Diabetes Center Berne and strategic funding of the medical faculty of the University of Bern. Calculations were performed on UBELIX, the HPC cluster at the University of Bern.

## References

Jan Boelts, Michael Deistler, Manuel Gloeckler, Álvaro Tejero-Cantero, Jan-Matthis Lueckmann, Guy Moss, Peter Steinbach, Thomas Moreau, Fabio Muratore, Julia Linhart, Conor Durkan, Julius Vetter, Benjamin Kurt Miller, Maternus Herold, Abolfazl Ziaeemehr, Matthijs Pals, Theo Gruner, Sebastian Bischoff, Nastya Krouglova, Richard Gao, Janne K. Lappalainen, Bálint Mucsányi, Felix Pei, Auguste Schulz, Zinovia Stefanidi, Pedro Rodrigues, Cornelius Schröder, Faried Abu Zaid, Jonas Beck, Jaivardhan Kapoor, David S. Greenberg, Pedro J. Gonçalves, and Jakob H. Macke. sbi reloaded: a toolkit for simulation-based inference workflows. Journal ofOpen Source Software, 10 (108):7754, 2025. doi: 10.21105/joss.07754. URL https://doi.org/10.21105/joss.07754.

Kyle Cranmer, Johann Brehmer, and Gilles Louppe. The frontier of simulation-based inference. Proceedings of the National Academy of Sciences, 117(48):30055–30062, 2020. doi: 10.1073/ pnas.1912789117. URL https://www.pnas.org/doi/abs/10.1073/pnas.1912789117.

A. P. Dempster, N. M. Laird, and D. B. Rubin. Maximum likelihood from incomplete data via the em algorithm. Journal ofthe Royal Statistical Society. Series B (Methodological), 39(1):1–38, 1977. ISSN 00359246. URL http://www.jstor.org/stable/2984875.

David Greenberg, Marcel Nonnenmacher, and Jakob Macke. Automatic posterior transformation for likelihood-free inference. In Ruslan Chaudhuri, Kamalika;Salakhutdinov, editor, Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings ofMachine Learning Research. PMLR, 09–15 Jun 2019. URL http://proceedings.mlr.press/v97/ greenberg19a.html.

David Lopez-Paz and Maxime Oquab. Revisiting classifier two-sample tests. In International Conference on Learning Representations, 2017. URL https://openreview.net/forum?id= SJkXfE5xx.

Jan-Matthis Lueckmann, Jan Boelts, David Greenberg, Pedro Goncalves, and Jakob Macke. Bench marking simulation-based inference. In Arindam Banerjee and Kenji Fukumizu, editors, Proceedings ofThe 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings ofMachine Learning Research, pages 343–351. PMLR, 13–15 Apr 2021.

George Papamakarios and Iain Murray. Fast ϵ-free inference of simulation models with bayesian conditional density estimation. In D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 29. Curran Associates, Inc., 2016. URL https://proceedings.neurips.cc/paper\_files/paper/2016/ file/6aca97005c68f1206823815f66102863-Paper.pdf.

Xinwei Shen, Diana Cai, Cheng Zhang, and David M. Blei. Simulation-based empirical bayes, 2026. URL https://arxiv.org/abs/2607.21843.

Vincent Stimper, David Liu, Andrew Campbell, Vincent Berenz, Lukas Ryll, Bernhard Schölkopf, and José Miguel Hernández-Lobato. normflows: A pytorch package for normalizing flows. Journal of Open Source Software, 8(86):5361, 2023. doi: 10.21105/joss.05361. URL https: //doi.org/10.21105/joss.05361.

Maxime Vandegar, Michael Kagan, Antoine Wehenkel, and Gilles Louppe. Neural empirical bayes: Source distribution estimation and its applications to simulation-based inference. In Proceedings of AISTATS 2021, 2021. URL https://arxiv.org/abs/2011.05836.

Julius Vetter, Guy Moss, Cornelius Schröder, Richard Gao, and Jakob H. Macke. Sourcerer: Samplebased maximum entropy source distribution estimation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id= 0cgDDa4OFr.

David J. Warne, Ruth E. Baker, and Matthew J. Simpson. Simulation and inference algorithms for stochastic biochemical reaction networks: from basic concepts to state-of-the-art. Journal ofThe Royal Society Interface, 16(151):20180943, 02 2019. ISSN 1742-5689. doi: 10.1098/rsif.2018. 0943. URL https://doi.org/10.1098/rsif.2018.0943.

Zikang Yao, Eugene Yee, and Fue-Sang Lien. Efficient probabilistic inference for expensiveto-evaluate black-box models for full-scale nuclear reactor thermal hydraulics. Energy, 348: 140531, 2026. ISSN 0360-5442. doi: https://doi.org/10.1016/j.energy.2026.140531. URL https: //www.sciencedirect.com/science/article/pii/S0360544226006341.

## A Convergence Proof

## A.1 Full KL Derivation

$$
D _ { \mathrm { K L } } \big ( p _ { o } ( x ) \| \pi _ { t + 1 } ( x ) \big ) = - H [ p _ { o } ( x ) ] - \int p _ { o } ( x ) \log \pi _ { t + 1 } ( x ) d x\tag{1}
$$

$$
= - H [ p _ { o } ( x ) ] - \int p _ { o } ( x ) \log \left( \int p _ { t } ( \theta \mid x ) { \frac { \pi _ { t + 1 } ( \theta ) p ( x \mid \theta ) } { p _ { t } ( \theta \mid x ) } } d \theta \right) d x\tag{2}
$$

$$
\leq - H [ p _ { o } ( x ) ] - \int \int p _ { o } ( x ) p _ { t } ( \theta \mid x ) \log \left( { \frac { \pi _ { t + 1 } ( \theta ) p ( x \mid \theta ) } { p _ { t } ( \theta \mid x ) } } \right) d \theta d x\tag{3}
$$

$$
= - H [ p _ { o } ( x ) ] - \int \int p _ { o } ( x ) p _ { t } ( \theta \mid x ) \Big [ \log \pi _ { t + 1 } ( \theta ) + \log p ( x \mid \theta ) - \log p _ { t } ( \theta \mid x ) \Big ] d \theta d x\tag{4}
$$

$$
= - H [ p _ { o } ( x ) ] - \int \pi _ { t + 1 } ( \theta ) \log \pi _ { t + 1 } ( \theta ) d \theta
$$

$$
- \iint p _ { o } ( x ) p _ { t } ( \theta \mid x ) \Big [ \log p ( x \mid \theta ) - \log p _ { t } ( \theta \mid x ) \Big ] d \theta d x\tag{5}
$$

$$
\leq - H [ p _ { o } ( x ) ] - \int \pi _ { t + 1 } ( \theta ) \log \pi _ { t } ( \theta ) d \theta
$$

$$
- \iint p _ { o } ( x ) p _ { t } ( \theta \mid x ) \Big [ \log p ( x \mid \theta ) - \log p _ { t } ( \theta \mid x ) \Big ] d \theta d x\tag{6}
$$

$$
= - H [ p _ { o } ( x ) ] - \iint p _ { o } ( x ) p _ { t } ( \theta \mid x ) \Big [ \log \pi _ { t } ( \theta ) + \log p ( x \mid \theta ) - \log p _ { t } ( \theta \mid x ) \Big ] d \theta d x\tag{7}
$$

$$
= - H [ p _ { o } ( x ) ] - \int \int p _ { o } ( x ) p _ { t } ( \theta \mid x ) \log \pi _ { t } ( x ) d \theta d x\tag{8}
$$

$$
= - H [ p _ { o } ( x ) ] - \int p _ { o } ( x ) \log \pi _ { t } ( x ) d x \ = \ D _ { \mathrm { K L } } \big ( p _ { o } ( x ) \| \pi _ { t } ( x ) \big ) .\tag{9}
$$

## A.2 Well-Definedness and Equality Condition

Well-definedness of (2): wherever $\pi _ { t + 1 } ( \theta ) p ( x \mid \theta ) > 0$ , both factors are positive, so $p ( x \mid \theta ) > 0 ;$ by $( \mathrm { A l } ) , \dot { \pi } _ { t } ( \theta ) > 0$ and $\pi _ { t } ( x ) > 0 , \operatorname { s o } p _ { t } ( \theta \mid x ) > 0$ exactly where needed; no support relation between $\pi _ { t + 1 }$ and $\pi _ { t }$ beyond (A1) is required.

(3): Jensen on concave log under $p _ { t } ( \theta \mid x ) d \theta$ . (4): splits the log-product before integrating. (5) and (7): Fubini, via (A2), to move between $\textstyle \int p _ { o } ( x ) \bar { p _ { t } } ( \theta \mid x )$ dx and $\pi _ { t + 1 } ( \theta )$ . (6) applies Gibbs inequality, using that $\pi _ { t + 1 }$ maximizes $\int \pi _ { t + 1 } \log { \mathfrak { C } }$ over q, so swapping in log $\pi _ { t }$ cannot decrease the bound. (8): the exact-posterior identity $\pi _ { t } ( \theta ) p ( x \mid \theta ) / p _ { t } ( \theta \mid x ) = \pi _ { t } ( x )$ , constant in θ.

Equality condition. $D _ { \mathrm { K L } } ( p _ { o } \| \pi _ { t + 1 } ) = D _ { \mathrm { K L } } ( p _ { o } \| \pi _ { t } ) { \mathrm { i f f } } \pi _ { t + 1 } = \pi _ { t } { \mathrm { a . e } } $ ., by strictness of Gibbs’ inequality unless the two priors agree, provided the displayed KL quantities are finite.

## B Algorithm Details

## B.1 PA (Separate Flows)

Algorithm 2 PA with separate flows   
Require: Simulator $p ( x \mid \theta )$ , data ${ \mathcal { D } } ,$ initial prior $\phi ^ { ( 0 ) }$ , rounds T   
1: repeat   
2: E-step: simulate $\theta _ { j } \sim \pi _ { \phi ^ { ( t ) } } , x _ { j } \sim p ( x \mid \theta _ { j } ) ; \psi ^ { ( t + 1 ) }  \arg \operatorname* { m i n } _ { \psi } \mathcal { L } _ { \mathrm { N P E } } ( \psi ; \pi _ { \phi ^ { ( t ) } } )$   
3: M-step: for each $x _ { i } \in \mathcal { D }$ , draw $\theta _ { s } \sim p _ { \psi ^ { ( t + 1 ) } } ( \theta \mid x _ { i } ) ; \phi ^ { ( t + 1 ) }  \arg \operatorname* { m i n } _ { \phi } \mathcal { L } _ { \mathrm { s o u r c e } } ( \phi )$   
4: $t \gets t ^ { - } + 1$   
5: until $t = T$ or converged

## B.2 PA-Shared (Shared Flow)

Algorithm 3 PA with shared flow   
Require: Simulator $p ( x \mid \theta )$ , data D, initial $\phi ^ { ( 0 ) }$ , dropout $p _ { \emptyset }$ , rounds $T$   
1: repeat   
2: E-step: simulate $\theta _ { j } \sim \pi _ { \phi ^ { ( t ) } } , x _ { j } \sim p ( x \mid \theta _ { j } )$ , apply CFG dropout; $( \phi _ { \mathrm { E } } ^ { ( t + 1 ) } , \xi ^ { ( t + 1 ) } ) \ $   
arg min $\mathcal { L } _ { \mathrm { E } } ( \phi , \xi )$   
3: M-step: embed $e _ { i } = h _ { \xi ^ { ( t + 1 ) } } \big ( x _ { i } \big )$ ; draw $\theta _ { s } \sim f _ { \phi _ { \mathrm { E } } ^ { ( t + 1 ) } } ( z , e _ { i } )$ ▷ frozen E-step output, before   
M-step’s own gradient   
4: ${ \phi } ^ { ( t + 1 ) } \gets \arg \operatorname* { m i n } _ { \phi } \mathcal { L } _ { \mathrm { p r i o r } } ( \phi )$ , initialized from $\phi _ { \mathrm { E } } ^ { ( t + 1 ) }$ , at c = e<sub>∅</sub>; t ← t + 1   
5: until $t = T$ or converged

## C Task Definitions

We utilize three tasks from the sbibm simulation-based inference benchmark suite [Lueckmann et al., 2021]. We reuse the corresponding simulators and treat their true priors as one of the true source distributions.

## C.1 Two Moons

A two-dimensional task whose true prior is Unif(−1, 1) per dimension. Given $\theta = \left( \theta _ { 1 } , \theta _ { 2 } \right)$ , data are generated by combining a fixed-radius arc with a translation depending on θ:

$$
\begin{array} { r l } & { \quad x \mid \theta = \left( \stackrel { r \cos \alpha + 0 . 2 5 } { r \sin \alpha } \right) + \left( - \vert \theta _ { 1 } + \theta _ { 2 } \vert / \sqrt { 2 } \right) , \qquad \alpha \sim \mathrm { U n i f } ( - \frac { \pi } { 2 } , \frac { \pi } { 2 } ) , \ r \sim \mathcal { N } ( 0 . 1 , 0 . 0 1 ^ { 2 } ) . } \\ & { \quad \theta \in \mathbb { R } ^ { 2 } , x \in \mathbb { R } ^ { 2 } . } \end{array}
$$

## C.2 SLCP

A five-dimensional task whose true prior is Unif(−3, 3) per dimension. The observation is four i.i.d. draws from a 2-D Gaussian whose mean and covariance are nonlinear functions of θ:

$$
x = ( x _ { 1 } , \ldots , x _ { 4 } ) , \quad x _ { i } \sim \mathcal { N } ( m _ { \theta } , S _ { \theta } ) , \qquad m _ { \theta } = \left( \theta _ { 2 } \right) , \quad S _ { \theta } = \left( { s _ { 1 } ^ { 2 } } _ { s } { \rho } s _ { 1 } s _ { 2 } \right) ,
$$

with $s _ { 1 } = \theta _ { 3 } ^ { 2 } , s _ { 2 } = \theta _ { 4 } ^ { 2 } , \rho =$ tanh $\theta _ { 5 }$ . Concatenating the four draws gives $x \in \mathbb { R } ^ { 8 }$

## C.3 Lotka–Volterra

A predator-prey ODE model, included as the one real (non-toy) simulator in our evaluation. Four rate parameters $\theta = ( \alpha , \beta , \gamma , \delta )$ govern

$$
\frac { d X } { d t } = \alpha X - \beta X Y , \qquad \frac { d Y } { d t } = - \gamma Y + \delta X Y ,
$$

with fixed initial condition $( X ( 0 ) , Y ( 0 ) ) = ( 3 0 , 1 )$ and integration horizon $T = 2 0$ . The true (sbibm) prior is

$$
\alpha , \gamma \sim \mathrm { { L o g N o r m a l } } ( - 0 . 1 2 5 , 0 . 5 ) , \quad \beta , \delta \sim \mathrm { { L o g N o r m a l } } ( - 3 , 0 . 5 )
$$

Observations are noisy log-population counts at 10 evenly spaced time points for each species, $x _ { 1 , i } \sim \mathrm { L o g N o r m a l } ( \mathrm { l o } _ { \mathbf { g } } ^ { \sim } X _ { i } ^ { \sim } , \hat { 0 } . 1 \big ) , x _ { 2 , i } \sim \mathrm { L o g N o r m a l } ( \mathrm { l o g } Y _ { i } ^ { \sim } , 0 . \dot { 1 } ) , \mathrm { g i v i n g } x ^ { \star } \in \mathbb { R } ^ { 2 0 }$

## D Architecture and Optimization

All hyperparameters are fixed across tasks, seeds, and both experiments unless noted.

## D.1 PA (Separate Flows)

• Disjoint parameters for the source $p _ { \phi } ( \theta )$ and the posterior $q _ { \psi } ( \theta \mid x )$

• Source flow $p _ { \phi } \mathbf { : }$ normflows [Stimper et al., 2023], diagonal-Gaussian base, $6 \times$ [autoregressive rational-quadratic spline coupling (2 bins, hidden $6 4 ) + \mathrm { L U }$ linear permute].

• Posterior $q _ { \psi } \mathbf { : }$ sbi [Boelts et al., 2025] NPE with the nsf density estimator. Training pairs are drawn from $p _ { \phi }$

• E-step: sample $n _ { \mathrm { s i m } }$ parameters $\theta \sim p _ { \phi }$ , simulate, retrain the NPE (batch size of 256, max epochs of 50, early-stop patience 10).

• M-step: target $\begin{array} { r } { \hat { p } ( \theta ) = \frac { 1 } { N } \sum _ { i } q _ { \psi } ( \theta \mid x _ { i } ^ { \mathrm { r e a l } } ) } \end{array}$ ; draw 300 posterior samples per $x _ { i } ^ { \mathrm { r e a l } }$ and fit $p _ { \phi }$ by maximum likelihood (300 steps, batch 512, gradient-norm clip 5).

## D.2 PA-Shared (Shared Flow)

• A single conditional flow ϕ: the same base and 6-block structure as the source flow of PA, but each spline coupling additionally consumes a 32-dim context.

$p _ { \phi } ( \theta ) = \mathrm { f o w } ( z , c { = } e _ { \mathcal { O } } )$ and $q _ { \phi } ( \theta \mid x ) = \operatorname { f l o w } ( z , c { = } h _ { \xi } ( x ) )$ , with $e _ { \mathcal { O } } \in \mathbb { R } ^ { 3 2 }$ a learnable null embedding (init. $\mathcal { N } ( 0 , 0 . 0 2 ^ { 2 } ) )$ .

• Embedder $h _ { \xi } \colon \mathrm { M L P ~ } d _ { x } \to 6 4 \to 6 4 \to 3 2 ,$ , SiLU (a CNN for image-valued x). Inputs are z-scored per feature using the statistics of $\{ x _ { i } ^ { \mathrm { r e a l } } \}$

• Classifier-free guidance: during the E-step the context is replaced by $e _ { \emptyset }$ with probability $p _ { \mathrm { u n c o n d } } = 0 . 2 ;$ at inference $c = e _ { \infty } + w \left( h _ { \xi } ( x ) - e _ { \infty } \right)$ with $w = 1$ throughout.

• E-step: sample from the null-context branch, simulate, train ϕ and ξ jointly on the conditional NLL of $\theta \mid h _ { \xi } ( x )$ (batch size of 256, max epochs of 60, patience 10, gradient-norm clip 5).

• M-step: target $p _ { \phi } ( \theta ) = \mathbb { E } _ { x ^ { \mathrm { r e a l } } } [ q _ { \phi } ( \theta \mid x ) ]$ ; fit the null-context branch by maximum likelihood (300 steps, batch size of 256) on 100 posterior samples per $x _ { i } ^ { \mathrm { r e a l } }$

• Shared $\phi \Rightarrow { \mathrm { s n a p s h o t } } \phi ^ { ( t ) }$ before the E-step, draw the M-step samples from the frozen $\phi ^ { ( t ) }$ combined with the updated embedder $\xi ^ { ( t + 1 \bar { ) } }$ , then restore the E-step-updated weights before the M-step gradient steps.

## D.3 Optimization Configuration

Table 4: Optimization configuration for PA and PA-Shared.
<table><tr><td></td><td>PA</td><td>PA-Shared</td></tr><tr><td>Init pre-fit to round-0 prior</td><td>800 steps, Adam 3e-3, batch 256 sbi NPE (Adam), batch 256, ≤50</td><td>800 steps, Adam 3e-3, batch 256</td></tr><tr><td>E-step optimizer</td><td>ep, patience 10</td><td>Adam 1e-3 (flow+embedder), batch  $2 5 6 , \leq 6 0$  ep, patience 10</td></tr><tr><td>M-step optimizer</td><td>Adam 1e-3, 300 steps, batch 512</td><td>Adam 1e-3 (shared), 300 steps, batch 256</td></tr></table>

## D.4 Protocol

• Budget (matched, all methods): $T = 8$ rounds; 4000 $( \theta , x )$ pairs per E-step; 32,000 total. No accumulation across rounds — each E-step uses fresh draws from the current source.

• Observed set: $N = 1 0 { , } 0 0 0$ , drawn once from the true prior.

• Tasks (sbibm): two\_moons $( d _ { \theta } \mathrm { { = } 2 , ~ } d _ { x } \mathrm { { = } 2 ) , ~ } \mathtt { s l c p } \left( 5 , \mathtt { \otimes } 8 \right)$ , lotka\_volterra (4, 20). Seeds: 0, 1, 2.

• Metric: data-space C2ST per round (classifier two-sample test, 1000 samples per side, between x simulated from the round-t learned source and $x ^ { \mathrm { r e a l } } ; 0 . 5 =$ indistinguishable). We report the performance in the last round and aggregate it over the 3 seeds.

• Compute: a single RTX 4090; the simulator runs on CPU.