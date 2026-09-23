# On Basis Function Selection for Sparse Gaussian Process Regression

Marnix Van Soom AI Lab Vrije Universiteit Brussel Pleinlaan 2, 1050 Brussels, Belgium

Ivan De Boi InViLab University of Antwerp Groenenborgerlaan 179, 2020 Antwerp, Belgium

## Abstract

Sparse Gaussian processes achieve $\mathcal { O } ( N )$ inference by replacing the kernel with an appropriate expansion in a fixed basis $\{ \phi _ { j } \}$ on the input space. Given a compute budget $M \ll { \bar { N } } ,$ , practitioners conventionally truncate the basis to its first M entries. Nothing in the formalism, however, prevents one from selecting only those M basis functions that matter for the data at hand. This would avoid spending budget on basis functions where there is no signal, but it requires a criterion for ranking the candidates. We propose three such criteria derived from an informationtheoretic view of the basis-function selection problem. Each criterion matches a different state of knowledge at selection time: (no data), (no prior), and an (inbetween) state. We then study the performance of truncation vs. selection strategies on six UCI regression benchmarks across three basis families: Hilbert-space Gaussian processes (HSGP), variational Fourier features (VFF), and variational inducing spherical harmonics (VISH). We observe that the (no data) criterion is a safe default, matching or improving on truncation for HSGP, VFF, and VISH, with substantial gains for VISH and improvements over a recently developed selection heuristic for that basis family. The data-aware (no-prior) and (in-between) criteria provide substantial gains over truncation specifically for HSGP, which is the most broadly used of the three families in practice.

## 1 Introduction

Gaussian processes [17] are a powerful class of models for regression. Given N noisy observations $\mathbf { y } = ( y _ { 1 } , \dots , y _ { N } )$ at inputs $\boldsymbol { X } ^ { \prime } = ( x _ { 1 } , \ldots , x _ { N } ) \in \mathbb { R } ^ { N \times D }$ , one models the response through a latent function f with the prior $f \sim \mathcal { G P } ( 0 , k _ { \theta } )$ , and learns the kernel hyperparameters θ from the data. Vanilla inference is, however, $\mathcal { O } ( N ^ { 3 } )$ in time because of the kernel-matrix solve, which is prohibitive for the moderately large N encountered in practice.

Sparse GPs [12] avoid this cost by replacing the kernel with an expansion in a basis $\{ \phi _ { j } \}$ on the input space, informally written as

$$
f ( x ) = \sum _ { j = 1 } ^ { \infty } w _ { j } \phi _ { j } ( x ) , \qquad w \sim { \mathcal { N } } \big ( 0 , \Sigma ( \theta ) \big ) .\tag{1}
$$

The sparse-GP method fixes both the basis $\{ \phi _ { j } \}$ and the covariance $\Sigma ( \theta )$ on the coefficients, and typically also requires additional user choices, such as the input domain on which the GP is approximated. One such choice stands out: the number $M \ll N$ of basis functions actually used for inference. Computing the marginal likelihood and the predictive mean scales as $\mathcal { O } ( N M ^ { 2 } + M ^ { p } )$ with p depending on the method, so M is essentially the user’s compute budget. The conventional way to spend that budget is to truncate the expansion in Equation (1) to its first M entries, in some ordering of the index $j .$

![](images/ca45bcdabb082a742d8bfeab9f78be8223553e5c8140de55ab7cfbf1ea313f11.jpg)  
truncation

![](images/546247d06fc0a4f3729f9d1953da071e4660aa8f583b4b05f7b28710625cec08.jpg)

![](images/e6b0f812d6f8e86f305ee1a1458aba5485e052bf918d40bac65bc1a2fdaf621b.jpg)  
(no prior)

![](images/f6bd2b6f97f58d1e7e6db42593e61619ecdc5460d7ea1a536c972abc2b23aead.jpg)  
Figure 1: The four basis-function selection strategies we consider in the paper, illustrated on a toy data set. From left to right: truncation $M _ { d } = ( 5 , \bar { 2 } )$ , the (no data) criterion $I _ { j } = \lambda _ { j }$ , the (no prior) criterion $P _ { j } = | a _ { j } | ^ { 2 }$ , and the (in-between) criterion $\tilde { H } _ { j } = \lambda _ { j } | a _ { j } | ^ { 2 }$ , computed as the pointwise product of the previous two. Selected basis functions are marked by black dots, and the test NLL is reported underneath each panel. Heatmap colours indicate the strength of the criterion (e.g. $I _ { j _ { 1 } , j _ { 2 } }$ in panel 2) at each grid point. On this example ${ \tilde { H } } _ { j }$ performs best by combining the low-frequency basis functions favoured by $I _ { j }$ with the higher-energy ones that $P _ { j }$ alone prefers.

For example, take a regression problem in $D = 2$ input dimensions and apply the Hilbert-space sparse-GP method, which is the most common of the three methods considered here (see Section 2 for the construction). The expansion in Equation (1) then takes the form

$$
f ( x _ { 1 } , x _ { 2 } ) \ = \ \sum _ { j _ { 1 } = 1 } ^ { \infty } \sum _ { j _ { 2 } = 1 } ^ { \infty } w _ { j _ { 1 } , j _ { 2 } } \phi _ { j _ { 1 } , j _ { 2 } } ( x _ { 1 } , x _ { 2 } ) , \qquad w _ { j _ { 1 } , j _ { 2 } } \sim \mathcal N \big ( 0 , \lambda _ { j _ { 1 } , j _ { 2 } } ( \theta ) \big ) ,\tag{2}
$$

where the basis function $\phi _ { j _ { 1 } , j _ { 2 } }$ is a product of two one-dimensional sinusoids, indexed by per-axis frequencies $( j _ { 1 } , j _ { 2 } )$ , and the prior variance $\lambda _ { j _ { 1 } , j _ { 2 } } ( \theta )$ is the kernel’s spectral density evaluated at the frequency corresponding to $( j _ { 1 } , j _ { 2 } )$ . Suppose now that our compute budget allows only $M = 1 0$ basis functions out of the infinite expansion: which ten pairs $( j _ { 1 } , j _ { 2 } )$ should we keep, that is, which frequencies should we model?

Figure 1 (left panel) shows one such allocation on a toy data set: the choice $M _ { d } = ( 5 , 2 )$ arranges its basis functions as a rectangle in the $( j _ { 1 } , j _ { 2 } )$ grid, with the test negative log-likelihood (NLL) reported underneath the panel. Such a rectangular allocation is the convention in available HSGP implementations: the user picks a count $M _ { d }$ per axis,<sup>1</sup> and the method then keeps every basis function with $j _ { d } \leq M _ { d }$ , giving $\begin{array} { r } { M = \prod _ { d = 1 } ^ { D } M _ { d } } \end{array}$ basis functions in total. This rule, however, is not unique even at fixed M: a budget of $M = \tilde { 1 0 }$ admits the allocations $( 5 , 2 ) , ( 2 , 5 ) , ( 1 0 , 1 )$ , and any other partition with the same product, and the user is left to pick by heuristics and trial and error.

This non-uniqueness in truncation points to a deeper fact: nothing in the sparse-GP formalism actually requires us to truncate sequentially at all. We can instead rank the candidates by any other criterion and keep the top M by that criterion. We call this selection, in contrast with the truncation baseline. The question is then which criterion to choose, and whether a cheap one can offer gains over truncation.

This paper proposes three such criteria, derived from an information-theoretic view of the problem. They are shown in the grey panels of Figure 1 alongside the truncation baseline. Each criterion produces a scalar value (the score) at every candidate index $j = ( j _ { 1 } , j _ { 2 } )$ , and selection keeps the top M candidates by that value.

We show in Section 3 that the criteria correspond to different states of knowledge at selection time. The (no data) criterion $I _ { j } = \lambda _ { j } ( \theta )$ ranks candidates by the kernel-prior weight alone (panel 2). The (no prior) criterion $P _ { j } = | a _ { j } | ^ { 2 }$ ranks by the squared data projection $a _ { j } : = \phi _ { j } ( X ) ^ { \top } y$ alone (panel 3). The (in-between) criterion $\tilde { H } _ { j } = \lambda _ { j } | a _ { j } | ^ { 2 }$ uses both factors (panel 4). Black dots mark the selected top-M = 10 basis functions, and each criterion picks a different ten. The resulting fit, and therefore the test NLL reported under each panel, varies with the choice of criterion.

HSGP truncation can only carve rectangular subsets out of the candidate grid. The three selection criteria are not bound by that shape. Even without looking at the data, the (no data) criterion $I _ { j }$ already departs from the rectangle and improves substantially on truncation. On this toy example, the (in-between) criterion $\tilde { H } _ { j }$ does best, notably better than the rectangular truncation.

We shall see that this pattern does not hold in general. We test whether selection improves over truncation on six UCI regression benchmarks across three sparse-GP methods: HSGP, variational Fourier features (VFF), and variational inducing spherical harmonics (VISH). The three methods are different but related, each with its own basis $\phi _ { j }$ and prior variance $\lambda _ { j } ( \theta )$ . Ranking by the (no data) criterion $I _ { j }$ is a safe default that matches or improves on truncation for all three methods, with the largest gains on VISH where it also outperforms the recently proposed phase-truncation heuristic of Eleftheriadis et al. [6]. The data-aware (no prior) criterion $P _ { j }$ and (in-between) criterion $\tilde { H } _ { j }$ offer substantial gains over truncation specifically for HSGP, the most broadly used of the three methods in practice.

The rest of the paper develops these ideas. Sections 2 and 3 fix notation and derive the three criteria. Sections 4–6 report the per-method experiments. Section 7 places the criteria in context, and Section 8 discusses what we learned.

## 2 Background

We consider the standard Gaussian process regression setting: a Gaussian process $f \sim \mathcal { G P } ( 0 , k _ { \theta } )$ on an input space ${ \mathcal { X } } \ni x = ( x _ { 1 } , \ldots , x _ { D } )$ observed at N inputs $X = ( x _ { 1 } , \dots , x _ { N } ) ^ { \intercal }$ with additive Gaussian noise $y _ { i } = f ( x _ { i } ) + \varepsilon _ { i } , \varepsilon _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ . The prior mean is zero throughout this paper, with any mean shift absorbed by standardising the response in pre-processing. Exact GP inference scales as $\mathrm { \dot { \cal O } } ( N ^ { 3 } )$ in compute and is infeasible at the dataset sizes we typically care about. We therefore work with sparse GPs throughout. The three methods we consider (HSGP, VFF, and VISH) each specify the sparse GP through a basis-function expansion of $f .$

Each method supplies an infinite basis-function expansion of the form Equation $( 1 ) . ^ { 2 }$ The index j is in general a multi-index whose structure is set by the method. In practice we keep only M of these basis functions, either by truncation (which keeps the first M in the natural ordering of $j )$ or by selection (which scores a finite candidate grid $\{ \phi _ { j } \} _ { j = 1 } ^ { J }$ and keeps the top M by score). The candidate grid size J is therefore needed only for selection, and in general $M \ll \mathsf { \bar { J } } \ll \dot { N }$ when N is large. The selection problem itself is to choose a subset $\mathcal { M } \subset \{ 1 , \dotsc , J \}$ of size $M ,$ , and to fit θ on that subset. The three methods specialise the basis and the prior as follows.

Hilbert-space Gaussian processes (HSGP) [24, 19] use Dirichlet Laplacian eigenfunctions on a box $[ - L , \bar { L } ] ^ { D }$ , indexed by $j = ( j _ { 1 } , \dots , j _ { D } )$ , with prior variance $\lambda _ { j } ( \theta ) = S _ { \theta } ( \omega _ { j } )$ at the Laplacian frequency vector $\omega _ { j } = ( \pi j _ { 1 } / 2 L , \ldots , \pi j _ { D } / 2 L ) ,$ ), and are the most broadly used of the three methods in practice. Details in Appendix C.

Variational Fourier features (VFF) [7] use an additive model $\begin{array} { r } { f ( x ) = \sum _ { d = 1 } ^ { D } f _ { d } ( x _ { d } ) } \end{array}$ where each $f _ { d }$ is expanded in windowed cosine and sine features at per-axis frequencies $j _ { d } .$ Candidates are indexed by $j = ( d , j _ { d } )$ , with prior variance $\lambda _ { j } ( \theta ) \approx S _ { \theta } ( \omega _ { j _ { d } } ^ { - } ) / T$ the kernel’s spectral density at the harmonic frequency on the per-axis box of length T. Details in Appendix D.

Variational inducing spherical harmonics (VISH) [5, 6] use spherical harmonics on $S ^ { D - 1 }$ for inputs lifted to the unit sphere by $z _ { n } = ( x _ { n } , 1 ) / \| ( x _ { n } , 1 ) \|$ , indexed by $j = ( \ell , k )$ for degree ℓ and orientation k, with prior variance $\lambda _ { j } ( \theta ) = \lambda _ { \ell } ( \theta )$ shared across orientations within a degree by Funk–Hecke. We use the order-1 arc-cosine kernel of Cho and Saul [3] throughout. Details in Appendix E.

For each method we follow the authors’ recommended training procedure: the closed-form marginal likelihood for HSGP, and the closed-form collapsed Titsias bound for VFF and VISH. We reproduce the published truncation baselines as faithfully as possible, with the full experimental protocol in Appendix G.

## 3 The selection criteria

Each candidate index j identifies a basis-function coefficient $u _ { j }$ in the expansion of Equation (1), with marginal prior $u _ { j } \sim \mathcal { N } ( 0 , \lambda _ { j } ( \theta ) )$ ) in all three methods (Appendix A). We want to know whether $u _ { j }$ is worth keeping in M. The natural quantity is how much observing the data y changes the belief about $u _ { j }$ , measured as the Kullback–Leibler divergence from posterior to prior:

$$
H _ { j } ( \theta , y ) : = D _ { \mathrm { K L } } \big ( p ( u _ { j } \mid y ) \| p ( u _ { j } ) \big ) .\tag{3}
$$

This is the per-basis-function information gain: the same $D _ { \mathrm { K L } }$ objective that Rasmussen and Williams [18] use for placing inducing points and Seeger et al. [21], Krause et al. [9] use for sensor selection, applied here to the discrete index $j$ that enumerates a fixed candidate basis. A coefficient $u _ { j }$ with large $H _ { j }$ is one whose belief is shifted strongly by the data, so a large $H _ { j }$ flags a candidate worth keeping. Since $H _ { j } \ \geq \ 0$ we use it directly as a non-negative scoring rule for the basis-function selection problem.

The prior $p ( u _ { j } ) = \mathcal { N } ( 0 , \lambda _ { j } ( \theta ) )$ is univariate Gaussian by construction, and the marginal posterior $p ( u _ { j } \mid \boldsymbol { y } )$ is univariate Gaussian by conjugacy in Bayesian linear regression on the selected basis. Equation (3) therefore has a closed form. With the per-basis-function data projection and signal-tonoise ratio

$$
a _ { j } : = \phi _ { j } ( \boldsymbol { X } ) ^ { \top } \boldsymbol { y } , \qquad \rho _ { j } : = \frac { \lambda _ { j } ( \theta ) \| \phi _ { j } ( \boldsymbol { X } ) \| ^ { 2 } } { \sigma ^ { 2 } } ,\tag{4}
$$

this closed form is

$$
H _ { j } ( \theta , y ) = \textstyle { \frac { 1 } { 2 } } \Big [ \log ( 1 + \rho _ { j } ) - \frac { \rho _ { j } } { 1 + \rho _ { j } } \Big ] + \frac { \lambda _ { j } ( \theta ) | a _ { j } | ^ { 2 } } { 2 \sigma ^ { 4 } ( 1 + \rho _ { j } ) ^ { 2 } } ,\tag{5}
$$

under the empirical-decoupling assumption $\Phi ^ { \top } \Phi \approx \mathrm { d i a g } ( \| \phi _ { j } ( X ) \| ^ { 2 } )$ , the standard regime in the basis-function case (Appendix B). The first bracket is a variance-shrinkage term that is independent of the data. The second bracket is the data-driven term. Evaluating Equation (5) requires both the noise scale $\sigma ^ { 2 }$ and the kernel eigenvalues $\lambda _ { j } ( \theta )$ , neither of which is fit yet at selection time.

Three limits, three criteria Three limits of $H _ { j }$ correspond to three states of knowledge at selection time about how the prior weight and the data interact, and each gives a closed-form ranking criterion on the candidate basis.

(i) The no-data limit: the eigenvalue criterion $I _ { j } .$ . Under the prior, $a _ { j }$ is mean-zero, and a short calculation (Appendix B) gives the data-averaged information gain

$$
\begin{array} { r } { \mathbb { E } _ { y } \big [ H _ { j } ( \theta , y ) \big ] ~ = ~ \frac { 1 } { 2 } \log ( 1 + \rho _ { j } ) , } \end{array}\tag{6}
$$

monotone in $\rho _ { j }$ and therefore in the prior weight $\lambda _ { j } ( \theta )$ at fixed θ and $\sigma ^ { 2 }$ . Ranking by Equation (6) therefore reduces to ranking by

$$
I _ { j } ( \theta ) : = \lambda _ { j } ( \theta ) .\tag{7}
$$

At fixed $\theta , I _ { j }$ ranks candidates by prior weight alone. It is the closest per-basis-function analogue of conventional truncation, since for stationary kernels the spectral density decays radially and $I _ { j }$ keeps the lowest-frequency basis functions first. The two need not agree, however: practitioner-default truncation enforces a structural shape (rectangular cube, harmonic shells, contiguous frequencies), while $I _ { j }$ ranks every candidate freely (Figure 1, panels 1 and 2).

(ii) The no-prior limit: the data-energy criterion $P _ { j }$ . Dropping the kernel factor in Equation (5) leaves the data-driven piece, $| \phi _ { j } ( X ) ^ { \top } y | ^ { 2 } / [ 2 \sigma ^ { 4 } ( 1 + \rho _ { j } ) ^ { 2 } ]$ . At fixed θ and $\sigma ^ { 2 }$ the prefactors are basis-function-independent up to $( 1 + \rho _ { j } ) ^ { 2 }$ , which itself is a function of $\lambda _ { j }$ . Ignoring that residual prior dependence gives the data-energy criterion

$$
P _ { j } ( y ) : = | \phi _ { j } ( X ) ^ { \top } y | ^ { 2 } ,\tag{8}
$$

the squared projection of the data onto $\phi _ { j }$ . A second, more principled, route arrives at the same ranking: $\mathrm { i f } \ \sigma ^ { 2 }$ and $\lambda _ { j }$ are jointly estimated from the data and substituted into the full Equation (5), the resulting per-basis-function score is rank-equivalent to $P _ { j }$ (Appendix B). The fully data-driven $H _ { j }$ therefore collapses to the data-only ordering.

Algorithm 1 Selecting basis functions by score   
1: Enumerate $J \gg M$ candidate basis functions $\{ \phi _ { j } \} _ { j = 1 } ^ { J }$ from the chosen basis family.   
2: Choose a starting point $\theta _ { 0 }$ for the kernel hyperparameters.   
3: Compute the kernel weights $\lambda _ { j } ( \theta _ { 0 } )$ for all candidates in $\mathcal { O } ( J )$ time.   
4: Compute the data projections $\dot { a } _ { j } : = \phi _ { j } ( X ) ^ { \top } y$ for all candidates in $\mathcal { O } ( N J )$ time.   
5: Choose one ranking score:   
$I _ { j } ( \theta _ { 0 } ) = \lambda _ { j } ( \theta _ { 0 } ) , \qquad P _ { j } ( y ) = | a _ { j } | ^ { 2 } , \qquad \tilde { H } _ { j } ( \theta _ { 0 } , y ) = \lambda _ { j } ( \theta _ { 0 } ) | a _ { j } | ^ { 2 } .$   
6: Rank candidates greedily in descending order by the chosen score in $\mathcal { O } ( J )$ time.   
7: Break ties by axis order, with axis 1 most important.   
8: Return the top-M indices.

(iii) The weak-observation limit: the criterion $\tilde { H } _ { j }$ . Selection happens before the kernel hyperparameters are fit, so the practical setting is one of weak observation: the score is evaluated at a starting point $\theta _ { 0 }$ rather than at a maximum-likelihood estimate. Taking $\rho _ { j } \to 0$ in Equation (5),

$$
H _ { j } ( \theta , y ) = \frac { \lambda _ { j } ( \theta ) | a _ { j } | ^ { 2 } } { 2 \sigma ^ { 4 } } + O ( \rho _ { j } ^ { 2 } ) ,\tag{9}
$$

the variance-shrinkage bracket drops to second order, only the data-driven term survives, and the noise scale $\sigma ^ { 2 }$ factors out of the ranking. Define

$$
\tilde { H } _ { j } ( \theta , y ) : = \lambda _ { j } ( \theta ) \cdot \big | \phi _ { j } ( X ) ^ { \top } y \big | ^ { 2 } .\tag{10}
$$

The weak-observation criterion retains both the kernel and the data factor, while $\sigma ^ { 2 }$ drops out: ranking depends only on the basis $\phi _ { j }$ , the kernel eigenvalues at $\theta _ { 0 }$ , and the data y.

A single pass over the data The data factor $| a _ { j } | ^ { 2 }$ in $P _ { j }$ and $\tilde { H } _ { j }$ has a periodogram structure: it is a windowed periodogram of y against $\{ \phi _ { j } \} _ { j = 1 } ^ { J }$ , and the full vector $\Phi ^ { \top } { \boldsymbol y }$ is a identical to a $\mathcal { O } ( N J )$ transform of the data over the candidate basis. In practice the user should pick J as large as the candidate-precomputation budget allows: a larger J widens the frequency or degree coverage of the candidate basis before any are dropped. Since the basis $\phi _ { j }$ never depends on $\bar { \theta }$ in any of the three methods, this transform is computed once, before any hyperparameter fit, and yields $P _ { j }$ and ${ \tilde { H } } _ { j }$ at any starting point $\theta _ { 0 }$ . Its cost is small relative to a single hyperparameter-fit step (which itself involves an $\mathcal { O } ( N \bar { M } ^ { \bar { 2 } } )$ linear solve), so it is effectively amortised by the downstream fit.

Algorithm Algorithm 1 summarises the selection step for any of the three criteria as a single rank-and-truncate pass. It is important to note that the proposed criteria are heuristics. None of them is the canonical $\bar { H _ { j } }$ at the eventual fitted hyperparameters, and the greedy rank-and-truncate approach comes with no theoretical guarantee on downstream test performance. Whether they nonetheless do useful work is an empirical question, which the next three sections take up.

## 4 Experiment I: Hilbert-space Gaussian processes

We sweep six UCI regression benchmarks (concrete, energy, kin8nm, power, yacht, airfoil) over $M \in \{ 1 \dot { 6 } , 3 2 , 6 4 , 1 2 8 , 2 5 6 \}$ across 10 random 90:10 train/test splits with a Matérn-5/2 ARD kernel. Construction, candidate-basis size, fit objective, and the non-uniform truncation baseline are in Appendix C (and the shared dataset and optimisation protocol in Appendix G).

Findings Both data-aware criteria improve on the (no data) $I _ { j }$ on most cells, with $P _ { j }$ slightly stronger on average and ${ \tilde { H } } _ { j }$ sitting between $I _ { j }$ and $P _ { j }$ in many cells. The improvement is largest at small $M ,$ where allocating budget to the basis functions the data actually activates pays off most, and shrinks toward larger M as the budget catches up with the signal. The non-uniform truncation baseline tracks $I _ { j }$ closely on every cell and is slightly worse on average, an empirical sanity check that the per-basis-function eigenvalue ordering is at least as good as the structural product rule. The HSGP candidate grid is anisotropic by construction with active axes unknown before fitting, and the data projection $| a _ { j } | ^ { 2 }$ gives a cheap early signal that the data-aware criteria pick up directly while $I _ { j }$ has to wait for the hyperparameter fit to discover it.

![](images/5db31e3503ba22e9ffdc2f166cb8155fefa0c4f02770005a25463028e89026f4.jpg)  
Figure 2: Hilbert-space Gaussian processes (HSGP) on six UCI regression benchmarks. Median test NLL versus compute budget $M$ across 10 random 90:10 train/test splits, with interquartile bands. The data-aware (in-between) $\tilde { H } _ { j }$ and (no prior) $P _ { j }$ improve on the (no data) $I _ { j }$ on most cells, with $P _ { j }$ slightly stronger on average. The non-uniform per-axis truncation baseline tracks $I _ { j }$ within noise, slightly worse on average. Truncation x-axis at the nearest reachable budget under $\begin{array} { r } { M = \prod _ { d } M _ { d } } \end{array}$ (Appendix C).

## 5 Experiment II: Variational Fourier features

We use the same six UCI benchmarks and split protocol as for HSGP, with a per-axis Matérn-5/2 kernel, the candidate basis flattened across axes and ranked globally, and the closed-form collapsed Titsias bound fit by L-BFGS-B from the unit starting point. Construction, the periodic-extension diagonalisation, and the per-axis truncation baseline are in Appendix D.

Findings Figure 3 shows that $\tilde { H } _ { j }$ is roughly flat against $I _ { j }$ across most cells, while $P _ { j }$ is reliably worse than both. The per-axis truncation baseline sits on top of $I _ { j }$ at essentially every cell: the two rules pick almost identical basis function sets, because the per-axis eigenvalue order is already the lowest-frequency-first order that truncation enforces. On smooth UCI regressions the data projection $| a _ { j } | ^ { 2 }$ mostly adds noise to an ordering that the eigenvalues already get right.

## 6 Experiment III: Variational inducing spherical harmonics

We use the same six UCI benchmarks and split protocol as for HSGP, with the order-1 arc-cosine kernel [3] on the lifted inputs and the gpfy reference implementation [6] for the spherical-harmonic basis and its Funk–Hecke spectrum. Only the basis-selection rule changes between curves, and we compare against two practitioner-default truncation rules from the literature: the cumulative-shell rule of Dutordoir et al. [5] and the phase-truncation rule of Eleftheriadis et al. [6]. Construction, the closed-form arc-cosine zero pattern, and the two truncation baselines are in Appendix E.

Findings Against $I _ { j } ,$ the data-aware criteria do not help on this family. The (no data) $I _ { j }$ already captures the right structural prior: spherical-harmonic eigenvalues drop fast with degree, and on smooth UCI regressions the signal energy is concentrated at low degree, so reordering by data energy mostly perturbs an already-good ranking. The cumulative-shell truncation curve, however, behaves more dramatically: at cumulative-shell budgets that end on an even degree it selects bit-for-bit the same basis functions as $I _ { j } ,$ but whenever the cutoff $L ^ { \star }$ lands on an odd degree $\ell \geq 3$ the curve jumps upward. The cause is a closed-form property of the order-1 arc-cosine kernel: its Funk–Hecke expansion has $\mu _ { \ell } = 0$ for all odd $\ell \geq 3 \left[ 1 \right.$ , Appendix D.2], independent of the sphere dimension $D ,$ so filling an odd shell spends budget on coefficients the kernel forces to zero variance, and the (no data) ranking skips those shells automatically. Against the phase-truncation baseline, every score-based criterion is consistently better, with the gap reaching one to two orders of magnitude in test NLL at small M on energy and power: the single global $m ^ { \star }$ knob cannot adapt when the data favour more basis functions at one degree than another, and it inherits the same odd-shell trap. The phase-truncation sweep also takes about 10.5 seconds per fit versus 1.4 seconds for the scoreselected VISH sweep, with about 18× as many L-BFGS-B iterations, because the phase variables are optimised inside the fit whereas $I _ { j }$ fixes a subset before fitting.

![](images/b759d8750a14b9a28deb52d4e4ca209ac7402c0ed1f142f69a8d5011d0e93a7b.jpg)  
Figure 3: Variational Fourier features (VFF) on the same setup as Figure 2 (Matérn- ${ \cdot } 5 / 2$ per axis, closed-form collapsed Titsias bound). The data-aware (in-between) $\tilde { H } _ { j }$ and (no prior) $P _ { j }$ neither help nor hurt against the (no data) $I _ { j }$ at larger budgets, with $\tilde { H } _ { j }$ slightly behind on average. The conventional VFF per-axis truncation baseline sits on top of $I _ { j }$ at every cell. Truncation x-axis at the nearest reachable budget under $M = D ( 2 M _ { \star } - 1 )$ (Appendix D).

## 7 Related work

Active-set and information-theoretic selection in Gaussian processes Sparse-GP active-set methods have long used greedy and information-theoretic criteria to decide which scalar linear functionals of $f$ enter the approximation. The informative vector machine of Lawrence et al. [10], fast forward selection for sparse GP regression [21], sparse online GPs [4], and sparse greedy GP regression [22] all construct a small active set this way, and the same KL information gain $D _ { \mathrm { K L } } ( \bar { p } ( u _ { j } \mid y ) \parallel p ( u _ { j } ) )$ ) that we use for $H _ { j }$ underlies several of these scores [18, 9]. More recent work continues this line by placing Bayesian or task-specific structure on inducing locations and their number, for example through point-process priors [26], Bayesian inference over inducing locations [20], or quality-diversity allocation for Bayesian optimisation [14]. For continuous inducing-point locations $z \in \mathcal { X }$ the criterion is differentiable and is typically optimised jointly with the variational posterior [23], but all of these methods select, infer, or allocate data-indexed inducing inputs as a sub-routine of inference itself. Our setting is different. The basis families we consider already define a fixed candidate set of linear functionals (HSGP eigenfunctions, VFF Fourier features, VISH spherical harmonics), and the question is only how to spend a given budget M inside that set. Once this candidate basis is fixed, each criterion is a scalar function of the prior weight $\lambda _ { j }$ and the data projection $a _ { j } = \phi _ { j } ( X ) ^ { \top } y$ , and all candidates can be ranked in one pass before any posterior fit. The contribution is therefore not a new active-set sparse GP algorithm, but a basis-index selection rule that can replace the default truncation rule inside existing basis-function sparse GP constructions.

![](images/8eacc250a6246f7d60da570045e13bb62f21efa662e6d849e3cccd86ae0e5b7a.jpg)  
Figure 4: Variational inducing spherical harmonics (VISH) on the same setup as Figure 2 (arc-cosine order-1 kernel, spherical harmonics on $S ^ { D } )$ . Against the (no data) $I _ { j }$ , the data-aware (in-between) $\tilde { H } _ { j }$ does not improve on most cells. The cumulative-shell truncation curve jumps upward whenever the cutoff $L ^ { \star }$ lands on an odd degree (see body). Against the Eleftheriadis et al. [6] phase-truncation baseline, every score-based criterion is consistently better, often by orders of magnitude at small M. Both truncation x-axes at the nearest reachable budget under each rule’s structural constraint (Appendix E).

Matching pursuit and correlation screening The (no prior) score $P _ { j } ~ = ~ | a _ { j } | ^ { 2 }$ is the same correlation-screening step that initialises matching pursuit and its orthogonal variant [13, 15], in which dictionary atoms are picked by their alignment with the signal or current residual. Two differences matter. Matching pursuit is iterative: after an atom is selected the residual is updated, and in orthogonal matching pursuit the selected atoms are re-orthogonalised. Our selection step is a non-iterative rank-and-truncate pass that fixes the GP basis once, before any fit. The GP setting also supplies a prior variance $\lambda _ { j }$ for each candidate coefficient, so the (no data) $I _ { j } = \lambda _ { j }$ and (in-between) $\tilde { H } _ { j } = \lambda _ { j } | a _ { j } | ^ { 2 }$ are not pure correlation-screening rules. They combine the observed alignment with the kernel-implied plausibility of the candidate, which is what makes the same selection idea meaningful across HSGP, VFF, and VISH, whose basis indices have different geometries but all carry GP prior weights.

Spectral and eigenfunction feature methods A different line of work changes the feature construction itself. Sparse-spectrum GPs choose a small set of spectral frequencies and learn their locations as model parameters [11]. Variational orthogonal features construct stationary-kernel features for cheaper variational inference [2], and data-dependent eigenfunction methods select eigenfunctions of an empirical Gram matrix by evidence maximisation [16]. These ask which feature family or spectral approximation should define the approximation. Our setting is narrower but complementary: the feature family is fixed by a published sparse-GP construction, and we only choose the subset of its candidate indices.

## 8 Discussion

The work done here is essentially a preliminary investigation of a gap in the sparse-GP literature. The three methods we considered all recommend truncation as the default way to spend the basis budget. Yet for a fixed compute budget M (the realistic inference setting), there is freedom in choosing which basis functions to actually use, and the truncation default does not exercise it. Asking whether the candidate basis should be selected instead turns out to be cheap enough to test, and informative enough to separate from a broad benchmark of sparse-GP approximations. The investigation has taught us two things.

![](images/b953ed3248096af7ed88269850590fbb1e11cee594c6a3a36260aec59cd4f9ce.jpg)

![](images/72c5a6395c56b991d08127cc8dbd670f454aa38dc55e67561ec0b550ec333d38.jpg)  
HSGP signal band $K \in [ 2 , 4 ] \ u \bullet _ { } I _ { j }$ (no data) $\ P _ { j }$ (in-between)

![](images/283bc05add4f3939fdf3ad1ed876bb76f802c4df963c5d0844ff259ceb99379b.jpg)  
Figure 5: Toy bandpass example (details in Appendix F). (a) Signal in the integer-frequency band $K \in [ 2 , 4 ]$ plus noise. (b) Score spectra: the (no data) $I _ { j }$ is monotone low-pass and picks the lowest frequencies (circles), while the (in-between) $\ddot { H } _ { j }$ rises inside the band and picks there (squares). (c) HSGP fits at $M = 1 0 \colon I _ { j }$ over-smooths through the band, while ${ \tilde { H } } _ { j }$ tracks the wiggles.

The first is that the (no data) criterion $I _ { j }$ is a strong default: ranking every candidate freely by its kernel-prior weight is at least as good as the standard truncation rule for each method. On VISH it is indeed much better than either of the two practitioner-default truncation rules we compared against: an odd-shell zero pattern in the order-1 arc-cosine kernel kills entire structural blocks that the standard truncation rules waste budget on, while $I _ { j }$ avoids them automatically (Section 6). The same simple ranking also defeats the phase-truncation baseline of Eleftheriadis et al. [6], with substantial speedups in addition to the accuracy gain (Section 6).

The second is that the data-aware (no prior) and (in-between) criteria are method-specific. On HSGP the candidate grid (a high-dimensional version of the rectangle in Figure 1) has unknown active axes before fitting, so the data projection $| a _ { j } | ^ { 2 }$ is the cheapest probe of where the signal lives, and both $P _ { j }$ and $\tilde { H } _ { j }$ give an essentially free improvement over truncation. On VFF and VISH the basis is naturally ordered to favour low-frequency or low-degree functions, which is what smooth UCI regressions need, and $I _ { j }$ alone exploits this. The data-aware criteria are most likely to help when the data has structure that low-frequency truncation misses, as in the toy bandpass example of Figure 5.

Limitations and future work The eigenvalue-dependent criteria $I _ { j }$ and $\tilde { H } _ { j }$ evaluate $\lambda _ { j }$ at a single guessed starting point $\theta _ { 0 }$ rather than integrating over plausible hyperparameters, and the closed-form $H _ { j }$ is exact only under the empirical-decoupling assumption $\begin{array} { r } { \tilde { \Phi } ^ { \dagger } \Phi \tilde { \mathbf { \Phi } } \approx \mathrm { d i a g } ( \| \phi _ { j } ( X ) \| ^ { 2 } ) } \end{array}$ ). The three methods also differ in how their basis is constructed (HSGP native multi-dimensional, VFF additive across axes, VISH native multi-dimensional on the lifted sphere), so cross-method differences in selection behaviour can reflect this construction confound as well as the criteria. The empirics are preliminary, with six UCI regressions at five budgets per method and test RMSE corroborating the test-NLL pattern (Appendix H). Whether the patterns generalise to other kernels, basis families, or larger-scale regressions remains for future work.

Conclusion This paper addressed a gap in the basis-function sparse-GP literature: the practitioner has a fixed compute budget M to spend across the candidate basis, but the default answer is structural truncation. We asked whether a simple one-pass selection criterion can do better, and the answer turns out to be yes, with the size of the win depending on the method. The (no data) $I _ { j }$ ties or beats truncation everywhere and substantially improves over it on VISH, while the data-aware (no prior) $P _ { j }$ and (in-between) $\ddot { H } _ { j }$ criteria further improve on HSGP, which is the most adopted method in practice.

## References

[1] Francis Bach. Breaking the curse of dimensionality with convex neural networks. Journal of Machine Learning Research, 18(19):1–53, 2017. Appendix D.2 gives the closed-form Funk– Hecke multipliers for the order-1 arc-cosine (ReLU) kernel, with the parity property $\mu _ { \ell } = 0$ for odd $\ell \geq 3$

[2] David R. Burt, Carl Edward Rasmussen, and Mark van der Wilk. Convergence of Sparse Variational Inference in Gaussian Processes Regression. Journal ofMachine Learning Research, 21(131):1–63, 2020. ISSN 1533-7928.

[3] Youngmin Cho and Lawrence Saul. Kernel Methods for Deep Learning. In Advances in Neural Information Processing Systems, volume 22. Curran Associates, Inc., 2009.

[4] Lehel Csató and Manfred Opper. Sparse on-line Gaussian processes. Neural Computation, 14 (3):641–668, 2002.

[5] Vincent Dutordoir, Nicolas Durrande, and James Hensman. Sparse Gaussian Processes with Spherical Harmonic Features. In Proceedings of the 37th International Conference on Machine Learning, pages 2793–2802. PMLR, November 2020.

[6] Stefanos Eleftheriadis, Dominic Richards, and James Hensman. Sparse Gaussian Processes with Spherical Harmonic Features Revisited, March 2023.

[7] James Hensman, Nicolas Durrande, and Arno Solin. Variational fourier features for gaussian processes. The Journal ofMachine Learning Research, 18(1):5537–5588, 2017.

[8] Markelle Kelly, Rachel Longjohn, and Kolby Nottingham. The UCI machine learning repository. https://archive.ics.uci.edu, 2023.

[9] Andreas Krause, Ajit Singh, and Carlos Guestrin. Near-optimal sensor placements in Gaussian processes: Theory, efficient algorithms and empirical studies. In Journal of Machine Learning Research, volume 9, pages 235–284, 2008.

[10] Neil D Lawrence, Matthias Seeger, and Ralf Herbrich. Fast sparse Gaussian process methods: The informative vector machine. In Advances in Neural Information Processing Systems, volume 15, 2002.

[11] Miguel Lázaro-Gredilla, Joaquin Quiñonero-Candela, Carl Edward Rasmussen, and Aníbal R Figueiras-Vidal. Sparse Spectrum Gaussian Process Regression. Journal of Machine Learning Research, 11:1865–1881, 2010.

[12] Felix Leibfried, Vincent Dutordoir, S. T. John, and Nicolas Durrande. A Tutorial on Sparse Gaussian Processes and Variational Inference, December 2022.

[13] Stéphane G Mallat and Zhifeng Zhang. Matching pursuits with time-frequency dictionaries. IEEE Transactions on Signal Processing, 41(12):3397–3415, 1993.

[14] Henry B. Moss, Sebastian W. Ober, and Victor Picheny. Inducing point allocation for sparse Gaussian processes in high-throughput Bayesian optimisation. In International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pages 5213–5230. PMLR, 2023.

[15] Yagyensh C Pati, Ramin Rezaiifar, and Perinkulam S Krishnaprasad. Orthogonal matching pursuit: Recursive function approximation with applications to wavelet decomposition. In Asilomar Conference on Signals, Systems, and Computers, pages 40–44, 1993.

[16] Yuan Qi, Ahmed H Abdel-Gawad, and Thomas P Minka. Sparse-posterior Gaussian processes for general likelihoods. In Uncertainty in Artificial Intelligence, 2010.

[17] Carl Edward Rasmussen and Christopher K. I. Williams. Gaussian Processes for Machine Learning. Adaptive Computation and Machine Learning. MIT Press, Cambridge, Mass, 2006. ISBN 978-0-262-18253-9.

[18] Carl Edward Rasmussen and Christopher K. I. Williams. Gaussian Processes for Machine Learning. MIT Press, 2006. Section 8.3 introduces the per-coefficient KL information gain as a criterion for active inducing-set selection in sparse GPs.

[19] Gabriel Riutort-Mayol, Paul-Christian Bürkner, Michael R. Andersen, Arno Solin, and Aki Vehtari. Practical Hilbert space approximate Bayesian Gaussian processes for probabilistic programming. arXiv:2004.11408 [stat], April 2020.

[20] Simone Rossi, Markus Heinonen, Edwin Bonilla, Zheyang Shen, and Maurizio Filippone. Sparse Gaussian processes revisited: Bayesian approaches to inducing-variable approximations. In International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings ofMachine Learning Research, pages 1837–1845. PMLR, 2021.

[21] Matthias W. Seeger, Christopher K. I. Williams, and Neil D. Lawrence. Fast Forward Selection to Speed Up Sparse Gaussian Process Regression. In International Workshop on Artificial Intelligence and Statistics, pages 254–261. PMLR, January 2003.

[22] Alex J. Smola and Peter L. Bartlett. Sparse greedy Gaussian process regression. In Advances in Neural Information Processing Systems, volume 13, pages 619–625. MIT Press, 2001.

[23] Edward Snelson and Zoubin Ghahramani. Sparse Gaussian processes using pseudo-inputs. In Advances in Neural Information Processing Systems, volume 18, 2006.

[24] Arno Solin and Simo Särkkä. Hilbert space methods for reduced-rank Gaussian process regression. Statistics and Computing, 30(2):419–446, March 2020. ISSN 1573-1375. doi: 10.1007/s11222-019-09886-w.

[25] Michalis Titsias. Variational Learning of Inducing Variables in Sparse Gaussian Processes. In Proceedings of the Twelfth International Conference on Artificial Intelligence and Statistics, pages 567–574. PMLR, April 2009.

[26] Anders Kirk Uhrenholt, Valentin Charvet, and Bjørn Sand Jensen. Probabilistic selection of inducing points in sparse Gaussian processes. In Uncertainty in Artificial Intelligence, volume 161 of Proceedings of Machine Learning Research, pages 1035–1044. PMLR, 2021.

## A Sparse GPs and basis function expansions

The main text uses the language of finite basis expansions because that is where the selection problem is most transparent: once a candidate dictionary has been fixed, one has to decide which coefficients to keep. Sparse GPs in the sense of Leibfried et al. [12], however, are slightly broader objects than finite expansions. We spell out the connection here so that the basis-expansion view used throughout the paper is not mistaken for the most general sparse-GP construction.

A GP is already a basis function expansion. A Gaussian process $f \sim \mathcal { G P } ( 0 , k )$ on a domain X admits a Mercer expansion of its kernel,

$$
k ( x , x ^ { \prime } ) = \sum _ { j = 1 } ^ { \infty } \lambda _ { j } \phi _ { j } ( x ) \phi _ { j } ( x ^ { \prime } ) ,
$$

in an orthonormal basis $\{ \phi _ { j } \} _ { j = 1 } ^ { \infty }$ of $L ^ { 2 } ( \mathcal { X } , \mu )$ for a natural measure $\mu ,$ with non-negative eigenvalues $\lambda _ { 1 } \ge \lambda _ { 2 } \ge \cdots \ : \mathbf { B y }$ Karhunen–Loève, a draw $f \sim \mathcal { G P } ( 0 , k )$ is then

$$
f ( x ) = \sum _ { j = 1 } ^ { \infty } w _ { j } \phi _ { j } ( x ) , \qquad w _ { j } \overset { \mathrm { i i d } } { \sim } \mathcal { N } ( 0 , \lambda _ { j } ) .
$$

Thus a GP is already a basis function expansion. What is infinite-dimensional is the number of random coefficients, not the expansion form itself.

The Leibfried sparse GP, in general. The general definition of Leibfried et al. [12] does not require Mercer. A sparse GP starts from a GP $f \sim \mathcal { \bar { G P } } ( 0 , k )$ conditioned on a finite set of inducing variables $\mathbf { u } = ( u _ { 1 } , \dots , u _ { M } )$ , each a scalar linear functional of $f ,$ together with some chosen distribution $q ( \mathbf { u } ) = \mathcal { N } ( \mathbf { m } _ { u } , S _ { u u } )$ . Marginalising out gives the sparse-GP marginal

$$
f ( \cdot ) \ \sim \ \mathcal { G P } \Big ( \mu ( \cdot ) + k _ { \cdot u } K _ { u u } ^ { - 1 } ( \mathbf { m } _ { u } - \mu _ { u } ) , \ k ( \cdot , \cdot ^ { \prime } ) - k _ { \cdot u } K _ { u u } ^ { - 1 } \left( K _ { u u } - S _ { u u } \right) K _ { u u } ^ { - 1 } k _ { u ^ { \prime } } \Big ) .
$$

The covariance is the prior kernel minus a rank-M correction, not a rank-M kernel itself, and so the random function $f$ retains infinite-dimensional uncertainty around the conditional mean. A sparse GP per Leibfried is therefore not a finite basis function expansion in general.

Mercer connects the two. Take the inducing variables to be Mercer coefficients, $u _ { j } : = \langle f , \phi _ { j } \rangle$ for the first M Mercer eigenfunctions. By orthonormality and Mercer, $K _ { u u } = \mathrm { d i a g } ( \mathsf { \bar { \lambda } } _ { 1 } , \ldots , \lambda _ { M } ^ { - } )$ and $k _ { x u } [ j ] = \lambda _ { j } \phi _ { j } ( x )$ . Substituting into Leibfried’s covariance with $S _ { u u } = K _ { u u }$ , the sparse-GP covariance becomes the tail of the Mercer expansion,

$$
\mathrm { C o v } _ { \mathrm { s p a r s e } } ( f ( x ) , f ( x ^ { \prime } ) ) = \sum _ { j > M } \lambda _ { j } \phi _ { j } ( x ) \phi _ { j } ( x ^ { \prime } ) ,
$$

and the GP itself decomposes as

$$
f ( x ) = \underbrace { \sum _ { j = 1 } ^ { M } u _ { j } \phi _ { j } ( x ) } _ { \mathrm { r a n k - } M \mathrm { b a s i s } \mathrm { e x p a n s i o n } } + \underbrace { h ( x ) } _ { \mathrm { M e r c e r t a i l } } ,
$$

with $u _ { j } \sim { \mathcal { N } } ( 0 , \lambda _ { j } )$ and h a zero-mean GP whose covariance is the tail of the Mercer expansion. The finite basis function expansion view drops the residual h. The full sparse-GP view keeps it. Both are valid sparse-GP constructions, and the difference is the orthogonal complement of span $\{ \phi _ { j } \} _ { j = 1 } ^ { M } .$

What each family in this paper does. HSGP [24, 19] drops the residual: the model is an explicit basis function expansion, with Dirichlet Laplacian eigenfunctions $\phi _ { j }$ and prior variances $\lambda _ { j } ( \theta )$ from the kernel’s spectral density, and inference is closed-form Bayesian linear regression on the truncated basis. VFF [7] keeps the residual: the inducing variables are RKHS Fourier projections, and Hensman et al. comment explicitly that the variational form does not discard the orthogonal complement. VISH [5, 6] keeps the residual on the same logic with spherical-harmonic projections. Appendices C–E give the construction of each family.

Why the residual does not affect our criteria. The criteria $I _ { j } ( \theta ) , P _ { j } ( y ) , \tilde { H } _ { j } ( \theta , y )$ are derived from the per-basis-function KL information gain $H _ { j } ( \theta , y ) = D _ { \mathrm { K L } } \big ( p ( u _ { j } \mid y ) \| p ( u _ { j } ) \big )$ . The prior $p ( u _ { j } ) = \mathcal { N } ( 0 , \lambda _ { j } ( \theta ) )$ depends only on the kernel eigenvalue at $\phi _ { j }$ , not on whether the residual is kept or dropped. The marginal posterior $p ( u _ { j } \mid \boldsymbol { y } )$ depends on the data through the projection $a _ { j } = \phi _ { j } ( X ) ^ { \top } y$ and on the noise scale $\sigma ^ { 2 }$ , and under the empirical-decoupling assumption $\Phi ^ { \top } \Phi \approx$ $\bar { \mathrm { d i a g } } ( c _ { j } )$ takes the same value in both pictures: the residual h contributes only to the unmodelled uncertainty in $f ( x )$ at unseen test inputs, not to the marginal posterior on the j-th Mercer coefficient. The criteria, and the selection problem they pose, are therefore identical in both pictures.

Conventions used in the main text. Given this equivalence at the criterion level, the main text uses the basis-expansion picture as its default exposition. It poses the selection-versus-truncation question cleanly (which Mercer coefficients to keep), unifies the three families at the model-and-prior level (each picks $\{ \phi _ { j } \}$ and $\lambda _ { j } ( \theta ) _ { , } ^ { \dag }$ , and does not commit to any inference machinery. The full sparse-GP construction is invoked per family only when reporting the fitting procedure each family uses downstream of selection: the closed-form marginal likelihood for HSGP, and the collapsed Titsias bound for VFF and VISH.

## B Derivation of the criteria

We derive the closed form Equation (5), the data-averaged form $\begin{array} { r } { \mathbb { E } _ { y } [ H _ { j } ] = \frac { 1 } { 2 } \log ( 1 + \rho _ { j } ) } \end{array}$ , and the rank-equivalence of the fully data-fit $H _ { j }$ with the no-prior criterion ${ \bf { \bar { \mathit { P } } } } _ { j }$ . The same argument extends to arbitrary scalar linear functionals of f (inducing points, Mercer projections, windowed Fourier projections). The basis-function specialisation used in the main text follows.

Closed form for $H _ { j } .$ . Under the empirical-decoupling assumption $\Phi ^ { \top } \Phi$ ≈ $\mathrm { d i a g } ( c _ { j } )$ with $c _ { j } : =$ $\| \phi _ { j } ( X ) \| ^ { 2 }$ , the marginal posterior on the basis-function coefficient $u _ { j }$ is Gaussian with variance $v _ { j } ^ { \mathrm { p o s t } } = v _ { j } / ( 1 + \rho _ { j } )$ and mean $\mu _ { j } ( y ) = \lambda _ { j } a _ { j } / ( \sigma ^ { 2 } + \lambda _ { j } c _ { j } )$ , where $v _ { j } = \lambda _ { j } ( \theta )$ is the prior variance and $\bar { \rho _ { j } } = \lambda _ { j } c _ { j } / \sigma ^ { 2 }$ is the per-basis-function signal-to-noise ratio. Plugging into the univariate Gaussian KL gives Equation (5).

Data-averaged form: derivation of $I _ { j }$ . Under the prior, $a _ { j } = \phi _ { j } ( X ) ^ { \top } y$ has variance $\mathrm { V a r } ( a _ { j } ) =$ $\sigma ^ { 2 } c _ { j } ( 1 + \rho _ { j } ) , \operatorname { s o } \mathbb { E } _ { y } [ | a _ { j } | ^ { 2 } ] = \sigma ^ { 2 } c _ { j } ( 1 + \rho _ { j } )$ . Substituting into the data-driven term in Equation (5),

$$
\mathbb { E } _ { y } \bigg [ \frac { \lambda _ { j } | a _ { j } | ^ { 2 } } { 2 \sigma ^ { 4 } ( 1 + \rho _ { j } ) ^ { 2 } } \bigg ] = \frac { \lambda _ { j } c _ { j } } { 2 \sigma ^ { 2 } ( 1 + \rho _ { j } ) } = \frac { \rho _ { j } } { 2 ( 1 + \rho _ { j } ) } ,
$$

and adding the variance-shrinkage bracket $\frac { 1 } { 2 } [ \log ( 1 + \rho _ { j } ) - \rho _ { j } / ( 1 + \rho _ { j } ) ]$ collapses the data-driven and shrinkage pieces:

$$
\begin{array} { r } { \mathbb { E } _ { y } [ H _ { j } ( \theta , y ) ] \ = \ \frac { 1 } { 2 } \log ( 1 + \rho _ { j } ) , } \end{array}
$$

which is monotone in $\rho _ { j }$ at fixed $\sigma ^ { 2 } , c _ { j }$ and therefore equivalent in rank to $\lambda _ { j } ( \theta )$ . This is the criterion $I _ { j }$ of Equation $( 7 )$

Rank-equivalence of the fully data-fit $H _ { j }$ with $P _ { j }$ . If $\sigma ^ { 2 }$ and $\lambda _ { j }$ are jointly fit by maximum likelihood at fixed $j ,$ the optimum gives $\hat { \rho } _ { j } = \operatorname* { m a x } ( 0 , | a _ { j } | ^ { 2 } / ( \sigma ^ { 2 } c _ { j } ) ^ { - } 1 )$ ). Substituting back into Equation (5) reduces $H _ { j }$ to a monotone function of $| a _ { j } | ^ { 2 }$ alone, which is rank-equivalent to $P _ { j } =$ $| \bar { a _ { j } } | ^ { 2 }$

## C Hilbert-space Gaussian processes (HSGP)

For HSGP [24, 19], the candidate dictionary is fixed by the geometry of a box and a Dirichlet boundary condition, while the kernel enters through spectral weights on that dictionary. The details below give the basis $\{ \phi _ { j } \}$ , the prior variances $\bar { \lambda _ { j } } ( \theta )$ , the downstream fit after selection, and the per-axis truncation baseline reported in Section 4.

The basis on a box. Take the input domain to be $\Omega = [ - L , L ] ^ { D }$ . The Dirichlet eigenproblem for the Laplacian on Ω,

$$
- \nabla ^ { 2 } \phi _ { j } ( x ) = \mu _ { j } \phi _ { j } ( x ) , \qquad x \in \Omega , \qquad \left. \phi _ { j } \right| _ { \partial \Omega } = 0 ,
$$

has eigenfunctions

$$
\phi _ { j } ( x ) = \prod _ { d = 1 } ^ { D } L ^ { - 1 / 2 } \sin \left( \frac { \pi j _ { d } ( x _ { d } + L ) } { 2 L } \right) , \qquad j = ( j _ { 1 } , \dots , j _ { D } ) \in \mathbb { N } _ { + } ^ { D } ,
$$

with eigenvalues

$$
\mu _ { j } = \sum _ { d = 1 } ^ { D } \biggl ( \frac { \pi j _ { d } } { 2 L } \biggr ) ^ { 2 } .
$$

The $\{ \phi _ { j } \}$ are orthonormal in $L ^ { 2 } ( \Omega )$ and depend only on $\Omega$ and the boundary condition. In particular they do not depend on the kernel hyperparameters θ.

Approximate Mercer interpretation. For a stationary kernel $k _ { \theta }$ on $\mathbb { R } ^ { D }$ with spectral density $S _ { \theta }$ the HSGP construction approximates the kernel by

$$
k _ { \theta } ( x , x ^ { \prime } ) \approx \sum _ { j } \lambda _ { j } ( \theta ) \phi _ { j } ( x ) \phi _ { j } ( x ^ { \prime } ) , \qquad \lambda _ { j } ( \theta ) = S _ { \theta } ( \omega _ { j } ) ,
$$

where $\omega _ { j } = ( \pi j _ { 1 } / 2 L _ { 1 } , \ldots , \pi j _ { D } / 2 L _ { D } )$ is the per-axis Laplacian frequency vector (the box $\Omega =$ $\Pi _ { d } [ - L _ { d } , L _ { d } ]$ is allowed to be anisotropic, and the kernel can be ARD). The approximation comes from restricting to a compact Ω, imposing Dirichlet boundary conditions, and truncating to finitely many modes. It is asymptotically exact as the per-axis box lengths $L _ { d } $ ∞ for inputs away from ∂Ω and with sufficient mode count. Throughout this paper we use the box $L _ { d } = 1 . 2 \cdot \mathrm { m a x } _ { i } \left| x _ { i d } \right| \mathrm { p e r }$ axis, following the convention of Riutort-Mayol et al. [19].

Inference under Gaussian likelihood. Once a subset M of size M is selected, the model is the finite linear-Gaussian system $y = \Phi u + \varepsilon$ with $\Phi _ { n j } = \phi _ { j } ( x _ { n } )$ , basis-function coefficients $u = ( u _ { j } ) _ { j \in { \mathcal { M } } } \sim { \mathcal { N } } ( 0 , \Lambda _ { \theta } ) , \Lambda _ { \theta } = \mathrm { d i a g } ( \lambda _ { j } ( \theta ) ) _ { j \in { \mathcal { M } } \mid }$ , and $\varepsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { N } )$ . The marginal likelihood,

$$
\begin{array} { r } { \log p ( \boldsymbol { y } \mid \boldsymbol { \theta } , \boldsymbol { \mathcal { M } } ) = - \frac { 1 } { 2 } \left[ \log \left| \boldsymbol { \Sigma } _ { \boldsymbol { y } } \right| + \boldsymbol { y } ^ { \top } \boldsymbol { \Sigma } _ { \boldsymbol { y } } ^ { - 1 } \boldsymbol { y } + N \log 2 \pi \right] , \qquad \boldsymbol { \Sigma } _ { \boldsymbol { y } } = \Phi \boldsymbol { \Lambda } _ { \boldsymbol { \theta } } \Phi ^ { \top } + \sigma ^ { 2 } \boldsymbol { I } _ { N } , } \end{array}
$$

is closed-form, so the downstream fit is L-BFGS-B maximisation of this objective rather than variational inference.

Truncation baseline. The conventional HSGP library default fixes per-axis counts $( M _ { 1 } , \ldots , M _ { D } )$ and keeps every basis function with $j _ { d } \le M _ { d }$ on every axis, giving a total of $\begin{array} { r } { M = \prod _ { d } M _ { d } } \end{array}$ basis functions. For our budget range $M \in \left[ 1 6 , 2 5 6 \right]$ this rule is coarse-grained: a single uniform ${ \mathrm { \hat { M } } } _ { d } = M ,$ on every axis only lands on $\bar { M } = \bar { M } _ { \star } ^ { D }$ , which covers at most one or two budgets per dataset. We therefore use a non-uniform extension to obtain a fair comparison curve on the same $M { \mathrm { - g r i d } }$ as the selection criteria. For each target budget we pick the most-uniform integer factorisation $\Pi _ { d } M _ { d }$ breaking ties by smallest standard deviation across axes, and assign the descending-sorted tuple to axes ordered by descending training-input variance. This defines a reproducible non-uniform truncation baseline on the same $M { \cdot } \mathrm { g r i d }$ as the selection criteria, used in Figure 2.

## D Variational Fourier features (VFF)

For VFF [7], the candidate basis is Fourier rather than Laplacian, and the multi-dimensional candidate basis used here is assembled from separate one-dimensional Fourier blocks rather than from their tensor product. We describe the one-dimensional features, the diagonalisation approximation used for scoring, the Gaussian-likelihood objective, and the per-axis truncation baseline used in Section 5.

Fourier features on an interval. Take the input domain to be $[ a , b ]$ with length $T = b - a$ and use the normalised measure $d \mu ( x ) = d x / T$ . Define harmonic frequencies $\omega _ { m } = 2 \pi m / T$ for $m = 1 , 2 , . . .$ . and the orthonormal Fourier features

$$
\phi _ { 0 } ( x ) = 1 , \qquad \phi _ { c , m } ( x ) = \sqrt { 2 } \cos \bigl ( \omega _ { m } ( x - a ) \bigr ) , \qquad \phi _ { s , m } ( x ) = \sqrt { 2 } \sin \bigl ( \omega _ { m } ( x - a ) \bigr ) .
$$

These are orthonormal in $L ^ { 2 } ( [ a , b ] , \mu )$ and depend only on the interval and the frequency grid. For the multi-dimensional case we use $\begin{array} { r } { f ( x ) = \sum _ { d = 1 } ^ { D } f _ { d } ( x _ { d } ) } \end{array}$ with a separate 1D VFF basis per axis. The total candidate count is $\textstyle \sum _ { d } M _ { d }$ , not the tensor-product $\Pi _ { d } M _ { d }$ , so the construction scales linearly in $D$ at fixed per-axis count.

Periodic-extension diagonalisation. The Fourier features are not exact Mercer eigenfunctions of a stationary kernel on a finite interval: boundary effects prevent perfect diagonalisation. Hensman et al. [7] show that for Matérn kernels the exact $K _ { u u }$ is diagonal-plus-low-rank, with the diagonal proportional to $1 / S _ { \theta } ( \omega _ { m } )$ and rank-one corrections capturing the boundary effects. We adopt the standard periodic-extension approximation, in which the kernel is extended periodically with period T and diagonalises in the Fourier basis with spectral weights

$$
\lambda _ { m } ( \theta ) \approx \frac { S _ { \theta } ( \omega _ { m } ) } { T } .
$$

The approximation is asymptotically exact as $T$ grows large relative to the kernel’s correlation length, and inherits boundary error at finite $\rvert _ { T }$ . For our criteria the diagonal piece suffices because the criteria depend only on the per-basis-function eigenvalue $\lambda _ { m }$ and the data projection.

Inference under Gaussian likelihood. A Gaussian variational posterior $q ( \mathbf { u } ) = \mathcal { N } ( \mathbf { m } , S )$ on the inducing variables $\mathbf { u } = ( u _ { j } ) _ { j \in \mathcal { M } }$ , the per-feature coefficients with prior variance $\lambda _ { j } ( \theta )$ , is fit by maximising the standard sparse-GP ELBO. For Gaussian likelihood the optimum admits a closed form (the collapsed Titsias bound), and the bound itself becomes the marginal-likelihood objective that we maximise over θ by L-BFGS-B [25, 7]. The closed-form gradients and predictive expressions follow Hensman et al. directly.

Truncation baseline. The conventional VFF library default fixes a per-axis frequency count $M _ { \star }$ and keeps frequencies $j _ { d } \in \{ 0 , 1 , \ldots , M _ { \star } - 1 \}$ on every axis. Each non-DC frequency contributes both a cosine and a sine basis function while $j _ { d } = 0$ contributes only a cosine, so the total count is $M = D ( 2 M _ { \star } - 1 )$ . For each dataset we sweep M<sub>⋆</sub> such that $M \in [ 1 6 , 2 5 6 ]$ ] and thin to five log-spaced points (e.g. on combined cycle power plant with $D = 4$ this lands at $\bar { M } \in \{ 2 0 , 3 6 , 6 0 , 1 \bar { 2 } 4 , 2 5 2 \}$ and on kin8nm with $D = \mathring { 8 } \mathrm { ~ a t ~ } \hat { M } \in \{ 2 \mathring { 4 } , 4 0 , 5 6 , 1 2 0 , 2 4 8 \} \}$ ). This is the truncation curve reported in Figure 3.

## E Variational inducing spherical harmonics (VISH)

For VISH [5, 6], the candidate dictionary is organised by spherical-harmonic degree after the Euclidean inputs have been lifted to the sphere. The details below give that construction, the order-1

arc-cosine kernel [3] used throughout, the closed-form zero pattern responsible for the cumulativeshell jump in Figure 4, and the two truncation baselines used in Section 6.

Input lifting and spherical harmonics. Given Euclidean inputs $x _ { n } \in \mathbb { R } ^ { p }$ , define

$$
z _ { n } = \frac { ( x _ { n } , 1 ) } { \| ( x _ { n } , 1 ) \| } \ \in \ S ^ { D - 1 } , \qquad D = p + 1 .
$$

All harmonic features are evaluated at $z _ { n }$ , not at the raw Euclidean input. Spherical harmonics $\phi _ { \ell , k } : S ^ { D - 1 } \to \mathbb { R }$ are indexed by degree $\ell = 0 , 1 , 2 , \ldots$ . and orientation $\mathbf { \bar { \xi } } k = 1 , \mathbf { \bar { \xi } } , \dots , N ( D , \ell )$ , with multiplicity

$$
N ( D , \ell ) = \frac { ( 2 \ell + D - 2 ) ( \ell + D - 3 ) ! } { \ell ! ( D - 2 ) ! } , \qquad D \ge 3 ,
$$

and are orthonormal under the normalised surface measure $d \mu ( z ) = d \omega ( z ) / \Omega _ { D - 1 }$

Funk–Hecke and degree-only eigenvalues. A kernel $k _ { \theta } \mathrm { o n } S ^ { D - 1 }$ is zonal if $k _ { \theta } ( z , z ^ { \prime } ) = \kappa _ { \theta } ( z ^ { \top } z ^ { \prime } )$ i.e. depends only on the angle between inputs. For a zonal kernel, the Funk–Hecke theorem says that spherical harmonics are eigenfunctions of the kernel integral operator with eigenvalues that depend only on the degree:

$$
\lambda _ { \ell , k } ( \theta ) = \lambda _ { \ell } ( \theta ) \quad { \mathrm { f o r ~ a l l ~ } } k = 1 , \ldots , N ( D , \ell ) .
$$

This gives a shell decomposition indexed by degree ℓ, where all $N ( D , \ell )$ orientations within a shel share the same prior variance.

The order-1 arc-cosine kernel. We use the order-1 arc-cosine kernel [3] throughout, with $\lambda _ { \ell } ( \theta )$ obtained from the gpfy reference implementation [6]. A closed-form property of this kernel is that its Funk–Hecke expansion has $\lambda _ { \ell } \overset { \_ } { = } 0$ for all odd $\ell \geq 3 .$ , independent of the sphere dimension D [1, Appendix D.2]. Only the shells $\ell \in \{ 0 , 1 \} \cup \{ 2 , 4 , 6 , . . . \}$ carry positive prior variance. In the dimensions used by the lifted UCI datasets $\dot { ( } D \in \{ 5 , 6 , 7 , 9 , 1 2 \} \dot { ) }$ , the same pattern is visible numerically: the magnitudes of the nonzero shells decay smoothly with $\ell ,$ while the odd shells $\ell \geq 3$ sit at machine precision. The pattern is a property of the kernel function, not the sphere geometry, and it has a sharp practical consequence for the cumulative-shell truncation baseline below.

Inference under Gaussian likelihood. Inducing variables are RKHS projections, $\begin{array} { r l } { u _ { \ell , k } } & { { } = } \end{array}$ $\langle f , \phi _ { \ell , k } \rangle _ { \mathcal { H } }$ . The prior $K _ { u u }$ is diagonal in the spherical-harmonic basis with entries $1 / \lambda _ { \ell }$ in the RKHS coordinate. The Mercer coordinate $c _ { \ell , k } = \lambda _ { \ell } u _ { \ell , k }$ has prior variance $\lambda _ { \ell } ,$ which is what the criteria use. A Gaussian variational posterior $q ( \mathbf { u } ) = \mathcal { N } ( \mathbf { m } , \bar { \boldsymbol { S } } )$ is fit by maximising the standard sparse-GP ELBO, which for Gaussian likelihood admits a closed form. We maximise the resulting collapsed bound over θ by L-BFGS-B [5, 25]. The diagonal $K _ { u u }$ structure is preserved under arbitrary subset selection.

Two truncation baselines. The VISH comparisons use the two practitioner-default truncation rules from the literature.

The cumulative-shell rule of Dutordoir et al. [5] fills harmonic shells in degree order: pick a cutoff degree $L ^ { \star }$ and keep every harmonic at $\ell \leq L ^ { \star }$ , for a total of $\begin{array} { r } { M = \sum _ { \ell = 0 } ^ { L ^ { \star } } N ( D , \ell ) } \end{array}$ basis functions. This is the direct VISH analogue of the per-axis cube rule used for HSGP. On the order-1 arc-cosine kernel, however, this rule can select zero-variance shells: whenever $L ^ { \star }$ is an odd degree $\ell \geq 3$ , every harmonic at that degree has zero prior variance and contributes no information to the fit. This is the cause of the upward jump in the black truncation curve in Figure 4.

The phase-truncation rule of Eleftheriadis et al. [6] keeps min $( m ^ { \star } , N ( D , \ell ) )$ orthogonalised phase features per shell up to a larger cutoff degree, where the phases are parameterised by learnable phase vectors and computed via the addition theorem as Gegenbauer evaluations $C _ { \ell } ^ { ( \alpha ) } ( z ^ { \top } v _ { \ell , m } )$ with $\dot { \alpha } = ( D - 2 ) / 2$ . This rule has a single integer knob $m ^ { \star }$ on top of the cutoff, and we choose $m ^ { \star }$ so that the resulting basis-function count is closest to the target M. It is data-blind (the phases are optimised at fitting time, not at selection time) and inherits the same odd-shell trap as cumulative-shell truncation.

## F Toy figure details

The two illustrative figures in the main text serve a narrower purpose than the UCI sweeps: they make visible what a ranking rule is doing before any benchmark comparison is considered. We give the data-generation and fitting details for Figure 1 (the four-panel grid in Section 1) and Figure 5 (the one-dimensional bandpass example in Section 3) so that the examples can be read as concrete constructions rather than as additional empirical claims.

Figure 1 (motivational grid, two-dimensional HSGP). We use the AT and V features (axes 0 and 1) of the UCI combined cycle power plant regression [8] to build a $D = 2$ HSGP problem with $M _ { \mathrm { p e r - d i m } } = 5 .$ , giving a $5 \times 5 = 2 5$ candidate basis on a box $[ - L _ { d } , L _ { d } ] ^ { 2 }$ with $L _ { d } = 1 . 2 \cdot \operatorname* { m a x } _ { i } \left| x _ { i d } \right|$ per axis. The kernel is Matérn-5/2 with ARD lengthscales. For each of the four selection strategies (rectangular truncation $M _ { d } = ( 5 , 2 )$ , the (no data) criterion $I _ { j }$ , the (no prior) criterion $P _ { j }$ , the (in-between) criterion $\tilde { H } _ { j } )$ , we select the top- $M = 1 0$ candidates at the unit starting point $\theta _ { 0 }$ and fit the kernel hyperparameters by ${ \bf L - B F G } { \bar { \bf S } } { \bf - B }$ on the closed-form marginal likelihood from the same starting point, reporting test NLL on the held-out split (seed 0). The corresponding script is experiments/exp1\_hsgp\_lbfgs/motivational\_figure.py.

Figure 5 (one-dimensional bandpass example). We sample N noisy observations of a synthetic bandpass-sinc signal on $t \in [ 0 , T ]$ at sampling frequency $\bar { f } _ { s } = 5 1 2 \ : ( T = 1 , N = 5 1 2$ regularlyspaced points), with bandpass cutoffs $\omega _ { 0 } = 3 0$ and $\omega _ { 1 } =$ 100 and additive Gaussian noise of standard deviation $\sigma _ { \mathrm { n o i s e } } = 0 . 3$ . The HSGP basis uses $M _ { \mathrm { c a n d } } = 5 1 2$ Dirichlet sine eigenfunctions on a box of length $1 . 2 \cdot T$ . The score-spectrum panels show $I _ { j }$ and ${ \tilde { H _ { j } } }$ at $M = 1 0$ alongside the corresponding HSGP fits. The corresponding script is experiments/exp1\_hsgp\_lbfgs/bandpass\_figure.py.

## G Experimental details for UCI

Datasets and splits. Six UCI regression benchmarks [8]: airfoil self-noise $( N = 1 5 0 3 , D = 5 )$ , concrete compressive strength $( N = 1 0 3 0 , D = 8 )$ , energy efficiency $( N = 7 6 8 , D = 8 )$ , kin8nm $( N = 8 1 9 2 , \bar { D } = 8 )$ , combined cycle power plant $( N = 9 5 6 8 , D = 4 )$ , and yacht hydrodynamics $( N = 3 0 8 , D = 6 )$ . Each cell uses 10 random 90:10 train/test splits with seeds $0 , \ldots , 9$ . Inputs and targets are standardised to zero mean and unit variance using training-set statistics.

Optimisation protocol (HSGP, VFF, VISH). For HSGP, VFF, and VISH, each cell is fit with a single L-BFGS-B run over $( \log \ell , \log \sigma _ { \mathrm { s i g } } ^ { 2 } , \log \sigma _ { \mathrm { n o i s e } } ^ { 2 } )$ from the unit starting point $\theta _ { 0 } = ( \log \ell _ { d } =$ 0, log $\sigma _ { \mathrm { s i g } } ^ { 2 } = 0 , \log \sigma _ { \mathrm { n o i s e } } ^ { 2 } = \log 0 . 1 )$ . The same starting point is used for selection and for the subsequent kernel-hyperparameter fit. We do not multi-start. Variability across the 10 splits is reported in every per-family figure as the median test NLL with shaded interquartile bands, robust to occasional optimisation outliers.

HSGP setup. For HSGP, we use a Matérn-5/2 ARD kernel on a box $[ - L _ { d } , L _ { d } ] ^ { D }$ per axis with $L _ { d } = 1 . 2 \cdot \operatorname* { m a x } _ { i } \left| x _ { i d } \right|$ . The candidate dictionary is $\{ 1 , \ldots , M _ { \mathrm { p e r - d i m } } \} ^ { D }$ , with $M _ { \mathrm { p e r - d i m } }$ chosen so that the total candidate count satisfies $J \le 8 0 0 0$ . The fit objective is the closed-form marginal likelihood of the truncated linear-Gaussian model described in Appendix C. The non-uniform truncation baseline is constructed as documented in Appendix C.

VFF setup. For VFF, we use the per-axis Matérn- $5 / 2 K _ { u u }$ blocks of Hensman et al. [7]. The candidate dictionary is $\{ ( d , j _ { d } ) : 1 \stackrel { - } { \leq } d \leq D , 0 \leq j _ { d } < M _ { \mathrm { p e r - d i m } } \}$ . It is treated as a flat set and ranked globally. The fit objective is the closed-form collapsed Titsias bound described in Appendix D, and the truncation baseline is the one documented there.

VISH setup. For VISH, inputs are projected to $S ^ { D }$ via $z _ { n } = ( x _ { n } , 1 ) / \| ( x _ { n } , 1 ) \| \ [ 5 ]$ . Spherical harmonics and the order-1 arc-cosine Funk–Hecke spectrum are taken from the gpfy library (Apache-2.0) [6, 3], with $\ell _ { \mathrm { m a x } }$ chosen per D to fit $\mathrm { g p f y ^ { \prime } s }$ pre-computed fundamental-set tables. The fit objective is the collapsed Titsias bound on the gpfy spherical-harmonic basis described in Appendix E. Cumulative-shell truncation cuts at the largest degree $L ^ { \star }$ satisfying $\begin{array} { r } { \sum _ { \ell < L ^ { \star } } N ( D , \ell ) \le \dot { \cal M } } \end{array}$ . The Eleftheriadis phase-truncation baseline uses $\mathrm { g p f y ^ { \prime } s }$ phase\_truncation parameter, choosing the per-shell phase count $m ^ { \star }$ so the total basis-function count is closest to the target M.

Compute resources. The paper experiments, meaning the runs that produced the JSON files used for the reported figures, were run on a single workstation with an AMD Ryzen 7 3700X CPU (8 cores, 16 hardware threads), 64 GB RAM, and an NVIDIA RTX 2080 SUPER GPU (8 GB). Those JSON files report wall-clock time per dataset via elapsed\_seconds. Summing the seven UCI sweep modules used in the paper gives approximately 3.9 hours for the six reported datasets, with individual L-BFGS-B fits typically completing in seconds. Cached datasets, JSON outputs, and generated figures require less than 20 GB of disk. The anonymised reproduction archive is set to the JAX CPU backend by default for broader compatibility. This changes the out-of-the-box execution environment, not the algorithms themselves, since the implementations are JAX-based and can use GPU acceleration when the corresponding JAX installation and environment are configured.

## H Test RMSE on UCI for Experiments I, II, III

Figures H.1, H.2, and H.3 repeat the UCI comparison with median test root mean square error (RMSE) versus budget M across the same 10 train/test splits, with interquartile bands. The pattern across the three families is the same as the test-NLL pattern in the main text. On HSGP the data-aware criteria $\tilde { H } _ { j }$ and $P _ { j }$ improve on $I _ { j }$ on most cells. On VFF the three criteria are roughly tied and the per-axis truncation baseline tracks $I _ { j }$ closely. On VISH the (no data) $I _ { j }$ matches or exceeds the data-aware criteria, and the cumulative-shell truncation curve shows the same odd-shell jumps the test-NLL curve does. We use NLL as the headline metric in the body for two reasons: it captures the predictive distribution rather than only the point prediction, and it is the standard test metric for sparse GP regression in this literature.

![](images/bfdeeb8ed1a4e0aa7fe62c72266aa2fe1033ea0787d254aeefde573bc996f6a1.jpg)

Figure H.1: Experiment I: HSGP test RMSE on the same six UCI benchmarks and budgets as Figure 2.  
![](images/3344f569f8901c16b4c654c9d47d0fc1d9ede329d23599d3f067b8a930f67a08.jpg)  
Figure H.2: Experiment II: VFF test RMSE on the same six UCI benchmarks and budgets as Figure 3.

![](images/32b578bf6d48dccea99ac532a37758add3f27e38589085296eac6b1a95778d54.jpg)  
VISH truncation I<sub>j</sub> (no data) P<sub>j</sub> (no prior) H<sup>˜</sup><sub>j</sub> (in-between) Eleft.  
Figure H.3: Experiment III: VISH test RMSE on the same six UCI benchmarks and budgets as Figure 4.