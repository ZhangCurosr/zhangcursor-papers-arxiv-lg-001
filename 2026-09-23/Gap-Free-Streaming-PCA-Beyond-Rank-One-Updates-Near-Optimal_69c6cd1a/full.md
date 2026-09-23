# Gap-Free Streaming PCA Beyond Rank-One Updates: Near-Optimal Rates and Applications to Diferential Privacy

Anming Gu<sup>∗</sup> Syamantak Kumar<sup>†</sup> Kevin Tian<sup>‡</sup> Chutong Yang<sup>§</sup>

## Abstract

Streaming principal component analysis (PCA) seeks to recover a leading spectral subspace in a single pass over a data stream. We give a new analysis of the ubiquitous Oja’s algorithm [Oja82] for the most general, gap-free variant of this problem, where no eigengap assumptions are made on the underlying mean matrix, complemented by a nearly-matching lower bound. Prior works achieving near-optimal rates for streaming PCA either required gap assumptions [JJK<sup>+</sup>16, HNWW21], or were limited to rank-one updates [AZL17, Lia23]. Our proof only uses a second moment bound on the individual stochastic updates, bypassing the almost sure bounds needed by prior near-optimal analyses, and the analogous ofline matrix Bernstein bound. We also extend our result to a Rayleigh quotient notion of approximate PCA, addressing an open question of [JJK<sup>+</sup>16]. As our main application, we give gap-free diferentially private PCA guarantees for sub-Gaussian data, settling Conjecture 1.1 of [Bro26] up to logarithmic factors.

## Contents

1 Introduction 1   
1.1 Our results 1   
1.2 Our techniques 3   
1.3 Related work 5   
2 Preliminaries 6   
2.1 Notation . 6   
2.2 Main problem . 6   
2.3 Baseline via matrix Bernstein 7   
3 Oja’s Algorithm 8   
4 PCA Lower Bound 12   
5 High-Probability Guarantees 14   
6 Energy PCA Guarantees 18   
7 Application to Diferentially Private PCA 22   
7.1 Privacy 22   
7.2 Utility 24   
7.3 Private energy PCA 27   
8 Experiments 28   
8.1 Oja’s algorithm 28   
8.2 Private PCA 29   
AI Disclosure 31   
A Deferred Proofs from Section 6 35

## 1 Introduction

Let $\mathbf { A } _ { 1 } , \mathbf { A } _ { 2 } , \ldots \in \mathbb { R } ^ { d \times d }$ be i.i.d. stochastic matrices with common mean $\Sigma \succeq \mathbf { 0 } _ { d \times d }$ . In streaming principal component analysis (PCA), the goal is to recover a unit vector close to the largest eigenvector of Σ, while processing each update only once, ideally with small space overhead.

The classical Oja’s algorithm [Oja82] is perhaps the simplest method for this task: starting from a (randomly initialized) unit vector w , it repeatedly iterates

$$
\mathbf { w } _ { t }  \frac { ( \mathbf { I } _ { d } + \eta _ { t } \mathbf { A } _ { t } ) \mathbf { w } _ { t - 1 } } { \| ( \mathbf { I } _ { d } + \eta _ { t } \mathbf { A } _ { t } ) \mathbf { w } _ { t - 1 } \| _ { 2 } } .
$$

This update can be performed using O(d) auxiliary space, i.e., without storing a matrix explicitly. Oja’s algorithm is extremely well-studied [BDF13, Sha16, JJK<sup>+</sup>16, AZL17, HNWW21, Lia23], and is known to achieve near-optimal rates of convergence in various settings, under standard regularity assumptions on the sequence $\{ \mathbf { A } _ { t } \} _ { t \ge 1 }$ , such as a second moment bound and almost sure bound (Model 2). Notably, these are the same assumptions required by the matrix Bernstein concentration inequality (cf. Proposition 1), which solves the same stochastic eigenvector estimation problem nearoptimally, albeit in an ofline setting and using $O ( d ^ { 2 } )$ space.

We study the most general formulation of streaming PCA, where no gap assumptions are placed on Σ’s spectrum. PCA objectives become ill-conditioned when the leading eigenvalues are equal or close to equal, e.g., if $\lambda _ { 1 } ( \pmb { \Sigma } ) = \lambda _ { 2 } ( \pmb { \Sigma } )$ , then recovering the leading eigenvector is not even welldefined. A common alternative in such gap-free settings, popularized by [GH15, AZL16, AZL17], is to ask for a unit vector with little mass on eigenvectors whose eigenvalues are below $( 1 - \gamma ) \lambda _ { 1 } ( \pmb { \Sigma } )$ , for a parameter $\gamma \in ( 0 , 1 )$ ). We formalize this correlation PCA (cPCA) objective in Definition 1.

Perhaps surprisingly, all prior near-optimal rates for streaming PCA, via Oja’s algorithm or otherwise, either required an eigengap assumption on Σ [JJK<sup>+</sup>16, HNWW21], or were limited to the setting where every ${ \bf A } _ { t }$ is rank-one [AZL17, Lia23]. This motivates our work’s central question.

Does Oja’s algorithm achieve near-optimal convergence for streaming PCA,

without eigengap assumptions on Σ or rank restrictions on the $\{ \mathbf { A } _ { t } \} _ { t \ge 1 } ?$

## 1.1 Our results

Our main result (Theorem 1) answers this question afirmatively. In fact, its convergence guarantee holds under qualitatively weaker regularity assumptions (Model 1) than used by prior work. Assuming a bound on the standard matrix variance parameter,

$$
V : = \operatorname* { m a x } \left\{ \left\| \mathbb { E } [ ( \mathbf { A } _ { t } - \Sigma ) ( \mathbf { A } _ { t } - \Sigma ) ^ { \top } ] \right\| _ { \mathrm { o p } } , \left\| \mathbb { E } [ ( \mathbf { A } _ { t } - \Sigma ) ^ { \top } ( \mathbf { A } _ { t } - \Sigma ) ] \right\| _ { \mathrm { o p } } \right\} ,
$$

Theorem 1 shows that with constant probability, Oja’s algorithm returns a $( \gamma , \Delta ) \mathrm { - c P C A ~ ( i . e . }$ , has squared correlation at most $\Delta$ with the eigenspace below $( 1 - \gamma ) \lambda _ { 1 } ( \pmb { \Sigma } ) )$ , using<sup>1</sup>

$$
\widetilde { O } \left( \frac { V } { \lambda _ { 1 } ^ { 2 } \gamma ^ { 2 } \Delta } + \frac { 1 } { \gamma } \right)
$$

online samples ${ \bf A } _ { t }$ . The first term in the above rate is complemented with a nearly-matching lower bound in Theorem 2, and the second term is a consequence of the standard convergence rate of the (ofline) power method, in the special deterministic setting where all $\mathbf A _ { t } = \Sigma$

Interestingly, Theorem 1 holds under weaker requirements than earlier convergence analyses of $\mathrm { O j a ^ { \prime } s }$ algorithm. In particular, it only posits a matrix variance bound V (Model 1), and circumvents the almost sure bound (Model 2) typically used by prior works on streaming PCA, as well as the matrix Bernstein inequality. As a tradeof, it only ofers a constant success probability (more generally, Theorem 1’s sample complexity scales inverse-polynomially in the failure probability $\zeta )$ In Theorem 3, we give an alternative result that leverages geometric aggregation to achieve a $\mathrm { p o l y l o g } ( \frac { 1 } { \zeta } )$ sample complexity overhead. This result analyzes an extension of Oja’s algorithm to block matrices (Algorithm 2), and requires $d \cdot \mathrm { p o l y l o g } ( \frac { 1 } { \zeta } )$ auxiliary space.

In Section 6, we also consider the energy objective $\mathbf { w } ^ { \top } \pmb { \Sigma } \mathbf { w } \geq ( 1 - \alpha ) \lambda _ { 1 }$ (ePCA, Definition 2). While a black-box cPCA-to-ePCA conversion (Lemma 8, [JKL<sup>+</sup>24]) exists, its combination with Theorem 1 leads to a suboptimal sample complexity by a factor of ${ \frac { 1 } { \alpha } } ,$ . Instead, we give a multiscale reduction-based analysis in Theorem 4 that shows Oja’s algorithm returns an α-ePCA using

$$
\widetilde O \left( \frac { V } { \lambda _ { 1 } ^ { 2 } \alpha ^ { 2 } } + \frac { 1 } { \alpha } \right)
$$

samples. Here also, the first term is complemented with a nearly matching lower bound (Corollary 2). This result addresses an open question posed by Section 6 of $[ \mathrm { J } \mathrm { J } \mathrm { K } ^ { + } 1 6 ]$

Finally, as our main application, we consider the setting of diferentially private PCA, i.e., where the goal is to solve PCA subject to $( \varepsilon , \delta ) – \mathrm { D P }$ (Definition 3). A prior work by [LKJO22] achieved a near-optimal convergence rate for this problem under an eigengap. We give an analogous sample bound in the gap-free setting: for publicly known $\nu , \lambda _ { 1 }$ , Theorem $5$ returns an $( \varepsilon , \delta ) – \mathrm { D P }$ estimator that is $\mathrm { ~ a ~ } ( \gamma , \Delta ) { \mathrm { - c P C A } }$ with high probability, using

$$
\widetilde { O } \left( \frac { d \nu ^ { 4 } } { \gamma ^ { 2 } \lambda _ { 1 } ^ { 2 } \Delta } + \frac { d \nu ^ { 2 } } { \varepsilon \gamma \lambda _ { 1 } \sqrt { \Delta } } \right)
$$

samples. Our result is stated directly under sub-Gaussianity (Definition 4). For Gaussian data, $\nu ^ { 2 } =$ $\lambda _ { 1 }$ , and the polynomial dependence matches Corollary 5.2 of [LKJO22], with spectral resolution γ replacing the relative eigengap. We obtain a slightly better $\gamma$ dependence than [LKJO22] by avoiding minibatches, instead taking full passes to obtain an improved sensitivity tradeof. Its analysis uses Rényi diferential privacy to compose the Gaussian queries and control adaptive clipping. Further, applying the ePCA analysis to the same algorithm gives, under the same ν-sub-Gaussian model, an α-ePCA with sample complexity (Theorem 6),

$$
\widetilde { \cal O } \left( { \frac { d \nu ^ { 4 } } { \lambda _ { 1 } ^ { 2 } \alpha ^ { 2 } } } + { \frac { d \nu ^ { 2 } } { \varepsilon \lambda _ { 1 } \alpha } } \right) .
$$

In particular, for Gaussian data with a publicly known $\lambda _ { 1 }$ , a setting where $\nu ^ { 2 } = O ( \lambda _ { 1 } )$ , our new sample complexity bound above matches the rate conjectured by [Bro26] up to logarithmic factors. The main outstanding questions left by Theorem 6 are to remove the remaining polylogarithmic overhead, and to privately estimate $\lambda _ { 1 }$ from samples.

<table><tr><td>Work</td><td>General updates</td><td>Gap-free</td><td>Near-optimal rate</td></tr><tr><td>[Sha16], Corollary 1</td><td>V</td><td>V</td><td></td></tr><tr><td>[JJK+16], Theorem 3</td><td>V</td><td></td><td>V</td></tr><tr><td>[AZL17], Theorem 2</td><td></td><td>√</td><td>V</td></tr><tr><td>[HNWW21], Theorem 3</td><td>V</td><td></td><td>√</td></tr><tr><td>[Lia23], Theorem 3.3</td><td></td><td>V</td><td>V</td></tr><tr><td>This work, Theorem 1</td><td>V</td><td>√</td><td>V</td></tr></table>

Table 1: Representative streaming PCA guarantees. “Near-optimal” means the rate matches the lower bound (Theorem 2) up to logarithmic factors and low-order terms. “General updates” means no rank restrictions are placed, and only a statistical assumption (e.g., Models 1 or 2) is used.

## 1.2 Our techniques

Our main result, Theorem 1, follows from a new analysis of Oja’s algorithm that leads to arguably a simpler convergence proof than in prior works, e.g., [JJK<sup>+</sup>16]. We begin by overviewing this new strategy, and provide an overview of our auxiliary results (Theorems 2, 3, 4, 5, and 6).

Expected trace as a potential. Our analysis starts from the operator viewpoint of [JJK<sup>+</sup>16]. Writing the unnormalized $\mathrm { O j a }$ iterate as being induced by the random operator

$$
\mathbf { B } _ { t } = \left( \mathbf { I } _ { d } + \eta _ { t } \mathbf { A } _ { t } \right) \cdot \cdot \cdot \left( \mathbf { I } _ { d } + \eta _ { 1 } \mathbf { A } _ { 1 } \right) ,
$$

their analysis controls the ratio between the energy of $\mathbf { B } _ { t }$ in the orthogonal complement of the leading eigenvector $\mathbf { v } _ { 1 }$ and the energy along $\mathbf { v } _ { 1 }$ (reproduced as Lemma 2). At the population level, these two quantities evolve at rates governed by $\lambda _ { 2 }$ and $\lambda _ { 1 }$ , respectively, so their separation is driven by the eigengap $\lambda _ { 1 } - \lambda _ { 2 }$ . This is precisely what becomes problematic for a gap-free objective.

Our proof departs from this strategy, and instead compares $\mathbf { B } _ { t }$ with its population counterpart

$$
\mathbf { C } _ { t } = ( \mathbf { I } _ { d } + \eta _ { t } \pmb { \Sigma } ) \cdot \cdot \cdot ( \mathbf { I } _ { d } + \eta _ { 1 } \pmb { \Sigma } ) = \mathbb { E } [ \mathbf { B } _ { t } ] .
$$

Let P denote the orthogonal projector onto eigenvectors with eigenvalues below $( 1 - \gamma ) \lambda _ { 1 }$ . Our starting point is the following consequence of the triangle inequality,

$$
\frac { \left\| \mathbf { P B } _ { t } \right\| _ { \mathrm { F } } } { \left\| \mathbf { B } _ { t } \right\| _ { \mathrm { F } } } \leq \frac { \left\| \mathbf { P C } _ { t } \right\| _ { \mathrm { F } } } { \left\| \mathbf { C } _ { t } \right\| _ { \mathrm { F } } } + \left\| \frac { \mathbf { B } _ { t } } { \left\| \mathbf { B } _ { t } \right\| _ { \mathrm { F } } } - \frac { \mathbf { C } _ { t } } { \left\| \mathbf { C } _ { t } \right\| _ { \mathrm { F } } } \right\| _ { \mathrm { F } } \leq \frac { \left\| \mathbf { P C } _ { t } \right\| _ { \mathrm { F } } } { \left\| \mathbf { C } _ { t } \right\| _ { \mathrm { F } } } + \frac { 2 \left\| \mathbf { B } _ { t } - \mathbf { C } _ { t } \right\| _ { \mathrm { F } } } { \left\| \mathbf { C } _ { t } \right\| _ { \mathrm { F } } } ,\tag{1}
$$

where the last inequality holds by a derivation in (8). The left-hand side above is precisely the quantity that Lemma 2 seeks to control in order to yield cPCA guarantees.

This inequality splits our bound into two terms: a deterministic center (depending only on $\mathbf { C } _ { t } )$ and the relative deviation of a random $\mathbf { B } _ { t }$ . The first term is simple to control using analyses of the standard power method. To bound the second term, since $\mathbb { E } [ { \bf B } _ { t } ] = { \bf C } _ { t }$ , we have E $\| \mathbf { B } _ { t } - \mathbf { C } _ { t } \| _ { \mathrm { F } } ^ { 2 } =$ $\mathbb { E } [ \mathrm { T r } ( \mathbf { B } _ { t } \mathbf { B } _ { t } ^ { \top } ) ] - \mathrm { T r } ( \mathbf { C } _ { t } \mathbf { C } _ { t } ^ { \top } )$ , suggesting the use of $\mathbb { E } [ \mathrm { T r } ( \mathbf { B } _ { t } \mathbf { B } _ { t } ^ { \top } ) ]$ as our potential.

The heart of our new analysis is Lemma 3, which precisely achieves the required control of the expected trace, assuming only a matrix variance bound. Concretely, we show that under Model 1, Oja’s algorithm with step sizes $\{ \eta _ { s } \} _ { s \ge 1 }$ satisfies

$$
\mathbb { E } \left[ \mathrm { T r } ( \mathbf { B } _ { t } \mathbf { B } _ { t } ^ { \top } ) \right] \leq \exp \left( V \sum _ { s \in [ t ] } \eta _ { s } ^ { 2 } \right) \mathrm { T r } ( \mathbf { C } _ { t } \mathbf { C } _ { t } ^ { \top } ) .
$$

The proof of Lemma 3 inductively shows a majorization relationship between the spectra of $\mathbb { E } [ \mathbf { B } _ { t } \mathbf { B } _ { t } ^ { \top } ]$ and a scaled population counterpart $\mathbf { C } _ { t } \mathbf { C } _ { t } ^ { \top }$ , by using the von Neumann trace inequality and our matrix variance assumption to bound the efect of each increment.

Lower bound. Finally, we complement Theorem 1 with a lower bound in Theorem 2, which obtains matching dependences in all parameters up to polylogarithmic factors. Qualitatively similar lower bounds to Theorem 1 (e.g., Theorem 32, $[ \mathrm { G H J ^ { + } 1 6 } ]$ , and Theorem $6 , [ \mathrm { A Z L 1 7 } ] )$ were already known, and our main contribution is to slightly strengthen the construction to hold for the entire range of V and $\lambda _ { 1 }$ . In particular, our proof builds upon the lower bound construction of [AZL17].

Gap-free probability boosting. A standard strategy for boosting the success probability of PCA under an eigengap is to apply geometric aggregation (e.g., Lemma 3.10, [KS24]). Unfortunately, a direct output aggregation fails in a gap-free setting: when the leading eigenvalue has multiplicity, even two exact solutions may be orthogonal. Nonetheless, our proof strategy for Theorem 1 proceeds by arguing constant probability closeness of each $\mathbf { B } _ { t }$ to the population matrix $\mathbf { C } _ { t }$ , making it amenable to an intermediate geometric aggregation step. Our Algorithm 2 applies independent Oja products $\mathbf { B } _ { t } ^ { ( r ) }$ across $\begin{array} { r } { R = O ( \log ( \frac { 1 } { \zeta } ) ) } \end{array}$ disjoint streams and initializes each stream with the same Gaussian matrix G, using a slightly oversampled dimension (i.e., with $\begin{array} { r } { O ( \log ( \frac { 1 } { \zeta } ) ) } \end{array}$ random vectors rather than a single vector). Together with standard results on the concentration of Gaussian traces, we show that we can aggregate these disjoint streams to a center compatible with the strategy in (1), at a relatively mild $\mathrm { p o l y l o g } ( \frac { 1 } { \zeta } )$ cost to the sample complexity and space overhead.

Energy PCA. We next consider an energy PCA guarantee for Oja’s algorithm. A direct cPCAto-ePCA reduction (e.g. Lemma 8, [JKL<sup>+</sup>24]) results in a suboptimal sample complexity scaling as $\textstyle { \frac { 1 } { \alpha ^ { 3 } } }$ for an α-ePCA guarantee. In Proposition 2, we consider a multiscale cPCA, with simultaneous guarantees on the projections to eigenvalues below a specified threshold $( 1 - u ) \lambda _ { 1 }$ for all choices of $u \in ( 0 , 1 )$ , as opposed to just $u = \gamma$ . By integrating over $u ,$ we are able to obtain a sample complexity scaling as in $\textstyle { \frac { 1 } { \alpha ^ { 2 } } }$ in Theorem 4, which we also show is tight in Corollary 2.

Application to DP PCA. Private PCA is a natural application of Theorem 1. The DP-PCA method of [LKJO22] forms minibatch covariance estimates and adds Gaussian perturbations, so its efective Oja updates are general matrix-valued rather than rank one. Their utility analysis invokes the gapped Oja guarantee of $[ \mathrm { J } \mathrm { J } \mathrm { K } ^ { + } 1 6 ]$ , and consequently depends on $\lambda _ { 1 } - \lambda _ { 2 }$

Compared to [LKJO22], our analysis also yields an improved dependence on the gap parameter γ, set to $\begin{array} { r } { 1 - \frac { \lambda _ { 2 } } { \lambda _ { 1 } } } \end{array}$ in their setting. We reuse the full dataset at every Oja step, rather than splitting it into fresh minibatches as in [LKJO22]. This choice improves the sensitivity of each update by a factor of $b ,$ where b is the number of mini-batches, while leading to b passes over each sample. By paying for these passes using advanced composition (or Rényi DP [Mir17], to give slightly tighter guarantees), this only incurs an $\approx \sqrt { b }$ overhead, the source of our savings.

Interestingly, our analysis directly uses the algorithm’s privacy to argue about its correctness. This need arises due to a dependency between a currently estimated subspace and the data, which would afect clipping thresholds. We instead use a near-independence guarantee implied by DP to save a poly(d) factor in the threshold magnitude, which directly reflects in our sample complexity.

## 1.3 Related work

Streaming and gap-free PCA. Finite-sample analyses of streaming PCA include incremental PCA [BDF13], memory-optimal block methods [MCJ13], and stochastic power or matrixfactorization methods [SOR15]. Other variants address Markovian data [KS23], sparse leading eigenvectors [KS24], entrywise uncertainty quantification [KPS25], and low-precision computation [DKPS25]. Most closely related to our work, [JJK<sup>+</sup>16] obtained the first near-optimal gapped rates for general, possibly nonsymmetric matrix updates, while [LWLZ18] give near-optimal gapped guarantees for sub-Gaussian PCA. Relatedly, [Sha16] gives an early eigengap-free guarantee permitting general PSD stochastic matrices, but with a slower objective rate and low success probability from random initialization. Later, [AZL17] established an eficient global near-optimal gap-free analysis for rank-one streaming k-PCA, and [Lia23] obtained sharp gap-free rates for sub-Gaussian data, again requiring rank-one updates. In another direction, [HNWW21] extends nearly ofline-optimal streaming-PCA guarantees to arbitrary-rank updates under an eigengap.

We note that this work focuses on the 1-PCA problem, i.e., approximating the top eigenvector of a population average Σ from samples. We leave open the analogous question for k-PCA for $k > 1$ where a similar situation holds in the current literature: [AZL17] gave a gap-free result for k-PCA under rank-one updates, and [HNWW21] removed the rank restriction, but used an eigengap.

Noisy power methods. Under Model 1 and an eigengap assumption $\lambda _ { 2 } ( \Sigma ) \leq ( 1 - \gamma ) \lambda _ { 1 } ( \Sigma )$ [HP14] gives a suboptimal sample complexity scaling as $\begin{array} { r } { \widetilde { O } ( \frac { V } { \lambda _ { 1 } ^ { 2 } \gamma ^ { 3 } \Delta } + \frac { 1 } { \gamma } ) } \end{array}$ for minibatched stochastic matrix-vector products, even before accounting for their additional projected-noise condition (see the statement of their Corollary 1.1). This incurs an extra factor of $\textstyle { \frac { 1 } { \gamma } }$ in the leading term compared with Theorem 1. Later, [BDWY16] replaces a dependence on $\lambda _ { 1 } - \lambda _ { 2 }$ by $\lambda _ { 1 } - \lambda _ { q + 1 }$ , but requires maintaining at least q directions. Notably, both results hold only in the gapped setting.

Diferentially private PCA. For arbitrary row-bounded datasets, early approaches sample a direction using the exponential mechanism [CSS13], while Analyze Gauss [DTTZ14] adds a symmetric Gaussian matrix to the empirical covariance and then extracts its leading eigenspace. Notably, when applying such results to i.i.d. sub-Gaussian data, the resulting sample complexity is at least $d ^ { 1 . 5 }$ up to logarithmic factors. Specializing to i.i.d. statistical models, [LKO22] use robust one-dimensional scores within a propose-test-release framework to obtain nearly information-theoretically optima private PCA under sub-Gaussian and hypercontractive assumptions, although the resulting estimator is not computationally eficient. The black-box reduction of [HKMN23] converts suitable robust estimators into private mean and covariance estimators, from which PCA can be obtained by postprocessing when covariance error controls the desired subspace. Closest to our algorithm, [LKJO22] give a single-pass minibatched $\mathrm { O j a }$ method with nearly optimal rates for sub-Gaussian data under an eigengap. Subsequent specialized results obtain minimax rates for rank-r spiked covariance models [CXZ24] and robustness to heavy tails and contamination under elliptical models [KJ25].

## 2 Preliminaries

In Section 2.1, we give basic notation used throughout the paper, and in Section 2.2, we state the main streaming PCA problem we consider. In Section 2.3, we state a baseline result in the ofline setting via the matrix Bernstein theorem, under a slight strengthening of the problem formulation. We defer preliminaries on diferential privacy, used in our main application, to Section 7.

## 2.1 Notation

We use $X \perp Y$ to denote that random variables X and Y are independent. We use $\mathbb { I } _ { \mathcal { E } }$ to denote the   
0-1 indicator random variable of an event E. For two measures $\pi , \mu$ over the same sample space $\Omega$   
which we identify with corresponding distributions, $\begin{array} { r } { \mathrm { T V } ( \pi , \mu ) : = \frac { 1 } { 2 } \int _ { \Omega } | \pi - \mu | \mathrm { d } \omega } \end{array}$ denotes their TV   
distance and KL $\begin{array} { r } { _ { i } ( \pi \| \mu ) : = \int } \end{array}$ π log <sup>π</sup><sub>µ</sub> dω denotes their KL divergence. µ

Vectors are denoted in lowercase boldface and matrices in uppercase boldface. We use $\mathbf { 0 } _ { d }$ and $\mathbf { 1 } _ { d }$ to denote the all-zeroes and all-ones vectors in $\mathbb { R } ^ { d } , \mathbf { I } _ { d }$ to denote the $d \times d$ identity matrix, and ${ \bf 0 } _ { m \times n }$ to denote the $m \times n$ all-zeroes matrix. We use [d] to denote $\{ i \in \mathbb { N } : 1 \leq i \leq d \}$ . For $p \geq 1$ including $p = \infty$ we use $\left\| \cdot \right\| _ { p }$ to denote the $\ell _ { p }$ norm of a vector, and $\| \cdot \| _ { S _ { p } }$ to denote the Schatten-p norm of a matrix. The set $\mathbb { S } ^ { d \times d }$ denotes all $d \times d$ symmetric matrices, and $\mathbb { S } _ { \succ { \bf 0 } } ^ { d \times d }$ denotes the subset of positive semidefinite matrices. We use $\mathcal { N } ( \mathbf { m } , \pmb { \Sigma } )$ to denote the multivariate Gaussian with mean m $\in \mathbb { R } ^ { d }$ and covariance $\pmb { \Sigma } \in \mathbb { S } _ { \succ \mathbf { 0 } } ^ { d \times d }$ . We use $\left\| \cdot \right\| _ { \mathrm { o p } }$ to denote the $( \ell _ { 2 }$ induced) operator norm of a matrix, and $\lVert \cdot \rVert _ { \mathrm { F } }$ to denote its Frobenius norm, i.e., Schatten-2 norm. We use $\lambda _ { i } ( \cdot )$ to denote the $i ^ { \mathrm { t h } }$ largest eigenvalue of a symmetric matrix, and $\operatorname { T r } ( \cdot )$ for the trace. We say a matrix is orthonormal if its columns $\left\{ { { \bf { u } } _ { i } } \right\}$ satisfy $\langle \mathbf { u } _ { i } , \mathbf { u } _ { j } \rangle = \mathbb { I } _ { i = j }$ . For unit vectors u, v we define

$$
\begin{array} { r } { \mathrm { m } _ { \mathrm { s i g n } } \left( \mathbf { u } , \mathbf { v } \right) : = \operatorname* { m i n } \left. \left\| \mathbf { u } - \mathbf { v } \right\| _ { 2 } , \left\| \mathbf { u } + \mathbf { v } \right\| _ { 2 } \right. . } \end{array}
$$

Lemma 1. $\mathrm { m } _ { \mathrm { s i g n } }$ satisfies the triangle inequality.

Proof. For unit u, v, w, if $\begin{array} { r } { \mathrm { m } _ { \mathrm { s i g n } } \left( \mathbf { u } , \mathbf { w } \right) = \| \mathbf { u } - \sigma \mathbf { w } \| _ { 2 } , \mathrm { m } _ { \mathrm { s i g n } } \left( \mathbf { w } , \mathbf { v } \right) = \| \mathbf { w } - \tau \mathbf { v } \| _ { 2 } , \mathrm { f o r } \left( \sigma , \tau \right) \in \{ \pm 1 \} ^ { 2 } , } \end{array}$

$$
\begin{array} { r } { \operatorname* { m } _ { \mathrm { s i g n } } \left( \mathbf { u } , \mathbf { v } \right) \leq \left\| \mathbf { u } - \sigma \tau \mathbf { v } \right\| _ { 2 } \leq \left\| \mathbf { u } - \sigma \mathbf { w } \right\| _ { 2 } + \left\| \sigma \mathbf { w } - \sigma \tau \mathbf { v } \right\| _ { 2 } = \operatorname* { m } _ { \mathrm { s i g n } } \left( \mathbf { u } , \mathbf { w } \right) + \operatorname* { m } _ { \mathrm { s i g n } } \left( \mathbf { w } , \mathbf { v } \right) . } \end{array}
$$

## 2.2 Main problem

To state our main problem, we recall the following helpful definition from $[ \mathrm { J K L ^ { + } 2 4 } ]$ , which has emerged as a useful gap-free notion of PCA in the literature [GH15, AZL16, AZL17].

Definition 1 (cPCA). Let $( \gamma , \Delta ) \in ( 0 , 1 ) ^ { 2 }$ , and let $\pmb { \Sigma } \in \mathbb { S } ^ { d \times d }$ . We say that a unit vector $\mathbf { v } \in \mathbb { R } ^ { d }$ is a $( \gamma , \Delta ) \mathrm { - c P C A }$ (correlation PCA) of Σ if, letting orthonormal $\mathbf { S } \in \mathbb { R } ^ { d \times r }$ have the same column span as the eigenspace of Σ corresponding to eigenvalues $< ( 1 - \gamma ) \lambda _ { 1 } ( \pmb { \Sigma } )$

$$
\left\| \mathbf { S } ^ { \top } \mathbf { V } \right\| _ { \mathrm { F } } ^ { 2 } \leq \Delta .
$$

We now state the main statistical model we consider in this paper.

Model 1. Fix $\lambda _ { 1 } > 0$ and $V > 0$ . Let $\{ \mathbf { A } _ { t } \in \mathbb { R } ^ { d \times d } \} _ { t \in [ n ] }$ be i.i.d. with $\mathbb { E } \mathbf { A } _ { t } = \pmb { \Sigma } \in \mathbb { S } _ { \succ \mathbf { 0 } } ^ { d \times d }$ , and

$$
\left\| \mathbf { \Sigma } \right\| _ { \mathrm { o p } } = \lambda _ { 1 } , \quad \operatorname* { m a x } \left\{ \left\| \mathbb { E } \left[ \left( \mathbf { \Sigma } \mathbf { - } \mathbf { A } _ { t } \right) \left( \mathbf { \Sigma } \mathbf { - } \mathbf { A } _ { t } \right) ^ { \top } \right] \right\| _ { \mathrm { o p } } , \left\| \mathbb { E } \left[ \left( \mathbf { \Sigma } \mathbf { - } \mathbf { A } _ { t } \right) ^ { \top } \left( \mathbf { \Sigma } \mathbf { - } \mathbf { A } _ { t } \right) \right] \right\| _ { \mathrm { o p } } \right\} \leq V .
$$

The main problem this paper focuses on is computing a cPCA of Σ, given access to $\{ \mathbf { A } _ { t } \} _ { t \in [ n ] }$ arising from Model 1. Our algorithms’ sample complexities will depend on five parameters: $( V , \bar { \lambda _ { 1 } } )$ from Model 1, $( \gamma , \Delta )$ from Definition 1, and the failure probability, denoted $\zeta \in ( 0 , 1 )$

We consider this problem in two settings: the batch setting where one can arbitrarily manipulate the $\{ \mathbf { A } _ { t } \} _ { t \in [ n ] }$ , and the streaming setting, our main focus. In the streaming setting, the $\{ \mathbf { A } _ { t } \} _ { t \in [ n ] }$ are given in a stream, and once we receive ${ \bf A } _ { t }$ we can perform an update and then it is discarded from memory. The goal is to solve the cPCA problem with low external memory, ideally $O ( d )$

The main algorithm we consider for streaming PCA is Oja’s algorithm (Algorithm 1).

Algorithm 1: $\operatorname { O j a } \ ( \{ \mathbf { A } _ { t } , \eta _ { t } \} _ { t \in [ n ] } )$   
1 Input: $\{ \mathbf { A } _ { t } \in \mathbb { R } ^ { d \times d } , \eta _ { t } > 0 \} _ { t \in [ n ] }$   
2 $\mathbf { w } _ { 0 }  \mathbf { g } / \| \mathbf { g } \| _ { 2 } .$ , where $\mathbf { g } \sim \mathcal { N } ( \mathbf { 0 } _ { d } , \mathbf { I } _ { d } )$   
3 for $t \in [ n ]$ do   
4 $\mathbf { w } _ { t }  ( \mathbf { I } _ { d } + \eta _ { t } \mathbf { A } _ { t } ) \mathbf { w } _ { t - }$ −1   
5 $\mathbf { w } _ { t } \gets \mathbf { w } _ { t } / \left\| \mathbf { w } _ { t } \right\| _ { 2 }$   
6 end   
7 return ${ \bf w } _ { n }$

Note that Algorithm 1 is clearly a streaming algorithm. We introduce some helpful notation:

$$
\begin{array} { r l } & { \mathbf { B } _ { t } : = \left( \mathbf { I } _ { d } + \eta _ { t } \mathbf { A } _ { t } \right) \cdot \cdot \cdot \left( \mathbf { I } _ { d } + \eta _ { 1 } \mathbf { A } _ { 1 } \right) , } \\ & { \mathbf { C } _ { t } : = \left( \mathbf { I } _ { d } + \eta _ { t } \mathbf { \Sigma } \right) \cdot \cdot \cdot \left( \mathbf { I } _ { d } + \eta _ { 1 } \mathbf { \Sigma } \right) . } \end{array}\tag{2}
$$

With this notation, the updates in Algorithm 1 are equivalent to $\begin{array} { r } { \mathbf { w } _ { t } \gets \frac { \mathbf { B } _ { t } \mathbf { g } } { \| \mathbf { B } _ { t } \mathbf { g } \| _ { 2 } } } \end{array}$ . Further, $\mathbf { C } _ { t }$ helps track the unnormalized iterates for the corresponding updates using the average matrix Σ.

## 2.3 Baseline via matrix Bernstein

As a baseline, we recall a folklore result that in the batch setting, any approximate cPCA of the empirical covariance (with appropriate parameters) also solves the statistical cPCA problem. This result is stated under a slight strengthening of Model 1 that imposes a probability 1 bound on each sample $\mathbf { A } _ { t } ,$ but naturally yields a high-probability guarantee unlike Theorem 1.

Model 2. Fix $\lambda _ { 1 } > 0 , V > 0$ , and $M > 0$ . Let $\{ \mathbf { A } _ { t } \} _ { t \in [ n ] }$ from Model 1 additionally satisfy

$$
\begin{array} { r } { \left\| \sum - \mathbf { A } _ { t } \right\| _ { \mathrm { o p } } \leq M \ w i t h \ p r o b a b i l i t y \ 1 . } \end{array}
$$

Proposition 1 (Gap-free PCA via matrix Bernstein). Under Model 2, let $\begin{array} { r } { \widehat { \mathbf { \Sigma } } : = \frac { 1 } { 2 n } \sum _ { t \in [ n ] } ( \mathbf { A } _ { t } + \mathbf { A } _ { t } ^ { \top } ) } \end{array}$ and let v be any $( \frac { \gamma } { 6 } , \frac { \Delta } { 4 } ) – c P C A ~ f o r ~ \widehat { \Sigma }$ . Then for any $\zeta \in ( 0 , \frac { 1 } { 3 } )$ , v is a $( \gamma , \Delta ) { - } c P C A$ for Σ with $p r o b a b i l i t y \ge 1 - \zeta .$ , if for an appropriate constant,

$$
n = \Omega \left( \left( \frac { V } { \lambda _ { 1 } ^ { 2 } \gamma ^ { 2 } \Delta } + \frac { M } { \lambda _ { 1 } \gamma \sqrt { \Delta } } \right) \log { \left( \frac { d } { \zeta } \right) } \right) .
$$

Proof. This is almost the statement of Proposition 1, [Tia26], up to the assumptions on $\{ \mathbf { A } _ { t } \} _ { t \in [ n ] }$ The proof of Proposition 1, [Tia26] shows the result if with probability $\geq 1 - \zeta$

$$
\left\| \widehat { \boldsymbol { \Sigma } } - \boldsymbol { \Sigma } \right\| _ { \mathrm { o p } } \leq \frac { \lambda _ { 1 } \gamma \sqrt { \Delta } } { 4 } .
$$

To show this, set $\mathbf { E } _ { t } : = \mathbf { A } _ { t } - \Sigma$ and $\begin{array} { r } { \mathbf { E } : = \frac { 1 } { n } \sum _ { t \in [ n ] } \mathbf { E } _ { t } } \end{array}$ . The $\mathbf { E } _ { t }$ are independent and mean zero, so the bounds from Model 2 and the matrix Bernstein inequality (Theorem 6.6.1, [Tro15]) prove the above bound on $\| \mathbf { E } \| _ { \mathrm { o p } }$ . The claim follows from Jensen’s inequality and $\begin{array} { r } { \widehat { { \pmb \Sigma } } - { \pmb \Sigma } = \frac { 1 } { 2 } ( { \bf E } + \widehat { { \bf E } } ^ { \top } ) } \end{array}$ □

## 3 Oja’s Algorithm

In this section we give a new analysis of Oja’s algorithm (Algorithm 1), yielding cPCA guarantees in the general gap-free setting of Model 1. To simplify notation, we let orthonormal S span the eigenspace of Σ corresponding to eigenvalues $< ( 1 - \gamma ) \lambda _ { 1 } ( \pmb { \Sigma } )$ (in line with Definition 1), and denote the associated orthogonal projector by $\mathbf { P } : = \mathbf { S } \mathbf { S } ^ { \top } . ~ \mathrm { A l s o } .$ we require one helper fact.

Fact 1. Let $\mathbf { x } , \mathbf { y } , \mathbf { c } \in \mathbb { R } _ { > 0 } ^ { d }$ have nonincreasing coordinates, and suppose y weakly majorizes x. Then

$$
\sum _ { j \in [ k ] } \mathbf { c } _ { j } \mathbf { x } _ { j } \leq \sum _ { j \in [ k ] } \mathbf { c } _ { j } \mathbf { y } _ { j } \ f o r \ a l l \ k \in [ d ] .
$$

Proof. For all $k \in [ d ]$ let $\mathbf { s } _ { k } : = \textstyle \sum _ { j \in [ k ] } \mathbf { x } _ { j }$ and $\mathbf { t } _ { k } : = \textstyle \sum _ { j \in [ k ] } \mathbf { y } _ { j }$ . Then Abel’s summation formula gives

$$
\sum _ { j \in [ k ] } \mathbf { c } _ { j } ( \mathbf { y } _ { j } - \mathbf { x } _ { j } ) = \mathbf { c } _ { k } ( \mathbf { t } _ { k } - \mathbf { s } _ { k } ) + \sum _ { j \in [ k - 1 ] } ( \mathbf { c } _ { j } - \mathbf { c } _ { j + 1 } ) ( \mathbf { t } _ { j } - \mathbf { s } _ { j } ) \geq 0 .
$$

We first analyze a one-step power method, analogously to Lemma 3.1, $[ \mathrm { J } \mathrm { J } \mathrm { K } ^ { + } 1 6 ]$

Lemma 2. Let $\zeta \in ( 0 , \frac { 1 } { 3 } )$ , let $\mathbf { B } \in \mathbb { R } ^ { d \times d }$ not be the all-zeroes matrix, and let $\mathbf { V } \in \mathbb { R } ^ { d \times r }$ be orthonormal. $I f \mathbf { g } \sim \mathcal { N } ( \mathbf { 0 } _ { d } , \mathbf { I } _ { d } )$ , then with probability $\geq 1 - \zeta$ 2

$$
\frac { \left\| \mathbf { V } ^ { \top } \mathbf { B } \mathbf { g } \right\| _ { 2 } ^ { 2 } } { \left\| \mathbf { B } \mathbf { g } \right\| _ { 2 } ^ { 2 } } \leq \frac { 9 0 \log \left( \frac { 1 } { \zeta } \right) } { \zeta ^ { 2 } } \cdot \frac { \left\| \mathbf { V } \mathbf { V } ^ { \top } \mathbf { B } \right\| _ { \mathrm { F } } ^ { 2 } } { \left\| \mathbf { B } \right\| _ { \mathrm { F } } ^ { 2 } } .
$$

Proof. Define $\mathbf { H } : = \mathbf { B } ^ { \top } \mathbf { B }$ and $\mathbf { K } : = \mathbf { B } ^ { \top } \mathbf { V } \mathbf { V } ^ { \top } \mathbf { B }$ . Then our goal is to bound

$$
\frac { \left\| \mathbf { V } ^ { \top } \mathbf { B g } \right\| _ { 2 } ^ { 2 } } { \left\| \mathbf { B g } \right\| _ { 2 } ^ { 2 } } = \frac { \mathbf { g } ^ { \top } \mathbf { K g } } { \mathbf { g } ^ { \top } \mathbf { H } \mathbf { g } } .
$$

For the denominator, standard Gaussian anti-concentration (e.g., Lemma A.2.1, [KS24]) shows

$$
\mathbf { g } ^ { \top } \mathbf { H } \mathbf { g } \geq \frac { \zeta ^ { 2 } } { 4 e } \mathrm { T r } ( \mathbf { H } )
$$

with probability at least $1 - { \frac { \zeta } { 2 } }$ . Similarly, for the numerator, standard $\chi ^ { 2 }$ concentration bounds (e.g., Lemma 1, [LM00]) along with $\mathrm { T r } ( \mathbf { K } ^ { 2 } ) \leq \mathrm { T r } ( \mathbf { K } ) ^ { 2 } , \| \mathbf { K } \| _ { \mathrm { o p } } \leq \mathrm { T r } ( \mathbf { K } )$ , gives

$$
\operatorname* { P r } \Big ( \mathbf { g } ^ { \top } \mathbf { K } \mathbf { g } > ( 1 + 2 \sqrt { t } + 2 t ) \mathrm { T r } ( \mathbf { K } ) \Big ) \leq \exp ( - t ) \mathrm { ~ f o r ~ a l l ~ } t > 0 .
$$

Plugging in $\begin{array} { r } { t = \log ( \frac { 2 } { \zeta } ) \leq 2 \log ( \frac { 1 } { \zeta } ) } \end{array}$ and combining the above three displays gives the result. □

In Lemmas 3 and 4, we derive bounds on the ratio in Lemma 2 for $\mathbf { V }  \mathbf { S } .$ , as $\mathbf { B }  \mathbf { B } _ { t }$ undergoes the updates of Algorithm 1. We begin by tracking a trace-based potential.

Lemma 3. Under Model 1 and notation (2), the iterates of Algorithm 1 satisfy, for all $t \in [ n ]$

$$
\mathbb { E } \left[ \operatorname { T r } \left( \mathbf { B } _ { t } \mathbf { B } _ { t } ^ { \top } \right) \right] \leq \exp \left( V \sum _ { s \in [ t ] } \eta _ { s } ^ { 2 } \right) \operatorname { T r } \left( \mathbf { C } _ { t } \mathbf { C } _ { t } ^ { \top } \right) .
$$

Proof. Let $\lambda _ { j } : = \lambda _ { j } ( \pmb { \Sigma } )$ for shorthand. We prove inductively that, for every $s \geq 0$ and $k \in [ d ]$

$$
\sum _ { j \in [ k ] } \lambda _ { j } \left( \mathbb { E } [ \mathbf { B } _ { s } \mathbf { B } _ { s } ^ { \top } ] \right) \leq \left( \prod _ { r \in [ s ] } ( 1 + V \eta _ { r } ^ { 2 } ) \right) \sum _ { j \in [ k ] } \prod _ { r \in [ s ] } ( 1 + \eta _ { r } \lambda _ { j } ) ^ { 2 } .\tag{3}
$$

Then, taking $s  t , k  d ,$ and using $1 + x \leq e ^ { x }$ proves the claim, since all $\mathbf { I } _ { d } + \eta _ { s } \pmb { \Sigma }$ commute.

Clearly (3) holds for $s = 0$ (where we take empty products as 1). For the inductive step, suppose that (3) holds for $s - 1$ and all $k \in [ d ]$ . Upon expanding, we have

$$
\mathbb { E } [ \mathbf { B } _ { s } \mathbf { B } _ { s } ^ { \top } ] = ( \mathbf { I } _ { d } + \eta _ { s } \boldsymbol { \Sigma } ) \mathbb { E } [ \mathbf { B } _ { s - 1 } \mathbf { B } _ { s - 1 } ^ { \top } ] ( \mathbf { I } _ { d } + \eta _ { s } \boldsymbol { \Sigma } ) + \eta _ { s } ^ { 2 } \mathbb { E } \left[ ( \mathbf { A } _ { s } - \boldsymbol { \Sigma } ) \mathbb { E } [ \mathbf { B } _ { s - 1 } \mathbf { B } _ { s - 1 } ^ { \top } ] ( \mathbf { A } _ { s } - \boldsymbol { \Sigma } ) ^ { \top } \right] .\tag{4}
$$

We bound the two terms separately in order to apply (3). For the first term, for any PSD $\mathbf { C } , \mathbf { H } .$ letting Q be the projector onto any top-k eigenspace of CHC, and using $\mathbf { C Q C } \preceq \mathbf { C ^ { 2 } }$ 2

$$
\begin{array} { r l r } {  { \sum _ { j \in [ k ] } \lambda _ { j } ( \mathbf { C } \mathbf { H } \mathbf { C } ) = \langle \mathbf { Q } , \mathbf { C } \mathbf { H } \mathbf { C } \rangle = \mathrm { T r } ( \mathbf { H } \mathbf { C } \mathbf { Q } \mathbf { C } ) } } \\ & { } & { \leq \displaystyle \sum _ { j \in [ k ] } \lambda _ { j } ( \mathbf { H } ) \lambda _ { j } ( \mathbf { C } \mathbf { Q } \mathbf { C } ) \leq \sum _ { j \in [ k ] } \lambda _ { j } ( \mathbf { C } ) ^ { 2 } \lambda _ { j } ( \mathbf { H } ) . } \end{array}
$$

For the second term, observe that for every rank-k orthogonal projector Q,

$$
\mathbf { 0 } _ { d \times d } \preceq \mathbb { E } [ ( \mathbf { A } _ { s } - \Sigma ) ^ { \top } \mathbf { Q } ( \mathbf { A } _ { s } - \Sigma ) ] \preceq V \mathbf { I } _ { d } , \quad \mathrm { T r } \left( \mathbb { E } \left[ ( \mathbf { A } _ { s } - \Sigma ) ^ { \top } \mathbf { Q } ( \mathbf { A } _ { s } - \Sigma ) \right] \right) \leq k V .
$$

Then by the von Neumann trace inequality,

$$
\mathrm { T r } \left( \mathbf { Q } \mathbb { E } [ ( \mathbf { A } _ { s } - \Sigma ) \mathbf { H } ( \mathbf { A } _ { s } - \Sigma ) ^ { \top } ] \right) = \mathrm { T r } \left( \mathbf { H } \mathbb { E } [ ( \mathbf { A } _ { s } - \Sigma ) ^ { \top } \mathbf { Q } ( \mathbf { A } _ { s } - \Sigma ) ] \right) \le V \sum _ { j = 1 } ^ { k } \lambda _ { j } ( \mathbf { H } ) .
$$

By supremizing this over rank-k projectors Q, for every $\mathbf { H } \in \mathbb { S } _ { \succ \mathbf { 0 } } ^ { d \times d }$ and $k \in [ d ]$

$$
\sum _ { j \in [ k ] } \lambda _ { j } \Bigl ( \mathbb { E } [ ( \mathbf { A } _ { s } - \pmb { \Sigma } ) \mathbf { H } ( \mathbf { A } _ { s } - \pmb { \Sigma } ) ^ { \top } ] \Bigr ) \leq V \sum _ { j \in [ k ] } \lambda _ { j } ( \mathbf { H } ) ,
$$

which bounds the second term. Combining the above displays into (4), and using the triangle inequality of the Ky Fan norm,

$$
\begin{array} { r l } {  { \sum _ { j \in [ k ] } \lambda _ { j } ( \mathbb { E } [ \mathbf { B } _ { s } \mathbf { B } _ { s } ^ { \top } ] ) \leq \sum _ { j \in [ k ] } \big ( ( 1 + \eta _ { s } \lambda _ { j } ) ^ { 2 } + V \eta _ { s } ^ { 2 } \big ) \lambda _ { j } ( \mathbb { E } [ \mathbf { B } _ { s - 1 } \mathbf { B } _ { s - 1 } ^ { \top } ] ) } \quad } & { } \\ & { \leq ( 1 + V \eta _ { s } ^ { 2 } ) \sum _ { j \in [ k ] } ( 1 + \eta _ { s } \lambda _ { j } ) ^ { 2 } \lambda _ { j } ( \mathbb { E } [ \mathbf { B } _ { s - 1 } \mathbf { B } _ { s - 1 } ^ { \top } ] ) } \\ & { \leq ( \prod _ { r \in [ s ] } ( 1 + V \eta _ { r } ^ { 2 } ) ) \sum _ { j \in [ k ] } \prod _ { r \in [ s ] } ( 1 + \eta _ { r } \lambda _ { j } ) ^ { 2 } } \end{array}
$$

where the third line applies Fact 1 with

$$
\mathbf { c } _ { j } \gets ( 1 + \eta _ { s } \lambda _ { j } ) ^ { 2 } , \quad \mathbf { x } _ { j } \gets \lambda _ { j } ( \mathbb { E } [ \mathbf { B } _ { s - 1 } \mathbf { B } _ { s - 1 } ^ { \top } ] ) , \quad \mathbf { y } _ { j } \gets \left( \prod _ { r \in [ s - 1 ] } ( 1 + V \eta _ { r } ^ { 2 } ) \right) \prod _ { r \in [ s - 1 ] } ( 1 + \eta _ { r } \lambda _ { j } ) ^ { 2 } ,
$$

where y weakly majorizes x is the inductive hypothesis. Thus (3) holds as desired.

Lemma 4. Under Model 1 and notation (2), let $\zeta \in ( 0 , \frac { 1 } { 3 } )$ , let $t \in [ n ]$ , and suppose $\begin{array} { r } { \eta _ { s } \le \frac { 1 } { \lambda _ { 1 } } } \end{array}$ for all $s \in [ t ]$ . If we let $\begin{array} { r } { q _ { t } : = V \sum _ { s \in [ t ] } \eta _ { s } ^ { 2 } } \end{array}$ , and $\begin{array} { r } { q _ { t } \le \frac { \zeta } { 2 } } \end{array}$ , then with probability ≥ 1 − ζ,

$$
\frac { \| \mathbf { P B } _ { t } \| _ { \mathrm { F } } ^ { 2 } } { \| \mathbf { B } _ { t } \| _ { \mathrm { F } } ^ { 2 } } \le 2 d \exp \left( - \gamma \lambda _ { 1 } \sum _ { s \in [ t ] } \eta _ { s } \right) + \frac { 8 \left( \exp ( q _ { t } ) - 1 \right) } { \zeta } .
$$

Proof. First, observe that since $\begin{array} { r } { \| \mathbf { C } _ { t } \| _ { \mathrm { F } } ^ { 2 } \geq \| \mathbf { C } _ { t } ^ { 2 } \| _ { \mathrm { o p } } = \prod _ { s \in [ t ] } ( 1 + \eta _ { s } \lambda _ { 1 } ) ^ { 2 } } \end{array}$ , P commutes with all of the $\mathbf { I } _ { d } + \eta _ { s } \pmb { \Sigma }$ , and the corresponding eigenvalues of $\Sigma \ \mathrm { a r e } \leq ( \mathsf { \bar { 1 } } - \gamma ) \lambda _ { 1 }$

$$
\frac { \| \mathbf { P } \mathbf { C } _ { t } \| _ { \mathrm { F } } ^ { 2 } } { \| \mathbf { C } _ { t } \| _ { \mathrm { F } } ^ { 2 } } \leq d \prod _ { s \in [ t ] } \left( \frac { 1 + \eta _ { s } ( 1 - \gamma ) \lambda _ { 1 } } { 1 + \eta _ { s } \lambda _ { 1 } } \right) ^ { 2 } \leq d \exp \left( - \gamma \lambda _ { 1 } \sum _ { s \in [ t ] } \eta _ { s } \right) .\tag{5}
$$

Next, independence of the stream in Model 1 shows that $\mathbb { E } [ { \bf B } _ { t } ] = { \bf C } _ { t }$ , so Lemma 3 gives

$$
\begin{array} { r } { \mathbb { E } \left\| \mathbf { B } _ { t } - \mathbf { C } _ { t } \right\| _ { \mathrm { F } } ^ { 2 } = \mathbb { E } \left\| \mathbf { B } _ { t } \right\| _ { \mathrm { F } } ^ { 2 } - \left\| \mathbf { C } _ { t } \right\| _ { \mathrm { F } } ^ { 2 } \leq \left( \exp \left( q _ { t } \right) - 1 \right) \left\| \mathbf { C } _ { t } \right\| _ { \mathrm { F } } ^ { 2 } . } \end{array}\tag{6}
$$

Thus, by Markov’s inequality, we have with probability $\geq 1 - \zeta$ that

$$
\frac { \| \mathbf { B } _ { t } - \mathbf { C } _ { t } \| _ { \mathrm { F } } } { \| \mathbf { C } _ { t } \| _ { \mathrm { F } } } \leq \sqrt { \frac { \exp ( q _ { t } ) - 1 } { \zeta } } < 1 ,\tag{7}
$$

where the last inequality used our assumption $\begin{array} { r } { q _ { t } \le \frac { \zeta } { 2 } } \end{array}$ . Under this event, we have $\mathbf { B } _ { t } \neq \mathbf { 0 } _ { d \times d }$ by the triangle inequality. Further, for non-zero $\mathbf { X } , \mathbf { Y }$

$$
\left. \frac { \mathbf { X } } { \Vert \mathbf { X } \Vert _ { \mathrm { F } } } - \frac { \mathbf { Y } } { \Vert \mathbf { Y } \Vert _ { \mathrm { F } } } \right. _ { \mathrm { F } } \leq \left. \frac { \mathbf { X } - \mathbf { Y } } { \Vert \mathbf { Y } \Vert _ { \mathrm { F } } } \right. _ { \mathrm { F } } + \left. \mathbf { X } \cdot \left( \frac { 1 } { \Vert \mathbf { X } \Vert _ { \mathrm { F } } } - \frac { 1 } { \Vert \mathbf { Y } \Vert _ { \mathrm { F } } } \right) \right. _ { \mathrm { F } } \leq 2 \frac { \Vert \mathbf { X } - \mathbf { Y } \Vert _ { \mathrm { F } } } { \Vert \mathbf { Y } \Vert _ { \mathrm { F } } } .\tag{8}
$$

Finally, applying (8) with $\left( \mathbf { X } , \mathbf { Y } \right) \gets \left( \mathbf { B } _ { t } , \mathbf { C } _ { t } \right)$ implies

$$
\begin{array} { r l } { \frac { \left\| \mathbf { P B } _ { t } \right\| _ { \mathrm { F } } } { \left\| \mathbf { B } _ { t } \right\| _ { \mathrm { F } } } \leq \left\| \mathbf { P } \frac { \mathbf { C } _ { t } } { \left\| \mathbf { C } _ { t } \right\| _ { \mathrm { F } } } \right\| _ { \mathrm { F } } + \left\| \mathbf { P } \left( \frac { \mathbf { B } _ { t } } { \left\| \mathbf { B } _ { t } \right\| _ { \mathrm { F } } } - \frac { \mathbf { C } _ { t } } { \left\| \mathbf { C } _ { t } \right\| _ { \mathrm { F } } } \right) \right\| _ { \mathrm { F } } } & { } \\ { \leq \frac { \left\| \mathbf { P C } _ { t } \right\| _ { \mathrm { F } } } { \left\| \mathbf { C } _ { t } \right\| _ { \mathrm { F } } } + \left\| \frac { \mathbf { B } _ { t } } { \left\| \mathbf { B } _ { t } \right\| _ { \mathrm { F } } } - \frac { \mathbf { C } _ { t } } { \left\| \mathbf { C } _ { t } \right\| _ { \mathrm { F } } } \right\| _ { \mathrm { F } } } & { } \\  \leq \sqrt { d \exp \left( - \gamma \lambda _ { 1 } \sum _ { s \in [ t ] } \eta _ { s } \right) + 2 \sqrt { \frac { \exp \left( q _ { t } \right) - 1 } { \zeta } } , } & { } \end{array}
$$

where we used (5) and (7) in the last line. The conclusion follows from $( a + b ) ^ { 2 } \leq 2 ( a ^ { 2 } + b ^ { 2 } )$ □

At this point, we are ready to give our main bound in this section.

Theorem 1. Let $\zeta \in ( 0 , \frac { 1 } { 3 } )$ and $( \gamma , \Delta ) \in ( 0 , 1 ) ^ { 2 }$ . Under Model 1, if

$$
n = \Omega \left( \frac { V } { \lambda _ { 1 } ^ { 2 } \gamma ^ { 2 } \Delta } \cdot \frac { \log ^ { 2 } \left( \frac { d } { \Delta \zeta } \right) \log \left( \frac { 1 } { \zeta } \right) } { \zeta ^ { 3 } } + \frac { \log \left( \frac { d } { \Delta \zeta } \right) } { \gamma } \right) ,
$$

for an appropriate constant, then there exists $\beta \in \mathbb { R } _ { > 0 }$ such that taking $\begin{array} { r } { \eta _ { t } = \frac { \log ( \frac { 1 4 4 0 d } { \Delta \zeta ^ { 3 } } ) } { \gamma \lambda _ { 1 } ( \beta + t ) } ~ f o r ~ t \in [ n ] } \end{array}$ the output of Algorithm 1 is $a ~ ( \gamma , \Delta ) { - } c P C A$ of Σ with probability $\geq 1 - \zeta$

Proof. For shorthand, denote $\begin{array} { r } { L : = \log ( \frac { 1 4 4 0 d } { \Delta \zeta ^ { 3 } } ) } \end{array}$ , and for a large enough constant $C ,$ let

$$
\beta : = \operatorname* { m a x } \left\{ \frac { 8 L } { \gamma } , \frac { C V L ^ { 2 } } { \gamma ^ { 2 } \lambda _ { 1 } ^ { 2 } \Delta } \cdot \frac { \log \left( \frac { 1 } { \zeta } \right) } { \zeta ^ { 3 } } \right\} ,
$$

and $n \geq 4 \beta$ , so that all $\begin{array} { r } { \eta _ { s } \le \frac { 1 } { \lambda _ { 1 } } } \end{array}$ . Moreover, $\begin{array} { r } { \sum _ { s \in [ n ] } \frac { 1 } { ( \beta + s ) ^ { 2 } } \le \int _ { \beta } ^ { \infty } \frac { \mathrm { d } x } { x ^ { 2 } } = \frac { 1 } { \beta } } \end{array}$ , so for large enough $C ,$

$$
q _ { n } = V \sum _ { s \in [ n ] } \eta _ { s } ^ { 2 } \leq \frac { V L ^ { 2 } } { \gamma ^ { 2 } \lambda _ { 1 } ^ { 2 } \beta } \leq \frac { \Delta } { 2 } \cdot \frac { \zeta ^ { 3 } } { 3 2 \cdot 3 6 0 \cdot \log \left( \frac { 2 } { \zeta } \right) } .
$$

Thus, $\begin{array} { r } { q _ { n } \leq \frac { \zeta } { 4 } } \end{array}$ , so Lemmas 2 and 4 both apply at failure probability ${ \frac { \zeta } { 2 } } ,$ , and combining gives

$$
\frac { \left\| \mathbf { S } ^ { \top } \mathbf { B } _ { n } \mathbf { g } \right\| _ { 2 } ^ { 2 } } { \left\| \mathbf { B } _ { n } \mathbf { g } \right\| _ { 2 } ^ { 2 } } \leq \frac { 3 6 0 \log \left( \frac { 2 } { \zeta } \right) } { \zeta ^ { 2 } } \left( 2 d \exp \left( - \gamma \lambda _ { 1 } \sum _ { s \in [ n ] } \eta _ { s } \right) + \frac { 1 6 ( \exp ( q _ { n } ) - 1 ) } { \zeta } \right) ,
$$

with probability $\geq 1 - \zeta$ over the randomness of $\mathbf { g } \sim \mathcal { N } ( \mathbf { 0 } _ { d } , \mathbf { I } _ { d } )$ and Model 1. Condition on this event henceforth. For the first term above, since $n \geq 4 \beta$ , an integral comparison gives

$$
\gamma \lambda _ { 1 } \sum _ { s \in [ n ] } \eta _ { s } = L \sum _ { s \in [ n ] } { \frac { 1 } { \beta + s } } \geq L \int _ { 1 } ^ { n + 1 } { \frac { \mathrm { d } x } { \beta + x } } \geq L \log \left( { \frac { \beta + n + 1 } { \beta + 1 } } \right) \geq L .
$$

Combining the above three displays concludes the proof, upon simplifying using $\exp ( q _ { n } ) - 1 \leq 2 q _ { n }$ and $\begin{array} { r } { \log ( \frac { 2 } { \zeta } ) \leq \frac { 1 } { \zeta } } \end{array}$ , in the relevant parameter regimes. □

## 4 PCA Lower Bound

We give an information-theoretic lower bound that shows the leading-order parameter dependence in Theorem 1 is qualitatively tight, for any choice of $V , \lambda _ { 1 }$ . We state our result under Model 1, but our hard instance is even more well-behaved: the matrices $\mathbf { A } _ { i }$ are always PSD.

To begin, we require a standard formulation of Le Cam’s two point method.

Lemma 5 (Theorem 2.2(i), [Tsy09]). Let $P _ { 0 } , P _ { 1 }$ be probability distributions on the same measurable space Ω, and let $\phi : \Omega \to \{ 0 , 1 \}$ be a (possibly randomized) function. Then,

$$
\operatorname* { P r } _ { \omega \sim P _ { 0 } } \left[ \phi ( \omega ) = 1 \right] + \operatorname* { P r } _ { \omega \sim P _ { 1 } } \left[ \phi ( \omega ) = 0 \right] \ge 1 - \mathrm { T V } \left( P _ { 0 } , P _ { 1 } \right) .
$$

We can now state and prove our lower bound.

Theorem 2. Fix any choice of $\lambda _ { 1 } > 0 , V > 0 , \gamma \in ( 0 , \frac { 1 } { 4 } )$ , and $\Delta \in ( 0 , { \frac { 1 } { 4 } } )$ . There is no algorithm A that takes as input $\{ \mathbf { A } _ { i } \} _ { i \in [ n ] }$ from Model 1, and outputs a $( \gamma , \Delta ) { - } c P C A$ of Σ with probability $\geq \frac { 2 } { 3 }$ even assuming that $\mathbf { A } _ { i } \in \mathbb { S } _ { \succeq \mathbf { 0 } } ^ { d \times d }$ for all $i \in [ n ]$ , unless for an appropriate constant,

$$
n = \Omega \left( { \frac { V } { \lambda _ { 1 } ^ { 2 } \gamma ^ { 2 } \Delta } } \right) .
$$

Proof. We begin by defining matrices used in our construction. Let $\alpha : = 1 . 5$ arcsin $\sqrt { \Delta }$ , and

$$
\mathbf { u } _ { 0 } : = \left( \begin{array} { l } { \cos \alpha } \\ { - \sin \alpha } \end{array} \right) , \quad \mathbf { v } _ { 0 } : = \left( \begin{array} { l } { \sin \alpha } \\ { \cos \alpha } \end{array} \right) , \quad \mathbf { u } _ { 1 } : = \left( \begin{array} { l } { \cos \alpha } \\ { \sin \alpha } \end{array} \right) , \quad \mathbf { v } _ { 1 } : = \left( \begin{array} { l } { - \sin \alpha } \\ { \cos \alpha } \end{array} \right) .
$$

Also, let

$$
\pmb { \Sigma } _ { 0 } : = \lambda _ { 1 } \pmb { \mathbf { u } } _ { 0 } \pmb { \mathbf { u } } _ { 0 } ^ { \top } + ( 1 - 2 \gamma ) \lambda _ { 1 } \pmb { \mathbf { v } } _ { 0 } \pmb { \mathbf { v } } _ { 0 } ^ { \top } , \quad \pmb { \Sigma } _ { 1 } : = \lambda _ { 1 } \pmb { \mathbf { u } } _ { 1 } \pmb { \mathbf { u } } _ { 1 } ^ { \top } + ( 1 - 2 \gamma ) \lambda _ { 1 } \pmb { \mathbf { v } } _ { 1 } \pmb { \mathbf { v } } _ { 1 } ^ { \top } .
$$

If u is a $( \gamma , \Delta ) \mathrm { - c P C A }$ of $\Sigma _ { 0 }$ , then $\begin{array} { r } { \mathrm { m } _ { \mathrm { s i g n } } ( \mathbf { u } , \mathbf { u } _ { 0 } ) \leq 2 \sin ( \frac { \alpha } { 3 } ) } \end{array}$ , and a similar bound holds for $\Sigma _ { 1 }$ . We observe two helpful reformulations of $\Sigma _ { 0 } , \Sigma _ { 1 }$ used in our constructions. First,

$$
\begin{array} { r l } & { \qquad \Sigma _ { i } = { \bf D } + ( 2 i - 1 ) \beta { \bf H } \mathrm { ~ f o r ~ } i \in \{ 0 , 1 \} , \mathrm { ~ w h e r e ~ } \beta : = 2 \gamma \lambda _ { 1 } \sin ( \alpha ) \cos ( \alpha ) , } \\ & { \qquad { \bf D } : = \lambda _ { 1 } \left( \cos ^ { 2 } \alpha + ( 1 - 2 \gamma ) \sin ^ { 2 } \alpha \right. \qquad \quad 0 \qquad } \\ & { \qquad 0 \left. \sin ^ { 2 } \alpha + ( 1 - 2 \gamma ) \cos ^ { 2 } \alpha \right) , \quad { \bf H } : = \left( \begin{array} { c c } { 0 } & { 1 } \\ { 1 } & { 0 } \end{array} \right) . } \end{array}\tag{9}
$$

Second, letting the diagonal elements of D be $d _ { 1 }$ and $d _ { 2 }$ , and $s : = d _ { 1 } + d _ { 2 } = 2 ( 1 - \gamma ) \lambda _ { 1 }$

$$
\begin{array} { r l } & { \qquad \pmb { \Sigma } _ { i } = s \left( \left( \frac { 1 } { 2 } + ( 2 i - 1 ) \eta \right) \mathbf { w } \mathbf { w } ^ { \top } + \left( \frac { 1 } { 2 } - ( 2 i - 1 ) \eta \right) \mathbf { z } \mathbf { z } ^ { \top } \right) , \mathrm { ~ w h e r e ~ } \eta : = \frac { \beta } { 2 \sqrt { d _ { 1 } d _ { 2 } } } , } \\ & { \qquad \quad \mathbf { w } : = \frac { 1 } { \sqrt { s } } \left( \sqrt { d _ { 1 } } \right) , \quad \mathbf { z } : = \frac { 1 } { \sqrt { s } } \left( \sqrt { d _ { 1 } } \right) . } \end{array}\tag{10}
$$

Now suppose there is an algorithm A as in the theorem statement, and define $\phi : ( \mathbb { S } _ { \succ \mathbf { 0 } } ^ { d \times d } ) ^ { n }  \{ 0 , 1 \}$ as follows. Given matrices $\{ \mathbf { A } _ { i } \} _ { i \in [ n ] }$ , let $\mathbf { u } : = \mathcal { A } ( \{ \mathbf { A } _ { i } \} _ { i \in [ n ] } )$ be the assumed algorithm’s output. Then we let $\phi$ be the composition of A with the map $\mathbf { u }  i \in \{ 0 , 1 \}$ , where $i = 0$ if $\begin{array} { r } { \mathrm { m } _ { \mathrm { s i g n } } ( \mathbf { u } , \mathbf { u } _ { 0 } ) \leq } \end{array}$ $\mathrm { m } _ { \mathrm { s i g n } } ( \mathbf { u } , \mathbf { u } _ { 1 } )$ and $i = 1$ otherwise. Because $\begin{array} { r } { \mathrm { m } _ { \mathrm { s i g n } } ( \mathbf { u } _ { 0 } , \mathbf { u } _ { 1 } ) = 2 \sin \alpha > 4 \sin ( \frac { \alpha } { 3 } ) } \end{array}$ , Lemma 1 implies that $\phi$ identifies $i \in \{ 0 , 1 \}$ whenever u is a $( \gamma , \Delta ) \mathrm { - c P C A }$ of the corresponding $\dot { \Sigma } _ { i }$

We next define our distributions on the $\{ \mathbf { A } _ { i } \} _ { i \in [ n ] }$ . We split into two cases depending on $V .$ . In each case, we define single-sample laws $p _ { 0 } , p _ { 1 }$ with means $\Sigma _ { 0 } , \Sigma _ { 1 }$ , and let $P _ { 0 } : = p _ { 0 } ^ { \otimes n } , P _ { 1 } : = p _ { 1 } ^ { \otimes n }$ be the $n { \mathrm { - f o l d } }$ product laws. We show that our laws satisfy Model 1, and bound $\mathrm { K L } ( p _ { 0 } \| p _ { 1 } )$ . In this setting, existence of the stated algorithm A implies that

$$
\operatorname* { P r } _ { \omega \sim P _ { 0 } } \left[ \phi ( \omega ) = 1 \right] \leq \frac { 1 } { 3 } , \quad \operatorname* { P r } _ { \omega \sim P _ { 1 } } \left[ \phi ( \omega ) = 0 \right] \leq \frac { 1 } { 3 } .\tag{11}
$$

Case 1: $V \leq ( 1 - 2 \gamma ) \lambda _ { 1 } ^ { 2 }$ . We follow the notation (9). Let $t : = \sqrt { V + \beta ^ { 2 } }$ . We define the law of $\mathbf { A } \sim p _ { i }$ for $i \in \{ 0 , 1 \}$ as follows. First, we draw $\sigma \in \{ \pm 1 \}$ with $\begin{array} { r } { \mathbb { E } [ \sigma ] = \frac { ( 2 i - 1 ) \beta } { t } } \end{array}$ , so that

$$
\sigma = \left\{ \begin{array} { l l } { 1 } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } \frac { 1 } { 2 } + \frac { ( 2 i - 1 ) \beta } { 2 t } } \\ { - 1 } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } \frac { 1 } { 2 } - \frac { ( 2 i - 1 ) \beta } { 2 t } } \end{array} , \right.
$$

We then set $\mathbf { A } \gets \mathbf { D } + \sigma t \mathbf { H }$ . Observe that $\mathbb { E } _ { p _ { i } } [ \mathbf { A } ] = \mathbf { D } + ( 2 i - 1 ) \beta \mathbf { H } = \Sigma _ { i }$ from (9),

$$
\mathbb { E } _ { p _ { i } } \left[ \left( \pmb { \Sigma } _ { i } - \mathbf { A } \right) ^ { 2 } \right] = \left( t ^ { 2 } - \beta ^ { 2 } \right) \mathbf { H } ^ { 2 } = V \mathbf { I } _ { 2 } ,
$$

and A has positive entries on the diagonal, with

$$
\operatorname* { d e t } \left( \mathbf { A } \right) = \operatorname* { d e t } \left( \mathbf { D } \right) - t ^ { 2 } = ( 1 - 2 \gamma ) \lambda _ { 1 } ^ { 2 } + \beta ^ { 2 } - t ^ { 2 } \geq 0 .
$$

Thus, draws from both $P _ { 0 }$ and $P _ { 1 }$ are valid instances of Model 1. We also have

$$
\mathrm { K L } \left( p _ { 0 } \| p _ { 1 } \right) = \frac { \beta } { t } \log \left( \frac { 1 + \frac { \beta } { t } } { 1 - \frac { \beta } { t } } \right) \leq \frac { 2 \beta ^ { 2 } } { t ^ { 2 } - \beta ^ { 2 } } \leq \frac { 1 8 \gamma ^ { 2 } \lambda _ { 1 } ^ { 2 } \Delta } { V } ,
$$

where we computed the KL between two distributions on {±1} with probabilities $\begin{array} { r } { \frac { 1 } { 2 } \pm \frac { \beta } { 2 t } } \end{array}$ , and used the bounds $\beta ^ { 2 } \le 9 \gamma ^ { 2 } \lambda _ { 1 } ^ { 2 } \Delta$ and x log $\textstyle { \frac { 1 + x } { 1 - x } } \leq { \frac { 2 x ^ { 2 } } { 1 - x ^ { 2 } } }$ valid for $x \in [ 0 , 1 )$

Case 2: $V > ( 1 - 2 \gamma ) \lambda _ { 1 } ^ { 2 }$ . We follow the notation (9), (10). Let $\begin{array} { r } { R : = \lambda _ { 1 } + \frac { V } { \lambda _ { 1 } } } \end{array}$ and $\begin{array} { r } { q : = \frac { s } { R } < 1 } \end{array}$ . We define the law of $\mathbf { A } \sim p _ { i }$ for $i \in \{ 0 , 1 \}$ as follows. We set

$$
\mathbf { A } = \left\{ \begin{array} { l l } { R \mathbf { w } \mathbf { w } ^ { \top } } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } q \left( \frac { 1 } { 2 } + ( 2 i - 1 ) \eta \right) } \\ { R \mathbf { z } \mathbf { z } ^ { \top } } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } q \left( \frac { 1 } { 2 } - ( 2 i - 1 ) \eta \right) . } \\ { \mathbf { 0 } _ { 2 \times 2 } } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } 1 - q } \end{array} \right.
$$

Observe that $\mathbb { E } _ { p _ { i } } [ \mathbf { A } ] = \pmb { \Sigma } _ { i }$ from (10), and A is clearly always PSD. Further, because $R \lambda _ { 1 } - \lambda _ { 1 } ^ { 2 } = V$ and $R ( 1 - 2 \gamma ) \lambda _ { 1 } - ( 1 - 2 \gamma ) ^ { 2 } \lambda _ { 1 } ^ { 2 } = ( 1 - 2 \gamma ) ( V + 2 \gamma \lambda _ { 1 } ^ { 2 } ) \leq V$ , we have

$$
\begin{array} { r } { \mathbb { E } _ { p _ { i } } \left[ \left( \pmb { \Sigma } _ { i } - \mathbf { A } \right) ^ { 2 } \right] = R \pmb { \Sigma } _ { i } - \pmb { \Sigma } _ { i } ^ { 2 } \preceq V \mathbf { I } _ { 2 } . } \end{array}
$$

Thus, draws from both $P _ { 0 }$ and $P _ { 1 }$ are valid instances of Model 1. We can finally directly compute

$$
\mathrm { K L } \left( p _ { 0 } \| p _ { 1 } \right) = ( 2 \eta q ) \log \frac { 1 + 2 \eta } { 1 - 2 \eta } \le \frac { 8 q \eta ^ { 2 } } { 1 - 4 \eta ^ { 2 } } = \frac { 2 q \beta ^ { 2 } } { d _ { 1 } d _ { 2 } - \beta ^ { 2 } } \le \frac { 8 \beta ^ { 2 } } { V } \le \frac { 7 2 \gamma ^ { 2 } \lambda _ { 1 } ^ { 2 } \Delta } { V } ,
$$

where we used $\begin{array} { r } { d _ { 1 } d _ { 2 } - \beta ^ { 2 } = ( 1 - 2 \gamma ) \lambda _ { 1 } ^ { 2 } , q \le \frac { 2 \lambda _ { 1 } } { R } \le \frac { 2 \lambda _ { 1 } ^ { 2 } } { V } } \end{array}$ , and our earlier bound $\beta ^ { 2 } \le 9 \gamma ^ { 2 } \lambda _ { 1 } ^ { 2 } \Delta$

In summary, in all regimes of $( V , \lambda _ { 1 } )$ , there are instances $P _ { 0 } , P _ { 1 }$ of Model 1 such that $P _ { 0 } = p _ { 0 } ^ { \otimes n }$ $P _ { 1 } = p _ { 1 } ^ { \otimes n }$ , the mean of $p _ { i }$ is $\Sigma _ { i }$ for $i \in \{ 0 , 1 \}$ , and $\begin{array} { r } { \mathrm { K L } ( p _ { 0 } \| p _ { 1 } ) = O ( \frac { \gamma ^ { 2 } \lambda _ { 1 } ^ { 2 } \Delta } { V } ) } \end{array}$

By tensorization of KL divergence, we have for suficiently small $\begin{array} { r } { n = o ( \frac { V } { \lambda _ { 1 } ^ { 2 } \gamma ^ { 2 } \Delta } ) } \end{array}$ that $\begin{array} { r } { \mathrm { T V } \left( P _ { 0 } , P _ { 1 } \right) < \frac { 1 } { 3 } } \end{array}$ Combining with Lemma 5 and (11), this contradicts existence of $\mathcal { A }$ □

## 5 High-Probability Guarantees

Theorem 1 achieves a near-optimal rate of error as a function of the parameters in Model 1 and Definition 1, but only provides a low-confidence guarantee (i.e., with polynomial dependence on $\textstyle { \frac { 1 } { \zeta } } \int$ We next give a confidence amplification procedure with only polylogarithmic overheads, without worsening the dependence on either of the cPCA parameters $( \gamma , \Delta )$

Notably, standard confidence boosting procedures based on a direct geometric aggregation $\left( \mathrm { e . g . } \right.$ Lemma 3.10, [KS24]) do not work, since if the leading eigenvalue has multiplicity larger than one, even two exact cPCAs may be orthogonal. Instead, we aggregate sketches of the unnormalized Oja product, piggybacking of of closeness guarantees from Lemma 4. The main technical novelty in this section is that we require a polylogarithmic-dimension sketch (rather than the single Gaussian used in Algorithm 1), so that the population-level sketch concentrates with high probability.

We begin with a standard Gaussian trace estimate.

Lemma 6. Let $\mathbf { H } \succeq \mathbf { 0 }$ be fixed and let $\mathbf { G } \in \mathbb { R } ^ { d \times s }$ have i.i.d. $\mathcal { N } ( 0 , 1 )$ entries. Then,

$$
\frac { s } { 2 } \mathrm { T r } ( \mathbf { H } ) \leq \mathrm { T r } ( \mathbf { G } ^ { \top } \mathbf { H } \mathbf { G } ) \leq 2 s \mathrm { T r } ( \mathbf { H } ) ,\tag{12}
$$

with probability $\geq 1 - \exp ( - \frac { s } { 1 6 } )$ for each inequality.

Proof. Let the columns of G be $\{ \mathbf { g } _ { i } \} _ { i \in [ s ] }$ . Then, the middle term in (12) is $\begin{array} { r } { X : = \sum _ { j \in [ s ] } \mathbf { g } _ { j } ^ { \top } \mathbf { H } \mathbf { g } _ { j } } \end{array}$ The claim follows from $\chi ^ { 2 }$ concentration. In particular, Lemma 1, [LM00] gives

$$
\begin{array} { r } { \mathrm { P r } \left( X - s \mathrm { T r } ( \mathbf { H } ) \geq 2 \sqrt { s \mathrm { T r } ( \mathbf { H } ^ { 2 } ) x } + 2 \left. \mathbf { H } \right. _ { \mathrm { o p } } x \right) \leq \exp ( - x ) } \\ { \mathrm { P r } \left( s \mathrm { T r } ( \mathbf { H } ) - X \geq 2 \sqrt { s \mathrm { T r } ( \mathbf { H } ^ { 2 } ) x } \right) \leq \exp ( - x ) . } \end{array}
$$

Using $\begin{array} { r } { x  \frac { s } { 1 6 } , \mathrm { T r } ( \mathbf H ^ { 2 } ) \leq \mathrm { T r } ( \mathbf H ) ^ { 2 } } \end{array}$ , and $\| \mathbf { H } \| _ { \mathrm { o p } } \leq \mathrm { T r } ( \mathbf { H } )$ proves the claim.

Next, we show that independent product sketches cluster around a common population sketch.

Lemma 7. Let $\mathbf { G } \in \mathbb { R } ^ { d \times R }$ have i.i.d. $\mathcal { N } ( 0 , 1 )$ entries. Under Model 1, if $\begin{array} { r } { \eta _ { s } \le \frac { 1 } { \lambda _ { 1 } } } \end{array}$ for all $s \in [ t ]$ and $\begin{array} { r } { q _ { t } : = V \sum _ { s \in [ t ] } \eta _ { s } ^ { 2 } \leq \frac { 1 } { 1 2 8 } } \end{array}$ , then with probability $\geq 1 - 3 \exp ( - \frac { R } { 1 6 } )$ over G, the following hold. First,

$$
\| \mathbf { P } \mathbf { Z } _ { \star , t } \| _ { \mathrm { F } } ^ { 2 } \leq 4 d \exp \left( - \gamma \lambda _ { 1 } \sum _ { s \in [ t ] } \eta _ { s } \right) , \ f o r \ \mathbf { Z } _ { \star , t } : = \frac { \mathbf { C } _ { t } \mathbf { G } } { \| \mathbf { C } _ { t } \mathbf { G } \| _ { \mathrm { F } } } .\tag{13}
$$

Second, conditioning on $\mathbf { G }$ , over the randomness of Model $^ { 1 , }$

$$
\operatorname* { P r } \left[ \left\| \frac { \mathbf { B } _ { t } \mathbf { G } } { \left\| \mathbf { B } _ { t } \mathbf { G } \right\| _ { \mathrm { F } } } - \mathbf { Z } _ { \star , t } \right\| _ { \mathrm { F } } \leq 8 \sqrt { 2 q _ { t } } \right] \geq \frac { 3 } { 4 } .\tag{14}
$$

Proof. Throughout the proof, let $\mathbf { H } _ { t } : = \mathbb { E } [ ( \mathbf { B } _ { t } - \mathbf { C } _ { t } ) ^ { \top } ( \mathbf { B } _ { t } - \mathbf { C } _ { t } ) ]$ . Then Lemma 6 implies

$$
\| { \bf C } _ { t } { \bf G } \| _ { \mathrm { F } } ^ { 2 } \geq \frac { R } { 2 } \left\| { \bf C } _ { t } \right\| _ { \mathrm { F } } ^ { 2 } , \quad \left\| { \bf P } { \bf C } _ { t } { \bf G } \right\| _ { \mathrm { F } } ^ { 2 } \leq 2 R \left\| { \bf P } { \bf C } _ { t } \right\| _ { \mathrm { F } } ^ { 2 } , \quad \mathrm { T r } \left( { \bf G } ^ { \top } { \bf H } _ { t } { \bf G } \right) \leq 2 R \mathrm { T r } \left( { \bf H } _ { t } \right) ,\tag{15}
$$

all hold with the requisite failure probability. Condition on these events henceforth. Combining the first two events in (15) with (5) then immediately implies our first claim, (13).

Next, recall from (6) and our range on $q _ { t }$ that

$$
\begin{array} { r } { { \mathrm { T r } } \left( \mathbf { H } _ { t } \right) = \mathbb { E } \left\| \mathbf { B } _ { t } - \mathbf { C } _ { t } \right\| _ { \mathrm { F } } ^ { 2 } \leq 2 q _ { t } \left\| \mathbf { C } _ { t } \right\| _ { \mathrm { F } } ^ { 2 } . } \end{array}
$$

Combining with the first and third events in (15) then gives

$$
\begin{array} { r } { \mathbb { E } \left\| \left( \mathbf { B } _ { t } - \mathbf { C } _ { t } \right) \mathbf { G } \right\| _ { \mathrm { F } } ^ { 2 } = \operatorname { T r } \left( \mathbf { G } ^ { \top } \mathbf { H } _ { t } \mathbf { G } \right) \leq 2 R \operatorname { T r } ( \mathbf { H } _ { t } ) \leq 4 q _ { t } R \left\| \mathbf { C } _ { t } \right\| _ { \mathrm { F } } ^ { 2 } \leq 8 q _ { t } \left\| \mathbf { C } _ { t } \mathbf { G } \right\| _ { \mathrm { F } } ^ { 2 } . } \end{array}
$$

Thus, by Markov’s inequality, with probability $\geq \frac { 3 } { 4 }$ over Model 1,

$$
\frac { \| ( \mathbf { B } _ { t } - \mathbf { C } _ { t } ) \mathbf { G } \| _ { \mathrm { F } } } { \| \mathbf { C } _ { t } \mathbf { G } \| _ { \mathrm { F } } } \leq 4 \sqrt { 2 q _ { t } } \leq \frac { 1 } { 2 } .
$$

Thus we have $\mathbf { B } _ { t } \mathbf { G } \neq \mathbf { 0 } _ { d \times R }$ on this event, so again applying (8) (with $( \mathbf { X } , \mathbf { Y } ) \gets ( \mathbf { B } _ { t } \mathbf { G } , \mathbf { C } _ { t } \mathbf { G } ) )$ , and using the definition of $\mathbf { Z } _ { \star , t }$ , concludes the proof of (14). □

Algorithm 2: BoostedSketch ${ \mathrm { O j } } \mathsf { a } ( \{ \mathbf { A } _ { t } , \eta _ { t } \} _ { t \in [ n ] } , R , \tau )$   
1 Input: $\{ \mathbf { A } _ { t } \in \mathbb { R } ^ { d \times d } , \eta _ { t } > 0 \} _ { t \in [ n ] } , R \in \mathbb { N } , \tau > 0$   
2 $\mathbf { G } \gets d \times R$ matrix with i.i.d. $\mathcal { N } ( 0 , 1 )$ entries   
3 for $r \in [ R ]$ do   
4 $\mathbf { Y } _ { r } \gets \mathbf { G }$   
5 end   
6 for $j \in [ \lfloor \frac { n } { R } \rfloor ]$ do   
7 for $r \in [ R ]$ do   
8 1 $\mathbf { Y } _ { r }  ( \mathbf { I } _ { d } + \eta _ { ( j - 1 ) R + r } \mathbf { A } _ { ( j - 1 ) R + r } ) \mathbf { Y } _ { r }$   
9 end   
10 end   
11 for $r \in [ R ]$ do   
12 $\mathbf { Z } _ { r }  \dot { \mathbf { Y } } _ { r } / \| \mathbf { Y } _ { r } \| _ { \mathrm { F } } ,$ or a fixed unit-Frobenius-norm matrix if ${ \bf Y } _ { r } = { \bf 0 } _ { d \times R }$   
13 end   
14 for $r \in [ R ]$ do   
15 $\mathcal { N } _ { r } \overset { \cdot } {  } \{ r ^ { \prime } \in [ R ] : \| \mathbf { Z } _ { r } - \mathbf { Z } _ { r ^ { \prime } } \| _ { \mathrm { F } } \leq 2 \tau \}$   
16 end   
17 i ← any index with $\begin{array} { r } { | \mathcal { N } _ { i } | \ge \frac { R } { 2 } } \end{array}$ , else $i \gets 1$   
18 $\mathbf { w } _ { n } \gets \mathrm { a n y }$ top left singular vector of $\mathbf { Z } _ { i }$   
19 Return: ${ \bf w } _ { n }$

We give our full high-probability method in Algorithm 2, which uses a shared Gaussian matrix across all sample blocks, making the population center common to the independent runs. This lets us apply geometric aggregation to boost the guarantee (14). We make two further observations: first, although it is written with a specified horizon $n ,$ Algorithm 2 is an online algorithm, as the iterations do not use knowledge of n. Second, it is implementable using $O ( d R ^ { 2 } ) = O ( d \log ^ { 2 } ( \frac { 1 } { \zeta } ) )$ space, for the eventual $\begin{array} { r } { R = O ( \log ( \frac { 1 } { \zeta } ) ) } \end{array}$ in Theorem 3, by storing only the ${ \bf Y } _ { r }$

We first give the guarantee for a common stepsize schedule across the R runs.

Lemma 8. Under Model 1, let $R , T \in \mathbb { N }$ and run R independent Oja products for T steps with common step sizes $\eta _ { t }$ , using the shared Gaussian initialization, G, and aggregation rule of Algorithm 2. $L e t \gamma \in ( 0 , 1 )$ and $\begin{array} { r } { 0 < \tau < \frac { 1 } { 4 \sqrt { R } } . \ I f \ r _ { 0 } < \eta _ { t } \le \frac { 1 } { \lambda _ { 1 } } } \end{array}$ and

$$
V \sum _ { t = 1 } ^ { T } \eta _ { t } ^ { 2 } \le \frac { \tau ^ { 2 } } { 1 2 8 } , \qquad 4 d \exp \left( - \gamma \lambda _ { 1 } \sum _ { t = 1 } ^ { T } \eta _ { t } \right) \le \tau ^ { 2 } ,
$$

then the aggregated output is a $( \gamma , 1 6 R \tau ^ { 2 } ) – c P C A$ of Σ with probability $\geq 1 - 4 \exp ( - \frac { R } { 7 2 } )$

Proof. Set $n = R T$ and index the update in round j of run r by $\mathbf { A } _ { ( j - 1 ) R + r } .$ , for $j \in [ T ]$ and $r \in [ R ]$ Let P project onto eigenvectors of Σ with eigenvalues below $( 1 - \gamma ) \lambda _ { 1 }$ . Define

$$
\mathbf { C } _ { j } : = ( \mathbf { I } _ { d } + \eta _ { j } \pmb { \Sigma } ) \cdot \cdot \cdot ( \mathbf { I } _ { d } + \eta _ { 1 } \pmb { \Sigma } ) ,
$$

$$
\mathbf { B } _ { j } ^ { ( r ) } : = ( \mathbf { I } _ { d } + \eta _ { j } \mathbf { A } _ { ( j - 1 ) R + r } ) \cdot \cdot \cdot ( \mathbf { I } _ { d } + \eta _ { 1 } \mathbf { A } _ { r } ) ,
$$

and

$$
\mathbf { Z } _ { \star } : = \frac { \mathbf { C } _ { T } \mathbf { G } } { \| \mathbf { C } _ { T } \mathbf { G } \| _ { \mathrm { F } } } , \qquad \mathbf { Z } _ { r } : = \frac { \mathbf { B } _ { T } ^ { ( r ) } \mathbf { G } } { \left\| \mathbf { B } _ { T } ^ { ( r ) } \mathbf { G } \right\| _ { \mathrm { F } } } ,
$$

with the algorithm’s fallback definition when $\mathbf { B } _ { T } ^ { ( r ) } \mathbf { G } \ = \ \mathbf { 0 } _ { d \times R }$ . By Lemma $7$ with $t  T$ and $\begin{array} { r } { q _ { T }  V \sum _ { j \in [ T ] } \eta _ { j } ^ { 2 } \leq \frac { \tau ^ { 2 } } { 1 2 8 } } \end{array}$ , with probability at least $1 - 3 \exp ( - \frac { R } { 1 6 } )$ over ${ \bf G } , \| { \bf P } { \bf Z } _ { \star } \| _ { \mathrm { F } } ^ { 2 } \le \tau ^ { 2 }$ . Further, for each run,

$$
\operatorname* { P r } ( \| \mathbf { Z } _ { \star } - \mathbf { Z } _ { r } \| _ { \mathrm { F } } \leq \tau \mid \mathbf { G } ) \geq \frac { 3 } { 4 } .
$$

Conditioning on this event for $\mathbf { G } ,$ the runs are independent. By a Chernof bound, the set

$$
\mathcal { G } : = \{ r \in [ R ] \mid \Vert \mathbf { Z } _ { r } - \mathbf { Z } _ { \star } \Vert _ { \mathrm { F } } \leq \tau \}
$$

has size at least $\frac { 2 R } { 3 }$ with conditional probability $\geq 1 - \exp ( - \frac { R } { 7 2 } )$ . Every such candidate has at least $\frac { 2 R } { 3 }$ neighbors within distance $2 \tau$ . Conversely, any candidate with at least $\textstyle { \frac { R } { 2 } }$ such neighbors has a neighbor in ${ \mathcal { G } } ,$ and is therefore within distance $3 \tau$ of $\mathbf { Z } _ { \gamma }$ <sub>⋆</sub> by the triangle inequality.

Let $\widehat { \mathbf { Z } } : = \mathbf { Z } _ { i }$ be the matrix selected in Line 17. On the preceding events, which hold with probability $\begin{array} { r } { \ge 1 - 3 \exp ( - \frac { R } { 1 6 } ) - \exp ( - \frac { R } { 7 2 } ) \ge 1 - 4 \exp ( - \frac { R } { 7 2 } ) } \end{array}$

$$
\left\| \mathbf { P } \widehat { \mathbf { Z } } \right\| _ { \mathrm { F } } \leq \left\| \mathbf { P } \mathbf { Z } _ { \star } \right\| _ { \mathrm { F } } + \left\| \widehat { \mathbf { Z } } - \mathbf { Z } _ { \star } \right\| _ { \mathrm { F } } \leq 4 \tau .
$$

Let ${ \bf w } _ { n }$ be a top left singular vector of $\widehat { \mathbf { Z } } .$ Since $\| \widehat { \mathbf Z } \| _ { \mathrm { F } } = 1$ and rank $( { \widehat { \mathbf { Z } } } ) \leq R$ , its top singular value is at least $R ^ { - 1 / 2 }$ , so $\begin{array} { r } { \| \mathbf { P } \tilde { \mathbf { Z } } \| _ { \mathrm { F } } ^ { 2 } \geq \frac { 1 } { R } \| \mathbf { P } \mathbf { w } _ { n } \| _ { 2 } ^ { 2 } } \end{array}$ by expanding the singular value decomposition. Thus,

$$
\left\| \mathbf { P } \mathbf { w } _ { n } \right\| _ { 2 } ^ { 2 } \leq R \left\| \mathbf { P } \mathbf { \widehat { Z } } \right\| _ { \mathrm { F } } ^ { 2 } \leq 1 6 R \tau ^ { 2 } .
$$

We conclude by proving our main high-probability result, Theorem 3.

Theorem 3. Let $\zeta \in ( 0 , \frac { 1 } { 3 } )$ and $( \gamma , \Delta ) \in ( 0 , 1 ) ^ { 2 }$ . Under Model 1, if

$$
n = \Omega \left( \frac { V } { \lambda _ { 1 } ^ { 2 } \gamma ^ { 2 } \Delta } \log ^ { 2 } \left( \frac { d } { \Delta \zeta } \right) \log ^ { 2 } \left( \frac { 1 } { \zeta } \right) + \frac { \log \left( \frac { d } { \Delta \zeta } \right) \log \left( \frac { 1 } { \zeta } \right) } { \gamma } \right)
$$

for an appropriate constant, then there exist choices of the inputs $R , \tau , \{ \eta _ { t } \} _ { t \geq 1 }$ , such that the output of Algorithm $\mathcal { Q }$ is a $( \gamma , \Delta ) { - } c P C A$ of Σ with probability $\geq 1 - \zeta$

Proof. Throughout the proof, for suficiently large $C > 0$ and small $c > 0$ , we take

$$
R : = \left\lceil C \log \left( \frac { 1 } { \zeta } \right) \right\rceil , \quad L : = C \log \left( \frac { d } { \Delta \zeta } \right) , \quad \tau : = c \sqrt { \frac { \Delta } { R } } .
$$

For an iteration $t \in [ n ]$ such that $t = ( j - 1 ) R + r$ for some $r \in [ R ]$ , we also choose

$$
\eta _ { t } : = \frac { L } { \gamma \lambda _ { 1 } ( \beta + j ) } , \mathrm { ~ w h e r e ~ } \beta : = C \left( \frac { L } { \gamma } + \frac { V R L ^ { 2 } } { \gamma ^ { 2 } \lambda _ { 1 } ^ { 2 } \Delta } \right) .
$$

Let $\begin{array} { r } { T : = \ \lfloor \frac { n } { R } \rfloor } \end{array}$ and write $\alpha _ { j } ~ : = ~ \eta _ { ( j - 1 ) R + r }$ for the common stepsize in round $j ~ \in ~ [ T ]$ , which is independent of $r \in [ R ]$ . By the bound on $\beta$ and $n ,$ we ensure $T \geq 4 \beta$ and $\begin{array} { r } { \alpha _ { j } \le \frac { 1 } { \lambda _ { 1 } } } \end{array}$ . Then, the same integral comparison as in the proof of Theorem 1 yields

$$
q _ { T } = V \sum _ { j \in [ T ] } \alpha _ { j } ^ { 2 } \le \frac { V L ^ { 2 } } { \gamma ^ { 2 } \lambda _ { 1 } ^ { 2 } \beta } \le \frac { 1 } { 1 2 8 } , \quad \gamma \lambda _ { 1 } \sum _ { j \in [ T ] } \alpha _ { j } \ge L ,
$$

if C is suficiently large relative to c. These imply

$$
4 d \exp \left( - \gamma \lambda _ { 1 } \sum _ { j \in [ T ] } \alpha _ { j } \right) \leq 4 d e ^ { - L } \leq \tau ^ { 2 } , \quad 8 \sqrt { 2 q _ { T } } \leq c \sqrt { \frac { \Delta } { R } } = \tau .
$$

Applying Lemma 8 to these R runs of T updates, with $\eta _ { j }  \alpha _ { j }$ and radius τ , gives

$$
\| \mathbf { P } \mathbf { w } _ { n } \| _ { 2 } ^ { 2 } \leq 1 6 R \tau ^ { 2 } = 1 6 c ^ { 2 } \Delta \leq \Delta ,
$$

for $c \leq { \frac { 1 } { 4 } }$ , with failure probability at most $4 \exp ( - \frac { R } { 7 2 } ) \leq \zeta$ when C is suficiently large. □

## 6 Energy PCA Guarantees

In this section, we show how to extend the approach of Section 3 to an alternative gap-free notion of PCA often considered in the literature, energy PCA, which asks for a direction capturing nearly the largest possible variance. We use the following definition from $\mathrm { [ J K L ^ { + } 2 4 ] }$

Definition 2 (ePCA). Let $\alpha \in ( 0 , 1 )$ , and let $\pmb { \Sigma } \in \mathbb { S } _ { \succ \mathbf { 0 } } ^ { d \times d }$ . We say that a unit vector $\mathbf { w } \in \mathbb { R } ^ { d }$ is an $\alpha { \mathrm { - e P C A } }$ (energy PCA) of Σ $i f \mathbf { w } ^ { \top } \pmb { \Sigma } \mathbf { w } \geq ( 1 - \alpha ) \lambda _ { 1 } ( \pmb { \Sigma } )$

Lemma 8 of $[ \mathrm { J K L ^ { + } 2 4 } ]$ shows that a $( \gamma , \Delta ) \mathrm { - c P C A }$ is also a $( \gamma + \Delta ) { \mathrm { - e P C A } }$ . Taking $\begin{array} { r } { \gamma = \Delta  \frac { \alpha } { 2 } } \end{array}$ in Theorem 1 therefore gives an $\alpha { \mathrm { - e P C A } }$ , but the resulting sample complexity scales as $\frac { V } { \lambda _ { 1 } ^ { 2 } \alpha ^ { 3 } }$ , which is suboptimal in its dependence on $\textstyle { \frac { 1 } { \alpha } }$ . This conversion bounds the squared projection onto eigenvalues below a single threshold $( 1 - \gamma ) \bar { \lambda _ { 1 } }$ . To avoid the lossy conversion, we will use the following identity that expresses $\begin{array} { r } { 1 - \frac { \mathbf w ^ { \top } \Sigma \mathbf w } { \lambda _ { 1 } } } \end{array}$ as an integral over thresholds $( 1 - u ) \lambda _ { 1 }$

Proposition 2 (Multiscale $\mathrm { c P C A - t o - e P C A } )$ . Let $\pmb { \Sigma } \in \mathbb { S } _ { \geq \mathbf { 0 } } ^ { d \times d }$ with $\lambda _ { 1 } : = \lambda _ { 1 } ( \Sigma ) > 0$ , and let $\mathbf { w } \in \mathbb { R } ^ { d }$ be a unit vector. For every $u \in \mathsf { \Gamma } ( 0 , 1 )$ , let $\mathbf { P } _ { u } \in \mathbb { S } _ { \succ \mathbf { 0 } } ^ { d \times d }$ project onto the eigenvectors of Σ with eigenvalues $< ( 1 - u ) \lambda _ { 1 }$ , and define $\Delta _ { u } : = \| \mathbf { P } _ { u } \mathbf { w } \| _ { 2 } ^ { 2 }$ . Then

$$
\mathbf { I } _ { d } - \frac { \pmb { \Sigma } } { \lambda _ { 1 } } = \int _ { 0 } ^ { 1 } \mathbf { P } _ { u } \mathrm { d } u .\tag{16}
$$

In particular,

$$
1 - \frac { \mathbf { w } ^ { \top } \pmb { \Sigma } \mathbf { w } } { \lambda _ { 1 } } = \int _ { 0 } ^ { 1 } \Delta _ { u } \mathrm { d } u .\tag{17}
$$

Consequently, if $\alpha \in ( 0 , 1 )$ and $\begin{array} { r } { \int _ { 0 } ^ { 1 } \Delta _ { u } \mathrm { d } u \leq \alpha } \end{array}$ , then w is an $\alpha { - } e P C A ~ o f \Sigma$

Proof. Write $\begin{array} { r } { \pmb { \Sigma } = \sum _ { j \in [ d ] } \lambda _ { j } \mathbf { u } _ { j } \mathbf { u } _ { j } ^ { \top } } \end{array}$ . Then since

$$
\mathbf { P } _ { u } = \sum _ { j \in [ d ] } \mathbb { I } _ { \left\{ u < 1 - \frac { \lambda _ { j } } { \lambda _ { 1 } } \right\} } \mathbf { u } _ { j } \mathbf { u } _ { j } ^ { \top } ,
$$

integrating each indicator gives the matrix identity in (16). Taking the quadratic form with w proves the integral identity (17) and the claim. □

We apply Proposition 2 to the output ${ \bf w } _ { n }$ of Algorithm 1, so henceforth $\Delta _ { u } : = \| \mathbf { P } _ { u } \mathbf { w } _ { n } \| _ { 2 } ^ { 2 }$ . Suppose $\begin{array} { r } { \eta _ { t } \le \frac { 1 } { \lambda _ { 1 } } } \end{array}$ for all $t \in [ n ]$ and $V \sum _ { t \in [ n ] } \eta _ { t } ^ { 2 }$ is a suficiently small constant. For each fixed $u \in ( 0 , 1 )$ combining Lemmas 2 and 4 with $\gamma  u$ and fixed constant ζ gives, for a universal constant $C _ { i }$

$$
\Delta _ { u } \leq C \operatorname* { m i n } \left\{ 1 , d \exp \left( - u \lambda _ { 1 } \sum _ { t \in [ n ] } \eta _ { t } \right) + V \sum _ { t \in [ n ] } \eta _ { t } ^ { 2 } \right\} .
$$

These bounds hold with constant probability for each fixed $u ,$ but need not hold simultaneously for all u. Even if the bound held simultaneously for every u, integrating it would only give

$$
\int _ { 0 } ^ { 1 } \Delta _ { u } \mathrm { d } u = O \left( \frac { \log ( e d ) } { \lambda _ { 1 } \sum _ { t \in [ n ] } \eta _ { t } } + V \sum _ { t \in [ n ] } \eta _ { t } ^ { 2 } \right) .\tag{18}
$$

For step sizes $\begin{array} { r } { \eta _ { t } = \frac { L } { n + t } } \end{array}$ , making both terms in the integral bound (18) at most α requires

$$
L = \Omega \left( \frac { \log ( e d ) } { \lambda _ { 1 } \alpha } \right) , \quad n = \Omega \left( \frac { V L ^ { 2 } } { \alpha } \right) = \Omega \left( \frac { V \log ^ { 2 } ( e d ) } { \lambda _ { 1 } ^ { 2 } \alpha ^ { 3 } } \right) ,
$$

which is still suboptimal. The following lemma gives a sharper bound on $\int _ { 0 } ^ { 1 } \Delta _ { u } \mathrm { d } u$

Lemma 9. Under Model 1 with the products $\mathbf { } \mathbf { B } _ { t } , \mathbf { C } _ { t }$ defined in (2), suppose $\begin{array} { r } { 0 < \eta _ { s } \le \frac { 1 } { \lambda _ { 1 } } } \end{array}$ for all $s \in [ n ]$ . For $u \in ( 0 , 1 )$ , define $\Delta _ { u } : = \| \mathbf { P } _ { u } \mathbf { w } _ { n } \| _ { 2 } ^ { 2 }$ for the projector ${ \bf P } _ { u }$ in Proposition 2. There are universal constants $c , C > 0$ such that, if $\begin{array} { r } { V \sum _ { s \in [ n ] } \eta _ { s } ^ { 2 } \le c , } \end{array}$ then, with probab $\begin{array} { r } { \mathit { n l i t y } \ge \frac { 3 } { 4 } } \end{array}$

$$
\int _ { 0 } ^ { 1 } \Delta _ { u } \mathrm { d } u \leq C \left( \frac { \log ( e d ) } { \lambda _ { 1 } \sum _ { s \in [ n ] } \eta _ { s } } + V \sum _ { i \in [ n ] } \eta _ { i } ^ { 2 } \operatorname* { m i n } \left. 1 , \frac { 1 } { \lambda _ { 1 } \sum _ { s = i + 1 } ^ { n } \eta _ { s } } \right. \right) .\tag{19}
$$

Here the minimum for $i = n$ is interpreted as 1.

The proof retains the dependence on u in the bound on $\mathbb { E } \| \mathbf { P } _ { u } \mathbf { B } _ { n } \| _ { \mathrm { F } } ^ { 2 }$ . After division by $\| \mathbf { C } _ { n } \| _ { \mathrm { F } } ^ { 2 }$ , this bound contains, up to a universal constant,

$$
V \sum _ { i \in [ n ] } \eta _ { i } ^ { 2 } \exp \left( - \frac { u \lambda _ { 1 } } 2 \sum _ { s = i + 1 } ^ { n } \eta _ { s } \right) .
$$

Integrating each summand over u gives the factor min $\{ 1 , 1 / ( \lambda _ { 1 } \sum _ { s = i + 1 } ^ { n } \eta _ { s } ) \}$ in (19), up to a universal constant. In Appendix $\mathrm { A } ,$ we provide a proof of Lemma 9 where we integrate the bounds on $\mathbb { E } \| \mathbf { P } _ { u } \mathbf { B } _ { n } \| _ { \mathrm { F } } ^ { 2 }$ over u before applying Markov’s inequality and controlling the normalization of ${ \bf w } _ { n }$ This avoids a simultaneous union bound over all thresholds $u \in ( 0 , 1 )$

We next evaluate the error bound for the schedule used in both this section and Section 7.3.

Lemma 10. Under Model 1, let $a \geq 1 , n \geq 4 a _ { ; }$ and $\begin{array} { r } { \eta _ { t } = \frac { a } { \lambda _ { 1 } ( n / 4 + t ) } } \end{array}$ for $t \in [ n ]$ . There are universal constants $c , C > 0$ such that, $\begin{array} { r } { i f V \sum _ { t \in [ n ] } \eta _ { t } ^ { 2 } \leq c , } \end{array}$ then with probability at least $\frac 3 4$ ,

$$
1 - { \frac { \mathbf { w } _ { n } ^ { \mathsf { T } } \pm \mathbf { w } _ { n } } { \lambda _ { 1 } } } \leq C \left( { \frac { \log ( e d ) } { a } } + { \frac { V a \log ( e a ) } { \lambda _ { 1 } ^ { 2 } n } } \right) .\tag{20}
$$

Proof. Our strategy is to bound the two terms in Lemma 9 for this schedule, then apply Proposition 2 to obtain the ePCA bound. Since $\begin{array} { r } { \eta _ { t } \lambda _ { 1 } \le \frac { 4 a } { n } \le 1 } \end{array}$ , Lemma 9 applies. The schedule satisfies

$$
\lambda _ { 1 } \sum _ { t \in [ n ] } \eta _ { t } \geq \frac { 4 a } { 5 } , \quad \lambda _ { 1 } \sum _ { s = i + 1 } ^ { n } \eta _ { s } \geq \frac { 4 a ( n - i ) } { 5 n } , \quad \eta _ { i } ^ { 2 } \leq \frac { 1 6 a ^ { 2 } } { \lambda _ { 1 } ^ { 2 } n ^ { 2 } } .
$$

The bound on $\lambda _ { 1 } \sum _ { t \in [ n ] } \eta _ { t }$ controls the first term in Lemma 9 by $O ( { \frac { \log ( e d ) } { a } } )$ . To bound the remaining variance sum, we write $k = n - i$ and use the bounds on $\textstyle \lambda _ { 1 } \sum _ { s = i + 1 } ^ { n } \eta _ { s }$ and $\eta _ { i } ^ { 2 }$ :

$$
\sum _ { i \in [ n ] } \eta _ { i } ^ { 2 } \operatorname* { m i n } \left\{ 1 , \frac { 1 } { \lambda _ { 1 } \sum _ { s = i + 1 } ^ { n } \eta _ { s } } \right\} \leq \frac { 1 6 a ^ { 2 } } { \lambda _ { 1 } ^ { 2 } n ^ { 2 } } \left( 1 + \sum _ { k \in [ n - 1 ] } \operatorname* { m i n } \left\{ 1 , \frac { 5 n } { 4 a k } \right\} \right) = O \left( \frac { a \log ( e a ) } { \lambda _ { 1 } ^ { 2 } n } \right) .
$$

Finally, applying Lemma 9 with our choices of $\eta _ { t }$ , and Proposition 2 with $\mathbf { w } \gets \mathbf { w } _ { n }$ , gives with probability at least $\frac 3 4$

$$
1 - \frac { \mathbf { w } _ { n } ^ { \top } \Sigma \mathbf { w } _ { n } } { \lambda _ { 1 } } = \int _ { 0 } ^ { 1 } \Delta _ { u } \mathrm { d } u \leq C \left( \frac { \log ( e d ) } { a } + \frac { V a \log ( e a ) } { \lambda _ { 1 } ^ { 2 } n } \right) .
$$

We therefore obtain the following ePCA guarantee.

Theorem 4. Under Model 1, let $\alpha \in ( 0 , \textstyle { \frac { 1 } { 4 } } )$ . Then if

$$
n = \Omega \left( \frac { V \log ^ { 2 } \left( \frac { d } { \alpha } \right) } { \lambda _ { 1 } ^ { 2 } \alpha ^ { 2 } } + \frac { \log \left( \frac { d } { \alpha } \right) } { \alpha } \right)
$$

for an appropriate constant, then there exists $a \in \mathbb { R } _ { > 0 }$ such that taking $\begin{array} { r } { \eta _ { t } = \frac { a } { \lambda _ { 1 } ( n / 4 + t ) } } \end{array}$ for $t \in [ n ]$ the output of Algorithm 1 is an $\alpha { - } e P C A$ of Σ with probability $\geq \frac { 3 } { 4 }$

Proof. Let $\begin{array} { r } { L : = \log ( \frac { d } { \alpha } ) } \end{array}$ and take $\begin{array} { r } { a = \frac { C _ { 1 } L } { \alpha } } \end{array}$ for a suficiently large universal constant $C _ { 1 }$ . Then our assumptions ensure $n \geq$ 4a and

$$
V \sum _ { t \in [ n ] } \eta _ { t } ^ { 2 } \leq \frac { 4 V a ^ { 2 } } { \lambda _ { 1 } ^ { 2 } n } \leq c ,
$$

where c is the constant in Lemma 10. Applying Lemma 10 with $a \gets \frac { C _ { 1 } L } { \alpha }$ and using $\log ( e a ) = O ( L )$ gives, with probability at least $\frac 3 4$

$$
1 - \frac { \mathbf { w } _ { n } ^ { \top } \pmb { \Sigma } \mathbf { w } _ { n } } { \lambda _ { 1 } } = O \left( \frac { L } { a } + \frac { V a L } { \lambda _ { 1 } ^ { 2 } n } \right) = O \left( \frac { \alpha } { C _ { 1 } } + \frac { C _ { 1 } \alpha } { C } \right) .
$$

Choosing $C _ { 1 }$ and then C suficiently large makes the error at most $\alpha$

Although Theorem 4 is stated with a constant failure probabili $\mathrm { t y , }$ it is straightforward to use holdout samples to reduce its failure probability.

Corollary 1. Let $( \zeta , \alpha ) \in ( 0 , \frac { 1 } { 4 } ) ^ { 2 }$ . Under Model 1, if

$$
n = \Omega \left( \frac { V \log ^ { 2 } \left( \frac { d } { \alpha } \right) \log ( \frac { 1 } { \zeta } ) } { \lambda _ { 1 } ^ { 2 } \alpha ^ { 2 } } + \frac { \log \left( \frac { d } { \alpha } \right) \log ( \frac { 1 } { \zeta } ) } { \alpha } \right)
$$

for an appropriate constant, we can obtain an α-ePCA of Σ with probability $\geq 1 - \zeta$

Proof. Fix some unit vector $\mathbf { w } \in \mathbb { R } ^ { d }$ . We claim that we can estimate its quadratic form $\mathbf { w } ^ { \top } \pmb { \Sigma }$ w up to additive error $\frac { \alpha \lambda _ { 1 } } { 4 }$ using $O \big ( \textstyle { \frac { V } { \alpha ^ { 2 } \lambda _ { 1 } ^ { 2 } } } \big )$ holdout samples. To see this, under Model 1, a single sample quadratic form $\mathbf { w } ^ { \top } \mathbf { A } _ { t } \mathbf { w }$ is unbiased for $\mathbf { w } ^ { \top } \pmb { \Sigma } \mathbf { w }$ , and has variance at most $V { : }$

$$
\begin{array} { r } { \mathbb { E } \left[ \left( \mathbf { w } ^ { \top } \left( \mathbf { A } _ { t } - \pmb { \Sigma } \right) \mathbf { w } \right) ^ { 2 } \right] \leq \mathbb { E } \left\| \left( \mathbf { A } _ { t } - \pmb { \Sigma } \right) \mathbf { w } \right\| _ { 2 } ^ { 2 } \leq V . } \end{array}
$$

Thus, averaging independent estimates and applying Chebyshev’s inequality gives the claim.

Now, calling Theorem $4 \ O ( \log { \frac { 1 } { \zeta } } )$ times independently with $\alpha  \frac { \alpha } { 2 }$ implies that with probability $\geq 1 - \frac { \zeta } { 2 }$ , at least one of the outputs will be an ${ \frac { \alpha } { 2 } } { \mathrm { - e P C A } }$ . Taking the median of $\begin{array} { r } { O ( \log { \frac { 1 } { \zeta } } ) } \end{array}$ estimates of the quadratic form $\mathbf { w } ^ { \top } \pmb { \Sigma } \mathbf { w }$ obtained by each output w, using independent holdout samples, then yields the unit vector with largest quadratic form up to additive error $\frac { \alpha \lambda _ { 1 } } { 2 }$ , concluding the proof. Note that the same holdout samples are simultaneously accurate for each output by applying independence and taking a union bound, and do not dominate the stated sample complexity. □

Matching lower bound. We briefly conclude the section by showing a matching lower bound, up to logarithmic factors, by appealing to Theorem 2.

Corollary 2. Fix $d \geq 2$ and any choice of $\lambda _ { 1 } > 0 , V > 0 , \alpha \in ( 0 , \frac { 1 } { 3 2 } )$ . There is no algorithm A that takes as input $\{ \mathbf { A } _ { i } \} _ { i \in [ n ] }$ from Model 1, and outputs an α-ePCA of Σ with $\begin{array} { r } { p r o b a b i l i t y \ge \frac { 2 } { 3 } } \end{array}$ , even assuming that $\mathbf { A } _ { i } \in \mathbb { S } _ { \succeq \mathbf { 0 } } ^ { d \times d }$ for all $i \in [ n ]$ , unless for an appropriate constant,

$$
n = \Omega \left( { \frac { V } { \lambda _ { 1 } ^ { 2 } \alpha ^ { 2 } } } \right) .
$$

Proof. By Lemma $7 , \ \mathrm { [ J K L ^ { + } 2 4 ] }$ with $k = 1$ , an $\alpha { \mathrm { - e P C A } }$ of Σ is also a $( \gamma , \frac { \alpha } { \gamma } ) \mathrm { - c P C A }$ for every $\gamma \in ( \alpha , 1 )$ , in the notation of Definition 1. Then, taking $\gamma $ 8α and $\Delta  { \frac { 1 } { 8 } }$ in Theorem 2 gives

$$
n = \Omega \left( \frac { V } { \lambda _ { 1 } ^ { 2 } ( 8 \alpha ) ^ { 2 } \cdot \frac { 1 } { 8 } } \right) = \Omega \left( \frac { V } { \lambda _ { 1 } ^ { 2 } \alpha ^ { 2 } } \right) .
$$

## 7 Application to Diferentially Private PCA

In this section, we give our application to diferentially private PCA. Given i.i.d. sub-Gaussian samples with covariance Σ, we seek a cPCA or ePCA subject to the following privacy guarantee.

Definition 3 (Diferential privacy). We say that a randomized algorithm $\mathcal { A } : ( \mathbb { R } ^ { d } ) ^ { n }  \Omega$ is $( \varepsilon , \delta )$ diferentially private if for all measurable ${ \mathcal { E } } \subseteq \Omega$ , and all $S , S ^ { \prime } \in ( \mathbb { R } ^ { d } ) ^ { n }$ difering in one entry,

$$
\operatorname* { P r } [ A ( S ) \in { \mathcal { E } } ] \leq \exp ( \varepsilon ) \operatorname* { P r } [ A ( S ^ { \prime } ) \in { \mathcal { E } } ] + \delta .
$$

Our utility analysis holds under the following assumption on the dataset.

Definition 4 (ν-sub-Gaussianity). A distribution D on $\mathbb { R } ^ { d }$ is ν-sub-Gaussian if, for every $\mathbf { v } \in \mathbb { R } ^ { d }$ 2

$$
\mathbb { E } _ { \mathbf { x } \sim \mathcal { D } } \left[ \exp \left( \left. \mathbf { x } - \mathbb { E } \mathbf { x } , \mathbf { v } \right. \right) \right] \leq \exp \left( \frac { \nu ^ { 2 } \left. \mathbf { v } \right. _ { 2 } ^ { 2 } } { 2 } \right) .
$$

Model 3. The samples $\mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { n }$ are drawn i.i.d. from a mean-zero, ν-sub-Gaussian distribution with covariance $\pmb { \Sigma } \in \mathbb { S } _ { \succ \mathbf { 0 } } ^ { d \times d }$ and $\lambda _ { 1 } = \left. \pmb { \Sigma } \right. _ { \mathrm { o p } } > 0$

Throughout this section, we treat $n , \nu , \lambda _ { 1 }$ as public parameters, fixed independently of the dataset. As is standard for statistical DP algorithms, our privacy guarantee will hold regardless of the input dataset, and our utility guarantee will hold assuming that the dataset follows Model 3.

We next state our main algorithm in Algorithm 3, which is patterned of of the DP-PCA algorithm of [LKJO22]. After clipping the input dataset, the algorithm simply adds an appropriate Gaussian perturbation to each iterate of an empirical power method, which we show can be cast as an instance of Model 1. One major diference between Algorithm 3 and the variant in [LKJO22] is that we do not subdivide our dataset into minibatches, and instead use full-batch iterations; this diference ends up shaving a roughly $\gamma ^ { - 1 / 2 }$ factor from our final sample complexity.

## 7.1 Privacy

We next prove that Algorithm 3 satisfies $( \varepsilon , \delta ) – \mathrm { D P }$ when $\sigma$ in Line 15 is appropriately chosen. Our proof is standard, and proceeds via Rényi $D P ,$ an alternative privacy accounting strategy that is particularly well-suited to the Gaussian mechanism. For brevity, we defer background on the Gaussian mechanism to Appendix A of [DR14], and background on Rényi DP to [Mir17].

Lemma 11. For $\varepsilon \in ( 0 , 1 ]$ and $\delta \in ( 0 , \frac { 1 } { 3 } )$ , the output of Algorithm 3 is $( \varepsilon , \delta ) \ – D P \ i f$

$$
\sigma ^ { 2 } \geq { \frac { 1 2 R T R _ { 1 } R _ { 2 } } { n ^ { 2 } \varepsilon ^ { 2 } } } \log \left( { \frac { 1 } { \delta } } \right) .\tag{21}
$$

Moreover, every U<sub>r</sub> on Line 8 is $( \varepsilon , \delta ) \ – D P$ for all $r \in [ R ]$ , at every iteration $t \in [ T ]$

Proof. The algorithm only accesses the dataset $\{ \mathbf { x } _ { i } \} _ { i \in [ n ] }$ in the nested for loops from Lines 6 to 18. Fix the initialization and preceding noisy answers at the beginning of one loop, indexed by $t \in [ T ]$ and $r \in [ R ]$ , so that ${ \bf Y } _ { r }$ and $\mathbf { U } _ { r }$ are fixed. Each clipped summand to $\mathbf { Q } _ { r , t }$ satisfies

$$
\left\| \mathbf { z } _ { i } \mathbf { z } _ { i } ^ { \top } \mathbf { U } _ { r } \right\| _ { \mathrm { F } } = \left\| \mathbf { z } _ { i } \right\| _ { 2 } \left\| \mathbf { U } _ { r } ^ { \top } \mathbf { z } _ { i } \right\| _ { 2 } \leq \sqrt { R _ { 1 } R _ { 2 } } ,
$$

Algorithm 3: PrivBoostedSketch $\mathsf { O j a } \big ( \{ \mathbf { x } _ { i } \} _ { i \in [ n ] } , \{ \eta _ { t } \} _ { t \in [ T ] } , R , T , R _ { 1 } , R _ { 2 } , \tau , \sigma \big )$   
1 Input: $\begin{array} { r } { \{ \mathbf { x } _ { i } \in \mathbb { R } ^ { d } \} _ { i \in [ n ] } , \ \{ \eta _ { t } > 0 \} _ { t \in [ T ] } , \ ( R , T ) \in \mathbb { N } ^ { 2 } , \ ( R _ { 1 } , R _ { 2 } , \tau , \sigma ) \in \mathbb { R } _ { > 0 } ^ { 4 } } \end{array}$   
2 $\mathbf { G } \gets d \times R$ matrix with i.i.d. $\mathcal { N } ( 0 , 1 )$ entries   
3 for $r \in [ R ]$ do   
4 $\mathbf { Y } _ { r } \gets \mathbf { G }$   
5 end   
6 for $t \in [ T ]$ do   
7 for $r \in [ R ]$ do   
8 $\mathbf { U } _ { r } \mathbf { D } _ { r } \mathbf { V } _ { r } ^ { \top }$ ← compact SVD of $\mathbf { Y } _ { r } ,$ with $s _ { r } : = \mathrm { r a n k } ( \mathbf Y _ { r } )$   
9 $\mathbf { Q } _ { r , t } \gets \mathbf { 0 } _ { d \times s _ { r } }$   
10 for $i \in [ n ]$ do   
$/ /$ Use multiplier 1 if a denominator in either clipping step is zero.   
11 $\mathbf { s } _ { i } \gets \mathbf { x } _ { i } \operatorname* { m i n } \{ 1 , \sqrt { R _ { 1 } } / \left. \mathbf { x } _ { i } \right. _ { 2 } \}$   
12 $\mathbf { z } _ { i }  \mathbf { s } _ { i }$ min $\{ 1 , \sqrt { R _ { 2 } } / \left. \mathbf { U } _ { r } ^ { \top } \mathbf { s } _ { i } \right. _ { 2 } \}$   
13 $\begin{array} { r } { \mathbf Q _ { r , t }  \mathbf Q _ { r , t } + \frac { 1 } { n } \mathbf z _ { i } ( \mathbf z _ { i } ^ { \top } \mathbf U _ { r } ) } \end{array}$   
14 end   
15 $\mathbf { H } _ { r , t } \gets d \times s _ { r }$ matrix with i.i.d. ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ entries   
16 $\mathbf { Y } _ { r } \gets \mathbf { Y } _ { r } + \eta _ { t } ( \mathbf { Q } _ { r , t } + \mathbf { H } _ { r , t } ) \mathbf { D } _ { r } \mathbf { V } _ { r } ^ { \top }$   
17 end   
18 end   
19 for $r \in [ R ]$ do   
20 $\mathbf { Z } _ { r }  \dot { \mathbf { Y } } _ { r } / \| \mathbf { Y } _ { r } \| _ { \mathrm { F } } ,$ or a fixed unit-Frobenius-norm matrix if ${ \bf Y } _ { r } = { \bf 0 } _ { d \times R }$   
21 end   
22 for $r \in [ R ]$ do   
23 $\mathcal { N } _ { r } \stackrel {  } {  } \{ r ^ { \prime } \in [ R ] : \| \mathbf { Z } _ { r } - \mathbf { Z } _ { r ^ { \prime } } \| _ { \mathrm { F } } \leq 2 \tau \}$   
24 end   
25 $i \gets$ any index with $| { \mathcal { N } } _ { i } | \geq R / 2 .$ , else $i \gets 1$   
26 $\mathbf { w } _ { n } \gets \mathrm { a n y }$ top left singular vector of $\mathbf { Z } _ { i }$   
27 Return: ${ \bf w } _ { n }$

and $\mathbf { z } _ { i }$ is a deterministic function of $\mathbf { U } _ { r }$ and $\mathbf { x } _ { i } ,$ so replacing one sample $\mathbf { x } _ { i }$ changes $\mathbf { Q } _ { r , t }$ in Frobenius norm by $\textstyle \leq s : = { \frac { 2 } { n } } { \sqrt { R _ { 1 } R _ { 2 } } }$ . Thus, Proposition 7 and Corollary 3 of [Mir17] show that the noisy answer $\mathbf { Q } _ { r , t } + \mathbf { H } _ { r , t } { \mathrm { ~ i s ~ } } ( p , \frac { p s ^ { 2 } } { 2 \sigma ^ { 2 } } ) { \mathrm { - R D P } }$ , and ${ \bf Y } _ { r }$ is updated by a deterministic function of this answer and the preceding state. Now RDP composition (Proposition 1, [Mir17]) over RT iterations shows that the transcript of all ${ \bf Y } _ { r }$ is $( p , \rho ) – \mathrm { R D P }$ , where

$$
\rho = { \frac { p s ^ { 2 } R T } { 2 \sigma ^ { 2 } } } .
$$

Finally, taking $p = 1 + \frac { 2 \log ( \frac { 1 } { \delta } ) } { \varepsilon }$ , and using the lower bound on $\sigma ^ { 2 }$ in (21), implies

$$
\frac { \log \left( \frac { 1 } { \delta } \right) } { p - 1 } \leq \frac { \varepsilon } { 2 } , \quad \rho \leq \frac { p s ^ { 2 } n ^ { 2 } \varepsilon ^ { 2 } } { 2 4 R _ { 1 } R _ { 2 } \log \left( \frac { 1 } { \delta } \right) } = \frac { p \varepsilon ^ { 2 } } { 6 \log \left( \frac { 1 } { \delta } \right) } \leq \frac { \varepsilon ^ { 2 } } { 6 \log \left( \frac { 1 } { \delta } \right) } + \frac { \varepsilon } { 3 } \leq \frac { \varepsilon } { 2 } .
$$

Proposition 3 of [Mir17] then shows that the transcript of all of the ${ \bf Y } _ { r }$ is $( \varepsilon , \delta ) – \mathrm { D P } .$ . The privacy of the algorithm’s output and all $\mathbf { U } _ { r }$ then follows, as postprocessings of the transcript. □

## 7.2 Utility

We next give our utility analysis. For convenience, denote the dataset and empirical covariance by

$$
S : = \{ \mathbf { x } _ { i } \} _ { i \in [ n ] } , \quad \widehat { \Sigma } : = \frac { 1 } { n } \sum _ { i \in [ n ] } \mathbf { x } _ { i } \mathbf { x } _ { i } ^ { \top } .
$$

We first show how to couple iterates of Algorithm 3 to an instance of Model 1. To begin, we show that with high probability, the clipping events on Lines 11 and 12 never occur. This step requires using our earlier privacy guarantee to handle a dependence between U and the dataset S.

Lemma 12. Under Model 3, let A be an $( \varepsilon , \delta ) \ – D P$ mechanism whose output $\mathbf { U } = { \mathcal { A } } ( S )$ is an orthonormal matrix with at most R columns. For every $i \in [ n ]$ and $u > 0$

$$
\operatorname* { P r } \left( \left\| \mathbf { U } ^ { \top } \mathbf { x } _ { i } \right\| _ { 2 } ^ { 2 } > 3 \nu ^ { 2 } ( R + u ) \right) \leq \exp ( \varepsilon - u ) + \delta .
$$

Proof. Replace $\mathbf { x } _ { i }$ by an independent copy to form $S ^ { ( i ) }$ , and set $\mathbf { U ^ { \prime } } = \mathcal { A } ( S ^ { ( i ) } )$ . Conditional on $\mathbf { U } ^ { \prime }$ the vector $\mathbf { x } _ { i }$ remains mean-zero and ν-sub-Gaussian. Theorem 2.1 of [HKZ12], applied to $( \mathbf { U } ^ { \prime } ) ^ { \top } { \mathbf { x } } _ { i }$ gives $\operatorname* { P r } ( \left\| ( \mathbf { U } ^ { \prime } ) ^ { \top } \mathbf { x } _ { i } \right\| _ { 2 } ^ { 2 } > 3 \nu ^ { 2 } ( R + u ) ) \leq \exp ( - u )$ . Applying Definition 3 to the neighboring datasets $S , S ^ { ( i ) }$ and averaging over the independent copy gives the claim. □

For a failure probability parameter $\zeta \in ( 0 , 1 )$ , set

$$
R _ { 1 } : = 3 \nu ^ { 2 } \left( d + \log \left( \frac { 8 n } { \zeta } \right) \right) , \qquad R _ { 2 } : = 3 \nu ^ { 2 } \left( R + 2 \log \left( \frac { 1 6 n R T } { \zeta } \right) \right) .\tag{22}
$$

We use the smaller privacy failure parameter

$$
\delta _ { * } : = \operatorname* { m i n } \left\{ \delta , { \frac { \zeta } { 1 6 n R T } } \right\} ,\tag{23}
$$

so that the additive privacy errors can be summed over all sample projections.

Lemma 13. Under Model 3, let $\varepsilon \in ( 0 , 1 ] , \delta \in ( 0 , \frac { 1 } { 3 } )$ , and $\zeta \in ( 0 , 1 )$ . Choose $R _ { 1 } , R _ { 2 }$ as in (22), and suppose $\sigma ^ { 2 }$ satisfies the lower bound in (21) with $\delta  \delta _ { * }$ from (23). There exist matrices

$$
\mathbf { A } _ { ( t - 1 ) R + r } : = \widehat { \pmb { \Sigma } } + \mathbf { G } _ { r , t } ,
$$

where the $\mathbf { G } _ { r , t } \in \mathbb { R } ^ { d \times d }$ have independent ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ entries, such that the updates in Algorithm 3 can be coupled to $\mathbf { Y } _ { r }  ( \mathbf { I } _ { d } + \eta _ { t } \mathbf { A } _ { ( t - 1 ) R + r } ) \mathbf { Y } _ { \eta }$ with probability $\geq 1 - \frac { \zeta } { 4 }$ . Conditional on the dataset, these matrices are i.i.d. with mean $\widehat { \pmb { \Sigma } }$ and satisfy both variance bounds in Model 1 with $V = d \sigma ^ { 2 }$

Proof. We first show that neither clipping step changes any sample, except with probability $\frac { \zeta } { 4 }$ . We then choose the Gaussian matrices so that the private and Oja updates agree whenever no clipping occurs. By Theorem 2.1 of $\mathrm { [ H K Z 1 2 ] }$ and the choice of $R _ { 1 }$

$$
\operatorname* { P r } \left( \exists i \in [ n ] : \| \mathbf { x } _ { i } \| _ { 2 } ^ { 2 } > R _ { 1 } \right) \leq n \exp \left( - \log \left( \frac { 8 n } { \zeta } \right) \right) = \frac { \zeta } { 8 } .
$$

By Lemma 11 with $\delta  \delta _ { * }$ , each subspace $\mathbf { U } _ { r }$ computed by the algorithm is $( \varepsilon , \delta _ { * } ) – \mathrm { D P }$ . For Line 12, apply Lemma 12 with $\begin{array} { r } { u = 2 \log ( \frac { 1 6 n \bar { R } T } { \zeta } ) } \end{array}$ , so that $R _ { 2 } = 3 \nu ^ { 2 } ( R + u )$ . Since $\begin{array} { r } { \varepsilon \le 1 \le \log ( \frac { 1 6 n R T } { \zeta } ) } \end{array}$ , for every $( i , r , t ) \in [ n ] \times [ R ] \times [ T ]$

$$
\mathrm { P r } \left( \left\| \mathbf { U } _ { r } ^ { \top } \mathbf { x } _ { i } \right\| _ { 2 } ^ { 2 } > R _ { 2 } \right) \leq \exp ( \varepsilon - u ) + \delta _ { * } \leq \frac { \zeta } { 1 6 n R T } + \frac { \zeta } { 1 6 n R T } = \frac { \zeta } { 8 n R T } .
$$

These bounds apply to the subspaces computed by the algorithm, including when earlier samples were clipped. A union bound gives total failure probability at most $\begin{array} { r } { \frac { \zeta } { 8 } + n R T \cdot \frac { \zeta } { 8 n R T } = \frac { \zeta } { 4 } } \end{array}$ . On the complementary event, $\mathbf { z } _ { i } = \mathbf { x } _ { i }$ for every sample in every update.

Now draw the $d \times d$ matrices $\mathbf { G } _ { r , t }$ with independent ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ entries, independently across updates and independently of the data and initialization. Set ${ \bf H } _ { r , t } = { \bf G } _ { r , t } { \bf U } _ { r }$ Given the dataset and all preceding updates, $\mathbf { U } _ { r }$ is fixed and orthonormal, so $\mathbf { H } _ { r , t }$ has independent ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ entries, as required by the algorithm. Whenever no clipping occurs, $\mathbf { Q } _ { r , t } = \widehat { \pmb { \Sigma } } \mathbf { U } _ { r }$ , and the SVD identity gives

$$
\mathbf { Y } _ { r } + \eta _ { t } ( \widehat { \pmb { \Sigma } } \mathbf { U } _ { r } + \mathbf { G } _ { r , t } \mathbf { U } _ { r } ) \mathbf { D } _ { r } \mathbf { V } _ { r } ^ { \top } = \big ( \mathbf { I } _ { d } + \eta _ { t } ( \widehat { \pmb { \Sigma } } + \mathbf { G } _ { r , t } ) \big ) \mathbf { Y } _ { r } .
$$

Thus, starting from the same initialization, the private and $\mathrm { O j a }$ iterates agree with probability at least $1 - { \frac { \zeta } { 4 } }$ . Conditional on the dataset, the matrices ${ \bf A } _ { ( t - 1 ) R + r }$ are i.i.d. with mean $\widehat { \pmb { \Sigma } }$ , and

$$
\mathbb { E } [ \mathbf { G } _ { r , t } \mathbf { G } _ { r , t } ^ { \top } ] = \mathbb { E } [ \mathbf { G } _ { r , t } ^ { \top } \mathbf { G } _ { r , t } ] = d \sigma ^ { 2 } \mathbf { I } _ { d } .
$$

This verifies Model 1 with $V = d \sigma ^ { 2 }$ , conditional only on the dataset.

We now combine the coupling with the analysis of Algorithm 2 to obtain a private cPCA guarantee.

Theorem 5. Under Model 3, let $\begin{array} { r } { \varepsilon \in ( 0 , 1 ] , ( \delta , \zeta ) \in ( 0 , \frac { 1 } { 3 } ) ^ { 2 } } \end{array}$ , and $( \gamma , \Delta ) \in ( 0 , 1 ) ^ { 2 }$ . If

$$
n = \Omega \left( \frac { \nu ^ { 4 } \left( d + \log \left( \frac { 1 } { \zeta } \right) \right) } { \lambda _ { 1 } ^ { 2 } \gamma ^ { 2 } \Delta } + \frac { d \nu ^ { 2 } \log \left( \frac { 1 } { \zeta } \right) \log \left( \frac { 2 d } { \Delta } \log \left( \frac { 1 } { \zeta } \right) \right) \log \left( \frac { n } { \zeta } \right) \sqrt { \log \left( \frac { n } { \delta \zeta } \right) } } { \varepsilon \lambda _ { 1 } \gamma \sqrt { \Delta } } \right)\tag{24}
$$

for an appropriate constant, there is a choice of inputs to Algorithm 3 that gives an $( \varepsilon , \delta ) \ – D P$ algorithm returning $a ~ ( \gamma , \Delta ) { - } c P C A$ of Σ with $p r o b a b i l i t y \ge 1 - \zeta$

Proof. We run Algorithm 3 with

$$
R = \left\lceil 7 2 \log \left( \frac { 8 } { \zeta } \right) \right\rceil , \quad L = \log \left( \frac { 2 5 6 d R } { \Delta } \right) , \quad T = \left\lceil \frac { 4 0 L } { \gamma } \right\rceil , \quad \eta _ { t } = \frac { 8 L } { \lambda _ { 1 } ( 1 0 L + \gamma t ) } , \quad \tau = \frac { 1 } { 8 } \sqrt { \frac { \Delta } { R } } .
$$

Choose $R _ { 1 } , R _ { 2 }$ as in (22), $\delta _ { * }$ as in (23), and set

$$
\sigma ^ { 2 } = \frac { 1 2 R T R _ { 1 } R _ { 2 } } { n ^ { 2 } \varepsilon ^ { 2 } } \log \left( \frac { 1 } { \delta _ { * } } \right) .\tag{25}
$$

Privacy. Lemma 11 with $\delta  \delta _ { * }$ <sub>∗</sub> gives $( \varepsilon , \delta _ { * } ) – \mathrm { D P }$ , and hence, $( \varepsilon , \delta ) – \mathrm { D P }$

Utility. Our utility proof strategy is to obtain a cPCA of $\widehat { \pmb { \Sigma } }$ and transfer it to Σ. For this, we first bound $\| \widehat { \pmb { \Sigma } } - \pmb { \Sigma } \| _ { \mathrm { o p } }$ , then couple the private updates to Oja and apply Lemma 8. We combine these guarantees on their common success event. Since $\lambda _ { 1 } \le \nu ^ { 2 }$ , the first term in the sample complexity (24) and Theorem 6.5 of [Wai19] give, with probability $\geq 1 - \frac { \zeta } { 4 }$

$$
\left\| \widehat { \boldsymbol { \Sigma } } - \boldsymbol { \Sigma } \right\| _ { \mathrm { o p } } \leq \frac { \lambda _ { 1 } \gamma \sqrt { \Delta } } { 4 } , \qquad \frac { 3 } { 4 } \lambda _ { 1 } \leq \left\| \widehat { \boldsymbol { \Sigma } } \right\| _ { \mathrm { o p } } \leq \frac { 5 } { 4 } \lambda _ { 1 } .\tag{26}
$$

Let $\mathcal { E } _ { \mathrm { c o v } }$ denote the event that the covariance bounds in (26) hold. As in the proof of Proposition 1, it sufices on this event to obtain a $( \frac { \gamma } { 6 } , \frac { \Delta } { 4 } ) – \mathrm { c P C A }$ of $\widehat { \pmb { \Sigma } }$

Apply Lemma 13 with the chosen $R , T , R _ { 1 } , R _ { 2 } , \sigma$ and failure parameter $\zeta .$ . Let $\mathcal E _ { \mathrm { c p l } }$ denote the event that the two sequences of iterates agree throughout; then $\mathrm { P r } ( \mathcal { E } _ { \mathrm { c p l } } ^ { c } ) \leq \frac { \zeta } { 4 }$ . We analyze these $\mathrm { O j a }$ updates conditional on a dataset satisfying the covariance bounds in (26); they are independent with mean $\widehat { \pmb { \Sigma } }$ and variance $V = d \sigma ^ { 2 }$

To obtain the required cPCA of $\widehat { \Sigma } .$ we check the hypotheses of Lemma 8 for the $R$ coupled runs of $T$ updates with $\bar { ( \Sigma , \gamma , V ) }  ( \widehat { \Sigma } , \frac { \gamma } { 6 } , d \sigma ^ { 2 } )$ , the chosen schedule $\{ \eta _ { t } \} _ { t \in [ T ] }$ , and radius τ . Its step-size condition holds since η<sub>t</sub> $\left\| \hat { \boldsymbol { \Sigma } } \right\| _ { \mathrm { o p } } \leq 1$ . Moreover, our parameter choices give

$$
4 d \exp \left( - \frac { \gamma } { 6 } \left. \widehat { \pmb { \Sigma } } \right. _ { \mathrm { o p } } \sum _ { t \in [ T ] } \eta _ { t } \right) \leq \tau ^ { 2 } , \quad q _ { T } : = d \sigma ^ { 2 } \sum _ { t \in [ T ] } \eta _ { t } ^ { 2 } \leq \frac { \tau ^ { 2 } } { 1 2 8 } .\tag{27}
$$

The exponential bound in (27) uses the choices of $T$ and $\eta _ { t }$ . The bound on $q _ { T }$ follows by substituting $R _ { 1 } , R _ { 2 }$ from (22) and $\sigma$ from (25), and using the second term in the sample bound (24); the sample bound (24) ensures $R , T = O ( n )$ and hence $\begin{array} { r } { \log ( \frac { 1 } { \delta _ { * } } ) = O ( \log \frac { n } { \delta \zeta } ) } \end{array}$

Having verified the hypotheses of Lemma $^ { 8 , }$ we apply it for every fixed $S$ in $\mathcal { E } _ { \mathrm { c o v } }$ to get

$$
\big \| \mathbf { P w } _ { R T } \big \| _ { 2 } ^ { 2 } \leq 1 6 R \tau ^ { 2 } = \frac { \Delta } { 4 }
$$

with failure probability at most 4 exp $\begin{array} { r } { ( - \frac { R } { 7 2 } ) \le \frac { \zeta } { 2 } } \end{array}$ , where P projects onto eigenvectors of $\widehat { \pmb { \Sigma } }$ with eigenvalues below $( 1 - \frac { \gamma } { 6 } ) \| \widehat { \pmb { \Sigma } } \| _ { \mathrm { o p } }$ . Thus, if $\mathcal { E } _ { \mathrm { O j a } }$ denotes the event that the coupled $\mathrm { O j a }$ output is a $( \frac { \gamma } { 6 } , \frac { \Delta } { 4 } ) – \mathrm { c P C A }$ of $\widehat { \Sigma }$ , then $\begin{array} { r } { \operatorname* { P r } ( \mathcal { E } _ { \mathrm { { O j a } } } ^ { c } \mid S ) \le \frac { \zeta } { 2 } } \end{array}$

We have thus obtained the required cPCA of $\widehat { \Sigma }$ . On $\mathcal { E } _ { \mathrm { c o v } } \cap \mathcal { E } _ { \mathrm { c p l } } \cap \mathcal { E } _ { \mathrm { O j a } }$ , the private output agrees with this Oja output, and the covariance bounds in (26), via the proof of Proposition 1, make it a $( \gamma , \Delta ) \mathrm { - c P C A }$ of Σ. The proof follows by noting that

$$
\operatorname* { P r } ( \mathcal { E } _ { \mathrm { c o v } } ^ { c } ) + \operatorname* { P r } ( \mathcal { E } _ { \mathrm { c p l } } ^ { c } ) + \operatorname* { P r } ( \mathcal { E } _ { \mathrm { O j a } } ^ { c } \cap \mathcal { E } _ { \mathrm { c o v } } ) \leq \frac { \zeta } { 4 } + \frac { \zeta } { 4 } + \frac { \zeta } { 2 } = \zeta .
$$

## 7.3 Private energy PCA

We conclude by giving an analogous private ePCA guarantee for Oja’s algorithm.

Theorem 6. Under Model 3, let $\alpha \in ( 0 , \frac { 1 } { 4 } ) , \varepsilon \in ( 0 , 1 ]$ , and $\delta \in ( 0 , \frac { 1 } { 3 } )$ . If

$$
n = \Omega \left( \frac { \nu ^ { 4 } d } { \lambda _ { 1 } ^ { 2 } \alpha ^ { 2 } } + \frac { d \nu ^ { 2 } \log ^ { 2 } \left( \frac { n d } { \alpha } \right) \sqrt { \log \left( \frac { n d } { \alpha \delta } \right) } } { \varepsilon \lambda _ { 1 } \alpha } \right)\tag{28}
$$

for an appropriate constant, there are choices of the inputs to Algorithm 3 that give an $( \varepsilon , \delta ) \ – D P$ algorithm returning an $\alpha { - } e P C A$ of Σ with probability $\geq \frac { 2 } { 3 }$

Proof. We run Algorithm 3 with $R = \tau = 1$ and

$$
a \gets \frac { C _ { 3 } \log \left( \frac { d } { \alpha } \right) } { \alpha } , \quad T \gets \lceil C _ { 4 } a \rceil , \quad \eta _ { t } \gets \frac { a } { \lambda _ { 1 } ( T / 4 + t ) } ,
$$

for suficiently large universal constants $C _ { 3 } , C _ { 4 }$ . We choose $R _ { 1 } , R _ { 2 }$ from (22) with $\begin{array} { r } { \zeta = \frac { 1 } { 1 2 } , } \end{array}$ set $\begin{array} { r } { \delta _ { * } = \operatorname* { m i n } \{ \delta , \frac { 1 } { 1 9 2 n T } \} } \end{array}$ as in (23), and choose σ as in (21) with $\delta  \delta _ { * }$ . The privacy proof is identical to Theorem 5, so we focus on the utility proof.

First, Lemma 13 with $( R , \zeta ) \gets ( 1 , \frac { 1 } { 1 2 } )$ gives coupled Oja updates with mean $\widehat { \pmb { \Sigma } }$ and variance $V = d \sigma ^ { 2 }$ . Let $\mathcal E _ { \mathrm { c p l } }$ denote the event that Lemma 13 succeeds; then $\begin{array} { r } { \mathrm { P r } ( \mathcal { E } _ { \mathrm { c p l } } ^ { c } ) \leq \frac { 1 } { 4 8 } } \end{array}$

Next, let $\mathcal { E } _ { \mathrm { c o v } }$ denote the event $\begin{array} { r } { \| \widehat { \pmb { \Sigma } } - \pmb { \Sigma } \| _ { \mathrm { o p } } \le \frac { \alpha \lambda _ { 1 } } { 8 } } \end{array}$ . The first term in (28) and Theorem 6.5 of [Wai19] give $\begin{array} { r } { \mathrm { P r } ( \mathcal { E } _ { \mathrm { c o v } } ^ { c } ) \leq \frac { 1 } { 4 8 } } \end{array}$ . Fix a dataset S in $\mathcal { E } _ { \mathrm { c o v } }$ and analyze the Oja run conditional only on S.

To apply Lemma 10 to $\widehat { \Sigma } ,$ we express the step sizes using its top eigenvalue. Writing $\widehat { \lambda } _ { 1 } : = \| \widehat { \pmb { \Sigma } } \| _ { \mathrm { o p } }$ and $\begin{array} { r } { a ^ { \prime } : = \frac { a \widehat \lambda _ { 1 } } { \lambda _ { 1 } } } \end{array}$ , the schedule becomes $\begin{array} { r } { \eta _ { t } = \frac { a ^ { \prime } } { \widehat { \lambda } _ { 1 } ( T / 4 + t ) } } \end{array}$ , with $\begin{array} { r } { ( 1 - \frac { \alpha } { 8 } ) a \leq a ^ { \prime } \leq ( 1 + \frac { \alpha } { 8 } ) a } \end{array}$ and $T \geq 4 a ^ { \prime }$

The schedule now has the form required by Lemma 10; it remains to check its variance condition. Using $V = d \sigma ^ { 2 }$ , σ from (21) with $( R , \delta ) \gets ( 1 , \delta _ { * } )$ , and the sample bound (28), we obtain

$$
V \sum _ { t = 1 } ^ { T } \eta _ { t } ^ { 2 } \le \frac { 4 V a ^ { 2 } } { \lambda _ { 1 } ^ { 2 } T } = \frac { 4 8 d R _ { 1 } R _ { 2 } a ^ { 2 } \log { \left( \frac { 1 } { \delta _ { * } } \right) } } { n ^ { 2 } \varepsilon ^ { 2 } \lambda _ { 1 } ^ { 2 } } \le c ,
$$

for a suficiently small universal constant $c > 0$ . We may therefore apply Lemma 10 with $( \pmb { \Sigma } , n , a ) \gets$ $( \widehat \Sigma , T , a ^ { \prime } )$ . For every fixed S in $\mathcal { E } _ { \mathrm { c o v } }$ , it gives, with probability at least $\frac 3 4$ ,

$$
1 - \frac { \mathbf { w } _ { T } ^ { \top } \widehat { \mathbf { \xi } } \mathbf { \mathbf { \xi } } \mathbf { \mathbf { \xi } } \mathbf { \mathbf { \dot { { z } } } } \mathbf { w } _ { T } } { \widehat { \lambda } _ { 1 } } \leq C _ { 0 } \left( \frac { \log ( e d ) } { a ^ { \prime } } + \frac { V a ^ { \prime } \log ( e a ^ { \prime } ) } { \widehat { \lambda } _ { 1 } ^ { 2 } T } \right) \leq \frac { \alpha } { 2 } .
$$

The error bound $\frac { \alpha } { 2 }$ uses $\begin{array} { r } { R _ { 1 } R _ { 2 } = O ( \nu ^ { 4 } d \log ^ { 2 } ( \frac { n d } { \alpha } ) ) } \end{array}$ , $\begin{array} { r } { \log ( \frac { 1 } { \delta _ { * } } ) = O ( \log ( \frac { n d } { \alpha \delta } ) ) } \end{array}$ ), and the sample bound (28), with $C _ { 3 }$ and then C suficiently large. Let $\mathcal { E } _ { \mathrm { { O j a } } }$ denote this $\mathrm { \frac { \alpha } { 2 } \mathrm { - e P C A } }$ guarantee for the coupled Oja output $\mathbf { w } _ { T } ;$ ; then $\textstyle \operatorname* { P r } ( { \mathcal { E } } _ { \mathrm { { O j a } } } ^ { c } \mid S ) \leq { \frac { 1 } { 4 } }$

We have obtained an $\mathrm { \frac { \alpha } { 2 } \mathrm { - e P C A } }$ of $\widehat { \Sigma } ;$ it remains to transfer this guarantee to Σ. On $\mathcal { E } _ { \mathrm { c o v } } \cap \mathcal { E } _ { \mathrm { c p l } } \cap \mathcal { E } _ { \mathrm { O j a } }$ ， coupling and the covariance bound give

$$
\mathbf { w } _ { n } ^ { \top } \pmb { \Sigma } \mathbf { w } _ { n } \geq \left( 1 - \frac { \alpha } { 2 } \right) \widehat { \lambda } _ { 1 } - \frac { \alpha \lambda _ { 1 } } { 8 } \geq ( 1 - \alpha ) \lambda _ { 1 } .
$$

The private output ${ \bf w } _ { n }$ is an $\alpha { \mathrm { - e P C A } }$ of Σ on these events. The failure probability follows from

$$
\operatorname* { P r } ( \mathcal { E } _ { \mathrm { c o v } } ^ { c } ) + \operatorname* { P r } ( \mathcal { E } _ { \mathrm { c p l } } ^ { c } ) + \operatorname* { P r } ( \mathcal { E } _ { \mathrm { O j a } } ^ { c } \cap \mathcal { E } _ { \mathrm { c o v } } ) \leq \frac { 1 } { 4 8 } + \frac { 1 } { 4 8 } + \frac { 1 } { 4 } < \frac { 1 } { 3 } .
$$

We remark that the success probability of Theorem 6 can be boosted using holdout samples, analogously to Corollary 1. For brevity, we omit this extension.

Remark 1 (Gaussian specialization). For Gaussian data, $\nu ^ { 2 } = \lambda _ { 1 }$ , so Theorem 6 matches the sample complexity in Brown’s Conjecture 1.1 [Bro26] up to logarithmic factors when $\lambda _ { 1 }$ is known.

## 8 Experiments

We conclude by providing empirical evaluations of Algorithm 1 (Section 8.1) and Algorithm 3 (Section 8.2), to complement our theoretical results. Code for all experiments can be found here.

## 8.1 Oja’s algorithm

We first evaluate Oja’s algorithm (Algorithm 1) on synthetic streams with nearly tied leading eigenvalues, by comparing it against the top principal component of the empirical covariance. Note that this empirical estimator is not applicable in streaming settings, and serves only as a baseline.

Our experiments study the performance of these two algorithms under the same sample size. In our experiments, we set $d = 5 0$ and set the population mean to $\Sigma = \mathbf { Q } \mathbf { A } \mathbf { Q } ^ { \top }$ , where Λ is a diagonal matrix and Q is a Haar-distributed orthonormal matrix. The first three eigenvalues in Λ are fixed at $( 1 , 0 . 9 9 , 0 . 9 8 )$ , and the remaining eigenvalues are independently drawn from Unif(0, 0.95) and sorted in decreasing order. We generate our matrix stream as

$$
\mathbf { A } _ { t } : = \pmb { \Sigma } + \frac { 1 } { \sqrt { d } } \mathbf { G } _ { t } , \quad [ \mathbf { G } _ { t } ] _ { i j } \overset { \mathrm { i i d } } { \sim } \mathcal { N } ( 0 , 1 ) .
$$

It is straightforward to check that this is an instance of Model 1 with $V = 1$

We set $\gamma = 0 . 0 5$ and measure the cPCA success rate from Definition 1 with $\Delta = 0 . 1$ . For the output ${ \bf w } _ { n }$ and an orthonormal eigenbasis $[ \mathbf { v } _ { i } ] _ { i \in [ d ] }$ , the cPCA error is

$$
\mathrm { e r r } ( \mathbf { w } _ { n } , \pmb { \Sigma } ) : = \sum _ { i : \lambda _ { i } < ( 1 - \gamma ) \lambda _ { 1 } } \left( \mathbf { v } _ { i } ^ { \top } \mathbf { w } _ { n } \right) ^ { 2 } .
$$

![](images/de2b1da93413861502d516f51449784c2d14a21d7725788680559738babb9321.jpg)

![](images/74804b964814a5404d46eb7125213a857c0333ce9d7c121e0f74bcf3be8dfc52.jpg)  
Figure 1: Oja’s algorithm with $\begin{array} { r } { \eta _ { t } = \frac { 1 6 } { 3 0 + t } } \end{array}$ and ofline empirical PCA across 100 fresh paired trials. The left figure shows the cPCA error with 95% Student-t confidence intervals. The right figure shows the fraction of trials with cPCA error at most $\Delta = 0 . 1$ with 95% Wilson intervals.

We implemented Algorithm 1 with

$$
\eta _ { t } = \frac { c } { \lambda _ { 1 } ( \beta + t ) } ,
$$

where $\lambda _ { 1 } = 1$ , and we performed a grid search for the pair of c and $\beta$ that achieved the smallest mean cPCA error, over the choices

$$
c \in \{ 0 . 5 , 1 , 2 , 4 , 8 , 1 6 , 3 2 \} , \quad \beta \in \{ 0 , 1 , 3 , 1 0 , 3 0 , 1 0 0 \} .
$$

The baseline returns a unit eigenvector corresponding to the largest eigenvalue of

$$
\widehat { \boldsymbol { \Sigma } } : = \frac { 1 } { 2 n } \sum _ { t \in [ n ] } \left( \mathbf { A } _ { t } + \mathbf { A } _ { t } ^ { \top } \right) ,
$$

i.e., the symmetrized empirical covariance. As shown in Figure 1, Oja’s algorithm with tuned step sizes achieves results comparable to the baseline, but in the streaming setting.

## 8.2 Private PCA

We next evaluate Algorithm 3 for private PCA on synthetic Gaussian samples, comparing it against the AnalyzeGauss algorithm of [DTTZ14], which clips samples and noises the empirical covariance matrix entrywise. Up to logarithmic factors, our Theorem 5 and Theorem 6 of [DTTZ14] show that the sample complexity of Algorithm 3 and AnalyzeGauss respectively scale $\mathrm { a s ^ { 2 } }$

$$
\approx \frac { d } { \gamma ^ { 2 } \Delta } + \frac { d } { \varepsilon \gamma \sqrt { \Delta } } , \approx \frac { d } { \gamma ^ { 2 } \Delta } + \frac { d ^ { 1 . 5 } } { \varepsilon \gamma \sqrt { \Delta } } .\tag{29}
$$

Observe that unless d is somewhat large or $\gamma , \Delta$ are somewhat small, the identical first term in each of the above expressions dominates. Thus, we expect our algorithm to have improved performance over AnalyzeGauss only in regimes with moderately large d and small $\gamma , \Delta$

In the following experiment, we set $d = 5 0 0 0$ and vary $n \in \{ 1 0 , 2 0 , 3 0 , 5 0 , 1 0 0 \} \times 1 0 ^ { 6 }$ . The population mean $\Sigma = \mathbf { Q } \mathbf { A } \mathbf { Q } ^ { \top }$ follows the exact same distribution as in Section 8.1, i.e., the first three eigenvalues are $( 1 , 0 . 9 9 , 0 . 9 8 )$ , and the remaining eigenvalues are independently drawn from $\mathsf { U n i f } ( 0 , 0 . 9 5 )$ At each sample size, we run 20 independent trials. We use the same cPCA error bound and success criterion of $\Delta = 0 . 1$ as before, and vary $\gamma \in \{ \frac { 1 } { 4 } , \frac { 1 } { 2 0 } \}$ to measure the efect of this parameter. Finally, we set our DP parameters to $\varepsilon = 1$ and $\delta = 1 0 ^ { - 6 }$ , and as hyperparameters to Algorithm 3, we use

$$
R = 3 , T = 5 0 0 , \eta _ { t } = \frac { 3 2 } { 1 0 0 + t } , \tau = \frac { 1 } { 8 } \sqrt { \frac { \Delta } { R } } .
$$

The step sizes $\eta _ { t }$ were picked using another grid search, selected from the same choices as used in Section 8.1. We choose the clipping thresholds from (22) with $\nu = \lambda _ { 1 } = 1$ and $\zeta = 0 . 1$ . The noise scale $\sigma$ is selected according to Lemma 11, which guarantees DP.

We next briefly describe the AnalyzeGauss baseline from [DTTZ14]. We used the same norm clipping threshold $R _ { 1 }$ , i.e., we follow Line 11 of Algorithm 3 to produce clipped samples $\{ \mathbf { s } _ { i } \} _ { i \in [ n ] }$ AnalyzeGauss then outputs a leading eigenvector of $\widehat { \pmb { \Sigma } } _ { \mathrm { c l i p } } + \mathbf { H }$ , where

$$
\widehat { \pmb { \Sigma } } _ { \mathrm { c l i p } } : = \frac { 1 } { n } \sum _ { i \in [ n ] } \mathbf { s } _ { i } \mathbf { s } _ { i } ^ { \top } , \quad \sigma _ { \mathrm { A G } } ^ { 2 } : = \frac { 4 R _ { 1 } ^ { 2 } \log ( 1 . 2 5 / \delta ) } { n ^ { 2 } \varepsilon ^ { 2 } } ,
$$

and H is a symmetric matrix with the upper triangle sampled i.i.d. from $\mathcal { N } ( 0 , \sigma _ { \mathrm { A G } } ^ { 2 } )$

In Figure 2, we show that Algorithm 3 achieves lower mean cPCA error than AnalyzeGauss under the given parameters. As expected from (29), Algorithm 3 performs better when $\gamma$ is smaller. This improvement becomes less drastic when n is very large, because rearranging (29) shows that

$$
\Delta \approx \operatorname* { m a x } \left( \frac { d } { \gamma ^ { 2 } n } , \left( \frac { d } { \varepsilon \gamma n } \right) ^ { 2 } \right)
$$

is dominated by the first term for large n. In such regimes, our error decay matches AnalyzeGauss.

We also note that, consistently with our theory, this finding appears to require a moderately large dimension to emerge: for example, when $d = 3 0 0 0$ and all other parameter settings remain fixed, AnalyzeGauss achieves lower error than our algorithm.

## Acknowledgments

SK and KT thank Ankit Pensia and Gavin Brown for several insightful discussions on this problem. SK and CY gratefully acknowledge support from the Amazon AI PhD Fellowship. We thank the NSF AI Institute for Foundations of Machine Learning (IFML) for supporting this project, and the Texas Advanced Computing Center (TACC) for providing the computing resources used.

![](images/cb4c488b02565563bffcb7b67d2e41b28f6080cd5be4234db25101cf58d17c07.jpg)

![](images/84cc3df8c166e0cac4ad35f27db7e0c8c6f48af5b31b323bba9baea1053c613b.jpg)

![](images/f19440f95ee64dbf921adc5bf0a04969b8b49f4f1939b340821f6bfc12e1a71c.jpg)

![](images/2bd9884629911534b72d7ebd1be8ac96e02587a12374cd4c397165d9b0c6fdd6.jpg)  
Figure 2: The left two figures show the cPCA error with 95% Student-t confidence intervals. The right two figures show the fraction of trials with cPCA error at most $\Delta = 0 . 1$ with 95% Wilson intervals. The top figures are for $\gamma = 0 . 0 5$ and bottom figures are for $\gamma = 0 . 2 5$

## AI Disclosure

The authors began the line of inquiry in this paper after discovering the connection between gap-free DP PCA and a gap-free Oja’s algorithm in Section 7, and the lack of a gap-free, general rank analysis of Oja’s algorithm. We used ChatGPT 5.5 and 5.6 Pro models to explore approaches for Theorem 1, primarily to aid with strategies for proving Lemma 3, but the final proof strategy was developed by the authors. After completing all of our cPCA results, we learned about the statement of Conjecture 1.1 in [Bro26] (which asked specifically for private ePCA) in personal communications with Gavin Brown. We then discovered the reduction in Proposition 2 in conversations with ChatGPT 5.6 Pro, allowing us to extend our cPCA results to ePCA. The manuscript was written solely by the authors, who take full responsibility for the organization and presentation of all results.

## References

[AZL16] Zeyuan Allen-Zhu and Yuanzhi Li. Even faster SVD decomposition yet without agonizing pain. In Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, pages 974–982, 2016.

[AZL17] Zeyuan Allen-Zhu and Yuanzhi Li. First eficient convergence for streaming k-pca: a global, gap-free, and near-optimal rate. In 2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS), pages 487–492. IEEE, 2017.

[BDF13] Akshay Balsubramani, Sanjoy Dasgupta, and Yoav Freund. The fast convergence of incremental PCA. In Advances in Neural Information Processing Systems 26, 2013.

[BDWY16] Maria-Florina Balcan, Simon Shaolei Du, Yining Wang, and Adams Wei Yu. An improved gap-dependency analysis of the noisy power method. In Proceedings of the 29th Annual Conference on Learning Theory, volume 49 of Proceedings of Machine Learning Research, pages 284–309. PMLR, 2016.

[Bro26] Gavin Brown. Gap-free, computationally eficient private PCA. Open Problems, Workshop on the Intersections of Diferential Privacy and Sublinear Algorithms, TTIC, 2026. Conjecture 1.1, p. 2; July 27–29, 2026.

[CSS13] Kamalika Chaudhuri, Anand D. Sarwate, and Kaushik Sinha. A near-optimal algorithm for diferentially-private principal components. Journal of Machine Learning Research, 14:2905–2943, 2013.

[CXZ24] T. Tony Cai, Dong Xia, and Mengyue Zha. Optimal diferentially private PCA and estimation for spiked covariance matrices. arXiv preprint arXiv:2401.03820, 2024.

[DKPS25] Sanjoy Dasgupta, Syamantak Kumar, Shourya Pandey, and Purnamrita Sarkar. Low precision streaming PCA. In Advances in Neural Information Processing Systems 38, pages 157961–157996. Curran Associates, Inc., 2025.

[DR14] Cynthia Dwork and Aaron Roth. The algorithmic foundations of diferential privacy. Found. Trends Theor. Comput. Sci., 9(3-4):211–407, 2014.

[DTTZ14] Cynthia Dwork, Kunal Talwar, Abhradeep Thakurta, and Li Zhang. Analyze Gauss: Optimal bounds for privacy-preserving principal component analysis. In Proceedings of the 46th Annual ACM Symposium on Theory of Computing, pages 11–20. ACM, 2014.

[GH15] Dan Garber and Elad Hazan. Fast and simple PCA via convex optimization. CoRR, abs/1509.05647, 2015.

[GHJ<sup>+</sup>16] Dan Garber, Elad Hazan, Chi Jin, Sham M. Kakade, Cameron Musco, Praneeth Netrapalli, and Aaron Sidford. Faster eigenvector computation via shift-and-invert preconditioning. In Proceedings of the 33rd International Conference on Machine Learning, ICML 2016, volume 48 of JMLR Workshop and Conference Proceedings, pages 2626– 2634. JMLR.org, 2016.

[HKMN23] Samuel B. Hopkins, Gautam Kamath, Mahbod Majid, and Shyam Narayanan. Robustness implies privacy in statistical estimation. In Proceedings of the 55th Annual ACM Symposium on Theory of Computing, pages 497–506. ACM, 2023.

[HKZ12] Daniel Hsu, Sham M. Kakade, and Tong Zhang. A tail inequality for quadratic forms of subgaussian random vectors. Electronic Communications in Probability, 17(52):1–6, 2012.

[HNWW21] De Huang, Jonathan Niles-Weed, and Rachel Ward. Streaming k-PCA: Eficient guarantees for Oja’s algorithm, beyond rank-one updates. In Proceedings of the Thirty-Fourth Conference on Learning Theory, volume 134 of Proceedings of Machine Learning Research, pages 2463–2498, 2021.

[HP14] Moritz Hardt and Eric Price. The noisy power method: A meta algorithm with applications. In Advances in Neural Information Processing Systems 27, pages 2861–2869, 2014.

[JJK<sup>+</sup>16] Prateek Jain, Chi Jin, Sham M Kakade, Praneeth Netrapalli, and Aaron Sidford. Streaming pca: Matching matrix bernstein and near-optimal finite sample guarantees for oja’s algorithm. In Conference on learning theory, pages 1147–1164. PMLR, 2016.

[JKL<sup>+</sup>24] Arun Jambulapati, Syamantak Kumar, Jerry Li, Shourya Pandey, Ankit Pensia, and Kevin Tian. Black-box k-to-1-pca reductions: Theory and applications. In The Thirty Seventh Annual Conference on Learning Theory, volume 247 of Proceedings of Machine Learning Research, pages 2564–2607. PMLR, 2024.

[KJ25] Minwoo Kim and Sungkyu Jung. Robust and diferentially private principal component analysis. Statistical Analysis and Data Mining: The ASA Data Science Journal, 18(6):e70053, 2025.

[KPS25] Syamantak Kumar, Shourya Pandey, and Purnamrita Sarkar. Beyond sin-squared error: Linear time entrywise uncertainty quantification for streaming PCA. In Proceedings of the Forty-First Conference on Uncertainty in Artificial Intelligence, volume 286 of Proceedings of Machine Learning Research, pages 2396–2430. PMLR, 2025.

[KS23] Syamantak Kumar and Purnamrita Sarkar. Streaming PCA for Markovian data. In Advances in Neural Information Processing Systems 36, pages 64650–64662. Curran Associates, Inc., 2023.

[KS24] Syamantak Kumar and Purnamrita Sarkar. Oja’s algorithm for streaming sparse PCA. In Advances in Neural Information Processing Systems 37, pages 74528–74578. Curran Associates, Inc., 2024.

[Lia23] Xin Liang. On the optimality of Oja’s algorithm for online PCA. Statistics and Computing, 33(3):62, 2023.

[LKJO22] Xiyang Liu, Weihao Kong, Prateek Jain, and Sewoong Oh. DP-PCA: Statistically optimal and diferentially private PCA. In Advances in Neural Information Processing Systems 35, 2022.

[LKO22] Xiyang Liu, Weihao Kong, and Sewoong Oh. Diferential privacy and robust statistics in high dimensions. In Proceedings of the Thirty-Fifth Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 1167–1246. PMLR, 2022.

[LM00] Béatrice Laurent and Pascal Massart. Adaptive estimation of a quadratic functional by model selection. The Annals of Statistics, 28(5):1302–1338, 2000.

[LWLZ18] Chengtao Li, Mengdi Wang, Han Liu, and Tong Zhang. Near-optimal stochastic approximation for online principal component estimation. Mathematical Programming, 167(1):75–97, 2018.

[MCJ13] Ioannis Mitliagkas, Constantine Caramanis, and Prateek Jain. Memory limited, streaming PCA. In Advances in Neural Information Processing Systems 26, 2013.

[Mir17] Ilya Mironov. Rényi diferential privacy. In 2017 IEEE 30th Computer Security Foundations Symposium (CSF), pages 263–275. IEEE, 2017.

[Oja82] Erkki Oja. Simplified neuron model as a principal component analyzer. Journal of Mathematical Biology, 15(3):267–273, 1982.

[Sha16] Ohad Shamir. Convergence of stochastic gradient descent for PCA. In Proceedings of the 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 257–265. PMLR, 2016.

[SOR15] Christopher De Sa, Kunle Olukotun, and Christopher Ré. Global convergence of stochastic gradient descent for some non-convex matrix problems. In Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 2332–2341. PMLR, 2015.

[Tia26] Kevin Tian. CS395T: Continuous Algorithms, Part XI: Low-Rank Approximation. Lecture notes, University of Texas at Austin, 2026.

[Tro15] Joel A Tropp. An introduction to matrix concentration inequalities. Foundations and trends® in machine learning, 8(1-2):1–230, 2015.

[Tsy09] Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. Springer Series in Statistics. Springer, 2009.

[Wai19] Martin J. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint. Cambridge University Press, 2019.

## A Deferred Proofs from Section 6

In this section, we prove Lemma 9, our multiscale cPCA guarantee on Algorithm 1.

Our strategy is to first bound $\mathbb { E } \| \mathbf { P } _ { u } \mathbf { B } _ { n } \| _ { \mathrm { F } } ^ { 2 }$ for each threshold $u ,$ and then give a normalization argument needed to integrate this bound. Throughout, work under Model 1 with the products $\mathbf { B } _ { s } , \mathbf { C } _ { s }$ defined in (2) and $\begin{array} { r } { 0 < \eta _ { s } \le \frac { 1 } { \lambda _ { 1 } } } \end{array}$ . For convenience, we also define

$$
q _ { n } : = V \sum _ { s \in [ n ] } \eta _ { s } ^ { 2 } , \qquad Z _ { s } : = \Vert \mathbf { C } _ { s } \Vert _ { \mathrm { F } } ^ { 2 } , \qquad N _ { s } ( u ) : = \mathbb { E } \Vert \mathbf { P } _ { u } \mathbf { B } _ { s } \Vert _ { \mathrm { F } } ^ { 2 } ,
$$

where ${ \bf P } _ { u }$ is the projector from Proposition 2.

Lemma 14. For every $u \in ( 0 , 1 )$ ,

$$
\frac { N _ { n } ( u ) } { Z _ { n } } \le \exp ( q _ { n } ) \left( \operatorname* { m i n } \left\{ 1 , d ( 1 + q _ { n } ) \exp \left( - \frac { u \lambda _ { 1 } } { 2 } \sum _ { s \in [ n ] } \eta _ { s } \right) \right\} + V \sum _ { i \in [ n ] } \eta _ { i } ^ { 2 } \exp \left( - \frac { u \lambda _ { 1 } } { 2 } \sum _ { s = i + 1 } ^ { n } \eta _ { s } \right) \right) .\tag{30}
$$

Proof. Our proof strategy is to bound the initial contribution and the variance introduced at each update separately. For this, we first expand the second-moment recurrence, then divide by $Z _ { n }$ and bound the two contributions. Finally, we combine these bounds with Lemma 3.

Define $r _ { s } ( u ) : = 1 + \eta _ { s } ( 1 - u ) \lambda _ { 1 }$ . Taking the trace against ${ \bf P } _ { u }$ in the second moment recurrence (4) and using the variance bound in Model 1 then gives

$$
N _ { s } ( u ) \leq r _ { s } ( u ) ^ { 2 } N _ { s - 1 } ( u ) + V \eta _ { s } ^ { 2 } \mathbb { E } \left. \mathbf { B } _ { s - 1 } \right. _ { \mathrm { F } } ^ { 2 } .
$$

Iterating from $N _ { 0 } ( u ) \leq d$ and using Lemma 3 with $t \gets i - 1$ for $i > 1$ (and ${ \bf B } _ { 0 } = { \bf C } _ { 0 } = { \bf I } _ { d }$ for $i = 1 )$ , we obtain

$$
N _ { n } ( u ) \leq \underbrace { d \prod _ { s \in [ n ] } r _ { s } ( u ) ^ { 2 } } _ { \mathrm { i n i t i a l ~ c o n t r i b u t i o n } } + \underbrace { V \exp ( q _ { n } ) \sum _ { i \in [ n ] } \eta _ { i } ^ { 2 } Z _ { i - 1 } \prod _ { s = i + 1 } ^ { n } r _ { s } ( u ) ^ { 2 } } _ { \mathrm { v a r i a n c e ~ t e r m } } .\tag{31}
$$

The initial contribution comes from $N _ { 0 } ( u ) \ \leq \ d ,$ multiplied by the recurrence factors over all n updates. The variance term sums the contribution introduced at each update $i ,$ multiplied by the factors from the subsequent updates. We first bound the initial contribution after dividing by $Z _ { n } .$ by comparing with the product for $\lambda _ { 1 }$ . The assumption $\eta _ { s } \lambda _ { 1 } \leq 1$ gives

$$
\left( \frac { 1 + \eta _ { s } \mu } { 1 + \eta _ { s } \mu ^ { \prime } } \right) ^ { 2 } \leq \exp ( - \eta _ { s } ( \mu ^ { \prime } - \mu ) ) \qquad \mathrm { f o r ~ } 0 \leq \mu \leq \mu ^ { \prime } \leq \lambda _ { 1 } .\tag{32}
$$

Since $\begin{array} { r } { Z _ { n } \ge \prod _ { s \in [ n ] } ( 1 + \eta _ { s } \lambda _ { 1 } ) ^ { 2 } } \end{array}$ , the first term in (31), divided by $Z _ { n }$ , is at most

$$
d \exp \left( - u \lambda _ { 1 } \sum _ { s \in [ n ] } \eta _ { s } \right) .
$$

It remains to bound the variance terms in (31) after division by $Z _ { n }$ . For the ith summand, expand

$$
Z _ { i - 1 } = \sum _ { j \in [ d ] } \prod _ { s < i } ( 1 + \eta _ { s } \lambda _ { j } ) ^ { 2 }
$$

and separate the eigenvalues at $\begin{array} { r } { ( 1 - \frac { u } { 2 } ) \lambda _ { 1 } } \end{array}$ . Above this threshold, we compare the remaining factors with those for the same eigenvalue in $Z _ { n } ;$ below it, we compare all factors with those for $\lambda _ { 1 }$ . In both cases the compared eigenvalues difer by at least $\frac { u \lambda _ { 1 } } { 2 }$ . For $\begin{array} { r } { \lambda _ { j } \ge ( 1 - \frac { u } { 2 } ) \lambda _ { 1 } } \end{array}$ , (32) gives

$$
\prod _ { s < i } ( 1 + \eta _ { s } \lambda _ { j } ) ^ { 2 } \prod _ { s = i + 1 } ^ { n } r _ { s } ( u ) ^ { 2 } \leq \exp \left( - \frac { u \lambda _ { 1 } } { 2 } \sum _ { s = i + 1 } ^ { n } \eta _ { s } \right) \prod _ { s \in [ n ] } ( 1 + \eta _ { s } \lambda _ { j } ) ^ { 2 } .
$$

Summing over $j$ with $\begin{array} { r } { \lambda _ { j } \ge ( 1 - \frac { u } { 2 } ) \lambda _ { 1 } } \end{array}$ bounds their total contribution by $\begin{array} { r } { Z _ { n } \exp ( - \frac { u \lambda _ { 1 } } { 2 } \sum _ { s = i + 1 } ^ { n } \eta _ { s } ) } \end{array}$ For $\begin{array} { r } { \lambda _ { j } < ( 1 - \frac { u } { 2 } ) \lambda _ { 1 } } \end{array}$ , the scalar ratio bound in (32) and $\Pi _ { s \in [ n ] } ( 1 + \eta _ { s } \lambda _ { 1 } ) ^ { 2 } \leq Z _ { n }$ give

$$
\prod _ { s < i } ( 1 + \eta _ { s } \lambda _ { j } ) ^ { 2 } \prod _ { s = i + 1 } ^ { n } r _ { s } ( u ) ^ { 2 } \leq \prod _ { s \in [ n ] } \left( 1 + \eta _ { s } \left( 1 - \frac { u } { 2 } \right) \lambda _ { 1 } \right) ^ { 2 } \leq Z _ { n } \exp \left( - \frac { u \lambda _ { 1 } } { 2 } \sum _ { s \in [ n ] } \eta _ { s } \right) .
$$

There are at most d such $j ,$ so the two ranges together give

$$
\frac { Z _ { i - 1 } \prod _ { s = i + 1 } ^ { n } r _ { s } ( u ) ^ { 2 } } { Z _ { n } } \leq \exp \left( - \frac { u \lambda _ { 1 } } { 2 } \sum _ { s = i + 1 } ^ { n } \eta _ { s } \right) + d \exp \left( - \frac { u \lambda _ { 1 } } { 2 } \sum _ { s \in [ n ] } \eta _ { s } \right) .
$$

Substituting the bounds for both terms into (31) and using $\begin{array} { r } { V \sum _ { i \in [ n ] } \eta _ { i } ^ { 2 } = q _ { n } } \end{array}$ gives the claimed bound without the minimum with 1. To obtain that minimum, apply Lemma 3 with $t \gets n$ and use $\| \mathbf { P } _ { u } \mathbf { B } _ { n } \| _ { \mathrm { F } } \leq \| \mathbf { B } _ { n } \| _ { \mathrm { F } }$ to get $\frac { N _ { n } ( u ) } { Z _ { n } } \leq \exp ( q _ { n } )$ . Combining this bound on $\frac { N _ { n } ( u ) } { Z _ { n } }$ with the bound obtained from (31), using min $\{ 1 , \stackrel { \cdot \cdot } { x } + y \} \leq \operatorname* { m i n } \{ 1 , x \} + y$ for $x , y \geq 0$ proves the claim. □

Lemma 15. There are universal constants $c , C > 0$ such that, $i f q _ { n } \leq c$ , then for any fixed $\mathbf { D } \succeq \mathbf { 0 } _ { d \times d }$ and independent $\mathbf { g } \sim \mathcal { N } ( \mathbf { 0 } _ { d } , \mathbf { I } _ { d } )$ , with probability at least $\frac { 3 } { 4 } , \mathbf { B } _ { n } \mathbf { g } \neq \mathbf { 0 } _ { d }$ and

$$
\frac { \mathbf { g } ^ { \top } \mathbf { B } _ { n } ^ { \top } \mathbf { D B } _ { n } \mathbf { g } } { \| \mathbf { B } _ { n } \mathbf { g } \| _ { 2 } ^ { 2 } } \leq C \frac { \mathbb { E } \mathrm { T r } ( \mathbf { D B } _ { n } \mathbf { B } _ { n } ^ { \top } ) } { Z _ { n } } .
$$

Proof. Our proof strategy is to control the numerator and denominator of the normalized output using bounds on $\mathbf { B } _ { n }$ . We first lower bound $\| \mathbf { B } _ { n } \| _ { \mathrm { F } } ^ { 2 }$ and upper bound $\mathrm { T r } ( { \bf D B } _ { n } { \bf B } _ { n } ^ { \top } )$ ). We then condition on $\mathbf { B } _ { n } ,$ apply the Gaussian quadratic-form bounds, and combine the three events.

Choose c so that $\textstyle \exp ( c ) - 1 \leq { \frac { 1 } { 4 8 } }$ . Since $\mathbb { E } \mathbf { B } _ { n } = \mathbf { C } _ { n }$ , Lemma 3 with $t \gets n$ gives

$$
\begin{array} { r } { \mathbb { E } \left\| \mathbf { B } _ { n } - \mathbf { C } _ { n } \right\| _ { \mathrm { F } } ^ { 2 } = \mathbb { E } \left\| \mathbf { B } _ { n } \right\| _ { \mathrm { F } } ^ { 2 } - Z _ { n } \leq ( \exp ( q _ { n } ) - 1 ) Z _ { n } . } \end{array}
$$

Markov’s inequality gives events

$$
\mathcal E _ { B } : = \left\{ \| { \bf B } _ { n } - { \bf C } _ { n } \| _ { \mathrm { F } } ^ { 2 } \le \frac { Z _ { n } } { 4 } \right\} , \qquad \mathcal E _ { D } : = \left\{ \mathrm { T r } ( { \bf D } { \bf B } _ { n } { \bf B } _ { n } ^ { \top } ) \le 1 2 \mathbb { E } \mathrm { T r } ( { \bf D } { \bf B } _ { n } { \bf B } _ { n } ^ { \top } ) \right\} ,
$$

each with failure probability at most $\textstyle { \frac { 1 } { 1 2 } }$ . Fix $\mathbf { B } _ { n }$ in $\mathcal { E } _ { B } \cap \mathcal { E } _ { D } ;$ then $\begin{array} { r } { \left\| \mathbf { B } _ { n } \right\| _ { \mathrm { F } } ^ { 2 } \geq \frac { Z _ { n } } { 4 } } \end{array}$ , and g remains an independent standard Gaussian.

We have thus obtained the required bounds on $\mathbf { B } _ { n } ;$ it remains to control the ratio of quadratic forms in $\mathbf { g } .$ The proof of Lemma 2 applies with $\mathbf { H } = \mathbf { B } _ { n } ^ { \top } \mathbf { B } _ { n }$ and ${ \bf K } = { \bf B } _ { n } ^ { \top } { \bf D } { \bf B } _ { n }$ . Its Gaussian quadratic-form bounds require only $\mathbf { H } , \mathbf { K } \succeq \mathbf { 0 } _ { d \times d }$ and H $\neq \mathbf { 0 } _ { d \times d }$ . With $\zeta = \textstyle { \frac { 1 } { 1 2 } }$ , it gives an event

$$
\mathcal { E } _ { g } : = \left\{ \| \mathbf { B } _ { n } \mathbf { g } \| _ { 2 } ^ { 2 } > 0 , \quad \frac { \mathbf { g } ^ { \top } \mathbf { B } _ { n } ^ { \top } \mathbf { D } \mathbf { B } _ { n } \mathbf { g } } { \| \mathbf { B } _ { n } \mathbf { g } \| _ { 2 } ^ { 2 } } \leq C _ { 0 } \frac { \mathrm { T r } ( \mathbf { D } \mathbf { B } _ { n } \mathbf { B } _ { n } ^ { \top } ) } { \| \mathbf { B } _ { n } \| _ { \mathrm { F } } ^ { 2 } } \right\} ,
$$

with conditional failure probability at most $\textstyle { \frac { 1 } { 1 2 } }$ , for a universal constant $C _ { 0 }$ . On $\mathcal { E } _ { B } \cap \mathcal { E } _ { D } \cap \mathcal { E } _ { g }$ , the claimed bound holds with $C = 4 8 C _ { 0 }$ . The total failure probability is at most $\textstyle 3 \cdot { \frac { 1 } { 1 2 } } = { \frac { 1 } { 4 } }$ □

Proof of Lemma 9. Our strategy is to bound the integral by an expected trace using Lemma 15, then estimate this trace using Lemma 14. For this, we first write the integral as a quadratic form. Set $\begin{array} { r } { \mathbf { D } : = \mathbf { I } _ { d } - \frac { \pmb { \Sigma } } { \lambda _ { 1 } } \succeq \mathbf { 0 } _ { d \times d } } \end{array}$ . Proposition 2, applied to $\Sigma ,$ , gives $\begin{array} { r } { \mathbf { D } = \int _ { 0 } ^ { 1 } \mathbf { P } _ { u } } \end{array}$ du. On the event $\mathbf { B } _ { n } \mathbf { g } \neq \mathbf { 0 } _ { d }$ write $\mathbf { w } _ { n } = \mathbf { B } _ { n } \mathbf { g } / \left. \mathbf { B } _ { n } \mathbf { g } \right. _ { 2 }$ and $\Delta _ { u } = \| \mathbf { P } _ { u } \mathbf { w } _ { n } \| _ { 2 } ^ { 2 }$ . Then

$$
\int _ { 0 } ^ { 1 } \Delta _ { u } \mathrm { d } u = \mathbf { w } _ { n } ^ { \top } \mathbf { D } \mathbf { w } _ { n } = \frac { \mathbf { g } ^ { \top } \mathbf { B } _ { n } ^ { \top } \mathbf { D } \mathbf { B } _ { n } \mathbf { g } } { \left\| \mathbf { B } _ { n } \mathbf { g } \right\| _ { 2 } ^ { 2 } } .
$$

Applying Lemma 15 with $\begin{array} { r } { \mathbf { D } \gets \mathbf { I } _ { d } - \frac { \Sigma } { \lambda _ { 1 } } } \end{array}$ and using linearity of trace and expectation therefore gives, with probability at least $\frac 3 4$

$$
\int _ { 0 } ^ { 1 } \Delta _ { u } \mathrm { d } u \leq C \frac { \mathrm { E T r } ( \mathbf { D B } _ { n } \mathbf { B } _ { n } ^ { \top } ) } { Z _ { n } } = C \int _ { 0 } ^ { 1 } \frac { N _ { n } ( u ) } { Z _ { n } } \mathrm { d } u .\tag{33}
$$

We have reduced the desired bound to the integral of $\frac { N _ { n } ( u ) } { Z _ { n } }$ in (33). It remains to apply Lemma 14 and integrate its two terms. For $A \geq 1$ and $x > 0$ , direct integration gives

$$
\int _ { 0 } ^ { 1 } \operatorname* { m i n } \left\{ 1 , A \exp \left( - { \frac { u x } { 2 } } \right) \right\} { \mathrm { d } } u \leq { \frac { 2 ( 1 + \log A ) } { x } } , \qquad \int _ { 0 } ^ { 1 } \exp \left( - { \frac { u x } { 2 } } \right) { \mathrm { d } } u \leq 2 \operatorname* { m i n } \left\{ 1 , { \frac { 1 } { x } } \right\} { \mathrm { . } }\tag{34}
$$

To conclude, we apply Lemma 14 for each $u \in ( 0 , 1 )$ . For its first term, use the first integration bound in (34) with $A = d ( 1 + q _ { n } ) $ and $x = \lambda _ { 1 } \sum _ { s \in \left[ n \right] } \eta _ { s }$ . For its $i ^ { \mathrm { t h } }$ variance summand with $i < n ,$ use the second integration bound with $\begin{array} { r } { x = \lambda _ { 1 } \sum _ { s = i + 1 } ^ { n } \eta _ { s } \mathrm { , } } \end{array}$ ; for $i = n$ , the exponential is identically 1. Taking $c \leq 1$ , we have $\exp ( q _ { n } ) = { \cal O } ( 1 )$ and $1 + \log ( { d ( 1 + q _ { n } ) } ) = O ( \log ( e d ) )$ , so

$$
\int _ { 0 } ^ { 1 } \frac { N _ { n } ( u ) } { Z _ { n } } \mathrm { d } u \leq C ^ { \prime } \left( \frac { \log ( e d ) } { \lambda _ { 1 } \sum _ { s \in [ n ] } \eta _ { s } } + V \sum _ { i \in [ n ] } \eta _ { i } ^ { 2 } \operatorname* { m i n } \left\{ 1 , \frac { 1 } { \lambda _ { 1 } \sum _ { s = i + 1 } ^ { n } \eta _ { s } } \right\} \right) ,
$$

where $C ^ { \prime } > 0$ is universal and the minimum for $i = n$ is interpreted as 1. Substituting this bound on $\int _ { 0 } ^ { 1 } \frac { N _ { n } ( u ) } { Z _ { n } }$ du into (33) proves (19). □