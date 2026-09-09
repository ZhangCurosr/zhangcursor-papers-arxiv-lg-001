# High-Magnetization Sampling at Low Temperatures: Ising Models and Bayesian Sparse Linear Regression

Syamantak Kumar<sup>∗</sup> Purnamrita Sarkar<sup>†</sup> Kevin Tian<sup>‡</sup> Yusong Zhu<sup>§</sup>

## Abstract

Sparsity is a powerful structural resource in optimization and statistics. We develop frameworks for leveraging sparsity in sampling problems over the Hamming slice $\mathcal { X } _ { k } ^ { d } : = \{ \mathbf { x } \in \{ \pm 1 \} ^ { d }$ : $| \{ i : { \bf x } _ { i } = 1 \} | = k \}$ , in high-dimensional regimes where $k \ \ll \ d \ ( \mathrm { i . e . }$ , where $\mathcal { X } _ { k } ^ { d }$ is highlymagnetized). We use our frameworks to design improved samplers for canonical problems in the study of Ising models and Bayesian sparse linear regression.

Our first main result considers the Sherrington-Kirkpatrick (SK) model, restricted to fixedmagnetization slices $\mathcal { X } _ { k } ^ { d }$ . We give a polynomial-time sampler for fixed-magnetization SK models at any inverse temperature $\beta > 0 .$ , under arbitrary external fields, provided $k \leq c _ { \beta } d$ for an appropriate constant $c _ { \beta }$ . By combining this result with an annealing strategy for estimating normalizing constants, this yields polynomial-time samplers for the SK model at arbitrarily low temperatures, under a suficiently strong external field strength h. In the large $\beta$ limit, our framework permits sampling at field strengths within constant factors of the Almeida-Thouless line delineating the replica symmetric and replica symmetry breaking regions [dAT78], improving polynomially over the $h ( \beta )$ required by recent work of [BAR26].

Our second main result concerns the measurement complexity of polynomial-time Bayesian sparse linear regression. Recent work by [KSTZ25] shows how to sample from the canonical Gaussian spike-and-slab posterior model with expected sparsity k, at any signal-to-noise ratio, $\mathrm { g i }$ ven $n \gtrsim \bar { k } ^ { 3 } \log ^ { 3 }$ d Gaussian measurements. We improve this to $n \stackrel { . } { \sim } k ^ { 1 . 5 } \log ^ { 2 } d + k \log ^ { 3 } d ,$ using a common sparsity-aware framework underlying our results on Ising models.

## Contents

1 Introduction 1   
1.1 Our results 3   
1.2 Our techniques 4   
1.3 Related work 7   
2 Preliminaries 8   
2.1 Notation . 8   
2.2 Markov chains . 10   
2.3 Statistical models 12   
3 Sparse Dobrushin Condition 13   
3.1 Basic analysis . . . 14   
3.2 Fixed-magnetization Ising models 15   
4 Spectral Mixing via Trickle Down 18   
4.1 Trickle down framework 19   
4.2 Simple suficient conditions for fast mixing . 22   
4.3 Linear magnetization in low-temperature SK models 25   
5 Annealing 28   
5.1 Estimating normalizing constants . 29   
5.2 Approximate bounded-magnetization sampling 30   
5.3 Bounded-magnetization Ising models 31   
6 Bayesian Sparse Linear Regression 33   
6.1 Setup 33   
6.2 Trickle down for support posterior 35   
6.3 Main result 42   
AI Disclosure 44   
A Sampling Near the Almeida–Thouless Line 51   
A.1 Preliminaries 52   
A.2 Sampling around 1 54   
A.3 Sampling around the mean 55   
B Scaling of Infinite ∆-Regular Tree Threshold 58

## 1 Introduction

Harnessing sparsity is a central theme in modern high-dimensional optimization and statistics. From a sample complexity perspective, sparsity is often a blessing: classical results on Gelfand widths [Kas77, GG84] imply that, in the well-studied sparse linear regression (SLR) problem, only n ≈ k log d noisy measurements $\mathbf { y } = \mathbf { X } \pmb { \theta } ^ { \star } + \pmb { \xi }$ are information-theoretically suficient to estimate a k-sparse signal $\pmb { \theta } ^ { \star } \in \mathbb { R } ^ { d }$ up to the noise level $\| { \pmb \xi } \| _ { 2 }$ . From an algorithm design perspective, however, imposing sparsity can create highly complex, nonconvex landscapes. Seminal work in compressed sensing overcame this tension by identifying structural conditions, such as the restricted isometry property, under which sparse recovery is possible in polynomial time [CT05, CT06, CRT06, Don06], leading to a broad theory of sparsity-aware optimization [Wai19].

Our goal is to develop an analogous framework for exploiting sparse structure in high-dimensional sampling. In the problems we study, sparsity restricts the number of simultaneously “active” coordinates without fixing their locations. A central motivating problem in this work is sampling from the Sherrington-Kirkpatrick (SK) model (cf. Model 1) under sparsity constraints. Interestingly, understanding sampling algorithms for the SK model under sparsity has other consequences, including improved algorithms for posterior sampling in Bayesian sparse linear regression.

SK model. The SK model is a well-studied special case of the Ising model, a measure over vectors of spins from the hypercube $\mathbf { x } \in \mathcal { X } ^ { d } : = \{ \pm 1 \} ^ { d }$ . Such x can be identified with a set $S \subseteq [ d ]$ , the locations of positive spins $\mathbf { x } _ { i } = 1$ . In Ising models, the underlying measure is

$$
\mu ( \mathbf { x } ) \propto \exp \left( \beta \left( \frac { 1 } { 2 } \mathbf { x } ^ { \top } \mathbf { J } \mathbf { x } + \mathbf { h } ^ { \top } \mathbf { x } \right) \right) , \quad \mathbf { x } \in \mathcal { X } ^ { d } ,\tag{1}
$$

governed by an interaction matrix $\mathbf { J } \in \mathbb { R } ^ { d \times d }$ , an external field h $\in \mathbb { R } ^ { d }$ , and an inverse temperature $\beta > 0$ Ising models are a canonical testbed across statistical physics, machine learning, and theoretical computer science [WJ08, LPW09, Tal10]. For suitable J, the Gibbs landscape in (1) is known to undergo qualitative phase transitions as $\beta$ varies [Tal10]. Eficient algorithms exist for sampling under the SK model, where J is drawn from the Gaussian orthogonal ensemble (GOE), for β up to a universal constant c [EKZ22, AJK<sup>+</sup>22, AMS22, AKV24, DLSS26], while conditional hardness has been demonstrated at a higher constant threshold $c ^ { \prime } > c \ \mathrm { [ A M S 2 2 ] }$

Sparsity from field strength. As intuition for our central high-magnetization SK model, to be introduced next, we describe a well-studied analog: the SK model under a strong positive external field $\mathbf { h } = \theta \mathbf { 1 } _ { d }$ . When θ is large, (1) is biased toward $\mathbf { 1 } _ { d } .$ driving the minority-spin set to become sparse. The high-field strength model thus exhibits a soft form of sparsity.

On the algorithmic side, [BAR26] proved polynomial-time mixing of the Glauber dynamics for the SK model at every inverse temperature $\beta > 0$ , under a high enough field strength $h : = \beta \theta$ . They derive their sampler as a consequence of a more general result that proves mixing in Ising models (1), under a condition on the sparse operator norm of J at a constant sparsity scale $k = \Theta ( n )$

On the geometric side, the Almeida–Thouless (AT) line is the standard benchmark for replica symmetry in the (β, h)-plane. This line, derived by [dAT78] using the replica method (and described in Appendix $\mathrm { A } )$ , predicts a qualitative shift in the behavior of (1) as h grows compared to $\beta .$ . This line provides a natural field strength scale against which to compare [BAR26] to our results.

The role of sparsity in [BAR26] is implicit: a large field strength h causes independent replicas to have high overlap, so their disagreement set is typically sparse. The relevant interaction depends on a principal submatrix of J, giving a dependence on sparse operator norms. This suggests a complementary question: can high magnetization itself, imposed as a hard constraint rather than induced by an external field, make sampling in the SK model tractable?

Sparsity from high magnetization. High-magnetization regimes have long played an important role in statistical physics, e.g., the study of spontaneous magnetization and large deviations [Yan52, Bon14, ADCS14, Ell12]. For measures on $\chi ^ { d }$ , high magnetization is a natural analog of sparsity. Under our convention, the positive spins are the active coordinates: an element x of

$$
\mathcal { X } _ { k } ^ { d } : = \Big \{ \mathbf { x } \in \mathcal { X } ^ { d } : \left| \{ i : \mathbf { x } _ { i } = 1 \} \right| = k \Big \}\tag{2}
$$

has exactly k positive spins and, for $k \leq \frac { d } { 2 }$ , magnetization magnitude $d - 2 k$ . Taking $k \ll d$ imposes a sparsity constraint, while maintaining an outcome space of exponential size $\begin{array} { r } { \binom { d } { k } \approx \exp ( k \log \frac { d } { k } ) } \end{array}$ We study the SK model restricted to both the fixed-magnetization slice $\mathcal { X } _ { k } ^ { d }$ , and its boundedmagnetization counterpart $\mathcal { X } _ { < k } ^ { d } : = \bigcup _ { i = 0 } ^ { k } \mathcal { X } _ { i } ^ { d }$ . We now state our first central problem.

For every fixed $\beta < \infty$ , is there a polynomial-time sampler from the

fixed-magnetization SK model whenever $k \leq c _ { \beta } d$ for a fixed $c _ { \beta } > 0 ?$

Theorem 1 answers this question afirmatively, taking the algorithm as the down-up walk.

Bayesian SLR. Our second motivating example is a Bayesian variant of SLR. SLR is often phrased as an optimization problem: given noisy measurements $( \mathbf { X } , \mathbf { y } = \mathbf { X } \pmb { \theta } ^ { \star } + \pmb { \xi } )$ , return a k-sparse $\widehat { \pmb { \theta } } \in \mathbb { R } ^ { d }$ (approximately) minimizing the residual norm $\| \mathbf { X } { \widehat { \pmb { \theta } } } - \mathbf { y } \| _ { 2 }$ . When X is RIP, low residual error implies accurate estimation [CRT06], so optimization produces a good point estimate of $\pmb { \theta } ^ { \star }$

In many applications, however, it is preferable to sample $\widehat { \pmb { \theta } }$ from a distribution over plausible signals, $\mathrm { e . g . }$ , to model uncertainty in support estimation (variable selection) [MB88, GM93]. In the wellestablished Bayesian SLR model, the noise $\boldsymbol { \xi }$ is Gaussian with coordinatewise variance $\sigma ^ { 2 }$ , where $\sigma ^ { - 1 } > 0$ is a signal-to-noise ratio. For an appropriate prior π over sparse signals $\pmb { \theta } ^ { \star }$ , the goal is then to sample from the posterior induced by the observations:

$$
\widehat { \pmb { \theta } } \sim \pi \left( \cdot \mid \mathbf { X } , \mathbf { y } \right) , \mathrm { ~ w h e r e ~ } \mathbf { y } = \mathbf { X } \pmb { \theta } ^ { \star } + \pmb { \xi } , \quad \pmb { \theta } ^ { \star } \sim \pi , \quad \pmb { \xi } \sim \mathcal { N } \left( \mathbf { 0 } _ { n } , \sigma ^ { 2 } \mathbf { I } _ { n } \right) .\tag{3}
$$

Samples from the posterior density can then be used in downstream tasks, $\mathrm { e . g . }$ , constructing credible intervals. In the statistics literature, π is often taken to be the spike-and-slab prior,

$$
\pi = \bigotimes _ { i \in [ d ] } \left( 1 - { \frac { k } { d } } \right) \delta _ { 0 } + { \frac { k } { d } } { \mathcal { N } } ( 0 , 1 ) ,\tag{4}
$$

where each signal coordinate $\pmb { \theta } _ { i } ^ { \star }$ is independently set to 0 except with low probability (so the expected sparsity is k). The induced spike-and-slab posterior sampling problem is often called the “theoretical gold standard” for modeling uncertainty in variable selection [JS04, CPS09, IR11, CvdV12, Roc18, PS19]. Unfortunately, this task poses a notorious computational challenge [CSHVdV15], and many heuristics have been developed as approximations [BRG21].

Recently, several works developed provable methods for this sampling problem [YWJ16, MW26], including an algorithm by [KSTZ25] which samples from the posterior density (3), (4) given a sublinear-in-d, $n \gtrsim k ^ { 3 } \log ^ { 3 } d$ measurements. The counterpart optimization problem is known to be feasible even when $n \approx k$ log d, motivating our second central problem.

What is the measurement threshold n at which spike-and-slab posterior sampling admits polynomial-time algorithms, for any signal-to-noise ratio?

We show that the complex SLR posterior density admits enough structure to be captured by our analysis framework for the high-magnetization SK model, and give a polynomial-time posterior sampling algorithm at $n \gtrsim k ^ { 1 . 5 } \mathrm { p o l y l o g } ( d )$ measurements (Theorem 2).

## 1.1 Our results

In this section, we overview our main results. To obtain these results, we develop a suite of technical tools for exploiting sparsity in sampling, described at more depth in Section 1.2.

High-magnetization SK models. Our first main result concerns sampling in high-magnetization SK models, where J is GOE and the external field h is arbitrary. As a benchmark, [AMS22] showed conditional hardness for sampling in SK models at large inverse temperatures $\beta > c ^ { \prime }$ for a constant $c ^ { \prime } .$ . We show that after fixing the magnetization $k \leq c _ { \beta } d$ of the spin vector $\mathbf { x } \in \mathcal { X } _ { k } ^ { d }$ , the SK model admits polynomial-time sampling at any temperature.

Theorem 1 (Informal; see Theorem 4). For any fixed $\beta > 0 , \textbf { h } \in \mathbb { R } ^ { d }$ , and $\mathbf { J } \sim \mathrm { G O E } ( d )$ , there is a constant $c _ { \beta } > 0$ such that if $k \leq c _ { \beta } d _ { ; }$ we can sample from the SK model (1) restricted to the Hamming slice $\mathcal { X } _ { k } ^ { d }$ in polynomial time, with high probability over J.

The algorithm in Theorem 1 is the canonical down-up walk over the Hamming slice $\mathcal { X } _ { k } ^ { d } .$ , and we bound its runtime by a relatively mild $\widetilde O ( d ^ { 2 } + d k ^ { 2 } \operatorname* { m a x } ( 1 , \beta \left\| \mathbf { h } \right\| _ { \infty } ) ) . ^ { 1 }$ Beyond Theorem 1, Section 4 develops a more general framework for proving Poincaré inequalities on the down-up walk for Ising models, using the trickle down (“local-to-global”) theorem of [Opp18, AL20]. For example, we state an analogous result for Gaussian Hopfield models in Corollary 4.

Due to Theorem 1, we recover a variant of the main result of [BAR26] by exploiting the relationship between high field strength and high magnetization.<sup>2</sup> We do note that [BAR26] directly analyze the Glauber dynamics, perhaps the simplest sampling algorithm over $\chi ^ { d }$ , whereas we use an annealing scheme (Section 5) to reduce bounded-magnetization sampling to fixed-magnetization sampling.

Our framework yields other interesting consequences beyond the linear sparsity regime in Theorem 1. For example, for $k = O ( 1 )$ independent of $d \to \infty$ , our results imply polynomial-time sampling in high-magnetization SK models for polynomial inverse temperatures $\beta ,$ up to $O ( { \sqrt { d / \log ( d ) } } )$ (cf. Corollary 3). Prior results exploiting notions of sparsity (albeit diferent from ours) to sample from Ising models at subconstant temperatures only tolerated $\beta \approx \log d$ [CDKP22, KPPY25]. This comparison is discussed at more length in Section 1.3 and Appendix B.

The Almeida-Thouless line. From a quantitative perspective, the AT line is a useful benchmark for comparing our sampling result with [BAR26]. The AT line predicts a phase transition in the geometry of the SK model under an external field $\begin{array} { r } { \mathbf { h } = \frac { h } { \beta } \mathbf { 1 } _ { d } . } \end{array}$ once the field strength $h \ge h _ { \mathrm { A T } } ( \beta )$ is large enough as a function of $\beta _ { i }$ , where $h _ { \mathrm { A T } } ( \beta ) = ( 1 + o ( 1 ) ) \beta \sqrt { 2 \log \beta }$ (Lemma 23).

Leveraging a simple reduction from high field strength sampling to high-magnetization sampling (Lemma 24), we show in Theorem 6 that Theorem 1 implies sampling from the SK model whenever $h \geq h _ { \mathrm { H M } } ( \beta )$ for a threshold $h _ { \mathrm { H M } } ( \beta ) = \sqrt { 2 } ( 1 + o ( 1 ) ) h _ { \mathrm { A T } } ( \beta )$ , i.e., within constant factors of the AT line. We also show in Theorem 7 that, if given access to the signs of the mean vector E[x] under the SK model, a modification of our sampler succeeds at any field strength $h \geq ( 1 + o ( 1 ) ) h _ { \mathrm { A T } } ( \beta )$ . By comparison, the result of [BAR26] applies whenever $h = \Omega ( \beta ^ { 2 } \sqrt { \log \beta } )$ , a polynomial factor larger (see discussion in Appendix A). The tighter range of h tolerated by our framework is a result of our basic sampler in Theorem 1 applying for an arbitrary external field h.

Bayesian SLR. Finally, we apply our sparse sampling frameworks to spike-and-slab posterior sampling. We obtain a state-of-the-art measurement complexity for a canonical variant of the problem, stated in (3), (4), and studied by [KSTZ25, MW26].

Theorem 2 (Informal; see Theorem 5). Let $\mathbf { X } \in \mathbb { R } ^ { n \times d }$ have i.i.d. entries distributed as $\textstyle { \mathcal { N } } ( 0 , { \frac { 1 } { n } } )$ . If

$$
n = \Omega \left( \left( k + \log \left( \frac { 1 } { \delta } \right) \right) ^ { 1 . 5 } \log ^ { 2 } \left( \frac { d } { \delta } \right) + \left( k + \log \left( \frac { 1 } { \delta } \right) \right) \log ^ { 3 } \left( \frac { d } { \delta } \right) \right) ,
$$

then, for every $\sigma > 0$ , there is a polynomial-time algorithm that samples from $\pi ( \cdot \mid \mathbf { X } , \mathbf { y } )$ , defined in (3) and (4), within total variation distance δ, with probability at least 1 − δ over the model.

As in prior work, there are two sources of failure in Theorem 2. Namely, the model (3), (4) may fail to produce a sparse signal $\pmb { \theta } ^ { \star }$ or bounded noise ξ (inhibiting tractability of the problem), and the sampling algorithm itself has an approximation error. Our formal result, Theorem 5, is more general and can handle nonuniform inclusion weights in the prior (see Model 3). Moreover, the runtime of Theorem 2 is relatively practical, e.g., it scales linearly in nd.

Our $n \approx k ^ { 1 . 5 } \log ^ { 2 } d$ requirement improves quadratically in its dependence on k upon the previous state-of-the-art sampler by [KSTZ25], which uses $n = \Omega ( k ^ { 3 } \log ^ { 3 } d )$ measurements (see also [MW26], who gave a result in the regime $n = \Omega ( d ) )$ ). Interestingly, the $k ^ { 1 . 5 }$ bottleneck appears inherent to our approach (discussed in the following Section 1.2). This motivates the tantalizing open question of whether spike-and-slab posterior sampling is tractable at $n = \Omega ( k \log d )$ measurements, which would close the gap between frequentist and Bayesian SLR.

## 1.2 Our techniques

In this section, we overview the main proof ideas behind our sparse Dobrushin (Section 3) and trickle down frameworks (Section 4), our annealing reduction from bounded-magnetization to fixedmagnetization sampling (Section 5), and our application to Bayesian SLR (Section 6).

Sparse Dobrushin condition. In Section 3, we give a warm-up path coupling analysis of the down-up walk illustrating why restricting $k \ll d$ can make low-temperature sampling easier. For a measure supported on k-sized subsets, we compare the conditional laws of the up step from neighboring (k − 1)-sized cores, after excluding the coordinates on which the two cores difer. If the resulting total variation discrepancy is $O \big ( \frac { 1 } { k } \big )$ , the walk contracts in Hamming distance and mixes in $\begin{array} { r } { O \big ( k \log { \frac { k } { \varepsilon } } \big ) } \end{array}$ steps (Lemma 3). This framework depends on a sparse variant of the classical Dobrushin influence matrix [Dob68], so we term it a sparse Dobrushin condition (Definition 1).

For fixed-magnetization Ising models, this discrepancy is controlled by $\beta$ max $i \neq j \ | { \bf J } _ { i j } |$ , independently of the external field h. Sparsity sets the required discrepancy bound $\begin{array} { r } { \mathrm { a t } \approx \frac { 1 } { k } , \mathrm { i . e . } } \end{array}$ , a relaxed bound at higher magnetizations. The maximum entry magnitude under the SK model scales as ≈ ${ \sqrt { \log d / d } } ,$ so this warm-up result already shows a variant of Theorem 1 at the higher magnetization level $k \leq c _ { \beta } \sqrt { d / \log d }$ . The sparse Dobrushin condition is simple and broadly applicable, but it cannot exploit cancellation among signed interactions. This motivates our main technique for extending to the range $k = \Theta ( d )$ , based on spectral expansion and the trickle down theorem.

Spectral mixing via trickle down. To obtain sharper parameter ranges in fixed-magnetization Ising models, we leverage the trickle down theorem [Opp18, AL20], a foundational result in the study of high-dimensional expansion. For a measure π over $\textstyle { \binom { \mathcal { U } } { k } }$ , and a core $R \in \left( \begin{array} { c } { { \mathcal { U } } } \\ { { k - 2 } } \end{array} \right)$ , define the link graph of $R$ on $\mathcal { U } \backslash R$ to have edge weights $\mathbf { W } _ { i j } : = \pi ( R \cup \{ i , j \} )$ , and let P be the corresponding random walk matrix. The trickle down theorem (Lemma 6) shows that if we can show $\begin{array} { r } { \lambda _ { 2 } ( \mathbf { P } ) = O ( \frac { 1 } { k } ) } \end{array}$ for all cores $R ,$ then the down-up walk satisfies a $O \big ( \frac { 1 } { k } \big )$ -Poincaré inequality. Thus, the global mixing problem reduces to proving uniform spectral expansion of all the induced P.

For a fixed-magnetization Ising model, each link has a particularly useful form. Fix a core $R ,$ and let $\mathbf { r } \in \{ \pm 1 \} ^ { d }$ be the spin vector whose positive coordinates are $R .$ For some interaction matrix K, Lemma 8 shows that the link weights satisfy, for an appropriate vector $\mathbf { a } \in \mathbb { R } ^ { d \setminus R }$ ，

$$
\mathbf { W } _ { i j } \propto \mathbf { a } _ { i } \mathbf { a } _ { j } \exp ( \mathbf { K } _ { i j } ) , \quad \mathbf { K } _ { i j } : = 4 \beta \mathbf { J } _ { i j } \quad ( i \neq j ) .
$$

We view W as a bounded perturbation (parameterized by K) of the rank-one weights $\mathbf { a } \mathbf { a } ^ { \top }$ : when K is the all-zeroes matrix, $\lambda _ { 2 } ( \mathbf { W } ) \leq 0$ follows simply by a rank argument.<sup>3</sup> Our main trickle down framework gives a tighter characterization of $\lambda _ { 2 } ( \mathbf { P } )$ as a function of K. Specifically, Lemma 9 shows using a second-order Taylor expansion of the exponential that if

$$
\begin{array} { r } { \left| \mathbf { u } ^ { \top } \mathbf { K } \mathbf { u } \right| \leq \tau \left\| \mathbf { u } \right\| _ { 1 } \left\| \mathbf { u } \right\| _ { 2 } + \tau ^ { 2 } \left\| \mathbf { u } \right\| _ { 1 } ^ { 2 } , } \end{array}\tag{5}
$$

then $\lambda _ { 2 } ( { \bf P } ) = { \cal O } ( \tau ^ { 2 } )$ . This sets the required bound on $\tau$ at $\approx k ^ { - 1 / 2 }$ for the trickle down argument. As a point of comparison, the $\begin{array} { r } { \tau ^ { 2 } \approx \frac { 1 } { k } } \end{array}$ term in this argument, along with $\mathbf { u } ^ { \top } \mathbf { K } \mathbf { u } \leq \left( \operatorname* { m a x } | \mathbf { K } _ { i j } | \right) \| \mathbf { u } \| _ { 1 } ^ { 2 } .$ already qualitatively recovers the sparse Dobrushin condition (Lemma 10).

We next show that the mixed norm condition (5) allows more fine-grained control of K, in terms of its sparse operator norms $\| \mathbf { K } \| _ { s , \mathrm { o p } } ,$ i.e., the largest $\| \mathbf { K } _ { S \times S } \| _ { \mathrm { o p } }$ among any s-sized sets S. This argument proceeds by applying a shelling decomposition, a classic technique from the sparse recovery literature [CRT06] that places the “efective sparsity” of a vector u on the scale of $\frac { \| \mathbf { u } \| _ { 1 } ^ { 2 } } { \| \mathbf { u } \| _ { 2 } ^ { 2 } }$

A basic application of this strategy (Lemma 11) shows that if $\| \mathbf { K } \| _ { s , \mathrm { o p } } \lesssim \tau \sqrt { s } + \tau ^ { 2 } s$ at all scales $s \in [ d ]$ , then (5) holds. Plugging in standard bounds on the sparse operator norms of various matrix ensembles now already gives Theorem 4 up to a logarithmic loss in the tolerated k range (Corollary 3), as well as our strongest conclusion for Gaussian Hopfield Ising models (Corollary 4). Our final application to SK models in the linear magnetization regime $k = \Theta ( d )$ (Theorem 4) uses more fine-grained estimates of sparse operator norms for GOE matrices.

Annealing. We take a brief detour to discuss a complementary part of our framework: a technique for lifting fixed-magnetization samplers to the bounded-magnetization setting (measures supported on $S \subseteq \mathcal { U } : | S | \leq k )$ . Our approach is based on the fact that bounded-magnetization measures are mixtures of fixed-magnetization measures, with weights proportional to normalizing constants. That is, to approximate a global density π on $\mathcal { U } _ { \leq k }$ to total variation $\delta ,$ it is enough to estimate

$$
Z _ { i } : = \sum _ { \omega \in \mathcal { U } _ { i } } \pi ( \omega ) \ \mathrm { f o r \ a l l } \ 0 \leq i \leq k
$$

to multiplicative error $O ( \delta )$ . We formalize this argument in Lemma 14.

Conveniently, there is a rich literature in theoretical computer science reducing between the problems of counting (e.g., normalization constant estimation) and sampling. An existing result, Theorem 6 in [Kol18], is essentially black-box applicable to our setting, giving δ-multiplicative estimates to any $Z _ { i }$ by using oracle calls to fixed-magnetization samplers. By building upon the estimator of [Kol18], we state our generic reduction from bounded-magnetization sampling to fixed-magnetization sampling in Lemma 15, and give an example of its use for the SK model in Corollary 5.

Spike-and-slab posterior sampling. We finally turn to our second main application: Bayesian SLR with a spike-and-slab prior, as studied by [KSTZ25, MW26]. The primary challenge is to correctly sample the support $S : = \mathrm { s u p p } ( \pmb { \theta } ^ { \star } ) \subseteq [ d ]$ from the posterior density

$$
\pi _ { \mathrm { s u p p } } ( S \mid \mathbf { X } , \mathbf { y } ) : = \pi ( \{ \pmb { \theta } : \mathrm { s u p p } ( \pmb { \theta } ) = S \} \mid \mathbf { X } , \mathbf { y } ) .
$$

As derived in prior work (cf. Fact 3), this density is proportional to a closed-form expression:

$$
\pi _ { \mathrm { s u p p } } ( S ) \propto \left( \frac { k } { d - k } \right) ^ { | S | } \exp \left( \frac { 1 } { 2 } \left. \mathbf { b } _ { S } \right. _ { \mathbf { A } _ { S } ^ { - 1 } } ^ { 2 } \right) \frac { 1 } { \sqrt { \operatorname* { d e t } \mathbf { A } _ { S } } } ,\tag{6}
$$

where A, b are induced by the measurements $( \mathbf { X } , \mathbf { y } )$ and defined in (12). Note that the first term in the above expression can be absorbed into an external field.

Our posterior sampler has three components. The first follows [KSTZ25] and uses a sparse recovery preprocessing step (Proposition 3) that identifies coordinates whose inclusion is nearly deterministic from $( \mathbf { X } , \mathbf { y } )$ , providing regularity to the residual posterior density. The second applies our annealing procedure from Section 5 to reduce the problem to a fixed-magnetization variant of (6).

The remaining step is to use our trickle down framework to demonstrate mixing of the down-up walk on fixed-magnetization slices of the residual posterior. The induced weight matrices W after pinning a core are more complex than in the SK setting, because of the inverse and determinantal terms in (6). In particular, the resulting perturbation K depends on the pinned core through a Schur complement correction, which yields various dependencies. Due to a union bound over all possible cores, our uniform estimate of $\tau$ in (5) scales as $\begin{array} { r } { n ^ { - 1 / 2 } + \frac { k } { n } \ ( \mathrm { e . g . } } \end{array}$ , see (53) in Lemma 20), and setting this to $k ^ { - 1 / 2 }$ as required by the trickle down framework gives a $n \gtrsim k ^ { 1 . 5 }$ bottleneck. Removing this $\sqrt { k }$ factor beyond the measurement complexity of (frequentist) sparse recovery is an exciting problem, that we leave open as a testbed of “average-case” trickle down theorems avoiding the union bound over worst-case dependencies sufered by our approach.

## 1.3 Related work

Fixed- and bounded-magnetization sampling. The most conceptually relevant prior algorithmic works are by [CDKP22, KPPY25], both of which study fixed-magnetization problems that exhibit improved phase transitions or critical $\beta$ as the sparsity (positive spin count) k becomes small. For example, Theorem 2 in [CDKP22] shows that for ferromagnetic Ising models with bounded degree $\Delta _ { i }$ , there is a critical “tree threshold” $\beta _ { c } ( \Delta ) = O ( \textstyle { \frac { 1 } { \Delta } } )$ such that for $\beta > \beta _ { c } ( \Delta )$ , the model undergoes a computational phase transition at a certain magnetization level. Notably, the sparsity tradeof required by their paper beyond $\beta > \beta _ { c } ( \Delta )$ decays exponentially in $\beta \colon$ we give a more formal derivation in Appendix B, but $\mathrm { e . g . }$ , for constant $\Delta .$ , they require

$$
k = d \exp \left( - \Theta _ { \Delta } ( \beta ) \right) ,
$$

which only implies a nontrivial setting $( k \neq 0 )$ up to $\beta \approx \log d .$ Similarly, Theorem 1 of [KPPY25] shows that the Kawasaki dynamics [Kaw66] (a standard magnetization-conservative dynamics) exhibits a phase transition at $\beta > \beta _ { c } ( \Delta )$ , but their magnetization thresholds inherit the same exponential decay in $\beta ,$ , and hence also apply only up to $\beta \approx \log d .$ On the other hand, our results are derived through more general sparsity-aware analysis frameworks (Definition 1) that continue to improve as $k  1$ , tolerating $\beta$ up to $\mathrm { p o l y } ( d )$ . This improvement is closer in spirit to existing results in the sparse recovery literature, where general structural conditions (e.g., RIP) yield sample complexity requirements that improve monotonically with the sparsity level k.

There have been other works that study sampling in high-magnetization regimes, that are motivated by giving analysis frameworks for natural sampling dynamics at fixed magnetization (e.g., the down-up walk or Kawasaki dynamics), rather than obtaining improved thresholds as $k  1$ . For example, [BBD24] study the local Kawasaki dynamics on random ∆-regular graphs, and prove rapid mixing for $\beta = \bar { O ( \Delta ^ { - 1 / 2 } ) }$ ; notably, their thresholds do not improve as $k  1$ . Similarly, a line of work has obtained estimates for the mixing time of the down-up walk that improve as k decreases [ALGV19, CGM19, AKV24], e.g., that it mixes in $O ( k \log k )$ steps for strongly log-concave measures, but their focus was not the relationship between k and the allowable inverse temperature $\beta .$

High-field strength spin glasses. The Almeida-Thouless line [dAT78] is the canonical benchmark for the replica symmetry phase transition in the SK model with a homogeneous external field $\theta \mathbf { 1 } _ { d } .$ From a geometric perspective, a recent work of [Lop26] established replica symmetry throughout the AT region, and [KN26] obtained quantitative overlap concentration in the strict AT region. Stronger forms of quantitative overlap concentration in a more restrictive region (cf. Definition 3) have also appeared in the literature [Tal11, JT17]; in particular, we use a bound from [RW26] in our reduction from high-field strength sampling to high-magnetization sampling (Appendix A). On the algorithmic side, [BAR26] prove polynomial-time mixing of the Glauber dynamics in the SK model at suficiently high field strength, by exploiting replica overlap and sparse operator norm bounds. We quantitatively improve upon the field strength tolerance of [BAR26] to within constant factors of the AT line, or within $1 + o ( 1 )$ if granted the signs of the mean vector.

Other applications of sparsity in sampling. Recent works by [BAR26, DLSS26] both use sparse operator norm bounds to derive mixing times for sampling from Ising models, related to the analytical core of our work (e.g., Lemma 11). Theorem 1 of [BAR26] shows that if the sparse operator norm is bounded at a sparsity level $k \ = \ \Theta ( d )$ , then the Glauber dynamics mixes in polynomial time under a suficiently large external field. In a similar spirit, Section 7 of [DLSS26] uses sparse operator norm bounds to prove rapid mixing of the SK model on small Hamming balls, but only tolerates $\beta < \textstyle { \frac { 1 } { 2 } }$ , as opposed to the arbitrary $\beta > 0$ handled by our Theorem 1.

More generally, a recurring theme in high-dimensional sampling is that sparsity and fixed-cardinality constraints can influence algorithmic landscapes. Recent work on discrete sampling has obtained sharp thresholds and improved mixing guarantees for fixed-size structures such as independent sets and matchings. In particular, [DP23] identified the computational threshold for approximately counting and sampling independent sets of a given size in bounded-degree graphs, [JMPV23] proved optimal $O ( k \log n )$ mixing of the down-up walk for independent sets of size $k ,$ and [JM24] established polynomial-time mixing of the down-up walk for matchings of size k. Recent progress on Ising models has also extended beyond dense mean-field settings: for example, [LMRW24] proves near-linear time mixing of the Glauber dynamics for sparse random Ising models, including the Viana–Bray spin glass, and also treats certain interaction matrices arising from stochastic block models.

Bayesian sparse linear regression. Spike-and-slab priors and their induced posteriors are classical tools for Bayesian variable selection [MB88, GM93]. A large statistical literature studies posterior contraction and uncertainty quantification for sparse priors, including spike-and-slab formulations and closely related continuous relaxations [CSHVdV15, Roc18]. Notably, in the moderate signal-tonoise ratio (SNR) regime, the support posterior measure is known to be highly multimodal, which poses an algorithmic challenge [CSHVdV15]. In comparison, the modes collapse at a high SNR, and the posterior nearly reduces to the prior under a low SNR.

On the computational side, many works target posterior modes or tractable approximations, including expectation maximization, Lasso-based procedures, and variational Bayes approximations [RG14, RG18, RS22, MS22]. While these works demonstrate the empirical performance of their algorithms, they do not yield end-to-end algorithmic guarantees for the sampling problem we consider. Among provable posterior samplers, [YWJ16] analyzes a Metropolis-Hastings chain under a truncated sparsity prior, which applies only under high or low signal-to-noise ratio regimes (see discussion after their Eq. (10)), while [MW26] gives a polynomial-time sampler via measure decomposition when the number of measurements n grows at least linearly with the ambient dimension d. A complementary difusion-based approach for linear inverse problems was recently developed in [BH24] (and analyzed in continuous time), though their framework does not seem to directly apply to spike-and-slab posterior sampling at $n \ll d .$ The closest prior work is [KSTZ25], which gave the first provable spike-and-slab posterior samplers that apply at arbitrary SNRs while allowing n to remain sublinear in d. Our result quadratically improves upon the k dependence of [KSTZ25].

## 2 Preliminaries

In this section we develop preliminaries for the rest of the paper. Section 2.1 provides notation used throughout. Section 2.2 gives basic notation and facts about Markov chains used in our analysis. Section 2.3 introduces the main statistical models we consider for our applications.

## 2.1 Notation

General notation. We use $\lesssim \gtrsim$ and ≈ in informal exposition only to suppress polylogarithmic factors in problem parameters; formal statements have all dependences explicitly stated.

For $n \in \mathbb { N }$ we let $[ n ] : = \{ i \in \mathbb { N } : i \leq n \}$ . We reserve uppercase and lowercase boldface for matrices and vectors respectively. We use ${ \bf 1 } _ { d }$ and $\mathbf { 0 } _ { d }$ to denote the all-ones and all-zeroes vectors in $\mathbb { R } ^ { d }$ $\mathbf { I } _ { d }$ to denote the identity in $\mathbb { R } ^ { d }$ , and ${ \bf 0 } _ { m \times n }$ is the $m \times n$ all-zeroes matrix. We denote the entrywise (Hadamard) product of equal-length vectors a, b by a ◦ b. $\mathbb { S } ^ { d \times d }$ and $\mathbb { S } _ { \geq 0 } ^ { d \times d }$ respectively denote the symmetric and positive semidefinite $d \times d$ matrices, where $\preceq$ is the Loewner partial order. We use nnz to denote the number of nonzero entries of a vector or matrix, and supp denotes the corresponding index set. $\mathbf { e } _ { i }$ denotes the $i ^ { \mathrm { t h } }$ standard basis vector, and we define the row and column selectors $\mathbf { M } _ { : i } : = \mathbf { M } \mathbf { e } _ { i } , \mathbf { M } _ { i : } : = \mathbf { M } ^ { \top } \mathbf { e } _ { i }$ . For $\mathbf { M } \in \mathbb { R } ^ { m \times n }$ and $( S , T ) \subseteq [ m ] \times [ n ] , \mathbf { M } _ { S \times T }$ means the appropriate submatrix; our convention is to transpose before indexing, so $\mathbf { M } _ { T \times S } ^ { \top } = ( \mathbf { M } _ { S \times T } ) ^ { \top }$

For $1 \leq p \leq \infty , \| \cdot \| _ { p }$ denotes the vector $\ell _ { p }$ norm, and for $1 \leq p , q \leq \infty$ and a matrix M, the associated operator norm is $\begin{array} { r } { \left\| \mathbf { M } \right\| _ { p \to q } : = \operatorname* { m a x } _ { \mathbf { v } \in \mathbb { R } ^ { d } : \left\| \mathbf { v } \right\| _ { p } \leq 1 } \left\| \mathbf { M } \mathbf { v } \right\| _ { q } . } \end{array}$ . For any $\mathbf { M } \in \mathbb { S } _ { \succeq \mathbf { 0 } } ^ { d \times d }$ we define $\| \mathbf { v } \| _ { \mathbf { M } } ^ { 2 } : = \mathbf { v } ^ { \top } \mathbf { M } \mathbf { v }$ . We use $\lVert \cdot \rVert _ { \mathrm { F } }$ and $\left\| \cdot \right\| _ { \mathrm { o p } }$ to denote the Frobenius and $( 2  2 )$ operator norms. For $\mathbf { M } \in \mathbb { S } ^ { d \times d }$ we let $\pmb { \lambda } ( \mathbf { M } )$ be its eigenvalues sorted so $\lambda _ { 1 } ( \mathbf { M } ) \geq . . . \geq \lambda _ { d } ( \mathbf { M } )$ ; we define $\sigma$ to similarly return the sorted singular values of its input. We define $\omega < 2 . 3 7 3 \ [ \mathrm { A D V ^ { + } 2 5 } ]$ so that multiplying, inverting, and eigendecomposing $d \times d$ matrices takes $O ( d ^ { \omega } )$ time [Str69, PC99].

We frequently use the following “restricted” or “sparse” quantities to parameterize our results: for $k \in [ d ]$ , the top-k norm of a vector $\mathbf { v } \in \mathbb { R } ^ { d }$ and matrix $\mathbf { M } \in \mathbb { S } ^ { d \times d }$ are

$$
\begin{array} { c } { \| \mathbf { v } \| _ { k , p } : = \displaystyle \operatorname* { s u p } _ { S \subseteq [ d ] : | S | = k } \| \mathbf { v } _ { S } \| _ { p } \mathrm { ~ f o r ~ a l l ~ } p \geq 1 , } \\ { \| \mathbf { M } \| _ { k , \mathrm { o p } } : = \displaystyle \operatorname* { s u p } _ { \mathbf { v } \in \mathbb { R } ^ { d } \atop \| \mathbf { v } \| _ { 2 } \leq 1 , \operatorname { n n z } ( \mathbf { v } ) \leq k } \left| \mathbf { v } ^ { \top } \mathbf { M } \mathbf { v } \right| = \displaystyle \operatorname* { s u p } _ { S \subseteq [ d ] : | S | = k } \| \mathbf { M } _ { S \times S } \| _ { \mathrm { o p } } . } \end{array}\tag{7}
$$

For two subsets $S , T$ with the same size of the same universe $u ,$ we use $\Delta _ { \mathrm { H a m } } ( S , T ) \in \mathbb { N } \cup \{ 0 \}$ to mean the Hamming distance between $S , T$ , which we define as half the number of difering elements.

Probability. For a state space $\Omega ,$ we let ${ \mathcal { P } } ( \Omega )$ denote all probability measures over Ω. For an event ${ \mathcal { E } } \subseteq \Omega$ , we let I(E) denote the corresponding 0-1 indicator random variable, and $\mu ( \mathcal { E } )$ denote the probability of the event. We let $\mathbb { E } [ \cdot ]$ and Var[·] denote the expectation and variance. For jointly distributed scalar random variables $X , Y$ $\operatorname { C o v } ( X , Y ) : = \mathbb { E } [ X Y ] - \mathbb { E } [ X ] \mathbb { E } [ Y ]$ denotes their covariance; if the random variables are instead vector-valued, $\mathbf { C o v } ( \pmb { X } , \pmb { Y } )$ is a matrix of appropriate dimension. We abbreviate $\mathbf { C o v } ( X ) : = \mathbf { C o v } ( X , X )$ . We frequently use the following distances between $\mu , \nu \in \mathscr { P } ( \Omega )$ , where dω denotes a counting measure if Ω is discrete:

$$
\mathrm { T V } \left( \mu , \nu \right) : = \frac { 1 } { 2 } \int _ { \Omega } | \mu ( \omega ) - \nu ( \omega ) | \mathrm { d } \omega = \operatorname* { s u p } _ { \varepsilon \subseteq \Omega } \mu ( \mathcal { E } ) - \nu ( \mathcal { E } ) ,
$$

$$
\chi ^ { 2 } \left( \mu \| \nu \right) : = \int _ { \Omega } \left( \frac { \mu ( \omega ) } { \nu ( \omega ) } - 1 \right) ^ { 2 } \nu ( \omega ) \mathrm { d } \omega .
$$

We denote the set of couplings of $\mu \in \mathcal { P } ( \Omega ) , \nu \in \mathcal { P } ( \Omega ^ { \prime } )$ (joint measures on $\Omega \times \Omega ^ { \prime }$ whose marginals agree with $\mu , \nu )$ , by $\Gamma ( \mu , \nu )$ . It is standard that when $\Omega = \Omega ^ { \prime }$ , an alternative definition of the total variation distance is $\begin{array} { r } { \mathrm { T V } \left( \mu , \nu \right) = \operatorname* { i n f } _ { \gamma \in \Gamma \left( \mu , \nu \right) } \mathbb { P } _ { \left( \omega , \omega ^ { \prime } \right) \sim \gamma } [ \omega \neq \omega ^ { \prime } ] } \end{array}$ (Proposition 4.7, [LPW09]).

We denote the multivariate normal distribution with specified mean and covariance by $\scriptstyle { \mathcal { N } } ( \mu , \Sigma )$ We let $\operatorname { B e r n } ( p )$ be the distribution on {0, 1} with $\mathbb { E } _ { X \sim \mathrm { B e r n } ( p ) } [ X ] = p$ . We use $\otimes _ { i \in [ n ] } \pi _ { i }$ to denote

a product measure with specified marginals, and we use $\delta _ { \omega }$ to mean a Dirac measure at ω. When $Z \in \mathbb { R } _ { > 0 } ^ { \mathcal { Z } }$ are indexed by a set I, we let Multinomial(Z) denote a draw $i \in \mathcal { Z }$ where Law(i) ∝ Z.

## 2.2 Markov chains

Let $\mathcal { T } = \{ \mathcal { T } _ { \omega } \} _ { \omega \in \Omega }$ be a set of transition distributions for a Markov chain on a state space Ω. For an arbitrary measure $\mu \in \mathscr { P } ( \Omega )$ we let $\tau _ { \mu }$ denote the marginal law of $\omega ^ { \prime }$ where $\omega \sim \mu$ and $\omega ^ { \prime } \sim \mathcal { T } _ { \omega }$ We say that π is a stationary measure for T if ${ \mathcal { T } } \pi = \pi ,$ i.e.,

$$
\int \mathcal { T } _ { \omega ^ { \prime } } ( \omega ) \pi ( \omega ^ { \prime } ) \mathrm { d } \omega ^ { \prime } = \pi ( \omega ) , \mathrm { ~ f o r ~ a l l ~ } \omega \in \Omega .
$$

We call a Markov chain T reversible if it has stationary measure π, and

$$
\pi ( \omega ) \mathcal { T } _ { \omega } ( \omega ^ { \prime } ) = \pi ( \omega ^ { \prime } ) \mathcal { T } _ { \omega ^ { \prime } } ( \omega ) , \mathrm { ~ f o r ~ a l l ~ } ( \omega , \omega ^ { \prime } ) \in \Omega \times \Omega .
$$

We often consider sampling from discrete measures over the hypercube with fixed-magnetization or bounded-magnetization. For shorthand we always let $\mathcal { X } : = \{ \pm 1 \}$ , and $\begin{array} { r } { \mathcal { X } _ { k } ^ { d } : = \{ \mathbf { x } \in \mathcal { X } ^ { d } : \sum _ { i \in [ d ] } \mathbf { x } _ { i } = } \end{array}$ $- d + 2 k \}$ , where the quantity $\begin{array} { r } { \sum _ { i \in [ d ] } \mathbf { x } _ { i } = \mathbf { 1 } _ { d } ^ { \top } \mathbf { x } } \end{array}$ is the magnetization. In other words, $\mathcal { X } _ { k } ^ { d }$ is the slice of the hypercube $\chi ^ { d }$ corresponding to elements with exactly k copies of 1 and $d - k$ copies of −1. For the analogous bounded-magnetization problem, we similarly define $\begin{array} { r } { \mathcal { X } _ { \leq k } ^ { d } : = \bigcup _ { j = 0 } ^ { k } \mathcal { X } _ { j } ^ { d } } \end{array}$

It is often helpful to associate elements of $\mathcal { X } _ { k } ^ { d }$ with subsets of [d] with size k (the locations of the 1s). We define $\mathrm { s e t } ( \mathbf { x } ) \subseteq [ d ]$ for $\mathbf { x } \in \mathcal { X } ^ { d }$ and vec $( S ) \in \mathcal { X } ^ { d }$ for $S \subseteq [ d ]$ in the natural way.

We frequently consider the (k-)down-up walk Markov chain. This Markov chain can be defined for any measure π supported on $\mathcal { X } _ { k } ^ { d }$ where $k \in [ d ]$ . We denote its transitions by ${ \mathcal { T } } ^ { \mathsf { D } \mathsf { U } , \pi }$ and k is inferred from the definition of π. The transition $\mathcal { T } _ { \mathbf { x } } ^ { \mathsf { D U } , \pi }$ is defined as follows for $\mathbf { x } \in \mathcal { X } _ { k } ^ { d }$ , where $S : = \mathrm { s e t } ( \mathbf { x } )$

1. (Down step.) A uniformly random $T \subseteq S$ with $| T | = k - 1$ is chosen.

2. (Up step.) A set $S ^ { \prime } \supseteq T$ with $| S ^ { \prime } | = k$ is sampled $\propto \pi ( S ^ { \prime } )$ , and we step to $\mathbf { x } ^ { \prime } = \mathbf { v e c } ( S ^ { \prime } )$

It is standard that ${ \mathcal { T } } ^ { \mathsf { D } \mathsf { U } , \pi }$ is reversible with stationary measure π (Definitions 6 and 7, and Corollary 11, [KO20]). More formal pseudocode is provided in Algorithm 1.

Spectral theory. Let T be a reversible Markov chain, and have stationary measure $\pi \in \mathscr { P } ( \Omega )$ We define the associated Dirichlet form by its action on two functions $f , g : \Omega \to \mathbb { R } : ^ { 4 }$

$$
\mathcal { E } _ { \mathcal { T } } ( f , g ) : = \int f ( \omega ) g ( \omega ) \pi ( \omega ) \mathrm { d } \omega - \iint f ( \omega ) g ( \omega ^ { \prime } ) \pi ( \omega ) \mathcal { T } _ { \omega } ( \omega ^ { \prime } ) \mathrm { d } \omega \mathrm { d } \omega ^ { \prime } ,
$$

and note that the following identity holds:

$$
\mathcal { E } _ { \mathcal { T } } ( f , f ) = \frac { 1 } { 2 } \iint ( f ( \omega ) - f ( \omega ^ { \prime } ) ) ^ { 2 } \pi ( \omega ) \mathcal { T } _ { \omega } ( \omega ^ { \prime } ) \mathrm { d } \omega \mathrm { d } \omega ^ { \prime } .\tag{8}
$$

For $\lambda \in ( 0 , 2 ]$ , we say that T satisfies a λ-Poincaré inequality if, for all $f : \Omega  \mathbb { R }$

$$
\operatorname { V a r } _ { \pi } [ f ] \leq { \frac { 1 } { \lambda } } { \mathcal { E } } _ { { \mathcal { T } } } ( f , f ) .
$$

Bounding the Poincaré constant λ results in an estimate of the mixing time of the Markov chain through the comparison inequality $\chi ^ { 2 } \left( \mu \| \boldsymbol { \pi } \right) \geq 4 \mathrm { T V } \left( \mu , \boldsymbol { \pi } \right) ^ { 2 }$ , and the following fact.

Lemma 1 (Chapters 12 and 13, [LPW09]). Let T be a reversible Markov chain with stationary measure $\pi ,$ and let lazy(T) denote the Markov chain that in each step transitions according to T with $p r o b a b i l i t y \ { \frac { 1 } { 2 } }$ , and otherwise does not transition. Then if T satisfies a λ-Poincaré inequality, if we let $\pi _ { t }$ be the law of an iterate taking $t \in \mathbb N$ steps of lazy(T) starting from $\pi _ { 0 }$ , we have

$$
\chi ^ { 2 } ( \pi _ { t } \| \pi ) \leq \left( 1 - \frac { \lambda } { 2 } \right) ^ { t } \chi ^ { 2 } ( \pi _ { 0 } \| \pi ) .
$$

Path coupling. In Section 3, we develop a new tool for bounding mixing times on fixed-magnetization or bounded-magnetization measures. This tool is based on path coupling, which we introduce here.

Lemma 2 (Path coupling). Let T be a Markov chain with stationary measure $\pi \in \mathcal { P } ( \Omega )$ Let m : $\Omega \times \Omega \to \mathbb { N } \cup \{ 0 \}$ be an integer-valued metric. Assume that for any ω, $\omega ^ { \prime } \in \Omega$ with m $( \omega , \omega ^ { \prime } ) = k$ there exists a path $\omega = \omega _ { 0 } , \omega _ { 1 } , \ldots , \omega _ { k } = \omega ^ { \prime }$ such that for every $i \in [ k ]$

$$
\begin{array} { r } { \mathrm { m } ( \omega _ { i - 1 } , \omega _ { i } ) = 1 \quad a n d \quad \mathcal { T } _ { \omega _ { i - 1 } } ( \omega _ { i } ) > 0 . } \end{array}\tag{9}
$$

Assume furthermore that there exists $\alpha \in ( 0 , 1 )$ such that for any $( \omega , \omega ^ { \prime } ) \in \Omega \times \Omega$ with m $( \omega , \omega ^ { \prime } ) = 1$ there exists $\gamma \in \Gamma ( \mathcal { T } _ { \omega } , \mathcal { T } _ { \omega ^ { \prime } } )$ with $\mathbb { E } ( \psi , \psi ^ { \prime } ) { \sim } \gamma [ \mathrm { m } ( \psi , \psi ^ { \prime } ) ] \leq 1 - \alpha$ . Then for any $\pi _ { 0 } \in \mathscr { P } ( \Omega )$ and $\epsilon \in ( 0 , \frac { 1 } { 2 } )$ ，

$$
\mathrm { T V } \left( \boldsymbol { T } ^ { T } \boldsymbol { \pi } _ { 0 } , \boldsymbol { \pi } \right) \leq \epsilon , \ f o r T \geq \frac { 1 } { \alpha } \left( \log \frac { \mathrm { d i a m } ( \Omega ) } { \epsilon } \right) , \ w h e r e \ \mathrm { d i a m } ( \Omega ) : = \operatorname* { s u p } _ { \omega , \omega ^ { \prime } \in \Omega \times \Omega } \mathrm { m } ( \omega , \omega ^ { \prime } ) .
$$

Proof. Fix any pair of initial states $\omega , \omega ^ { \prime } \in \Omega$ and let $k : = \mathrm { m } ( \omega , \omega ^ { \prime } )$ Choose a path $\omega =$ $\omega _ { 0 } , \omega _ { 1 } , \ldots , \omega _ { k } = \omega ^ { \prime }$ meeting the conditions (9). For all $i \in [ k ]$ let $\gamma _ { i } \in \Gamma ( \mathcal { T } _ { \omega _ { i - 1 } } , \mathcal { T } _ { \omega _ { i } } )$ satisfy

$$
\begin{array} { r } { \mathbb { E } _ { ( \psi , \psi ^ { \prime } ) \sim \gamma _ { i } } \left[ \mathrm { m } ( \psi , \psi ^ { \prime } ) \right] \leq 1 - \alpha . } \end{array}
$$

From these couplings we can define a joint measure µ over the product space of $\psi _ { i } \sim \mathcal { T } _ { \omega _ { i } }$ for all $0 \leq i \leq k$ , such that each marginal on adjacent pairs $( \omega _ { i - 1 } , \omega _ { i } )$ agrees with $\gamma _ { i }$ . This construction is standard and follows from the “gluing lemma” on couplings: see e.g., Lemma 14.3, [LPW09].

Now draw $( \psi _ { 0 } , \ldots , \psi _ { k } ) \sim \mu$ . We have

$$
\mathbb { E } _ { \mu } \left[ \mathbf { m } ( \psi _ { 0 } , \psi _ { k } ) \right] \leq \mathbb { E } _ { \mu } \left[ \sum _ { i \in [ k ] } \mathbf { m } ( \psi _ { i - 1 } , \psi _ { i } ) \right] = \sum _ { i \in [ k ] } \mathbb { E } _ { \gamma _ { i } } \left[ \mathbf { m } ( \psi _ { i - 1 } , \psi _ { i } ) \right] \leq ( 1 - \alpha ) \mathbf { m } ( \omega , \omega ^ { \prime } ) .
$$

This gives a one-step coupling showing a contraction in m. Thus, after $\begin{array} { r } { T \geq \frac { 1 } { \alpha } \log \frac { \mathrm { d i a m } ( \Omega ) } { \epsilon } } \end{array}$ steps, if we independently draw $( \omega _ { 0 } , \omega _ { 0 } ^ { \prime } ) \sim \pi _ { 0 } \times \pi$ , and then iterate the above construction to produce coupled iterates $( \omega _ { t } , \omega _ { t } ^ { \prime } )$ for all $t \in [ T ]$ such that $\omega _ { t } \sim \mathcal T _ { \omega _ { t - 1 } }$ and $\omega _ { t } ^ { \prime } \sim \mathcal { T } _ { \omega _ { t - 1 } ^ { \prime } }$ , we have

$$
\mathbb { E } \left[ \mathrm { m } ( \omega _ { T } , \omega _ { T } ^ { \prime } ) \right] \leq ( 1 - \alpha ) ^ { T } \mathbb { E } \left[ \mathrm { m } ( \omega _ { 0 } , \omega _ { 0 } ^ { \prime } ) \right] \leq ( 1 - \alpha ) ^ { T } \mathrm { d i a m } ( \Omega ) \leq \epsilon .
$$

Because m takes values in $\mathbb { N } \cup \{ 0 \}$ , and we have $\operatorname { m } ( \omega , \omega ^ { \prime } ) \geq 1$ whenever $\omega \neq \omega ^ { \prime }$

$$
\mathbb { I } ( \omega _ { T } \neq \omega _ { T } ^ { \prime } ) \le \ m ( \omega _ { T } , \omega _ { T } ^ { \prime } ) .
$$

The conclusion follows from the coupling definition of total variation, as ω<sub>T</sub> $\sim \mathcal { T } ^ { T } \pi _ { 0 } , \omega _ { T } ^ { \prime } \sim \mathcal { T } ^ { T } \pi$ .

## 2.3 Statistical models

We describe the statistical models that induce the main structured distributions we consider.

Ising model. An Ising model is specified by an interaction matrix $\mathbf { J } \in \mathbb { S } ^ { d \times d }$ and optionally, an external field $\mathbf { h } \in \mathbb { R } ^ { d }$ . It induces a Gibbs measure π over $\chi ^ { d }$ , with an additional parameter $\beta > 0$ governing the inverse temperature:

$$
\pi ( \mathbf x ) \propto \exp \left( \beta \left( \frac { 1 } { 2 } \mathbf x ^ { \top } \mathbf J \mathbf x + \mathbf h ^ { \top } \mathbf x \right) \right) \cdot \mathbb { I } ( \mathbf x \in \mathcal X ^ { d } ) .\tag{10}
$$

We often refer to fixed-magnetization or bounded-magnetization Ising models, where the magnetization level $k \in [ d ]$ is clear from context. Under these models, our goal is to sample from the Gibbs measure π in (10), further conditioned on $\mathbf { x } \in \mathcal { X } _ { k } ^ { d }$ or $\mathbf { x } \in \mathcal { X } _ { \leq k } ^ { d }$ respectively.

The literature on Ising models considers a variety of statistical models for the interaction matrix.   
We summarize a few standard parameterizations here.

Model 1 (Sherrington-Kirkpatrick model, [SK75]). In the Sherrington-Kirkpatrick (SK) model, J is drawn from the Gaussian orthogonal ensemble GOE $( d ) \colon \mathbf { J } \in \mathbb { S } ^ { d \times d }$ has $\begin{array} { r } { \mathbf { J } _ { i j } \sim _ { \mathrm { i . i . d . } } N ( 0 , \frac { 1 } { d } ) } \end{array}$ for all $( i , j ) \in [ d ] \times [ d ]$ with $i < j$ , and $\begin{array} { r } { { \mathbf J } _ { i i } \sim _ { \mathrm { i . i . d . } } N ( 0 , \frac { 2 } { d } ) } \end{array}$ for all $i \in [ d ] . ^ { 5 }$

Model 2 (Gaussian Hopfield model, [BvEN99, BG08, $\mathrm { H R P ^ { + } 2 0 l } )$ . In the Gaussian Hopfield model,<sup>6</sup> J is drawn from the Wishart ensemble with n degrees of freedom: $\begin{array} { r } { \mathbf { J } \in \mathbb { S } ^ { d \times d } \ h a s \ \mathbf { J } = \frac { 1 } { n } \mathbf { G } ^ { \top } \mathbf { G } } \end{array}$ , where $\mathbf { G } \in \mathbb { R } ^ { n \times d }$ has $\mathbf { G } _ { i j } \sim _ { \mathrm { i . i . d . } } \mathcal { N } ( 0 , 1 )$ for all $( i , j ) \in [ n ] \times [ d ]$

Our algorithms’ guarantees will depend on an appropriate norm of J (and sometimes, h). Here we present some standard estimates for the random matrix ensembles in Models 1, 2.

Fact 1 (Section 2.5, [Ver18] and Theorem 2.3.5, [AGZ10]). For any $\delta \in ( 0 , \frac { 1 } { 2 } )$ , the following hold under Model 1 for an appropriate constant $C > 0$ , simultaneously with $p r o b a b i l i t y \ge 1 - \delta$

$$
\begin{array} { r } { 1 . \ \operatorname* { m a x } _ { ( i , j ) \in [ d ] \times [ d ] } | \mathbf { J } _ { i j } | \leq C \sqrt { \log ( d / \delta ) / d } . } \end{array}
$$

$$
\begin{array} { r } { \mathcal { Q } . \| \mathbf { J } \| _ { \mathrm { o p } } \leq 2 + C \sqrt { \log ( 1 / \delta ) / d } . } \end{array}
$$

Fact 2 (Lemma 6.26, [Wai19] and Exercise 4.7.3, [Ver18]). For any $\delta , \epsilon \in ( 0 , \frac { 1 } { 2 } )$ , the following hold under Model 2 for an appropriate constant $C > 0$ , simultaneously with $p r o b a b i l i t y \ge 1 - \delta$

$$
\begin{array} { r } { I . \ I f n \geq C \log ( \frac { d } { \delta } ) \cdot \frac { 1 } { \epsilon ^ { 2 } } , \operatorname* { m a x } _ { ( i , j ) \in [ d ] \times [ d ] } | { \bf J } _ { i j } - \mathbb { I } ( i = j ) | \leq \epsilon . } \end{array}
$$

$$
\begin{array} { r } { \mathrm { ~ \mathcal { Z } . ~ } I f n \geq C ( k \log ( \frac { e d } { k } ) + \log ( \frac { 1 } { \delta } ) ) \cdot \frac { 1 } { \epsilon ^ { 2 } } \mathrm { ~ } f o r \mathrm { ~ } k \in [ d ] , \mathrm { ~ } \| \mathbf { J } \| _ { k , \mathrm { o p } } \leq 1 + \epsilon . } \end{array}
$$

Remark 1. We focus on Gaussian G in Model 2 for simplicity, although Fact 2 holds for any $\begin{array} { r } { \mathbf { J } = \frac { 1 } { n } \mathbf { G } ^ { \top } \mathbf { G } } \end{array}$ where the entries of G are drawn i.i.d. from a 1-sub-Gaussian distribution (see Section ${ \it 2 . 5 , \Delta \Psi } \mathrm { \it / V e r } { \it 1 8 } \mathrm { \it / ) }$ . Our analyses only rely on the properties of Model 2 in Fact 2, so they apply to any sub-Gaussian ensemble as well. This captures other common Hopfield model instances, $e . g .$ Rademacher G as often considered in the associative memory literature [Lit74, PF77, Hop82].

Bayesian sparse linear regression. Our second main application considers Bayesian sparse linear regression, i.e., sampling from the posterior distribution of a sparse linear model. We focus on a canonical parameterization of the problem, induced by the spike-and-slab prior [MB88, GM93] (see also [Chi96, Gew96]) and Gaussian measurement noise, a standard formulation recently studied by the sampling algorithms community [KSTZ25, MW26].

Model 3 (Spike-and-slab posterior sampling). Let $\mathbf { q } \in ( 0 , 1 ) ^ { d }$ and $\sigma > 0$ be known, and let $\bar { k } : =$ $\| \mathbf { q } \| _ { 1 }$ . Let $\mathbf { X } \in \mathbb { R } ^ { n \times d }$ have entries $\sim _ { \mathrm { i . i . d . } } \mathcal { N } ( 0 , \frac { 1 } { n } )$ , and suppose that we observe $( \mathbf { X } , \mathbf { y } )$ where

$$
\theta ^ { \star } \sim \pi : = \bigotimes _ { i \in [ d ] } \left( ( 1 - \mathbf { q } _ { i } ) \delta _ { 0 } + \mathbf { q } _ { i } \mathcal { N } ( 0 , 1 ) \right) , \quad \xi \sim \mathcal { N } ( \mathbf { 0 } _ { n } , \sigma ^ { 2 } \mathbf { I } _ { n } ) , \quad \mathbf { y } = \mathbf { X } \pmb { \theta } ^ { \star } + \pmb { \xi } ,\tag{11}
$$

and $\pmb { \theta } ^ { \star }$ and $\boldsymbol { \xi }$ are independent. Our goal is to sample from the posterior $\pi ( \cdot \mid \mathbf { X } , \mathbf { y } )$

When designing algorithms for Model 3, there is a reparameterization in terms of $\pi _ { \mathrm { s u p p } }$ the distribution of $\operatorname { s u p p } ( \pmb { \theta } ^ { \star } ) \mid \mathbf { X } , \mathbf { y }$ . Indeed, the main algorithmic challenge is sampling from π<sub>supp</sub>.

Fact 3 (Lemma 8, [KSTZ25]). $\theta ^ { \star } \sim \pi ( \cdot \mid \mathbf { X } , \mathbf { y } )$ in Model 3 can be equivalently generated as follows.

1. First, $S \subseteq [ d ]$ is sampled from $\pi _ { \mathrm { s u p p } }$ where

$$
\pi _ { \mathrm { s u p p } } ( S ) \propto \left( \prod _ { i \in S } { \frac { \mathbf { q } _ { i } } { 1 - \mathbf { q } _ { i } } } \right) \exp \left( { \frac { 1 } { 2 } } \left\| \mathbf { b } _ { S } \right\| _ { \mathbf { A } _ { S } ^ { - 1 } } ^ { 2 } \right) { \frac { 1 } { \sqrt { \operatorname* { d e t } \mathbf { A } _ { S } } } } ,\tag{12}
$$

$$
\mathbf { A } _ { S } \in \mathbb { R } ^ { S \times S } : = \frac { 1 } { \sigma ^ { 2 } } \left[ \mathbf { X } ^ { \top } \mathbf { X } \right] _ { S \times S } + \mathbf { I } _ { S } , \quad \mathbf { b } _ { S } \in \mathbb { R } ^ { S } : = \frac { 1 } { \sigma ^ { 2 } } \mathbf { X } _ { S : \mathbf { y } } ^ { \top } .
$$

2. Second, $\theta ^ { \star } \mid S , \mathbf { X } , \mathbf { y }$ is sampled from $\mathcal { N } ( \mathbf { A } _ { S } ^ { - 1 } \mathbf { b } _ { S } , \mathbf { A } _ { S } ^ { - 1 } )$

## 3 Sparse Dobrushin Condition

In this section, we give our first technical result: a simple suficient condition for rapid mixing of the down-up walk, patterned of of the classical Dobrushin uniqueness condition [Dob68, Wu06].

The applications of our sparse Dobrushin framework to Ising models (Theorem 3 and Corollary 1) in this section generally obtain weaker parameter tradeofs than our framework in Section 4 does. Nonetheless, this section serves as a useful proof-of-concept of the temperature improvements achievable in fixed-magnetization settings. We include Theorem 3 both due to its ease of applicability, and because the samplers resulting from it are formally incomparable to those in Section 4, as it trades of a faster mixing time for a stricter requirement on relevant parameters.

To ease notation, this section works in a more abstract formulation than in Section 2.2, where we use the down-up walk to sample subsets $S \subseteq { \mathcal { U } } .$ , from a distribution π supported on

$$
\mathcal { U } _ { k } : = \{ S \subseteq \mathcal { U } : | S | = k \} .\tag{13}
$$

Here, U is a discrete universe of candidate elements. This straightforwardly captures the setting of the down-up walk in Section 2.2 by equating $\mathcal { U } \equiv [ d ]$ and $S \equiv \mathbf { x } : = \mathbf { v e c } ( S )$ . We provide pseudocode implementing a one-step transition of the down-up walk in Algorithm 1. For brevity, we define

$$
\operatorname { d o w n } ( S ) : = \{ T \in \mathcal { U } _ { k - 1 } : T \subset S \} \mathrm { ~ f o r ~ a l l ~ } S \in \mathcal { U } _ { k } ,
$$

$\operatorname { u p } ( T ) : = \{ S \in \mathcal { U } _ { k } : S \supset T \}$ for all $T \in \mathcal { U } _ { k - 1 }$

Algorithm 1: DU(S, U, π)   
1 Input: $S \in \mathcal { U } _ { k }$ , discrete universe U, $\pi \in \mathcal { P } ( \mathcal { U } _ { k } )$   
2 Output: Sample $S ^ { \prime } \in \mathcal { U } _ { k }$ from $\tau _ { S } ^ { \mathsf { D U } , \pi }$   
3 $T \sim _ { \mathrm { u n i f } } .$ down(S)   
4 $S ^ { \prime } \sim \pi ( \cdot \mid \cdot \in \operatorname { u p } ( T ) )$   
5 return $S ^ { \prime }$

We state our general framework in Section 3.1 and apply it to Ising models in Section 3.2.

## 3.1 Basic analysis

We begin by stating our new sparse Dobrushin condition.

Definition 1 (Sparse Dobrushin condition). Let $\pi \in \mathcal { P } ( \mathcal { U } _ { k } )$ with full support where U is a discrete universe with $| \mathcal { U } | \geq k _ { \ast }$ , and let $\alpha \in ( 0 , 1 )$ . We say π satisfies an α-sparse Dobrushin condition if

$$
\begin{array} { r } { \operatorname { T V } \left( \pi _ { U \parallel V } , \pi _ { V \parallel U } \right) \leq \alpha , \ f o r \ a l l \ ( U , V ) \in \mathcal { U } _ { k - 1 } \times \mathcal { U } _ { k - 1 } \ w i t h \ \Delta _ { \operatorname { H a m } } ( U , V ) = 1 , } \end{array}
$$

where for all $( U , V ) \in \mathcal { U } _ { k - 1 } \times \mathcal { U } _ { k - 1 }$ , we define $\pi _ { U \parallel V } ( i ) \in { \mathcal { P } } ( { \mathcal { U } } \setminus ( U \cup V ) )$ by

$$
\pi _ { U \parallel V } ( i ) : = \frac { \pi ( U \cup \{ i \} ) } { \sum _ { j \in \mathcal { U } \backslash ( U \cup V ) } \pi ( U \cup \{ j \} ) }\tag{14}
$$

The utility of Definition 1 reveals itself through the following bound.

Lemma 3. Assume that $\pi \in \mathcal { P } ( \mathcal { U } _ { k } )$ satisfies an α-sparse Dobrushin condition, and let $( S , T ) \in$ $\mathcal { U } _ { k } \times \mathcal { U } _ { k } \ s a t i s f y \ \Delta _ { \mathrm { H a m } } ( S , T ) = 1$ . Then there exists a coupling $\gamma \in \Gamma ( \mathcal { T } _ { S } ^ { \mathsf { D U } , \pi } , \mathcal { T } _ { T } ^ { \mathsf { D U } , \pi } )$ satisfying

$$
\mathbb { E } _ { ( S ^ { \prime } , T ^ { \prime } ) \sim \gamma } \left[ \Delta _ { \mathrm { H a m } } ( S ^ { \prime } , T ^ { \prime } ) \right] \leq 1 - \frac { 1 } { k } + \alpha .
$$

Proof. Let $W : = S \cap T$ and assume $S = W \cup \{ s \}$ and $T = W \cup \{ t \}$ . We construct the coupling explicitly. First, for the down step (Line 3) we couple the transitions as follows.

1. Draw $u \sim _ { \mathrm { u n i f . } } W$ and $r \sim _ { \mathrm { u n i f . } } [ 0 , 1 ]$ independently.

2. If $\begin{array} { r } { r < \frac { 1 } { k } . } \end{array}$ , we set $S ^ { \downarrow } = T ^ { \downarrow } = W$ . If $\begin{array} { r } { r \geq \frac { 1 } { k } } \end{array}$ , we set $S ^ { \downarrow } = S \setminus \{ u \}$ and $T ^ { \downarrow } = T \setminus \{ u \}$

If $S ^ { \downarrow } = T ^ { \downarrow }$ , we can perfectly couple the subsequent up step (Line 4), yielding $\Delta _ { \mathrm { H a m } } ( S ^ { \prime } , T ^ { \prime } ) = 0$

In the other case, $S ^ { \downarrow } \neq T ^ { \downarrow }$ . Let $\rho$ denote the optimal coupling of $( \pi _ { S ^ { \downarrow } \parallel T ^ { \downarrow } } , \pi _ { T ^ { \downarrow } \parallel S ^ { \downarrow } } )$ (see Definition 1) inducing their TV distance. Define the marginal probabilities of selecting the disjoint elements as

$$
q _ { S ^ { \downarrow } } = \frac { \pi ( S ^ { \downarrow } \cup \{ t \} ) } { \sum _ { j \notin S ^ { \downarrow } } \pi ( S ^ { \downarrow } \cup \{ j \} ) } , \quad q _ { T ^ { \downarrow } } = \frac { \pi ( T ^ { \downarrow } \cup \{ s \} ) } { \sum _ { j \notin T ^ { \downarrow } } \pi ( T ^ { \downarrow } \cup \{ j \} ) } .
$$

Without loss of generality (by symmetry of the statement), assume $q _ { S ^ { \downarrow } } \geq q _ { T ^ { \downarrow } }$

For the up step (Line 4) in the case $S ^ { \downarrow } \ne T ^ { \downarrow }$ , we couple the transitions as follows.

1. With probability $q _ { T ^ { \downarrow } }$ , set $S ^ { \prime } = T ^ { \prime } = S ^ { \downarrow } \cup \{ t \} = T ^ { \downarrow } \cup \{ s \}$

2. With probability $q _ { S ^ { \downarrow } } - q _ { T ^ { \downarrow } } ,$ set $S ^ { \prime } = S ^ { \downarrow } \cup \{ t \}$ , draw $j \sim \pi _ { T ^ { \downarrow } \parallel S ^ { \downarrow } }$ , and set $T ^ { \prime } { = } T ^ { \downarrow } \cup \{ j \}$

3. With probability $1 - q _ { S ^ { \downarrow } }$ , draw $( j , j ^ { \prime } ) \sim \rho ,$ and set $S ^ { \prime } = S ^ { \downarrow } \cup \{ j \}$ and $T ^ { \prime } { = } T ^ { \downarrow } \cup \{ j ^ { \prime } \}$

We verify that this is a coupling. The first marginal sets $S ^ { \prime } = S ^ { \downarrow } \cup \{ t \}$ with probability $q _ { S ^ { \downarrow } }$ and otherwise samples from the correct conditional distribution over $\mathcal { U } \setminus T \cup \{ u \}$ . Similarly, the second marginal sets $T ^ { \prime } = T ^ { \downarrow } \cup \{ s \}$ correctly, and otherwise samples from the correct conditional distribution over $\mathcal { U } \setminus S \cup \{ t \}$ . Also, in the third case, the probability $\Delta _ { \mathrm { H a m } } ( S ^ { \prime } , T ^ { \prime } ) \neq 1$ is

$$
\mathbb { P } \left[ j \neq j ^ { \prime } \right] \leq \alpha .
$$

Finally, we can compute the total expected Hamming distance:

$$
\begin{array} { r l r } {  { \mathbb { E } _ { ( S ^ { \prime } , T ^ { \prime \prime } ) \sim \gamma } [ \Delta _ { \mathrm { H a m } } \big ( S ^ { \prime } , T ^ { \prime } \big ) ] \le ( 1 - \frac { 1 } { k } ) ( q _ { T ^ { \downarrow } } \cdot 0 + ( q _ { S ^ { \downarrow } } - q _ { T ^ { \downarrow } } ) \cdot 1 + ( 1 - q _ { S ^ { \downarrow } } ) \cdot ( ( 1 - \alpha ) + 2 \alpha ) ) } } \\ & { } & \\ & { } & { \le ( 1 - \frac { 1 } { k } ) ( 1 + \alpha ) \le 1 - \frac { 1 } { k } + \alpha . } \end{array}\tag{15}
$$

It is clear that the down-up walk over $\mathcal { U } _ { k }$ satisfies the condition in (9) with $\mathrm { ~ m ~ } = \Delta _ { \mathrm { H a m } }$ . Thus, applying Lemma 3 within the framework of Lemma 2 gives the following result.

Proposition 1. Let $\pi \in \mathcal { P } ( \mathcal { U } _ { k } )$ satisfy a $\alpha \leq \frac { 1 } { 2 k }$ -sparse Dobrushin condition, and let $\epsilon \in ( 0 , \frac { 1 } { 2 } )$ . Then $\begin{array} { r } { i f T = \Omega ( k \log \frac { k } { \epsilon } ) } \end{array}$ for a suficiently large constant, we have for any $\pi _ { 0 } \in \mathcal { P } ( \mathcal { U } _ { k } )$ ，

$$
\operatorname { T V } \left( ( { \mathcal { T } } ^ { \mathsf { D U } , \pi } ) ^ { T } \pi _ { 0 } , \pi \right) \leq \epsilon .
$$

Of course, the parameter α in Proposition 1 can be taken to be any constant factor smaller than $\textstyle { \frac { 1 } { k } }$ Proposition 1 also explains our naming choice, as its conclusion becomes stronger (as a function of the sparse Dobrushin condition parameter) as k gets smaller, i.e., π is supported on sparser sets.

## 3.2 Fixed-magnetization Ising models

In this section, we demonstrate how to apply Proposition 1 to fixed-magnetization Ising models:

$$
\pi ( \mathbf x ) \propto \exp \left( \beta \left( \frac { 1 } { 2 } \mathbf x ^ { \top } \mathbf J \mathbf x + \mathbf h ^ { \top } \mathbf x \right) \right) \cdot \mathbb { I } _ { \mathbf x \in \mathcal X _ { k } ^ { d } } .\tag{16}
$$

We first require a helper tool to control the sparse Dobrushin condition parameter.

Lemma 4. Let $\pi \in \mathcal { P } ( \Omega ) , \mu \in \mathcal { P } ( \Omega )$ have $\pi \propto P$ and $\mu \propto Q$ for unnormalized densities $P , Q$ . Then

$$
\operatorname* { s u p } _ { \omega \in \Omega } \left| \log \frac { P ( \omega ) } { Q ( \omega ) } \right| \le \Delta \implies \mathrm { T V } \left( \pi , \mu \right) \le \Delta .
$$

Proof. First observe that

$$
\operatorname* { s u p } _ { \omega \in \Omega } \left| \log \frac { \pi ( \omega ) } { \mu ( \omega ) } \right| \leq \operatorname* { s u p } _ { \omega \in \Omega } \left| \log \frac { P ( \omega ) } { Q ( \omega ) } \right| + \left| \log \frac { \int _ { \Omega } Q ( \omega ^ { \prime } ) \mathrm { d } \omega ^ { \prime } } { \int _ { \Omega } P ( \omega ^ { \prime } ) \mathrm { d } \omega ^ { \prime } } \right| \leq 2 \operatorname* { s u p } _ { \omega \in \Omega } \left| \log \frac { P ( \omega ) } { Q ( \omega ) } \right| \leq 2 \Delta .
$$

Let $\begin{array} { r } { L ( \omega ) : = \frac { \pi ( \omega ) } { \mu ( \omega ) } } \end{array}$ so $\exp ( - 2 \Delta ) \le L ( \omega ) \le \exp ( 2 \Delta )$ for all $\omega \in \Omega$ . Using $\pi = L \mu$

$$
\mathrm { T V } \left( \pi , \mu \right) = \frac { 1 } { 2 } \int _ { \Omega } \left| \pi ( \omega ) - \mu ( \omega ) \right| \mathrm { d } \omega = \frac { 1 } { 2 } \int _ { \Omega } \mu ( \omega ) \left| L ( \omega ) - 1 \right| \mathrm { d } \omega = \frac { 1 } { 2 } \mathbb { E } _ { \mu } \left[ \left| L - 1 \right| \right] .
$$

Next we use the following convexity bound: if ϕ is convex on $[ a , b ]$ and random variable $X \in [ a , b ]$ then writing $X = t a + ( 1 - t ) b$ with $\begin{array} { r } { t = \frac { b - X } { b - a } \in [ 0 , 1 ] } \end{array}$ gives

$$
\phi ( X ) \leq t \phi ( a ) + ( 1 - t ) \phi ( b ) .
$$

Taking expectations yields

$$
\mathbb { E } [ \phi ( X ) ] \leq \frac { b - \mathbb { E } [ X ] } { b - a } \phi ( a ) + \frac { \mathbb { E } [ X ] - a } { b - a } \phi ( b ) .
$$

We apply this bound with $X = L , \mathbb { E } _ { \mu } [ L ] = 1 , \phi ( u ) = | u - 1 | , \mathrm { a n d } [ a , b ] = [ \exp ( - 2 \Delta ) , \exp ( 2 \Delta ) ]$ Since $a < 1 < b , \phi ( a ) = 1 - a$ and $\phi ( b ) = b - 1$ , so

$$
\mathrm { T V } \left( \pi , \mu \right) = \frac { 1 } { 2 } \mathbb { E } _ { \mu } [ \phi ( L ) ] \leq \frac { ( b - 1 ) ( 1 - a ) } { b - a } .
$$

Now set $\textstyle r : = { \frac { b } { a } } > 1$ and write $b = a r$ . Since $a \leq 1 \leq a r$ , we have $a \in [ \textstyle { \frac { 1 } { r } } , 1 ]$ . Define

$$
f _ { r } ( a ) : = { \frac { ( a r - 1 ) ( 1 - a ) } { a r - a } } = { \frac { ( r a - 1 ) ( 1 - a ) } { a ( r - 1 ) } } .
$$

A direct derivative computation shows that $f _ { r }$ is maximized at $a = r ^ { - 1 / 2 }$ , and hence

$$
\mathrm { T V } \left( \pi , \mu \right) \leq f _ { r } ( r ^ { - 1 / 2 } ) = \frac { \sqrt { r } - 1 } { \sqrt { r } + 1 } = \operatorname { t a n h } \left( \frac { 1 } { 4 } \log r \right) \leq \Delta .
$$

Lemma 4 lets us conclude that for fixed-magnetization Ising models $( \mathrm { i . e . , ~ ( 1 0 ) }$ restricted to $\boldsymbol { \mathcal { X } } _ { k } ^ { d } )$ , the sparse Dobrushin condition parameter can be controlled by the largest of-diagonal entry of J.

Lemma 5. Let π be a fixed-magnetization Ising model (16) with

$$
\beta \operatorname* { m a x } _ { ( i , j ) \in [ d ] \times [ d ] } | \mathbf { J } _ { i j } | \leq \alpha .
$$

Then π satisfies a 8α-sparse Dobrushin condition.

Proof. Write $\mathbf { J }  \beta \mathbf { J }$ for simplicity in this proof, so we will prove the result when $\beta = 1$ without loss of generality. Let $( \mathbf { v e c } ( U ) , \mathbf { v e c } ( V ) ) \in \mathcal { X } _ { k - 1 } ^ { d } \times \mathcal { X } _ { k - 1 } ^ { d }$ have $\Delta _ { \mathrm { H a m } } ( U , V ) = 1$ , and denote $W : = U \cap V .$ $U = W \cup \{ u \}$ , and $V = W \cup \{ v \}$ . Note that the distributions $\pi _ { U \parallel V }$ and $\pi _ { V \| U }$ defined in (14) are supported on the same set $\Omega : = [ d ] \setminus ( U \cup V )$ , so we are in the setting of Lemma 4.

Let $i \in [ d ] \setminus ( U \cup V ) , X : = U \cup \{ i \} , Y : = V \cup \{ i \}$ , and $\mathbf { x } : = \mathbf { v e c } ( X ) = 2 \mathbf { 1 } _ { X } - \mathbf { 1 } _ { d } , \mathbf { y } : = \mathbf { v e c } ( Y ) =$ $2 { \bf 1 } _ { Y } - { \bf 1 } _ { d }$ . Observe that by expanding definitions and cancelling similar terms,

$$
\begin{array} { r l } & { \log \left( \frac { \exp \left( \frac { 1 } { 2 } { \bf x } ^ { \top } { \bf J } { \bf x } + { \bf h } ^ { \top } { \bf x } \right) } { \exp \left( \frac { 1 } { 2 } { \bf x } ^ { \top } { \bf J } { \bf y } + { \bf h } ^ { \top } { \bf y } \right) } \right) = \left( \frac { 1 } { 2 } { \bf x } ^ { \top } { \bf J } { \bf x } + { \bf h } ^ { \top } { \bf x } \right) - \left( \frac { 1 } { 2 } { \bf y } ^ { \top } { \bf J } { \bf y } + { \bf h } ^ { \top } { \bf y } \right) } \\ & { \qquad = \left( { \bf 2 1 } _ { X } ^ { \top } { \bf J } { \bf 1 } _ { X } + 2 \left( { \bf h } - { \bf J } { \bf 1 } _ { d } \right) ^ { \top } { \bf 1 } _ { X } + \frac { 1 } { 2 } { \bf 1 } _ { d } ^ { \top } { \bf J } { \bf 1 } _ { d } - { \bf h } ^ { \top } { \bf 1 } _ { d } \right) } \\ & { \qquad - \left( { \bf 2 1 } _ { Y } ^ { \top } { \bf J } { \bf 1 } _ { Y } + 2 \left( { \bf h } - { \bf J } { \bf 1 } _ { d } \right) ^ { \top } { \bf 1 } _ { Y } + \frac { 1 } { 2 } { \bf 1 } _ { d } ^ { \top } { \bf J } { \bf 1 } _ { d } - { \bf h } ^ { \top } { \bf 1 } _ { d } \right) } \\ & { \qquad = 4 \left( \mathbf { e } _ { i } + \mathbf { e } _ { W } \right) ^ { \top } { \bf J } ( { \bf e } _ { u } - { \bf e } _ { v } ) + 2 \left( \mathbf { J } _ { u u } - { \bf J } _ { v v } \right) + 2 ( { \bf h } - { \bf J } { \bf 1 } _ { d } ) ^ { \top } ( { \bf e } _ { u } - { \bf e } _ { v } ) } \\ &  \qquad = 4 { \bf e } _ { i } ^ { \top } { \bf J } \end{array}
$$

Further, we have

$$
\begin{array} { r } { \pi _ { U \parallel V } ( X ) \propto \exp \left( \frac { 1 } { 2 } \mathbf { x } ^ { \top } \mathbf { J x } + \mathbf { h } ^ { \top } \mathbf { x } \right) \propto \exp \left( \frac { 1 } { 2 } \mathbf { x } ^ { \top } \mathbf { J x } + \mathbf { h } ^ { \top } \mathbf { x } - 2 \mathbf { J } _ { u u } - 2 ( \mathbf { h } + \mathbf { J } ( 2 \mathbf { 1 } _ { W } - \mathbf { 1 } _ { d } ) ) ^ { \top } \mathbf { e } _ { u } \right) , } \\ { \pi _ { V \parallel U } ( Y ) \propto \exp \left( \frac { 1 } { 2 } \mathbf { y } ^ { \top } \mathbf { J y } + \mathbf { h } ^ { \top } \mathbf { y } \right) \propto \exp \left( \frac { 1 } { 2 } \mathbf { y } ^ { \top } \mathbf { J y } + \mathbf { h } ^ { \top } \mathbf { y } - 2 \mathbf { J } _ { v v } - 2 ( \mathbf { h } + \mathbf { J } ( 2 \mathbf { 1 } _ { W } - \mathbf { 1 } _ { d } ) ) ^ { \top } \mathbf { e } _ { v } \right) . } \end{array}
$$

Above we used that $W$ and u are fixed in the definition of $\pi _ { U \parallel V } ( \cdot )$ , so we can bring terms involving only these indices into the unnormalized density; a similar argument holds for $\pi _ { V \| U } ( \cdot )$ . Thus, Lemma 4 applies with $( P , Q )$ set to the right-hand sides above, so its conclusion holds with

$$
\Delta = \operatorname* { m a x } _ { \mathbf { \Phi } ( u , v , i ) \in [ d ] \times [ d ] \times [ d ] } \left. 4 \mathbf { e } _ { i } ^ { \top } \mathbf { J } ( \mathbf { e } _ { u } - \mathbf { e } _ { v } ) \right. \leq 8 \alpha .
$$

Importantly, the bound in Lemma 5 is independent of h. Intuitively, this follows because the conditional distributions in (14) exclude all elements in $U \cup V$ , including the non-shared elements, which induce the only diference in the external field. By combining Lemma 5 and Proposition 1, we thus obtain a mixing time bound for fixed-magnetization Ising models.

Theorem 3. Let $\delta \in ( 0 , \frac { 1 } { 2 } )$ and let π be induced by a fixed-magnetization Ising model (16) satisfying

$$
\beta \operatorname* { m a x } _ { \mathbf { \Phi } ( i , j ) \in [ d ] \times [ d ] \atop { i \neq j } } | \mathbf { J } _ { i j } | \leq \frac { 1 } { 1 6 k } .\tag{17}
$$

Then $\begin{array} { r } { i f T = \Omega ( k \log \frac { k } { \delta } ) } \end{array}$ for a suficiently large constant, we have for any $\pi _ { 0 } \in \mathcal { P } ( \mathcal { X } _ { k } ^ { d } )$ ，

$$
\begin{array} { r } { \mathrm { T V } \left( ( \mathcal { T } ^ { \mathsf { D U } , \pi } ) ^ { T } \pi \phantom { } _ { 0 } , \pi \right) \leq \delta . } \end{array}
$$

We conclude by briefly stating example implications of Theorem 3 and Lemma 5 for sampling from the Gibbs distributions induced by Models 1 and 2.

Corollary 1. Let $\delta \in ( 0 , \frac { 1 } { 2 } )$ and let π be defined as in (16). If $\pi _ { 0 } \in \mathcal { P } ( \mathcal { X } _ { k } ^ { d } )$ and $\begin{array} { r } { T = \Omega ( k \log { \frac { k } { \delta } } ) } \end{array}$ for an appropriate constant, $\bar { \mathrm { T V } } ( ( \mathcal { T } ^ { \mathsf { D U } , \pi } ) ^ { T } \pi _ { 0 } , \pi ) \leq \delta$ , under any of the following conditions.

1. Under Model 1 with probability ≥ 1 − δ, $\begin{array} { r } { i f \beta = O ( \frac { 1 } { k } \sqrt { d / \log ( d / \delta ) } ) } \end{array}$ for an appropriate constant.

2. Under Model 2 with probability $\begin{array} { r } { \geq 1 - \delta , \ i f \ n = \Omega ( \operatorname* { m a x } ( 1 , ( \beta k ) ^ { 2 } ) \log \frac { d } { \delta } ) } \end{array}$ for an appropriate constant.

Proof. It sufices to use high-probability bounds on the maximum of-diagonal entry of J (via Facts 1 and 2), combined with Theorem 3 and Lemma 5. □

We give a brief discussion of the runtime of Algorithm 1 in the Ising model setting.

Remark 2 (Runtime of down-up walk for Ising model). For the Ising model, Algorithm 1 can be implemented in time $O ( d )$ per iteration, after $O ( d ^ { 2 } )$ time preprocessing. We believe this is folklore, but sketch a proof here. Our implementation maintains a set $S \subseteq [ d ]$ and applies Algorithm 1 to it. Clearly, Line 3 is implementable in $O ( k )$ time. Next, the law $o f \left\{ i \right\} = S ^ { \prime } \setminus T$ in Line 4 is

$$
\propto \exp \left( \frac { \beta } { 2 } \left( 2 \mathbf { 1 } _ { T \cup \{ i \} } - \mathbf { 1 } _ { d } \right) ^ { \top } \mathbf { J } \left( 2 \mathbf { 1 } _ { T \cup \{ i \} } - \mathbf { 1 } _ { d } \right) + \beta \left. \mathbf { h } , 2 \mathbf { 1 } _ { T \cup \{ i \} } - \mathbf { 1 } _ { d } \right. \right)
$$

$$
\propto \exp \left( \beta \left( 2 { \bf J } _ { i i } + 4 \left. { \bf J } { \bf e } _ { i } , { \bf 1 } _ { T } \right. - 2 \left. { \bf J } { \bf e } _ { i } , { \bf 1 } _ { d } \right. + 2 { \bf h } _ { i } \right) \right) .
$$

Therefore, it is enough to maintain the quantities, for all $i \in [ d ] \setminus T$

$$
\langle \mathbf { J } \mathbf { e } _ { i } , \mathbf { 1 } _ { T } \rangle = \sum _ { j \in T } \mathbf { J } _ { i j } , \quad \mathbf { J } _ { i i } , \quad \langle \mathbf { J } \mathbf { 1 } _ { d } , \mathbf { e } _ { i } \rangle , \quad \mathbf { h } _ { i } .
$$

After $O ( d ^ { 2 } )$ time preprocessing, we store the latter three (constant) quantities, and we can update the first in $O ( d )$ time per call to Algorithm 1, since at most 2 coordinates change in T. Finally, given these values the sampling on Line 4 can be performed in time $O ( d )$

## 4 Spectral Mixing via Trickle Down

In this section, we develop a second approach to prove mixing bounds on fixed-magnetization measures, based on the trickle down theorem [Opp18, AL20] from the literature on high-dimensional expanders (we recommend Section 5 of the excellent survey [GK23] as an introduction to this topic). This framework, summarized abstractly in Section 4.1, achieves tighter parameter tradeofs in our applications to Ising models than its sparse Dobrushin counterpart in Section 3.

In Section 4.2, we begin by giving a more interpretable suficient condition for our framework, and two applications, as warmups. Specifically, we show that our trickle down framework qualitatively subsumes the sparse Dobrushin condition, and implies fast mixing for the SK model (Model 1) in the near-proportional magnetization regime, $\begin{array} { r } { k = O _ { \beta } ( \frac { d } { \log d } ) } \end{array}$ . We also derive an analogous result for the Gaussian Hopfield model (Model 2). Finally, we conclude with our strongest result on the SK model, handling the proportional regime $k = O _ { \beta } ( d )$ , in Section 4.3.

## 4.1 Trickle down framework

We develop our framework for analyzing fast mixing on fixed-magnetization Ising models in three parts. We begin by recalling preliminaries on spectral graph theory, and a statement of the trickle down theorem of [Opp18, AL20]. We then state a suficient condition (21) for applying the trickle down theorem, when the weights of the distribution in question are governed by a small perturbation of a product graph. Finally, we specialize this framework to Ising models.

Trickle down. Let [d] index a finite vertex set. For an edge weight matrix $\mathbf { W } \in \mathbb { R } _ { \geq 0 } ^ { d \times d }$ , we define its associated random walk matrix P to be the degree-normalized W, i.e.,

$$
\mathbf { P } : = \mathbf { D } ^ { - 1 } \mathbf { W } , { \mathrm { ~ w h e r e ~ } } \mathbf { D } : = \mathbf { d i a g } \left( \mathbf { W } \mathbf { 1 } _ { d } \right) .\tag{18}
$$

This section only considers reversible P, associated with symmetric edge weight matrices W.

We next state the trickle down theorem [Opp18, AL20], which bounds the Poincaré constant of the down-up walk induced by a measure in $\mathcal { P } ( \mathcal { X } _ { k } ^ { d } )$ , in terms of the worst spectral gap among certain restrictions of the measure. It is proven by using the law of total variance, which gives recursive relationships among the spectral gaps of various restricted down-up walks. We defer additional background to [Opp18, AL20], and simply state a suficient form for our purposes here.

For a measure $\pi \in \mathcal { P } ( \mathcal { X } _ { k } ^ { d } )$ and a set $R \subseteq [ d ]$ with $| R | = k - 2$ , define the link graph of R to be the weighted graph on vertices [d] \ R with edge weight matrix W given by

$$
\mathbf { W } _ { i j } = \pi ( R \cup \{ i , j \} ) { \mathrm { ~ f o r ~ a l l ~ } } ( i , j ) \in ( [ d ] \setminus R ) \times ( [ d ] \setminus R ) , \ i \neq j .
$$

Here we associate the set $R \cup \{ i , j \}$ with an element of $\mathcal { X } _ { k } ^ { d }$ with those positive coordinates, per our convention. We denote the associated random walk matrix by $\mathbf { P } _ { R }$ , following (18).

Lemma 6 (Theorem 2.5, [Opp18] and Theorem 3.1, [AL20]). For $\pi \in \mathcal { P } ( \mathcal { X } _ { k } ^ { d } )$ with full support, if

$$
\lambda _ { 2 } \left( { \bf P } _ { R } \right) \le \frac { \alpha } { k - 1 } f o r a l l R \in \binom { [ d ] } { k - 2 } ,
$$

for some $\alpha < 1$ , then ${ \mathcal { T } } ^ { \mathsf { D } \mathsf { U } , \pi }$ satisfies a λ-Poincaré inequality, for $\textstyle \lambda = { \frac { 1 - \alpha } { k } }$

Spectral gap for rank-one perturbations. To apply Lemma 6, we require tools for bounding the spectral gaps of the link graphs induced by each $\bar { R } \in \mathsf { ( } _ { k - 2 } ^ { [ d ] } )$ . The next piece of our framework, Lemma 7, shows such a spectral gap for graphs where the edge weight matrix W is induced by an appropriately-bounded perturbation K of a rank-one matrix aa<sup>⊤</sup>.

Remark 3. An instructive warmup is when K is the all-zeroes matrix in Lemma 7, in which case

$$
\mathbf { W } = \mathbf { a } \mathbf { a } ^ { \top } - \mathbf { d i a g } \left( \mathbf { a } \right) ^ { 2 }\tag{19}
$$

is a rank-one matrix with its diagonal removed. Because $\lambda _ { 2 } ( { \bf a } { \bf a } ^ { \top } ) = 0$ and diag $\left( \mathbf { a } \right) ^ { 2 } \in \mathbb { S } _ { \succ \mathbf { 0 } } ^ { d \times d }$ , the min-max characterization of eigenvalues gives $\lambda _ { 2 } ( \mathbf { W } ) \leq 0$ . The same strategy, applied to the similar matrix D<sup>−</sup> $^ { - 1 / 2 } \mathbf { W } \mathbf { D } ^ { - 1 / 2 }$ , implies $\lambda _ { 2 } ( \mathbf { P } ) \leq 0$ , following the notation (18). Lemma 7 robustly extends this bound to the case where W in (19) is perturbed by a bounded matrix K.

Lemma 7. Let $\mathbf { a } \in \mathbb { R } _ { > 0 } ^ { d }$ , and let $\mathbf { K } \in \mathbb { S } ^ { d \times d }$ satisfy ${ \bf K } _ { i i } = 0$ for all $i \in [ d ]$ . Define

$$
\mathbf { W } _ { i j } = \{ \begin{array} { l l } { \mathbf { a } _ { i } \mathbf { a } _ { j } \exp { ( \mathbf { K } _ { i j } ) } } & { i \neq j } \\ { 0 } & { i = j } \end{array} , \ f o r \ a l l \ ( i , j ) \in [ d ] \times [ d ] .\tag{20}
$$

Also, let $m ( \mathbf { K } ) : = \operatorname* { m a x } _ { ( i , j ) \in [ d ] \times [ d ] } | \mathbf { K } _ { i j } |$ , let $\mathbf { X } \in \mathbb { S } ^ { d \times d }$ have ${ \bf { X } } _ { i j } = \exp ( { \bf { K } } _ { i j } ) - 1$ entrywise, and let

$$
\rho ( \mathbf { K } ) : = \operatorname* { m a x } \left( 0 , \operatorname* { s u p } _ { \mathbf { u } \in \mathbb { R } ^ { d } : \operatorname { n n z } ( \mathbf { u } ) > 1 } \frac { \mathbf { u } ^ { \top } \left( \mathbf { X } - \mathbf { I } _ { d } \right) \mathbf { u } } { \left\| \mathbf { u } \right\| _ { 1 } ^ { 2 } - \left\| \mathbf { u } \right\| _ { 2 } ^ { 2 } } \right) .\tag{21}
$$

Then, following the notation (18), $\lambda _ { 2 } ( \mathbf { P } ) \leq \exp ( m ( \mathbf { K } ) ) \rho ( \mathbf { K } )$

Proof. Let $ { \mathbf { N } } : =  { \mathbf { D } } ^ { - 1 / 2 }  { \mathbf { W } }  { \mathbf { D } } ^ { - 1 / 2 }$ , so that $\lambda _ { 2 } ( { \bf P } ) = \lambda _ { 2 } ( { \bf N } )$ . We first write N as the sum of a rank-one matrix and a correction. For $\mathbf { A } : = \mathbf { d i a g } \left( \mathbf { a } \right)$

$$
\mathbf { N } = \mathbf { D } ^ { - { \frac { 1 } { 2 } } } \mathbf { a } \mathbf { a } ^ { \top } \mathbf { D } ^ { - { \frac { 1 } { 2 } } } + \mathbf { D } ^ { - { \frac { 1 } { 2 } } } \mathbf { A } \left( \mathbf { X } - \mathbf { I } _ { d } \right) \mathbf { A } \mathbf { D } ^ { - { \frac { 1 } { 2 } } } .
$$

Because the min-max characterization of eigenvalues gives

$$
\begin{array} { r } { \lambda _ { 2 } \left( \mathbf { N } \right) \leq \lambda _ { 2 } \left( \mathbf { D } ^ { - \frac { 1 } { 2 } } \mathbf { a } \mathbf { a } ^ { \top } \mathbf { D } ^ { - \frac { 1 } { 2 } } \right) + \lambda _ { 1 } \left( \mathbf { D } ^ { - \frac { 1 } { 2 } } \mathbf { A } \left( \mathbf { X } - \mathbf { I } _ { d } \right) \mathbf { A } \mathbf { D } ^ { - \frac { 1 } { 2 } } \right) , } \end{array}
$$

it is enough to bound the second term above. We next have

$$
\begin{array} { r l } & { \lambda _ { 1 } \left( \mathbf { D } ^ { - \frac { 1 } { 2 } } \mathbf { A } \left( \mathbf { X } - \mathbf { I } _ { d } \right) \mathbf { A } \mathbf { D } ^ { - \frac { 1 } { 2 } } \right) = \underset { \mathbf { y } \in \mathbb { R } ^ { d } : \mathbf { y } \neq \mathbf { 0 } _ { d } } { \operatorname* { s u p } } \frac { \mathbf { y } ^ { \top } \mathbf { D } ^ { - \frac { 1 } { 2 } } \mathbf { A } \left( \mathbf { X } - \mathbf { I } _ { d } \right) \mathbf { A } \mathbf { D } ^ { - \frac { 1 } { 2 } } \mathbf { y } } { \left\| \mathbf { y } \right\| _ { 2 } ^ { 2 } } } \\ & { \quad \quad \quad \quad = \underset { \mathbf { u } \in \mathbb { R } ^ { d } : \mathbf { u } \neq \mathbf { 0 } _ { d } } { \operatorname* { s u p } } \frac { \mathbf { u } ^ { \top } \left( \mathbf { X } - \mathbf { I } _ { d } \right) \mathbf { u } } { \mathbf { u } ^ { \top } \mathbf { A } ^ { - 2 } \mathbf { D } \mathbf { u } } . } \end{array}\tag{22}
$$

If the supremum is achieved by a 1-sparse u, then the lemma statement holds, because $\rho ( { \bf K } ) \ge 0$ and any 1-sparse u has a negative numerator in (22), because X has an all-zeroes diagonal.

We now bound the denominator of (22) when $\mathrm { n n z } ( \mathbf { u } ) > 1$

$$
\begin{array} { r l } & { \mathbf { u } ^ { \top } \mathbf { A } ^ { - 2 } \mathbf { D } \mathbf { u } = \displaystyle \sum _ { i \in [ d ] } \frac { \mathbf { u } _ { i } ^ { 2 } \mathbf { D } _ { i i } } { \mathbf { a } _ { i } ^ { 2 } } = \displaystyle \sum _ { i \in [ d ] } \left( \frac { \mathbf { u } _ { i } ^ { 2 } } { \mathbf { a } _ { i } } \sum _ { j \in [ d ] : j \neq i } \mathbf { a } _ { j } \exp ( \mathbf { K } _ { i j } ) \right) } \\ & { \qquad \geq \exp \left( - m ( \mathbf { K } ) \right) \displaystyle \sum _ { i \in [ d ] } \frac { \mathbf { u } _ { i } ^ { 2 } ( \| \mathbf { a } \| _ { 1 } - \mathbf { a } _ { i } ) } { \mathbf { a } _ { i } } } \\ & { \qquad = \exp \left( - m ( \mathbf { K } ) \right) \left( \displaystyle \sum _ { i \in [ d ] } \frac { \mathbf { u } _ { i } ^ { 2 } \| \mathbf { a } \| _ { 1 } } { \mathbf { a } _ { i } } - \| \mathbf { u } \| _ { 2 } ^ { 2 } \right) \geq \exp \left( - m ( \mathbf { K } ) \right) \left( \| \mathbf { u } \| _ { 1 } ^ { 2 } - \| \mathbf { u } \| _ { 2 } ^ { 2 } \right) . } \end{array}
$$

The last line applied the Cauchy-Schwarz inequality. Plugging this into (22) gives the result.

As a simple application of Lemma 7, we rederive a standard mixing result on the Curie-Weiss model. Model 4 (Curie-Weiss model, [Wei07, Ell12]). In the Curie-Weiss model, $\mathbf { J } = \textstyle \frac { 1 } { d } \mathbf { 1 } _ { d } \mathbf { 1 } _ { d } ^ { \top }$

Corollary 2. Let $\pi \in \mathcal { P } ( \mathcal { X } _ { k } ^ { d } )$ be induced by a fixed-magnetization Ising model (16), under the Curie-Weiss model $( M o d e l \downarrow )$ . Then for any $\beta \in \mathbb { R } , \ T ^ { \mathsf { D U } , \pi }$ satisfies a <sup>1</sup><sub>k</sub> -Poincaré inequality.

Proof. Recall that the Curie-Weiss interaction matrix is $\mathbf { J } = \textstyle \frac { 1 } { d } \mathbf { 1 } _ { d } \mathbf { 1 } _ { d } ^ { \top }$ . Because $\mathbf { 1 } _ { d } ^ { \top } \mathbf { x }$ is constant over $\mathbf { x } \in \mathcal { X } _ { k } ^ { d }$ , identifying each x with its set S of positive coordinates, we have

$$
\pi ( S ) \propto \exp \left( 2 \beta \mathbf { h } ^ { \top } \mathbf { 1 } _ { S } \right) .\tag{23}
$$

Now for the random walk matrix $\mathbf { P } _ { R }$ associated with the link graph of $R \in \binom { [ d ] } { k - 2 }$ , we have that the associated edge weights W follow (20) with K set to the all-zeroes matrix, and

$$
\mathbf { a } _ { i } = \exp \left( 2 \beta \mathbf { h } _ { i } \right) { \mathrm { ~ f o r ~ a l l ~ } } i \in [ d ] \setminus R .
$$

Therefore Lemma 7 applies with $\mathbf K = \mathbf 0$ and gives a bound of $\alpha = 0$ for use with Lemma 6. □

Corollary 2 rephrases the following proof: the fixed-magnetization Curie-Weiss model is a product distribution restricted to a Hamming slice (23), which Theorem 1.1, [ALGV19] proves a Poincaré inequality for. We include this example to illustrate a trivial case of our framework.

Specialization to Ising model. We next derive a generic application of the framework given by Lemmas 6 and 7 to Ising models. Consider a fixed-magnetization Ising model (16), where

$$
\pi ( \mathbf x ) \propto \exp \left( \beta \left( \frac { 1 } { 2 } \mathbf x ^ { \top } \mathbf J \mathbf x + \mathbf h ^ { \top } \mathbf x \right) \right) \cdot \mathbb { I } _ { \mathbf x \in \mathcal X _ { k } ^ { d } } .
$$

Ising models are invariant to changes in the diagonal of J, so without loss of generality, we explicitly assume in this section that J has zero diagonal, $\mathrm { i . e . , } \mathbf { J } _ { i i } = 0$ for all $i \in [ d ]$

Lemma 8. For a fixed-magnetization Ising model π (16), and following the notation (21), if

$$
\rho ( 4 \beta \mathbf { J } ) \exp ( { m ( 4 \beta \mathbf { J } ) } ) \leq \frac { 1 } { 2 ( k - 1 ) } ,
$$

then ${ \mathcal { T } } ^ { \mathsf { D } \mathsf { U } , \pi }$ satisfies a <sup>1</sup><sub>2k</sub> -Poincaré inequality.

Proof. Following the notation of Lemma 6, it is enough to show that for all $R \in \binom { [ d ] } { k - 2 }$

$$
\lambda _ { 2 } \left( { \bf P } _ { R } \right) \le \frac { 1 } { 2 ( k - 1 ) } .
$$

We prove this using Lemma 7. Fix some R for the remainder of this proof, and let $S : = [ d ] \backslash R$ . Let $\mathbf { r } \in \mathcal { X } _ { k - 2 } ^ { d }$ have positive coordinates with indices R, and consider some $\mathbf { x } = \mathbf { r } + 2 ( \mathbf { e } _ { i } + \mathbf { e } _ { j } ) \in \mathcal { X } _ { k } ^ { d } , \mathrm { i . e . }$ 2 corresponding to the set $R \cup \{ i , j \}$ . We have

$$
\beta \left( { \frac { 1 } { 2 } } \mathbf { x } ^ { \top } \mathbf { J } \mathbf { x } + \mathbf { h } ^ { \top } \mathbf { x } \right) = \beta \left( { \frac { 1 } { 2 } } \mathbf { r } ^ { \top } \mathbf { J } \mathbf { r } + \mathbf { h } ^ { \top } \mathbf { r } \right) + 2 \beta \left( \mathbf { h } + \mathbf { J } \mathbf { r } \right) ^ { \top } \left( \mathbf { e } _ { i } + \mathbf { e } _ { j } \right) + 4 \beta \mathbf { J } _ { i j } .
$$

The first term is a constant for all pairs $( i , j ) \in S \times S$ . Therefore, up to a proportionality constant, the edge weights are given by (20), where for all $( i , j ) \in S \times S$ 2

$$
\mathbf { a } _ { i } = \exp \left( 2 \beta \left( \mathbf { h } + \mathbf { J } \mathbf { r } \right) ^ { \top } \mathbf { e } _ { i } \right) , \quad \mathbf { K } _ { i j } = 4 \beta \mathbf { J } _ { i j } .
$$

The conclusion now follows from Lemma 7 and the assumption, because excluding R from the coordinates can only decrease both $m ( 4 \beta \mathbf { J } )$ and $\rho ( 4 \beta \mathbf { J } )$ . □

## 4.2 Simple suficient conditions for fast mixing

Section 4.1 gives a generic strategy for sampling in fixed-magnetization Ising models. By combining Lemmas $6 , 7 ,$ and 8, our task reduces to bounding $\rho ( \mathbf { K } )$ and $m ( \mathbf { K } )$ for $\mathbf { K } \gets 4 \beta \mathbf { J }$ . The bottleneck is typically to control $\rho ( \mathbf { K } )$ , whose definition (21) is somewhat opaque.

In Lemma 9, we give a more interpretable suficient condition for applying this framework. We show that one specialization of this condition qualitatively recovers the sparse Dobrushin condition, and that another implies improvements in the same applications as considered in Corollary 1.

Mixed-norm quadratic form bound. Our first strategy for controlling $\rho ( \mathbf { K } )$ decomposes ${ \bf { X } } _ { i j } = { \bf { \Phi } }$ $\exp ( \mathbf { K } _ { i j } ) - 1$ into a linear term in $\mathbf { K } _ { i j }$ , and a high-order term. The high-order contribution to the quadratic form in X is folded into our assumption (24), which implies a bound on $\rho ( \mathbf { K } )$ .

Lemma 9. In the setting of Lemma $\delta ,$ define $\mathbf { K } : = 4 \beta \mathbf { J }$ . Then, if

$$
\left| { \mathbf { u } ^ { \top } } \mathbf { K } \mathbf { u } \right| \leq \tau \left\| { \mathbf { u } } \right\| _ { 1 } \left\| { \mathbf { u } } \right\| _ { 2 } + \tau ^ { 2 } \left\| { \mathbf { u } } \right\| _ { 1 } ^ { 2 } f o r \ a l l \ { \mathbf { u } } \in \mathbb { R } ^ { d } ,\tag{24}
$$

for some $\tau \in [ 0 , \frac { 1 } { 4 } ]$ , we have $\rho ( \mathbf { K } ) \exp ( m ( \mathbf { K } ) ) \leq 9 \tau ^ { 2 }$

Proof. First, note that (24) implies a bound on $m ( \mathbf { K } )$ : taking $\mathbf { u }  \mathbf { e } _ { i } + \mathbf { e } _ { j }$ gives

$$
| \mathbf { K } _ { i j } | \leq \sqrt { 2 } \tau + 2 \tau ^ { 2 } \leq 2 \tau .
$$

Therefore, $m ( \mathbf { K } ) \leq 2 \tau$ . Further, if we decompose $\mathbf { X } = \mathbf { K } + \mathbf { R }$ , then

$$
0 \leq \mathbf { R } _ { i j } \leq \operatorname* { s u p } _ { | x | \leq 2 \tau } \exp ( x ) - 1 - x \leq 2 { \sqrt { e } } \tau ^ { 2 } , { \mathrm { ~ f o r ~ a l l ~ } } ( i , j ) \in [ d ] \times [ d ] .
$$

Now by the triangle inequality, and Young’s inequality applied to (24),

$$
\begin{array} { r l r } {  {  \mathbf { u } ^ { \top } \mathbf { X } \mathbf { u }  \leq  \mathbf { u } ^ { \top } \mathbf { K } \mathbf { u }  +  \mathbf { u } ^ { \top } \mathbf { R } \mathbf { u }  } } \\ & { } & { \leq \displaystyle \frac { 1 } { 2 }  \mathbf { u }  _ { 2 } ^ { 2 } + ( 1 + \frac { 1 } { 2 } + 2 \sqrt { e } ) \tau ^ { 2 }  \mathbf { u }  _ { 1 } ^ { 2 } \leq \displaystyle \frac { 1 } { 2 }  \mathbf { u }  _ { 2 } ^ { 2 } + 5 \tau ^ { 2 }  \mathbf { u }  _ { 1 } ^ { 2 } . } \end{array}
$$

We thus have

$$
\mathbf { u } ^ { \mathsf { T } } \mathbf { X } \mathbf { u } - \| \mathbf { u } \| _ { 2 } ^ { 2 } \leq - { \frac { 1 } { 2 } } \left\| \mathbf { u } \right\| _ { 2 } ^ { 2 } + 5 \tau ^ { 2 } \left\| \mathbf { u } \right\| _ { 1 } ^ { 2 } \leq 5 \tau ^ { 2 } \left( \left\| \mathbf { u } \right\| _ { 1 } ^ { 2 } - \left\| \mathbf { u } \right\| _ { 2 } ^ { 2 } \right) .
$$

Therefore, $\rho ( \mathbf { K } ) \leq 5 \tau ^ { 2 }$ , and the conclusion follows from $\begin{array} { r } { \operatorname* { s u p } _ { \tau \in [ 0 , \frac { 1 } { 4 } ] } \exp \left( 2 \tau \right) \le \frac { 9 } { 5 } } \end{array}$

Recovering a sparse Dobrushin condition. We next observe that the $\tau ^ { 2 } \left. \mathbf { u } \right. _ { 1 } ^ { 2 }$ term alone in (24) already qualitatively recovers the sparse Dobrushin condition of Theorem 3.

Lemma 10. In the setting of Lemma 8, if $\begin{array} { r } { m ( \beta \mathbf { J } ) \le \frac { 1 } { 7 2 k } , \ T ^ { \mathsf { D U } , \pi } } \end{array}$ satisfies a ${ \frac { 1 } { 2 k } } - P$ oincaré inequality.

Proof. By combining Lemmas 8 and 9, it sufices to show that (24) holds with $\textstyle \tau ^ { 2 } = { \frac { 1 } { 1 8 k } }$ . Under the assumption on $m ( \beta \mathbf { J } )$ , we have the desired

$$
\left| { \bf u } ^ { \top } { \bf K } { \bf u } \right| \leq m ( { \bf K } ) \left\| { \bf u } \right\| _ { 1 } ^ { 2 } = m ( 4 \beta { \bf J } ) \left\| { \bf u } \right\| _ { 1 } ^ { 2 } \leq \frac { 1 } { 1 8 k } \left\| { \bf u } \right\| _ { 1 } ^ { 2 } .
$$

We remark that compared to Theorem 3, Lemma 10 loses a constant factor in the allowable temperature range, and only implies mixing in $\chi ^ { 2 }$ divergence, which typically loses $\mathrm { a } \approx k$ factor compared to analogous mixing time bounds in Hamming distance.

Applications to Models 1 and 2. Our second application of Lemma 9 controls the τ required in (24) via the largest $\| \mathbf { K } _ { S \times S } \| _ { \mathrm { o p } }$ , appropriately normalized by |S|, over all $S \subseteq [ d ]$ . As intuition for why, if u is the 0-1 indicator vector for some S,

$$
\frac { | \mathbf { u } ^ { \top } \mathbf { K } \mathbf { u } | } { \| \mathbf { u } \| _ { 1 } \| \mathbf { u } \| _ { 2 } } \leq | S | \left\| \mathbf { K } _ { S \times S } \right\| _ { \mathrm { o p } } \cdot \frac { 1 } { | S | ^ { 1 . 5 } } = \frac { \| \mathbf { K } _ { S \times S } \| _ { \mathrm { o p } } } { \sqrt { | S | } } ,
$$

Lemma 11 uses a shelling decomposition to make this intuition rigorous. Interestingly, the shelling decomposition is a standard strategy for passing to continuous notions of sparsity [CRT06], further strengthening connections between our paper’s toolkit and the broader literature.

Lemma 11. Let $\mathbf { K } \in \mathbb { S } ^ { d \times d }$ , and suppose that

$$
\| \mathbf { K } \| _ { s , \mathrm { o p } } \leq \alpha \sqrt { s } + \alpha ^ { 2 } s ~ f o r ~ a l l ~ s \in [ d ] .\tag{25}
$$

Then (24) holds with $\tau : = 3 2 \alpha$

Proof. Fix a vector $\mathbf { u } \neq \mathbf { 0 } _ { d } ,$ , and let

$$
\sigma : = \left\lceil \frac { \| \mathbf { u } \| _ { 1 } ^ { 2 } } { \| \mathbf { u } \| _ { 2 } ^ { 2 } } \right\rceil
$$

which can intuitively be thought of as a numerical analog of the sparsity of u.

Sort the coordinates of u (relabel [d] by a permutation) so that $| \mathbf { u } _ { 1 } | \geq | \mathbf { u } _ { 2 } | \geq . . . \geq | \mathbf { u } _ { d } |$ . Now partition [d] into consecutive blocks ${ \cal B } _ { 1 } = [ \sigma ] , { \cal B } _ { 2 } = [ 2 \sigma ] \backslash { \cal B } _ { 1 } , . .$ . of size at most σ each. For each $\ell \geq 2$ , monotonicity of the coordinates gives

$$
\left\| \mathbf { u } _ { B _ { \ell } } \right\| _ { 2 } \leq \sqrt { \sigma } \left\| \mathbf { u } _ { B _ { \ell } } \right\| _ { \infty } \leq \frac { 1 } { \sqrt { \sigma } } \left\| \mathbf { u } _ { B _ { \ell - 1 } } \right\| _ { 1 } ,
$$

so summing, we have

$$
\sum _ { \ell } \left\| \mathbf { u } _ { B _ { \ell } } \right\| _ { 2 } \leq \left\| \mathbf { u } \right\| _ { 2 } + \frac { 1 } { \sqrt { \sigma } } \left\| \mathbf { u } \right\| _ { 1 } \leq 2 \left\| \mathbf { u } \right\| _ { 2 } .
$$

Now we decompose u<sup>⊤</sup>Ku blockwise, and control each block using α: because each union $B _ { \ell } \cup B _ { \ell ^ { \prime } }$ is 2σ-sparse, and each blockwise contribution is only supported on this set,

$$
\begin{array} { r l r } {  {  \mathbf { u } ^ { \top } \mathbf { K } \mathbf { u }  \leq 2 ( \alpha \sqrt { \sigma } + \alpha ^ { 2 } \sigma ) \sum _ { \ell , \ell ^ { \prime } } \| \mathbf { u } _ { B _ { \ell } } \| _ { 2 } \| \mathbf { u } _ { B _ { \ell ^ { \prime } } } \| _ { 2 } } } \\ & { } & \\ & { } & { \leq 2 ( \alpha \sqrt { \sigma } + \alpha ^ { 2 } \sigma ) ( \sum _ { \ell } \| \mathbf { u } _ { B _ { \ell } } \| _ { 2 } ) ^ { 2 } \leq 8 ( \alpha \sqrt { \sigma } + \alpha ^ { 2 } \sigma ) \| \mathbf { u } \| _ { 2 } ^ { 2 } . } \end{array}\tag{26}
$$

Finally, using $\left\| \mathbf { u } \right\| _ { 2 } \leq 2 \sigma ^ { - 1 / 2 } \left\| \mathbf { u } \right\| _ { \ L }$ <sub>1</sub> gives the claim.

We now derive an application to the SK model (Model 1). This application only uses the $\alpha \sqrt { s }$ term in (25), due to the operator norm behavior of entrywise sub-Gaussian matrices. Per our convention in this section, we use $\mathbf { J } _ { i i } = 0$ for all $i \in [ d ]$ , which does not afect the Gibbs measure (16).

Corollary 3. Let $\delta \in ( 0 , \frac { 1 } { 2 } )$ and let $\pi$ be induced by a fixed-magnetization Ising model (16) under the SK model (Model 1). Further, assume that for an appropriate constant,

$$
\beta = O \left( \sqrt { \frac { d } { k \log \frac { d } { \delta } } } \right) .
$$

Then with probability $\geq 1 - \delta , \mathcal { T } ^ { \mathsf { D U } , \pi }$ satisfies a <sup>1</sup>2k -Poincaré inequality.

Proof. We first claim that for Model 1 with zero diagonal,

$$
\operatorname* { m a x } _ { s \in [ d ] } \frac { \Vert \mathbf { J } \Vert _ { s , \mathrm { o p } } } { \sqrt { s } } \leq C \sqrt { \frac { \log \frac { d } { \delta } } { d } } ,\tag{27}
$$

with probability $\geq 1 - \delta$ , for a universal constant C. To see this, fix $s \in [ d ]$ . With probability $\begin{array} { r l } { \ge 1 - \frac { \delta } { d ^ { s + 1 } } } & { { } } \end{array}$ , Corollary 3.9, [BvH16] shows that for a fixed $S \in ( \mathbb { I } )$

$$
\| \mathbf { J } _ { S \times S } \| _ { \mathrm { o p } } \leq C \sqrt { \frac { s \log { \frac { d } { \delta } } } { d } } .\tag{28}
$$

Now a union bound over the $\leq d ^ { s }$ possible $S ,$ and the d possible $s \in [ d ]$ , shows (27).

Finally, for $\mathbf { K } = 4 \beta \mathbf { J }$ , the assumed range on $\beta$ implies that, following the notation (25), $3 2 \alpha \ \leq$ $( 1 8 k ) ^ { - 1 / 2 }$ . Also, Lemma 11 implies that (24) holds with $\tau = 3 2 \alpha$ . Combining gives $9 \tau ^ { 2 } \leq \frac { 1 } { 2 k }$ in Lemma 9, and then the claim follows from Lemma 8. □

Corollary 3 directly improves the allowable temperature range in Corollary 1’s SK model specialization by $\mathrm { ~ a ~ } \sqrt { k }$ factor. Unfortunately, it does not permit taking arbitrary $\beta = O ( 1 )$ unless k is suficiently sublinear in $d ,$ i.e., smaller than $\frac { d } { \log d }$ . This is an inherent artifact of using the bound (27), because the maximum of $\Omega ( d )$ Gaussians grows with ${ \sqrt { \log d } }$ , causing an obstruction at $| S | = 2$ However, it is not inherent to the SK model, and in Section 4.3, we show how to further shave this extraneous logarithmic factor, by more directly controlling $\rho ( 4 \beta \mathbf { J } )$ in Lemma 8.

We conclude the section with a similar improvement upon Corollary 1’s Gaussian Hopfield model specialization. Crucially, to be compatible with the two-regime sub-exponential concentration of the operator norms of Wishart matrices, our proof uses both of the terms in (25).

Corollary 4. Let $\delta \in ( 0 , \frac { 1 } { 2 } )$ and let $\pi$ be induced by a fixed-magnetization Ising model (16) under the Gaussian Hopfield model (Model 2). Further, assume that for an appropriate constan $^ { \mathrm { , } t \mathrm { , } }$

$$
n = \Omega \left( \operatorname* { m a x } ( \beta , \beta ^ { 2 } ) k \log \frac { d } { \delta } \right) .
$$

Then with probability $\geq 1 - \delta , \mathcal { T } ^ { \mathsf { D U } , \pi }$ satisfies a $\scriptstyle { \frac { 1 } { 2 k } }$ -Poincaré inequality.

Proof. We claim that for Model 2 with zero diagonal, for each fixed $S \subseteq [ d ]$ with $| S | = s$

$$
\left\| \mathbf { J } _ { S \times S } \right\| _ { \mathrm { o p } } \leq C \left( \sqrt { \frac { s \log \frac { d } { \delta } } { n } } + \frac { s \log \frac { d } { \delta } } { n } \right) ,\tag{29}
$$

with probabilit $\begin{array} { r } { \mathrm { ~ y ~ } \ge 1 - \frac { \delta } { d ^ { s + 1 } } } \end{array}$ , for a universal constant C. At this point, the proof follows identically to that of Corollary 3, using the tighter bound in (25). To see that (29) holds, let $\begin{array} { r }  \mathbf { J } = \frac { 1 } { n } \mathbf { G } ^ { \top } \mathbf { G } \cdot \mathbf { \bar { G } } \cdot \mathbf { \bar { \Pi } } \end{array}$ −D for $\mathbf { G } \in \mathbb { R } ^ { n \times d }$ where the $\mathbf { G } _ { i j }$ are i.i.d. Gaussian, and D is a diagonal matrix agreeing with the diagonal of $\textstyle { \frac { 1 } { n } } \mathbf { G } ^ { \top } \mathbf { G }$ . Then (29) follows because with probability $\geq 1 - \delta$

$$
\begin{array} { r } { \left\| \frac { 1 } { n } [ \mathbf { G } ^ { \top } \mathbf { G } ] _ { S \times S } - \mathbf { I } _ { S } \right\| _ { \mathrm { o p } } = O \left( \sqrt { \frac { s + \log \frac { 1 } { \delta } } { n } } + \frac { s + \log \frac { 1 } { \delta } } { n } \right) , } \\ { \| \mathbf { D } _ { S \times S } - \mathbf { I } _ { S } \| _ { \mathrm { o p } } = O \left( \sqrt { \frac { \log \frac { s } { \delta } } { n } } + \frac { \log \frac { s } { \delta } } { n } \right) , } \end{array}\tag{30}
$$

where the first bound above uses Exercise 4.7.3 of [Ver18], and the second uses Theorem 3.1.1 of [Ver18] with a union bound over all s of the diagonal coordinates. Now (29) follows by substituting $\textstyle { \bar { \delta } } \gets { \frac { \bar { \delta } } { d ^ { s + 1 } } }$ above and applying the triangle inequality. □

## 4.3 Linear magnetization in low-temperature SK models

We conclude with a tighter analysis of the quantities required by Lemma 7 for the SK model, that removes the extraneous logarithmic factor from Corollary 3. The results in this section hold assuming a high-probability event under Model 1, captured in the following lemma.

Lemma 12. Let $\delta \in ( 0 , \frac { 1 } { 2 } )$ and $C > 0$ be a suficiently large universal constant. Then the following events simultaneously hold with probability $\underline { { \boldsymbol { \mathbf { \Pi } } } } \geq 1 - \delta$ over Model 1.

1. For all $S \subseteq [ d ]$ with $| S | = s \in [ d ]$

$$
\left\| \mathbf { J } _ { S \times S } \right\| _ { \mathrm { o p } } \leq C \left( \sqrt { \frac { s \log \frac { e d } { s } + \log \frac { d } { \delta } } { d } } \right) .
$$

2. We have

$$
\operatorname* { m a x } _ { ( i , j ) \in [ d ] \times [ d ] } | \mathbf { J } _ { i j } | \leq C \sqrt { \frac { \log \frac { d } { \delta } } { d } } , \quad \| \mathbf { J 1 } _ { d } \| _ { \infty } \leq C \sqrt { \log \frac { d } { \delta } } .
$$

3. For all nonzero $\mathbf { u } \in \mathbb { R } ^ { d }$ with $\begin{array} { r } { q : = \frac { \| \mathbf { u } \| _ { 2 } ^ { 2 } } { \| \mathbf { u } \| _ { 1 } ^ { 2 } } } \end{array}$

$$
\left| \mathbf { u } ^ { \mathsf { T } } \mathbf { J } \mathbf { u } \right| \leq C \left\| \mathbf { u } \right\| _ { 1 } ^ { 2 } \left( { \sqrt { \frac { q \log ( e d q ) } { d } } } + q { \sqrt { \frac { \log { \frac { d } { \delta } } } { d } } } \right) .
$$

4. $\begin{array} { r } { L e t \mathbf { H } : = \mathbf { J } \circ \mathbf { J } - \frac { 1 } { d } ( \mathbf { 1 } _ { d } \mathbf { 1 } _ { d } ^ { \top } - \mathbf { I } _ { d } ) } \end{array}$ . Then,

$$
\left\| \mathbf { H } \right\| _ { \mathrm { o p } } \leq C \left( { \sqrt { \frac { \log { \frac { d } { \delta } } } { d } } } + { \frac { \log { \frac { d } { \delta } } } { d } } \right)
$$

Proof. We allot $\mathrm { ~ a ~ } \frac { \delta } { 3 }$ failure probability for Items 1, 2, and $^ { 4 , }$ and Item 3 will follow from Item 1. Item 1 follows from the calculation in (27), using the tighter estimate $\begin{array} { r } { \binom { d } { s } = \exp ( O ( s \log \frac { e d } { s } ) ) } \end{array}$

Both parts of Item 2 follow from standard bounds on the maximum of $\mathrm { p o l y } ( d )$ i.i.d. Gaussians, where the variance of each entry of $\mathbf { J } \mathbf { 1 } _ { d }$ is at most 1.

Item 3 follows from the same shelling decomposition argument as in Lemma 11. Concretely, perform the same decomposition into blocks $B _ { 1 } , B _ { 2 } , \ldots$ . of size at most $\sigma = \textstyle \lceil { \frac { 1 } { q } } \rceil$ . Then the same argument as in (26), combined with the estimate on $\| \mathbf { J } \| _ { 2 \sigma , \mathrm { o p } }$ already derived in Item 1, yields

$$
\left| \mathbf { u } ^ { \top } \mathbf { J } \mathbf { u } \right| \leq 8 C \left( \sqrt { \frac { \log ( e d q ) } { q d } } + \sqrt { \frac { \log \frac { d } { \delta } } { d } } \right) \left\| \mathbf { u } \right\| _ { 2 } ^ { 2 } .
$$

The claim then follows by adjusting the constant C, and substituting the definition of $q .$ There is an edge case when $2 \sigma \geq d ,$ but in this case $\| \mathbf { J } \| _ { \mathrm { o p } }$ satisfies the required bound (see Fact 1).

Finally, Item 4 asks to bound the operator norm of a matrix with i.i.d. sub-exponential entries. Concretely, let $\mathbf { E } _ { i j } : = \mathbf { e } _ { i } \mathbf { e } _ { j } ^ { \top } + \mathbf { e } _ { j } \mathbf { e } _ { i } ^ { \top }$ for each $1 \leq i < j \leq d .$ . Then $\begin{array} { r } { \mathbf { H } = \sum _ { 1 \leq i < j \leq d } ( \mathbf { J } _ { i j } ^ { 2 } - \frac { 1 } { d } ) \mathbf { E } _ { i j } } \end{array}$ A straightforward calculation shows that the sub-exponential matrix Bernstein inequality (e.g., Theorem 6.2 in [Tro12] with $\begin{array} { r } { \sigma ^ { 2 } = R = O ( \frac { 1 } { d } ) ) } \end{array}$ now applies, which concludes the proof. □

We now use Items 2, 3, and 4 of Lemma 12 to derive estimates on the parameters $\rho ( \mathbf { K } )$ and $m ( \mathbf { K } )$ defined in Lemma 7, when $\mathbf { K } = 4 \beta \mathbf { J }$ as derived in Lemma 8.

Lemma 13. Assume the success of the events in Lemma 12, and let $\beta \in \mathbb { R } ^ { + } , \bar { \beta } : = \operatorname* { m a x } ( \beta , 1 )$ , and $\mathbf { K } : = 4 \beta \mathbf { J }$ . Then for a suficiently large universal constant $C ^ { \prime } > 0$ , assuming

$$
\Delta : = \sqrt { \frac { \log \frac { d } { \delta } } { d } } + \frac { \log \frac { d } { \delta } } { d } \leq \frac { 1 } { C ^ { \prime } \bar { \beta } ^ { 2 } \log ( e \bar { \beta } ) } ,\tag{31}
$$

we have

$$
\rho ( { \bf K } ) \leq \frac { C ^ { \prime } \bar { \beta } ^ { 2 } \log ( e \bar { \beta } ) } { d } , \quad m ( { \bf K } ) \leq C ^ { \prime } \bar { \beta } ^ { 2 } \Delta .
$$

Proof. The bound on $m ( \mathbf { K } )$ follows from the first condition in Item 2 of Lemma 12, and any $C ^ { \prime } \geq 4 C$

To bound $\rho ( \mathbf { K } )$ , we follow the notation of Lemma $^ { 7 , }$ so ${ \bf X } _ { i j } = \exp ( { \bf K } _ { i j } ) - 1$ entrywise. We also decompose $\mathbf { X } = \mathbf { K } + \mathbf { R }$ as in Lemma 9. For the linear term, letting $\mathbf { u } \in \mathbb { R } ^ { d }$ have $\begin{array} { r } { q : = \frac { \| \mathbf { u } \| _ { 2 } ^ { 2 } } { \| \mathbf { u } \| _ { 1 } ^ { 2 } } } \end{array}$

$$
\begin{array} { r } { \left| \mathbf { u } ^ { \top } \mathbf { K } \mathbf { u } \right| = 4 \beta \left| \mathbf { u } ^ { \top } \mathbf { J } \mathbf { u } \right| \leq 4 \beta C \left\| \mathbf { u } \right\| _ { 1 } ^ { 2 } \left( \sqrt { \frac { q \log ( e d q ) } { d } } + q \sqrt { \frac { \log \frac { d } { \delta } } { d } } \right) } \\ { \leq \left\| \mathbf { u } \right\| _ { 1 } ^ { 2 } \left( \frac { q } { 4 } + 4 \beta C q \Delta + \frac { C ^ { \prime \prime } \bar { \beta } ^ { 2 } \log ( e \bar { \beta } ) } { d } \right) , } \end{array}\tag{32}
$$

where the second line used the scalar inequality, for an appropriate $C ^ { \prime \prime }$ depending on $C ,$

$$
4 \beta C \sqrt { s \log ( e s ) } \leq 4 \bar { \beta } C \sqrt { s \log ( e s ) } \leq \frac { s } { 4 } + C ^ { \prime \prime } \bar { \beta } ^ { 2 } \log ( e \bar { \beta } ) ,
$$

valid for any $s , \bar { \beta } \geq 1$ . For the residual term, we first establish the entrywise bound

$$
0 \leq \exp ( \mathbf { K } _ { i j } ) - \mathbf { K } _ { i j } - 1 = \mathbf { R } _ { i j } \leq \mathbf { K } _ { i j } ^ { 2 } \leq 1 6 \beta ^ { 2 } \mathbf { J } _ { i j } ^ { 2 } ,
$$

as we have shown $m ( \mathbf { K } ) \leq C ^ { \prime } \bar { \beta } ^ { 2 } \Delta \leq 1$ already. Thus, letting $\mathbf { w } _ { i } : = | \mathbf { u } _ { i } |$ for all $i \in [ d ]$

$$
\begin{array} { r l } & { \left| \mathbf { u } ^ { \top } \mathbf { R } \mathbf { u } \right| \leq 1 6 \beta ^ { 2 } \mathbf { w } ^ { \top } \left( \mathbf { J } \circ \mathbf { J } \right) \mathbf { w } } \\ & { \qquad = 1 6 \beta ^ { 2 } \mathbf { w } ^ { \top } \mathbf { H } \mathbf { w } + \displaystyle \frac { 1 6 \beta ^ { 2 } } { d } \mathbf { w } ^ { \top } \left( \mathbf { 1 } _ { d } \mathbf { 1 } _ { d } ^ { \top } - \mathbf { I } _ { d } \right) \mathbf { w } } \\ & { \qquad \leq 1 6 \beta ^ { 2 } C \Delta \left\| \mathbf { u } \right\| _ { 2 } ^ { 2 } + \displaystyle \frac { 1 6 \beta ^ { 2 } \left\| \mathbf { u } \right\| _ { 1 } ^ { 2 } } { d } } \end{array}\tag{33}
$$

where the second line used the definition of H from Item 4, and the last line applied Item 4 and our definition of $\Delta$ . Finally, by combining (32) and (33),

$$
\begin{array} { r l } & { \mathbf { u } ^ { \top } { \mathbf { X } } { \mathbf { u } } - \| { \mathbf { u } } \| _ { 2 } ^ { 2 } \leq \left( - \frac { 3 } { 4 } + 4 \beta C \Delta + 1 6 \beta ^ { 2 } C \Delta \right) \| { \mathbf { u } } \| _ { 2 } ^ { 2 } + \left( \frac { C ^ { \prime \prime } \bar { \beta } ^ { 2 } \log ( e \bar { \beta } ) + 1 6 \beta ^ { 2 } } { d } \right) \| { \mathbf { u } } \| _ { 1 } ^ { 2 } } \\ & { \qquad \leq - \displaystyle \frac { 1 } { 2 } \| { \mathbf { u } } \| _ { 2 } ^ { 2 } + \frac { C ^ { \prime } \bar { \beta } ^ { 2 } \log ( e \bar { \beta } ) } { d } \| { \mathbf { u } } \| _ { 1 } ^ { 2 } , } \end{array}
$$

for an appropriate $C ^ { \prime }$ . We now obtain the desired bound on $\rho ( \mathbf { K } )$ by noting $\begin{array} { r } { \frac { C ^ { \prime } \bar { \beta } ^ { 2 } \log ( e \bar { \beta } ) } { d } \leq \frac { 1 } { 2 } } \end{array}$ □

Theorem 4. Let $\delta \in ( 0 , \frac { 1 } { 2 } )$ , and let π be induced by a fixed-magnetization Ising model (16), where

$$
k = { \cal O } \left( \frac { d } { \bar { \beta } ^ { 2 } \log ( e \bar { \beta } ) } \right)
$$

for a suficiently small constant, defining $\bar { \beta } : = \operatorname* { m a x } ( 1 , \beta )$ . Also, assume that the condition (31) holds. With probabili $t y \ge 1 - \delta$ over the SK model (Model 1), if

$$
T = \Omega \left( k \log \frac { 1 } { \delta } + k ^ { 2 } \left( \log ( d ) + \beta \left( \left\| \mathbf { h } \right\| _ { \infty } + \sqrt { \log \frac { d } { \delta } } \right) \right) \right)
$$

for a suficiently large constant, we have for any $\pi _ { 0 } \in \mathcal { P } ( \mathcal { X } _ { k } ^ { d } )$ ，

$$
\mathrm { T V } \left( \left( \mathsf { l a z y } ( \mathcal T ^ { \mathsf { D U } , \pi } ) \right) ^ { T } \pi _ { 0 } , \pi \right) \le \delta .
$$

Proof. The failure probability comes from Lemma 12, so henceforth condition on its success. Combining Lemmas $7 , 8 ,$ and 13 implies that $\tau ^ { \mathrm { { D U , \pi } } }$ satisfies $\mathrm { ~ a ~ } ~ \frac { 1 } { 2 k } – \mathrm { P I }$ , for the assumed range on k. Lemma 1 now shows the $\chi ^ { 2 }$ divergence of the lazy down-up walk contracts by a factor of $\Omega ( \textstyle { \frac { 1 } { k } } )$ in each iteration. The conclusion follows if we can bound the initial $\chi ^ { 2 }$ divergence:

$$
\chi ^ { 2 } \left( \pi _ { 0 } \| \pi \right) = \mathbb { E } _ { \pi } \left[ \left( \frac { \pi _ { 0 } } { \pi } \right) ^ { 2 } \right] - 1 \leq \frac { 1 } { \pi _ { \operatorname* { m i n } } ^ { 2 } } , \ \mathrm { w h e r e } \ \pi _ { \operatorname* { m i n } } : = \operatorname* { m i n } _ { \mathbf { x } \in \mathcal { X } _ { k } ^ { d } } \pi ( \mathbf { x } ) .
$$

Thus it remains to control $\pi _ { \mathrm { m i n } }$ . Identify $\mathbf { x } \in \mathcal { X } _ { k } ^ { d }$ with $S \in ( \mathbb { \vphantom { | } } _ { k } ^ { [ d ] } )$ , so $\mathbf { x } = 2 \mathbf { 1 } _ { S } - \mathbf { 1 } _ { d }$ . Then for

$$
F ( \mathbf { x } ) = 2 \mathbf { 1 } _ { S } ^ { \top } \mathbf { J } \mathbf { 1 } _ { S } - 2 \mathbf { 1 } _ { S } ^ { \top } \mathbf { J } \mathbf { 1 } _ { d } + 2 \mathbf { h } ^ { \top } \mathbf { 1 } _ { S } ,
$$

the Ising measure is $\propto \exp ( \beta { \cal F } )$ . On the events in Items 1 and 2 of Lemma 12,

$$
\left| 2 \mathbf { 1 } _ { S } ^ { \top } \mathbf { J } \mathbf { 1 } _ { S } \right| = O \left( k ( 1 + \Delta ) \right) = O ( k ) , \quad \left| 2 \mathbf { 1 } _ { S } ^ { \top } \mathbf { J } \mathbf { 1 } _ { d } \right| = O \left( k \sqrt { \log \frac { d } { \delta } } \right) ,
$$

where we used that (31) gives $\Delta = O ( 1 )$ . Hence, by bounding the range of $\beta F$ over $\mathcal { X } _ { k } ^ { d }$

$$
\begin{array} { r l } & { \pi _ { \operatorname* { m i n } } \geq \frac { 1 } { { \binom { [ d ] } { k } } } \exp \left( { - O \left( k + \beta k \left( \| \mathbf { h } \| _ { \infty } + \sqrt { \log \frac { d } { \delta } } \right) \right) } \right) } \\ & { \quad \quad = \exp \left( { - O \left( k \log ( d ) + \beta k \left( \| \mathbf { h } \| _ { \infty } + \sqrt { \log \frac { d } { \delta } } \right) \right) } \right) . } \end{array}
$$

The conclusion follows by taking $\begin{array} { r } { T = \Omega ( k \log { \frac { 1 } { \pi _ { \mathrm { m i n } } } } ) } \end{array}$ and simplifying.

We make two brief remarks on the statement of Theorem 4. First, the condition (31) is a lower bound on the failure probability δ that must apply for Theorem 4 to hold. This condition is relatively mild, and even permits taking $\delta = \exp ( - \Omega ( d ) )$ for appropriate constants. Second, the use of $\bar { \beta } = \operatorname* { m a x } ( \beta , 1 )$ in the statement makes the k upper bound uniform over all $\beta \geq 0$ . In particular, if the assumptions hold at a target $\beta ^ { \star } \geq 0$ , they hold simultaneously for every intermediate $\beta \in [ 0 , \beta ^ { \star } ]$ which becomes relevant in our applications of annealing in Section 5.

## 5 Annealing

Let $f : \mathcal { X } ^ { d }  \mathbb { R }$ and $\beta \geq 0$ parameterize a Gibbs measure π<sub>β</sub> ∝ exp(βf). In this section, we develop a general framework for sampling from bounded-magnetization restrictions over $\chi ^ { d }$

$$
\pi _ { \le k , \beta } ( \mathbf { x } ) \propto \exp { ( \beta f ( \mathbf { x } ) ) \cdot \mathbb { I } _ { \mathbf { x } \in \mathcal { X } _ { \le k } ^ { d } } } ,\tag{34}
$$

leveraging samplers for fixed-magnetization measures for $0 \leq s \leq k$

$$
\pi _ { \boldsymbol { s } , \beta } ( \mathbf { x } ) \propto \exp \left( \beta f ( \mathbf { x } ) \right) \cdot \mathbb { I } _ { \mathbf { x } \in \mathcal { X } _ { s } ^ { d } } ,\tag{35}
$$

e.g., those constructed in Sections 3 and 4, as black boxes. Our approach decomposes the task into two stages: we first construct an approximate sampler for the cardinality s, and then, conditioned on this size, invoke the corresponding fixed-size sampler for $\pi _ { s , \beta }$ to obtain a sample $\sim \pi _ { \leq k , \beta }$

In Section 5.1, we employ an annealing-based technique (adapted from [Kol18]) to estimate the normalizing constants of the fixed-magnetization measures $\pi _ { s , \beta }$ . In Section 5.2, we integrate these estimates into a unified framework for approximately sampling from bounded-magnetization measures $\pi _ { \leq k , \beta }$ . Finally, in Section 5.3, we instantiate our framework for bounded-magnetization variants of the Ising models in Sections 3 and 4, and derive the resulting mixing time bounds.

## 5.1 Estimating normalizing constants

We first recall an estimation procedure for normalization constants of a discretely-supported measure $\pi \in \mathscr { P } ( \Omega )$ , based directly on [Kol18]. For some fixed $f : \Omega \to { \mathbb { R } }$ , where $| \Omega | < \infty$ , let

$$
Z ( \beta ) : = \sum _ { \omega \in \Omega } \exp ( \beta f ( \omega ) )\tag{36}
$$

to be the normalizing constant of the tempered Gibbs distribution $\pi _ { \beta } \propto \exp ( \beta f )$ at inverse temperature $\beta \geq 0$ . Observe that $Z ( 0 ) = | \Omega |$ is known exactly. For a target inverse temperature $\beta ^ { \star }$ and error tolerance $\epsilon \in ( 0 , 1 )$ , our goal is to produce an estimate $\widehat { Z }$ such that

$$
\begin{array} { r } { ( 1 - \epsilon ) Z ( \beta ^ { \star } ) \le \widehat { Z } \le ( 1 + \epsilon ) Z ( \beta ^ { \star } ) . } \end{array}\tag{37}
$$

We now state a consequence of the estimation procedure of [Kol18].

Proposition 2. Let $f : \Omega \to [ - R , R ]$ for $R \geq 0$ and let $\beta ^ { \star } \ge 0 , ( \delta , \epsilon ) \in ( 0 , \frac { 1 } { 2 } ) ^ { 2 }$ . There is an algorithm EstimateZ $( f , \beta ^ { \star } , \delta , \epsilon , \mathcal { A } )$ , where A is an algorithm that samples from

$$
\pi _ { \beta } \in \mathcal P ( \Omega ) \propto \exp \left( \beta f \right)
$$

for any $\beta \in [ 0 , \beta ^ { \star } ]$ . The output of EstimateZ satisfies (37) with probability $\geq 1 - \delta$ . Further, it uses

$$
N = O \left( \frac { 1 + \beta ^ { \star } R } { \epsilon ^ { 2 } } \log \left( \frac { 1 } { \delta } \right) \right)\tag{38}
$$

calls to A.

Proof. We first obtain (37) with probability $> \ \frac { 1 } { 2 }$ using $O \big ( \frac { 1 + R \beta ^ { \star } } { \epsilon ^ { 2 } } \big )$ calls to ${ \mathcal { A } } ,$ at which point the result follows by taking the median of $O ( \log ( \frac { 1 } { \delta } ) )$ copies via a standard Chernof bound argument.

I ${ \ o } \beta ^ { \star } = 0 \ \mathrm { o r } \ R = 0$ , the claim is immediate by outputting |Ω|. Otherwise, define

$$
H ( \omega ) : = \frac { 2 R - f ( \omega ) } { R } , \quad t ^ { \star } : = R \beta ^ { \star } ,
$$

so that if we define $\begin{array} { r } { Z _ { H } ( t ) : = \sum _ { \omega \in \Omega } \exp ( - t H ( \omega ) ) } \end{array}$ to be the corresponding normalizing constant for $- H$ at inverse temperature t, then

$$
Z _ { H } ( t ) = \exp \left( - 2 t \right) Z \left( { \frac { t } { R } } \right) , { \mathrm { ~ f o r ~ a l l ~ } } t \in [ 0 , t ^ { \star } ] .
$$

Thus to produce the required estimate (37), it is enough to estimate

$$
Q : = \frac { Z _ { H } ( 0 ) } { Z _ { H } ( t ^ { \star } ) } = \exp \left( 2 R \beta ^ { \star } \right) \frac { Z ( 0 ) } { Z ( \beta ^ { \star } ) }
$$

to multiplicative error $1 \pm \epsilon$ . Also, observe that $q : = \log Q$ satisfies $q \in [ t ^ { \star } , 3 t ^ { \star } ]$ , because the range of $H ( \omega )$ is [1, 3]. Theorem 6 in [Kol18] with $n = 3$ and $q = O ( 1 + R \beta ^ { \star } )$ , and its suggested parameters $d = 6 4 , m = { \cal O } ( 1 )$ , and $r = O ( \epsilon ^ { - 2 } )$ , now gives the claim. We note that Theorem 6 in [Kol18] is stated with an expected query complexity, but Markov’s inequality converts this into a deterministic runtime with a constant failure probability that can be folded into the estimator’s failure. □

## 5.2 Approximate bounded-magnetization sampling

In this section, we develop a framework for approximate sampling from $\pi { \leq } k , \beta ^ { \star }$ , defined in (34), assuming access to samplers for the densities $\pi _ { s , \beta } ~ ( 3 5 )$ for all $0 \leq s \leq k$ and $0 \le \beta \le \beta ^ { \star }$

The key observation is that $\pi _ { \leq k , \beta } ,$ ⋆ admits the decomposition

$$
\pi _ { \leq k , \beta ^ { \star } } = \sum _ { i = 0 } ^ { k } \alpha _ { i } \pi _ { i , \beta ^ { \star } } , \quad \alpha _ { i } = \frac { Z _ { i } ( \beta ^ { \star } ) } { \sum _ { j = 0 } ^ { k } Z _ { j } ( \beta ^ { \star } ) } , \quad Z _ { i } ( \beta ) : = \sum _ { \mathbf { x } \in \mathcal { X } _ { \ast } ^ { d } } \exp \left( \beta f ( \mathbf { x } ) \right) .\tag{39}
$$

We first establish Lemma 14, which shows that accurate estimates of the mixture weights α and component distributions $\pi _ { i , \beta ^ { \star } }$ sufice to guarantee an accurate approximation of $\pi { \leq } k , \beta ^ { \star }$

Lemma 14. Let I be an index set and assume that $\pi , \pi ^ { \prime } \in \mathcal { P } ( \Omega )$ admit the following decompositions:

$$
\pi = \mathbb { E } _ { i \sim \rho } [ \mu _ { i } ] , \quad \pi ^ { \prime } = \mathbb { E } _ { j \sim \rho ^ { \prime } } [ \mu _ { j } ^ { \prime } ] , \quad \rho , \rho ^ { \prime } \in \mathcal { P } ( \mathcal { T } ) , \quad \mu _ { i } , \mu _ { i } ^ { \prime } \in \mathcal { P } ( \Omega ) \ f o r \ a l l \ i \in \mathcal { T } .
$$

Then,

$$
\mathrm { T V } \left( \pi , \pi ^ { \prime } \right) \leq \mathrm { T V } \left( \rho , \rho ^ { \prime } \right) + \operatorname* { s u p } _ { i \in \mathcal { I } } \mathrm { T V } \left( \mu _ { i } , \mu _ { i } ^ { \prime } \right)
$$

Proof. Let $\pi _ { m } : = \mathbb { E } _ { i \sim \rho } [ \mu _ { i } ^ { \prime } ]$ . By the triangle inequality of TV distance,

$$
\mathrm { T V } \left( \pi , \pi ^ { \prime } \right) \leq \mathrm { T V } \left( \pi , \pi _ { m } \right) + \mathrm { T V } \left( \pi _ { m } , \pi ^ { \prime } \right) .
$$

For the first term, convexity of $| \cdot | \mathrm { ~ y ~ }$ ields

$$
\mathrm { T V } \left( \pi , \pi _ { m } \right) = \frac { 1 } { 2 } \sum _ { \omega \in \Omega } \left| \mathbb { E } _ { i \sim \rho } [ \mu _ { i } ( \omega ) - \mu _ { i } ^ { \prime } ( \omega ) ] \right| \leq \mathbb { E } _ { i \sim \rho } \left[ \frac { 1 } { 2 } \sum _ { \omega \in \Omega } \left| \mu _ { i } ( \omega ) - \mu _ { i } ^ { \prime } ( \omega ) \right| \right] \leq \operatorname* { s u p } _ { i \in \mathcal { I } } \mathrm { T V } \left( \mu _ { i } , \mu _ { i } ^ { \prime } \right) .
$$

For the second term, couple $i \sim \rho , i ^ { \prime } \sim \rho ^ { \prime }$ to minimize $\mathbb { P } [ i \neq i ^ { \prime } ]$ , and then couple the draws from $\mu _ { i } ^ { \prime } = \mu _ { i { \prime } } ^ { \prime }$ whenever $i = i ^ { \prime }$ . This produces diferent samples with probability $\leq \mathrm { T V } \left( \rho , \rho ^ { \prime } \right)$ □

We now present our sampler for bounded-magnetization measures (34) in Algorithm 2, and establish correctness in Lemma 15. An obstacle to directly applying Proposition 2 is that only approximate samplers $\hat { \pi } _ { i , \beta }$ are available in place of $\pi _ { i , \beta } ;$ this is addressed via a coupling argument.

Lemma 15. Let $f : \mathcal { X } _ { < k } ^ { d } \to [ - R , R ]$ for $R \geq 0$ and let $\beta ^ { \star } \ge 0 , \delta \in ( 0 , \frac { 1 } { 2 } )$ . For all $0 \leq s \leq k _ {  }$ $0 \le \beta \le \beta ^ { \star }$ , assume that algorithm $\mathcal { A } _ { s } ( { \boldsymbol { \beta } } , \delta ^ { \prime } )$ returns a sample within $\delta ^ { \prime }$ total variation distance of $\pi _ { s , \beta }$ defined in (35). Algorithm $\mathcal { Q }$ uses N calls, each to some $\mathcal { A } _ { s } ( \beta , \frac { \delta } { 6 N } )$ where

$$
N = O \left( \frac { k ( 1 + \beta ^ { \star } R ) \log ( \frac { k } { \delta } ) } { \delta ^ { 2 } } \right) .\tag{40}
$$

Further, its output x satisfies

$$
\begin{array} { r } { \mathrm { T V } \left( \mathrm { L a w } ( \mathbf { x } ) , \pi _ { \leq k , \beta ^ { \star } } \right) \leq \delta . } \end{array}
$$

Proof. For simplicity, in this proof we denote $\begin{array} { r } { \hat { \pi } _ { s , \beta } : = \operatorname { L a w } ( \mathcal { A } _ { s } ( \beta , \frac { \delta } { 6 N } ) ) } \end{array}$ . Fix an optimal coupling between $\hat { \pi } _ { s , \beta }$ and $\pi _ { s , \beta }$ for each oracle call to $A _ { s } .$ By assumption, each call produces an exact sample from $\pi _ { s , \beta }$ with probability at least $1 - { \frac { \delta } { 6 N } }$ . A union bound over the N calls implies that, with probability at least $1 - \textstyle { \frac { \delta } { 6 } }$ , all oracle samples are exact.

Next, note that our choice of N satisfies (38) with $\epsilon  \frac { \delta } { 6 }$ and $\delta \gets \frac { \delta } { 1 2 k }$ . Thus, all of the $k + 1 \leq 2 k$ estimates $\{ \widehat { Z } _ { s } \} _ { s = 0 } ^ { k }$ computed on Line 4 are correct with probability at least $1 - { \frac { \delta } { 6 } }$ . Altogether, by Proposition 2, we have that with probability at least $1 - { \frac { \delta } { 3 } }$ that each $\widehat { Z } _ { s }$ satisfies

$$
\frac { \hat { Z } _ { s } } { Z _ { s } } \in \left[ 1 - \frac { \delta } { 6 } , 1 + \frac { \delta } { 6 } \right] .
$$

Applying Lemma 4 and the estimate x $\in [ \pm { \frac { \delta } { 6 } } ] \implies \log ( 1 + x ) \in [ \pm { \frac { \delta } { 3 } } ]$ yields

$$
\mathrm { T V } \left( \mathrm { L a w } ( s ) , \mathrm { M u l t i n o m i a l } ( Z ) \right) \leq { \frac { \delta } { 3 } } ,
$$

where s is the sampled index on Line 6, and $Z _ { i } : = Z _ { i } ( \beta ^ { \star } )$ for all $0 \leq i \leq k$ as defined in (39). Combining this with Lemma 14 and the failure probability of $\frac { \delta } { 3 }$ on the final sample, we obtain

$$
\mathrm { T V } \left( \mathrm { L a w } ( \mathbf { x } ) , \pi _ { \leq k , \beta ^ { \star } } \right) \leq \frac { 2 \delta } { 3 }
$$

on the above high-probability event. Finally, the total failure probability from earlier was ${ \frac { \delta } { 3 } } ,$ and contributes additively in total variation. Therefore,

$$
\begin{array} { r } { \mathrm { T V } \left( \mathrm { L a w } ( \mathbf { x } ) , \pi _ { \leq k , \beta ^ { \star } } \right) \leq \delta . } \end{array}
$$

Algorithm 2: BMGibbsSampler(k, $\beta ^ { \star } , f , \{ \mathcal { A } _ { s } \} _ { s = 0 } ^ { k } , \delta )$   
1 Input: Magnetization bound $k \in [ d ]$ , target inverse temperature $\beta ^ { \star } \geq 0 , f : \mathcal { X } _ { < k } ^ { d } \to [ - R , R ] .$   
approximate samplers $\{ \mathcal { A } _ { s } \} _ { s = 0 } ^ { k }$ such that for all $\begin{array} { r } { 0 \leq \beta \leq \beta ^ { \star } , \delta ^ { \prime } \in ( 0 , \frac { 1 } { 2 } ) , \mathcal { A } _ { s } ( \beta , \delta ^ { \prime } ) } \end{array}$ returns a   
sample x with TV $( \operatorname { L a w } ( \mathbf { x } ) , \overbar { \pi _ { s , \beta } } ) \leq \delta ^ { \prime }$ , failure probability $\delta \in ( 0 , \frac { 1 } { 2 } )$   
2 Output: Sample x such that TV (Law $( \mathbf { x } ) , \pi _ { \leq k , \beta ^ { \star } } ) \leq \delta$ where $\pi _ { \le k , \beta ^ { \star } } \propto \exp ( \beta ^ { \star } f ) \cdot \mathbb { I } _ { \cdot \in \mathcal { X } _ { \le k } ^ { d } }$   
3 for $i = 0 , 1 , \ldots , k$ do   
4 $\begin{array} { r } { \widehat { Z } _ { i } \gets \mathsf { E s t i m a t e } Z ( f _ { i } , \beta ^ { \star } , \frac { \delta } { 1 2 k } , \frac { \delta } { 6 } , \mathcal { A } _ { i } ( \cdot , \frac { \delta } { 6 N } ) ) } \end{array}$ where $f _ { i } = f$ with domain $\mathcal { X } _ { i } ^ { d } .$ , and N is as defined   
in (40)   
5 end   
6 $s \sim \mathrm { M u l t i n o m i a l } ( \widehat { Z } )$   
7 return $\mathbf { x } \sim \mathcal { A } _ { s } ( \beta ^ { \star } , \frac { \delta } { 3 } )$

## 5.3 Bounded-magnetization Ising models

In this section, we show how to apply Lemma 15 to bounded-magnetization Ising models, where

$$
f ( \mathbf { x } ) = { \frac { 1 } { 2 } } \mathbf { x } ^ { \top } \mathbf { J } \mathbf { x } + \mathbf { h } ^ { \top } \mathbf { x }
$$

as in (34). To apply our framework, we require an upper bound on $| f |$ over the domain $\boldsymbol { \mathcal { X } } _ { \leq k } ^ { d }$ . We derive such a bound, which is slightly tightened by the observation that shifting the potential by a constant does not afect the Gibbs measure. The same strategy can be applied in any setting where f has large magnitude but small variation over its domain.

## Lemma 16. We have

$$
\begin{array} { r } { | f ( \mathbf { x } ) - f ( - \mathbf { 1 } _ { d } ) | \leq ( 2 k + 4 d ) \| \mathbf { J } \| _ { 2 k , \mathrm { o p } } + 2 \| \mathbf { h } \| _ { k , 1 } , \qquad f o r \ a l l \ \mathbf { x } \in \mathcal { X } _ { \leq k } ^ { d } . } \end{array}
$$

Proof. Let $S : = \mathrm { s e t } ( \mathbf { x } )$ and $\mathbf { v } : = \mathbf { 1 } _ { S }$ . Because $\mathbf { x } - \left( - \mathbf { 1 } _ { d } \right) = 2 \mathbf { v } , \mathbf { x } + \left( - \mathbf { 1 } _ { d } \right) = 2 \mathbf { v } - 2 \mathbf { 1 } _ { d } $ , we obtain

$$
\begin{array} { r } { | f ( \mathbf { x } ) - f ( - \mathbf { 1 } _ { d } ) | \leq 2 \left| \mathbf { v } ^ { \top } \mathbf { J } \mathbf { v } \right| + 2 \left| \left( \mathbf { h } - \mathbf { J } \mathbf { 1 } _ { d } \right) ^ { \top } \mathbf { v } \right| \leq 2 k \left\| \mathbf { J } \right\| _ { 2 k , \mathrm { o p } } + 2 \left\| \mathbf { h } \right\| _ { k , 1 } + 2 \left| \mathbf { 1 } _ { d } ^ { \top } \mathbf { J } \mathbf { v } \right| , } \end{array}
$$

where the last step exploits the fact that v is k-sparse and $\| \mathbf { v } \| _ { 2 } \leq { \sqrt { k } }$ . Finally, the conclusion follows from the bound

$$
\left| \mathbf { 1 } _ { d } ^ { \top } \mathbf { J } \mathbf { v } \right| \leq \sum _ { i \in [ m ] } \left| \mathbf { 1 } _ { S _ { i } } ^ { \top } \mathbf { J } \mathbf { v } \right| \leq 2 d \left\| \mathbf { J } \right\| _ { 2 k , \mathrm { o p } } ,
$$

for an arbitrary partition of [d] into k-sparse sets $S _ { 1 } \cup S _ { 2 } \cup \ldots \cup S _ { m }$ , where $\begin{array} { r } { m \leq \frac { d } { k } + 1 \leq \frac { 2 d } { k } } \end{array}$ □

By instantiating Lemma 15 with the fixed-magnetization samplers from Sections 3 and 4, we derive samplers for the analogous bounded-magnetization Gibbs measures. For brevity, we only present the bounded-magnetization generalization of Theorem 4 here.

Corollary 5. In the setting of Theorem 4, let π be as in (16) with restriction set $\boldsymbol { \mathcal { X } } _ { \leq k } ^ { d }$ instead of $\mathcal { X } _ { k } ^ { d }$ . There is an algorithm that returns a sample x with TV $( \mathrm { L a w } ( \mathbf { x } ) , \pi ) \leq \delta$ , with $p r o b a b i l i t y \ge 1 - \delta$ over the SK model (Model 1), in time

$$
O \left( \frac { d k \rho \log \left( \frac { k } { \delta } \right) } { \delta ^ { 2 } } \cdot \left( k \log \left( \frac { k \rho } { \delta } \right) + \beta k ^ { 2 } \left( \left\| \mathbf { h } \right\| _ { \infty } + \log \frac { d } { \delta } \right) \right) \right) , \ w h e r e \ \rho : = d + \beta \left\| \mathbf { h } \right\| _ { k , 1 } .
$$

Proof. The algorithm is Algorithm 2 with $\beta ^ { \star }  \beta$ , where we use

$$
\left( \mathsf { I a z y } \left( \mathcal { T } ^ { \mathsf { D U } , \pi _ { s , \beta } } \right) \right) ^ { T } \pi _ { 0 }
$$

as $\mathcal { A } _ { s } ( \beta , \frac { \delta } { 6 N } )$ for all $0 \leq s \leq k$ , for an arbitrary $\pi _ { 0 } \in \mathcal { P } ( \mathcal { X } _ { s } ^ { d } )$ . Theorem 4 states that we need

$$
T = O \left( k \log \frac { N } { \delta } + \beta k ^ { 2 } \left( \left\| \mathbf { h } \right\| _ { \infty } + \log \frac { d } { \delta } \right) \right) ,
$$

merging terms for simplicity. Under the success of Item 1 in Lemma 12, the assumed range on k in Theorem 4, and the condition (31), we have $\beta \| \mathbf { J } \| _ { 2 k , \mathrm { { o p } } } = O ( 1 )$ . Thus, Lemma 16 gives

$$
\beta ^ { \star } R = O ( \rho ) \implies N = O \left( \frac { k \rho \log \left( \frac { k } { \delta } \right) } { \delta ^ { 2 } } \right)
$$

in our application of Lemma 15. The conclusion follows from the implementation in Remark 2.

## 6 Bayesian Sparse Linear Regression

In this section, we apply the local-to-global framework developed earlier to design an approximate sampler for the spike-and-slab posterior in Model 3. As in [KSTZ25], we first use a sparse recovery preprocessing step to remove coordinates whose inclusion is determined from the observations, up to negligible posterior mass. We then sample directly from the preprocessed support posterior.

In Section 6.1, we recall the preprocessing reduction of [KSTZ25]. In Section 6.2, we prove rapid mixing of every fixed-size restriction of the exact support posterior and lift these samplers to the bounded-size posterior using tools from Section 5. Finally, Section 6.3 combines the pieces.

## 6.1 Setup

Throughout this section, we follow the shorthand

$$
\mathbf { E } : = \mathbf { X } ^ { \top } \mathbf { X } - \mathbf { I } _ { d } , \quad L : = \log \frac { d } { \delta }
$$

to simplify the statement of bounds, where δ is a specified failure probability. We begin with a technical lemma used to decompose the output of the [KSTZ25] reduction.

Lemma 17. Let $\mathbf { M } \in \mathbb { S } ^ { d \times d }$ and let $\mathbf { v } \in \mathbb { R } ^ { d }$ satisfy $| \mathrm { s u p p } ( \mathbf { v } ) | \leq r$ . Then following the notation (7),

$$
\begin{array} { r } { \left. \mathbf { M } \mathbf { v } \right. _ { s , 2 } \leq \left. \mathbf { M } \right. _ { r + s , \mathrm { o p } } \left. \mathbf { v } \right. _ { 2 } \ f o r \ a l l \ s \in [ d - r ] . } \end{array}
$$

Proof. Fix $T \subseteq [ d ]$ with $| T | \leq s$ , and let $S : = T \cup \operatorname { s u p p } ( \mathbf { v } )$ so $| S | \le r + s$ . Then,

$$
\begin{array} { r } { \left\| \left[ \mathbf { M } \mathbf { v } \right] _ { T } \right\| _ { 2 } \leq \left\| \mathbf { M } _ { S \times S } \right\| _ { \mathrm { o p } } \left\| \mathbf { v } \right\| _ { 2 } \leq \left\| \mathbf { M } \right\| _ { r + s , \mathrm { o p } } \left\| \mathbf { v } \right\| _ { 2 } . } \end{array}
$$

We next state a variant of the preprocessing strategy used by [KSTZ25]. To obtain our improved sample complexity, we require a somewhat more fine-grained guarantee on its output than the $\ell _ { \infty }$ error bound used in prior works [KSTZ25, CLTZ26]. We state our required property in the form of a decomposition (42), which splits the preprocessing output vector z into two terms: a short vector (with bounded sparse $\ell _ { 2 }$ norm), and a flat vector (with bounded $\ell _ { \infty }$ norm). Notably, this strategy is reminiscent of a similar decomposition used algorithmically by $\mathrm { | K L L ^ { + } 2 3 | }$

Proposition 3 (Section 3.1, Lemma 8, [KSTZ25] and Theorem 3, [CLTZ26]). In the setting of Model 3, let $\delta \in ( 0 , 1 )$ . Then with $p r o b a b i l i t y \ge 1 - \delta$ over the randomness in Model 3, $i f k : =$ $\begin{array} { r } { 2 4 ( \bar { k } + \log \frac { 1 } { \delta } ) , \mathrm { ~ \mathbf ~ X ~ } \sim _ { \mathrm { i . i . d . ~ } } \mathcal { N } ( 0 , \frac { 1 } { n } ) } \end{array}$ and $\begin{array} { r } { n = \Omega ( k \log { \frac { d } { \delta } } ) } \end{array}$ for a large enough constant, there is a subset $\mathcal { U } \subseteq [ d ]$ and a vector $\mathbf { z } \in \mathbb { R } ^ { d }$ that can be computed in time $\begin{array} { r } { O ( n d \log \frac { d } { \delta \operatorname* { m i n } \{ 1 , \sigma \} } ) } \end{array}$ , satisfying

$$
| \mathcal { U } ^ { c } | = O ( k ) , \quad \mathrm { T V } \left( \widehat { \pi } _ { \mathrm { s u p p } } , \pi _ { \mathrm { s u p p } } \right) \leq \delta ,
$$

where $\hat { \pi } _ { \mathrm { s u p p } }$ is supported on $S \cup \mathcal { U } ^ { c }$ where $S \subseteq \mathcal { U } , \mathcal { U } ^ { c } : = [ d ] \backslash \mathcal { U }$ , with

$$
\hat { \pi } _ { \mathrm { s u p p } } ( S \cup \mathcal { U } ^ { c } ) \propto \left( \prod _ { i \in S } \frac { \mathbf { q } _ { i } } { 1 - \mathbf { q } _ { i } } \right) \exp \left( \frac { 1 } { 2 } \left. \mathbf { z } _ { S \cup \mathcal { U } ^ { c } } \right. _ { \mathbf { A } _ { S \cup \mathcal { U } ^ { c } } ^ { - 1 } } ^ { 2 } \right) \frac { 1 } { \sqrt { \operatorname* { d e t } \mathbf { A } _ { S \cup \mathcal { U } ^ { c } } } } \cdot \mathbb { I } ( | S | \leq k ) .\tag{41}
$$

Moreover, z admits a decomposition ${ \bf z } = { \bf z } ^ { ( 0 ) } + { \bf z } ^ { ( 1 ) }$ , such that for any constant $a ,$ , there exist constants $C _ { 0 } , C _ { 1 } > 0$ (where $C _ { 1 }$ depends only on a) with

$$
\left\| \mathbf { z } ^ { ( 0 ) } \right\| _ { \infty } \leq C _ { 0 } \left( \sigma + { \frac { 1 } { \sigma } } \right) \sqrt { L } , \quad \left\| \mathbf { z } ^ { ( 1 ) } \right\| _ { a k , 2 } \leq { \frac { C _ { 1 } k L } { \sigma { \sqrt { n } } } } .\tag{42}
$$

Proof. We explain how to derive this result from [KSTZ25, CLTZ26], as it is not stated in this form. First, $\mathcal { U } ^ { c }$ is set to $\operatorname { s u p p } ( \widehat { \pmb \theta } )$ where $\widehat { \pmb { \theta } }$ is the estimator used by Lemma 6 of [KSTZ25] satisfying

$$
\left. \widehat { \pmb { \theta } } - \pmb { \theta } ^ { \star } \right. _ { \infty } = O \left( \sigma \sqrt { L } \right) ,\tag{43}
$$

with probability $\geq 1 - \frac { \delta } { 8 }$ . Also, $\| \pmb \theta ^ { \star } \| _ { \infty } = O ( \sqrt { L } )$ under Model 3 with probability $\geq 1 - \frac { \delta } { 8 }$ , so

$$
\left\| \widehat { \pmb \theta } \right\| _ { \infty } = O \left( ( 1 + \sigma ) \sqrt { L } \right) .\tag{44}
$$

The existence of such an estimator $\widehat { \pmb { \theta } }$ that takes inputs $( \mathbf { X } , \mathbf { y } )$ , runs within the stated runtime, and satisfies (43) and $| \mathrm { s u p p } ( \widehat { \pmb { \theta } } ) | = { \cal O } ( k )$ follows from Theorem 3, [CLTZ26] for Gaussian ensembles. We note that the bound in (43) is obtained by using the tighter error bound for Gaussian observation matrices, discussed at the end of Page 15, [CLTZ26]. The closeness of $\pi _ { \mathrm { s u p p } }$ and $\hat { \pi } _ { \mathrm { s u p p } }$ then follows from Lemma 6 in [KSTZ25], where the form of $\hat { \pi } _ { \mathrm { s u p p } }$ comes from Fact 3.

Next, following Eq. (23) in [KSTZ25], we let

$$
\begin{array} { r l } & { \mathbf { z } : = \cfrac { 1 } { \sigma ^ { 2 } } \mathbf { X } ^ { \top } \left( \mathbf { X } \theta ^ { \star } + \xi - \mathbf { X } \widehat { \theta } \right) - \widehat { \theta } } \\ & { \quad = \cfrac { 1 } { \sigma ^ { 2 } } \left( \mathbf { X } ^ { \top } \xi + \theta ^ { \star } - \widehat { \theta } \right) - \widehat { \theta } + \underbrace { \frac { 1 } { \sigma ^ { 2 } } \mathbf { E } \left( \theta ^ { \star } - \widehat { \theta } \right) } _ { : = \mathbf { z } ^ { ( 0 ) } } . } \end{array}
$$

Clearly z can be computed given knowledge of ${ \widehat { \pmb \theta } } ,$ because $\mathbf { y } = \mathbf { X } \pmb { \theta } ^ { \star } + \pmb { \xi }$ is given as an input. We now verify the conditions (42). For $\mathbf { z } ^ { ( 0 ) }$ , the stated bound in (42) follows by combining Lemma 1 of [KSTZ25], which gives $\| \mathbf { X } ^ { \top } \pmb { \xi } \| _ { \infty } = O ( \sigma \sqrt { L } )$ except with probability $\frac { \delta } { 4 }$ , with (43) and (44).

For $\mathbf { z } ^ { ( 1 ) }$ , we first condition on nn ${ \mathfrak { a } } ( \theta ^ { \star } ) \leq k$ , which occurs with probability $\geq 1 - \frac { \delta } { 4 }$ by Corollary 1, [KSTZ25]. Thus, ${ \pmb \theta } ^ { \star } - \widehat { \pmb \theta }$ is bk-sparse for a constant b. The bound in (42) then follows from Lemma 17 with $\mathbf { v }  \pmb { \theta } ^ { \star } - \hat { \pmb { \theta } }$ , and $s \gets a k$ , where

$$
\left\| \theta ^ { \star } - \widehat \theta \right\| _ { 2 } = O ( \sigma \sqrt { k L } ) , \quad \left\| { \bf E } \right\| _ { ( a + b ) k , \mathrm { o p } } = O \left( \sqrt { \frac { k L } { n } } \right) .
$$

The first inequality above follows from (43) and our sparsity bound. For the second, following the proof of Corollary 4 up until (30), and adjusting the failure probability to union bound over subsets as in that proof, shows that for all $s \in [ d ]$ , with probability $\geq 1 - \frac { \delta } { 4 }$

$$
\left\| \mathbf { E } \right\| _ { s , \mathrm { o p } } = O \left( { \sqrt { \frac { s L } { n } } } + { \frac { s L } { n } } \right) .\tag{45}
$$

Finally, the claim follows from a union bound over all five random events in the proof.

Remark 4. For notational simplicity, in Section 6.2 we present the argument assuming $\mathcal { C } : = \mathcal { U } ^ { c } = \emptyset$ in Proposition 3. The general case of C is identical up to a universal constant-factor enlargement of the sparsity k. Indeed, write the full-support potential in (41) as

$$
F ( T ) : = \sum _ { i \in T } \log { \frac { \mathbf { q } _ { i } } { 1 - \mathbf { q } _ { i } } } + { \frac { 1 } { 2 } } \mathbf { z } _ { T } ^ { \top } \mathbf { A } _ { T } ^ { - 1 } \mathbf { z } _ { T } - { \frac { 1 } { 2 } } \log \operatorname* { d e t } \mathbf { A } _ { T } ~ f o r ~ a l l ~ { \mathcal { C } } \subseteq T \subseteq [ d ] .
$$

Then the posterior on the collapsed support $\emptyset \subseteq S \subseteq { \mathcal { U } }$ is $\propto \exp ( F c ( S ) ) \mathbb { I } ( | S | \le k )$ , where $F c ( S ) : =$ $F ( { \mathcal { C } } \cup S )$ . All conclusions in Section 6.2 go through unchanged after lifting by C and adjusting constant factors to account for $| { \mathcal { C } } | = O ( k )$ . For example, Lemma 19 applies verbatim after replacing $R  \mathcal { C } \cup R _ { \mathrm { : } }$ , and since we only considered $| R | \leq k$ , the same sparsity bounds hold up to constant factors after this adjustment. Similarly, Lemma 20 considers diferences between potentials $F ( S )$ which goes through unchanged after replacing $F  F _ { C }$ and $S  { \mathcal { C } } \cup S$

In the rest of the section, we fix the universe U returned by Proposition 3. We write $\textstyle \mathcal { U } _ { \leq k } : = \bigcup _ { s = 0 } ^ { k } \mathcal { U } _ { s }$ analogously to (13). Conditioned on the success of Proposition 3, it is enough to sample from the density $\hat { \pi } _ { \mathrm { s u p p } }$ in (41). We write the (unnormalized) density as $\exp ( F ( S ) )$ , where we define

$$
F ( S ) = \sum _ { i \in S } \log { \frac { \mathbf { q } _ { i } } { 1 - \mathbf { q } _ { i } } } + { \frac { 1 } { 2 } } \mathbf { z } _ { S } ^ { \top } \mathbf { A } _ { S } ^ { - 1 } \mathbf { z } _ { S } - { \frac { 1 } { 2 } } \log \operatorname* { d e t } \mathbf { A } _ { S } .
$$

We now manipulate this expression to be on a more convenient scale. In particular, let

$$
\gamma : = \frac { \sigma ^ { 2 } } { 1 + \sigma ^ { 2 } } , \quad \tau : = \frac { 1 } { 1 + \sigma ^ { 2 } } , \quad \mathbf { s } : = \sqrt { \gamma \tau } \mathbf { z } , \quad \mathbf { L } _ { S } : = \gamma \mathbf { A } _ { S } = \mathbf { I } _ { S } + \tau \mathbf { E } _ { S } \mathrm { ~ f o r ~ a l l ~ } S \subseteq [ d ] .\tag{46}
$$

Under this scaling, and defining $\mathbf { s } ^ { ( 0 ) } : = \sqrt { \gamma \tau } \mathbf { z } ^ { ( 0 ) }$ and $\mathbf { s } ^ { ( 1 ) } : = \sqrt { \gamma \tau } \mathbf { z } ^ { ( 1 ) }$ , (42) implies

$$
\left\| \mathbf { s } ^ { ( 0 ) } \right\| _ { \infty } \leq C _ { 0 } \sqrt { L } , \quad \left\| \mathbf { s } ^ { ( 1 ) } \right\| _ { a k , 2 } \leq \frac { C _ { 1 } k L } { \sqrt { n } } .\tag{47}
$$

Moreover, again using the notation (46) and combining all linear terms, we obtain

$$
F ( S ) = { \bf h } ^ { \top } { \bf 1 } _ { S } + \frac { 1 } { 2 \tau } { \bf s } _ { S } ^ { \top } { \bf L } _ { S } ^ { - 1 } { \bf s } _ { S } - \frac { 1 } { 2 } \log \operatorname* { d e t } { \bf L } _ { S } , \mathrm { ~ w h e r e ~ } { \bf h } _ { i } : = \log \frac { { \bf q } _ { i } } { 1 - { \bf q } _ { i } } + \frac { 1 } { 2 } \log \gamma .\tag{48}
$$

Finally, for $\beta \in [ 0 , 1 ]$ and $0 \leq s \leq k$ , we define

$$
\begin{array} { r } { \nu _ { \leq s , \beta } ( S ) \propto \exp ( \beta F ( S ) ) \mathbb { I } ( S \in \mathcal { U } _ { \leq s } ) , \quad \nu _ { s , \beta } ( S ) \propto \exp ( \beta F ( S ) ) \mathbb { I } ( S \in \mathcal { U } _ { s } ) . } \end{array}\tag{49}
$$

With this notation, our target support posterior is $\nu = \nu { \le } k , 1$

## 6.2 Trickle down for support posterior

In this section, we apply the trickle down framework of Section 4 to sampling under Model 3. We begin by collecting several additional estimates we require of our draw from the model.

Lemma 18. Under Model 3, assume $n = \Omega ( k L )$ for an appropriate constant. Under the success of the event in Proposition 3, we have the following additional guarantee, for a universal constant $C > 0$ , and $\alpha : = \sqrt { L / n }$ . Simultaneously for every $R \in { \mathcal { U } } _ { \leq k } , B : = { \mathcal { U } } \setminus R$ , and u, $\mathbf { v } \in \mathbb { R } ^ { d }$ 2

$$
\| \mathbf { E } \| _ { 2 k , \mathrm { o p } } \leq C \alpha \sqrt { k } , \quad \operatorname* { m a x } _ { ( i , j ) \in [ d ] \times [ d ] } | \mathbf { E } _ { i j } | \leq C \alpha , \quad \| \mathbf { E } _ { R \times B } \mathbf { u } _ { B } \| _ { 2 } \leq C \alpha \left( \sqrt { k } \| \mathbf { u } \| _ { 2 } + \| \mathbf { u } \| _ { 1 } \right) ,
$$

and

$$
\left| { \bf u } ^ { \top } { \bf E } { \bf v } \right| \leq C \left( \alpha \left( \left\| { \bf u } \right\| _ { 1 } \left\| { \bf v } \right\| _ { 2 } + \left\| { \bf u } \right\| _ { 2 } \left\| { \bf v } \right\| _ { 1 } \right) + \alpha ^ { 2 } \left\| { \bf u } \right\| _ { 1 } \left\| { \bf v } \right\| _ { 1 } \right) .\tag{50}
$$

Proof. The only event we condition on in this proof is that (45) holds. The first two inequalities then follow by plugging $s = 2 k$ and $s = 2$ into (45), and using our lower bound on n. For inequality (50), assume $\mathbf { u } , \mathbf { v } \neq \mathbf { 0 } _ { d }$ , else the claim is immediate. Then let

$$
t ^ { 2 } : = \frac { \| \mathbf { v } \| _ { 1 } } { \| \mathbf { u } \| _ { 1 } } \implies t \| \mathbf { u } \| _ { 1 } + \frac { 1 } { t } \| \mathbf { v } \| _ { 1 } = 2 \sqrt { \| \mathbf { u } \| _ { 1 } \| \mathbf { v } \| _ { 1 } } .
$$

Since

$$
\mathbf { u } ^ { \mathsf { T } } \mathbf { E } \mathbf { v } = { \frac { 1 } { 4 } } \left( t \mathbf { u } + { \frac { 1 } { t } } \mathbf { v } \right) ^ { \mathsf { T } } \mathbf { E } \left( t \mathbf { u } + { \frac { 1 } { t } } \mathbf { v } \right) - { \frac { 1 } { 4 } } \left( t \mathbf { u } - { \frac { 1 } { t } } \mathbf { v } \right) ^ { \mathsf { T } } \mathbf { E } \left( t \mathbf { u } - { \frac { 1 } { t } } \mathbf { v } \right) ,
$$

the claim follows from Lemma 11 applied to the two terms, the triangle inequality, and

$$
\begin{array} { r l } & { \left( t \left\| \mathbf { u } \right\| _ { 1 } + \displaystyle \frac { 1 } { t } \left\| \mathbf { v } \right\| _ { 1 } \right) \left( t \left\| \mathbf { u } \right\| _ { 2 } + \displaystyle \frac { 1 } { t } \left\| \mathbf { v } \right\| _ { 2 } \right) = 2 \left( \left\| \mathbf { u } \right\| _ { 1 } \left\| \mathbf { v } \right\| _ { 2 } + \left\| \mathbf { u } \right\| _ { 2 } \left\| \mathbf { v } \right\| _ { 1 } \right) , } \\ & { \quad \quad \quad \quad \left( t \left\| \mathbf { u } \right\| _ { 1 } + \displaystyle \frac { 1 } { t } \left\| \mathbf { v } \right\| _ { 1 } \right) ^ { 2 } = 4 \left\| \mathbf { u } \right\| _ { 1 } \left\| \mathbf { v } \right\| _ { 1 } . } \end{array}
$$

For the third inequality, applying (50) and the fact that $\| \mathbf { E } _ { R \times B } \mathbf { u } \| _ { 2 } = \mathbf { v } ^ { \top } \mathbf { E } \mathbf { u } _ { B }$ for some unit vector v supported only on R, such that $\| \mathbf { v } \| _ { 1 } \leq { \sqrt { k } }$ , yields the claim upon simplifying with $\alpha \sqrt { k } = { \cal O } ( 1 )$ □

Conditioned on the success of the event in Lemma 18, we next show how to apply our framework in Section 4 to sample from the posterior density $\nu _ { \leq k , 1 } \ ( 4 9 )$ . We begin with an exact characterization of the potential gain by including a set of coordinates.

Lemma 19. For a fixed $R \subseteq { \mathcal { U } }$ , let $B : = \mathcal { U } \setminus R$ and define

$$
\mathbf { R } _ { R } : = \mathbf { E } _ { B \times B } - \tau \mathbf { E } _ { B \times R } \mathbf { L } _ { R } ^ { - 1 } \mathbf { E } _ { R \times B } , \quad \mathbf { r } _ { R } : = \mathbf { s } _ { B } - \tau \mathbf { E } _ { B \times R } \mathbf { L } _ { R } ^ { - 1 } \mathbf { s } _ { R } .\tag{51}
$$

Then, for every $T \subseteq B$ , letting ${ \bf R } _ { T } : = [ { \bf R } _ { R } ] _ { T \times T }$ and $\mathbf { r } _ { T } : = [ \mathbf { r } _ { R } ] _ { T }$

$$
F ( R \cup T ) - F ( R ) = \mathbf { h } ^ { \top } \mathbf { 1 } _ { T } + { \frac { 1 } { 2 \tau } } \mathbf { r } _ { T } ^ { \top } \left( \mathbf { I } _ { T } + \tau \mathbf { R } _ { T } \right) ^ { - 1 } \mathbf { r } _ { T } - { \frac { 1 } { 2 } } \log \operatorname* { d e t } \left( \mathbf { I } _ { T } + \tau \mathbf { R } _ { T } \right) .
$$

Proof. With the coordinates ordered as $R , T$

$$
\mathbf { L } _ { R \cup T } = \left( \begin{array} { c c } { \mathbf { L } _ { R } } & { \tau \mathbf { E } _ { R \times T } } \\ { \tau \mathbf { E } _ { T \times R } } & { \mathbf { L } _ { T } } \end{array} \right) .
$$

Its Schur complement with respect to the R block is

$$
\mathbf { L } _ { T } - \tau ^ { 2 } \mathbf { E } _ { T \times R } \mathbf { L } _ { R } ^ { - 1 } \mathbf { E } _ { R \times T } = \mathbf { I } _ { T } + \tau \mathbf { R } _ { T } .
$$

Consequently, det ${ \bf L } _ { R \cup T } = \operatorname* { d e t } ( { \bf L } _ { R } ) \operatorname* { d e t } \left( { \bf I } _ { T } + \tau { \bf R } _ { T } \right)$ . The block inverse formula also gives

$$
\begin{array} { r l } & { { \mathbf { s } } _ { R \cup T } ^ { \top } { \mathbf { L } } _ { R \cup T } ^ { - 1 } { \mathbf { s } } _ { R \cup T } = { \mathbf { s } } _ { R } ^ { \top } { \mathbf { L } } _ { R } ^ { - 1 } { \mathbf { s } } _ { R } + \left( { \mathbf { s } } _ { T } - \tau { \mathbf { E } } _ { T \times R } { \mathbf { L } } _ { R } ^ { - 1 } { \mathbf { s } } _ { R } \right) ^ { \top } \left( { \mathbf { I } } _ { T } + \tau { \mathbf { R } } _ { T } \right) ^ { - 1 } \left( { \mathbf { s } } _ { T } - \tau { \mathbf { E } } _ { T \times R } { \mathbf { L } } _ { R } ^ { - 1 } { \mathbf { s } } _ { R } \right) } \\ & { \qquad = { \mathbf { s } } _ { R } ^ { \top } { \mathbf { L } } _ { R } ^ { - 1 } { \mathbf { s } } _ { R } + { \mathbf { r } } _ { T } ^ { \top } \left( { \mathbf { I } } _ { T } + \tau { \mathbf { R } } _ { T } \right) ^ { - 1 } { \mathbf { r } } _ { T } . } \end{array}
$$

Substituting both of the above displays into (48) now proves the claim.

We now give the main technical result in this section, which provides various estimates required to apply the parameter bounds from Lemma 9 to an appropriate interaction matrix.

Lemma 20. Assume the events of Proposition 3 and Lemma 18 hold. Let $R \in \mathcal { U } { \leq } k$ and $B : = \mathcal { U } \backslash R$ and following notation of Lemma 19, let $\mathbf { K } \in \mathbb { R } ^ { B \times B }$ have zero diagonal and satisfy

$$
\mathbf { K } _ { i j } : = F \left( R \cup \{ i , j \} \right) - F \left( R \cup \{ i \} \right) - F \left( R \cup \{ j \} \right) + F ( R ) \ f o r \ a l l \ ( i , j ) \in B \times B , \ i \neq j .
$$

Then the following bounds hold for a universal constant $C ^ { \prime } > 0$ , letting $\alpha : = \sqrt { L / n }$

1. For all $\mathbf { u } \in \mathbb { R } ^ { B }$ , following the notation (51),

$$
\left| \sum _ { \stackrel { \left( i , j \right) \in B \times B } { i \neq j } } \mathbf { R } _ { i j } \mathbf { u } _ { i } \mathbf { u } _ { j } \right| \leq C ^ { \prime } \left( \alpha ^ { 2 } k \left\| \mathbf { u } \right\| _ { 2 } ^ { 2 } + \alpha \left\| \mathbf { u } \right\| _ { 1 } \left\| \mathbf { u } \right\| _ { 2 } + \alpha ^ { 2 } \left\| \mathbf { u } \right\| _ { 1 } ^ { 2 } \right) .
$$

2. For all $( i , j ) \in B \times B , i \neq j$

$$
\mathbf { { K } } _ { i j } = - \mathbf { { R } } _ { i j } \mathbf { { r } } _ { i } \mathbf { { r } } _ { j } + \epsilon _ { i j } , ~ f o r ~ \left| \epsilon _ { i j } \right| \leq C ^ { \prime } \left( \alpha ^ { 2 } + \alpha ^ { 4 } k ^ { 2 } \right) \left( 1 + \mathbf { { r } } _ { i } ^ { 2 } + \mathbf { { r } } _ { j } ^ { 2 } \right) .
$$

3. For all $\mathbf { u } \in \mathbb { R } ^ { B }$

$$
\left\| \mathbf { r } \circ \mathbf { u } \right\| _ { 1 } \leq C ^ { \prime } \sqrt { L } \left( \left\| \mathbf { u } \right\| _ { 1 } + \frac { k L } { \sqrt { n } } \left\| \mathbf { u } \right\| _ { 2 } \right) , \quad \left\| \mathbf { r } \circ \mathbf { u } \right\| _ { 2 } \leq C ^ { \prime } \left( \sqrt { L } + \frac { k L } { \sqrt { n } } \right) \left\| \mathbf { u } \right\| _ { 2 } ,
$$

and

$$
\sum _ { i \in B } | \mathbf { u } _ { i } | \mathbf { r } _ { i } ^ { 2 } \leq C ^ { \prime } \left( L \left\| \mathbf { u } \right\| _ { 1 } + \frac { k ^ { 2 } L ^ { 2 } } { n } \left\| \mathbf { u } \right\| _ { 2 } \right) .
$$

Proof. We proceed with the three claims in order. First, observe that by Lemma 18 and $| R | \leq k$

$$
\left\| \mathbf { L } _ { R } ^ { - 1 } \right\| _ { \mathrm { o p } } \leq \left( 1 - C \tau \alpha \sqrt { k } \right) ^ { - 1 } \leq 2 ,\tag{52}
$$

using $\tau \leq 1$ and taking n large enough. Also, again by applying Lemma 18,

$$
\begin{array} { r } { \| { \bf R } \| _ { \mathrm { m a x } } : = \underset { ( i , j ) \in B \times B } { \operatorname* { m a x } } | { \bf R } _ { i j } | \leq \underset { ( i , j ) \in B \times B } { \operatorname* { m a x } } | { \bf E } _ { i j } | + \left\| { \bf L } _ { R } ^ { - 1 } \right\| _ { \mathrm { o p } } \left\| { \bf E } _ { R \times \{ i \} } \right\| _ { 2 } \left\| { \bf E } _ { R \times \{ j \} } \right\| _ { 2 } } \\ { \leq C \alpha + 2 \left( C \alpha ( \sqrt { k } + 1 ) \right) ^ { 2 } \leq C \alpha + 8 C ^ { 2 } \alpha ^ { 2 } k \leq \frac { 1 } { 2 } . } \end{array}\tag{53}
$$

The last inequality again used our lower bound on n. Hence,

$$
\left. \sum _ { ( i , j ) \in { \cal B } \times { \cal B } } { \bf R } _ { i j } { \bf u } _ { i } { \bf u } _ { j } \right. \leq \left. { \bf u } ^ { \top } { \bf R } { \bf u } \right. + \left\| { \bf R } \right\| _ { \operatorname* { m a x } { \bf \Pi } } \left\| { \bf u } \right\| _ { 2 } ^ { 2 }
$$

$$
\leq \left| \mathbf { u } ^ { \top } \mathbf { R } \mathbf { u } \right| + 8 C ^ { 2 } \alpha ^ { 2 } k \left\| \mathbf { u } \right\| _ { 2 } ^ { 2 } + C \alpha \left\| \mathbf { u } \right\| _ { 1 } \left\| \mathbf { u } \right\| _ { 2 } ,
$$

and by again applying Lemma 18, we have

$$
\begin{array} { r l } & { \left| { \bf { u } } ^ { \top } { \bf { R } } { \bf { u } } \right| \leq \left| { \bf { u } } ^ { \top } { \bf { E } } { \bf { u } } \right| + 2 \left\| { \bf { E } } _ { R \times B } { \bf { u } } \right\| _ { 2 } ^ { 2 } } \\ & { \qquad \leq 2 C \left( \alpha \left\| { \bf { u } } \right\| _ { 1 } \left\| { \bf { u } } \right\| _ { 2 } + \alpha ^ { 2 } \left\| { \bf { u } } \right\| _ { 1 } ^ { 2 } \right) + 4 C ^ { 2 } \left( \alpha ^ { 2 } k \left\| { \bf { u } } \right\| _ { 2 } ^ { 2 } + \alpha ^ { 2 } \left\| { \bf { u } } \right\| _ { 1 } ^ { 2 } \right) . } \end{array}
$$

Combining the above two displays proves Item 1.

Next, for all $i \in B$ , let $d _ { i } : = 1 + \tau { \bf R } _ { i i }$ so that $d _ { i } \in [ 1 \pm \tau \| \mathbf { R } \| _ { \operatorname* { m a x } } ] \subseteq [ \frac { 1 } { 2 } , \frac { 3 } { 2 } ]$ . Also, let

$$
D _ { i j } : = \operatorname* { d e t } \left( \mathbf { I } _ { \{ i , j \} } + \tau \mathbf { R } _ { \{ i , j \} } \right) = d _ { i } d _ { j } - \tau ^ { 2 } \mathbf { R } _ { i j } ^ { 2 }
$$

so that

$$
\left( \mathbf { I } _ { \{ i , j \} } + \tau \mathbf { R } _ { \{ i , j \} } \right) ^ { - 1 } = \left( \begin{array} { c c } { d _ { i } } & { \tau \mathbf { R } _ { i j } } \\ { \tau \mathbf { R } _ { i j } } & { d _ { j } } \end{array} \right) ^ { - 1 } = \frac { 1 } { D _ { i j } } \left( \begin{array} { c c } { d _ { j } } & { - \tau \mathbf { R } _ { i j } } \\ { - \tau \mathbf { R } _ { i j } } & { d _ { i } } \end{array} \right) .
$$

Then, applying Lemma 19 with $T = \emptyset , \{ i \} , \{ j \} , \{ i , j \}$ , the linear terms cancel, so

$$
\begin{array} { l } { { \displaystyle { \bf K } _ { i j } = \frac { 1 } { 2 \tau D _ { i j } } \left( d _ { j } { \bf r } _ { i } ^ { 2 } + d _ { i } { \bf r } _ { j } ^ { 2 } - 2 \tau { \bf R } _ { i j } { \bf r } _ { i } { \bf r } _ { j } \right) - \frac { 1 } { 2 \tau } \left( \frac { { \bf r } _ { i } ^ { 2 } } { d _ { i } } + \frac { { \bf r } _ { j } ^ { 2 } } { d _ { j } } \right) - \frac { 1 } { 2 } \log \left( D _ { i j } \right) + \frac { 1 } { 2 } \log \left( d _ { i } d _ { j } \right) } }  \\ { { \displaystyle { \bf \Lambda } = \frac { \tau { \bf R } _ { i j } ^ { 2 } \left( \frac { { \bf r } _ { i } ^ { 2 } } { d _ { i } } + \frac { { \bf r } _ { j } ^ { 2 } } { d _ { j } } \right) - 2 { \bf R } _ { i j } { \bf r } _ { i } { \bf r } _ { j } } } { 2 D _ { i j } } - \frac { 1 } { 2 } \log \left( 1 - \frac { \tau ^ { 2 } { \bf R } _ { i j } ^ { 2 } } { d _ { i } d _ { j } } \right) }  \\ { { \displaystyle { \bf \Lambda } = - \frac { { \bf R } _ { i j } { \bf r } _ { i } { \bf r } _ { j } } { D _ { i j } } + \frac { \tau { \bf R } _ { i j } ^ { 2 } } { 2 D _ { i j } } \left( \frac { { \bf r } _ { i } ^ { 2 } } { d _ { i } } + \frac { { \bf r } _ { j } ^ { 2 } } { d _ { j } } \right) - \frac { 1 } { 2 } \log \left( 1 - \frac { \tau ^ { 2 } { \bf R } _ { i j } ^ { 2 } } { d _ { i } d _ { j } } \right) } . } \end{array}\tag{54}
$$

Using the estimates $| d _ { i } - 1 | , | d _ { j } - 1 | , | D _ { i j } - 1 | = O ( \| \mathbf { R } \| _ { \operatorname* { m a x } } ) = O ( \alpha + \alpha ^ { 2 } k )$ and Taylor expanding each error term (since (53) bounds $\lVert \mathbf { R } \rVert _ { \operatorname* { m a x } }$ by an arbitrary constant), gives for large enough $C ^ { \prime } { \mathrm { . } }$

$$
\begin{array} { r l } & { | { \bf R } _ { i j } { \bf r } _ { i } { \bf r } _ { j } | \cdot | \frac { 1 } { D _ { i j } } - 1 | \leq \displaystyle \frac { C ^ { \prime } } { 6 } ( \alpha + \alpha ^ { 2 } k ) ^ { 2 } ( { \bf r } _ { i } ^ { 2 } + { \bf r } _ { j } ^ { 2 } ) , } \\ & { \quad \frac { \tau { \bf R } _ { i j } ^ { 2 } } { 2 D _ { i j } } ( \frac { { \bf r } _ { i } ^ { 2 } } { d _ { i } } + \frac { { \bf r } _ { j } ^ { 2 } } { d _ { j } } ) \leq \displaystyle \frac { C ^ { \prime } } { 6 } ( \alpha + \alpha ^ { 2 } k ) ^ { 2 } ( { \bf r } _ { i } ^ { 2 } + { \bf r } _ { j } ^ { 2 } ) , } \\ & { \quad \displaystyle | \log ( 1 - \frac { \tau ^ { 2 } { \bf R } _ { i j } ^ { 2 } } { d _ { i } d _ { j } } ) | \leq \displaystyle \frac { C ^ { \prime } } { 6 } ( \alpha + \alpha ^ { 2 } k ) ^ { 2 } . } \end{array}
$$

Combining these bounds within (54), and using $( a + b ) ^ { 2 } \leq 2 ( a ^ { 2 } + b ^ { 2 } )$ , then yields Item 2.

To conclude, let $\mathbf { c } : = \mathbf { E } _ { B \times R } \mathbf { L } _ { R } ^ { - 1 } \mathbf { s } _ { R }$ and $\mathbf { d } : = \mathbf { s } _ { B } ^ { ( 1 ) } - \boldsymbol { \tau } \mathbf { c } ,$ , so that following Lemma 19 and $\mathbf { s } = \mathbf { s } ^ { ( 0 ) } + \mathbf { s } ^ { ( 1 ) }$ we have $\mathbf { r } _ { R } = \mathbf { s } _ { B } ^ { ( 0 ) } + \mathbf { d }$ . Recall from (47) that ${ \bf s } _ { B } ^ { ( 0 ) }$ has all entries bounded by $O ( \sqrt { L } )$ , such that

$$
\| \mathbf { s } ^ { ( 0 ) } \circ \mathbf { u } \| _ { 1 } = O ( { \sqrt { L } } ) \| \mathbf { u } \| _ { 1 } , \quad \| \mathbf { s } ^ { ( 0 ) } \circ \mathbf { u } \| _ { 2 } = O ( { \sqrt { L } } ) \| \mathbf { u } \| _ { 2 } , \quad \sum _ { i \in B } | \mathbf { u } _ { i } | ( \mathbf { s } _ { i } ^ { ( 0 ) } ) ^ { 2 } = O \left( L \right) \| \mathbf { u } \| _ { 1 } .
$$

By choosing $C ^ { \prime }$ large enough in Item 3, all of the above contributions fit asymptotically within the claimed budgets. It thus sufices to prove Item 3 using d to reweight u rather than r.

Next, we bound $\| \mathbf { d } \| _ { k , 2 }$ . Let $T \subseteq B$ index the largest k coordinates of d by magnitude. By (47) and $n = \Omega ( k L )$ , for every $R \in \mathcal { U } _ { \leq k }$

$$
\| \mathbf { s } _ { R } \| _ { 2 } \leq \sqrt { k } \left\| \mathbf { s } ^ { ( 0 ) } \right\| _ { \infty } + \left\| \mathbf { s } ^ { ( 1 ) } \right\| _ { k , 2 } = O ( \sqrt { k L } ) .
$$

Then, (52) and the third bound in Lemma 18 imply, for a large enough constant $C ^ { \prime \prime }$

$$
\| \mathbf { c } _ { T } \| _ { 2 } \leq 2 C \alpha \sqrt { k } \cdot O ( \sqrt { k L } ) \leq k L \sqrt { \frac { C ^ { \prime \prime } } { n } } .
$$

Combining with (47) and $\tau \leq 1$ gives $\begin{array} { r } { \| \mathbf { d } _ { T } \| _ { 2 } \leq k L \sqrt { \frac { C ^ { \prime \prime } } { n } } } \end{array}$ after adjusting $C ^ { \prime \prime }$

Thus, letting the $i ^ { \mathrm { t h } }$ largest magnitude amongst d’s coordinates be denoted $d _ { ( i ) }$ , we have shown

$$
d _ { ( i ) } ^ { 2 } \leq { \frac { C ^ { \prime \prime } k ^ { 2 } L ^ { 2 } } { n } } \cdot { \frac { 1 } { \operatorname* { m i n } ( i , k ) } } , { \mathrm { ~ f o r ~ a l l ~ } } i \in | B | .\tag{55}
$$

Now, similarly denoting the $i ^ { \mathrm { t h } }$ largest magnitude amongst u’s coordinates by $u _ { ( i ) }$ ,

$$
\begin{array} { r l r } {  { \| \mathbf { d } \circ \mathbf { u } \| _ { 1 } \leq \sqrt { \frac { C ^ { \prime \prime } k ^ { 2 } L ^ { 2 } } { n } } \cdot ( \sum _ { i \in [ [ B ] ] } \frac { u _ { ( i ) } } { \sqrt { i } } + \frac { u _ { ( i ) } } { \sqrt { k } } ) } } \\ & { } & { \leq C ^ { \prime } \sqrt { L } \| \mathbf { u } \| _ { 1 } + \sqrt { \frac { C ^ { \prime \prime } k ^ { 2 } L ^ { 2 } } { n } } \cdot \sqrt { ( \sum _ { i \in [ [ B ] ] } \frac { 1 } { i } ) \| \mathbf { u } \| _ { 2 } ^ { 2 } } \leq C ^ { \prime } \sqrt { L } \| \mathbf { u } \| _ { 1 } + \frac { C ^ { \prime } k L ^ { 1 . 5 } } { \sqrt { n } } \| \mathbf { u } \| _ { 2 } , } \end{array}\tag{56}
$$

where the second inequality was by Cauchy-Schwarz. Similarly,

$$
\left. \mathbf { d } \circ \mathbf { u } \right. _ { 2 } ^ { 2 } \leq \frac { C ^ { \prime \prime } k ^ { 2 } L ^ { 2 } } { n } \cdot \left( \sum _ { i \in [ | B | ] } \frac { u _ { ( i ) } ^ { 2 } } { i } + \frac { u _ { ( i ) } ^ { 2 } } { k } \right) \leq \frac { C ^ { \prime \prime } k ^ { 2 } L ^ { 2 } } { n } \left. \mathbf { u } \right. _ { 2 } ^ { 2 } .
$$

Finally, recalling $n = \Omega ( k L )$ , and applying the Cauchy-Schwarz inequality as in (56),

$$
\sum _ { i \in B } | \mathbf { u } _ { i } | \mathbf { d } _ { i } ^ { 2 } \leq \sum _ { i \in [ | B | ] } u _ { ( i ) } d _ { ( i ) } ^ { 2 } \leq \frac { C ^ { \prime \prime } k ^ { 2 } L ^ { 2 } } { n } \cdot \left( \sum _ { i \in [ | B | ] } \frac { u _ { ( i ) } } { i } + \frac { u _ { ( i ) } } { k } \right) \leq \frac { C ^ { \prime } k ^ { 2 } L ^ { 2 } } { n } \| \mathbf { u } \| _ { 2 } + C ^ { \prime } L \left\| \mathbf { u } \right\| _ { 1 } .
$$

We are finally ready to give our application of the trickle down framework.

Proposition 4. Assume the events of Proposition 3 and Lemma 18 hold, and that

$$
n = \Omega \left( k ^ { 1 . 5 } L ^ { 2 } + k L ^ { 3 } \right)
$$

for a suficiently large constant. Then for all $2 \leq s \leq k$ and $\beta \in [ 0 , 1 ]$ , following the notation (49), $\mathcal { T } ^ { \mathsf { D U } , \nu _ { s , \beta } }$ satisfies a <sup>1</sup> -Poincaré inequality.

Proof. Let $R \in \mathcal { U } _ { s - 2 }$ , and define $\mathbf { K } \in \mathbb { R } ^ { B \times B }$ following the notation in Lemma 20. We first claim

$$
\mathbf { u } ^ { \top } \mathbf { K } \mathbf { u } \leq \frac { 9 } { 1 6 } \left\| \mathbf { u } \right\| _ { 2 } ^ { 2 } + \frac { 1 } { 8 k } \left\| \mathbf { u } \right\| _ { 1 } ^ { 2 } , \mathrm { ~ f o r ~ a l l ~ } \mathbf { u } \in \mathbb { R } ^ { B } .\tag{57}
$$

To see this, decompose K as in Item 2 of Lemma 20, and by homogeneity, assume that $\left\| \mathbf { u } \right\| _ { 1 } = 1$ and $\| \mathbf { u } \| _ { 2 } = q$ . By combining the estimates in Items 1 and 3, and using that

$$
\begin{array} { r l } { \underset { \underset { \{ \vphantom { ( \vphantom { ( \sqrt { \pi } ) \pi } ) ( ( \frac { L } { \sqrt { \pi } ) ( \frac { R } { \sqrt { \pi } ) ( T } ) ( \frac { R } { \sqrt { \pi } ) ( T } )  \kern - delimiterspace } ( T ) } } { \sqrt { \pi } \mathrm { ~ B R } } } } { \sum } =  & { \biggr ( \sigma ( x ^ { 2 } k ( L + \frac { k ^ { 2 } L ^ { 2 } } { n } ) \sigma ^ { 2 } ) } \\ & { \qquad + O ( \mathrm { s W i } ( 1 + \frac { k L \sigma L } { \sqrt { n } } ) ( \sqrt { T } + \frac { k L } { \sqrt { n } } ) \sigma ) } \\ & { \quad + O ( \sigma ^ { 2 } L ( 1 + \frac { k ^ { 2 } L ^ { 2 } \sigma ^ { 2 } } { n } ) ) } \\ & { = O ( ( \frac { k L ^ { 2 } } { n } + \frac { k ^ { 3 } L ^ { 2 } } { n ^ { 2 } } + \frac { k L ^ { 2 } \sigma ^ { 3 } } { n } + \frac { k ^ { 2 } L ^ { 5 } } { n ^ { 4 } \Sigma ^ { 5 } } + \frac { k ^ { 2 } L ^ { 4 } } { n ^ { 2 } } ) q ^ { 2 } ) } \\ &  \underset {  \mathrm { ( ) } \vphantom { ( \frac { L ^ { 2 } \sigma ^ { 2 } } { \sqrt { n } } ) ( \frac { L ^ { 2 } \sigma ^ { 3 } } { \sqrt { n } } ) ( \frac { R } { \sqrt { \pi } } ) } { \sigma } ) \mathrm { ~ O ~ } ( \frac { L ^ { 2 } } { \sqrt { n } } ) } \\ & { \leq \frac { 1 } { 4 } q ^ { 2 } + \frac { 1 } { 1 6 \sqrt { n } } \frac { 1 } { n ^ { 4 } } + \frac { 1 } { 3 2 k } \sum _ { i } ^ { 5 } \frac { 1 } { n ^ { 4 } } + \frac { 1 } { 1 6 k } . } \end{array}
$$

where the hidden constants above depend only on $C ^ { \prime }$ . The last line then follows by taking n large enough as stated. Next, for the error term in Item 2, recalling $\| \mathbf { u } \| _ { 1 } = 1$ ，

$$
\begin{array} { r l } & { \quad \displaystyle \sum _ { ( i , j ) \in \mathbb { S } } \epsilon _ { i j 1 } \mathbf { u } _ { 1 i } \mathbf { u } _ { j } \bigg | = O \left( \alpha ^ { 2 } + \alpha ^ { 4 } k ^ { 2 } \right) \cdot \displaystyle \sum _ { ( i , j ) \in \mathbb { S } } \left| \mathbf { u } _ { 1 i } \mathbf { u } _ { j } \right| \left( 1 + \mathbf { r } _ { \mathrm { s } } ^ { 2 } + \mathbf { r } _ { j } ^ { 2 } \right) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & & { \quad \quad = O \left( \alpha ^ { 2 } + \alpha ^ { 4 } k ^ { 2 } \right) \cdot \left( 1 + 2 \displaystyle \sum _ { j \in \mathbb { S } } | \mathbf { u } _ { 1 i } | \mathbf { r } _ { j } ^ { 2 } \right) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad = O \left( \left( \alpha ^ { 2 } + \alpha ^ { 4 } k ^ { 2 } \right) \left( L + \frac { k ^ { 2 } L ^ { 2 } q } { n } \right) \right) } \\ & { \quad \quad \quad \quad \quad - O \left( \left( \frac { k ^ { 2 } L ^ { 3 } } { n ^ { 2 } } + \frac { k ^ { 4 } L ^ { 4 } } { n ^ { 3 } } \right) q \right) + O \left( \frac { L ^ { 2 } } { n } + \frac { k ^ { 2 } L ^ { 3 } } { n ^ { 2 } } \right) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \leq \frac { 1 } { 1 6 \sqrt { k } } q ^ { 4 } + \frac { 1 } { 3 2 k } \leq \frac { 1 } { 1 6 } q ^ { 2 } + \frac { 1 } { 1 6 k ^ { 2 } } , } \end{array}\tag{58}
$$

where we used Item 3 in the third line, and again simplified by taking n large enough. Combining the above two displays gives the desired (57).

Next, we observe that the link graph induced by R, in the sense of Lemma 6, has edge weight matrix W exactly given in the form required by Lemma 7, with K defined in Lemma 20 and

$$
\mathbf { a } _ { i } : = \exp \left( \beta \left( F \left( R \cup \left\{ i \right\} \right) - \frac { 1 } { 2 } F ( R ) \right) \right) \mathrm { ~ f o r ~ a l l ~ } i \in B .
$$

Indeed, for $( i , j ) \in B \times B$ with $i \neq j$

$$
\mathbf { a } _ { i } \mathbf { a } _ { j } \exp \left( \beta \mathbf { K } _ { i j } \right) = \exp \left( \beta F ( R \cup \{ i , j \} ) \right)
$$

as is required by the link graph. We now provide bounds on $m ( \beta \mathbf { K } )$ and $\rho ( \beta \mathbf { K } )$ as used in Lemma 7.

To bound $m ( \beta \mathbf { K } )$ , recall from the proof of Lemma 20 $( \mathrm { i . e . , } \ \mathbf { r } _ { R } = \mathbf { s } _ { B } ^ { ( 0 ) } + \mathbf { d }$ , (47), and (55)) that we showed $\begin{array} { r } { \left\| \mathbf { r } _ { R } \right\| _ { \infty } = \overset { \cdot } { O } ( \sqrt { L } + \frac { k L } { \sqrt { n } } ) } \end{array}$ . Thus, combining with (53) and Item 2 shows that

$$
m ( \beta \mathbf { K } ) \leq m ( \mathbf { K } ) = O \left( \left( \alpha + \alpha ^ { 2 } k \right) \left( L + \frac { k ^ { 2 } L ^ { 2 } } { n } \right) \right) + O \left( \left( \alpha ^ { 2 } + \alpha ^ { 4 } k ^ { 2 } \right) \left( L + \frac { k ^ { 2 } L ^ { 2 } } { n } \right) \right) \leq \frac { 1 } { 1 6 } .
$$

To bound $\rho ( \beta \mathbf { K } )$ , it sufices to restrict to $\left. \mathbf { u } \right. _ { 1 } = 1 , \left. \mathbf { u } \right. _ { 2 } = q$ by homogeneity. We follow the proof of Lemma 9, and write $\mathbf { X } = \beta \mathbf { K } + \mathbf { Q }$ where $\mathbf { X } _ { i j } = \exp ( \beta \mathbf { K } _ { i j } ) - 1$ entrywise. By Taylor expansion, we have that $| \mathbf { Q } _ { i j } | \leq \beta ^ { 2 } \mathbf { K } _ { i j } ^ { 2 }$ . Again, using Item 2, denoting $\begin{array} { r } { \quad A : = \alpha ^ { 2 } + \alpha ^ { 4 } k ^ { 2 } = \frac { L } { n } + \frac { k ^ { 2 } L ^ { 2 } } { n ^ { 2 } } \leq 1 } \end{array}$ 7

$$
\mathbf { K } _ { i j } ^ { 2 } = O \left( A \mathbf { r } _ { i } ^ { 2 } \mathbf { r } _ { j } ^ { 2 } + A ^ { 2 } \left( 1 + \mathbf { r } _ { i } ^ { 4 } + \mathbf { r } _ { j } ^ { 4 } \right) \right) .
$$

Thus, applying the bounds from Item 3, and simplifying,

$$
\begin{array} { r l } & { \left| { { \bf { u } } ^ { \top } } { \bf { { Q } } } { \bf { u } } \right| = O \left( A \left( { L ^ { 2 } } + \frac { { k ^ { 4 } { L ^ { 4 } } { q ^ { 2 } } } } { n ^ { 2 } } \right) + A ^ { 2 } \left( 1 + { \left. { { \bf { r } } _ { H } } \right. _ { \infty } ^ { 2 } } \right) \left( 1 + \underset { i \in B } { \sum } \left| { { { \bf { u } } _ { i } } } { \bf { r } } _ { i } ^ { 2 } \right. \right) \right) } \\ & { \qquad = O \left( A \left( { L ^ { 2 } } + \frac { { k ^ { 4 } { L ^ { 4 } } { q ^ { 2 } } } } { n ^ { 2 } } \right) + A ^ { 2 } \left( L + \frac { { k ^ { 2 } { L ^ { 2 } } } } { n } \right) \left( L + \frac { { k ^ { 2 } { L ^ { 2 } } q } } { n } \right) \right) } \\ & { \qquad = O \left( \frac { A { k ^ { 4 } { L ^ { 4 } } } } { n ^ { 2 } } q ^ { 2 } \right) + O \left( \left( \frac { A ^ { 2 } { k ^ { 2 } } { L ^ { 3 } } } { n } + \frac { A ^ { 2 } { k ^ { 4 } } { L ^ { 4 } } } { n ^ { 2 } } \right) q \right) } \\ & { \qquad + O \left( A L ^ { 2 } + A ^ { 2 } L ^ { 2 } + \frac { A ^ { 2 } { k ^ { 2 } } { L ^ { 3 } } } { n } \right) } \\ & { \qquad \leq \frac { 1 } { 3 2 } q ^ { 2 } + \frac { 1 } { 1 6 \sqrt { k } } q + \frac { 1 } { 3 2 k } \leq \frac { 1 } { 1 6 } q ^ { 2 } + \frac { 1 } { 1 6 k } . } \end{array}
$$

Above, the first line used a similar simplification as in (58), as well as our bound on $\| \mathbf { r } _ { R } \| _ { \infty }$ . The last inequality used our lower bound on n to simplify the various terms.

In conclusion, combining with (57), we have shown that

$$
\mathbf { u } ^ { \top } \left( \mathbf { X } - \mathbf { I } _ { B } \right) \mathbf { u } \leq { \frac { 1 } { 4 k } } \left( \left\| \mathbf { u } \right\| _ { 1 } ^ { 2 } - \left\| \mathbf { u } \right\| _ { 2 } ^ { 2 } \right) \ \Longrightarrow \ \rho ( \beta \mathbf { K } ) \leq { \frac { 1 } { 4 k } } .
$$

The rest of the proof follows analogously to Lemma 8, using our bounds on $\rho ( { \boldsymbol { \beta } } \mathbf { K } ) , m ( { \boldsymbol { \beta } } \mathbf { K } )$ □

We conclude the section by bounding the range of the potential, for use with Section 5.

Lemma 21. Assume the events of Proposition 3 and Lemma 18 hold, and that

$$
n = \Omega \left( k ^ { 1 . 5 } + k L ^ { 3 } \right)
$$

for a suficient constant. Then following the notation (48),

$$
\operatorname* { m a x } _ { S \in \mathcal { U } _ { \leq k } } \left| F ( S ) \right| = O \left( k \left( 1 + \sigma ^ { 2 } \right) L + k \left\| \mathbf { h } \right\| _ { \infty } \right) .
$$

Proof. For every $S \in \mathcal { U } _ { \leq k }$ , Lemma 18 gives $\| \mathbf { E } _ { S \times S } \| _ { \mathrm { o p } } \leq \frac { 1 } { 2 }$ , so the spectrum of ${ \mathbf { L } } _ { S } ^ { - 1 }$ is in $[ \frac { 2 } { 3 } , 2 ]$ Moreover, (47) and $n = \Omega ( k L )$ give $\| \mathbf { s } _ { S } \| _ { 2 } = O ( { \sqrt { k L } } )$ . Therefore,

$$
\frac { 1 } { 2 \tau } \mathbf { s } _ { S } ^ { \top } \mathbf { L } _ { S } ^ { - 1 } \mathbf { s } _ { S } \leq \frac { 1 } { \tau } \left\| \mathbf { s } _ { S } \right\| _ { 2 } ^ { 2 } = O \left( k ( 1 + \sigma ^ { 2 } ) L \right) .
$$

Further, $\mathbf { | h ^ { \intercal } 1 } _ { S } | \leq k \left\| \mathbf { h } \right\| _ { \infty }$ . The claim follows, as |log det $\mathbf { L } _ { S } | = O ( k )$ using our spectrum bound.

For completeness, we record that under the event of Lemma 18, combining the mixing guarantee in Lemma 1, the spectral gap in Proposition 4, and the Lemma 21, shows that it sufices to take

$$
T = \Omega \left( k ^ { 2 } \left( \left( 1 + \sigma ^ { 2 } \right) L + \left\| \mathbf { h } \right\| _ { \infty } \right) \log \left( \frac { 1 } { \delta ^ { \prime } } \right) \right)\tag{59}
$$

steps of $\mathsf { l a z y } ( \mathcal T ^ { \mathsf { D U } , \nu _ { s , \beta } } )$ , for a suficient constant, to sample from within $\delta ^ { \prime } ~ \mathrm { T V }$ from $\nu _ { s , \beta } ~ ( 4 9 )$

## 6.3 Main result

In this section, we finally put together the pieces to give our main sampling result for spike-and-slab posterior densities under Model 3. We use the notation $F c ( S ) : = F ( { \mathcal { C } } \cup S )$ defined in Remark 4, and for notational simplicity, we also let $\nu _ { s , \beta } ^ { \mathscr { C } } ( S ) \propto \exp ( \beta F _ { \mathscr { C } } ( \dot { S } ) ) \mathbb { I } ( S \in \mathcal { U } _ { s } )$

Algorithm 3: SASPosteriorSampler $( \mathbf { X } , \mathbf { y } , \sigma , \mathbf { q } , \delta )$   
1 Input: $\mathbf { X } \in \mathbb { R } ^ { n \times d } , \mathbf { y } \in \mathbb { R } ^ { n } , \sigma > 0 , \mathbf { q } \in ( 0 , 1 ) ^ { d } ,$ and $\delta \in ( 0 , \frac { 1 } { 2 } )$   
2 Output: Sample θ such that TV $( \operatorname { L a w } ( { \pmb \theta } ) , \pi ( \cdot \mid \mathbf { X } , \mathbf { y } ) ) \leq { \bar { \delta } }$ with probability $\geq 1 - \delta$ under   
Model 3   
3 $( \mathbf { z } , \mathcal { U } ) \gets$ output of Proposition 3, with error parameter $\frac { \delta } { 3 }$   
4 $\begin{array} { r } { \dot { k }  \dot { 2 } 4 ( \bar { k } + \log ( \frac { 3 } { \delta } ) ) } \end{array}$   
5 $c  { } u ^ { c }$   
6 $A _ { 0 } $ algorithm that always outputs $\emptyset$   
7 for $s \in [ k ]$ do   
8 $\mathbf { \mathcal { A } } _ { s } ( \beta , \delta ^ { \prime } ) \gets ( \mathsf { l a z y } ( \mathsf { T } ^ { \mathsf { D U } , \nu _ { s , \beta } ^ { c } } ) ) ^ { T }$ applied to an arbitrary start, for T as in (59)   
9 end   
10 S ∼ BMGibbsSampler $\cdot ( k , 1 , F c , \{ \mathcal { A } _ { s } \} _ { s = 0 } ^ { k } , \frac { \delta } { 3 } )$   
11 $\widetilde { S }  S \cup \mathcal { U } ^ { c }$   
12 return $\pmb { \theta } \sim \mathcal { N } ( \mathbf { A } _ { \widetilde { S } } ^ { - 1 } \mathbf { b } _ { \widetilde { S } } , \mathbf { A } _ { \widetilde { S } } ^ { - 1 } )$ following Fact 3

Theorem 5. Let $\delta \in ( 0 , \frac { 1 } { 2 } )$ , and following the notation of Model 3, let $\begin{array} { r } { k = 2 4 ( \bar { k } + \log ( \frac { 3 } { \delta } ) ) } \end{array}$ . Suppose

$$
n = \Omega \left( k ^ { 1 . 5 } \log ^ { 2 } \left( \frac { d } { \delta } \right) + k \log ^ { 3 } \left( \frac { d } { \delta } \right) \right)
$$

for a suficiently large constant, and assume that $\mathbf { q } \in [ \eta , 1 - \eta ] ^ { d } ~ f o r ~ \eta > 0$ . Then, with probability $\geq 1 - \delta$ over the randomness of Model ${ \mathcal { B } } ,$ the output of Algorithm 3 satisfies

$$
\operatorname { T V } \left( \operatorname { L a w } ( { \pmb \theta } ) , \pi ( \cdot \mid \mathbf { X } , \mathbf { y } ) \right) \leq \delta .
$$

Defining $\begin{array} { r } { Q : = k ( 1 + \sigma ^ { 2 } ) \log ( \frac { d } { \delta } ) + k \log ( \frac { 1 } { \eta } + \frac { 1 } { \eta \sigma } ) } \end{array}$ , the algorithm runs in time

$$
O \left( \frac { n d k ^ { 3 } Q ^ { 2 } } { \delta ^ { 2 } } \log \left( \frac { k } { \delta } \right) \log \left( \frac { k Q } { \delta } \right) \right) .
$$

Proof. Throughout the proof, condition on the success of Proposition 3 and the event in Lemma 18, which give the failure probability over Model 3. Under these events, the extended draw $\widetilde { S }$ in Algorithm 3 is within total variation distance δ from the support posterior $\pi ( \operatorname { s u p p } ( \cdot ) \mid \mathbf { X } , \mathbf { y } )$ . This is because the density $\hat { \pi } _ { \mathrm { s u p p } }$ in (41) is within $\mathrm { T V } \ \frac { \delta } { 3 }$ of the support posterior by Proposition 3 and Remark 4, and the guarantees of BMGibbsSampler in Lemma 15 imply S is within $\mathrm { T V } \ \frac { \delta } { 3 }$ of an exact sample from $\hat { \pi } _ { \mathrm { s u p p } }$ . Finally, the sample $\theta \mid { \widetilde { S } }$ is exact (Fact 3) and cannot increase TV.

It remains to bound the implementation cost. Observe that Lemma 21 shows that the potential $F _ { \mathcal { C } }$ is bounded by $O ( Q )$ under the assumption $\mathbf { q } \in [ \eta , 1 - \eta ] ^ { d }$ , so Lemma 15 uses

$$
N = O \left( \frac { k Q \log ( \frac { k } { \delta } ) } { \delta ^ { 2 } } \right)
$$

calls to algorithms $\mathcal { A } _ { s } ( \beta , \frac { \delta } { 6 N } )$ , each using T steps of the down-up walk where T is defined in (59).   
We claim that each step can be implemented in $O ( n d k )$ time, which gives the runtime claim.

To prove the implementation cost of $\mathcal { A } _ { s }$ for $s \in [ k ]$ , fix some $R \in \left( \begin{array} { c } { { \mathcal { U } } } \\ { { s - 1 } } \end{array} \right)$ that is the result of dropping an element (i.e., after Line 3 of Algorithm 1 has completed). For simplicity, we consider the case when $\mathcal { C } = \emptyset$ . In the general case every core R appearing below is replaced by $\mathcal { C } \cup R$ , whose size remains $O ( k )$ . To implement Line 4, Lemma 19 gives

$$
F _ { \mathcal { C } } ( R \cup \{ j \} ) - F _ { \mathcal { C } } ( R ) = { \bf h } _ { j } + \frac { 1 } { 2 \tau } \frac { { \bf r } _ { j } ^ { 2 } } { d _ { j } } - \frac { 1 } { 2 } \log d _ { j } ,
$$

where we followed the notation of (54), so

$$
\begin{array} { r } { r _ { j } = { \bf s } _ { j } - \tau { \bf E } _ { \{ j \} \times R } { \bf L } _ { R } ^ { - 1 } { \bf s } _ { R } , \quad d _ { j } : = 1 + \tau { \bf R } _ { j j } = 1 + \tau { \bf E } _ { j j } - \tau ^ { 2 } { \bf E } _ { \{ j \} \times R } { \bf L } _ { R } ^ { - 1 } { \bf E } _ { R \times \{ j \} } . } \end{array}
$$

We can compute $\mathbf { E } _ { B \times R } = [ \mathbf { X } ^ { \top } \mathbf { X } - \mathbf { I } _ { d } ] _ { B \times R }$ in time $O ( n d k )$ , and similarly we can compute and invert ${ \bf L } _ { R } = { \bf I } _ { R } + \tau { \bf E } _ { R }$ in this time. We can also compute all diagonal entries of E in time $O ( n d )$ . Thus, computing all of the $r _ { j }$ takes time $O ( n d k )$ , and computing each $d _ { j }$ takes time $O ( k ^ { 2 } ) = O ( n k )$ to compute the relevant quadratic form, so computing all of them takes time $O ( n d k )$ as well.

## Acknowledgments

We thank Thuy-Duong (June) Vuong for her participation at an earlier stage of this project, as well as Sidhanth Mohanty for several helpful conversations. We also thank an anonymous FOCS reviewer for making a suggestion that led to our strategy in Section 4. SK gratefully acknowledges funding support from the Amazon AI PhD Fellowship. KT and $\mathrm { Y Z }$ thank the NSF AI Institute for Foundations of Machine Learning (IFML) for supporting this project.

## AI Disclosure

A preliminary version of this paper, consisting of Sections 3, 5, and 6 (at a measurement complexity $n \gtrsim k ^ { 2 } )$ , was previously submitted to FOCS, with all ideas contributed by the authors. Based on a reviewer’s suggestion, the authors used GPT 5.6 Pro to explore applications of the trickle down theorem to sharpen our results. Specifically, the key perturbation strategy in Lemma 7 was suggested by GPT, which led to a weaker variant of Theorem 1 (Corollary 3). The authors then built on this approach in our final applications in Theorems 1 and 2. We also acknowledge the use of LLMs in understanding the literature on the Almeida-Thouless line, as well as the prior works [CDKP22, KPPY25], which helped us prepare Appendices A and B. The manuscript was written solely by the authors, who take full responsibility for the organization and presentation of all results.

## References

[ADCS14] Michael Aizenman, Hugo Duminil-Copin, and Vladas Sidoravicius. Random currents and continuity of ising model’s spontaneous magnetization. Communications in Mathematical Physics, 334(2):719–742, July 2014.

[ADV<sup>+</sup>25] Josh Alman, Ran Duan, Virginia Vassilevska Williams, Yinzhan Xu, Zixuan Xu, and Renfei Zhou. More asymmetry yields faster matrix multiplication. In Proceedings of the 2025 Annual ACM-SIAM Symposium on Discrete Algorithms, SODA 2025, pages 2005–2039. SIAM, 2025.

[AGS85] Daniel J Amit, Hanoch Gutfreund, and Haim Sompolinsky. Storing infinite numbers of patterns in a spin-glass model of neural networks. Physical review letters, 55(14):1530, 1985.

[AGS87] Daniel J Amit, Hanoch Gutfreund, and Haim Sompolinsky. Statistical mechanics of neural networks near saturation. Annals of physics, 173(1):30–67, 1987.

[AGZ10] Greg W Anderson, Alice Guionnet, and Ofer Zeitouni. An introduction to random matrices. Cambridge university press, 2010.

[AJK<sup>+</sup>22] Nima Anari, Vishesh Jain, Frederic Koehler, Huy Tuan Pham, and Thuy-Duong Vuong. Entropic independence: optimal mixing of down-up random walks. In STOC ’22: 54th Annual ACM SIGACT Symposium on Theory of Computing, pages 1418– 1430. ACM, 2022.

[AKV24] Nima Anari, Frederic Koehler, and Thuy-Duong Vuong. Trickle-down in localization schemes and applications. In Proceedings of the 56th Annual ACM Symposium on Theory of Computing, STOC 2024, pages 1094–1105. ACM, 2024.

[AL20] Vedat Levi Alev and Lap Chi Lau. Improved analysis of higher order random walks and applications. In Proceedings of the 52nd annual ACM SIGACT symposium on theory of computing, pages 1198–1211, 2020.

[ALGV19] Nima Anari, Kuikui Liu, Shayan Oveis Gharan, and Cynthia Vinzant. Log-concave polynomials II: high-dimensional walks and an FPRAS for counting bases of a matroid.

In Moses Charikar and Edith Cohen, editors, Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing, STOC 2019, pages 1–12. ACM, 2019.

[AMS22] Ahmed El Alaoui, Andrea Montanari, and Mark Sellke. Sampling from the sherrington-kirkpatrick gibbs measure via algorithmic stochastic localization. In 63rd IEEE Annual Symposium on Foundations of Computer Science, FOCS 2022, pages 323–334. IEEE, 2022.

[BAR26] Afonso S. Bandeira, Ahmed El Alaoui, and Almut Rödder. Mixing of glauber dynamics on high overlap gibbs measures, 2026.

[BBD24] Roland Bauerschmidt, Thierry Bodineau, and Benoit Dagallier. Kawasaki dynamics beyond the uniqueness threshold. Probability Theory and Related Fields, 192(1–2):267– 302, 2024.

[BG08] Adriano Barra and Francesco Guerra. About the ergodic regime in the analogical hopfield neural networks: moments of the partition function. Journal of mathematical physics, 49(12), 2008.

[BH24] Joan Bruna and Jiequn Han. Provable posterior sampling with denoising oracles via tilted transport. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, 2024.

[Bon14] Claudio Bonati. The peierls argument for higher dimensional ising models. European Journal of Physics, 35(3):035002, March 2014.

[BRG21] Ray Bai, Veronika Rockova, and Edward I. George. Spike-and-slab meets lasso: A review of the spike-and-slab lasso. Handbook of Bayesian Variable Selection, pages 81–108, 2021.

[BvEN99] Anton Bovier, Aernout CD van Enter, and Beat Niederhauser. Stochastic symmetrybreaking in a gaussian hopfield model. Journal of statistical physics, 95(1):181–213, 1999.

[BvH16] Afonso S. Bandeira and Ramon van Handel. Sharp nonasymptotic bounds on the norm of random matrices with independent entries. The Annals of Probability, 44(4):2479– 2506, 2016.

[BY22] Christian Brennecke and Horng-Tzer Yau. The replica symmetric formula for the sk model revisited. Journal of Mathematical Physics, 63(7), 2022.

[CDKP22] Charlie Carlson, Ewan Davies, Alexandra Kolla, and Will Perkins. Computational thresholds for the fixed-magnetization ising model. In Proceedings of the 54th Annual ACM SIGACT Symposium on Theory of Computing, STOC 2022, page 1459–1472. Association for Computing Machinery, 2022.

[CGM19] Mary Cryan, Heng Guo, and Giorgos Mousa. Modified log-sobolev inequalities for strongly log-concave distributions. In 60th IEEE Annual Symposium on Foundations of Computer Science, FOCS 2019, pages 1358–1370. IEEE Computer Society, 2019.

[Chi96] H. Chipman. Bayesian variable selection with related predictors. The Canadian Journal of Statistics, 24:17–36, 1996.

[CLTZ26] Ziyun Chen, Jerry Li, Kevin Tian, and Yusong Zhu. Separating oblivious and adaptive models of variable selection. arXiv preprint arXiv:2602.16568, 2026.

[CPS09] Carlos M. Carvalho, Nicholas G. Polson, and James G. Scott. Handling sparsity via the horseshoe. In Proceedings of the Twelfth International Conference on Artificial Intelligence and Statistics, AISTATS 2009, volume 5 of JMLR Proceedings, pages 73–80. JMLR.org, 2009.

[CRT06] Emmanuel J Candès, Justin Romberg, and Terence Tao. Robust uncertainty principles: Exact signal reconstruction from highly incomplete frequency information. IEEE Transactions on information theory, 52(2):489–509, 2006.

[CSHVdV15] Ismael Castillo, Johannes Schmidt-Hieber, and Aad Van der Vaart. Bayesian linear regression with sparse priors. The Annals of Statistics, 43(5):1986–2018, 2015.

[CT05] Emmanuel J Candes and Terence Tao. Decoding by linear programming. IEEE transactions on information theory, 51(12):4203–4215, 2005.

[CT06] Emmanuel J Candes and Terence Tao. Near-optimal signal recovery from random projections: Universal encoding strategies? IEEE transactions on information theory, 52(12):5406–5425, 2006.

[CvdV12] Ismaël Castillo and Aad van der Vaart. Needles and straw in a haystack: Posterior concentration for possibly sparse sequences. The Annals of Statistics, 40(4):2069–2101, August 2012.

[dAT78] Jairo RL de Almeida and David J Thouless. Stability of the sherrington-kirkpatrick solution of a spin glass model. Journal of Physics A: Mathematical and General, 11(5):983–990, 1978.

[DLSS26] Ewan Davies, Holden Lee, Juspreet Singh Sandhu, and Jonathan Shi. Potential hessian ascent III: sampling the sherrington-kirkpatrick model at beta < 1/2. CoRR, abs/2605.03718, 2026.

[Dob68] PL Dobruschin. The description of a random field by means of conditional probabilities and conditions of its regularity. Theory of Probability & Its Applications, 13(2):197– 224, 1968.

[Don06] David L Donoho. Compressed sensing. IEEE Transactions on information theory, 52(4):1289–1306, 2006.

[DP23] Ewan Davies and Will Perkins. Approximately counting independent sets of a given size in bounded-degree graphs. SIAM Journal on Computing, 52(2):618–640, 2023.

[EKZ22] Ronen Eldan, Frederic Koehler, and Ofer Zeitouni. A spectral condition for spectral gap: fast mixing in high-temperature ising models. Probability theory and related fields, 182(3):1035–1051, 2022.

[Ell12] Richard S Ellis. Entropy, large deviations, and statistical mechanics. Springer Science & Business Media, 2012.

[Gew96] J. Geweke. Variable selection and model comparison in regression. Bayesian Statistics, 5:609–620, 1996.

[GG84] Andrei Yur’evich Garnaev and Efim Davydovich Gluskin. The widths of a euclidean ball. In Doklady Akademii Nauk, volume 277, pages 1048–1052. Russian Academy of Sciences, 1984.

[GK23] Roy Gotlib and Tali Kaufman. Nowhere to go but high: a perspective on highdimensional expanders. In International Congress of Mathematicians, pages 4842– 4871. European Mathematical Society-EMS-Publishing House GmbH, 2023.

[GM93] Edward I George and Robert E McCulloch. Variable selection via gibbs sampling. Journal of the American Statistical Association, 88(423):881–889, 1993.

[Hop82] John J Hopfield. Neural networks and physical systems with emergent collective computational abilities. Proceedings of the national academy of sciences, 79(8):2554– 2558, 1982.

[HRP<sup>+</sup>20] Firas Hamze, Jack Raymond, Christopher A Pattison, Katja Biswas, and Helmut G Katzgraber. Wishart planted ensemble: A tunably rugged pairwise ising model with a first-order phase transition. Physical Review E, 101(5):052102, 2020.

[IR11] Hemant Ishwaran and J. Sunil Rao. Consistency of spike and slab regression. Statistics & Probability Letters, 81(12):1920–1928, 2011.

[JM24] Vishesh Jain and Clayton Mizgerd. Rapid Mixing of the Down-Up Walk on Matchings of a Fixed Size. In Approximation, Randomization, and Combinatorial Optimization. Algorithms and Techniques (APPROX/RANDOM 2024), volume 317 of Leibniz International Proceedings in Informatics (LIPIcs), pages 63:1–63:13, 2024.

[JMPV23] Vishesh Jain, Marcus Michelen, Huy Tuan Pham, and Thuy-Duong Vuong. Optimal mixing of the down-up walk on independent sets of a given size. In 2023 IEEE 64th Annual Symposium on Foundations of Computer Science (FOCS), pages 1665–1681. IEEE, 2023.

[JS04] Iain M. Johnstone and Bernard W. Silverman. Needles and straw in haystacks: Empirical bayes estimates of possibly sparse sequences. Annals of Statistics, 32(4):1594– 1649, 2004.

[JT17] Aukosh Jagannath and Ian Tobasco. Some properties of the phase diagram for mixed p-spin glasses. Probability Theory and Related Fields, 167(3):615–672, 2017.

[Kas77] Boris Sergeevich Kashin. Diameters of some finite-dimensional sets and classes of smooth functions. Izvestiya Rossiiskoi Akademii Nauk. Seriya Matematicheskaya, 41(2):334–351, 1977.

[Kaw66] Kyozi Kawasaki. Difusion constants near the critical point for time-dependent ising models. ii. Physical Review, 148(1):375, 1966.

[KLL<sup>+</sup>23] Jonathan A. Kelner, Jerry Li, Allen Liu, Aaron Sidford, and Kevin Tian. Semirandom sparse recovery in nearly-linear time. In The Thirty Sixth Annual Conference

on Learning Theory, COLT 2023, volume 195 of Proceedings of Machine Learning Research, pages 2352–2398. PMLR, 2023.

[KN26] Seiichiro Kusuoka and Shuta Nakajima. A quantitative replica-symmetric bound of sherrington–kirkpatrick model in the entire de almeida–thouless region. arXiv preprint arXiv:2608.23413, 2026.

[KO20] Tali Kaufman and Izhar Oppenheim. High order random walks: Beyond spectral gap. Combinatorica, 40(2):245–281, 2020.

[Kol18] Vladimir Kolmogorov. A faster approximation algorithm for the gibbs partition function. In Conference On Learning Theory, COLT 2018, volume 75 of Proceedings of Machine Learning Research, pages 228–249. PMLR, 2018.

[KPPY25] Aiya Kuchukova, Marcus Pappik, Will Perkins, and Corrine Yap. Fast and slow mixing of the kawasaki dynamics on bounded-degree graphs. Random Structures & Algorithms, 67(4), 2025.

[KSTZ25] Symantak Kumar, Purnamrita Sarkar, Kevin Tian, and Yusong Zhu. Spike-and-slab posterior sampling in high dimensions. In The Thirty Eighth Annual Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pages 3407–3462. PMLR, 2025.

[Lit74] William A Little. The existence of persistent states in the brain. Mathematical biosciences, 19(1-2):101–120, 1974.

[LMRW24] Kuikui Liu, Sidhanth Mohanty, Amit Rajaraman, and David X. Wu. Fast mixing in sparse random ising models. In 65th IEEE Annual Symposium on Foundations of Computer Science, FOCS 2024, pages 120–128. IEEE, 2024.

[Lop26] Patrick Lopatto. Replica symmetry up to the de almeida-thouless line in the sherrington-kirkpatrick model. arXiv preprint arXiv:2604.11921, 2026.

[LPW09] David Asher Levin, Yuval Peres, and Elizabeth Wilmer. Markov Chains and Mixing Times. American Mathematical Society, 2009.

[Lyo89] Russell Lyons. The ising model and percolation on trees and tree-like graphs. Communications in Mathematical Physics, 125(2):337–353, 1989.

[MB88] Toby J Mitchell and John J Beauchamp. Bayesian variable selection in linear regression. Journal of the American Statistical Association, 83(404):1023–1032, 1988.

[Mos06] Elchanan Mossel. Ising model on trees. Lecture notes for STAT 206A, University of California, Berkeley, 2006. Lecture 20.

[MS22] Sumit Mukherjee and Subhabrata Sen. Variational inference in high-dimensional linear regression. Journal of Machine Learning Research, 23:1–56, 2022.

[MW26] Andrea Montanari and Yuchen Wu. Provably eficient posterior sampling for sparse linear regression via measure decomposition. Journal of the American Statistical Association, pages 1–19, 2026.

[Opp18] Izhar Oppenheim. Local spectral expansion approach to high dimensional expanders part i: Descent of spectral gaps. Discrete & Computational Geometry, 59(2):293–330, 2018.

[PC99] Victor Y. Pan and Zhao Q. Chen. The complexity of the matrix eigenproblem. In Proceedings of the Thirty-First Annual ACM Symposium on Theory of Computing, pages 507–516. ACM, 1999.

[PF77] Leonid A Pastur and Alexander L Figotin. Exactly soluble model of a spin glass. Soviet Journal of Low Temperature Physics, 3(6):378–383, 1977.

[PS19] Nicholas G. Polson and Lei Sun. Bayesian ℓ<sub>0</sub>-regularized least squares. Applied Stochastic Models in Business and Industry, 35(3):717–731, 2019.

[RG14] Veronika Ročková and Edward I. George. Emvs: The em approach to bayesian variable selection. Journal of the American Statistical Association, 109(506):828–846, 2014.

[RG18] Veronika Ročková and Edward I. George. The spike-and-slab lasso. Journal of the American Statistical Association, 113(521):431–444, 2018.

[Roc18] Veronika Rockova. Bayesian estimation of sparse signals with a continuous spike-andslab prior. Annals of Statistics, 46(1):401–437, 2018.

[RS22] Kolyan Ray and Botond Szabó. Variational bayes for high-dimensional linear regression with sparse priors. Journal of the American Statistical Association, 117(539):1270–1281, 2022.

[RW26] Amit Rajaraman and David X Wu. Markov chains approximate message passing. In Proceedings of the 58th Annual ACM Symposium on Theory of Computing, pages 1192–1199, 2026.

[SK75] David Sherrington and Scott Kirkpatrick. Solvable model of a spin-glass. Physical review letters, 35(26):1792, 1975.

[Str69] Volker Strassen. Gaussian elimination is not optimal. Numerische Mathematik, 13:354–356, 1969.

[Tal10] Michel Talagrand. Mean field models for spin glasses: Volume I: Basic examples, volume 54. Springer Science & Business Media, 2010.

[Tal11] Michel Talagrand. Mean Field Models for Spin Glasses: Volume II: Advanced Replica-Symmetry and Low Temperature, volume 55. Springer Science & Business Media, 01 2011.

[Tro12] Joel A Tropp. User-friendly tail bounds for sums of random matrices. Foundations of computational mathematics, 12(4):389–434, 2012.

[Ver18] Roman Vershynin. High-dimensional probability: An introduction with applications in data science, volume 47. Cambridge university press, 2018.

[Wai19] Martin J Wainwright. High-dimensional statistics: A non-asymptotic viewpoint, volume 48. Cambridge university press, 2019.

[Wei07] Pierre Weiss. L’hypothèse du champ moléculaire et la propriété ferromagnétique. Journal de Physique Théorique et Appliquée, 6(1):661–690, 1907.

[WJ08] Martin J Wainwright and Michael I Jordan. Graphical models, exponential families, and variational inference. Foundations and Trends® in Machine Learning, 1(1-2):1– 305, 2008.

[Wu06] Liming Wu. Poincaré and transportation inequalities for gibbs measures under the dobrushin uniqueness condition. The Annals of Probability, 34(5), 2006.

[Yan52] Chen Ning Yang. The spontaneous magnetization of a two-dimensional ising model. Physical Review, 85(5):808, 1952.

[YWJ16] Yun Yang, Martin J Wainwright, and Michael I Jordan. On the computational complexity of high-dimensional bayesian variable selection. The Annals of Statistics, 2016.

## A Sampling Near the Almeida–Thouless Line

In this section, we give an application to sampling near the Almeida-Thouless $( A T )$ line. The AT line is parameterized by $\beta > 0$ and a field strength $h > 0$ , and considers the specialized SK model

$$
\pi _ { \beta , h } ( \mathbf { x } ) \propto \exp \left( \frac { \beta } { 2 } \mathbf { x } ^ { \top } \mathbf { J } \mathbf { x } + h \mathbf { 1 } _ { d } ^ { \top } \mathbf { x } \right) , \quad \mathbf { x } \in \mathcal { X } ^ { d } ,\tag{60}
$$

where $\mathbf { J } \sim \mathrm { G O E } ( d )$ follows Model 1, and we set the diagonal of J to zero without loss of generality. Thus, h denotes the coeficient of the linear term $\propto \mathbf { 1 } _ { d }$

The AT line delineates a region in $\mathbb { R } _ { \geq 0 } ^ { 2 } ,$ , representing a pair of parameters $( \beta , h )$ in (60). This line was originally derived in [dAT78] via the replica method, with the prediction that models induced by h above the line (i.e., large enough as a function of $\beta )$ are replica symmetric, and that pairs below the line exhibit replica symmetry breaking. This prediction was recently established rigorously for the SK model with a homogeneous external field [Lop26]. However, our application requires a stronger quantitative concentration estimate, so our results hold above the slightly more restrictive weak AT line (Definition 3), leveraging bounds by [Tal11, JT17] as presented by [RW26].

For large h, (60) favors x with more positive spins. Following this convention, we write

$$
N _ { - } ( { \bf x } ) : = \left| \left\{ i \in [ d ] : x _ { i } = - 1 \right\} \right| , \qquad \mathcal { X } _ { \leq k } ^ { d , - } : = \left\{ { \bf x } \in \mathcal { X } ^ { d } : N _ { - } ( { \bf x } ) \leq k \right\} .
$$

Our earlier results use the number of positive spins as the sparsity parameter; the two conventions are equivalent under the global spin flip ${ \bf x }  - { \bf x } .$ . Moreover, on every fixed-magnetization slice, $h \mathbf { 1 } _ { d } ^ { \top } \mathbf { x }$ is constant, so all fixed-size mixing analyses for (60) are independent of h.

We now define the regions determined by the AT line.

Definition 2 (AT condition, [dAT78]). For $\beta , h > 0$ , let $q = q ( \beta , h )$ be the unique solution to

$$
q = \mathbb { E } \left[ \operatorname { t a n h } ^ { 2 } \left( \beta \sqrt { q } Z + h \right) \right] , \qquad Z \sim \mathcal { N } ( 0 , 1 ) .
$$

Define

$$
\alpha _ { \mathrm { A T } } ( \beta , h ) : = \beta ^ { 2 } \mathbb { E } \left[ \mathrm { s e c h } ^ { 4 } \left( \beta \sqrt { q } Z + h \right) \right] .
$$

We say that $( \beta , h )$ satisfies the $A T$ condition if $\alpha _ { \mathrm { A T } } ( \beta , h ) \leq 1$ . The AT boundary $h _ { \mathrm { A T } } ( \beta )$ is characterized by $\alpha _ { \mathrm { A T } } ( \beta , h _ { \mathrm { A T } } ( \beta ) ) = 1$ . We also define $q _ { \mathrm { A T } } ( \beta ) : = q ( \beta , h _ { \mathrm { A T } } ( \beta ) )$

Definition 3 (Weak AT condition, [BY22]). With $q = q ( \beta , h )$ as in Definition 2, define

$$
\alpha _ { \mathrm { w A T } } ( \beta , h ) : = \beta ^ { 2 } \mathbb { E } \left[ { \mathrm { s e c h } } ^ { 2 } \left( \beta { \sqrt { q } } Z + h \right) \right] = \beta ^ { 2 } ( 1 - q ) .
$$

We say that $( \beta , h )$ satisfies the weak AT condition if $\alpha _ { \mathrm { w A T } } ( \beta , h ) \leq 1$ . The weak AT boundary $h _ { \mathrm { W A T } } ( \beta )$ is characterized by $\alpha _ { \mathrm { w A T } } ( \beta , h _ { \mathrm { w A T } } ( \beta ) ) = 1$ . We also define $q _ { \mathrm { w A T } } ( \beta ) : = q ( \beta , h _ { \mathrm { w A T } } ( \beta ) )$

Since sech $\mathfrak { l } ^ { 4 } ( u ) \leq \mathrm { s e c h } ^ { 2 } ( u )$ for all u, we have that $\alpha _ { \mathrm { A T } } ( \beta , h ) \leq \alpha _ { \mathrm { w A T } } ( \beta , h )$ for all (β, h). Thus, the weak AT region (above the weak AT line) is a subset of the AT region.

## A.1 Preliminaries

In this section, we prove two key preliminary results. The first (Lemma 23) computes asymptotics of the boundaries $h _ { \mathrm { A T } } , \ h _ { \mathrm { w A T } }$ as a function of $\beta .$ The second (Lemma 24) formalizes a reduction from large field strength h to concentration on high-magnetization states.

Boundary asymptotics. We first recall a common Laplace estimate that will be used repeatedly. Boundary asymptotics. We first recall a common Laplace estimate that will be used repeatedly.

Lemma 22. Let $\begin{array} { r } { \phi ( t ) : = \frac { 1 } { \sqrt { 2 \pi } } \exp ( - \frac { t ^ { 2 } } { 2 } ) } \end{array}$ be the Gaussian density, and let $p \in \{ 2 , 4 \}$ . Suppose that there are positive $\{ q _ { \beta } , h _ { \beta } \} _ { \beta > 0 }$ satisfying, as $\begin{array} { r } { \beta \to \infty , q _ { \beta } \to 1 , b _ { \beta } : = \frac { h _ { \beta } } { \beta } \to \infty } \end{array}$ , and $\begin{array} { r } { \frac { b _ { \beta } } { \beta }  0 } \end{array}$ . Then

$$
\mathbb { E } \left[ \mathrm { s e c h } ^ { p } \left( \beta \sqrt { q \beta } Z + \beta b _ { \beta } \right) \right] = \frac { I _ { p } } { \beta \sqrt { q \beta } } \phi \left( \frac { b _ { \beta } } { \sqrt { q \beta } } \right) \left( 1 + o ( 1 ) \right) ,\tag{61}
$$

where

$$
I _ { 2 } = \int _ { \mathbb R } \mathrm { s e c h } ^ { 2 } ( u ) \mathrm { d } u = 2 , \qquad I _ { 4 } = \int _ { \mathbb R } \mathrm { s e c h } ^ { 4 } ( u ) \mathrm { d } u = \frac 4 3 .
$$

Proof. We first perform a change of variables $u = \beta \big ( \sqrt { q _ { \beta } } z + b _ { \beta } \big )$ , which gives

$$
\mathbb { E } \left[ \mathrm { s e c h } ^ { p } \left( \beta \sqrt { q \beta } Z + \beta b _ { \beta } \right) \right] = \frac { \phi \left( b _ { \beta } / \sqrt { q \beta } \right) } { \beta \sqrt { q \beta } } \int _ { \mathbb { R } } \mathrm { s e c h } ^ { p } ( u ) \mathrm { e x p } \left( \frac { b _ { \beta } u } { \beta q _ { \beta } } - \frac { u ^ { 2 } } { 2 \beta ^ { 2 } q _ { \beta } } \right) \mathrm { d } u .\tag{62}
$$

Since $q _ { \beta } \to 1$ and $\frac { b _ { \beta } } { \beta }  0$ , the factor ex $\rho \big ( \frac { b _ { \beta } u } { \beta q _ { \beta } } - \frac { u ^ { 2 } } { 2 \beta ^ { 2 } q _ { \beta } } \big )$ converges pointwise to one. Moreover, for suficiently large $\beta , \frac { b _ { \beta } } { \beta q _ { \beta } }  0$ , so we have

$$
\exp \left( \frac { b _ { \beta } u } { \beta q _ { \beta } } - \frac { u ^ { 2 } } { 2 \beta ^ { 2 } q _ { \beta } } \right) \leq \exp \left( \frac { | u | } { 2 } \right) .
$$

For $p \in \{ 2 , 4 \} , \mathrm { s e c h } ^ { p } ( u ) \mathrm { e x p } ( \frac { | u | } { 2 } )$ is integrable. Applying dominated convergence then gives (61).

We now derive the (identical) asymptotics of $h _ { \mathrm { A T } }$ and $h _ { \mathrm { w A T } }$

Lemma 23. As $\beta \to \infty$

$$
h _ { \mathrm { A T } } ( \beta ) = \sqrt { 2 } \beta \sqrt { \log \beta } \left( 1 + O \left( \frac { 1 } { \log \beta } \right) \right) , \quad h _ { \mathrm { w A T } } ( \beta ) = \sqrt { 2 } \beta \sqrt { \log \beta } \left( 1 + O \left( \frac { 1 } { \log \beta } \right) \right) .
$$

Proof. We begin by verifying the conditions of Lemma 22. For $\star \in \{ \mathrm { A T } , \mathrm { w A T } \}$ , write

$$
q _ { \star } : = q _ { \star } ( \beta ) , \quad b _ { \star } : = \frac { h _ { \star } ( \beta ) } { \beta } .
$$

We also drop the index $\beta$ from these two sequences for simplicity. First, along the weak AT boundary, $1 - q _ { \mathrm { w A T } } = \beta ^ { - 2 }$ , so $q _ { \mathrm { w A T } }  1$ . Similarly, along the AT boundary,

$$
\mathbb { E } \left[ \mathrm { s e c h } ^ { 4 } \left( \beta \sqrt { q _ { \mathrm { A T } } } Z + h _ { \mathrm { A T } } ( \beta ) \right) \right] = \frac { 1 } { \beta ^ { 2 } } .
$$

Since $1 - q _ { \mathrm { A T } } = \mathbb { E } [ \mathrm { s e c h } ^ { 2 } ( \beta \sqrt { q _ { \mathrm { A T } } } Z + h _ { \mathrm { A T } } ( \beta ) ) ]$ , Cauchy-Schwarz gives $1 - q _ { \mathrm { A T } } \leq \beta ^ { - 1 }$ , so $q _ { \mathrm { A T } }  1$

Next, for either value of $\star ,$ the boundary equation also implies $b _ { \star }  \infty$ . Indeed, if $b _ { \star }$ were bounded along a subsequence, then restricting the integral (62) to $u \in [ - 1 , 1 ]$ would already give a lower bound of order $\beta ^ { - 1 }$ , because $q _ { \star }  1 , b _ { \star }$ is bounded, and sec $\mathrm { \Omega } ^ { p } ( u ) = \Omega ( 1 )$ for $u \in [ - 1 , 1 ]$ . This would contradict the boundary equations, which say that (62) evaluates to $\beta ^ { - 2 }$

Finally, we claim that $\frac { b _ { \star } } { \beta }  0$ . Otherwise, $b _ { \star } ~ \geq ~ \epsilon \beta$ along a subsequence for some $\epsilon > 0$ . Now, split the integral (62) along the events $Z \ge - b _ { \star } / 2 \sqrt { q _ { \star } }$ or $Z \le - b _ { \star } / 2 \sqrt { q _ { \star } }$ . In the former region, the change of variables gives $u \ge \frac { \epsilon \beta ^ { 2 } } { 2 }$ , so the corresponding integral is $\exp ( - \Omega ( \beta ^ { 2 } ) )$ . Similarly, the latter region has a probability bounded by $\exp ( - \Omega ( \beta ^ { 2 } ) )$ ), and sech is pointwise bounded. Thus, the entire integral is $\exp ( - \Omega ( \beta ^ { 2 } ) )$ , again contradicting that it equals $\beta ^ { - 2 }$ by definition.

Lemma 22 therefore applies and we obtain that

$$
\phi \left( \frac { b _ { \mathrm { A T } } } { \sqrt { q _ { \mathrm { A T } } } } \right) = \frac { 3 \sqrt { q _ { \mathrm { A T } } } } { 4 \beta } \left( 1 + o ( 1 ) \right) , \quad \phi \left( \frac { b _ { \mathrm { w A T } } } { \sqrt { q _ { \mathrm { w A T } } } } \right) = \frac { \sqrt { q _ { \mathrm { w A T } } } } { 2 \beta } \left( 1 + o ( 1 ) \right) .
$$

Expanding with the definition of $\phi ,$ and then taking logarithms, then gives for $\star \in \{ \mathrm { A T } , \mathrm { w A T } \}$

$$
{ \frac { b _ { \star } ^ { 2 } } { 2 q _ { \star } } } = \log \beta + { \cal O } ( 1 ) \implies b _ { \star } ^ { 2 } = 2 q _ { \star } \log \beta + { \cal O } ( 1 ) = 2 \log \beta - 2 ( 1 - q _ { \star } ) \log \beta + { \cal O } ( 1 ) .
$$

Since $1 - q _ { \mathrm { A T } } \leq \beta ^ { - 1 }$ and $1 - q _ { \mathrm { w A T } } = \beta ^ { - 2 }$ , the desired claims follow:

$$
b _ { \star } ^ { 2 } = 2 \log \beta + O ( 1 ) \implies b _ { \star } = \sqrt { 2 \log \beta } \left( 1 + O \left( \frac { 1 } { \log \beta } \right) \right) .
$$

Magnetization from field strength. To conclude the section, we show that taking h large in (60) implies that $\pi _ { \beta , h }$ is concentrated on high-magnetization states. For $s \in [ 0 , 1 ]$ , we let $H _ { 2 } ( s ) : =$ −s log $s - ( 1 - s ) \log ( 1 - s )$ denote the binary entropy, with the convention 0 log $0 = 0$

Lemma 24. Fix $\delta \in ( 0 , \frac { 1 } { 2 } )$ . With probability at least $1 - \delta$ over J in Model 1, the following holds simultaneously for every $\bar { \rho } \in ( 0 , \frac { 1 } { 2 } )$ and $\gamma \geq 0$ . If

$$
h \geq \frac { H _ { 2 } ( \rho ) } { 2 \rho } + \beta \sqrt { \frac { 2 ( 1 - \rho ) } { \rho } \left( H _ { 2 } ( \rho ) + \frac { 1 } { d } \log \frac { d } { \delta } \right) } + \gamma ,\tag{63}
$$

then

$$
\begin{array} { r } { \mathbb { P } _ { \mathbf { x } \sim \pi _ { \beta , h } } \left[ N _ { - } ( \mathbf { x } ) > \rho d \right] \leq d \exp \left( - 2 \gamma \rho d \right) . } \end{array}
$$

Proof. For a fixed $S \subseteq [ d ]$ , define $\begin{array} { r } { Y _ { S } : = \sum _ { \stackrel { i \in S } { j \not \in S } } \mathbf { J } _ { i j } } \end{array}$ . We claim that simultaneously for all $S \subseteq [ d ]$

$$
Y _ { S } \geq - t \left( { \frac { | S | } { d } } \right) , { \mathrm { ~ w h e r e ~ } } t ( s ) : = d { \sqrt { 2 s ( 1 - s ) \left( H _ { 2 } ( s ) + { \frac { 1 } { d } } \log { \frac { d } { \delta } } \right) } } ,\tag{64}
$$

with probability $\geq 1 - \delta$ . To see this, fix some $r \in [ d ]$ . Under Model 1, we have $Y _ { S } \sim { \mathcal { N } } ( 0 , d s ( 1 - s ) )$ for all $| S | = r$ and $\textstyle s : = { \frac { r } { d } }$ . Then the standard Gaussian tail bound gives

$$
\mathbb { P } [ - Y _ { S } \ge t ( s ) ] \le \exp \left( - d H _ { 2 } ( s ) \right) \frac { \delta } { d } .
$$

Since $\binom { d } { r } \leq \exp ( d H _ { 2 } ( s ) )$ , we conclude that (64) holds for all $| S | = r$ with probability $\geq 1 - \frac { \delta } { d }$ , and then a union bound over all nontrivial layers $r \in [ d ]$ gives the claim.

Next, letting w be the unnormalized weight in (60), a direct calculation gives log $\frac { w ( S ) } { w ( \emptyset ) } = - 2 h | S | -$ $2 \beta Y _ { S }$ . Thus, again letting $\begin{array} { r } { s = \frac { r } { d } } \end{array}$ for some $r \in [ d ]$ ，

$$
\frac { \sum _ { S : | S | = r } w ( S ) } { w ( \emptyset ) } \leq \exp \left( 2 s d \left( \frac { H _ { 2 } ( s ) } { 2 s } + \frac { \beta t ( s ) } { s d } - h \right) \right) .\tag{65}
$$

It remains to bound the right-hand side uniformly over $s \geq \rho .$ . First, we can directly check that $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } s } ( \frac { H _ { 2 } ( s ) } { s } ) = \frac { \log ( 1 - s ) } { s ^ { 2 } } < 0 . } \end{array}$ , so $\boxed { H _ { 2 } ( s ) }$ is decreasing in s. This implies

$$
\frac { t ( s ) } { s d } = \sqrt { \frac { 2 ( 1 - s ) } { s } \left( H _ { 2 } ( s ) + \frac { 1 } { d } \log \frac { d } { \delta } \right) } ~ 
$$

is decreasing in $s ,$ because every term in the square root is decreasing. Hence for every $s \geq \rho _ { ; }$ , the assumed lower bound (63) implies

$$
\frac { H _ { 2 } ( s ) } { 2 s } + \frac { \beta t ( s ) } { s d } - h \leq \frac { H _ { 2 } ( \rho ) } { 2 \rho } + \frac { \beta t ( \rho ) } { \rho d } - h \leq - \gamma .
$$

Summing (65) over all $r > \rho d$ and using that the partition function is $\geq w ( \emptyset )$ proves the claim.

## A.2 Sampling around ${ \bf 1 } _ { d }$

In this section, we give the basic variant of our result for sampling from (60). This variant shows that as $\beta \to \infty$ , when the field strength h exceeds the threshold $h _ { \mathrm { A T } } ( \beta )$ in Lemma 23 by roughly a $\sqrt { 2 }$ factor, we can sample from $\pi _ { \beta , h }$ in polynomial time.

Theorem 6. Let $\beta \geq 1$ and $\delta \in ( 0 , \frac { 1 } { 2 } )$ , and suppose that (31) holds with $\delta  { \frac { \delta } { 2 } } . \ I f$

$$
h \geq \frac { H _ { 2 } ( \rho _ { \beta } ) } { 2 \rho _ { \beta } } + \beta \sqrt { 2 \frac { 1 - \rho _ { \beta } } { \rho _ { \beta } } \left( H _ { 2 } ( \rho _ { \beta } ) + \frac { 1 } { d } \log \frac { 2 d } { \delta } \right) + \frac { 1 } { 2 \rho _ { \beta } d } } \log \frac { 2 d } { \delta } ,\tag{66}
$$

where $\begin{array} { r } { \rho _ { \beta } : = \frac { c } { \beta ^ { 2 } \log ( e \beta ) } } \end{array}$ for a universal constant $c > 0$ , then with probability at least $1 - \delta$ over the SK model (Model 1), there is a polynomial-time algorithm that outputs x satisfying TV (Law $( { \bf x } ) , \pi _ { \beta , h } ) \le$ δ. Furthermore, for $\begin{array} { r } { \delta = \mathrm { p o l y } ( \frac { 1 } { d } ) } \end{array}$ and fixed $\beta ,$ as $d \to \infty$ , the right-hand side of (66) converges to

$$
h _ { \mathrm { H M } } ( \beta ) : = \frac { H _ { 2 } ( \rho _ { \beta } ) } { 2 \rho _ { \beta } } + \beta \sqrt { 2 ( 1 - \rho _ { \beta } ) \frac { H _ { 2 } ( \rho _ { \beta } ) } { \rho _ { \beta } } } = 2 \beta \sqrt { \log \beta } ( 1 + o ( 1 ) ) .\tag{67}
$$

Proof. Throughout this proof, set

$$
k : = \lfloor \rho _ { \beta } d \rfloor , \quad \gamma : = \frac { 1 } { 2 \rho _ { \beta } d } \log \frac { 2 d } { \delta } .
$$

Applying Lemma 24 with failure probability $\frac { \delta } { 2 } { : }$ , sparsity lower bound parameter $\rho _ { \beta }$ , and $\gamma$ as defined above then implies that with probability $\geq 1 - \frac { \delta } { 2 }$ over J,

$$
\mathbb { P } _ { \mathbf { x } \sim \pi _ { \beta , h } } \left( N _ { - } ( \mathbf { x } ) > k \right) \le d \exp ( - 2 \gamma \rho _ { \beta } d ) \le \frac { \delta } { 2 } .
$$

Next, to sample from $\pi _ { \beta , h }$ conditioned on $N _ { - } ( \mathbf { x } ) \leq k ,$ assume $k \geq 1$ , else it sufices to output $\mathbf { 1 } _ { d } .$ By choosing c suficiently small, $k \leq \rho _ { \beta } d$ is in the range required by Corollary 5. Then applying Corollary 5 with accuracy and failure probability $\frac { \delta } { 2 }$ gives the desired sample $\mathbf { x } ,$ upon flipping 1s and −1s consistently with our convention in this section. A union bound then gives both the tota failure probability of δ over the draw $\mathbf { J } ,$ and the overall accuracy $\delta$ to the target $\pi _ { \beta , h }$

Finally, we prove the asymptotic claims. For fixed $\beta$ and $\begin{array} { r } { \delta = \operatorname { p o l y } ( \frac { 1 } { d } ) } \end{array}$ , the finite-d corrections in (66) vanish as $d \to \infty$ , so the limit is

$$
h _ { \mathrm { H M } } ( \beta ) = \frac { H _ { 2 } ( \rho _ { \beta } ) } { 2 \rho _ { \beta } } + \beta \sqrt { 2 ( 1 - \rho _ { \beta } ) \frac { H _ { 2 } ( \rho _ { \beta } ) } { \rho _ { \beta } } } .
$$

The conclusion follows because as $\beta  \infty .$ , we have

$$
\frac { H _ { 2 } ( \rho _ { \beta } ) } { \rho _ { \beta } } = \log \frac { 1 } { \rho _ { \beta } } + 1 + O ( \rho _ { \beta } ) , \quad \log \frac { 1 } { \rho _ { \beta } } = 2 \log \beta + o ( \log \beta ) .
$$

## A.3 Sampling around the mean

We next give a stronger variant of Theorem 6 that removes the $\sqrt { 2 }$ factor overhead in our lower bound on $h ,$ , assuming the ability to compute the signs of the mean magnetization vector. Concretely, Theorem 6 is not adapted to the actual realization of J. Instead, we show that knowledge of

$$
\pmb { \tau } _ { i } ^ { \star } : = \mathrm { s i g n } ( \mathbf { m } _ { i } ) , \mathrm { ~ w h e r e ~ } \mathbf { m } _ { i } : = \mathbb { E } _ { \pi _ { \beta , h } } \left[ \mathbf { x } _ { i } \right] \mathrm { ~ f o r ~ a l l ~ } i \in [ d ] ,
$$

for a fixed J, where $\mathrm { s i g n } ( 0 ) : = 1$ , allows us to recenter the algorithm and improve our h range. Roughly speaking, the idea is to use overlap concentration to redefine our notion of sparsity as disagreement with $\boldsymbol { \tau } _ { i } ^ { \star }$ , rather than disagreement with $\mathbf { 1 } _ { d }$ as used in Section A.2.

We next set up some notation for this section. For $\pmb { \tau } \in \mathcal { X } ^ { d }$ , let

$$
N _ { \pmb { \tau } } ( \mathbf { x } ) : = | \{ i \in [ d ] : \mathbf { x } _ { i } \neq \pmb { \tau } _ { i } \} | .
$$

Also, for two i.i.d. draws $( \mathbf { x } ^ { ( 1 ) } , \mathbf { x } ^ { ( 2 ) } ) \sim \pi _ { \beta , h } ^ { \otimes 2 }$ , we define the overlap quantity

$$
R _ { 1 , 2 } : = \frac { \langle \mathbf { x } ^ { ( 1 ) } , \mathbf { x } ^ { ( 2 ) } \rangle } { d } .
$$

Observe that independence gives $\begin{array} { r } { \mathsf { E } [ R _ { 1 , 2 } ] = \frac { 1 } { d } \left\| \mathbf { m } \right\| _ { 2 } ^ { 2 } } \end{array}$ , which is a quantity depending on the realized J. We next state a stronger result from [RW26] which shows that above the weak AT line, $R _ { 1 , 2 }$ concentrates around the deterministic quantity $q ( \beta , h )$ , independent of J.

Proposition 5 (Lemma 4.14, [RW26]). Suppose that $( \beta , h )$ satisfies the weak $A T$ condition in Definition 3. Then there exists $C _ { \beta , h } > 0$ such that

$$
\mathbb { E } _ { \mathbf { J } } \mathbb { E } _ { \pi _ { \beta , h } ^ { \otimes 2 } } \left[ \exp \left( \frac { d ( R _ { 1 , 2 } - q ( \beta , h ) ) ^ { 2 } } { C _ { \beta , h } } \right) \right] \leq 2 .\tag{68}
$$

As a corollary, we upgrade Proposition 5 into a high-probability distance bound $\tan \tau ^ { \star }$ , replacing the direct sparsity notion from Lemma 24. Our strategy is to first relate the random quantity $\| \mathbf { m } \| _ { 2 } ^ { 2 }$ to $q ( \beta , h )$ using Proposition 5, and then to relate $\| \mathbf { m } \| _ { 2 } ^ { 2 }$ to the Hamming distance $N _ { \tau } .$ ⋆ using (70).

Lemma 25. Suppose that $( \beta , h )$ satisfies the weak AT condition. Then there exists $C _ { \beta , h } > 0$ such that for every $\rho , \delta \in ( 0 , \frac 1 2 )$ , with probability at least $1 - \delta$ over J,

$$
\pi _ { \beta , h } \left( N _ { \tau ^ { \star } } > \rho d \right) \leq \frac { 1 - q ( \beta , h ) } { 2 \rho } + \frac { 1 } { 2 \rho } \sqrt { \frac { C _ { \beta , h } \log \frac { 2 } { \delta } } { d } } .\tag{69}
$$

Proof. Set $q _ { d } ( \mathbf { J } ) : = \frac { \| \mathbf { m } \| _ { 2 } ^ { 2 } } { d }$ , and recall that $\mathbb { E } _ { \pi _ { \beta , h } ^ { \otimes 2 } } [ R _ { 1 , 2 } ] = q _ { d } ( \mathbf { J } )$ . We first derive

$$
\frac 1 d \mathbb { E } _ { \pi _ { \beta , h } } \left[ N _ { \pmb { \tau } ^ { \star } } ( \mathbf { x } ) \right] = \frac 1 { 2 d } \sum _ { i = 1 } ^ { d } \left( 1 - | m _ { i } | \right) \leq \frac 1 2 \left( 1 - q _ { d } ( \mathbf { J } ) \right) .\tag{70}
$$

Applying Jensen’s inequality conditionally on J to (68) then gives

$$
\mathbb { E } _ { \mathbf { J } } \left[ \exp \left( \frac { d \left( q _ { d } ( \mathbf { J } ) - q ( \beta , h ) \right) ^ { 2 } } { C _ { \beta , h } } \right) \right] \leq 2 .
$$

Hence, for every $t > 0$

$$
\mathbb { P } _ { \mathbf { J } } \left[ q _ { d } ( \mathbf { J } ) < q ( \beta , h ) - t \right] \leq 2 \exp \left( - \frac { d t ^ { 2 } } { C _ { \beta , h } } \right) .
$$

On the complementary event, (70) and Markov’s inequality under $\pi _ { \beta , h }$ give

$$
\pi _ { \beta , h } \left( N _ { \tau ^ { \star } } > \rho d \right) \leq \frac { 1 - q ( \beta , h ) + t } { 2 \rho } .
$$

Taking $t = { \sqrt { \frac { C _ { \beta , h } \log { \frac { 2 } { \delta } } } { d } } }$ proves the claim.

Lemma 25 shows that to apply our bounded-magnetization SK sampler (Corollary 5), we have reduced the problem to making $1 - q ( \beta , h )$ smaller than the sparsity parameter $\rho _ { \beta } \approx ( \beta ^ { 2 } \log \beta ) ^ { - 1 }$ The weak AT condition only gives $1 - q ( \beta , h ) \le \beta ^ { - 2 }$ , so we choose a slightly larger field strength, still asymptotic to the weak AT scale in Lemma 23:

$$
h _ { \mathrm { O C } } ( \beta ) : = \beta \sqrt { 2 \log \beta + 2 \log \log \beta + 2 \log \log \log \beta } , \quad q _ { \mathrm { O C } } ( \beta ) : = q \left( \beta , h _ { \mathrm { O C } } ( \beta ) \right) .\tag{71}
$$

Lemma 26 next shows that the field strength in (71) satisfies the weak AT condition, and thus we can bound the sparsity of its induced model using Lemma 25.

Lemma 26. There exist universal constants $C , \beta _ { 0 } > 0$ such that, for every $\beta \geq \beta _ { 0 }$ , and $h \geq h _ { \mathrm { O C } } ( \beta )$

$$
1 - q ( \beta , h ) \leq { \frac { C } { \beta ^ { 2 } \log \beta \log \log \beta } } .\tag{72}
$$

Consequently, $( \beta , h )$ satisfies the weak AT condition.

Proof. We first claim that $q ( \beta , h )$ is nondecreasing in h if $( \beta , h )$ strictly satisfies the weak AT condition, so it sufices to prove the result for $h = h _ { \mathrm { O C } } ( \beta )$ . Let $T ( q , h ) : = \mathbb { E } \operatorname { t a n h } ^ { 2 } ( \beta \sqrt { q } Z + h )$ , so that by definition, $q = T ( q , h )$ . Then, performing a Gaussian integration by parts gives

$$
\frac { \partial } { \partial q } T ( q , h ) = \beta ^ { 2 } \mathbb { E } \left[ \mathrm { s e c h } ^ { 2 } \left( \beta \sqrt { q } Z + h \right) \left( 1 - 3 \operatorname { t a n h } ^ { 2 } \left( \beta \sqrt { q } Z + h \right) \right) \right] \leq \beta ^ { 2 } ( 1 - q ) < 1 ,
$$

where the last inequality used the weak AT condition. Similarly,

$$
\frac { \partial } { \partial h } T ( q , h ) = 2 \mathbb { E } \left[ \operatorname { t a n h } \left( \beta \sqrt { q } Z + h \right) \operatorname { s e c h } ^ { 2 } \left( \beta \sqrt { q } Z + h \right) \right] > 0 ,
$$

because tanh(·) sech $\mathrm { \Omega _ { l } { } ^ { 2 } } ( \cdot )$ is odd and positive on $\mathbb { R } _ { > 0 }$ , and $\beta { \sqrt { q } } Z + h$ is centered around the positive value $h > 0$ . Now, implicit diferentiation gives

$$
q ^ { \prime } ( h ) = \frac { \partial } { \partial q } T ( q ( h ) , h ) q ^ { \prime } ( h ) + \frac { \partial } { \partial h } T ( q ( h ) , h ) \implies q ^ { \prime } ( h ) = \frac { \frac { \partial } { \partial h } T ( q ( h ) , h ) } { 1 - \frac { \partial } { \partial q } T ( q ( h ) , h ) } > 0 ,
$$

as desired. For the rest of the proof we take $h = h _ { \mathrm { O C } } ( \beta )$

Write $\begin{array} { r } { b _ { \beta } : = \frac { h _ { \mathrm { O C } } ( \beta ) } { \beta } } \end{array}$ . We first verify Lemma $2 2 \mathrm { { ^ { , } s } }$ hypotheses. From (71), the conditions $b _ { \beta } $ ∞ and $\frac { b _ { \beta } } { \beta }  0$ are immediate. Moreover, from the fixed-point equation defining $q _ { \beta } : = q _ { \mathrm { O C } } ( \beta )$ ，

$$
1 - q _ { \beta } = \mathbb { E } \left[ \mathrm { s e c h } ^ { 2 } \left( \beta \sqrt { q _ { \beta } } Z + \beta b _ { \beta } \right) \right] \leq \mathrm { s e c h } ^ { 2 } \left( \frac { \beta b _ { \beta } } { 2 } \right) + \mathbb { P } \left[ Z < - \frac { b _ { \beta } } { 2 } \right] \to 0 ,
$$

where the inequality split the expectation based on whether $Z \geq - \frac { b _ { \beta } } { 2 }$ or not, and used that sec ${ \mathrm { 1 } } ^ { 2 } \leq 1$ pointwise and $q _ { \beta } \leq 1$ . Thus, Lemma 22 applies with $p = 2$ , yielding

$$
1 - q _ { \beta } = \frac { 2 } { \beta \sqrt { q _ { \beta } } } \phi \left( \frac { b _ { \beta } } { \sqrt { q _ { \beta } } } \right) ( 1 + o ( 1 ) ) \leq \frac { C } { \beta } \phi ( b _ { \beta } ) ,
$$

for a universal constant C, where the last inequality used that $q _ { \beta } \leq 1$ , ϕ is decreasing on $\mathbb { R } _ { \geq 0 } .$ , and $q _ { \beta } \to 1$ as $\beta \to \infty$ . Finally, substituting the definition (71) gives

$$
\phi ( b _ { \beta } ) = \frac { 1 } { \sqrt { 2 \pi } \beta \log \beta \log \log \beta } ,
$$

and combining the above two displays proves (72). Finally, $\begin{array} { r } { \beta ^ { 2 } ( 1 - q _ { \beta } ) \le \frac { C } { \log \beta \log \log \beta } < 1 } \end{array}$ as $\beta \to \infty$ which verifies the weak AT condition with strict inequality. Since we earlier showed $q ^ { \prime } ( h ) > 0$ whenever $\beta ^ { 2 } ( 1 - q ( h ) ) < 1$ , all $h > h _ { \mathrm { O C } } ( \beta )$ also strictly satisfy the weak AT condition. □

We are finally ready to give our main result.

Theorem 7. Let $\beta \geq 1$ and $\delta \in ( 0 , \frac { 1 } { 2 } )$ , and suppose that (31) holds with $\delta  \frac { \delta } { 2 }$ Assume that $h \geq h _ { \mathrm { O C } } ( \beta )$ defined in (71), and define $\begin{array} { r } { \rho _ { \beta } : = \frac { c } { \beta ^ { 2 } \log ( e \beta ) } } \end{array}$ for a universal constant $c > 0$ . Also assume

$$
\log \log \beta \geq \frac { 2 C } { \delta } , \quad d \geq \frac { 4 C _ { \beta , h } \log \frac { 4 } { \delta } } { \delta ^ { 2 } \rho _ { \beta } ^ { 2 } } ,\tag{73}
$$

where C is a universal constant and $C _ { \beta , h }$ is from Lemma 25, and that we are given

$$
\tau _ { i } ^ { \star } : = \mathrm { s i g n } \left( \mathbb { E } _ { \pi _ { \beta , h } } \left[ \mathbf { x } _ { i } \right] \right) \ f o r \ a l l \ i \in [ d ] .
$$

Then with probability at least $1 - \delta$ over the SK model $( M o d e l \ 1 )$ , there is a polynomial-time algorithm that outputs x satisfying TV $( \operatorname { L a w } ( \mathbf { x } ) , \pi _ { \beta , h } ) \leq \delta .$ . Further,

$$
h _ { \mathrm { O C } } ( \beta ) = \sqrt { 2 } \beta \sqrt { \log \beta } \left( 1 + o ( 1 ) \right) .
$$

Proof. Let $\mathbf { D } : = \mathbf { d i a g } \left( \pmb { \tau } ^ { \star } \right)$ throughout. We first claim that the fixed-magnetization sampler in Theorem 4 (and hence, the bounded-magnetization sampler in Corollary 5) is invariant to replacing $\mathbf { J }  \mathbf { D J D }$ . To see this, all but one of the estimates in Lemma 12 remain unchanged under this replacement, because Du has the same $\ell _ { 2 }$ and $\ell _ { 1 }$ norms as u for any vector u, and $( { \bf D J D } ) \circ ( { \bf D J D } ) =$ J ◦ J. The only diference is that $\| \mathbf { D J D 1 } _ { d } \| _ { \infty }$ could be larger because $\mathbf { D 1 } _ { d }$ is no longer independent of J, but the same bound holds up to a $\sqrt { d }$ factor. This factors into Theorem 4’s initial $\chi ^ { 2 }$ divergence bound, which only afects the claimed runtime by a polynomial factor after taking logarithms.

To complete the proof, take $k : = \lfloor \rho _ { \beta } d \rfloor$ . Lemmas 25 and 26, and our assumed bounds (73), imply

$$
\mathbb { P } _ { \mathbf { x } \sim \pi _ { \beta , h } } \left[ N _ { \pmb { \tau } ^ { \star } } ( \mathbf { x } ) > k \right] \leq \frac { \delta } { 2 } .
$$

The rest of the proof is identical to Theorem 6, under the change of variables $\mathbf { x }  \mathbf { D x }$

In summary, Theorem 7 covers a range of h with a lower bound $h _ { \mathrm { O C } } ( \beta )$ roughly a $\sqrt { 2 }$ factor smaller than Theorem 6’s $h _ { \mathrm { H M } } ( \beta )$ . In particular, up to $\mathrm { ~ a ~ } 1 + o ( 1 )$ factor, $h _ { \mathrm { O C } } ( \beta )$ matches the AT and weak AT thresholds $h _ { \mathrm { A T } } ( \beta )$ and $h _ { \mathrm { w A T } } ( \beta )$ derived in Lemma 23. In comparison, a recent work by [BAR26] derives a similar polynomial-time sampling result, but examining their proof (particularly, Corollary 1.2 combined with the improvement in Remark 3.3) implies their threshold on the field strength h scales as $h = \Omega ( \beta ^ { 2 } \sqrt { \log \beta } )$ , i.e., roughly a $\beta$ factor larger than $h _ { \mathrm { A T } } ( \beta )$

We remark that Theorem 7 is a conditional result that requires access to $\tau ^ { \star }$ , the signs of the mean magnetization vector; we leave the eficient computation of $\tau ^ { \star }$ as an important open problem. Additionally, our definition of $h _ { \mathrm { O C } } ( \beta )$ includes potentially unnecessary low-order terms, arising due to a discrepancy between the weak AT region’s definition (which gives $1 - q ( \beta , h ) \le \beta ^ { - 2 } )$ with the requirements of our sampler in Corollary 5 (which requires a sparsity parameter $\rho \approx ( \beta ^ { 2 } \log \beta ) ^ { - 1 } )$ It is also an interesting open problem to improve the thresholds imposed by our approach, with the goal of eficient sampling across the entire AT region.

## B Scaling of Infinite ∆-Regular Tree Threshold

In this section, we provide a calculation that explains asymptotics induced by the “tree threshold” $\beta _ { c } ( \Delta )$ , which parameterizes the results of [CDKP22, KPPY25]. Concretely, $\begin{array} { r } { \beta _ { c } ( \Delta ) = \log ( \frac { \Delta } { \Delta - 2 } ) } \end{array}$ is a critical inverse temperature under which the Ising model on the infinite ∆-regular tree undergoes a phase transition. When $\beta < \beta _ { c } ( \Delta )$ , there is a unique fixed point of a certain message-passing recursion on the infinite tree, and when $\beta > \beta _ { c } ( \Delta )$ , two new fixed points appear. We defer an overview to $\scriptstyle \left[ \mathrm { L y o } 8 9 . \right.$ , Mos06]; here we calculate consequences of this threshold asymptotically.

The main algorithmic result of [CDKP22] at low temperatures $\beta > \beta _ { c } ( \Delta )$ is their Theorem $2 ( \mathrm { a } )$ They show that there is a corresponding critical magnetization $\eta _ { \Delta , \beta , 1 } ^ { + }$ at which the fixed-magnetization Ising model undergoes a computational phase transition. This $\eta _ { \Delta , \beta , 1 } ^ { + }$ is the expected spin at the root of the infinite ∆-regular tree, at one of the new fixed points emerging when $\beta > \beta _ { c } ( \Delta )$ (see Section 4, [CDKP22] for these calculations). Note that [CDKP22] defines the magnetization $\eta \in [ - 1 , 1 ]$ as (in our notation) $- 1 + { \frac { 2 k } { d } }$ , so that $k = d$ corresponds to $\eta = 1$ (and $k = 0$ corresponds to $\eta = - 1 )$ . Rearranging, Theorem $2 ( \mathrm { a } )$ of [CDKP22] applies at sparsity levels

$$
\eta < - \eta _ { \Delta , \beta , 1 } ^ { + } \implies k < \frac { d } { 2 } \left( 1 - \eta _ { \Delta , \beta , 1 } ^ { + } \right) .
$$

It remains to understand $1 - \eta _ { \Delta , \beta , 1 } ^ { + }$ . From Section 1.1, [CDKP22], letting L be the largest root of

$$
L = \left( \Delta - 1 \right) \operatorname { a r c t a n h } \left( \operatorname { t a n h } ( L ) \operatorname { t a n h } \left( { \frac { \beta } { 2 } } \right) \right) ,
$$

denoting $\eta _ { \beta } : = \eta _ { \Delta , \beta , 1 } ^ { + }$ for short,

$$
\eta _ { \beta } = \operatorname { t a n h } \left( L + \operatorname { a r c t a n h } \left( \operatorname { t a n h } ( L ) \operatorname { t a n h } \left( { \frac { \beta } { 2 } } \right) \right) \right) .
$$

Let $\begin{array} { r } { A : = \operatorname { a r c t a n h } ( \operatorname { t a n h } ( L ) \operatorname { t a n h } ( \frac { \beta } { 2 } ) ) = \frac { 1 } { \Delta - 1 } L } \end{array}$ . Then,

$$
\operatorname { t a n h } A = \operatorname { t a n h } \left( ( \Delta - 1 ) A \right) \operatorname { t a n h } \left( { \frac { \beta } { 2 } } \right) \implies \operatorname { t a n h } \left( { \frac { \beta } { 2 } } \right) = { \frac { \operatorname { t a n h } A } { \operatorname { t a n h } \left( ( \Delta - 1 ) A \right) } } .
$$

Now using the approximation tan $\begin{array} { r } { \mathbf { \mu } ( c ) = 1 - \Theta ( \exp ( - 2 c ) ) } \end{array}$ for large $c ,$ we obtain $A = { \textstyle \frac { \beta } { 2 } } + O _ { \Delta } ( 1 )$ Finally, plugging this back into our definition of $\eta _ { \beta }$ , we have

$$
1 - \eta _ { \beta } = 1 - \operatorname { t a n h } \left( \Delta A \right) = \Theta \left( \exp \left( - 2 \Delta A \right) \right) = \exp \left( - \Delta \beta + O _ { \Delta } ( 1 ) \right) .
$$

Therefore, for the sparsity regime

$$
k < d \exp \left( { - \Delta \beta + O _ { \Delta } ( 1 ) } \right)\tag{74}
$$

to be meaningful (i.e., the inequality above does not hold only when $k = 0 )$ , the result of [CDKP22] is limited to inverse temperatures of $\beta = O ( \log d )$ for constant-degree graphs.

The main algorithmic result of the subsequent work [KPPY25] is their Theorem 1.1, which is parameterized at a slightly diferent critical magnetization $\eta _ { \beta , a }$ , which is at least as large as the $\eta _ { \beta }$ from before. Therefore, Theorem 1.1 of [KPPY25] is also restricted to the regime (74).

Finally, we note that the comparison in this section is purely a statement about the allowable temperatures that our sparsity-aware framework tolerates (vs. the prior works [CDKP22, KPPY25]), for our models of interest. For example, directly applying our results to the graph-based Ising models in these prior works only permits rapid mixing in the high-temperature regime $\begin{array} { r } { \beta = O ( \frac { 1 } { k } ) } \end{array}$ . However, for other well-studied models, e.g., Models 1 and 2, our results allow taking inverse temperatures as large as $\beta = \mathrm { p o l y } ( d )$ when k is suficiently small.