# Eigenvalue-Decomposition Cost Denoising as an Alternative to Predict-then-Optimize for Shortest-Path Problems

Henry Aldridge-Krawciw

Irene Aldridge

## Abstract

Predict-then-optimize methods such as Smart “Predict, then Optimize” (SPO+) of Elmachtoub and Grigas [2022] learn a mapping from contextual features to unknown edge costs and then solve the induced combinatorial problem on the predicted costs. This approach is powerful but relies on the predictive model being well specified: when the true cost-generating process is nonlinear in the features and the predictor is linear, SPO+’s performance degrades as the misspecification grows. We propose and evaluate a structurally diferent remedy for a specific but common setting: when the decision-maker observes many noisy realizations of the same underlying cost process, the realized cost vectors themselves can be treated as a noisy signal and denoised directly, via eigenvalue decomposition (equivalently, Principal Component Analysis) of their covariance matrix, before ever invoking a predictive model. We instantiate this idea on the 5 × 5 grid shortest-path benchmark introduced by Elmachtoub and Grigas [2022], retaining only the top-k eigenvectors of the training cost covariance matrix and projecting new noisy cost observations onto that subspace prior to solving with Dijkstra’s algorithm [Dijkstra, 1959]. We find that the choice of k is decisive: keeping only k=2 eigenvectors discards real signal and underperforms even the naive noisy-cost baseline, while setting k=5 to match the true latent feature dimension makes eigenvalue-denoised Dijkstra the best-performing method at every misspecification level tested, outperforming SPO+ by a wide margin under high mis specification.

## 1 Introduction

Many operational decisions are shortest-path problems in disguise: routing a vehicle, scheduling a sequence of tasks, or moving a packet through a network, all reduce to finding a minimum-cost path through a graph whose edge costs are not known in advance but must be estimated from data [Vera et al., 2021, Chen et al., 2004]. The dominant paradigm for this class of problems is predict-thenoptimize: fit a model that maps observed features to edge costs, then feed the predicted costs into an of-the-shelf combinatorial solver. The Smart “Predict, then Optimize” (SPO+) framework of Elmachtoub and Grigas [2022] refined this paradigm by replacing the usual squared-error training loss with a convex surrogate of the true decision loss, so that the predictive model is trained to make good decisions rather than merely accurate point predictions of cost.

SPO+ and its relatives [Bertsimas and Kallus, 2020, Ban and Rudin, 2019, Donti et al., 2017, Wilder et al., 2019] all share an implicit assumption: that a hypothesis class exists (in SPO+’s case, linear functions of the features) rich enough to capture the true cost-generating process reasonably well. When this assumption fails – when the true process is a nonlinear function of the features but the predictor is constrained to be linear – the predictor is misspecified, and SPO+’s decision quality degrades as the degree of nonlinearity increases, a pattern we reproduce and quantify in Section 5.

This paper asks a diferent question. Suppose the decision-maker does not need to predict costs from features at all, but instead directly observes many independent, noisy realizations of the cost vector for the same underlying network – e.g. repeated daily commutes over the same road network, where each day’s travel times are a noisy draw from a common but unknown structure. In this setting, the natural tool is not a supervised predictor but an unsupervised one: treat the collection of observed cost vectors as a data matrix, and denoise it. If the true cost signal occupies a low-dimensional subspace of the full edge-cost space – which it will if it is driven by a small number of latent factors, as in the Elmachtoub and Grigas [2022] synthetic design – then an eigenvalue decomposition (equivalently, PCA [Pearson, 1901, Hotelling, 1933]) of the observed cost covariance matrix should recover that subspace, and reconstructing each new observation using only its top-k eigencomponents should suppress much of the idiosyncratic noise while retaining the signal. This is exactly the logic behind optimal singular-value shrinkage for matrix denoising [Gavish and Donoho, 2014], and behind the use of eigendecomposed covariance structure in decision-making more broadly – most famously in mean-variance portfolio theory [Markowitz, 1952], and, in the specific context of decision-theoretic regret, in the closed-form regret-equals-covariance characterization of Aldridge [2026]. See also Aldridge [2025] for a complementary discussion of reordering the predict/optimize pipeline.

We make this idea concrete for the shortest-path setting and compare it head-to-head against three natural baselines: Dijkstra’s algorithm run on the (unobservable, oracle) true costs; Dijkstra run naively on the raw noisy/misspecified costs; and SPO+. Our central empirical finding is that the eigenvalue-denoising approach is highly sensitive to the number of retained components k: an undersized k discards real signal and underperforms doing nothing at all, while a correctly sized k makes the method the best of the four across every misspecification level we test, including levels at which SPO+ itself breaks down.

## 2 Related Work

Shortest paths and predict-then-optimize. Dijkstra’s algorithm [Dijkstra, 1959] remains the standard exact solver for nonnegative-weight shortest-path problems, and constrained and largescale variants continue to be an active area [Vera et al., 2021, Chen et al., 2004]. When edge costs are not directly observed, predict-then-optimize methods learn them from contextual features. Elmachtoub and Grigas [2022] introduced the SPO and SPO+ loss functions, convex surrogates for the true regret of a decision induced by a predicted cost vector, and showed both statistical and computational advantages over training a predictor with a generic loss (e.g. squared error) that ignores the downstream optimization structure. Related decision-focused learning frameworks include the predictive-prescriptive framework of Bertsimas and Kallus [2020], the data-driven newsvendor analysis of Ban and Rudin [2019], and end-to-end task-based learning through the optimization layer itself [Donti et al., 2017, Wilder et al., 2019]. All of these methods are supervised: they require paired (feature, realized-cost) training data and a hypothesis class for the feature-to-cost map.

Eigenvalue decomposition and denoising. Principal Component Analysis [Pearson, 1901, Hotelling, 1933] is the classical tool for finding the low-dimensional subspace that best explains the variance of a data matrix, and is equivalent to an eigenvalue decomposition of the (empirical) covariance matrix. When a data matrix is a low-rank signal plus noise, keeping only the leading eigen/singular components is a standard and asymptotically principled denoising strategy; Gavish and Donoho [2014] characterize the optimal number of components to retain (and the optimal shrinkage of their singular values) under a spiked-covariance noise model. Unlike predict-thenoptimize, this denoising approach requires no feature-to-cost hypothesis class at all – only repeated observations of the cost vector – and is consequently immune to the specific failure mode of predictor misspecification that aflicts $\mathrm { S P O + }$

Regret and covariance. Closest in spirit to our evaluation methodology is Aldridge [2026], which shows that for stochastic linear programs (including shortest path), expected regret relative to acting on the mean cost decomposes exactly into a covariance term between the realized cost and the realized optimal decision, with no residual for continuous-cost LPs. We use ordinary realizedcost regret (relative to the oracle optimum) throughout this paper as our evaluation metric, and note that the covariance-based estimator of that reference could serve as a cheaper drop-in replacement for the Sample Average Approximation regret estimates we compute directly, an avenue we leave to future work. Aldridge [2025] separately considers reordering the predict/optimize pipeline itself, a complementary idea to the denoise-only approach studied here.

## 3 Methodology

## 3.1 Grid and Cost Generation

We use the same $5 \times 5$ directed acyclic grid graph as Elmachtoub and Grigas [2022]’s shortest-path experiments: 25 nodes arranged in a grid, edges permitted only rightward and downward, giving $d = 4 0$ edges, a source at the top-left corner, and a sink at the bottom-right corner. For a feature vector $x \in \mathbb { R } ^ { p } \left( p = 5 \right)$ drawn i.i.d. $N ( 0 , I _ { p } )$ , and a fixed random matrix $B \in \{ - 1 , + 1 \} ^ { d \times p }$ , the true (noise-free) cost of edge e is

$$
c _ { e } ^ { \mathrm { t r u e } } ( x ) = \Big [ \frac { 1 } { \sqrt { p } } ( B x ) _ { e } + 3 \Big ] ^ { \mathrm { d e g } } ,\tag{1}
$$

where deg $\in \{ 1 , 2 , 4 \}$ controls the degree of nonlinearity in the features – deg = 1 is linear (well specified for a linear predictor), and larger deg increasingly misspecifies any linear model. The observed (misspecified/noisy) cost is

$$
c _ { e } ^ { \mathrm { n o i s y } } ( x ) = c _ { e } ^ { \mathrm { t r u e } } ( x ) \cdot \varepsilon _ { e } , \qquad \varepsilon _ { e } \sim \mathrm { U n i f o r m } [ 1 - \bar { \varepsilon } , 1 + \bar { \varepsilon } ] ,\tag{2}
$$

with $\bar { \varepsilon } = 0 . 5$ throughout. This is exactly the synthetic design of Elmachtoub and Grigas [2022], with the true and noisy costs exposed separately so that we can evaluate decision quality against the noise-free ground truth while only ever acting on noisy observations.

## 3.2 Baselines

(1) Dijkstra on true costs. Solves the shortest path directly on $c ^ { \mathrm { t r u e } } ( x )$ for each test instance. This is an oracle upper bound: no real decision-maker has access to $c ^ { \mathrm { t r u e } }$ , but it defines the zero-regret reference point for all other methods.

(2) Dijkstra on misspecified (noisy) costs. Solves the shortest path directly on the raw observation $c ^ { \mathrm { n o i s y } } ( x )$ , ignoring that it is a noisy realization rather than the true expected cost. This is the naive baseline.

(3) SPO+ [Elmachtoub and Grigas, 2022]. A linear predictor $\hat { c } ( x ) = W x$ is trained on $( x _ { i } , c _ { i } ^ { \mathrm { n o i s y } } )$ pairs from a training set, using the SPO+ subgradient method (Algorithm 1), and the shortest path is then solved on $\hat { c } ( x )$ for each test instance.

Algorithm 1 SPO+ training (subgradient descent), following Elmachtoub and Grigas [2022]   
1: Initialize W ← small random matrix   
2: for epoch $= 1 , \ldots , E$ do   
3: for each training example $( x _ { i } , c _ { i } , z _ { i } ^ { * } )$ in random order do   
4: $\hat { c } \gets W x _ { i }$   
5: $z _ { \mathrm { s p o } } \gets \mathrm { a r g } \operatorname* { m i n } _ { z \in \mathcal { Z } } \left( 2 \hat { c } - c _ { i } \right) ^ { \top } z$ ▷ shortest-path oracle   
6: $\nabla _ { \hat { c } } \gets 2 \left( z _ { i } ^ { * } - z _ { \mathrm { s p o } } \right)$   
7: $W \gets W - \eta \nabla _ { \hat { c } } x _ { i } ^ { \top }$ ▷ plus $\ell _ { 2 }$ regularization   
8: end for   
9: end for

## 3.3 Eigenvalue-Decomposition Denoising

Our proposed method (4) requires no feature-to-cost hypothesis class. Instead, from a training sample of n noisy cost vectors $C \in \mathbb { R } ^ { n \times d }$ (rows are training instances, columns are edges), we compute the empirical mean $\begin{array} { r } { \bar { c } = \frac { 1 } { n } \sum _ { i } C _ { i } } \end{array}$ , the centered data $\tilde { C } = C - \bar { c }$ , and the empirical covariance

$$
\Sigma \ = \ \frac 1 n \tilde { C } ^ { \top } \tilde { C } \ \in \ \mathbb { R } ^ { d \times d } .\tag{3}
$$

Because Σ is symmetric positive semi-definite, its eigenvalue decomposition $\Sigma = V \Lambda V ^ { \top }$ has real, non-negative eigenvalues; let $V _ { k } \ \in \ \mathbb { R } ^ { d \times k }$ collect the eigenvectors associated with the k largest eigenvalues. For a new noisy observation $c ^ { \mathrm { n o i s y } }$ , the denoised reconstruction is the projection onto the afine subspace spanned by these top-k directions:

$$
c ^ { \mathrm { d e n o i s e d } } = \bar { c } + V _ { k } V _ { k } ^ { \top } \left( c ^ { \mathrm { n o i s y } } - \bar { c } \right) .\tag{4}
$$

$V _ { k }$ and ¯c are fit once on the training set and then applied prospectively to each test instance, exactly as W is fit once and applied prospectively in SPO+ – both methods use only training data to build a fixed object that is applied, unchanged, to new test instances. The shortest path is then solved on $c ^ { \mathrm { d e n o i s e d } }$ with Dijkstra.

The rationale for Eq. (4) is that under the generative model of Eq. (1), the true cost signal is driven by only $p = 5$ latent coordinates (linearly, before the deg power is applied), so it plausibly concentrates in a low-dimensional subspace of the d = 40-dimensional edge-cost space, whereas the multiplicative noise $\varepsilon _ { e }$ in Eq. (2) is comparatively unstructured across edges. Retaining only the top-k eigendirections should therefore capture a disproportionate share of the signal while discarding a disproportionate share of the noise – provided k is large enough to actually span the signal subspace. We test this directly by comparing k = 2 against k = 5 (the true latent dimension) in Section 5.

## 4 Experimental Setup

For each misspecification level deg $\in \{ 1 , 2 , 4 \}$ we draw $n _ { \mathrm { t r a i n } } = 2 0 0$ training instances and $n _ { \mathrm { t e s t } } =$ 300 test instances from Eqs. (1)–(2), sharing a single random matrix B across the train/test split. SPO+ is trained for 50 epochs of subgradient descent with learning rate 0.02 and $\ell _ { 2 }$ penalty 10<sup>−4</sup> (Algorithm 1). The eigen-denoiser (Eq. (4)) is fit once on the training cost matrix, for $k \in \{ 2 , 5 \}$ All four methods are evaluated on the same test instances, and every method’s chosen path is scored by its true cost $c ^ { \mathrm { t r u e } } ( x ) ^ { \top } z$ , so that method (1) always attains the minimum possible cost by construction; we report the other three methods’ mean regret relative to that oracle optimum, regret $= \mathbb { E } \left[ \frac { \underline { { c } } ^ { \mathrm { t r u e } \top } z - c ^ { \mathrm { t r u e } \top } z ^ { * } } { c ^ { \mathrm { t r u e } \top } z ^ { * } } \right]$

## 5 Results

## 5.1 Undersized Denoising (k = 2)

Table 1 reports results when only the top 2 eigenvectors are retained, capturing between 33% and 43% of the training cost variance depending on deg. At every misspecification level, eigenvaluedenoised Dijkstra is the worst of the four methods, including worse than doing nothing (the naive noisy-cost baseline).

Table 1: Mean regret relative to the true-cost optimum, k = 2 retained eigenvectors.
<table><tr><td>deg</td><td>Noisy (naive)</td><td>SPO+</td><td>Eigen-denoised (k=2)</td><td>Var. explained</td></tr><tr><td>1</td><td>6.82%</td><td>3.61%</td><td>13.41%</td><td>33.4%</td></tr><tr><td>2</td><td>5.18%</td><td>6.59%</td><td>25.86%</td><td>42.7%</td></tr><tr><td>4</td><td>4.55%</td><td>26.12%</td><td>77.71%</td><td>39.4%</td></tr></table>

## 5.2 Correctly-Sized Denoising (k = 5)

Table 2 repeats the experiment with k = 5, matching the true latent feature dimension p. The ranking reverses completely: eigenvalue-denoised Dijkstra becomes the best method at every misspecification level, and its advantage over SPO+ widens sharply as deg grows – at deg = 4, SPO+’s regret (26.12%) is nearly double that of the denoised method (14.67%), and both are far worse than the denoised method’s own regret at lower deg.

Table 2: Mean regret relative to the true-cost optimum, k = 5 retained eigenvectors.
<table><tr><td>deg</td><td>Noisy (naive)</td><td>SPO+</td><td>Eigen-denoised (k=5)</td><td>Var. explained</td></tr><tr><td>1</td><td>6.82%</td><td>3.61%</td><td>1.60%</td><td>60.3%</td></tr><tr><td>2</td><td>5.18%</td><td>6.59%</td><td>1.74%</td><td>76.3%</td></tr><tr><td>4</td><td>4.55%</td><td>26.12%</td><td>14.67%</td><td>70.9%</td></tr></table>

## 5.3 Discussion

Two patterns are worth isolating. First, SPO+’s regret grows sharply with deg (3.61% → 6.59% → 26.12%): as the true cost becomes more nonlinear in the features, a linear predictor becomes more misspecified, and no amount of SPO-consistent training can fix a hypothesis class that cannot represent the target function. Second, the eigen-denoiser’s regret is comparatively insensitive to deg when k is correctly sized $( 1 . 6 0 \%  1 . 7 4 \%  1 4 . 6 7 \% )$ : because the denoiser never attempts to model the feature-to-cost functional form at all, it is not directly harmed by that function becoming more nonlinear – it only needs the realized cost vectors to continue concentrating in a k-dimensional subspace, which they do for any deg given how Eq. (1) is constructed. The method’s sensitivity to deg that does remain (1.60% at deg = 1 growing to 14.67% at deg = 4) is attributable to the increasing dispersion of the underlying cost distribution at higher deg, which the fraction of variance explained (70–76% at k = 5, rather than 100%) does not fully capture.

These results should not be read as “denoising dominates predict-then-optimize.” Rather, the two methods fail in diferent regimes and for diferent reasons: SPO+ fails when the hypothesis class cannot represent the true cost function; eigenvalue denoising fails when k is chosen without regard to the efective rank of the underlying signal. In our setting the correct k happened to be knowable in advance because we generated the data ourselves; in practice, k would need to be chosen by a model-selection procedure (e.g. cross-validated reconstruction error, or the spiked-covariance threshold of Gavish and Donoho [2014]) rather than assumed.

## 6 Limitations and Future Work

Our experiments use a single, small (25-node) grid and a synthetic cost process whose true rank is known by construction; real cost data will not come with a known target rank, and selecting k well is itself a nontrivial estimation problem [Gavish and Donoho, 2014]. The denoising approach also requires repeated observations of costs on a fixed network topology, which SPO+ does not require – SPO+ can price out an edge cost never seen at training time as long as its features are in-distribution, whereas the eigen-denoiser has no mechanism to generalize beyond the covariance structure it was fit on. A natural next step is a hybrid: use SPO+ (or any feature-based predictor) to produce an initial cost estimate, and use eigenvalue denoising of the residual between predicted and observed costs to further clean the signal before optimizing, potentially combining the strengths of both approaches. We also flag, per Aldridge [2026], that the covariance-based regret estimator could replace our direct Sample Average Approximation of regret at a fraction of the computationa cost for larger problems, which we leave to future work.

## 7 Conclusion

We introduced and evaluated an eigenvalue-decomposition-based alternative to predict-then-optimize for shortest-path problems with misspecified, noisy costs. The method requires no feature-to-cost model and instead denoises repeated cost observations directly via PCA of their covariance matrix. Its performance is highly sensitive to the number of retained components: an undersized choice underperforms doing nothing, while a correctly sized choice outperforms SPO+ substantially, particularly under high model misspecification where SPO+’s linear hypothesis class struggles. The two approaches are complementary rather than competing, and combining them is a promising direction for future work.

## References

Irene Aldridge. Optimize, then predict. SSRN Working Paper, 2025.

Irene Aldridge. Regret equals covariance: A closed-form characterization for stochastic optimization. arXiv:2605.14019 [econ.EM], 2026.

Gah-Yi Ban and Cynthia Rudin. The big data newsvendor: Practical insights from machine learning. Operations Research, 67(1):90–108, 2019.

Dimitris Bertsimas and Nathan Kallus. From predictive to prescriptive analytics. Management Science, 66(3):1025–1044, 2020.

Shigang Chen, Meongchul Song, and Sartaj Sahni. Two techniques for fast computation of constrained shortest paths. In IEEE Global Telecommunications Conference, 2004. GLOBECOM ’04, volume 3, pages 1348–1352, 2004.

Edsger W. Dijkstra. A note on two problems in connexion with graphs. Numerische Mathematik, 1(1):269–271, 1959.

Priya Donti, Brandon Amos, and J. Zico Kolter. Task-based end-to-end model learning in stochastic optimization. In Advances in Neural Information Processing Systems (NeurIPS), volume 30, 2017.

Adam N. Elmachtoub and Paul Grigas. Smart “Predict, then Optimize”. Management Science, 68 (1):9–26, 2022.

Matan Gavish and David L. Donoho. The optimal hard threshold for singular values is $4 / { \sqrt { 3 } } .$ IEEE Transactions on Information Theory, 60(8):5040–5053, 2014.

Harold Hotelling. Analysis of a complex of statistical variables into principal components. Journal of Educational Psychology, 24(6):417–441, 1933.

Harry Markowitz. Portfolio selection. The Journal of Finance, 7(1):77–91, 1952.

Karl Pearson. On lines and planes of closest fit to systems of points in space. Philosophical Magazine, 2(11):559–572, 1901.

Alberto Vera, Siddhartha Banerjee, and Samitha Samaranayake. Computing constrained shortestpaths at scale. Operations Research, 70, 2021.

Bryan Wilder, Bistra Dilkina, and Milind Tambe. Melding the data-decisions pipeline: Decisionfocused learning for combinatorial optimization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 1658–1665, 2019.