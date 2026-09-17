# A General Kernel Framework for Non-CND Distance Measures Using |D|-Dimensional Sparse Landmark Embeddings

Marcus M. Noack Applied Mathematics and Computational Research Division, Lawrence Berkeley National Laboratory Berkeley, CA 94720, USA MarcusNoack@lbl.gov

Maher B. Alghalayini Applied Mathematics and Computational Research Division, Lawrence Berkeley National Laboratory Berkeley, CA 94720, USA

Mark D. Risser   
Climate and Ecosystem Sciences Division,   
Lawrence Berkeley National Laboratory Berkeley, CA 94720, USA

## Abstract

Kernel methods, and Gaussian Processes (GPs) in particular, require a Hilbertian distance measure—one whose square is conditionally negative definite (CND)—to guarantee positive semi-definiteness (PSD) of the kernel matrix; a condition that fails for many natural input spaces, including smooth manifolds and spaces of probability distributions. We propose the Sparse Landmark Embedding (SLE) kernel, which eliminates this requirement entirely. Each input is embedded into a sparse feature vector via compactly supported bump functions centered at all |D| training points; applying any standard PSD kernel in this embedding space yields a kernel that is provably PSD for arbitrary distance measures. The compact support automatically controls embedding sparsity, keeping kernel matrices well-conditioned and computationally tractable despite the high ambient dimension. We provide theoretical guarantees on PSD, sparsity, stability, and universal approximation, and demonstrate, using geodesic and Wasserstein distances, that the SLE kernel matches or substantially exceeds domain-specific baselines in both predictive accuracy and uncertainty quantification.

## 1 Introduction

Modern machine learning applications increasingly require flexible, probabilistic models that can handle diverse data structures while quantifying uncertainty. Gaussian Process (GP) regression has emerged as a flexible kernel-based method for approximating unknown functions from limited observed data [Rasmussen and Williams, 2006]. A GP defines a normal prior probability distribution $\mathcal { N } ( \mathbf { m } , \mathbf { K } )$ over an arbitrary set of function values $\mathbf { f } = [ f ( x _ { 1 } ) , f ( x _ { 2 } ) , \dot { \mathbf { \Omega } } . . . , \dot { f ( x _ { N } ) } ] ^ { T }$ , where $x \in \mathcal { X }$ with a mean function $m ( x ) -$ often assumed to be zero for simplicity — and a covariance matrix $\mathbf { K } = \mathrm { C o v } ( \mathbf { f } , \mathbf { f } )$ ${ \bf K } \in \mathrm { \overline { { \mathbb { R } } } } ^ { N \times N }$ . The true underlying function generating the data is assumed to be $f ( x )$ . The observed dataset $\mathbf { \mathcal { D } } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { | \mathcal { D } | }$ with cardinality |D| is assumed to result from the functional relationship $y _ { i } = f ( x _ { i } ) + \epsilon ( x _ { i } )$ , where the noise $\epsilon ( x _ { i } )$ is drawn from a Gaussian distribution with zero mean. The Gaussian prior is commonly assumed to be defined over function values at the data points; that means $N = | \bar { \mathcal { D } } |$ . We denote the collection of all inputs by X and the corresponding outputs by $\mathbf { y } .$ The covariance matrix is calculated by applying a positive semi-definite (PSD) kernel function to positional arguments; i.e., $\mathbf { K } = \mathrm { C o v } ( \mathbf { f } , \mathbf { \bar { f } } ) \mathbf { \bar { \Pi } } = [ k ( \mathbf { \bar { \boldsymbol { x } } } _ { i } , \mathbf { \bar { \boldsymbol { x } } } _ { j } ) ] _ { i , j = 1 } ^ { N } .$ . Stationary kernels depend only on the distance between inputs; $k ( x _ { i } , x _ { j } ) = k ( | x _ { i } - x _ { j } | ) \colon$ ; most non-stationary kernels also use some form of distance between input pairs in their formulation. Positive definiteness of such stationary and non-stationary kernels is guaranteed when the square of the underlying distance metric is conditionally negative definite (CND)—equivalently, that it be Hilbertian [Berg et al., 1984]. This property is not generally satisfied for distance measures on non-Euclidean spaces.

This issue is particularly apparent when input data lie on manifolds (e.g., endowed with the geodesic distance), are probability distributions (Wasserstein distance), strings, trees, or graphs, many of which naturally admit distance measures that are not CND. For example, the Wasserstein distance $W _ { 2 }$ between distributions is not CND in general [Bachoc et al., 2020], and geodesic distances on Riemannian manifolds similarly fail this property [Feragen et al., 2015, Haasdonk and Burkhardt, 2007]. As a result, naively applying kernels can yield indefinite Gram matrices, violating the mathematical requirements of GPs and other kernel methods.

A variety of workarounds have been developed. For instance, for smooth manifolds, intrinsic heat or diffusion kernels provide PSD alternatives at the expense of increased computational burden and the need for geometric information [Lafon and Lee, 2006]. For distributions, the sliced Wasserstein distance [Bonneel et al., 2015] offers a computationally tractable and CND alternative, but may sacrifice accuracy and efficiency.

One particularly interesting approach to handle non-CND distances without distorting geometry is to move from distance-based kernels to landmark-based embedding kernels, which map inputs into finite-dimensional feature spaces defined by distances to a selected set of reference points. Rather than relying on a single pairwise distance, these methods construct feature vectors of the form $\boldsymbol { \phi } ( \boldsymbol { x } ) = \left[ d ( \overline { { \boldsymbol { x } } } , \boldsymbol { \ell } _ { 1 } ) , \ldots , \overline { { d ( \boldsymbol { x } , \boldsymbol { \ell } _ { m } ) } } \right]$ , where $\{ \ell _ { i } \} _ { i = 1 } ^ { m }$ are landmark points drawn from the data or placed strategically. Once embedded, a Euclidean-distance kernel — such as any Matérn kernel — can be applied to these representations, ensuring positive semidefiniteness even when the original distance is not CND. This idea connects to early Nyström and inducing-point approximations in kernel methods [Williams and Seeger, 2001, Drineas et al., 2005, Titsias, 2009, Hensman et al., 2013], as well as more recent landmark-based constructions for learning on manifolds and distributions [Jayasumana et al., 2013]. Unlike sliced or projected distances, landmark embeddings preserve richer structural information, making them a well-suited candidate for extending Gaussian processes to non-Euclidean and non-CND settings.

Despite their conceptual appeal, landmark-based kernels introduce significant practical challenges. The first difficulty lies in choosing or learning landmark locations. If landmarks are selected heuristically (e.g., via k-means or random subsampling), they may fail to capture important geometric or topological features of the data manifold, leading to degraded predictive performance [El Alaoui and Mahoney, 2015, Musco and Musco, 2017]. Conversely, jointly learning landmark positions as model parameters introduces a highly nonconvex optimization problem, in which gradients must propagate through distance computations and kernel evaluations, often resulting in unstable training dynamics. A second major challenge stems from the embedding’s dimensionality. As the number of landmarks grows, each input is mapped to a feature vector in $\mathbb { R } ^ { m }$ , where m may be in the hundreds or thousands. While increasing m improves geometric fidelity, it exacerbates the curse of dimensionality, leading to poor predictive performance and uncertainty quantification. Moreover, high-dimensional embeddings tend to yield poorly conditioned kernel matrices, which complicates both hyperparameter learning and posterior inference. Thus, practical deployment of landmark-based Gaussian processes requires a careful balance between expressivity (large m) and tractability (small m), a regime for which no principled selection mechanisms currently exist.

In this paper, we propose Sparse Landmark Embedding (SLE) kernels that treat all data points as landmarks — therefore avoiding the need to select their positions — and leverage bump-functionbased embeddings for automatic dimensionality reduction. Our kernel operates on arbitrary distance measures, mitigates the need for CND distance approximations, and is computationally stable. An overview is given in Figure 1. Our contributions are: (i) the SLE kernel construction itself, which renders the use of $a l l \ | \ D |$ training points as landmarks tractable and thereby eliminates the landmark-selection problem; (ii) theoretical guarantees on positive semi-definiteness, sparsity, conditioning, local geometric fidelity, and universal approximation (Theorems 1–9, Proposition 1); (iii) a non-stationary extension that preserves the PSD guarantee for arbitrary spatially varying bump parameters (Appendix B); and (iv) an experimental evaluation against three domain-specific baselines on manifold- and distribution-valued GP regression, together with controlled ablation studies that isolate the effect of the bump embedding relative to raw-distance landmark embeddings (Appendix E).

![](images/1473cc87463e375abfa1e2831104a96d94a022156f78cb1397c1529921078a16.jpg)

![](images/10f139ce007e7d224988f1f93575fa1f564c8570e265ea585c19ec95ac666796.jpg)

![](images/3e973b16a9de9159d49de2c6adf3f4ecb82f618ce241d6b5dcec6c71d0efba3d.jpg)

![](images/bf0579331e18d85599001dd290d8eb5ca9a8ad7ceec03faf7f8e1bd2fea5b2e9.jpg)

![](images/dd83aa513053e5b47bd432b8856b873ffb96af51ab6e8aea1bbbb1c9a029160a.jpg)  
Figure 1: We propose the Sparse Landmark Embedding kernel $k _ { S L E }$ , a general (non-)stationary kernel for Gaussian Processes (GPs). Panels (a) and (b) show a standard GP regression task performed with the proposed SLE kernel (a) and a Matérn kernel $( \nu = 3 / 2 )$ (b), demonstrating the comparable behavior of the two kernels in a standard regression scenario, with the proposed kernel yielding a lower prediction error (RMSE) and better uncertainty quantification (CRPS). In simple cases, our kernel structurally and empirically resembles well-known stationary kernels such as the RBF and other Matérn kernels (c). Unlike standard stationary kernels, the proposed SLE kernel does not rely on a conditionally negative-definite (CND) distance metric, making it applicable to input spaces that lack such a metric. This includes smooth manifolds, where geodesic distances can be used without restriction (d), as well as sets of distributions, where exact Wasserstein distances can be employed directly (e).

## 2 Related work

A foundational requirement for kernel methods in general, and GPs in particular, is that the kernel be positive semi-definite (PSD), with distance-based kernels (e.g., RBF, Matérn) being PSD only when the associated distance metric is conditionally negative definite (CND) [Berg et al., 1984]. When this condition is violated, applying standard stationary and non-stationary kernels can yield indefinite covariance matrices and unstable inference. This phenomenon has been studied in several disciplines.

Manifolds. In the context of smooth Riemannian manifolds, the geodesic distance $d _ { \mathcal { M } } ( \boldsymbol { x } , \boldsymbol { x } ^ { \prime } )$ is not CND for general manifolds [Feragen et al., 2015]. This prohibits the direct use of stationary radial kernels. Extrinsic approaches seek to embed the manifold in Euclidean space and use chordal distances, at the cost of geometric distortion. Intrinsically, heat kernels and spectral Laplace-Beltrami kernels leverage the underlying geometry to guarantee the PSD property, following from the spectral properties of the associated self-adjoint operators [Borovitskiy et al., 2021, 2020]. However, these methods can be computationally demanding and require access to geometric features such as Laplacian eigenfunctions [Lafon and Lee, 2006].

Distributions. Probability distributions endowed with the Wasserstein metric face an analogous challenge: $W _ { 2 }$ is not CND except in specific cases, for example, in one dimension [Bachoc et al., 2020]. Consequently, kernels such as $\exp ( - W _ { 2 } ^ { 2 } ( \mu , \nu ) / \sigma ^ { 2 } )$ are indefinite in general. The sliced Wasserstein distance [Bonneel et al., 2015] improves the situation as it is CND, but can incur both computational cost and reduced accuracy. Other alternatives, such as entropically regularized Sinkhorn distances, partly restore the practical PSD property but are not guaranteed in all cases.

Structured Data. For data structured as sequences (strings) or trees, edit distances like Levenshtein and tree edit distance are common, but not CND. The resulting radial kernels are usually indefinite. To circumvent this, structured kernels that rely on feature vectors of n-grams, subsequences, or subtrees have been proposed, guaranteeing PSD via explicit inner-product constructions [Lodhi et al., 2002, Chen et al., 2023].

Graphs. In graph domains, shortest-path distances are not generally CND, rendering direct radial kernels indefinite. PSD kernels such as diffusion, resistance, or commute-time kernels are constructed from the spectral or stochastic properties of graph Laplacians [Von Luxburg et al., 2008, Nikolentzos et al., 2021, Ralaivola et al., 2005], ensuring mathematical validity.

Landmark/Nyström Methods. A prominent set of alternatives use landmarks or Nyström methods, embedding each point as a feature vector of its distances to selected landmarks [Williams and Seeger, 2001, Drineas et al., 2005, Jayasumana et al., 2013]. Once embedded, a standard Euclidean kernel can be applied, preserving PSDness independently of the original metric’s CND property. These ideas underlie scalable GP approximations like inducing points [Titsias, 2009, Hensman et al., 2013]. However, landmark-based embeddings inevitably involve critical choices about the number and placement of landmarks. Heuristic selections (e.g., k-means, subsampling) may not capture underlying geometric features, reducing predictive performance [El Alaoui and Mahoney, 2015, Musco and Musco, 2017], while learning landmarks involves challenging non-convex optimization. Embedding dimensionality also introduces a tradeoff between expressivity and tractability, complicated by the curse of dimensionality and associated numerical instabilities.

Closely related to this line of work, [Wu et al., 2018] propose D2KE, a general framework for constructing PSD kernels from arbitrary dissimilarities by embedding inputs via distances to a reference set and applying a standard kernel in the resulting feature space. The SLE kernel can be viewed as a specific instantiation of this framework, distinguished by three design choices: (i) the use of compactly supported bump functions rather than raw distances as the embedding map, which induces automatic sparsity and avoids the curse of dimensionality; (ii) the use of all training points as landmarks, eliminating the landmark selection problem; and (iii) a specific focus on the GP setting with theoretical guarantees on conditioning, stability, and universal approximation. We emphasize that these differences are structural rather than incremental. D2KE embeds inputs via raw distances to a small, randomly sampled reference set, producing a dense embedding whose dimension must be kept limited to avoid ill-conditioning and distance concentration — precisely the expressivity/tractability tradeoff described in Section 4.2. The bump construction changes the character of the embedding — sparse, compactly supported, and local — and it is exactly this change that renders the use of all |D| training points as landmarks feasible, eliminates landmark selection, and yields the conditioning and sparsity guarantees of Theorems 4–6, none of which have analogues in the D2KE framework. The ablation studies in Appendix E constitute a controlled empirical comparison against precisely this raw-distance (D2KE-style) embedding.

In summary, while a rich ecosystem of kernels and workarounds has been developed to extend GP regression beyond Euclidean domains, most require nontrivial geometric information, incur high computational cost, or sacrifice expressive fidelity. Our work contributes to this landscape by proposing a computationally efficient, provably PSD kernel that operates directly on the native distance measure of the input space — without projections, slicing, or surrogate metrics (made precise in Proposition 1) — and is suitable for arbitrary distances and learning in abstract non-Euclidean spaces.

## 3 Background

A major challenge in kernel design is ensuring the covariance function remains PSD when non-Euclidean or data-driven distances are used. Kernels built from a distance $d ( x , x ^ { \prime } )$ whose square is not conditionally negative definite (CND) may yield indefinite Gram matrices, compromising both mathematical consistency and numerical stability of GP inference [Berg et al., 1984]. Hilbertian distance metrics guarantee, via Schoenberg’s theorem [Berg et al., 1984], that kernels of the form $k ( x , x ^ { \prime } ) = \exp ( - d ( x , x ^ { \prime } ) ^ { 2 } / \sigma ^ { 2 } )$ are PSD. A distance $d ( x , \bar { x ^ { \prime } } )$ is called Hilbertian if $( X , d )$ can be isometrically embedded into a Hilbert space H, i.e., there exists $\phi : X \to \mathcal { H }$ such that $d ( x , x ^ { \prime } ) =$ $\lVert \phi ( \boldsymbol { x } ) - \phi ( \dot { \boldsymbol { x } } ^ { \prime } ) \rVert _ { \mathcal { H } }$ . Not all common distances satisfy this property: while the Euclidean distance is Hilbertian, the Wasserstein-2 distance in dimensions greater than one is not [Peyré and Cuturi, 2019], and neither are geodesic, $l _ { 1 }$ (in dimension $\geq 2 )$ , and other common distances, precluding their direct use in standard kernel methods. In this work, we propose a kernel construction that guarantees PSD even when the underlying distance is not CND.

## 4 Methodology

Our goal is to construct a kernel that (i) is provably positive semi-definite (PSD) for arbitrary, potentially non-CND distance measures, (ii) operates directly on the native distance d of the input space, without replacing it by a projected, sliced, or otherwise distorted surrogate (made precise in Proposition 1), and (iii) remains computationally tractable. We build toward this construction in three steps: first establishing the core theoretical insight that motivates landmark embeddings, then identifying the practical obstacles of naive implementations, and finally showing how compactly supported bump functions resolve both obstacles simultaneously.

## 4.1 The core insight: geometry-free PSD kernels via embeddings

The fundamental observation underlying our approach is the following. Let X be any input space equipped with an arbitrary distance $d ( \cdot , \cdot )$ , not necessarily CND, and let $\bar { \phi } : \mathcal { X }  \mathbb { R } ^ { m }$ be any mapping into a Euclidean space. If $h : \mathbb { R } ^ { m } \times \bar { \mathbb { R } } ^ { m } \to$ R is a PSD kernel on $\mathbb { R } ^ { m }$ , then the composed kernel

$$
k ( x , x ^ { \prime } ) = h ( \phi ( x ) , \phi ( x ^ { \prime } ) )\tag{1}
$$

is automatically PSD on $x ,$ for any choice of ϕ. This follows directly from the definition of positive semi-definiteness: for any finite set $\{ x _ { 1 } , \dots , x _ { N } \} \subset \mathcal { X }$ and any $\mathbf { c } \in \mathbb { R } ^ { N }$

$$
\sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } c _ { i } c _ { j } k ( x _ { i } , x _ { j } ) = \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } c _ { i } c _ { j } h ( \phi ( x _ { i } ) , \phi ( x _ { j } ) ) \geq 0 ,\tag{2}
$$

since $h$ is PSD on $\mathbb { R } ^ { m }$ and $\{ \phi ( x _ { i } ) \}$ is simply a finite collection of points in $\mathbb { R } ^ { m }$ . Crucially, this guarantee is entirely independent of the geometry of X and of whether $d ( \cdot , \cdot )$ is CND. The geometry of X enters only through ϕ, which can be constructed from d in any way we choose.

Remark 1 (Requirements on $d ) .$ . No properties whatsoever are required of d for the validity of the resulting kernel: Theorem 1 places no conditions on $d ( \cdot , \cdot )$ , since positive semi-definiteness is inherited entirely from the kernel applied in the embedding space. In particular, d need not satisfy the triangle inequality, need not be symmetric, and may be noisy or inconsistent. Only auxiliary results require more of d (Theorems 8, 2, 5).

This insight suggests a general strategy: encode the geometry of $( \mathcal { X } , d )$ into $\phi ,$ and then apply a standard Euclidean kernel h in the embedding space. The remaining question is how to design ϕ so that it faithfully represents the geometry of X while keeping the kernel computationally tractable.

## 4.2 Landmark embeddings and their limitations

A natural choice for $\phi$ is a landmark embedding [Gao et al., 2019, Schölkopf and Smola, 2002]: given a set of reference points $\{ \ell _ { 1 } , \ldots , \ell _ { m } \} \subset \mathcal { X }$ , embed each input by its distances to all landmarks,

$$
\phi ( \boldsymbol { x } ) = [ d ( \boldsymbol { x } , \ell _ { 1 } ) , \boldsymbol { \cdot } \boldsymbol { \cdot } \cdot , d ( \boldsymbol { x } , \ell _ { m } ) ] ^ { \top } \in \mathbb { R } ^ { m } .\tag{3}
$$

This construction is intuitive and general: it uses only the distance $d ,$ makes no assumptions about the geometry of $x ,$ and produces a Euclidean feature vector to which any standard kernel can be applied. However, naive landmark embeddings face two fundamental and coupled difficulties.

Landmark selection. Heuristic selection (e.g., k-means, random subsampling) may miss important geometric or topological features of the data, degrading predictive performance [El Alaoui and Mahoney, 2015, Musco and Musco, 2017], while jointly optimizing landmark positions introduces a highly nonconvex problem with gradients propagating through distance and kernel evaluations, often yielding unstable training dynamics.

Dimensionality. Increasing the number of landmarks m improves geometric fidelity but subjects the embedding to the curse of dimensionality: pairwise Euclidean distances concentrate, kernel matrices become poorly conditioned, and both hyperparameter learning and posterior inference deteriorate [El Alaoui and Mahoney, 2015].

These two problems are fundamentally coupled. Good geometric coverage of $\mathcal { X }$ requires many landmarks, but many landmarks produce high-dimensional, ill-conditioned embeddings. Any principled solution must address both simultaneously.

## 4.3 Bump-function embeddings: resolving both problems at once

We resolve both problems through a single design choice: replacing the raw distance embedding with a compactly supported bump-function embedding, and using all |D| training points as landmarks. Note that |D| therefore serves simultaneously as the dataset cardinality and the embedding dimension — this is not a notational coincidence but a deliberate design choice that eliminates the need to separately specify or optimize the number of landmarks.

Throughout this work, a bump function $b ( \cdot )$ is any function of the distance that is (i) compactly supported on $[ 0 , r )$ for a radius $r > 0 .$ , (ii) smooth on its support, and (iii) strictly positive (and strictly decreasing) on its support. Our specific choice, defined in Eq. (4) below and visualized in Appendix I, is one member of this admissible family; any other function with these properties may be substituted without affecting the guarantees of this paper.

The compact support of the bump functions [Noack and Funke, 2017] is the key mechanism. Because each bump function is exactly zero beyond a radius r from its center, a given input x will activate only the bump functions of nearby landmarks — those within distance r. The embedding vector $\phi ( x )$ is therefore sparse: most of its $| \mathcal D |$ entries are exactly zero, with only a small number of nonzero entries corresponding to the local neighborhood of x. This sparsity has two immediate consequences.

First, it resolves the dimensionality problem. Although the ambient dimension of the embedding is |D|, the effective dimension — the number of nonzero entries — remains small and controlled by the radius $r ,$ independently of |D|. The curse of dimensionality is therefore avoided: pairwise distances in the embedding space remain informative, and kernel matrices remain well-conditioned as |D| grows (Theorems 4 and $\begin{array} { r } { 6 ; } \end{array}$ empirically verified in Appendix G).

Second, it makes landmark selection trivial. Because the embedding is sparse, using all $| \mathcal D |$ training points as landmarks is computationally tractable. This choice guarantees maximal geometric coverage by construction, entirely bypassing the landmark selection problem and its associated nonconvex optimization.

Finally, the embedding is faithful to the native distance in the following precise sense: no projection, slicing, or surrogate metric is ever introduced — the embedding is a function of the exact native distance profile — and the map from local distance profiles to embeddings is injective.

Proposition 1 (Local geometric fidelity). Fix landmarks $\{ x _ { i } \} _ { i = 1 } ^ { | \mathcal { D } | }$ and bump parameters, and let $b ( \cdot ; a , r _ { i } , \beta )$ be strictly decreasing on its support $[ 0 , r _ { i } )$ for every i. Thenfor any $x , x ^ { \prime } \in { \mathcal { X } }$

$$
\phi ( x ) = \phi ( x ^ { \prime } ) \quad \Longleftrightarrow \quad d ( x , x _ { i } ) = d ( x ^ { \prime } , x _ { i } ) \ f o r e \nu e r y \ i \ w i t h \ \operatorname* { m i n } \big ( d ( x , x _ { i } ) , \ d ( x ^ { \prime } , x _ { i } ) \big ) < r _ { i } .
$$

That is, two inputs receive identical embeddings ifand only iftheir distances to all landmarks within reach agree exactly; the embedding discards only far-field information (distances beyond the bump radii), which is a deliberate consequence oflocality, and distorts nothing within it. The proofis given in Appendix A.2. An empirical distortion analysis comparing embedding-space distances to native distances,for SLE versus the sliced Wasserstein surrogate, is provided in Appendix H.

## 4.4 The Sparse Landmark Embedding (SLE) kernel

Let $\{ x _ { 1 } , x _ { 2 } , \dotsc , x _ { | D | } \}$ denote the training data points and let $d ( \cdot , \cdot )$ be a possibly non-CND distance metric, e.g., a geodesic distance on a manifold or the Wasserstein distance between distributions. We define the normalized bump function as

$$
b ( d ; a , r , \beta ) = \left\{ { \begin{array} { l l } { a \exp \left( - { \frac { \displaystyle \beta } { \displaystyle 1 - d ^ { 2 } / r ^ { 2 } } } \right)} + \beta  } & { { \mathrm { i f } } d < r , } \\ { 0 } & { { \mathrm { o t h e r w i s e , } } } \end{array}  \right.\tag{4}
$$

where $a > 0$ is the amplitude, $r > 0$ is the support radius, and $\beta > 0$ is a shape parameter controlling the flatness of the bump. This gives rise to the sparse landmark embedding

$$
\phi ( x ) = \left[ b ( d ( x , x _ { 1 } ) ; a , r , \beta ) , b ( d ( x , x _ { 2 } ) ; a , r , \beta ) , \dots , b ( d ( x , x _ { | \mathcal { D } | } ) ; a , r , \beta ) \right] ^ { \top } \in \mathbb { R } ^ { | \mathcal { D } | } .\tag{5}
$$

Any stationary and non-stationary kernel can now be applied to the embedding space. For example, applying the RBF kernel in the embedding space yields the particular SLE kernel:

$$
k _ { \mathrm { S L E - R B F } } ( \boldsymbol { x } , \boldsymbol { x ^ { \prime } } ) = \sigma ^ { 2 } \exp \left( - \frac { \lVert \phi ( \boldsymbol { x } ) - \phi ( \boldsymbol { x ^ { \prime } } ) \rVert ^ { 2 } } { 2 \ell ^ { 2 } } \right) ,\tag{6}
$$

where $\sigma ^ { 2 } > 0$ is the signal variance and $\ell > 0$ is the length scale. By the argument of Theorem 1, k is immediately PSD on X for any distance metric d. Any other standard kernel that is PSD on $\mathbb { R } ^ { | \mathcal { D } | }$ — including the entire Matérn family — can be used in place of the RBF kernel, yielding a corresponding SLE variant. We write SLE-RBF, SLE (Matérn $\nu = 3 / 2 )$ , etc. to indicate the inner kernel applied to the embedding, and simply SLE when the inner kernel is clear from context; all experiments in Section 5 use Matérn inner kernels, matched to the kernel order of the respective baseline. The SLE kernel maintains the original kernel’s expressivity and universal approximation properties (Theorem 3) and can be reduced in certain conditions to a stationary kernel in the original domain (Theorem 7). The SLE kernel’s differentiability properties are inherited from the kernel applied to the embedding (Theorem 8). See Theorem 9 for some notes on scaling properties of the kernel. Ablation studies demonstrating the effect of the bump function in the embedding are presented in Appendix E. All bump parameters, including the radius r, are hyperparameters learned by marginal-likelihood maximization with data-adaptive bounds (see Appendix D.1); r thus plays the role of, and is selected by the same mechanism as, a length scale in a standard stationary kernel. Sensitivity analyses over r, the bump amplitude, and the choice of inner kernel are reported in Appendix F.

Non-stationary extension. Because the PSD guarantee of Theorem 1 holds for any embedding ϕ, all bump parameters may vary freely as functions of position in X without endangering validity — a property most non-stationary kernel constructions do not enjoy. The natural use case is data whose local complexity varies across the domain (e.g., PDE solution fields with shocks or boundary layers), where a single global radius forces a compromise between resolving fine structure and retaining long-range correlation. We develop this extension, including the PSD proof and the roles of the individual parameter fields, in Appendix B.

## 5 Experiments and results

We evaluate the proposed SLE kernel across three settings of increasing geometric complexity: two GP regression examples on smooth manifolds using geodesic distances, and one on sets of probability distributions using the Wasserstein-2 distance. In all experiments, predictive performance is assessed via root mean square error (RMSE), continuous ranked probability score (CRPS), and prediction interval coverage probability (PICP) at the nominal 95% level, with CRPS and PICP serving as the primary indicators of uncertainty calibration quality. Throughout, lower is better for RMSE and CRPS, and closer to the nominal 0.95 is better for PICP. The manifold benchmark problems (meshes, target functions, and kernel orders) are taken directly from the baseline publications [Borovitskiy et al., 2020, Mostowsky et al., 2025] to preclude benchmark selection in our favor. Sensitivity analyses for the bump radius, amplitude, and inner kernel are reported in Appendix F, empirical condition-number measurements in Appendix G, and wall-clock runtime comparisons in Appendix J.

## 5.1 Dragon manifold

We benchmark the proposed SLE kernel with geodesic distances against the Riemannian Matérn kernel [Borovitskiy et al., 2020], which is defined via stochastic partial differential equations based on Laplace–Beltrami eigenpairs. The Dragon mesh from the referenced work consists of 100,179 vertices used as $\mathrm { G P }$ input points, with output values defined as the sine of the geodesic distance from the dragon’s snout. Following the referenced work, the data are assumed noiseless $( 1 0 ^ { - 5 }$ nugget) with a zero prior mean. Predictive performance was evaluated across training sizes of 50–1000 points, with 30 randomly sampled datasets per size. The Riemannian Matérn kernel was tested with 100, 500, and 1000 eigenpairs. Table 1 reports the mean and standard error of RMSE, CRPS, and PICP for training sizes of 400, 600, and 800 points; the complete results appear in Figure 2 in Appendix C. The SLE kernel consistently outperformed the Riemannian Matérn kernel across all training sizes and eigenpair configurations. More information is included in Appendix D.1.

Table 1: Test RMSE, CRPS, and PICP (95%) for Riemannian Kernel variants and SLE (Matérn) for the Dragon example. Values reported as mean ± standard error. Dashes indicate unavailable values due to instability in the computation of the posterior covariance. Best performing method in bold. “–” is used for repeatedly unstable executions (see Appendix C).
<table><tr><td rowspan="2">Metric</td><td rowspan="2">Model</td><td colspan="3">Training Size</td></tr><tr><td>400</td><td>600</td><td>800</td></tr><tr><td rowspan="4">RMSE (↓)</td><td>Riem. (100 eigenpairs)</td><td> $0 . 1 7 1 \pm 0 . 0 0 6$ </td><td> $0 . 1 4 4 \pm 0 . 0 0 1$ </td><td> $0 . 1 3 7 \pm 0 . 0 0 1$ </td></tr><tr><td>Riem. (500 eigenpairs)</td><td> $0 . 2 4 2 \pm 0 . 0 2 0$ </td><td> $4 . 2 5 7 \pm 0 . 5 8 0$ </td><td> $0 . 2 9 7 \pm 0 . 0 4 9$ </td></tr><tr><td>Riem. (1000 eigenpairs)</td><td> $0 . 1 0 9 \pm 0 . 0 0 2$ </td><td> $0 . 1 2 0 \pm 0 . 0 0 6$ </td><td> $0 . 4 7 3 \pm 0 . 0 7 3$ </td></tr><tr><td>SLE</td><td> $\mathbf { 0 . 0 6 2 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 0 4 2 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 0 3 4 \pm 0 . 0 0 1 }$ </td></tr><tr><td rowspan="4">CRPS (↓)</td><td>Riem. (100 eigenpairs)</td><td> $0 . 1 1 1 \pm 0 . 0 0 1$ </td><td> $0 . 1 0 1 \pm 0 . 0 0 0$ </td><td> $0 . 0 9 8 \pm 0 . 0 0 0$ </td></tr><tr><td>Riem. (500 eigenpairs)</td><td> $0 . 1 0 0 \pm 0 . 0 1 8$ </td><td> $1 . 0 5 7 \pm 0 . 1 4 7$ </td><td> $0 . 0 9 0 \pm 0 . 0 0 6$ </td></tr><tr><td>Riem. (1000 eigenpairs)</td><td></td><td></td><td></td></tr><tr><td>SLE</td><td> $\mathbf { 0 . 0 3 0 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 0 2 0 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 0 1 5 \pm 0 . 0 0 0 }$ </td></tr><tr><td rowspan="4">PICP (95%)  $(  0 . 9 5 )$ </td><td>Riem. (100 eigenpairs)</td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>Riem. (500 eigenpairs)</td><td> $0 . 7 8 0 \pm 0 . 0 0 7$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>Riem. (1000 eigenpairs)</td><td> $\mathbf { 0 . 9 1 2 \pm 0 . 0 0 4 }$ </td><td> $0 . 8 6 9 \pm 0 . 0 0 7$ </td><td> $0 . 6 0 9 \pm 0 . 0 2 5$ </td></tr><tr><td>SLE</td><td> $0 . 9 1 1 \pm 0 . 0 1 8$ </td><td> $\mathbf { 0 . 9 4 4 \pm 0 . 0 0 7 }$ </td><td> $\mathbf { 0 . 9 3 7 \pm 0 . 0 0 5 }$ </td></tr></table>

## 5.2 Teddy Bear manifold

We further benchmark the SLE kernel against the Geometric kernel [Mostowsky et al., 2025], a more recent manifold kernel. The Teddy Bear mesh is reproduced from the referenced work and consists of 1,598 vertices used as GP input points, with output values defined as a random sample from the prior reported therein. Predictive performance was evaluated across training sizes of 50–800 points, with 30 randomly sampled datasets per size. Table 2 reports the mean and standard error of RMSE, CRPS, and PICP for training sizes of 200, 300, and 400 points; the complete results appear in Figure 3 in Appendix C. Both kernels achieve comparable RMSE across all training sizes, indicating similar posterior mean accuracy. However, the two kernels differ substantially in uncertainty quantification. The SLE kernel consistently achieves lower CRPS and maintains a PICP near the nominal 95% level across all training sizes, indicating well-calibrated predictive uncertainty. The Geometric kernel produced PICP values between 8% and 27%, reflecting severe overconfidence in which the 95% prediction intervals capture only a small fraction of the true test values. These results were obtained using the reference implementation of [Mostowsky et al., 2025] without modification, confirming that the result reflects the behavior of the published method rather than an implementation artifact. We stress that, in contrast to the spectral-truncation instability observed on the Dragon manifold, the Geometric kernel is numerically stable in this experiment and matches SLE in RMSE; the reported

PICP therefore reflects the published method’s calibration behavior in its stable operating regime, not an instability artifact. More information is discussed in Appendix D.2.

Table 2: Test RMSE, CRPS, and PICP (95%) for Geometric Kernel and SLE models for the Teddy Bear example. Values reported as mean ± standard error. Best performing method in bold.
<table><tr><td rowspan="2">Metric</td><td rowspan="2">Model</td><td colspan="3">Training Size</td></tr><tr><td>200</td><td>300</td><td>400</td></tr><tr><td rowspan="2">RMSE (↓)</td><td>Geometric Kernel</td><td> $4 2 . 1 4 7 \pm 0 . 3 1 6$ </td><td> $3 6 . 9 6 5 \pm 0 . 3 2 7$ </td><td> $3 4 . 0 9 4 \pm 0 . 3 3 8$ </td></tr><tr><td>SLE</td><td> $\mathbf { 4 1 . 8 2 5 \pm 0 . 4 8 5 }$ </td><td> ${ \bf 3 6 . 7 0 2 \pm 0 . 5 5 8 }$ </td><td> $\mathbf { 3 4 . 9 7 6 \pm 0 . 5 3 1 }$ </td></tr><tr><td rowspan="2">CRPS (↓)</td><td>Geometric Kernel SLE</td><td> $2 3 . 7 8 1 \pm 0 . 2 6 2$ </td><td> $1 8 . 8 0 5 \pm 0 . 1 7 8$ </td><td> $1 6 . 1 4 2 \pm 0 . 1 5 3$ </td></tr><tr><td></td><td> ${ \bf 1 8 . 1 3 4 \pm 0 . 1 9 7 }$ </td><td> $\mathbf { 1 4 . 6 7 2 \pm 0 . 1 8 3 }$ </td><td> $\mathbf { 1 3 . 0 4 1 \pm 0 . 1 3 0 }$ </td></tr><tr><td>PICP (95%)</td><td>Geometric Kernel</td><td> $0 . 1 0 0 \pm 0 . 0 0 3$ </td><td> $0 . 1 1 9 \pm 0 . 0 0 3$ </td><td> $0 . 1 3 8 \pm 0 . 0 0 3$ </td></tr><tr><td>(→ 0.95)</td><td>SLE</td><td> $\mathbf { 0 . 9 3 2 \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 9 4 1 \pm 0 . 0 0 3 }$ </td><td> $\mathbf { 0 . 9 2 9 \pm 0 . 0 0 3 }$ </td></tr></table>

## 5.3 X-ray scattering data disguised as distributions

We evaluate the SLE kernel on a dataset of 500 synthetic small-angle X-ray scattering (SAXS) images designed to mimic real-world SAXS patterns from oriented soft-matter thin films, such as block copolymer and liquid crystal systems, measured at synchrotron beamlines (Appendix D.3 Figure 4). Each 64 × 64 image represents the 2D reciprocal-space intensity pattern of a multi-domain lamellar sample, consisting of a fundamental scattering arc at wavevector $q ^ { * }$ and a second harmonic at $2 q ^ { * }$ The key structural parameter is the inter-harmonic coupling disorder $\sigma _ { \mathrm { c o u p } } ,$ which controls the degree to which the two arcs within each domain remain collinear. As $\sigma _ { \mathrm { c o u p } }$ increases, the harmonic arcs decohere, reducing the effective Young’s modulus of the material along the measurement axis — the quantity used as the $\mathrm { G P }$ output y. The SLE kernel was applied with the full Wasserstein distance and benchmarked against the $\nu = 3 / 2$ Matérn kernel with the sliced Wasserstein distance across training sizes of 150, 200, and 250 images, with a fixed test set of 100 held-out images and 30 random dataset draws per configuration. As reported in Table 3, the SLE kernel achieved comparable or slightly better RMSE and CRPS, with small differences across all training sizes. The key finding is not superiority but competitiveness: the SLE kernel matches a domain-adapted baseline that uses the sliced Wasserstein approximation, while operating directly on the true $\dot { W _ { 2 } }$ distance without any geometric preprocessing. To decompose the contribution of the bump embedding from that of the exact $W _ { 2 }$ distance, we additionally evaluate the SLE kernel applied on top of the sliced Wasserstein distance (SLE – Sliced Wass in Table 3). The SLE–Sliced Wasserstein variant achieves calibration comparable to the other two methods, with PICP within a few points of nominal at every training size, indicating that the calibration benefit of the bump construction persists regardless of the underlying distance. However, RMSE and CRPS for SLE–Sliced Wasserstein are higher than for both SLE–Wass and Matérn–Sliced Wass across all training sizes, suggesting that discarding far-field information via the bump radius compounds with the geometric distortion introduced by slicing. This indicates that the accuracy gains of the SLE kernel derive primarily from operating on exact distances rather than from the bump embedding alone, while the calibration gains are attributable to the sparse, local structure of the embedding itself, independent of the base distance to which it is applied.

## 6 Discussion and conclusion

We propose the Sparse Landmark Embedding (SLE) kernel, a general-purpose kernel for Gaussian process regression that operates on arbitrary distance measures, including those that are not conditionally negative definite. By embedding inputs via compactly supported bump functions centered at all training points, the SLE kernel is provably PSD for any input geometry, avoids the landmark selection problem, and mitigates the curse of dimensionality through automatic sparsity. Theoretical analysis establishes guarantees of PSD, sparsity scaling, stability, universal approximation, and connections to standard stationary kernels in the limit. We evaluated the SLE kernel against three domain-specific baselines: the Riemannian Matérn kernel [Borovitskiy et al., 2020] and Geometric Matérn kernel [Mostowsky et al., 2025] for GP regression on smooth manifolds, and the sliced Wasserstein Matérn kernel [Bachoc et al., 2020] for GP regression over sets of distributions. Experiments were conducted on the Dragon and Teddy Bear Manifolds and a synthetic SAXS dataset, covering a range of geometric complexities and dataset sizes. Across all settings, the SLE kernel achieved predictive accuracy — as measured by RMSE — that was comparable to or better than the domain-specific baselines for all considered training dataset sizes (Tables 1, 2, and 3). The more striking finding concerns uncertainty quantification as measured by CRPS and Probability Coverage (PICP); the SLE kernel consistently produced better-calibrated predictive uncertainties than both baselines across all experimental settings. We attribute this to two structural properties: the kernel operates on distance measures native to the input space — geodesic distances on manifolds, exact Wasserstein distances between distributions — capturing true geometry rather than a distorted proxy (Proposition 1); and the sparse embedding produces well-conditioned kernel matrices (Theorem 4, verified empirically in Appendix G), avoiding the variance underestimation that can arise from ill-conditioned Gram matrices.

Table 3: Test RMSE, CRPS, and PICP (95%) for SLE (Matérn) and Matérn models for the SAXS data. Values are reported as mean ± standard error of 30 random trials. Best-performing method in bold.
<table><tr><td rowspan="2">Metric</td><td rowspan="2">Model</td><td colspan="3">Training Size</td></tr><tr><td>150</td><td>200</td><td>250</td></tr><tr><td rowspan="3">RMSE (↓)</td><td>Matérn - Sliced Wass</td><td> $0 . 4 8 9 \pm 0 . 0 1 0$ </td><td> $0 . 4 4 5 \pm 0 . 0 1 1$ </td><td> $0 . 3 5 8 \pm 0 . 0 0 9$ </td></tr><tr><td>SLE - Wass</td><td> $\mathbf { 0 . 4 6 7 \pm 0 . 0 1 0 }$ </td><td> $\mathbf { 0 . 3 9 6 \pm 0 . 0 0 9 }$ </td><td> $\mathbf { 0 . 3 3 4 \pm 0 . 0 0 7 }$ </td></tr><tr><td>SLE - Sliced Wass</td><td> $0 . 6 7 2 \pm 0 . 0 2 3$ </td><td> $0 . 5 7 5 \pm 0 . 0 1 4$ </td><td> $0 . 4 9 7 \pm 0 . 0 1 1$ </td></tr><tr><td rowspan="3">CRPS (↓)</td><td>Matérn - Sliced Wass</td><td> $0 . 2 2 1 \pm 0 . 0 0 4$ </td><td> $0 . 1 8 8 \pm 0 . 0 0 4$ </td><td> $0 . 1 4 8 \pm 0 . 0 0 5$ </td></tr><tr><td>SLE - Wass</td><td> $\mathbf { 0 . 2 1 0 \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 1 7 4 \pm 0 . 0 0 3 }$ </td><td> $\mathbf { 0 . 1 4 6 \pm 0 . 0 0 3 }$ </td></tr><tr><td>SLE - Sliced Wass</td><td> $0 . 2 8 8 \pm 0 . 0 0 6$ </td><td> $0 . 2 5 6 \pm 0 . 0 0 6$ </td><td> $0 . 2 2 0 \pm 0 . 0 0 6$ </td></tr><tr><td rowspan="3">PICP (95%) (→ 0.95)</td><td>Matérn - Sliced Wass</td><td> $\mathbf { 0 . 9 4 0 \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 9 4 5 \pm 0 . 0 0 4 }$ </td><td> $0 . 9 6 3 \pm 0 . 0 0 4$ </td></tr><tr><td> $\mathrm { S L E - W a s s }$ </td><td> $0 . 9 3 4 \pm 0 . 0 0 6$ </td><td> $\mathbf { 0 . 9 4 5 \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 9 5 9 \pm 0 . 0 0 4 }$ </td></tr><tr><td>SLE - Sliced Wass</td><td> $0 . 9 2 2 \pm 0 . 0 0 4$ </td><td> $0 . 9 3 3 \pm 0 . 0 0 3$ </td><td> $0 . 9 3 7 \pm 0 . 0 0 4$ </td></tr></table>

We emphasize that the three experiments probe three distinct baseline regimes, and the calibration advantage of SLE is not attributable to baseline instability. On the Dragon manifold, the Riemannian Matérn kernel requires explicit access to the Laplace–Beltrami eigenpairs of the manifold, which are expensive to compute and introduce approximation error that grows with geometric complexity. This instability becomes particularly pronounced when the training dataset size approaches the number of eigenpairs used, causing a sharp deterioration in predictive performance. This is a structural property of the published spectral-truncation construction — the truncation level is a parameter the method itself requires — and away from the instability (e.g., the 1000-eigenpair variant at 400 training points) the baseline behaves well yet SLE still outperforms it on all three metrics; having no truncation parameter to mis-set is precisely SLE’s practical advantage here. The SLE kernel, by contrast, requires only pairwise geodesic distances and exhibits stable, monotonically improving performance as training size increases. On the Teddy Bear manifold, the Geometric kernel is numerically stable and matches SLE in point accuracy; its severe overconfidence (PICP of 8–27%) therefore reflects the published method’s calibration behavior in its stable operating regime, obtained with the authors reference implementation. On the distribution-valued SAXS dataset, the sliced Wasserstein Matérn kernel replaces the true Wasserstein-2 distance with its sliced approximation in order to recover the CND property. The geometric distortion introduced by slicing can degrade both predictive accuracy and uncertainty quantification, particularly when the distributions are high-dimensional or multimodal [Nadjahi et al., 2019]. The SLE kernel operates directly on the true $W _ { 2 }$ distance, avoiding this distortion entirely. Here the baseline behaves entirely well, and we claim competitiveness rather than superiority.

Limitations. The most significant limitation is the dependence on pairwise distance computations between all test points and all |D| training landmarks. While the sparse embedding ensures that kernel evaluations are cheap (Theorem 9), forming the full distance matrix still requires $\bar { \boldsymbol { O } } ( N \cdot | \boldsymbol { D } | )$ distance computations. Approximate nearest neighbor methods or hierarchical distance approximations — e.g., cover trees or vantage-point trees, which require only the distance function and exploit the fact that landmarks beyond radius r contribute nothing — could mitigate this cost; we note this cost is shared by all distance-based competitors in our experiments (Appendix J). A second limitation concerns the choice of bump radii $r ( x _ { i } )$ . Although the sparsity and PSD properties hold for any choice of radii, predictive performance is sensitive to their values, and principled data-driven selection of r(·) remains challenging; in practice, we learn a global r by marginal-likelihood maximization with data-adaptive bounds, and the sensitivity study in Appendix F indicates a broad well-performing region around the likelihood-selected value. A third limitation follows from the deliberately local design: information about distance relationships beyond the bump radii is discarded by construction (Proposition 1 guarantees fidelity of local distance profiles only), so tasks driven by genuinely global geometric structure may require larger radii, trading sparsity for reach. A final cautionary statement: There are too many types of inputs and distance metrics to establish broad practical generality of the proposed method in this paper. What we aim to do is provide a tool that might help in cases where natural CND distances are unavailable.

Author Contributions. M.M.N.: Ideation, Kernel derivation, Performance comparisons, Software development, Manuscript; M.D.R.: Ideation, Kernel derivation, Manuscript; M.B.A.: Kernel derivation, Performance comparisons, Data curation, Test executions, Manuscript.

Acknowledgments This work was supported by

• The Center for Advanced Mathematics for Energy Research Applications (CAMERA), which is jointly funded by the Advanced Scientific Computing Research (ASCR) and Basic Energy Sciences (BES) within the Department of Energy’s Office of Science, under Contract No. DE-AC02-05CH11231.

• The U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research’s Applied Mathematics Competitive Portfolios program under Contract No. AC02- 05CH11231.

• The U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research’s Applied Mathematics program under Contract No. DE-AC02-05CH11231 at Lawrence Berkeley National Laboratory.

Data and Code Availability Statement. We will make all code and data available upon publication.

Ethics Statement. The authors declare no conflicts of interest.

## References

François Bachoc, Alexandra Suvorikova, David Ginsbourger, Jean-Michel Loubes, and Vladimir Spokoiny. Gaussian processes with multidimensional distribution inputs via optimal transport and Hilbertian embedding. Electronic Journal ofStatistics, 14(2):2742–2772, 2020.

Christian Berg, Jens Peter Reus Christensen, and Paul Ressel. Harmonic Analysis on Semigroups: Theory ofPositive Definite and Related Functions, volume 100. Springer, 1984. ISBN 9780387136418.

Nicolas Bonneel, Julien Rabin, Gabriel Peyré, and Hanspeter Pfister. Sliced and Radon Wasserstein barycenters of measures. Journal ofMathematical Imaging and Vision, 51(1):22–45, 2015.

Viacheslav Borovitskiy, Alexander Terenin, Peter Mostowsky, and Marc Peter Deisenroth. Matérn Gaussian processes on Riemannian manifolds. Advances in Neural Information Processing Systems, 33:12426–12437, 2020.

Viacheslav Borovitskiy, Iskander Azangulov, Alexander Terenin, Peter Mostowsky, Marc Deisenroth, and Nicolas Durrande. Matérn Gaussian processes on graphs. In Proceedings of the 24th International Conference on Artificial Intelligence and Statistics (AISTATS), pages 2593–2601. Proceedings of Machine Learning Research (PMLR), 2021.

Maximillian Chen, Caitlyn Chen, Xiao Yu, and Zhou Yu. FastKASSIM: A fast tree kernel-based syntactic similarity metric. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics (EACL), pages 211–231, 2023.

Petros Drineas, Michael W. Mahoney, and Nello Cristianini. On the Nyström method for approximating a Gram matrix for improved kernel-based learning. Journal of Machine Learning Research, 6: 2153–2175, 2005.

Ahmed El Alaoui and Michael W. Mahoney. Fast randomized kernel ridge regression with statistical guarantees. In Advances in Neural Information Processing Systems, volume 28, 2015.

Aasa Feragen, François Lauze, and Søren Hauberg. Geodesic exponential kernels: When curvature and linearity conflict. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3032–3042, 2015.

Tingran Gao, Shahar Z. Kovalsky, and Ingrid Daubechies. Gaussian process landmarking on manifolds. SIAM Journal on Mathematics ofData Science, 1(1):208–236, 2019.

Bernard Haasdonk and Hans Burkhardt. Invariant kernel functions for pattern analysis and machine learning. Machine Learning, 68(1):35–61, 2007.

James Hensman, Nicolò Fusi, and Neil D. Lawrence. Gaussian processes for big data. In Proceedings ofthe 29th Conference on Uncertainty in Artificial Intelligence (UAI), pages 282–290, 2013.

Sadeep Jayasumana, Richard Hartley, Mathieu Salzmann, Hongdong Li, and Mehrtash Harandi. Kernel methods on the Riemannian manifold of symmetric positive definite matrices. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 73–80, 2013.

Stéphane Lafon and Ann B. Lee. Diffusion maps and coarse-graining: A unified framework for dimensionality reduction, graph partitioning, and data set parameterization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 28(9):1393–1403, 2006.

Huma Lodhi, Craig Saunders, John Shawe-Taylor, Nello Cristianini, and Chris Watkins. Text classification using string kernels. Journal of Machine Learning Research, 2(Feb):419–444, 2002.

Charles A. Micchelli, Yuesheng Xu, and Haizhang Zhang. Universal kernels. Journal ofMachine Learning Research, 7:2651–2667, 2006.

Peter Mostowsky, Vincent Dutordoir, Iskander Azangulov, Noémie Jaquier, Michael John Hutchinson, Aditya Ravuri, Leonel Rozo, Alexander Terenin, and Viacheslav Borovitskiy. The GeometricKernels package: Heat and Matérn kernels for geometric learning on manifolds, meshes, and graphs. Journal ofMachine Learning Research, 26(276):1–14, 2025.

Cameron Musco and Christopher Musco. Recursive sampling for the Nyström method. Advances in Neural Information Processing Systems, 30, 2017.

Kimia Nadjahi, Alain Durmus, Umut Simsekli, and Roland Badeau. Asymptotic guarantees for learning generative models with the sliced-Wasserstein distance. Advances in Neural Information Processing Systems, 32, 2019.

Giannis Nikolentzos, Giannis Siglidis, and Michalis Vazirgiannis. Graph kernels: A survey. Journal ofArtificial Intelligence Research, 72:943–1027, 2021.

Marcus M. Noack and Simon W. Funke. Hybrid genetic deflated Newton method for global optimisation. Journal of Computational and Applied Mathematics, 325:97–112, 2017.

Christopher J. Paciorek and Mark J. Schervish. Nonstationary covariance functions for Gaussian process regression. Advances in Neural Information Processing Systems, 16, 2003.

Gabriel Peyré and Marco Cuturi. Computational optimal transport: With applications to data science. Foundations and Trends in Machine Learning, 11(5–6):355–607, 2019.

Liva Ralaivola, Sanjay J. Swamidass, Hiroto Saigo, and Pierre Baldi. Graph kernels for chemical informatics. Neural Networks, 18(8):1093–1110, 2005.

Carl Edward Rasmussen and Christopher K. I. Williams. Gaussian Processes for Machine Learning. MIT Press, Cambridge, MA, 2006.

Bernhard Schölkopf and Alexander J. Smola. Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond. MIT Press, Cambridge, MA, 2002.

Ingo Steinwart. On the influence of the kernel on the consistency of support vector machines. Journal ofMachine Learning Research, 2:67–93, 2001.

Michalis Titsias. Variational learning of inducing variables in sparse Gaussian processes. In Proceedings of the 12th International Conference on Artificial Intelligence and Statistics (AISTATS), pages 567–574. Proceedings of Machine Learning Research (PMLR), 2009.

Ulrike Von Luxburg, Mikhail Belkin, and Olivier Bousquet. Consistency of spectral clustering. The Annals ofStatistics, 36(2):555–586, 2008.

Christopher K. I. Williams and Matthias Seeger. Using the Nyström method to speed up kernel machines. In Advances in Neural Information Processing Systems, volume 13, 2001.

Lingfei Wu, Ian En-Hsu Yen, Fangli Xu, Pradeep Ravikumar, and Michael Witbrock. D2KE: From distance to kernel and embedding. arXiv preprint arXiv:1802.04956, 2018.

## A Theoretical Properties and Proofs

In this section, we present the properties of the proposed kernel, focusing on positive semi-definiteness, automatic dimensionality reduction, expressivity, stability, high-dimensional effects, connection to stationary kernels, and smoothness.

## A.1 Positive Semi-Definiteness (PSD) of the Kernel

Theorem 1. Let

$$
k ( x , x ^ { \prime } ) = \sigma ^ { 2 } \exp { \left( - \frac { \| \varphi ( x ) - \varphi ( x ^ { \prime } ) \| ^ { 2 } } { 2 \ell ^ { 2 } } \right) }
$$

where $\varphi : \mathcal { X }  \mathbb { R } ^ { m }$ is any mapping, and $\| \cdot \|$ is the standard Euclidean norm. Then k is a positive semi-definite (PSD) kernel on $\bar { \mathcal { X } } .$

Proof. Let $\{ x _ { 1 } , \dotsc , x _ { N } \} \subset { \mathcal { X } }$ be any finite collection of points, and let $z ^ { ( i ) } = \varphi ( x _ { i } ) \in \mathbb { R } ^ { m }$ for $i = 1 , \ldots , N$ . Consider the Gram matrix with entries

$$
K _ { i j } = k ( x _ { i } , x _ { j } ) = \sigma ^ { 2 } \exp \left( - \frac { \| z ^ { ( i ) } - z ^ { ( j ) } \| ^ { 2 } } { 2 \ell ^ { 2 } } \right) .
$$

The function $\begin{array} { r } { ( z , z ^ { \prime } ) \mapsto \exp \left( - \frac { \| z - z ^ { \prime } \| ^ { 2 } } { 2 \ell ^ { 2 } } \right) } \end{array}$ defines a positive semi-definite kernel on $\mathbb { R } ^ { m }$ , as follows from Bochner’s theorem since it is the Fourier transform (characteristic function) of a finite Gaussian measure. Therefore, for any real vector $\mathbf { c } \in \mathbb { R } ^ { N }$

$$
\sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } c _ { i } c _ { j } K _ { i j } \ge 0 .
$$

Hence, k is a positive semi-definite kernel for any choice of mapping $\varphi .$

## A.2 Local Geometric Fidelity: Proof of Proposition 1

ProofofProposition 1. Fix i and write $b _ { i } ( \cdot ) = b ( \cdot ; a , r _ { i } , \beta )$ , which by assumption is strictly decreasing — hence injective — on its support $[ 0 , r _ { i } )$ , and identically zero on $[ r _ { i } , \infty )$ . Consider the i-th embedding coordinates $\phi _ { i } ( x ) = b _ { i } ( d ( x , \dot { x _ { i } } ) )$ and $\phi _ { i } ( x ^ { \prime } ) = b _ { i } ( \dot { d } ( x ^ { \prime } , x _ { i } ) )$

$( \Leftarrow )$ If min $\begin{array} { r l r } { \mathfrak { \ i } ( d ( x , x _ { i } ) , d ( x ^ { \prime } , x _ { i } ) ) } & { { } \geq } & { r _ { i } , } \end{array}$ , then both coordinates are zero and agree. If min $( d ( x , x _ { i } ) , d ( x ^ { \prime } , x _ { i } ) ) \ < \ r _ { i }$ and $d ( x , x _ { i } ) = d ( x ^ { \prime } , x _ { i } )$ , the coordinates agree trivially. Hence the stated distance condition implies $\phi ( \boldsymbol { x } ) = \phi ( \boldsymbol { x } ^ { \prime } )$

$( \Rightarrow )$ Suppose $\phi _ { i } ( x ) ~ = ~ \phi _ { i } ( x ^ { \prime } )$ and min $( d ( x , x _ { i } ) , d ( x ^ { \prime } , x _ { i } ) ) ~ < ~ r _ { i } ;$ without loss of generality $d ( x , x _ { i } ) < r _ { i } ,$ so ϕ $\mathfrak { \xi } _ { i } ( x ) = b _ { i } ( d ( x , x _ { i } ) ) > 0$ by strict positivity on the support. Then $\phi _ { i } ( \bar { x ^ { \prime } } ) > 0$ as well, forcing $d ( x ^ { \prime } , x _ { i } ) < r _ { i } .$ , and injectivity of $b _ { i }$ on $[ 0 , r _ { i } )$ yields $d ( x , x _ { i } ) = d ( x ^ { \prime } , x _ { i } )$ . Applying this to every coordinate i gives the claim.

Consequently, the embedding is an injective function of the local distance profile $\{ d ( x , x _ { i } )$ $d ( x , x _ { i } ) < r _ { i } \}$ : within the reach of the bumps, the exact native distances are encoded without projection or surrogate, and only far-field information (distances beyond the radii) is discarded.

## A.3 Automatic Dimensionality Reduction and Sparsity

Theorem 2. Let X be any metric space equipped with a (not necessarily CND) distance $d ( \cdot , \cdot )$ Consider a collection of m landmark points $L = \{ x _ { 1 } , \ldots , x _ { m } \} \subset { \mathcal { X } }$ , and for each i let $r _ { i } > 0$ denote the radius ofthe compactly supported bumpfunction centered at $x _ { i } .$ . For any $x \in \mathcal { X } ,$ , define the embedding

$$
\varphi ( x ) = \left[ b ( d ( x , x _ { 1 } ) ; a _ { 1 } , r _ { 1 } , \beta _ { 1 } ) , \mathrm { ~ } . . . , b ( d ( x , x _ { m } ) ; a _ { m } , r _ { m } , \beta _ { m } ) \right] ^ { \top } \in \mathbb { R } ^ { m } ,
$$

where $b ( d ; a , r , \beta )$ is supported on $[ 0 , r )$ (that $i s , b ( d ; a , r , \beta ) = 0 i f d \geq r )$

Then at most $| \{ i : d ( x , x _ { i } ) < r _ { i } \}$ elements $o f \varphi ( x )$ are nonzero; all others are exactly zero. In particular, ifthe radii $\{ r _ { i } \}$ are small compared to the spacing ofthe points in $L ,$ and x is randomly sampled according to some probability measure $\mu$ on $x ,$ , then the expected number ofnonzero entries is

$$
\mathbb { E } _ { x \sim \mu } \left[ \| \varphi ( x ) \| _ { 0 } \right] = \sum _ { i = 1 } ^ { m } \mathbb { P } _ { x \sim \mu } \left[ d ( x , x _ { i } ) < r _ { i } \right] .
$$

If all radii $r _ { i } = r$ and $\mu$ is sufficiently distributed across the domain then

$$
\begin{array} { r } { \mathbb { E } _ { x \sim \mu } \left[ \| \varphi ( x ) \| _ { 0 } \right] = m \cdot p _ { r } \quad w h e r e \quad p _ { r } : = \mathbb { P } _ { x \sim \mu } [ d ( x , x _ { i } ) < r ] \ll 1 , } \end{array}
$$

so the embedding is sparse: as m increases and r is fixed, this expected count can be kept small relative to m.

Proof. The bump function $b ( d ( x , x _ { i } ) ; a _ { i } , r _ { i } , \beta _ { i } )$ is nonzero if and only if $d ( x , x _ { i } ) < r _ { i } ;$ otherwise, it is zero by definition. Thus, in the embedding vector $\varphi ( x )$ , the i-th coordinate is zero unless x lies within radius $r _ { i }$ of landmark $x _ { i }$ . The set of indices with nonzero entries is thus $S _ { x } = \{ i : d ( x , x _ { i } ) <$ $r _ { i } \}$ , so $\| \varphi ( x ) \| _ { 0 } = | S _ { x } |$ . Averaging over x drawn from $\mu$ yields the expected sparsity as stated. If, for each $i , p _ { i } : = \mathbb { P } _ { x \sim \mu } [ d ( x , x _ { i } ) < r _ { i } ]$ is small $( \mathrm { e . g . }$ ., because $r _ { i }$ is much less than the typical inter-landmark spacing), then the expected number of nonzero coordinates is $\sum _ { i = 1 } ^ { m } p _ { i }$ , which can be made much less than m by suitable choice of $r _ { i }$ . In the case where all radii are equal, this simplifies as above. □

## A.4 Expressivity and Universal Approximation

Theorem 3. Let X be a compact metric space and let $C ( \mathcal X )$ denote the space ofcontinuousfunctions on $\mathcal { X } .$ . Let $k ( x , x ^ { \prime } )$ be the kernel defined as

$$
k ( x , x ^ { \prime } ) = \sigma ^ { 2 } \exp \left( - \frac { \| \varphi ( x ) - \varphi ( x ^ { \prime } ) \| ^ { 2 } } { 2 \ell ^ { 2 } } \right) ,
$$

where the feature map $\varphi : \mathcal { X }  \mathbb { R } ^ { m }$ is constructed from compactly supported, smooth bump functions centered at locations $\{ x _ { i } \}$ with tunable radii $\{ r _ { i } \}$ and amplitudes $\{ a _ { i } \}$ , and assume that $\varphi$ is continuous and injective on $\mathcal { X }$ (guaranteed whenever the local distance profiles separate the points $o f { \mathcal { X } } ;$ cf. Proposition 1). Then the associated reproducing kernel Hilbert space (RKHS) is dense in $C ( \mathcal X )$ , i.e., for every $f \in C ( \mathcal { X } )$ and every $\epsilon > 0 ,$ there exists a function g in the RKHS such that

$$
\operatorname* { s u p } _ { x \in \mathcal { X } } | f ( x ) - g ( x ) | < \epsilon .
$$

Proof. Since $\mathcal { X }$ is compact and $\varphi$ is continuous and injective, $\varphi$ is a homeomorphism onto its image $Z = \varphi ( \mathcal { X } ) \subset \mathbb { R } ^ { m }$ , which is compact. The Gaussian kernel $\begin{array} { r } { k _ { \mathrm { R B F } } ( z , z ^ { \prime } ) = \exp \left( - \frac { \| z - z ^ { \prime } \| ^ { 2 } } { 2 \ell ^ { 2 } } \right) } \end{array}$ is universal on compact subsets of $\mathbb { R } ^ { m }$ [Micchelli et al., 2006, Steinwart, 2001], so its RKHS $\mathcal { H } _ { \mathrm { R B F } }$ is dense in $\dot { C } ( Z )$ . Given $f \in C \bar { ( \mathcal { X } ) }$ and $\epsilon > 0$ , the function $f \circ \varphi ^ { - 1 }$ is continuous on $Z ;$ choose $\tilde { g } \in \mathcal { H } _ { \mathrm { R B F } }$ with $\begin{array} { r } { \operatorname* { s u p } _ { z \in Z } | f ( \varphi ^ { - 1 } ( z ) ) - \tilde { g } ( z ) | < \epsilon } \end{array}$ . The RKHS of the composed kernel $k ( x , x ^ { \prime } ) = \sigma ^ { 2 } k _ { \mathrm { R B F } } ( \varphi ( x ) , \varphi ( x ^ { \prime } ) )$ consists exactly of functions of the form $\tilde { h } \circ \varphi$ with $\tilde { h } \in \mathcal { H } _ { \mathrm { R B F } }$ (restricted to Z) [Schölkopf and Smola, 2002, Ch. 4], so $g : = \tilde { g } \circ \varphi$ lies in the RKHS of k and

$$
\operatorname* { s u p } _ { x \in \mathcal { X } } | f ( x ) - g ( x ) | = \operatorname* { s u p } _ { z \in Z } \left| f ( \varphi ^ { - 1 } ( z ) ) - { \tilde { g } } ( z ) \right| < \epsilon .
$$

## A.5 Stability and Conditioning of the Kernel Matrix

Theorem 4. Let X be a metric space, and let $L = \{ x _ { 1 } , \ldots , x _ { m } \} \subset \mathcal { X }$ be a set of m landmarks, each with a compactly supported bump function embedding as in the previous theorem:

$$
\varphi ( x ) = [ b ( d ( x , x _ { 1 } ) ) , b ( d ( x , x _ { 2 } ) ) , \ldots , b ( d ( x , x _ { m } ) ) ] ^ { \top } \in \mathbb { R } ^ { m } ,
$$

where $b ( d )$ is nonzero $i f$ and only $i f d < r$ for somefixed support radius r. Consider the kernel

$$
k ( x , x ^ { \prime } ) = \sigma ^ { 2 } \exp \left( - \frac { \| \varphi ( x ) - \varphi ( x ^ { \prime } ) \| ^ { 2 } } { 2 \ell ^ { 2 } } \right) .
$$

Let $\{ z _ { 1 } , \dots , z _ { N } \}$ be a dataset, and let $K \in \mathbb { R } ^ { N \times N }$ be the Gram matrix with entries $K _ { i j } = k ( z _ { i } , z _ { j } )$ Assumption (A1) (bounded overlap). Assume the expected number ofoverlapping nonzero entries in $\varphi ( z _ { i } )$ and $\varphi ( z _ { j } )$ is bounded above by $s \ll m ,$ , independent ofm as m increases.

Then, under Assumption (A1), as m grows, the Gram matrix K remains well-conditioned: its condition number is bounded above by a constant depending on the maximal overlap s and the kernel parameters, but not on m.

Proof. [Structural sketch] As shown in the sparsity theorem, for each datapoint $z _ { i } ,$ , the embedding $\varphi ( z _ { i } )$ has at most s nonzero entries. Further, for most pairs $( z _ { i } , z _ { j } )$ , the supports of $\varphi ( z _ { i } )$ and $\varphi ( z _ { j } )$ do not overlap, so $\| \varphi ( z _ { i } ) - \varphi ( z _ { j } ) \| ^ { 2 } = \| \varphi ( z _ { i } ) \| ^ { 2 } + \| \varphi ( z _ { j } ) \| ^ { 2 }$

Thus, most off-diagonal entries of K take the form

$$
k ( z _ { i } , z _ { j } ) = \sigma ^ { 2 } \exp \left( - \frac { \| \varphi ( z _ { i } ) \| ^ { 2 } + \| \varphi ( z _ { j } ) \| ^ { 2 } } { 2 \ell ^ { 2 } } \right) = k _ { 0 } ( z _ { i } ) k _ { 0 } ( z _ { j } ) ,
$$

where $\begin{array} { r } { k _ { 0 } ( z ) = \exp \left( - \frac { \| \varphi ( z ) \| ^ { 2 } } { 2 \ell ^ { 2 } } \right) } \end{array}$ . This structure yields a Gram matrix that is block-diagonal (or close to it) with small off-diagonal entries except within overlapping support, where blocks of size $s \times s$ may appear.

Block-diagonal or banded matrices with small block size always have condition numbers bounded by a constant (given by the maximal block condition number) and are thus resistant to the ill-conditioning that arises when all entries are dense and m is large, as seen in standard high-dimensional RBF kernels.

Therefore, for any m, the condition number of K is controlled by the overlap s and the kernel parameters $( \sigma ^ { 2 } , \ell )$ , but is not adversely affected by increasing m. □

Remark 2 (Scope of Theorem 4). Assumption (A1) need not hold uniformly across datasets $- e . g .$ under strongly clustered sampling with radii large relative to cluster diameters — and the argument above is structural rather thanfully quantitative. Appendix G therefore verifies the predicted behavior empirically, reporting Gram-matrix condition numbers as a function of training size for the SLE embedding and the raw-distance embedding at their respective likelihood-optimized hyperparameters.

## A.6 Sparsity Scaling of Embedding with Number of Landmarks

Theorem 5. Let X be a metric space endowed with distance $d ( \cdot , \cdot )$ , and let $L = \{ x _ { 1 } , \ldots , x _ { m } \} \subset \mathcal { X }$ be a set ofm landmarks. For each i, let $r _ { i } > 0$ , and define the compactly supported bumpfunction $b _ { i } ( x ) = b ( d ( x , x _ { i } ) ; a _ { i } , r _ { i } , \beta _ { i } )$ which is nonzero ifand only $i f d ( x , x _ { i } ) < r _ { i } .$ . For any $x \in { \mathcal { X } } ,$ , define the embedding vector

$$
\varphi ( x ) = [ b _ { 1 } ( x ) , b _ { 2 } ( x ) , \ldots , b _ { m } ( x ) ] ^ { \intercal } \in \mathbb { R } ^ { m } .
$$

Suppose $r _ { i } = r$ for all i, andfix a probability measure µ on X. Denote

$$
p _ { m } = \mathbb { P } _ { x \sim \mu } \left[ d ( x , x _ { i } ) < r \right]
$$

(where by symmetry, this does not depend on i iflandmarks are spread in a regularfashion and m is large).

Then the expected proportion ofnonzero entries in $\varphi ( x ) f o r x \sim \mu$ satisfies

$$
\mathbb { E } _ { x \sim \mu } \left[ \frac { \| \varphi ( x ) \| _ { 0 } } { m } \right] = p _ { m } ,
$$

so the expected number of nonzero entries is $m p _ { m }$ . If the landmarks become dense but r is fixed and small relative to the typical inter-point distance, then $p _ { m } \ll 1$ and the embedding becomes increasingly sparse as m grows.

Proof. For any $x \in \mathcal { X }$ , the i-th entry of $\varphi ( x )$ is nonzero if and only if $d ( x , x _ { i } ) < r$ . Thus,

$$
\| \varphi ( x ) \| _ { 0 } = \sum _ { i = 1 } ^ { m } \mathbb { I } \{ d ( x , x _ { i } ) < r \} .
$$

Taking expectation over $x \sim \mu ,$ , linearity of expectation gives

$$
\mathbb { E } _ { x \sim \mu } [ \| \varphi ( x ) \| _ { 0 } ] = \sum _ { i = 1 } ^ { m } \mathbb { P } _ { x \sim \mu } [ d ( x , x _ { i } ) < r ] .
$$

If the distribution of landmarks is regular and each $p _ { m } : = \mathbb { P } _ { x \sim \mu } [ d ( x , x _ { i } ) < r ]$ is (approximately) the same for all $i ,$ then

$$
\mathbb { E } _ { x \sim \mu } [ \| \varphi ( x ) \| _ { 0 } ] = m p _ { m } .
$$

Dividing by m yields the expected proportion. For small fixed $r$ compared to the domain size or typical landmark spacing, $p _ { m }$ can be made arbitrarily small and does not increase with m. Thus, even as m increases, the expected number of nonzero coordinates remains small compared to $m ,$ so the embedding is sparse. □

## A.7 Mitigation of Distance Concentration in High Dimensions

Theorem 6. Let X be a space in which standard Euclidean embeddings are subject to distance concentration (i.e., as the feature dimension $m  \infty$ , pairwise distances between random points become nearly equal). Let $\varphi : \mathcal { X }  \mathbb { R } ^ { m }$ be the compactly supported bump-function embedding defined as in previous theorems, so that each component $\varphi _ { i } ( x ) = b ( d ( x , x _ { i } ) ; a _ { i } , r _ { i } , \beta _ { i } )$ is nonzero if and only $i f d ( x , x _ { i } ) < r _ { i }$ for landmark $x _ { i } .$

Consider the kernel:

$$
k ( x , x ^ { \prime } ) = \sigma ^ { 2 } \exp \left( - \frac { \| \varphi ( x ) - \varphi ( x ^ { \prime } ) \| ^ { 2 } } { 2 \ell ^ { 2 } } \right) .
$$

Then, as m increases, provided the radii $\{ r _ { i } \}$ remain small relative to the domain, thefollowing hold:

1. Locality. The overlap $\langle \varphi ( x ) , \varphi ( x ^ { \prime } ) \rangle$ ⟩ is nonzero for a pair $( x , x ^ { \prime } )$ if and only if they fall within the support ofat least one common bump, i.e., $d ( x , x _ { i } ) < r _ { i }$ and $d ( x ^ { \prime } , x _ { i } ) < r _ { i }$ for some i.

2. Suppression of Distance Concentration. For most pairs $( x , x ^ { \prime } )$ , φ(x) and $\varphi ( x ^ { \prime } )$ have disjoint support, so that $\| \varphi ( x ) - \varphi ( x ^ { \prime } ) \| ^ { 2 } = \| \varphi ( x ) \| ^ { 2 } + \| \varphi ( x ^ { \prime } ) \| ^ { 2 }$ , making $k ( x , x ^ { \prime } )$ small, often exactly zero. $O n l y f o r$ nearby x, x<sup>′</sup> will $k ( x , x ^ { \prime } )$ be large.

3. Preservation of Informative Local Structure. The nonzero entries in the kernel matrix reflect local neighborhoods determined by the supports of the bump functions, preserving meaningful similarity relations in high dimensions and overcoming the loss ofdiscriminative power associated with distance concentration.

Consequently, the kernel does not sufferfrom the distance concentration effect typically observed in high-dimensional Euclideanfeature spaces. An empirical verification ofthe predicted conditioning behavior — which would be thefirst casualty ofdistance concentration — is provided in Appendix G.

Proof. 1. By the construction of the bump embedding, $\varphi _ { i } ( x )$ is nonzero only if $d ( x , x _ { i } ) < r _ { i }$ . Thus, for both $\varphi _ { i } ( x )$ and $\varphi _ { i } ( x ^ { \prime } )$ to be nonzero requires that both x and $x ^ { \prime }$ are within $r _ { i }$ of $x _ { i } .$ If this is not the case for any i, then $\varphi ( x )$ $\varphi ( x ^ { \prime } )$ have disjoint support.

2. In high dimensions, for randomly selected $x , x ^ { \prime }$ , the likelihood that they share support in any coordinate $i ( \mathrm { i . e . }$ , that x and $x ^ { \prime }$ both fall within the small ball of radius $r _ { i }$ around $x _ { i } )$ is vanishingly small as m increases, assuming the supports $r _ { i }$ are fixed and small relative to the domain or interlandmark distances. Thus, $\| \varphi ( \breve { x } ) - \varphi ( \dot { x } ^ { \prime } ) \| ^ { 2 } = \mathsf { \breve { \| } } \varphi ( x ) \| ^ { 2 } + \| \varphi ( x ^ { \prime } ) \| ^ { 2 }$ for most pairs, making $k ( x , x ^ { \prime } )$ small or exactly zero except for local neighborhoods.

3. Nontrivial (large) $k ( x , x ^ { \prime } )$ values can only arise if there is substantial overlap in the supports of $\varphi ( x )$ and $\varphi ( x ^ { \prime } )$ , i.e., x and $x ^ { \prime }$ are close to at least one common landmark. This means the kernel matrix is supported only on genuinely local neighborhoods, and entry magnitudes retain their informativeness even as m grows.

Therefore, the notorious phenomenon of distances becoming non-informative in high dimensions is avoided: the kernel remains locally discriminative and informative due to the sparsity and locality of the embedding. □

## A.8 Reducibility to Standard Stationary Kernels

Theorem 7. Let X be an input space equipped with a distance function $d ( \cdot , \cdot )$ , and let $L \ =$ $\{ x _ { 1 } , \dots , x _ { m } \} \subset { \mathcal { X } }$ be a set of landmarks. Define the embedding

$$
\boldsymbol { \varphi } ( \boldsymbol { x } ) = \left[ b ( d ( x , x _ { 1 } ) ; a _ { 1 } , r _ { 1 } , \beta _ { 1 } ) , b ( d ( x , x _ { 2 } ) ; a _ { 2 } , r _ { 2 } , \beta _ { 2 } ) , \dots , b ( d ( x , x _ { m } ) ; a _ { m } , r _ { m } , \beta _ { m } ) \right] ^ { \top } ,
$$

where each bump function $b ( d ; a , r , \beta )$ is continuous and strictly positive for $d \ < \ r ,$ , and zero otherwise. Consider the kernel

$$
k ( x , x ^ { \prime } ) = \sigma ^ { 2 } \exp \left( - \frac { \| \varphi ( x ) - \varphi ( x ^ { \prime } ) \| ^ { 2 } } { 2 \ell ^ { 2 } } \right) .
$$

Suppose for all i, $r _ { i } $ ∞ and $a _ { i } , \beta _ { i }$ are fixed so that $b ( \cdot )$ becomes a globally supported, smooth, strictly positive function of $\therefore d ( x , x _ { i } )$

Then, for all $x , x ^ { \prime } \in \mathcal { X }$

1. $\varphi ( x )$ is a dense feature vector depending only on the set $\{ d ( x , x _ { i } ) \} _ { i }$

2. $k ( x , x ^ { \prime } )$ reduces to a function that depends on $\{ d ( x , x _ { i } ) \}$ <sub>i</sub> and $\{ d ( x ^ { \prime } , x _ { i } ) \} _ { i } ,$

3. Ifd is a (conditionally) negative definite metric, then as $m  \infty$ and with suitable choice $o f b ( \cdot )$ , the kernel converges to a stationary RBF kernel $\begin{array} { r } { k _ { R B F } ( x , x ^ { \prime } ) = \exp { \left( - \frac { d ( x , x ^ { \prime } ) ^ { 2 } } { 2 \tilde { \ell } ^ { 2 } } \right) } } \end{array}$ on $( \mathcal { X } , d )$

Proof. 1. When all $r _ { i } \to \infty$ , for any $x \in \mathcal { X }$ and any $i , d ( x , x _ { i } ) < r _ { i }$ always holds. Therefore, each coordinate $b ( d ( x , x _ { i } ) ; a _ { i } , r _ { i } , \beta _ { i } )$ is strictly positive and only depends on $d ( x , x _ { i } )$ .

2. The vector $\varphi ( x )$ encodes the global structure of x with respect to all landmarks, and the difference $\varphi ( x ) - \varphi ( x ^ { \prime } )$ depends only on the vector differences $\{ b ( d ( \stackrel { . } { x } , x _ { i } ) ) - b ( d ( x ^ { \prime } , x _ { i } ) ) \} _ { i = 1 } ^ { m }$

3. If d is (conditionally) negative definite, the classic result for kernel methods states that the standard RBF kernel $\begin{array} { r } { k _ { \mathrm { R B F } } ( x , x ^ { \prime } ) = \exp \left( - \frac { d ( x , x ^ { \prime } ) ^ { 2 } } { 2 \tilde { \ell } ^ { 2 } } \right) } \end{array}$ is positive-definite and stationary on $( \mathcal { X } , d )$ For sufficiently large m and appropriately chosen, smooth, global bump functions, the feature embedding $\varphi ( x )$ can be made to approximate an injective mapping from $\mathcal { X }$ into $\mathbb { R } ^ { m }$ such that $\| \varphi ( x ) - \bar { \varphi } ( \dot { x } ^ { \prime } ) \|$ encodes $d ( x , x ^ { \prime } )$ up to a scale. Thus, in the limit $r _ { i } \to \infty$ and $m  \infty , k ( x , x ^ { \prime } )$ converges to the standard RBF kernel over $d ( \cdot , \cdot )$

Therefore, the bump-embedding kernel recovers the standard RBF kernel on the original space when the bumps become globally supported. □

## A.9 Continuity and Smoothness

Theorem 8. Let X be a topological space, and let d $: \mathcal { X } \times \mathcal { X }  \mathbb { R }$ be a continuous function. Consider a collection of smooth, compactly supported bump functions $b ( d ; a , r , \beta )$ that are $C ^ { \infty }$ (infinitely differentiable) on their support. Define the embedding

$$
\boldsymbol { \varphi } ( \boldsymbol { x } ) = \left[ b ( d ( x , x _ { 1 } ) ; a _ { 1 } , r _ { 1 } , \beta _ { 1 } ) , b ( d ( x , x _ { 2 } ) ; a _ { 2 } , r _ { 2 } , \beta _ { 2 } ) , \dots , b ( d ( x , x _ { m } ) ; a _ { m } , r _ { m } , \beta _ { m } ) \right] ^ { \top } ,
$$

for a set offixed landmarks $\{ x _ { i } \} _ { i = 1 } ^ { m }$ . The kernel is given by

$$
k ( x , x ^ { \prime } ) = \sigma ^ { 2 } \exp \left( - \frac { \| \varphi ( x ) - \varphi ( x ^ { \prime } ) \| ^ { 2 } } { 2 \ell ^ { 2 } } \right) .
$$

$H d ( x , x _ { i } )$ is smooth in x, and b is smooth in d, then $k ( x , x ^ { \prime } )$ is smooth (infinitely differentiable) as a function of each argument.

Proof. Since $d ( x , x _ { i } )$ is assumed smooth in x (for all fixed $x _ { i } )$ , and $b ( \cdot )$ is $C ^ { \infty }$ as a function of $d ,$ each coordinate of $\varphi ( x )$ is a composition of smooth functions and hence is $C ^ { \infty }$ in x. Therefore, $\varphi ( x )$ is $C ^ { \infty }$ as a mapping from X to $\hat { \mathbb { R } } ^ { m }$

The Euclidean norm, squaring, and difference are all smooth operations in $\mathbb { R } ^ { m }$ , so $F ( x , x ^ { \prime } ) =$ $\| \varphi ( x ) - \varphi ( x ^ { \prime } ) \| ^ { 2 }$ is a smooth function of both x and $x ^ { \prime } .$ . The function $k ( x , x ^ { \prime } )$ is then a composition of $F ( x , x ^ { \prime } )$ with the exponential function, which is also smooth.

Thus, $k ( x , x ^ { \prime } )$ is smooth in both arguments; that is, $k \in C ^ { \infty } ( \mathcal { X } \times \mathcal { X } )$

## A.10 Empirical Scaling and Complexity

Theorem 9. Let X be a metric space, let $L = \{ x _ { 1 } , \ldots , x _ { m } \} \subset \mathcal { X }$ be m landmarks, and let $b ( d ; a , r , \beta )$ denote a compactly supported bump function as in previous theorems. Define the embedding

$$
\varphi ( x ) = [ b ( d ( x , x _ { 1 } ) ; a _ { 1 } , r _ { 1 } , \beta _ { 1 } ) , b ( d ( x , x _ { 2 } ) ; a _ { 2 } , r _ { 2 } , \beta _ { 2 } ) , \dots , b ( d ( x , x _ { m } ) ; a _ { m } , r _ { m } , \beta _ { m } ) ] ^ { \top } \in \mathbb { R } ^ { m } .
$$

Let $s _ { x } = \| \varphi ( x ) \| _ { 0 }$ denote the number of nonzero entries in $\varphi ( x )$ . The kernel is given by

$$
k ( x , x ^ { \prime } ) = \sigma ^ { 2 } \exp \left( { - \frac { \| \varphi ( x ) - \varphi ( x ^ { \prime } ) \| ^ { 2 } } { 2 \ell ^ { 2 } } } \right) .
$$

Then:

1. For any pair $x , x ^ { \prime } \in { \mathcal { X } } ,$ computing $\| \varphi ( x ) - \varphi ( x ^ { \prime } ) \| ^ { 2 }$ and thus $k ( x , x ^ { \prime } )$ requires $\mathcal { O } ( s _ { x , x ^ { \prime } } )$ operations, where $s _ { x , x ^ { \prime } }$ is the number ofindices i such that at least one $o f \varphi _ { i } ( x ) o r \varphi _ { i } ( x ^ { \prime } )$ is nonzero $( i . e . ,$ , at most $s _ { x } + s _ { x ^ { \prime } } )$

2. Ifall bump radii $r _ { i }$ are small compared to the domain and the landmark set is sufficiently large, then $s _ { x } \ll m$ for typical $x ,$ so computational cost is sublinear in m.

3. The total number ofnonzero entries in the $N \times m$ embedding matrixfor N data points is $\mathcal { O } ( N \bar { s } )$ , where s¯ is the average sparsity per embedding, and all kernel matrix and matrix operation costs (e.g., matrix-vector products) scale accordingly.

Therefore, kernel evaluation and matrix operations scale with embedding sparsity (local bump overlap), not with thefull ambient embedding dimension m. Note that this concerns the kernel and linear-algebra stage; the cost of forming the distance profiles themselves is discussed in Appendix J.

Proof. 1. By construction, $b ( \cdot )$ is compactly supported, so for each x only a small fraction $s _ { x }$ of the coordinates in $\varphi ( x )$ are nonzero. The squared Euclidean distance $\| \varphi ( \dot { x } ) - \varphi ( x ^ { \prime } ) \| ^ { 2 }$ involves only dimensions where at least one entry is nonzero (i.e., the union of nonzero indices in $\varphi ( x )$ and $\varphi ( x ^ { \prime } ) )$ . Thus, its computation is $\mathcal { O } ( s _ { x , x ^ { \prime } } )$

2. If bump radii are small and landmarks are widely dispersed, $s _ { x }$ remains small and does not increase with m. Thus, the per-kernel evaluation and per-row storage cost are both $\mathcal { O } ( s _ { x } ) \ll m$

3. For a dataset of N points, the total number of floating-point operations for forming all $\varphi ( x ^ { ( i ) } )$ is $\mathcal { O } ( N \bar { s } )$ for average sparsity s¯. Matrix operations such as matrix-vector products with the Gram matrix K also scale with the number of nonzero overlaps between pairs of embeddings, yielding $\mathcal { O } ( N \bar { s } )$ scaling for sparse kernels, far more efficient than the $\mathcal { O } ( N m \bar { ) }$ scaling of dense embeddings.

Thus, the complexity is governed by embedding sparsity rather than by the full embedding dimension $m ,$ as claimed. □

## B The Non-Stationary SLE Kernel

In the stationary formulation of the SLE kernel, each bump function $b ( d ( x , x _ { i } ) ; a _ { i } , r _ { i } , \beta _ { i } )$ carries the same amplitude a, radius r, and shape parameter $\beta .$ So far, we have treated these as free hyperparameters to be learned globally. However, a more powerful and principled choice is to let them vary as functions (arbitrary parametric, NNs, polynomial) of position in $x ,$ , making the kernel explicitly non-stationary: the similarity structure it encodes can differ across different regions of the input space.

Concretely, we allow each landmark $x _ { i }$ to carry its own local parameters $a _ { i } = a ( x _ { i } ) , \quad r _ { i } =$ $r ( x _ { i } ) , \quad \bar { \beta } _ { i } = \beta ( x _ { i } )$ , that are functions defined on X. These functions can be specified by the user based on prior knowledge of the input domain, or learned from data. This yields the non-stationary SLE kernel, in the case of RBF,

$$
k _ { \mathrm { S L E - R B F } } ^ { \mathrm { N S } } ( \boldsymbol { x } , \boldsymbol { x ^ { \prime } } ) = \sigma ^ { 2 } \exp \left( - \frac { \lVert \phi ^ { \mathrm { N S } } ( \boldsymbol { x } ) - \phi ^ { \mathrm { N S } } ( \boldsymbol { x ^ { \prime } } ) \rVert ^ { 2 } } { 2 \ell ^ { 2 } } \right) ,\tag{7}
$$

where the non-stationary embedding is

$$
\phi ^ { \mathrm { N S } } ( x ) = \left[ b ( d ( x , x _ { 1 } ) ; a ( x _ { 1 } ) , r ( x _ { 1 } ) , \beta ( x _ { 1 } ) ) , \dots , b ( d ( x , x _ { | \mathcal { D } | } ) ; a ( x _ { | \mathcal { D } | } ) , r ( x _ { | \mathcal { D } | } ) , \beta ( x _ { | \mathcal { D } | } ) ) \right] ^ { \top } .\tag{8}
$$

The non-stationarity enters entirely through the domain-varying bump parameters, and the PSD property is unaffected, as the following proposition confirms.

Proposition 2 (PSD of the Non-Stationary SLE Kernel). Let $a ( \cdot ) , r ( \cdot )$ , and $\beta ( \cdot )$ be arbitrary positivevaluedfunctions on $\mathcal { X } .$ . Then $k _ { \mathrm { S L E } } ^ { \mathrm { N S } }$ as defined in $E q .$ (7) is a positive semi-definite kernel on X,for any distance $d ( \cdot , \cdot )$ , whether or not it is CND.

Proof. The non-stationary embedding $\phi ^ { \mathrm { N S } } : \mathcal { X } \xrightarrow { } \mathbb { R } ^ { | \mathcal { D } | }$ is a mapping into Euclidean space, regardless of how its parameters vary across $\mathcal { X } .$ . By the argument of Theorem 1, any kernel of the form $k ( x , x ^ { \prime } ) = h ( \phi ( x ) , \phi ( x ^ { \prime } ) )$ with h PSD on $\dot { \mathbb { R } } ^ { | \mathcal { D } | }$ is PSD on $\mathcal { X } .$ Since the RBF kernel is PSD on $\mathbb { R } ^ { | \mathcal { D } | }$ ， the result follows immediately. □

The three parameter fields $a ( \cdot ) , r ( \cdot )$ , and $\beta ( \cdot )$ each control a distinct aspect of the non-stationarity and can be defined via any parametric function to avoid an excessive number of hyperparameters. The individual roles of the parameter fields are described next.

## B.1 Non-Stationary Parameter Fields

Radius $r ( x _ { i } )$ . The support radius controls the spatial reach of landmark $x _ { i } { : }$ how large a neighborhood around $x _ { i }$ contributes to the similarity structure. In regions where the function being modeled varies rapidly, smaller radii are appropriate, encoding the intuition that only very nearby points should be considered similar. In smoother regions, larger radii allow information to propagate further. Varying $r ( \cdot )$ therefore adapts the effective length scale of the kernel to local function complexity, analogously to the input-dependent length scales of non-stationary kernels such as those proposed by Paciorek and Schervish [2003].

Amplitude $a ( x _ { i } )$ . The amplitude controls the contribution of landmark $x _ { i }$ to the overall embedding. Landmarks in regions of high data density or high functional relevance can be upweighted, while those in sparse or uninformative regions can be downweighted. This provides a mechanism for the kernel to allocate representational capacity unevenly across $\mathcal { X } ,$ , analogously to signal variance modulation in non-stationary GP models.

Shape $\beta ( x _ { i } )$ . The shape parameter controls the profile of the bump: how steeply similarity decays with distance from $x _ { i }$ within the support. Large $\beta$ produces a bump that is nearly flat near $x _ { i }$ and drops sharply near the boundary $^ { r , }$ while small β produces a smoother, more gradual decay. Varying $\beta ( \cdot )$ therefore allows the kernel to encode different local smoothness assumptions in different parts of $\chi$

Signal Variance $\sigma ( x )$ . In addition, we can make the signal variance non-stationary by considering $\sigma ^ { 2 } = \sigma ^ { 2 } ( x ) = \sigma ( x ) \overset { \sim } { \sigma ( x ) }$

Together, these four spatially varying parameters give the non-stationary SLE kernel considerable flexibility. We note that the stationary SLE kernel is recovered as the special case where $a ( \cdot ) , r ( \cdot )$ and $\beta ( \cdot )$ are constant functions.

Remark 3 (Sparsity Under Non-Stationarity). The sparsity properties established in Theorems 2 and 5 carry over directly to the non-stationary case. For any $x \in \mathcal { X } ,$ , the i-th entry of $\phi ^ { \mathrm { N S } } ( x )$ is nonzero if and only $i f d ( x , x _ { i } ) < r ( x _ { i } )$ . The expected number of nonzero entries is therefore $\begin{array} { r } { \sum _ { i = 1 } ^ { | { D } | } \mathbb { P } _ { \boldsymbol { x } \sim \boldsymbol { \mu } } [ d ( \boldsymbol { x } , \boldsymbol { x } _ { i } ) < r ( \boldsymbol { x } _ { i } ) ] } \end{array}$ , which remains small provided the radii $r ( x _ { i } )$ are small relative to the typical inter-point spacing. Non-stationarity in $r ( \cdot )$ thus affects the local sparsity pattern but does not compromise the overall sparsity of the embedding.

## C Complete Manifold Benchmarking Results

Here we present the full performance curves across all evaluated training dataset sizes for the GP regression experiments on the Dragon and Teddy Bear manifolds introduced in Section 5. These figures complement the summary statistics reported in Tables 1 and 2 of the main text, and provide a complete view of how predictive accuracy and uncertainty quantification evolve with training dataset size.

For the Dragon manifold, the Riemannian Matérn kernel approximates the covariance matrix via a truncated spectral expansion $K = \Phi _ { X } \mathrm { d i a g } ( S ) \Phi _ { X } ^ { \top }$ , where l eigenpairs are retained [Borovitskiy et al., 2020]. This truncation causes the kernel matrix to become rank-deficient when the number of training points n approaches l, leading to numerical failure of the Cholesky decomposition and, consequently, of model training. The resulting instability is visible as sharp spikes in RMSE and CRPS at n ≈ l $( n = \{ 1 0 0 , 5 0 0 , 1 0 0 0 \} )$ in Figure 2b–c, where divergent values are indicated by dashed lines rather than reported numerically. Most critically, this instability corrupts uncertainty quantification: the posterior variance becomes negative, requiring it to be clamped to zero and collapsing the predictive distribution to a point estimate. This renders CRPS undefined and drives PICP to 0%, explaining the empty cells in Table 1. In contrast, the SLE kernel maintains a stable PICP near the nominal 95% level and a smoothly decreasing CRPS with increasing training size, demonstrating reliable, calibrated uncertainty quantification without a spectral truncation parameter. For the Teddy Bear manifold, the full results are shown in Figure 3. In contrast to the previous tests, the Geometric kernel is stable across dataset sizes, so these results represent robust, stable runs.

![](images/1b6da82803ccba2cad1f1fed93f36058df3624d56d4a73aec305c36d3963a819.jpg)

b)  
![](images/e0f3a70d23a69711176e4f9a2b65be801eda6cd4428d954d09774573f13b55ab.jpg)

c)  
![](images/855aae4f70b865ef3ab8b139ed417960e65612814b21286d40b41e11103851e1.jpg)

d)  
![](images/bb04adf768104aed2759f7b942e2effe3238c6a6da030af28e0902ab8a4cc2c7.jpg)  
Figure 2: Benchmarking the SLE kernel against the Riemannian Matérn kernel [Borovitskiy et al., 2020] on the Dragon manifold. (a) Ground truth and SLE predicted output values represented by the color of the mesh vertices. (b) Test RMSE, (c) CRPS, and (d) PICP (95% interval) as a function of training dataset size for the SLE kernel and the Riemannian Matérn kernel with 100, 500, and 1000 eigenpairs. Dashed lines indicate training sizes where results are omitted due to numerical divergence caused by ill-conditioning of the Riemannian kernel matrix when $n \approx l ;$ this instability also accounts for the degraded performance of the 500-eigenpair variant relative to the 100-eigenpair variant at certain training sizes. Error bars represent the standard error of the mean across 30 trials.

a)  
![](images/bba2844bdc29c503b5ba0d8aa8378e0a90b16b252396c2b40e6497099aaaf39c.jpg)

Yb)  
![](images/3d16a45a7d9208d52248385416c7d6fb90b3280715c48d43bd8dcafdde93dce2.jpg)

c)  
![](images/4d71129918c7895f38ee067c63cdd7fc9e50255980603f9eb1b3231636217606.jpg)

d)  
![](images/c13e01c6e36a8343c190b983234690b6b962dddc98a110d4c7f1a7ffd6c90bd1.jpg)  
Figure 3: Benchmarking the SLE kernel against the Geometric kernel [Mostowsky et al., 2025] on the Teddy Bear manifold. (a) Ground truth and SLE predicted output values represented by the color of the mesh vertices. (b) Test RMSE, (c) CRPS, and (d) PICP (95% interval) as a function of training dataset size for the SLE kernel and the Geometric kernel. Error bars represent the standard error of the mean across 30 trials.

## D Additional Information on Experiments

## D.1 Dragon Manifold

Dataset. The experiment is conducted on the Stanford Dragon, a standard 3D benchmark geometry represented as a triangulated surface mesh. The mesh and associated scalar field values are generated using Firedrake. The mesh contains two arrays: ‘vertices’ of shape (N, 3) storing the 3D Cartesian coordinates of all $N = 1 0 0 , 1 7 9$ mesh vertices, and ‘ground truth’ of shape $( N , )$ storing the corresponding output. A precomputed symmetric geodesic distance matrix of shape $N \times N$ is also required. Each entry (i, j) contains the shortest-path distance between vertex i and vertex j measured along the mesh surface. The exact geodesic distance matrix was computed once using a graph-based shortest-path algorithm and stored for later extraction. This pre-computation took approximately 12 hours on a single CPU. The dominant memory cost is the $N \times N$ pairwise distance matrix and the derived covariance matrix, both of which store $N ^ { 2 }$ floating-point values. Once the distance matrix is available, training the GP model is inexpensive: with 100 training points, hyperparameter optimization completes in under a minute, scaling to roughly 3–4 minutes for 1000 points on a single CPU.

Experimental Design. To assess how performance scales with training set size, the GP is trained across multiple sizes, with 30 independent random trials per size. Training configurations (random seeds and training indices) are pre-generated and stored in a JSON file. A fixed held-out test set of 5,000 points is shared across all experiments to ensure consistent evaluation. An initial single exploratory run (seed 2355, $n _ { \mathrm { t r a i n } } = 1 , 0 0 0 , n _ { \mathrm { t e s t } } = 5 , 0 0 0 )$ is first conducted using global hyperparameter optimization to verify the setup before launching the full experiment.

GP Model Setup. The GP model is implemented using the ‘gpCAM‘ library with three components. They are implemented to match the Riemannian Kernel implementation discussed in [Borovitskiy et al., 2020].

Prior Mean. A zero prior mean is used throughout.

Noise. A homoscedastic zero-mean white noise with variance of $1 0 ^ { - 5 }$ is used, reflecting non-noisy data, while still ensuring numerical stability.

Kernel. The geodesic SLE kernel is used. Each input point is mapped to a feature vector by evaluating the bump function at its geodesic distances to all $n _ { \mathrm { t r a i n } }$ training points, and a $\nu = 3 / 2$ Matérn kernel is then applied to the Euclidean distance between these embeddings. Concretely, for a point x, the i-th component of its feature vector is:

$$
\phi ( x ) _ { i } = \tt b u m p ( { d } _ { g } ( x , z _ { i } ) , \it r , \beta , \beta = 1 ) , \quad i = 1 , \ldots , n _ { \tt t r a i n }
$$

where $d _ { g }$ denotes geodesic distance and the bump function is:

$$
\mathsf { b u m p } ( d , r , \beta ) = \left\{ \begin{array} { l l } { \displaystyle \mathrm { e x p } \left( \frac { - \beta } { 1 - d ^ { 2 } / r ^ { 2 } } + \beta \right) } & { \mathrm { i f } d < r } \\ { 0 } & { \mathrm { o t h e r w i s e } } \end{array} \right.
$$

The kernel value between two points is then:

$$
k ( x , x ^ { \prime } ) = \sigma _ { f } ^ { 2 } \cdot \left( 1 + \frac { \sqrt { 3 } \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { 2 } } { \ell } \right) \exp \left( - \frac { \sqrt { 3 } \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { 2 } } { \ell } \right)
$$

The kernel has three hyperparameters: signal variance $\sigma _ { f } ^ { 2 } ,$ bump radius $r ,$ and Matérn length scale ℓ. Since $\mathbf { \dot { \ g p C A M } } ^ { \circ }$ passes 3D coordinates rather than vertex indices to the kernel, vertex indices are recovered at evaluation time via a k-d tree nearest-neighbor lookup over the full vertex array.

Hyperparameter Bounds and Initialization. Bounds are set adaptively per trial. The signal variance is bounded between $0 . 0 1 \cdot \mathrm { V a r } ( y _ { \mathrm { t r a i n } } )$ and $1 0 \cdot \mathrm { V a r } ( y _ { \mathrm { t r a i n } } )$ , anchoring it to the observed scale of the target field. The bump radius is bounded between the minimum and maximum non-zero pairwise geodesic distances within the training set. The Matérn length scale is given broad, uninformative bounds of [0.01, 100]. All hyperparameters are initialized to the midpoint of their respective bounds.

Training and Evaluation. Hyperparameters are optimized by maximizing the log marginal likelihood. Optimization is performed using Markov Chain Monte Carlo (MCMC) with up to 4,000 iterations, providing more thorough exploration of the hyperparameter space. Results are saved incrementally after each trial, so the experiment can be interrupted and resumed without data loss.

Four metrics are computed on the held-out test set: RMSE on the test set to measure fit and generalization; CRPS (mean and standard deviation) as a proper scoring rule evaluating the full predictive distribution; and PICP at the 95% level, measuring the fraction of test points covered by the posterior predictive interval. A well-calibrated model should achieve $\mathrm { P I C P \approx 0 . 9 5 }$

## D.2 Teddy Bear Manifold

Dataset. The second experiment is conducted on a teddy bear mesh, loaded from an .obj file using the Mesh class from the Geometric Kernels library [Mostowsky et al., 2025]. The geodesic distance matrix is computed using the same graph-based shortest-path approach as above and loaded directly into memory. This pre-computation took approximately one hour on a single CPU. Once the distance matrix is available, training the GP model on the teddy bear dataset takes approximately 4 seconds for 100 training points and 2921 seconds ( 49 minutes) for 1000 training points on a single CPU.

Experimental Design. The design mirrors that of the Dragon experiment: multiple training set sizes are evaluated with 30 independent random trials per size, using a fixed held-out test set across all trials. An initial exploratory run is performed with seed 3256, $n _ { \mathrm { t r a i n } } = 5 0$ , and $n _ { \mathrm { t e s t } } = 5 0$ , using global optimization to verify the setup.

GP Model Setup. The same GP framework is used, with the following differences.

Input representation. Rather than passing 3D vertex coordinates to the kernel, vertex indices are passed directly as integer-valued inputs of shape $( n , 1 )$ . This eliminates the need for the k-d tree nearest-neighbor lookup used in the Dragon experiment, since geodesic distances can be indexed directly.

Kernel. The same geodesic SLE kernel structure is used, with the bump function defined as above. However, the inner stationary kernel is replaced with a $\nu = 5 / 2$ Matérn: $k ( x , x ^ { \prime } ) =$ $\begin{array} { r } { \sigma _ { f } ^ { 2 } \cdot \left( 1 + \frac { \sqrt { 5 } D } { \ell } + \frac { 5 D ^ { 2 } } { 3 \ell ^ { 2 } } \right) \exp \left( - \frac { \sqrt { 5 } D } { \ell } \right) , \quad D = \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { 2 } } \end{array}$ . The $\nu = 5 / 2$ Matérn is used here to be consistent with the kernel order adopted in [Mostowsky et al., 2025].

Bump amplitude. Unlike the Dragon experiment, where the bump amplitude was fixed at 1, here it is treated as a free hyperparameter $^ { a , }$ allowing the model to control the scale of the feature embedding independently of the signal variance.

Noise. Rather than a fixed noise level, the noise variance is also treated as a learnable hyperparameter, reflecting greater uncertainty about the noise level in this dataset.

Hyperparameter Bounds. The model has five hyperparameters: signal variance $\sigma _ { f } ^ { 2 } \in [ 1 0 ^ { 2 } , 1 0 ^ { 6 } ]$ bump radius r bounded by the minimum and maximum non-zero pairwise geodesic distances within the training set; bump amplitude $a \in [ 0 . 1 , 5 0 ]$ ; Matérn length scale $\bar { \ell } \in [ 1 0 ^ { - 3 } , 4 0 ] ;$ ; and noise variance $\in [ 1 0 ^ { - 6 } , \bar { 1 0 } ]$ . All hyperparameters are initialized at the midpoint of their bounds.

Training and Evaluation. Hyperparameters are optimized by maximizing the log marginal likelihood using global optimization with up to 4,000 iterations. The same four metrics are reported: train and test RMSE, CRPS, and PICP at the 95% level.

## D.3 X-Ray Scattering Data Disguised as Distributions

Dataset. The third experiment uses a dataset of 500 synthetic Small Angle X-ray Scattering (SAXS) images designed to mimic real-world patterns from oriented soft-matter thin films measured at synchrotron beamlines; examples are shown in Appendix Figure 4. The target output is the effective elastic modulus, $y ,$ computed along the measurement axis. Two precomputed pairwise distance matrices between images are used: a Wasserstein distance (WD) matrix and a Sliced Wasserstein distance (SWD) matrix with 50 projections, both of full size $n \times n$ where n is the total number of images. The experiment is run separately for each metric, and results are compared against a baseline GP using a standard $\nu = 3 / 2$ Matérn kernel applied directly to the sliced Wasserstein distances, without the bump embedding. The Wasserstein and sliced Wasserstein distance matrices were precomputed once and stored for later use; this pre-computation took approximately one hour on 32 CPUs. Once the distance matrix is available, training the GP model on the SAXS dataset takes approximately 11 seconds for 50 training points and 537 seconds ( 9 minutes) for 500 training points on a single CPU.

![](images/cce6bf0dd12cb397601f0fef173e71921fe841d74c64092dd9dc651697fc3745.jpg)  
Figure 4: Three examples of the 500 SAXS images for our computational experiments.

Experimental Design. The same multi-trial design is used, with 30 independent random trials per training set size and a fixed held-out test set across all trials. An initial exploratory run is performed with a 70/30 train/test split (random state 3345) and global optimization to verify the setup. As in the mesh experiments, image indices are passed as integer-valued inputs of shape $( n , 1 )$ , and the selected distance matrix is indexed directly.

GP Model Setup. The SLE kernel with the bump function and $\nu = 3 / 2$ Matérn inner kernel is used, as described above. The key structural difference from the mesh experiments is that the pairwise distances here are not geodesic distances on a surface but rather optimal transport distances between image distributions, specifically WD or SWD. The bump embedding therefore maps each image into a feature vector encoding its transport-distance neighborhood structure relative to the training set. Two additional hyperparameters are introduced compared to the Dragon experiment. The prior mean is no longer fixed at zero but is instead a learnable constant $\mu _ { 0 } .$ , bounded between the minimum and maximum observed training output values. This is appropriate here since the elastic modulus has a non-zero global mean that may vary across trials. The noise variance is also treated as a free hyperparameter, as in the teddy bear experiment.

Hyperparameter Bounds. The model has six hyperparameters in total: signal variance $\sigma _ { f } ^ { 2 } \in$ $[ 1 , 1 0 ^ { 1 0 } ]$ ; bump radius r bounded between the minimum and twice the maximum non-zero pairwise distance within the training set (the upper bound is doubled to allow the bump to cover the full range of distances); bump amplitude $a \in [ 0 . 0 1 , 1 0 ]$ ; Matérn length scale $\ell \in [ 1 0 ^ { \dot { - } 4 } , 1 0 0 ]$ ; noise variance $\in [ 1 0 ^ { - 5 } , 1 0 ] ;$ and mean offset $\mu _ { 0 } \in [ 0 , 1 0 ]$ . All hyperparameters are initialized at the midpoint of their bounds.

Training and Evaluation. Hyperparameters are optimized by maximizing the log marginal likelihood using global optimization with up to 10,000 iterations. The same four metrics are reported: train and test RMSE, CRPS, and PICP at the 95% level.

## E Ablation Study

To assess the contribution of the bump function in the SLE kernel, we conduct an ablation study comparing two embedding strategies: the proposed bump embedding, which applies a compactly supported smooth mask to the geodesic distances, and a distance-based embedding, which uses the raw geodesic distance vector to train landmarks directly. We note that the distance-based embedding is precisely the raw-distance embedding map underlying D2KE [Wu et al., 2018] and classical landmark constructions, so this ablation doubles as a controlled empirical comparison against that family, isolating the contribution of the bump construction (cf. Section 2). All other components of the kernel are held identical, including the inner Matérn covariance and the hyperparameter optimization procedure. We evaluate both variants on the Dragon and Teddy Bear meshes across a range of training set sizes, using three metrics: RMSE for predictive accuracy, CRPS for probabilistic sharpness, and PICP at the 95% level for uncertainty calibration. The two manifolds offer complementary perspectives: the Dragon presents a geometrically intricate surface with thin features, while the Teddy Bear is a smoother, more compact shape.

## E.1 Dragon Manifold

For the Dragon ablation, we use a target function constructed as a sum of sinusoids of geodesic distances from multiple well-separated source vertices with incommensurate periods. This deconfounded ground truth ensures that the function is genuinely manifold-defined but cannot be reduced to a one-dimensional function of distance from any single source, providing a fair test of both embeddings.

Figure 5 reports the ablation results on the Dragon mesh. In terms of point prediction Figure 5(a), the two embeddings track each other closely across the entire training range, with no meaningful difference in RMSE. The same pattern is observed in CRPS Figure 5(b), where the two methods produce nearly identical probabilistic sharpness throughout. The picture changes for uncertainty calibration Figure 5(c). The bump embedding reaches the nominal 95% PICP earlier and more reliably than the distance-based embedding, with a clear advantage maintained up to approximately 650 training points. Beyond that point, both methods converge toward the target coverage, and the distinction becomes minor. This indicates that, even when the two methods produce comparable point predictions, the sparse bump representation yields better-calibrated predictive variances across most of the practically relevant training range — a regime where principled uncertainty quantification is most valuable.

a)  
![](images/5facdb19bf69235c96a9dcc565bfc47b569aa5f35499b29919b3c86cbe85c40d.jpg)

b)  
![](images/16ce5b196c0e5a086b434fe50475154ad8074295bbd3ebebe4b70386a5793f28.jpg)

c)  
![](images/22fa8e857f252a7733575be70f682b21a02dd61f14db5c5d3b2f82f81ffc42e9.jpg)  
Figure 5: Ablation study comparing bump embedding and distance-based embedding on the Dragon mesh. (a) Test RMSE, (b) Test CRPS, and (c) Test PICP at the 95% confidence level, each as a function of training set size, averaged over 30 independent trials with error bars denoting one standard error. The two embeddings achieve essentially identical point prediction and probabilistic sharpness, while the bump embedding produces better-calibrated predictive intervals up to approximately 650 training points, after which the two methods converge to the nominal 95% coverage.

## E.2 Teddy Bear Manifold

For the Teddy Bear ablation, we use the same ground truth as in the main experiments (see Appendix D.2 for full experimental details), allowing the ablation to be interpreted directly in the context of the corresponding evaluation. Figure 6 reports the results. In terms of RMSE Figure 6(a), the distance-based embedding is competitive at small training sizes, where the embedding dimensionality remains manageable relative to the number of observations. As the training set grows, however, the raw distance embedding operates in an increasingly high-dimensional feature space without any regularization of its structure, and predictive accuracy degrades relative to the bump embedding. The bump embedding acts as a sparse, locally adaptive dimensionality reduction: each point is described only by its relationships to nearby landmarks, producing a compact, geometrically meaningful repre sentation that remains well-conditioned as the training size scales. The bump embedding opens a growing advantage beyond approximately 400 training points and achieves substantially lower error at 800 points.

The benefit of the bump embedding extends to uncertainty quantification on this manifold. Figure 6(b) shows that while the distance-based embedding achieves marginally lower CRPS at very small training sizes, the bump embedding overtakes it at approximately 200 training points and maintains consistently superior probabilistic sharpness thereafter, with the gap widening at larger training sizes. Figure 6(c) further confirms this picture: both methods converge toward the nominal 95% PICP, but the bump embedding does so faster, with tighter error bars and more stable calibration across the full training size range.

Taken together, the Dragon and Teddy Bear ablations show that the relative merit of the bump embedding depends on the geometric complexity of the manifold and the training regime. On smoother manifolds such as the Teddy Bear, the bump embedding yields clear improvements in both prediction and uncertainty quantification at moderate to large training sizes. On more intricate manifolds such as the Dragon, the bump embedding matches the distance-based embedding in point prediction and probabilistic sharpness while providing better-calibrated uncertainty estimates across most of the training range. In both cases, the bump embedding provides better-calibrated uncertainty in the small-data regime, where principled uncertainty quantification matters most.

a)  
![](images/5c857e6dd39c78676c17c2727c94c6205336e67b98171718af7a2f88692911e1.jpg)

b)  
![](images/18e6be8f3848dd90a27640e96899d5f3fe55e87139cf8965c12d58511689f824.jpg)

c)  
![](images/306acf1ae3ec6c26a6f3afd65adb187b4a40089abaa7edf71d07744e157d61cc.jpg)  
Figure 6: Ablation study comparing bump embedding and distance-based embedding on the teddy bear mesh. (a) Test RMSE, (b) Test CRPS, and (c) Test PICP at the 95% confidence level, each reported as a function of training set size (50–800 points), averaged over 30 independent trials with error bars denoting one standard error. The distance-based embedding is competitive at small training sizes but degrades relative to the bump embedding as the feature space dimensionality grows with the number of landmarks. The bump embedding, which induces a sparse and locally adaptive representation of the manifold geometry, achieves lower prediction error and better-calibrated uncertainty estimates at moderate to large training sizes, with the crossover occurring at approximately 400 points for RMSE and 200 points for CRPS.

## F Sensitivity Analyses

This section addresses the sensitivity of the SLE kernel to its central design choices: the bump support radius r, the bump amplitude $^ { a , }$ and the choice of inner kernel applied to the embedding. We recall that in all main experiments the radius and amplitude are not hand-tuned but learned by marginal-likelihood maximization with data-adaptive bounds (Appendix D.2), while the inner kernel family $( \mathrm { e . g }$ ., Matérn $\nu = 3 / 2$ or $\nu = 5 / 2 )$ is fixed by design choice, matched to the kernel order of the corresponding baseline; the analyses here characterize how performance varies away from the likelihood-selected radius and amplitude values, and how sensitive results are to the inner kernel choice itself. All three sweeps are conducted on the Teddy Bear manifold, following the same optimization procedure described in Appendix D.2; the specific training size and number of independent trials used for each sweep are stated in the corresponding subsection below. A fixed held-out test set is shared across all grid points and trials in every sweep, consistent with the evaluation protocol used elsewhere in the paper.

## F.1 Bump Radius r

The radius sweep uses a fixed training size of 300 points, the $\nu = 5 / 2$ Matérn inner kernel (consistent with the main Teddy Bear experiment, Appendix D.2), and 15 independent random trials per grid point. At each grid point, the radius r is held fixed at its grid value—swept between the minimum and maximum nonzero pairwise geodesic distances within the training set (the same data-adaptive bounds used for r during marginal-likelihood optimization elsewhere in the paper), up to $r \approx 5 6 -$ while the remaining hyperparameters (signal variance, bump amplitude, Matérn length scale, and noise variance) are re-optimized by marginal-likelihood maximization, following the same optimization procedure described in Appendix D.2. This isolates the effect of the radius after the model has been allowed to compensate through its remaining degrees of freedom, rather than showing raw sensitivity under an otherwise frozen model.

Figure 7 reports test RMSE, CRPS, and PICP as a function of r, together with the corresponding re-optimized log marginal likelihood. Predictive accuracy is highly sensitive to r at the small end of the range: as r shrinks toward zero, each embedding coordinate activates only a vanishingly small neighborhood, starving the kernel of local distance information even after re-optimizing the remaining hyperparameters, and both RMSE and CRPS rise sharply. Both metrics reach a minimum around $r \approx 1 3 \ – 1 5$ and then increase only mildly and monotonically thereafter, settling into a broad, shallow plateau for r beyond approximately 20 that persists to the upper bound of the sweep. PICP shows a markedly different pattern: coverage is reasonable near $r = 0$ , dips sharply to roughly 0.50 at very small nonzero radii—a regime in which the embedding, even with the remaining hyperparameters re-optimized, is expressive enough to fit the mean well but too locally constrained to produce wellcalibrated variance estimates—and then recovers quickly, exceeding 0.90 by $r \approx 1 0$ and drifting upward toward the nominal 0.95 level as r grows further, with the closest approach to the target near the upper end of the sweep.

The re-optimized log-likelihood tracks this same transition (Figure 7d): it rises sharply from its worst value at $r \approx 0$ and plateaus for r beyond roughly 15–20, mirroring the RMSE/CRPS plateau and indicating that the marginal-likelihood surface itself, not just the point-prediction metrics, favors radii in this broad mid-to-large range over very small ones. Taken together, these results indicate a mild tension between sharpness and calibration: the radius minimizing RMSE and CRPS $( r \approx 1 3 – 1 5 )$ is somewhat smaller than the radius optimizing PICP, though the accuracy cost of choosing a larger, better-calibrated radius is small, since RMSE and CRPS remain within the flat plateau across this region. The value selected by unconstrained marginal-likelihood maximization in the main Teddy Bear experiment $( r \approx 3 0 $ , dotted line in Figure 7) falls within this plateau, consistent with the intended role of r as a data-adaptive length-scale analog rather than a parameter requiring manual tuning.

## F.2 Bump Amplitude a

The amplitude sweep uses the same protocol as the radius sweep: a fixed training size of 600 points (it was 300 in the radius sweep), the $\nu = 5 / 2$ Matérn inner kernel, and 13 independent trials per grid point, with signal variance, bump radius, length scale, and noise variance re-optimized at each fixed value of a by marginal-likelihood maximization, following Appendix D.2. The sweep spans $a \in [ 0 , 5 0 ]$ , the same bound used during hyperparameter learning in the main experiment.

Figure 8 shows that the test metrics (RMSE, CRPS, PICP, and log-likelihood) are essentially flat over $a \in [ 0 , 2 2 ]$ : RMSE and CRPS sit at or near their best values, log-likelihood is at or near its maximum, and PICP is mildly below the nominal 0.95 level. Beyond $a \approx 2 2$ , this stability breaks down: log-likelihood declines steadily for the remainder of the sweep, RMSE and CRPS both worsen substantially, and PICP drifts upward past nominal coverage toward mild over-confidence-in-reverse (over-coverage, $\approx 0 . 9 6 )$ by $a = 5 0$

Figure 9 makes explicit the mechanism underlying this pattern by tracking the re-optimized hyperparameters themselves rather than only the resulting predictive metrics. Over $a \in [ 0 , 2 2 $ ], the flatness of the test metrics in Figure 8 is not because the underlying model is static — it is because two hyperparameters are actively compensating for the growing amplitude. Because a rescales every nonzero entry of the embedding before the Euclidean distance $\| \phi ( x ) - \phi ( x ^ { \prime } ) \|$ is formed, increasing a inflates typical inter-point distances in embedding space; the re-optimized Matérn length scale ℓ and bump radius r both increase steadily over this range (Figure 9b–c) to offset this, keeping the effective kernel — and hence predictive performance — nearly unchanged. Signal variance (Figure 9a), by contrast, remains comparatively flat and noisy over this same range, indicating it plays little role in the compensation while ℓ and r still have room to grow.

![](images/b02107d9a561af16887768415984fa9466669a2954ed83d8b70cb7f93b73e0c6.jpg)

![](images/c248ac0e3babd0c55c6f6c0770bbcae7eab8987e556993aa25a11022390af67e.jpg)

c)  
![](images/71e659596277aef9acc5ed4a2ff0a9b8a5cca5c2ce714bc594d38c4a40ea2958.jpg)

d)  
![](images/5b6662538a2ca3c69060b899b37df334548bbe745f82678ab6d0d69a087f37be.jpg)  
Figure 7: Sensitivity of the SLE kernel to the bump radius r on the Teddy Bear manifold, at a fixed training size of 300 points with the $\nu = 5 / 2$ Matérn inner kernel and 15 independent trials per grid point. At each grid point r is held fixed while the remaining hyperparameters (signal variance, bump amplitude, Matérn length scale, and noise variance) are re-optimized by marginallikelihood maximization, isolating the effect of r after the model has been allowed to compensate through its other degrees of freedom. (a) Test RMSE, (b) Test CRPS, (c) Test PICP at the 95% level (dashed horizontal line marks nominal coverage), and (d) the corresponding re-optimized log marginal likelihood, each as a function of r. The red dotted vertical line marks the radius selected by unconstrained marginal-likelihood maximization in the main Teddy Bear experiment (Appendix D.2). Shaded bands denote one standard error across trials.

This compensation is only possible while ℓ and r have room to grow, and both reach their fixed upper bounds within the sweep. The length scale saturates first, reaching its configured upper bound of 40 at $a \approx 2 2$ (Figure 9c); the radius continues increasing for a time afterward, reaching its own upper bound of 56 only around $a \approx 3 5$ (Figure 9b). Once the length scale can no longer increase to track a, it is no longer sufficient on its own to keep the effective kernel unchanged, and the model falls back on two alternate mechanisms: signal variance begins declining steadily from that point onward (Figure 9a), and noise variance jumps sharply, from a small, stable value below $1 0 ^ { \div 2 }$ to roughly 0.6–0.8 (Figure 9d). This saturation-and-fallback sequence — length scale saturating first, radius following, then signal variance and noise variance absorbing the remainder — is the direct explanation for the divergence in RMSE, CRPS, and log-likelihood beyond $a \approx 2 2$ in Figure 8, rather than any qualitative change in the kernel’s locality. The sharp rise in noise variance, in particular, explains the mild PICP over-coverage observed in the same regime, since inflated noise variance directly widens predictive intervals, regardless of whether the underlying fit is improving. Noise variance drops again at $a = 5 0$ , the extreme edge of the sweep range; since predictive performance is already substantially degraded throughout this saturated regime and this point simply marks the boundary of the search space rather than a qualitatively new operating condition, we do not interpret this final drop further.

![](images/3adf6db76a4937a88da7df763a4080b8e38d25ffd852b7291ab8533be8f9ba75.jpg)

![](images/c516ca0e0735e7c0cc24c2ec1eaac7a6cc145d12d69364df3ea17ec9a4fc0f89.jpg)

c)  
![](images/a57b0451557f62bc80508f26624591e7930484e8bb4ea92dd304181b1f16abe6.jpg)

![](images/d035d54d08d2f5e05b016a940ebeca947d9c8dc10126512a4d24ee3b1731e140.jpg)  
Figure 8: Sensitivity of the SLE kernel to the bump amplitude a on the Teddy Bear manifold, at a fixed training size of 600 points with the $\nu = 5 / 2$ Matérn inner kernel and 13 independent trials per grid point. At each grid point, a is held fixed while the remaining hyperparameters (signal variance, bump radius, Matérn length scale, and noise variance) are re-optimized by marginal-likelihood maximization, isolating the effect of a after the model has been allowed to compensate through its other degrees of freedom. (a) Test RMSE, (b) Test CRPS, (c) Test PICP at the 95% level (dashed horizontal line marks nominal coverage), and (d) the corresponding re-optimized log marginal likelihood, each as a function of a. The red dotted vertical line marks the amplitude selected by unconstrained marginal-likelihood maximization in the main Teddy Bear experiment (Appendix D.2). Shaded bands denote one standard error across trials.

The amplitude selected by unconstrained marginal-likelihood maximization in the main Teddy Bear experiment $( a \approx 1 2 . 2 5$ , red dotted line in Figures 8 and 9 falls well within the region where ℓ and r can still freely compensate for a, comfortably below the saturation point at a ≈ 22 where predictive performance begins to degrade. This also clarifies why fixing $a = 1$ in the Dragon experiment, rather than learning it, incurs no cost: at a value well inside this compensating region, amplitude, length scale, and radius trade off freely, and the model retains its full expressivity regardless of which specific value of a is chosen within this regime.

## F.3 Inner Kernel

For this sweep, the three most common choices of stationary kernel—Matérn $\nu = 5 / 2$ , Matérn $\nu = 3 / 2$ , and RBF — are each applied to the same bump embedding at a fixed training size of 300 points, with all remaining hyperparameters (signal variance, bump radius, bump amplitude, kernel length scale, and noise variance) independently re-optimized for each kernel by marginal-likelihood maximization, following the same protocol used elsewhere in Appendix D.2. This isolates the effect of the inner kernel’s functional form from the effect of the embedding itself, which is held fixed across all three configurations.

Figure 10 shows that predictive accuracy is essentially insensitive to the choice of inner kernel: the RMSE (a) and CRPS (b) distributions for all three kernels overlap substantially, with nearly identical medians and interquartile ranges, and no kernel is a clear or consistent winner across 15 trials. This is consistent with the sparse bump embedding doing the bulk of the representational work, with the inner kernel’s functional form acting as a comparatively minor modulation on top of an already well-conditioned, locally structured input space.

![](images/b814ab7ea2d18a131b25c90a69d6bc9b3c6fcb234cb3b87142fbda4b144bd016.jpg)

![](images/cf164b00ee0178c2d4a87c67453ab21ef0841c1690a088d1d26f1809e383995f.jpg)

c)  
![](images/35bebf30742c73e2a637a325eb9110cf8c60612d3578efc640e66027d7a9f1c5.jpg)

d)  
![](images/a2841db5ce403827f07056eef255778104d4b5a3e5a5bc6b998baaeb9399a2e0.jpg)  
Figure 9: Re-optimized hyperparameters as a function of bump amplitude $^ { a , }$ corresponding to the sweep in Figure 8. At each grid point, signal variance (a), bump radius (b), Matérn length scale (c), and noise variance (d) are re-optimized by marginal-likelihood maximization while a is held fixed. The dash-dotted black line marks the amplitude at which the length scale saturates at its upper bound $( a \approx 2 2 )$ ; the dotted red line marks the amplitude selected by unconstrained marginal-likelihood maximization in the main Teddy Bear experiment $( a = 1 2 . 2 5 )$ . Dashed black lines in (b) and (c) mark the configured upper bounds on bump radius (56) and length scale (40), respectively. Shaded bands denote one standard deviation across trials.

Calibration and marginal likelihood, however, do show a modest but consistent separation between kernel choices that predictive accuracy alone does not reveal. Both Matérn variants achieve median PICP closer to the nominal 0.95 level (panel c), with Matérn $\nu = 3 / 2$ slightly ahead of $\nu = 5 / 2$ RBF, by contrast, shows both a lower median PICP $\mathrm { ( \approx 0 . 9 2 ) }$ and a wider spread extending well below nominal coverage, indicating a mild but noticeable tendency toward overconfident intervals relative to the Matérn kernels. This pattern is not mirrored in the log-likelihood panel (d): RBF achieves the highest (least negative) median log-likelihood of the three, with Matérn $\nu = 3 / 2$ the lowest, while Matérn $\nu = 5 / 2$ falls in between. In other words, the kernel most favored by the marginal-likelihood objective (RBF) is also the kernel with the weakest calibration on held-out data, while the best-calibrated kernel (Matérn $\nu = 3 / 2 )$ has the lowest training-time marginal likelihood of the three. This is a useful reminder that marginal-likelihood maximization selects for in-sample fit and is not a direct proxy for held-out calibration, and it provides a concrete, data-driven justification for the paper’s choice to fix the inner kernel to the Matérn family, matched to the kernel order of the corresponding baseline, rather than treating it as a free hyperparameter selected by likelihood alone.

Taken together, these results support a qualified version of the intended narrative for this section: the inner kernel’s effect on point-prediction accuracy is negligible, consistent with the embedding— not the inner kernel— driving predictive performance, but its effect on uncertainty calibration is real, if modest, and argues for the deliberate, baseline-matched choice of a Matérn inner kernel used throughout the main experiments rather than for treating the inner kernel as an arbitrary or inconsequential design choice.

a)  
![](images/8ff61e2730adbfa5feb086df000469102d228f15b3f17978e0e6c6d343444f9b.jpg)

b)  
![](images/b4bfbff2bec621661ae9cd6b1e58f15f6b7ac4ec5f80fd2e049d32ec08987895.jpg)

![](images/410b6bd3551b6b5a053fce6c6fe21dee9c7f0350476504ce9dd77f89a3feab80.jpg)

d)  
![](images/54b6080ee1df63d2ab37acb80f079f7fcda4d0005d2696fc14dd75818eb5ed1b.jpg)  
Figure 10: Sensitivity of the SLE kernel to the choice of inner kernel applied to the bump embedding, on the Teddy Bear manifold at a fixed training size of 300 points. For each of three inner kernel choices — Matérn $\nu = 5 / 2$ , Matérn $\nu = 3 / 2$ , and RBF — the remaining hyperparameters (signal variance, bump radius, bump amplitude, kernel length scale, and noise variance) are independently re-optimized by marginal-likelihood maximization, following the protocol described in Appendix 10.2. (a) Test RMSE, (b) Test CRPS, (c) Test PICP at the 95% level (dashed horizontal line marks nominal coverage), and (d) the corresponding final log marginal likelihood. Box plots show the median (orange line), mean (white diamond), interquartile range (box), and full range excluding outliers (whiskers); individual trial values are overlaid as jittered points, with outliers outlined in black. Results are computed across 15 independent trials per kernel.

## G Empirical Conditioning of the Gram Matrix

Theorem 4 predicts, under the bounded-overlap Assumption (A1), that the SLE Gram matrix remains well-conditioned as the number of landmarks |D| grows, in contrast to dense raw-distance embeddings. Here we test this prediction directly. For the one-dimensional test function of Figure 1, Figure 11 shows that the condition number is far better behaved for the SLE kernel — growing only as $\kappa _ { 2 } ~ \sim ~ N ^ { 0 . 3 8 }$ over $N \in \mathsf { [ 2 0 , 1 0 0 0 ] }$ , compared to $\kappa _ { 2 } ~ \sim ~ N ^ { \approx 1 }$ for the native (dense) distance-tolandmarks embedding. Both kernels are evaluated at the same initial hyperparameters used in the runtime benchmark (Table 5), and both matrices are regularized by the identical noise nugget $\sigma _ { n } ^ { 2 } = 0 . 0 1$ that the GP marginal-likelihood solve actually uses, so the comparison isolates the effect of the embedding itself rather than the regularizer. At small N, the two embeddings are indistinguishable $( \kappa _ { \mathrm { n a t i v e } } / \kappa _ { \mathrm { S L E } } \approx 0 . 9$ at $N = 2 0 )$ , because the nugget floor dominates the smallest singular value on both sides; as N grows, the sparse-support geometry of the bump embedding caps the effective feature dimension while the dense embedding’s landmark coordinates progressively concentrate, driving the ratio to $\kappa _ { \mathrm { n a t i v e } } / \kappa _ { \mathrm { S L E } } \approx 9 . 8$ at $N = \mathrm { \check { 1 } 0 0 0 }$

![](images/68bea22fdb3f0cf5951cf16c82812147d3a3e6d046807d8bd524c6946e5753ff.jpg)  
Figure 11: Gram-matrix condition number $\kappa _ { 2 } ( K + \sigma _ { n } ^ { 2 } I )$ with $\sigma _ { n } ^ { 2 } = 0 . 0 1$ for the SLE kernel versus the native (dense) distance-to-landmarks embedding on the 1D benchmark of Figure 1. Lines are means and shaded bands min–max over 5 random dataset draws; both kernels use the initial hyperparameters of Table 5 (no training). SLE conditioning grows only as $\kappa _ { 2 } \sim N ^ { 0 . 3 8 }$ while the native embedding grows as $\kappa _ { 2 } \sim N ^ { \approx 1 }$ (≈ 9.8× worse at $N = 1 0 0 0 )$ , consistent with Theorem 4: compact bump support decouples the conditioning of K from the ambient embedding dimension |D|.

We note that this constitutes a conservative test of Theorem 4: in this benchmark the bump radius spans a constant fraction of the domain, so the average overlap s¯ grows linearly with N (Table 5) and Assumption (A1) is deliberately not enforced — yet conditioning still degrades dramatically more slowly than for the dense embedding, and the gap opens exactly as the ambient embedding dimension grows. This isolates dimensionality, rather than the embedding per se, as the cause of the dense embedding’s degradation, consistent with the ablation results of Appendix E. Compact support thus decouples the conditioning of K from |D|, keeping SLE Gram matrices amenable to a numerically stable Cholesky factorization — and, consequently, to reliable posterior inference and hyperparameter learning — in regimes where the native embedding is already losing precision.

## H Empirical Distortion Analysis

Proposition 1 establishes that the SLE embedding is an injective function of the exact local distance profile, introducing no surrogate metric. Here, we complement this with an empirical comparison of embedding-space distances to native distances, alongside the sliced Wasserstein approximation. The analysis is performed on the SAXS distance matrices of Appendix D.3 using the embedding map directly, with all 500 images serving as landmarks; it characterizes the map itself and involves no fitted model.

Figure 12a plots the embedding distance $\| \phi ( x ) - \phi ( x ^ { \prime } ) \|$ against the true $W _ { 2 }$ distance for all 124,750 pairs, at $r = 0 . 1 8 , \beta = 1$ and $a = 1$ , giving a Spearman rank correlation of $\rho = 0 . 9 7 6$ . Figure 12b repeats the comparison against the sliced Wasserstein distance $( \rho = 0 . 9 6 8 )$ , and Figure 12c compares the sliced surrogate to the true $W _ { 2 }$ directly $( \rho = 0 . 9 9 2 )$ . The closeness of the values in (a) and (b) is a direct consequence of (c): since the sliced approximation is itself highly rank-correlated with the true $W _ { 2 }$ distance, the SLE embedding’s rank fidelity to one target is necessarily close to its rank fidelity to the other. Rank correlation is invariant to the bump amplitude, since a rescales every embedding coordinate uniformly, but not to β, which is held at 1 throughout.

Within the bump reach, the embedding distance is a faithful increasing function of the native distance, with the scatter in Figure 12a being tight and the ordering essentially preserved. Beyond the reach, the relationship folds over, and this is a direct consequence of compact support rather than a distortion of the geometry. Once two inputs are separated by more than $r ,$ neither lies in the support of the other’s bump, the coordinates that encode their mutual distance vanish, and the embedding distance reduces to the disjoint-support identity $\left( \| \phi ( x ) \| ^ { 2 } + \| \phi ( x ^ { \prime } ) \| ^ { 2 } \right) ^ { 1 / 2 }$ (Theorem 6). This residual quantity measures how densely each input’s own neighborhood is populated rather than how far apart the two inputs are, and since the most widely separated pairs tend to lie in sparser regions, it decreases with $W _ { 2 }$ over the far field. Far pairs are therefore no longer ordered by the embedding. This is precisely the far-field information that Proposition 1 states is discarded, and that Section 6 records as a limitation of the deliberately local design; at $r = 0 . 1 8$ it concerns the 6.7% of pairs separated by more than the radius.

a)  
![](images/abe4fcd0fd8dae45b67c6da807a43b85ad4131fc7389f0ffe6db0142c0d653d4.jpg)  
b)

![](images/53230d6b15c531863840b4c3b66ede9f56be873640bb20079fdf3a52af548fb3.jpg)  
d)

c)  
![](images/cc147fbfac471d9f75e664a437f723fc1c2c74b1191152e15984c829f5982c35.jpg)

![](images/0f465b8fd6318b19343e5bed1401c3f00d3cf8cc6c9e897fe1e775df0a2669ed.jpg)

![](images/ea3fa3b4c7cca399f145a55c768143a9bac9eb04095b776c5a64d50f7834a313.jpg)  
Figure 12: Empirical distortion analysis on the SAXS distance matrices, over all 124,750 pairs of the 500 images. (a) SLE embedding distance $\| \phi ( x ) - \phi ( x ^ { \prime } ) \|$ against the true Wasserstein-2 distance, at bump radius $r = 0 . 1 8 .$ , shape $\beta = 1$ , and amplitude $a = 1$ . (b) The same embedding distance against the sliced Wasserstein distance (50 projections). (c) Sliced Wasserstein against true ${ \bar { W } } _ { 2 } .$ . (d) The same comparison as (a), at a larger bump radius $r = 0 . 3$ . (e) Rank correlation between the embedding distance and the true $W _ { 2 }$ as a function of the bump radius r, over the same pair set. The green dotted line marks the globally supported (dense) limit obtained once r exceeds the largest pairwise distance, and the black dashed line marks that distance. Spearman rank correlations are inset in panels (a)–(d). The analysis uses the embedding map applied directly to the precomputed distance matrices and is independent of any fitted GP model.

Figure 12d repeats this comparison at a substantially larger radius, $r = 0 . 3$ , larger than the largest pairwise distance in the dataset. At this radius every pair lies within reach of every bump, so the embedding is fully dense and the fold-over visible in Figure 12a is eliminated entirely: the scatter is monotonic across the full range of native distances. The resulting rank correlation, $\rho = 0 . 9 4 4$ is nonetheless slightly below the peak value obtained at $r = 0 . 1 8$ , illustrating directly the shallowmaximum behavior quantified by the full sweep in Figure 12e: enlarging the radius removes the far-field fold-over but does not, on this dataset, improve rank fidelity beyond what is already achieved by a much sparser embedding.

Figure 12e shows that this behavior is not an artifact of the particular radius chosen. Rank fidelity rises steeply with r and varies by less than 0.03 for all $r \geq 0 . 1 5$ , remaining high out to and beyond the largest pairwise distance in the dataset; the values shown in Figures 12a and 12d are drawn from this range. Two features of the curve are worth noting. First, the shallow maximum near $r \approx 0 . 1 8$ lies marginally above the dense limit reached once r exceeds the data diameter and every bump is globally supported, so compact support costs nothing in rank fidelity relative to a dense embedding on this dataset while retaining exactly zero entries (Theorem 2). Second, at very small radii, the correlation is mildly negative: below $r \approx 0 . 0 7 ,$ almost no pair of inputs shares a landmark in common support, every embedding distance is governed by the density term above, and the ordering it induces runs weakly counter to the native one for the reason given in the preceding paragraph. This regime is far from any radius of practical interest and is shown only for completeness.

Taken together, the comparison confirms that the SLE embedding preserves the ordering of the native Wasserstein geometry within the region its bumps span, without introducing any projection or surrogate metric, and locates the boundary of that region exactly where Proposition 1 places it.

## I Bump Function Visualization

Section 4 introduces the bump function $b ( d ; a , r , \beta )$ of Eq. (4) (repeated here for reference)

$$
b ( d ; a , r , \beta ) = \left\{ \begin{array} { l l } { { a \exp \left( - \displaystyle \frac { \beta } { 1 - d ^ { 2 } / r ^ { 2 } } + \beta \right) , } } & { { d < r , } } \\ { { 0 , } } & { { d \geq r , } } \end{array} \right.\tag{9}
$$

as the map applied to each coordinate of the sparse landmark embedding $\begin{array} { r l } { \phi ( { \boldsymbol { x } } ) } & { { } = } \end{array}$ $\left[ b ( d ( x , x _ { 1 } ) ; a , r , \beta ) , \dots , b ( d ( x , x _ { | \mathcal { D } | } ) ; a , r , \beta ) \right] ^ { \intercal }$ . Figure 13 visualizes this function for fixed amplitude and support radius $( a = 1 , r = 1 )$ across three shape parameters, $\beta \in \{ 0 . 5 , 1 , 5 \}$ , isolating the three properties on which the paper’s guarantees depend: compact support on $[ 0 , r )$ (Theorems 2 and 5, which give the sparsity of $\phi$ and hence of the SLE embedding), $C ^ { \infty }$ smoothness on the support and at the boundary $d = r$ (Theorem 8, which gives smoothness of the resulting kernel), and strict positivity and monotonic decay on $[ 0 , r )$ (the hypothesis of Proposition 1, which gives injectivity of $\phi$ with respect to the local distance profile).

All three curves in Figure 13 are strictly positive and strictly decreasing on [0, 1) and drop to exactly zero at $d = r = 1$ , with all derivatives vanishing at the boundary rather than producing a kink — this is what allows a training point $x _ { i }$ whose distance from x satisfies $d ( x , x _ { i } ) \geq r$ to contribute exactly zero to the embedding coordinate $\phi _ { i } ( x )$ , rather than a small but nonzero value, which is the mechanism underlying the sparsity results of Section 4. The shape parameter $\beta$ controls how the decay is distributed within the support. At $\beta = 0 . 5$ the bump remains close to its peak value a over most of $[ 0 , r )$ and falls off steeply only as d approaches the boundary, so that a landmark contributes a nearly uniform weight until x nears the edge of its support. $\operatorname { A t } \beta = 5 ,$ , the bump decays rapidly from the origin and is already small well before the boundary is reached, so that a landmark’s contribution is sharply concentrated on its immediate neighborhood, while the outer portion of the support contributes little. The $\beta = 1$ curve used throughout this work is intermediate between the two. Note that all three profiles share the same support radius r and therefore the same sparsity pattern: $\beta$ changes the weighting within the support, not which coordinates are nonzero. In the main text, $\beta$ is held fixed at 1 and r is learned by marginal-likelihood maximization with data-adaptive bounds (Appendices D.1–D.3); the sensitivity of predictive performance to r and to the amplitude a is reported separately in Appendix F.

## J Computational Cost

Per kernel evaluation, the cost is $\mathcal { O } ( s )$ , where s is the number of overlapping nonzero embedding entries (Theorem 9), independent of the ambient embedding dimension $| \mathcal D |$ . The dominant cost is forming the distance profiles between evaluation points and landmarks: $\mathcal { O } ( N \cdot | \mathcal { D } | )$ distance evaluations if computed naively. Two observations put this cost in context. First, it is shared by every distance-based competitor considered in this work: the sliced Wasserstein baseline computes the same N · |D| (sliced) distances, and the Riemannian and Geometric kernels additionally require a Laplace–Beltrami eigendecomposition of the full mesh. In all our experiments, the distance matrix is precomputed once and shared across all methods and trials. Second, because only landmarks within radius r contribute to the embedding, metric-space indexing structures — cover trees, vantage-point trees, or approximate nearest-neighbor search, which require only the distance function and no coordinate representation — reduce test-time distance computation to range queries.

![](images/03dc36364aa4fa48762a64f9c7b00eb423ee5f05dd3b0a4abadca0652d2f4d91.jpg)  
Figure 13: The bump function $b ( d ; a , r , \beta )$ of Eq. (9), plotted versus distance d for fixed $a = 1 , r = 1$ and three values of the shape parameter $\bar { \beta } \in \{ 0 . 5 , 1 , 5 \}$ . The dashed vertical line marks the support boundary $d = r ,$ beyond which each embedding coordinate $\phi _ { i } ( x ) = b ( d ( x , x _ { i } ) ; a , r , \beta )$ is identically zero. All curves attain the peak value a at $d = 0$ , are strictly positive and strictly decreasing on $[ 0 , r )$ and vanish smoothly (all derivatives $ 0 )$ at $d = r$ , illustrating the compact support, $C ^ { \infty }$ smoothness, and strict monotonicity relied on by Theorems $2 , 5 ,$ and 8 and Proposition 1. The three curves differ only in how the decay is distributed across the support: larger $\beta$ decays more rapidly from the origin and lies below the smaller- $\cdot \beta$ curves at every $d \in ( 0 , r )$ , with the half-maximum crossing moving inward from d $\approx 0 . 7 6 r$ at $\beta = 0 . 5$ to d ≈ 0.35 r at $\beta = 5$

Memory and time relative to a standard distance-based kernel. Relative to a conventional distance-based kernel (e.g., a Matérn kernel applied directly to a CND distance), the SLE kernel requires no additional memory: both approaches consume the same $N \times | \mathcal { D } |$ pairwise distance matrix and produce the same $N \times N$ Gram matrix. The sparse embedding adds only $O ( N { \bar { s } } )$ nonzero entries, where s¯ is the average number of active bumps per point (Theorem 9), and need not be stored at all, since each embedding row can be formed on the fly from the corresponding distance row. In compute time, a plain distance kernel evaluates each entry from a single precomputed distance in $O ( 1 )$ , whereas the SLE kernel compares two sparse distance profiles in $O ( s ) { \ ; }$ since $s \ll | D |$ and is independent of |D|, this overhead is a small constant factor rather than a change in scaling, as the measured runtimes in Table 4 confirm. Spectral baselines (Riemannian, Geometric) store an $N \times l$ eigenvector matrix in place of a distance matrix, so the memory comparison there is an equal trade instead of an overhead. Table 4 reports wall-clock training (including full hyperparameter optimization) and prediction times, peak memory, and empirical sparsity s¯ for SLE and all baselines on a single CPU.

Table 5 reports runtimes and RAM usage for the 1-dimensional synthetic function shown in Figure 1. The run was set up as follows. The SLE implementation builds a KD-tree over the landmark set once and reuses it across every likelihood evaluation of the MCMC sampler. For each query point, only landmark–query pairs with $d \leq r$ are materialized (cKDTree.sparse\_distance\_matrix), the resulting embedding is stored as a CSR sparse matrix with s¯ nonzeros per row, and the pairwise squared distance is formed with sparse matrix multiplication. The native kernel serves as a reference for the same landmark idea without the bump; distances to all |D| landmarks are computed and stored densely. Kernel timings measure a single $K = k ( X , X )$ evaluation at the initial hyperparameters; training timings measure a full MCMC hyperparameter run (200 samples, identical settings on all three kernels). All values are mean ± standard error over 3 random dataset draws on the same single-CPU host. Empirical scaling $( y \sim N ^ { \alpha } )$ fitted over $N \in \left\lceil 5 0 , 5 0 0 \right\rceil$ : kernel evaluation $\alpha _ { \mathrm { S L E } } = 1 . 7 5$ $\alpha _ { \mathrm { M a t \acute { e } r n } } = 1 . 4 5 , \alpha _ { \mathrm { n a t i v e } } = 2 . 5 1 $ training $\alpha _ { \mathrm { S L E } } = 2 . 1 6$ , α<sub>Matérn</sub> = 1.84, $\alpha _ { \mathrm { n a t i v e } } = 2 . 6 0$

Comparison with the manuscript. Appendix J predicts $\mathcal { O } ( N \cdot | \mathcal { D } | )$ distance work and an $\mathcal { O } ( N ^ { 2 } \bar { s } )$ Gram assembly for SLE versus $\bar { \mathcal { O } } ( N ^ { 2 } )$ for a raw-distance stationary kernel, i.e. a constant-factor overhead instead of a change in scaling order. Consistent with this, the SLE exponent exceeds that of the Matérn baseline by $\Delta \alpha \approx 0 . 3 0$ for kernel evaluation and $\Delta \alpha \approx 0 . 3 2$ for training; both empirical exponents are below their asymptotic ideals of 2 and 3 because at $N \leq 5 0 0$ Python and BLAS fixed overheads still contribute meaningfully to the timings. The native embedding, which materializes the full $N \times | \mathcal { D } |$ distance matrix without sparsification, tracks SLE closely $( \alpha _ { \mathrm { n a t i v e } } = 2 . 5 1 )$ because in this experiment s is comparable to |D|: the reported timings are taken at the initial radius $r = 0 . 1 5$ rather than the MLE-selected radius, so the average sparsity grows as s¯ $\mathit { \bar { \Psi } } \sim N ^ { 1 } \ ( \bar { s } \approx 1 3 . 9$ at $N = 5 0 ;$ $\bar { s } \approx 1 3 5 . 6 \mathrm { a t } N = 5 0 0 )$ ). The manuscript’s $\bar { s } = \mathcal { O } ( 1 )$ regime requires r to shrink with $N$ , which occurs under marginal-likelihood maximization (Appendix D.1) but is not represented here; in that regime, the SLE kernel would separate more markedly from the native baseline. Peak $\Delta { \sf R A M }$ for kernel evaluation scales as $N ^ { 0 . 9 6 ^ { \bullet } } ( \mathrm { S L E } ) , N ^ { 1 . 2 1 }$ (Matérn), and $N ^ { 1 . 9 0 }$ (native).

The native (dense) embedding baseline is fastest for small training sets — at $N = 5 0$ its single-kernel evaluation is roughly 2.5× faster than SLE — because it avoids the KD-tree query, CSR construction, and sparse-matmul overhead of the SLE implementation entirely. However, its kernel-evaluation walltime scales as $N ^ { 2 . 5 1 }$ against SLE’s $N ^ { 1 . 7 5 }$ and Matérn’s $N ^ { 1 . 4 5 }$ , so by $N = 5 0 0$ it becomes roughly 2× slower than SLE and $4 \times$ slower than Matérn, with substantially larger run-to-run variance $( 1 0 7 . 5 \pm 5 8 . 8 $ ms vs. $4 8 . 5 \pm 0 . 7$ ms for $\operatorname { S L E } ) ;$ this crossover directly reflects the dense-embedding pathology that motivates the sparse SLE construction in the first place. The most decisive quantitative gap between the three kernels is memory: peak kernel-evaluation $\Delta { \sf R A M }$ scales as $N ^ { \hat { 1 } . 9 0 }$ for the native embedding versus $N ^ { 0 . 9 6 }$ for SLE and $N ^ { 1 . 2 1 }$ for Matérn, providing direct empirical support for the memory-scaling claim in this section.

Table 4: Wall-clock runtimes (single CPU; distance/eigenpair precomputation reported separately since it is shared or method-specific).
<table><tr><td>Dataset</td><td>Model</td><td>Precompute</td><td>Train</td><td>Predict</td></tr><tr><td rowspan="2">Dragon (n=1000)</td><td>SLE</td><td>12 h (geodesics, shared)</td><td>3-4 min</td><td>0.5 s</td></tr><tr><td>Riemannian (500 ep)</td><td>2–3 min</td><td>2–3 min</td><td>7.5 s</td></tr><tr><td rowspan="2">Teddy Bear (n=400)</td><td>SLE</td><td>1 h (geodesics, shared)</td><td>10.49 s</td><td>0.0821 s</td></tr><tr><td>Geometric Kernel</td><td>1 s</td><td>22.59 s</td><td>0.11 s</td></tr><tr><td rowspan="2">SAXS (n=250)</td><td>SLE – Wass</td><td>1 h on 32 CPUs</td><td>62.73 s</td><td>0.02 s</td></tr><tr><td>Matérn – Sliced Wass</td><td>1 h on 32 CPUs</td><td>4.3 s</td><td>0.01 s</td></tr></table>

Table 5: Wall-clock time and peak resident memory for the proposed Sparse Landmark Embedding (SLE) kernel, a standard Matérn $( \nu = 3 / 2 )$ baseline, and the native (dense) distance-to-landmarks embedding kernel of Eq. (3), on the 1D analytic benchmark of Figure 1.
<table><tr><td colspan="2"></td><td colspan="2">Kernel evaluation</td><td colspan="2">Full training (MCMC)</td><td rowspan="2">s</td></tr><tr><td>|D|</td><td>Model</td><td>Time (ms)</td><td>∆RAM (MiB)</td><td>Time (s)</td><td>∆RAM (MiB)</td></tr><tr><td rowspan="3">50</td><td>SLE</td><td> $1 . 1 9 \pm 0 . 1 4$ </td><td> $1 . 3 6 \pm 0 . 0 1$ </td><td> $0 . 1 4 \pm 0 . 0 1$ </td><td> $2 . 3 2 \pm 0 . 1 1$ </td><td>13.9</td></tr><tr><td>Matérn</td><td> $1 . 1 0 \pm 0 . 0 1$ </td><td> $1 . 1 2 \pm 0 . 0 4$ </td><td> $0 . 1 3 \pm 0 . 0 1$ </td><td> $1 . 4 8 \pm 0 . 0 5$ </td><td>一</td></tr><tr><td>Native</td><td> $0 . 4 6 \pm 0 . 0 1$ </td><td> $0 . 1 4 \pm 0 . 0 4$ </td><td> $0 . 0 7 \pm 0 . 0 0$ </td><td> $2 . 4 3 \pm 0 . 0 5$ </td><td>一</td></tr><tr><td rowspan="3">100</td><td>SLE</td><td> $1 . 4 2 \pm 0 . 0 1$ </td><td> $1 . 5 6 \pm 0 . 0 5$ </td><td> $0 . 1 5 \pm 0 . 0 1$ </td><td> $2 . 6 6 \pm 0 . 0 3$ </td><td>27.2</td></tr><tr><td>Matérn</td><td> $1 . 7 1 \pm 0 . 0 4$ </td><td> $1 . 2 4 \pm 0 . 0 1$ </td><td> $0 . 1 4 \pm 0 . 0 1$ </td><td> $1 . 2 6 \pm 0 . 2 8$ </td><td>一</td></tr><tr><td>Native</td><td> $0 . 8 3 \pm 0 . 0 2$ </td><td> $0 . 4 0 \pm 0 . 0 4$ </td><td> $0 . 1 0 \pm 0 . 0 0$ </td><td> $2 . 6 7 \pm 0 . 0 8$ </td><td>一</td></tr><tr><td rowspan="3">150</td><td>SLE</td><td> $2 . 1 5 \pm 0 . 0 3$ </td><td> $1 . 8 1 \pm 0 . 1 0$ </td><td> $1 . 5 6 \pm 0 . 1 6$ </td><td> $3 . 0 9 \pm 0 . 0 9$ </td><td>40.8</td></tr><tr><td>Matérn</td><td> $2 . 7 7 \pm 0 . 0 1$ </td><td> $2 . 1 5 \pm 0 . 1 3$ </td><td> $1 . 3 9 \pm 0 . 1 8$ </td><td> $1 . 6 3 \pm 0 . 0 9$ </td><td>一</td></tr><tr><td>Native</td><td> $1 . 7 1 \pm 0 . 0 2$ </td><td> $1 . 1 0 \pm 0 . 0 5$ </td><td> $1 . 6 1 \pm 0 . 1 1$ </td><td> $2 . 9 2 \pm 0 . 0 4$ </td><td>一</td></tr><tr><td rowspan="3">200</td><td>SLE</td><td> $4 . 0 1 \pm 0 . 3 4$ </td><td> $3 . 3 4 \pm 0 . 0 4$ </td><td> $2 . 8 5 \pm 0 . 5 5$ </td><td> $4 . 0 8 \pm 0 . 4 2$ </td><td>54.2</td></tr><tr><td>Matérn</td><td> $4 . 6 2 \pm 0 . 0 3$ </td><td> $3 . 6 3 \pm 0 . 0 6$ </td><td> $2 . 2 4 \pm 0 . 2 2$ </td><td> $2 . 8 2 \pm 0 . 6 0$ </td><td>一</td></tr><tr><td>Native</td><td> $4 . 4 1 \pm 0 . 3 3$ </td><td> $2 . 2 1 \pm 0 . 0 1$ </td><td> $2 . 4 5 \pm 0 . 2 9$ </td><td> $3 . 5 8 \pm 0 . 1 1$ </td><td>一</td></tr><tr><td rowspan="3">250</td><td>SLE</td><td> $5 . 3 4 \pm 0 . 0 7$ </td><td> $4 . 2 6 \pm 0 . 0 5$ </td><td> $3 . 0 9 \pm 0 . 3 2$ </td><td> $6 . 4 8 \pm 0 . 0 3$ </td><td>67.5</td></tr><tr><td>Matérn</td><td> $6 . 9 1 \pm 0 . 6 7$ </td><td> $4 . 7 5 \pm 0 . 0 5$ </td><td> $2 . 1 3 \pm 0 . 1 2$ </td><td> $3 . 9 7 \pm 0 . 5 8$ </td><td></td></tr><tr><td>Native</td><td> $7 . 7 6 \pm 1 . 6 6$ </td><td> $3 . 0 5 \pm 0 . 0 3$ </td><td> $3 . 5 3 \pm 0 . 4 4$ </td><td> $6 . 6 9 \pm 0 . 0 5$ </td><td>一</td></tr><tr><td rowspan="3">300</td><td>SLE</td><td> $8 . 5 5 \pm 0 . 7 1$ </td><td> $5 . 0 0 \pm 0 . 2 1$ </td><td> $4 . 7 0 \pm 0 . 8 5$ </td><td> $8 . 2 6 \pm 0 . 2 0$ </td><td>81.1</td></tr><tr><td>Matérn</td><td> $8 . 2 8 \pm 0 . 0 6$ </td><td> $6 . 1 0 \pm 0 . 0 4$ </td><td> $2 . 6 4 \pm 0 . 3 8$ </td><td> $5 . 9 7 \pm 0 . 0 2$ </td><td>一</td></tr><tr><td>Native</td><td> $1 0 . 1 0 \pm 0 . 2 0$ </td><td> $4 . 1 8 \pm 0 . 0 1$ </td><td> $4 . 7 7 \pm 0 . 2 9$ </td><td> $8 . 8 9 \pm 0 . 1 1$ </td><td>一</td></tr><tr><td rowspan="3">350</td><td>SLE</td><td> $2 0 . 1 7 \pm 0 . 1 3$ </td><td> $6 . 1 4 \pm 0 . 1 1$ </td><td> $5 . 0 8 \pm 0 . 6 5$ </td><td> $1 1 . 6 3 \pm 0 . 6 3$ </td><td>94.6</td></tr><tr><td>Matérn</td><td> $1 3 . 2 2 \pm 0 . 1 3$ </td><td> $7 . 9 4 \pm 0 . 0 9$ </td><td> $3 . 5 5 \pm 0 . 1 2$ </td><td> $7 . 8 6 \pm 0 . 0 3$ </td><td></td></tr><tr><td>Native</td><td> $2 1 . 1 0 \pm 4 . 8 9$ </td><td> $5 . 3 1 \pm 0 . 0 4$ </td><td> $8 . 9 8 \pm 1 . 7 4$ </td><td> $1 2 . 3 5 \pm 0 . 0 6$ </td><td>一</td></tr><tr><td rowspan="3">400</td><td>SLE</td><td> $2 7 . 2 6 \pm 0 . 1 6$ </td><td> $7 . 5 3 \pm 0 . 0 5$ </td><td> $1 0 . 4 5 \pm 1 . 1 0$ </td><td> $1 5 . 1 1 \pm 0 . 0 6$ </td><td>108.4</td></tr><tr><td>Matérn</td><td> $1 7 . 3 8 \pm 0 . 6 4$ </td><td> $1 0 . 5 0 \pm 0 . 1 0$ </td><td> $4 . 6 2 \pm 0 . 1 7$ </td><td> $9 . 6 3 \pm 0 . 0 2$ </td><td>一</td></tr><tr><td>Native</td><td> $5 7 . 6 8 \pm 3 3 . 9 4$ </td><td> $6 . 8 6 \pm 0 . 0 4$ </td><td> $1 0 . 4 3 \pm 2 . 3 8$ </td><td> $1 4 . 9 5 \pm 0 . 0 1$ </td><td>一</td></tr><tr><td rowspan="3">450</td><td>SLE</td><td> $3 7 . 0 6 \pm 0 . 3 8$ </td><td> $7 . 9 2 \pm 0 . 2 9$ </td><td> $1 1 . 1 3 \pm 0 . 9 3$ </td><td> $1 7 . 0 6 \pm 0 . 4 7$ </td><td>122.2</td></tr><tr><td>Matérn</td><td> $2 2 . 0 9 \pm 1 . 1 7$ </td><td> $1 1 . 8 3 \pm 0 . 2 7$ </td><td> $5 . 2 7 \pm 0 . 8 4$ </td><td> $1 1 . 9 7 \pm 0 . 0 5$ </td><td></td></tr><tr><td>Native</td><td> $7 9 . 1 7 \pm 4 3 . 0 6$ </td><td> $8 . 3 7 \pm 0 . 0 1$ </td><td> $1 5 . 6 6 \pm 3 . 2 6$ </td><td> $1 8 . 2 0 \pm 0 . 0 3$ </td><td>一</td></tr><tr><td rowspan="3">500</td><td>SLE</td><td> $4 8 . 4 9 \pm 0 . 7 0$ </td><td> $1 0 . 8 7 \pm 0 . 1 0$ </td><td> $1 3 . 0 9 \pm 0 . 5 5$ </td><td> $2 0 . 6 4 \pm 0 . 1 0$ </td><td>135.6</td></tr><tr><td>Matérn</td><td> $2 4 . 9 6 \pm 0 . 0 2$ </td><td> $1 3 . 8 8 \pm 0 . 2 0$ </td><td> $7 . 0 5 \pm 0 . 3 1$ </td><td> $1 4 . 5 5 \pm 0 . 0 8$ </td><td>一</td></tr><tr><td>Native</td><td> $1 0 7 . 4 8 \pm 5 8 . 8 1$ </td><td> $1 0 . 2 3 \pm 0 . 0 4$ </td><td> $1 6 . 8 2 \pm 3 . 5 3$ </td><td> $2 1 . 9 4 \pm 0 . 0 4$ </td><td>一</td></tr></table>