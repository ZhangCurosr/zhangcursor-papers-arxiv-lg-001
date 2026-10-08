# EDGE ACCURACY IS NOT ENOUGH: WHY DYNAMICS-LEARNED STRUCTURE FAILS TO TRANSFER TO INVERSE PROBLEMS

Nicholas Tan Jerome   
Institute for Data Processing and Electronics (IPE)   
Karlsruhe Institute of Technology (KIT)   
Karlsruhe, Germany   
nicholas.tanjerome@kit.edu   
Fangnian Wang   
Institute for Thermal Energy Technology   
and Safety (ITES)   
Karlsruhe Institute of Technology (KIT)   
Karlsruhe, Germany   
fangnian.wang@kit.edu

## ABSTRACT

A natural strategy for inverse problems with scarce labelled data is to transfer relational structure learned from abundant forward-simulation data. We show this strategy fails systematically, even when it satisfies the standard theoretical justification for why structure should help. We prove that approximate structure provides estimation-error benefits whenever the edge error satisfies $\Delta < n ^ { 2 } - k n$ (Theorem 3), reducing sample complexity from O(n<sup>2</sup>) to $O ( k n + \Delta )$ . Structure learned via Neural Relational Inference (NRI) from dynamics prediction satisfies this condition, yet on a source-localisation task across 180 CFD-simulated hydrogen-leak scenarios and 180 acoustic scenarios, it degrades performance by 116% and 201% respectively relative to a flexible, task-optimised attention baseline, while a physics-based prior (Green’s function) degrades by only 69–72%. Four independent lines of evidence show this is not a tuning failure: NRI improves only 0.5% when given 18× more training data (versus 16.6% for the taskoptimised baseline, p < 0.001); performance is insensitive to the NRI edge threshold across a wide range; the dynamics-learned graph overlaps the task-optimal graph on only 6% of edges (Jaccard similarity); and two further dynamics-derived structure estimators (correlation- and mutual-information-based) show no measurable benefit over a structure-free baseline, with the correlation-based estimator performing markedly worse. We formalise this gap as a statement about approximation error that the edge-accuracy condition cannot control, and we provide a lightweight transferability test (Jaccard similarity against a partially-observed target-task graph) that separates successful from failed transfer in all four domain/structure pairs we evaluate, using under an hour of computation and 15– 20% of target-domain data; we present this as a promising heuristic calibrated on a small number of cases rather than a validated general threshold.

## 1 INTRODUCTION

Inverse problems — inferring causes from observed effects — are pervasive in scientific computing, but training data for the inverse task is often expensive or dangerous to collect, while forwardsimulation data is comparatively cheap. This motivates a recurring idea across scientific machine learning: learn relational structure from abundant forward dynamics, then reuse that structure to constrain the inverse model. The idea is intuitive — if two sensors co-vary under the governing physics, their relationship should also constrain where an unobserved source could be — and it underlies pretraining strategies from molecular property prediction (Zhu et al., 2022) to gene regulatory network inference (Theodoris et al., 2023).

We test this idea on source localisation, a canonical inverse problem, using hydrogen leak detection in underground parking facilities as our primary case study, a safety-critical sensor-placement problem also studied via CFD-informed optimisation (Wang et al., 2026): forward CFD simulation is hours per scenario, but the underlying transport dynamics are comparatively easy to generate, making this an attractive setting for forward-to-inverse structure transfer. We find that the strategy fails, and the failure is not explained by insufficient data or poor hyperparameter choices.

Our theoretical starting point is unsurprising: known sparse structure reduces sample complexity from $O ( n ^ { 2 } ) { \mathrm { ~ t o ~ } } O ( k n ) $ (Theorem 1), and this benefit degrades gracefully under edge-level errors as long as $\Delta < n ^ { 2 } -$ kn (Theorem 3). The surprising part is empirical: structure learned by Neural Relational Inference (NRI) from forward dynamics satisfies this condition comfortably, yet it underperforms even a fixed, physics-agnostic baseline. We show through data-scaling, thresholdablation, and direct structural comparison that this is because the edge-accuracy condition bounds estimation error, not approximation error — and approximation error is what dominates when the forward and inverse tasks have different optimal structures.

Contributions. (1) We prove a robustness condition under which approximate structure still reduces sample complexity (Theorem 3), and show empirically that satisfying it is insufficient when the structure-learning objective is misaligned with the downstream task. (2) Through four independent lines of evidence across two physically distinct domains (CFD hydrogen dispersion and acoustic propagation) and three different dynamics-derived structure estimators, we show dynamicslearned structure fails to transfer while flexible, task-optimised attention and physics-based priors both succeed to different degrees. (3) We give a practical transferability test, based on Jaccard over lap between candidate structure and a partially-observed target-task graph, that separates successful from failed transfer in all four domain/structure pairs we evaluate, using a fraction of the target data and under an hour of compute; given the small number of pairs, we present the specific $J \geq 0 . 3 0$ threshold as an initial calibration rather than a validated general rule (Section 5.4).

## 2 RELATED WORK

Source localization. Sparse-sensor source localization is addressed by two largely separate literatures. Physics-based approaches combine a forward dispersion or transport model with sparse observations, typically via Bayesian source-term estimation, to infer an unknown source’s location and strength (Hirst et al., 2013) — the same governing-physics strategy our Green’s Function baseline uses directly rather than learning it. A separate, more recent literature instead learns the sensorreading-to-location mapping end-to-end from data, predominantly with deep networks operating on raw or transformed sensor signals (Grumiaux et al., 2022), analogous to our Learned Attention baseline. Neither line of work tests whether a graph structure learned from a distinct forward-dynamics objective — rather than the physics itself, and rather than the task’s own labels — can substitute for either strategy; that substitution is what we evaluate.

Structure transfer in scientific ML. Learning interaction graphs from forward dynamics to support downstream inference is common in molecular modelling (Zhu et al., 2022), systems biology (Theodoris et al., 2023), and elsewhere, but the conditions under which such structure transfers to a downstream inverse task are rarely characterised directly; most prior work either assumes transfer succeeds or evaluates it only on tasks similar to the structure-learning objective itself.

Neural Relational Inference. NRI (Kipf et al., 2018) learns a latent interaction graph from observed dynamics via a graph-structured VAE, and has been extended to dynamic graphs (Graber & Schwing, 2020), more efficient message passing (Chen et al., 2021), and partial observability (Sun et al., 2019). We take NRI as our primary object of study, rather than a hand-crafted heuristic, because it is the dynamics-derived estimator whose edge error is provably low enough to satisfy Theorem 3’s condition (Section 4): if edge-accuracy alone were sufficient for transfer, NRI is precisely where that should show, making it the sharpest available test rather than a straw man chosen for being easy to beat. None of the extensions above evaluate whether the learned graph transfers to a task with a different objective than dynamics prediction; that is the question we study.

Sample complexity of structured models. Our theoretical results build on Rademacher complexity bounds for structured hypothesis classes (Bartlett & Mendelson, 2002; Wainwright, 2019; Mohri et al., 2018). We apply this machinery to attention-based localisers and use it to make precise what edge-level structure error does and does not control.

## 3 THEORETICAL ANALYSIS

Setup. n sensors at positions $P = \{ p _ { 1 } , . . . , p _ { n } \}$ in a bounded domain $\Omega \subset \mathbb { R } ^ { 3 }$ observe a source $s \sim P _ { S }$ through readings $r _ { i } = c ( p _ { i } , s ) + \epsilon _ { i }$ , with ϵ sub-Gaussian noise. An attention-based localiser predicts $\begin{array} { r } { \hat { s } ( r , \mathsf { \bar { P } } ) = \sum _ { i = 1 } ^ { \mathsf { \bar { m } } } } \end{array}$ sof $\operatorname { m a x } ( \operatorname { s c o r e } _ { j } ) \cdot q _ { j }$ over m query points, where score<sub>j</sub> $\in \mathbb { R } ^ { n }$ is any bounded, architecture-specific compatibility function between query j and the n sensor encodings (e.g. scaled dot-product $q _ { i } ^ { \top } k _ { i } / \sqrt { d }$ for Learned Attention, or a fixed kernel of sensor–query distance for Green’s Function; see Section 4), with attention weights $\alpha _ { j i } ~ \in ~ [ 0 , 1 ]$ summing to one over sensors. A structure graph $G = ( V , E )$ restricts $\alpha _ { j i } > 0 \mathrm { ~ t o ~ } ( i , \dot { j } ) \in E .$ with maximum in-degree (sparsity) k. The unstructured class $\mathcal { H } _ { \mathrm { f u l l } }$ has effective dimension $O ( n ^ { 2 } )$ (for $m = \Theta ( n )$ query points); a structured class $\mathcal { H } _ { G }$ has dimension $| E | \leq k n$

Assumption 1 (Boundedness). There exists $B > 0$ such that $\| s \| \leq B$ for all s in the support of $P _ { S } ,$ , and $\| \hat { s } ( r , P ) \| \le B f o r$ every localiser sˆ in every hypothesis class considered $( \mathcal { H } _ { \mathrm { f u l l } } , \ \mathcal { H } _ { G } ,$ , and $\mathcal { H } _ { \tilde { G } } )$

Remark 1 (Scope of the theory across architectures). Theorems $_ { I - 3 }$ are statedfor attention-based localisers, but the underlying argument is a dimension-counting one: any hypothesis class whose effective degrees offreedom shrinkfrom $O ( n ^ { 2 } ) t o O ( k n + \Delta )$ under a sparsity mask G will inherit the same estimation-versus-approximation-error decomposition, regardless of whether the masked component is attention, a masked MLP, or a sparse message-passing layer. What is specific to attention is the exact constant in the Rademacher-complexity bound (Appendix $A ) _ { \cdot }$ the qualitative claim — that edge-accuracy controls estimation error but not approximation error — is not. We do not prove this extension for other architectures here andflag it as future work (Discussion), but we do not expect the negative result to be an artefact ofthe attention parameterisation specifically.

Theorem 1 (Sample complexity with known structure). For a structure graph with sparsity $k ,$ with probability at least $1 ~ - ~ \delta$ over m i.i.d. samples, $\begin{array} { r l } { L ( \hat { s } _ { G } ) \ - \ \operatorname* { i n f } _ { \hat { s } \in \mathcal { H } _ { G } } L ( \hat { s } ) } & { { } \leq } \end{array}$ $\begin{array} { r } { O \bigg ( B ^ { 2 } \sqrt { \frac { k n \log m } { m } } + B ^ { 2 } \sqrt { \frac { \log ( 1 / \delta ) } { m } } \bigg ) } \end{array}$ , where L is the expected squared localisation error and B is the boundedness constant ofAssumption 1. Consequently, achieving ε-optimal excess risk requires $m = \Omega ( k n / \varepsilon ^ { 2 } )$ with structure versus $\Omega ( n ^ { 2 } / \varepsilon ^ { 2 } )$ without it — an improvement of $\Theta ( n / k )$

Assumption 2 (Restricted eigenvalue). Let $\hat { \Sigma }$ denote the empirical Gram matrix of the sensor time series usedfor structure recovery. There exists $\kappa > 0$ such that min $| S | \leq 2 k , v \neq 0  v ^ { \top } \hat { \Sigma } v / \| v _ { S } \| _ { 2 } ^ { 2 } \geq \kappa ,$ where the minimum is over index sets S of size at most 2k and $v _ { S }$ denotes the restriction of v to S. This is the standard condition under which LASSO recovers sparse support in high-dimensional linear regression (Wainwright, 2019).

Theorem 2 (Structure learnability from dynamics). Let $G ^ { \star }$ be the true k-sparse interaction graph on n nodes. (a) Any algorithm recovering $G ^ { \star }$ requires Ω(k log n) samples, even noiselessly. (b) For linear dynamics $X _ { t + 1 } = A ^ { \star } X _ { t } + \eta _ { t }$ with supp $\mathbf { \partial } ^ { \prime } A ^ { \star } ) = \mathbf { \bar { G } } ^ { \star }$ , under Assumption 2, LASSO recovers $G ^ { \star }$ exactly with probability 1 − δ from $O ( k \log ( n / \delta ) )$ samples.

Theorem 1 assumes the optimal localiser actually lies in ${ \mathcal { H } } _ { G } ;$ when it does not, an additional approximation error term $\begin{array} { r } { \operatorname* { i n f } _ { \hat { s } \in \mathcal { H } _ { G } } L ( \hat { s } ) - \operatorname* { i n f } _ { \hat { s } } L ( \hat { s } ) } \end{array}$ appears in the excess risk relative to the unrestricted optimum, and this bound does not control it. Theorem 2 establishes that structure can be learned efficiently from dynamics — so the obstacle we study below is not learnability, but whether the structure learned this way is the structure the inverse task needs. We next ask: how much does the benefit of Theorem 1 survive when G is only approximately known?

Theorem 3 (Robustness to approximate structure). Let $G ^ { \star }$ be the true structure with sparsity $k ,$ and $\tilde { G }$ a learned approximation with edge error $\Delta = | E ( \tilde { G } ) \triangle E ( G ^ { \star } )$ |. Then (i) dim $( \mathcal { H } _ { { \tilde { G } } } ) \leq k n + \Delta ,$ (ii) the sample complexity for ε-optimal localisation over $\mathcal { H } _ { { \tilde { G } } } i s O ( ( k n + \Delta ) / \varepsilon ^ { 2 } )$ ; and (iii) structure learning helps over the unstructured baseline whenever $\bar { \Delta < } n ^ { 2 } - k n$

For any $k < n$ this permits $\Delta = \Omega ( n ^ { 2 } )$ edge errors while still improving over the baseline $- \mathrm { ~ a ~ }$ generous tolerance. Critically, however, Theorem 3 bounds estimation error only.

Remark 2 (Insufficiency of edge-level accuracy). Let $\tilde { G } _ { \mathrm { d y n } }$ be structure learned from a dynamics objective and $G _ { \mathrm { l o c } } ^ { \star }$ the optimal graph for localisation. Even when $\Delta = | E ( \tilde { G } _ { \mathrm { d y n } } ) \triangle E ( G _ { \mathrm { l o c } } ^ { \star } ) |$ is small, the approximation-error gap in $\begin{array} { r } { \dot { \hat { s } } \in \mathcal { H } _ { \tilde { G } _ { \mathrm { d y n } } } \ L ( \hat { s } ) - \operatorname* { i n f } _ { \hat { s } \in \mathcal { H } _ { G _ { \mathrm { l o c } } ^ { \star } } } L ( \hat { s } ) } \end{array}$ can be large whenever the two tasks have different optimal structures. This gap vanishes only when $E ( \tilde { G } _ { \mathrm { d y n } } )$ and $E ( G _ { \mathrm { l o c } } ^ { \star } )$ are close in a task-relevant sense, not merely an edge-count sense.

Conjecture 1 (Approximation error dominance). When $\tilde { G }$ is learned from an objective misaligned with the inverse task, approximation error dominates estimation error regardless of ∆, sample size, or threshold selection, whenever the forward and inverse tasks have fundamentally different optimal structures $G _ { \mathrm { d y n } } ^ { \star } \neq G _ { \mathrm { l o c } } ^ { \star } .$

We do not prove Conjecture 1 in general: a proof would require a closed-form characterisation of both $G _ { \mathrm { d y n } } ^ { \star }$ and $G _ { \mathrm { l o c } } ^ { \star } .$ , which is not available for neural structure-learning objectives. Instead, Section 5 provides four independent empirical tests designed to rule out the alternative explanations (insufficient data, mistuned threshold, insufficient sparsity, architecture-specific quirk) that a defender of the transfer strategy could otherwise offer. Full proofs of Theorems 1–3 and 2 are in Appendix A.

## 4 METHODS

We compare four localiser architectures sharing identical inputs (sensor readings $r \in \mathbb { R } ^ { n }$ , positions $P \in \mathbb { R } ^ { n \times 3 } )$ and outputs $( \hat { s } \in \mathbb { R } ^ { 3 } )$ , with $m = 6 4$ query points:

MLP (no structure): a four-layer MLP over concatenated readings and positions — unlike the other three, it computes no sensor-query attention at all, so there is no mask to speak of. “No structure” refers to the sparsity mask defined in Section 3, not to model capacity; MLP represents $\mathcal { H } _ { \mathrm { f u l l } }$ , the baseline these masks are compared against.

Green’s Function: physics-based attention $\alpha _ { j i } \propto \exp ( - \| p _ { i } - q _ { j } \| ^ { 2 } / 2 R _ { D } ^ { 2 } )$ with $R _ { D } = 4 \mathbf { m }$ , the diffusion length scale. Fixed, sparse, and grounded in the governing transport physics.

Learned Attention: standard scaled dot-product attention with learned query/key projections. Despite having $O ( n d )$ projection parameters, any attention pattern is expressible, so this remains in $\mathcal { H } _ { \mathrm { f u l l } }$ with $\bar { O } ( n ^ { 2 } / \varepsilon ^ { 2 } )$ sample complexity.

NRI-Graph: an NRI encoder (Kipf et al., 2018) trained on the 60-timestep concentration sequence to predict edge probabilities $p ( e _ { i j } )$ between sensors and query points, minimising dynamics reconstruction loss plus an edge-sparsity penalty. The learned graph $\tilde { G } = \{ ( i , j ) : p ( e _ { i j } ) > \tau \} ( \tau = 0 . 3 ,$ selected by edge F1 on held-out dynamics data) constrains attention: $\alpha _ { j i } \propto p ( e _ { i j } )$ for $( i , j ) \in \tilde { G } , 0$ otherwise. Represents $\mathcal { H } _ { \tilde { G } }$ with dim $= O ( k n )$

NRI-Graph receives the most scrutiny in our experiments because it is the only method that can satisfy Theorem 3’s condition $( \Delta < \grave { n ^ { 2 } } - k n )$ while potentially suffering large approximation error per Remark 2 — making it the sharpest test of whether edge-accuracy is sufficient in practice. Two further dynamics-derived structure estimators (correlation- and mutual-information-based) are introduced in Section 5 as an architecture-independence check. Full architectural and training details for all methods are in Appendix B.

## 5 EXPERIMENTS

Domains. CFD: an underground parking facility $( 5 0 \times 3 0 \times 3 \mathrm { m } )$ with 180 simulated hydrogenleak scenarios (12 leak positions $\times ~ 5$ leak rates × 3 ventilation levels) (Wang et al., 2026), governed by buoyancy-driven incompressible Navier–Stokes transport with k-ε turbulence, validated against published release experiments to within ±10% (Xiao et al., 2017; Hu et al., 2021). Acoustic: 180 synthetic scenarios in a 2D room governed by a screened-Poisson (Helmholtz-type) equation. We treat CFD and acoustic as independent tests, not two variants of one task, because they differ along the axes that determine whether structure transfer could succeed: governing physics (parabolic advection–diffusion transport with turbulence versus an elliptic wave equation), source signature (a persistent, buoyancy-driven plume versus an instantaneous, oscillatory point emission), and noise/sensing characteristics (concentration sensors with slow, spatially-correlated drift versus pressure sensors with fast, largely independent readings). Any explanation of the results that appeals to a shared quirk of a single physical system would have to hold across both of these axes simultaneously. All localisers use $n = 1 5$ sensors, $m = 6 4$ queries, and are evaluated over 10 seeds. Full simulation and protocol details are in Appendix C.

Table 1: Structure comparison, $n = 1 5$ sensors, mean ± std over 10 seeds. The ranking (Attention > Green’s > NRI > MLP) is preserved across two domains governed by different physics.
<table><tr><td>Localiser</td><td></td><td>CFD RMSE (m) ↓ Acoustic RMSE (m) ↓</td></tr><tr><td>MLP (no structure)</td><td> $1 4 . 5 7 \pm 0 . 2 4$ </td><td> $4 . 3 5 \pm 0 . 0 7$ </td></tr><tr><td>Green&#x27;s Function</td><td> $8 . 9 6 \pm 0 . 0 7$ </td><td> $1 . 6 1 \pm 0 . 0 5$ </td></tr><tr><td>Learned Attention</td><td> ${ \bf 5 . 2 1 \pm 0 . 2 2 }$ </td><td> $\mathbf { 0 . 9 5 \pm 0 . 0 4 }$ </td></tr><tr><td>NRI-Graph</td><td> $1 1 . 2 6 \pm 0 . 1 0$ </td><td> $2 . 8 6 \pm 0 . 0 5$ </td></tr></table>

Table 2: Data scaling: RMSE (m) as a function of training scenarios m (CFD domain). NRI-Graph’s improvement is an order of magnitude smaller than Attention’s despite identical data growth.
<table><tr><td>m</td><td>MLP</td><td>Green&#x27;s</td><td>Attention</td><td>NRI-Graph</td></tr><tr><td>10</td><td> $1 5 . 9 4 \pm 1 . 2 8$ </td><td> $8 . 6 3 \pm 0 . 3 8$ </td><td> $6 . 2 6 \pm 0 . 5 4$ </td><td> $1 1 . 3 2 \pm 0 . 6 5$ </td></tr><tr><td>180</td><td> $1 4 . 5 7 \pm 0 . 2 4$ </td><td> $8 . 9 6 \pm 0 . 0 7$ </td><td> $5 . 2 1 \pm 0 . 2 2$ </td><td> $1 1 . 2 6 \pm 0 . 1 0$ </td></tr><tr><td>Improvement</td><td> $8 . 6 \%$ </td><td> $- 3 . 9 \%$ </td><td> $1 6 . 6 \%$ </td><td> $0 . 5 \% ^ { * * * }$ </td></tr></table>

<sup>∗∗∗</sup>NRI-Graph’s per-seed % improvement differs from Learned Attention’s at $p < 0 . 0 0 1$ (paired t-test,  
t(9) = 6.05). Full six-point scaling curve in Appendix C.

## 5.1 STRUCTURE COMPARISON FAILS TO FAVOUR DYNAMICS-LEARNED STRUCTURE

Table 1 shows Learned Attention achieves the best RMSE in both domains, beating Green’s Function by 42% and NRI-Graph by 54% in CFD, with the same ranking replicated in acoustic localisation (0.95 m versus 2.86 m, a 201% degradation for NRI). NRI-Graph improves over the unstructured MLP by only 23% in CFD, far short of the 3.5–5× sample-efficiency gain Theorem 3 would predict if the learned structure were well-aligned with the task.

## 5.2 FOUR LINES OF EVIDENCE AGAINST A TUNING EXPLANATION

The central question is whether NRI-Graph’s poor performance reflects estimation error (fixable with more data or a better threshold) or approximation error (not fixable, per Remark 2). These two explanations make different, testable predictions.

Evidence 1: data scaling. If NRI’s structure were merely noisily estimated, more dynamics training data should narrow the gap. Table 2 shows the opposite: with 18× more scenarios (10 → 180), NRI-Graph improves only 0.5%, while Learned Attention improves 16.6% (paired t-test on per-seed % improvement, $t ( 9 ) = 6 . 0 5 , p < 0 . 0 0 1 )$ and even the unstructured MLP improves 8.6%. A flat scaling curve under increasing data is the signature of approximation error, not estimation error.

Evidence 2: threshold insensitivity. If the failure were due to a mistuned edge threshold τ, performance should vary meaningfully across τ. Instead, $\tau \in [ 0 . 2 , 0 . 5 ]$ — spanning a wide range of graph density — yields an identical $1 1 . 4 9 \pm 0 . 1 1 \mathrm { m } .$ , and even the F1-optimal threshold $\tau = 0 . 1$ differs from this plateau by less than 0.1% relative. Denser top-k graphs (top-2, top-4 connectivity) degrade performance further, the opposite of what partial structural benefit would predict (full table in Table 8, Appendix C).

Evidence 3: structural overlap. We use Learned Attention’s graph as a proxy for the (unobservable) task-optimal structure $G _ { \mathrm { l o c } } ^ { \star } \colon$ it is unconstrained by any structural prior and empirically achieves the best RMSE in both domains (Table 1), so the spatial-influence pattern it converges to is the best available estimate of what structure the localisation task actually needs, independent of what any dynamics-learning objective produces. We extract binary adjacency structures directly from trained models and compute Jaccard similarity $J ( A , B ) = | \bar { A } \cap \mathsf { \bar { B } } | / | \bar { A } \cup B |$ against this proxy graph. NRI’s learned graph overlaps it on only $J \approx 0 . 0 \dot { 6 }$ of edges (CFD) and $\dot { J } \approx 0 . 1 9$ (acoustic), versus J ≈ 0.31 (CFD) and $J \approx 0 . 4 7$ (acoustic) for Green’s Function. Across both domains, structures achieving $J \geq 0 . 3 0$ correspond to $< 1 0 0 \%$ performance degradation from optimal, while $J < 0 . 2 0$ corresponds $\mathrm { t o } > 1 0 0 \%$ degradation (Table 3) — a pattern we exploit directly in Section 5.4.

Table 3: Transfer outcomes vs. estimated structural similarity to the task-optimal graph. A $J \geq 0 . 3 0$ threshold separates successful from failed transfer in all four cases.
<table><tr><td>Domain</td><td>Structure</td><td>RMSE</td><td>Degradation</td><td>Est. J</td></tr><tr><td rowspan="2">CFD</td><td>Green&#x27;s Function</td><td>8.96m</td><td>72%</td><td> $\approx 0 . 3 1$ </td></tr><tr><td>NRI Learned Graph</td><td>11.26m</td><td>116%</td><td> $\approx 0 . 0 6$ </td></tr><tr><td rowspan="2">Acoustic</td><td>Green&#x27;s Function</td><td>1.61 m</td><td>69%</td><td> $\approx 0 . 4 7$ </td></tr><tr><td>NRI Learned Graph</td><td>2.86 m</td><td>201%</td><td> $\approx 0 . 1 9$ </td></tr></table>

Table 4: Two additional dynamics-learned structure baselines (CFD domain, $n = 1 5 ,$ mean ± std over 10 seeds), neither derived from NRI. Correlation-Graph clearly underperforms the unstructured MLP; Mutual-Information-Graph is statistically indistinguishable from it. Neither provides a measurable benefit, ruling out an NRI-architecture-specific explanation for why dynamics-derived structure fails to help.
<table><tr><td>Method</td><td> $\mathrm { R M S E } \left( \mathrm { m } \right) \downarrow$ </td><td>Sparsity</td><td>Est. J</td></tr><tr><td>MLP (no structure)</td><td> $1 4 . 5 7 \pm 0 . 2 4$ </td><td></td><td>0.00</td></tr><tr><td>NRI-Graph</td><td> $1 1 . 2 6 \pm 0 . 1 0$ </td><td>97%</td><td>≈ 0.06</td></tr><tr><td>Correlation-Graph (best τ)</td><td> $1 8 . 6 1 \pm 0 . 0 1$ </td><td>5%</td><td>≈1.00</td></tr><tr><td>Mutual-Information-Graph (best pct.)</td><td> $1 4 . 4 8 \pm 0 . 0 0$ </td><td>80%</td><td>≈ 0.37</td></tr></table>

Evidence 4: the failure is not NRI-specific. A remaining objection is that NRI’s particular architecture, not the dynamics-learning objective in general, is responsible for the failure. We test this with two additional dynamics-learned structure baselines: a graph built from pairwise Pearson correlation between sensor–query time series, and one built from pairwise mutual information (via k-NN density estimation). Both are constructed purely from co-variation in the dynamics data, with no NRI-style learned encoder. Table 4 shows Correlation-Graph clearly underperforms the unstructured MLP (18.61 m versus 14.57 m), while Mutual-Information-Graph (14.48 m) is statistically indistinguishable from it — dynamics-derived structure here provides no measurable benefit even in the one case where it does not actively hurt. Correlation-Graph’s near-total edge overlap with the task-optimal proxy graph $( J \approx 1 . 0 0 )$ reflects its near-complete connectivity (only 5% sparsity, versus NRI’s 97%) rather than genuine structural alignment, and it still fails to help despite this overlap; Mutual-Information-Graph’s overlap $( J \approx 0 . 3 7 )$ exceeds the $J = 0 . 3 0$ transferability threshold used elsewhere in this paper, yet also yields no measurable improvement. This indicates that threshold, calibrated on Green’s Function and NRI-Graph alone (Section 5.4), does not straightforwardly extend to these two estimators, and that structural overlap without appropriately-scaled sparsity is not on its own informative about transfer.

Together, these four lines of evidence rule out the explanations a defender of forward-to-inverse structure transfer could otherwise offer (insufficient data, bad threshold, insufficient sparsity, architecture-specific quirk), and are consistent with Conjecture 1: the dynamics-learning objective converges to a graph that is accurate for predicting concentration evolution but poorly aligned with the spatial-influence structure that localisation requires.

## 5.3 SENSOR-COUNT SCALING CONFIRMS STRUCTURE ENABLES SCALABILITY

A further test of whether NRI-Graph’s structure is doing any real work: does it scale gracefully with more sensors, the way genuine structural constraints should? Table 5 shows the unstructured MLP degrades as n grows from 10 to 20 sensors in both domains (more parameters to estimate, same data), while all three structured methods improve — including NRI-Graph. This confirms NRI-Graph’s hypothesis class is genuinely constrained (consistent with Theorem 3’s estimationerror guarantee), even though that constraint is the wrong one for the task: NRI improves with scale but never closes the gap to Learned Attention.

Table 5: Sensor-count scaling. The unstructured MLP degrades with more sensors in both domains; all structured methods improve, confirming that NRI-Graph’s hypothesis class is genuinely sparsityconstrained even though the constraint is misaligned with the task.
<table><tr><td>n</td><td colspan="2">CFD RMSE (m) MLP NRI-Graph</td><td colspan="2">Acoustic RMSE (m) MLP NRI-Graph</td></tr><tr><td>10</td><td>14.02</td><td>12.37</td><td>3.60</td><td>3.43</td></tr><tr><td>20</td><td>15.17</td><td>10.53</td><td>5.19</td><td>2.41</td></tr><tr><td>Change</td><td>+8%</td><td>-15%</td><td>+44%</td><td>-30%</td></tr></table>

## 5.4 A PRACTICAL TRANSFERABILITY TEST

The pattern in Table 3 suggests a simple a priori test: (1) hold out 15–20% of target-task data; (2) train a minimal localiser on it; (3) extract its task-specific attention graph $\hat { G } _ { B } ;$ (4) compute $J ( { \tilde { G } } , { \hat { G } } _ { B } )$ for the candidate structure ${ \tilde { G } } ;$ (5) reject transfer if $\textit { J } < \ 0 . 3 0$ This procedure requires under an hour of computation and correctly identifies transfer viability in all four evalu ated domain/structure pairs (2 domains × {Green’s Function, NRI-Graph}). We emphasise that the $J \geq 0 . 3 0$ threshold is calibrated on the cases available here; broader validation across more transfer pairs is future work, and we report it as a practical heuristic rather than a proven guarantee.

## 6 DISCUSSION

Why forward and inverse objectives diverge. Forward dynamics optimises for temporal coherence — which variables co-evolve — while inverse localisation optimises for spatial constraint — which observations most restrict the location of an unseen cause. These are different questions even when they concern the same physical system, and nothing in the dynamics objective rewards a graph for being useful at the second question. This is consistent with our results: Green’s Function, derived directly from the spatial-influence structure of the transport equation rather than from dynamics, achieves the best non-learned transfer and the best out-of-distribution robustness (−0.7% acoustic OOD gap; Appendix B), because its structure is task-relevant by construction rather than incidentally.

Practical guidance. Use physics-based structure when reliable domain knowledge constrains the inverse task directly (not just the forward dynamics) and data is limited; use flexible, task-specific learning when > 10 scenarios per sensor are available and accuracy is paramount; only attempt structure transfer when $J \ge 0 . 3 0$ against a partially observed target-task graph, since satisfying the edge-accuracy condition of Theorem 3 alone is not sufficient — noting that this threshold is calibrated specifically on physics-based and NRI-learned structure (Section 5) and does not straightforwardly extend to other dynamics-derived estimators.

Limitations. Both domains are simulated; real-world validation with sensor deployments is the most direct way to test whether these conclusions hold outside simulation, and is the focus of on going work. Our theorems are proven for attention-based localisers; while Remark 1 argues the qualitative estimation/approximation decomposition should extend to other masked architectures, we have not proven this formally, and empirical confirmation on at least one non-attention architecture (e.g., a masked MLP) would strengthen the claim considerably. Finding a stylised setting (e.g., linear dynamics with a linear localiser) where Conjecture 1 can be proven rather than only tested empirically remains the most promising direction for closing the gap between Theorem 3 and practice.

## 7 CONCLUSION

Relational structure provably reduces sample complexity for inverse localisation, and this benefit degrades gracefully under edge-level structure error (Theorem 3). But satisfying that condition is not sufficient: structure learned from a forward dynamics objective can satisfy it while still failing to transfer, because the condition bounds estimation error and says nothing about task-objective alignment. Across two physically distinct domains, four independent lines of evidence — flat data scaling, threshold insensitivity, low structural overlap, and the failure or absence of benefit of two further dynamics-derived estimators — converge on the same explanation, and a simple Jaccardbased test built on this insight correctly predicts transfer outcomes for the four domain/structure pairs it was calibrated on (Section 5.4), though it does not straightforwardly extend to the two additional dynamics-derived estimators (Section 5). We expect this gap between edge-accuracy and task-alignment to recur in any forward-to-inverse transfer setting where the two objectives are not explicitly designed to share optimal structure.

## AI USE STATEMENT

In this work, we used generative AI tools to draft initial implementations of the localiser baselines, structure estimators, and simulation pipelines; to assist with post-hoc debugging of the experiment pipeline, including identifying a methodological issue in how one baseline’s evaluation was constructed; to draft interpretive prose describing what the results show; to draft and edit portions of the manuscript; and to source and summarise related literature. All AI-assisted code, debugging, interpretation, and literature suggestions were independently verified by the authors against the underlying data and original sources before inclusion, including independently re-running experiments to confirm the results in Tables 1–8. We did not use generative AI to develop the theoretical models, formulate or prove the mathematical claims (Theorems 1–3, Conjecture 1), propose or refine hypotheses, design the research methodology or experiments, generate or clean data, support qualitative data analysis, create or modify figures, suggest experimental parameters, or propose the paper’s structure or title; these, and the final interpretation of all findings, remain the authors’ own responsibility. Assistance with translation is not applicable. We take responsibility for the final content of this work, including text, claims, and artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

Code, NRI checkpoints, and experiment scripts are available at https://github.com/ nicolaisi/edge-accuracy-is-not-enough. CFD scenario data are available on Zenodo in four parts (part 1, part 2, part 3, part 4); the acoustic domain is fully reproducible from a provided generator script (under 15 seconds per full 180-scenario sweep on a single CPU core). Full proofs of all theoretical results are in Appendix A; complete architectural, training, and dataset details are in Appendices B–C.

## ETHICS STATEMENT

This work supports hydrogen safety monitoring and acoustic source localisation. We do not identify a path by which our specific contributions (sample-complexity theory and a structure-transferability diagnostic) facilitate harmful surveillance beyond what is already enabled by source-localisation methodology in general use.

## REFERENCES

Peter L. Bartlett and Shahar Mendelson. Rademacher and Gaussian complexities: Risk bounds and structural results. Journal ofMachine Learning Research, 3:463–482, 2002.

Siyuan Chen, Jiahai Wang, and Guoqing Li. Neural relational inference with efficient message passing mechanisms. Proceedings ofthe AAAI Conference on Artificial Intelligence, 35(8):7055– 7063, 2021. doi: 10.1609/aaai.v35i8.16868.

Colin Graber and Alexander G. Schwing. Dynamic neural relational inference. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8513–8522, June 2020. doi: 10.1109/CVPR42600.2020.00854.

Pierre-Amaury Grumiaux, Srdjan Kitic, Laurent Girin, and Alexandre Gu´ erin. A survey of sound´ source localization with deep learning methods. The Journal of the Acoustical Society of America, 152(1):107–151, 2022.

Bill Hirst, Philip Jonathan, Fernando Gonzalez del Cueto, David Randell, and Oliver Kosut. Locating and quantifying gas emission sources using remotely obtained concentration data. Atmospheric Environment, 74:141–158, 2013.

G. Hu, F. Wang, Q. Ba, J. Xiao, and T. Jordan. Numerical investigation of light gas release, stratification and dissolution in TH22 test facility using 3-D CFD code GASFLOW-MPI. International Journal of Hydrogen Energy, 46(46):23974–23987, 2021.

Thomas Kipf, Ethan Fetaya, Kuan-Chieh Wang, Max Welling, and Richard Zemel. Neural relational inference for interacting systems. In Jennifer Dy and Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 2688–2697. PMLR, 2018.

Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar. Foundations of Machine Learning. MIT Press, 2nd edition, 2018.

Chen Sun, Per Karlsson, Jiajun Wu, Joshua B. Tenenbaum, and Kevin Murphy. Stochastic prediction of multi-agent interactions from partial observations. In International Conference on Learning Representations (ICLR), 2019.

Christina V. Theodoris, Ling Xiao, Anant Chopra, Mark D. Chaffin, Zeina R. Al Sayed, Matthew C. Hill, Helene Mantineo, Elizabeth M. Brydon, Zexian Zeng, X. Shirley Liu, and Patrick T. Ellinor. Transfer learning enables predictions in network biology. Nature, 618(7965):616–624, 2023.

Martin J. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2019.

Fangnian Wang, Nicholas Tan Jerome, Thomas Jordan, and Frank Simon. Optimizing sensor placement for hydrogen leak detection in enclosed infrastructure: A comparative study using cfdinformed genetic algorithm and deepsets neural surrogate, 2026. URL https://arxiv.org/ abs/2607.26078.

J. Xiao, W. Breitung, M. Kuznetsov, H. Zhang, J. R. Travis, R. Redlinger, and T. Jordan. GASFLOW-MPI: A new 3-D parallel all-speed CFD code for turbulent dispersion and combustion simulations: Part I: Models, verification and validation. International Journal of Hydrogen Energy, 42(12):8346–8368, 2017.

Jingxuan Zhu, Juexin Wang, Weiwei Han, and Dong Xu. Neural relational inference to learn longrange allosteric interactions in proteins from molecular dynamics simulations. Nature Communications, 13:1661, 2022.

## A PROOFS OF THEORETICAL RESULTS

## A.1 PROOF OF THEOREM 1

Let $\hat { s } _ { G } = \mathrm { a r g }$ min $\boldsymbol { 1 } _ { \hat { s } \in \mathcal { H } _ { G } } \hat { L } ( \hat { s } )$ denote the empirical risk minimiser (ERM), and let $\begin{array} { r l } { \hat { s } _ { G } ^ { \star } } & { { } = } \end{array}$ arg min<sub>sˆ∈</sub> $\mathcal { H } _ { G } \ : L ( \hat { s } )$ the population risk minimiser. The excess risk decomposes as

$$
\begin{array} { r } { L ( \hat { s } _ { G } ) - L ( \hat { s } _ { G } ^ { \star } ) = \big ( L ( \hat { s } _ { G } ) - \hat { L } ( \hat { s } _ { G } ) \big ) + \big ( \hat { L } ( \hat { s } _ { G } ) - \hat { L } ( \hat { s } _ { G } ^ { \star } ) \big ) + \big ( \hat { L } ( \hat { s } _ { G } ^ { \star } ) - L ( \hat { s } _ { G } ^ { \star } ) \big ) . } \end{array}
$$

By definition of the ERM the middle term is non-positive, so $\begin{array} { r } { L ( \hat { s } _ { G } ) - L ( \hat { s } _ { G } ^ { \star } ) \leq 2 \operatorname* { s u p } _ { \hat { s } \in \mathcal { H } _ { G } } | L ( \hat { s } ) - } \end{array}$ $\hat { L } ( \hat { s } ) |$

By standard uniform-convergence bounds for bounded losses (e.g. Theorem 3.3 of Mohri et al., 2018), with probability at least $1 - \delta$

$$
\operatorname* { s u p } _ { \hat { s } \in \mathcal { H } _ { G } } \left. L ( \hat { s } ) - \hat { L } ( \hat { s } ) \right. \leq 2 \Re _ { m } ( \mathcal { L } \circ \mathcal { H } _ { G } ) + B ^ { 2 } \sqrt { \frac { \log ( 2 / \delta ) } { 2 m } } ,
$$

where $\Re _ { m }$ is the empirical Rademacher complexity and $\mathcal { L }$ is the squared-loss class. Since $\| s \| , \| \hat { s } ( r , P ) \| \le B$ for all $s \in \Omega$ and all localisers in the class (Assumption 1), $\lVert \hat { s } ( r , P ) - s \rVert \leq$

2B, and the squared loss $\ell ( z ) ~ = ~ z ^ { 2 }$ is 4B-Lipschitz on $[ 0 , 2 B ] ;$ by the contraction lemma, $\Re _ { m } ( \mathcal { L } \circ \mathcal { H } _ { G } ) \overset {  } { \leq } 4 B \Re _ { m } ( \mathcal { H } _ { G } )$

$\mathcal { H } _ { G }$ is parametrised by attention weights supported on $E ( G )$ , with effective dimension $d \_ =$ dim $( \mathcal { H } _ { G } ) = \vert E ( G ) \vert = ^ { \cdot } O ( k n )$ . By covering-number bounds for bounded hypothesis classes (Theorem 4.14 of Wainwright, 2019),

$$
\begin{array} { r } { \Re _ { m } ( \mathcal { H } _ { G } ) \leq B \sqrt { \frac { 2 d \log ( e m / d ) } { m } } \leq B \sqrt { \frac { 2 k n \log m } { m } } , } \end{array}
$$

where the final inequality holds in the regime $m \gg$ kn where learning is feasible at all (if $m \textless$ < kn even the structured class cannot be learned). Combining the three displays gives

$$
\begin{array} { r } { L ( \hat { s } _ { G } ) - L ( \hat { s } _ { G } ^ { \star } ) \leq 8 B ^ { 2 } \sqrt { \frac { 2 k n \log m } { m } } + 2 B ^ { 2 } \sqrt { \frac { \log ( 2 / \delta ) } { 2 m } } = O \bigg ( B ^ { 2 } \sqrt { \frac { k n \log m } { m } } \bigg ) . } \end{array}
$$

## A.2 PROOF OF THEOREM 2

Part (a) (lower bound). The class of k-sparse graphs on n nodes has cardinality $\begin{array} { r } { | \mathcal { G } _ { k } | \ge \binom { n ^ { 2 } } { k } / k ! \ge } \end{array}$ $( n ^ { 2 } / k ) ^ { k } / k ! \geq ( n / k ) ^ { 2 k }$ for $k \leq n$ , so log $| \mathcal { G } _ { k } | \ge 2 k \log ( n / k ) = \Omega ( k \log n )$ for $k = { O } ( n / \log n )$ By Fano’s inequality, distinguishing among $\left| \mathcal { G } _ { k } \right|$ hypotheses with probability at least $1 / 2$ requires mutual information $\begin{array} { r } { I ( X _ { 1 : m } ; G ^ { \star } ) \geq \frac { 1 } { 2 } \log \bar { | \mathcal { G } _ { k } | } - \bar { 1 } = \Omega ( k \log n ) } \end{array}$ . Under bounded signal-to-noise ratio, each sample carries $O ( 1 )$ bits, so $m = \Omega ( k \log n )$ samples are necessary.

Part (b) (LASSO achievability). For each node $j ,$ the dynamics $\begin{array} { r } { X _ { t + 1 , j } = \sum _ { i } A _ { i i } ^ { \star } X _ { t , i } + \eta _ { t , j } } \end{array}$ form a sparse linear regression with design matrix $X _ { t }$ and k-sparse coefficient vector $\check { \beta } _ { j } ^ { \star } = A _ { j , { \ i } } ^ { \star }$ <sub>:</sub>. Under Assumption 2 (restricted eigenvalue, constant κ), LASSO recovers supp $( \hat { \beta } _ { j } ) \ = \ \operatorname { s u p p } ( \beta _ { j } ^ { \star } )$ with probability at least $1 - \delta / n$ from m $\ge C k \log ( n / \delta ) / \kappa ^ { 2 }$ samples (Theorem $7 . 1 9$ of Wainwright, 2019). A union bound over all n nodes gives $\begin{array} { r } { \operatorname* { P r } ( \hat { G } \neq G ^ { \star } ) \leq \sum _ { j = 1 } ^ { n } \operatorname* { P r } ( \operatorname { s u p p } ( \hat { \beta } _ { j } ) \neq \operatorname { s u p p } ( \beta _ { j } ^ { \star } ) ) \leq } \end{array}$ $\boldsymbol { n } \cdot \delta / n = \delta . \boldsymbol { \mathbf { \mathit { \Theta } } }$

## A.3 PROOF OF THEOREM 3

Part (i). Decompose $| E ( \tilde { G } ) | = | E ( \tilde { G } ) \cap E ( G ^ { \star } ) | + | E ( \tilde { G } ) \setminus E ( G ^ { \star } ) | \leq | E ( G ^ { \star } ) | + | E ( \tilde { G } ) \setminus E ( G ^ { \star } ) | \leq$ $k n + \Delta$ , using $| E ( G ^ { \star } ) | \leq k n$ (sparsity k) and $| E ( \tilde { G } ) \setminus E ( G ^ { \star } ) | \leq \Delta$ (spurious edges are a subset of the symmetric difference). Since $\mathcal { H } _ { \tilde { G } }$ has one parameter per edge, dim $. ( \mathcal H _ { { \tilde { G } } } ) = | E ( { \tilde { G } } ) | \leq k n + \Delta$

Part (ii). Substituting dim $. ( \mathcal { H } _ { { \tilde { G } } } ) \leq k n + \Delta$ into the Rademacher-complexity bound from the proof of Theorem $, \Re _ { m } ( \mathcal { H } _ { \tilde { G } } ) \leq B \sqrt { 2 ( k n + \Delta ) }$ log m/m, and following the same argument, $L ( \hat { s } _ { \tilde { G } } ) -$ $\begin{array} { r } { \operatorname* { i n f } _ { \hat { s } \in \mathcal { H } _ { \tilde { G } } } L ( \hat { s } ) \leq O \big ( B ^ { 2 } \sqrt { ( k n + \Delta ) \log m / m } \big ) } \end{array}$ . Solving for m: $m _ { \mathrm { l o c } } = O ( ( k n + \Delta ) / \varepsilon ^ { 2 } )$

Part (iii). Structure learning improves over the unstructured baseline $( \dim ( \mathcal { H } _ { \mathrm { f u l l } } ) = O ( n ^ { 2 } ) )$ precisely when kn $+ \Delta < n ^ { 2 } , \bar { \mathrm { i . e . } } \ \dot { \Delta } < n ^ { 2 } - k n = n ( n - k ) . |$ 1

Corollary (end-to-end complexity). Combining dynamics-learning and localisation sample requirements, $m _ { \mathrm { t o t a l } } = m _ { \mathrm { d y n } } \dot { + } O \big ( ( k n + \Delta ( m _ { \mathrm { d y n } } ) \big ) / \varepsilon ^ { 2 } \big )$ , where $\Delta ( m _ { \mathrm { d y n } } )$ is the structure error achieved with $m _ { \mathrm { d y n } }$ dynamics samples. For linear dynamics where LASSO achieves $\Delta = 0$ with $m _ { \mathrm { d y n } } = { \cal O } ( k \log n )$ (Theorem 2b), $m _ { \mathrm { t o t a l } } = { O } ( k n / \varepsilon ^ { 2 } )$ , strictly better than $O ( n ^ { 2 } / \varepsilon ^ { 2 } )$ for sparse graphs.

## B IMPLEMENTATION DETAILS

## B.1 LOCALISER ARCHITECTURES

All four localisers share input format $( r \in \mathbb { R } ^ { n } , P \in \mathbb { R } ^ { n \times 3 } )$ and output format $( \hat { s } ~ \in ~ \mathbb { R } ^ { 3 } )$ , with $m = 6 4$ query points.

MLP. Concatenates $( r _ { 1 } , p _ { 1 } , \ldots , r _ { n } , p _ { n } ) \in \mathbb { R } ^ { 4 n }$ ; four-layer MLP $\nu [ 4 n  1 2 8  6 4  3 2  3 ]$ with ReLU activations, regressing directly to sˆ.

Green’s Function. Sensor encoding via a two-layer MLP mapping $( r _ { i } , p _ { i } ) \to h _ { i } \in \mathbb { R } ^ { 6 4 }$ ; attention $\alpha _ { j i } = \underline { { \mathrm { e x p } } } ( - \| p _ { i } - q _ { j } \| ^ { 2 } / 2 R _ { D } ^ { 2 } ) / \sum _ { i ^ { \prime } } ^ { } \exp ( - \| p _ { i ^ { \prime } } - q _ { j } \| ^ { 2 } / 2 R _ { D } ^ { 2 } )$ with $R _ { D } = 4$ m; query aggregation $\begin{array} { r } { z _ { j } = \sum _ { i } \alpha _ { j i } h _ { i } ; } \end{array}$ output head is a two-layer MLP with soft-argmax over $m = 6 4$ queries. $R _ { D }$ is selected from facility geometry and hydrogen dispersion scales, giving average degree $k \approx 5 . 7$

Learned Attention. Sensor encoding as above; query/key/value projections $q _ { j } = W _ { Q } e _ { j } + b _ { Q }$ $k _ { i } = W _ { K } h _ { i } + b _ { K } , v _ { i } = W _ { V } h _ { i } + b _ { V } \in \mathbb { R } ^ { 6 4 }$ (with $e _ { j }$ a learned query-point embedding); scaled dot-product attention $\alpha _ { j i } = \exp ( q _ { j } ^ { \top } k _ { i } / \sqrt { 6 4 } ) / \sum _ { i ^ { \prime } } \exp ( q _ { j } ^ { \top } k _ { i ^ { \prime } } / \sqrt { 6 4 } )$ ; aggregation and output head as above.

NRI-Graph. NRI encoder: two-layer GRU $( d _ { \mathrm { h i d d e n } } = 6 4 )$ for node embeddings from the 60- timestep sequence, two-layer MLP on concatenated node-embedding pairs for edge probabilities $p ( e _ { i j } ) \ = \ \mathrm { s o f t m a x } ( \mathrm { L i n e a r } ( \mathrm { M L P } ( [ z _ { i } ; z _ { j } ] ) ) )$ . Training minimises reconstruction loss $\mathbb { E } \Vert \hat { X } _ { t + 1 } ~ -$ $X _ { t + 1 } | | _ { 2 } ^ { 2 }$ plus an edge-sparsity penalty $\lambda \mathbb { E } [ p ( e _ { i j } = 1 ) ]$ with $\lambda = 0 . 0 1$ , for 50 epochs with Adam $( \mathrm { l r } = 1 0 ^ { - 3 } )$ . Threshold $\tau = 0 . 3$ selected by maximising edge F1 on held-out dynamics data; edges with $p ( e _ { i j } ) < \tau$ contribute $< 0 . 1 \%$ attention mass in practice. Localiser attention: $\alpha _ { j i } \propto p ( e _ { i j } )$ for $( i , j ) \in \tilde { G } , 0$ otherwise; sensor encoding, query aggregation, and output head as in Green’s Function.

## B.2 CORRELATION- AND MUTUAL-INFORMATION-BASED STRUCTURE ESTIMATORS

Correlation-Graph. Edges constructed from pairwise Pearson correlation $\begin{array} { r l } { \rho _ { i j } } & { { } = } \end{array}$ $\mathrm { C o v } ( X _ { 1 : T , i } , X _ { 1 : T , j } ) / \sqrt { \mathrm { V a r } ( X _ { 1 : T , i } ) \mathrm { V a r } ( X _ { 1 : T , j } ) }$ between sensor–query time series, with $( i , j ) \in { \tilde { G } }$ $\mathrm { i f f } \ | \rho _ { i j } | > \tau _ { \mathrm { c o r r } } . \ \mathrm { \ w e \ s w e e p \ } \tau _ { \mathrm { c o r r } } \in \{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 \} ;$ results in Table 4 report the best of these $( \tau _ { \mathrm { c o r r } } = 0 . 1 \colon \mathrm { R M S E } \ 1 8 . 6 1 \pm 0 . 0 1$ , average degree 14.2, sparsity 5%; $\tau _ { \mathrm { c o r r } } \stackrel { - } { = } 0 . 2  \colon 3 1 . 0 7 \pm 4 . 8 1 ;$ $\tau _ { \mathrm { c o r r } } = 0 . 3 \colon 2 6 . 2 7 \pm 4 . 8 6 ; \tau _ { \mathrm { c o r r } } = 0 . 4 \colon 2 4 . 1 2 \pm 0 . 0 9 )$

Mutual-Information-Graph. Edges constructed from pairwise mutual information via k-nearest-neighbours density estimation $\begin{array} { r l r l r l } { ( k } & { { } } & { = } & { { } } & { 5 } & { { } } \end{array}$ neighbours, sklearn.feature selection.mutual info regression), with edges retained above a percentile threshold of the MI distribution. We sweep the 60th/70th/80th percentile; Table 4 reports the best (80th percentile: RMSE $1 4 . 4 8 \pm 0 . 0 0$ , average degree 3.0, sparsity 80%; 70th percentile: $2 1 . 4 7 \pm 3 . 8 8$ ; 60th percentile: $2 5 . 5 6 \pm 1 . 5 0 )$

Both estimators use the same NRI encoder inputs (the 60-timestep concentration sequence) and the same downstream localiser architecture as NRI-Graph, isolating the structure-estimation method as the only varying factor.

## B.3 OUT-OF-DISTRIBUTION GENERALISATION

We additionally evaluate OOD generalisation by training on a subset of operating conditions and testing on a held-out condition: for CFD, training on ventilation rates $\mathrm { A } \bar { \mathrm { C H } } \in \bar { \left\{ 6 , 1 0 \right\} } \mathrm { h } ^ { - 1 }$ (120 scenarios) and testing on $\mathrm { A C H } = 3 \mathrm { h } ^ { - 1 }$ (60 scenarios); for acoustic, training on low/mid absorption and testing on high absorption. Table 6 reports train/test RMSE and the relative gap.

Table 6: Out-of-distribution generalisation. Green’s Function shows near-perfect acoustic robustness, consistent with its structure being derived directly from the governing physics rather than from data.
<table><tr><td></td><td colspan="3">CFD</td><td colspan="3">Acoustic</td></tr><tr><td>Localiser</td><td>Train</td><td>Test</td><td>Gap</td><td>Train</td><td>Test</td><td>Gap</td></tr><tr><td>Learned Attention</td><td>5.45</td><td>5.67</td><td> $+ 4 . 1 \%$ </td><td>1.05</td><td>1.15</td><td> $+ 1 0 . 1 \%$ </td></tr><tr><td>Green&#x27;s Function</td><td>8.88</td><td>9.44</td><td> $+ 6 . 3 \%$ </td><td>1.62</td><td>1.61</td><td>-0.7%</td></tr><tr><td>NRI-Graph</td><td>11.48</td><td>11.08</td><td> $- 3 . 4 \%$ </td><td>3.27</td><td>3.20</td><td>-2.0%</td></tr><tr><td>MLP (no structure)</td><td>15.39</td><td>16.92</td><td>+10.0%</td><td>4.78</td><td>5.03</td><td>+5.2%</td></tr></table>

Learned Attention generalises best in CFD, where the underlying physics is complex enough that a flexible, data-driven structure adapts better than a fixed prior. Green’s Function generalises best in the acoustic domain, where the diffusion-kernel structure closely matches the governing screened-Poisson equation. NRI-Graph’s small negative OOD gaps are consistent with a regularisation effect from its restrictive (if misaligned) structure, but this does not compensate for its poor absolute performance.

## C CFD AND ACOUSTIC SIMULATION DETAILS

## C.1 CFD DOMAIN

Facility geometry. An underground parking garage (Wang et al., 2026), 50 $\mathrm { m } \times 3 0 \mathrm { m } \times 3$ m (length × width × height), with two longitudinal lanes (40 m × 5 m) at $y = 7 . 5 , 2 2 . 5$ m and two transverse lanes (20 m × 5 m) at $x = 7 . 5 , 4 2 . 5 \mathrm { { m } ; }$ structural columns $( 0 . 5 \mathrm { m } \times 0 . 5 \mathrm { m } )$ on a 6 m × 6 m grid.

Ventilation. Mechanical ventilation supplied at one corner, exhausted at the diagonally opposite corner; fresh-air mass flow at 90% of exhaust to maintain slight negative pressure; six ceilingmounted jet fans $( 1 0 \mathrm { m / s }$ exit velocity). Three ventilation rates: $\mathrm { A C H } \in \{ 3 , 6 , 1 0 \} \mathrm { h } ^ { - 1 }$ (minimal code-compliant, typical accident-response, emergency-mode).

Governing equations. Three-dimensional incompressible Navier–Stokes, $\partial u / \partial t + ( u \cdot \nabla ) u =$ $- \rho ^ { - 1 } \nabla p \dot { + } \nu \mathbf { \dot { \nabla } } ^ { 2 } u + g .$ , coupled to a species-transport equation for hydrogen mass fraction; buoyancy via the Boussinesq approximation; k-ε turbulence with standard wall functions. The solver (GASFLOW) is validated against published hydrogen-release experiments in confined geometries to within ±10% (Xiao et al., 2017; Hu et al., 2021).

Mesh independence. Baseline cell size 0.25 m; a representative scenario $( \mathrm { A C H } = 6 \mathrm { h } ^ { - 1 }$ , leak rate 50 g/s) shows 3.6% relative error for a medium mesh (288K cells) versus a fine mesh (972K cells, 13 h CPU), versus 58.3% for a coarse mesh (36K cells, 0.25 h). The medium mesh is selected.

Scenario matrix. $1 2 \times 5 \times 3 = 1 8 0$ scenarios: 12 leak positions (four corner, four wall-adjacent, four mid-row, at z = 0.5 m fuel-tank height) × 5 leak rates $( 1 , 3 0 , 5 0 , 1 0 0 , 1 5 0 \mathrm { g } / \mathrm { s } ) \times 3$ ventilation levels. Each simulation runs 10 min under $\mathrm { A C H } = 3$ to establish a quiescent field, then the leak activates for 60 s with fields saved at 1 s intervals.

## C.2 ACOUSTIC DOMAIN

Governing equation. Advection-diffusion-reaction, $\partial u / \partial t = D \nabla ^ { 2 } u - v \cdot \nabla u + S ( x , y ) - \alpha u$ with $D = \mathrm { { \bar { 0 . 8 m ^ { 2 } / s } } }$ , background drift $v = ( 0 . 5 , 0 . 2 ) \mathrm { m } / \mathrm { s } ,$ and bulk absorption α. The steady-state field satisfies the screened-Poisson equation $( - D \nabla ^ { 2 } + v \cdot \nabla + \alpha ) u _ { s s } = S$ , solved via sparse linear algebra.

Geometry. 20 m × 15 m 2D room, 100 × 75 grid; two 1 m × 1 m obstacles as minor obstructions, analogous to the CFD facility’s columns; absorbing-wall and zero-flux-at-obstacle boundary conditions.

Scenario matrix. $1 2 \times 5 \times 3 = 1 8 0$ scenarios: 12 source positions (matching the CFD topology) × 5 source powers × 3 absorption coefficients $\alpha \in \{ 0 . 0 5 , 0 . 2 0 , 0 . 5 0 \}$ (analogous to $\mathrm { A C H } \in \{ 3 , \bar { 6 } , 1 0 \} )$ Each scenario produces 60 timesteps of transient build-up plus a steady-state field; the full 180- scenario sweep generates in ≈11 s on a single CPU core.

## C.3 TRAIN/VALIDATION/TEST PROTOCOL

All 180 scenarios pooled with an 80/10/10 split stratified over leak rate/source power and ventilation/absorption level, re-drawn for each of 10 seeds. For the data-scaling experiment (Table 2), a stratified subset of $m \in \{ 1 0 , 2 5 , 5 0 , 1 0 0 , 1 5 0 , 1 8 0 \}$ } scenarios is drawn for training with the remainder as test, averaged over 10 random draws per m. The full six-point curve (extending Table 2) is given in Table 7:

## C.4 THRESHOLD AND CONNECTIVITY ABLATION

The full threshold and top-k ablation summarised in Evidence 2 (Section 5) is given in Table 8:

Table 7: Full data-scaling curve (CFD domain), RMSE (m) as a function of training scenarios m, mean ± std over 10 seeds.
<table><tr><td>m</td><td>MLP</td><td>Green&#x27;s</td><td>Attention</td><td>NRI-Graph</td></tr><tr><td>10</td><td> $1 5 . 9 4 \pm 1 . 2 8$ </td><td> $8 . 6 3 \pm 0 . 3 8$ </td><td> $6 . 2 6 \pm 0 . 5 4$ </td><td> $1 1 . 3 2 \pm 0 . 6 5$ </td></tr><tr><td>25</td><td> $1 5 . 5 4 \pm 0 . 4 6$ </td><td> $8 . 0 2 \pm 0 . 2 8$ </td><td> $5 . 2 6 \pm 0 . 3 9$ </td><td> $1 1 . 3 2 \pm 0 . 2 9$ </td></tr><tr><td>50</td><td> $1 7 . 8 4 \pm 0 . 4 7$ </td><td> $9 . 6 1 \pm 0 . 2 1$ </td><td> $5 . 7 6 \pm 0 . 2 1$ </td><td> $1 2 . 1 0 \pm 0 . 2 7$ </td></tr><tr><td>100</td><td> $1 6 . 6 8 \pm 0 . 3 8$ </td><td> $9 . 3 5 \pm 0 . 1 7$ </td><td> $5 . 7 5 \pm 0 . 1 6$ </td><td> $1 1 . 5 8 \pm 0 . 1 9$ </td></tr><tr><td>150</td><td> $1 5 . 0 8 \pm 0 . 2 0$ </td><td> $9 . 2 5 \pm 0 . 1 5$ </td><td> $5 . 4 5 \pm 0 . 1 7$ </td><td> $1 1 . 6 1 \pm 0 . 1 5$ </td></tr><tr><td>180</td><td> $1 4 . 5 7 \pm 0 . 2 4$ </td><td> $8 . 9 6 \pm 0 . 0 7$ </td><td> $5 . 2 1 \pm 0 . 2 2$ </td><td> $1 1 . 2 6 \pm 0 . 1 0$ </td></tr></table>

Table 8: NRI-Graph ablation over edge threshold τ and top-k connectivity (CFD domain). Performance is flat across $\tau \in [ 0 . 2 , 0 . 5 ]$ and degrades under denser top-k connectivity, ruling out a simple sparsity-tuning explanation.
<table><tr><td>Configuration</td><td>Avg. degree</td><td>Sparsity</td><td>RMSE (m)</td></tr><tr><td> $\tau = 0 . 1$ </td><td>1.1</td><td>97%</td><td> $1 1 . 5 0 \pm 0 . 0 7$ </td></tr><tr><td> $\tau \in [ 0 . 2 , 0 . 5 ]$ </td><td>1.0</td><td>97%</td><td> $1 1 . 4 9 \pm 0 . 1 1$ </td></tr><tr><td> $\tau = 0 . 7$  (disconnected)</td><td>0.0</td><td>100%</td><td> $1 7 . 4 2 \pm 0 . 1 3$ </td></tr><tr><td>Top-2 connectivity</td><td>2.0</td><td>94%</td><td> $1 4 . 4 2 \pm 0 . 2 0$ </td></tr><tr><td>Top-4 connectivity</td><td>4.0</td><td>88%</td><td> $1 3 . 7 9 \pm 0 . 1 6$ </td></tr></table>

The wide plateau across $\tau \in [ 0 . 2 , 0 . 5 ]$ , the negligible difference at the F1-optimal threshold $( \tau =$ 0.1, within 0.1% relative of the plateau), and the degradation under denser top-k graphs together indicate that which edges are retained, not how many, is the binding constraint — consistent with an approximation-error rather than estimation-error explanation (Remark 2).