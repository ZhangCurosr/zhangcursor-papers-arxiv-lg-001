# Spectral Allocation: Why Muon Outperforms Adam, and How to Improve Muon

Xiaodong Wu<sup>∗</sup>   
Department of Engineering   
University of Cambridge xw338@cam.ac.uk

Chao Zhang Department of Electronic Engineering Tsinghua University

Wenyi Yu Department of Electronic Engineering Tsinghua University

Philip Woodland Department of Engineering University of Cambridge

## Abstract

Orthogonal optimisers such as Muon can substantially accelerate large language model pretraining relative to Adam, yet the mechanism remains incompletely understood. We investigate this through an out-of-sample spectral probing analysis of Transformer loss landscapes. At checkpoints along real training trajectories, we decompose each momentum buffer into its singular directions and estimate the loss-optimal step size along each direction on held-out data. The resulting spectral profile is anisotropic yet stable across batches and training stages, and consistent across the optimisers and model scales: a volatile head operating at the Edge-of-Stability supports a much smaller step size than the tolerant bulk, which permits substantially larger steps. This profile provides a unified spectral allocation account of why Muon outperforms Adam, which outperforms SGD. It also exposes a limitation of Muon’s uniform scaling: it still underutilises the bulk. Guided by this finding, we introduce Spectral-Aware Muon (SAMuon), which holds the head at the Muon scale and amplifies the bulk using a static spectral prior. We provide two variants: the complete SAMuon follows the measured profile using a low-rank randomised SVD and the simplified SAMuon-lite uses a two-level approximation via rank-one power iteration. Neither method adds persistent optimiser state or notable extra FLOPs beyond Muon at scale, and the idealised exact-whitening versions of both retain Muon’s asymptotic convergence rate under standard assumptions. Across “modded-nanogpt” models from 124M to 1B parameters, both variants outperform tuned AdamW and Muon (Scion implementation) baselines in all evaluated model-scale and batch-size configurations. SAMuon requires 13.3%–24.0% fewer training tokens to reach the same validation loss as Muon, while SAMuon-lite retains most of this gain with near-zero wall-clock overhead.

## 1 Introduction

The pretraining of Transformer-based decoder-only Large Language Models (LLMs) [Radford et al., 2019] is constrained by the efficiency of optimisers like Adam [Kingma and Ba, 2015]. Recently, orthogonal optimisers such as Muon [Jordan et al., 2024b, Liu et al., 2025] have emerged as a promising research direction. By whitening the (momentum) gradient matrix to generate updates, these methods achieve clear convergence acceleration compared to Adam, particularly during the transient phase (the early, non-asymptotic stage) of pretraining [Wen et al., 2026].

The empirical success of orthogonal optimisers has prompted a number of theoretical explanations, yet the true source of their fast convergence remains an open question. The dominant view frames Muon as spectral-norm steepest descent [Bernstein and Newhouse, 2024, Chen et al., 2026, Bernstein and Newhouse, 2025, Kovalev, 2025], which accounts for its stability and learning-rate transfer but, as Lau et al. [2025] note, does not by itself explain the observed convergence advantage on highly curved pretraining objectives, since whitening “effectively removes all curvature information”. A more recent line of explanation turns directly to curvature: Su [2025] showed that orthogonalisation is loss-optimal under an isotropic curvature model, and concurrent work by Wang et al. [2026] attributes the advantage of Muon over Adam and gradient descent to Muon’s lower directional-sharpness penalty. These insights are real progress. However, the isotropic curvature model [Su, 2025] is, as our measurements show, violated in practice. Meanwhile, the analysis in Wang et al. [2026] relies on an aggregate, in-sample Hessian diagnostic (computed only on the gradient batch) which compresses the update’s entire spectral geometry into a single scalar and consequently misses the crucial curvature information hidden in the full spectrum. In this paper, we directly probe the behaviour of the full spectrum of the momentum buffer on held-out batches, the exact behaviour we aim to improve in practical training scenarios. This allows us to ask the key question that previous work leaves open: what is the curvature behaviour along each spectral direction of the momentum buffer in real Transformer training, and what does this imply for optimiser design? Answering it gives a spectral lens on the loss landscape and a unified view for understanding and improving optimisers.

Our paper is based on an out-of-sample spectral probing analysis. Probing the singular basis of the momentum buffer that orthogonal optimisers whiten, we measure the optimal step size that the loss landscape of held-out batches allows along each singular direction. The resulting loss-optimal step profile is highly anisotropic and stable across different batches, training stages and model scales. The profile is highly forgiving across the tolerant bulk and drops sharply near a single volatile head, with the decline being close to linear in log rank over the leading ranks. Viewed in this way, the major optimiser families fall on a single axis: momentum-based stochastic gradient descent (SGD) allocates step size in proportion to each singular value, exactly opposite to what the landscape suggests; the coordinate-wise rescaling of Adam dampens but cannot overcome this misallocation; and Muon, by whitening to a uniform scale, reallocates step mass from the head into the more tolerant bulk. This yields, to our knowledge, the first unified Spectral Allocation account of why Muon outperforms Adam, which in turn outperforms SGD in Transformer pretraining.

In addition, the same view suggests that there is still room for improvement in the uniform scaling of Muon. The dominant singular direction, termed the Volatile Head, carries the majority of the gradient signal yet allows the smallest step size. It sits at the “Edge-of-Stability” (EoS) [Cohen et al., 2021] and its optimal step size coincides with the set global learning rate. Meanwhile, the remaining Tolerant Bulk allows step sizes several times larger and is mostly flat beyond the leading few ranks. Therefore, the uniform spectral allocation of Muon realises only a small fraction of the per-iteration loss reduction that an idealised, per-rank-optimal spectral allocation would permit (the right panel of Figure 1). This loss reduction could likely be recovered by amplifying the bulk while holding the head at the Muon scale.

To make use of this potential improvement without incurring significant computational overhead, we propose Spectral-Aware Muon (SAMuon). Rather than learning the profile online like SOAP, which applies Adam in Shampoo’s eigenbasis [Vyas et al., 2025], SAMuon imposes a static, head-anchored prior on the whitened update. Two algorithms hold the volatile head at the Muon scale that limits the step and increase the scale of the rest of the spectrum. Two variants of the SAMuon algorithm are provided: the complete SAMuon algorithm employs a profile that closely follows the empirically measured log-rank-linear shape through a rank-k randomised singular value decomposition (SVD) of the momentum buffer at each update step; the simplified SAMuon-lite algorithm keeps a deliberately coarse two-level profile which amplifies the whole bulk uniformly to scale γ while pinning only the head using power iteration at each step. Both variants add no persistent optimiser state beyond the single buffer of Muon and negligible extra floating-point operations (FLOPs). Note that both methods introduce only a single tuned scale hyperparameter γ; setting γ = 1 recovers Muon for both variants.

We validate both SAMuon variants on “modded-nanogpt” [Jordan et al., 2024a] pretraining following the protocol in Pethick et al. [2025], across models from 124M to 1B parameters and batch sizes from 1024 to 4096. Hyperparameters are tuned per batch size at 124M and transferred across model scales. The 124M budget is ∼4× the Chinchilla-optimal allocation [Hoffmann et al., 2022], placing training in the over-trained regime where optimiser gains are hardest to retain [Wen et al., 2026]. By better exploiting the tolerant bulk, the complete SAMuon improves on AdamW and the Muon (Scion implementation) baseline in each trained model and batch size combination, reaching the same final validation loss as Muon with 13.3%–24.0% fewer tokens. SAMuon-lite recovers most of this improvement at nearly the same wall-clock speed as Muon due to the more efficient implementation.

## Our contributions:

1) Out-of-sample spectral probing and a unified account of optimiser performance. To answer the question of why the uniform whitening of Muon works so well with the loss landscape in Transformer pretraining, we introduce an out-of-sample spectral probing framework which measures the optimal step size that the Transformer loss landscape tolerates along each singular direction of the momentum buffer. It identifies a training-stable structure: a profile that is mostly flat across the bulk and drops sharply near a single volatile head. Viewed as a spectral allocation, this gives a unified account of why the performance of Muon exceeds that of Adam, which in turn exceeds that of SGD. It also provides insights into how to improve Muon.

2) The SAMuon optimisers. Based on the findings of the spectral probing analysis, we propose two variants of Spectral-Aware Muon (SAMuon), which distil this measured structure into two static, head-anchored priors on the spectrum of the momentum buffer. SAMuon follows the measured log-rank-linear shape via a rank-k randomised SVD at each update step, and SAMuon-lite keeps a two-level simplification via power iteration at each step, with nearly the same wall-clock speed as Muon. Neither adds persistent optimiser state over Muon. Both add only one additional tunable hyperparameter and negligible extra FLOPs.

3) Empirical and theoretical validation. On the “modded-nanogpt” pretraining benchmark, using models from 124M to 1B parameters and matched hyperparameter tuning for all methods, both proposed variants improve over AdamW and the strong Muon (Scion [Pethick et al., 2025]) baseline in each trained model and batch size combination. SAMuon shows a 13.3%–24.0% token-efficiency improvement over Muon, and SAMuon-lite matches or nearly matches this performance with a token-efficiency improvement of 13.3%–22.1%. A convergence guarantee covering the idealised exact-whitening versions of both variants is also provided.

## 2 Related work

Theoretical Interpretations of Muon’s Success. The prevailing view of Muon is norm-geometric, i.e. orthogonalisation is the steepest-descent step under the spectral norm. This view is developed along three threads: as a norm/duality derivation of the update [Bernstein and Newhouse, 2024, 2025, Jordan et al., 2024b, Large et al., 2024, Yang et al., 2023, Chen et al., 2026], as a linear-minimisation-oracle or non-Euclidean trust-region step with attendant convergence guarantees [Pethick et al., 2025, Kovalev, 2025, Li and Hong, 2025] and as an implicit bias towards spectral-norm max-margin solutions [Fan et al., 2025]. Collectively, these works establish that orthogonalisation is the steepest-descent step for the linearised loss under a constraint on the update’s spectral norm. Such spectral-norm control yields width- and depth-robust update scales and learning-rate transfer, while ensuring that the resulting iterates carry a characterisable spectral-norm max-margin bias. What they do not address is the curvature of the landscape inside the norm ball, which is the most direct link to the training acceleration. Su [2025] makes this missing premise explicit, showing that gradient orthogonalisation is loss-optimal under an isotropic curvature model in which all post-whitening directions are equally easy to descend, an assumption our measurements show is violated in real Transformer training. Most recently, and arguably closest to ours, the concurrent work of Wang et al. [2026] shows that the advantage of Muon over Adam and gradient descent stems from a smaller normalised directionalsharpness penalty, achieved by balancing update energy across curvature groups, rather than from spectral-norm properties; they further verify that the in-sample Hessian eigenbasis aligns closely with the gradient’s singular basis, directly supporting the per-basis construction of our spectral probes. Our spectral probing analysis can be viewed as an extension of this work, measuring the full anisotropic profile at per-rank granularity along real Transformer training trajectories in an out-of-sample manner. This measured out-of-sample spectral profile is more closely aligned with practical training scenarios (where convergence on unseen batches is concerned) and enables us to develop a low-cost, improved Muon optimiser (SAMuon), whose notable empirical improvement, in turn, verifies this account.

Spectral Analysis of Deep Neural Networks. Empirical studies of the Hessian eigenspectrum have long established that neural network loss landscapes are highly anisotropic, the spectrum splitting into a bulk concentrated near zero and a small number of outlier eigenvalues orders of magnitude larger [Sagun et al., 2018]. For Transformers, Zhang et al. [2024] analyse this structure at the granularity of parameter blocks, attributing the failure of single-learning-rate SGD, relative to coordinate-wise Adam, to this inter-block curvature disparity, i.e. “block heterogeneity”. Complementary to their work, we argue that the distribution of curvature within each block across the singular directions of the update is another key aspect that separates orthogonal optimisers from Adam (Section 4.3). There is also a body of work discussing the spectral anisotropy of the gradient itself [Refael et al., 2025, Cao et al., 2026, Magakyan et al., 2026], instead of the curvature behaviour that is our focus. Specifically, Magakyan et al. [2026] provides independent evidence for our central finding: truncating the bottom singular directions of the Muon update markedly degrades its performance, consistent with our finding that the advantage of Muon resides in better utilising the spectral bulk.

Adaptive Methods and Practical Muon Variants. Beyond convergence speed, a central appeal of Muon [Jordan et al., 2024b] is its low memory overhead. Muon maintains only a single momentum buffer, rather than the two moments used by Adam [Kingma and Ba, 2015] and the even heavier memory requirement of most second-order methods. Kronecker-Factored Approximate Curvature (K-FAC) [Martens and Grosse, 2015], Shampoo [Gupta et al., 2018], and SOAP [Vyas et al., 2025] precondition with Kronecker-factored curvature statistics whose factors scale with model size, and some recent Muon variants reintroduce the same burden in pursuit of adaptivity: AdaMuon [Si et al., 2025] adds an element-wise second moment, Newton–Muon [Du and Su, 2026] an explicit activation-covariance right preconditioner, and Mousse [Zhang et al., 2026] a full set of Shampoo Kronecker factors. Each improves on Muon but loses its memory advantage, carrying additional state that scales with the model. A complementary hybrid, COSMOS [Liu et al., 2026], instead confines adaptive (SOAP-style) preconditioning to the leading eigen-subspace of the gradient and applies a low-cost orthogonalised update to the remainder. Its design echoes our central finding, but is more complex than our head-only static treatment. LiMuon [Huang et al., 2026] likewise applies a randomised low-rank SVD to the momentum buffer, but towards memory compression of the update rather than shaping its spectral allocation, i.e. the same computational primitive as SAMuon but with a different purpose. Two lines of research instead keep the footprint close to the single buffer used by Muon. NorMuon [Li et al., 2026] adds only a per-neuron second-moment vector to rescale the orthogonalised update; its operation is complementary to ours and potentially integrates with our method rather than competing with it. Scion [Pethick et al., 2025] is a renormalised Muon which also replaces the auxiliary Adam optimiser used for embeddings with a lightweight signed update. SAMuon retains this single-buffer footprint while exploiting the within-block spectral structure through a fixed, measurement-derived shaping of the update spectrum, at negligible additional FLOPs and with a single tuned extra hyperparameter (γ; the profile rank of SAMuon follows a fixed width rule and is not tuned). Note that we build our work directly on Scion, which is our primary baseline and is referred to as Muon in Section 6.

## 3 Preliminaries

The Muon optimiser [Jordan et al., 2024b] operates independently on each 2D weight matrix $\mathbf { W } ^ { ( t ) } \in$ $\mathbb { R } ^ { m \times n }$ of the target model, where we take $m \leq n$ without loss of generality. At step t, given the mini-batch gradient $\mathbf { G } ^ { ( t ) } = \nabla _ { \mathbf { W } } \mathcal { L } ( \mathbf { W } ^ { ( t ) } ; \boldsymbol { B } _ { t } )$ , Muon first integrates the gradient into a momentum buffer $\mathbf { M } ^ { ( t ) }$ , and it is this buffer that is orthogonalised to produce the update:

$$
\mathbf { M } ^ { ( t ) } = \mu \mathbf { M } ^ { ( t - 1 ) } + \left( 1 - \mu \right) \mathbf { G } ^ { ( t ) } ,\tag{1}
$$

$$
\mathbf { O } ^ { ( t ) } = \mathrm { N S } \big ( \mathbf { M } ^ { ( t ) } \big ) \approx \big ( \mathbf { M } ^ { ( t ) } \mathbf { M } ^ { ( t ) \top } \big ) ^ { - 1 / 2 } \mathbf { M } ^ { ( t ) } = \mathbf { U } \mathbf { V } ^ { \top } ,\tag{2}
$$

$$
\mathbf { W } ^ { ( t + 1 ) } = \mathbf { W } ^ { ( t ) } - \eta \kappa \mathbf { O } ^ { ( t ) } ,\tag{3}
$$

where $\mu$ is the momentum coefficient, $\eta$ is the learning rate and κ is a fixed optimiser-specific scaling factor. For the Scion implementation of Muon used in our paper, the fixed scale for a weight matrix is $\kappa = \sqrt { d _ { \mathrm { o u t } } / d _ { \mathrm { i n } } }$ , where $d _ { \mathrm { o u t } }$ and $d _ { \mathrm { i n } }$ are the matrix’s original output and input dimensions, respectively [Pethick et al., 2025]. For simplicity, we also assume $\mathbf { M } ^ { ( t ) }$ to have full row rank (with rank $r \ : = \ : m )$ . Here and throughout the paper, we write the (reduced) SVD of any matrix as A = US $\begin{array} { r } { \mathbf V ^ { \top } = \sum _ { i } \sigma _ { i } \pmb { u } _ { i } \pmb { v } _ { i } ^ { \top } } \end{array}$ , where S is the diagonal matrix of singular values $\sigma _ { 1 } \geq \sigma _ { 2 } \geq \cdot \cdot \cdot \geq \sigma _ { m } \geq 0$ ${ \mathbf { } } { \mathbf { } } _ { { \dot { \mathbf { \imath } } } } , v _ { i }$ are the corresponding left and right singular vectors, and the index i is the spectral rank index of the singular component $\sigma _ { i } \ b { u } _ { i } \ b { v } _ { i } ^ { \top }$ ; in Equation (2), U and V are from this SVD of the buffer $\mathbf { M } ^ { ( t ) }$ . The operator $\mathrm { N S } ( \cdot )$ denotes the Newton-Schulz iteration, which approximates the polar factor $\mathbf { U V } ^ { \top }$ of its input using only a small number of matrix products, avoiding an explicit SVD. The iteration-specific coefficients and finite-iteration response used in our experiments are reported in appendix $\mathrm { { G . 4 . } }$ In spectral terms, exact whitening maps every non-zero singular value to one. Thus, under exact whitening, all m singular components within each weight matrix receive the same coefficient $\eta \kappa ;$ the custom finite $4 \times 3$ scheme closely approximates this uniform allocation over the operative range. The analysis in this paper focuses on this uniform spectral scaling.

## 4 Spectral probing for the loss-optimal step profile

The Muon optimiser [Jordan et al., 2024b] achieves rapid convergence by applying a uniform scaling to the spectrum of the momentum buffer and removing gradient anisotropy. To find why this isotropic whitening improves over methods such as SGD and Adam, and to understand its structural suboptimality, we must investigate the true underlying geometry of the loss landscape. To this end, we conduct a fine-grained spectral analysis of the Transformer loss landscapes, measuring the loss-optimal step across the target spectrum.

## 4.1 Methodology: Spectral probing

To construct the optimal step profile of the loss landscape, we propose a Spectral Probing procedure that measures the loss landscape along each direction in the spectral basis of the momentum buffer across the entire model in an offline manner, drawing on the directional-probing methodology of Wu et al. [2024].

Per-Rank Probe Construction. Let $\mathcal { L } ( \pmb \theta _ { t } ; B )$ be the loss on a batch $\boldsymbol { B }$ at the checkpoint parameter state $\theta _ { t } ,$ , where t denotes the training iteration of the checkpoint. We probe the spectrum of the updated momentum buffer for the $\tilde { N _ { \mathrm { W } } } = 7 2$ two-dimensional weight matrices in the model, indexed by j. For each matrix, we integrate the gradient $\mathbf { G } _ { j }$ of a gradient batch $B _ { \mathrm { g } }$ into the momentum buffer $\mathbf { M } _ { j }$ stored at the checkpoint, $\mathbf { M } _ { j }  \mu \mathbf { M } _ { j } + \dot { ( 1 - \mu ) } \mathbf { G } _ { j }$ (with $\mu$ the momentum coefficient of the underlying optimiser), exactly reproducing the buffer state the optimiser operates on when generating parameter updates in practical training scenarios. Crucially, the response on a disjoint out-of-sample batch $B _ { \mathrm { o } } ^ { - } \neq B _ { \mathrm { g } }$ is measured, $i . e .$ the loss reduction an update step achieves on unseen data, instead of the gradient batch that was used to form the update. This protocol is thus denoted as out-of-sample spectral probing. Note that an in-sample measurement used by curvature analyses in prior works reports a far more permissive spectral tail that largely reflects refitting the gradient batch rather than generalisation on unseen data (see Figure 8 in appendix D). The SVD of the updated momentum buffer is $\begin{array} { r } { \mathbf { M } _ { j } = \sum _ { d } \sigma _ { j , d } \mathbf { \boldsymbol { u } } _ { j , d } \mathbf { \boldsymbol { v } } _ { j , d } ^ { \top } , } \end{array}$ . The singular components are indexed by their rank $d , i . e .$ the position of $\sigma _ { j , d }$ in the descending spectrum, and grouped across weight matrices by rank. This grouping is well-defined for GPT-2-style architectures because every probed matrix shares the same minimum dimension, the model width $d _ { \mathrm { m o d e l } }$ , so each matrix carries exactly $d _ { \mathrm { m o d e l } }$ singular components and the d-th component can be aggregated model-wide. Our analysis therefore focuses on intra-block spectral behaviour instead of inter-block heterogeneity. The rank-d probe $\Delta \theta ^ { ( d ) }$ collects the d-th singular component $\mathbf { P } _ { j , d }$ of every probed matrix to form a pseudo parameter update

$$
\mathbf { P } _ { j , d } = - \kappa _ { j } \pmb { u } _ { j , d } \pmb { v } _ { j , d } ^ { \top } , \qquad \Delta \pmb { \theta } ^ { ( d ) } = \mathrm { v e c } \left( \{ \mathbf { P } _ { j , d } \} _ { j = 1 } ^ { N _ { \mathrm { W } } } \right) \quad ( d = 1 , \dots , d _ { \mathrm { m o d e l } } ) ,\tag{4}
$$

where vec concatenates the entries of all $N _ { \mathrm { W } }$ matrices into a single vector acting on the parameter state $\pmb \theta _ { t } .$ , and $\kappa _ { j }$ is the fixed per-weight-matrix scaling of Scion for the j-th matrix. Note that $\mathbf { P } _ { j , d }$ has the same orientation and scaling as the rank-d component that the actual Muon (Scion) optimiser applies to the j-th momentum buffer, which makes the probe’s loss-optimal step size directly comparable with the effective learning rate used in the Muon (Scion) optimiser’s trajectory (the red dashed line in the middle panel of Figure 1). Also, every matrix contributes exactly one rank-one component to each probe, which prevents the curvature measurements from being confounded by unequal numbers of aggregated components from different matrices.

Measurement via Local-Quadratic Approximation. For each probe, we primarily measure the loss-optimal step size, defined as the step size that minimises the out-of-sample loss along the probe direction, $\begin{array} { r } { \eta _ { d } ^ { * } = \arg \operatorname* { m i n } _ { \eta \ge 0 } \mathcal { L } ( \pmb { \theta } _ { t } + \eta \Delta \pmb { \theta } ^ { ( d ) } ; \mathcal { B } _ { \mathrm { o } } ) } \end{array}$ . Running an exact line search for all ranks of every checkpoint is prohibitively expensive. Therefore, we estimate $\eta _ { d } ^ { * }$ from the local-quadratic approximation (LQA) of the loss along the probe direction,

$$
\begin{array} { r } { \mathcal { L } ( \pmb { \theta } _ { t } + \eta \Delta \pmb { \theta } ; \mathcal { B } _ { \mathrm { o } } ) \approx \mathcal { L } ( \pmb { \theta } _ { t } ; \mathcal { B } _ { \mathrm { o } } ) + \eta \pmb { g } ^ { \top } \Delta \pmb { \theta } + \frac { 1 } { 2 } \eta ^ { 2 } \Delta \pmb { \theta } ^ { \top } \mathbf { H } \Delta \pmb { \theta } , } \end{array}\tag{5}
$$

where $\textbf {  { g } }$ and H denote the full-parameter gradient and Hessian of the out-of-sample batch loss $\mathcal { L } ( \cdot ; B _ { \mathrm { o } } )$ at $\theta _ { t }$ . Minimising Equation (5) over η yields

$$
\eta _ { d } ^ { * } = - \frac { { \pmb g } ^ { \top } \Delta { \pmb \theta } ^ { ( d ) } } { \Delta { \pmb \theta } ^ { ( d ) \top } { \bf H } \Delta { \pmb \theta } ^ { ( d ) } } .\tag{6}
$$

The directional curvature in the denominator does not require an explicit Hessian, since rearranging Equation (5) at a small displacement ε gives the finite-difference estimate

$$
\Delta { \pmb \theta } ^ { \top } { \bf H } \Delta { \pmb \theta } \approx \frac { 2 } { \varepsilon ^ { 2 } } \Big [ { \mathcal L } \big ( \pmb \theta _ { t } + \varepsilon \Delta { \pmb \theta } ; { \mathcal B } _ { \mathrm { o } } \big ) - { \mathcal L } \big ( \pmb \theta _ { t } ; { \mathcal B } _ { \mathrm { o } } \big ) - \frac { } { \varepsilon } { \pmb \theta } ^ { \top } \Delta { \pmb \theta } \Big ] ,\tag{7}
$$

which reuses the pre-computed loss and gradient and consumes only one additionalforward pass per probe. The resulting collection $\{ \eta _ { d } ^ { * } \} _ { d = 1 } ^ { d _ { \mathrm { m o d e l } } }$ constitutes the out-of-sample loss-optimal spectral step profile of the loss landscape (the measured profile for short) at checkpoint $\theta _ { t } ,$ which Figure 1 tracks across training iterations. Interpreting this profile relies on local quadraticity and, for a positive interior optimum, on the probe being a descent direction on the held-out batch and having positive directional curvature. These assumptions are stated in appendix $\mathbf { A } ,$ which also validates the local quadraticity of the Transformer pretraining loss landscape and reports the filtering of probes that fail either the descent or curvature check. Such probes account for only 0.52% of the sample and are excluded from the final spectral profile.

![](images/619c124302d342ccebe20fe76b60a8aa71b6e0a7a035a2039202a2e4469b9b1d.jpg)

![](images/82f2a8b86f3e484e338fd6c9785c384d5c7a349438336d17f3e99c74c7a78949.jpg)

![](images/2e0ed1168a5fc927cbbf93fc99ee9c67d12f552ee86a4187d1f8d3aa47b268eb.jpg)  
Figure 1: Out-of-sample loss-optimal step profile across training. Spectral probing (Section 4) of the updated momentum $b u f f e r ,$ along a Muon (Scion) trajectory of a 64M-parameter “moddednanogpt” model (exact run in Pethick et al. [2025]), with checkpoints from iterations 200–3500 in the constant-learning-rate stage. Each point shows the mean over 20 held-out batches, and error bars indicate the standard deviation across batches. The horizontal axis runs from the deep tail (rank 512) to the head (rank 1) in log scale. Left: cumulative spectral energy, concentrated in the leading ranks (the head alone carries the majority). Middle: per-rank optimal spectral allocation $\eta _ { d } ^ { * }$ in log scale (the LQA-estimated loss-optimal step size along each probe): this spectral profile is mostlyflat across the bulk and drops sharply near the volatile head; the head’s optimal step coincides with the actual (uniform) allocation of Muon (red dashed line, 0.01221), pinned at the Edge-of-Stability [Cohen et al., 2021]. Right: cumulative loss reduction (LQA estimate), accumulated for each probe from the deep tail to the head and normalised by the per-iteration optimal total, under two allocation policies: each rank at its own optimal allocation (solid, reaching 100% by construction) versus every rank at the actual uniform allocation of Muon (dashed). The head-pinned allocation of Muon realises only a small fraction of this idealised per-iteration optimum.

## 4.2 Observations: a stable, highly anisotropic loss-optimal step profile

Figure 1 maps the spectral profile of the momentum buffer along the Muon (Scion) training trajectory of a 64M “modded-nanogpt” model [Jordan et al., 2024a, Pethick et al., 2025]. Each data point is averaged over 20 held-out batches, with error bars indicating the standard deviation. We probe only 32 ranks at each checkpoint to reduce the probing cost and interpolate the resulting profile for visualisation. The foremost observation from this figure is that the measured profile as a whole is highly anisotropic yet stable. The loss-optimal step spans nearly an order of magnitude and follows a characteristic shape that is mostly flat across the bulk and drops sharply near the head. After an initial burn-in phase of ∼200 iterations, the profile settles into this shape and persists across training, remaining nearly sample-invariant with narrow error bars across batches. This stability makes a static structural prior viable in Section 5, rather than an adaptively learned one. Note that this shape is not an artefact of the choice of optimiser or model scale, since measured profiles with the same characteristics also appear when probing a standard AdamW trajectory (see Figure 5 in appendix B) and a larger model (see Figure 6 in appendix C). In the remainder of this section, we refine this picture with three structural details and a further finding on its slow emergence during early training.

The update signal concentrates in a Volatile Head operating at the Edge-of-Stability. The leading ranks carry the dominant share of the buffer’s spectral energy (the low-rank structure described in [Cao et al., 2026, Refael et al., 2025]), yet these are the least stable directions in the spectrum: their loss-optimal steps are nearly an order of magnitude smaller than those of the bulk. Crucially, the actual Muon update scale coincides with this limit (red dashed line in the middle panel of Figure 1). This provides direct empirical evidence that the head operates at the Edge-of-Stability [Cohen et al., 2021], and that this single direction limits the maximum step size of all the spectral directions in standard Muon.

The Tolerant Bulk is mostly flat and under-exploited. Below the head, the loss-optimal step climbs sharply over the leading few ranks, with the largest single gap between the rank-1 and rank-2 probes, and then stays mostlyflat across the bulk at several times the scale of the head (Figure 1), yet Muon drives all directions at the same head-pinned scale. This causes a gap across the bulk of the spectrum. By allowing the bulk directions to use their own much larger loss-optimal step sizes instead of Muon’s uniform spectral allocation, a much larger per-iteration loss reduction can be achieved (the right panel of Figure 1). Note that this cumulative loss reduction is a very optimistic idealisation, assuming all probes to be Hessian-conjugate, so we interpret it qualitatively: the headroom is large, although its exact size is uncertain. This idealised headroom motivates bulk amplification, although cross-rank interactions determine how much can be practically recovered in a joint update.

The near-head decline is log-rank-linear. Replotting the same per-rank optimal allocation on a linear scale against log rank (the left panel of Figure 2) makes the shape of the near-head decline directly visible. It can be seen that the loss-optimal step increases approximately linearly with log rank from the head to the bulk plateau, and stays mostly flat beyond. The measured profile can be described by two parameters, the plateau level and the log-rank slope of the decline (equivalently, its onset rank). This parametric description is stable across checkpoints and model scales in the same way as the profile itself. We emphasise that this is an empirical characterisation of the measurement, but it is directly applicable, and Section 5 applies it to the spectral allocation used by SAMuon.

The head gap emerges slowly over training. The gap between the head and the rest of the spectrum is not present at initialisation. Rather, it develops gradually over the first few hundred iterations. When we probe the same trajectory across training iterations (Figure 11 in appendix G), the head gap (rank-2/rank-1 optimal spectral allocation ratio, depicted in the left panel of Figure 11) starts from near-isotropic (≈ 2 at iteration 50; exact isotropy would give 1) and slowly climbs to its final level (≈ 5; this measured head gap anchors the γ search range of our experiments in Section 6) within a few hundred steps, after which it remains stable. The head’s spectral energy and directional curvature also concentrate over the same window at a similar pace. One consequence is that the gap that SAMuon exploits must open before it can be exploited, which is precisely why the spectral shaping requires a slow warmup rather than being applied from the first step (Section 5).

## 4.3 Why Muon Outperforms Adam, and How to Improve Muon

The measured spectral profile supplies a missing convergence-side account for the success of orthogonal optimisers and quantifies the potential room for improvement.

Spectral Re-Allocation Perspective. Because our probing measures the loss-optimal step profile of the momentum buffer from which momentum-based optimisers generate their update steps, the comparison between optimisers reduces to a question of spectral allocation. Momentum SGD applies the buffer directly, i.e. its update $\begin{array} { r } { \mathbf { M } ^ { ( t ) } = \sum _ { d } \sigma _ { d } \mathbf { \boldsymbol { u } } _ { d } \mathbf { \boldsymbol { v } } _ { d } ^ { \top } } \end{array}$ assigns each rank a scale proportional to its singular value $\sigma _ { d } .$ . Muon applies the whitened buffer $\begin{array} { r } { \mathrm { N S } ( \mathbf { M } ^ { ( t ) } ) = \sum _ { d } \mathbf { u } _ { d } \pmb { v } _ { d } ^ { \top } } \end{array}$ , assigning a uniform scale to every singular direction. To visualise this comparison, Figure 3 overlays each optimiser’s spectral allocation against the measured optimal profile. Viewing each method as a spectral allocation places SGD, Adam, and Muon on a single axis and yields a novel unified account of why Muon outperforms Adam, which in turn outperforms SGD, without relying on the separate, optimiser-specific arguments of prior work. The remainder of this section discusses the consequences.

![](images/c8c44d9a50807649c4f8310f00e19a2b9bb469e513e7a34ccc244f822b9f5cc6.jpg)

![](images/9bd1baeb7f53ff6df1ec92901fb7fb147de1c7f5799e95beebbb1832ebc25224.jpg)  
Figure 2: The near-head decline of the loss-optimal step profile is close to linear in log rank, across model scales. We replot the probed spectral profile with a linear y-axis against log rank. Left: the width-512 (64M) model (the middle panel of Figure 1); Right: the width-768 (124M) model (see appendix C). At both scales, the loss-optimal step decreases approximately linearly with log rank from the bulk plateau to the head. The visually estimated plateau onset shifts from rank ∼ 32 at width 512 to rank ∼ 40 at width 768, consistent with the square-root width heuristic used for SAMuon in Section 5.

SGD Issues and Adam Workaround. The poor performance of SGD on Transformer training is well documented [Zhang et al., 2024], and our probing analysis makes its failure mode selfevident, since its signal-proportional allocation runs directly counter to the measured loss-optimal step profile. Worse still, because the Transformer gradient energy concentrates in the leading ranks [Cao et al., 2026, Refael et al., 2025], SGD allocates almost its entire update along the most unstable directions. Adam provides a workaround through coordinate-wise scaling, which implicitly dampens this allocation and explains its advantage over SGD in Transformer training. This provides a new, within-block account of the Adam advantage. While Zhang et al. [2024] attributes the Adam–SGD gap to curvature heterogeneity across parameter blocks, our probing identifies the anisotropic spectral curvature within parameter blocks as a complementary within-block mechanism that concurrent work by Wang et al. [2026] independently identifies as a driving factor. Note that the Adam workaround is spectral-agnostic, since it softens the head’s weight but has no mechanism to overcome its dominance of the update spectrum. The measured allocation confirms this point: projected onto the buffer’s singular basis, the true Adam update lifts the bulk’s relative scale by an order of magnitude over SGD, but the head still dominates (Figure 3).

Advantage and Limit of Muon’s Uniform Scaling. Muon applies a far more direct spectral correction. By assigning a uniform scale across the spectrum, it amplifies rank d by $\sigma _ { 1 } / \sigma _ { d }$ relative to the SGD spectral allocation, which greatly improves utilisation of the bulk (Figure 3). Since the measured loss-optimal step profile is anti-correlated with $\sigma _ { d } ,$ this reallocation pushes in precisely the direction the landscape demands. The whitening greatly increases the relative update scale of the tolerant bulk, where the landscape allows far larger steps (Figure 1). We argue that this spectral reallocation, rather than the spectral-norm stability emphasised by existing analyses, is the geometric source of the rapid convergence of Muon, directly addressing the open question raised in Section 1. The measured loss-optimal step is mostly flat away from the head and several times the head’s Edge-of-Stability step size, whereas Muon, pinned by that single head direction, drives every rank at the head’s conservative scale (Figure 1). The allocation of Muon is therefore approximately correct in shape across the bulk but is also systematically too small in scale because of the limiting volatile head.

![](images/0d75e68e90dc59783087a2ac1f79cd5ed6db1496357be249c3487cbf39621dfb.jpg)  
Figure 3: Optimisers differ in how they scale the momentum-buffer spectrum, and SAMuon / SAMuon-lite match the measured optimum most closely. The updates generated by different optimisers implicitly assign a per-rank scaling to the singular components of the momentum buffer, and we overlay these allocations against the optimal<sup>∗</sup> profile (the iteration-1000 out-of-sample lossoptimal step size profile of Muon from Figure 1). All profiles are head-normalised to 1 and evaluated at the iteration-1000 checkpoint, with the Adam allocation drawn from the AdamW trajectory (see appendix E). Left: scaling per spectral rank (log axes). Right: the same allocations as a function of normalised singular value σ (linear axes). SAMuon-lite (γ = 5) raises the bulk to a single uniform level well above the Muon scale while holding the head at the same scale as Muon, and SAMuon $( \gamma = 7 . 0 7 )$ additionally follows the measured log-rank-linear decline over the leading ranks, the closest shape match to the measured optimum profile. From SGD → Adam → Muon → SAMuon-lite → SAMuon, each successive optimiser aligns its allocation more closely with what the spectral profile suggests.

Towards Structural Alignment (SAMuon). The remaining question is how to make use of this headroom. Eigen-adaptive methods such as SOAP [Vyas et al., 2025] can be viewed as learning this anisotropic profile through their adaptive second moments, but at a high computational and memory cost due to explicit eigendecompositions, large adaptive optimiser states, and greatly increased algorithm complexity. Our analysis shows that this expensive adaptive machinery could be avoided. The measured profile is structurally stable across batches and training stages. This means it can be distilled into a static prior at near-zero additional cost that holds the single volatile head at the step size it already takes, and raises the remainder of the spectrum towards the tolerance the probe measures, either uniformly or following the measured log-rank-linear shape (the SAMuon-lite and SAMuon curves in Figure 3). Section 5 implements this idea in SAMuon and SAMuon-lite, using low-cost, transient low-rank estimates of the head region.

## 5 Spectral-Aware Muon (SAMuon)

Our spectral probing shows that the out-of-sample optimal step is mostlyflat across the bulk of the spectrum, far above the much smaller step the volatile head tolerates (Figure 1). The head sits at the Edge-of-Stability and limits the global step size: the mostly flat bulk could absorb a uniformly larger update, but Muon, by whitening every direction to the same scale, can only use the head’s conservative step for the whole spectrum. To address this issue, we propose Spectral-Aware Muon (SAMuon). As detailed in Algorithm 1, SAMuon retains the momentum and whitening mechanisms of Muon but applies a head-anchored spectral allocation to the whitened update: it holds the head at the same scale as Muon while increasing the rest of the spectrum towards its measured tolerance, realised through a low-cost, transient low-rank estimate of the head region that adds no persistent optimiser state. We instantiate this allocation into two variants: SAMuon, which follows the measured log-rank-linear shape of Figure 2; and SAMuon-lite, which is a two-level simplification that runs at nearly the same wall-clock speed as Muon.

## 5.1 Head-anchored spectral allocation: from two levels to the measured profile

Momentum Integration. Both instantiations maintain a single momentum buffer per 2D weight matrix, updated exactly as in the Muon baseline (Equation (1)). The spectral modification below changes only how the buffer is orthogonalised, leaving its temporal dynamics untouched.

Head-anchored scaling. SAMuon and SAMuon-lite distil the measured profile of Section 4 (mostly flat across the bulk, log-rank-linear near the head, see Figure 2) into a static allocation that anchors the head and rescales the rest of the spectrum. Let $\begin{array} { r } { \mathrm { N S } ( \bar { \mathbf { M } } ^ { ( t ) } ) \approx \sum _ { i } { \mathbf { u } } _ { i } { \mathbf { v } } _ { i } ^ { \top } } \end{array}$ be the whitened update used in Muon; under exact whitening, it drives every singular direction at unit scale. Let $( { \pmb u } _ { i } , { \pmb v } _ { i } ) _ { i \leq k }$ be the leading k singular pairs of the buffer $\mathbf { M } ^ { ( t ) }$ . The SAMuon update to each weight matrix $\mathbf { O } _ { \mathrm { S A } } ^ { ( t ) }$ raises the whole whitened momentum buffer to scale $\gamma _ { t }$ and then restores the leading k directions to scales $s _ { i } ( t )$

$$
{ \bf O } _ { \mathrm { S A } } ^ { ( t ) } \ : = \ : \gamma _ { t } \mathrm { N S } \big ( \mathbf { M } ^ { ( t ) } \big ) \ : - \ : \sum _ { i = 1 } ^ { k } \big ( \gamma _ { t } - s _ { i } ( t ) \big ) \ : u _ { i } v _ { i } ^ { \top } ,\tag{8}
$$

Under exact whitening, this allocates rank $i \leq k$ the coefficient $s _ { i } ( t )$ and every other direction the coefficient $\gamma _ { t }$ . The target profile anchors the head at the same unit scale as Muon, $s _ { 1 } ( t ) = 1$ , so it remains at the Edge-of-Stability step size that whitening already respects. With the finite $4 \times 3$ Newton–Schulz implementation, these target allocations are approximate; its response is quantified in appendix G.4. The scales rise monotonically to the level of the tail, $\gamma _ { t } .$ , with $1 = s _ { 1 } ( t ) \leq s _ { i } ( t ) \leq \gamma _ { t }$ Both variants start from the uniform profile of Muon and warm up gradually to the target profile. A warmup weight $w _ { t } \in [ 0 , 1 ]$ , rising from 0 to 1 over the warmup horizon (cosine schedule, see appendix G), sets $\gamma _ { t } = 1 + ( \gamma - 1 ) w _ { t }$ and $s _ { i } ( t ) = 1 + ( s _ { i } - 1 )$ w for target scales $s _ { i }$ . Training begins at exactly Muon $( w _ { t } = 0$ collapses Equation (8) to $\mathrm { N S } ( \mathbf { M } ^ { ( t ) } ) )$ , and the target anisotropic profile takes effect only after the head-bulk gap in the spectral profile has emerged (appendix G). Setting $\gamma = 1$ recovers the standard Muon update exactly for SAMuon / SAMuon-lite, making the Muon baseline a strict ablation.

SAMuon: the measured log-rank-linear profile. SAMuon follows the empirical shape of Figure 2 directly so that the target scales increase linearly in log rank from the head to the level of the tail,

$$
s _ { i } = 1 + ( \gamma - 1 ) \frac { \log i } { \log k } , 1 \leq i \leq k ( k \geq 2 ) ,\tag{9}
$$

and all ranks beyond k receive the fixed tail scale γ. The value of k is set by the measured onset of the plateau, $k = 3 2$ on the width-512 probing model, and transfers across widths by a square-root rule inspired by Figure 2, i.e. $k ( d _ { \mathrm { m o d e l } } ) = \left| 3 2 \sqrt { d _ { \mathrm { m o d e l } } / 5 1 2 } \right|$ k. The leading k singular pairs are obtained from a randomised low-rank SVD of the buffer at each update step (torch.svd\_lowrank; Halko et al., 2011). We emphasise that this shape and scaling rule are an empirical distillation of the measurement and not a theoretically derived optimum. We defer the theoretical investigation of this measurement to future work. Under this rule, SAMuon attains the best or joint-best final loss at every batch size and model scale (Section 6.2).

SAMuon-lite: a two-level simplification at Muon speed. $\mathrm { A t } k = 1$ , SAMuon-lite does not need a ramp (Equation (9) is not invoked), and only the anchor $s _ { 1 } = 1$ remains with a uniformly boosted remainder. This deliberately coarse two-level allocation captures the profile’s dominant feature of a single volatile head with a far more tolerant bulk. The single leading pair $( \mathbf { \boldsymbol { u } } _ { 1 } , \mathbf { \boldsymbol { v } } _ { 1 } )$ is obtained from a few steps of power iteration on $\mathbf { M } ^ { ( t ) }$ (reusing the buffer, with no explicit SVD) at $O ( m n )$ cost, a handful of matrix-vector products, so SAMuon-lite runs at nearly the same wall-clock speed as Muon. For both SAMuon variants, the weights are then updated as $\mathbf { W } ^ { ( t + 1 ) } = \mathbf { W } ^ { ( t ) } - \eta \kappa \mathbf { O } _ { \mathrm { S A } } ^ { ( t ) }$

## 5.2 Algorithm analysis: complexity and convergence guarantee

SAMuon preserves the core efficiency of standard orthogonal optimisers. Memory-wise, both instantiations maintain exactly one dense momentum buffer per weight matrix, identical to Muon (mn) and strictly smaller than Adam (2mn) and SOAP (6mn). The head estimate adds only a transient rank-k factor pair $( k ( m + n ) \ll m n )$ , not persistent state, so SAMuon remains the most memory-frugal orthogonal optimiser of the recent Muon variants. In FLOPs, the per-step overhead over Muon is negligible for both: $O ( m n )$ for the power iteration of SAMuon-lite and $O ( m n k )$ for the rank-k randomised SVD of SAMuon, with k on the order of $\sqrt { d _ { \mathrm { m o d e l } } } \ll r .$ , both small beside the $O ( m n r )$ Newton-Schulz whitening. Wall-clock behaviour, however, currently separates the two. Measured per training iteration on the 1B model (appendix $\mathrm { G } . 7 ) ,$ , the power iteration of SAMuon-lite adds 0.5% to the Muon iteration time (5.3% of the NS whitening), so it runs at nearly the same speed as Muon, whereas the torch.svd\_lowrank call of SAMuon adds $7 . 4 \%$ , nearly as much as the Newton-Schulz whitening itself. It is left for future work to improve the current svd\_lowrank implementation, which could greatly reduce its wall-clock cost given the low theoretical FLOP count.

Algorithm 1 SAMuon / SAMuon-lite, shown for a single 2D weight matrix   
1: Input: initial weights $\mathbf { W } ^ { ( 1 ) }$ , momentum buffer $\mathbf { M } ^ { ( 0 ) } = \mathbf { 0 } ,$ Scion matrix scaling $\kappa = \sqrt { d _ { \mathrm { o u t } } / d _ { \mathrm { i n } } }$   
2: Hyperparameters: η (effective LR), µ (momentum), γ (tail boost), T (total steps); profile rank k (width   
rule, appendix G) and target scales $\{ s _ { i } \} _ { i = 1 } ^ { k }$ (Equation (9)) for SAMuon only   
3: for $t { \overset { - } { = } } 1 , \ldots , T$ do   
4: Compute gradient $\mathbf { G } ^ { ( t ) }$ on batch $B _ { t }$   
5: $\mathbf { M } ^ { ( t ) }  \bar { \mu } \mathbf { M } ^ { ( t - 1 ) } + ( 1 - \mu ) \mathbf { G } ^ { ( t ) }$   
6: $\gamma _ { t }  1 + ( \gamma - 1 ) $ w<sub>t</sub> ▷ cosine-scheduled spectral warmup   
7: Variant #1: SAMuon (log-rank-linear profile):   
8: $( \boldsymbol { \mathbf { \mathit { u } } } _ { i } , \boldsymbol { \mathbf { \mathit { v } } } _ { i } ) _ { i \leq k } \gets \mathrm { L o w R a n k S V D } ( \mathbf { M } ^ { ( t ) } , k )$ ▷ torch.svd\_lowrank()   
9: $\begin{array} { r } { s _ { i } ( t ) \gets \bar { 1 } + ( s _ { i } - 1 ) w _ { t } , \mathrm { f o r } i = 1 , \dots , k } \end{array}$ ▷ head region spectral warmup   
10: $\begin{array} { r } { \mathbf { O } _ { \mathrm { S A } } ^ { ( t ) }  \gamma _ { t } \mathrm { N S } ( \mathbf { M } ^ { ( t ) } ) - \sum _ { i = 1 } ^ { k } ( \gamma _ { t } - s _ { i } ( t ) ) \mathbf { \delta } \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } } \end{array}$ ▷ Equation (8)   
11: Variant #2: SAMuon-lite $( \bar { k } = 1$ , simple two-level profile):   
12: $( \mathbf { \boldsymbol { u } } _ { 1 } , \mathbf { \boldsymbol { v } } _ { 1 } ) \gets \mathrm { P o w e r I t e r } ( \mathbf { \mathbf { \boldsymbol { M } } } ^ { ( t ) } )$   
13: $\mathbf { O } _ { \mathrm { S A } } ^ { ( t ) }  \gamma _ { t } \mathrm { N S } ( \mathbf { M } ^ { ( t ) } ) - ( \gamma _ { t } - 1 ) \mathbf { \mathfrak { u } } _ { 1 } \mathbf { \mathfrak { v } } _ { 1 } ^ { \top }$ ▷ Equation (8) with k = 1   
14: $\mathbf { W } ^ { ( t + 1 ) } \gets \mathbf { W } ^ { ( t ) } - \eta \kappa \mathbf { O } _ { \mathrm { S A } } ^ { ( t ) }$   
15: end for

Theoretical Convergence. We provide a convergence guarantee for the idealised exact-whitening versions of both variants (assuming perfect Newton-Schulz whitening) in appendix F (Theorem F.4). For any head-anchored allocation with scales in $[ 1 , \gamma ]$ , the guarantee preserves the convergence orders of Muon up to γ-dependent constants: $\mathcal { O } ( T ^ { - 1 / 4 } )$ in the stochastic setting and $\mathcal { O } ( T ^ { - 1 / 2 } )$ in the deterministic setting [Shen et al., 2025], covering both proposed algorithms.

## 6 Experiments

We evaluate the SAMuon and SAMuon-lite instantiations on Large Language Model (LLM) pretraining, following the rigorous experimental protocol established in Pethick et al. [2025] for training models with norm-constrained updates. Both instantiations are implemented directly on top of the codebase for Scion [Pethick et al., 2025].

## 6.1 Experimental setup

Architecture and Training Protocol. Our experimental setup follows the exact configuration in Pethick et al. [2025] to ensure a rigorous and fair comparison with the baselines. We train “moddednanogpt” models [Jordan et al., 2024a, Pethick et al., 2025] at three scales: 124M (width 768), 300M (width 1280), and 1B (width 2560) parameters. The models deviate from the standard GPT-2 by adopting modern architectural improvements: replacing LayerNorm with RMSNorm; utilising Rotary Positional Embeddings (RoPE) [Su et al., 2024] in place of absolute embeddings; and removing bias terms to maximise arithmetic intensity. All runs train on the FineWeb dataset [Penedo et al., 2024] (100B-token sample). The 124M and 300M models use a 10B-token budget and the 1B model a 20B-token budget, with the iteration count matched to the budget for each batch size. The batch size is swept over 1024, 2048, and 4096 sequences. For the 124M model the budget is roughly 4× the Chinchilla-optimal allocation [Hoffmann et al., 2022] (∼1.7× at 300M and ${ \sim } 1 \times$ at 1B), so the 124M runs train well into the over-trained regime. We employ a trapezoidal learning rate schedule consisting of a constant phase followed by a linear decay over the final 28.5% of training tokens (tuned schedule for Muon by Pethick et al. [2025]). A 5% linear warmup phase is included for the AdamW baseline, and a 30% spectral warmup phase (for $w _ { t }$ in Algorithm 1) is applied to the SAMuon variants at the beginning of training. Each optimiser configuration is evaluated with a single seed. Within each model-scale and batch-size setting, all optimiser runs use the same model initialisation and identical data ordering. Finally, all runs execute on NVIDIA A100 (80GB) GPUs in bfloat16 mixed precision.

Baselines. We compare against two primary optimisers: AdamW [Loshchilov and Hutter, 2019], the standard adaptive baseline $( \beta _ { 1 } ~ = ~ 0 . 9 , \beta _ { 2 } ~ = ~ 0 . 9 5 )$ used in most LLM training setups; and Muon (the Unconstrained Scion implementation by Pethick et al. [2025]). We chose Scion over the original Muon implementation due to its simpler hyperparameter tuning and better performance. A detailed discussion is provided in appendix G.1. Since SAMuon is implemented directly on the Scion codebase, Scion serves as the strict ablation baseline (equivalent to either instantiation with γ = 1).

Hyperparameter Tuning Protocol. All hyperparameters are tuned per batch size on the 124M model, and transferred across model scales following the convention of Pethick et al. [2025]. For AdamW we sweep the learning rate on a half-power-of-two logarithmic grid. For Muon we sweep the spectral radius (radius\_2d), the learning-rate factor Scion applies to the 2D, Muon-effective modules, which plays for orthogonal updates the role that the learning rate plays for AdamW, over a 2-spaced grid {35.36, 50, 70.71, 100, 141.42}. The tuned Muon optimum is 70.71/50/70.71 at batch sizes 1024/2048/4096. Both SAMuon variants run at the fixed spectral radius 50, the Scion default, and they sweep the tail boost γ over {3.54, 5, 7.07, 10, 14.14}. The value of k for SAMuon follows the fixed width rule (k = 39/50/71 at widths 768/1280/2560; appendix G) and is not tuned. Finally, the hyperparameters are transferred directly across larger model scales, following the transfer rule found in Pethick et al. [2025] which uses a constant optimal radius for Scion and a halved learning rate per model-scale step for AdamW. γ is assumed to be constant across model scales and is transferred without retuning, which makes the SAMuon variants relatively under-tuned compared to the baselines (see discussion in Section 7). Refer to appendix G for the full hyperparameter sweeps and other experimental details.

## 6.2 Main results

Table 1 reports the final validation loss and the token-efficiency improvement over the model-scale (124M/300M/1B) × batch-size (1024/2048/4096) grid. Analysis of the results is provided below.

SAMuon variants outperform the Muon (Scion) baseline by a large margin across model scales and batch sizes. Both variants attain a lower final validation loss than the tuned Muon baseline in all trained cells. SAMuon improves on that baseline by 0.0187–0.0295 at 124M, 0.0196–0.0228 at 300M, and 0.0137–0.0191 at 1B, which translates to a token-efficiency improvement of 13.3%–24.0% across the grid. For the simpler SAMuon-lite variant, the corresponding improvements are 0.0184–0.0275, 0.0173–0.0203, and 0.0134–0.0173, which translate to a token-efficiency improvement of 13.3%– 22.1%. This demonstrates that both SAMuon variants make effective use of the headroom identified in the spectral profile and achieve better convergence across models trained at the Chinchilla-optimal token budget or beyond.

SAMuon variants’ improvement over Muon grows with batch size. A thorough batch-size sweep is provided for the 124M model in the first three rows of Table 1, and a partial batch-size sweep is provided for larger models in the rest of the same table. A consistent trend can be observed across model scales: the token-efficiency improvements of SAMuon and SAMuon-lite over Muon generally grow with batch size. They increase from 20.3% and 20.3% (1024) to 24.0% and 22.1% (4096) on 124M, from 17.7% and 15.9% (1024) to 18.8% and 17.1% (2048) on 300M, and from 13.3% and 13.3% (1024) to 15.6% and 14.4% (4096) on 1B. One possible explanation is that larger batches provide a less noisy estimate of the spectral structure in the momentum buffer.

SAMuon matches or outperforms SAMuon-lite, and the gap generally widens with batch size. Across the model-scale and batch-size grid, SAMuon attains a final validation loss at least as low as SAMuon-lite in all cells in Table 1, with a difference of 0.0003–0.0027 at 124M, 0.0023–0.0025 at 300M, and 0.0003–0.0018 at 1B, which translates to a small token-efficiency advantage of 0.0–2.3 (percentage) points. At batch size 1024 on the 124M and 1B models the two variants are effectively tied, with a loss difference of 0.0003 and an equal token-efficiency improvement. Elsewhere the margin favours SAMuon, which suggests that the more precise log-rank-linear allocation provides an edge in optimisation by following a profile closer to the observation in our spectral probing analysis. Additionally, it can be observed that the gap between SAMuon and SAMuon-lite generally widens with batch size, which grows from 0.0 (1024) to 1.9 (percentage) points (4096) on 124M, from 0.0 (1024) to 1.2 points (4096) on 1B, and holds level on 300M at 1.8 (1024) and 1.7 points (2048). One possible explanation is that larger batches provide a less noisy estimate of the spectral structure in the momentum buffer, allowing SAMuon to benefit more from its finer-grained allocation. Note that the simpler two-level allocation of SAMuon-lite remains a cost-effective alternative that recovers most of SAMuon’s token-efficiency improvement over Muon, while offering a simpler implementation and lower per-iteration wall-clock cost.

Table 1: Final validation loss and token-efficiency improvement across the model-scale × batch-size grid. The hyperparameters for each optimiser are tuned for every batch size setup on the 124M model. The larger model scales (300M and 1B) inherit the tuned hyperparameters from the corresponding batch size of the 124M model, following the scaling convention of Pethick et al. [2025] (see Section 6.1). The final loss is reported at the end of the training budget (10B tokens for 124M and 300M, 20B tokens for 1B). The token-efficiency improvement (improv.) is the token budget saving $1 - T ^ { \star } / N$ of a SAMuon variant over the Muon baseline, where N is the schedule length of the tuned Muon baseline and $T ^ { \star }$ the shortest schedule with which the variant still reaches the same final loss (appendix H). The best (unrounded) loss and token-efficiency improvement in each row are highlighted in bold.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Batch</td><td rowspan="2">Tokens</td><td rowspan="2">AdamW</td><td rowspan="2">Muon</td><td colspan="2">SAMuon</td><td colspan="2">SAMuon-lite</td></tr><tr><td>loss</td><td>improv.</td><td>loss</td><td>improv.</td></tr><tr><td rowspan="3">124M</td><td>1024</td><td>10B</td><td>3.2105</td><td>3.1622</td><td>3.1435</td><td>20.3%</td><td>3.1438</td><td>20.3%</td></tr><tr><td>2048</td><td>10B</td><td>3.2410</td><td>3.1726</td><td>3.1484</td><td>23.8%</td><td>3.1511</td><td>21.5%</td></tr><tr><td>4096</td><td>10B</td><td>3.3050</td><td>3.1953</td><td>3.1658</td><td>24.0%</td><td>3.1678</td><td>22.1%</td></tr><tr><td rowspan="2">300M</td><td>1024</td><td>10B</td><td>3.0297</td><td>2.9836</td><td>2.9640</td><td>17.7%</td><td>2.9663</td><td>15.9%</td></tr><tr><td>2048</td><td>10B</td><td>3.0658</td><td>2.9941</td><td>2.9713</td><td>18.8%</td><td>2.9738</td><td>17.1%</td></tr><tr><td rowspan="2">1B</td><td>1024</td><td>20B</td><td>2.7599</td><td>2.7251</td><td>2.7114</td><td>13.3%</td><td>2.7117</td><td>13.3%</td></tr><tr><td>4096</td><td>20B</td><td>2.8067</td><td>2.7413</td><td>2.7222</td><td>15.6%</td><td>2.7240</td><td>14.4%</td></tr></table>

## 7 Discussion

The offline probing analysis for new optimiser design. The proposed out-of-sample spectral probing based on saved model checkpoints turns optimiser design into a problem that can be efficiently measured. The SAMuon variants show that a static, head-anchored prior derived from the measurement can provide large improvements on Muon at near-zero additional FLOPs. We believe this shows a promising direction for future work on optimiser design, where researchers can probe the loss landscape with saved checkpoints efficiently. This process can generate rich design signal (e.g. to discover the under-exploited spectral capacity in Muon in this paper) and save the cost of thorough full-training sweeps in early design stages.

SAMuon variants are currently under-tuned relative to the Muon (Scion) baseline. The SAMuon variants reported in Section 6.2 are under-tuned relative to their Muon baselines in two ways, which might understate their performance. Firstly, the SAMuon variants follow the same schedule as the Muon baseline in Pethick et al. [2025], which is optimised for Muon. Our preliminary investigation has shown that the SAMuon variants can benefit from a much longer linear decay phase. Properly tuning the learning rate schedule for the SAMuon variants specifically would likely improve their performance. Secondly, the hyperparameters for the SAMuon variants are transferred directly from the tuned hyperparameters for the 124M model to larger model-scale setups without re-tuning, while the hyperparameters for Muon and AdamW are transferred across model scales by following the scaling rules tuned and discovered in Pethick et al. [2025]. This difference makes the hyperparameters for the SAMuon variants relatively under-tuned for larger model-scale setups. We leave a more thorough hyperparameter/schedule tuning for the SAMuon variants to future work.

Is there a better spectral profile? SAMuon-lite adopts the simplest two-level profile, which achieves a 13.3%–22.1% token-efficiency improvement in training. Meanwhile, the richer log-rank-linear profile adopted in SAMuon provides a small yet consistent improvement on top of SAMuon-lite of up to 2.3 (percentage) points across all training setups. This is evidence that following the measured optimal spectral profile is beneficial, though the margin is small relative to the gap over Muon. The natural next step is to ask whether there is a better spectral profile than the measured one. First and foremost, the current loss-optimal step profile does not consider cross-probe interactions. A more refined scaling profile/analysis that takes cross-probe curvature interactions into account could yield a much better spectral profile. Another possible direction is to investigate whether the profile should adapt to its drift over the training trajectory, beyond the current early warmup. A better-scheduled or dynamic allocation mechanism that addresses this long-term drift could be beneficial. Finally, the current profile only considers intra-block spectral allocation, where the spectral profile is averaged over all probed layers/modules in the model. A preliminary investigation of per-module gradient alignment in appendix I shows clear inter-block differences. Thus, a more refined profile that assigns block-dependent allocations could be beneficial. We leave these questions for future work.

## 8 Conclusion

In this work, we contributed an out-of-sample spectral analysis of the Transformer pretraining loss landscape in order to understand why whitening the momentum buffer accelerates convergence so effectively. Probing the spectrum of the momentum buffer along real training trajectories, we mapped the optimal per-rank step size that the loss landscape tolerates along each singular direction and found it to have a stable, characteristic shape. This shape is mostly flat across the bulk of the spectrum, declining sharply near a single volatile head which sits at the Edge-of-Stability and allows a far smaller step size than the rest, with the decline tracing a close-to-linear ramp in log rank over the leading ranks. Viewed as a spectral allocation, this measured profile gives a unified account of why Muon outperforms Adam, which in turn outperforms SGD. It shows the limit of uniform whitening, which is constrained by the volatile head and under-exploits the more tolerant bulk of the spectrum.

Guided by this measured profile, we proposed Spectral-Aware Muon (SAMuon), which employs static, head-anchored priors on the whitened update that hold the volatile head at the same scale as Muon and increase the remainder of the spectrum towards its measured tolerance. SAMuon implements the empirically measured log-rank-linear shape using a rank-k randomised SVD of the momentum buffer, while the simplified SAMuon-lite keeps a coarse two-level allocation with power iteration and runs at nearly the same wall-clock speed as Muon. Neither adds persistent optimiser state over Muon, and a single convergence guarantee covers the idealised exact-whitening versions of both variants. Under the shared tuning-and-transfer protocol, with token budgets from approximately one to four times Chinchilla-optimal, SAMuon reaches the same final loss as the tuned Muon (Scion) baseline with a 13.3%–24.0% token-efficiency improvement in all trained model and batch size combinations. SAMuon-lite matches or nearly matches that improvement, and the improvement generally grows with batch size. These gains are a validation of the analysis. Aligning an optimiser’s update with the measured loss-optimal step profile translates directly into better convergence. Furthermore, SAMuon matches or improves on the two-level SAMuon-lite in every setting, with the clearest margins at larger batch sizes. This suggests that the complete spectral profile, not merely the head/bulk split, carries a meaningful signal. We hope the profile and the probing methodology behind it serve as a reusable starting point for spectral-allocation optimiser design.

## References

Jeremy Bernstein and Laker Newhouse. Old optimizer, new norm: An anthology. In 16th Annual Workshop on Optimization for Machine Learning (OPT 2024), 2024. URL https://opt-ml. org/papers/2024/paper93.pdf.

Jeremy Bernstein and Laker Newhouse. Modular duality in deep learning. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id= hErdffTsLu.

Hengjie Cao, Mengyi Chen, Yifeng Yang, Fang Dong, Ruijun Huang, Jixian Zhou, Anrui Chen, Mingzhi Dong, Yujiang Wang, Jinlong Hou, Yuan Cheng, Fan Wu, Fan Yang, Tun Lu, Ning Gu, and Li Shang. Metis: Training LLMs with FP4 quantization. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id= I2ZrCi5O84.

Lizhang Chen, Jonathan Li, and Qiang Liu. Muon optimizes under spectral norm constraints. Transactions on Machine Learning Research, 2026. URL https://openreview.net/forum? id=Blz4hjxLwU.

Jeremy M Cohen, Simran Kaur, Yuanzhi Li, J Zico Kolter, and Ameet Talwalkar. Gradient descent on neural networks typically occurs at the edge of stability. In International Conference on Learning Representations, 2021.

Zhehang Du and Weijie Su. The Newton–Muon optimizer. arXiv preprint arXiv:2604.01472, 2026. URL https://arxiv.org/abs/2604.01472.

Chen Fan, Mark Schmidt, and Christos Thrampoulidis. Implicit bias of spectral descent and Muon on multiclass separable data. In Advances in Neural Information Processing Systems, 2025. URL https://arxiv.org/abs/2502.04664. Spotlight; arXiv:2502.04664.

Vineet Gupta, Tomer Koren, and Yoram Singer. Shampoo: Preconditioned stochastic tensor optimization. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 1842–1850. PMLR, 10–15 Jul 2018. URL https://proceedings.mlr.press/v80/gupta18a.html.

N. Halko, P. G. Martinsson, and J. A. Tropp. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Review, 53:217–288, 2011. doi: 10.1137/090771806.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. An empirical analysis of compute-optimal large language model training. In Advances in Neural Information Processing Systems, volume 35, 2022.

Feihu Huang, Yuning Luo, and Songcan Chen. LiMuon: Light and fast Muon optimizer for large models. In International Conference on Machine Learning (ICML), 2026. URL https: //arxiv.org/abs/2509.14562.

Keller Jordan, Jeremy Bernstein, Brendan Rappazzo, Vlado Boza, Jiacheng You, Franz Cesista, Braden Koszarsky, et al. modded-nanogpt: Speedrunning the NanoGPT baseline, 2024a. URL https://github.com/KellerJordan/modded-nanogpt. Additional contributors credited in the repository as @fernbear.bsky.social and @Grad62304977.

Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024b. URL https: //kellerjordan.github.io/posts/muon/.

Kimi Team. Kimi K2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025. URL https://arxiv.org/abs/2507.20534.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Yoshua Bengio and Yann LeCun, editors, 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015. URL http://arxiv.org/abs/1412.6980.

Dmitry Kovalev. Understanding gradient orthogonalization for deep learning via non-Euclidean trust-region optimization, 2025. URL https://arxiv.org/abs/2503.12645.

Tim Large, Yang Liu, Minyoung Huh, Hyojin Bahng, Phillip Isola, and Jeremy Bernstein. Scalable optimization in the modular norm. In Advances in Neural Information Processing Systems, volume 37, pages 73501–73548, 2024. URL https://openreview.net/forum?id=SFxAjB7UXx. arXiv:2405.14813.

Tim Tsz-Kit Lau, Qi Long, and Weijie Su. PolarGrad: A class of matrix-gradient optimizers from a unifying preconditioning perspective, 2025. URL https://arxiv.org/abs/2505.21799.

Jiaxiang Li and Mingyi Hong. A note on the convergence of Muon. arXiv preprint arXiv:2502.02900, 2025. URL https://arxiv.org/abs/2502.02900.

Zichong Li, Liming Liu, Chen Liang, Weizhu Chen, and Tuo Zhao. NorMuon: Making Muon more efficient and scalable. In International Conference on Machine Learning (ICML), 2026. URL https://arxiv.org/abs/2510.05491.

Jingyuan Liu, Zizheng Zhang, Hantian Zhou, Zhen Huang, Yibo Wang, Heting Zhao, Shifan Song, et al. Muon is scalable for LLM training. arXiv preprint arXiv:2502.16982, 2025. URL https: //arxiv.org/abs/2502.16982.

Liming Liu, Zhenghao Xu, Zixuan Zhang, Hao Kang, Zichong Li, Chen Liang, Weizhu Chen, and Tuo Zhao. COSMOS: A hybrid adaptive optimizer for efficient training of large language models. In International Conference on Learning Representations (ICLR), 2026. URL https: //openreview.net/forum?id=j2QTOOtM8R.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.

Gagik Magakyan, Pablo Parrilo, and Asuman Ozdaglar. Spectral scaling laws of Muon, 2026. URL https://arxiv.org/abs/2606.04058.

James Martens and Roger Grosse. Optimizing neural networks with Kronecker-factored approximate curvature. In Proceedings ofthe 32nd International Conference on Machine Learning, volume 37 of Proceedings ofMachine Learning Research, pages 2408–2417. PMLR, 2015. URL https: //proceedings.mlr.press/v37/martens15.html.

Guilherme Penedo, Hynek Kydlícek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colinˇ Raffel, Leandro von Werra, and Thomas Wolf. The FineWeb datasets: Decanting the web for the finest text data at scale. In Thirty-eighth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://arxiv.org/abs/2406.17557.

Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu, Antonio Silveti-Falls, and Volkan Cevher. Training deep learning models with norm-constrained LMOs. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu, editors, Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 49069–49104. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/pethick25a.html.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI, 2019. URL https://cdn.openai.com/better-language-models/language\_models\_are\_ unsupervised\_multitask\_learners.pdf.

Yehonathan Refael, Guy Smorodinsky, Tom Tirer, and Ofir Lindenbaum. SUMO: Subspace-aware moment-orthogonalization for accelerating memory-efficient LLM training. In Advances in Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id= DIjRvEKOeG.

Levent Sagun, Utku Evci, V. Ugur Guney, Yann Dauphin, and Leon Bottou. Empirical analysis of the Hessian of over-parametrized neural networks, 2018. URL https://openreview.net/forum? id=rJrTwxbCb.

Wei Shen, Ruichuan Huang, Minhui Huang, Cong Shen, and Jiawei Zhang. On the convergence analysis of Muon, 2025. URL https://arxiv.org/abs/2505.23737.

Chongjie Si, Debing Zhang, and Wei Shen. AdaMuon: Adaptive Muon optimizer. arXiv preprint arXiv:2507.11005, 2025. URL https://arxiv.org/abs/2507.11005.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. doi: 10.1016/j.neucom.2023.127063.

Weijie Su. Isotropic curvature model for understanding deep learning optimization: Is gradient orthogonalization optimal? arXiv preprint arXiv:2511.00674, 2025. URL https://arxiv.org/ abs/2511.00674.

Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, and Sham M. Kakade. SOAP: Improving and stabilizing Shampoo using Adam for language modeling. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=IDxZhXrpNf.

Shuche Wang, Fengzhuo Zhang, Jiaxiang Li, Dirk Bergemann, and Zhuoran Yang. Why Muon outperforms Adam: A curvature perspective, 2026. URL https://arxiv.org/abs/2606. 04662.

Kaiyue Wen, David Leo Wright Hall, Tengyu Ma, and Percy Liang. Fantastic pretraining optimizers and where to find them. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=2J51qUZ0iG.

Xiaodong Wu, Wenyi Yu, Chao Zhang, and Philip Woodland. An improved empirical Fisher approximation for natural gradient descent. In Proceedings ofthe 38th International Conference on Neural Information Processing Systems, NeurIPS ’24, Red Hook, NY, USA, 2024. Curran Associates Inc. ISBN 9798331314385.

Greg Yang, James B. Simon, and Jeremy Bernstein. A spectral condition for feature learning. arXiv preprint arXiv:2310.17813, 2023. URL https://arxiv.org/abs/2310.17813.

Yechen Zhang, Shuhao Xing, Junhao Huang, Kai Lv, Yunhua Zhou, Xipeng Qiu, Qipeng Guo, and Kai Chen. Mousse: Rectifying the geometry of Muon with curvature-aware preconditioning. arXiv preprint arXiv:2603.09697, 2026. URL https://arxiv.org/abs/2603.09697.

Yushun Zhang, Congliang Chen, Tian Ding, Ziniu Li, Ruoyu Sun, and Zhi-Quan Luo. Why transformers need Adam: A Hessian perspective. In Proceedings of the 38th International Conference on Neural Information Processing Systems, NeurIPS ’24, Red Hook, NY, USA, 2024. Curran Associates Inc. ISBN 9798331314385.

## A Spectral Probing: Assumptions and Validation

This section states the assumptions and retention conditions required to interpret the local-quadratic estimate in Section 4.1, reports the filtering of probes that fail these conditions, and validates the local-quadratic approximation against an exact line-search reference.

Assumption A.1 (Local Quadraticity). Along each probe direction $\{ \pmb { \theta } _ { t } + \eta \Delta \pmb { \theta } ^ { ( d ) } : \eta \geq 0 \}$ , the batch loss is well-approximated by its second-order Taylor expansion (Equation (5)) over the step range containing the loss minimum.

Assumption A.2 (Held-Out Descent Direction). For every retained rank- $- d$ probe, the directional derivative of the held-out loss is negative, $\pmb { g } ^ { \top } \Delta \pmb { \theta } ^ { ( d ) } < 0 .$ , where $\mathbf { \pmb { g } }$ is the gradient of the held-out batch loss at $\theta _ { t }$ . Thus, a positive step along the probe direction initially decreases the held-out loss.

Assumption A.3 (Positive Directional Curvature). The probed curvature $\Delta \pmb { \theta } ^ { ( d ) \top } \mathbf { H } \Delta \pmb { \theta } ^ { ( d ) }$ is positive. Together with assumption A.2, this ensures that the quadratic in Equation (5) has a positive interior minimum. All probes retained in our analysis satisfy this condition.

Filtering ill-behaved probes. The LQA yields a positive interior optimum under assumptions A.2 and A.3. We therefore discard any probe that fails either check before aggregating or plotting the results. Across the 64M and 124M out-of-sample spectral probing analyses, 31 of the 6,400 sampled rank probes (0.48%) fail the descent-direction check, while two additional rank probes (0.03%) have negative curvature, giving 33 filtered rank probes (0.52%) in total. The two negative-curvature cases occur at the deepest ranks late in training and are numerically negligible, with curvature magnitudes no greater than $\mathbf { \bar { 4 . 0 \times 1 0 ^ { - 7 } } }$ ; the non-descent cases are likewise concentrated late in training. Thus, every probe included in the analysis satisfies both conditions, while the filtering affects only a tiny fraction of the measurements.

Validation against an exact reference: golden-section line search. For every LQA probe, the finite-difference curvature estimate of Equation (7) uses the fixed displacement $\varepsilon = 0 . 3$ . To further test assumption A.1 directly, we measure the exact optimal step of a probe by a one-dimensional line search along the probe direction, free from any quadratic assumptions. An exponential coarse search first brackets the loss minimum, and then a golden-section refinement in $\log _ { 1 0 }$ -space locates it to a target precision. The search operates purely on forward evaluations of the batch loss but requires tens of evaluations per probe, compared with the single additional forward evaluation required by the LQA estimate of Equation (7). We report the loss-optimal step in update-norm units as $s _ { d } ^ { * } = \| \eta _ { d } ^ { * } \Delta \pmb { \theta } ^ { ( d ) } \| _ { 2 } = \eta _ { d } ^ { * } \| \Delta \pmb { \theta } ^ { ( d ) } \| _ { 2 }$ , where the Scion matrix scales $\kappa _ { j }$ are already included in $\Delta \theta ^ { ( d ) }$ Figure 4 compares $s _ { d } ^ { * }$ estimated by the LQA with the line-search reference over 24 checkpoint–batch analyses spanning training stages and both Muon (Scion) and AdamW trajectories (773 probes in total). The two agree closely across more than four orders of magnitude of update norm: the median relative deviation remains near 5% throughout the operationally relevant range $( s _ { d } ^ { * } \lesssim 1 0 )$ which covers the volatile head and the bulk of the spectrum. Deviations grow only for the flattest tail directions $( s _ { d } ^ { * } \gtrsim 1 0 ^ { 2 } )$ , where the loss basin is extremely shallow, so the precise location of the minimum is both harder for the reference search to pin down and less consequential for the resulting profile. This close agreement confirms that the pretraining loss is locally near-quadratic along the probed directions (across training stages and update orientations), justifying the single-forward-pass LQA estimate used in our spectral analysis.

## B Spectral Probing of an AdamW Trajectory

The spectral profile in Figure 1 is measured on checkpoints produced by a Muon run, which raises a natural question: is the measured structure a property of the Transformer loss landscape, or an artefact of the particular trajectory that the update rule of Muon carves through it? To answer this, we repeat the complete out-of-sample probing analysis of Section 4 on a standard AdamW trajectory of the same 64M-parameter model under the identical protocol. The gradient of a gradient batch ${ \dot { B _ { \mathrm { g } } } }$ is integrated into the first-moment (momentum) buffer stored at each AdamW checkpoint, and the same Scion-scaled per-rank probes of Equation (4) are measured against a held-out batch with the validated LQA estimate. Figure 5 shows the same overall structure as the Muon trajectory. The loss-optimal step rises by more than an order of magnitude from the volatile head into a mostly flat, far more tolerant bulk, and the deep tail again contributes only a modest residual share of the cumulative loss reduction.

![](images/5b4ea94acc3563c4b312796d51f5e645b122b280cd413d8630581207235293c6.jpg)

![](images/e8316d46cc1bb85832d137c0cabd6578de9d6d86b0f8e66ff6ab2e9383bd383f.jpg)  
Figure 4: Validation of the local-quadratic estimate against the exact line search. Left: modelwide loss-optimal update norm $s _ { d } ^ { * } ,$ estimated by the LQA (x-axis) and by the golden-section linesearch reference (y-axis), for the 773 probe directions sampled from 24 checkpoint–batch analyses spanning all training stages and both Muon and AdamW trajectories; the dashed line marks perfect agreement $( y = x )$ . Right: relative difference between the two as a function of the LQA estimate, with the windowed median and interquartile range overlaid. The median deviation stays near 5% over the operationally relevant range, rising only for extremely flat directions where the loss minimum is shallow.

The one systematic difference appears at the head and is expected. The whitened update of Muon drives every singular direction at the same shared global scale, so the head sits at the Edge-of-Stability and its measured optimal step is pinned at that scale (the middle panel of Figure 1). The AdamW update, in contrast, is not normalised. The exact update norm realised along the trajectory varies, and the head’s loss-optimal step size inherits this variation, so it varies relatively more freely across checkpoints. As a consequence, the optimal step size profile is more variable across checkpoints than along the Muon trajectory (at early checkpoints it even rises across the bulk before the sharp near-head decline), although the overall trend of a volatile head and a more tolerant bulk still holds. This close structural agreement shows that the measured anisotropic profile persists across the evaluated Muon and AdamW trajectories, rather than being specific to the optimiser that generated the trajectory.

## C Spectral Probing at Width 768

The probing analyses of Section 4 were run on a width-512 (64M) model. To check that the measured structure is not specific to that scale, the complete out-of-sample probing analysis is repeated on a Muon trajectory of the 124M (width-768) model, matching the smallest trained scale of Section 6 (probing-run setup in appendix G). Figure 6 gives the same three-panel analysis as Figure 1; the near-head decline of the optimal step size with respect to the log rank axis is shown in the right panel of Figure 2 in the main text.

The qualitative observations in Section 4 persist at width 768. The profile remains highly anisotropic and stable across checkpoints, with a volatile head near the applied Muon scale, a mostly flat tolerant bulk, and an approximately log-rank-linear near-head transition.

The most relevant visible scale effect is a modest upward shift in the approximate plateau-onset rank, from about 32 at width 512 to about 40 at width 768 (the right panel of Figure 2). Because the transition is gradual, these values should be interpreted as approximate rather than sharply identified thresholds. This shift is consistent with the square-root width heuristic used in Section 5, which predicts k ≈ 39 at width 768. With only two directly probed widths, however, we do not treat this agreement as evidence establishing a general scaling law.

![](images/9661d4aa20b62bdfd6fd5c8cfacc77b32b4b8a9ba49b5deae5765f6796f3ad69.jpg)  
rank (512 1, log)

![](images/101f5f83bc7b33ad38f009900c3e4c217c64fb2a4bc90c8f57ce184bc1aaaa38.jpg)

![](images/b5df76a54a6e5563703f5cf6dec5a296ea3c36e4c2aecd7f4704ceb87df5f717.jpg)  
Figure 5: Out-of-sample loss-optimal step profile along an AdamW trajectory (same 64Mparameter “modded-nanogpt” model as Figure 1, checkpoints from iterations 200–3500). Out-ofsample spectral probing (Section 4) of the updated first-moment (momentum) buffer, with each per-rank probe’s optimal step size measured on a disjoint out-of-sample batch; the horizontal axis runs from the deep tail (rank 512) to the head (rank 1, log scale). Left: cumulative spectral energy, again concentrated in the leading ranks. Middle: per-rank optimal spectral allocation $\eta _ { d } ^ { * }$ (the lossoptimal step along each probe; LQA estimate; log axis). As with Muon, the bulk is mostly flat and the optimal allocation drops sharply near the head, which tolerates a far smaller step. Because the AdamW update is unnormalised, the realised update norm varies, and the head’s loss-optimal step therefore varies more freely across checkpoints; the slower convergence of AdamW also prolongs the burn-in, giving a more unstable profile in early checkpoints (iterations 200–500) that settles to the same volatile-head shape later (iterations 1000–3500). Right: cumulative loss reduction (LQA estimate): as for Muon, the leading ranks account for most of the achievable reduction and the deep tail contributes a modest residual share. The close structural agreement with Figure 1 shows that the volatile-head and tolerant-bulk structure persists across the evaluated Muon and AdamW trajectories.

![](images/a0909cfe26da5b0522fa5cc9181b813d662a2f95f443c776121761d266532b25.jpg)

![](images/0a113b5f03edc19e8ee2dc4ad564b53573344572116b19fed3e806739db1699d.jpg)

![](images/ba8bc272acdc05edb5b7cd93962786832cb4dde160704a402df4c106dabce128.jpg)  
Figure 6: Out-of-sample loss-optimal step profile at width 768 (124M model, Muon trajectory). Panels, axes, and protocol as in Figure 1. The qualitative structure of the width-512 analysis is reproduced.

## D In-Sample Spectral Probing

The out-of-sample probing of Section 4 measures the step size that reduces loss the most on unseen data. For completeness, we also report the in-sample variant, in which each probe’s optimal step size is measured on the same batch whose gradient formed the buffer $( B _ { \mathrm { o } } = \bar { B } _ { \mathrm { g } } )$ . This is the quantity that earlier curvature analyses implicitly probe. We report it in the same rank-organised, three-panel layout as the main-text out-of-sample figure (Figure 1), so the two regimes can be compared panel for panel.

Observation. Figure 7 reports the in-sample profile along the same Muon trajectory, in the same rank-organised, three-panel layout as the out-of-sample figures. The qualitative structure matches the out-of-sample analysis (a volatile head pinned at the Edge-of-Stability and a far more tolerant bulk), but two quantitative differences appear. First, the in-sample optimal allocation declines monotonically from the deep tail to the head rather than holding a mostly flat bulk, and the anisotropy spans two to three orders of magnitude, wider than the out-of-sample gap. Second, the cumulative loss reduction under the optimal allocation accrues steadily from the tail, so the bulk and tail appear far more productive than out-of-sample, where the reduction concentrates at the head. Both effects are consistent with in-sample probing partly measuring the optimiser’s ability to refit the gradient batch that formed the update; the out-of-sample analysis in the main text removes this confounder and yields the more conservative, generalisation-relevant profile that motivates SAMuon. The per-module sign alignment of appendix I supports the refitting explanation of the in-sample behaviour: tail directions align poorly with the gradients of unseen batches, so their large in-sample tolerance mainly reflects refitting of the sampled batch. The same in-sample inflation of the bulk and tail also occurs along an AdamW trajectory (Figure 9), confirming that this batch-refitting confounder is a property of the probe, not of the optimiser that generated the trajectory.

![](images/77e8730129a67f2e9064c3341f90f35fdcc3c04e89e79bfc1327278a11877e5f.jpg)

![](images/b883ff48de481bf82bd16ce93c882cb0ae3fef5652b8a85bb71b197146306b35.jpg)

![](images/94b11e021bef0130ffd074ae6530d2e2f07d465d8f1f84319ea65fb9335d36c3.jpg)  
Figure 7: In-sample loss-optimal step profile along a Muon trajectory, in the same panel layout as the out-of-sample Figure 1 (same 64M-parameter “modded-nanogpt” model and checkpoints), but with each probe’s optimal step size measured on the same batch whose gradient formed the buffer $( B _ { \mathrm { o } } = \bar { B } _ { \mathrm { g } } )$ . Two differences from the out-of-sample profile stand out. First, the entire bulk is raised well above its out-of-sample level (middle panel), widening the anisotropy. Second, the deep tail contributes far more to the idealised cumulative loss reduction (right panel). Both indicate that, in-sample, the bulk mainly yields refitting of the batch that formed the update, a gain that does not transfer to unseen data, which is exactly the confounder the main-text out-of-sample protocol removes.

Figure 8 overlays the two regimes directly along the same Muon trajectory. The out-of-sample optimal allocation (solid) is uniformly more conservative than the in-sample one (dashed) across the entire bulk, with the gap widest in the deep tail, where the in-sample allocation runs several times larger. The per-probe loss-reduction panel tells the same story from the capacity side. For the in-sample case, every probe across the spectrum yields a large reduction; out-of-sample, the tail’s contribution collapses and the reduction climbs only towards the head. The in-sample tail gain that does not survive out-of-sample is refitting of the gradient batch, while the residual that does survive is the generalisation-relevant headroom that SAMuon targets. Note that the same in-sample-versusout-of-sample gap holds along an AdamW trajectory (Figure 10), so this separation is generalisable rather than specific to Muon. Curvature-based accounts such as Wang et al. [2026] implicitly measure the in-sample quantity, which our out-of-sample protocol deliberately discounts.

## E Construction of the Spectral Allocation Comparison

This section details the construction of the curves in Figure 3, which compares the spectral scaling each optimiser applies against the measured loss-optimal step profile. All curves are computed at checkpoint 1000 of the out-of-sample probing analyses (Section 4), averaged across out-of-sample batches, and head-normalised so that the top singular direction (rank 1, largest σ) sits at 1. A single checkpoint is shown for visual clarity. The left panel plots the spectral allocation against spectral rank (log axes). The right panel plots the same allocation against singular value σ (linear axes), making explicit the transfer function each optimiser applies to the spectrum.

• Loss-optimal step profile: the per-rank loss-optimal step size measured by out-of-sample spectral probing (the measured profile of the Muon trajectory at checkpoint 1000 in Figure 1).

• SGD: a virtual update that assigns each rank a scale proportional to its average singular value, that is, the allocation produced when SGD applies the momentum buffer directly.

![](images/bf6eafcaacd7c9db26c84deecb6050217b61b69591ecc05227908618c97198b8.jpg)

![](images/9b78b3094e3021e44fa8940b1fbab752107f0f5ad15ce4b02add1e8ff0eec52a.jpg)  
Figure 8: In-sample versus out-of-sample spectral profiles along a Muon trajectory (64Mparameter “modded-nanogpt”, checkpoints from iterations 200–3500). Solid curves are the out-ofsample (main-text) measurement; dashed curves are the in-sample variant. Left: per-rank optimal spectral allocation. The in-sample allocation is systematically higher across the entire bulk, several times the out-of-sample level. Both collapse together at the head, pinned at the actual Muon scale (red dashed, 0.01221). Right: per-probe optimal loss reduction (log axis). In-sample, every probe across the spectrum, particularly the deep tail, yields a large reduction; out-of-sample, the per-probe reduction in the tail is far smaller and only climbs towards the head. The gap between the two reflects that the tail spectrum of the gradient batch mainly contributes to overfitting that batch, with limited transferable capacity to out-of-sample batches.

![](images/053007883448d262b0b9b2ea580cc35efae979d23b464a0a4d3adb9f771fb15e.jpg)

![](images/b91ce2829738a7f8fc5c62f4f1ea6416349403b9e61cc7f873b00a7307e9bed5.jpg)

![](images/57ed1e371c6fca31e60525afe095c49fe1e192e45f6229a44c588c0bb4057333.jpg)  
Figure 9: In-sample loss-optimal step profile along an AdamW trajectory, the AdamW analogue of the Muon Figure 7 and the in-sample counterpart of the out-of-sample Figure 5 (same 64Mparameter modded-nanogpt model and checkpoints; three-panel layout as Figure 1), with each probe’s optimal step size measured on the same batch whose gradient formed the buffer $( B _ { \mathrm { o } } = B _ { \mathrm { g } } )$ As for Muon (Figure 7), the bulk is lifted above its out-of-sample level (middle panel) and the deep tail contributes far more of the idealised cumulative loss reduction (right panel), the same batch-refitting signatures that the main-text out-of-sample protocol removes.

• Adam: the true Adam update $\Delta ^ { \mathrm { A } }$ projected onto each rank of the singular basis of its own buffer, $a _ { i } = { \pmb u } _ { i } ^ { \top } \Delta ^ { \mathrm { A } } { \pmb v } _ { i }$ , that is, the diagonal of $\mathbf { U } ^ { \top } \Delta ^ { \mathrm { A } } \mathbf { V }$ . Only these diagonal terms are tracked. The off-diagonal terms $\pmb { u } _ { i } ^ { \top } \pmb { \Delta } ^ { \mathrm { A } } \pmb { v } _ { j }$ with $i \neq j ,$ which describe how Adam couples distinct singular directions, are ignored. This update is sampled from checkpoint 1000 of the AdamW trajectory.

• Muon: exact whitening, a uniform scale across the spectrum.

• SAMuon-lite: the two-level allocation of Section 5 with $k = 1$ , holding the head at the same scale as Muon and amplifying the bulk uniformly $\tan \gamma = 5$ , the measured head gap.

• SAMuon: the log-rank-linear allocation of Equation (9) at $\gamma = 7 . 0 7$ , its tuned optimum at all batch sizes (Table 3), ramping from the anchored head to the flat tail boost over the leading k ranks.

![](images/cf4309d93d04f9f949aa13331e8e0bfe9ea238cf4a9398b235845a195a5eb722.jpg)

![](images/f8ec23bf6c164f8ab9a1acca38afaf5db43ed20af70c277d38a22a9fa8c3e363.jpg)  
Figure 10: In-sample versus out-of-sample spectral profiles along an AdamW trajectory (64Mparameter “modded-nanogpt”, checkpoints from iterations 200–3500), the AdamW analogue of the Muon Figure 8; panels and axes as there, with solid curves out-of-sample and dashed in-sample. As for Muon, the in-sample allocation is systematically more permissive across the bulk (widest in the deep tail) and every in-sample probe yields a large loss reduction, whereas out-of-sample the tail’s contribution collapses; the gap is the batch-refitting capacity that does not transfer to unseen data.

Both the SAMuon and SAMuon-lite allocations are drawn at their post-warmup target profiles. The comparison summarises the argument of Section 4 that the SGD allocation is anti-correlated with the measured profile; Adam dampens but does not overcome the head’s dominance; Muon reallocates several times the scale towards the bulk relative to SGD; SAMuon-lite holds the head at the same scale as Muon and lifts the bulk uniformly towards the measured profile; and SAMuon follows the measured near-head decline itself, thereby providing the closest alignment among the five methods.

## F Convergence proof for the idealised SAMuon update

In this section, we establish convergence guarantees for the idealised exact-whitening form of the SAMuon / SAMuon-lite variants of Algorithm 1. By extending the convergence analysis framework of standard Muon (see Theorem 4.1 in Shen et al. [2025]) to accommodate the head-anchored spectral allocation of Section 5.1, we prove convergence to an ϵ-nuclear-norm stationary point. Although the spectral shaping differs substantially from uniform whitening, the analysis depends on it only through its bounds, so a single argument covers SAMuon, SAMuon-lite, and other related profiles while mathematically formalising the trade-off introduced by the tail boost γ.

## F.1 Setup and notation

Following the standard matrix-structured analysis of Muon, we state the result for one twodimensional parameter matrix, matching the blockwise form of the optimiser update. Accordingly, the theorem below is a single-matrix guarantee; SAMuon applies the same update independently to every eligible matrix in the model. Consider the task of minimising a non-convex objective function $l : \dot { \mathbb { R } ^ { m \times n } }  \mathbb { R }$ . Let $\mathbf { W } _ { t } \in \mathbb { R } ^ { m \times n }$ denote the parameter matrix at iteration $t ,$ and let $r = \operatorname* { m i n } ( m , n )$ denote the maximum possible rank of the matrix.

The optimiser maintains an exponential moving average (EMA) of the stochastic gradients, initialised at zero as in Algorithm 1, and reshapes its spectrum with the head-anchored allocation of Section 5.1.

For $t = 1 , \dots , T$ , the update rule is

$$
{ \bf M } _ { 0 } = { \bf 0 } ,\tag{10}
$$

$$
\mathbf { M } _ { t } = \mu \mathbf { M } _ { t - 1 } + ( 1 - \mu ) \mathbf { G } _ { t }\tag{11}
$$

$$
\mathbf { Q } _ { t } = \mathrm { S A M u o n } ( \mathbf { M } _ { t } )\tag{12}
$$

$$
\mathbf { W } _ { t + 1 } = \mathbf { W } _ { t } - \eta \mathbf { Q } _ { t }\tag{13}
$$

where $\eta$ is a constant effective step size, $\mu \in [ 0 , 1 )$ is the momentum parameter, and $\mathbf { G } _ { t }$ is the stochastic mini-batch gradient. Because this is a single-matrix result, the fixed positive Scion scale κ is absorbed into η. As in Section 3, each momentum buffer is assumed to have rank r. Here SAMuon(·) denotes the anisotropic spectral scaling of Equation (8). Following Shen et al. [2025], we idealise the Newton-Schulz iteration as exact whitening. Then each singular direction receives exactly its prescribed scale, with the head remaining at the same unit scale as in Muon, while $\mathbf { Q } _ { t }$ shares the singular vectors of ${ { \bf { M } } _ { t } }$ . The guarantee therefore covers this idealised exact-whitening update, as is standard for analyses of this type; the finite-step Newton-Schulz, randomised-SVD, and power-iteration approximations run by the deployed algorithms lie outside the bound.

Consequently, if the Singular Value Decomposition (SVD) of the momentum buffer is $\mathbf { M } _ { t } = \mathbf { U } \pmb { \Sigma } \mathbf { V } ^ { \top }$ the preconditioned update evaluates to $\mathbf { Q } _ { t } = \mathbf { U } \hat { \pmb { \Sigma } } \mathbf { V } ^ { \top }$ with modified singular values

$$
\hat { \sigma } _ { i } = \left\{ \begin{array} { l l } { s _ { i } ( t ) } & { i \leq k \quad \mathrm { ( t h e ~ h e a d ~ r e g i o n , ~ w i t h ~ t h e ~ h e a d ~ a n c h o r e d ~ a t ~ } \hat { \sigma } _ { 1 } = s _ { 1 } ( t ) = 1 ) } \\ { \gamma _ { t } } & { i > k \quad \mathrm { ( t h e ~ b o o s t e d ~ t a i l ) } , } \end{array} \right.\tag{14}
$$

following the head-anchored allocation of Equation (8); the two-level SAMuon-lite is the case $k = 1$ and SAMuon takes the log-rank-linear scales of Equation (9). The analysis below depends on this scaling only through its bounds, with the amplification fixed at $\gamma \geq 1$ . As the warmup raises the whole profile monotonically from the all-ones Muon scaling towards its target over the first $T _ { 0 }$ steps (so $1 \leq s _ { i } ( t ) \leq \gamma _ { t } \leq \gamma$ throughout), all modified singular values satisfy

$$
1 \ \leq \ { \hat { \sigma } } _ { i } \ \leq \ \gamma \qquad { \mathrm { f o r ~ a l l ~ } } i \ { \mathrm { a n d ~ } } t .\tag{15}
$$

All bounds below are therefore stated in terms of the post-warmup amplification γ, and hold uniformly over training.

## F.2 Assumptions

We rely on the following standard assumptions regarding the loss landscape and the stochastic oracle, following Shen et al. [2025]. Both landscape assumptions are idealisations relative to Transformer practice: a single global smoothness constant is assumed and the variance bound is assumed uniform along the trajectory.

Assumption F.1 (L-Smoothness). The objective function l has L-Lipschitz continuous gradients with respect to the Frobenius norm. For any W, $\mathbf { W ^ { \prime } } \in \mathbb { R } ^ { m \times n }$

$$
l ( \mathbf { W } ) \leq l ( \mathbf { W } ^ { \prime } ) + \langle \nabla l ( \mathbf { W } ^ { \prime } ) , \mathbf { W } - \mathbf { W } ^ { \prime } \rangle + \frac { L } { 2 } \| \mathbf { W } - \mathbf { W } ^ { \prime } \| _ { F } ^ { 2 }\tag{16}
$$

Assumption F.2 (Bounded Variance). Let $\mathcal { F } _ { t }$ denote the optimisation history before $\mathbf { G } _ { t }$ is sampled. At each step t, the stochastic oracle provides a conditionally unbiased gradient estimator with bounded conditional variance:

$$
\mathbb { E } [ \mathbf { G } _ { t } \mid \mathcal { F } _ { t } ] = \nabla l ( \mathbf { W } _ { t } ) ,\tag{17}
$$

$$
\mathbb { E } [ \| \mathbf { G } _ { t } - \nabla l ( \mathbf { W } _ { t } ) \| _ { F } ^ { 2 } \mid \mathcal { F } _ { t } ] \leq \frac { \sigma ^ { 2 } } { B } ,\tag{18}
$$

where B is the mini-batch size.

Assumption F.3 (Lower-Bounded Objective). The objective function $l ( \mathbf { W } )$ is bounded from below by a global minimum $l ^ { * }$

## F.3 Main convergence theorems

Theorem F.4 (Ergodic Convergence of the Idealised SAMuon Update). Under assumptions F.1 to $F . 3 ,$ using the idealised exact-whitening SAMuon update with constant learning rate η for T iterations yields the following bound on the expected nuclear norm of the true gradient:

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } [ \| \nabla l ( \mathbf { W } _ { t } ) \| _ { * } ] \leq \frac { l ( \mathbf { W } _ { 1 } ) - l ^ { * } } { \eta T } + \frac { L r \gamma ^ { 2 } \eta } { 2 } } \\ & { \quad \quad \quad \quad + \sqrt { r } ( 1 + \gamma ) \left( \frac { \sigma \sqrt { 1 - \mu } } { \sqrt { B ( 1 + \mu ) } } + \frac { \mu \| \nabla l ( \mathbf { W } _ { 1 } ) \| _ { F } } { ( 1 - \mu ) T } + \frac { \sqrt { r } \gamma \eta \mu L } { 1 - \mu } \right) . } \end{array}\tag{19}
$$

The following choices recover the stochastic and deterministic convergence orders of Muon while retaining the same zero-buffer initialisation as Algorithm 1.

Corollary F.5 (Stochastic and Deterministic Convergence Rates). Let $\mathcal { D } = l ( \mathbf { W } _ { 1 } ) - l ^ { * }$ and regard γ as fixed.

(i) In the stochastic setting, set $B ~ = ~ 1 , ~ \eta ~ = ~ \sqrt { ( 1 - \mu ) \mathcal { D } / ( r T L ) }$ , and $1 ~ - ~ \mu ~ =$ min $\{ \sqrt { L \mathcal { D } } / ( \sigma \sqrt { T } ) , 1 \}$ . Then

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } [ \| \nabla l ( \mathbf { W } _ { t } ) \| _ { * } ] \le \mathcal { O } _ { \gamma } \left( \sqrt [ 4 ] { \frac { r ^ { 2 } L \mathcal { D } \sigma ^ { 2 } } { T } } + \sqrt { \frac { r L \mathcal { D } } { T } } + \frac { \sqrt { r } \sigma } { \sqrt { T } } \right) .\tag{20}
$$

(ii) In the deterministic setting $( \sigma = 0 )$ , set $\eta = \sqrt { \mathcal { D } / ( r T L ) }$ and hold $\mu < 1$ fixed independently of T. Then

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \| \nabla l ( \mathbf { W } _ { t } ) \| _ { * } \leq \mathcal { O } _ { \gamma } \left( \sqrt { \frac { r L \mathcal { D } } { T } } \right) .\tag{21}
$$

The boundsfollow by substituting these choices into Theorem F.4 and using the standard consequence $\| \nabla l ( \mathbf { W } _ { 1 } ) \| _ { F } \le \sqrt { 2 L \mathcal { D } }$ of L-smoothness and lower boundedness; the zero-initialisation term is lower order in both cases. Here $\mathcal { O } _ { \gamma }$ suppresses constants depending on thefixed amplification bound γ (and, in the deterministic case, on thefixed momentum coefficient).

Remark F.6. Setting $\gamma = 1$ (and hence $\gamma _ { t } \equiv 1 , \hat { \sigma } _ { i } \equiv 1 )$ specialises Theorem F.4 to Muon with the zero-buffer initialisation of Algorithm 1. This initialisation produces the vanishing initial-gradient term in the theorem; the first-gradient initialisation analysed by Shen et al. [2025] instead produces a vanishing initial-noise term. Both conventions give the stochastic and deterministic convergence orders stated in corollary F.5.

Comparing against this Muon baseline makes the trade-off of the tail boost explicit: while $\gamma$ accelerates descent across the spectrum, it linearly amplifies the stochastic noise floor $\scriptstyle { \big ( } { \frac { \sigma } { \sqrt { B } } } { \big ) }$ and quadratically amplifies the deterministic penalty. Of our design choices, the head anchor $( \hat { \sigma } _ { 1 } = 1 )$ rests on the empirical Edge-of-Stability measurement of Section 4 rather than on the bound, which is indifferent to which rank carries which scale. Furthermore, the warmup schedule withholds the boost during early training, precisely when the momentum tracking error $\| \bar { \boldsymbol { \Delta } } _ { t } \| _ { F }$ that the amplification multiplies is at its largest. The bound itself is shape-agnostic within [1, γ]: the log-rank-linear profile of SAMuon moderates the scales of the leading ranks, so it satisfies the same guarantee at the same γ while pushing only the tail to the peak scale.

## F.4 Detailed proofs

Proof of Theorem F.4. Step 1: The Descent Lemma and the Anisotropic Penalty We begin by substituting the SAMuon parameter update $\mathbf { W } _ { t + 1 } - \mathbf { W } _ { t } = - \eta \mathbf { Q } _ { i }$ into the L-smoothness inequality (assumption F.1):

$$
l ( \mathbf { W } _ { t + 1 } ) \leq l ( \mathbf { W } _ { t } ) - \eta \langle \nabla l ( \mathbf { W } _ { t } ) , \mathbf { Q } _ { t } \rangle + \frac { L \eta ^ { 2 } } { 2 } \| \mathbf { Q } _ { t } \| _ { F } ^ { 2 }\tag{22}
$$

Because $\mathbf { Q } _ { t }$ is constrained by the spectral bounds of Equation (15), its singular values are bounded above by γ. The squared Frobenius norm is geometrically bounded independently of the raw gradient scale:

$$
\Vert \mathbf { Q } _ { t } \Vert _ { F } ^ { 2 } = \sum _ { i = 1 } ^ { r } \hat { \sigma } _ { i } ^ { 2 } \leq \sum _ { i = 1 } ^ { r } \gamma ^ { 2 } = r \gamma ^ { 2 }\tag{23}
$$

## Step 2: Decomposing the Descent Direction

We evaluate the inner product between the true gradient and the anisotropic update by adding and subtracting the momentum buffer $\mathbf { M } _ { t }$

$$
\left. \nabla l ( \mathbf { W } _ { t } ) , \mathbf { Q } _ { t } \right. = \left. \mathbf { M } _ { t } , \mathbf { Q } _ { t } \right. + \left. \nabla l ( \mathbf { W } _ { t } ) - \mathbf { M } _ { t } , \mathbf { Q } _ { t } \right.\tag{24}
$$

## Step 3: The Nuclear Norm Alignment

Since $\mathbf { M } _ { t }$ and $\mathbf { Q } _ { t }$ share identical singular vectors by design, their trace inner product collapses into

a weighted sum of the singular values of $\mathbf { M } _ { t }$ . Because Equation (15) guarantees $\hat { \sigma } _ { i } \geq 1$ for all dimensions, we obtain:

$$
\langle \mathbf { M } _ { t } , \mathbf { Q } _ { t } \rangle = \sum _ { i = 1 } ^ { r } \sigma _ { i } \hat { \sigma } _ { i } \geq \sum _ { i = 1 } ^ { r } \sigma _ { i } = \| \mathbf { M } _ { t } \| _ { * }\tag{25}
$$

## Step 4: Bounding the Amplified Tracking Error

Let $\Delta _ { t } = \nabla l ( \mathbf { W } _ { t } ) - \mathbf { M }$ <sub>t</sub> represent the tracking error. Applying the Cauchy-Schwarz inequality, followed by the upper bound $\| \mathbf { Q } _ { t } \| _ { F } \leq \sqrt { r } \gamma$ derived from Equation (23) gives:

$$
\langle \Delta _ { t } , \mathbf { Q } _ { t } \rangle \geq - \| \Delta _ { t } \| _ { F } \| \mathbf { Q } _ { t } \| _ { F } \geq - \sqrt { r } \gamma \| \Delta _ { t } \| _ { F }\tag{26}
$$

Substituting Equations (25) and (26) into Equation (24) yields:

$$
\langle \nabla l ( \mathbf { W } _ { t } ) , \mathbf { Q } _ { t } \rangle \geq \| \mathbf { M } _ { t } \| _ { * } - \sqrt { r } \gamma \| \Delta _ { t } \| _ { F }\tag{27}
$$

Relating $\| \mathbf { M } _ { t } \| _ { * }$ to the true gradient via the reverse triangle inequality $( \| \mathbf { A } \| _ { * } \geq \| \mathbf { B } \| _ { * } - \| \mathbf { A } - \mathbf { B } \| _ { * } )$ and the matrix norm bound $\| \Delta _ { t } \| _ { * } \leq \sqrt { r } \| \Delta _ { t } \| _ { F }$ gives:

$$
\| \mathbf { M } _ { t } \| _ { * } \geq \| \nabla l ( \mathbf { W } _ { t } ) \| _ { * } - \sqrt { r } \| \Delta _ { t } \| _ { F }\tag{28}
$$

Combining these establishes a lower bound for the descent direction, exposing the $\gamma$ tracking multiplier:

$$
\langle \nabla l ( \mathbf { W } _ { t } ) , \mathbf { Q } _ { t } \rangle \geq \| \nabla l ( \mathbf { W } _ { t } ) \| _ { * } - \sqrt { r } ( 1 + \gamma ) \| \Delta _ { t } \| _ { F }\tag{29}
$$

## Step 5: Bounding the EMA Tracking Error

Define the moving average of the true gradients by $\mathbf { C } _ { 0 } = \mathbf { 0 }$ and

$$
\mathbf { C } _ { t } = \mu \mathbf { C } _ { t - 1 } + ( 1 - \mu ) \nabla l ( \mathbf { W } _ { t } ) .\tag{30}
$$

The tracking error decomposes as

$$
\| \boldsymbol { \Delta } _ { t } \| _ { F } \leq \| \mathbf { M } _ { t } - \mathbf { C } _ { t } \| _ { F } + \| \nabla l ( \mathbf { W } _ { t } ) - \mathbf { C } _ { t } \| _ { F } .\tag{31}
$$

For the first term, unrolling the two moving averages gives

$$
\mathbf { M } _ { t } - \mathbf { C } _ { t } = ( 1 - \mu ) \sum _ { i = 1 } ^ { t } \mu ^ { t - i } \big ( \mathbf { G } _ { i } - \boldsymbol { \nabla } l ( \mathbf { W } _ { i } ) \big ) .\tag{32}
$$

The conditional unbiasedness in assumption F.2 makes the summands a martingale-difference sequence. Hence the cross terms vanish after taking expectations, and Cauchy–Schwarz gives

$$
\mathbb { E } [ \| \mathbf { M } _ { t } - \mathbf { C } _ { t } \| _ { F } ] \le \sqrt { \mathbb { E } [ \| \mathbf { M } _ { t } - \mathbf { C } _ { t } \| _ { F } ^ { 2 } ] }\tag{33}
$$

$$
\leq \frac { \sigma } { \sqrt { B } } ( 1 - \mu ) \sqrt { \sum _ { i = 1 } ^ { t } \mu ^ { 2 ( t - i ) } } \leq \frac { \sigma \sqrt { 1 - \mu } } { \sqrt { B ( 1 + \mu ) } } .\tag{34}
$$

For the second term, let $\mathbf { D } _ { t } = \nabla l ( \mathbf { W } _ { t } ) - \mathbf { C } _ { t }$ . Since $\mathbf { C } _ { 0 } = \mathbf { 0 } , \mathbf { D } _ { 1 } = \mu \nabla l ( \mathbf { W } _ { 1 } )$ . For $t \geq 2$ L-smoothness and Equation (23) give

$$
\| \mathbf { D } _ { t } \| _ { F } \leq \mu \| \mathbf { D } _ { t - 1 } \| _ { F } + \mu \| \nabla l ( \mathbf { W } _ { t } ) - \nabla l ( \mathbf { W } _ { t - 1 } ) \| _ { F }\tag{35}
$$

$$
\leq \mu \| \mathbf { D } _ { t - 1 } \| _ { F } + \mu L \eta \sqrt { r } \gamma .\tag{36}
$$

Unrolling this recursion yields

$$
\| { \bf D } _ { t } \| _ { F } \leq \mu ^ { t } \| \nabla l ( { \bf W } _ { 1 } ) \| _ { F } + \frac { \sqrt { r } \gamma \eta \mu L } { 1 - \mu } .\tag{37}
$$

Combining the two components therefore gives

$$
\mathbb { E } [ \| \mathbf { \Delta } \mathbf { a } _ { t } \| _ { F } ] \le \frac { \sigma \sqrt { 1 - \mu } } { \sqrt { B ( 1 + \mu ) } } + \mu ^ { t } \| \nabla l ( \mathbf { W } _ { 1 } ) \| _ { F } + \frac { \sqrt { r } \gamma \eta \mu L } { 1 - \mu } .\tag{38}
$$

## Step 6: Telescoping the Descent Bound

Taking the expectation of the descent lemma (Equation (22)) and substituting Equations (23) and (29) gives:

$$
\mathbb { E } [ l ( \mathbf { W } _ { t + 1 } ) ] \le \mathbb { E } [ l ( \mathbf { W } _ { t } ) ] - \eta \mathbb { E } [ \| \nabla l ( \mathbf { W } _ { t } ) \| _ { * } ] + \eta \sqrt { r } ( 1 + \gamma ) \mathbb { E } [ \| \mathbf { \Delta } \mathbf { a } _ { t } \| _ { F } ] + \frac { L \eta ^ { 2 } r \gamma ^ { 2 } } { 2 }\tag{39}
$$

Rearranging gives the following bound on the expected nuclear norm of the true gradient:

$$
\eta \mathbb { E } [ \Vert \nabla l ( \mathbf { W } _ { t } ) \Vert _ { * } ] \le \mathbb { E } [ l ( \mathbf { W } _ { t } ) ] - \mathbb { E } [ l ( \mathbf { W } _ { t + 1 } ) ] + \eta \sqrt { r } ( 1 + \gamma ) \mathbb { E } [ \Vert \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } _ { t } \Vert _ { F } ] + \frac { L \eta ^ { 2 } r \gamma ^ { 2 } } { 2 }\tag{40}
$$

Summing this inequality from t = 1 to T, noting that $l ( \mathbf { W } _ { T + 1 } ) \geq l ^ { * }$ (assumption F.3), and dividing by ηT yields:

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } [ \| \nabla l ( \mathbf { W } _ { t } ) \| _ { * } ] \le \frac { l ( \mathbf { W } _ { 1 } ) - l ^ { * } } { \eta T } + \frac { L r \gamma ^ { 2 } \eta } { 2 } + \frac { \sqrt { r } ( 1 + \gamma ) } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } [ \| \pmb { \Delta } _ { t } \| _ { F } ]\tag{41}
$$

Substituting Equation (38) and using $T ^ { - 1 } \sum _ { t = 1 } ^ { T } \mu ^ { t } \leq \mu / ( ( 1 - \mu ) T )$ proves Theorem F.4. □

## G Detailed experimental setup

In this section, we provide the exact hyperparameters and architectural details to ensure full reproducibility of our results. Our codebase and training pipeline are built directly upon the open-source implementation provided by Pethick et al. [2025].

## G.1 Motivation for using Scion as the “Muon” baseline

In our experiments, Muon serves as a crucial strong baseline. Instead of the vanilla Muon implementation, we are in fact using the improved Muon version, Scion [Pethick et al., 2025]. We select Scion over the vanilla Muon implementation for three strategic reasons: (i) Efficiency: Scion replaces the auxiliary Adam optimiser for embeddings with a lightweight signed update rule, reducing memory overhead; (ii) Stability: It employs a theoretically grounded normalisation scheme that decouples the learning rate from model scale, facilitating consistent hyperparameter tuning; and (iii) Performance: Scion consistently demonstrates improved empirical convergence and large-batch scalability over vanilla Muon.

## G.2 Model architecture

All models used in this work are decoder-only Transformers based on the GPT-2 architecture, modified with RMSNorm and Rotary Positional Embeddings (RoPE) (refer to Pethick et al. [2025] for detailed modifications). The 64M and 124M models used for the spectral probing analyses, and the 124M, 300M, and 1B models used for the training experiments, all share the same depth (12 layers) and head dimension, so width is the only scale axis among them. The vocabulary size is padded to 50,304 to improve kernel efficiency. The complete architectural specifications are summarised in Table 2.

Table 2: Model architectures. The 64M, 124M, 300M and 1B models used for probing and/or training experiments. All utilise a head dimension of 128 and a depth of 12 layers.
<table><tr><td>Specification</td><td>64M</td><td>124M</td><td>300M</td><td>1B</td></tr><tr><td>Layers (L)</td><td>12</td><td>12</td><td>12</td><td>12</td></tr><tr><td>Model Dimension  $( d _ { \mathrm { m o d e l } } )$ </td><td>512</td><td>768</td><td>1280</td><td>2560</td></tr><tr><td>Attention Heads (H)</td><td>4</td><td>6</td><td>10</td><td>20</td></tr><tr><td>Head Dimension  $( d _ { h } )$ </td><td>128</td><td>128</td><td>128</td><td>128</td></tr><tr><td>Vocabulary Size</td><td>50,304</td><td>50,304</td><td>50,304</td><td>50,304</td></tr><tr><td>Context Length</td><td>1024</td><td>1024</td><td>1024</td><td>1024</td></tr></table>

## G.3 Training hyperparameters

All experiments train on the FineWeb dataset [Penedo et al., 2024] (100B-token sample; ODC-By 1.0 licence) at global batch sizes of 1M, 2M, and 4M tokens (1024, 2048, and 4096 sequences). The 124M and 300M models use a ∼10B-token budget, with iteration counts of 9600, 4800, and 2400 at the three batch sizes, and the 1B model a ∼20B-token budget. For the 124M model this budget is ∼4× the Chinchilla-optimal allocation [Hoffmann et al., 2022], falling to ${ \sim } 1 . 7 \times$ at 300M and ${ \sim } 1 \times$ at 1B.

Optimiser Settings. All hyperparameters are swept per batch size on the 124M model and transferred across model scales, with the single exception of the AdamW learning rate, which is halved at each scale step $( 2 ^ { - 8 . 5 }  2 ^ { - 9 . 5 }  2 ^ { - 1 0 . 5 }$ at batch size 1024) following the optimal-learning-rate trend for AdamW reported in Pethick et al. [2025]. For AdamW we sweep the learning rate on a half-power-of-two grid around the Pethick et al. [2025] value. For Muon we sweep the spectral radius radius\_2d (the learning-rate factor on the 2D, Muon-effective modules) over the ${ \sqrt { 2 } } .$ -spaced grid {35.36, 50, 70.71, 100, 141.42}. Since this sweep is flat around its optimum, both SAMuon instantiations instead run at the fixed radius 50, the Scion default, and sweep only the tail boost $\gamma$ over {3.54, 5, 7.07, 10, 14.14}, a grid centred on the measured head gap. The complete sweeps are reported in Table 5, and the per-batch optima are collected in Table 3.

Probing-run training setup. The main-text spectral probing analyses are run on the 64M and 124M models, following the Pethick et al. [2025] protocol: 5100 training steps at batch size 512, using the optimiser hyperparameters specified by Pethick et al. [2025]. For the smallest 64M model, the probed Muon trajectory reaches a final validation loss of 3.4292, and the AdamW trajectory probed in appendix B reaches 3.4913 under the same protocol. The width-768 (124M) Muon trajectory probed in appendix C follows exactly the 124M setup of the same protocol, with the learning rate transferred from the 64M model following Pethick et al. [2025], and reaches a final validation loss of 3.2812.

Profile warmup schedule. The spectral shaping is itself warmed up. In the full-budget training runs, the cosine warmup weight $\begin{array} { r } { w _ { t } = \frac { 1 } { 2 } \big ( 1 - \cos ( \bar { \pi } \operatorname* { m i n } ( t / T _ { 0 } , 1 ) ) \big ) } \end{array}$ spans the first $3 0 \%$ of training iterations $( T _ { 0 } = 0 . 3 T )$ , interpolating the whole profile from the Muon scaling through $\gamma _ { t } = 1 + ( \gamma - 1 ) w _ { t }$ and $s _ { i } ( t ) = 1 + ( s _ { i } - 1 ) w _ { t } .$ , after which it is held at its target. The warmup span mirrors the trapezoidal schedule’s final linear-decay phase (28.5% of training) with a matching ∼30% ramp at the start. This lets the model first settle onto a stable Muon trajectory before the shaping fully engages, and applies identically to both instantiations (for SAMuon-lite, it reduces to the single-scale ramp of $\gamma _ { t } ) .$ The shortened runs used only for the token-efficiency measurement instead retain the fixed warmup horizon $T _ { 0 } = 0 . 3 N$ of the corresponding Muon baseline, as detailed in appendix H.

This warmup is not arbitrary since the spectral anisotropy itself emerges slowly over early training. Figure 11 tracks the out-of-sample profile along the Muon trajectory and shows that the head gap rises from near-isotropic (≈ 2 at iteration 50) towards its final level $( \approx 5 )$ over the first few hundred iterations, while the head’s concentration of spectral energy and directional curvature also climbs at the same time. Applying the full shaping from the start would therefore overdrive a spectrum that is not yet highly anisotropic. Warming the profile instead lets the optimiser track the anisotropy as it emerges and supplies a measurement-grounded justification for delaying the full shaping (the span itself mirrors the schedule’s decay phase, as above). The final head gap also anchors the tail-boost search. The γ grid of our experiments is centred on the measured gap (≈5), and the optima sit at or moderately above it, namely $\gamma ^ { * } = 7 . 0 7$ for SAMuon at all batch sizes and between 7.07 and 10 for SAMuon-lite (Table 5).

## G.4 Newton–Schulz Parameters

All Muon-based optimisers in our experiments use the same custom $4 \times 3$ Newton–Schulz scheme: four iterations, each requiring three matrix multiplications. For a normalised input $\mathbf { X } _ { 1 }$ , iteration ℓ applies

$$
\begin{array} { r } { \mathbf { X } _ { \ell + 1 } = a _ { \ell } \mathbf { X } _ { \ell } + b _ { \ell } \left( \mathbf { X } _ { \ell } \mathbf { X } _ { \ell } ^ { \top } \right) \mathbf { X } _ { \ell } + c _ { \ell } \left( \mathbf { X } _ { \ell } \mathbf { X } _ { \ell } ^ { \top } \right) ^ { 2 } \mathbf { X } _ { \ell } , \qquad \ell = 1 , \dots , 4 . } \end{array}\tag{42}
$$

The iteration-specific coefficients in Table 4 are optimised to produce a stable response close to one across the operative range of normalised singular values. Figure 12 shows the resulting response after four iterations; for normalised input singular values $\sigma \geq 0 . 0 2$ , the maximum response error is 0.0069.

![](images/26cc7cdf11e8e67dcecc949799a88410d188b7d2ae230365e4c59a142e89151e.jpg)

![](images/4600d21e709f73748a41ea98ce962005d953a11fb3f99ba8ecc29e051d3c7dcd.jpg)

![](images/1219407894623208164cf30e8b200f14fe46eb098f32f2b7dbfafa0f0828bf87.jpg)  
Figure 11: The spectral anisotropy emerges over early training, motivating the slow warmup. Out-of-sample probing along the Muon trajectory (64M-parameter “modded-nanogpt”), against training iteration (log axis). Left: the head gap (rank-2 over rank-1 optimal spectral allocation ratio, the largest single gap in the measured profile) rises from near-isotropic (≈ 2) at iteration 50 towards its final level of ≈ 5 (red dotted), on which the $\gamma$ search grid of our experiments is centred, within a few hundred iterations, and stays there; the isotropic-curvature assumption (= 1) is violated throughout and increasingly so. Middle: the head’s concentration of spectral energy, $\sigma _ { 1 } ^ { 2 } / \sum _ { i } \sigma _ { i } ^ { 2 }$ Right: the head’s concentration of directional curvature, $\left( \Delta \pmb { \theta } ^ { ( 1 ) \top } \mathbf { H } \Delta \pmb { \theta } ^ { ( 1 ) } \right) / \sum _ { d } \Delta \pmb { \theta } ^ { ( d ) \top } \mathbf { H } \overline { { \Delta } } \tilde { \pmb { \theta } } ^ { ( d ) }$ using the rank-d directional curvature of Equation (7); both concentrate over the same window, from ∼70% to ∼85–95%. Early in training the spectrum is closer to isotropic, so SAMuon and SAMuon-lite delay the full anisotropic shaping and warm the profile from the uniform scaling of Muon to its target over the first 30% of training (much slower than standard warmup schedules).

Table 3: Optimiser hyperparameters. All values are tuned on the 124M model per batch size, with optima listed as bs 1024/2048/4096 and the full sweeps in Table 5. Both SAMuon instantiations run at the fixed spectral radius 50. The profile rank k of SAMuon follows the fixed width rule described below and is not tuned. All values transfer across model scales, except the AdamW learning rate, which is halved at each scale step.
<table><tr><td>Parameter</td><td>AdamW</td><td>Muon (Scion)</td><td>SAMuon-lite</td><td>SAMuon</td></tr><tr><td colspan="5">Global Settings</td></tr><tr><td>Weight Decay</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Gradient Clip</td><td>1.0</td><td>None</td><td>None</td><td>None</td></tr><tr><td>LR Schedule</td><td>Trapezoidal</td><td>Trapezoidal</td><td>Trapezoidal</td><td>Trapezoidal</td></tr><tr><td>LR Warmup</td><td>5%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td colspan="5">Hyperparameter sweeps at 124M (optima at bs 1024/2048/4096)</td></tr><tr><td>LR / radius grid</td><td> $2 ^ { x }$  , half steps</td><td>{35.36, . . . , 141.42}</td><td>fixed at 50</td><td>fixed at 50</td></tr><tr><td>Optimal LR / radius</td><td> $2 ^ { - 8 . 5 } / 2 ^ { - 9 } / 2 ^ { ^ { \cdot } - 8 . 5 }$ </td><td>70.71/50/70.71</td><td>50</td><td>50</td></tr><tr><td>Tail boost γ grid</td><td></td><td></td><td>{3.54, . . . , 14.14} {3.54, . . . , 14.14}</td><td></td></tr><tr><td>Optimal γ</td><td></td><td></td><td> $1 0 / 7 . 0 7 / 1 0$ </td><td>7.07 (all)</td></tr><tr><td>Profile rank k</td><td></td><td></td><td>1</td><td>39/50/71</td></tr><tr><td>Profile warmup</td><td></td><td></td><td>30% (cosine)</td><td>30% (cosine)</td></tr><tr><td>Transfer to 300M and 1B</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="5"></td></tr><tr><td>Across model scales | LR halved per step</td><td></td><td>unchanged</td><td>unchanged</td><td>k by width rule</td></tr></table>

Table 4: Coefficients of the custom $4 \times 3$ Newton–Schulz scheme.
<table><tr><td>Iteration l</td><td> $a \ell$ </td><td> $b _ { \ell }$ </td><td> $c _ { \ell }$ </td></tr><tr><td>1</td><td>5.30697775</td><td>-9.73226547</td><td>4.52926445</td></tr><tr><td>2</td><td>3.99123669</td><td>-4.20899105</td><td>1.18235242</td></tr><tr><td>3</td><td>2.66316843</td><td>-2.16650701</td><td>0.56325209</td></tr><tr><td>4</td><td>1.93040931</td><td>-1.31219244</td><td>0.38289258</td></tr></table>

![](images/e7628a5a762916dcad854328c7ccb734ffc93fd41289f9dddda611ff3f4b0bfc.jpg)  
Figure 12: Response of the custom $4 \times 3$ Newton–Schulz scheme after four iterations. The dashed line denotes exact orthogonalisation, which maps every non-zero singular value to one. For normalised input singular values $\sigma \geq 0 . 0 2$ , the maximum response error is 0.0069.

## G.5 Full sweep results

Table 5 reports all sweep results underlying the tuned hyperparameters in Table $^ { 3 , }$ covering the AdamW learning rate, the Muon spectral radius, and the tail boost of both SAMuon instantiations. Note that SAMuon attains a best-of-sweep loss at least as low as SAMuon-lite at all three batch sizes, with the two effectively tied at batch size 1024, and that its optimum is at $\gamma = 7 . 0 7$ in all cases whereas the SAMuon-lite optimum varies.

## G.6 SAMuon / SAMuon-lite configurations

Both instantiations run at the fixed spectral radius 50 and add the tail boost γ, swept per batch size on the 124M model, with the selected optima summarised in Table 3 and the complete sweeps reported in Table 5. The tail boost is ramped from the Muon value over the cosine profile warmup spanning the first 30% of training. Setting $\gamma = 1$ recovers Muon exactly. SAMuon-lite captures the head with a rank-1 power iteration (Section 5). SAMuon instead requires the leading k singular pairs of the momentum buffer, computed per step with torch.svd\_lowrank. Its profile rank follows the width rule $k ( d _ { \mathrm { { m o d e l } } } ) = \left| 3 2 \sqrt { d _ { \mathrm { { m o d e l } } } / 5 1 2 } \right|$ below, i.e. $k = 3 9 / 5 0 / 7 1$ at the trained widths 768/1280/2560, held fixed and not tuned.

Low-rank estimator hyperparameters. For the 124M, 300M, and 1B models, respectively, SAMuon uses $k = 3 9 / 5 0 / 7 1$ , with torch.svd\_lowrank parameters $q \ : = \ : 4 4 / 5 5 / 7 6$ (five oversampling dimensions) and $\dot { n } _ { \mathrm { i t e r } } = 4 / 5 / 6$ . At the same model scales, SAMuon-lite uses a rank-one power iteration with $1 0 / 1 2 / 1 4$ iterations.

Choice of the profile rank k. The plateau onset measured on the width-512 probing model is k ≈ 32 (the left panel of Figure 2); the width rule $k ( d _ { \mathrm { m o d e l } } ) = \left| 3 2 \sqrt { d _ { \mathrm { m o d e l } } / 5 1 2 } \right.$ transfers this calibration across model widths, giving $k = 3 9 / 5 0 / 7 1$ at the trained widths $7 6 8 / 1 \bar { 2 } 8 0 / 2 5 6 0$ . We present the square-root rule as a calibrated heuristic rather than a derived law. However, we believe this heuristic should be reliable due to the following two reasons. First, the profile is mostly flat beyond the log-rank-linear band, so a moderately mis-set k mostly re-labels plateau ranks whose target scale is close to γ regardless. Second, the sweeps of Table 5 are flat around their optima, indicating that the allocation tolerates moderate mis-specification of its scales. Under this rule SAMuon attains the best or joint-best final loss at every trained width (Section 6.2). A power-law dependence of the spectral structure of the Muon momentum matrices on model size has been independently documented by Magakyan et al. [2026]; the randomised low-rank estimate itself is accurate and well-conditioned at these ranks [Halko et al., 2011], and LiMuon [Huang et al., 2026] provides a precedent for running a randomised SVD of the momentum buffer inside a Muon-style optimiser. Directly measuring the plateau onset at the two largest trained widths, rather than transferring it, is left for future work;

Table 5: Hyperparameter searches at 124M (width 768, FineWeb-100B, ∼10B tokens), for each batch size. Entries are final validation loss, and the per-batch optimum of each sweep is in bold. AdamW is swept over learning rate (2<sup>x</sup>), Muon over its spectral radius, and both SAMuon instantiations over the tail boost γ at the re-anchored radius 50, the Scion default. The Muon optimum shifts with batch size (70.71/50/70.71), whereas the SAMuon optimum is $\gamma = 7 . 0 7$ at all batch sizes. Searches at the superseded radius 70.71 are omitted for the SAMuon instantiations.
<table><tr><td>Hyperparameter</td><td>bs 1024</td><td>bs 2048</td><td>bs 4096</td></tr><tr><td colspan="3">AdamW — learning rate (log2)</td></tr><tr><td>-10.5</td><td>3.2363</td><td></td></tr><tr><td>-10</td><td></td><td>3.2620</td></tr><tr><td>-9.5 -9</td><td>3.2113</td><td>3.3266</td></tr><tr><td></td><td>3.2410</td><td></td></tr><tr><td>-8.5 3.2105</td><td></td><td>3.3050</td></tr><tr><td>-8</td><td>3.2462</td><td></td></tr><tr><td>-7.5</td><td>3.2360</td><td>3.3392</td></tr><tr><td>-7</td><td>3.2963</td><td></td></tr><tr><td>-6.5</td><td>一</td><td>3.4059</td></tr><tr><td>Muon — spectral radius</td><td></td><td></td></tr><tr><td>35.36</td><td>3.1625 3.1768</td><td>3.2046</td></tr><tr><td>50</td><td>3.1641 3.1726</td><td>3.1979</td></tr><tr><td>70.71</td><td>3.1622 3.1743</td><td>3.1953</td></tr><tr><td>100</td><td>3.1641 3.1757</td><td>3.1965</td></tr><tr><td>141.42</td><td>3.1649 3.1761</td><td>3.1986</td></tr><tr><td>SAMuon — γ at radius 50</td><td></td><td></td></tr><tr><td>3.54</td><td>3.1473 3.1535</td><td>3.1734</td></tr><tr><td>5</td><td>3.1502</td><td>3.1686</td></tr><tr><td>7.07</td><td>3.1445 3.1484</td><td>3.1658</td></tr><tr><td>10</td><td>3.1435</td><td></td></tr><tr><td>14.14</td><td>3.1442 3.1486 3.1504</td><td>3.1662</td></tr><tr><td>SAMuon-lite — γ at radius 50</td><td></td><td></td></tr><tr><td>3.54</td><td>3.1462 3.1553</td><td>3.1724</td></tr><tr><td>5</td><td>3.1445 3.1523</td><td>3.1699</td></tr><tr><td>7.07</td><td>3.1449 3.1511</td><td>3.1680</td></tr><tr><td>10</td><td>3.1516</td><td>3.1678</td></tr><tr><td></td><td>3.1438</td><td></td></tr><tr><td>14.14</td><td>3.1550</td><td></td></tr></table>

the width-768 probing of appendix C shows a modest shift of the approximate plateau onset in the direction predicted by the rule, although additional widths are needed to test the proposed scaling.

## G.7 Wall-clock cost of the optimiser update step

To quantify the per-iteration overhead discussed in Section 5.2, we time one full training iteration of the 1B model under its real batch-size-1024 configuration on a single NVIDIA RTX 6000 Ada GPU; this benchmark serves the wall-clock evaluation only, while the training runs of Section 6.1 execute on A100 GPUs. Per iteration, the GPU performs 64 microbatch forward+backward passes. Table 6 reports the GPU wall-clock breakdown. The forward+backward computation dominates at 91.2% of the Muon iteration, with the Newton-Schulz whitening accounting for the remaining 8.8%. On top of this, the rank-1 power iteration of SAMuon-lite adds 41.2 ms (0.5% of the Muon iteration), confirming its wall-clock parity with Muon. The rank-k torch.svd\_lowrank call of SAMuon adds 657.8 ms (7.4%), nearly as much as the Newton-Schulz whitening itself despite its far lower FLOP count (O(mnk) against O(mnr), Section 5.2), which reflects the unoptimised kernel rather than the arithmetic cost of the method.

## H Measuring the token-efficiency improvement

For each model and batch-size setting, we train the tuned-Muon baseline for N iterations and use its final validation loss as the target. We then ask how short a SAMuon or SAMuon-lite run can be while

Table 6: Per-GPU wall-clock cost of one training iteration (1B model, batch-size-1024 configuration), measured on a single NVIDIA RTX 6000 Ada GPU. The full Muon iteration is the forward+backward computation plus the Newton-Schulz whitening (8916 ms in total), and percentages are relative to this total. The SAMuon and SAMuon-lite rows give the additional cost of each variant’s head estimate on top of the Muon iteration.
<table><tr><td>Component</td><td>ms/iter</td><td>% of Muon iter</td></tr><tr><td>Forward + backward (×64 microbatches, compiled, bf16)</td><td>8133.8</td><td>91.2%</td></tr><tr><td>Muon: Newton-Schulz (4 × 3)</td><td>782.1</td><td>8.8%</td></tr><tr><td>SAMuon: + torch.svd_lowrank</td><td>657.8</td><td>7.4%</td></tr><tr><td>SAMuon-lite: + power iteration (bf16)</td><td>41.2</td><td>0.5%</td></tr></table>

Table 7: Token-efficiency improvement of SAMuon over tuned Muon at matched final validation loss. N is the tuned-Muon schedule length, and $T ^ { \star }$ is the estimated SAMuon or SAMuon-lite schedule length that matches Muon’s final loss, found using Algorithm 2 and checked with a confirmation run. SAMuon uses spectral radius 50; Muon uses its tuned radius for each batch size. The shaping warmup remains fixed at 0.30N absolute steps, making the reported improvements conservative. The 1B runs use 20B tokens; the others use 10B.
<table><tr><td colspan="4"></td><td colspan="2">Token-effic. improv.  $1 - T ^ { \star } / N$ </td></tr><tr><td>Model</td><td>Batch</td><td>Tokens</td><td>N</td><td>SAMuon</td><td>SAMuon-lite</td></tr><tr><td rowspan="3">124M (w768)</td><td>1024</td><td>10B</td><td>9600</td><td>20.3%</td><td>20.3%</td></tr><tr><td>2048</td><td>10B</td><td>4800</td><td>23.8%</td><td>21.5%</td></tr><tr><td>4096</td><td>10B</td><td>2400</td><td>24.0%</td><td>22.1%</td></tr><tr><td rowspan="3">300M (w1280)</td><td>1024</td><td>10B</td><td>9600</td><td>17.7%</td><td>15.9%</td></tr><tr><td>2048</td><td>10B</td><td>4800</td><td>18.8%</td><td>17.1%</td></tr><tr><td>4096</td><td>10B</td><td></td><td>not trained</td><td></td></tr><tr><td rowspan="3">1B (w2560)</td><td>1024</td><td>20B</td><td>19200</td><td>13.3%</td><td>13.3%</td></tr><tr><td>2048</td><td>20B</td><td></td><td>not trained</td><td></td></tr><tr><td>4096</td><td>20B</td><td>4800</td><td>15.6%</td><td>14.4%</td></tr></table>

still reaching that target. Every candidate run of length T uses a complete learning-rate schedule, including the final warmdown, compressed into T iterations. We therefore compare final losses from fully annealed runs, rather than comparing the loss curves at an intermediate iteration.

If $T ^ { \star }$ is the estimated number of iterations needed to match the target, the token-efficiency improvement is

$$
{ \mathrm { i m p r o v e m e n t } } = 1 - { \frac { T ^ { \star } } { N } } .\tag{43}
$$

For example, matching the baseline after 0.8N iterations gives a 20% improvement.

## H.1 Iteration matching

Longer fully annealed runs generally achieve lower final validation loss over the range considered. We therefore search for the run length at which the variant reaches the baseline target. We start with candidate lengths of 0.70N and 0.90N, corresponding to improvements of 30% and 10%. The shorter run should finish above the target loss, whereas the longer run should match or improve upon it. If they do not bracket the target, we expand the search interval.

Within this interval, false-position interpolation uses the two endpoint losses to propose the next run length. We execute a complete, fully annealed run at that length and replace the shorter or longer endpoint according to whether the run misses or reaches the target. We repeat this process until the two endpoints differ by at most 0.005N iterations. Finally, we estimate $\hat { T } ^ { \star }$ by linear interpolation between the runs nearest the target and execute a slightly longer confirmation run to check that the target is reached.

Final validation losses vary slightly between runs, partly because the optimiser uses a randomised low-rank SVD. The search therefore retains executed runs on both sides of the target rather than relying on a single interpolated estimate. The additional, slightly longer run provides a conservative check that the reported matched length reaches the target despite this variation.

Algorithm 2 Iteration matching for the token-efficiency measurement.   
Require: tuned-Muon schedule length N and final validation loss; SAMuon variant   
1: Verify that a fully annealed N-iteration variant run reaches the baseline loss   
2: Set the shorter and longer candidate lengths to 0.70N and 0.90N   
3: Train and evaluate fully annealed variant runs at both candidate lengths   
4: Expand the interval if necessary until the shorter run misses the target and the longer run reaches   
it   
5: while the candidate lengths differ by more than 0.005N do   
6: Choose the next length by false-position interpolation between the endpoint lengths and   
losses   
7: Train and evaluate a fully annealed variant run at the proposed length   
8: if the final loss is above the baseline target then   
9: Replace the shorter endpoint with this run   
10: else   
11: Replace the longer endpoint with this run   
12: end if   
13: end while   
14: Estimate $T ^ { \star }$ by linear interpolation between the runs nearest the target   
15: Run a schedule half the final interval width longer than $T ^ { \star }$ and confirm that it reaches the target   
16: return $1 - T ^ { \star } / N$

The SAMuon shaping warmup is fixed at 0.30N absolute iterations and is not shortened with the candidate schedule. A matched run with $T < N$ therefore spends more than 30% of its iterations in warmup (between 34.6% and 39.5% for the settings in Table 7). This puts the shortened runs at a disadvantage, so the reported token-efficiency improvements are conservative.

## I Per-Module Sign Alignment

The main-text probe of Section 4 applies each weight matrix’s rank-d direction in the fixed orientation $- \mathbf { \boldsymbol { u } } _ { j , d } \mathbf { \boldsymbol { v } } _ { j , d } ^ { \top }$ that a real optimiser uses, and clusters by rank across weight matrices. The resulting profile therefore averages over an inter-block sign structure, i.e. how often a weight matrix’s buffer direction agrees with the out-of-sample gradient, which Figure 13 records. This agreement measures how trustworthy a direction is. A direction that agrees with the gradients of unseen batches makes transferable progress, whereas a direction that flips sign from batch to batch has a greater chance of disrupting training. The tail probes align poorly under this measure, which explains why the in-sample analysis of appendix D is far more optimistic along the tail: in-sample, each direction is evaluated on the batch whose gradient formed the buffer, where its alignment is much better. The alignment also differs by weight-matrix type. The attention projections, and the query and key projections in particular, show the worst tail alignment, so their tail directions deserve the least trust; the QK-clip mechanism used in large-scale Muon training [Kimi Team, 2025] restrains exactly these matrices, so the probing framework predicts training behaviour observed in practice. This cross-matrix agreement further determines whether contributions add or cancel within a rank, and is one source of the mostly flat bulk.

![](images/e1375d8dadc7a4e02acff1fee9ea12c50ca8d851e2c9886c8415a0849443361e.jpg)  
Figure 13: Per-weight-matrix sign-alignment probability. For the checkpoint at iteration 1000, the probability that a weight matrix’s rank-d buffer direction ${ \pmb u } _ { j , d } { \pmb v } _ { j , d } ^ { \top }$ is aligned (has positive projection) with the out-of-sample gradient, estimated over 20 held-out batches. Rows group the six weight types (attention query, key, value, and output projections; multilayer perceptron (MLP) up- and down-projections), each shown for all twelve layers L0–L11; columns run from the deep tail (rank 512) to the head (rank 1, log axis). Blue marks near-certain alignment, red near-certain anti-alignment, and white a coin-flip. Two patterns stand out. First, the top few ranks are almost always aligned (the deep-blue right edge across all weight matrices), so the head’s orientation is effectively deterministic. Second, in the tail the four attention projections drift towards 0.5 and below, i.e. their low-energy directions flip sign freely from batch to batch, whereas the MLP projections stay strongly aligned across almost the entire spectrum. The query and key projections show the worst tail alignment, so their tail directions are the least trustworthy; compare this observation with the QK-clip practice of Kimi Team [2025]. This weight-matrix-dependent sign-flip structure is the inter-block variation that our per-rank clustering averages over (Section 7); the cross-batch cancellation it produces in the tail is one source of the mostly flat bulk.