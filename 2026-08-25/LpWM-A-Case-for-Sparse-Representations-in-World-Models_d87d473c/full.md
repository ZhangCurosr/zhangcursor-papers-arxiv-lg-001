# LpWM: A Case for Sparse Representations in World Models

Yilun Kuang\* NYU, AMI Labs

Yash Dagade Duke University

Quentin Le Lidec NYU, AMI Labs

Lucas Maes Mila, AMI Labs

Randall Balestriero Brown University, AMI Labs

Yann LeCun NYU, AMI Labs

## Abstract

Joint-embedding predictive architectures (JEPAs) learn latent dynamics for planning and avoid representation collapse by matching features to maximum-entropy distributions such as isotropic Gaussians, yielding dense representations. However, it is unclear whether dense representations are the most favorable geometry for modeling dynamics. In this work, we ask whether a different geometry—sparse representations—can make action-conditioned latent dynamics easier to model, and what dynamical structure emerges from such representations. We first show that nonlinear Lipschitz dynamics can be approximated arbitrarily well by actionconditioned linear dynamics in a sufficiently high-dimensional one-hot latent space, with rollout error vanishing as the dimension grows. This motivates distributed sparse representations as a practical relaxation of one-hot sparsity. We introduce LpWorldModel (LpWM), a JEPA model regularized with Rectified Distribution Matching Regularization (RDMReg) to match encoder features to a Rectified Generalized Gaussian distribution, yielding non-negative sparse codes. Empirically, sparsity lowers the predictor complexity required for successful planning: on PushT, sparse LpWM outperforms dense LeWM by up to 57% in planning success at intermediate predictor capacities. This advantage also extends beyond Gaussian distribution matching, with LpWM outperforming dense VICReg representations across multiple predictor families. We further find that the learned sparse representations are mode-factored, with support encoding discrete dynamical regimes and feature magnitudes capturing continuous within-regime state. Together, these results suggest that sparse representations can reduce the predictor complexity required for control while revealing interpretable structure.

## 1 Introduction

Joint-embedding predictive architectures (JEPA) have emerged as a practical recipe for model-based control from pixels: an encoder maps observations to a latent space, a predictor rollouts the latent forward under actions, and planning is carried out by model-predictive control (MPC) in latent space [LeCun et al., 2022]. In the case of end-to-end training, where the encoder and predictor are jointly trained, an anti-collapse loss is necessary to prevent the feature representation from degenerating to a constant or low-rank subspaces [Jing et al., 2021].

Prior methods such as PLDM [Sobal et al., 2025] use VCReg [Bardes et al., 2022], while LeWM [Maes et al., 2026b] uses SIGReg [Balestriero and LeCun, 2025] to prevent collapse through information maximization, yielding dense representations in which all latent coordinates are nonzero almost surely. While effective for preventing collapse, it remains unclear whether dense features are the most favorable geometry for modeling controlled dynamics.

![](images/f6b700156d503c8708b75f6ba5fafb61545422f59ef392411d8707300ecabde7.jpg)  
(a) LpWM Schematic Diagram

![](images/a907715b4e257f7afaf9b956b0adf2a59f5f6549486f6438995616c7e0d1f02d.jpg)  
(b) PushT Planning Performance.  
Figure 1: LpWorldModel: a JEPA model with sparse representations. (a) An encoder f maps observation $\mathbf { o } _ { t }$ to non-negative, sparse latents $\mathbf { z } _ { t }$ via a terminal (Rep)ReLU function. The predictor g takes inputs $\mathbf { z } _ { t }$ and actions $\mathbf { a } _ { t }$ to predict $\hat { \mathbf { z } } _ { t + 1 } .$ , compared against the encoded target $\mathbf { z } _ { t + 1 } = f ( \mathbf { o } _ { t + 1 } )$ RDMReg [Kuang et al., 2026b] matches per-timestep latent marginals to a Rectified Generalized Gaussian, preventing collapse and inducing sparsity. (b). Arrows indicate the success-rate difference between LpWM and LeWM: ↑ denotes higher performance for LpWM, while ↓ denotes lower performance. LpWM outperforms LeWM with more linear predictor variants (MLPoLTV(k), MLPoLTI(k), and LTI(k). See full specifications in Table 1) by up to 57% in success rate, while performing on par with LeWM using the high-capacity DiT predictors. This suggests that sparse representations induce a latent space with simpler dynamics to learn compared to their dense counterparts.

In this work, we ask whether sparse representations provide a more favorable geometry for modeling action-conditioned dynamics. Our motivation comes from an idealized limit: for Lipschitz controlled dynamics on a compact state space, sufficiently high-dimensional one-hot representations can approximate the dynamics with action-conditioned linear latent dynamics (Proposition 1), with rollout error vanishing as the dimension grows (Corollary 1). While exact one-hot codes are impractical, this suggests distributed sparse representations, in which a nontrivial fraction of latent coordinates are exactly zero, as a learnable relaxation that may simplify the dynamics.

We introduce LpWorldModel (LpWM), a JEPA world model regularized with RDMReg [Kuang et al., 2026b] to learn non-negative, sparse codes. RDMReg matches rectified feature distributions to Rectified Generalized Gaussian targets, whose underlying Generalized Gaussian distributions are maximum-entropy under expected $\ell _ { p } .$ -norm constraints with $p \leq 1$ yields sparse targets.

Empirically, sparsity lowers the predictor complexity required for successful planning: predictors that fail over dense codes succeed over sparse ones. The resulting representations are also mode-factored: their support encodes discrete dynamics regimes, while feature magnitudes encode continuous withinregime state. Together, these results suggest that sparse latent geometry can make dynamics both easier to model and more interpretable.

## Contributions.

• LpWorldModel. We introduce LpWM, a JEPA world model that uses RDMReg to learn non-negative sparse latent representations for latent dynamics modeling and planning.

• Sparse representations for latent linearization. We formalize the intuition that sufficiently high-dimensional one-hot representations can approximate nonlinear controlled dynamics with linear latent transitions, motivating distributed sparse codes as a practical relaxation.

• Lower-complexity latent dynamics. We show empirically that sparsity reduces the predictor capacity needed for successful planning: while Wall dynamics are already solved by linear predictors, on the harder PushT environment sparse LpWM substantially outperforms dense LeWM with intermediate-capacity predictors, by up to 57% in success rate.

• Mode-factored latent dynamics. We find that sparse representations organize dynamics into interpretable factors, with support encoding discrete dynamical regimes and feature magnitudes capturing continuous within-regime state.

## 2 Method

## 2.1 LpWM: JEPA with Rectified Distribution Matching Regularization (RDMReg)

Let $\mathbf { z } _ { t + 1 } = f _ { \pmb { \theta } } ( \mathbf { x } _ { t + 1 } ) \in \mathbb { R } ^ { d }$ be the encoder representation of the observation $\mathbf { x } _ { t + 1 }$ and $\hat { \mathbf { z } } _ { t + 1 } =$ $g _ { \phi } ( \mathbf { z } _ { t } , \mathbf { a } _ { t } )$ be the predictor output with action inputs $\mathbf { a } _ { t } \in \mathbb { R } ^ { d _ { a } }$ . The action-conditioned JEPA uses the following loss function to jointly train the encoder $f _ { \theta }$ and the predictor $g _ { \phi } \mathrm {  i }$

$$
\operatorname* { m i n } _ { \theta , \phi } \quad \| \hat { \mathbf { z } } _ { t + 1 } - \mathbf { z } _ { t + 1 } \| _ { 2 } + \lambda _ { \mathrm { R D M R e g } } \cdot \mathcal { R } ( \mathbf { z } _ { t + 1 } )\tag{1}
$$

where $\lambda _ { \mathrm { R D M R e g } }$ is a hyperparameter and $\mathcal { R } ( \cdot )$ is an anti-collapse regularizer. We focus on the setting where $\mathcal { R } ( \cdot )$ is the Rectified Distribution Matching Regularization (RDMReg) [Kuang et al., 2026b]:

$$
\begin{array} { r } { \mathcal { R } ( \mathbf { z } ) = \mathbb { E } _ { \mathbf { c } \sim \mathrm { U n i f } ( \mathbb { S } _ { \ell _ { 2 } } ^ { d - 1 } ) } \left[ \mathcal { L } ( \mathbb { P } _ { \mathbf { c } ^ { \top } \mathbf { z } } \vert \vert \mathbb { P } _ { \mathbf { c } ^ { \top } \mathbf { y } } ) \right] } \end{array}\tag{2}
$$

where the random projection vector $\mathbf { c } \in \mathbb { R } ^ { d }$ is sampled uniformly from the $\ell _ { 2 }$ unit sphere $\mathbb { S } _ { \ell _ { 2 } } ^ { d - 1 } : =$ $\{ \mathbf { x } \in \mathbb { R } ^ { d } \mid \| \mathbf { x } \| _ { 2 } = 1 \}$ , the target random vector $\begin{array} { r } { \mathbf { y } \sim \prod _ { i = 1 } ^ { d } } \end{array}$ $\operatorname { R e L U } ( \mathcal G N _ { p } ( \mu , \sigma ) )$ follows the Rectified Generalized Gaussian (RGG) distribution, $\mathbb { P } _ { x }$ denotes the probability distribution for the random variable $x ,$ and $\mathcal { L } ( \cdot \| \cdot )$ is a loss function that compute discrepancies between probability distributions. In our case, we follow Kuang et al. [2026b] and set $\mathcal { L } ( \cdot \| \cdot )$ to be the 2-Wasserstein distance. By default we choose $\mu = 0 , \sigma = \sqrt { { 1 } / { 2 } }$ , and $p = 1$ and hence our target distribution reduces to the Rectified Laplace distribution [Kuang et al., 2026b]. Note that when $\mu = 0 , \sigma = 1 , p = 2$ , and in the absence of ReLU, we recover dense isotropic Gaussian used in LeWM [Maes et al., 2026b]

## 2.2 Architecture and Rectification

Following Maes et al. [2026b], our encoder $f _ { \theta }$ is a ViT [Dosovitskiy et al., 2021] with the output CLS token passed through a three-layer MLP projector to produce the embedding $\mathbf { z } \in \mathbb { R } ^ { d }$ . The predictor $g _ { \phi }$ is a Transformer with AdaLN-zero action conditioning [Peebles and ${ \bar { \mathrm { X i e } } } ,$ 2023] and a final three-layer MLP projector that predicts the next embedding $\hat { \mathbf { z } } _ { t + 1 }$

To induce exact zeros, both encoder and predictor MLP projector outputs pass through a reparameterized ReLU [Wang et al., 2024, Kuang et al., 2026b]: RepReL $\bar { \mathrm { U } } ( x \bar { ) } = \mathrm { s g } ( \mathrm { R e } \bar { \mathrm { L U } } ( x ) \bar { ) } +$ GeLU $\mathrm { J } ( x ) - \mathrm { s g } ( \mathrm { G e L U } ( x ) )$ , where $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operator. In the forward pass, RepReL $\mathrm { U } ( x ) = \mathrm { R e L U } ( x )$ , yielding exact zeros, while in the backward pass gradients flow through GeLU(x) to mitigate the dying-ReLU pathology [Hendrycks and Gimpel, 2023]. Importantly, LpWM does not rely on stop-gradient for collapse prevention: standard ReLU with RDMReg trains without collapse and yields exactly sparse representations [Kuang et al., 2026b]. We therefore use RepReLU as an optimization safeguard against dead neurons rather than as an essential component of the method. See Figure 1a for LpWM architecture.

## 2.3 Planning with the Learned World Model

At test time the world model serves as the dynamics model for model-predictive control (MPC): we encode the current and goal observation images to $\mathbf { z } _ { 0 }$ and $\mathbf { z } _ { g } .$ and search for an action sequence whose predicted latent rollout minimizes the goal cost $\mathcal { C } = \lVert \hat { \mathbf { z } } _ { T } - \mathbf { z } _ { g } \rVert _ { 2 }$ using the Cross-Entropy Method (CEM) [Rubinstein, 1999, Zhou et al., 2025]. See additional experimental details in Appendix D.

## 3 Does Sparsity Simplify Latent Dynamics?

In the following sections, we ask whether a sparse code makes the action-conditioned latent dynamics simpler by requiring a less expressive predictor than a dense code. We first show that a sufficiently

![](images/fac5ad0416c21e753ed9527b22cbd67d54b304d4c531644450278b524a5e4300.jpg)  
(a) Support Jaccard Heatmap.

![](images/8f112600cc485c00da7bd30bfc2502fe40a45cb90b972e0a4055b0ab68d70a17.jpg)  
(b) State Decoding.

![](images/6864fe6bec52f1062e82579de05f84535e9637b919e9529b7e753e41036cc92a.jpg)  
(c) Planning Evaluation  
Figure 2: The sparse support recovers the piecewise structure of the dynamics. (a) Support Jaccard heatmap over a $2 0 \times 2 0$ grid: the Jaccard index of each cell's support with a fixed reference cell (white +) is high within the reference zone and drops across zone boundaries (dashed), for a model trained with rendered zones (top) and one with no visual cues (bottom). (b) The binary support decodes the ground-truth zone identity as accurately as the full embedding, while agent position is decoded better from the magnitudes than from the support. (c) Planning success vs. horizon (closed-loop, 3-seed mean±std). Goal locations of the dot agent are sampled randomly. Both sparse variants beat dense LeWM at every horizon.

high-dimensional sparse code renders the latent dynamics exactly linear under appropriate assumptions (Proposition 1). We then relax this to a learnable distributed sparse code and test, along a predictor-complexity ladder (Table 1), whether a simpler predictor plans better under sparsity.

## 3.1 Linearization of Nonlinear Dynamics with Sparse Representations

In Proposition 1, we show that compact controlled systems with Lipschitz dynamics admit finite one-hot sparse encodings with exactly linear latent dynamics. This is closely related to classical symbolic abstractions based on state-space quantization [Pola et al., 2008], which we reinterpret as one-hot sparse latent codes.

Proposition 1 (Linear Approximation of Nonlinear Dynamics with Sparse Representations). Let $\boldsymbol { \mathcal { X } } \subset \mathbb { R } ^ { d }$ be a compact state space and let $\mathcal { A } \subset \mathbb { R } ^ { p }$ be a compact continuous action space. Consider a deterministic discrete-time controlled dynamical system $\mathbf { x } _ { t + 1 } = f ( \mathbf { x } _ { t } , \mathbf { a } _ { t } )$ where $f : \mathcal { X } \times \mathcal { A }  \mathcal { X }$ Assume that f is uniformly Lipschitz in the state variable, i.e. there exists a finite constant $L \geq 0$ such that $\| f ( \mathbf { x } , \mathbf { a } ) - { f ( \mathbf { y } , \mathbf { a } ) } \| \leq L \| \mathbf { x } - \mathbf { y } \|$ for every $\mathbf { x } , \mathbf { y } \in { \mathcal { X } }$ and $\mathbf { a } \in { \mathcal { A } }$ Then, for every $\varepsilon > 0$ there exist 1) a finite latent dimension $N ; 2 )$ a one-hot sparse encoder $E _ { \varepsilon } : \mathcal { X } \to \bar { \mathbb { R } ^ { N } } ; 3 )$ a decoder $D _ { \varepsilon } : \{ \mathbf { e } _ { 1 } , \dots , \mathbf { \bar { e } } _ { N } \} \to \mathcal { X } ;$ and 4) an action-conditioned family of matrices $P _ { \varepsilon } : \mathcal { A }  \{ 0 , 1 \} N \times N$ $\mathbf { a } \mapsto \mathbf { P } _ { \varepsilon } ( \mathbf { a } )$ , such that the latent dynamics ${ \bf z } _ { t + 1 } = { \bf P } _ { \varepsilon } ( { \bf a } _ { t } ) { \bf z } _ { t }$ are exactly linear in $\mathbf { z } _ { t } ,$ and the decoded one-step prediction satisfies

$$
\operatorname* { s u p } _ { \mathbf { x } _ { t } \in \mathcal { X } , \mathbf { a } _ { t } \in \mathcal { A } } \| f ( \mathbf { x } _ { t } , \mathbf { a } _ { t } ) - D _ { \varepsilon } ( \mathbf { P } _ { \varepsilon } ( \mathbf { a } _ { t } ) E _ { \varepsilon } ( \mathbf { x } _ { t } ) ) \| \leq ( L + 1 ) \varepsilon .\tag{3}
$$

## Proof. See Appendix G.1

In Corollary 1, we consider one example of a compact domain and show that the rollout error with respect to the ground-truth dynamics vanishes as the encoding dimension grows.

Corollary 1 (Sparsity in High-Dimensions). Let the state space be $\mathcal { X } = [ 0 , 1 ] ^ { d }$ and consider dividing each coordinate into n equal intervals, producing $N = n ^ { d }$ grid cells. We represent each cell by its center with the covering radius $\begin{array} { r } { \varepsilon _ { N } = \frac { \sqrt { d } } { 2 } N ^ { - 1 / d } } \end{array}$ . Hence the one-step prediction error satisfies

$$
\operatorname* { s u p } _ { \mathbf { x } , \mathbf { a } } \| f ( \mathbf { x } , \mathbf { a } ) - D _ { \varepsilon _ { N } } ( \mathbf { P } _ { \varepsilon _ { N } } ( \mathbf { a } ) E _ { \varepsilon _ { N } } ( \mathbf { x } ) ) \| \leq \frac { \sqrt { d } } { 2 } ( L + 1 ) N ^ { - 1 / d } = O ( N ^ { - 1 / d } ) .\tag{4}
$$

For any fixed rollout horizon H, $\begin{array} { r } { \| { { \bf x } _ { H } } - \hat { { \bf x } } _ { H } \| \le \frac { \sqrt { d } } { 2 } N ^ { - 1 / d } \sum _ { k = 0 } ^ { H } L ^ { k } } \end{array}$ and lim $\mathbf { \xi } _ { | N  \infty  } \| \mathbf { x } _ { H } - \hat { \mathbf { x } } _ { H } \| = 0$

Table 1: Predictors with Different Levels of Complexity. $\hat { \mathbf { z } } _ { t + 1 }$ is the predicted latent, $\mathbf { z } _ { t - i }$ past latents, and $\mathbf { a } _ { t }$ the action. AdaLN predictors are Transformer-based models with Adaptive Layer Normalization for action conditioning, i.e. DiT architecture [Peebles and Xie, 2023]. The Deep-AdaLN(k) predictor is the one used in LeWM [Maes et al., 2026b]. LTI denotes a linear time-invariant predictor with fixed operators $\mathbf { A } _ { i } , \mathbf { B } ,$ while LTV denotes a linear time-varying predictor with statedependent operators $\mathbf { \bar { A } } _ { i } ( \mathbf { z } _ { t } ) , \mathbf { B } ( \mathbf { z } _ { t } )$ . We use MLPo to denote one extra nonlinear transform. The history k denotes context length, with 1 the single-frame case. The output link is σ = identity for dense LeWM and $\sigma = \mathrm { R e p R e L U }$ for sparse LpWM. See Appendix B for details.
<table><tr><td>Predictor</td><td>Layers</td><td>History (k)</td><td colspan="3">Functional form</td></tr><tr><td>Deep-AdaLN(k)</td><td>6</td><td>3</td><td></td><td> $\hat { \mathbf { z } } _ { t + 1 } = \sigma \Big ( \mathrm { A d a L N } ^ { ( 6 ) } \big ( \mathbf { z } _ { t - k + 1 : t } , \mathbf { a } _ { t } \big ) \Big )$ </td><td rowspan="5"></td></tr><tr><td>Shallow-AdaLN(k)</td><td>1</td><td>3</td><td></td><td> $\hat { \mathbf { z } } _ { t + 1 } = \sigma \Big ( \mathrm { A d a L N } ^ { ( 1 ) } ( \mathbf { z } _ { t - k + 1 : t } , \mathbf { a } _ { t } ) \Big )$ </td></tr><tr><td>MLPoLTV(k)</td><td>1</td><td>3</td><td></td><td> $\begin{array} { r } { \hat { { \bf { z } } } _ { t + 1 } = \sigma \Big ( { \bf { W } } \mathrm { R e L U } ( \sum _ { i = 0 } ^ { k - 1 } { \bf { A } } _ { i } ( { \bf { z } } _ { t } ) { \bf { z } } _ { t - i } + { \bf { B } } ( { \bf { z } } _ { t } ) { \bf { a } } _ { t } ) \Big ) } \end{array}$ </td></tr><tr><td>MLPoLTI(k)</td><td>1</td><td>3</td><td></td><td> $\hat { \mathbf { z } } _ { t + 1 } = \sigma \Big ( \mathbf { W } \operatorname { R e L U } ( \sum _ { i = 0 } ^ { k - 1 } \mathbf { A } _ { i } \mathbf { z } _ { t - i } + \mathbf { B } \mathbf { a } _ { t } ) \Big )$ </td></tr><tr><td>LTI(k)</td><td>1</td><td>3</td><td></td><td> $\begin{array} { r } { \hat { \mathbf { z } } _ { t + 1 } = \sigma \Big ( \sum _ { i = 0 } ^ { k - 1 } \mathbf { A } _ { i } \mathbf { z } _ { t - i } + \mathbf { B } \mathbf { a } _ { t } \Big ) } \end{array}$ </td></tr><tr><td>LTI(1)</td><td>1</td><td>1</td><td></td><td> $\hat { \mathbf { z } } _ { t + 1 } = \sigma ( \mathbf { A } \mathbf { z } _ { t } + \mathbf { B } \mathbf { a } _ { t } )$ </td><td rowspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Proof. See Appendix G.2.

We note that the approximation error of one-step prediction under an exact one-hot sparse encoding decreases as $O ( N ^ { - 1 / d } )$ , and therefore suffers from the curse of dimensionality. Nonetheless, Corollary 1 illustrates a useful trade-off: higher-dimensional sparse representations can simplify the latent transition map, motivating distributed sparse codes as a practical relaxation that may reduce the complexity of the learned dynamics model. In the following experiments, we test whether this simplification enables successful planning with less expressive predictors. In the special case where the learned latent dynamics reduce further to a standard linear controlled system, classical control methods such as LQR become directly applicable under standard assumptions [Kalman et al., 1960].

## 3.2 Experiment Designs

In practice, exact high-dimensional one-hot encodings are infeasible to optimize, and we need exponentially large feature dimensions to linearize the latent dynamics. LpWM instead learns distributed sparse codes whose sparsity is controlled by the target-distribution parameter $\mu .$ Although these codes maybe not linearize the dynamics exactly, we hypothesize that they make the latent dynamics simpler to model than dense representations, so that lower-capacity predictors can suffice.

To test this, we consider a list of predictor options in Table 1, ranging from nonlinear DiT predictors with AdaLN action conditioning [Peebles and Xie, 2023] to linear time-invariant (LTI) state-space models. We train sparse LpWM and dense LeWM with the same encoder architecture described in Section 2.2, but replace the standard DiT predictor with candidates in Table 1 across latent dimensions D ∈ {384, 768, 1536, 2048, 4096} with their corresponding parameter counts in Table 2.

Additionally, we test a variant of LeWM in which SIGReg [Balestriero and LeCun, 2025] is replaced with VICReg [Bardes et al., 2022], yielding another JEPA model with dense representations. For simplicity, we use VICReg to denote this LeWM variant in Figure 3b.

Since our experiment design involves comparing models under width scaling, we use maximum update parameterization (μP) to stabilize training across widths, with details in Appendix F. Experiments are conducted on Wall and PushT environments (Appendix C).

## 3.3 Results

In Figure 3a, we observe that for the Wall environment, even the simplest linear LTI(1) predictor reaches nearly 100% closed-loop CEM success rate for both LeWM and LpWM, leaving little room to distinguish sparse and dense representations.

Table 2: Predictor parameter counts across latent dimensions D. Counts are for the predictor alone for each D used in our experiments. LTI, MLPoLTI, and MLPoLTV are attention-free and far smaller than the AdaLN Transformers.
<table><tr><td>Predictor</td><td>D=384</td><td>D=768</td><td>D=1536</td><td>D=2048</td><td>D=4096</td></tr><tr><td>Deep-AdaLN(k)</td><td>25.8M</td><td>62.2M</td><td>166.9M</td><td>260.2M</td><td>822.4M</td></tr><tr><td>Shallow-AdaLN(k)</td><td>5.6M</td><td>13.0M</td><td>33.1M</td><td>50.4M</td><td>151.1M</td></tr><tr><td>MLPLTV(k)</td><td>0.81M</td><td>3.10M</td><td>12.1M</td><td>21.4M</td><td>84.7M</td></tr><tr><td>MLPoLTI(k)</td><td>0.74M</td><td>2.95M</td><td>11.8M</td><td>21.0M</td><td>83.9M</td></tr><tr><td>LTI(k)</td><td>0.59M</td><td>2.36M</td><td>9.44M</td><td>16.8M</td><td>67.1M</td></tr><tr><td>LTI(1)</td><td>0.30M</td><td>1.18M</td><td>4.72M</td><td>8.39M</td><td>33.6M</td></tr></table>

![](images/532ca76c94fa6ed54d083a7380e7028db99be8aaa366f4b263b8b61e7b2e107c.jpg)  
(a) Wall Environment. LeWM vs. LpWM

![](images/c11724ec73a48035f94c23c630b43a4928c07a68a6aa492c54086f2915767b10.jpg)  
(b) PushT Environment. VICReg vs. LpWM  
Figure 3: Linear predictor is sufficient for the wall environment regardless of dense or sparse embeddings, and LpWM outperforms VICReg-based JEPA across predictor settings in PushT. (a) We plot the planning success rate of the linear LTI(1) predictor against the nonlinear Deep-AdaLN(k), where each dot denotes one setting of method and feature dimension. For simplicity, we use μ = \* to denote LpWM with the target distribution parameter µ. We show the open-loop evaluation on top and closed-loop results at bottom. Both are ran using CEM planner. We observe that the closed-loop success rate for LTI(1) predictor using SIGReg is already approaching 100%, indicating that the wall environment can already be described well by a linear dynamics model in the latent space. (b) We have already shown LeWM vs. LpWM in Figure 1b. Here we compare LpWM with a variant of LeWM where we replaced SIGReg with VICReg in the PushT environment across predictor settings. As oppposed to SIGReg, we observe consistent improvement from LpWM over VICReg even for the Deep-AdaLN(k) predictor, and the gap between sparse LpWM and dense methods (i.e. VICReg here) continues to hold for MLPoLTV(k) and MLPoLTI(k) predictors. See full hyperparameter sweeps in Appendix H.2.

We therefore turn to the harder PushT environment with results shown in Figure 1b. At the lowest predictor capacity, LTI(1) fails for both LeWM and LpWM. Thus PushT differs from Wall in that the underlying dynamics cannot be modelled well by a linear predictor in practice.

At the highest capacity, Deep-AdaLN(k) and Shallow-AdaLN(k) perform similarly for both representations, indicating that the predictor complexity saturates and we don't observe significant advantages of sparse LpWM against dense LeWM.

The difference emerges at intermediate capacity: on PushT, sparse LpWM improves planning success over dense LeWM by 24%–57% with MLPoLTI(k), 36%–45% with MLPoLTV(k), and 11%–23% with LTI(k). These performance gaps are verified across extensive hyperparameter sweeps for both sparse LpWM and dense LeWM in Appendix H.1. We also report the $\ell _ { 0 }$ norms of the LpWM embeddings in Table 3.

![](images/75b569c4b3e4090cb6d2990e8b979cee9a5eefe28526fab7e3c55df25ed95a48.jpg)  
(a) Temporal Jaccard Regularization

![](images/1ef54de90d021ba53c8da08d2b67135fd78d7643c0d36b47058a94caa86b60b2.jpg)  
(b) Example Episode.

![](images/1411d49166d7a9ec23bc43893cc1b311b459278c545c8ea6397590f305897376.jpg)  
(c) Support Raster Plot.  
Figure 4: Temporal-Jaccard (TJ) makes the cube support track contact, not effector motion. (a) As TJ strength increases, support-instability correlation with effector motion drops (0.87→0.40) while cube-motion (0.26→0.80) and gripper-contact (0.05→0.61) rise—at no change in planning success. (b) One example episode: without TJ (top) support instability tracks effector displacement $( r _ { \mathrm { e f f } } { = } 0 . 8 6 )$ ; with TJ (bottom) it tracks cube displacement $( r _ { \mathrm { c u b e } } { = } 0 . 8 0 )$ , spiking inside the shaded contact windows. (c) Support raster: without TJ the support rapidly transitions across frame; with TJ it is smoothly varying in time, reorganizing almost during the shaded contact window.

Additionally, Figure 3b shows that $\mathrm { L p W M }$ outperforms VICReg across Deep-AdaLN(k), MLPoLTV(k), and MLPoLTI(k) predictors. This indicates that the benefit of sparsity extends beyond comparison with the specific dense isotropic Gaussian targets used in LeWM to more general dense representations. Moreover, LpWM outperforms VICReg even with the high-capacity Deep-AdaLN(k) predictor. We hypothesize that this is because second-order moment constraints in VICReg alone can be insufficient as anti-collapse targets because they leave the target distribution underspecified.

Overall, these results show that the benefit of sparsity depends on predictor capacity relative to dynamics complexity. In the simple Wall environment, even linear predictors are sufficient, leaving little room for sparsity to improve performance. In the harder PushT environment, sparsity provides substantial gains at intermediate capacities, but these gains diminish with sufficiently expressive predictors. As the underlying dynamics become more complex, the advantages of sparsity may extend to increasingly expressive predictors, potentially making sparse representations a preferable default over dense representations for modeling latent dynamics. We leave a systematic investigation of this hypothesis to future work.

## 4 Does Sparsity Lead to Interpretable Latent Dynamics?

Section 3 shows that sparse codes can simplify latent prediction. We now ask what structure emerges in these representations. We call a representation mode-factored when its binary support primarily identifies the discrete dynamics regime, while feature magnitudes capture finer within-regime state. We first test whether this structure emerges naturally in Piecewise, where the dynamics mode is known by construction, and then examine whether it extends to OGBench-Cube [Park et al., 2025], where contact provides a natural temporal regime change.

## 4.1 Discovering Piecewise-Affine Dynamics

Piecewise is a 2D navigation environment partitioned into zones with different force fields, yielding piecewise-affine dynamics (Appendix C.2). Empirically, we observe that success rate reaches 84.7% for LpWM versus 65.3% for LeWM with random goals, while performance is saturated for goals sampled from the set of evaluation episodes for both methods (Table 4).

To test whether the sparse support recovers the ground-truth piecewise structure, we discretize the Piecewise environment into a $2 0 \times 2 0$ grid, yielding 400 possible agent locations. For each location, we compute the corresponding embedding and binarize it into a support vector in $\{ 0 , 1 \} ^ { D }$ . For two binary support vectors x, $\mathbf { y } \in \{ 0 , 1 \} ^ { D }$ , we measure their overlap using the Jaccard index

$$
J ( \mathbf { x } , \mathbf { y } ) = \frac { \sum _ { i = 1 } ^ { D } \mathbb { 1 } _ { \mathbf { x } _ { i } = 1 \wedge \mathbf { y } _ { i } = 1 } } { \sum _ { i = 1 } ^ { D } \mathbb { 1 } _ { \mathbf { x } _ { i } = 1 \vee \mathbf { y } _ { i } = 1 } } ,\tag{5}
$$

where $\wedge , \vee$ are logical “and", “or" symbols. We defer further details and its differentiable relaxation given in Appendix E. We then select a reference cell, marked by + in Figure $^ { 2 \mathrm { a , } }$ and compute its Jaccard similarity with every other cell. The resulting similarity is high within the reference zone and drops sharply across zone boundaries (dashed), a pattern that persists even when the zones contain no visual cues (bottom). This indicates that $\mathrm { L p W M }$ recovers the underlying dynamics regime from action-conditioned prediction rather than appearance alone, organizing the mode structure into which features are active.

Linear probes further confirm a factorized code (Figure 2b): the binary support decodes the groundtruth zone at near-perfect accuracy for sparse LpWM, as accurately as the full embedding, while the continuous agent position is decoded better from the magnitudes. Therefore, the support carries the discrete zone mode, and the magnitudes carry the within-zone agent's state.

Finally, this structure also accompanies strong long-horizon planning (Figure 2c): sparse LpWM outperforms dense LeWM at every horizon H, with the gap widening as H increases.

## 4.2 From Navigation to Contact-Rich Dynamics

Does the sparse code also separate distinct phases of the dynamics in contact-rich manipulation? On OGBench-Cube, contact provides a natural regime change between free arm motion and arm-cube interaction. As a temporal analogue of the spatial partition in Piecewise, we measure the support instability $1 - J ( \mathbf { z } _ { t } , \mathbf { z } _ { t + 1 } )$ between consecutive frames and compute its Pearson correlation with effector motion, cube motion, and gripper contact.

Unlike Piecewise, sparsity alone does not determine which temporal factor is encoded by the support. RDMReg constrains only the per-frame marginal, and without an explicit temporal prior the support tends to follow the dominant fast-varying signal. Accordingly, without temporal regularization, support instability behaves largely as a motion detector: it correlates strongly with effector motion, with $r \approx 0 . 8 7$ , but only weakly with contact, with $r \approx 0 . 0 5$ (Figure 4a).

To test whether the support can instead track slower physical regime changes, we additionally consider an optional temporal Jaccard (TJ) loss that encourages support stability across adjacent frames. As the TJ weight increases, support instability becomes less aligned with effector motion and increasingly aligned with cube motion and contact (Figure 4b); for example, the correlation with cube motion increases from $r _ { \mathrm { c u b e } } = 0 . 2 1$ to 0.80. The support raster in Figure 4c shows the same effect visually.

These results highlight a limitation of vanilla LpWM: sparsity alone may not ensure that support transitions correspond to semantically meaningful dynamical events, and without temporal structure the support can instead behave largely as a motion detector. However, they also suggest that, with an appropriate temporal prior, sparse support can become an interpretable tracker of physical regime changes such as object interaction and contact. We leave the design of temporal regularization for more complex dynamics to future work.

## 5 Conclusion

We introduced LpWM, an action-conditioned JEPA regularized with RDMReg to produce nonnegative, exactly sparse latents. Motivated by a one-hot linearization result, we showed empirically that sparsity simplifies the action-conditioned latent dynamics: at a fixed, intermediate predictor capacity, a shallow predictor plans over sparse codes where it fails over dense ones on PushT. We further found the sparse code is mode-factored—its support encodes the discrete dynamics regime (94–99% decodable on Piecewise) and, with a temporal-Jaccard prior, tracks contact on OGBench-Cube. Hence sparse representations can reduce the predictor complexity for control while leading to more interpretable dynamics.

## Acknowledgement

We thank Amir Bar, Raktim Goswami, Ying Wang, Wancong Zhang, Gaoyue Zhou, and Kefei Zhu for helpful discussions. This work was supported in part by AFOSR under grant FA95502310139 and NSF Award 1922658. This work was also supported through AMI Labs and NYU IT High Performance Computing resources, services, and staff expertise.

## References

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture, 2023. URL https://arxiv.org/abs/2301.08243.

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-jepa 2: Self-supervised video models enable understanding, prediction and planning, 2025. URL https://arxiv.org/abs/2506.09985.

Randall Balestriero and Yann LeCun. Lejepa: Provable and scalable self-supervised learning without the heuristics, 2025. URL https://arxiv.org/abs/2511.08544.

Randall Balestriero, Mark Ibrahim, Vlad Sobal, Ari Morcos, Shashank Shekhar, Tom Goldstein, Florian Bordes, Adrien Bardes, Gregoire Mialon, Yuandong Tian, Avi Schwarzschild, Andrew Gordon Wilson, Jonas Geiping, Quentin Garrido, Pierre Fernandez, Amir Bar, Hamed Pirsiavash, Yann LeCun, and Micah Goldblum. A cookbook of self-supervised learning, 2023. URL https://arxiv.org/abs/2304.12210.

Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning, 2022. URL https://arxiv.org/abs/2105.04906.

Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video, 2024. URL https://arxiv.org/abs/2404.08471.

Steven L Brunton, Joshua L Proctor, and J Nathan Kutz. Discovering governing equations from data by sparse identification of nonlinear dynamical systems. Proceedings of the national academy of sciences, 113(15):3932–3937, 2016.

Deep Chakraborty, Yann LeCun, Tim G. J. Rudner, and Erik Learned-Miller. Improving pretrained self-supervised embeddings through effective entropy maximization, 2025. URL https: //arxiv.org/abs/2411.15931.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations, 2020. URL https://arxiv. org/abs/2002. 05709.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale, 2021. URL https://arxiv.org/abs/2010.11929.

Aleksandr Ermolov, Aliaksandr Siarohin, Enver Sangineto, and Nicu Sebe. Whitening for selfsupervised representation learning, 2021. URL https://arxiv. org/abs/2007.06346.

Raktim Gautam Goswami, Prashanth Krishnamurthy, and Farshad Khorrami. Data-driven deep learning based feedback linearization of systems with unknown dynamics. In 2023 American Control Conference (ACC), pages 66–71. IEEE, 2023.

Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, and Michal Valko. Bootstrap your own latent: A new approach to self-supervised learning, 2020. URL https://arxiv.org/abs/2006. 07733.

Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models, 2022. URL https://arxiv.org/abs/2010.02193.

Olivier J Henaff, Robbe LT Goris, and Eero P Simoncelli. Perceptual straightening of natural videos. Nature neuroscience, 22(6):984–991, 2019.

Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus), 2023. URL https: //arxiv. org/abs/1606.08415.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. URL https: //arxiv.org/abs/2106.09685.

Jiaming Hu, Yan Zheng, and Tian Wang. Scale: State-calibrated latent embeddings for jepa planning in the right geometry, 2026. URL https://arxiv.org/abs/2608.16287.

Arnav Kumar Jain, Shivakanth Sujit, Shruti Joshi, Vincent Michalski, Danijar Hafner, and Samira Ebrahimi-Kahou. Learning robust dynamics through variational sparse gating, 2022. URL https://arxiv.org/abs/2210.11698.

Li Jing, Pascal Vincent, Yann LeCun, and Yuandong Tian. Understanding dimensional collapse in contrastive self-supervised learning. arXiv preprint arXiv:2110.09348, 2021.

Rudolf Emil Kalman et al. Contributions to the theory of optimal control. Bol. soc. mat. mexicana, 5 (2):102–119, 1960.

Milan Korda and Igor Mezić. Linear predictors for nonlinear dynamical systems: Koopman operator meets model predictive control. Automatica, 93:149–160, 2018.

Yilun Kuang, Yash Dagade, Deep Chakraborty, Erik Learned-Miller, Randall Balestriero, Tim G. J. Rudner, and Yann LeCun. Radial-vcreg: More informative representation learning through radial gaussianization, 2026a. URL https://arxiv.org/abs/2602.14272.

Yilun Kuang, Yash Dagade, Tim G. J. Rudner, Randall Balestriero, and Yann LeCun. Rectified lpjepa: Joint-embedding predictive architectures with sparse and maximum-entropy representations, 2026b. URL https://arxiv.org/abs/2602.01456.

Yann LeCun et al. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62(1):1–62, 2022.

Timothée Lesort, Natalia Díaz-Rodríguez, Jean-François Goudou, and David Filliat. State representation learning for control: An overview. Neural Networks, 108:379–392, December 2018. ISSN 0893-6080. doi: 10.1016/j.neunet.2018.07.006. URL http://dx.doi.org/10.1016/j. neunet.2018.07.006.

Lucas Maes, Quentin Le Lidec, Luiz Facury, Nassim Massaudi, Ayush Chaurasia, Francesco Capuano, Richard Gao, Taj Gillin, Dan Haramati, Damien Scieur, Yann LeCun, and Randall Balestriero. stable-worldmodel: A platform for reproducible world modeling research and evaluation, 2026a. URL https://arxiv.org/abs/2605.21800.

Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. Leworldmodel: Stable end-to-end joint-embedding predictive architecture from pixels, 2026b. URL https: //arxiv.org/abs/2603.19312.

Seohong Park, Kevin Frans, Benjamin Eysenbach, and Sergey Levine. Ogbench: Benchmarking offline goal-conditioned rl, 2025. URL https://arxiv.org/abs/2410.20092.

William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023. URL https: //arxiv.org/abs/2212.09748.

Giordano Pola, Antoine Girard, and Paulo Tabuada. Approximately bisimilar symbolic models for nonlinear control systems. Automatica, 44(10):2508–2516, 2008.

Joshua L Proctor, Steven L Brunton, and J Nathan Kutz. Dynamic mode decomposition with control. SIAM Journal on Applied Dynamical Systems, 15(1):142–161, 2016.

Reuven Rubinstein. The cross-entropy method for combinatorial and continuous optimization. Methodology and computing in applied probability, 1(2):127–190, 1999.

Mikuláš Ružička. Anwendung mathematisch-statistischer methoden in der geobotanik: Synthetische bearbeitung von aufnahmen. Biológia, Bratislava, 13:647–661, 1958.

Xu Shang, Masih Haseli, Jorge Cortés, and Yang Zheng. On the existence of koopman linear embeddings for controlled nonlinear systems. arXiv preprint arXiv:2602.14537, 2026.

Vlad Sobal, Wancong Zhang, Kyunghyun Cho, Randall Balestriero, Tim G. J. Rudner, and Yann LeCun. Learning from reward-free offline data: A case for planning with latent dynamics models, 2025. URL https://arxiv.org/abs/2502.14819.

Basile Terver, Randall Balestriero, Megi Dervishi, David Fan, Quentin Garrido, Tushar Nagarajan, Koustuv Sinha, Wancong Zhang, Mike Rabbat, Yann LeCun, and Amir Bar. A lightweight library for energy-based joint-embedding predictive architectures, 2026. URL https://arxiv.org/ abs/2602.03604.

Yifei Wang, Qi Zhang, Yaoyu Guo, and Yisen Wang. Non-negative contrastive learning, 2024. URL https://arxiv.org/abs/2403.12459.

Ying Wang, Oumayma Bounou, Yann LeCun, and Mengye Ren. Adajepa: An adaptive latent world model, 2026a. URL https://arxiv.org/abs/2606.32026.

Ying Wang, Oumayma Bounou, Gaoyue Zhou, Randall Balestriero, Tim G. J. Rudner, Yann LeCun, and Mengye Ren. Temporal straightening for latent planning, 2026b. URL https: //arxiv. org/ abs/2603.12231.

Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer, 2022. URL https://arxiv. org/abs/2203.03466.

Greg Yang, James B. Simon, and Jeremy Bernstein. A spectral condition for feature learning, 2024. URL https://arxiv.org/abs/2310.17813.

Thomas Yerxa, Yilun Kuang, Eero Simoncelli, and SueYeon Chung. Learning efficient coding of natural images with maximum manifold capacity representations, 2023. URL https://arxiv. org/abs/2303.03307.

Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stéphane Deny. Barlow twins: Self-supervised learning via redundancy reduction, 2021. URL https://arxiv.org/abs/2103.03230.

Wancong Zhang, Basile Terver, Artem Zholus, Soham Chitnis, Harsh Sutaria, Mido Assran, Randall Balestriero, Amir Bar, Adrien Bardes, Yann LeCun, and Nicolas Ballas. Hierarchical planning with latent world models, 2026. URL https://arxiv.org/abs/2604.03208.

Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. Dino-wm: World models on pre-trained visual features enable zero-shot planning, 2025. URL https://arxiv. org/abs/2411.04983.

## A Related Work

Latent world models for planning. Joint-Embedding Predictive Architectures (JEPA) predict in a learned latent space rather than reconstructing pixels [Assran et al., 2023, Bardes et al., 2024]. Equipped with action-conditioning, JEPA-style world models can be viewed as latent dynamics model of the environment, and can be used for planning with model predictive control [Assran et al., 2025, Zhou et al., 2025, Sobal et al., 2025, Maes et al., 2026b]. Hierarchical JEPA has also been shown to outperform standard flat JEPA models with improved planning performances [Zhang et al., 2026].

Table 3: Sparsity of the LpWM encoder representations across predictor types and latent dimensions. Each entry is the fraction of non-zero coordinates $\mathbb { E } [ \lVert \mathbf { z } \rVert _ { 0 } ] \bar { / } D$ in the sparse code for the target distribution with $\{ \mu = 0 , \sigma = \sqrt { 1 / 2 } , p = 1 \}$ , measured on the validation set at the tuned-best cell for the LpWM run paired with each predictor at latent dimension D. Dense LeWM has active fraction 1.0 by construction. The learned code stays \~ 30–65% active regardless of predictor or dimension, confirming the codes underlying the planning comparison are genuinely sparse.
<table><tr><td>D</td><td>Deep-AdaLN(k)</td><td>Shallow-AdaLN(k)</td><td>MLPoLTV(k)</td><td>MLPoLTI(k)</td><td>LTI(k)</td><td>LTI(1)</td></tr><tr><td>384</td><td>0.34</td><td>0.33</td><td>0.37</td><td>0.28</td><td>0.41</td><td>0.63</td></tr><tr><td>768</td><td>0.44</td><td>0.37</td><td>0.54</td><td>0.53</td><td>0.41</td><td>0.63</td></tr><tr><td>1536</td><td>0.49</td><td>0.39</td><td>0.49</td><td>0.54</td><td>0.41</td><td>0.60</td></tr><tr><td>2048</td><td>0.42</td><td>0.46</td><td>0.51</td><td>0.46</td><td>0.43</td><td>0.61</td></tr><tr><td>4096</td><td>0.48</td><td>0.49</td><td>0.47</td><td>0.43</td><td>0.44</td><td>0.56</td></tr></table>

Feature Regularization for Latent World Models. JEPA-style world models plan in the latent space, and hence proper regularizations of the embedding matters for both anti-collapse and also better planning performances. Preventing collapse without negative examples [Chen et al., 2020] is the central design problem of non-contrastive SSL [Balestriero et al., 2023]. The stop-gradient and EMA techniques [Grill et al., 2020] have been used in DINO-WM [Zhou et al., 2025], V-JEPA 2 AC [Assran et al., 2025] and related work. More recently, information maximization based techniques have also been proposed, which aims at reducing feature redundancies [Zbontar et al., 2021, Ermolov et al., 2021, Yerxa et al., 2023] and maximizing feature entropy through distribution-matching regularizations [Chakraborty et al., 2025, Balestriero and LeCun, 2025, Kuang et al., 2026a,b].

Beyond collapse preventions, other techniques have been proposed to improve the latent space for planning and control. Wang et al. [2026b] show that adding a temporal straightening loss [Henaff et al., 2019] reduces the curvature of the embeddings across time and hence enables better planning. SCALE [Hu et al., 2026] aligns latent distances with task-relevant state differences to make the planner-facing geometry more informative. Wang et al. [2026a] also considers adapting the final layer features online to improve planning performance and achieves better out-of distribution generalization. It's also helpful to use inverse-dynamics modeling [Lesort et al., 2018], temporal smoothness [Sobal et al., 2025], which can lead to better downstream planning performances [Terver et al., 2026].

Sparse and Discrete Representations in World Models. Discrete latent representations have previously been explored in world models. DreamerV2 [Hafner et al., 2022] replaces Gaussian stochastic states with a product of categorical variables, which can equivalently be viewed as a structured sparse binary representation, and finds that categorical latents substantially improve performance over Gaussian ones. The authors hypothesize that sparsity may contribute to this improvement and that categorical states may better capture non-smooth transitions. Relatedly, Variational Sparse Gating [Jain et al., 2022] introduces sparsity into latent dynamics by sparsely updating recurrent-state dimensions through stochastic binary gates. In contrast, we study sparsity directly as a geometry for continuous latent representations: LpWM learns distributed, non-negative sparse codes whose support is not fixed by a categorical grouping. Our focus is also different in that we ask whether sparsity reduces the complexity of the action-conditioned transition model itself, and whether the learned support recovers discrete dynamical regimes while the nonzero magnitudes retain continuous within-regime state.

Linearizing Nonlinear Dynamics. Koopman operator theory studies nonlinear autonomous systems through lifted observables with linear evolution, with extensions to controlled systems enabling linear prediction and MPC [Proctor et al., 2016, Korda and Mezić, 2018]. Recent work further characterizes the conditions for which exact finite-dimensional Koopman embeddings exist for controlled nonlinear systems [Shang et al., 2026]. Related approaches include SINDy, which identifies sparse nonlinear governing equations from candidate function libraries [Brunton et al., 2016]. Datadriven feedback linearization also learns state and input transformations of yielding linear controlled dynamics [Goswami et al., 2023].

Table 4: Success Rate (%) over 50 Test Samples in the Piecewise Environments with Varying Numbers of Zones. Piecewise $( n \times n )$ denotes the environment with varying numbers of zones. Goal images are either sampled from the evaluation dataset or randomly sampled from all possible locations of the dot agent. H denotes the horizon and R represents the receding-horizon. Results are shown as mean±std over 3 seeds. Bold denotes the best performance.
<table><tr><td colspan="2"></td><td>Goals from Evaluation Set</td><td colspan="2">Random Goals</td></tr><tr><td>Env</td><td>Method</td><td> $H = R = 5$ </td><td> $H = 5 , R = 1$ </td><td> $H = R = 5$ </td></tr><tr><td rowspan="3">Piecewise  $( 2 \times 2 )$ </td><td>LpWM (Ours)</td><td>99.33±1.15</td><td>84.67±4.16</td><td>59.33±2.31</td></tr><tr><td>LpWM+TJ (Ours)</td><td>99.33±1.15</td><td>82.67±5.03</td><td>68.67±4.62</td></tr><tr><td>LeWM</td><td>99.33±1.15</td><td>65.33±4.16</td><td>36.00±3.46</td></tr><tr><td rowspan="3">Piecewise  $\left( 3 \times 3 \right)$ </td><td>LpWM (Ours)</td><td>100.00±0.00</td><td>58.67±1.15</td><td>56.00±4.00</td></tr><tr><td>LpWM+TJ (Ours)</td><td>100.00±0.00</td><td>60.00±2.00</td><td>58.00±5.29</td></tr><tr><td>LeWM</td><td>100.00±0.00</td><td>50.00±5.29</td><td>47.33±6.11</td></tr></table>

## B Predictor Designs

## B.1 Overall Designs

Table 1 lists the predictors we compare, in order of decreasing complexity. Deep- and Shallow-AdaLN(k) are 6- and 1-block Transformers with AdaLN-zero action conditioning [Peebles and Xie, 2023]; LTI(k) and LTI(1) are linear (time-invariant) state-space maps over k and 1 history lags; and MLPoLTI(k) places a single nonlinear readout W ReLU(·) on top of the linear core. One caveat is that we use the RepReLU link function in LpWM when using LTI(k) and $\mathbf { L T I } ( 1 )$ predictors, so we don't have exact linear dynamics models.

We have also designed MLPoLTV(k), a data-dependent (linear-time-varying) predictor that moves beyond the fixed weight matrices in the linear-time invariant case.

## B.2 MLPLTV(k)

MLPoLTV(k) retains the MLPoLTI form $\begin{array} { r } { \hat { { \bf z } } _ { t + 1 } = \sigma \big ( { \bf W } \mathrm { R e L U } ( \sum _ { i = 0 } ^ { k - 1 } { \bf A } _ { i } ( { \bf z } _ { t } ) { \bf z } _ { t - i } + { \bf B } ( { \bf z } _ { t } ) { \bf a } _ { t } ) \big ) } \end{array}$ but makes the operator matrices state-dependent. Naively, designing a linear map with a matrix-valued output given a vector-valued input will require a $O ( \bar { D } ^ { 3 } )$ tensor for the feature dimension D.

Instead, we consider a structured matrix parameterization, where the state emits only a small gate that modulates a fixed low-rank correction:

$$
\mathbf { A } _ { i } ( \mathbf { z } _ { t } ) = \mathbf { A } _ { i } + \mathbf { U } _ { i } \mathrm { d i a g } \big ( \mathbf { g } _ { i } ( \mathbf { z } _ { t } ) \big ) { \mathbf { V } } _ { i } ^ { \top } , \qquad \mathbf { g } ( \mathbf { z } _ { t } ) = \mathrm { s i g m o i d } ( \mathbf { G } \mathbf { z } _ { t } ) \in [ 0 , 1 ] ^ { r } ,\tag{6}
$$

where $\mathbf { A } _ { i }$ is a fixed base operator, $\mathbf { V } _ { i } ^ { \top } : \mathbb { R } ^ { D }  \mathbb { R } ^ { r }$ and $\mathbf { U } _ { i } \colon \mathbb { R } ^ { r }  \mathbb { R } ^ { D }$ are learned low-rank factors, and the gate ${ \bf g } ( { \bf z } _ { t } )$ —a single projection $\mathbf { G } \colon \mathbb { R } ^ { D } \to \mathbb { R } ^ { ( k + 1 ) r }$ shared across the k lags and the action term—selects among r fixed low-rank “modes" $( r { = } 1 6$ in our experiments). The action operator $\mathbf { B } ( \mathbf { z } _ { t } )$ is gated identically. This costs $O ( D r )$ with no $D ^ { 3 }$ term. Crucially, only $\mathbf { z } _ { t }$ drives the gate, so the action $\mathbf { a } _ { t }$ still enters affinely: the dynamics core stays control-affine up to the shared nonlinear readout.

We zero-initialize the up-projections $\mathbf { U } _ { i }$ (LoRA-style [Hu et al., 2021]) as an override after $\mu \mathrm { P }$ initialization, so at initialization every correction vanishes and MLPoLTV(k) coincides exactly with MLPoLTI(k). It is therefore a strict superset that recovers the fixed-operator model unless the data reward state-dependence.

## C Environment

## C.1 TwoRoom, PushT, and OGBench Cube

We use the saved dataset in Zhou et al. [2025] for TwoRoom, PushT, and in Maes et al. [2026a] for OGBench Cube. During planning, we fix the goal location to be 25 steps away from the starting location and select episodes from the evaluation set. The action block size is fixed to be 5.

![](images/682eef6fcdb6e944cb4d98fb276e3b79b2c0ca4caa13e5c9079ad90e8f10759c.jpg)  
(a) Wall

![](images/de83636b28354229d8dec4cfd34d56691144257da18fdba9446149f6e05351d7.jpg)  
(b) PushT

![](images/f3b5670bda5ba1ce4a229be082192edb6c9f197df4a0e15e1ffda3440fb44981.jpg)  
(c) Piecewise

![](images/0ab823640fc3443f14372aacce18adbcc19373e8eeee8866e67fc9b006bbf6c4.jpg)  
(d) OGBench-Cube  
Figure 5: Environments. (a) Wall: 2D navigation through a doorway (red agent). (b) PushT: push a T-shaped block to a target pose (green). (c) Piecewise: an open room partitioned into force-field zones (colored) with a known ground-truth segmentation. (d) OGBench-Cube: a robot arm manipulating a cube.

## C.2 Piecewise

Piecewise. To measure what the sparse code encodes about the dynamics, we use a synthetic, analytically-known environment called Piecewise [Maes et al., 2026b]. An open $2 2 4 \times 2 2 4$ room (no walls) is divided into a $g \times g$ grid of zones. Each zone i applies a fixed additive bias to motion, $\mathbf { p } _ { t + 1 } = \mathbf { p } _ { t } + \mathbf { a } _ { t }$ · speed $+  { \mathbf { b } } [ \mathrm { z o n e } (  { \mathbf { p } } _ { t } ) ]$ , with radial biases $\mathbf { b } _ { i } = \mathbf { b } \mathrm { i a s } .$ \_scale · (cos $\theta _ { i } .$ sin $\theta _ { i } )$ $\theta _ { i } = 2 \pi i / g ^ { 2 }$ The dynamics are thus locally linear but globally piecewise-affine, with a known ground-truth zone segmentation and an exact expert policy (it inverts the motion equation). This is a controlled environment where the latent mode is known, so we can test directly whether the sparse support recovers it. We use $g \in \{ 2 , 3 \}$ (i.e. 4 and 9 zones) in our experiments.

## C.3 Visualization of Environments

See Figure 5 for example snapshots of the environments. Note that there is another version of the Piecewise environment without the background color rendering as we have shown in Figure 2a.

## D Experimental Details

Our experiments comprise two complementary studies implemented in separate experimental pipelines. Section 3 systematically varies predictor complexity and latent dimension to test whether sparsity reduces the complexity of the learned dynamics in Wall and PushT environments. The interpretability study in Section 4 instead fixes the model configuration and probes the structure of the learned sparse support in the Piecewise and OGBench-Cube environments. The two pipelines instantiate the same underlying LpWM formulation but differ in architecture and optimization details, summarized in Table 5. All comparisons and ablations are performed within the corresponding experimental pipeline.

In both pipelines, we follow the settings in Section 2.1 and Section 2.2 for the loss, architecture, and link function designs. The model is an action-conditioned JEPA trained end-to-end. The encoder and predictor outputs pass through a link function σ, and per-timestep latent marginals are matched by 2-Wasserstein distance to a target distribution obtained by applying the same link to a Generalized Gaussian with shape parameter p and location or mean-shift parameter $\mu .$ Dense LeWM uses $\sigma =$ identity with an isotropic Gaussian target $( p = 2 )$ , whereas sparse $\mathrm { L p W M }$ uses $\sigma = \mathrm { ( R e p ) R e L U }$ with a Rectified Laplace target $( p = 1 )$ , yielding sparse representations.

Experimental Details for Section 3. We follow the settings in DINO-WM [Zhou et al., 2025] and set the context length (k) for the Wall environment to be $k = 1$ . PushT uses $k = 3$ because velocity is not recoverable from a single observation. For each predictor type and latent feature dimension configuration, we sweep the RDMReg weight $\lambda _ { \mathrm { R D M R e g } }$ and the $\mu \mathrm { P }$ base learning rate $\ell _ { \mathrm { m u P } } ( \mathrm { A p p e n d i x }$ H) and report the best-performing cell in the corresponding grid. We also swept the hyperparameter for LeWM and its VICReg variants in Appendix H to make sure we obtain strong baseline performances.

Table 5: Experimental configurations for Section 3 and Section 4. We list out the experimental pipelines in the following table.
<table><tr><td rowspan="2"></td><td colspan="2">Section 3</td><td colspan="2">Section 4</td></tr><tr><td>Wall</td><td>PushT</td><td>Piecewise</td><td>OGBench-Cube</td></tr><tr><td>Encoder</td><td></td><td>from-scratch ViT, 12 layers, width 384</td><td></td><td>from-scratch ViT-Tiny, width 192</td></tr><tr><td>Predictor</td><td></td><td>See Table 1</td><td></td><td>Deep-AdaLN(k)</td></tr><tr><td>Dimension (D)</td><td></td><td>{384, 768, 1536, 2048, 4096}</td><td></td><td>192</td></tr><tr><td>History k</td><td>1</td><td>3</td><td>3</td><td>3</td></tr><tr><td>Frameskip</td><td>5</td><td>5</td><td>5</td><td>5</td></tr><tr><td>Batch size</td><td>128</td><td>64</td><td>128</td><td>128</td></tr><tr><td>Epochs</td><td>20</td><td>2</td><td>10</td><td>10</td></tr><tr><td>Optimizer</td><td>Adam, µP (with base width 384)</td><td></td><td></td><td>AdamW</td></tr><tr><td>Base LR</td><td></td><td>See Appendix H</td><td></td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Grad. clip</td><td></td><td>0.0</td><td></td><td>1.0</td></tr><tr><td>Reg. weight λ</td><td></td><td>See Appendix H</td><td>25</td><td>10</td></tr><tr><td>Number of projections</td><td></td><td>8192</td><td></td><td>1024</td></tr><tr><td>Parameter μ</td><td></td><td>{0, −1, −2}</td><td>0</td><td>0</td></tr><tr><td>Temporal Jaccard</td><td>off</td><td>off</td><td>{0.0, 0.01}</td><td>{0.0, 0.005, 0.01, 0.05, 0.1}</td></tr></table>

Experimental Details for Section 4. We follow the settings in stable-worldmodel [Maes et al., 2026a]. For both the Piecewise and OGBench-Cube environments, we use a from-scratch ViT-Tiny encoder (width 192, CLS token), latent dimension D = 192, and a depth-6 Deep-AdaLN(k) predictor with history $k = 3$ . Models are trained for 10 epochs using AdamW with learning rate $5 \times 1 0 ^ { - 5 }$ weight decay $1 0 ^ { - 3 }$ , a linear-warmup cosine schedule with warmup over approximately 1% of the optimization steps, batch size 128, bf16 precision, and gradient clipping at 1.0. These experiments do not use $\mu \mathrm { P }$ since the feature dimension is relatively small, i.e. D = 192.

Planning and Evaluation. All experiments use the cross-entropy method (CEM) to optimize action sequences in latent space by minimizing the MSE between predicted latent rollouts and the encoded goal representation. Both experimental pipelines use 300 candidate action sequences per CEM iteration, retain the top 30 best-performing action sequences, and run CEM for 30 iterations with an initial variance scale of 1.0 and planning horizon $\dot { H } = 5$ . With a frameskip of 5, the goal observation is therefore 25 raw environment steps ahead.

In Section 3, we report both open-loop and closed-loop planning success rates. Open-loop evaluation executes a single CEM plan without replanning and therefore provides a more direct measure of accumulated model-prediction quality. Closed-loop evaluation uses receding-horizon MPC, replanning after each block of 5 executed actions, with at most 10 replanning iterations. Wall and PushT are evaluated over 50 trajectories whose goals are drawn from the evaluation dataset.

In Section 4, we additionally evaluate Piecewise using randomly sampled goal images, since performance for both LeWM and LpWM saturates when goals are sampled from the evaluation set. In Table 4, we also consider a receding horizon of 1. For OGBench-Cube, we follow the evaluation protocol of Maes et al. [2026b], where goal observations are sampled only 25 raw environment steps ahead. This short horizon can make the planning task relatively easy, potentially leading to performance saturation and limiting the discriminative power of the benchmark. We leave evaluation under more challenging longer-horizon and larger-frameskip settings to future work.

## E Jaccard Index

## E.1 Hard and Soft Jaccard index

The standard Jaccard index between two sets A and B is Jac $( A , B ) = | A \cap B | / | A \cup B |$ . For two binary vectors $\mathbf { x } , \mathbf { y } \in \{ 0 , 1 \} ^ { D }$ this is $\begin{array} { r } { J ( \mathbf { x } , \mathbf { y } ) = \sum _ { i = 1 } ^ { D } \mathbb { 1 } _ { \mathbf { x } _ { i } = 1 \wedge \mathbf { y } _ { i } = 1 } / \sum _ { i = 1 } ^ { D } \mathbb { 1 } _ { \mathbf { x } _ { i } = 1 \vee \mathbf { y } _ { i } = 1 } } \end{array}$ , where ∧, ∨ are logical “and”, $\ " \mathrm { o r } ^ { \prime \ }$ symbols. Because the encoder's output is non-negative but not binary, directly optimizing this set Jaccard is non-differentiable. We therefore use the soft relaxation [Ružička, 1958],

for $\mathbf { x } , \mathbf { y } \in \mathbb { R } _ { \ge 0 } ^ { D } .$

$$
J _ { S } ( \mathbf { x } , \mathbf { y } ) = \frac { \sum _ { i = 1 } ^ { D } \operatorname* { m i n } ( \mathbf { x } _ { i } , \mathbf { y } _ { i } ) } { \sum _ { i = 1 } ^ { D } \operatorname* { m a x } ( \mathbf { x } _ { i } , \mathbf { y } _ { i } ) + \epsilon } ,\tag{7}
$$

where $\operatorname* { m i n } ( \cdot , \cdot )$ and $\operatorname* { m a x } ( \cdot , \cdot )$ are smooth proxies for $\wedge$ and $\vee ;$ a higher $J _ { S }$ means more similar supports, and $J _ { S }$ recovers the exact set Jaccard when its arguments are binary.

## E.2 Temporal Jaccard Loss

Enforcing the feature distribution to be sparse independently across time steps via RDMReg imposes no constraints on the desired temporal transition. Therefore we also consider the optional temporal Jaccard loss (TJ) as a slowness prior on the binary support vector. The temporal Jaccard loss penalizes rapid support transitions between consecutive frames and hence encourages the support to be smoothly varying across time while not interfering with the global sparsity constraints brought by RDMReg.

Temporal Jaccard loss. Given encoder embeddings $\mathbf { Z } \in \mathbb { R } _ { \geq 0 } ^ { B \times T \times D }$ , where B is batch size, T is the temporal dimension, and D is the feature dimension, the temporal jaccard loss is the mean support instability over all consecutive frame pairs:

$$
\mathcal { L } _ { \mathrm { T J } } = \frac { 1 } { B ( T - 1 ) } \sum _ { b = 1 } ^ { B } \sum _ { t = 1 } ^ { T - 1 } \Big ( 1 - J _ { S } \big ( \mathbf { Z } [ b , t , : ] , \mathbf { Z } [ b , t + 1 , : ] \big ) \Big )\tag{8}
$$

where $\mathbf { Z } [ b , t , : ] \in \mathbb { R } ^ { D }$ denotes the slice of the embedding tensor at batch index b and time index t. Minimizing ${ \mathcal { L } } _ { \mathrm { T J } }$ encourages the support between embeddings of consecutive frames to be smoothly varying, and hence reduces the chance of drastic support changes between temporally adjacent frames.

## F Maximal-Update Parameterization $( \mu \mathbf { P } )$ for JEPA

Our predictor-complexity experiments vary the latent dimensions over more than an order of magnitude, from $D = 3 8 4$ to $D = 4 0 9 6$ . A naive parameterization uses one learning rate for all layers in the neural network as D varies, so performance differences across widths can be confounded by width-specific optimization effects rather than properties of the learned representation. We therefore use maximal-update parameterization $( \mu \mathrm { P } )$ [Yang et al., 2022] to preserve feature-learning dynamics across width and enable learning-rate transfer.

For a dense linear map $\mathbf { h } = \mathbf { W } \mathbf { x }$ with $\mathbf { W } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ $\mu \mathrm { P }$ chooses initialization and learning-rate scalings such that both the hidden activations and their training-induced updates remain of order $\Theta ( \sqrt { d _ { \mathrm { o u t } } } )$ as width changes. Equivalently, the weight and update operators remain at the appropriate spectral scale for stable feature learning [Yang et al., 2024]. For Adam-type optimizers, this gives an effective learning-rate scaling proportional to the inverse fan-in,

$$
{ \eta } _ { \bf { w } } \propto \frac { 1 } { { d } _ { \mathrm { { i n } } } } .\tag{9}
$$

We apply this rule to all width-dependent dense maps in the JEPA encoder and predictor. Let $D _ { 0 }$ denote the reference latent width at which a base learning rate $\eta _ { \mathrm { b a s e } }$ is selected. In our experiment, we fix $D _ { 0 } = 3 8 4$ . For parameter matrices whose fan-in scales linearly with the latent dimension D, we use the following scaling equation:

$$
\eta _ { \bf W } ( D ) = \eta _ { \mathrm { b a s e } } \frac { D _ { 0 } } { D } .\tag{10}
$$

Width-independent parameter groups retain their base learning rate. Initialization follows the corresponding $\mu \mathrm { P }$ fan-in scaling. For all other layers or components like layer norm or bias, we follow the design recipe in Appendix B.1 in Yang et al. [2022].

Given that $\mu \mathrm { P }$ has not been demonstrated in JEPA architectures before for hyperparameter transfer across model width to our best knowledge, we don't intend to use $\mu \mathrm { P }$ to reduce the hyperparameter tuning efforts. Instead, we use $\mu \mathrm { P }$ here simply to ensure stable feature learning, i.e. maintaining a stable spectral norm of the weight matrices to be around $\lVert \mathbf { W } \rVert _ { * } = \Theta ( \sqrt { d _ { \mathrm { o u t } } / d _ { \mathrm { i n } } } )$

In Appendix H, we show the $\mu \mathrm { P }$ parameterization of learning rate can transfer across width most of the time, although there are occasional exceptions. We intend to investigate this more in future work.

## G Proofs of Linearization

## G.1 Proof of Proposition 1

Proof. Since X is a compact subset of $\mathbb { R } ^ { d }$ , the open cover of X

$$
\mathcal { X } \subseteq \bigcup _ { \mathbf { c } \in \mathcal { X } } B _ { \varepsilon } ( \mathbf { c } )\tag{11}
$$

contains a finite subcover. In other words, for every $\varepsilon > 0$ , there exists $C _ { \varepsilon } = \{ \mathbf { c } _ { 1 } , \dots , \mathbf { c } _ { N } \} \subset \mathcal { X }$ such that

$$
\mathcal { X } \subseteq \bigcup _ { \mathbf { c } \in C _ { \varepsilon } } B _ { \varepsilon } ( \mathbf { c } )\tag{12}
$$

where $B _ { \varepsilon } ( \mathbf { c } ) : = \{ \mathbf { x } \in \mathcal { X } : \| \mathbf { x } - \mathbf { c } \| < \varepsilon \}$ . Now consider a deterministic quantization map $Q _ { \varepsilon } : \mathcal { X } $ $\{ 1 , \ldots , N \}$ such that $\left\| \mathbf { x } - \mathbf { c } _ { Q _ { \varepsilon } ( \mathbf { x } ) } \right\| \leq \varepsilon$ for all $\mathbf { x } \in \mathcal { X }$

Let's define the encoder as $E _ { \varepsilon } ( \mathbf { x } ) = \mathbf { e } _ { Q _ { \varepsilon } ( \mathbf { x } ) }$ , where $\mathbf { e } _ { i }$ is the i-th standard basis vector of $\mathbb { R } ^ { N }$ . Hence $\| E _ { \varepsilon } ( \mathbf { x } ) \| _ { 0 } = 1$ for every $\mathbf { x } \in \mathcal { X }$ , i.e. $E _ { \varepsilon } ( \cdot )$ is a one-hot encoder.

Correspondingly, we also define the decoder by $D _ { \varepsilon } ( \mathbf { e } _ { i } ) = \mathbf { c } _ { i }$ . Now fix any action $\mathbf { a } \in { \mathcal { A } } .$ For each representative $\mathbf { c } _ { i } .$ , define $\tau _ { \mathbf { a } } ( i ) : = Q _ { \varepsilon } ( f ( \mathbf { c } _ { i } , \mathbf { a } ) )$ as the transition dynamics map. Then the transition matrix $\mathbf { P } _ { \varepsilon } ( \mathbf { a } )$ can be defined column-wise by $\mathbf { P } _ { \varepsilon } ( \mathbf { a } ) \mathbf { e } _ { i } = \mathbf { e } _ { \tau _ { \mathbf { a } } ( i ) }$

Thus every column of $\mathbf { P } _ { \varepsilon } ( \mathbf { a } )$ contains exactly one nonzero entry, and

$$
\mathbf { z } _ { t + 1 } = \mathbf { P } _ { \varepsilon } ( \mathbf { a } _ { t } ) \mathbf { z } _ { t }\tag{13}
$$

is exactly linear in z. It remains to bound the decoded one-step error. Fix arbitrary $\mathbf { x } _ { t } \in \mathcal { X }$ and ${ \mathbf { a } } _ { t } \in { \mathcal { A } } ,$ and let $i : = Q _ { \varepsilon } ( \mathbf { x } _ { t } )$ . Then by the definition of the encoder we have $E _ { \varepsilon } ( \mathbf { x } _ { t } ) = \mathbf { e } _ { i }$ with $\left\| \mathbf { x } _ { t } - \mathbf { c } _ { i } \right\| \leq \varepsilon$ . We can rollout the latent dynamics to obtain

$$
\mathbf { P } _ { \varepsilon } ( \mathbf { a } _ { t } ) E _ { \varepsilon } ( \mathbf { x } _ { t } ) = \mathbf { P } _ { \varepsilon } ( \mathbf { a } _ { t } ) \mathbf { e } _ { i } = \mathbf { e } _ { \tau _ { \mathbf { a } _ { t } } ( i ) }\tag{14}
$$

The decoded state is thus

$$
\hat { \mathbf { x } } _ { t + 1 } : = D _ { \varepsilon } ( \mathbf { P } _ { \varepsilon } ( \mathbf { a } _ { t } ) E _ { \varepsilon } ( \mathbf { x } _ { t } ) ) = \mathbf { c } _ { Q _ { \varepsilon } ( f ( \mathbf { c } _ { i } , \mathbf { a } _ { t } ) ) } .\tag{15}
$$

Let the ground-truth next state be

$$
\mathbf { x } _ { t + 1 } = f ( \mathbf { x } _ { t } , \mathbf { a } _ { t } ) .\tag{16}
$$

By the triangle inequality, we have

$$
\begin{array} { r l } & { \| \mathbf { x } _ { t + 1 } - \hat { \mathbf { x } } _ { t + 1 } \| = \Big \| f ( \mathbf { x } _ { t } , \mathbf { a } _ { t } ) - \mathbf { c } _ { Q _ { \varepsilon } ( f ( \mathbf { c } _ { i } , \mathbf { a } _ { t } ) ) } \Big \| } \\ & { \qquad \leq \| f ( \mathbf { x } _ { t } , \mathbf { a } _ { t } ) - f ( \mathbf { c } _ { i } , \mathbf { a } _ { t } ) \| + \Big \| f ( \mathbf { c } _ { i } , \mathbf { a } _ { t } ) - \mathbf { c } _ { Q _ { \varepsilon } ( f ( \mathbf { c } _ { i } , \mathbf { a } _ { t } ) ) } \Big \| . } \end{array}\tag{17}
$$

(18)

Since $f$ is Lipschitz continuous in state, we know

$$
\| f ( \mathbf { x } _ { t } , \mathbf { a } _ { t } ) - f ( \mathbf { c } _ { i } , \mathbf { a } _ { t } ) \| \leq L \| \mathbf { x } _ { t } - \mathbf { c } _ { i } \| \leq L \varepsilon .\tag{19}
$$

Now because $f ( \mathbf { c } _ { i } , \mathbf { a } _ { t } ) \in \mathcal { X }$ , the compactness property implies that

$$
\left\| f ( \mathbf { c } _ { i } , \mathbf { a } _ { t } ) - \mathbf { c } _ { Q _ { \varepsilon } ( f ( \mathbf { c } _ { i } , \mathbf { a } _ { t } ) ) } \right\| \leq \varepsilon .\tag{20}
$$

Therefore the approximation error is upper bounded by

$$
\begin{array} { r } { \| \mathbf { x } _ { t + 1 } - \hat { \mathbf { x } } _ { t + 1 } \| \leq L \varepsilon + \varepsilon = ( L + 1 ) \varepsilon . } \end{array}\tag{21}
$$

Since the bound is uniform over $\mathbf { x } _ { t }$ and $\mathbf { a } _ { t }$

$$
\operatorname* { s u p } _ { \mathbf { x } _ { t } \in \mathcal { X } , \mathbf { a } _ { t } \in \mathcal { A } } \| f ( \mathbf { x } _ { t } , \mathbf { a } _ { t } ) - D _ { \varepsilon } ( \mathbf { P } _ { \varepsilon } ( \mathbf { a } _ { t } ) E _ { \varepsilon } ( \mathbf { x } _ { t } ) ) \| \leq ( L + 1 ) \varepsilon .\tag{22}
$$

Now since the Lipschitz constant $L$ is fixed, we know that $( L + 1 ) \varepsilon \to 0$ as $\varepsilon \to 0$ . Therefore, we arrive at

$$
\operatorname* { l i m } _ { \varepsilon \to 0 } \operatorname* { s u p } _ { \mathbf x _ { t } \in \mathcal X , \mathbf a _ { t } \in \mathcal A } \left\| f ( \mathbf x _ { t } , \mathbf a _ { t } ) - D _ { \varepsilon } ( \mathbf P _ { \varepsilon } ( \mathbf a _ { t } ) E _ { \varepsilon } ( \mathbf x _ { t } ) ) \right\| = 0 .\tag{23}
$$

Lemma 1 (Finite-horizon rollout bound). Under the assumptions of Proposition 1, let $\mathbf { x } _ { t + 1 } ~ =$ $f ( \mathbf { x } _ { t } , \mathbf { a } _ { t } ) f o r t = 0 , \ldots , H - 1$ be the ground-truth trajectory under an arbitrary action sequence $\mathbf { a } _ { 0 } , \dots , \mathbf { a } _ { H - 1 } \in \mathcal { A }$ Initialize ${ \bf z } _ { 0 } = E _ { \varepsilon } ( { \bf x } _ { 0 } )$ and recursively deine the latent transition as $\mathbf { z } _ { t + 1 } =$ ${ \bf P } _ { \varepsilon } ( { \bf a } _ { t } ) { \bf z } _ { t } . L e t \hat { \bf x } _ { t } = D _ { \varepsilon } ( { \bf z } _ { t } )$ be the decoded state. Then

$$
\| { \mathbf x } _ { H } - \hat { { \mathbf x } } _ { H } \| \le \varepsilon \sum _ { k = 0 } ^ { H } L ^ { k } .\tag{24}
$$

Proof. Define the decoded quantization map as $q _ { \varepsilon } ( \mathbf { x } ) : = D _ { \varepsilon } ( E _ { \varepsilon } ( \mathbf { x } ) ) = \mathbf { c } _ { Q _ { \varepsilon } ( \mathbf { x } ) }$ . By construction, $\| \mathbf { x } - q _ { \varepsilon } ( \mathbf { x } ) \| \leq \varepsilon$ for all $\mathbf { x } \in \mathcal { X }$ . Since every latent state $\mathbf { z } _ { t }$ is one-hot, we know that if $\mathbf { z } _ { t } = \mathbf { e } _ { i }$ , then $\hat { \mathbf { x } } _ { t } = D _ { \varepsilon } ( \mathbf { z } _ { t } ) = \mathbf { c } _ { i }$ . Since the latent dynamics give

$$
\mathbf { z } _ { t + 1 } = \mathbf { P } _ { \varepsilon } ( \mathbf { a } _ { t } ) \mathbf { e } _ { i } = \mathbf { e } _ { Q _ { \varepsilon } ( f ( \mathbf { c } _ { i } , \mathbf { a } _ { t } ) ) } .\tag{25}
$$

We know that

$$
\hat { \mathbf { x } } _ { t + 1 } = D _ { \varepsilon } ( \mathbf { z } _ { t + 1 } ) = q _ { \varepsilon } \big ( f \big ( \hat { \mathbf { x } } _ { t } , \mathbf { a } _ { t } \big ) \big ) .\tag{26}
$$

Define the scalar rollout error as $e _ { t } : = \| \mathbf { x } _ { t } - \hat { \mathbf { x } } _ { t } \|$ . Then

$$
\begin{array} { r l } & { e _ { t + 1 } = \| f ( \mathbf { x } _ { t } , \mathbf { a } _ { t } ) - q _ { \varepsilon } ( f ( \hat { \mathbf { x } } _ { t } , \mathbf { a } _ { t } ) ) \| } \\ & { \qquad \leq \| f ( \mathbf { x } _ { t } , \mathbf { a } _ { t } ) - f ( \hat { \mathbf { x } } _ { t } , \mathbf { a } _ { t } ) \| + \| f ( \hat { \mathbf { x } } _ { t } , \mathbf { a } _ { t } ) - q _ { \varepsilon } ( f ( \hat { \mathbf { x } } _ { t } , \mathbf { a } _ { t } ) ) \| } \\ & { \qquad \leq L e _ { t } + \varepsilon . } \end{array}\tag{27}
$$

(28)

$\mathrm { A t } \ t = 0 , e _ { 0 } = \| \mathbf { x } _ { 0 } - \hat { \mathbf { x } } _ { 0 } \| \leq \varepsilon$ . Rolling out H steps give us

$$
e _ { H } \leq L ^ { H } e _ { 0 } + \varepsilon \sum _ { k = 0 } ^ { H - 1 } L ^ { k } \leq L ^ { H } \varepsilon + \varepsilon \sum _ { k = 0 } ^ { H - 1 } L ^ { k } = \varepsilon \sum _ { k = 0 } ^ { H } L ^ { k } .\tag{29}
$$

Hence we arrive at our conclusion

$$
\| { \mathbf x } _ { H } - \hat { { \mathbf x } } _ { H } \| \le \varepsilon \sum _ { k = 0 } ^ { H } L ^ { k } .\tag{30}
$$

## G.2 Proof of Corollary 1

Proof. Partition each coordinate interval [0, 1] into n subintervals of equal length $\begin{array} { r } { h \ = \ \frac { 1 } { n } } \end{array}$ . The resulting partition of $[ 0 , 1 ] ^ { d }$ consists of $n ^ { d } = N$ axis-aligned hypercubes, each with side length h. Let $\mathbf { c } _ { i }$ denote the center of one such hypercube. For any point x in that cell, each coordinate differs from the corresponding coordinate of $\mathbf { c } _ { i }$ by at most $h / \dot { 2 }$ . Hence

$$
| \mathbf { x } _ { j } - \mathbf { c } _ { i , j } | \leq { \frac { h } { 2 } } , \qquad j = 1 , \ldots , d .\tag{31}
$$

Therefore, using the Euclidean norm,

$$
\| \mathbf { x } - \mathbf { c } _ { i } \| ^ { 2 } = \sum _ { j = 1 } ^ { d } | \mathbf { x } _ { j } - \mathbf { c } _ { i , j } | ^ { 2 } \leq \sum _ { j = 1 } ^ { d } \left( \frac { h } { 2 } \right) ^ { 2 } = d \frac { h ^ { 2 } } { 4 } .\tag{32}
$$

Taking square roots gives $\begin{array} { r } { \| \mathbf { x } - \mathbf { c } _ { i } \| \leq \frac { \sqrt { d } } { 2 } h } \end{array}$ . Since $h = 1 / n$ , we arrive at

$$
\varepsilon _ { N } = \frac { \sqrt { d } } { 2 n } = \frac { \sqrt { d } } { 2 } N ^ { - 1 / d } .\tag{33}
$$

Based on Proposition 1, the one-step decoded prediction error is bounded by

$$
\operatorname* { s u p } _ { \mathbf { x } , \mathbf { a } } \| f ( \mathbf { x } , \mathbf { a } ) - D _ { \varepsilon _ { N } } ( \mathbf { P } _ { \varepsilon _ { N } } ( \mathbf { a } ) E _ { \varepsilon _ { N } } ( \mathbf { x } ) ) \| \leq ( L + 1 ) \varepsilon _ { N } .\tag{34}
$$

Substituting the expression for $\varepsilon _ { N }$ gives

$$
\operatorname* { s u p } _ { \mathbf { x } , \mathbf { a } } \| f ( \mathbf { x } , \mathbf { a } ) - D _ { \varepsilon _ { N } } ( \mathbf { P } _ { \varepsilon _ { N } } ( \mathbf { a } ) E _ { \varepsilon _ { N } } ( \mathbf { x } ) ) \| \leq ( L + 1 ) \frac { \sqrt { d } } { 2 } N ^ { - 1 / d } = \frac { \sqrt { d } } { 2 } ( L + 1 ) N ^ { - 1 / d } .\tag{35}
$$

Since $d$ and L are fixed constants,

$$
\operatorname* { s u p } _ { \mathbf { x } , \mathbf { a } } \| f ( \mathbf { x } , \mathbf { a } ) - D _ { \varepsilon _ { N } } ( \mathbf { P } _ { \varepsilon _ { N } } ( \mathbf { a } ) E _ { \varepsilon _ { N } } ( \mathbf { x } ) ) \| = O ( N ^ { - 1 / d } ) .\tag{36}
$$

Similarly, the finite-horizon rollout bound from Lemma 1 gives

$$
\left\| { \mathbf { x } } _ { H } - \hat { \mathbf { x } } _ { H } \right\| \leq \varepsilon _ { N } \sum _ { k = 0 } ^ { H } L ^ { k } .\tag{37}
$$

Substituting the covering radius, we have

$$
\| { \bf x } _ { H } - \hat { \bf x } _ { H } \| \leq \frac { \sqrt { d } } { 2 } N ^ { - 1 / d } \sum _ { k = 0 } ^ { H } L ^ { k } .\tag{38}
$$

Since $H , d ,$ and $L$ are fixed, it follows that

$$
\operatorname* { l i m } _ { N  \infty } \| { \bf x } _ { H } - \hat { \bf x } _ { H } \| = 0 .\tag{39}
$$

## H Hyperparameter Sweeps

## H.1 LpWM and LeWM Sweeps in the PushT Environment

To ensure full reproducibility and transparency, we perform and report hyperparameter sweeps underlying Figure 1b.

We run a grid search over the RDMReg regularization weight $\lambda _ { \mathrm { R D M R e g } } \in \{ 0 . 0 1 , 0 . 1 , 1 . 0 , 1 0 . 0 \}$ and the muP (or µP) learning rate $\ell _ { \mathrm { m u P } } ~ \in ~ \{ 0 . 0 5 , 0 . 0 0 5 , 0 . \bar { 0 } 0 0 5 , 0 . 0 0 0 0 5 \}$ for each fixed feature dimension $d \in \{ \dot { 3 } 8 4 , 7 6 8 , 1 5 \bar { 3 } 6 , 2 0 4 8$ , 4096}, method (LeWM and LpWM with target distribution $\begin{array} { r } { \prod _ { i = 1 } ^ { d } \operatorname { R e L U } ( \operatorname { L a p l a c e } ( \mu = 0 , \sigma = \sqrt { 1 / 2 } ) ) } \end{array}$ , and predictor setting in Table 1. To ensure fair comparison and standardized hyperparameter sweeps, we use our sliced Wasserstein distance formulation as the distribution-matching loss for isotropic Gaussian targets instead of the Epps-Pulley loss in SIGReg [Balestriero and LeCun, 2025]. They are both in fact instantiations of Cramer-Wold theorem, and only differ in one-sample vs. two-sample style distribution matching [Kuang et al., 2026b].

Due to compute constraints, we fix one seed for training but average over 3 seeds during planning evaluations with Cross-Entropy Methods (CEM) [Rubinstein, 1999]. In practice, we execute our sweep sequentially. So there are some predictor types where we have done a very thorough grid search. After we have determined that some hyperparameters are always leading to bad performances (e.g. usually learning rate like 0.05 is too large), we drop them in our subsequent sweeps so the resulting figures for different predictor types usually cover different subset of the full grid.

In Figure 9, we observe that $\ell _ { \mathrm { m u P } } \in \{ 0 . 0 5 , 0 . 0 0 5 \}$ almost always leads to 0% success rate. Subsequent hyperparameter sweeps shown in Figure 6, Figure 7, Figure 8, Figure 11, and Figure 10 selectively remove portions of the grid to save compute.

## H.2 VICReg Sweep in the PushT Environment

We also perform extensive hyperparameter sweeps for JEPA regularized with VICReg. Basically, we take the LeWM setting and replaced the SIGReg regularizer with VICReg. The resulting model doesn't have a name yet, and we abuse the notation and simply call it VICReg here.

In Figure 12, Figure 13, and Figure 14, we show the sweep results for the predictors Deep-AdaLN(k) MLPoLTV(k), and MLPoLTI(k) respectively.

Deep-AdaLN(k) — Open-Loop plan success over λRDmReg × Learning Rate θ (mean ± std, 3 seeds)  
![](images/9dc816556699db679904f12a8906bb75c0207c634dc503f2cd111d9441a2600f.jpg)  
(a) Open-Loop.

Deep-AdaLN(k) — Closed-Loop plan success over λRDmReg × Learning Rate θ (mean ± std, 3 seeds)  
![](images/2b53b936bfa4e8303a92a9097b95b7e57fb4faf1c8402a3e646ac050fd820fe3.jpg)  
(b) Closed-Loop.  
Figure 6: Deep-AdaLN(k) Predictor. Hyperparameter Sweeps for the PushT Environment.

Shallow-AdaLN(k) — Open-Loop plan success over λRDmReg × Learning Rate θ (mean ± std, 3 seeds)

![](images/d238d5f7d89b152d6872a163c2fc85f219818a84b1362b05ae27ef4a1d1a5b68.jpg)  
(a) Open-Loop.

Shallow-AdaLN(k) — Closed-Loop plan success over λRdmReg × Learning Rate θ (mean ± std, 3 seeds)  
![](images/5f4511dbbb5b3c29f5b1dada6151934a51d928cbc22b82d0eb11be6fc0e20935.jpg)  
(b) Closed-Loop.  
Figure 7: Shallow-AdaLN(k) Predictor. Hyperparameter Sweeps for the PushT Environment.

![](images/853189123f4ae7df4e6be905a87d2bacfbcc06bed00427e4603024391da8995b.jpg)  
(a) Open-Loop.

MLP-LTV(k) — Closed-Loop plan success over λRDmReg × Learning Rate θ (mean ± std, 3 seeds)  
![](images/756f5a227783822aba44066576f5583dc9e12def12326d24becfc7634eabdee0.jpg)  
(b) Closed-Loop.  
Figure 8: MLPoLTV(k) Predictor. Hyperparameter Sweeps for the PushT Environment.

![](images/f8129c256291e5d7f830debdd76b02b1dc712415a2f6b5e5934df1b7e7ba3ec9.jpg)  
(a) Open-Loop.

MLP-LTI(k) — Closed-Loop plan success over λRDmReg × Learning Rate θ (mean ± std, 3 seeds)  
![](images/4893acde043a39ed6cae0e58758a071ce5543d65fbe6f5d1af18154ba2c6b1f6.jpg)  
(b) Closed-Loop.  
Figure 9: MLPoLTI(k) Predictor. Hyperparameter Sweeps for the PushT Environment.

![](images/7aa75a271cc1955a2120969adc04cee7fba5058c4ffd6afbe39fac4d25ad349b.jpg)  
(a) Open-Loop.

LTI(1) — Closed-Loop plan success over λRdmReg × Learning Rate θ (mean ± std, 3 seeds)  
![](images/4642965edb58cca57c0710eed40f8b930b3183669a0312a0b1da52f9bd7fccde.jpg)  
Figure 10: LTI(1) Predictor. Hyperparameter Sweeps for the PushT Environment.  
(b) Closed-Loop.

![](images/4821eca4082d8b6a2ff7ea7e817c634db2698aee92778dadf61dce1e2374a9ad.jpg)  
(a) Open-Loop.

LTI(k) — Closed-Loop plan success over λRdmReg × Learning Rate θ (mean ± std, 3 seeds)  
![](images/585bc4dcc9f54f8c72abea97703db1c0a88189d81ba654bdd67c2c6fdc961d7e.jpg)  
(b) Closed-Loop.  
Figure 11: LTI(k) Predictor. Hyperparameter Sweeps for the PushT Environment.

![](images/0d8e35f7f1ca275436e487281aab43c46fbf95e152642b38379839d8a9a04713.jpg)  
(a) Open-Loop.

Dense-VICReg — Deep-AdaLN(k) — Closed-Loop plan success (mean ± std, 3 seeds)  
![](images/c85c19c8439a11dcb48450ffa0f27a75a4e90a6aec12a8dcebc13d25ebdc8d7d.jpg)  
(b) Closed-Loop.  
Figure 12: Deep-AdaLN(k) Predictor. Hyperparameter Sweeps for VICReg-based JEPA models in the PushT Environment.

![](images/fc2e15d113e3f1defe35c1de8be95efe29434f1d1c098a19ffbc59ffcb45e818.jpg)  
(a) Open-Loop.

Dense-VICReg — MLP-LTV(k) — Closed-Loop plan success (mean ± std, 3 seeds)  
![](images/43fe92531e4820e338cef34e5afc2488e531b833a6941a022be50ed17a555343.jpg)  
(b) Closed-Loop.  
Figure 13: MLPoLTV(k) Predictor. Hyperparameter Sweeps for VICReg-based JEPA models in the PushT Environment.

![](images/4a60d9583a6b8bc65ad1b1f885fe7f0903c87af43c2823b7ede7db3fa180cde6.jpg)  
(a) Open-Loop.

Dense-VICReg — MLP-LTI(k) — Closed-Loop plan success (mean ± std, 3 seeds)  
![](images/1352895ac572dcd7cb1fc6a3ecff80a9b9cb086d08eb7350e2daaf7868be240c.jpg)  
(b) Closed-Loop.  
Figure 14: MLPoLTI(k) Predictor. Hyperparameter Sweeps for VICReg-based JEPA models in the PushT Environment.