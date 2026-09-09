# Geometry-Aware Bayesian Parameter-Efficient Fine-Tuning on the Stiefel Manifold via Stein Variational Gradient Descent

Quang-Duy Tran<sup>∗†§</sup>, Trung Le<sup>‡</sup>, Bao Duong<sup>∗</sup>, Phuoc Nguyen<sup>∗</sup>, Thin Nguyen<sup>∗</sup>

<sup>∗</sup>Deakin Applied Artificial Intelligence Initiative, Deakin University, Geelong, Australia

<sup>†</sup>Department of Computer Science, Aalto University, Espoo, Finland

<sup>‡</sup>Department of Data Science & AI, Monash University, Melbourne, Australia

quang-duy.tran@aalto.fi, trunglm@monash.edu, {b.duong, phuoc.nguyen, thin.nguyen}@deakin.edu.au

<sup>§</sup>This work was completed at the Deakin Applied Artificial Intelligence Initiative, Deakin University.

Abstract—Several geometry-aware approaches to low-rank adaptation have emerged for parameter-efficient fine-tuning of large pre-trained models. These methods aim to take full advantage of the geometric structure of low-rank manifolds for improving the efficiency in subspace utilization and reducing redundancy by enforcing orthogonality constraints during optimization. The strong empirical results of these techniques have motivated further study into whether predictions from such geometry-based adaptation methods could be overconfident. In this paper, we build on the singular value decomposition factorization of adapters to develop a framework based on Stein variational gradient descent (SVGD). In this formulation, the lowrank matrices are transported along the Stiefel manifold to match the targeted distributions while retaining their crucial geometric structure. Since this geometry-aware SVGD approach provides multiple solutions during inference, it supports uncertainty quantification and produces better-calibrated adapters on the Stiefel manifold. Extensive experiments show that our method delivers strong model calibration and attains higher prediction accuracy than SVGD and related uncertainty estimation methods that are formulated in Euclidean space.

Index Terms—PEFT, Bayesian Inference, Stiefel Manifold, Approximate Inference

## I. INTRODUCTION

Various large pre-trained models, such as large language models (LLMs) [1]–[3], are emerging as valuable tools for safety-critical domains [4], [5]. To efficiently adapt these pre-trained models to a specific domain, parameter-efficient fine-tuning (PEFT) formulations are often utilized to create general instruction-following models [6]–[9]. Low-rank adaptation (LoRA) [6] treats the fine-tuned weights ∆W as an additive adapter to the pre-trained weights $\mathbf { W } _ { 0 } \in \mathbb { R } ^ { m \times n }$ as $\mathbf { W } : = \mathbf { W } _ { 0 } + \Delta \mathbf { W }$ and reduces the computational cost with a low-rank decomposition $\begin{array} { r } { \Delta \mathbf { W } : = \frac { \alpha } { r } \mathbf { B } \mathbf { A } } \end{array}$ , where $\mathbf { B } \in \mathbb { R } ^ { m \times r }$ $\mathbf { A } \in \mathbb { R } ^ { r \times n }$ , r is the rank such that r ≪ min (m, n), and α is a scale factor.

However, despite showing improved accuracy, it has been shown that fine-tuning can break the calibration of the models and cause them to produce overconfident predictions [1], [10], [11]. Bayesian deep learning [12]–[14] has been a principled choice for estimating the epistemic uncertainty as well as allowing the production of better calibrated predictions through Bayesian posterior predictive distributions of these models. Although these methods are often challenging in large models due to the high dimensionality of the parameters, applying them to a significantly reduced subset of parameters during PEFT is more feasible. In addition, due to a lower amount of training data during, fine-tuning is more prone to epistemic uncertainty compared to pre-training [15]. Hence, various Bayesian inference studies have been presented [15]– [17] to address the uncertainty and calibration during LoRA fine-tuning.

On the other hand, to further enhance the rank efficiency of LoRA, approaches [8], [9], [18] have been introduced to enforce orthogonality between the r basis vectors of the upprojection matrix, as in Stiefel-LoRA [9], or both the basis vectors of up-projection and down-projection matrices, as in StelLA [8]. These constraints are based on the assumption that the weights belong to the Stiefel manifolds and geometryaware optimization on Riemannian manifold [19] can straightforwardly be applied to enforce the orthogonality. With this constraint, these methods can fully exploit all subspace dimensions of the rank, leading to promising performance enhancements over diverse fine-tuning tasks [8], [9].

Despite promising predictive results, these methods lack evaluations on calibration and epistemic uncertainty. As a result, we present this study to examine the calibration errors of geometry-aware low-rank fine-tuning adaptations (specifically, StelLA [8]) and proposed a Bayesian inference framework on the Stiefel manifold for evaluating their uncertainty and improving the calibration. Due to the complexity of distributional definitions for Bayesian inference on the manifold, particle-based variational inference approaches, notably Stein variational gradient descent (SVGD) [20], [21], are ideal for this setting due to their approximation flexibility compared to model-based variational inference and better iterationeffectiveness compared to Monte Carlo methods [21]. As a result, we choose SVGD [20], [21] as our inference engine and solve to find the optimal update function to iteratively transport the particles on the Riemannian/Stiefel manifold to the target distributions. Through evaluations on diverse experimental settings, our proposed approach—Riemannian Stein variational gradient descent for PEFT on the Stiefel manifold (StePS)— has demonstrated favorable accuracy and expected calibration errors compared to related methods on both in-distribution and out-of-distribution settings of the commonsense reasoning fine-tuning benchmarks.

Contributions: The key contributions of this work can be highlighted as follows:

• We address the uncertainty estimation and calibration problems in geometry-aware PEFT by performing the Bayesian inference on the Stiefel manifold, allowing explicit and principled estimation of the epistemic uncertainty and enhancing the calibration of the predictions via the Bayesian posterior predictive distribution.

• We find the optimal transport function with the gradient flow formulation on the Stiefel manifold, resulting in the closed-formed solution for steepest descent to iteratively transport the particles to match the target distributions on the manifold.

• The effectiveness of our approach is evaluated through extensive experiments on fine-tuning and inference on both in-distribution and out-of-distribution data, demonstrating beneficial improvements on predictive accuracy and model calibration results compared to related Bayesian PEFT frameworks.

## II. RELATED WORK

## A. Geometry-Aware Parameter-Efficient Fine-Tuning

Low-rank adaptation (LoRA) [6] is a foundational parameter-efficient fine-tuning method that freezes the pretrained backbone weights and injects trainable low-rank updates $( \Delta \mathbf { W } : = \mathbf { B } \mathbf { A } )$ into selected weight matrices. While LoRA is simple and effective, it treats the factors as unconstrained Euclidean parameters, which can ignore the vital geometric structure of the low-rank matrix spaces. Recent work has begun to utilize geometric structure into the lowrank adaptation. Stiefel-LoRA [9] directly constrains the B to lie on the Stiefel manifold to enforce the orthonormal columns that span across the adapted subspace. The A factor remains unconstrained and is optimized conventionally.

SVD-like tri-factor low-rank formulations of the adapter $( \Delta \mathbf { W } : = \mathbf { U S V } ^ { \top } )$ , as adopted in, for example, [8], [18], [22], [23], provide a more meaningful geometric structure, where both U and V ideally form orthonormalized bases with respect to the subspace dimensions. Different strategies have been used to enforce or preserve the orthogonality. AdaLoRA [22] introduces additional regularization terms that “softly” encourages orthonormality of the factors. GeoLoRA [23] implicitly preserves low-rank orthogonality by updating the adapter parameters along a dynamical low-rank gradient flow, where orthonormality is maintained as an invariant of the integration scheme rather than enforced explicitly. Both PoLAR [18] and StelLA [8] impose explicit manifold constraints on the factors. Specifically, PoLAR employs a landing algorithm with infeasibility penalty to guide the factors toward the Stiefel manifold, whereas StelLA performs the optimization directly on the Stiefel manifold using tangent-space projection and retraction operations. Consequently, due to its principled geometry constraints, StelLA is particularly well-suited for geometry-aware Bayesian inference.

## B. Uncertainty Quantification of Large Models

Large pre-trained models tend to exhibit good calibration during pre-training [1], [10], [11]. However, they are often failed to express reliable predictive uncertainty after finetuning [15], [16], [24]. To address this issue, model-based uncertainty quantification methods [13]–[17], [24] can be integrated to the fine-tuning adapters to allow the models to efficiently provide epistemic uncertainty estimation and improved calibration.

While straightforward methods, such as Monte Carlo Dropout [13] and Deep Ensemble [14], [24], can be directly applied, they generally yield limited enhancements in model calibration. Laplace-LoRA [15] employs Laplace approximation to LoRA parameters to evaluate the uncertainty around the Maximum A Posteriori (MAP) solution. However, its posthoc nature can lead to suboptimal uncertainty estimation. Model-based variational inference methods, BLoB [16] and ScalaBL [17], allow the joint optimization of the approximated variational posterior, which can enhance the efficiency and calibration. Nevertheless, their stochastic optimization procedures can be unstable and often requires additional techniques for enhancing the training stability. Despite the advantages, existing Bayesian PEFT methods are predominantly designed for Euclidean parameters, lacking the ability to capture inherent geometric structure of the low-rank parameters. This motivates the development of geometry-aware Bayesian approaches.

Beyond model-based variational inference, particle-based variational inference methods provide an alternative approach to uncertainty quantification. Stein variational gradient descent (SVGD) [20] approximates the posterior distribution with a pre-defined set of samples, called “particles”, which are deterministically transported toward the target distribution via functional gradient flows. Compared to the modelbased approaches, SVGD avoids the restrictive distributional assumptions and allows more expressive approximations of the posterior distribution with a trade-off in computational cost. To account for non-Euclidean parameters, Riemannian SVGD [21] extends SVGD to parameters on Riemannian manifolds, enabling the inference to be performed directly on constrained parameter spaces. However, its application to large-scale deep learning models and PEFT remains unexplored.

## III. PRELIMINARIES

A. Tri-Factor Factorization of Adapters and Optimization on the Stiefel Manifold

a) SVD-Based Adaptation: We follow the problem setting of StelLA [8] by considering a low-rank factorization based on truncated singular value decomposition (SVD) of

the fine-tuning adaptation ∆W for a pre-trained weight matrix $\mathbf { W } _ { 0 } \in \mathbb { R } ^ { m \times n }$ as

$$
\Delta \mathbf { W } : = \frac { \alpha } { r } \mathbf { U S V } ^ { \top } ,\tag{1}
$$

where $\mathbf { U } \ \in \ \mathbb { R } ^ { m \times r } , \ \mathbf { V } \ \in \ \mathbb { R } ^ { n \times r } , \ \mathbf { S } \ : = \ \mathrm { d i a g } \left( \mathbf { s } \right) , \mathbf { s } \ \in \ \mathbb { R } ^ { r }$ $r \ll \operatorname* { m i n } \left( m , n \right)$ , and α is a scale factor. Since U and V define the orthonormal bases for the output and input subspaces, constraining them to the Stiefel manifold is necessary to retain their orthonormality during optimization.

Definition 1 (Stiefel Manifold). The Stiefel manifold St $( k , r )$ of orthonormal matrices in $\mathbb { R } ^ { k \times r } ~ ( r \leq k )$ is defined as

$$
\operatorname { S t } \left( k , r \right) : = \left\{ \mathbf { M } \in \mathbb { R } ^ { k \times r } \mid \mathbf { M } ^ { \top } \mathbf { M } = \mathbf { I } _ { r } \right\} .\tag{2}
$$

b) Optimization on the Stiefel Manifold: Operations on the Stiefel manifold [8], [19] requires additional choices relating to Riemannian geometry, including the local tangent space $\mathcal { T } _ { \mathbf { M } } \operatorname { S t } \left( k , r \right)$ at a point $\textbf { M } \in \ S \mathrm { t } \left( k , r \right)$ containing matrices $\mathbf { Z }$ such that $\mathbf { Z } ^ { \top } \mathbf { M } + \mathbf { M } ^ { \top } \mathbf { Z } = \mathbf { 0 }$ and the canonical metric $\begin{array} { r } { g _ { \mathbf { M } } \left( \mathbf { Z } , \mathbf { T } \right) : = \operatorname { t r } \left( \mathbf { Z } ^ { \top } \left( \mathbf { I } _ { k } - \frac { 1 } { 2 } \mathbf { M } \mathbf { M } ^ { \top } \right) \mathbf { T } \right) } \end{array}$ between two matrices $\mathbf { Z } , \mathbf { T } \in \mathcal { T } _ { \mathbf { M } } \mathrm { S t } \left( \dot { k } , r \right)$ . Given an optimization objective $\mathcal { L } : \mathrm { S t } ( k , r )  \mathbb { R }$ , the Riemannian gradient $\mathrm { g r a d } _ { \mathbf { M } } \mathcal { L } \left( \mathbf { M } \right) \in$ $\mathcal { T } _ { \mathbf { M } } \operatorname { S t } \left( k , r \right)$ of $\mathcal { L }$ can be computed from its Euclidean gradient $\nabla _ { \mathbf { M } } \mathcal { L } \left( \mathbf { M } \right)$ by

$$
\mathrm { g r a d } _ { \mathbf { M } } \mathscr { L } \left( \mathbf { M } \right) : = \nabla _ { \mathbf { M } } \mathscr { L } \left( \mathbf { M } \right) - \mathbf { M } ^ { \top } \left( \nabla _ { \mathbf { M } } \mathscr { L } \left( \mathbf { M } \right) \right) \mathbf { M } .\tag{3}
$$

However, the modification made by the optimizer can cause the update step to deviate from the tangent space. Hence, for an update $\Delta$ obtained after the optimizer step, we need to project it back to $\mathcal { T } _ { \mathbf { M } } \operatorname { S t } \left( k , r \right)$ using the tangent-space projection function:

$$
\mathrm { P r o j } _ { \mathbf { M } } \left( \Delta \right) : = \Delta - \mathbf { M } \mathrm { s y m m } \left( \mathbf { M } ^ { \top } \Delta \right) ,\tag{4}
$$

where symm $\begin{array} { r } { \mathbf { ( A ) } = \frac { 1 } { 2 } \left( \mathbf { A } ^ { \top } + \mathbf { A } \right) } \end{array}$ . After updating along the tangent space with $\Delta _ { \top } : = \mathrm { P r o j } _ { \bf M } \left( \Delta \right)$ , the next point can be obtained from a retraction function chosen as

$$
\begin{array} { r } { \mathrm { R e t r } _ { \mathbf { M } } \left( \Delta _ { \top } \right) : = \mathrm { u f } \left( \mathbf { M } + \Delta _ { \top } \right) , } \end{array}\tag{5}
$$

where uf (·) returns the orthogonal matrix from a polar decomposition. These geometric operations are visualized in Fig. 1.

## B. Stein Variational Gradient Descent for Bayesian Inference

a) Gradient Flow in Probability Space: Let us consider a problem of efficiently sampling from a distribution $p \left( \pmb { \theta } \right) \propto$ $\exp \left( - \beta \Psi \left( \pmb { \theta } \right) \right)$ over a subspace $\mathbf { \Theta } \Theta \subseteq \mathbb { R } ^ { d }$ , where $\Psi \left( \cdot \right)$ is the energy function. In this case, $p$ is the solution of optimization problem:

$$
\operatorname* { m i n } _ { \rho \in \mathscr { P } ( \Theta ) } \left\{ \mathscr { F } \left( \rho \right) : = \int \beta \Psi d \rho + \int \log \rho d \rho \right\} ,\tag{6}
$$

where $\mathcal { P } \left( \Theta \right)$ is the space of distributions over $\Theta$ with the Wasserstein distance [25]. The gradient flow of $\mathcal { F }$ in the Wasserstein space [25] at time s is described by the continuity equation

$$
\partial _ { s } \rho _ { s } + \nabla _ { \pmb { \theta } } \cdot \left( \rho _ { s } \nabla _ { \pmb { \theta } } \frac { \partial \mathcal { F } } { \partial \rho _ { s } } \right) = 0 ,\tag{7}
$$

where $\frac { \partial \mathcal { F } } { \partial \rho _ { s } }$ is the first functional derivative of ${ \mathcal F } .$

![](images/41cd50910cf968322a1eb72093dbcf1c507f0dba6e30501d52b1eb9aca3d15e9.jpg)  
Fig. 1. An example of an update step on the Riemannian manifold $\mathcal { M } .$ Starting from a point M on the manifold with a Euclidean vector $\Delta , \Delta$ is first projected (depicted in blue) to the local tangent space $\mathcal { T } _ { \mathbf { M } } \mathcal { M }$ of M to obtain $\Delta \tau$ . Then, the retraction operation $\mathrm { R e t r } _ { \mathbf { M } } \left( \Delta _ { \top } \right)$ (depicted in red) is conducted to find the next point on the manifold corresponding $\Delta \top$ on the tangent space.

b) Steepest Descent Solution: To solve the gradient flow problem, with each a current distribution $\rho _ { t }$ at the time t, we need to find the velocity field $T \left( \pmb { \theta } \right) = \pmb { \theta } + \eta \phi ^ { * } \left( \pmb { \theta } \right)$ by finding the steepest descent direction:

$$
\phi ^ { * } : = \underset { \phi } { \operatorname { a r g m i n } } \left. \frac { \partial } { \partial \eta } \mathcal { F } \left( T _ { \sharp } \rho _ { t } \right) \right. _ { \eta = 0 } ,\tag{8}
$$

where $T _ { \sharp } \rho _ { t }$ is the push-forward distribution of $\rho _ { t }$ by T. With an additional assumption that $\phi$ belongs to a reproducing kernel Hilbert space (RKHS) $\mathcal { H } _ { \kappa } ^ { d }$ defined by a positive semidefinite kernel $\kappa ( \cdot , \cdot ) : \Theta \times \Theta  \mathbb { R }$ , Stein variational gradient descent (SVGD) [20] achieves a closed-form solution for the optimal update step as follows

$$
\boldsymbol { \phi } ^ { * } \left( \pmb { \theta } \right) = \int \left[ - \beta \kappa \left( \pmb { \theta } ^ { \prime } , \pmb { \theta } \right) \nabla _ { \pmb { \theta } ^ { \prime } } \Psi \left( \pmb { \theta } ^ { \prime } \right) + \nabla _ { \pmb { \theta } ^ { \prime } } \kappa \left( \pmb { \theta } ^ { \prime } , \pmb { \theta } \right) \right] d \rho _ { t } \left( \pmb { \theta } ^ { \prime } \right)\tag{9}
$$

In practice, this formula can be approximated with M particles as follows

$$
\phi ^ { \ast } \left( \pmb { \theta } _ { j } \right) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \left[ - \beta \kappa \left( \pmb { \theta } _ { i } , \pmb { \theta } _ { j } \right) \nabla _ { \pmb { \theta } _ { i } } \Psi \left( \pmb { \theta } _ { i } \right) + \nabla _ { \pmb { \theta } _ { i } } \kappa \left( \pmb { \theta } _ { i } , \pmb { \theta } _ { j } \right) \right] .\tag{10}
$$

## IV. STEPS: RIEMANNIAN SVGD FOR BAYESIAN PEFT ON THE STIEFEL MANIFOLD

The orthonormality offered by the Stiefel manifold allows the full utilization of the subspace rank as well as disentanglement between the dimensions. This is particularly beneficial with low-rank adapters since $r \ll \operatorname* { m i n } \left( m , n \right)$ . Despite the ability to enhance the expressiveness of the subspace and the effectiveness in learning, research regarding estimating the uncertainty with models having parameters on the Stiefel manifold. As mentioned in the previous section SVGD is appropriate for this task due to its flexibility. However, the canonical Euclidean SVGD cannot directly be applied to the adaptation for Bayesian inference. In this section, we introduce the necessary adjustments to the optimization problem and the transportation map, and present the necessary formulation to find the optimal solution with these adjustments.

## A. Gradient Flow on the Riemannian Manifold

To define the optimization problem for U and V in the SVD-based adaptation, we consider a Riemannian manifold M with a corresponding set of distributions over it $\mathcal { P } \left( \mathcal { M } \right)$ in place of the Euclidean subspace Θ. The target distribution $p _ { \mathcal { M } } \left( \pmb { \theta } \right)$ over the manifold that we aim to sample is defined as follows

$$
p _ { \mathcal M } : = \operatorname * { a r g m i n } _ { \rho \in \mathcal P ( \mathcal M ) } \left\{ \mathcal F \left( \rho \right) : = \int \beta \Psi d \rho + \int \log \rho d \rho \right\} .\tag{11}
$$

Similar to SVGD [20], we start from an initial prior distribution $\rho _ { 0 }$ and iteratively transport it to match $p _ { { \mathcal { M } } }$ via a flow of distribution $\{ \rho _ { t } \} _ { t > 0 }$ on the manifold. Assume that at the time step t, we are at the distribution $\rho _ { t } \in \mathcal P \left( \mathcal M \right)$ , our objective is to learn a transportation map T to transport it to the next distribution $\rho _ { t + 1 } : = T _ { \sharp } \rho _ { t } \in { \mathscr { P } \left( \mathcal { M } \right) }$ that further reduces the objective function $\mathcal { F } \left( \cdot \right)$

Proposition 2 (Transportation Map on the Riemannian Manifold). Let the Riemannian manifold M be equipped with $\operatorname { P r o j } _ { \theta }$ and Retr<sub>θ</sub> respectively being the projection map to the tangent space $\mathcal { T } _ { \theta } \mathcal { M }$ at a point θ and the retraction operation from $\mathcal { T } _ { \theta } \mathcal { M }$ back to M. We can formulate the transportation map as follows

$$
T \left( \pmb \theta \right) : = \mathrm { R e t r } _ { \pmb \theta } \left( \eta \mathrm { P r o j } _ { \pmb \theta } \left( \phi \left( \pmb \theta \right) \right) \right) ,\tag{12}
$$

where ϕ is afunction in the RKHS $\mathcal { H } _ { \kappa } ^ { \mathrm { d i m } ( \mathcal { M } ) }$ with the kernel κ.

## B. Steepest Descent Solution for the Stiefel Manifold

a) Theoretical Development for the Riemannian Manifold: With the aforementioned transportation map, we aim to find the optimal $\phi ^ { * }$ by solving

$$
\phi ^ { * } : = \left. \underset { \phi \in \mathcal { H } _ { \kappa } ^ { \mathrm { d i m } ( \mathcal { M } ) } } { \mathrm { a r g m i n } } \ \frac { \partial } { \partial \eta } \mathcal { F } \left( T _ { \sharp } \rho _ { t } \right) \right| _ { \eta = 0 } ,\tag{13}
$$

where $T _ { \sharp } \rho _ { t }$ is the distribution obtained by transporting $\rho _ { t }$ through the map $T .$

Let us denote $\rho _ { t } ^ { [ T ] } : = T _ { \sharp } \rho _ { t }$ , we derive $\mathcal { F } \left( \rho _ { t } ^ { \left[ T \right] } \right)$ as follows:

$$
\begin{array} { r l } & { \mathcal { F } \left( \rho _ { t } ^ { [ T ] } \right) = \beta \displaystyle \int \Psi d \rho _ { t } ^ { [ T ] } + \int \log \rho _ { t } ^ { [ T ] } d \rho _ { t } ^ { [ T ] } } \\ & { \quad \quad \quad \quad = \beta \displaystyle \int \Psi \left( T \left( \pmb { \theta } \right) \right) d \rho _ { t } + \int \log \rho _ { t } ^ { [ T ] } \left( T \left( \pmb { \theta } \right) \right) d \rho _ { t } . } \end{array}\tag{14}
$$

Let us consider a sufficiently small step size $\eta > 0$ so that $T$ is a bijection. As a result, using the formula of the density change for a bijection: $\rho _ { t } \left( \pmb { \theta } \right) = \bar { \rho _ { t } ^ { \left[ T \right] } } \left( T \left( \pmb { \theta } \right) \right) \left| \operatorname* { d e t } \left( \nabla _ { \pmb { \theta } } T \left( \pmb { \theta } \right) \right) \right.$ |, we can rewrite the objective function of interest as:

$$
\begin{array} { r l r } {  { \mathcal { F } ( \rho _ { t } ^ { [ T ] } ) = \beta \int \Psi ( T ( \pmb { \theta } ) ) d \rho _ { t } } } \\ & { } & { + \int [ \log \rho _ { t } ( \pmb { \theta } ) - \log | \operatorname* { d e t } ( \nabla _ { \pmb { \theta } } T ( \pmb { \theta } ) ) | ] d \rho _ { t } } \\ & { } & { = \beta \int \Psi ( \mathrm { R e t r } _ { \pmb { \theta } } ( \eta \mathrm { P r o j } _ { \pmb { \theta } } ( \phi ( \pmb { \theta } ) ) ) ) d \rho _ { t } + \mathrm { c o n s t } } \end{array}
$$

$$
- \int \log \left| \operatorname* { d e t } \left( \nabla _ { \theta } \operatorname { R e t r } _ { \theta } \left( \eta \operatorname { P r o j } _ { \theta } \left( \phi \left( \theta \right) \right) \right) \right) \right| d \rho _ { t } .\tag{15}
$$

Let $\mathcal { F } _ { 1 } \left( \cdot \right)$ and $\mathcal { F } _ { 2 } \left( \cdot \right)$ respectively denote the first and second term of $\mathcal { F } \left( \cdot \right)$ defined in (15). The derivative of the first term can be calculated as

$$
\begin{array} { r l } & { \displaystyle \frac { \partial } { \partial \eta } \mathcal { F } _ { 1 } \left( { \boldsymbol \rho } _ { t } ^ { [ T ] } \right) \bigg \vert _ { \eta = 0 } } \\ & { = \beta \int \langle \nabla _ { { \boldsymbol \theta } } \Psi \left( { \boldsymbol \theta } \right) , D _ { \eta } \mathrm { R e t r } _ { { \boldsymbol \theta } } \left( { \bf 0 } \right) \left[ \mathrm { P r o j } _ { { \boldsymbol \theta } } \left( { \boldsymbol \phi } \left( { \boldsymbol \theta } \right) \right) \right] \rangle d \rho _ { t } } \\ & { = \beta \int \langle \nabla _ { { \boldsymbol \theta } } \Psi \left( { \boldsymbol \theta } \right) , \mathrm { P r o j } _ { { \boldsymbol \theta } } \left( { \boldsymbol \phi } \left( { \boldsymbol \theta } \right) \right) \rangle d \rho _ { t } } \\ & { = \beta \int \langle \mathrm { P r o j } _ { { \boldsymbol \theta } } \left( \nabla _ { { \boldsymbol \theta } } \Psi \left( { \boldsymbol \theta } \right) \right) , { \boldsymbol \phi } \left( { \boldsymbol \theta } \right) \rangle d \rho _ { t } , } \end{array}\tag{16}
$$

where $\operatorname { P r o j } _ { \pmb { \theta } } \left( \nabla _ { \pmb { \theta } } \Psi \left( \pmb { \theta } \right) \right)$ can be computed using the Riemannian gradient in (3). Furthermore, the derivative of the second term can be formulated as

$$
\left. \frac { \partial } { \partial \eta } \mathcal { F } _ { 2 } \left( \rho _ { t } ^ { \left[ T \right] } \right) \right. _ { \eta = 0 } = \mathrm { t r } \left( J \left( \eta \right) ^ { - 1 } J ^ { \prime } \left( \eta \right) \right) \Big \vert _ { \eta = 0 } = \mathrm { t r } \left( J ^ { \prime } \left( \eta \right) \vert _ { \eta = 0 } \right) ,
$$

where $J \left( \eta \right) : = \nabla _ { \pmb { \theta } } \mathrm { R e t r } _ { \pmb { \theta } } \left( \eta \mathrm { P r o j } _ { \pmb { \theta } } \left( \phi \left( \pmb { \theta } \right) \right) \right)$ with $J \left( \mathbf { 0 } \right) = \mathbf { I } .$ It appears that

$$
\begin{array} { l } { { J ^ { \prime } \left( \eta \right) \vert _ { \eta = 0 } = \left. \displaystyle \frac { \partial } { \partial \eta } \nabla _ { \theta } \mathrm { R e t r } _ { \theta } \left( \eta \mathrm { P r o j } _ { \theta } \left( \phi \left( \theta \right) \right) \right) \right. _ { \eta = 0 } } } \\ { { \displaystyle ~ = \nabla _ { \theta } \left[ \left. \displaystyle \frac { \partial } { \partial \eta } \mathrm { R e t r } _ { \theta } \left( \eta \mathrm { P r o j } _ { \theta } \left( \phi \left( \theta \right) \right) \right) \right. _ { \eta = 0 } \right] } } \\ { { \displaystyle ~ = \nabla _ { \theta } \left[ D _ { \eta } \mathrm { R e t r } _ { \theta } \left( \mathbf { 0 } \right) \left[ \mathrm { P r o j } _ { \theta } \left( \phi \left( \theta \right) \right) \right] \right. } } \\ { { \displaystyle ~ = \nabla _ { \theta } \mathrm { P r o j } _ { \theta } \left( \phi \left( \theta \right) \right) . } } \end{array}\tag{17}
$$

Therefore, we can reach

$$
\begin{array} { r l r } & { } & { \left. \frac { \partial } { \partial \eta } \mathcal { F } \left( \rho _ { t } ^ { \left[ T \right] } \right) \right. _ { \eta = 0 } = \beta \int \langle \mathrm { P r o j } _ { \pmb { \theta } } \left( \nabla _ { \pmb { \theta } } \Psi \left( \pmb { \theta } \right) \right) , \phi \left( \pmb { \theta } \right) \rangle d \rho _ { t } } \\ & { } & { - \displaystyle \int \mathrm { t r } \left( \nabla _ { \pmb { \theta } } \mathrm { P r o j } _ { \pmb { \theta } } \left( \phi \left( \pmb { \theta } \right) \right) \right) d \rho _ { t } . \quad ( } \end{array}\tag{18}
$$

b) Theoretical Development for the Stiefel Manifold: Without loss of generality, we assume that $\pmb \theta \in \mathrm { S t } ( k , r )$ is a matrix lying on the Stiefel manifold. Additionally, more complex model parameter can be thought as a collection of matrices on the Stiefel manifolds.

Through some manipulations (see Appendix $\mathbf { A } )$ , we arrive at the following result:

$$
\left. \frac { \partial } { \partial \eta } \mathcal { F } \left( \rho _ { t } ^ { \left[ T \right] } \right) \right| _ { \eta = 0 } = \left. u _ { t } \left( \cdot \right) , \phi \left( \cdot \right) \right. _ { \mathcal { H } _ { \kappa } ^ { \mathrm { d i m } \left( \mathcal { M } \right) } } ,
$$

where we have found that

$$
\boldsymbol { u } _ { t } \left( \cdot \right) = \int \left[ \beta \boldsymbol { \kappa } \left( \pmb { \theta } , \cdot \right) \mathrm { P r o j } _ { \pmb { \theta } } \left( \nabla _ { \pmb { \theta } } \Psi \left( \pmb { \theta } \right) \right) - \mathrm { P r o j } _ { \pmb { \theta } } \left( \nabla _ { \pmb { \theta } } \boldsymbol { \kappa } \left( \pmb { \theta } , \cdot \right) \right) \right] d \rho _ { t } .
$$

As a result, the steepest descent is satisfied when $\phi ^ { * } \left( \cdot \right) =$ $- u _ { t } \left( \cdot \right)$ . This result is formally presented in Theorem 3 below.

```latex
Algorithm 1 Riemannian Stein Variational Gradient Descent for Bayesian PEFT on the Stiefel Manifold (StePS)
Require: Training dataset D, pre-trained weight $\mathbf { W } _ { 0 } ,$ , rank $r ,$ scale factor α, number of particles M, a step function of a
Euclidean optimizer step $( \cdot , \cdot )$ for minimization, and number of iterations $T$
1: for $i \gets 1$ to M do
2: Randomly initialize $\mathbf { U } _ { i , 0 } \in \mathrm { S t } \left( m , r \right)$ and $\mathbf { V } _ { i , 0 } \in \mathrm { S t } \left( n , r \right)$ and set $\mathbf { S } _ { i , 0 }  \mathbf { I } _ { r }$
3: end for
4: for $i \gets 0$ to $T - 1$ do
5: for $i \gets 1$ to M do
6: Compute $\begin{array} { r } { \mathbf { W } _ { i , t } \gets \mathbf { W } _ { 0 } + \frac { \alpha } { r } \mathbf { U } _ { i , t } \mathbf { S } _ { i , t } \mathbf { V } _ { i , t } ^ { \top } } \end{array}$
7: Compute the loss $\mathcal { L } _ { \mathcal { D } } \left( \mathbf { W } _ { i , t } ^ { ' } \right)$ using (22)
8: Compute the Euclidean gradients $\begin{array} { r } { \mathbf { \bar { V } } _ { \mathbf { U } _ { i , t } } \mathcal { L } _ { \mathcal { D } } \left( \mathbf { W } _ { i , t } \right) , \nabla _ { \mathbf { S } _ { i , t } } \mathcal { L } _ { \mathcal { D } } \left( \mathbf { W } _ { i , t } \right) , } \end{array}$ , and $\nabla _ { \mathbf { V } _ { i , t } } \mathcal { L } _ { \mathcal { D } } \left( \mathbf { W } _ { i , t } \right)$
9: Compute Riemannian gradients gra $\mathsf { l } _ { \mathbf { U } _ { i , t } } \mathscr { L } _ { \mathcal { D } } \left( \mathbf { W } _ { i , t } \right)$ and grad ${ \bf { \cdot v } } _ { i , t } \ { \mathcal { L } } _ { { D } } \left( { { \bf { W } } _ { i , t } } \right)$ using (3)
10: end for
11: for $i \gets 1$ to M do
12: Compute the update $\phi ^ { * }$ for $\mathbf { U } _ { i , t }$ and $\mathbf { V } _ { i , t }$ using (20), and for $\mathbf { S } _ { i , t }$ using (10) with the RBF kernel in (21)
13: end for
14: for $i \gets 1$ to M do
15: Project $\phi ^ { * }$ to the tangent space by $\phi _ { \top } ^ { * } ( \mathbf { U } _ { i , t } )  \mathrm { P r o j } _ { \mathbf { U } _ { i , t } } ( \phi ^ { * } ( \mathbf { U } _ { i , t } ) )$ and $\phi _ { \top } ^ { * } ( \mathbf { V } _ { i , t } )  \mathrm { P r o j } _ { \mathbf { V } _ { i , t } } ( \phi ^ { * } ( \mathbf { V } _ { i , t } ) )$ using (4)
16: Compute the next values by $\tilde { \mathbf { U } } _ { i , t } ~ \gets ~ \mathrm { s t e p } \left( \mathbf { U } _ { i , t } , - \phi _ { \top } ^ { * } \left( \mathbf { U } _ { i , t } \right) \right) , ~ \mathbf { S } _ { i , t + 1 } ~ \gets ~ \mathrm { s t e p } \left( \mathbf { S } _ { i , t } , - \phi ^ { * } \left( \mathbf { S } _ { i , t } \right) \right)$ , and $\tilde { \mathbf { V } } _ { i , t } ~ \gets$
step $( \mathbf { V } _ { i , t } , - \phi _ { \top } ^ { * } \left( \mathbf { V } _ { i , t } \right) )$
17: Project the update step to the tangent spaces by $\begin{array} { r l r } { \Delta _ { \top , \mathbf { U } _ { i , t } } } & { { }  } & { \mathrm { P r o j } _ { \mathbf { U } _ { i , t } } ( \tilde { \mathbf { U } } _ { i , t } - \mathbf { U } _ { i , t } ) } \end{array}$ and $\Delta \tau _ { } { , } { \mathbf { v } } _ { i , t } \gets$
Pro $\mathbf { \sigma } _ { \mathbf { V } _ { i , t } } \left( \tilde { \mathbf { V } } _ { i , t } - \mathbf { V } _ { i , t } \right)$ using (4)
18: Update and retract back to the manifold by $\mathbf { U } _ { i , t + 1 }  \mathrm { R e t r } _ { \mathbf { U } _ { i , t } } ( \Delta _ { \top , \mathbf { U } _ { i , t } } )$ and $\mathbf { V } _ { i , t + 1 }  \mathrm { R e t r } _ { \mathbf { V } _ { i , t } } ( \Delta _ { \top , \mathbf { V } _ { i , t } } )$ using (5)
19: end for
20: end for
21: return Final particles $\left\{ { \mathbf { U } } _ { i , T } , { \mathbf { S } } _ { i , T } , { \mathbf { V } } _ { i , T } \right\} _ { i = 1 } ^ { M }$
```

Theorem 3 (Steepest Descent for the Stiefel Manifold). We can choose the optimal solution $\phi ^ { * }$ for the transportation map in (12) to solve the optimization problem in (11) as follows

$$
\begin{array} { r l } & { \boldsymbol { \phi } ^ { * } \left( \pmb { \theta } \right) = \displaystyle \int \left[ - \beta \kappa \left( \pmb { \theta } ^ { \prime } , \pmb { \theta } \right) \mathrm { P r o j } _ { \pmb { \theta } ^ { \prime } } \left( \nabla _ { \pmb { \theta } ^ { \prime } } \Psi \left( \pmb { \theta } ^ { \prime } \right) \right) \right. } \\ & { \qquad \quad \left. + \mathrm { P r o j } _ { \pmb { \theta } ^ { \prime } } \left( \nabla _ { \pmb { \theta } ^ { \prime } } \kappa \left( \pmb { \theta } ^ { \prime } , \pmb { \theta } \right) \right) \right] d \rho _ { t } \left( \pmb { \theta } ^ { \prime } \right) . } \end{array}\tag{19}
$$

## C. Practical Implementation

To practically implement the expectation in the optimal solution in (19), we follow SVGD [20] and approximate with M particles $\{ \pmb { \theta } _ { i } \} _ { i = 1 } ^ { M }$ as

$$
\begin{array} { r l r } {  { \hat { \phi } ^ { * } ( \pmb { \theta } _ { j } ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } [ - \beta \kappa ( \pmb { \theta } _ { i } , \pmb { \theta } _ { j } ) \mathrm { P r o j } _ { \pmb { \theta } _ { i } } ( \nabla _ { \pmb { \theta } _ { i } } \Psi ( \pmb { \theta } _ { i } ) )  } } \\ & { } & {  + \mathrm { P r o j } _ { \pmb { \theta } _ { i } } ( \nabla _ { \pmb { \theta } _ { i } } \kappa ( \pmb { \theta } _ { i } , \pmb { \theta } _ { j } ) ) ] . } \end{array}\tag{20}
$$

Regarding the choice for the kernel $\kappa ,$ we opt for the radial basis function (RBF) kernel:

$$
\kappa _ { \mathrm { R B F } } \left( \pmb { \theta } ^ { \prime } , \pmb { \theta } \right) : = \exp \left( - \frac { \left\| \pmb { \theta } ^ { \prime } - \pmb { \theta } \right\| _ { F } ^ { 2 } } { h } \right) ,\tag{21}
$$

where $h > 0$ is the bandwidth of the kernel and $\left\| \cdot \right\| _ { F }$ denotes the Frobenius norm. Note that this choice is equivalent up to a multiplicative constant to the commonly used von Mises– Fisher (vMF) kernel $\kappa _ { \mathrm { v M F } } \left( { \pmb \theta } ^ { \prime } , { \pmb \theta } \right) ~ : = ~ \mathbf { \bar { e x p } } \left( \gamma \mathrm { t r } \left( { \pmb \theta } ^ { \prime } ^ { \top } { \pmb \theta } \right) \right)$ with $\gamma = 2 / h$ for the Stiefel manifold [21]. Consequently, this choice enables a unified kernel formulation that applies consistently to parameters on both the Stiefel manifold and the Euclidean space. In particular, we compute a joint kernel for all parameters (i.e., U, S, and V) of the adapters and perform additional Stiefel tangent-space projection and retraction for U and V. Regarding the value for the kernel bandwidth h, we adaptively evaluate the bandwidth at each step with $h : = { \mathrm { m e d } } ^ { \dot { 2 } } /$ log M as proposed in SVGD [20], where med is the median of the pairwise distances between each pair $\pmb \theta _ { i }$ and $\theta _ { j } ~ ( i \neq j )$

The selection for the energy function Ψ in (11) is the empirical loss $\mathcal { L } _ { \mathcal { D } } \left( \mathbf { W } \right)$ computed over a training set $\mathcal { D } : =$ $\{ ( \bar { \bf x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ as follows:

$$
\mathcal { L } _ { \mathcal { D } } \left( \mathbf { W } \right) : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \ell \left( f _ { \mathbf { W } } \left( \mathbf { x } _ { i } \right) , y _ { i } \right) ,\tag{22}
$$

where $f _ { \mathbf { W } } \left( \mathbf { x } _ { i } \right)$ is the prediction made by the model f with the parameters W and ℓ is a loss function depending on the current task (e.g., cross-entropy loss in classification task). The detailed optimization process of our method is presented in Algorithm 1.

## V. EXPERIMENTS

## A. Experimental Settings

a) Fine-Tuning Task: Following the experimental set-ups of prior works [15]–[17], we evaluate our approach using a collection of commonsense reasoning benchmarks. We finetune Llama2-7B [2] to answer multiple-choice questions where the answers are evaluated via next-token logits corresponding to possible answers for each dataset. The fine-tuning objective is minimizing the Negative Log-Likelihood of the correct answer tokens.

We consider two settings in the evaluation: in-distribution and out-of-distribution. The former one involves six tasks: Winogrande-Small (WG-S) and Winogrande-Medium (WG-M) [26], ARC-Challenge (ARC-C) and ARC-Easy (ARC-E) [27], OpenBookQA (OBQA) [28], and BoolQ [29]. For each of them, the models are fine-tuned on the training set and evaluated on the test set. In the latter out-of-distribution setting, the models are fine-tuned on the training set of OBQA and evaluated on the test sets of ARC-C and ARC-E, as well as on the MMLU-Chemistry and MMLU-Physics [30] to assess the generalization ability under smaller and larger distribution shifts. The details of these evaluation datasets are available in Table I.

b) Evaluation Metrics: For evaluation, we use accuracy to evaluate the ability of the models to correctly answer the questions. In addition, we utilize the Expected Calibration Error (ECE) [31] and Negative Log-Likelihood (NLL) to evaluate the uncertainty estimation ability of these models. Higher accuracy, and lower ECE and NLL values are preferable, which demonstrate that a model can provide accurate predictions without overconfidence.

c) Baseline Methods: We compare our StePS framework against a range of uncertainty quantification methods for LoRA [6] and StelLA [8]. For the standard LoRA and StelLA training, we report two deterministic configurations: Maximum Likelihood Estimation (MLE), without weight decay regularization, and Maximum A Posteriori (MAP) estimation, with weight decay regularization. We further include standard Bayesian deep learning baselines for both LoRA and StelLA, consisting of Monte Carlo Dropout (MCD) [13] and Deep Ensemble [14], [24]. For recent state-of-the-art approaches for LoRA, we consider Laplace-LoRA [15], BLoB [16], and ScalaBL [17] for comparison. Finally, as a particle-based variational inference baseline, we include Stein variational gradient descent (SVGD) [20] applied in the Euclidean parameter space for LoRA.

d) Implementation Details: We incorporate the adapters into the query and value projections of each self-attention layer, as well as into the softmax output head of the LLM as in prior works [15]–[17]. For both LoRA and StelLA, we choose the rank $r = 1 6$ and the scale factor $\alpha = 3 2 $ All approaches are trained for 5, 000 optimization steps using the AdamW optimizer [32], [33], with a batch size of 4, a linear learning rate scheduler, a warm-up ratio of 0.06, and the maximum learning rate of $1 0 ^ { - 4 }$ . For MAP baselines, a weight decay value of $1 0 ^ { - 5 }$ is applied. During training and evaluation, the frozen base weights of the models are quantized to 8-bit, while trainable parameters remain in 32-bit precision. Except for MLE and MAP baselines, all methods are evaluated using 4 inference samples or particles to ensure a leveling ground for comparison among stochastic and particle-based methods. All remaining method-specific hyperparameters are set to the default values specified by each method. All experiments are performed on a single 32GB NVIDIA V100 or 48GB NVIDIA L40S GPU with 3 random initializations. The implementation of StePS is available at github.com/quangdzuytran/StePS. Additional experimental results are available at Appendix B.

TABLE I  
OVERVIEW OF THE COMMONSENSE REASONING DATASETS USED IN THE EXPERIMENTS.
<table><tr><td>Dataset</td><td>No. Classes</td><td>Train Size</td><td>Test Size</td></tr><tr><td>WG-S [26]</td><td>2</td><td>0.64K</td><td>1.27K</td></tr><tr><td>WG-M [26]</td><td>2</td><td>2.56K</td><td>1.27K</td></tr><tr><td>ARC-C [27]</td><td>4</td><td>1.12K</td><td>0.30K</td></tr><tr><td>ARC-E [27]</td><td>4</td><td>2.25K</td><td>0.57K</td></tr><tr><td>OBQA [28]</td><td>4</td><td>4.96K</td><td>0.50K</td></tr><tr><td>BoolQ [29]</td><td>2</td><td>2.49K</td><td>3.27K</td></tr><tr><td>MMLU-Chemistry* [30]</td><td>4</td><td></td><td>0.10K</td></tr><tr><td>MMLU-Physics* [30]</td><td>4</td><td>一</td><td>0.10K</td></tr></table>

\*MMLU datasets are only used for out-of-distribution evaluations.

## B. In-Distribution Inference Results

The in-distribution results of all methods are presented in Table II. Across six benchmarks, Deep Ensemble, SVGD, and StePS consistently achieve higher accuracy than other benchmarks. Among these approaches, StePS attains relatively low Expected Calibration Errors (ECE) and Negative Log-Likelihood (NLL), achieving the lowest values on several datasets and the second-lowest on most others. Notably, StePS maintains strong calibration while simultaneously achieving the highest accuracy on five out of six datasets. Compared to the MLE and MAP variants of StelLA, StePS also exhibits improvements in all evaluation metrics, which provides a clear evidence that our geometry-aware variational inference approach enhances both accuracy and reliability. Furthermore, StelLA-based methods consistently outperform their corresponding LoRA-based counterparts. This confirms the effectiveness of the Stiefel manifold constraints in learning expressive low-rank parameters.

While Laplace-LoRA and ScalaBL score competitively high accuracy on several benchmarks, their calibration improvements are relatively minimal compared to other baselines for uncertainty quantification. Although BLoB generally obtains the competitive calibration results with lowest ECE and NLL on most benchmarks. These calibration gains are not accompanied by meaningful predictive accuracy. In contrast, our method, StePS, delivers simultaneous improvements in accuracy and calibration that provides a more balanced tradeoff. Overall, these results further highlight the advantages of combining geometry-aware optimization with particle-based variational inference for parameter-efficient fine-tuning.

TABLE II IN-DISTRIBUTION EXPERIMENTS USING LLAMA2-7B.
<table><tr><td rowspan="2">Adapter</td><td rowspan="2">Method</td><td colspan="10">Datasets</td><td colspan="3"></td></tr><tr><td>WG-S</td><td></td><td>ARC-C</td><td></td><td>ARC-E</td><td></td><td></td><td>WG-M</td><td></td><td>OBQA</td><td></td><td>BoolQ</td><td></td></tr><tr><td colspan="10"></td><td colspan="7"></td></tr><tr><td rowspan="9">LoRA</td><td>MLE</td><td>70.94 ± 0.65</td><td></td><td></td><td> $6 8 . 5 8 \pm 2 . 6 3$ </td><td></td><td> $8 6 . 4 4 \pm \ : \ : 0 . 8 7$ </td><td></td><td> $7 6 . 6 6 \pm \ : 1 . 0 4$ </td><td></td><td> $8 2 . 1 3 \pm \ : \ : 0 . 4 1$ </td><td></td><td></td><td>88.56 ± 0.28</td></tr><tr><td>MAP</td><td> $7 0 . 7 8 \pm \ : 1 . 3 2$ </td><td></td><td></td><td> $6 8 . 5 8 \pm 1 . 7 2$ </td><td></td><td> $8 6 . 2 1 \pm \ : \ : 0 . 9 4$ </td><td></td><td> $7 6 . 2 9 \pm \ : \ : 0 . 7 7$ </td><td></td><td> $8 2 . 0 0 \pm \ : \ : 0 . 4 3$ </td><td></td><td>88.44 ± 0.28 88.51 ± 0.14</td><td></td></tr><tr><td>MCD</td><td>70.75 ± 0.13 72.78 ± 0.62</td><td></td><td> $6 9 . 3 7 \pm 1 . 5 2$ </td><td></td><td></td><td> $8 6 . 1 5 \pm 0 . 5 4$ </td><td></td><td> $6 8 . 2 8 \pm 1 2 . 6 5$ </td><td></td><td> $8 3 . 8 0 \pm 1 . 3 0$ </td><td></td><td></td><td></td></tr><tr><td>Ensemble</td><td>70.75 ± 0.52</td><td></td><td> $\underline { { 7 0 . 2 7 \pm } } 0 . 8 3$ </td><td></td><td></td><td> $\mathbf { 8 7 . 2 1 \pm { \ : \ : 0 . 1 7 } }$ </td><td></td><td> $7 7 . 5 6 \pm \ : 0 . 1 9$ </td><td></td><td> $8 3 . 7 3 \pm \ : \ : 0 . 4 7$ </td><td></td><td>89.09 ± 0.04</td><td></td></tr><tr><td>Laplace</td><td></td><td></td><td> $\overline { { 6 9 . 1 4 \pm 1 . 5 2 } }$ </td><td></td><td></td><td> $8 6 . 0 9 \pm \ : \ : 1 . 6 3$ </td><td></td><td>75.87 ± 0.79</td><td></td><td> $8 2 . 0 7 \pm \ : \ : 0 . 9 3$ </td><td></td><td>88.04 ± 0.13</td><td></td></tr><tr><td>BLoB</td><td>52.80 ± 1.99</td><td></td><td> $6 4 . 0 8 \pm \ : 1 . 7 7$ </td><td></td><td></td><td> $7 0 . 8 9 \pm 1 5 . 8 8$ </td><td></td><td>62.29 ± 9.08</td><td></td><td> $7 9 . 6 7 \pm 1 . 6 4$ </td><td></td><td>86.84 ± 0.27</td><td></td></tr><tr><td>ScalaBL</td><td>71.55 ± 0.78</td><td></td><td> $6 8 . 3 6 \pm \ : 1 . 1 5$ </td><td></td><td></td><td> $8 7 . 0 3 \pm \ : \ : 0 . 3 3$ </td><td></td><td>77.56 ± 0.13</td><td></td><td> $8 3 . 3 3 \pm \ : \ : 0 . 6 6$ </td><td></td><td>88.30 ± 0.05</td><td></td></tr><tr><td>SVGD</td><td> $7 3 . 0 7 \pm 0 . 9 1$ </td><td></td><td>69.37 ± 0.97</td><td></td><td> $8 6 . 6 8 \pm \ : \ : 0 . 5 4$ </td><td></td><td></td><td> $7 8 . 1 9 \pm 0 . 4 8$ </td><td></td><td> $\underline { { 8 4 . 3 3 \pm \ : \ : 0 . 3 4 } }$ </td><td></td><td>88.84 ±</td><td>0.17</td></tr><tr><td rowspan="5">MLE StelLA</td><td></td><td> $7 1 . 3 6 \pm 0 . 7 6$ </td><td></td><td>68.47 ± 1.11</td><td></td><td> $8 6 . 4 4 \pm \ : \ : 0 . 2 9$ </td><td></td><td></td><td> $7 6 . 7 7 \pm 0 . 3 6$ </td><td></td><td> $8 2 . 6 7 \pm \ : 1 . 0 6$ </td><td></td><td>88.60 ± 0.21</td><td></td></tr><tr><td>MAP</td><td>71.36 ± 0.76</td><td></td><td> $6 8 . 4 7 \pm 1 . 1 1$ </td><td></td><td> $8 6 . 4 4 \pm \ : \ : 0 . 2 9$ </td><td></td><td></td><td> $7 6 . 7 7 \pm 0 . 3 6$ </td><td></td><td> $8 2 . 6 7 \pm \ : 1 . 0 6$ </td><td></td><td></td><td>88.60 ± 0.21</td></tr><tr><td>MCD</td><td> $7 1 . 3 3 \pm \ : 0 . 6 5$ </td><td></td><td> $6 8 . 8 1 \pm \ : 1 . 8 8$ </td><td></td><td> $8 6 . 2 7 \pm \ : \ : 0 . 8 0$ </td><td></td><td></td><td> $7 7 . 0 3 \pm \ : \ : 0 . 9 7$ </td><td></td><td> $8 1 . 5 3 \pm \ : \ : 0 . 8 1$ </td><td></td><td></td><td> $8 8 . 5 7 \pm 0 . 3 1$ </td></tr><tr><td>Ensemble</td><td> $\underline { { 7 3 . 1 3 \pm } } 0 . 8 7$ </td><td></td><td> $6 9 . 8 2 \pm 1 . 8 8$ </td><td></td><td> $8 7 . 0 9 \pm \ : \ : 0 . 4 1$ </td><td></td><td></td><td> $\underline { { 7 8 . 9 3 \pm \ 0 . 1 6 } }$ </td><td></td><td> $8 3 . 8 0 \pm 0 . 3 3$ </td><td></td><td></td><td> ${ \bf 8 9 . 1 8 \pm \delta 0 . 1 2 }$ </td></tr><tr><td>StePS (Ours)</td><td>73.58 ± 1.13</td><td></td><td></td><td>70.50 ± 1.30</td><td> $8 6 . 8 0 \pm 0 . 6 6$ </td><td></td><td></td><td> $\mathbf { 7 9 . 1 7 \pm 0 . 1 9 }$ </td><td></td><td> $\mathbf { 8 4 . 5 0 \pm { \ : \ : 0 . 6 5 } }$ </td><td></td><td>89.18 ± 0.08</td><td></td></tr><tr><td colspan="10">Expected Calibration Error (↓)</td><td colspan="7"></td></tr><tr><td colspan="10">27.87 ± 0.55 28.60 ± 2.81</td><td colspan="7"></td></tr><tr><td rowspan="8">LoRA</td><td>MLE</td><td>28.44 ± 1.55</td><td></td><td>29.46 ± 1.28</td><td></td><td>11.86 ± 1.27 12.59 ± 0.76</td><td></td><td></td><td>20.44 ± 1.20 19.58 ± 0.82</td><td></td><td>14.84 ± 0.83  $1 4 . 3 6 \pm 0 . 4 0$ </td><td></td><td>4.98 ± 0.38 5.04 ± 0.43</td><td></td></tr><tr><td>MAP MCD</td><td>27.34 ±</td><td>0.60</td><td>27.73 ± 1.45</td><td></td><td></td><td></td><td></td><td>12.82 ± 8.88</td><td></td><td> $1 2 . 9 3 \pm \ : 1 . 1 1$ </td><td></td><td></td><td>4.77 ± 0.46</td></tr><tr><td>Ensemble</td><td>24.11 ± 1.11</td><td></td><td>26.38 ± 0.82</td><td></td><td>12.41 ± 0.59</td><td></td><td></td><td>17.00 ± 0.57</td><td></td><td> $1 2 . 7 9 \pm \ : 0 . 3 2$ </td><td></td><td></td><td>4.88 ± 0.82</td></tr><tr><td></td><td> $2 8 . 0 7 \pm \ : \ : 0 . 3 5$ </td><td></td><td>28.28 ± 1.71</td><td></td><td>11.05 ± 0.27</td><td></td><td></td><td> $2 1 . 0 1 \pm \ : \ : 0 . 9 0$ </td><td></td><td> $1 4 . 0 6 \pm \ : \ : 0 . 8 1$ </td><td></td><td></td><td></td></tr><tr><td>Laplace</td><td>2.72 ± 0.80</td><td></td><td></td><td></td><td></td><td>12.62 ± 1.51</td><td></td><td></td><td></td><td></td><td></td><td></td><td>5.20 ± 0.14</td></tr><tr><td>BLoB</td><td> $2 7 . 5 5 \pm \ : 1 . 0 0$ </td><td></td><td></td><td> $\mathbf { 7 . 6 7 \pm 1 . 6 6 }$ </td><td></td><td> $1 0 . 5 1 \pm \ : \ : 4 . 2 9$ </td><td></td><td> $\mathbf { 3 . 3 7 \pm { \ : \ : 0 . 1 8 } }$ </td><td></td><td> $\mathbf { 6 . 9 2 \pm \ : 3 . 1 0 }$ </td><td></td><td></td><td> ${ \bf 1 . 5 1 \pm \delta 0 . 2 9 }$ </td></tr><tr><td>ScalaBL</td><td></td><td></td><td> $2 9 . 7 2 \pm \ : 1 . 0 3$ </td><td></td><td> $1 1 . 7 4 \pm 0 . 6 2$ </td><td></td><td></td><td> $1 9 . 3 4 \pm \ : \ : 0 . 4 8$ </td><td></td><td> $1 3 . 0 2 \pm \ : 1 . 0 9$ </td><td></td><td></td><td> $5 . 1 5 \pm \ : 0 . 2 8$ </td></tr><tr><td>SVGD</td><td>19.79 ± 0.83</td><td></td><td> $2 1 . 1 0 \pm 1 . 0 6$ </td><td></td><td> $\underline { { 1 0 . 1 3 \pm \ : \ : 0 . 3 9 } }$ </td><td></td><td></td><td> $1 1 . 4 4 \pm \ : 5 . 2 9$ </td><td></td><td> $1 0 . 8 7 \pm \ : \ : 0 . 3 7$ </td><td></td><td> $4 . 8 7 \pm \ : \ : 0 . 0 9$ </td><td></td></tr><tr><td></td><td></td><td>27.31 ± 1.29</td><td></td><td> $2 8 . 2 6 \pm \ : \ : 0 . 8 7$ </td><td></td><td> $1 1 . 7 6 \pm \ : \ : 0 . 3 5$ </td><td></td><td>19.82 ± 0.61</td><td></td><td> $1 3 . 4 5 \pm \ : 1 . 0 8$ </td><td></td><td></td><td>5.05 ± 0.40</td></tr><tr><td rowspan="5">StelLA</td><td>MLE</td><td>27.31 ± 1.29</td><td></td><td> $2 8 . 2 6 \pm \ : \ : 0 . 8 7$ </td><td></td><td> $1 1 . 7 6 \pm \ : \ : 0 . 3 5$ </td><td></td><td></td><td></td><td></td><td> $1 3 . 4 5 \pm \ : 1 . 0 8$ </td><td></td><td>5.05 ± 0.40</td><td></td></tr><tr><td>MAP</td><td>26.85 ± 1.05</td><td></td><td></td><td> $2 7 . 8 8 \pm \ : 2 . 7 4$ </td><td></td><td></td><td></td><td>19.82 ± 0.61</td><td></td><td></td><td> $1 4 . 6 1 \pm \ : \ : 0 . 7 6$ </td><td></td><td></td></tr><tr><td>MCD</td><td>24.11 ± 1.17</td><td></td><td></td><td></td><td>11.74 ± 0.80</td><td></td><td></td><td>18.90 ± 0.90</td><td></td><td></td><td></td><td></td><td>4.98 ± 0.23</td></tr><tr><td>Ensemble StePS (Ours)</td><td>17.68 ± 1.22</td><td></td><td></td><td></td><td> $1 1 . 0 2 \pm \ : 0 . 1 2$ </td><td></td><td></td><td>15.09 ± 1.15</td><td></td><td> $1 2 . 6 4 \pm \ : \ : 0 . 6 9$  9.23 ± 0.47</td><td></td><td>4.59 ± 0.00</td><td>4.15 ± 0.25</td></tr><tr><td colspan="10"</table>

The reported results are mean±std of the metrics evaluated on the test sets over 3 random initializations. Bold and underlined results denote the best and second-best mean values on each metric for each dataset.

## C. Out-of-Distribution Inference Results

form their respective LoRA-based variants on most out-ofdistribution benchmarks. This suggests that the Stiefel manifold formulation yields more robust low-rank representations under distribution shifts, especially when the domain gaps are wider as observed on the MMLU datasets.

Table III presents the out-of-distribution evaluation results under both smaller and larger distribution shifts. In general, all methods achieve satisfactory results on datasets with smaller distribution shifts, ARC-C and ARC-E, while demonstrating obvious degradation in accuracy and calibration on the datasets with larger distribution shifts, MMLU-Chemistry and MM-LU-Physics. This behavior aligns with previous observations that domain shifts involving specialized knowledge can create a great challenge for fine-tuned LLMs [16], [17]. A consistent pattern also emerges where StelLA-based methods outper-

Although StePS does not always have the highest predictive accuracy, its performance remains highly competitive in comparison to the baselines. Crucially, StePS achieves robust and consistent calibration results with the second-lowest ECE and NLL over all datasets. While BLoB obtains the lowest ECE and NLL scores, its accuracy still remains lower than

TABLE III  
OUT-OF-DISTRIBUTION EXPERIMENTS USING LLAMA2-7B.
<table><tr><td rowspan="2">Adapter</td><td rowspan="2">Method</td><td colspan="2">In-Distribution</td><td colspan="3">Smaller Distribution Shift</td><td colspan="4">Larger Distribution Shift</td></tr><tr><td>OBQA</td><td></td><td>ARC-C</td><td></td><td>ARC-E</td><td></td><td>MMLU-Chemistry</td><td></td><td>MMLU-Physics</td></tr><tr><td colspan="10">Accuracy (↑)</td></tr><tr><td rowspan="9"></td><td>MLE</td><td> $8 2 . 1 3 \pm \ : 0 . 4 1$ </td><td></td><td> $7 0 . 1 6 \pm \ : \ : 0 . 4 2$ </td><td></td><td> $7 5 . 4 1 \pm 1 . 7 1$ </td><td></td><td> $3 4 . 0 0 \pm \ : 1 . 6 3$ </td><td></td><td> $2 2 . 6 7 \pm \ : 1 . 2 5$ </td></tr><tr><td>MAP</td><td> $8 2 . 0 0 \pm \ : \ : 0 . 4 3$ </td><td></td><td> $7 0 . 1 6 \pm \ : \ : 0 . 4 2$ </td><td></td><td> $7 5 . 6 5 \pm 2 . 0 4$ </td><td></td><td> $3 4 . 0 0 \pm \ : 1 . 6 3$ </td><td></td><td>23.33 ± 0.47</td></tr><tr><td>MCD</td><td>83.80 ± 1.30</td><td></td><td> $6 9 . 0 3 \pm \ : 1 . 8 4$ </td><td></td><td> $7 6 . 5 3 \pm \ : 0 . 2 2$ </td><td></td><td> $3 7 . 6 7 \pm \ : 3 . 3 0$ </td><td></td><td>26.33 ± 3.09</td></tr><tr><td>Ensemble</td><td>83.73 ± 0.47</td><td></td><td> $\mathbf { 7 1 . 7 3 \pm { \ : \ : 1 . 1 1 } }$ </td><td></td><td> $7 7 . 5 8 \pm \ : 0 . 5 4$ </td><td></td><td> $3 9 . 6 7 \pm 2 . 6 2$ </td><td></td><td>27.33 ± 1.25</td></tr><tr><td>Laplace</td><td>82.07 ± 0.93</td><td></td><td> $7 0 . 1 6 \pm 2 . 2 5$ </td><td></td><td> $\overline { { 7 6 . 2 3 \pm \ : \ : 0 . 3 8 } }$ </td><td></td><td> $3 6 . 3 3 \pm \ : 1 . 8 9$ </td><td>28.67 ± 5.19</td><td></td></tr><tr><td>BLoB</td><td>79.67 ± 1.64</td><td></td><td> $6 8 . 9 8 \pm \ : 1 . 8 4$ </td><td></td><td> $7 5 . 8 9 \pm 1 . 1 6$ </td><td></td><td> $3 9 . 2 4 \pm \ : 2 . 7 3$ </td><td>31.25 ± 0.00</td><td></td></tr><tr><td>ScalaBL SVGD</td><td>83.33 ± 0.66 84.33 ± 0.34</td><td></td><td> $6 9 . 2 1 \pm \ : 1 . 2 8$ </td><td></td><td> $7 6 . 6 7 \pm \ : 1 . 1 9$ </td><td></td><td> $3 6 . 4 6 \pm 2 . 5 5$ </td><td></td><td>29.51 ± 2.73</td></tr><tr><td></td><td></td><td></td><td> $6 8 . 8 1 \pm \ : 1 . 7 7$ </td><td></td><td> $7 6 . 5 3 \pm \ : 0 . 3 3$ </td><td></td><td> $3 3 . 6 7 \pm \ : 3 . 8 6$ </td><td></td><td>31.00 ± 2.16</td></tr><tr><td rowspan="5">StelLA</td><td>MLE</td><td> $8 2 . 6 7 \pm \ : 1 . 0 6$ </td><td></td><td> $7 0 . 0 5 \pm \ : 1 . 3 9$ </td><td></td><td> $7 7 . 0 0 \pm \ : \ : 0 . 8 4$ </td><td></td><td> $3 9 . 0 0 \pm 2 . 1 6$ </td><td>27.33 ± 3.40</td><td></td></tr><tr><td>MAP</td><td> $8 2 . 6 7 \pm \ : 1 . 0 6$ </td><td></td><td> $7 1 . 2 8 \pm 1 . 1 0$ </td><td></td><td> $7 7 . 0 0 \pm \ : \ : 0 . 3 3$ </td><td></td><td> $\underline { { 4 0 . 0 0 \pm } } ~ 3 . 5 6$ </td><td></td><td>29.33 ± 0.94</td></tr><tr><td>MCD</td><td> $8 1 . 5 3 \pm \ : \ : 0 . 8 1$   $8 3 . 8 0 \pm 0 . 3 3$ </td><td></td><td> $\overline { { 6 9 . 0 3 \pm \ : \ : 0 . 5 7 } }$   $7 0 . 9 5 \pm \ : \ : 0 . 2 8$ </td><td></td><td> $7 7 . 2 9 \pm \ : 1 . 9 8$ </td><td></td><td> $\overline { { 3 3 . 3 3 \pm \ 3 . 4 0 } }$ </td><td></td><td> $3 1 . 0 0 \pm \ : 3 . 5 6$ </td></tr><tr><td>Ensemble</td><td> $\mathbf { 8 4 . 5 0 \pm { \ : \ : 0 . 6 5 } }$ </td><td></td><td> $7 0 . 5 0 \pm \ : \ : 0 . 6 9$ </td><td></td><td> $7 7 . 4 0 \pm \ : 0 . 5 0$ </td><td></td><td> $3 9 . 3 3 \pm \ : \ : 0 . 4 7$ </td><td></td><td> $3 1 . 3 3 \pm \ : 2 . 8 7$ </td></tr><tr><td>StePS (Ours)</td><td></td><td></td><td></td><td></td><td> $\mathbf { 7 7 . 7 6 \pm . 0 . 2 2 }$ </td><td></td><td> $\mathbf { 4 0 . 6 7 \pm { \ : \ : 1 . 7 0 } }$ </td><td></td><td> $\mathbf { 3 2 . 6 7 \pm 4 . 1 1 }$ </td></tr><tr><td colspan="10">Expected Calibration Error (↓)</td></tr><tr><td rowspan="9">LoRA</td><td>MLE</td><td>14.84 ± 0.83  $1 4 . 3 6 \pm 0 . 4 0$ </td><td></td><td>23.55 ± 0.77  $2 3 . 6 4 \pm \ : \ : 0 . 7 3$ </td><td></td><td> $1 8 . 0 8 \pm \ : 1 . 3 4$   $1 7 . 9 0 \pm \ : 1 . 6 0$ </td><td></td><td> $2 9 . 6 9 \pm \ : 1 . 1 1$   $3 0 . 1 9 \pm \ : \ : 0 . 4 1$ </td><td></td><td>39.29 ± 0.38 38.75 ± 1.14</td></tr><tr><td>MAP MCD</td><td> $1 2 . 9 3 \pm \ : 1 . 1 1$ </td><td></td><td> $2 3 . 7 2 \pm 2 . 1 5$ </td><td></td><td> $1 6 . 7 4 \pm \ : \ : 0 . 1 6$ </td><td></td><td> $2 7 . 4 5 \pm \ : 3 . 3 3$ </td><td></td><td>33.54 ± 3.51</td></tr><tr><td>Ensemble</td><td> $1 2 . 7 9 \pm \ : 0 . 3 2$ </td><td></td><td> $2 1 . 5 4 \pm 1 . 3 9$ </td><td></td><td> $1 5 . 4 4 \pm \ : 0 . 6 5$ </td><td></td><td> $2 5 . 3 6 \pm 2 . 0 0$ </td><td></td><td>32.55 ± 0.41</td></tr><tr><td>Laplace</td><td> $1 4 . 0 6 \pm \ : \ : 0 . 8 1$ </td><td></td><td> $2 2 . 1 1 \pm 1 . 0 1$ </td><td></td><td> $1 6 . 7 3 \pm \ : 0 . 7 4$ </td><td></td><td> $2 7 . 0 4 \pm \ : \ : 0 . 9 6$ </td><td></td><td> $3 1 . 9 9 \pm \ : 6 . 0 2$ </td></tr><tr><td>BLoB</td><td> $\mathbf { 6 . 9 2 \pm \ : 3 . 1 0 }$ </td><td></td><td></td><td></td><td></td><td></td><td> $\mathbf { 1 3 . 9 6 \pm \ : \ : 0 . 5 0 }$ </td><td></td><td></td></tr><tr><td>ScalaBL</td><td> $1 3 . 0 2 \pm \ : 1 . 0 9$ </td><td></td><td> $\mathbf { 1 2 . 0 1 \pm \ : \ : 1 . 2 4 }$ </td><td></td><td> $\mathbf { 8 . 0 8 \pm . 0 . 8 7 }$ </td><td></td><td> $2 6 . 0 5 \pm 2 . 2 3$ </td><td> $\mathbf { 1 8 . 1 0 \pm . 1 . 5 1 }$ </td><td></td></tr><tr><td>SVGD</td><td> $1 0 . 8 7 \pm \ : \ : 0 . 3 7$ </td><td></td><td> $2 3 . 3 3 \pm \ : 0 . 7 2$   $2 3 . 1 7 \pm 1 . 3 3$ </td><td></td><td> $1 6 . 2 7 \pm \ : 1 . 0 1$ </td><td></td><td> $2 9 . 5 5 \pm \ : 6 . 1 6$ </td><td></td><td> $3 1 . 9 5 \pm \ : 1 . 7 0$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td> $1 5 . 4 7 \pm 0 . 6 2$ </td><td></td><td></td><td></td><td> $3 0 . 1 8 \pm \ : 3 . 1 5$ </td></tr><tr><td>MLE MAP</td><td> $1 3 . 4 5 \pm \ : 1 . 0 8$ </td><td></td><td> $2 3 . 5 9 \pm 2 . 2 8$ </td><td></td><td> $1 7 . 0 6 \pm \ : \ : 0 . 7 5$ </td><td></td><td> $2 8 . 3 3 \pm \ : 0 . 7 9$ </td><td></td><td> $3 8 . 3 1 \pm \ : 3 . 3 3$ </td></tr><tr><td rowspan="5">StelLA</td><td></td><td> $1 3 . 4 5 \pm \ : 1 . 0 8$ </td><td></td><td> $2 1 . 3 7 \pm 1 . 2 3$ </td><td> $1 6 . 5 4 \pm \ : \ : 0 . 5 0$ </td><td></td><td> $2 7 . 7 2 \pm \ : 0 . 6 6$ </td><td></td><td>35.53 ± 1.76</td><td></td></tr><tr><td>MCD</td><td> $1 4 . 6 1 \pm \ : \ : 0 . 7 6$ </td><td></td><td> $2 2 . 9 8 \pm \ : \ : 0 . 6 5$ </td><td></td><td> $1 6 . 2 0 \pm \ : 1 . 2 2$ </td><td> $3 4 . 9 2 \pm \ : 5 . 0 2$ </td><td></td><td>35.30 ± 3.01</td><td></td></tr><tr><td>Ensemble</td><td> $1 2 . 6 4 \pm \ : \ : 0 . 6 9$ </td><td></td><td> $2 1 . 4 7 \pm \ : \ : 0 . 2 3$ </td><td></td><td> $1 3 . 8 6 \pm 0 . 5 4$ </td><td></td><td> $2 3 . 6 0 \pm \ : \ : 0 . 9 0$ </td><td></td><td>31.13 ± 2.84</td></tr><tr><td>StePS (Ours)</td><td>9.23 ± 0.47</td><td></td><td> $\underline { { 1 8 . 1 2 \pm \ : \ : 0 . 2 7 } }$ </td><td></td><td> $\underline { { 1 2 . 3 6 \pm } } 0 . 2 6$ </td><td></td><td>22.34 ± 1.33</td><td></td><td>30.10 ± 2.21</td></tr><tr><td colspan="10">Negative Log-Likelihood (↓)</td></tr><tr><td rowspan="9"></td><td>MLE</td><td> $0 . 9 7 \pm \ : \ : 0 . 0 6$ </td><td></td><td> $1 . 5 0 \pm \ : \ : 0 . 0 3$ </td><td></td><td> $1 . 1 6 \pm \ : \ : 0 . 0 6$ </td><td></td><td> $1 . 8 1 \pm \ : \ : 0 . 1 2$ </td><td></td><td> $1 . 9 5 \pm \ : \ : 0 . 1 1$ </td></tr><tr><td>MAP MCD</td><td> $0 . 9 3 \pm \ : \ : 0 . 0 4$   $0 . 8 8 \pm \ : \ : 0 . 0 5$ </td><td></td><td> $1 . 5 1 \pm \ : 0 . 0 4$ </td><td></td><td> $1 . 1 7 \pm \ : 0 . 0 4$ </td><td></td><td> $1 . 8 1 \pm \ : \ : 0 . 1 2$ </td><td></td><td> $1 . 9 5 \pm \ : \ : 0 . 1 1$ </td></tr><tr><td>Ensemble</td><td> $0 . 8 1 \pm \ : \ : 0 . 0 2$ </td><td></td><td> $1 . 5 1 \pm \ : \ : 0 . 0 9$   $1 . 3 9 \pm \ : \ : 0 . 0 2$ </td><td></td><td> $1 . 1 0 \pm \ : \ : 0 . 0 4$ </td><td> $1 . 7 1 \pm \ : \ : 0 . 0 2$ </td><td> $1 . 8 1 \pm \ : \ : 0 . 0 6$ </td><td> $1 . 8 8 \pm \ : \ : 0 . 0 8$   $1 . 8 1 \pm \ : \ : 0 . 0 2$ </td><td></td></tr><tr><td>Laplace</td><td> $0 . 8 8 \pm \ : \ : 0 . 0 3$ </td><td></td><td> $1 . 4 3 \pm \ : \ : 0 . 0 7$ </td><td></td><td> $1 . 0 2 \pm \ : \ : 0 . 0 2$   $1 . 0 9 \pm \ : \ : 0 . 0 1$ </td><td></td><td> $1 . 7 0 \pm \ : \ : 0 . 0 9$ </td><td> $1 . 7 8 \pm \ : \ : 0 . 0 8$ </td><td></td></tr><tr><td>BLoB</td><td> $\mathbf { 0 . 5 6 \pm 0 . 0 4 }$ </td><td></td><td> $\mathbf { 0 . 9 0 \pm . 0 . 0 7 }$ </td><td></td><td> $\mathbf { 0 . 7 0 \pm } \ \mathbf { 0 . 0 4 }$ </td><td></td><td> $\mathbf { 1 . 4 3 \pm 0 . 0 2 }$ </td><td> $\mathbf { 1 . 5 0 \pm } \ \mathbf { 0 . 0 2 }$ </td><td></td></tr><tr><td>ScalaBL</td><td> $0 . 8 8 \pm \ : \ : 0 . 0 7$ </td><td></td><td></td><td></td><td></td><td></td><td> $1 . 6 6 \pm \ : \ : 0 . 0 5$ </td><td></td><td> $1 . 7 4 \pm \ : \ : 0 . 0 7$ </td></tr><tr><td>SVGD</td><td> $0 . 8 0 \pm \ : \ : 0 . 0 4$ </td><td></td><td> $1 . 4 5 \pm \ : \ : 0 . 0 4$ </td><td></td><td> $1 . 0 8 \pm \ : \ : 0 . 0 6$ </td><td></td><td></td><td></td><td></td></tr><tr><td>MLE</td><td></td><td></td><td> $1 . 4 2 \pm \ : \ : 0 . 0 3$ </td><td></td><td> $1 . 0 4 \pm \ : \ : 0 . 0 2$ </td><td></td><td> $1 . 8 7 \pm \ : \ : 0 . 1 1$ </td><td></td><td> $1 . 9 0 \pm \ : \ : 0 . 1 3$ </td></tr><tr><td rowspan="5">StelLA</td><td></td><td> $0 . 8 7 \pm \ : \ : 0 . 0 4$ </td><td></td><td> $1 . 5 7 \pm \ : 0 . 1 2$   $1 . 4 7 \pm \ : \ : 0 . 0 7$ </td><td></td><td> $1 . 1 3 \pm \ : 0 . 0 3$   $1 . 0 7 \pm \ : \ : 0 . 0 6$ </td><td></td><td> $1 . 7 7 \pm \ : \ : 0 . 1 1$   $1 . 8 0 \pm \ : \ : 0 . 0 3$ </td><td></td><td> $2 . 0 2 \pm \ : \ : 0 . 0 4$   $1 . 9 9 \pm \ : \ : 0 . 1 4$ </td></tr><tr><td>MAP MCD</td><td> $0 . 8 7 \pm \ : \ : 0 . 0 4$   $0 . 9 4 \pm \ : \ : 0 . 0 2$ </td></table>

The reported results are mean±std of the metrics evaluated on the test sets over 3 random initializations Bold and underlined results denote the best and second-best mean values on each metric for each dataset.

## VI. CONCLUSION

other baselines on ARC-C and ARC-E datasets. As a result, StePS still demonstrates a more favorable balance between the predictive performance and uncertainty calibration on the out-of-distribution setting.

this, we derive a closed-form update from the steepest descent of the associated gradient flow on the Stiefel manifold, providing a clear theoretical foundation for our method. The extensive in-distribution and out-of-distribution evaluations of our methods on diverse benchmarks demonstrate the effectiveness of our approach. StePS delivers promising results with better accuracy and comparable model calibration on in-distribution and out-of-distribution settings. In future work, we will explore model-based variational inference on the Riemannian manifold to further enhance the computational efficiency of geometryware Bayesian PEFT.

In this work, we have proposed a framework based on Riemannian Stein variational gradient descent, StePS, to both quantify the uncertainty and calibrate geometry-aware PEFT. StePS allows explicit and principled estimation of the epistemic uncertainty by iteratively transporting the particles on the Stiefel manifold toward the target distribution. To achieve

[1] OpenAI, “GPT-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.

[2] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale et al., “Llama 2: Open foundation and fine-tuned chat models,” arXiv preprint arXiv:2307.09288, 2023.

[3] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan et al., “The Llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[4] J. Clusmann, F. R. Kolbinger, H. S. Muti, Z. I. Carrero, J.-N. Eckardt, N. G. Laleh, C. M. L. Loffler, S.-C. Schwarzkopf, M. Unger, G. P.¨ Veldhuizen et al., “The future landscape of large language models in medicine,” Communications Medicine, vol. 3, no. 1, p. 141, 2023.

[5] Y. Zhang, X. Chen, B. Jin, S. Wang, S. Ji, W. Wang, and J. Han, “A comprehensive survey of scientific large language models and their applications in scientific discovery,” in Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2024, pp. 8783–8817.

[6] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in Proceedings of International Conference on Learning Representations (ICLR), 2022.

[7] Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi, “Self-Instruct: Aligning language models with selfgenerated instructions,” in Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2023, pp. 13 484– 13 508.

[8] Z. Li, S. Sajadmanesh, J. Li, and L. Lyu, “StelLA: Subspace learning in low-rank adaptation using Stiefel manifold,” in Advances on Neural Information Processing Systems (NeurIPS), vol. 38, 2025.

[9] J. Park, M. Kang, S. Lee, H. Lee, S. Kim, and J. Lee, “Riemannian optimization for LoRA on the Stiefel manifold,” in Findings of the Association for Computational Linguistics: EMNLP 2025, 2025, pp. 20 971–20 985.

[10] Z. Jiang, J. Araki, H. Ding, and G. Neubig, “How can we know when language models know? On the calibration of language models for question answering,” Transactions of the Association for Computational Linguistics, vol. 9, pp. 962–977, 2021.

[11] S. Kadavath, T. Conerly, A. Askell, T. Henighan, D. Drain, E. Perez, N. Schiefer, Z. Hatfield-Dodds, N. DasSarma, E. Tran-Johnson et al., “Language models (mostly) know what they know,” arXiv preprint arXiv:2207.05221, 2022.

[12] C. Blundell, J. Cornebise, K. Kavukcuoglu, and D. Wierstra, “Weight uncertainty in neural network,” in Proceedings of the International Conference on Machine Learning (ICML), ser. Proceedings of Machine Learning Research, vol. 37, 2015, pp. 1613–1622.

[13] Y. Gal and Z. Ghahramani, “Dropout as a Bayesian approximation: Representing model uncertainty in deep learning,” in Proceedings of the International Conference on Machine Learning (ICML), ser. Proceedings of Machine Learning Research, vol. 48, 2016, pp. 1050–1059.

[14] B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and scalable predictive uncertainty estimation using deep ensembles,” Advances in Neural Information Processing Systems (NIPS), vol. 30, 2017.

[15] A. X. Yang, M. Robeyns, X. Wang, and L. Aitchison, “Bayesian lowrank adaptation for large language models,” in Proceedings of the International Conference on Learning Representations (ICLR), 2024.

[16] Y. Wang, H. Shi, L. Han, D. N. Metaxas, and H. Wang, “BLoB: Bayesian low-rank adaptation by backpropagation for large language models,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 37, 2024, pp. 67 758–67 794.

[17] C. Samplawski, A. D. Cobb, M. Acharya, R. Kaur, and S. Jha, “Scalable Bayesian low-rank adaptation of large language models via stochastic variational subspace inference,” in Proceedings of the Conference on Uncertainty in Artificial Intelligence (UAI), ser. Proceedings of Machine Learning Research, vol. 286, 2025, pp. 3587–3604.

[18] K. Lion, L. Zhang, B. Li, and N. He, “PoLAR: Polar-decomposed low-rank adapter representation,” in Proceedings of the International Conference on Machine Learning (ICML) Workshop on Efficient Systems for Foundation Models, 2025.

[19] P.-A. Absil, Optimization Algorithms on Matrix Manifolds. Princeton University Press, 2008.

[20] Q. Liu and D. Wang, “Stein variational gradient descent: A general purpose Bayesian inference algorithm,” in Advances in Neural Information Processing Systems (NIPS), vol. 29, 2016.

[21] C. Liu and J. Zhu, “Riemannian Stein variational gradient descent for Bayesian inference,” in Proceedings ofthe AAAI Conference on Artificial Intelligence (AAAI), vol. 32, no. 1, 2018, pp. 3627–3634.

[22] Q. Zhang, M. Chen, A. Bukharin, P. He, Y. Cheng, W. Chen, and T. Zhao, “Adaptive budget allocation for parameter-efficient fine-tuning,” in Proceedings of the International Conference on Learning Representations (ICLR), 2023.

[23] S. Schotthofer, E. Zangrando, G. Ceruti, F. Tudisco, and J. Kusch,¨ “GeoLoRA: Geometric integration for parameter efficient fine-tuning,” in Proceedings of the International Conference on Learning Representations (ICLR), 2025.

[24] O. Balabanov and H. Linander, “Uncertainty quantification in fine-tuned LLMs using LoRA ensembles,” in Proceedings of the International Conference on Learning Representations (ICLR) Workshop on Quantify Uncertainty and Hallucination in Foundation Models: The Next Frontier in Reliable AI, 2025.

[25] L. Ambrosio, N. Gigli, and G. Savare,´ Gradient Flows: In Metric Spaces and in the Space of Probability Measures, ser. Lectures in Mathematics. ETH Zurich. Basel Boston Berlin: Birkh¨ auser, 2008.¨

[26] K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi, “WinoGrande: An adversarial Winograd schema challenge at scale,” Communications of the ACM, vol. 64, no. 9, pp. 99–106, 2021.

[27] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord, “Think you have solved question answering? try ARC, the AI2 reasoning challenge,” arXiv preprint arXiv:1803.05457, 2018.

[28] T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal, “Can a suit of armor conduct electricity? a new dataset for open book question answering,” in Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018, pp. 2381–2391.

[29] C. Clark, K. Lee, M.-W. Chang, T. Kwiatkowski, M. Collins, and K. Toutanova, “BoolQ: Exploring the surprising difficulty of natural yes/no questions,” in Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2019, pp. 2924–2936.

[30] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, “Measuring massive multitask language understanding,” in Proceedings of the International Conference on Learning Representations (ICLR), 2021.

[31] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proceedings of the International Conference on Machine Learning (ICML), ser. Proceedings of Machine Learning Research, vol. 70, 2017, pp. 1321–1330.

[32] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in Proceedings of the International Conference for Learning Representations (ICLR)), 2015.

[33] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proceedings of the International Conference on Learning Representations (ICLR), 2019.

## APPENDIX A PROOF OF THEOREM 3

Proof. Let us recall the formula in (18) as follow

$$
\begin{array} { r l r } & { } & { \left. \frac { \partial } { \partial \eta } \mathcal { F } \left( \rho _ { t } ^ { \left[ T \right] } \right) \right. _ { \eta = 0 } = \beta \int \langle \mathrm { P r o j } _ { \pmb { \theta } } \left( \nabla _ { \pmb { \theta } } \Psi \left( \pmb { \theta } \right) \right) , \phi \left( \pmb { \theta } \right) \rangle d \rho _ { t } } \\ & { } & { - \displaystyle \int \mathrm { t r } \left( \nabla _ { \pmb { \theta } } \mathrm { P r o j } _ { \pmb { \theta } } \left( \phi \left( \pmb { \theta } \right) \right) \right) d \rho _ { t } . \quad \mp } \end{array}\tag{18}
$$

The first term in (18) can be derived as

$$
\begin{array} { r l } & { \beta \displaystyle \int \left. \mathrm { P r o j } _ { \pmb \theta } \left( \nabla _ { \pmb \theta } \Psi \left( \pmb \theta \right) \right) , \phi \left( \pmb \theta \right) \right. d \rho _ { t } } \\ & { = \beta \displaystyle \int \left. \mathrm { P r o j } _ { \pmb \theta } \left( \nabla _ { \pmb \theta } \Psi \left( \pmb \theta \right) \right) , \left. \kappa \left( \pmb \theta , \cdot \right) , \phi \left( \cdot \right) \right. _ { \mathcal { H } _ { \kappa } ^ { d } } \right. d \rho _ { t } } \\ & { = \beta \displaystyle \int \left. \kappa \left( \pmb \theta , \cdot \right) \mathrm { P r o j } _ { \pmb \theta } \left( \nabla _ { \pmb \theta } \Psi \left( \pmb \theta \right) \right) , \phi \left( \cdot \right) \right. _ { \mathcal { H } _ { \kappa } ^ { d } } d \rho _ { t } , } \end{array}\tag{23}
$$

where $\mathcal { H } _ { \kappa }$ is the RKHS with the kernel κ and $d = k \times r$ . For the second term, we consider the derivative of the tangentprojection in (4) as

$$
\begin{array} { r l } & { \nabla _ { \pmb \theta } \mathrm { P r o j } _ { \pmb \theta } \left( \phi \left( \pmb \theta \right) \right) } \\ & { = \nabla _ { \pmb \theta } \left[ \phi \left( \pmb \theta \right) - \frac { 1 } { 2 } \pmb \theta \left[ \pmb \theta ^ { \top } \phi \left( \pmb \theta \right) + \phi \left( \pmb \theta \right) ^ { \top } \pmb \theta \right] \right] . } \end{array}\tag{24}
$$

The trace of its first part can be formulated as

$$
\begin{array} { r } { \mathrm { t r } \big ( \nabla _ { \pmb { \theta } } \phi \left( \pmb { \theta } \right) \big ) = \langle \nabla _ { \pmb { \theta } } \kappa \left( \pmb { \theta } , \cdot \right) , \phi \left( \cdot \right) \rangle _ { \mathcal { H } _ { \kappa } ^ { d } } . } \end{array}\tag{25}
$$

For the remaining part, let us denote ${ \pmb { \alpha } } = { \pmb { \theta } } { \pmb { \theta } } ^ { T }$ and $\delta _ { p , q } \left( \theta \right) =$ $\textstyle \sum _ { k } \gamma _ { p k } \phi _ { k q } ( \pmb \theta )$ . We flatten the matrix $\alpha \phi \left( \theta \right)$ into a vector of $k \times r$ dimensions. The trace of its derivative becomes

$$
\begin{array} { r l } { \displaystyle \sum _ { y ^ { \prime } \neq z } \nabla _ { y ^ { \prime } , z } \delta _ { \alpha \epsilon } ( \theta ) = \sum _ { \nu ^ { \prime } \neq z } \sum _ { \nu ^ { \prime } \neq z } \nabla _ { \alpha ^ { \prime } , \alpha ^ { \prime } \neq \nu } ( \theta ) } \\ { = \displaystyle \sum _ { x , x } \sum _ { \nu ^ { \prime } \neq z } \nabla _ { \alpha ^ { \prime } , \alpha ^ { \prime } \neq \nu } ( \theta ) } \\ { =  \sum _ { \nu ^ { \prime } \neq z } \sum _ { \nu ^ { \prime } \neq z } \nabla _ { \alpha ^ { \prime } , \alpha ^ { \prime } \neq \nu } ( \theta ) , \ i \omega _ { \alpha ^ { \prime } } ( z )  _ { \nu , \alpha } } \\ { =  \sum _ { \nu ^ { \prime } \neq z } \sum _ { \nu ^ { \prime } \neq z } \nabla _ { \alpha ^ { \prime } , \alpha ^ { \prime } \neq \nu ^ { \prime } } ( \theta ) , \ i \omega _ { \alpha ^ { \prime } } ( z )  _ { \nu , \alpha } } \\ { =  \sum _ { \nu ^ { \prime } \neq z } \sum _ { \nu ^ { \prime } \neq z } \nabla _ { \alpha ^ { \prime } , \alpha ^ { \prime } \neq \nu } ( \theta ) , \ i \omega _ { \alpha ^ { \prime } } ( z )  _ { \nu , \alpha } } \\ { =  \sum _ { \nu ^ { \prime } \neq z } \sum _ { \nu ^ { \prime } \neq z } \nabla _ { \alpha ^ { \prime } , \alpha ^ { \prime } \neq \nu } ( \theta ) , \ i \omega _ { \alpha ^ { \prime } } ( z )  _ { \nu , \alpha } } \\ { = \displaystyle \sum _ { y ^ { \prime } \neq z }  \sum _ { \nu ^ { \prime } \neq z } \nabla _ { \alpha ^ { \prime } , \alpha ^ { \prime } \neq \nu ^ { \prime } } ( \theta ) , \ i \omega _ { \alpha ^ { \prime } } ( z )  _ { \nu , \alpha } } \\  = \displaystyle \sum _ { \nu ^ { \prime } \neq z }  \sum _  \nu ^  \prime \end{array}
$$

Let us denote $\begin{array} { r } { \lambda _ { p q } \left( \pmb { \theta } \right) = \sum _ { k , t } \theta _ { p k } \phi _ { t k } \left( \pmb { \theta } \right) \theta _ { t q } } \end{array}$ , the trace of its derivative can be formulated as

$$
\begin{array} { r l } { \displaystyle \sum _ { p , q } \nabla _ { \theta _ { p q } \lambda _ { p q } } \left( \theta \right) = \displaystyle \sum _ { p , q } \displaystyle \sum _ { k , t } \theta _ { p k } \nabla _ { \theta _ { p q } } \phi _ { t k } \left( \theta \right) \theta _ { t q } } & { } \\ { = \displaystyle \sum _ { t , k } \displaystyle \sum _ { p , q } \theta _ { t q } \nabla _ { \theta _ { t k } } \phi _ { p q } \left( \theta \right) \theta _ { p k } } & { } \\ { = \displaystyle \sum _ { p , q } \left. \displaystyle \sum _ { t , k } \theta _ { t q } \nabla _ { \theta _ { t k } } \kappa \left( \theta , \cdot \right) \theta _ { p k } , \phi \left( \cdot \right) \right. _ { \mathscr { H } _ { s } } } & { } \\ { = \displaystyle \left. \theta \nabla _ { \theta ^ { K } } \left( \theta , \cdot \right) ^ { \top } \theta , \phi \left( \cdot \right) \right. _ { \mathscr { H } _ { d } } . } & { \quad { ( 2 7 } } \end{array}
$$

Combining (25)–(27) yields the trace of (24) as follow

$$
\begin{array} { r l r } & { \mathrm { t r } \left( \nabla _ { \theta } \mathrm { P r o j } _ { \theta } \left( \phi \left( \theta \right) \right) \right) } & \\ & { = \left. \nabla _ { \theta ^ { K } } \left( \theta , \cdot \right) - \frac { 1 } { 2 } \theta \left[ \theta ^ { \top } \nabla _ { \theta ^ { K } } \left( \theta , \cdot \right) + \nabla _ { \theta ^ { K } } \left( \theta , \cdot \right) ^ { \top } \theta \right] , \phi \left( \cdot \right) \right. } & \\ & { = \langle \mathrm { P r o j } _ { \theta } \left( \nabla _ { \theta ^ { K } } \left( \theta , \cdot \right) \right) , \phi \left( \cdot \right) \rangle _ { \mathcal { H } _ { \kappa } ^ { d } } . } & { ( 2 8 ) } \end{array}\tag{H<sup>d</sup><sub>κ</sub>}
$$

Finally, from (23) and (28), we reach the following form of (18):

$$
 \frac { \partial } { \partial \eta } \mathcal { F } ( \rho _ { t } ^ { [ T ] } )  _ { \eta = 0 } = \int  [ \beta \kappa ( \pmb { \theta } , \cdot ) \mathrm { P r o j } _ { \pmb { \theta } } ( \nabla _ { \pmb { \theta } } \Psi ( \pmb { \theta } ) ) 
$$

$$
- \left. \mathrm { P r o j } _ { \pmb { \theta } } \left( \nabla _ { \pmb { \theta } } \kappa \left( \pmb { \theta } , \cdot \right) \right) \right] , \phi \left( \cdot \right) \rangle _ { \mathcal { H } _ { \kappa } ^ { d } } d \rho _ { t } .\tag{29}
$$

This concludes our proof as the rest is obvious.

## APPENDIX B ADDITIONAL EXPERIMENTAL RESULTS

## A. Runtime

Table IV presents the estimated runtime (both training and evaluation) for all methods on the Winogrande-Small (WG-S) [26] dataset. From these results, it is apparent that modelbased variational inference baselines, Laplace-LoRA, BLoB, and ScalaBL, have an advantage in their scalability due to their inherent design. Deep Ensemble, SVGD, and StePS, require substantially longer runtime due to the need of evaluating multiple solutions in every forward pass. In addition, geometryaware optimizations on the Stiefel manifold demand more computation with the tangent-space projection and retraction. However, these extra costs are minimal compared to the costs of ensemble evaluations.

TABLE IV  
ESTIMATED RUNTIME COMPARISON ON WG-S DATASET.
<table><tr><td>Adapter</td><td>Method</td><td>Runtime (h)</td></tr><tr><td rowspan="6">LoRA</td><td>MLE/MAP/MCD</td><td>1.4</td></tr><tr><td>Ensemble</td><td>10.7</td></tr><tr><td>Laplace</td><td>1.7</td></tr><tr><td>BLoB</td><td>2.8</td></tr><tr><td>ScalaBL</td><td>1.6</td></tr><tr><td>SVGD</td><td>9.1</td></tr><tr><td rowspan="3">StelLA</td><td>MLE/MAP/MCD</td><td>1.7</td></tr><tr><td>Ensemble</td><td>11.1</td></tr><tr><td>StePS (Ours)</td><td>10.5</td></tr></table>

## B. Ablation Study on the Number of Particles

We train StePS on the Winogrande-Small (WG-S) [26] dataset for 4 epochs with $M \ = \ 1 , 2 , 4 , 6 ,$ and 8 particles to examine the effect of number of particles on the model performance. The accuracy, ECE, NLL, and estimated runtime results over 3 random restarts are presented in Table V. From these results, it is apparent that the accuracy of the model increases with more particles involved in the approximation. However, starting from 6 particles, the ECE scores also begin to increase with the NLL values slightly fluctuating. Overall, $M \ = \ 4$ achieves the best trade-off between accuracy and calibration. Additionally, using a lower number of particles also facilitates the training process with shorter runtime and reduced computational cost.

TABLE V  
ABLATION STUDY ON THE NUMBER OF PARTICLES ON WG-S DATASET.
<table><tr><td>M</td><td>Acc (↑)</td><td>ECE (↓)</td><td>NLL (↓)</td><td>Runtime (h)</td></tr><tr><td>1</td><td> $6 6 . 3 8 \pm 0 . 5 8$ </td><td> $2 0 . 4 9 \pm 1 . 7 4$ </td><td> $0 . 8 8 \pm 0 . 0 8$ </td><td>0.47</td></tr><tr><td>2</td><td> $6 8 . 8 8 \pm 0 . 7 8$ </td><td> $1 3 . 1 3 \pm 0 . 6 1$ </td><td> $0 . 7 0 \pm 0 . 0 1$ </td><td>0.93</td></tr><tr><td>4</td><td> $6 9 . 2 2 \pm 0 . 8 4$ </td><td> $1 1 . 9 1 \pm 0 . 2 7$ </td><td> $0 . 6 6 \pm 0 . 0 1$ </td><td>1.55</td></tr><tr><td>6</td><td> $7 0 . 0 2 \pm 1 . 2 5$ </td><td> $1 1 . 9 2 \pm 0 . 8 8$ </td><td> $0 . 6 5 \pm 0 . 0 1$ </td><td>2.39</td></tr><tr><td>8</td><td> $7 0 . 6 0 \pm 0 . 2 1$ </td><td> $1 2 . 4 8 \pm 0 . 4 2$ </td><td> $0 . 6 6 \pm 0 . 0 1$ </td><td>3.10</td></tr></table>