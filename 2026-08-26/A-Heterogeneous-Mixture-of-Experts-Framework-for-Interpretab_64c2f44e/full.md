# A Heterogeneous Mixture of Experts Framework for Interpretable Machine Learning

Soham Chatterjee<sup>a,∗</sup>, Rwitobroto Dey<sup>a</sup>, Smarajit Bose<sup>b</sup>

<sup>a</sup>Indian Statistical Institute, 203 B.T. Road, Kolkata, 700108, India

<sup>b</sup>Interdisciplinary Statistical Research Unit, Indian Statistical Institute, 203 B.T. Road, Kolkata, 700108, India

## Abstract

Mixture-of-Experts (MoE) models provide a flexible framework for partitioning complex prediction problem into simpler local learning tasks through an input-dependent gating mechanism. Existing interpretable MoE approaches, such as Mixture of Decision Trees (MoDT), achieve transparency by employing homogeneous decision-tree experts, but this restricts the model to a single inductive bias across all regions of the feature space. We extend the MoDT framework by introducing heterogeneous expert families comprising decision trees, linear support vector machines, and quadratic discriminant analysis under a common probabilistic gating mechanism. To ensure coherent likelihood-based inference, non-probabilistic experts are calibrated to produce conditional class probabilities, allowing parameter estimation within the generalized Expectation-Maximization framework of MoDT. We further establish theoretical monotone ascent guarantees for the proposed heterogeneous gating updates, providing a justification for the optimization procedure. Experiments on a diverse collection of synthetic and real-world benchmark datasets demonstrate that the proposed framework adaptively specializes experts according to local data geometry, yielding interpretable expert assignments while achieving predictive performance competitive with homogeneous MoDT and Random Forests. The proposed approach combines interpretability, adaptive inductive bias selection, and probabilistic coherence within a unified mixture-of-experts framework.

Keywords: mixture of experts, interpretable machine learning, decision trees, expectation–maximization, heterogeneous experts, gating mechanism

## 1. Introduction

Mixture-of-experts (MoE) models provide a principled framework for combining multiple predictive models through an input-dependent gating mechanism that dynamically allocates observations to specialized experts. Introduced by Jacobs et al. [1] and subsequently generalized through hierarchical formulations by Jordan and Jacobs [2], MoE architectures decompose complex prediction problems into simpler local subproblems, allowing diferent experts to specialize in diferent regions of the input space. This adaptive specialization mechanism offers an appealing compromise between model flexibility and statistical eficiency, and has become an influential paradigm in machine learning, ensemble learning, and probabilistic modeling [3–6].

From a statistical learning perspective, the success of MoE models stems from their ability to exploit hetero geneous structure within the data. Rather than relying on a single global hypothesis class, the gating mechanism allows diferent experts to dominate in regions where their inductive biases are most appropriate. This idea is closely aligned with a central principle of statistical learning theory: no single model class is uniformly optima across all learning problems, and predictive performance depends critically on matching model assumptions to the underlying data structure [7–10].

Despite their flexibility, classical MoE models have often faced criticism regarding interpretability. Many modern MoE architectures employ highly expressive neural-network gating mechanisms and complex expert models whose decision processes are dificult to understand, reflecting a broader challenge in interpretable machine learning [11–13]. This challenge has motivated a growing body of research on interpretable machine learning and interpretable mixtures of experts[14, 15]. Examples include MOET [16], which employs tree-based experts in reinforcement learning, Interpretable Mixture of Experts [17], which seeks globally understandable expert decompositions, InterpretCC [18], which focuses on user-centric interpretability, and recent intrinsically interpretable MoE architectures [19]. These works demonstrate increasing interest in preserving the adaptive specialization advantages of MoE models while maintaining transparency and explainability.

Among existing interpretable MoE approaches, the Mixture of Decision Trees (MoDT) framework proposed by Brüggenjürgen et al. [20] is particularly attractive. MoDT combines a linear softmax gating mechanism with shallow decision-tree experts, thereby preserving the probabilistic structure of classical MoE models while providing interpretable rule-based decision paths. By restricting experts to shallow trees, MoDT ofers localized explanations and transparent expert assignments, making it especially suitable for scientific and tabular-data applications where interpretability is a primary concern.

However, despite these advantages, MoDT inherits an important structural limitation from its design: all experts belong to the same hypothesis class. Consequently, every region of the feature space is modeled using decision trees, regardless of whether an alternative model family would provide a more natural representation. While decision trees are highly interpretable and flexible, their axis-aligned partition structure may be ineficient for representing smooth linear separators, elliptical decision boundaries, anisotropic Gaussian classconditionals, or other geometric structures that arise naturally in many real-world datasets [21–23].

From a statistical perspective, this restriction may be viewed as a form of local model mismatch. Although the gating mechanism partitions the input space adaptively, the set of admissible local models remains globally fixed. Consequently, adaptation occurs only through parameter variation within a single hypothesis class rather than through the selection of fundamentally diferent inductive biases. In practice, this often necessitates increasingly complex tree structures to approximate decision boundaries that could be represented more naturally by alternative model classes such as support vector machines [24] or quadratic discriminant models [7].

This observation motivates the central idea of the present work. Many datasets exhibit heterogeneous local structure in which diferent regions are more naturally represented by diferent model families. For example, one region may be approximately linearly separable and therefore well modeled by a support vector machine, while another may exhibit covariance-driven quadratic boundaries better captured by quadratic discriminant analysis, and yet another may require rule-based axis-aligned partitions naturally represented by decision trees.

The original MoE framework already provides a principled mechanism for routing observations toward the most appropriate expert through responsibility-based gating [1, 2]. Extending this framework to allow experts with heterogeneous inductive biases is therefore both natural and statistically appealing.

Building on the interpretability of MoDT and the probabilistic foundations of classical MoE models, we propose a heterogeneous mixture-of-experts framework that integrates decision trees, linear support vector machines, and quadratic discriminant analysis experts under a common linear softmax gating mechanism. To ensure coherent probabilistic inference, expert outputs are represented as conditional class probabilities and parameter estimation is performed through a generalized Expectation–Maximization procedure [25, 26]. Unlike existing interpretable MoE approaches that rely on a single expert family, the proposed framework enables adaptive region-specific inductive bias selection while preserving transparency in both expert assignment and prediction.

The contributions of this work are threefold. First, we introduce an interpretable heterogeneous mixtureof-experts architecture that combines complementary expert families within a unified probabilistic framework. Second, we develop a generalized EM estimation procedure together with theoretical results establishing monotone ascent properties of the proposed gating updates. Third, through experiments on a collection of real-world and synthetic datasets, we demonstrate that the proposed framework achieves predictive performance comparable to strong baselines while providing meaningful expert specialization patterns that reveal how diferent inductive biases are selected across regions of the feature space.

## 2. Problem Formulation and Discussion

We consider a supervised multi-class classification problem in which we observe independent and identically distributed samples

$$
\mathcal { D } = \{ ( \boldsymbol { x } _ { i } , \boldsymbol { y } _ { i } ) \} _ { i = 1 } ^ { n }
$$

drawn from an unknown joint distribution P(X Y) defined on the product space $\chi _ { \times } y$ . Here $\chi \subset \mathbb { R } ^ { p + 1 }$ denotes the feature space, where each $\mathbf { \nabla } _ { \mathbf { \boldsymbol { x } } _ { i } }$ is augmented with an intercept term, and $y = \{ 1 , \ldots , C \}$ denotes a finite label space.

Our objective is to construct an interpretable estimator of the conditional distribution $P ( Y \mid x )$ that generalizes to unseen data. We assume that the true conditional mechanism exhibits region-specific structural heterogeneity, in the sense that no single hypothesis class provides a globally parsimonious representation of the Bayes decision boundary.

Specifically, we posit the existence of a latent categorical variable $Z \in \{ 1 , \ldots , e \}$ such that the conditional distribution admits the representation

$$
P ( Y = y \mid { \pmb x } ) = \sum _ { j = 1 } ^ { e } P ( Z = j \mid { \pmb x } ) P ( Y = y \mid { \pmb x } , Z = j ) .\tag{1}
$$

This formulation induces a soft partition of the input space, allowing distinct conditional mechanisms to dominate in diferent regions.

Classical MoE models [1] operationalize (1) using parametric gating functions and homogeneous expert families. The MoDT framework utilizes a linear gating function and constrains experts to interpretable treebased components, thereby enhancing transparency. However, restricting all experts to a single hypothesis class imposes a common inductive bias across regions of the feature space. When the underlying conditional structure deviates from axis-aligned partitions, this constraint may induce regional model misspecification, forcing adaptation through increased model complexity rather than appropriate bias selection.

The statistical problem addressed in this work is therefore the following: can one construct a coherent and interpretable mixture estimator that permits heterogeneous expert families while preserving likelihood-based inference and stable specialization behaviour?

To this end, we define a heterogeneous model space

$$
\mathcal { F } = \bigcup _ { j = 1 } ^ { e } \mathcal { F } _ { j } ,
$$

where each $\mathcal { F } _ { j }$ denotes a distinct hypothesis class with complementary inductive bias. In the present work, the expert families include: (i) shallow decision trees for rule-based axis-aligned structure, (ii) linear support vector machines for maximum-margin hyperplanar separation, and (iii) quadratic discriminant models for classconditional covariance structure.

Let Θ denote the parameters of a linear softmax gating mechanism

$$
g _ { j } ( \pmb { x } ; \mathbf { \Theta } \Theta ) = \frac { \exp ( \pmb { x } ^ { \top } \pmb { \theta } _ { j } ) } { \sum _ { k = 1 } ^ { e } \exp ( \pmb { x } ^ { \top } \pmb { \theta } _ { k } ) } ,
$$

and let $\Psi = ( \psi _ { 1 } , \ldots , \psi _ { e } )$ denote the collection of expert-specific parameters. The observed-data likelihood under the heterogeneous mixture model is

$$
\ell ( \Theta , \Psi ) = \sum _ { i = 1 } ^ { n } \log \Biggl ( \sum _ { j = 1 } ^ { e } g _ { j } ( x _ { i } ; \Theta ) p _ { j } ( y _ { i } \mid x _ { i } ; \psi _ { j } ) \Biggr ) ,\tag{2}
$$

where $p _ { j } ( y \mid \pmb { x } ; \psi _ { j } )$ denotes the conditional density associated with expert j.

For expert classes that do not inherently produce probabilistic outputs $( \mathrm { e . g . }$ , margin-based linear support vector machines), we employ calibrated mappings from decision scores to conditional class probabilities (e.g., via Platt scaling)[27–29] to ensure that (2) defines a coherent likelihood function. This guarantees compatibility with likelihood-based estimation consistent with modern probabilistic machine learning formulations [30] and preserves probabilistic interpretability of the mixture.

Parameter estimation proceeds via a generalized Expectation Maximization [26] procedure that iteratively maximizes (2). The resulting estimator $( \hat { \mathbf { \Theta } } _ { } \hat { \mathbf { \Xi } } _ { } ^ { \hat { \Psi } } )$ is designed to achieve region-adaptive inductive bias selection: in regions where a single hypothesis class is suficient, the gating mechanism may concentrate its mass on that expert, whereas in structurally heterogeneous regimes, responsibility is distributed across complementary experts.

The goal is thus to reconcile interpretability, coherence, and heterogeneous inductive bias within a unified likelihood-based framework.

## 3. Model Specification

Let $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { n } \in \mathbb { R } ^ { p + 1 }$ denote the observed covariate vectors, treated as fixed and non-random. Each vector is written as $\pmb { x } _ { i } = ( 1 , z _ { i } ^ { \top } ) ^ { \top }$ , where the first coordinate corresponds to an intercept term and $z _ { i } \in \mathbb { R } ^ { p }$ denotes the non-intercept covariates. Define the design matrix

$$
\begin{array} { r } { X = \left[ \begin{array} { l } { \pmb { x } _ { 1 } ^ { \top } } \\ { \vdots } \\ { \qquad \pmb { x } _ { n } ^ { \top } } \end{array} \right] \in \mathbb { R } ^ { n \times ( p + 1 ) } . } \end{array}
$$

Throughout this section we assume that

$$
\operatorname { r a n k } ( X ) = p + 1 .
$$

This condition is imposed to ensure injectivity of the normalized softmax gating parametrization and to avoid degeneracy arising from linear dependence among covariates.

## 3.1. Parameter Space

We assume that the heterogeneous mixture model is indexed by (Θ Ψ), where $\boldsymbol { \Theta } = ( \pmb { \theta } _ { 1 } , \dots , \pmb { \theta } _ { e } )$ denotes the gating parameters and $\Psi = ( \pmb { \psi } _ { 1 } , \dots , \pmb { \psi } _ { e } )$ denotes the collection of expert-specific parameters.

The gating parameters satisfy

$$
\begin{array} { r } { \pmb { \theta } _ { j } \in \mathbb { R } ^ { p + 1 } , \qquad j = 1 , \dotsc , e . } \end{array}
$$

As established below in Section 4, the softmax parametrization exhibits an additive redundancy; after normalization, the free gating parameter space is identified with R<sup>(p+1)(e−1)</sup>.

Further, it is assumed that each expert j belongs to a finite-dimensional parametric family indexed by

$$
\pmb { \psi } _ { j } \in \Psi _ { j } \subset \mathbb { R } ^ { d _ { j } } ,
$$

where $d _ { j } < \infty$ denotes the dimension of the jth expert model. This finite-dimensional Euclidean representation reflects the parametric nature of the expert families. The assumption $\pmb { \psi } _ { j } \in \mathbb { R } ^ { d _ { j } }$ ensures that each expert family admits a smooth coordinate representation suitable for likelihood-based analysis.

We assume that the overall parameter space

$$
\pmb { \Omega } = \{ ( \Theta , \Psi ) : \pmb { \theta } _ { j } \in \mathbb { R } ^ { p + 1 } , \pmb { \psi } _ { j } \in \pmb { \Psi } _ { j } \}
$$

is compact. Combined with continuity of the log-likelihood, this guarantees existence of maximizers of the observed-data likelihood, and that the target parameter lies in the interior of Ω.

## 4. Estimation via Expectation–Maximization

In Section 2, the heterogeneous mixture model was specified through the conditional density

$$
p ( y _ { i } \mid x _ { i } ; \Theta , \Psi ) = \sum _ { j = 1 } ^ { e } g _ { j } ( x _ { i } ; \Theta ) p _ { j } ( y _ { i } \mid x _ { i } ; \psi _ { j } ) ,\tag{3}
$$

where $g _ { j } ( \pmb { x } _ { i } ; \pmb { \Theta } )$ denotes the softmax gating probability and $p _ { j } ( y _ { i } \mid x _ { i } ; \psi _ { j } )$ denotes the conditional density associated with expert j.

Direct maximization of the observed-data log-likelihood

$$
\ell ( \Theta , \Psi ) = \sum _ { i = 1 } ^ { n } \log \left( \sum _ { j = 1 } ^ { e } g _ { j } ( x _ { i } ; \Theta ) p _ { j } ( y _ { i } \mid x _ { i } ; \psi _ { j } ) \right)\tag{4}
$$

is analytically intractable due to the coupling between the gating and expert parameters induced by the logarithm of the mixture sum.

To facilitate optimization, we introduce latent allocation variables

$$
Z _ { i } \in \{ 1 , \ldots , e \} ,
$$

where $Z _ { i } = j$ indicates that observation i is generated by expert j. This latent-variable representation naturally leads to an Expectation–Maximization (EM) framework [25, 31].

The proposed estimation procedure combines:

1. a fully probabilistic E-step derived from the heterogeneous mixture likelihood, and

2. a computationally eficient regression-based generalized M-step inspired by the Mixture of Decision Trees framework.

The resulting algorithm therefore constitutes a generalized Expectation–Maximization (GEM) procedure.

## 4.1. Complete-Data Representation

If the latent variables $Z _ { 1 } , \ldots , Z _ { n }$ were observed, the complete-data likelihood would factorize as

$$
L _ { c } ( \boldsymbol { \Theta } , \boldsymbol { \Psi } ) = \prod _ { i = 1 } ^ { n } \prod _ { j = 1 } ^ { e } \left[ g _ { j } ( x _ { i } ; \boldsymbol { \Theta } ) p _ { j } ( y _ { i } | x _ { i } ; \psi _ { j } ) \right] ^ { \mathbb { 1 } ( Z _ { i } = j ) } ,
$$

where 1(·) denotes the indicator function.

Taking logarithms yields the complete-data log-likelihood:

$$
\ell _ { c } ( \boldsymbol { \Theta } , \boldsymbol { \Psi } ) = \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { e } \mathbb { 1 } ( Z _ { i } = j ) \Big [ \log g _ { j } ( \boldsymbol { x } _ { i } ; \boldsymbol { \Theta } ) + \log p _ { j } ( y _ { i } \mid \boldsymbol { x } _ { i } ; \boldsymbol { \psi } _ { j } ) \Big ] .\tag{5}
$$

Because the latent allocations are unobserved, EM proceeds by iteratively replacing the indicator variables by their posterior conditional expectations.

## 4.2. E-Step

At iteration $t ,$ let $( \mathbf { \Theta } \mathbf { \Theta } ^ { ( t ) } , \Psi ^ { ( t ) } )$ denote the current parameter estimates. The E-step computes the posterior distribution of the latent allocation variables conditioned on the observed data:

$$
\tau _ { i j } ^ { ( t ) } = P ( Z _ { i } = j \mid \boldsymbol { y } _ { i } , \boldsymbol { x } _ { i } ; \mathbf { \Theta } ^ { ( t ) } , \boldsymbol { \Psi } ^ { ( t ) } ) ,
$$

where i ranges from 1 to n and j varies from 1 to $e .$

Applying Bayes’ theorem yields

$$
\tau _ { i j } ^ { ( t ) } = \frac { g _ { j } ( \pmb { x } _ { i } ; \pmb { \Theta } ^ { ( t ) } ) p _ { j } ( y _ { i } \mid \pmb { x } _ { i } ; \pmb { \psi } _ { j } ^ { ( t ) } ) } { \sum _ { k = 1 } ^ { e } g _ { k } ( \pmb { x } _ { i } ; \pmb { \Theta } ^ { ( t ) } ) p _ { k } ( y _ { i } \mid \pmb { x } _ { i } ; \pmb { \psi } _ { k } ^ { ( t ) } ) } .\tag{6}
$$

The quantities $\tau _ { i j } ^ { ( t ) }$ are referred to as posterior responsibilities and satisfy

$$
0 \leq \tau _ { i j } ^ { ( t ) } \leq 1 , \qquad \sum _ { j = 1 } ^ { e } \tau _ { i j } ^ { ( t ) } = 1 .
$$

Unlike heuristic confidence-based responsibility assignments, these posterior probabilities arise directly from the latent-variable mixture likelihood and therefore possess a rigorous probabilistic interpretation.

Define the posterior responsibility matrix

$$
\pmb { T } ^ { ( t ) } = ( \tau _ { i j } ^ { ( t ) } ) _ { i , j } \in \mathbb { R } ^ { n \times e } .
$$

The auxiliary EM functional is then given by

$$
{ \mathcal { Q } } ( \Theta , \Psi \mid \Theta ^ { ( t ) } , \Psi ^ { ( t ) } ) = \mathbb { E } \Big [ \ell _ { c } ( \Theta , \Psi ) \mid { \mathcal { D } } ; \Theta ^ { ( t ) } , \Psi ^ { ( t ) } \Big ] .
$$

Substituting the complete-data log-likelihood yields

$$
\mathcal { Q } ( \boldsymbol { \Theta } , \Psi ) = \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { e } \tau _ { i j } ^ { ( t ) } \Big [ \log g _ { j } ( \boldsymbol { x } _ { i } ; \boldsymbol { \Theta } ) + \log p _ { j } ( y _ { i } \mid \boldsymbol { x } _ { i } ; \psi _ { j } ) \Big ] .\tag{7}
$$

The auxiliary function separates additively into gating-specific and expert-specific components:

$$
\mathcal { Q } ( \Theta , \Psi ) = \mathcal { Q } _ { \mathrm { g a t e } } ( \Theta ) + \sum _ { j = 1 } ^ { e } \mathcal { Q } _ { j } ( \psi _ { j } ) ,
$$

where

$$
Q _ { \mathrm { g a t e } } ( \Theta ) = \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { e } \tau _ { i j } ^ { ( t ) } \log g _ { j } ( x _ { i } ; \Theta ) ,\tag{8}
$$

and

$$
Q _ { j } ( \pmb { \psi } _ { j } ) = \sum _ { i = 1 } ^ { n } \tau _ { i j } ^ { ( t ) } \log p _ { j } ( y _ { i } \mid \pmb { x } _ { i } ; \pmb { \psi } _ { j } ) .\tag{9}
$$

This separability permits independent optimization of the experts and the gating mechanism.

Since exact maximization of the gating likelihood requires iterative multinomial logistic optimization, we instead adopt the regression-based surrogate update proposed in MoDT [20]. Consequently, the resulting procedure is a generalized EM (GEM) algorithm rather than an exact EM algorithm.

## 4.3. M-Step

In the M-step, the auxiliary function is optimized with respect to the parameters $( \Theta , \Psi )$

## 4.3.1. Expert Updates

For each expert j, the optimization problem reduces to

$$
\widehat { \psi } _ { j } ^ { ( t + 1 ) } = \arg \operatorname* { m a x } _ { \psi _ { j } \in \Psi _ { j } } \sum _ { i = 1 } ^ { n } \tau _ { i j } ^ { ( t ) } \log p _ { j } ( y _ { i } \mid \pmb { x } _ { i } ; \psi _ { j } ) .\tag{10}
$$

Thus, each expert update corresponds to a weighted maximum likelihood estimation problem with observation weights given by the posterior responsibilities.

Accordingly:

• decision tree experts are trained using weighted impurity minimization,

• support vector machine experts are trained using weighted margin optimization together with probability calibration,

• quadratic discriminant experts are estimated using weighted class means and covariance matrices.

Since the expert objectives are mutually independent, all experts may be optimized separately and in parallel.

## 4.3.2. Regression-Based Gating Update

The gating objective is

$$
Q _ { \mathrm { g a t e } } ( \Theta ) = \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { e } \tau _ { i j } ^ { ( t ) } \log g _ { j } ( x _ { i } ; \Theta ) .\tag{11}
$$

Exact maximization of this objective corresponds to multinomial logistic regression with soft labels $\tau _ { i j } ^ { ( t ) }$ Although the resulting optimization problem is convex, it does not admit a closed-form solution and typically requires iterative numerical procedures such as IRLS or quasi-Newton optimization at every EM iteration.

To avoid this repeated high-dimensional optimization, we follow the computational strategy adopted in the MoDT [20] framework and replace the exact multinomial likelihood maximization by a least-squares surrogate update. Specifically, rather than directly optimizing the softmax log-likelihood, we model the discrepancy between the posterior responsibilities and the current gating assignments through a linear regression approximation.

This surrogate formulation yields a computationally eficient closed-form update while preserving the directional alignment between the posterior allocation structure obtained in the E-step and the gating probabilities induced by the current parameter estimates.

Define the current gating probability matrix

$$
\begin{array} { r } { \pmb { G } ^ { ( t ) } = \big ( g _ { j } ( \pmb { x } _ { i } ; \pmb { \Theta } ^ { ( t ) } ) \big ) _ { i , j } \in \mathbb { R } ^ { n \times e } . } \end{array}
$$

We define the residual responsibility matrix

$$
\pmb { R } ^ { ( t ) } = \pmb { T } ^ { ( t ) } - \pmb { G } ^ { ( t ) } .\tag{12}
$$

The matrix $\pmb { R } ^ { ( t ) }$ measures the discrepancy between the posterior expert allocations implied by the complete  
data likelihood, and the current gating allocations induced by the softmax gating network. Positive entries   
$R _ { i j } ^ { ( t ) } > 0$ indicate under-allocation of observation i to expert $j ,$ while negative entries indicate over-allocation. Let

$$
\begin{array} { r } { X = \left[ { \begin{array} { c } { \mathbf { x } _ { 1 } ^ { \top } } \\ { \vdots } \\ { \mathbf { x } _ { n } ^ { \top } } \end{array} } \right] \in \mathbb { R } ^ { n \times ( p + 1 ) } } \end{array}
$$

denote the design matrix. We seek a linear perturbation of the gating parameters that approximately aligns the gating probabilities with the posterior responsibilities. As in MoDT [20], this is achieved by solving the multivariate least-squares problem

$$
\widehat { \pmb { B } } ^ { ( t ) } = \arg \operatorname* { m i n } _ { \pmb { B } \in \mathbb { R } ^ { ( p + 1 ) \times e } } \left\| \pmb { R } ^ { ( t ) } - \pmb { X } \pmb { B } \right\| _ { \mathcal { F } } ^ { 2 } ,\tag{13}
$$

where $\| \cdot \| _ { \mathcal { F } }$ denotes the Frobenius norm. Equivalently,

$$
\widehat { \pmb { B } } ^ { ( t ) } = \arg \operatorname* { m i n } _ { { \pmb { B } } } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { e } \left( \tau _ { i j } ^ { ( t ) } - g _ { j } ( \pmb { x } _ { i } ; \pmb { \Theta } ^ { ( t ) } ) - x _ { i } ^ { \top } { \pmb { \beta } } _ { j } \right) ^ { 2 } .
$$

Assuming $X ^ { \top } X$ is nonsingular, the minimizer admits the closed-form expression

$$
\widehat { \pmb { B } } ^ { ( t ) } = ( \pmb { X } ^ { \top } \pmb { X } ) ^ { - 1 } \pmb { X } ^ { \top } \pmb { R } ^ { ( t ) } .\tag{14}
$$

The gating parameters are then updated according to

$$
\mathbf { \Theta } ^ { ( t + 1 ) } = \mathbf { \Theta } ^ { ( t ) } + \gamma _ { t } \widehat { \mathbf { B } } ^ { ( t ) } ,\tag{15}
$$

where $\gamma _ { t } > 0$ denotes a learning-rate parameter. This update may be interpreted as a first-order correction step that moves the gating probabilities toward the posterior allocation structure computed in the E-step.

## 4.4. Local Generalized EM Interpretation

The proposed gating update does not maximize $Q _ { \mathrm { g a t e } } ( \Theta )$ exactly. Instead, it performs an approximate ascent step based on the discrepancy between the posterior responsibilities obtained from the E-step and the current gating probabilities. Consequently, the resulting optimization procedure constitutes a generalized Expectation– Maximization (GEM) algorithm rather than an exact EM algorithm.

The following theorem establishes that the proposed least-squares gating update defines a monotone ascent step for the gating auxiliary objective under suitable learning-rate conditions.

## Theorem 1. Suppose that:

1. the design matrix satisfies rank $( X ) = p + 1$

2. the gating auxiliaryfunction

$$
Q _ { \mathrm { g a t e } } ( \Theta ) = \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { e } \tau _ { i j } ^ { ( t ) } \log g _ { j } ( x _ { i } ; \Theta )
$$

is twice continuously diferentiable.

Define $\pmb { R } ^ { ( t ) } = \pmb { T } ^ { ( t ) } - \pmb { G } ^ { ( t ) }$ and let

$$
\widehat { \pmb { B } } ^ { ( t ) } = ( \pmb { X } ^ { \top } \pmb { X } ) ^ { - 1 } \pmb { X } ^ { \top } \pmb { R } ^ { ( t ) } .
$$

Further define

$$
A _ { t } = \langle \nabla _ { \Theta } Q _ { \mathrm { g a t e } } ( \Theta ^ { ( t ) } ) , \widehat { \pmb { B } } ^ { ( t ) } \rangle
$$

and

$$
C _ { t } = \langle \widehat { \pmb { B } } ^ { ( t ) } , \nabla _ { \pmb { \Theta } } ^ { 2 } Q _ { \mathrm { g a t e } } ( \widetilde { \pmb { \Theta } } ^ { ( t ) } ) \widehat { \pmb { B } } ^ { ( t ) } \rangle ,
$$

where $\widetilde { \Theta } ^ { ( t ) } = \Theta ^ { ( t ) } + \eta _ { t } \gamma _ { t } \widehat { \pmb { B } } ^ { ( t ) }$ for some $\eta _ { t } \in ( 0 , 1 )$ .

Then:

1. $A _ { t } \geq 0 ;$ , with strict inequality whenever $\pmb { R } ^ { ( t ) }$ < $\operatorname { N u l l } ( P _ { X } )$ , where $P _ { X } = X ( X ^ { \top } X ) ^ { - 1 } X ^ { \top }$

2. the update $\mathbf { \Theta } ^ { ( t + 1 ) } = \mathbf { \Theta } ^ { ( t ) } + \gamma _ { t } \widehat { \mathbf { B } } ^ { ( t ) }$ satisfies $Q _ { \mathrm { g a t e } } ( \Theta ^ { ( t + 1 ) } ) > Q _ { \mathrm { g a t e } } ( \Theta ^ { ( t ) } )$ whenever $0 < \gamma _ { t } < 2 A _ { t } / | C _ { t } | .$

Proof. The softmax gating probabilities are given by

$$
g _ { j } ( \pmb { x } _ { i } ; \pmb { \Theta } ) = \frac { \exp ( \pmb { x } _ { i } ^ { \top } \pmb { \theta } _ { j } ) } { \sum _ { k = 1 } ^ { e } \exp ( \pmb { x } _ { i } ^ { \top } \pmb { \theta } _ { k } ) } .
$$

Therefore,

$$
\log g _ { j } ( \pmb { x } _ { i } ; \pmb { \Theta } ) = \pmb { x } _ { i } ^ { \top } \pmb { \theta } _ { j } - \log \left( \sum _ { k = 1 } ^ { e } \exp ( \pmb { x } _ { i } ^ { \top } \pmb { \theta } _ { k } ) \right) .
$$

Substituting into $Q _ { \mathrm { g a t e } }$ and using the identity $\begin{array} { r } { \sum _ { j = 1 } ^ { e } \tau _ { i j } ^ { ( t ) } = 1 } \end{array}$ yields

$$
Q _ { \mathrm { g a t e } } ( \Theta ) = \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { e } \tau _ { i j } ^ { ( t ) } \pmb { x } _ { i } ^ { \top } \pmb { \theta } _ { j } - \sum _ { i = 1 } ^ { n } \log \left( \sum _ { k = 1 } ^ { e } \exp ( \pmb { x } _ { i } ^ { \top } \pmb { \theta } _ { k } ) \right) .
$$

Diferentiating with respect to $\theta _ { j }$ gives

$$
\nabla _ { \pmb { \theta } _ { j } } \mathcal { Q } _ { \mathrm { g a t e } } ( \Theta ) = \sum _ { i = 1 } ^ { n } ( \tau _ { i j } ^ { ( t ) } - g _ { j } ( \pmb { x } _ { i } ; \Theta ) ) \pmb { x } _ { i } .
$$

Stacking the gradients columnwise gives

$$
\nabla _ { \Theta } \mathcal { Q } _ { \mathrm { g a t e } } ( \Theta ) = X ^ { \top } R ^ { ( t ) } .
$$

Using the definition of $\widehat { \pmb { B } } ^ { ( t ) }$

$$
A _ { t } = \mathrm { t r } \Big [ ( { \pmb R } ^ { ( t ) } ) ^ { \top } { \pmb P } _ { X } { \pmb R } ^ { ( t ) } \Big ] .
$$

Since $P _ { X }$ is symmetric positive semidefinite, $A _ { t } \ \geq \ 0 .$ , with equality if and only if ${ \pmb R } ^ { ( t ) } \in \mathrm { N u l l } ( { \pmb P } _ { X } )$ . This proves Part 1.

For the Hessian, diferentiating the softmax probabilities gives

$$
\frac { \partial g _ { i j } } { \partial \pmb { \theta } _ { \ell } } = g _ { i j } ( \delta _ { j \ell } - g _ { i \ell } ) \pmb { x } _ { i } ,
$$

where $\delta _ { j \ell }$ denotes the Kronecker delta. Define $\pmb { g } _ { i } = ( g _ { i 1 } , \ldots , g _ { i e } ) ^ { \top }$ and $\pmb { \Sigma } _ { i } = \mathrm { D i a g } ( \pmb { g } _ { i } ) - \pmb { g } _ { i } \pmb { g } _ { i } ^ { \top }$ . The Hessian may be written compactly as

$$
\nabla ^ { 2 } Q _ { \mathrm { g a t e } } = - \sum _ { i = 1 } ^ { n } [ \pmb { \Sigma } _ { i } \otimes ( \pmb { x } _ { i } \pmb { x } _ { i } ^ { \top } ) ] .
$$

This confirms $\nabla ^ { 2 } Q _ { \mathrm { g a t e } }$ is negative semi-definite, hence $C _ { t } \leq 0$

By the multivariate second-order Taylor expansion,

$$
\begin{array} { r } { Q _ { \mathrm { g a t e } } ( \Theta ^ { ( t + 1 ) } ) - Q _ { \mathrm { g a t e } } ( \Theta ^ { ( t ) } ) = \gamma _ { t } A _ { t } + \frac { \gamma _ { t } ^ { 2 } } { 2 } C _ { t } . } \end{array}
$$

Since $C _ { t } \leq 0 .$

$$
\begin{array} { r } { { \cal Q } _ { \mathrm { g a t e } } ( \Theta ^ { ( t + 1 ) } ) - { \cal Q } _ { \mathrm { g a t e } } ( \Theta ^ { ( t ) } ) \geq \gamma _ { t } \Big ( A _ { t } - \frac { \gamma _ { t } } { 2 } | C _ { t } | \Big ) > 0 } \end{array}
$$

whenever $0 < \gamma _ { t } < 2 A _ { t } / | C _ { t } |$ . This proves Part 2.

Theorem 2. Under the multinomial softmax gating parametrization,

$$
\frac { 2 A _ { t } } { | C _ { t } | } \geq 2 .
$$

Consequently, every learning-rate sequence satisfying $0 < \gamma _ { t } < 2$ guarantees monotone ascent of the gating auxiliary function:

$$
Q _ { \mathrm { g a t e } } ( \Theta ^ { ( t + 1 ) } ) \ge Q _ { \mathrm { g a t e } } ( \Theta ^ { ( t ) } )
$$

for all t. Hence, the resulting optimization procedure defines a valid generalized EM algorithm [32].

Proof. From Theorem 1,

$$
A _ { t } = \mathrm { t r } \Big [ ( { \pmb R } ^ { ( t ) } ) ^ { \top } { \pmb P } _ { X } { \pmb R } ^ { ( t ) } \Big ] = \| { \pmb P } _ { X } { \pmb R } ^ { ( t ) } \| _ { \mathcal { F } } ^ { 2 } = \| \widehat { \pmb X } \widehat { \pmb B } ^ { ( t ) } \| _ { \mathcal { F } } ^ { 2 } .
$$

To bound $| C _ { t } | ,$ we first show $\Sigma _ { i } \preceq I .$ For any $z \in \mathbb { R } ^ { e }$

$$
z ^ { \top } \Sigma _ { i } z = \sum _ { j = 1 } ^ { e } g _ { i j } z _ { j } ^ { 2 } - \left( \sum _ { j = 1 } ^ { e } g _ { i j } z _ { j } \right) ^ { 2 } \leq \sum _ { j = 1 } ^ { e } g _ { i j } z _ { j } ^ { 2 } \leq \sum _ { j = 1 } ^ { e } z _ { j } ^ { 2 } = \| z \| ^ { 2 } ,
$$

since $0 \leq g _ { i j } \leq 1$ for all $i , j .$ Hence $\Sigma _ { i } \preceq I .$

Setting $\pmb { \nu } _ { i } = \pmb { x } _ { i } ^ { \top } \widehat { \pmb { B } } ^ { ( t ) }$ and using the Kronecker representation of the Hessian gives

$$
| C _ { t } | = \sum _ { i = 1 } ^ { n } \pmb { \nu } _ { i } ^ { \top } \pmb { \Sigma } _ { i } \pmb { \nu } _ { i } \leq \sum _ { i = 1 } ^ { n } \| \pmb { \nu } _ { i } \| ^ { 2 } = \| \pmb { X } \pmb { \widehat { B } } ^ { ( t ) } \| _ { \mathcal { F } } ^ { 2 } = A _ { t } .
$$

Therefore $2 A _ { t } / | C _ { t } | \geq 2$ , and every $\gamma _ { t } \in ( 0 , 2 )$ satisfies the condition $0 < \gamma _ { t } < 2 A _ { t } / | C _ { t } |$ from Theorem 1.

The principal computational advantage of this formulation is that the gating update reduces to a simple closed-form multivariate regression problem, thereby avoiding repeated iterative optimization of the multinomial logistic objective. At the same time, the update remains directly driven by the discrepancy between the posterior responsibilities and the current gating probabilities, ensuring that the probabilistic structure learned during the E-step continues to guide the parameter updates. Moreover, Theorem 1 shows that the resulting least squares correction acts as a local ascent step for the gating auxiliary function for suficiently small learning rates, thereby providing the proposed procedure with a generalized EM interpretation [25].

The complete training procedure of the proposed heterogeneous mixture-of-experts framework is summarized in Algorithm 1. The algorithm alternates between the computation of posterior responsibilities in the E-step and the estimation of the expert and gating parameters in the M-step until convergence. The regressionbased gating update described in Section 4 is incorporated directly into the iterative optimization procedure.

## 5. Application

In this section the overall classification accuracy performance of the heterogeneous mixture model is evaluated and compared with that of the MoDT architecture. The heterogeneous mixture model is further contrasted with non-interpretable ML techniques such as Random Forest.

Experiments are conducted on several publicly available benchmark datasets to evaluate the proposed heterogeneous mixture framework. Categorical features are one-hot encoded if necessary to make them compatible with the underlying expert models. Unless otherwise stated, all experiments employ the training procedure de scribed in Algorithm 1. The same optimization settings and convergence criteria are used across all datasets to ensure a fair comparison between the proposed heterogeneous framework and the competing methods. As the Python library scikit-learn [33] is used for the decision trees within the implementation of the heterogeneous mixture model, the decision trees used for MoDT are also implemented using scikit-learn. Furthermore, the proposed framework is compared against Random Forests [34], which are widely recognized for their strong predictive performance. Although Random Forests are frequently regarded as strong predictive baselines, they typically provide considerably less transparency than interpretable mixture-based models. Consequently, achieving predictive performance comparable to Random Forests may itself be viewed as a favorable outcome in settings where interpretability is an important modeling objective. To allow a fair comparison, the same preprocessing steps included in the heterogeneous model are also applied to the training data of all alternative models.

For each dataset, the data is split into training data (75%) and test data (25%) using randomized train-test splits. The splitting procedure is repeated 10 times with diferent random seeds in order to obtain statistically stable performance estimates, and the mean classification accuracy and standard deviation over all repetitions are reported as the final performance.

Algorithm 1 Training Procedure of the Proposed Heterogeneous Mixture-of-Experts   
Require: Training data $\mathcal D = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n } ,$ expert families $\mathcal { F } = \{ \mathrm { D T } , \mathrm { S V M } , \mathrm { Q D A } \}$ , number of experts $e ,$ maximum iterations $T ,$ initial   
learning rate γ .   
Ensure: Estimated gating parameters ${ \widehat { \Theta } } ,$ expert parameters ${ \widehat { \Psi } } .$   
1: Initialize gating parameters $\Theta ^ { ( 0 ) }$   
2: Compute initial gating probabilities $G ^ { ( 0 ) }$   
3: for $t = 0 , \ldots , T - 1$ do   
4: E-step   
5: for each observation i do   
6: for each expert j do   
7: Compute posterior responsibility   
$\tau _ { i j } ^ { ( t ) } = \frac { g _ { j } ( x _ { i } ; \Theta ^ { ( t ) } ) p _ { j } ( y _ { i } | x _ { i } ; \psi _ { j } ^ { ( t ) } ) } { \sum _ { k = 1 } ^ { e } g _ { k } ( x _ { i } ; \Theta ^ { ( t ) } ) p _ { k } ( y _ { i } | x _ { i } ; \psi _ { k } ^ { ( t ) } ) } .$   
8: end for   
9: end for   
10: Form the responsibility matrix $T ^ { ( t ) } .$   
11:   
M-step   
12: for each expert j do   
13: if expert j is DT then   
14: Train a weighted Decision Tree.   
15: else if expert j is Linear SVM then   
16: Train a weighted Linear SVM.   
17: Calibrate probabilities using Platt scaling.   
18: else if expert j is QDA then   
19: Estimate weighted means and covariance matrices.   
20: end if   
21: end for   
22: Compute residual matrix $R ^ { ( t ) } = T ^ { ( t ) } - G ^ { ( t ) } ,$   
23: Compute $B ^ { ( t ) } = ( X ^ { \top } X ) ^ { - 1 } X ^ { \top } R ^ { ( t ) }$   
24: Update gating parameters $\Theta ^ { ( t + 1 ) } = \Theta ^ { ( t ) } + \gamma _ { t } B ^ { ( t ) }$   
25: Update learning rate $\gamma _ { t + 1 } = \eta \gamma _ { t }$   
26: if log-likelihood increment < ε then   
27: Break.   
28: end if   
29: end for   
30: Return ${ \widehat { \Theta } } , { \widehat { \Psi } } .$

The proposed heterogeneous model uses a fixed optimization configuration throughout all experiments. In particular, the initialization learning rate, the learning rate decay and the initialization procedure are fixed across all datasets and repetitions. We fix the initialization procedure to Random\_init(40) to minimize the variability of the initialization due to stochastic efects and to ensure reproducibility. As a result, experimental variability is introduced mainly by random train-test splits, rather than repeated hyperparameter tuning.

The motivation for using a fixed hyperparameter setting is two-fold. First, the main goal of this work is to assess the impact of heterogeneous expert composition in the MoDT framework, rather than optimizing datasetspecific hyperparameters. By fixing optimization settings, improvements in predictive performance can be more directly attributed to the heterogeneous expert architecture itself rather than to aggressive hyperparameter search procedures. Second, large-scale hyperparameter optimization can give over-optimistic estimates, and may mask the intrinsic behavior of heterogeneous expert specialization. Thus, the proposed evaluation protocol focuses on reproducibility, architectural fairness and robustness across multiple datasets under a common optimization setting.

In addition, the proposed approach allows for analysis beyond predictive performance. Because the heterogeneous mixture framework integrates decision trees, support vector machines and quadratic discriminant analysis experts in a common gating mechanism, expert utilization statistics are additionally analyzed to investigate specialization behavior across datasets and feature distributions.

We also conduct an expert specialization analysis for the proposed heterogeneous mixture framework. Specifically, we say that an expert dominates the heterogeneous gating mechanism for the corresponding dataset if the average utilization of a single expert over multiple experimental runs exceeds 90%. In these cases, the dominant expert model is also trained independently on the same dataset and the overall maximum predictive accuracy is reported. This analysis ofers additional insights into whether the heterogeneous framework benefits primarily from collaborative expert interaction or whether certain datasets are inclined to a particular expert type. For completeness, Algorithm 2 summarizes the prediction procedure together with the expert specialization analysis used throughout the experimental study. Besides generating class predictions, the algorithm records the average utilization of each expert and identifies datasets for which a single expert dominates the gating mechanism.

## 5.1. Application on Synthetic Datasets

Table 1 gives the descriptions of the generated datasets while Table 2 and Figure 1 present the experimental results on a diverse collection of synthetic benchmark datasets designed to capture various geometric decision boundaries and covariance structures. Figure 2 illustrates the expert utilisation behaviour across these datasets. The proposed heterogeneous mixture framework performs well on a number of nonlinear and covariance-sensitive datasets and at the same time provides insight into the behavior of expert specialization. The dominance of the QDA expert in the gating mechanism for datasets like Anisotropic Gaussian, Hierarchical Gaussian and Radial Sector Multiclass (Figure 2) indicates the suitability of covariance-based quadratic decision boundaries for these distributions. Furthermore, for Piecewise Linear Kink and Rotated Gaussian Slabs the heterogeneous framework reveals a significant collaboration between decision tree, SVM and QDA experts, which indicates that diferent experts specialize in diferent regions of the feature space. These observations demonstrate that the proposed heterogeneous gating framework is capable of adaptively selecting suitable expert models according to intrinsic geometric characteristics of the data distribution while achieving predictive performance competitive with strong ensemble baselines such as Random Forests.

Classification Accuracy - Synthetic Datasets  
![](images/0b97ec5bb26b82ec50894c449c14c90214551a06b53ec39d32ebd3a316fc795e.jpg)  
Figure 1: Mean classification accuracy on synthetic datasets. Error bars indicate one standard deviation over ten random train-test splits. Results are shown for Random Forest, MoDT (depth 2 and 3), and the proposed Hetero-Mix (depth 2 and 3).

Expert Utilisation - Synthetic Datasets  
![](images/50c74dbded66ad83ca683d966c8d1b1cd1b3e2b466fd4f2e8f895ebf073b1933.jpg)  
Figure 2: Average expert utilisation (%) on synthetic datasets for the proposed heterogeneous mixture model. Each stacked bar shows the proportion of samples assigned to the Decision Tree, SVM, and QDA experts by the gating mechanism, averaged over ten train-test splits.

Algorithm 2 Prediction and Expert Specialization Analysis   
Require: Test data, trained gating parameters Θb, trained experts Ψb.   
Ensure: Predicted labels and expert utilization statistics.   
1: for each test observation x do   
2: for each expert j do   
3: Compute gating probability   
$g _ { j } ( x ; \widehat { \Theta } ) .$   
4: Compute calibrated expert probability   
$p _ { j } ( y | x ; { \widehat { \psi } } _ { j } )$   
5: end for   
6: Compute mixture probability   
$\begin{array} { r } { P ( y | x ) = \sum _ { j = 1 } ^ { e } g _ { j } ( x ) p _ { j } ( y | x ) . } \end{array}$   
7: Predict by = arg maxy P(y|x)   
8: Assign the observation to the expert with maximum gating probability.   
9: end for   
10: Compute the average utilization of each expert.   
11: if average utilization of an expert exceeds 90% then   
12: Train the dominant expert independently on the same dataset.   
13: Evaluate its standalone predictive accuracy.   
14: Report maximum of standalone and heterogeneous model accuracies.   
15: else   
16: Report only the heterogeneous model accuracy.   
17: end if

Table 1: Description of synthetic datasets.
<table><tr><td>Dataset</td><td>Data Points</td><td>Classes</td><td>Description</td></tr><tr><td>overlapping_rings</td><td>3,000</td><td>2</td><td>Noisy concentric rings with overlap.</td></tr><tr><td>rotated_gaussian_close</td><td>2,000</td><td>2</td><td>Rotated Gaussian classes with overlapping quadratic bound- aries.</td></tr><tr><td>piecewise_linear_kink</td><td>3,000</td><td>2</td><td>Piecewise linear decision boundary with a kink.</td></tr><tr><td>anisotropic_gaussian</td><td>600</td><td>2</td><td>Two centered Gaussian classes with orthogonal anisotropic covariance structures.</td></tr><tr><td>blob_vs_ring</td><td>500</td><td>2</td><td>Central Gaussian blob versus outer ring structure.</td></tr><tr><td>multiclass_anisotropic_gaussians</td><td>2,400</td><td>4</td><td>Anisotropic Gaussian classes with different covariance ori- entations.</td></tr><tr><td>hierarchical_gaussians</td><td>3,000</td><td>6</td><td>Hierarchical Gaussian subclasses forming two superclusters.</td></tr><tr><td>radial_sector_multiclass</td><td>3,000</td><td>6</td><td>Angular sector-based classes with nonlinear radial bound- aries.</td></tr></table>

Qualitative Analysis ofDecision Boundaries and Expert Specialization.. Figures 3 and 4 compare the ground truth (or Bayes-optimal) decision boundaries, the decision boundaries learned by the MoDT, and the specializa tion patterns learned by the proposed heterogeneous Mixture of Experts framework. For the Piecewise Linear

![](images/41ff4bb05a5d0fad35198ede5a11151e3c26b5fe51230647713d3f61e6122be3.jpg)  
(a) Ground-truth decision boundary.

![](images/d26a2608015abbe269b1e5f9ad6b7bfdc27b92e4bc3f90efd7cefc6c01fe9ab8.jpg)  
(b) Homogeneous MoDT decision boundary.

Heterogeneous MoDT: Expert Boundaries in Their Own Regions  
![](images/54cea23ecd5f84d713734fddcbb46473e18306ae8e8f3ed3341c5ff6f438a012.jpg)  
(c) Expert specialization learned by the heterogeneous Mixture Model.

Figure 3: Comparison of decision boundaries and expert specialization on the Piecewise Linear Kink dataset. Subfigure (a) shows the ground-truth data-generating decision boundary consisting of two linear segments with opposite slopes that meet at a kink. Subfigure (b) presents the decision boundary learned by the MoDT composed exclusively of decision-tree experts. Subfigure (c) visualizes the specialization behavior learned by the proposed heterogeneous mixture framework. The shaded background denotes the dominant expert selected by the gating mechanism, where beige corresponds to the linear SVM expert, pink corresponds to the QDA expert, and blue corresponds to the decision-tree expert. Red points denote observations from class 1, while blue points denote observations from class 0. Expert decision boundaries are displayed only within the regions where the corresponding expert receives dominant gating responsibility.

![](images/8eb03f53dd4a927d5d1e84d0a2fb9285ff36bca873e77d5e3f6552185078af3c.jpg)  
(a) Estimated Bayes-optimal boundary.

![](images/b37e047182c30d023f14733c96e26116e1a3aa3295a74bf0824b2310672be835.jpg)  
(b) Homogeneous MoDT decision boundary.

Heterogeneous MoDT: Expert Boundaries Inside Their Assigned Regions  
![](images/c33270d5e1789940d1f618e4b7a9871eba508e31353bce7a9f544f4ee3315051.jpg)  
(c) Expert specialization learned by the heterogeneous Mixture Model.

Figure 4: Comparison of decision boundaries and expert specialization on the Rotated Gaussian Slabs dataset. Subfigure (a) shows th estimated Bayes-optimal decision boundary obtained using quadratic discriminant analysis. Subfigure (b) presents the decision boundary learned by the MoDT composed exclusively of decision-tree experts. Subfigure (c) visualizes the specialization behavior learned by the proposed heterogeneous mixture framework. The shaded background denotes the dominant expert selected by the gating mechanism, where pink corresponds to the QDA expert, and blue corresponds to the decision-tree expert. Diferent colour of points denotes observations from diferent classes. Expert decision boundaries are displayed only within the regions where the corresponding expert receives dominant gating responsibility.

Table 2: Comparison of models across synthetic datasets (mean accuracy ± std).
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Random Forest</td><td colspan="2">MoDT</td><td colspan="2">Hetero-Mix Expert</td><td rowspan="2">Avg. Expert Usage (%)</td></tr><tr><td>Depth 2</td><td>Depth 3</td><td>Depth 2</td><td>Depth 3</td></tr><tr><td>Overlapping Rings</td><td> $0 . 7 7 \pm 0 . 0 1$ </td><td> $0 . 7 8 \pm 0 . 0 1$ </td><td> $0 . 7 8 \pm 0 . 0 1$ </td><td> $0 . 7 7 \pm 0 . 0 1$ </td><td> ${ \bf 0 . 7 9 \pm 0 . 0 2 }$ </td><td>DT 75.4, QDA 24.6</td></tr><tr><td>Rotated Gaussian Slabs</td><td> $0 . 6 9 \pm 0 . 0 1$ </td><td> $0 . 7 2 \pm 0 . 0 1$ </td><td> $0 . 7 2 \pm 0 . 0 1$ </td><td> $0 . 7 2 \pm 0 . 0 1$ </td><td> $\mathbf { 0 . 7 3 \pm 0 . 0 1 }$ </td><td>DT 62.5, SVM 20.7, QDA 16.8</td></tr><tr><td>Piecewise Linear Kink</td><td> $\mathbf { 0 . 9 5 \pm 0 . 0 1 }$ </td><td> $0 . 9 2 \pm 0 . 0 1$ </td><td> $0 . 9 4 \pm 0 . 0 1$ </td><td> $0 . 9 1 \pm 0 . 0 1$ </td><td> $0 . 9 4 \pm 0 . 0 1$ </td><td>DT 18.5, SVM 39.6, QDA 41.9</td></tr><tr><td>Anisotropic Gaussian</td><td> $0 . 8 8 \pm 0 . 0 2$ </td><td> $0 . 8 6 \pm 0 . 0 2$ </td><td> $0 . 8 5 \pm 0 . 0 2$ </td><td> ${ \bf 0 . 8 9 \pm 0 . 0 2 }$ </td><td> $0 . 8 5 \pm 0 . 0 2$ </td><td>QDA 100</td></tr><tr><td>Inner Blob + Outer Ring</td><td> $\mathbf { 0 . 8 1 \pm 0 . 0 2 }$ </td><td> $\mathbf { 0 . 8 1 \pm 0 . 0 2 }$ </td><td> $0 . 8 0 \pm 0 . 0 2$ </td><td> $\mathbf { 0 . 8 1 \pm 0 . 0 3 }$ </td><td> $0 . 8 0 \pm 0 . 0 2$ </td><td>DT 44.1, QDA 55.9</td></tr><tr><td>Multiclass Anisotropic Gaussian</td><td> $0 . 6 4 \pm 0 . 0 2$ </td><td> $0 . 6 3 \pm 0 . 0 4$ </td><td> $0 . 6 4 \pm 0 . 0 2$ </td><td> ${ \bf 0 . 6 9 \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 6 9 \pm 0 . 0 2 }$ </td><td>QDA 100</td></tr><tr><td>Hierarchical Gaussians</td><td> $\mathbf { 0 . 9 9 \pm 0 . 0 1 }$ </td><td> $0 . 9 8 \pm 0 . 0 1$ </td><td> $0 . 9 8 \pm 0 . 0 1$ </td><td> $\mathbf { 0 . 9 9 \pm 0 . 0 1 }$ </td><td> $0 . 9 7 \pm 0 . 0 2$ </td><td>QDA 100 or DT 29.6, QDA 70.4</td></tr><tr><td>Radial Sector Multiclass</td><td> $0 . 9 0 \pm 0 . 0 1$ </td><td> $0 . 8 9 \pm 0 . 0 2$ </td><td> $0 . 8 9 \pm 0 . 0 1$ </td><td> $\mathbf { 0 . 9 2 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 9 2 \pm 0 . 0 1 }$ </td><td>QDA 100</td></tr></table>

Kink dataset, the ground-truth boundary consists of two linear segments with opposite slopes that meet at a kink, creating a decision structure that cannot be represented by a single global linear classifier. The MoDT successfully captures the presence of the kink but approximates the underlying geometry through a collection of piecewise axis-aligned partitions induced by tree experts. In contrast, the heterogeneous Mixture produces a more interpretable decomposition of the feature space by assigning the left branch primarily to the SVM expert, the right branch to the QDA expert, and a lower-central region to the decision tree expert. Consequently, each expert contributes a meaningful local decision boundary that reflects its underlying inductive bias, yielding a decomposition that more closely aligns with the geometry of the data-generating process.

A diferent specialization pattern emerges on the Multiclass Anisotropic Gaussian dataset, whose Bayesoptimal classifier is efectively linear. Although a linear support vector machine is also capable of representing such a decision boundary, the heterogeneous mixture assigns the overwhelming majority of the posterior responsibility to the QDA expert. This behaviour is consistent with the underlying data-generating process. The class-conditional distributions possess approximately equal covariance structures, under which the quadratic discriminant function reduces to linear discriminant analysis (LDA), yielding an essentially linear decision boundary. Consequently, the QDA expert naturally recovers the optimal decision rule while retaining the ability to model more general quadratic boundaries if required. The gating mechanism therefore has little incentive to allocate substantial responsibility to the linear SVM expert.

In contrast, the original MoDT is restricted to decision tree experts and must approximate the underlying linear boundary through a collection of axis-aligned partitions. Although the gating mechanism combines multiple tree experts, the resulting piecewise approximation cannot recover the globally linear Bayes decision boundary as faithfully, leading to comparatively lower predictive performance.

Since the average expert usage assigned to the QDA expert exceeds 90%, we additionally evaluate the corresponding standalone QDA classifier. Figure 4 presents both the Bayes-optimal (QDA) decision boundary and the boundary learned by the proposed heterogeneous mixture. Although the two boundaries are nearly identical, the comparison serves a diferent purpose. It demonstrates that the gating mechanism successfully identifies the statistically appropriate expert family and efectively reduces the heterogeneous mixture to the correct local model. Consequently, the figure illustrates not only accurate prediction but also the ability of the proposed framework to perform meaningful expert specialization through adaptive model selection.

These observations highlight a principal advantage of the proposed framework. Rather than enforcing a homogeneous hypothesis class across all experts, the heterogeneous mixture is capable of selecting the induc tive bias most appropriate for diferent regions of the feature space. In this example, the gating mechanism correctly recognises that a covariance-based discriminative model provides the most faithful representation of the underlying data-generating process, thereby recovering the Bayes-optimal decision boundary while preserving the interpretability of the overall mixture architecture. Taken together, these examples demonstrate that the proposed heterogeneous mixture framework does not enforce uniform participation among experts; rather, it adaptively selects and combines heterogeneous inductive biases according to the geometric and statistical characteristics of diferent regions of the feature space. The resulting decomposition improves interpretability while providing insight into how diferent expert classes collaborate to solve classification tasks of varying complexity.

Table 3: Summary of real-world datasets.
<table><tr><td>Dataset Name</td><td># Data Points</td><td># Classes</td><td># Features</td><td>Description</td></tr><tr><td>Adult</td><td>45,222</td><td>2</td><td>103</td><td>Census-based dataset used for income pre- diction.</td></tr><tr><td>GYM</td><td>2,000</td><td>2</td><td>4</td><td>Reinforcement learning state-action classi- fication dataset.</td></tr><tr><td>Breast Cancer</td><td>569</td><td>2</td><td>10</td><td>Predict breast cancer based on properties extracted from images.</td></tr><tr><td>Iris</td><td>150</td><td>3</td><td>4</td><td>Classify iris flower species based on sepal</td></tr><tr><td>Wine (Chemical)</td><td>178</td><td>3</td><td>13</td><td>and petal measurements. Classify wine cultivars using chemical</td></tr><tr><td>Congressional Voting Records</td><td>435</td><td>2</td><td>16</td><td>composition measurements. Classify political party from congressional roll-call voting records.</td></tr><tr><td>Connectionist Bench</td><td>208</td><td>2</td><td>60</td><td>Classify sonar signals as reflections from mines or rocks using acoustic measure-</td></tr><tr><td>Rice (Osmancik vs Cammeo)</td><td>3810</td><td>2</td><td>7</td><td>ments. Distinguish Osmancik and Cammeo rice varieties using morphological characteris- tics extracted from grain images.</td></tr></table>

Table 4: Comparison of models across real-world datasets (mean accuracy ± std).
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Random Forest</td><td colspan="2">MoDT</td><td colspan="2">Hetero-Mix Expert</td><td rowspan="2">Avg. Expert Usage (%)</td></tr><tr><td>Depth 2</td><td>Depth 3</td><td>Depth 2</td><td>Depth 3</td></tr><tr><td>Iris</td><td> $0 . 9 4 \pm 0 . 0 2$ </td><td> $0 . 9 3 \pm 0 . 0 3$ </td><td> $0 . 9 4 \pm 0 . 0 3$ </td><td> $\mathbf { 0 . 9 5 \pm 0 . 0 3 }$ </td><td> $0 . 9 3 \pm 0 . 0 2$ </td><td> $\mathrm { D T } 6 2 . 1 , \mathrm { Q D A } 3 7 . 9$ </td></tr><tr><td>Breast Cancer</td><td> $\mathbf { 0 . 9 4 \pm 0 . 0 2 }$ </td><td> $0 . 9 2 \pm 0 . 0 2$ </td><td> $0 . 9 3 \pm 0 . 0 2$ </td><td> $0 . 9 1 \pm 0 . 0 2$ </td><td> $0 . 9 2 \pm 0 . 0 2$ </td><td> $\mathrm { D T } 5 2 . 5 , \mathrm { S V M 0 . 1 } , \mathrm { Q D A } 4 7 . 4$ </td></tr><tr><td>Gym</td><td> $\mathbf { 0 . 9 4 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 9 4 \pm 0 . 0 1 }$ </td><td> $0 . 9 3 \pm 0 . 0 2$ </td><td> $0 . 9 3 \pm 0 . 0 1$ </td><td> $\mathbf { 0 . 9 4 \pm 0 . 0 1 }$ </td><td>DT 100</td></tr><tr><td>Adult</td><td> $_ { 0 . 7 8 9 }$ </td><td>0.783</td><td>0.849</td><td>0.850</td><td>0.838</td><td> $\mathrm { D T } 7 . 9 , \mathrm { S V M 9 0 . 2 , Q D A 1 . 9 }$ </td></tr><tr><td>Wine (Chemical)</td><td> $\mathbf { 0 . 9 9 \pm 0 . 0 1 }$ </td><td> $0 . 9 3 \pm 0 . 0 3$ </td><td> $0 . 9 2 \pm 0 . 0 3$ </td><td> $0 . 9 4 \pm 0 . 0 3$ </td><td> $0 . 9 4 \pm 0 . 0 2$ </td><td> $\mathrm { D T } 4 9 . 6 , \mathrm { Q D A } 5 0 . 4$ </td></tr><tr><td>Congressional Voting Records</td><td> $\mathbf { 0 . 9 6 \pm 0 . 0 1 }$ </td><td> $0 . 9 4 \pm 0 . 0 1$ </td><td> $0 . 9 4 \pm 0 . 0 1$ </td><td> $0 . 9 5 \pm 0 . 0 1$ </td><td> $0 . 9 5 \pm 0 . 0 1$ </td><td>DT 47.5, SVM 2.1, QDA 50.3</td></tr><tr><td>Connectionist</td><td> $\mathbf { 0 . 8 0 \pm 0 . 0 1 }$ </td><td> $0 . 6 9 \pm 0 . 0 6$ </td><td> $0 . 7 2 \pm 0 . 0 6$ </td><td> $0 . 7 1 \pm 0 . 0 5$ </td><td> $0 . 7 3 \pm 0 . 0 9$ </td><td>DT 32.6, SVM 0.3, QDA 67.2</td></tr><tr><td>Rice (Osmancik vs Cammeo)</td><td> $0 . 9 2 \pm 0 . 0 1$ </td><td> $0 . 9 2 \pm 0 . 0 1$ </td><td> $0 . 9 2 \pm 0 . 0 1$ </td><td> $\mathbf { 0 . 9 3 \pm 0 . 0 1 }$ </td><td> $0 . 9 2 \pm 0 . 0 1$ </td><td> $\mathrm { D T } 4 2 . 2 , \mathrm { S V M } 2 3 . 6 , \mathrm { Q D A } 3 4 . 2$ </td></tr></table>

## 5.2. Application on Real-World Datasets

To evaluate the proposed framework on practical classification problems, we consider eight widely used benchmark datasets drawn from diverse application domains. The Adult [35], Breast Cancer, Gym, and Iris [36] datasets were obtained from the publicly available implementation accompanying the original Mixture of Decision Trees (MoDT) framework [20], while the Connectionist (Sonar) [37], Wine [38], Congressional Voting Records, and Rice [39] datasets were obtained from the UCI Machine Learning Repository [40, 41]. Table 3 gives the descriptions of the datasets analyzed while Table 4 and Figure 5 summarize the classification performance of the proposed heterogeneous mixture framework on several real-world benchmark datasets. Figure 6 illustrates the corresponding expert utilisation patterns. In general, the heterogeneous mixture model provides competitive performance compared to Random Forests and the MoDT, while also providing interpretable expert specialization behavior. On several datasets (e.g., Adult, Iris, Rice), the heterogeneous model performs as well or slightly better than the MoDT architecture, which suggests that a mixture of heterogeneous experts can lead to better generalization across datasets with diferent underlying structures. Moreover, the expert utilization figures show interesting specialization trends. For example, the Adult dataset is mostly dominated by the SVM expert, indicating that there are roughly linearly separable regions in the feature space, whereas datasets such as Gym show almost total dominance of a single expert, indicating that some datasets are naturally adapted to particular expert structures. Conversely, the more balanced expert utilization in datasets such as Rice and Breast Cancer suggests that collaborative interaction among heterogeneous experts may play a role in predictive performance in more complex classification scenarios.

Classification Accuracy - Real-World Datasets  
![](images/9099548425631dd9546ec0098d4590e2d77ec53ddb4937aed44e8abcb96d9a95.jpg)  
Figure 5: Mean classification accuracy on real-world datasets. Error bars indicate one standard deviation over ten random train-test splits. Results are shown for Random Forest, MoDT (depth 2 and 3), and the proposed Hetero-Mix (depth 2 and 3).

![](images/1107552e35df2008b9827cab3f3474fb0bce14d7e3fd87afe27c7904ee63c184.jpg)  
Figure 6: Average expert utilisation (%) on real-world datasets for the proposed heterogeneous mixture model. Each stacked bar shows the proportion of samples assigned to the Decision Tree, SVM, and QDA experts by the gating mechanism, averaged over ten train-test splits.

## 6. Conclusion and Future Research

This work develops Heterogeneous Mixture of Experts (Hetero-MoE) as a framework for pattern classification in which the predictive mechanism can adapt not only through its parameters and spatial allocation of observations, but also through the choice of the underlying hypothesis class. The distinction is fundamental. In a conventional homogeneous mixture, the gating mechanism determines which expert is emphasized, but the experts themselves remain members of the same model family. In Hetero-MoE, the local representation is allowed to change with the structure of the problem: a decision tree can describe an axis-aligned rule, a linear support vector machine can represent a locally hyperplanar separation, and quadratic discriminant analysis can capture covariance-dependent class structure. The resulting model therefore treats the selection of an inductive bias as part of the pattern-recognition problem itself. Rather than asking only how accurately a decision boundary can be approximated, the framework also asks which interpretable form of decision boundary is appropriate in a given region of the feature space.

The principal strength of the proposed approach is consequently its ability to expose a level of model adaptation that is usually hidden inside a single predictive architecture. A flexible homogeneous learner may approximate several geometries simultaneously, but the resulting representation does not necessarily reveal which modeling assumption is responsible for the prediction in a particular region. Hetero-MoE makes this distinction explicit. The gating mechanism determines the relative contribution of expert families, while the experts retain their own recognizable statistical structure. The resulting decomposition provides a natural bridge between mixture modeling, local model selection, and interpretable pattern recognition. In this sense, the proposed framework is not simply a larger ensemble of classifiers; it makes the choice of predictive representation itself an object of estimation.

The empirical results demonstrate both the usefulness and the limits of this additional flexibility. Hetero-MoE improves upon or matches homogeneous MoDT on several problems whose local geometry is heterogeneous or covariance-sensitive, while its expert-utilization patterns often correspond to recognizable properties of the underlying recognition problem. In the Piecewise Linear Kink experiment, for example, the simul taneous participation of decision-tree, SVM, and QDA experts provides a local decomposition of a globally heterogeneous boundary rather than an approximation based exclusively on axis-aligned partitions. Simila specialization patterns are observed in the real-world experiments: the Adult data are predominantly associated with the SVM expert, Gym is efectively represented by the decision-tree expert, and datasets such as Rice and Breast Cancer exhibit more balanced utilization across expert families. These results demonstrate that expert utilization can serve not only as an optimization outcome but also as a descriptive representation of the statistical structure identified by the fitted classifier.

At the same time, the experiments show that heterogeneity is not universally beneficial. There are datasets on which Random Forest or homogeneous MoDT remains competitive or superior to Hetero-MoE. This observation is an important part of the contribution rather than a weakness to be concealed. It indicates that introducing additional hypothesis classes does not automatically improve generalization; the value of heterogeneity depends on the relationship between the data geometry, the available expert families, the routing mechanism, and the amount of information available to estimate the resulting local models. Hetero-MoE should therefore be viewed as an additional modeling degree of freedom rather than as a universal replacement for homogeneous classifiers. The more informative question for practitioners becomes not simply ‘Which classifier should be used?” but, when the data justify such a decomposition, ‘Which classifier should be used where?”

This positioning is complementary to recent developments in interpretable Pattern Recognition and mixtureof-experts modeling. Recent Pattern Recognition studies have considered intrinsically interpretable approaches based on prototypes, neural decision trees, and constrained or monotonic architectures [42–44], while recent MoE research has investigated intrinsic interpretability through structured expert architectures and routing [19]. More recent work in Pattern Recognition has also demonstrated the predictive value of heterogeneous neural experts combined through adaptive routing in fine-grained visual classification [45]. These approaches establish that specialization and routing remain active themes in contemporary pattern recognition, but they address settings that difer materially from the present one. In particular, Hetero-MoE combines heterogeneous, intrinsically interpretable statistical learners with distinct inductive biases inside a common probabilistic conditional model. Its principal distinction is therefore not that heterogeneous routing itself is new, but that heterogeneous classical hypothesis classes can be treated as explicitly selectable local recognition mechanisms while preserving probabilistic interpretability.

Accordingly, the results should not be interpreted as establishing universal state-of-the-art predictive accu racy across contemporary classification methods. Such a claim is neither required by the proposed framework nor supported uniformly by the experiments. Rather, the empirical evidence shows that Hetero-MoE can remain competitive with strong predictive baselines while providing information that is structurally diferent from that supplied by black-box ensembles or homogeneous interpretable mixtures. A Random Forest can be highly effective without exposing a region-wise decomposition into distinct statistical mechanisms, while homogeneous MoDT retains transparency but restricts every expert to the same hypothesis class. Hetero-MoE occupies the intermediate space between these two objectives by allowing predictive competition across fundamentally different interpretable mechanisms and by exposing the resulting allocation through the gating probabilities and expert responsibilities.

For practitioners in pattern recognition, this additional information can be useful at several stages of model development. Expert utilization can provide a form of structural diagnosis: near-complete dominance by one expert may indicate that the data are adequately represented by a simpler homogeneous model, whereas persistent use of several experts may indicate heterogeneous local structure. The learned specialization can also guide subsequent model development by revealing which kinds of decision mechanisms appear to be relevant before a practitioner adopts a substantially more complex black-box architecture. Moreover, the use of recognizable expert families provides a natural language for communicating model behavior to domain experts. A prediction can be associated with a tree-based rule, a linear separation, or a covariance-driven discriminant mechanism rather than only with an opaque latent representation. Because the individual experts remain independently interpretable, local portions of the fitted decision function can also be inspected, audited, or replaced without requiring the entire predictive architecture to be discarded.

A related strength is that the proposed framework separates three forms of adaptation that are often conflated in predictive modeling: adaptation of model parameters, adaptation of the spatial allocation of observations, and adaptation of the hypothesis class itself. Classical homogeneous mixtures provide the first two forms, whereas Hetero-MoE introduces the third. This creates a conceptual link between model selection and mixture modeling: rather than choosing a single global hypothesis class before estimation, the model allows the data to determine which interpretable predictive paradigm should govern diferent parts of the feature space. The resulting framework therefore provides a mechanism for representing heterogeneity in the form of a classifier rather than only in its fitted parameters.

The probabilistic formulation provides an additional strength. Expert responsibilities are obtained from the posterior distribution of the latent allocation variable rather than from an externally imposed similarity score or an independently defined heuristic confidence measure. Consequently, expert utilization has a direct probabilistic interpretation within the fitted conditional mixture. The calibration of the SVM outputs is important in this respect because it allows a margin-based classifier to participate in the same probabilistic mixture as the other experts. The generalized EM formulation consequently establishes a coherent connection between local prediction, posterior responsibility, and expert allocation. The accompanying monotone-ascent result for the gating auxiliary objective further shows that the regression-based gating correction is not merely a computa tional heuristic, but admits a formal generalized-EM interpretation under the stated learning-rate conditions.

The interpretability ofered by Hetero-MoE should nevertheless be understood in a precise sense. The present study evaluates interpretability primarily through the intrinsic transparency of the expert families, explicit gating probabilities, expert-utilization statistics, and visualization of local decision boundaries. These components provide structural information that is unavailable from a purely black-box predictor, but they do not constitute a complete human-centred assessment of whether the resulting explanations are useful, faithful, stable, or cognitively eficient. Recent work has emphasized the need to evaluate explanation quality explicitly rather than infer it solely from model architecture [46]. A useful continuation of the present work is therefore not to establish interpretability as a new problem, but to determine how well the specific explanation furnished by expert-family selection communicates the behavior of a heterogeneous classifier to practitioners and domain experts.

The present work nevertheless has several limitations. First, the performance and specialization behavior of the framework depend on the collection of expert families made available to the gating mechanism. If an appropriate expert class is absent, the model may still experience local model mismatch despite its heterogeneous structure. Second, the framework does not guarantee balanced utilization of all experts. Indeed, for some datasets the gating mechanism naturally collapses toward a single dominant expert, efectively recovering a homogeneous model. While such behavior is often informative and may itself reveal the most suitable inductive bias for the dataset, it reduces the practical benefits of maintaining a larger heterogeneous architecture.

A second limitation concerns the expert routing mechanism. The proposed framework inherits the probabilistic softmax gating formulation from the classical mixture-of-experts architecture [1, 2] while preserving the interpretable routing mechanism employed in MoDT [20]. Although this provides computational eficiency and a coherent probabilistic interpretation of expert responsibilities, a linear gating function may become restrictive when the true allocation boundaries are highly nonlinear, and likelihood-based responsibilities may not always yield the most informative or stable expert specialization. More flexible routing mechanisms are already well established in contemporary MoE systems; consequently, the unresolved question in the present setting is narrower: how can routing be made nonlinear or adaptive without losing the intrinsic interpretability of heterogeneous statistical experts? In particular, tree-based, additive, or other structured gates could be investigated so that the routing boundaries themselves remain interpretable while accommodating substantially more complex allocation geometries.

The dependence on a prespecified expert dictionary is particularly consequential for the broader vision of adaptive pattern recognition. At present, the framework can choose among the models supplied by the prac titioner, but it cannot discover an entirely new hypothesis class when none of the available experts is appro priate. While model-selection procedures for the number of experts and other aspects of MoE architecture have been studied in the broader literature, the unresolved question here is more specific: how should a heterogeneous conditional mixture choose which statistical learning paradigms should be present in its expert dictionary when those paradigms have diferent parameterizations, dimensions, optimization procedures, and interpretability properties? This problem is distinct from simply selecting the number of experts. A future model could, for example, jointly determine whether a decision-tree, margin-based, discriminant, additive, or prototype-based expert is warranted and whether its inclusion provides suficient predictive benefit to justify the additional complexity. Developing such a parsimonious expert-selection mechanism would turn the current practitioner-specified dictionary into a more fully data-adaptive component of the model.

The present work also leaves several important theoretical questions unanswered. Existing work has established substantial asymptotic theory for conventional mixtures-of-experts, including nonstandard behavio associated with latent expert allocation and the estimation of the number of experts [47]. Recent Bayesian developments have further studied posterior contraction, parameter estimation, identifiability, and model selection for softmax-gated mixtures-of-experts [48]. These results substantially reduce the scope of what can reasonably be described as theoretically unexplored in MoE methodology. The unresolved issue for Hetero-MoE is instead the extension of such theory to a conditional mixture whose experts belong to fundamentally diferent statistical families. In the present model, the components may have diferent parameter dimensions, diferent objective functions, diferent regularity properties, and diferent notions of local complexity. It therefore re mains to be determined which forms of identifiability, consistency, asymptotic normality, convergence rate, and model-selection guarantees continue to hold when heterogeneous interpretable experts coexist within the same conditional mixture. Establishing such results would provide a theoretical foundation specifically for hetero geneous statistical mixtures rather than for mixtures whose components are instances of a common parametric family.

The empirical results also reveal an interesting phenomenon that deserves further investigation. Although the proposed heterogeneous framework frequently outperforms or matches the homogeneous MoDT architecture [20], there also exist datasets for which the original homogeneous model achieves superior predictive performance. This observation suggests that heterogeneity is not universally advantageous and raises a more specific unresolved question concerning the relationship between data geometry, expert diversity, and model complexity. General MoE research already studies expert diversity, routing balance, and specialization; the question that remains open in the present setting is whether one can characterize, in terms of properties of the underlying classification problem, when the additional approximation flexibility obtained by heterogeneous hypothesis classes outweighs the additional estimation and routing complexity. Such a characterization would provide a principled basis for deciding when Hetero-MoE should be preferred to a homogeneous MoDT rather than relying on empirical trial alone.

A particularly promising empirical direction is therefore to move from asking whether Hetero-MoE per forms well on a collection of datasets to asking which measurable properties of a dataset predict the value of heterogeneous modeling. The current experiments already show substantial variation in expert utilization. A systematic follow-up could relate the gain of Hetero-MoE over homogeneous MoDT to quantities such as utilization entropy, the spatial separation of expert responsibilities, the complexity of local decision boundaries, and measures of local class-conditional geometry. The purpose would not be simply to produce another descrip tive visualization, but to determine whether these quantities can serve as empirical indicators of the conditions under which heterogeneous modeling is warranted. Such an analysis would directly operationalize the question of when heterogeneity is beneficial, which remains more specific to the present framework than the broader questions of expert diversity or routing optimization studied in contemporary MoE research.

The possibility of expert collapse also merits a more nuanced interpretation. In many modern MoE systems, severe imbalance in expert utilization is treated as a practical routing problem because it can produce ineficient computation or poorly trained experts. In Hetero-MoE, however, concentration on a single expert can have a second interpretation: it may indicate that the data provide little evidence for multiple inductive biases. The unresolved problem is therefore not simply how to force more balanced utilization, but how to distinguish undesirable optimization-induced collapse from statistically meaningful evidence that a homogeneous model is adequate. A principled criterion for making this distinction could allow expert collapse to become a modelsimplification signal rather than merely a failure mode. This would also provide a natural connection between routing, model selection, and interpretability.

The framework can be expanded beyond the specific collection of experts considered in this study. The same architecture can naturally incorporate generalized additive models, sparse linear models, probabilistic graphical models, Gaussian process experts, robust classifiers, prototype-based learners, or other interpretable statistical procedures. The important unresolved problem is not whether such models can be inserted mechanically into the mixture, but how their difering notions of complexity, regularization, uncertainty, and interpretability should be compared on a common basis during joint estimation. In particular, future work could investigate expert libraries tailored to particular Pattern Recognition domains and develop criteria by which additional expert families are admitted only when they ofer a demonstrable and interpretable improvement over the existing dictionary.

A second major direction is therefore the development of more flexible but still interpretable gating mechanisms. Although the current linear softmax gate is attractive for its simplicity, computational eficiency, and transparent parameterization, it may be insuficient when the regions in which diferent experts are appropriate are themselves highly nonlinear. Existing MoE literature already contains many powerful nonlinear and spatially adaptive routing mechanisms [45]; the more specific question for Hetero-MoE is how to obtain comparable routing flexibility while retaining an explicit statistical interpretation of both the gate and the expert assignment. Tree-based gates, sparse additive gates, piecewise-linear gates, or other structured routing functions are therefore natural candidates. In parallel, the routing objective could be extended to incorporate predictive uncertainty, calibration, expert complementarity, and stability. The unresolved issue is how these criteria should be combined without allowing routing to become so complex that the principal interpretability benefit of the heterogeneous architecture is lost.

The theoretical development of such extensions would also be valuable. The monotone-ascent result established here concerns the gating auxiliary objective and provides an optimization guarantee under specified learning-rate conditions; it does not establish global convergence or statistical optimality of the complete het erogeneous estimator. A more comprehensive theory would ideally connect the geometry of the expert classes to approximation error, estimation error, routing complexity, and identifiability, and would characterize the conditions under which heterogeneous mixtures ofer a genuine statistical advantage over homogeneous mixtures. In particular, it remains to be determined whether one can formulate oracle inequalities, risk comparisons, or data-dependent selection criteria that quantify when the gain from allowing multiple inductive biases justifies the additional complexity of the heterogeneous architecture.

Finally, the broader significance of the proposed framework extends beyond the specific collection of experts considered in this work. The central idea underlying mixture-of-experts models [1, 2] is that diferent regions of the feature space may require diferent predictive models. The present work extends this philosophy by allowing those local models to originate from fundamentally diferent statistical learning paradigms rather than a common hypothesis class. Consequently, the proposed methodology is not restricted to decision trees, linear support vector machines, and quadratic discriminant analysis. The same framework can naturally incorporate generalized additive models, sparse linear models, probabilistic graphical models, Gaussian process experts, robust classifiers, or other interpretable statistical learning procedures. We therefore view the proposed heterogeneous mixture-of-experts framework not merely as an extension of MoDT, but as a general methodology for combining heterogeneous statistical models within a unified probabilistic architecture. The central research opportunity is now to understand the statistical principles governing such heterogeneous specialization, to determine when it is beneficial, and to develop routing and selection mechanisms that preserve interpretability while admitting greater structural flexibility.

Taken together, the contribution of this work is broader than the particular empirical gains obtained by com bining three expert families. The deeper contribution is to make model heterogeneity itself an interpretable object of learning. Instead of treating the choice between a tree, a linear classifier, or a covariance-based discriminant model as a decision that must be made globally before fitting the classifier, Hetero-MoE allows tha choice to become spatially adaptive and probabilistically represented. This creates a model in which prediction, representation, and local model selection are learned jointly. For pattern-recognition practitioners, this provides a practical mechanism for exploiting heterogeneous local geometry without abandoning interpretable predictive mechanisms; for researchers, it motivates a more specific set of unanswered questions concerning the statistical value, identifiability, selection, and interpretation of heterogeneous hypothesis classes within a common condi tional mixture. In this sense, the significance of Hetero-MoE lies not in claiming that heterogeneous routing is new, but in demonstrating a concrete and interpretable formulation in which the inductive bias itselfcan become a spatially adaptive component ofa probabilistic pattern-recognition model.

## Code Availability

The oficial implementation of the proposed framework, including the source code, datasets, and experi mental scripts, is publicly available at GitHub Repository

## References

[1] R. A. Jacobs, M. I. Jordan, S. E. Nowlan, G. E. Hinton, Adaptive mixture of experts, Neural Comput. 3 (1991) 79–87.

[2] M. I. Jordan, R. A. Jacobs, Hierarchical mixtures of experts and the em algorithm, Neural Comput. 6 (1994) 181–214.

[3] T. G. Dietterich, Ensemble methods in machine learning, in: Proc. Int. Workshop Multiple Classifier Systems, 2000, pp. 1–15.

[4] G. McLachlan, D. Peel, Finite Mixture Models, John Wiley & Sons, New York, 2000.

[5] W. Gan, et al., Mixture of experts: A big data perspective, Inf. Fusion 118 (2025).

[6] Z.-H. Zhou, Ensemble Methods: Foundations and Algorithms, Chapman and Hall/CRC, 2012.

[7] T. Hastie, R. Tibshirani, J. Friedman, The Elements of Statistical Learning, 2nd Edition, Springer, New York, 2009.

[8] G. James, D. Witten, T. Hastie, R. Tibshirani, An Introduction to Statistical Learning, 2nd Edition, Springer, New York, 2021.

[9] V. N. Vapnik, Statistical Learning Theory, Wiley, 1998.

[10] S. Shalev-Shwartz, S. Ben-David, Understanding Machine Learning: From Theory to Algorithms, Cambridge University Press, 2014.

[11] C. Molnar, Interpretable Machine Learning, 2nd Edition, Lulu.com, 2022.

[12] R. Guidotti, A. Monreale, S. Ruggieri, F. Turini, F. Giannotti, D. Pedreschi, A survey of methods for explaining black box models, ACM Computing Surveys 51 (5) (2018) 93:1–93:42.

[13] C. Rudin, Interpretable machine learning: Fundamental principles and 10 grand challenges, Statistics Surveys 16 (2022) 1–85.

[14] M. T. Ribeiro, S. Singh, C. Guestrin, "why should i trust you?": Explaining the predictions of any classifier, in: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016, pp. 1135–1144.

[15] F. Doshi-Velez, B. Kim, Towards a rigorous science of interpretable machine learning, Tech. Rep. arXiv:1702.08608, arXiv (2017).

[16] M. Vasic, A. Petrovic, K. Wang, M. Nikolic, R. Singh, S. Khurshid, Moet: Interpretable and verifiable reinforcement learning via mixture of expert trees, arXiv preprint arXiv:1906.06717 (2019).

[17] A. A. Ismail, S. O. Arik, J. Yoon, A. Taly, S. Feizi, T. Pfister, Interpretable mixture of experts, arXiv preprint arXiv:2206.02107 (2022).

[18] V. Swamy, S. Montariol, J. Blackwell, J. Frej, M. Jaggi, T. Käser, Interpretcc: Intrinsic user-centric interpretability through global mixture of experts, arXiv preprint arXiv:2402.02933 (2024).

[19] X. Yang, C. Venhof, A. Khakzar, C. Schroeder De Witt, P. K. Dokania, A. Bibi, P. Torr, Mixture of experts made intrinsically interpretable, in: Proceedings of the 42nd International Conference on Machine Learning, Vol. 267 of Proceedings of Machine Learning Research, PMLR, 2025, pp. 71231–71248.

[20] S. Brüggenjürgen, N. Schaaf, P. Kerschke, M. F. Huber, Mixture of decision trees for interpretable machine learning, in: Proc. IEEE Int. Conf. Mach. Learn. Appl. (ICMLA), 2022, pp. 1175–1182.

[21] L. Breiman, J. H. Friedman, R. A. Olshen, C. J. Stone, Classification and Regression Trees, Wadsworth International Group, Belmont, CA, 1984.

[22] R. Piltaver, M. Lustrek, M. Gams, S. Martinciˇ c-Ipšiˇ c, What makes classification trees comprehensible?,ˇ Expert Syst. Appl. 62 (2016) 333–346.

[23] J. R. Quinlan, C4.5: Programs for Machine Learning, Morgan Kaufmann, 1993.

[24] C. Cortes, V. Vapnik, Support-vector networks, Mach. Learn. 20 (1995) 273–297.

[25] A. P. Dempster, N. M. Laird, D. B. Rubin, Maximum likelihood from incomplete data via the em algorithm, J. R. Stat. Soc. Ser. B 39 (1977) 1–22.

[26] C. M. Bishop, Pattern Recognition and Machine Learning, Springer, New York, 2006.

[27] J. C. Platt, Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods, in: Advances in Large Margin Classifiers, 1999, pp. 61–74.

[28] A. Niculescu-Mizil, R. Caruana, Predicting good probabilities with supervised learning, in: Proceedings of the 22nd International Conference on Machine Learning, 2005, pp. 625–632.

[29] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, On calibration of modern neural networks, in: Proceedings of the 34th International Conference on Machine Learning, 2017, pp. 1321–1330.

[30] K. P. Murphy, Probabilistic Machine Learning: An Introduction, MIT Press, 2022.

[31] G. J. McLachlan, T. Krishnan, The EM Algorithm and Extensions, 2nd Edition, Wiley, 2008.

[32] C. F. J. Wu, On the convergence properties of the em algorithm, The Annals of Statistics 11 (1) (1983) 95–103.

[33] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, E. Duchesnay, Scikit-learn: Machine learning in python, J. Mach. Learn. Res. 12 (2011) 2825–2830.

[34] L. Breiman, Random forests, Machine Learning 45 (1) (2001) 5–32.

[35] B. Becker, R. Kohavi, Adult, UCI Machine Learning Repository, DOI: https://doi.org/10.24432/C5XW20 (1996).

[36] R. A. Fisher, Iris, UCI Machine Learning Repository, DOI: https://doi.org/10.24432/C56C76 (1936).

[37] T. Sejnowski, R. Gorman, Connectionist Bench (Sonar, Mines vs. Rocks), UCI Machine Learning Repository, DOI: https://doi.org/10.24432/C5T01Q (1988).

[38] S. Aeberhard, M. Forina, Wine, UCI Machine Learning Repository, DOI: https://doi.org/10.24432/C5PC7J (1992).

[39] Rice (Cammeo and Osmancik), UCI Machine Learning Repository, DOI: https://doi.org/10.24432/C5MW4Z (2019).

[40] M. Kelly, R. Longjohn, K. Nottingham, The uci machine learning repository, https://archive.ics. uci.edu (2017).

[41] D. Dua, C. Graf, UCI machine learning repository (2017). URL http://archive.ics.uci.edu/ml

[42] J. Wen, H. Kong, W.-K. Wong, Z. Zhu, Characteristic discriminative prototype network with detailed interpretation for classification, Pattern Recognition 157 (2025) 110901. doi:10.1016/j.patcog.2024. 110901.

[43] J. Luo, S. Xu, Ncart: Neural classification and regression tree for tabular data, Pattern Recognition 154 (2024) 110578. doi:10.1016/j.patcog.2024.110578.

[44] V. Wargnier-Dauchelle, T. Grenier, F. Durand-Dubief, F. Cotton, M. Sdika, Explainable monotonic networks and constrained learning for interpretable classification and weakly supervised anomaly detection, Pattern Recognition 160 (2025) 111186. doi:10.1016/j.patcog.2024.111186.

[45] S. Yang, J. Wen, B. Fang, Fg-moe: Heterogeneous mixture of experts model for fine-grained visual classification, Pattern Recognition 175 (2026) 113050. doi:10.1016/j.patcog.2026.113050.

[46] M. Bello, R. Amador, M.-M. García, J. Del Ser, P. Mesejo, Ó. Cordón, The level of strength of an explanation: A quantitative evaluation technique for post-hoc xai methods, Pattern Recognition 161 (2025) 111221. doi:10.1016/j.patcog.2024.111221.

[47] W. Jiang, et al., Asymptotic properties of mixture-of-experts models, Neurocomputing 74 (9) (2011) 1444– 1449. doi:10.1016/j.neucom.2010.12.007.

[48] N. Bariletto, H. Nguyen, N. Ho, A. Rinaldo, On bayesian softmax-gated mixture-of-experts models, arXiv preprint arXiv:2604.20551 (2026). doi:10.48550/arXiv.2604.20551.