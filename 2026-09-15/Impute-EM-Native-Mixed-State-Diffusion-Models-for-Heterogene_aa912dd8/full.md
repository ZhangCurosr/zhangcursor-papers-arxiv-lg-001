# Impute-EM: Native Mixed-State Diffusion Models for Heterogeneous Data Imputation

Sergei Kholkin

Applied AI Institute

Kirill Sokolov

Applied AI Institute, Moscow State University

Dmitry Baranchuk

Yandex Research

Evgeny Burnaev

Applied AI Institute, AXXX

Alexander Korotin

Applied AI Institute, AXXX

Abstract—Missing values are ubiquitous in heterogeneous data mining, where numerical, categorical, and binary variables often coexist. Many imputation methods, especially diffusion-based ones, treat discrete variables through continuous surrogates such as onehot relaxations rather than modeling them natively. This creates a mismatch between the model state space and the mixed discrete and continuous structure of the data. We propose Impute-EM, an Expectation Maximization style framework that alternates between imputing missing entries with the current model and refitting a diffusion backbone on completed data. We instantiate Impute-EM with native mixed-state diffusion backbones for heterogeneous data, combining Gaussian and masked categorical components without one-hot relaxations. In exact settings, we characterize the update and show that the observed mask-indexed marginals match the targets at the limit, while making explicit that the full data distribution is generally non-identifiable from incomplete observations alone. Empirically, Impute-EM delivers the best distributional fidelity on mixed-type tabular imputation, on which downstream modeling relies, with text imputation serving as a controlled validation of the native discrete backbone.

## I. INTRODUCTION

Missing data is ubiquitous, and reliable imputation is a central step in data-driven systems: imputed tables are often reused for downstream modeling and analysis, so preserving the marginal and dependency structure of the data matters alongside recovering individual entries. This is especially important for heterogeneous data, where numerical, categorical, and binary variables often coexist across tabular data [1], audio data [2], and graph data [3].

Prior work on data imputation spans classical statistical methods, conventional machine learning, and modern deep models. Early approaches include nearest-neighbor imputation [4] and parametric density models such as Gaussian mixtures [5]. Recent predictive methods learn missing entries from observed entries using masking or graph-based structure [1, 3, 6]. Generative methods instead model the joint distribution of observed and missing entries and impute by conditional sampling [7–11]. These methods are attractive for heterogeneous imputation because they can represent uncertainty over plausible completions rather than returning only a single deterministic fill-in.

Despite the success of modern generative models, including diffusion models [12], training generative imputers from incomplete data remains challenging. A common approach is to optimize incomplete-data objectives that evaluate likelihood or reconstruction losses only on observed entries [8, 10]. These objectives are practical, but they do not directly specify how the unobserved coordinates should be completed during training. The Expectation Maximization algorithm provides a natural alternative by alternating between imputing missing values under the current model and updating the model on the resulting completed data [7, 13–15]. However, its use with native discrete and mixed-state diffusion backbones remains underexplored.

Moreover, recent diffusion-based imputation methods mostly rely on continuous diffusion models [10, 13], often representing categorical variables as one-hot vectors or continuous relaxations. This creates a mismatch between the model state space and the native structure of discrete variables, and it can become inefficient for high-dimensional categorical data [16]. Native discrete diffusion models and mixed continuous–categorical diffusion models are therefore a natural fit for heterogeneous imputation, but their use in this setting remains limited [1–3].

Our contributions are:

• We formulate missing-data learning as matching maskindexed observed marginals and derive the corresponding EM-style update, see Section IV-A. The formulation makes explicit that the full data distribution is generally non-identifiable from incomplete observations alone.

• We instantiate the EM-style update with native mixed-state diffusion backbones for heterogeneous data, combining Gaussian and masked categorical components without one-hot relaxations, see Section IV-B.

• We evaluate Impute-EM on mixed-type tabular imputation as the main experiment, with text imputation as a controlled validation of the native discrete backbone in isolation, and study how EM-style refinement affects diffusion-based imputation quality.

a) Notation: Let $\mathcal { X } = \mathbb { R } ^ { d _ { 1 } } \times \mathcal { S }$ , with $\begin{array} { r } { \pmb { S } = \prod _ { \ell = 1 } ^ { d _ { 2 } } \pmb { S } _ { \ell } } \end{array}$ , be a mixed continuous-discrete data space, where $\mathbb { R } ^ { d _ { 1 } }$ contains continuous coordinates and S collects the discrete coordinates. Each $S _ { \ell }$ is a finite discrete set, such as a categorical or binary variable. We write $D = d _ { 1 } + d _ { 2 }$ for the total number of variables and denote a sample by $x = ( x ^ { ( 1 ) } , \ldots , x ^ { ( D ) } ) \in \mathcal { X }$ . The set of probability distributions on X is denoted by $\mathcal { P } ( \mathcal { X } )$ . A mask is a binary vector $m \in \{ 0 , 1 \} ^ { D }$ , where $m _ { j } = 1$ indicates that coordinate j is observed and $m _ { j } = 0$ indicates that it is missing.

We write $x _ { m } = \{ x ^ { ( j ) } : m _ { j } = 1 \}$ for observed coordinates and $\overline { { x } } _ { m } = \{ x ^ { ( j ) } : m _ { j } = 0 \}$ for missing coordinates. For a distribution $p \in \mathcal { P } ( \mathcal { X } ) , p _ { m }$ denotes its marginal distribution on the observed coordinates indexed by $m .$

## II. BACKGROUND

In this section we do define the learning from missing data problem and describe the methodology of diffusion models, which are our algorithms backbone generative model.

## A. Problem Setup

Let $p ^ { \ast } ( x ) \in \mathcal { P } ( \mathcal { X } )$ denote the data distribution and let $x \in \mathcal { X }$ be a clean sample. In our setting, full samples are not observed directly. Instead, observations are partially revealed through observation masks. Such a mask is a binary vector $m \in \{ 0 , 1 \} ^ { D }$ , where $m _ { j } ~ = ~ 1$ means that coordinate $j$ is observed and $m _ { j } = 0$ means that it is missing. The mask partitions x into observed and missing subvectors:

$$
x _ { m } : = \{ x ^ { ( j ) } : m _ { j } = 1 \} , \qquad { \overline { { x } } } _ { m } : = \{ x ^ { ( j ) } : m _ { j } = 0 \} .\tag{1}
$$

We model masks as random variables, with

$$
\mu ( m ) \in \mathcal { P } ( \mathcal { M } ) , \quad \mathcal { M } \subseteq \{ 0 , 1 \} ^ { D }\tag{2}
$$

which reflects real-world settings where missingness is stochastic. The observed distribution is the mask-induced marginal:

$$
p _ { m } ( x _ { m } ) = \int p ^ { * } ( x _ { m } , \overline { { x } } _ { m } ) d \overline { { x } } _ { m } , \quad x \sim p ^ { * } , \ m \sim \mu .\tag{3}
$$

When clean sample entries are being dropped we inherently lose information about the clean data distribution $p ^ { * } ( x )$ and do have only the access to its partial marginals $p _ { m } ^ { * } ( x _ { m } )$ which in general are insufficient to uniquely identify the $p ^ { * }$ distribution [7]. The goal of the data imputation problem is to learn a conditional imputer that samples plausible missing values $\overline { { x } } _ { m }$ given observed values $x _ { m }$ , while matching the observed mask-indexed marginals available in the incomplete data. Two empirical setups are commonly considered: in-sample imputation, where the model imputes the training data, and out-of-sample imputation, where it is applied to unseen data. B. Diffusion models

Diffusion models learn a data-generating reverse process by inverting a prescribed forward corruption [12, 17]. The construction depends on the state space: continuous diffusion adds Gaussian noise for real-valued features [12, 17]; masked discrete diffusion operates natively on categorical variables [16, 18]; mixed-state diffusion combines both for heterogeneous data [19]. For imputation, this matters because observed coordinates should stay in their native domains while missing coordinates are resampled conditionally. Below, we describe the DDPM [12] and MDM [18] formulations adopted in our work.

a) Training: For continuous data, such as real-valued tabular features, diffusion models such as DDPM [12] or Score SDE [17] define a forward process that gradually adds Gaussian noise to a clean sample, from clean data $x _ { 0 }$ to nearly pure noise $x _ { T } \mathbf { \cdot }$

$$
\begin{array} { r } { x _ { t } = \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , \epsilon \sim \mathcal { N } ( 0 , I ) . } \end{array}\tag{4}
$$

The reverse transitions are learned through the DDPM training objective, which is commonly written as the noiseprediction loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d d p m } } ( \theta ) = \mathbb { E } _ { x _ { 0 } , t , \epsilon } \left[ \left. \epsilon - \epsilon _ { \theta } ( x _ { t } , t ) \right. ^ { 2 } \right] . } \end{array}\tag{5}
$$

For discrete data, such as text, categorical variables, or tokens, Gaussian noise is not naturally defined. Instead, masked diffusion models corrupt samples through masked transitions:

$$
q _ { \mathrm { M D M } } ( x _ { t } = a \mid x _ { 0 } ) = \alpha _ { t } { \bf 1 } \{ a = x _ { 0 } \} + ( 1 - \alpha _ { t } ) { \bf 1 } \{ a = M \} ,
$$

where $a \in S \cup \{ M \}$ <sup>(6)</sup> and M is a special mask token. The reverse transitions are trained by maximizing the probability assigned to the original discrete value:

$$
\mathcal { L } _ { \mathrm { m d m } } ( \theta ) = \mathbb { E } _ { x _ { 0 } , t , x _ { t } } \left[ - \log \hat { p } _ { \theta } ( x _ { 0 } \mid x _ { t } ) \right] , x _ { t } \sim q _ { \mathrm { M D M } } ( x _ { t } \mid x _ { 0 } ) .
$$

This formulation keeps discrete variables finite-valued throughout the diffusion process.

b) Generation process: In the continuous case, the learned reverse process then estimates the denoising direction at each noise level, ultimately transforming pure noise into a realistic data point:

$$
p _ { \theta } ( x _ { t - 1 } \mid x _ { t } ) = \mathcal { N } \big ( x _ { t - 1 } ; \mu _ { \theta } ( x _ { t } , t ) , \sigma _ { t } ^ { 2 } I \big ) ,
$$

$$
\mu _ { \theta } ( x _ { t } , t ) = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( x _ { t } - \frac { \beta _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \epsilon _ { \theta } ( x _ { t } , t ) \right) .\tag{7}
$$

In the discrete case, the learned reverse process iteratively predicts masked entries conditioned on the current corrupted state:

$$
\begin{array} { r l } & { p _ { \theta } ( x _ { t - 1 } \mid x _ { t } ) } \\ & { \quad = \mathrm { C a t } \Bigg ( x _ { t - 1 } ; \frac { ( 1 - \alpha _ { t - 1 } ) M + ( \alpha _ { t - 1 } - \alpha _ { t } ) p _ { \theta } ( x _ { 0 } \mid x _ { t } ) } { 1 - \alpha _ { t } } \Bigg ) . } \end{array}
$$

c) Mixed-state diffusion models: Many real-world appli cations contain both continuous and discrete variables. Mixed space diffusion models combine the two constructions above by applying Gaussian diffusion to continuous coordinates and masked categorical diffusion to discrete coordinates. The forward process is obtained by concatenating the corresponding corruptions, and the reverse process jointly denoises or resamples both parts. Training typically uses a summed objective,

$$
\mathcal { L } _ { \mathrm { m i x e d } } ( \theta ) = \mathcal { L } _ { \mathrm { d d p m } } ( \theta ) + \mathcal { L } _ { \mathrm { m d m } } ( \theta ) ,
$$

possibly with task-dependent weights between the continuous and discrete components.

## III. RELATED WORK

Data imputation methods aim to estimate plausible values for missing entries, or a conditional distribution over them, when values are lost during acquisition or were never measured. Imputation is central for heterogeneous tabular data [1, 7], token-based audio restoration [2], graph attributes [3], and other data mining settings with mixed variable types.

a) Classical methods: Early imputation methods include nearest-neighbor imputation [4], Gaussian mixtures [5], and HyperImpute automatic model selection [21]. More recently, deep generative models have framed imputation as conditional generation of missing values from observed entries. VAEbased methods such as MIWAE [8] and GP-VAE [22] model incomplete observations with latent variables, while GAIN [7] uses adversarial training. Diffusion-based approaches such as TabCSDI [11] and MissDiff [10] sample missing values conditional on observed entries. Many contemporary generative approaches, including ReMasker [1], MissDiff [10], and MIWAE [8], can be viewed as optimizing an incomplete-data likelihood, where the training objective is evaluated only on the observed components of each sample.

<table><tr><td>Method</td><td>Backbone Model</td><td>Practical Discrete Modeling</td><td>Empirical focus</td></tr><tr><td>GAIN [7]</td><td>GAN</td><td></td><td>Tabular imputation</td></tr><tr><td>MIRI [15]</td><td>Rectified Flow</td><td>X</td><td>Tabular imputation</td></tr><tr><td>MCFlow [9]</td><td>Normalizing Flow</td><td>√X</td><td>Tabular imputation</td></tr><tr><td>Diffputer [13]</td><td>Score Diffusion [17]</td><td> $\checkmark \pmb { x }$ </td><td>Tabular imputation</td></tr><tr><td>DiffEM [20]</td><td>Score Diffusion [17]</td><td>X</td><td>Image reconstruction</td></tr><tr><td>Impute-EM (ours)</td><td>General Diffusion [12, 18, 19]</td><td> $\checkmark$ </td><td>Text &amp; mixed-type tabular</td></tr></table>

TABLE I: Conceptual comparison of iterative, i.e., EM-like, imputation methods. The discrete variables column indicates whether the method handles discrete variables directly, where ✓✗ denotes a one-hot encoding workaround. The empirical focus column reports each method’s main evaluation domain.

b) Iterative methods: Iterative refinement is another long standing strategy for imputation, where initial guesses for missing values are repeatedly improved. The EM algorithm is the classical example [14], although early uses often relied on simple distributions such as Gaussian mixtures, Bernoulli models, or multinomial models [5]. Related modern methods include MCFlow, which iteratively imputes with normalizing flows [9], IGRM, which updates graph based friend networks during training [6], and HyperImpute, which iteratively refines model selection and imputations [21]. MIRI also uses an EM based iterative algorithm, but views imputation as reducing mutual information between the completed data and the missingness mask with rectified flows [15].

DiffPuter trains an EM-style iterative improvement procedure with a diffusion model for missing-data imputation [13]. DiffEM learns continuous diffusion models from corrupted observations by alternating conditional reconstruction in the E step and score matching in the M step [20]. DiffEM’s EM analysis covers general state spaces and guarantees observation consistency without identifiability, while its diffusion models and experiments remain continuous. Our work is closest to DiffPuter and DiffEM: these methods show that iterative reconstruction and model refitting can improve diffusion-based imputation or reconstruction, but they primarily use continuous diffusion backbones. Our focus is different: we instantiate the same EM-style principle with native mixed-state diffusion backbones for heterogeneous data.

AugMask [23] also studies training tabular diffusion models, including TabDiff, from incomplete data. Unlike our iterative EM approach, AugMask constructs fixed stochastic completions with auxiliary feature-wise models and uses them only as conditioning context under an observed-only denoising loss.

c) Disclaimer: This work was completed and submitted to ICDM 2026 by June 6, 2026, but could not be posted on arXiv during the review process under the conference policy.

## IV. METHOD

This section presents Impute-EM. We first formulate missing data learning as matching the observed marginals induced by a family of masks Section IV-A. We then describe the practical instantiation with native state space backbones Section IV-B. Finally, we characterize the exact nonparametric update and show convergence of the observed marginals as a supporting theoretical result Section IV-C.

## A. Learning from missing data with Impute-EM

Given the data distribution $p ^ { \ast } ( x ) \in \mathcal { P } ( \mathcal { X } )$ with the observation masks probability distribution $\mu ( m ) \in \mathcal P ( \mathcal M )$ , where masks are $\bar { m \in \{ 0 , 1 \} ^ { D } }$ . Following problem setup Section II-A, we observe only partial random vectors $x _ { m } \sim p _ { m } ^ { * } ( x _ { m } )$ The full distribution $p ^ { * } ( x )$ is generally not identifiable from incomplete observations alone [7]. Therefore, our objective is to learn a model $p ( x )$ whose observed marginals agree with the target marginals $p _ { m } ^ { * } ( x _ { m } )$ for the mask family used during training.

We treat $\mu ( m )$ as independent of x (missing completely at random, MCAR), which ensures the observed marginals $p _ { m } ^ { * } ( x _ { m } )$ are well-defined. In practice, Impute-EM can also be applied under MAR or MNAR: the E-step and M-step remain computationally valid, though under MNAR the observed marginals are biased by the missingness mechanism and the theoretical guarantee does not apply.

This leads to the following optimization problem:

$$
\begin{array} { r l } {  { \underset { p } { \operatorname { a r g m i n } } \mathbb { E } _ { m \sim \mu } [ \mathrm { K L } ( p _ { m } ^ { * } ( x _ { m } ) \| p _ { m } ( x _ { m } ) ) ] } \quad } & { } \\ & { = \underset { p } { \operatorname { a r g m a x } } \mathbb { E } \underset { x _ { m } \sim p _ { m } ^ { * } ( x _ { m } ) } { \sim } [ \log p _ { m } ( x _ { m } ) ] + C _ { 1 } . } \end{array}\tag{8}
$$

Where $C _ { 1 }$ is an entropy term that does not depend on $p .$ To solve this problem we can follow the variational inference approach and introduce a variational distribution $q ( \overline { { x } } _ { m } | x _ { m } )$ over the missing entries $\overline { { x } } _ { m }$ , which are treated as latent variables:

$$
\log p ( x _ { m } ) \geq \mathbb { E } _ { \overline { { x } } _ { m } \sim q ( \overline { { x } } _ { m } | x _ { m } ) } \left[ \log \frac { p ( \overline { { x } } _ { m } , x _ { m } ) } { q ( \overline { { x } } _ { m } | x _ { m } ) } \right] = \mathcal { L } ( p , q | x _ { m } ) ,\tag{9}
$$

where the argmax ${ \bf \Phi } _ { q } \mathcal { L } ( p , q )$ solution would be $q ( \overline { { x } } _ { m } | x _ { m } ) =$ $p ( \overline { { x } } _ { m } | x _ { m } )$ , for each m, which gives us the Expectation step. Then the full objective with lower bound:

$$
\begin{array} { r l } & { \mathbb { E } _ { m \sim \mu , \ x _ { m } \sim p _ { m } ^ { * } } \left[ \log p ( x _ { m } ) \right] \geq \mathbb { E } _ { m , \ x _ { m } \sim p _ { m } ^ { * } } \left[ \mathcal { L } ( p , q \mid x _ { m } ) \right] } \\ & { \phantom { \mathbb { E } _ { m \sim p _ { m } ^ { * } } } = \mathbb { E } _ { \phantom { \mu } \pi , \ x _ { m } \sim p _ { m } ^ { * } } \left[ \log p ( \overline { { x } } _ { m } , x _ { m } ) \right] + C _ { 2 } . } \end{array}\tag{10}
$$

where $C _ { 2 }$ is an entropy term that does not depend on $p .$

With fixed q, the rhs of (10) is just the log likelihood for the $p ( \overline { { x } } _ { m } , x _ { m } )$ w.r.t. data distribution $x = \{ \overline { { x } } _ { m } , x _ { m } \} \sim$ $q ( \overline { { x } } _ { m } | x _ { m } ) p _ { m } ^ { * } ( x _ { m } )$ . Then we can find the solution for argmax<sub>p</sub> $, { \mathcal { L } } ( p , q )$ by the regular likelihood optimization for the model $p ( x )$ , which gives us the Maximization step. We call the resulting EM algorithm Impute-EM:

• Expectation step: $q ^ { n } ( \overline { { x } } _ { m } | x _ { m } ) \gets p _ { m } ^ { n } ( \overline { { x } } _ { m } | x _ { m } )$

• Maximization step:

$$
\begin{array} { r l } & { p ^ { n + 1 } \gets \underset { p } { \mathrm { a r g m a x } } \mathbb { E } \underset { m \sim \mu , \ : x _ { m } \sim p _ { m } ^ { * } } { \mathrm { m } } \left[ \log p ( \overline { { x } } _ { m } , x _ { m } ) \right] } \\ & { \quad \quad \quad = \underset { p } { \mathrm { a r g m a x } } \mathcal { L } ( p , q ) . } \end{array}
$$

## B. Native Diffusion Backbones for Impute-EM

We instantiate the EM-style update with diffusion backbones that operate in the native state space of each variable, spanning continuous, discrete, and mixed types. The E-step samples missing entries from the current backbone and the M-step retrains it on the completed data.

a) Expectation step: The conditional inference of the current model $p _ { \theta }$ has to be performed at E-step. By our construction we have a conditional diffusion $p _ { \theta } ( \overline { { x } } _ { m } | x _ { m } )$ , which in continuous case would be the conditional mean predictor $\mu _ { \boldsymbol { \theta } } ( \overline { { x } } _ { m , t } , t | x _ { m } )$ or in discrete case the conditional data predictor $\hat { p } _ { \theta } ( \overline { { x } } _ { m } \mid \overline { { x } } _ { m , t } , x _ { m } )$ . In both cases the conditional generation would be just backward diffusion process inference with conditional noise predictor and conditional data predictor, similarly to methodology described in Section II-B. In detail, for the continuous case:

$$
\begin{array} { r l } & { p _ { \theta ^ { n } } ( \overline { { x } } _ { m , t - 1 } \mid \overline { { x } } _ { m , t } , x _ { m } ) } \\ & { \qquad = \mathcal { N } \big ( \overline { { x } } _ { m , t - 1 } ; \mu _ { \theta ^ { n } } ( \overline { { x } } _ { m , t } , t \mid x _ { m } ) , \sigma _ { t } ^ { 2 } I \big ) . } \end{array}
$$

For the discrete case:

$$
\begin{array} { l } { \displaystyle p _ { \theta ^ { n } } \bigl ( \overline { { x } } _ { m , t - 1 } ~ \big | ~ \overline { { x } } _ { m , t } , x _ { m } \bigr ) = } \\ { \displaystyle \mathrm { C a t } \biggl ( \overline { { x } } _ { m , t - 1 } ~ \biggl | ~ \frac { \bigl ( 1 - \alpha _ { t - 1 } \bigr ) M + \bigl ( \alpha _ { t - 1 } - \alpha _ { t } \bigr ) \hat { p } _ { \theta ^ { n } } \bigl ( \overline { { x } } _ { m } ~ \big | ~ \overline { { x } } _ { m , t } , x _ { m } \bigr ) } { 1 - \alpha _ { t } } \biggr . } \end{array}
$$

where $\alpha _ { t }$ coefficients which are determined by the forward process construction. Then such the probability distribution ${ \mathcal { D } } ^ { n } ( x )$ , would be the result of E-step:

$$
\begin{array} { r } { \mathcal { D } ^ { n } ( x ) = \mathbb { E } _ { m \sim \mu } [ \mathcal { D } ^ { n } ( \overline { { x } } _ { m } , x _ { m } ) ] = \mathbb { E } _ { m \sim \mu } [ p _ { \theta ^ { n } } \left( \overline { { x } } _ { m } \vert x _ { m } \right) p _ { m } ^ { * } ( x _ { m } ) ] } \end{array}\tag{11}
$$

In practice, thefinite dataset is used instead of ${ \mathcal { D } } ^ { n } ( x )$ . Notice, that E-step requires sampling from the learned model, which introduces additional computational overhead for diffusion models [12, 18]. Similar overhead is inherent to EM-based methods with diffusion backbones [13, 15, 20].

b) Maximization step: During the M step we have to fit the family of conditional generative models $p _ { \theta } ( \overline { { x } } _ { m } | x _ { m } )$ to ${ \mathcal { D } } ^ { n } ( x )$ data via likelihood maximization or equivalent optimize the following loss function:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c o n d } } ( \theta ) = - \mathbb { E } _ { x \sim \mathcal { D } ^ { n } ( x ) , m \sim \mu } \left[ \log p _ { \theta } ( \overline { { x } } _ { m } | x _ { m } ) \right] } \end{array}\tag{12}
$$

In practice, learning of conditional diffusion models is long standing practice [16] and is done by simply adding another input to the neural network and slightly modifying the loss function. Final loss function for continuous state space

Algorithm 1: Impute-EM   
Input: Initial parameters $\theta ^ { 0 }$ , observed data $\{ x _ { m } ^ { i } \} _ { i = 1 } ^ { L }$   
number of EM iterations N   
Output: Refined parameters $\theta ^ { N }$   
Initialize $n \gets 0 ;$   
while $n < N$ do   
Initialize dataset ${ \mathcal { D } } ^ { n }  \emptyset ;$   
for each $x _ { m } ^ { i } \in \{ x _ { m } ^ { i } \} _ { i = 1 } ^ { L }$ do   
Sample a batch   
${ x _ { i } ^ { n } } ^ { \setminus } = x ^ { i } \sim p _ { \theta ^ { n } } ( \overline { { x } } _ { m } \mid x _ { m } ^ { i } ) p _ { m } ^ { * } ( x _ { m } ^ { i } ) ;$   
$/ /$ Inference diffusion $\theta ^ { n }$   
Update dataset $\mathcal { D } ^ { n }  \mathcal { D } ^ { n } \cup x _ { i } ^ { n } ;$   
$\theta  \theta ^ { n } ;$   
while not converged do   
Update θ with an optimization step on   
$\begin{array} { r } { \mathcal { L } _ { \mathrm { c o n d } } ( \theta ) = - \mathbb { E } _ { x \sim \mathcal { D } ^ { n } , k \sim \mu } \left[ \log p _ { \theta } ( \overline { { x } } _ { k } | x _ { k } , k ) \right] ; } \end{array}$   
θ<sup>n+1</sup> ← θ;   
n ← n + 1;   
return $\theta ^ { N } ;$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c o n d } } ^ { \mathrm { d d p m } } ( \theta ) = \mathbb { E } \underset { t \sim \mathcal { U } [ 0 , T ] , \epsilon \sim \mathcal { N } ( 0 , I ) } { \sim } \left[ \Vert \epsilon - \epsilon _ { \theta } ( \overline { { x } } _ { m , t } , t | x _ { m } ) \Vert ^ { 2 } \right] } \\ { \quad \quad \quad \quad \quad \quad t \sim \mathcal { U } [ 0 , T ] , \epsilon \sim \mathcal { N } ( 0 , I ) } \end{array}\tag{13}
$$

where $\overline { { x } } _ { m , t } = \sqrt { \bar { \alpha } _ { t } } \overline { { x } } _ { m } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon$ represents the forward noising process, and $\epsilon _ { \theta }$ is the model prediction. The conditional structure is enforced by providing the mask m and observed values $x _ { m }$ as inputs to the network.

For discrete state space, we notice that MDM is naturally suited for the imputation problem since it already learns a family of conditional generative models by design [24] and carries the information about the masked values by special masked token. In that light we can use the neural network architecture as and just slightly alter the training procedure:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { c o n d } } ^ { \mathrm { m d m } } ( \theta ) = \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } ^ { n } ( \boldsymbol { x } ) , m \sim \mu } \left[ \mathrm { C E } \big ( e _ { \overline { { \boldsymbol { x } } } _ { m } } , \hat { p } _ { \theta } ( \cdot | \boldsymbol { x } _ { m } ) \big ) \right] } \\ & { \qquad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \qquad \quad \quad \quad \quad \quad \quad \frac { \overline { { t } } \sim \mathcal { U } [ [ 0 , T ] ] } { \overline { { \boldsymbol { x } } } _ { m , t } \sim q _ { \mathrm { M D M , t } } ( \overline { { \boldsymbol { x } } } _ { m , t } | \overline { { \boldsymbol { x } } } _ { m } ) } } \end{array}\tag{14}
$$

c) Heterogeneous mixed-state diffusion: In our implementation, the mixed state space model is built by concatenating continuous and discrete diffusion components, and the neural networks take the full multimodal input in one forward pass. The joint training objective would be just the sum of continuous and discrete losses:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c o n d } } ( \theta ) = \mathcal { L } _ { \mathrm { c o n d } } ^ { \mathrm { d d p m } } ( \theta ) + \mathcal { L } _ { \mathrm { c o n d } } ^ { \mathrm { m d m } } ( \theta ) , } \end{array}\tag{15}
$$

The other details including the full joint training objective and reverse sampling updates are delivered in Appendix B.

d) Initial model: Initialization is an important practical choice for Impute-EM. We set $p _ { 0 } ^ { \theta }$ to diffusion models trained via the incomplete likelihood objectives, see Appendix B for more details. On continuous domain we use training close to MissDiff [10]. On discrete domain we train a masked diffusion model as an incomplete data masked autoencoder, analogous to ReMasker [1].

![](images/acafd4e56e6fd9ab3f50ecc8417ab29f0e72e24c25212f56da5a14e6ba3f9928.jpg)  
(a) PPL ↓ versus mask rate. ReMasker MDLM and Clean data MDLM as baselines, compared with MDLM Impute-EM (ours).

![](images/dc0749e217651729cc5b6bf12e91739fc8a6416ec7b14697e51acbfc25848f2a.jpg)  
(b) PPL ↓ versus number of Impute-EM iterations across mask rates.  
Fig. 1: Text data imputation on text8 under MCAR per token.

We summarize the resulting practical procedure in Algorithm 1, which alternates E-step conditional diffusion sampling with M-step conditional training.

## C. Observed-marginal consistency of the exact update

We now characterize the exact nonparametric operator underlying Algorithm 1 and state the observed-marginal consistency result that supports it. The analysis is idealized: it assumes exact E-step sampling and exact M-step maximization. The practical diffusion approximations introduced in Section IV-B relax both.

Proposition 1 (Impute-EM update). Let $\mathcal { M } \subseteq \{ 0 , 1 \} ^ { D }$ be the family of observed index sets, and $\mu ( m ) \in \mathcal P ( \mathcal M )$ be the positive probability distribution of observed indices. For a joint distribution $p \in { \mathcal { P } } ( { \mathcal { X } } )$ , define the exact Impute-EM update.

$$
p ^ { n + 1 } ( x ) = \mathbb { E } _ { m \sim \mu } \left[ p _ { m } ^ { * } ( x _ { m } ) p ^ { n } ( \overline { { x } } _ { m } \mid x _ { m } ) \right] ,\tag{16}
$$

where $p _ { m } ^ { * }$ denotes the target marginal on the observed coordinates indexed by m.

See the proof in Appendix A. This proposition gives the closed-form update operator. The following theorem shows that at the limit of iterations the observed marginals converge to the targets.

Theorem 1 (Observed-marginal consistency of exact Impute-EM). Let $p ^ { * } ( x ) \in \mathcal { P } ( \mathcal { X } )$ be the data probability distribution, $\mathcal { M } \subseteq \{ 0 , 1 \} ^ { D }$ be the family of observed index sets and $\mu ( m ) \in \mathcal P ( \mathcal M )$ be the positive probability distribution of observed indices. Define the set of (8) minimizers:

$$
\mathcal { C } : = \{ q \in \mathcal { P } ( \mathcal { X } ) : q _ { m } = p _ { m } ^ { * } \forall m \in \mathcal { M } \} .\tag{17}
$$

![](images/28e1c840218d8a3b2baf53a4112cac670613a1658886a94008ae6291c448624e.jpg)  
Fig. 2: Overall Density ↑ under MCAR versus missing rate (Adult, Shoppers, Default, and Beijing). Overall Density is (Trend + Shape)/2.

Define starting probability distribution $p ^ { 0 } ( x )$ , such that there exists $r \in \mathcal { C }$ with $\mathrm { K L } ( r \| p ^ { 0 } ) < \infty ,$ , and define iterative updates:

$$
p ^ { n + 1 } ( x ) = \mathbb { E } _ { m \sim \mu } \left[ p _ { m } ^ { \ast } ( x _ { m } ) p ^ { n } ( \overline { { x } } _ { m } \mid x _ { m } ) \right] .
$$

Then, for every $m \in { \mathcal { M } } ,$

$$
\mathrm { K L } ( p _ { m } ^ { * } \| p _ { m } ^ { n } ) \to 0 \qquad a s \ n \to \infty .\tag{18}
$$

The proof is presented in Appendix A. The Theorem 1 provides that if the Impute-EM procedure is ran long enough, the $p ^ { n }$ is guaranteed to that satisfy the observed marginals $p _ { m } ^ { * } f o r$ all $m \in \mathcal { M }$ . This property is independent of the start of iteration $p _ { 0 }$ and the mask probability distribution $\mu ( m )$ under slight assumptions. However, there could be many such possible distributions $p$ with fitting marginals, $\mathrm { i } . \mathrm { e } . , p \in \mathcal { C }$ , and our theoretical result does not provide any information on what particular distribution would be the result of Impute-EM iteration, i.e., it is not necessarily $p ^ { * }$ . The theorem concerns the exact nonparametric operator. Its diffusion implementation is a parametric surrogate, so the result supports the target update but does not guarantee convergence of the finite-sample implementation.

## V. EXPERIMENTS

In this section, we evaluate Impute-EM on heterogeneous, mixed-type data imputation. Our central claim concerns this mixed-type setting, so the main experiment is on tabular data with TabDiff [19], which exercises the full mixed continuouscategorical backbone. Since, to our knowledge, native discrete diffusion has not previously been used as an EM-style trainfrom-incomplete-data backbone for imputation, we additionally study text imputation with a Masked Diffusion Language Model [16] on text8 as a controlled validation of this backbone in isolation, the degenerate case where all coordinates are categorical $( d _ { 1 } \ = \ 0 )$ . Our problem statement is generative, the goal is to recover the imputation probability distribution $p ( \overline { { x } } _ { m } | x _ { m } )$ rather than individual entries, so we evaluate with generative metrics (PPL for text, Trend and Shape for tabular data), and additionally report downstream ML-efficiency for practical utility.

![](images/1e6db0909ef75badf64f0409f54dee911f507dc2fe929a485c5df12d2e55208d.jpg)  
Fig. 3: ML-efficiency absolute deviation ↓ under MCAR versus missing rate (Adult, Shoppers, Default, and Beijing), relative to the MLE on the clean dataset. Lower is better. Beijing MLE is multiplied by 0.1 to keep the scale comparable across datasets.

## A. Discrete data: text

We use the ‘text8‘ dataset [25], tokenized at the character level, with vocabulary size 27 and sequence length 128. We consider several missing rates $p \in \{ 0 . 2 , 0 . 3 , 0 . 5 \}$ and randomly replace tokens in the dataset samples with a special mask token with probability p (MCAR per token). The missing tokens are generated only once for each data sample. Training is done on the train split and the imputation is done on validation split, following out-of-sample imputation setup.

We compare our method with the ReMasker MDLM, an MDLM trained as a masked autoencoder on observed data only via the incomplete likelihood following the ReMasker [1] methodology. The ReMasker MDLM and the initialization of Impute-EM were trained for 500k iterations, while MDLM Impute-EM (ours) ran 5 EM iterations of 20k training iterations each. All the experimental details can be seen in Appendix D.

Our metric is perplexity (PPL, lower is better), the exponen tial of the average negative log-likelihood per token. PPL scores how well the model’s per-token predictive distribution matches held-out tokens, which makes it both the standard distributional metric for masked language modeling [16] and directly tied to token-wise accuracy, hence suitable for measuring imputation ability. See Appendix D for the formal definition. The results are shown in Figure 1. It is evident that missing rates more than $p = 0 . 2$ yield a degradation in model performance which results in significant increase in perplexity. However, MDLM Impute-EM (ours) allows to recover the losses in perplexity w.r.t. Clean MDLM model, which is clearly visible from the Figure 1a. Additionally, we study the behavior of Impute-EM by the iterations number and report the results in Figure 1b.

It is clearly visible that Impute-EM iterations improve the imputation quality and the first few Impute-EM steps yield the most significant improvements.

## B. Heterogeneous data: tabular

Having validated the discrete backbone in isolation, we now turn to the full heterogeneous setting. We combine the naturally mixed-state TabDiff backbone [19] with Impute-EM, avoiding the one-hot relaxations used by some prior work [10, 11].

We consider four mixed-type datasets with both numerical and categorical features: Adult, Shoppers, Default, and Beijing (see Appendix D for descriptions). As baselines, we compare state-of-the-art approaches: DiffPuter [13] as a diffusion-based EM method, ReMasker [1] as a masked autoencoder baseline, and MissDiff [10] as an incomplete likelihood diffusion approach. Both training and imputation take place on the training data (in-sample imputation), and we run Tabdiff Impute-EM for 5 iterations. Further implementation details are in Appendix D.

For the wide performance ablation we follow the standard tabular imputation benchmark under the Missing Completely At Random (MCAR) setting, varying the missing rate across $p \in \{ 0 . 2 , 0 . 3 , 0 . 5 , 0 . 7 \}$ . We report Overall Density (the average of Shape and Trend marginal and dependency scores [19]) for generative fidelity, and ML-efficiency for downstream utility. Both are described in Appendix D.

To assess robustness under more challenging missingness, we additionally evaluate all methods under the Missing Not At Random (MNAR) setting on the same four datasets at a single high mask rate $p = 0 . 7$ . The MNAR mechanism is described in Appendix D. All models are trained with the same hyperparameters as in the MCAR experiments. Figure 4 reports Overall Density and ML-efficiency deviation, and the discriminative-metric counterpart is in Appendix C. TabDiff Impute-EM (ours) still holds its positions in this setting. Additional single-run results under MAR are reported in Appendix C-B.

<table><tr><td>Method</td><td>Adult</td><td>Shoppers</td><td>Default</td><td>Beijing</td><td>Avg.</td></tr><tr><td>ReMasker</td><td>0.194</td><td>0.143</td><td>0.155</td><td>0.253</td><td>0.186</td></tr><tr><td>DiffPuter</td><td>0.136</td><td>0.133</td><td>0.118</td><td>0.207</td><td>0.148</td></tr><tr><td>HyperImpute</td><td>0.093</td><td>0.086</td><td>0.089</td><td>0.165</td><td>0.108</td></tr><tr><td>MissDiff</td><td>0.154</td><td>0.196</td><td>0.269</td><td>0.219</td><td>0.210</td></tr><tr><td>TabDiff Impute-EM (ours)</td><td>0.060</td><td>0.050</td><td>0.071</td><td>0.102</td><td>0.071</td></tr></table>

TABLE II: Overall Density error ↓ on tabular datasets under MCAR. For each dataset the error is averaged over the mask rates. The last column reports the averaged error. The best method is highlighted in bold.

To isolate the contribution of Impute-EM from the TabDiff backbone, Appendix C reports an iteration-dynamics analysis across different metrics, showing that Impute-EM iterative updates substantially improve majority of them over the plain TabDiff (incomplete-likelihood) baseline. The same appendix contains runtime comparisons and additional metrics.

Across both MCAR and MNAR, TabDiff Impute-EM (ours) delivers the best distributional metrics and the best average downstream rank. Under MCAR, Figure 2 and Table II show an almost twofold reduction in distributional error w.r.t. the second best baseline across all mask rates, while Figure 3 and Table III give TabDiff Impute-EM the best average downstream rank from per-dataset ML-efficiency errors. Under MNAR at $p = 0 . 7$ , Figure 4 reproduces this picture, with TabDiff Impute-EM again leading on both Overall Density and ML-efficiency.

![](images/9732199ffdc213cd87aadcee4a3dc1bdeedda8bfec32f3e44f9b9c0311cb6767.jpg)

Absolute deviation from MLE baseline ↓  
![](images/6ad0accda1911af628b70f2e19a3ea3bcb0eda6896cc2444c867f71b8942b3c6.jpg)

Fig. 4: Under MNAR at mask rate $p = 0 . 7 \colon$ Overall Density ↑ and ML-efficiency absolute deviation ↓ across the four datasets. Beijing MLE is multiplied by 0.1 to keep the scale comparable across datasets.
<table><tr><td>Method</td><td>Adult</td><td>Shoppers</td><td>Default</td><td>Beijing</td><td>Rank</td></tr><tr><td>ReMasker</td><td>3</td><td>3</td><td>2</td><td>2</td><td>③</td></tr><tr><td>DiffPuter</td><td>2</td><td>2</td><td>4</td><td>1</td><td>2</td></tr><tr><td>HyperImpute</td><td>5</td><td>4</td><td>2</td><td>2</td><td>4</td></tr><tr><td>MissDiff</td><td>4</td><td>4</td><td>3</td><td>4</td><td>4</td></tr><tr><td>TabDiff Impute-EM (ours)</td><td>1</td><td>1</td><td>1</td><td>3</td><td>1</td></tr></table>

TABLE III: Ranking downstream performance on tabular datasets under MCAR. Ranks follow the averaged absolute deviation from the clean-data ML-efficiency in each dataset. The last column ranks methods by their average rank across datasets. Lower is better. The best method is highlighted in bold.

## VI. CONCLUSION, LIMITATIONS, AND FUTURE WORK

We introduced Impute-EM, an EM-style framework that instantiates diffusion imputation with native backbones for heterogeneous data, and showed empirically that it improves imputation quality on discrete text and achieves the best distributional metrics and the best average downstream rank on heterogeneous tabular data.

Our method has two main limitations, both shared with other diffusion-based EM imputers [7, 13, 20]: missing-data learning is non-identifiable, so we guarantee observed-marginal consistency rather than recovery of the true distribution, and each EM iteration requires conditional diffusion sampling followed by training, which is more expensive than singlestage imputation.

Future work includes faster conditional samplers, regularization to select among compatible distributions, and adaptation to other heterogeneous domains.

## REFERENCES

[1] T. Du, L. Melis, and T. Wang, “Remasker: Imputing tabular data with masked autoencoding,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 9002–9024.

[2] T. Dror, I. Shoham, M. Buchris, O. Gal, H. Permuter, G. Katz, and E. Nachmani, “Token-based audio inpainting via discrete diffusion,” arXiv preprint arXiv:2507.08333, 2025.

[3] J. You, X. Ma, Y. Ding, M. J. Kochenderfer, and J. Leskovec, “Handling missing data with graph representation learning,” in Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, Eds., vol. 33. Curran Associates, Inc., 2020, pp. 19 075–19 087. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2020/ file/dc36f18a9a0a776671d4879cae69b551-Paper.pdf

[4] D. M. P. Murti, U. Pujianto, A. P. Wibawa, and M. I. Akbar, “K-nearest neighbor (k-nn) based missing data imputation,” in 2019 5th International Conference on Science in Information Technology (ICSITech), 2019, pp. 83–88.

[5] P. J. García-Laencina, J.-L. Sancho-Gómez, and A. R. Figueiras-Vidal, “Pattern classification with missing data: a review,” Neural Computing and Applications, vol. 19, no. 2, pp. 263–282, 2010.

[6] J. Zhong, N. Gui, and W. Ye, “Data imputation with iterative graph reconstruction,” in Proceedings of the AAAI conference on artificial intelligence, vol. 37, no. 9, 2023, pp. 11 399–11 407.

[7] J. Yoon, J. Jordon, and M. van der Schaar, “GAIN: Missing data imputation using generative adversarial nets,” in Proceedings of the 35th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, J. Dy and A. Krause, Eds., vol. 80. PMLR, 10–15 Jul 2018, pp. 5689–5698. [Online]. Available: https://proceedings.mlr.press/v80/yoon18a.html

[8] P.-A. Mattei and J. Frellsen, “Miwae: Deep generative modelling and imputation of incomplete data sets,” in International conference on machine learning. PMLR, 2019, pp. 4413–4423.

[9] T. W. Richardson, W. Wu, L. Lin, B. Xu, and E. A. Bernal, “Mcflow: Monte carlo flow models for data imputation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 14 205–14 214.

[10] Y. Ouyang, L. Xie, C. Li, and G. Cheng, “Missdiff: Training diffusion models on tabular data with missing values,” arXiv preprint arXiv:2307.00467, 2023.

[11] S. Zheng and N. Charoenphakdee, “Diffusion models for missing value imputation in tabular data,” in NeurIPS 2022 First Table Representation Workshop.

[12] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840–6851, 2020.

[13] H. Zhang, L. Fang, Q. Wu, and P. Yu, “Diffputer: Empowering diffusion models for missing data imputation,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 63 164–63 185.

[14] A. P. Dempster, N. M. Laird, and D. B. Rubin, “Maximum likelihood from incomplete data via the em algorithm,” Journal of the royal statistical society: series B (methodological), vol. 39, no. 1, pp. 1–22, 1977.

[15] J. Yu, Q. Ying, L. Wang, Z. Jiang, and S. Liu, “Missing data imputation by reducing mutual information with rectified flows,” Advances in Neural Information Processing Systems, vol. 38, pp. 80 324–80 352, 2025.

[16] S. S. Sahoo, M. Arriola, Y. Schiff, A. Gokaslan, E. Marroquin, J. T. Chiu, A. Rush, and V. Kuleshov, “Simple and effective masked diffusion language models,” Advances in Neural Information Processing Systems, vol. 37, pp. 130 136–130 184, 2024.

[17] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum? id=PxTIG12RRHS

[18] J. Austin, D. D. Johnson, J. Ho, D. Tarlow, and R. Van Den Berg, “Structured denoising diffusion models in discrete state-spaces,” Advances in neural information processing systems, vol. 34, pp. 17 981–17 993, 2021.

[19] J. Shi, M. Xu, H. Hua, H. Zhang, S. Ermon, and J. Leskovec, “Tabdiff: a mixed-type diffusion model for tabular data generation,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 37 353– 37 375.

[20] D. Hosseintabar, F. Chen, G. Daras, A. Torralba, and C. Daskalakis, “Diffem: Learning from corrupted data with diffusion models via expectation maximization,” arXiv preprint arXiv:2510.12691, 2025.

[21] D. Jarrett, B. C. Cebere, T. Liu, A. Curth, and M. van der Schaar, “Hyperimpute: Generalized iterative imputation with automatic model selection,” in International Conference on Machine Learning. PMLR, 2022, pp. 9916–9937.

[22] V. Fortuin, D. Baranchuk, G. Rätsch, and S. Mandt, “Gp-vae: Deep probabilistic time series imputation,” in International conference on artificial intelligence and

statistics. PMLR, 2020, pp. 1651–1661.

[23] J. Kim, T. Park, and K. Lee, “Augmask: Training diffusion models on incomplete tabular data via stochastic augmentation and masking,” in Proceedings of the 43rd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 306. PMLR, 2026. [Online]. Available: https://arxiv.org/abs/2606.03347

[24] J. Ou, S. Nie, K. Xue, F. Zhu, J. Sun, Z. Li, and C. Li, “Your absorbing discrete diffusion secretly models the conditional distributions of clean data,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 64 972–65 009.

[25] M. Mahoney, “Large text compression benchmark,” 2011.

[26] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4195–4205.

## APPENDIX A

## PROOFS

Proof of the Proposition 1. Fix the current iterate $p ^ { n } \in { \mathcal { P } } ( { \mathcal { X } } )$ In the exact E-step, for each mask $m \in \mathcal { M }$ the variational conditional over the missing coordinates is

$$
q ^ { n } ( x _ { \bar { m } } \mid x _ { m } ) = p ^ { n } ( x _ { \bar { m } } \mid x _ { m } ) ,\tag{19}
$$

which induces the completed-data measure

$$
\pi _ { m } ( p ^ { n } ) ( x ) : = p _ { m } ^ { * } ( x _ { m } ) p ^ { n } ( x _ { \bar { m } } \mid x _ { m } ) .\tag{20}
$$

The M-step maximizes the expected complete-data loglikelihood, which in terms of the mixture

$$
\bar { \pi } ^ { n } ( x ) : = \mathbb { E } _ { m \sim \mu } \big [ \pi _ { m } ( p ^ { n } ) ( x ) \big ] = \mathbb { E } _ { m \sim \mu } \big [ p _ { m } ^ { * } ( x _ { m } ) p ^ { n } ( x _ { \bar { m } } \mid x _ { m } ) \big ]\tag{21}
$$

reads $\begin{array} { r } { \operatorname * { a r g m a x } _ { p \in \mathcal { P } ( \mathcal { X } ) } \int _ { \mathcal { X } } \log p ( x ) d \bar { \pi } ^ { n } ( x ) } \end{array}$ . Since $\mathrm { K L } ( \bar { \pi } ^ { n } \parallel p ) \geq$ 0,

$$
\int _ { \mathcal { X } } \log p d \bar { \pi } ^ { n } = \int _ { \mathcal { X } } \log \bar { \pi } ^ { n } d \bar { \pi } ^ { n } - \mathrm { K L } ( \bar { \pi } ^ { n } \| p ) \leq \int _ { \mathcal { X } } \log \bar { \pi } ^ { n } d \bar { \pi } ^ { n } ,\tag{22}
$$

with equality if and only if $p = \bar { \pi } ^ { n }$ . Hence the maximizer is $p ^ { n + 1 } = \bar { \pi } ^ { n }$ . Therefore the exact update is

$$
p ^ { n + 1 } ( x ) = \bar { \pi } ^ { n } ( x ) = \mathbb { E } _ { m \sim \mu } \big [ p _ { m } ^ { * } ( x _ { m } ) p ^ { n } ( x _ { \bar { m } } \mid x _ { m } ) \big ] ,\tag{23}
$$

which is precisely the claimed Impute-EM update.

Proof of the Theorem 1. Fix m $\in \mathcal { M }$ and let p satisfy $\mathrm { K L } ( r \| p ) < \infty$ for some $\boldsymbol { r } \in \mathcal { C }$ , and set $\pi _ { m } ( p ) ( x ) : = p ( \overline { { x } } _ { m } \mid$ $x _ { m } ) p _ { m } ^ { * } ( x _ { m } )$ . By the chain rule for relative entropy,

$$
\mathrm { K L } ( r \parallel p ) = \mathrm { K L } ( p _ { m } ^ { * } \parallel p _ { m } ) + \mathrm { K L } ( r \parallel \pi _ { m } ( p ) ) .\tag{24}
$$

By convexity of relative entropy in its second argument (the log-sum inequality), for measures $s _ { m }$ with $\mathrm { K L } ( r \parallel s _ { m } ) < \infty .$

$$
\mathrm { K L } ( r \| \sum _ { m \in \mathcal { M } } \mu ( m ) s _ { m } ) \leq \sum _ { m \in \mathcal { M } } \mu ( m ) \mathrm { K L } ( r \| s _ { m } ) .\tag{25}
$$

An induction using (24) and $\mu ( m ) > 0$ keeps $r \ll p ^ { n }$ and $\mathrm { K L } ( r \parallel p ^ { n } ) < \infty$ . Since $\begin{array} { r } { p ^ { n + 1 } = \sum _ { m } \mu ( m ) \pi _ { m } ( p ^ { n } ) } \end{array}$ , applying (25) and then (24) gives

$$
\begin{array} { r l } {  { \mathrm { K L } \big ( r \| p ^ { n + 1 } \big ) \le \sum _ { m \in \mathcal { M } } \mu ( m ) \mathrm { K L } ( r \| \pi _ { m } ( p ^ { n } ) ) } } \\ & { = \mathrm { K L } ( r \| p ^ { n } ) - \sum _ { m \in \mathcal { M } } \mu ( m ) \mathrm { K L } ( p _ { m } ^ { * } \| p _ { m } ^ { n } ) . } \end{array}
$$

Thus

$$
\mathrm { K L } \big ( r \| p ^ { n + 1 } \big ) + \sum _ { m \in \mathcal { M } } \mu ( m ) \mathrm { K L } ( p _ { m } ^ { * } \| p _ { m } ^ { n } ) \leq \mathrm { K L } ( r \| p ^ { n } ) .\tag{26}
$$

Summing (26) over n and letting $N \to \infty$

$$
\sum _ { n = 0 } ^ { \infty } \sum _ { m \in \mathcal { M } } \mu ( m ) \operatorname { K L } ( p _ { m } ^ { * } \parallel p _ { m } ^ { n } ) \leq \operatorname { K L } ( r \parallel p ^ { 0 } ) < \infty ,
$$

so each nonnegative term satisfies $\mathrm { K L } ( p _ { m } ^ { * } \| p _ { m } ^ { n } ) \to 0$ as $n $ ∞ for every $m \in \mathcal { M }$ □

## APPENDIX B ADDITIONAL METHODOLOGY CLARIFICATIONS

## A. Incomplete likelihood

We use the term incomplete likelihood for objectives that train a generative model using only observed coordinates, i.e.,

$$
\operatorname* { m a x } _ { \theta } \ \mathbb { E } _ { m \sim \mu , \ x _ { m } \sim p _ { m } ^ { * } } \left[ \log p _ { \theta , m } ( x _ { m } ) \right] ,
$$

or tractable diffusion/variational surrogates thereof. Under this terminology, MissDiff [10] can be viewed as an incompletelikelihood diffusion method, since its masked denoising scorematching loss is motivated as an upper bound on the negative observed-data likelihood. ReMasker [1] can also be viewed as a likelihood-based masked autoencoder: it artificially re-masks observed entries and trains the model to predict them from the remaining observed entries. Equivalently, it optimizes a conditional reconstruction likelihood over observed coordinates, rather than a full joint likelihood over complete data.

The ability to train models via incomplete likelihood naturally comes from the our M-step formulation, see Section IV-B.

## B. Mixed space diffusion models

Following TabDiff [19], we combine continuous and discrete diffusion components, writing $x ^ { \mathrm { n u m } }$ and x<sup>cat</sup> for the numerical and categorical coordinates. The mixed conditional diffusion loss is, writing pˆ<sub>θ</sub> for the conditional categorical prediction $\hat { p } _ { \theta } \big ( \cdot \mid x _ { \bar { m } , t } ^ { \mathrm { c a t } } , x _ { m } , m \big )$

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { m i x e d d i f f } } ( \theta ) = \mathbb { E } _ { \underset { t \sim \mathcal { U } [ 0 , T ] , \epsilon \sim \mathcal { N } ( 0 , I ) } { x \sim D ^ { n } , m \sim \mu } } \left[ \left\| \epsilon - \epsilon _ { \theta } \big ( x _ { \bar { m } , t } ^ { \mathrm { n u m } } , t \mid x _ { m } , m \big ) \right\| ^ { 2 } \right] } \\ & { \quad \quad \quad \quad + \mathbb { E } _ { \underset { t \sim \mathcal { U } [ 0 , T ] } { x \sim D ^ { n } , m \sim \mu } } \left[ \mathrm { C E } \big ( e _ { x _ { \bar { m } } ^ { \mathrm { c a t } } } , \hat { p } _ { \theta } \big ) \right] , \quad \quad \underset { \mathrm { t } } { \mathrm { \ell } } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \times _ { \bar { m } , t } ^ { \mathrm { c a t } } \sim q _ { \mathrm { M D M , t } } ( \cdot \vert x _ { \bar { m } } ^ { \mathrm { c a t } } ) } \end{array}\tag{27}
$$

![](images/c362707b2b1ebeb53c92d8a009b22859c7d928dcd7658b73bed0f1040dc7171f.jpg)  
Fig. 5: Metrics versus Impute-EM iteration on the tabular data, i.e., "default" dataset with 50% missingness.

with $x _ { \bar { m } , t } ^ { \mathrm { n u m } } = \sqrt { \bar { \alpha } _ { t } } x _ { \bar { m } } ^ { \mathrm { n u m } } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon$ . The conditional reverse process factorizes into numerical and categorical transitions:

$$
\begin{array} { r l } & { p _ { \theta } ( x _ { \bar { m } , t - 1 } \mid x _ { \bar { m } , t } , t , x _ { m } , m ) = \mathcal { N } \big ( x _ { \bar { m } , t - 1 } ^ { \mathrm { n u m } } ; \mu _ { \theta } , \sigma _ { t } ^ { 2 } I \big ) } \\ & { \quad \quad \cdot \operatorname { C a t } \left( x _ { \bar { m } , t - 1 } ^ { \mathrm { c a t } } \Big | \frac { \big ( 1 - \alpha _ { t - 1 } \big ) \mathtt { M } + \big ( \alpha _ { t - 1 } - \alpha _ { t } \big ) \hat { p } _ { \theta } } { 1 - \alpha _ { t } } \right) , } \end{array}\tag{28}
$$

where M is the categorical mask-token distribution (distinct from the family of observation masks M).

## APPENDIX C ADDITIONAL EXPERIMENTAL RESULTS

## A. Tabular data

a) Runtime analysis: Table IV compares training runtimes for the tabular methods in Section V-B. Our method is slower than the incomplete-likelihood baselines MissDiff [10] and Remasker [1] but only by a modest margin, and 3 to 4 times faster than Diffputer [13], whose E-step averages 10 imputations and pays a corresponding inference cost.

TABLE IV: Runtime comparison for training the imputation methods on default dataset with 50% missingness rate. All methods except HyperImpute were trained on an NVIDIA A100 GPU. HyperImpute utilizes only the CPU.
<table><tr><td>Method</td><td>Runtime</td></tr><tr><td>Remasker</td><td>1 hour</td></tr><tr><td>MissDiff</td><td>20 minutes</td></tr><tr><td>Diffputer</td><td>7 hours 18 minutes</td></tr><tr><td>HyperImpute</td><td>22 minutes</td></tr><tr><td>TabDiff</td><td>53 minutes</td></tr><tr><td>TabDiff Impute-EM (ours)</td><td>2 hours 41 minutes</td></tr></table>

b) Impute-EM iterations analysis: Figure 5 shows MLfficiency, Shape, and Trend versus Impute-EM iterations on he Default dataset with 50% missingness. All metrics improve with iterations, with most of the gain in the first few steps and convergence by step 5.

![](images/65831bf6292de7914b9a5ba8ca632fdf3a9c40b8f33c40f121b5b3d2030c9620.jpg)  
Fig. 6: Categorical accuracy ↑ under MCAR versus mask rate (Adult, Shoppers, Default, and Beijing).

![](images/f9c413baafcdb6bbea52e70cf5baac9944f12f58c5e36154637a4651988d8992.jpg)  
Fig. 7: Mean absolute error (MAE) ↓ on numerical features under MCAR versus mask rate (Adult, Shoppers, Default, and Beijing).

c) Discriminative Metrics: Our objective is distributional recovery rather than pointwise reconstruction, so discriminative metrics are less relevant for our method. We nevertheless report them on imputed features under the same MCAR setup as in Section V-B. Accuracy measures how well imputed categorical entries match the ground truth, and MAE measures mean absolute error on numerical features. Figures 6 and 7 show both metrics versus mask rate for Adult, Shoppers, Default, and Beijing. While TabDiff Impute-EM achieves the best distributional metrics and the best average downstream MLefficiency rank, it does not lead on discriminative metrics, as the compared baselines are designed for pointwise recovery [1, 10, 13]. Figure 8 reports the same metrics under MNAR at $p = 0 . 7$ , with the same conclusion.

![](images/7474712399e9080c1ba184267b1f37e8a30a8ca979a5794df8de10ed23ab61fa.jpg)  
Fig. 8: Discriminative metrics under MNAR at mask rate p = 0.7. Categorical accuracy ↑ and MAE ↓ across the four datasets.

![](images/d105ef93b284439784a9afb8d864f0e3d205392969490decdd47cce1324704a5.jpg)

![](images/8864ee803d781168322025150f7bbea9d5bd52d04d64448b71bc72fabd77b6c8.jpg)  
Fig. 9: Single-run MAR results under the DiffPuter 70% mask, split 0. Density Overall ↑ and absolute deviation from the cleandata MLE baseline ↓ across the four datasets. Beijing MLE deviation is multiplied by 0.1 to keep the scale comparable across datasets.

## B. MAR robustness

We additionally evaluate MAR (Missing At Random) robustness with the DiffPuter [13] 70% masking protocol, with Adult, Shoppers, Default, and Beijing datasets. All the methods are run once on the same test-set evaluation. TabDiff and DiffPuter use five EM iterations, while ReMasker [1] and MissDiff [10] are single-pass baselines. Figure 9 reports distributional fidelity and downstream utility. Figure 10 reports pointwise metrics. TabDiff Impute-EM has the highest Overall Density score on every dataset and deliveres less MLE error than the other methods overall.

![](images/b6de7810e03eefa15bf5af3cf47d885ca1428c2a8aab508d1dfa8f25a2bd6ff5.jpg)  
Fig. 10: Single-run MAR discriminative results under the DiffPuter 70% mask, split 0. Categorical accuracy ↑ and MAE ↓ across the four datasets. The MAE axis is capped at 1.0.

## APPENDIX D EXPERIMENTAL DETAILS

## A. Text data

a) Metrics: Perplexity (PPL) is the exponential of the average negative log-likelihood, PPL $\begin{array} { r } { \exp \bigl ( { - \mathbb { E } _ { x \sim p _ { \mathrm { d a t a } } } \bigl [ \frac { 1 } { N } \log \bar { p _ { \theta } } ( x _ { 1 } , \dots , \bar { x _ { N } } ) \bigr ] } \bigr ) } \end{array}$ where the loglikelihood is computed via the MDLM ELBO [16].

b) Implementation details: We utilize the MDLM [16] official code base:

## https://github.com/kuleshov-group/mdlm

The following hyperparameters are consistent across all experiments. During E-step dataset generation for MDM backward process inference, we use 64 function evaluations (NFE). As the backbone architecture, we use a DIT [26] with approximately 13M parameters. EMA of weights was used with 0.999 coefficient. The training took 10 hours for ReMasker MDLM and 17 hours for MDLM Impute-EM (including the initialization stage) on two NVIDIA-A100 GPUs.

## B. Tabular data

a) Missingness patterns: Missing Completely At Random (MCAR) means missingness is independent of all entries. It is the standard tabular imputation benchmark for its simplicity and reproducibility. For our Missing Not At Random (MNAR) robustness study at $p = 0 . 7 $ , we follow the MNAR protocol of DiffPuter [13]: the columns are split into two groups, a logistic model on the first group outputs the missing probabilities for the second group, and MCAR is then applied to the first group. As a result, missingness in the second group depends on the masked values of the first. For more details we refer the reader to DiffPuter [13].

b) Distributional metrics (Shape and Trend): Following TabDiff [19], we use the SDMetrics<sup>1</sup> Shape and Trend scores. Shape measures how well each column marginal matches the reference data, using the Kolmogorov-Smirnov statistic for numerical columns and total variation distance for categorical columns. Trend measures how well pairwise dependence matches, with Pearson correlations for numerical pairs and contingency-table frequencies for categorical pairs. Overall Density is their average (Shape + Trend)/2. Lower error is better.

TABLE V: Dataset statistics. # Num and # Cat count numerical and categorical columns and task target columns are counted separately.
<table><tr><td>Dataset</td><td># Rows</td><td>#Num</td><td># Cat</td><td># Train</td><td># Test</td><td>Task</td></tr><tr><td>Adult</td><td>32,561</td><td>6</td><td>7</td><td>22,792</td><td>9,769</td><td>Classification</td></tr><tr><td>Default</td><td>30,000</td><td>14</td><td>9</td><td>21,000</td><td>9,000</td><td>Classification</td></tr><tr><td>Shoppers</td><td>12,330</td><td>10</td><td>6</td><td>8,631</td><td>3,699</td><td>Classification</td></tr><tr><td>Beijing</td><td>41,828</td><td>6</td><td>5</td><td>29,320</td><td>12,528</td><td>Regression</td></tr></table>

c) Machine learning efficiency: Machine Learning Efficiency (ML-efficiency) evaluates whether imputed tables remain useful for the dataset’s supervised task [19]. We train an XGBoost model on the imputed training split (hyperparameters chosen on an 8:1 train-validation split) and evaluate on a heldout clean test set, using AUC-ROC for classification and RMSE for regression. Only training features are imputed, with scores averaged over 50 runs.

d) Data description: We use four mixed-type tabular benchmarks from the UCI Machine Learning Repository.<sup>2</sup> Each dataset is tied to a downstream supervised task used for ML-efficiency. Adult, Default, and Shoppers are classification problems. Beijing is a regression problem. In the Table V one can see the information about datset sizes and train test splits. All preprocessing follows DiffPuter [13], whose codebase also releases the preprocessed splits.<sup>3</sup>

e) Implementation details: Baseline methods, i.e., MissDiff, Remasker, Diffputer, were taken from Diffputer [13] official repository:

## https://github.com/hengruizhang98/DiffPuter

All the baseline methods hyperparameters were taken from Diffuputer [13]. The MissDiff and Remasker experiments took no longer than hour, Diffputer experiments took no longer than 12 hours. Models were trained on NVIDIA-A100 GPU.

As the baseline implementation for TabDiff Impute-EM we used the official TabDiff [19] github repository:

## https://github.com/minkaixu/tabdiff

As the backbone neural network the same multimodal MLP as in TabDiff was used. At the initialization stage the incomplete likelihood model was trained for 5000 epochs, while during each of the Impute-EM iterations the model was trained for 2000 epochs. During the E-step the samples were generated by TabDiff with 50 NFE. The model was trained with Adam optimizer, starting learning rate is 3e−4 with ReduceOnPlateou scheduler. EMA of weights was used with 0.997 coefficient. Each experiment took no longer than 12 hours on NVIDIA-A100 GPU.