# ADAPTIVE MEAN ESTIMATION BY IN-CONTEXT LEARNING: A GRADIENT-FLOW ANALYSIS

Martin Eppert<sup>1</sup>

Krishna Balasubramanian<sup>2</sup> Subhro Ghosh<sup>1</sup>

Jason Klusowski<sup>3</sup>

Yan Shuo Tan<sup>1</sup>   
<sup>1</sup>National University of Singapore <sup>2</sup>University of California, Davis <sup>3</sup>Princeton University

## ABSTRACT

Prior Fitted Networks (PFNs) such as TabPFN (Zhang et al., 2025) now rival established statistical procedures across prediction and estimation tasks. A natural explanation is that PFNs have the property of statistical adaptivity, that is, they perform nearly as well as a method tailored to the true data-generating model for a heterogeneous set of models, while not being told which model the data comes from. Prior work does not give a conclusive explanation of how gradient-based pretraining learns an adaptive estimator. We study how such adaptivity is learned in a controlled location-estimation problem. Each task is an unlabeled sample whose family is hidden: Gaussian data call for averaging, with error of order $n ^ { \bar { - } 1 }$ whereas uniform data are best estimated from their extremes, at the faster rate $n ^ { - 2 }$ . We also provide the example of a symmetric Gaussian mixture, for which a rate of $\sigma _ { n } ^ { 2 } / { \dot { n } }$ can be attained. On scalar inputs, softmax attention computes the derivative of the empirical cumulant-generating function. A single primitive therefore both supplies features that distinguish the families and forms estimators interpolating between the sample mean and the mid-range. Building on this, we combine attention experts through either a softmax mixture of experts or a gated linear unit (GLU), and analyze stagewise gradient flow. With $\widetilde \Omega ( n ^ { 1 + \epsilon } )$ pretraining tasks, the learned estimator is asymptotically efficient on Gaussian tasks, within a factor $n ^ { \epsilon }$ of the minimax rate on uniform tasks, and order-optimal on mixtures in a shrinking-variance regime. These guarantees extend to new locations and longer contexts. A risk decomposition separates expert error, routing error and normalization error, which clarifies the architectural contrast. Softmax gating enforces normalization and exact translation equivariance, whereas the GLU must learn it: its dynamics separate into fast bias removal followed by slow expert selection. End-to-end experiments recover the predicted specialization.

## 1 INTRODUCTION

In a landmark paper, Brown et al. (2020) showed that large language models can learn to perform new tasks from examples supplied in a prompt (also called a context), without updating their parameters at inference time, a capability they called in-context learning (ICL). Garg et al. (2023) formulated ICL as a new paradigm for statistical learning: whereas the classical approach fits a predictor to each new dataset through parameter optimization, ICL pretrains a reusable prediction procedure across datasets and applies it to a new dataset through context alone. Tabular foundation models (TFMs) have made this paradigm practical for tabular prediction, with evaluations on the broad TabArena and TALENT benchmarks showing that TFMs can even outperform previously state-of-the-art methods such as boosted tree ensembles (Erickson et al., 2025; Ye et al., 2025). Other recent studies show how the ICL paradigm can outperform well-established classical methods in estimating various statistical quantities, including heterogeneous treatment effects (Balazadeh et al., 2025), survival curves (Qi et al., 2026), mutual information (Peyrard & Cho, 2025), and Poisson means in empirical Bayes problems (Teh et al., 2025). These works suggest that the central theoretical question for ICL should not be whether it can learn to approximate an existing algorithm or statistical procedure, but how it is able to outperform established procedures across diverse statistical problems.

Several theoretical explanations have been developed by analyzing empirical risk minimization (ERM) of transformers. Bai et al. (2023) show that transformers can implement in-context algorithm selection, including choosing among ridge penalties using a train-validation split. Wakayama & Suzuki (2025) and Ma et al. (2025) establish approximation of the posterior predictive mean under hierarchical priors over regression-function classes. In particular, Ma et al. (2025) use posterior contraction to show that this predictor achieves optimal rates corresponding to the unknown smoothness and effective dimension of the task. These results suggest that part of ICL’s strength may lie in its ability to adjust the statistical procedure to the structure revealed by the context. This is a form of statistical adaptivity, which is when a single estimator attains a close to optimal accuracy appropriate to the underlying model for multiple models, without knowing that model in advance. Such adaptation is a longstanding objective in statistics, often achieved through procedures tailored to particular models and inferential targets (Bickel et al., 1993). Pretraining offers a potentially general recipe to acquiring this behavior, but guarantees for ERM do not establish that gradient-based training learns the required mechanisms.

Understanding how such adaptation is learned requires an analysis of pretraining itself. A complementary literature studies pretraining dynamics directly. Results include gradient-flow convergence for linear regression with linear attention (Zhang et al., 2024a;b), task allocation in multi-head softmax attention for multi-task linear regression (Chen et al., 2024a), and gradient-based learning of shared low-dimensional structure for nonlinear single-index regression (Oko et al., 2024). These results explain important forms of learned statistical behavior, including specialization and adaptation to shared structure.

In this paper, we study how adaptation can emergefrom gradient-based pretraining,focusing on a simple one-dimensional location-estimation problem. Here, each task draws a context comprising n i.i.d. observations from either a Gaussian or a uniform distribution, and the estimation target is the location parameter of the underlying distribution. The appropriate estimator depends on the distributional family. For Gaussian observations, a sample average is optimal, with mean-squared error of order $n ^ { - 1 }$ , whereas uniform observations permit the faster $n ^ { \stackrel {  } { - } 2 }$ rate through the sample mid-range. Neither estimator attains both rates. Moreover, the family label is hidden and has to be inferred from the context. During pretraining, the learner observes the target mean but receives neither the family label nor a prescribed estimator choice. We also consider a symmetric two-component Gaussian mixture as the alternative to the Gaussian family. Its shrinking component variance $\sigma _ { n } ^ { 2 }$ permits risk of order $\sigma _ { n } ^ { 2 } / n ,$ , providing a second example of adaptation to distributional structure.

Multi-head attention can be used to identify the distribution of the samples in context as well as to esti mate their location. With scalar observations, the query and key matrices of an attention head collapse to a single scalar which we denote by λ. Thus, attention effectively computes the exponentially tilted average $\begin{array} { r } { \mathbf { \bar { \Phi } } _ { h ( \mathbf { x } , \lambda ) } = \sum _ { i } x _ { i } e ^ { \lambda x _ { i } } / \sum _ { i } \bar { e } ^ { \lambda x _ { i } } } \end{array}$ . This is the derivative of the empirical cumulant-generating function (CGF). Since the population CGF can be used to characterize distributions, evaluating this function at several locations supplies features for distinguishing between the two families. The same operation also constructs the relevant estimators: setting $\lambda = 0$ gives the sample mean, while sending λ → ±∞ approaches the sample maximum and minimum, which are required to compute the mid-range.

We use these two roles of multi-head attention to construct a bank of estimators, or experts, as well as a feature embedding, that allows distinguishing different distributions. A learned gating parameter $\theta _ { i }$ is used within a gating layer, either a Softmax MoE or a Gated Linear Unit, to output a weighted combination of the outputs of the different experts. This weighted sum of the expert estimates forms the location estimate. Both of these architectures are minor modifications of commonly used layers within standard Transformer architectures (Shazeer, 2020). We analyze stagewise gradient flow that first learns the gating parameters $\theta _ { i }$ with the expert parameters $\lambda _ { i }$ fixed, then learns the expert parameters $\lambda _ { i }$ with the gating parameters $\theta _ { i }$ fixed. This is also done to simplify the proof structure, while our experiments demonstrate that this is not strictly necessary. Notably, our results are stronger than an expressivity statement, as they demonstrate that gradient flow on the empirical loss is able to learn how to do in context location estimation with close to optimal rates.

Our contributions are fourfold:

• Adaptive estimation and transfer. We establish family-specific risk guarantees for the output of stagewise pretraining under both the Softmax MoE and the GLU architecture. Both architectures achieve Gaussian asymptotic efficiency and nearly (up to a $n ^ { \epsilon }$ factor) optimal uniform estimation rates. For a Gaussian mixture distribution, we also obtain order-optimal rates. This implies that our architecture performs nearly as well as the sample mean on Gaussian data and the sample mid-range on uniform data. We also establish out-of-distribution risk bounds for longer contexts as well as for samples with a fixed location.

• Statistical mechanism and architectural comparison. Our results show how the risk can be decomposed into the variance of the experts, the bias of the final estimator, and the error accumulated through the gating mechanism (Lemma 3.5). They explain how distribution-sensitive weighting preserves the accuracy of the appropriate expert, and how the architectures differ: softmax enforces normalization of the weights and thus exact translation equivariance, whereas the GLU must learn approximate normalization of the weights (Theorem 3.1).

• Learning dynamics. We show how training learns the gating parameters and expert parameters required by the statistical analysis (Lemma 3.3 and Lemma 3.4). Since the loss admits a bias-variance decomposition, our analysis tracks the movement of the bias and variance separately. The major terms of the bias are created by the sum of the weights, while variance is accumulated through the variance of the separate experts. Additionally, we observe that the bias is scaled at a different order, proportional to the prior variance $\sigma _ { \mu } ^ { 2 } .$ , than the variance which is of order $\textstyle { \frac { 1 } { n } }$ where n is the sample size. Thus, bias and variance move at different timescales. We exploit this fact to characterize the dynamics of the parameters (Lemma B.8 and Lemma B.9). Subsequent training of the attention layer lets the experts specialize their predictions to their assigned distribution (Lemma 3.4). Here, our analysis exploits that the second layer has already converged to a near-perfect gating of the different distributions.

• Finite Sample Guarantees. Our analysis demonstrates that for $m = \widetilde \Omega ( n ^ { 1 + \epsilon } )$ pretraining tasks, the gradient flow on the empirical loss ${ \mathcal { L } } _ { m }$ converges to the correct solution.

![](images/4f054bcdc54ecd288378a0f4eb39ecadc0f59b5eb02849e240a4f0a53c3bc1c3.jpg)  
Figure 1: Illustration of our studied architecture. We also highlight the parts which were fixed for simplicity.

Notation. We write $[ n ] = \{ 1 , \dots , n \}$ . The norms $\| \cdot \| _ { 2 }$ and $\Vert \cdot \Vert _ { \mathrm { F } }$ denote the Euclidean and Frobenius norms, respectively. Parameter collections and their gradients are identified with stacked vectors. Let $e _ { i }$ be the i-th coordinate vector. The notation O, Θ, Ω, and o has its usual meaning, $\widetilde { \mathcal { O } }$ and $\widetilde \Omega$ suppress logarithmic factors, and $\lesssim$ denotes inequality up to an absolute multiplicative constant. Constants may change from line to line. Uppercase $X _ { i }$ and X denote random observations and contexts, while lowercase $x _ { i }$ and $\mathbf { x } = ( x _ { 1 } , \ldots , x _ { n } )$ denote their realizations. We define estimators as functions of $\mathbf { x }$ and evaluate their risks at the random context X. We use $\mathbb { E } _ { j } [ \cdot ] = \mathbb { E } [ \cdot \mid \tau = j ]$ for expectation under family $j ,$ including both the location prior and the conditional sample. We write $\mathbb { P } _ { j }$ for this family-conditional law. For $1 \leq p < \infty , \| Y \| _ { L ^ { p } ( \mathbb { P } _ { i } ) } = ( \mathbb { E } _ { j } [ | Y | ^ { p } ] ) ^ { 1 / p }$ , using the Euclidean norm for random vectors and the unconditional law when the family subscript is omitted.

## 2 PROBLEM AND ARCHITECTURES

We consider a location-estimation task. During training and inference, the learner is presented with observations from a distribution in context and is tasked with estimating their location. The learner is not explicitly told which distribution the observations are from. To allow a fair comparison, the distributions differ only in shape, not in variance.

Definition 2.1 (Adaptive mean-estimation tasks). Sample a location $\mu$ and a family label τ independently:

$$
\begin{array} { r } { \mu \sim \mathcal { N } ( 0 , \sigma _ { \mu } ^ { 2 } ) , \quad \quad \mathbb { P } ( \tau = \mathrm { G } ) = \mathbb { P } ( \tau = \mathrm { A } ) = \frac { 1 } { 2 } } \end{array}
$$

Conditional on $( \mu , \tau )$ , draw $\xi _ { i } \stackrel { \mathrm { i . i . d . } } { \sim } Q ,$ <sub>τ</sub> and set

$$
X _ { i } = \mu + \xi _ { i } , \qquad i \in [ n ] , \qquad \mathbf { X } = ( X _ { 1 } , \ldots , X _ { n } ) .
$$

The centered Gaussian law is $Q _ { \mathrm { G } } = \mathcal { N } ( 0 , 1 )$ . The alternative centered law $Q _ { \mathrm { A } }$ is either

Uniform:

$$
Q _ { \mathrm { A } } = Q _ { \mathrm { U } } = \mathrm { U n i f } [ - \sqrt { 3 } , \sqrt { 3 } ] ,
$$

Gaussian mixture:

$$
\begin{array} { r } { Q _ { \mathrm { A } } = Q _ { \mathrm { M } } = \frac { 1 } { 2 } \mathcal { N } ( - \sqrt { 1 - \sigma _ { n } ^ { 2 } } , \sigma _ { n } ^ { 2 } ) + \frac { 1 } { 2 } \mathcal { N } ( \sqrt { 1 - \sigma _ { n } ^ { 2 } } , \sigma _ { n } ^ { 2 } ) , } \end{array}
$$

where $\sigma _ { n } ^ { 2 } \in ( 0 , 1 )$ . In statements specific to one alternative, we write U or M in place of A. Write $P _ { \mu , j }$ for the law of $\mu + \xi$ with $\xi \sim Q _ { j }$ . Each observed task $( X , \mu )$ is the tuple of the context and the location and thus, the family label τ is never observed.

Risk and Lower Bounds. For $j \in \{ \mathrm { G } , \mathrm { A } \}$ , define the risk at location $\mu$ and its average over the location prior by

$$
R _ { j , n } ( { \widehat { \mu } } ; \mu ) : = \mathbb { E } _ { \mathbf { X } \sim P _ { \mu , j } ^ { \otimes n } } [ ( { \widehat { \mu } } ( \mathbf { X } ) - \mu ) ^ { 2 } ] , \qquad R _ { j , n } ( { \widehat { \mu } } ) : = \mathbb { E } _ { j } [ ( { \widehat { \mu } } ( \mathbf { X } ) - \mu ) ^ { 2 } ] .
$$

For the trained estimator, these expectations are over an independent test task with the trained parameters held fixed. For a fixed family j, define the location-minimax risk

$$
R _ { j , n } ^ { \star } : = \operatorname* { i n f } _ { \widetilde { \mu } } \operatorname* { s u p } _ { \mu \in \mathbb { R } } \mathbb { E } _ { { \mathbf { X } } \sim P _ { \mu , j } ^ { \otimes n } } [ ( \widetilde { \mu } ( { \mathbf { X } } ) - \mu ) ^ { 2 } ] .
$$

The sample mean has risk $1 / n$ under every family. For Gaussian data, the sample mean attains the exact location-minimax risk $n ^ { - 1 }$ . For uniform data, the sample mid-range attains the faster location-minimax rate $n ^ { - 2 }$ . For completeness, we give the proofs of these benchmarks in Section H.1. For the Gaussian mixture, revealing component labels reduces the problem to Gaussian location estimation with variance $\sigma _ { n } ^ { 2 } .$ , yielding the lower bound $\sigma _ { n } ^ { 2 } / n$ (see Lemma H.2).

## 2.1 ARCHITECTURE

We consider a two-layer transformer architecture that consists of one multi-head attention layer followed by a second layer, which is either a Softmax Mixture of Experts (MoE) layer or a Gated Linear Unit. In the following, we will fix some of the parameters of the transformer architecture to decide before training which attention heads will be used as estimators and which will be used to identify the distribution.

Definition 2.2 (Transformer architecture). Let K be the number of experts and $d \geq 3$ be the number of heads dedicated to identifying the distribution family. Let x be the context of size n, containing the input dataset. For $\lambda \in \mathbb { R }$ , define

$$
h ( \mathbf { x } , \lambda ) = \frac { \sum _ { i = 1 } ^ { n } e ^ { \lambda x _ { i } } x _ { i } } { \sum _ { i = 1 } ^ { n } e ^ { \lambda x _ { i } } } .\tag{1}
$$

Expert $k \in [ K ]$ is $g _ { k } ( { \bf x } ) = \frac { 1 } { 2 } \left( h ( { \bf x } , \lambda _ { k } ) + h ( { \bf x } , - \lambda _ { k } ) \right)$ with $\lambda _ { k } ( 0 ) \stackrel { \mathrm { \scriptsize ~ i . i . d . } } { \sim } \mathrm { \ U n i f } [ - 1 , 1 ]$ . For gating, sample $\omega _ { r } \stackrel { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , 1 ) , r \in [ d ]$ , and hold $\boldsymbol { \omega } = ( \omega _ { 1 } , \ldots , \omega _ { d } )$ fixed. We define $\phi _ { \omega } ( \mathbf { x } ) = $ $( h ( \mathbf { x } , \omega _ { 1 } ) , \hdots , h ( \mathbf { x } , \omega _ { d } ) ) ^ { \top }$ and then apply a LayerNorm with fixed parameters to obtain z

$$
\mathbf { z } ( \mathbf { x } ) = \mathrm { L a y e r N o r m } ( \phi _ { \omega } ( \mathbf { x } ) ) = \frac { P _ { d } \phi _ { \omega } ( \mathbf { x } ) } { \lVert P _ { d } \phi _ { \omega } ( \mathbf { x } ) \rVert _ { 2 } } ,\tag{2}
$$

where $P _ { d } = I _ { d } - \mathbf { 1 1 } ^ { \top } / d .$ For $k \in [ K ]$ , initialize $\theta _ { k } ( 0 ) \stackrel { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , \sigma _ { \theta } ^ { 2 } I _ { d } )$ . We will consider two different choices for w, the Softmax Mixture of Experts (3) and the Gated Linear Unit (4):

$$
w _ { k } ( { \mathbf { x } } ; \theta ) = \frac { \exp ( \langle \mathbf { z } ( { \mathbf { x } } ) , \theta _ { k } \rangle ) } { \sum _ { \ell = 1 } ^ { K } \exp ( \langle \mathbf { z } ( { \mathbf { x } } ) , \theta _ { \ell } \rangle ) } , \ : \ : ( 3 ) \qquad w _ { k } ( { \mathbf { x } } ; \theta ) = \sigma ( \langle \mathbf { z } ( { \mathbf { x } } ) , \theta _ { k } \rangle ) = \frac { 1 } { 1 + e ^ { - \langle \mathbf { z } ( { \mathbf { x } } ) , \theta _ { k } \rangle } } .\tag{4}
$$

Finally, we define the final network output as

$$
\widehat { \mu } _ { \theta , \lambda } ( \mathbf { x } ) = \sum _ { k = 1 } ^ { K } w _ { k } ( \mathbf { x } ; \theta ) g _ { k } ( \mathbf { x } ) .\tag{5}
$$

The parameters $\theta = \left( \theta _ { 1 } , \ldots , \theta _ { K } \right)$ and $\lambda = ( \lambda _ { 1 } , \ldots , \lambda _ { K } )$ are trained, while ω remains fixed.

The architecture in Definition 2.2 fixes some parameters to remove permutation invariances. The attention heads are split into the experts $g _ { k }$ , which implement the candidate estimators, and the heads forming the feature z, which identifies the distribution family. When the routing feature is replaced by its population value, the unbiased experts $g _ { k }$ give an unbiased Softmax MoE output, whereas the GLU must also learn normalization. For sample-dependent gates, softmax guarantees exact translation equivariance but need not be exactly unbiased. The feature does not depend on the location, ${ \bf z } ( { \bf X } + \mu { \bf 1 } ) = { \bf z } ( { \bf X } )$ , so its distribution depends only on the family. Under family j, ${ \bf z } ( { \bf X } )  \bar { \bf z } _ { i }$ in probability as n $\imath  \infty$ , where $\bar { \mathbf { z } } _ { j }$ is the centered and normalized vector of derivatives of the $\mathrm { C G F }$ of $Q _ { j }$ at $\omega _ { 1 } , \ldots , \omega _ { d }$ . Since $\bar { \mathbf { z } } _ { \mathrm { G } } \neq \bar { \mathbf { z } } _ { \mathrm { A } }$ almost surely (Lemma F.5), z distinguishes the families for large n.

Our architecture introduces a LayerNorm for the gating input to the second layer, as this simplifies calculations. We believe this is not strictly necessary. For details, see Appendix J.

## 2.2 TRAINING PROTOCOL

For our training dataset, we sample m i.i.d. tasks of size n from Definition 2.1 and denote our dataset by $\{ ( \mathbf { X } _ { b } , \mu _ { b } ) \} _ { b = 1 } ^ { m }$

Definition 2.3 (Stagewise gradient flow training). For either architecture, (3) or (4), define the population loss $\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } , \boldsymbol { \lambda } ) = \frac { 1 } { 2 } \mathbb { E } [ ( \widehat { \mu } ( \mathbf { X } ) - \boldsymbol { \mu } ) ^ { 2 } ] } \end{array}$ and the empirical loss $\begin{array} { r } { \hat { \mathcal { L } } _ { m } ( \theta , \lambda ) = \frac { 1 } { 2 m } \sum _ { b = 1 } ^ { m } ( \widehat { \mu } ( \mathbf { X } _ { b } ) - } \end{array}$ $\mu _ { b } ) ^ { 2 }$ . Training proceeds in two stages:

1. For $0 \leq t < t _ { \theta }$ , run $\dot { \theta } = - \nabla _ { \theta } \hat { \mathcal { L } } _ { m }$ with λ fixed.

2. For $t _ { \theta } \leq t < t _ { \lambda }$ , run $\dot { \lambda } = - \nabla _ { \lambda } \hat { \mathcal { L } } _ { m }$ with θ fixed.

The full population flow is the same procedure with $\hat { \mathcal { L } } _ { m }$ replaced by L in both stages.

## 3 MAIN RESULTS

Theorem 3.1 (Main result; proof in Section B.5). Let $K = 2$ be the number of experts. Fix any $\eta , \epsilon \in ( 0 , 1 )$ , a gating dimension $d \geq 3 ,$ , and sufficiently small $\sigma _ { \mu } ^ { 2 } > 0 \mathrm { : }$ , all independent of n. Sample $m = \widetilde { \Omega } ( n ^ { 1 + \epsilon } )$ independent tasks according to Definition 2.1, with $m \in \mathrm { p o l y } ( n )$ . In the Gaussian-mixture case, assume $n ^ { - 1 + \epsilon } \leq \sigma _ { n } ^ { 2 } \leq o ( 1 )$ . Initialize as in Definition 2.2, with fixed $\sigma _ { \theta } ^ { 2 } > 0 .$ , sufficiently small depending on η. Relabel the experts and permute θ accordingly so that $| \check { \lambda _ { 1 } } ( 0 ) | < | \check { \lambda _ { 2 } } ( 0 )$ . Let $\widehat { \mu } _ { \mathrm { t r a i n e d } } = \widehat { \mu } _ { \boldsymbol { \theta } ( t _ { \boldsymbol { \theta } } ) , \lambda ( t _ { \lambda } ) }$ , with $t _ { \lambda }$ as in Lemma 3.4. For sufficiently large n, with probability at least $1 - \eta$ over the initialization and training tasks,

$$
R _ { \mathbb { G } , n } ( \widehat { \mu } _ { \mathrm { t r a i n e d } } ) = \frac { 1 + o ( 1 ) } { n } , \qquad R _ { \mathbb { A } , n } ( \widehat { \mu } _ { \mathrm { t r a i n e d } } ) = \left\{ \begin{array} { l l } { \mathcal { O } ( n ^ { - 2 + \epsilon } ) , } & { u n i f o r m a l t e r n a t i v e , } \\ { \mathcal { O } ( \sigma _ { n } ^ { 2 } / n ) , } & { G a u s s i a n - m i x t u r e ~ a l t e r n a t i v e . } \end{array} \right.
$$

For the $G L U ,$ we have, for every $\mu \in \mathbb { R }$ and $j \in \{ \mathrm { G } , \mathrm { A } \}$

$$
R _ { j , n } ( \widehat { \mu } _ { \mathrm { t r a i n e d } } ; \mu ) \leq 2 R _ { j , n } ( \widehat { \mu } _ { \mathrm { t r a i n e d } } ) + \frac { C \mu ^ { 2 } } { n ^ { 2 } } .
$$

For the softmax MoE, for every $\mu \in \mathbb { R }$ and $j \in \{ \mathrm { G } , \mathrm { A } \}$

$$
R _ { j , n } ( \widehat { \mu } _ { \mathrm { t r a i n e d } } ; \mu ) = R _ { j , n } ( \widehat { \mu } _ { \mathrm { t r a i n e d } } ) .
$$

The Gaussian and Gaussian-mixture rates match the lower bounds of Proposition H.1 and Lemma H.2 up to constants, and the uniform rate is within a factor of $n ^ { \epsilon }$ of the minimax rate $n ^ { - 2 }$ . Table 1 summarizes the results. The Softmax MoE architecture has a stronger guarantee when $| \mu |$ is large.

Table 1: The terminal expert parameters and fresh-context risks certified by the main theorem.
<table><tr><td>Alternative family terminal</td><td> $| \lambda _ { 2 } |$ </td><td></td><td>Gaussian risk Alternative risk</td><td> $R _ { { \mathrm { A } } , n } ^ { \star }$ </td></tr><tr><td>Uniform</td><td> $n ^ { 1 - \epsilon }$ </td><td> $( 1 + o ( 1 ) ) / n$ </td><td> $\mathcal { O } ( n ^ { - 2 + \epsilon } )$ </td><td> $\asymp n ^ { - 2 }$ </td></tr><tr><td>Gaussian mixture</td><td> $\underline { { \log ( 1 / \sigma _ { n } ^ { 2 } ) } }$   $^ 4 { \sqrt { 1 - \sigma _ { n } ^ { 2 } } }$ </td><td> $( 1 + o ( 1 ) ) / n$ </td><td> $\mathcal { O } ( \sigma _ { n } ^ { 2 } / n )$ </td><td> $\geq \sigma _ { n } ^ { 2 } / n$ </td></tr></table>

This follows from its exact translation equivariance, while the GLU has to learn approximate normalization.

Corollary 3.2 (Length generalization; proof in Section D.3.7). Under the assumptions of Theorem $3 . I ,$ for sufficiently large n, with probability at least $1 - \eta$ over the initialization and pretraining tasks, for every $n ^ { \prime } \geq n$

$$
R _ { \mathrm { G } , n ^ { \prime } } ( \widehat { \mu } _ { \mathrm { t r a i n e d } } ) = \mathcal { O } \left( \frac { 1 } { n ^ { \prime } } + \frac { \log n ^ { \prime } } { n ^ { 2 } } \right) ,
$$

$$
R _ { \mathrm { A } , n ^ { \prime } } ( \widehat { \mu } _ { \mathrm { t r a i n e d } } ) = \left\{ \begin{array} { l l } { \mathcal { O } ( n ^ { - 1 + \epsilon } / n ^ { \prime } + n ^ { - 2 } ) , } & { U n i f o r m , } \\ { \mathcal { O } ( \sigma _ { n } ^ { 2 } / n ^ { \prime } + \log n ^ { \prime } / n ^ { 2 } ) , } & { G a u s s i a n - m i x t u r e . } \end{array} \right.
$$

## 3.1 PRETRAINING DYNAMICS

We establish the gate bound of Lemma 3.3 and the expert-parameter bounds of Lemma 3.4: first train the gating parameters with fixed experts, then specialize the expert parameters with fixed gates. Equation 3.4 supplies the resulting risk guarantees.

We specialize to two experts and two families. Define the population attention responses and feature direction by

$$
m _ { j } ( a ) = \frac { \mathbb { E } _ { j } \left[ e ^ { a ( X - \mu ) } ( X - \mu ) \right] } { \mathbb { E } _ { j } \left[ e ^ { a ( X - \mu ) } \right] } , \qquad \bar { \phi } _ { j } = \left( m _ { j } ( \omega _ { 1 } ) , \dots , m _ { j } ( \omega _ { d } ) \right) ^ { \top } , \qquad \bar { \mathbf { z } } _ { j } = \frac { P _ { d } \bar { \phi } _ { j } } { \| P _ { d } \bar { \phi } _ { j } \| _ { 2 } } .
$$

Let $\bar { w } _ { k j }$ denote $w _ { k }$ with ${ \bf z } ( { \bf X } )$ replaced by $\bar { \mathbf { z } } _ { j }$ , and $\bar { w } . _ { j } = ( \bar { w } _ { 1 j } , \bar { w } _ { 2 j } ) ^ { \top }$ . Throughout, bars indicate evaluation at population moments rather than expectation of the corresponding nonlinear statistic. In particular, $\bar { \mathbf { z } } _ { j } \neq \mathbb { E } _ { j } [ \mathbf { z } ( \mathbf { X } ) ]$ in general. In the proofs, $\Delta$ denotes deviation from the corresponding population-moment value. In particular, under family $j , \Delta \mathbf { z } = \mathbf { z } - \bar { \mathbf { z } } _ { j }$

## 3.2 STAGE 1: GATING DYNAMICS

Lemma 3.3 (Gating dynamics; proof in Section B.3 for the GLU). Under the assumptions of Theorem $3 . I ,$ run thefirst stage ofDefinition 2.3. For sufficiently large $n ,$ with probability at least $1 - \eta$ over the initialization and training tasks, thefirst stage ends at afinite time $t _ { \theta }$ with

$$
\operatorname* { m a x } \{ \| \bar { w } _ { \cdot \mathrm { G } } ( t _ { \theta } ) - e _ { 1 } \| _ { 1 } , \| \bar { w } _ { \cdot \mathrm { A } } ( t _ { \theta } ) - e _ { 2 } \| _ { 1 } \} = \mathcal { O } ( n ^ { - 1 } ) , \qquad \operatorname* { m a x } _ { q } ^ { \mathrm { m a x } } \| \theta _ { q } ( t _ { \theta } ) \| _ { 2 } = \mathcal { O } ( \log n ) .
$$

Proof sketch The population features $\bar { \mathbf { z } } _ { j }$ distinguish the families. Let $\pmb { \Sigma } ^ { ( j ) }$ be the asymptotic covariance matrix of $g$ given data from family j, that is,

$$
\mathbb { E } _ { j } [ ( g _ { i } - \mu ) ( g _ { k } - \mu ) ] = { \frac { \Sigma _ { i k } ^ { ( j ) } } { n } } + o ( n ^ { - 1 } ) .
$$

We show that, for the families we study, there must exist an expert i such that for all $k \neq i$ we have almost surely

$$
\Sigma _ { i i } ^ { ( j ) } < \Sigma _ { i k } ^ { ( j ) } < \Sigma _ { k k } ^ { ( j ) } .\tag{6}
$$

This implies that the optimal gating must exclusively choose the “best” expert. Finally, we observe that the main component of the loss can be identified as

$$
\mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \theta ) = \frac { 1 } { 4 } \sum _ { j } \left[ \sigma _ { \mu } ^ { 2 } \left( \sum _ { k = 1 } ^ { 2 } \bar { w } _ { k j } - 1 \right) ^ { 2 } + \frac { 1 } { n } \bar { w } _ { \cdot j } ^ { \top } \Sigma ^ { ( j ) } \bar { w } _ { \cdot j } \right] .\tag{7}
$$

For softmax, the normalization term vanishes as $\begin{array} { r } { \sum _ { k = 1 } ^ { 2 } { \bar { w } } _ { k j } = 1 } \end{array}$

Softmax MoE First, note that this problem is highly nonconvex due to w being generated by a softmax. Since $\textstyle \sum _ { i } w _ { i } = 1$ and the routing feature concentrates at $\bar { \mathbf { z } } _ { j }$ , the loss expansion gives $\begin{array} { r } { \mathcal { L } ( \theta ) = \mathcal { L } ^ { \mathrm { s m } } ( \theta ) \stackrel { \mathbf { \theta } } { + } o \left( 1 / n \right) } \end{array}$ , where

$$
\mathcal { L } ^ { \mathrm { s m } } ( \theta ) = \frac { 1 } { 4 n } \sum _ { j } \bar { w } _ { \cdot j } ^ { \top } \Sigma ^ { ( j ) } \bar { w } _ { \cdot j } .\tag{8}
$$

We can analyze the gradient flow on $\mathcal { L } ^ { \mathrm { s m } }$ and show global convergence. Correcting for the $o ( 1 / n )$ remainder finishes the proof.

Gated linear unit For the GLU, unbiasedness is not automatic even when the gates are evaluated at $\bar { \mathbf { z } } _ { j }$ . Our approach will be to analyze the gradient flow on $\mathcal { L } ^ { \mathrm { t r u n c a t e d } }$ and correct for the perturbation $\dot { \mathcal { L } } ( \theta ) - \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \theta )$ . The loss ${ \mathcal { L } } ^ { \mathrm { t r u n c a t e d } } ( \theta )$ has two distinct scales, where one scales with $\sigma _ { \mu } ^ { 2 }$ and one scales with $1 / n$ . The term $\begin{array} { r } { \sigma _ { \mu } ^ { 2 } \left( \sum _ { k = 1 } ^ { 2 } { \bar { w } } _ { k j } - 1 \right) ^ { 2 } } \end{array}$ corresponds (roughly) to the bias of the predictor. In the truncated loss, there exists a whole manifold of unbiased predictors, and we show that the gradient flow finds a solution near this manifold, while optimizing for the variance of the predictor. Choosing n sufficiently large lets us attack this problem using Tikhonov’s theorem and determine where on the manifold the gradient flow converges. Notably, $\bar { \mathcal { L } ( \boldsymbol { \theta } ) } - \mathcal { L } ^ { \dagger }$ <sup>truncated</sup>(θ) comes from the variance of w and thus scales with $\sigma _ { \mu } ^ { 2 }$ . Hence, the proof depends on $\sigma _ { \mu } ^ { 2 }$ being below some threshold. This is natural: a larger prior variance $\sigma _ { \mu } ^ { 2 }$ increases the cost of normalization error, amplifying inaccuracies in the gating and potentially leading to a collapse of the gating layer towards a single expert for some initializations.

## 3.3 STAGE 2: EXPERT DYNAMICS

For $t \geq t _ { \theta } , \theta$ is frozen. In this second stage, we analyze the dynamics of λ. We establish that the expert parameter assigned to samples from the Gaussian family moves toward zero, while uniform expert parameters grow toward endpoint estimation, and mixture expert parameters approach a certain limit. In the respective regimes, the leading expert-parameter flows satisfy $\dot { \lambda } _ { 1 } \asymp - \lambda _ { 1 } ^ { 3 } / n$ and $\dot { \lambda } _ { 2 } \times 1 / ( n \lambda _ { 2 } ^ { 2 } )$ assuming without loss of generality $\lambda _ { 2 } > 0$ . Controlling the residuals yields the following result.

Lemma 3.4 (Expert dynamics; proof in Section F.1). Under the assumptions ofTheorem $3 . I ,$ assume the first stage of Definition 2.3 ends at a finite time $t _ { \theta }$ satisfying

$$
\operatorname* { m a x } \{ \| \bar { w } _ { \cdot \mathrm { G } } - e _ { 1 } \| _ { 1 } , \| \bar { w } _ { \cdot \mathrm { A } } - e _ { 2 } \| _ { 1 } \} \leq C / n , \qquad \operatorname* { m a x } _ { q } \| \theta _ { q } \| _ { 2 } \leq C \log n ,
$$

and $| \lambda _ { 1 } ( t _ { \theta } ) | \le 1 , C ^ { - 1 } \le | \lambda _ { 2 } ( t _ { \theta } ) | \le 1$ ,for a constant $C < \infty$ independent of n. Use the training procedure in Definition 2.3. For sufficiently large $n ,$ with probability at least $1 - \eta ,$ uniformly over the possible $( \theta , \dot { \lambda } )$ at $t _ { \theta }$ satisfying these bounds, $\lceil \lambda _ { 1 } \rceil \leq 2$ until $| \lambda _ { 2 } |$ first reaches the value below, and this first time $t _ { \lambda }$ satisfies

$$
\left\{ \begin{array} { l l } { | \lambda _ { 2 } ( t _ { \lambda } ) | = n ^ { 1 - \epsilon } , \quad } & { t _ { \lambda } - t _ { \theta } \le C n ^ { 4 - 3 \epsilon } , \quad } & { U n i f o r m , } \\ { | \lambda _ { 2 } ( t _ { \lambda } ) | = \displaystyle \frac { \log ( 1 / \sigma _ { n } ^ { 2 } ) } { 4 \sqrt { 1 - \sigma _ { n } ^ { 2 } } } , \quad } & { t _ { \lambda } - t _ { \theta } \le C n / \sigma _ { n } ^ { 2 } , \quad G a u s s i a n m i x t u r e . } \end{array} \right.
$$

Moreover, $| \lambda _ { 1 } ( t _ { \lambda } ) | = o ( 1 )$ . The tasks used for training λ may also have been used for training θ.

## 3.4 EFFICIENCY OF THE FINAL ESTIMATOR

For every expert i, the error of the final estimator decomposes as

$$
\widehat { \mu } - \mu = \underbrace { g _ { i } - \mu } _ { \mathrm { e x p e r t \ e r r o r } } + \underbrace { \sum _ { k \neq i } w _ { k } ( g _ { k } - g _ { i } ) } _ { \mathrm { r o u t i n g \ e r r o r } } + \underbrace { \left( \sum _ { k } w _ { k } - 1 \right) g _ { i } } _ { \mathrm { n o r m a l i z a t i o n \ e r r o r } } .\tag{9}
$$

The normalization error vanishes for the softmax MoE, since $\textstyle \sum _ { k } w _ { k } = 1$ . Lemma 3.5 bounds the routing and normalization errors in $L ^ { 2 } ( \mathbb { P } _ { j } )$ through $\lVert \bar { w } _ { \cdot j } - e _ { i } \rVert _ { 1 }$

Lemma 3.5 (Accuracy of the final estimator; proof in Appendix C). Under the data model of Definition 2.1,fix $j \in \{ \mathrm { G } , \mathrm { A } \}$ . Condition on the trained $( \theta , { \bar { \lambda } } )$ and evaluate on an independent pair $( \mu , \mathbf { X } )$ , with max $_ k \parallel \theta _ { k } \parallel _ { 2 } \leq C$ log n. Then, for sufficiently large $n ,$ every expert $i \in [ 2 ]$ , and arbitrary λ,

$$
\sqrt { \mathbb { E } _ { j } \left[ ( \widehat { \mu } ( \mathbf { X } ) - \mu ) ^ { 2 } \right] } \leq \sqrt { \mathbb { E } _ { j } \left[ ( g _ { i } ( \mathbf { X } ) - \mu ) ^ { 2 } \right] } + C \| \bar { w } _ { \cdot j } - e _ { i } \| _ { 1 } \left( \sigma _ { \mu } ^ { 2 } + \sqrt { \mathbb { E } _ { j } \left[ \operatorname* { m a x } _ { \ell \in [ n ] } | X _ { \ell } - \mu | ^ { 4 } \right] } \right) ^ { 1 / 2 } .
$$

Proof sketch of Theorem 3.1. Under the conclusions of Lemma 3.4, the moment estimates in Lemma F.1, Lemma F.2, and Lemma F.3 give expert risks $( 1 + o ( 1 ) ) / n , \mathcal { O } ( n ^ { - 2 + \epsilon } )$ , and $\mathcal { O } ( \sigma _ { n } ^ { 2 } / n )$ respectively. Applying Lemma 3.5 and its proof with $i = 1$ for $j = \mathrm { G }$ and $i = 2$ for $j = \mathrm { A }$ , the gate bound of Lemma 3.3, and the sample-maximum moment bounds gives

$$
\Vert \widehat { \mu } - g _ { i } \Vert _ { L ^ { 2 } ( \mathbb { P } _ { j } ) } = \mathcal { O } \left( \frac { \sqrt { \sigma _ { \mu } ^ { 2 } + \log n } } { n } \right) .\tag{10}
$$

The triangle inequality then gives the stated risk bounds and preserves the Gaussian leading constant. The fixed-location claims follow from the translation identities and normalization estimate in Section D.3.6.

## 4 EXPERIMENTS

Figures 2 and 3 show joint training of the softmax MoE and GLU with the uniform alternative. We use $m = 8 0 0 0 , n = { \mathrm { 1 5 0 } } , K = 2 , { \overline { { d } } } = 1 0 .$ , and $\sigma _ { \mu } ^ { 2 } = 0 . 5$ , with 600,000 gradient-descent steps and constant step sizes 20 for θ and 100 for λ. Both learn family-specific selection as $| \lambda _ { 1 } |$ decreases toward zero and $| \lambda _ { 2 } |$ grows, with greater variation in the onset of specialization across GLU runs. Gaussian error stays near the sample-mean error, while uniform error falls below it toward the sample mid-range error. Training details, Gaussian-mixture results, and stagewise experiments are in Appendix I.

![](images/9165b37b8f86d3f348148e0fe27e93f13e34da0fa29d4d4e65f92ed95ce8530d.jpg)

![](images/47cd12c264ee93c79bc6f8d7d3a271c64678d857a0e2dcb60eed7a6c3e23598b.jpg)

![](images/82d7fc4c5a9f6fb6656699cd4d7193f171b58eb5288ffc5d3ef7486c06084ee9.jpg)  
Figure 2: Softmax MoE, joint training with the uniform alternative. Left: $( \bar { w } _ { 1 \mathrm { G } } , \bar { w } _ { 1 \mathrm { A } } )$ across runs, colored by iteration. Middle: median $| \lambda _ { 1 } |$ and $| \lambda _ { 2 } |$ , where thin lines are individual runs. Right: median validation error of $\widehat { \mu }$ on Gaussian and uniform contexts, with the sample mean and sample mid-range as references. Shading spans all ten runs.

![](images/eb3d8d08d0229bf5a1121025394fe642a842254fce197a0410b6fd79a3dff422.jpg)

![](images/159d3ba32b0e92f778ec124e4089ea840bac6a1a4dbd079898d659e7f30853bf.jpg)

![](images/92cf5a752212dddfd07200694d9527ea16b17e8847278b92e8dc37e94252983b.jpg)  
Figure 3: GLU, joint training with the uniform alternative, with panels and conventions as in Figure 2.

## 5 RELATED WORK

In-context learning of statistical procedures. Prior-data fitted networks and tabular foundation models learn prediction procedures by pretraining on synthetic tasks (Müller et al., 2024; Hollmann et al., 2025; Erickson et al., 2025), and related pretrained models estimate treatment effects, survival curves, mutual information, and Poisson means (Balazadeh et al., 2025; Qi et al., 2026; Peyrard & Cho, 2025; Teh et al., 2025). Their focus is prediction or posterior approximation, not how pretraining attains estimation rates that depend on an unknown family.

Statistical theory of pretrained predictors. Transformers can implement in-context algorithm selection (Bai et al., 2023), attain minimax rates over a known function class (Kim et al., 2024), and approximate Bayesian posterior predictors whose rates adapt to unknown structure (Wakayama & Suzuki, 2025; Ma et al., 2025); idealized pretrained Bayes estimators are similarly adaptive in empirical Bayes problems (Cannella et al., 2026). These guarantees concern the empirical risk minimizer or an idealized predictor; we follow the training trajectory and identify the internal parameter that carries the adaptation.

Training dynamics of attention. Gradient-based analyses of attention for in-context learning (Zhang et al., 2024a; Huang et al., 2023; Kim & Suzuki, 2024; Oko et al., 2024) include task allocation across softmax heads (Chen et al., 2024a), where the task handled by each head is fixed by the task structure. Here the family changes across contexts, so specialization must be paired with learned routing.

Adaptive estimation and architecture. Adaptivity is a classical objective in statistics (Lepskii, 1991; Donoho & Johnstone, 1995), including adaptive estimation of a symmetric location (Stone, 1975; Bickel et al., 1993); the rate gap in our problem stems from the irregularity of the uniform family (Ibragimov & Has’minskii, 1981). Our model is a small Set Transformer (Lee et al., 2019) whose second layer is a softmax mixture of experts (Jacobs et al., 1991) or a GLU (Shazeer, 2020). Appendix K discusses these connections in detail.

## 6 DISCUSSION

Our current analysis is limited by making some simplifying assumptions on the training procedure as well as the architecture. Without fixing parameters, we observe that the problem becomes much easier to learn in practice, but admits many possible optimal solutions making it much harder to study. We also observe that separating training into two stages slows down training while making the problem much easier to study. Future work should focus on addressing these issues. Additionally, we believe transformers are able to do parameter estimation for much larger families of distributions. Extending our analysis towards these would also be a natural next step. Finally, we observe that there is an upper bound to the prior variance $\sigma _ { \mu } ^ { 2 }$ , which seems to be unavoidable in a simple architecture. Still, finding an architecture that is able to handle arbitrary $\sigma _ { \mu } ^ { 2 }$ would be a clear next step.

## 7 CONCLUSION

We show that stagewise gradient flow on the empirical loss learns an adaptive location estimator without observing family labels. The estimator achieves Gaussian asymptotic efficiency, nearly optimal uniform rates, and order-optimal rates for Gaussian mixtures with shrinking component variance. Attention provides both distribution-sensitive features and candidate estimators, while the gating layer learns which expert to select. Softmax enforces normalization and exact translation equivariance, whereas the GLU learns approximate normalization before selecting experts.

## FUNDING STATEMENT

Y.S. Tan was supported by NUS Startup Grant A-8000448-00-00 and the Singapore Ministry of Education (MOE) AcRF Tier 1 Grants A-8002498-00-00 and A-8004458-00-00. Klusowski gratefully acknowledges support from the National Science Foundation through NSF CAREER award DMS-2239448 and from the Alfred P. Sloan Foundation through a Sloan Research Fellowship. Subhro Ghosh was supported in part by the NUS Dean’s Chair Associate Professorship E-146-00-0037-01 and the Singapore MOE grants A-8002014-00-00 and A-8003802-00-00.

## AI USE STATEMENT

In this work, we used generative AI tools to assist in the writing of proofs, formulate mathematical claims and implement the experiment code. Additionally, we used generative AI tools to polish the writing, find typesetting and grammar errors, suggest the structure of the main paper, and identify and summarize related work. We have reviewed all AI-assisted work. All proofs were verified by the authors. The experiment code was reviewed and tested by the authors. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## REFERENCES

Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models, 2023. URL https:// arxiv.org/abs/2211.15661.

Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, and Song Mei. Transformers as statisticians: Provable in-context learning with in-context algorithm selection. Advances in Neural Information Processing Systems, 2023.

Vahid Balazadeh, Hamidreza Kamkari, Valentin Thomas, Benson Li, Junwei Ma, Jesse C. Cresswell, and Rahul G. Krishnan. CausalPFN: Amortized causal effect estimation via in-context learning, 2025. URL https://arxiv.org/abs/2506.07918.

Rudolf Beran. Asymptotically efficient adaptive rank estimates in location models. The Annals of Statistics, 2(1):63–74, 1974. doi: 10.1214/aos/1176342613.

Nils Berglund. Perturbation theory of dynamical systems, 2001. URL https://arxiv.org/ abs/math/0111178.

P. J. Bickel. On Adaptive Estimation. The Annals of Statistics, 10(3):647 – 671, 1982. doi: 10.1214/aos/1176345863. URL https://doi.org/10.1214/aos/1176345863.

Peter J. Bickel, Chris A. J. Klaassen, Ya’acov Ritov, and Jon A. Wellner. Efficient and Adaptive Estimation for Semiparametric Models. Johns Hopkins University Press, 1993.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020. URL https: //arxiv.org/abs/2005.14165.

Dake Bu, Wei Huang, Andi Han, Atsushi Nitanda, Qingfu Zhang, Hau-San Wong, and Taiji Suzuki. Provable in-context vector arithmetic via retrieving task concepts, 2025. URL https://arxiv. org/abs/2508.09820.

Nick Cannella, Anzo Teh, Yanjun Han, and Yury Polyanskiy. Universal priors: solving empirical Bayes via Bayesian inference and pretraining, 2026. URL https://arxiv.org/abs/2602. 15136.

Siyu Chen, Heejune Sheen, Tianhao Wang, and Zhuoran Yang. Training dynamics of multi-head softmax attention for in-context learning: Emergence, convergence, and optimality. arXiv preprint arXiv:2402.19442, 2024a.

Xingwu Chen, Lei Zhao, and Difan Zou. How transformers utilize multi-head attention in in-context learning? a case study on sparse linear regression. arXiv preprint arXiv:2408.04532, 2024b.

Kyle Cranmer, Johann Brehmer, and Gilles Louppe. The frontier of simulation-based inference. Proceedings ofthe National Academy ofSciences, 117(48):30055–30062, 2020. ISSN 1091-6490. doi: 10.1073/pnas.1912789117. URL http://dx.doi.org/10.1073/pnas.1912789117.

Yann N. Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated convolutional networks, 2017. URL https://arxiv.org/abs/1612.08083.

David L. Donoho and Iain M. Johnstone. Adapting to unknown smoothness via wavelet shrinkage. Journal ofthe American Statistical Association, 90(432):1200–1224, 1995. doi: 10.1080/01621459. 1995.10476626.

Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Mutalik Desai, David Salinas, and Frank Hutter. TabArena: A living benchmark for machine learning on tabular data. In Advances in Neural Information Processing Systems, 2025. URL https://arxiv.org/ abs/2506.16791.

Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes, 2023. URL https://arxiv.org/abs/ 2208.01066.

Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. TabPFN: A transformer that solves small tabular classification problems in a second. International Conference on Learning Representations, 2023.

Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 637:319–326, 2025. doi: 10.1038/s41586-024-08328-6.

Yu Huang, Yuan Cheng, and Yingbin Liang. In-context convergence of transformers, 2023. URL https://arxiv.org/abs/2310.05249.

I. A. Ibragimov and R. Z. Has’minskii. Statistical Estimation: Asymptotic Theory. Springer New York, New York, NY, 1981. ISBN 978-1-4899-0027-2. doi: 10.1007/978-1-4899-0027-2.

Nikolaos Ignatiadis and Sid Kankanala. Compound decisions and empirical Bayes via Bayesian nonparametrics, 2026. URL https://arxiv.org/abs/2602.20115.

Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Geoffrey E. Hinton. Adaptive mixtures of local experts. Neural Computation, 3(1):79–87, 1991. doi: 10.1162/neco.1991.3.1.79.

Juno Kim and Taiji Suzuki. Transformers learn nonlinear features in context: Nonconvex mean-field dynamics on the attention landscape, 2024. URL https://arxiv.org/abs/2402.01258.

Juno Kim, Tai Nakamaki, and Taiji Suzuki. Transformers are minimax optimal nonparametric in-context learners, 2024. URL https://arxiv.org/abs/2408.12186.

Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, and Yee Whye Teh. Set transformer: A framework for attention-based permutation-invariant neural networks. In Proceedings ofthe 36th International Conference on Machine Learning, 2019.

E. L. Lehmann and George Casella. Theory of Point Estimation. Springer Texts in Statistics. Springer New York, NY, 2 edition, 1998. ISBN 978-0-387-98502-2. doi: 10.1007/b98854.

Oleg V. Lepskii. On a problem of adaptive estimation in Gaussian white noise. Theory of Probability and Its Applications, 35(3):454–466, 1991. doi: 10.1137/1135065.

Zihao Li, Yuan Cao, Cheng Gao, Yihan He, Han Liu, Jason M. Klusowski, Jianqing Fan, and Mengdi Wang. One-layer transformer provably learns one-nearest neighbor in context. arXiv preprint arXiv:2411.10830, 2024.

Tianyi Ma, Tengyao Wang, and Richard J. Samworth. Optimal in-context adaptivity and distributional robustness of transformers, 2025. URL https://arxiv.org/abs/2510.23254.

Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, and Frank Hutter. Transformers can do Bayesian inference, 2024. URL https://arxiv.org/abs/2112. 10510.

Thomas Nagler. Statistical foundations of prior-data fitted networks, 2023. URL https://arxiv. org/abs/2305.11097.

Kazusato Oko, Yujin Song, Taiji Suzuki, and Denny Wu. Pretrained transformer efficiently learns lowdimensional target functions in-context. In Advances in Neural Information Processing Systems, 2024. URL https://arxiv.org/abs/2411.02544.

Maxime Peyrard and Kyunghyun Cho. Meta-statistical learning: Supervised learning of statistical estimators. arXiv preprint arXiv:2502.12088, 2025.

Shi-ang Qi, Vahid Balazadeh, Michael Cooper, Russell Greiner, and Rahul G Krishnan. SurvivalPFN: Amortizing survival prediction via in-context Bayesian inference. In 2nd ICML Workshop on Foundation Modelsfor Structured Data, 2026. URL https://openreview.net/forum? id=PDik7bpFhE.

J. Shao. Mathematical Statistics. Springer Texts in Statistics. Springer, 2003. ISBN 9780387953823.

Noam Shazeer. GLU variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer, 2017. URL https://arxiv.org/abs/1701.06538.

Charles Stein. Efficient nonparametric testing and estimation. In Proceedings ofthe Third Berkeley Symposium on Mathematical Statistics and Probability, Volume 1: Contributions to the Theory of Statistics, volume 1, pp. 187–195. University of California Press, 1956.

C. J. Stone. Adaptive maximum likelihood estimators of a location parameter. Annals ofStatistics, 3 (2):267–284, 1975. doi: 10.1214/aos/1176343056.

Anzo Teh, Mark Jabbour, and Yury Polyanskiy. Solving empirical Bayes via transformers, 2025. URL https://arxiv.org/abs/2502.09844.

Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. Springer Series in Statistics. Springer New York, 2009. doi: 10.1007/b13794.

Tomoya Wakayama and Taiji Suzuki. In-context learning is provably Bayesian inference: A generalization theory for meta-learning, 2025. URL https://arxiv.org/abs/2510.10981.

Han-Jia Ye, Si-Yang Liu, Hao-Run Cai, Qi-Le Zhou, and De-Chuan Zhan. A closer look at deep learning methods on tabular datasets, 2025. URL https://arxiv.org/abs/2407.00956.

Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabás Póczos, Ruslan Salakhutdinov, and Alexander J. Smola. Deep sets. In Advances in Neural Information Processing Systems, 2017.

Qiong Zhang, Yan Shuo Tan, Qinglong Tian, and Pengfei Li. TabPFN: One model to rule them all? arXiv preprint arXiv:2505.20003, 2025. URL https://arxiv.org/abs/2505.20003.

Ruiqi Zhang, Spencer Frei, and Peter L. Bartlett. Trained transformers learn linear models in-context. Journal ofMachine Learning Research, 25(49):1–55, 2024a.

Ruiqi Zhang, Jingfeng Wu, and Peter L. Bartlett. In-context learning of a linear transformer block: Benefits of the MLP component and one-step GD initialization. Advances in Neural Information Processing Systems, 2024b.

## A OUTLINE

Proof map. Lemma B.5 gives the reparameterization used in the dynamics, and Lemma F.5 establishes the geometry of z. Appendix B assembles the main theorem from the fresh-context risk bound in Appendix C, the gating analysis in Appendix D, and the expert-parameter analysis in Appendix F. Appendix G collects auxiliary estimates, and Appendix H gives the Gaussianmixture benchmark. Appendix I reports additional experiments, and Appendix J discusses the role of LayerNorm. Appendix K gives an extended discussion of related work. See section 1 for notation.

## B PROOF OF THEOREM 3.1

## B.1 EXPERT HIERARCHY

Definition B.1 (Expert covariance and the better expert). Let $\pmb { \Sigma } ^ { ( j ) }$ be the matrix with entries $\Sigma _ { i k } ^ { ( j ) } = \Sigma _ { g } ( \lambda _ { i } , \lambda _ { k } )$ evaluated under the centered law $Q _ { j }$ . The symmetrized covariance formula is given in Definition F.9, with family-specific calculations in the expert-covariance appendix. Expert $i \in \{ 1 , 2 \}$ is better for family $j \in \{ \mathrm { G } , \mathrm { A } \}$ if, for the other expert $k \neq i ,$

$$
\Sigma _ { i i } ^ { ( j ) } < \Sigma _ { i k } ^ { ( j ) } < \Sigma _ { k k } ^ { ( j ) } .\tag{11}
$$

The experts satisfy the expert hierarchy if each family has a better expert. Its strict covariance gaps are the two positive differences in the ordering.

Lemma B.2 (Gaussian distribution satisfies the expert hierarchy). Let $Y \sim { \mathcal { N } } ( 0 , 1 )$ and $\lambda _ { 1 } , \lambda _ { 2 } \in \mathbb { R }$ with $0 < | \lambda _ { 1 } | < | \lambda _ { 2 } |$ |. Then

$$
\Sigma _ { g } ( \lambda _ { 1 } , \lambda _ { 1 } ) < \Sigma _ { g } ( \lambda _ { 1 } , \lambda _ { 2 } ) < \Sigma _ { g } ( \lambda _ { 2 } , \lambda _ { 2 } ) .\tag{12}
$$

Proof. The statement follows from $\Sigma _ { g } ( \lambda _ { 1 } , \lambda _ { 2 } ) = \cosh ( \lambda _ { 1 } \lambda _ { 2 } ) + \lambda _ { 1 } \lambda _ { 2 } \sinh ( \lambda _ { 1 } \lambda _ { 2 } )$ , which is strictly increasing in $| \lambda _ { 1 } \lambda _ { 2 } |$ □

Lemma B.3 (Gaussian mixture distribution satisfies the expert hierarchy). For every fixed $0 < \delta < 1$ there exists $\sigma _ { \delta } > 0$ such that for all $0 < \sigma _ { n } < \sigma _ { \delta } < 1 / \sqrt { 2 }$ and $0 \leq | \lambda _ { 1 } | < | \lambda _ { 2 } | \leq L _ { \operatorname* { m a x } } =$ $\left( 1 - \delta \right) \log ( 1 / \sigma _ { n } )$

$$
\Sigma _ { g } ( \lambda _ { 1 } , \lambda _ { 1 } ) > \Sigma _ { g } ( \lambda _ { 1 } , \lambda _ { 2 } ) > \Sigma _ { g } ( \lambda _ { 2 } , \lambda _ { 2 } ) .\tag{13}
$$

Proof. See Lemma F.16.

Lemma B.4 (Uniform distribution satisfies the expert hierarchy). Let $Y \sim \mathrm { U n i f } [ - { \sqrt { 3 } } , { \sqrt { 3 } } ]$ and $\lambda _ { 1 } , \lambda _ { 2 } \in [ - 1 , 1 ]$ with $| \lambda _ { 1 } | < | \lambda _ { 2 } |$ . Then

$$
\Sigma _ { g } ( \lambda _ { 1 } , \lambda _ { 1 } ) > \Sigma _ { g } ( \lambda _ { 1 } , \lambda _ { 2 } ) > \Sigma _ { g } ( \lambda _ { 2 } , \lambda _ { 2 } ) .\tag{14}
$$

Proof. See Corollary F.13.

In the parameter ranges above, $0 < | \lambda _ { 1 } | < | \lambda _ { 2 } |$ makes expert 1 better on Gaussian contexts and expert 2 better on alternative contexts.

## B.2 GATING-LAYER DYNAMICS

Lemma B.5 (Decomposition of the gating parameters). For the GLU, let $Z _ { j k } = \langle \bar { \bf z } _ { j } , \bar { \bf z } _ { k } \rangle$ and suppose $Z \succ 0$ . Each gating parameter has the decomposition

$$
\theta _ { q } = \theta _ { q } ^ { \parallel } + \theta _ { q } ^ { \perp } , \qquad \theta _ { q } ^ { \parallel } = \sum _ { \ell , j } ( Z ^ { - 1 } ) _ { \ell j } \sigma ^ { - 1 } ( \bar { w } _ { q j } ) \bar { \bf z } _ { \ell } , \qquad \theta _ { q } ^ { \perp } \perp \mathrm { s p a n } \{ \bar { \bf z } _ { j } \} _ { j \in \{ { \bf G } , { \bf A } \} } .
$$

Thus θ is equivalently parameterized by $( \bar { w } , \theta ^ { \perp } )$ . Moreover,

$$
\| \theta _ { q } \| _ { 2 } \leq \| \theta _ { q } ^ { \perp } \| _ { 2 } + \sqrt { \frac { 2 } { \lambda _ { \operatorname* { m i n } } ( Z ) } } \operatorname* { m a x } _ { j } \log { \frac { 1 } { \bar { w } _ { q j } ^ { \prime } } } .
$$

Proof. Taking the inner product of $\theta _ { q } ^ { \parallel }$ with $\bar { \mathbf { z } } _ { j }$ gives $\sigma ^ { - 1 } ( \bar { w } _ { q j } ) = \langle \theta _ { q } , \bar { \mathbf { z } } _ { j } \rangle$ , so the stated decomposition is orthogonal. Moreover,

$$
\| \theta _ { q } ^ { \parallel } \| _ { 2 } ^ { 2 } = \sum _ { i , j } \sigma ^ { - 1 } ( \bar { w } _ { q i } ) ( Z ^ { - 1 } ) _ { i j } \sigma ^ { - 1 } ( \bar { w } _ { q j } ) \leq \frac { 1 } { \lambda _ { \operatorname* { m i n } } ( Z ) } \sum _ { j } | \sigma ^ { - 1 } ( \bar { w } _ { q j } ) | ^ { 2 } .
$$

The conclusion follows from the triangle inequality and $| \log ( p / ( 1 - p ) ) | \leq \log ( 1 / ( p ( 1 - p ) ) )$ for $0 < p < 1$ □

For the truncated flow, $\dot { \theta } _ { q }$ lies in span $\{ \bar { \bf z } _ { j } \} _ { j \in \{ \mathrm { G } , \mathrm { A } \} }$ , so $\theta _ { q } ^ { \perp } ( t ) = \theta _ { q } ^ { \perp } ( 0 )$ . Under the assumptions of Lemma D.10, the full flow also has $\| \theta _ { q } ^ { \perp } ( t ) \| _ { 2 } = \mathcal { O } ( 1 )$ for $t _ { \mathrm { c r i t } } \leq t \leq t _ { \theta }$

Definition B.6 (Truncated loss). For fixed λ, define the bias loss

$$
\mathcal { L } ^ { \mathrm { b i a s } } ( \theta ) = \frac { \sigma _ { \mu } ^ { 2 } } { 4 } \sum _ { j } ( \bar { w } _ { 1 j } + \bar { w } _ { 2 j } - 1 ) ^ { 2 }
$$

and recall the truncated loss from (7)

$$
\mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \theta ) = \frac { 1 } { 4 } \sum _ { j } \left[ \sigma _ { \mu } ^ { 2 } \left( \sum _ { k = 1 } ^ { 2 } \bar { w } _ { k j } - 1 \right) ^ { 2 } + \frac { 1 } { n } \bar { w } _ { \cdot j } ^ { \top } \Sigma ^ { ( j ) } \bar { w } _ { \cdot j } \right] .
$$

Lemma B.7 (Stationary points of the truncated flow). Fix λ and $\sigma _ { \mu } ^ { 2 } > 0 ,$ , suppose the expert hierarchy ofDefinition B.1 holds, and let $k _ { j }$ denote the better expert infamily j. Let $e _ { 1 } = ( 1 , 0 ) ^ { \top } , e _ { 2 } = ( 0 , 1 ) ^ { \top }$ and suppose $Z \succ 0 .$ . The truncatedflow is

$$
\dot { \bar { w } } _ { q k } = - \frac { 1 } { 2 } \bar { w } _ { q k } ^ { \prime } \sum _ { j } Z _ { j k } \bar { w } _ { q j } ^ { \prime } S _ { q j } ( \bar { w } ) ,
$$

$$
S _ { q j } ( \bar { w } ) = \sigma _ { \mu } ^ { 2 } \left( \sum _ { i = 1 } ^ { 2 } \bar { w } _ { i j } - 1 \right) + \frac { 1 } { n } \sum _ { i = 1 } ^ { 2 } \Sigma _ { q i } ^ { ( j ) } \bar { w } _ { i j } .
$$

Its stationary points form the set

$$
\mathcal { S } = \left. \bar { w } ^ { \circ } \in [ 0 , 1 ] ^ { 2 \times 2 } : \bar { w } _ { q j } ^ { \circ } ( 1 - \bar { w } _ { q j } ^ { \circ } ) S _ { q j } ( \bar { w } ^ { \circ } ) = 0 f o r a l l q , j \right. ,
$$

with

$$
\begin{array} { r c l } { { \bar { w } ^ { \circ } \in { \mathcal S } } } & { { \longleftrightarrow } } & { { \bar { w } _ { \cdot j } ^ { \circ } \in \Bigl \{ 0 , e _ { 1 } , e _ { 2 } , e _ { 1 } + e _ { 2 } , } } \\ { { } } & { { } } & { { \displaystyle \frac { n \sigma _ { \mu } ^ { 2 } } { n \sigma _ { \mu } ^ { 2 } + \Sigma _ { 1 1 } ^ { ( j ) } } e _ { 1 } , \displaystyle \frac { n \sigma _ { \mu } ^ { 2 } } { n \sigma _ { \mu } ^ { 2 } + \Sigma _ { 2 2 } ^ { ( j ) } } e _ { 2 } \Bigr \} \quad ( j \in \{ \mathrm { G } , \mathrm { A } \} ) . } } \end{array}
$$

The unique optimum is

$$
\begin{array} { c } { { \displaystyle \bar { w } _ { \cdot j } ^ { * } = \frac { n \sigma _ { \mu } ^ { 2 } } { n \sigma _ { \mu } ^ { 2 } + \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } } e _ { k _ { j } } , } } \\ { { \displaystyle \mathrm { a r g m i n } \ell ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ) = \{ \bar { w } ^ { * } \} . } } \\ { { \displaystyle \bar { w } \in [ 0 , 1 ] ^ { 2 \times 2 } } } \end{array}
$$

Proof. See Section D.3.1.

Lemma B.8 (Convergence to the critical manifold). Consider the unperturbed flow

$$
\dot { \theta } = - \nabla _ { \theta } \mathcal { L } ^ { \mathrm { b i a s } } .
$$

Assume $Z \succ 0 ,$ , fixed $\sigma _ { \mu } ^ { 2 } > 0$ , and initialization $\theta ( 0 )$ such that ma $\mathfrak { c } _ { q , j } | \bar { w } _ { q j } ( 0 ) - 1 / 2 | \le \delta$ for sufficiently small δ. Then all gates remain in $[ 1 / 4 , 3 / 4 ]$ , and

$$
\operatorname* { m a x } _ { j } | \bar { w } _ { 1 j } ( t ) + \bar { w } _ { 2 j } ( t ) - 1 | \leq C \delta e ^ { - \alpha t } , \qquad \alpha = \frac { 9 \sigma _ { \mu } ^ { 2 } \lambda _ { \operatorname* { m i n } } ( Z ) } { 2 5 6 } .
$$

For the truncated flow at fixed $\lambda ,$ there is also a time $t _ { c r i t } = \mathcal { O } ( \log n )$ such that all gates remain in $[ 1 / 4 , 3 / 4 ]$ up to $t _ { c r i t }$ and

$$
\operatorname* { m a x } _ { j } \vert \bar { w } _ { 1 j } ( t _ { c r i t } ) + \bar { w } _ { 2 j } ( t _ { c r i t } ) - 1 \vert = \mathcal { O } ( n ^ { - 1 } ) .
$$

The same conclusion at $t _ { c r i t }$ holdsfor thefull populationflow atfixed λ ifma $\mathrm { x } _ { q } \| \theta _ { q } ( 0 ) \| _ { 2 }$ is bounded independently ofn. Under this initialization bound and the data conditions ofLemma $\ddot { D } . 4 , f i x \epsilon > 0$ and sample $m = \widetilde \Omega ( n ^ { 1 + \epsilon } )$ independent tasks, with $m \in \mathrm { p o l y } ( n )$ . With probability at least $1 - 1 / n$ the empirical flow $\dot { \theta } = - \nabla _ { \theta } \hat { \mathcal { L } } _ { m }$ has the same conclusion at $t _ { c r i t } = \mathcal { O } ( \log n )$ , with all gates in $[ 1 / 4 , 3 { \dot { / } } 4 ]$ and sup $\begin{array} { r } { \mathfrak { \ d } _ { 0 \le t \le t _ { c r i t } } \| \theta ( t ) \| _ { 2 } = \mathcal { O } ( 1 ) } \end{array}$ . Constants are independent ofn, but may depend on $\sigma _ { \mu } ^ { 2 } ,$ $Z ,$ and thefixed λ, ω.

## Proof. See Section D.3.2.

Lemma B.9 (Convergence below loss barrier). Fix λ independently of n and suppose the strict expert hierarchy holds. Assume $Z \succ 0$ and $Z _ { \mathrm { G A } } < \operatorname* { m i n } \{ Z _ { \mathrm { G G } } , \mathrm { \bar { Z } _ { A A } } \}$ . Suppose

$$
\bar { w } ( t _ { c r i t } ) \in [ 1 / 4 , 3 / 4 ] ^ { 2 \times 2 } , \qquad n \sigma _ { \mu } ^ { 2 } \operatorname* { m a x } _ { j } | \bar { w } _ { 1 j } ( t _ { c r i t } ) + \bar { w } _ { 2 j } ( t _ { c r i t } ) - 1 | \leq C _ { 0 } ,
$$

with $C _ { 0 }$ independent ofn. For the truncatedflow andfixed $\sigma _ { \mu } ^ { 2 } > 0 ,$ , there exist $S , \kappa > 0$ , independent of n, such that, for sufficiently large n,

$$
\mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t _ { c r i t } + n S ) ) < \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { \circ } ) - \frac { \kappa } { n }
$$

for every

$$
\bar { w } ^ { \circ } \in \left\{ \left( \begin{array} { c c } { 1 } & { 1 } \\ { 0 } & { 0 } \end{array} \right) , \left( \begin{array} { c c } { 0 } & { 0 } \\ { 1 } & { 1 } \end{array} \right) , \left( \begin{array} { c c } { 0 } & { 1 } \\ { 1 } & { 0 } \end{array} \right) \right\} .\tag{15}
$$

For thefull populationflow, assume also that ma $\mathfrak { c } _ { q } \| \theta _ { q } ^ { \perp } ( t _ { c r i t } ) \| _ { 2 }$ is bounded independently ofn and $\sigma _ { \mu } ^ { 2 } .$ . For sufficiently smallfixed $\sigma _ { \mu } ^ { 2 } > 0$ and then sufficiently large $n ,$ both $\mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t _ { c r i t } + n S ) )$ and $\mathcal { L } ( \theta ( t _ { c r i t } + n S ) )$ satisfy the same upper bound. Under these assumptions and the data and samplesize conditions of the empirical part of Lemma B.8, the same conclusions hold for the empirical flow with probability at least $1 - 1 / n$ , uniformly over possibly data-dependent initial parameters satisfying the stated bounds. For the full population and empirical flows, $\begin{array} { r } { \operatorname* { s u p } _ { t _ { c r i t } \leq t \leq t _ { c r i t } + n S } \| \theta ( t ) \| _ { 2 } = \mathcal { O } ( 1 ) } \end{array}$ with a bound independent ofn and sufficiently small $\sigma _ { \mu } ^ { 2 } .$

Proof. See Section D.3.3.

Lemma B.10 (Flow at fixed λ). Fix λ independently of n and suppose the strict inequalities in Definition B.1 hold. Assume $Z \succ 0 , Z _ { \mathrm { G A } } \geq \dot { 0 }$ , andfixed $\overset { \cdot } { \sigma _ { \mu } ^ { 2 } } > 0$ . Use Lemma $B . 7$ with $\bar { k } _ { \mathrm { G } } = 1$ and $k _ { \mathrm { { A } } } = 2 .$ . Suppose $\bar { w } ( t _ { c r i t } ) \in ( 0 , 1 ) ^ { 2 \times 2 }$ , L<sup>truncated</sup> $\cdot ( \bar { w } ( t _ { c r i t } ) ) \dot { = } \mathcal { O } ( 1 / n )$ , and,for somefixed $\kappa > 0 ,$

$$
\mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t _ { c r i t } ) ) \leq \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { \circ } ) - \frac { \kappa } { n }
$$

for every $\bar { w } ^ { \circ }$ in (15). For sufficiently large n, the truncated flow reaches a finite time $t _ { \theta } \geq t _ { c r i t }$ with

$$
\| \bar { w } _ { \cdot j } ( t _ { \theta } ) - e _ { k _ { j } } \| _ { 2 } \leq \frac { 1 } { n } + \frac { \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } } { n \sigma _ { \mu } ^ { 2 } + \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } } = \mathcal { O } ( n ^ { - 1 } ) .
$$

For the full population flow, assume that $\| \theta ( t _ { c r i t } ) \| _ { 2 }$ and $n \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t _ { c r i t } ) )$ are bounded independently ofn and $\sigma _ { \mu } ^ { 2 } .$ . For sufficiently small fixed $\dot { \sigma } _ { \mu } ^ { 2 } > 0$ and then sufficiently large n, let $C _ { 0 }$ be the constantfrom Lemma D.10 and set

$$
\rho = \frac { C _ { 0 } } { n \sigma _ { \mu } ^ { 2 } } .
$$

There is a time $t _ { \theta } \leq t _ { c r i t } + 2 C _ { 0 } ^ { 2 } n ^ { 2 } \sigma _ { \mu } ^ { 2 }$ with

$$
\| \bar { w } _ { \cdot j } ( t _ { \theta } ) - \bar { w } _ { \cdot j } ^ { * } \| _ { 2 } \leq \rho + \frac { \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } } { n \sigma _ { \mu } ^ { 2 } + \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } } , \qquad \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t _ { \theta } ) ) \leq \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t _ { c r i t } ) ) .
$$

In particular, the gate error is $\mathcal { O } ( n ^ { - 1 } )$ , and $\begin{array} { r } { \operatorname* { s u p } _ { t _ { c r i t } \leq t \leq t _ { \theta } } \| \theta ( t ) \| _ { 2 } = \mathcal { O } ( \log n ) } \end{array}$

Proof. See Section D.3.4.

Lemma B.11 (Empirical flow at fixed λ). Assume the data conditions ofLemma D.4 and thefull-flow assumptions ofLemma B.10, with $\| \theta ( t _ { c r i t } ) \| _ { 2 } \leq B$ and nL<sup>truncated</sup> $( \bar { w } ( t _ { c r i t } ) ) \leq c .$ Let $C _ { 0 }$ and ρ be as there. Fix $\epsilon > 0$ and sample m independent tasks according to Definition 2.1, where $m \in \mathrm { p o l y } ( n )$ and

$$
m \geq n ^ { 1 + \epsilon } .
$$

Consider $\boldsymbol { \dot { \theta } } = - \nabla _ { \theta } \boldsymbol { \hat { \mathcal { L } } } _ { m }$ from possibly data-dependent parameters $\theta ( t _ { c r i t } )$ satisfying these assumptions. For sufficiently large n, with probability at least $1 - 1 / n ,$ , there is a time $t _ { \theta } \stackrel { \triangledown } { \le } t _ { c r i t } + 2 C _ { 0 } ^ { 2 } n ^ { 2 } \sigma _ { \mu } ^ { 2 }$ with

$$
\| \bar { w } _ { \cdot j } ( t _ { \theta } ) - \bar { w } _ { \cdot j } ^ { * } \| _ { 2 } \leq \rho + \frac { \Sigma _ { k j k _ { j } } ^ { ( j ) } } { n \sigma _ { \mu } ^ { 2 } + \Sigma _ { k j k _ { j } } ^ { ( j ) } } , \qquad \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t _ { \theta } ) ) \leq \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t _ { c r i t } ) ) ,
$$

$$
\operatorname* { s u p } _ { t _ { c i l } \le t \le t _ { \theta } } \| \theta ( t ) \| _ { 2 } \le B + 3 C _ { 0 } \log ( 1 + c n \sigma _ { \mu } ^ { 2 } ) = \mathcal { O } ( \log n ) , \qquad \operatorname* { s u p } _ { t _ { c i l } \le t \le t _ { \theta } } \| \theta _ { q } ^ { \perp } ( t ) \| _ { 2 } = \mathcal { O } ( 1 ) ,
$$

and

$$
{ \mathcal { L } } ( \theta ( t _ { \theta } ) ) = { \mathcal { O } } ( n ^ { - 1 } ) , \qquad { \hat { \mathcal { L } } } _ { m } ( \theta ( t _ { \theta } ) ) = { \mathcal { O } } ( n ^ { - 1 } ) .
$$

Proof. See Section D.3.5.

## B.3 PROOF OF THE GLU PART OF LEMMA 3.3

Proof of the GLU part of Lemma 3.3. Since λ is fixed in the first stage, $\lambda ( t _ { \theta } ) = \lambda ( 0 )$ , and $| \lambda _ { k } ( 0 ) | \le$ 1 lies in the ranges of Lemma B.2, Lemma B.3 and Lemma B.4 for sufficiently large n. By those lemmas, $0 < | \bar { \lambda _ { 1 } } ( 0 ) | < | \lambda _ { 2 } ( 0 ) |$ makes expert 1 better in family G and expert 2 better in family $\mathrm { A }$ Under this hierarchy and $\bar { w } _ { 1 j } + \bar { w } _ { 2 j } = 1$ , selecting the better expert minimizes the variance in family $j .$ Lemma F.5 gives $Z \succ 0 , Z _ { \mathrm { G A } } \doteq 0$ and $Z _ { \mathrm { G A } } <$ < min $\{ Z _ { \mathrm { G G } } , \mathrm { \bar { Z } _ { A A } } \}$ for $d \geq 3 ,$ , almost surely over ω.

Apply Lemma B.8 to obtain the gate and parameter bounds at $t _ { \mathrm { c r i t } } .$ . Then Lemma B.9 gives, at $t _ { \mathrm { c r i t } } + n S$ , the loss bound below every suboptimal corner and $\lVert \theta ( t _ { \mathrm { c r i t } } + n S ) \rVert _ { 2 } = \mathcal { O } ( 1 )$ . Since the loss at each such corner is $\mathcal { O } ( n ^ { - 1 } )$ , these bounds supply the entry assumptions of Lemma B.11, with its initial time shifted to $t _ { \mathrm { c r i t } } + n S$ . Since $m = \widetilde \Omega ( n ^ { 1 + \epsilon } ) \ge n ^ { 1 + \epsilon / 2 }$ for sufficiently large $^ { n , }$ apply that lemma with $\epsilon / 2$ in place of ϵ to obtain a finite time $t _ { \theta }$ with $\lVert \bar { w } . _ { j } ( t _ { \theta } ) - \bar { w } _ { \cdot j } ^ { * } \rVert _ { 2 } ^ { - } = \bar { \mathcal { O } } ( n ^ { - 1 } )$ and ma $\mathrm { x } _ { q } \| \theta _ { q } ( t _ { \theta } ) \| _ { 2 } = \mathcal { O } ( \log n )$ . Lemma B.7 gives $\| \bar { w } _ { \cdot j } ^ { * } - e _ { k _ { j } } \| _ { 1 } = \mathcal { O } ( n ^ { - 1 } )$ , so

$$
\operatorname* { m a x } \{ \| \bar { w } _ { \cdot \mathrm { G } } ( t _ { \theta } ) - e _ { 1 } \| _ { 1 } , \| \bar { w } _ { \cdot \mathrm { A } } ( t _ { \theta } ) - e _ { 2 } \| _ { 1 } \} = \mathcal { O } ( n ^ { - 1 } ) .
$$

The probability that any of these first-stage conclusions fails is $\mathcal { O } ( n ^ { - 1 } )$ .

## B.4 ATTENTION-EXPERT DYNAMICS

Corollary B.12 (Accuracy with fixed gating parameters). Under the assumptions ofLemma 3.4, stop training λ at $t _ { \lambda }$ . For sufficiently large n, with probability at least $1 - \eta ,$ , the mean-squared errors on independent test samples satisfy

$$
\mathbb { E } _ { \mathbb { G } } [ ( \widehat { \mu } - \mu ) ^ { 2 } ] = \frac { 1 + o ( 1 ) } { n } , \qquad \mathbb { E } _ { \mathbb { A } } [ ( \widehat { \mu } - \mu ) ^ { 2 } ] = \left\{ \begin{array} { l l } { \mathcal { O } ( n ^ { - 2 + \epsilon } ) , } & { U n i f o r m , } \\ { \mathcal { O } ( \sigma _ { n } ^ { 2 } / n ) , } & { G a u s s i a n m i x t u r e . } \end{array} \right.
$$

Proof. See Section D.3.6.

## B.5 THEOREM ASSEMBLY

ProofofTheorem 3.1. Apply Lemma 3.3 for the respective architecture and freeze the gating parameters at $t _ { \theta }$ . For the softmax MoE, represent the frozen weights by the sigmoid parameters $( \theta _ { 1 } - \theta _ { 2 } , \theta _ { 2 } - \theta _ { 1 } )$ , which retain the O(log n) norm bound and give the same loss for every λ. Since $\lambda ( t _ { \theta } ) = \lambda ( 0 )$ , the expert parameters retain their initialization bounds, and $m = \widetilde { \Omega } ( n ^ { 1 + \epsilon } )$ satisfies the sample-size requirement of Lemma 3.4. Its uniform conclusion applies to the data-dependent parameters at $t _ { \theta }$ using the same training tasks and gives a finite target time $t _ { \lambda }$ . Finally, Corollary B.12 gives the stated test risks at $t _ { \lambda }$ , with joint probability at least $1 - \eta$ by the failure bounds below. The bounds at fixed $\mu$ follow from the translation identities in Section D.3.6.

Under Definition 2.2, after relabeling, $0 < | \lambda _ { 1 } ( 0 ) | < | \lambda _ { 2 } ( 0 ) |$ almost surely and $\mathbb { P } ( | \lambda _ { 2 } ( 0 ) | < C ^ { - 1 } ) =$ $C ^ { - 2 }$ . For sufficiently large $C > 1$ , independent of n, both $C ^ { - 2 } \leq \eta / 4$ and $\mathbb { P } ( \operatorname* { m a x } _ { q } \| \theta _ { q } ( 0 ) \| _ { 2 } >$ $C ) \leq \eta / 4$ . For the GLU, choose $\sigma _ { \theta } ^ { 2 }$ sufficiently small that $\begin{array} { r } { \mathbb { P } ( \operatorname* { m a x } _ { q } \| \theta _ { q } ( 0 ) \| _ { 2 } > \operatorname* { m i n } \{ \bar { C } , 4 \delta \} ) \leq } \end{array}$ $\eta / 4 ,$ , where $\delta > 0$ is the sufficiently small constant in Lemma B.8. On the complementary event, $\begin{array} { r } { \operatorname* { m a x } _ { q , j } | \bar { w } _ { q j } ( 0 ) - 1 / 2 | \leq \delta . } \end{array}$ since $\| \bar { \bf z } _ { j } \| _ { 2 } = 1$ and $| \sigma ( u ) - 1 / 2 | \le | u | / 4$ Thus the required initialization bounds hold jointly with probability at least $1 - \eta / 2$ . For sufficiently large $n ,$ the remaining failure probabilities are at most $\eta / 4$ per training stage, giving probability at least $1 - \eta$ in both Lemma 3.3 and Theorem 3.1. □

## C ACCURACY OF THE FINAL ESTIMATOR

ProofofLemma 3.5. It suffices to consider sigmoid gates, since for $K = 2$ the softmax weights equal the sigmoid weights with parameters $( \theta _ { 1 } - \theta _ { 2 } , \theta _ { 2 } - \theta _ { 1 } )$ , whose norms are still ${ \mathcal { O } } ( \log n )$ . Since $h ( { \mathbf { X } } + \mu { \mathbf { 1 } } , a ) = h ( { \mathbf { X } } , a ) + \mu .$ , centering the feature coordinates removes the location exactly:

$$
\mathbf { z } ( \mathbf { X } + \mu \mathbf { 1 } ) = \mathbf { z } ( \mathbf { X } ) .
$$

Thus z and the gates depend only on the centered observations and are independent of $\mu .$ . Each centered feature satisfies

$$
h ( \mathbf { X } , \omega _ { r } ) - \mu = \frac { n ^ { - 1 } \sum _ { \ell = 1 } ^ { n } e ^ { \omega _ { r } ( X _ { \ell } - \mu ) } ( X _ { \ell } - \mu ) } { n ^ { - 1 } \sum _ { \ell = 1 } ^ { n } e ^ { \omega _ { r } ( X _ { \ell } - \mu ) } } .
$$

The summands have finite moments of every order, uniformly in n also for the Gaussian mixture, since its laws are uniformly 1-sub-Gaussian and ω is fixed. The population denominators are positive, and by Lemma F.5 the centered population feature norms are bounded below, so the normalization map is locally Lipschitz there. Moment bounds for the empirical averages therefore give

$$
\mathbb { E } _ { j } \left[ \lVert \Delta \mathbf { z } \rVert _ { 2 } ^ { 2 p } \right] = \mathcal { O } ( n ^ { - p } )
$$

for every fixed integer $p \geq 1 { : }$ on a fixed neighborhood of the population averages this follows from local Lipschitz continuity, and outside that neighborhood it follows from $\| \Delta \mathbf { z } \| _ { 2 } \leq 2$ and Markov’s inequality applied to the averages. Using max $_ k \parallel \theta _ { k } \parallel _ { 2 } \leq C$ log n, another application of Markov’s inequality gives

$$
\mathbb { P } \left[ \operatorname* { m a x } _ { k } | \langle \theta _ { k } , \Delta \mathbf { z } \rangle | > 1 | \tau = j \right] = \mathcal { O } \left( n ^ { - p } ( \log n ) ^ { 2 p } \right) .
$$

Since ma $\mathrm { x } _ { k } \mid \langle \theta _ { k } , \Delta \mathbf { z } \rangle \mid \leq 2 C$ log n, splitting at the event above and choosing $p > 8 C$ gives

$$
\mathbb { E } _ { j } \left[ \exp \left( 4 \operatorname* { m a x } _ { k } | \left. \theta _ { k } , \Delta \mathbf { z } \right. | \right) \right] \leq e ^ { 4 } + \mathcal { O } \left( n ^ { 8 C - p } ( \log n ) ^ { 2 p } \right) \leq C .
$$

The sigmoid satisfies

$$
\begin{array} { r } { \sigma ( u + v ) \leq e ^ { | v | } \sigma ( u ) , \qquad 1 - \sigma ( u + v ) \leq e ^ { | v | } ( 1 - \sigma ( u ) ) . } \end{array}
$$

Applying these inequalities to each gate yields

$$
\left| 1 - w _ { i } ( \mathbf { X } ) \right| + \sum _ { k \neq i } { w _ { k } ( \mathbf { X } ) } \leq \exp \left( \operatorname* { m a x } _ { k } | \left. \theta _ { k } , \Delta \mathbf { z } \right. | \right) \| { \bar { w } } _ { \cdot j } - e _ { i } \| _ { 1 } .
$$

The difference from the selected expert is

$$
\widehat { \mu } - g _ { i } = \mu \left( \sum _ { k } w _ { k } - 1 \right) + \sum _ { k } ( w _ { k } - \mathbf { 1 } _ { \{ k = i \} } ) ( g _ { k } - \mu ) .
$$

The gates and centered experts depend only on the centered observations and are independent of $\mu .$ Since $\mathbb { E } [ \mu ] = 0$ , averaging over $\mu$ eliminates the cross term and gives

$$
\begin{array} { r l } & { \mathbb { E } _ { j } \left[ ( \widehat \mu - g _ { i } ) ^ { 2 } \right] = \sigma _ { \mu } ^ { 2 } \mathbb { E } _ { j } \left[ \left( \displaystyle \sum _ { k } w _ { k } - 1 \right) ^ { 2 } \right] + \mathbb { E } _ { j } \left[ \left( \displaystyle \sum _ { k } ( w _ { k } - \mathbf { 1 } _ { \{ k = i \} } ) \big ( g _ { k } - \mu \big ) \right) ^ { 2 } \right] } \\ & { \qquad \leq \| \bar { w } _ { \cdot j } - e _ { i } \| _ { 1 } ^ { 2 } \mathbb { E } _ { j } \left[ \exp \left( 2 \operatorname* { m a x } _ { k } \big | \langle \theta _ { k } , \Delta \mathbf { z } \rangle \big | \right) \left( \sigma _ { \mu } ^ { 2 } + \operatorname* { m a x } _ { \ell \in [ n ] } \big | X _ { \ell } - \mu \big | ^ { 2 } \right) \right] } \\ & { \qquad \leq C \| \bar { w } _ { \cdot j } - e _ { i } \| _ { 1 } ^ { 2 } \left( \sigma _ { \mu } ^ { 2 } + \sqrt { \mathbb { E } _ { j } \left[ \operatorname* { m a x } _ { \ell \in [ n ] } \big | X _ { \ell } - \mu \big | ^ { 4 } \right] } \right) . } \end{array}
$$

The first inequality uses $| g _ { k } - \mu | \leq \operatorname* { m a x } _ { \ell } | X _ { \ell } - \mu |$ , which holds for every $\lambda _ { k }$ , and the second uses Cauchy–Schwarz and the exponential moment bound above. The triangle inequality in $L ^ { 2 }$ proves the claim. □

The bound on $\mathbb { E } _ { j } \left\lceil ( \widehat { \mu } - g _ { i } ) ^ { 2 } \right\rceil$ is quadratic in $\left\| \bar { w } _ { \cdot j } - e _ { i } \right\| .$ <sub>1</sub> and uniform over $\lambda .$ It therefore applies throughout training of λ after θ is fixed.

## D GATING-LAYER DYNAMICS

## D.1 DERIVATION OF THE POPULATION GRADIENT FLOW

Remark D.1 (Layer normalization). Since the centered population feature norms are bounded below by Lemma F.5, layer normalization has bounded derivatives on a fixed neighborhood of each such vector. Composing the moment ratios with this map therefore preserves the orders in n of the Taylor expansions and their remainder bounds in the following gradient and loss calculations, although their coefficients change. Outside this neighborhood, the moment bounds for the empirical averages and $\| \mathbf { z } \| _ { 2 } = 1$ control the contribution.

Lemma D.2 (Population gradient expansion). Let $\theta _ { q }$ be the gating parameter vector $f o r$ expert $q \in \{ 1 , 2 \}$ , and let the loss function $\mathcal { L } ( \boldsymbol { \theta } )$ be defined as:

$$
\mathcal { L } ( \theta ) = \frac { 1 } { 2 } \mathbb { E } \left[ \left( \sum _ { i = 1 } ^ { 2 } w _ { i } ( \mathbf { X } ) g _ { i } - \mu \right) ^ { 2 } \right]
$$

Then

$$
\begin{array} { r l } { \nabla _ { \Phi _ { 6 } } \mathcal { L } ( \xi , \eta ) - \displaystyle \frac { 1 } { 2 } \sum _ { j = 1 } ^ { n } \{ \sigma _ { \alpha } ^ { 2 } ( \displaystyle \sum _ { k = 1 } ^ { 2 } \widehat { \sigma } _ { \alpha } - 1 ) \widehat { \sigma } _ { \alpha } ^ { \prime } \widehat { \sigma } _ { \alpha } + \frac { \widehat { \sigma } _ { \alpha \beta } ^ { \prime } } { n } \displaystyle \sum _ { i = 1 } ^ { 2 } \widehat { \sigma } _ { \alpha } \widehat { \sigma } _ { \alpha } ^ { \prime } \widehat { \sigma } _ { \alpha } ^ { \prime }  } & { } \\ {   + \sigma _ { \mu } ^ { 2 } \widehat { \sigma } _ { \alpha } ^ { \prime } \widehat { \sigma } _ { \alpha } ^ { \prime } \widehat { \sigma } _ { \alpha } ^ { \prime } \} \displaystyle \sum _ { i = 1 } ^ { 2 } \widehat { \sigma } _ { \alpha } ^ { \prime } \widehat { \sigma } _ { \alpha } ^ { \prime } \widehat { \sigma } _ { \alpha } ^ { \prime } \widehat { \sigma } _ { \alpha } ^ { \prime } \widehat { \sigma } _ { \alpha } ^ { \prime } \} \widehat { \sigma } _ { \alpha } ^ { \prime } } & { } \\ { + \sigma _ { \nu } ^ { 2 } ( \displaystyle \sum _ { i = 1 } ^ { 2 } \widehat { \sigma } _ { \alpha } \widehat { \sigma } _ { \alpha } - 1 ) [ \widehat { \sigma } _ { \alpha } ^ { \prime \prime } \widehat { \sigma } _ { \alpha } ^ { \prime } \widehat { \sigma } _ { \alpha } ^ { \prime } \widehat { \sigma } _ { \alpha } \widehat { \sigma } _ { \alpha } ^ { \prime } ] \sigma _ { \alpha } + \widehat { \sigma } _ { \alpha } ^ { \prime \prime } \widehat { \sigma } _ { \alpha } \widehat { \sigma } _ { \alpha } ^ { \prime } \widehat { \sigma } _ { \alpha } ^ { \prime } \widehat { \sigma } _ { \alpha } ^ { \prime } | } & { } \\  + \displaystyle \frac { \sigma _ { \alpha } ^ { 2 } }  n  \end{array}
$$

where $\bar { w } _ { i j } = \sigma ( \langle { \theta _ { i } , \bar { \bf z } _ { j } } \rangle ) , \bar { \bf z } _ { j }$ is obtained by applying layer normalization to the population moment ratios, $\Delta w _ { i } = w _ { i } ( \mathbf { z } ) - \bar { w } _ { i j }$ , and $\Delta w _ { q } ^ { \prime } = w _ { q } ^ { \prime } ( \mathbf { z } ) - \bar { w } _ { q j } ^ { \prime } .$ . Primes on gates denote derivatives of the

sigmoid with respect to its scalar logit, so $w _ { q } ^ { \prime } = w _ { q } ( 1 - w _ { q } )$ , with the same conventionfor population gates and higher derivatives. The matrix $\pmb { \Sigma } ^ { ( j ) }$ is defined in Definition $B . I ,$ , and $\Sigma _ { Z } ^ { ( j ) }$ is the asymptotic covariance matrix of $\sqrt { n } ( \mathbf { z } - \bar { \mathbf { z } } _ { j } )$ under family j.

Proof.

$$
\begin{array} { r l } { \mathbf { r } _ { \perp } ( x ) } &  = ( \begin{array} { l } { 1 } \\  2 ( \frac { 1 } { \sqrt { 3 } } ) ( \frac { 1 } { 2 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 2 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 } { 3 } ) ( \frac { 1 }   \end{array} \end{array}
$$

Taylor expansion of the gates and Lemma F.6 give

$$
\begin{array} { r c l } { { \nabla _ { \Theta } \bar { c } ( \xi ) = \displaystyle { \frac { 1 } { 2 } \sum _ { j } ^ { \prime } \Bigg \{ \sigma _ { j } ^ { 2 } ( \sum _ { k = 1 } ^ { 2 } \bar { w } _ { \xi } - 1 ) \bar { w } _ { \xi } ^ { \prime \prime } \bar { c } _ { j k } + \frac { w _ { \xi } ^ { \prime \prime } } { \kappa } \frac { \sigma _ { j } ^ { 2 } } { \omega _ { k } } \sum _ { i = 1 } ^ { \nu } \bar { c } _ { j k } ^ { \prime } \bar { c } _ { j k } ^ { \prime } } } } \\ { { } } & { { } } & { { } } \\ { { } } & { { + \sigma _ { j } ^ { 2 } \nu _ { \sigma _ { j } } ^ { \prime } \sigma _ { j } ^ { 2 } \nu _ { k } \sum _ { m ^ { \prime } \neq j } ^ { \nu } \bar { c } _ { j k } ^ { \prime } \bar { d } _ { 1 } ^ { \prime } \mathrm { H } _ { i } ^ { \prime } \mathrm { L i } \Bigg \{ \Delta \Bigg / { } } }  \\ { { } } & { { } } & { { } } \\ { { } } & { { + \sigma _ { i } ^ { 2 } ( \sum _ { m ^ { \prime } = 1 } ^ { 2 } \bar { w } _ { \xi } - 1 ) [ \frac { 1 } { w _ { \sigma } ^ { \prime \prime } } 2 \sigma _ { j } ^ { 2 } \nu _ { \sigma _ { j } } ^ { 2 } \overline { { \kappa } } [ \Delta \Omega ] + \bar { w } _ { \sigma } ^ { \prime \prime } \mathrm { H } _ { i } ^ { \prime } ( \Delta \Omega ) ] } } \\ { { } } & { { } } & { { } } \\ { { } } &   + \frac { C _ { j } ^ { 2 } } { \kappa } \sum _ { i = 1 } ^ { 2 } \bar { w } _ { \xi } ^ { \prime } [ \bar { w } _ { \sigma } ^ { \prime \prime } ( \bar { \theta } _ { 0 } ^ { \prime \prime } \Sigma _ { i } ^ { 2 } \ \end{array}
$$

Lemma D.3 (Population loss expansion). Let $\theta _ { q }$ be the gating parameter vectorfor expert $q \in \{ 1 , 2 \}$ , and let the loss function $\mathcal { L } ( \boldsymbol { \theta } )$ be defined as:

$$
\mathcal { L } ( \theta ) = \frac { 1 } { 2 } \mathbb { E } \left[ \left( \sum _ { i = 1 } ^ { 2 } w _ { i } ( \mathbf { X } ) g _ { i } - \mu \right) ^ { 2 } \right]
$$

Then the expected loss can be expanded as:

$$
\begin{array} { r l } { \mathcal { L } ( \theta ) = \cfrac { 1 } { 4 } \displaystyle \sum _ { j } \Biggl \{ \sigma _ { \mu } ^ { 2 } \left( \sum _ { i = 1 } ^ { 2 } \bar { w } _ { i j } - 1 \right) ^ { 2 } + \frac { 1 } { n } \sum _ { i = 1 } ^ { 2 } \sum _ { q = 1 } ^ { 2 } \bar { w } _ { i j } \bar { w } _ { q j } \Sigma _ { i q } ^ { ( j ) } } & { } \\ { + \frac { \sigma _ { \mu } ^ { 2 } } { n } \displaystyle \sum _ { i = 1 } ^ { 2 } \sum _ { q = 1 } ^ { 2 } \bar { w } _ { i j } ^ { \prime } \bar { w } _ { q j } ^ { \prime } \left( \theta _ { i } ^ { \top } \Sigma _ { Z } ^ { ( j ) } \theta _ { q } \right) } & { } \\ { + 2 \sigma _ { \mu } ^ { 2 } \left( \displaystyle \sum _ { i = 1 } ^ { 2 } \bar { w } _ { i j } - 1 \right) \displaystyle \sum _ { i = 1 } ^ { 2 } \bar { w } _ { i j } ^ { \prime } \theta _ { i } ^ { \top } \mathbb { E } _ { 2 } [ \Delta \mathbf { z } ] } & { } \\ { + \frac { \sigma _ { \mu } ^ { 2 } } { n } \left( \displaystyle \sum _ { i = 1 } ^ { 2 } \bar { w } _ { i j } - 1 \right) \displaystyle \sum _ { i = 1 } ^ { 2 } \bar { w } _ { i j } ^ { \prime } \left( \theta _ { i } ^ { \top } \Sigma _ { Z } ^ { ( j ) } \theta _ { i } \right) \Biggr \} + \mathcal { O } ( n ^ { - 3 / 2 } ) } \end{array}
$$

Proof. We can directly compute

$$
\begin{array} { r l }  \sum _ { j = 1 } ^ { N } \Bigg ( \frac { 1 } { N } \Bigg ) ^ { j } \frac { 1 } { N } \Bigg [ \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \int \end{array}
$$

## D.1.1 FLOW REPARAMETERIZATION

Use $S _ { q j }$ from Lemma B.7. Recall that we are studying the gradient flow

$$
\begin{array} { r l } { \mathbb { E } _ { k } - \mathbb { V } _ { k } \Big ( \xi \Big ( k \eta \Big ( \mathcal { T } _ { s } \Big ) } & { = \xi \Big ( k \eta \Big ( \mathcal { T } _ { s } \Big ) + \xi \Big ( \xi \Big ) } \\ & { - \frac { 1 } { 2 } \frac { 1 } { \eta } \Big \{ S _ { \xi } \Big ( k \eta ( \mathcal { T } _ { s } ) \Big \} \frac { \xi } { \eta } \Big ) \Big \{ S _ { \xi } \Big ( k \eta ( \mathcal { T } _ { s } ) \Big \} \frac { \xi } { \eta } , } \\ &  + \frac { 1 } { \eta } \Big \{ S _ { \xi } \Big ( k \eta ( \mathcal { T } _ { s } ) \Big \} \frac { \xi } { \eta } \Big \{ S _ { \xi } \Big ( k \eta \Big ) \} \frac { \xi } { \eta } \Big \{ Z _ { s } \Big ( k \eta \Big \} \} \\ & { + \frac { 1 } { \eta } \Big \{ S _ { \xi } \Big ( k \eta ( \mathcal { T } _ { s } ) \Big \} - \frac { \xi } { \eta } \Big \{ ( k \eta ^ { 2 } ) \Big \} \{ \frac { \xi } { \eta } \Big ( k \eta \Big ) + \frac { 1 } { \eta } \Big \{ S _ { \xi } \Big ( k \eta ( \mathcal { T } _ { s } ) \Big \} + \frac { \xi } { \eta } \Big \{ S _ { \xi } \Big ( k \eta ( \mathcal { T } _ { s } ) \Big \} \Big \} \Big . } \\ & {  + \frac { 1 } { \eta } \Big \{ S _ { \xi } \Big ( k \eta ( \mathcal { T } _ { s } ) \Big \} \Big \{ \frac { \xi } { \eta } \Big ( k \eta ( \mathcal { T } _ { s } ) \Big \} \{ \frac { \xi } { \eta } \Big ( k \eta \Big ) + \frac { 1 } { \eta } \Big \{ S _ { \xi } \Big ( k \eta ( \mathcal { T } _ { s } ) \Big \} \{ \frac { \xi } { \eta } \Big \} ( k \eta \Big ) } \\ &  + \frac { 1 } { \eta } \Big \{ S _ { \xi } \Big ( k \eta ( \mathcal { T } _  \end{array}
$$

We now re-parameterize the flow in terms of $\bar { w } = \left( \bar { w } _ { i j } \right) _ { i \in [ 2 ] , j \in \{ \mathrm { G } , \mathrm { A } \} }$ , using $\bar { w } _ { q k } = \sigma ( \langle \theta _ { q } , \bar { \mathbf { z } } _ { k } \rangle )$ ) and the chain rule.

$$
\begin{array} { r l } { \varepsilon _ { 3 } ^ { \prime \prime } } & { = \frac { \varepsilon _ { 3 } ^ { \prime \prime } } { \varepsilon _ { 3 } } } \\ & { - \varepsilon _ { 3 } ^ { \prime \prime } \left( \varepsilon _ { 3 } ^ { \prime \prime } + \varepsilon _ { 3 } ^ { \prime \prime } \right) } \\ & { - \varepsilon _ { 3 } ^ { \prime \prime } \left( \varepsilon _ { 3 } ^ { \prime \prime } + \varepsilon _ { 3 } ^ { \prime \prime } \right) } \\ & { + \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } } \\ & { - \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } } \\ & { + \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } } \\ & { - \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } } \\ & { - \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } \varepsilon _ { 3 } ^ { \prime \prime } } \\ &  + \varepsilon _ { 3 } ^  \prime  \end{array}
$$

## D.2 CONCENTRATION BOUNDS

## D.2.1 UNIFORM CONCENTRATION BOUND FOR THE EMPIRICAL GRADIENT

Lemma D.4 (Concentration bound for the gradient of θ). Let $B _ { d } ( r )$ denote the Euclidean ball of radius r in $\mathbb { R } ^ { d }$ . Fix $j \in \{ \mathrm { G } , \mathrm { A } \}$ and sample the m tasks independently conditional on $\tau _ { b } = j ,$ , writing $\bar { \mathbf { z } } = \bar { \mathbf { z } } _ { j }$ and $\bar { w } _ { k } = \bar { w } _ { k j } . \ : \dot { F i x } \sigma _ { \mu } ^ { 2 }$ and λ independently $o f n$ , with either data law from Definition 2.1. Let $0 < \beta < 1 / 2 , 0 < \delta < 1 , r = \mathcal { O } ( \log n )$ , and suppose m, $\delta ^ { - 1 } \in \mathrm { p o l y } ( n )$ . For sufficiently large n, with probability at least $1 - \delta ,$ , simultaneouslyfor all $\theta _ { 1 } , \theta _ { 2 } \in B _ { d } ( r )$ and all $q \in \{ 1 , 2 \}$ and $s \in [ d ]$

$$
\begin{array} { r l } & { \left| \displaystyle \frac { 1 } { m } \sum _ { b = 1 } ^ { m } \left[ \left( \sum _ { k = 1 } ^ { 2 } w _ { k } ( \mathbf { X } _ { b } ) g _ { k } ( \mathbf { X } _ { b } ) - \mu _ { b } \right) g _ { q } ( \mathbf { X } _ { b } ) w _ { q } ^ { \prime } ( \mathbf { X } _ { b } ) z _ { s } ( \mathbf { X } _ { b } ) \right] \right| } \\ & { \quad - \mathbb { E } _ { j } \left[ \left( \displaystyle \sum _ { k = 1 } ^ { 2 } w _ { k } ( \mathbf { X } ) g _ { k } - \mu \right) g _ { q } w _ { q } ^ { \prime } ( \mathbf { X } ) z _ { s } \right] \Bigg | } \\ & { \quad \le \left( \left| \displaystyle \sum _ { k = 1 } ^ { 2 } \bar { w } _ { k } - 1 \right| + n ^ { - 1 / 2 + \beta } \right) \bar { w } _ { q } ( 1 - \bar { w } _ { q } ) \widetilde { \mathcal { O } } ( m ^ { - 1 / 2 } ) . } \end{array}
$$

Proof. Set $M = n ^ { - 1 / 2 + \beta }$ and choose $R , C = \mathcal O ( \sqrt { \log n } ) , C \geq 1$ , with sufficiently large constants. Apply Lemma F.17 with $k = 1$ at the fixed $\pm \lambda _ { k }$ and $\omega _ { s } ,$ using deviations equal to a sufficiently small constant multiple of M. Since $n M ^ { 2 } / ( R ^ { 2 } e ^ { 2 R | \lambda | } ) = n ^ { 2 \beta - o ( 1 ) }$ , symmetry of the centered observations,

the fixed d and $\omega ,$ , and the Gaussian tail of $\mu$ give

$$
\mathcal { E } = \left\{ \operatorname* { m a x } _ { k } | g _ { k } - \mu | \leq M , \ | \mu | \leq C , \ \| \mathbf { z } - \bar { \mathbf { z } } \| _ { 2 } \leq M \right\} , \qquad \mathbb { P } _ { j } ( \mathcal { E } ^ { c } ) = \mathcal { O } ( n ^ { - A } )
$$

for any prescribed fixed $A > 0$ . Apply the deterministic covers in Lemma D.5 to the integrands multiplied by $\mathbf { 1 } _ { \mathcal { E } }$ . The discarded normalized integrand is bounded by

$$
\frac { 3 e ^ { 2 r } } { M } \left( \left| \mu \right| + \operatorname* { m a x } _ { \ell \in \left[ n \right] } \left| X _ { \ell } - \mu \right| \right) ^ { 2 } ,
$$

since $w _ { q } ^ { \prime } / \bar { w } _ { q } ^ { \prime } \ \leq \ e ^ { 2 r }$ and $\Gamma _ { 1 } \geq M$ . Its $L ^ { 2 }$ norm is polynomial in $n ,$ so Cauchy–Schwarz and sufficiently large $A$ make the discarded expectation $o ( m ^ { - 1 / 2 } )$ and the probability of discarding any training task at most $\delta / 2$ . Use failure probability $\delta / 2$ for the fixed number of coordinates in the helper bound, where $r M = { \overset { \cdot } { o } } ( 1 )$ ) and its remaining factors are logarithmic. □

Lemma D.5 (Concentration bound for the normalized gradient). $F i x \ j \in \{ \mathrm { G } , \mathrm { A } \}$ , a gate index $q \in [ 2 ]$ , a coordinate $s \in [ d ] ,$ , a radius $r > 0$ and afailure probability $\delta \in ( 0 , 1 )$ , and sample m tasks independently conditional on $\tau _ { b } = j ;$ , writing $\bar { \mathbf { z } } = \bar { \mathbf { z } } _ { j }$ and $\bar { w } _ { k } = \bar { w } _ { k j }$ . Suppose $M > 0 , C \geq 1$ , and, almost surely for a single task and every $k ,$

$$
| g _ { k } ( \mathbf { X } ) - { \boldsymbol { \mu } } | \leq M , \qquad | { \boldsymbol { \mu } } | \leq C , \qquad | | \mathbf { z } ( \mathbf { X } ) - { \bar { \mathbf { z } } } | | _ { 2 } \leq M , \qquad | | { \bar { \mathbf { z } } } | | _ { 2 } \leq C .
$$

Let

$$
w _ { k } ( { \mathbf { X } } ) = \sigma ( \langle \theta _ { k } , \mathbf { z } ( { \mathbf { X } } ) \rangle ) , \qquad \bar { w } _ { k } = \sigma ( \langle \theta _ { k } , \bar { \mathbf { z } } \rangle ) ,
$$

and define

$$
\Gamma _ { 1 } = \left| \sum _ { k = 1 } ^ { 2 } { \bar { w } } _ { k } - 1 \right| + M , \qquad \Gamma _ { 2 } = { \bar { w } } _ { q } ( 1 - { \bar { w } } _ { q } ) ,
$$

$$
\Psi _ { \theta , q , s } ( \mathbf { X } , \mu ) = \left( \sum _ { k = 1 } ^ { 2 } w _ { k } ( \mathbf { X } ) g _ { k } ( \mathbf { X } ) - \mu \right) g _ { q } ( \mathbf { X } ) w _ { q } ^ { \prime } ( \mathbf { X } ) z _ { s } ( \mathbf { X } ) .
$$

Then, with probability at least $1 - \delta ,$

$$
\begin{array} { r l r } {  { \operatorname* { s u p } _ { \theta _ { 1 } , \theta _ { 2 } \in B _ { d } ( r ) } \frac { 1 } { \Gamma _ { 1 } \Gamma _ { 2 } } | \frac { 1 } { m } \sum _ { b = 1 } ^ { m } \Psi _ { \theta , q , s } ( \mathbf { X } _ { b } , \mu _ { b } ) - \mathbb { E } _ { j } [ \Psi _ { \theta , q , s } ( \mathbf { X } , \mu ) ] | } } \\ & { } & { \lesssim 2 ( C + M + C r ) e ^ { r M } ( M + C ) ^ { 2 } } \\ & { } & { \times [ \frac { 2 d \log \{ 2 + 2 m r ( C + M ) ( 1 + M ^ { - 1 } ) \} + \log ( 2 / \delta ) } { m } ] ^ { 1 / 2 } . } \end{array}
$$

Proof. Here $L _ { \infty }$ denotes the supremum over the full support of a single task. All covers below are deterministic. Define

$$
\mathcal { F } _ { a } = \left\{ f _ { \theta } : ( \mathbf { z } , \mathbf { g } , \mu ) \mapsto \frac { 1 } { \Gamma _ { 1 } } \left( \sum _ { k = 1 } ^ { 2 } \sigma ( \boldsymbol { \theta } _ { k } ^ { \top } \mathbf { z } ) g _ { k } - \mu \right) : \theta _ { k } \in B _ { d } ( r ) \right\} .
$$

For $f _ { \theta } \in \mathcal { F } _ { a }$ we can write

$$
f _ { \theta } = \frac { 1 } { \Gamma _ { 1 } } \left( \sum _ { k = 1 } ^ { 2 } w _ { k } ( g _ { k } - \mu ) + \mu \sum _ { k = 1 } ^ { 2 } ( w _ { k } - \bar { w } _ { k } ) + \mu \left( \sum _ { k = 1 } ^ { 2 } \bar { w } _ { k } - 1 \right) \right) .
$$

Since $| w _ { k } - \bar { w } _ { k } | \leq r M / 4$ and $C \geq 1$ , this gives

$$
\| f _ { \theta } \| _ { \infty } \leq 2 ( C + M + C r ) \equiv B _ { a } .
$$

The normalizing factor also depends on $\theta ,$ with

$$
| \Gamma _ { 1 } ( \theta ) - \Gamma _ { 1 } ( \theta ^ { \prime } ) | \leq \frac { C } { 2 } \operatorname* { m a x } _ { k } \| \theta _ { k } - \theta _ { k } ^ { \prime } \| _ { 2 } .
$$

Using $\Gamma _ { 1 } \geq M$ and the sigmoid’s 1/4-Lipschitz bound, we obtain

$$
\| f _ { \theta } - f _ { \theta ^ { \prime } } \| _ { \infty } \leq \frac { ( C + M ) ^ { 2 } + B _ { a } C } { 2 M } \operatorname* { m a x } _ { k } \| \theta _ { k } - \theta _ { k } ^ { \prime } \| _ { 2 } \leq \frac { 2 B _ { a } ( C + M ) } { M } \operatorname* { m a x } _ { k } \| \theta _ { k } - \theta _ { k } ^ { \prime } \| _ { 2 } .
$$

A Euclidean cover of each $B _ { d } ( r )$ therefore gives

$$
\log \mathcal { N } ( u , \mathcal { F } _ { a } , L _ { \infty } ) \lesssim 2 d \log \left( 2 + \frac { 2 r B _ { a } ( C + M ) } { M u } \right) .\tag{16}
$$

Define

$$
\mathcal { F } _ { b } = \left\{ ( \mathbf { z } , \mathbf { g } , \mu ) \mapsto \frac { w _ { q } ^ { \prime } g _ { q } z _ { s } } { \Gamma _ { 2 } } : \theta _ { q } \in B _ { d } ( r ) \right\} .
$$

Since $\mid \frac { d } { d u }$ log $\sigma ^ { \prime } ( u ) | \leq 1$

$$
\frac { w _ { q } ^ { \prime } } { \Gamma _ { 2 } } \leq e ^ { r M } , \qquad \left\| \nabla _ { \theta _ { q } } \log \frac { w _ { q } ^ { \prime } } { \Gamma _ { 2 } } \right\| _ { 2 } \leq M + 2 C .
$$

Thus the functions in $\mathcal { F } _ { b }$ are bounded in absolute value by $e ^ { r M } ( M + C ) ^ { 2 }$ and have Lipschitz constant at most $e ^ { r M } ( M + C ) ^ { 2 } ( M + 2 C ) \mathrm { i n } \theta _ { q } ,$ including the dependence of $\Gamma _ { 2 } .$ Let $\mathcal { F }$ consist of the products $f _ { a } f _ { b }$ with the same parameters in both factors. These products are bounded in absolute value by $B _ { a } e ^ { r M } ( M + C ) ^ { 2 }$ . A deterministic parameter cover with mesh $[ 4 m ( C + M ) ( 1 + M ^ { - 1 } ) ] ^ { - 1 }$ gives

$$
\log \mathcal { N } \left( \frac { B _ { a } e ^ { r M } ( M + C ) ^ { 2 } } { m } , \mathcal { F } , L _ { \infty } \right) \lesssim 2 d \log \left( 2 + 2 m r ( C + M ) ( 1 + M ^ { - 1 } ) \right) .
$$

Hoeffding’s inequality and a union bound over this cover, followed by approximation in $L _ { \infty } , \mathrm { g i v e } .$ with probability at least $1 - \delta$

$$
\begin{array} { r l r } {  { \operatorname* { s u p } _ { f \in \mathcal { F } }  \frac { 1 } { m } \sum _ { b = 1 } ^ { m } f ( { \mathbf { X } } _ { b } ) - \mathbb { E } _ { j } [ f ]  \lesssim B _ { a } e ^ { r M } ( M + C ) ^ { 2 } } } \\ & { } & { \times [ \frac { 2 d \log ( 2 + 2 m r ( C + M ) ( 1 + M ^ { - 1 } ) ) + \log ( 2 / \delta ) } { m } ] ^ { 1 / 2 } . } \end{array}
$$

The approximation contributes at most $2 B _ { a } e ^ { r M } ( M + C ) ^ { 2 } / m$ , which is absorbed in this bound.

## D.2.2 UNIFORM CONCENTRATION FOR THE LOSS

Lemma D.6 (Local concentration for the loss). Under the assumptions ofLemma $D . 4 ,$ with probability at least $1 - \delta ,$ , uniformly over $\theta _ { 1 } , \theta _ { 2 } \in B _ { d } ( r )$ satisfying $\begin{array} { r } { \lvert \sum _ { k } \bar { w } _ { k } - 1 \rvert = \mathcal { O } ( n ^ { - 1 } ) } \end{array}$ with afixed implied constant,

$$
| \hat { \mathcal { L } } _ { m } ^ { ( j ) } ( \theta ) - \mathcal { L } ^ { ( j ) } ( \theta ) | = \widetilde O \left( \frac { n ^ { - 1 + \beta } } { \sqrt { m } } + \frac { n ^ { - 1 + 2 \beta } } { m } \right) .
$$

Proof. Use $\mathcal { E } , M$ , and C from the proof of Lemma D.4, and apply the covers in Lemma D.7 to $\mathcal { F } _ { s q } \mathbf { 1 } _ { \mathcal { E } }$ . The normalized squared residual has envelope

$$
{ \frac { 9 } { M ^ { 2 } } } \left( \left| \mu \right| + \operatorname* { m a x } _ { \ell \in \left[ n \right] } \left| X _ { \ell } - \mu \right| \right) ^ { 2 } .
$$

Cauchy–Schwarz makes the discarded expectation $o ( m ^ { - 1 } )$ , with the probability of discarding a training task absorbed in $\delta .$ The fixed-parameter moment-ratio argument in the proof of Lemma $3 . 5$ applied also to the paired experts, and independence of $\mu$ from the centered observations give

$$
\mathcal { L } ^ { ( j ) } = \mathcal { O } \left( \frac { ( 1 + r ) ^ { 2 } } { n } \right) = \widetilde { \mathcal { O } } ( n ^ { - 1 } ) , \qquad \Gamma _ { 1 } ^ { 2 } U = \widetilde { \mathcal { O } } ( n ^ { - 1 + 2 \beta } ) .
$$

Substitute into Lemma D.7. The discarded expectation is absorbed by its linear term.

Lemma D.7 (Uniform concentration for the loss). Fix $j \in \{ \mathrm { G } , \mathrm { A } \}$ and sample the m tasks independently conditional on $\tau _ { b } = j ,$ , writing $\bar { \mathbf { z } } = \bar { \mathbf { z } } _ { j }$ and $\bar { w } _ { k } = \bar { w } _ { k j }$ . Suppose $M > 0 , C \geq 1$ , and the following bounds hold almost surelyfor a single task and every k:

$$
\begin{array} { r } { | g _ { k } ( \mathbf { X } ) - \mu | \leq M } \\ { | \mu | \leq C } \end{array}
$$

$$
\| \mathbf { z } ( \mathbf { X } ) - \bar { \mathbf { z } } \| _ { 2 } \leq M
$$

$$
\| \bar { \mathbf { z } } \| _ { 2 } \leq C
$$

$$
w _ { k } ( { \mathbf { X } } ) = \sigma ( \langle \theta _ { k } , \mathbf { z } ( { \mathbf { X } } ) \rangle )
$$

$$
\bar { w } _ { k } = \sigma ( \langle \theta _ { k } , \bar { \bf z } \rangle )
$$

$$
\Gamma _ { 1 } = \left| \sum _ { k = 1 } ^ { 2 } { \bar { w } } _ { k } - 1 \right| + M
$$

$$
U = 4 ( C + M + C r ) ^ { 2 }
$$

Let

$$
\hat { \mathcal { L } } _ { m } ^ { ( j ) } ( \theta ) = \frac { 1 } { 2 m } \sum _ { b = 1 } ^ { m } \left( \sum _ { k = 1 } ^ { 2 } w _ { k } ( \mathbf { X } _ { b } ) g _ { k } ( \mathbf { X } _ { b } ) - \mu _ { b } \right) ^ { 2 } , \quad \mathcal { L } ^ { ( j ) } ( \theta ) = \frac { 1 } { 2 } \mathbb { E } _ { j } \left[ \left( \sum _ { k = 1 } ^ { 2 } w _ { k } ( \mathbf { X } ) g _ { k } ( \mathbf { X } ) - \mu \right) ^ { 2 } \right]
$$

Then, with probability at least $1 - \delta ,$ the following holds for all $\theta _ { 1 } , \theta _ { 2 } \in B _ { d } ( r )$

$$
\begin{array} { r l } & { \left| \hat { \mathcal { L } } _ { m } ^ { ( j ) } ( \theta ) - \mathcal { L } ^ { ( j ) } ( \theta ) \right| \lesssim \sqrt { \mathcal { L } ^ { ( j ) } ( \theta ) \Gamma _ { 1 } ^ { 2 } U } \frac { 2 d \log { ( 2 + 2 m r ( C + M ) ( 1 + M ^ { - 1 } ) ) } + \log ( 2 / \delta ) } { m } } \\ & { \qquad + \Gamma _ { 1 } ^ { 2 } U \frac { 2 d \log { \left( 2 + 2 m r ( C + M ) ( 1 + M ^ { - 1 } ) \right) } + \log ( 2 / \delta ) } { m } . } \end{array}
$$

Proof. Use ${ \mathcal { F } } _ { a }$ from the proof of Lemma D.5, with $B _ { a } = 2 ( C + M + C r )$ , and let

$$
{ \mathcal { F } } _ { s q } = \{ f _ { \theta } ^ { 2 } : f _ { \theta } \in { \mathcal { F } } _ { a } \} .
$$

For every $u \in \mathcal { F } _ { s q }$

$$
0 \leq u \leq B _ { a } ^ { 2 } = U , \qquad \mathrm { V a r } ( u ) \leq \mathbb { E } _ { j } [ u ^ { 2 } ] \leq U \mathbb { E } _ { j } [ u ] .
$$

Since squaring is $2 B _ { a }$ -Lipschitz on $[ - B _ { a } , B _ { a } ]$ , (16) gives

$$
\log \mathcal { N } ( u , \mathcal { F } _ { s q } , L _ { \infty } ) \lesssim 2 d \log \left( 2 + \frac { 2 r U ( C + M ) } { M u } \right) .
$$

Choose $u _ { 0 } = U / m$ and a deterministic u -cover $\mathcal { C } _ { u _ { 0 } } \subset \mathcal { F } _ { s q }$ in $L _ { \infty }$ over the full support. Then

$$
\log | \mathcal { C } _ { u _ { 0 } } | \lesssim 2 d \log \big ( 2 + 2 m r ( C + M ) ( 1 + M ^ { - 1 } ) \big ) .
$$

Bernstein’s inequality and a union bound give, simultaneously for every $v \in \mathcal { C } _ { u _ { 0 } }$ , with probability at least $1 - \delta$

$$
\left| \frac { 1 } { m } \sum _ { b = 1 } ^ { m } v ( \mathbf { X } _ { b } ) - \mathbb { E } _ { j } [ v ] \right| \leq \sqrt { \frac { 2 U \mathbb { E } _ { j } [ v ] \log ( 2 | \mathcal { C } _ { u _ { 0 } } | / \delta ) } { m } } + \frac { 2 U \log ( 2 | \mathcal { C } _ { u _ { 0 } } | / \delta ) } { 3 m } .
$$

For any $u \in \mathcal { F } _ { s q }$ choose v in this cover with $\| u - v \| _ { \infty } \leq U / m$ . This controls both the empirical average and the expectation, so

$$
\left| \frac { 1 } { m } \sum _ { b = 1 } ^ { m } u ( { \mathbf { X } } _ { b } ) - \mathbb { E } _ { j } [ u ] \right| \leq \left| \frac { 1 } { m } \sum _ { b = 1 } ^ { m } v ( { \mathbf { X } } _ { b } ) - \mathbb { E } _ { j } [ v ] \right| + \frac { 2 U } { m } , \qquad \mathbb { E } _ { j } [ v ] \leq \mathbb { E } _ { j } [ u ] + \frac { U } { m } .
$$

The resulting approximation terms are absorbed by the linear term in Bernstein’s bound. Finally, for $u = f _ { \theta } ^ { 2 }$

$$
\hat { \mathcal { L } } _ { m } ^ { ( j ) } ( \theta ) = \frac { \Gamma _ { 1 } ^ { 2 } } { 2 m } \sum _ { b = 1 } ^ { m } u ( \mathbf { X } _ { b } ) , \qquad \mathcal { L } ^ { ( j ) } ( \theta ) = \frac { \Gamma _ { 1 } ^ { 2 } } { 2 } \mathbb { E } _ { j } [ u ] .
$$

Multiplying by $\Gamma _ { 1 } ^ { 2 } / 2$ proves the claimed bound simultaneously for all parameters.

## D.3 ANALYSIS OF INDIVIDUAL STATIONARY POINTS: COMPLETE FLOW

## D.3.1 PROOF OF LEMMA B.7

Proof. Since $Z \succ 0$

$$
\begin{array} { r l } { \displaystyle \frac { d } { d t } { \mathcal L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ) = - \frac { 1 } { 4 } \sum _ { q = 1 } ^ { 2 } \left\| \sum _ { j } \bar { w } _ { q j } ^ { \prime } S _ { q j } ( \bar { w } ) \bar { \bf z } _ { j } \right\| _ { 2 } ^ { 2 } , } & { } \\ { \displaystyle \dot { \bar { w } } = 0 } & { \Longleftrightarrow \quad \bar { w } _ { q j } ^ { \prime } S _ { q j } ( \bar { w } ) = 0 \quad \mathrm { f o r ~ a l l ~ } q , j . } \end{array}
$$

For $k \neq k _ { j }$ , the strict hierarchy gives

$$
S _ { k j } ( \bar { w } ) - S _ { k j } ( \bar { w } ) = \frac { \Sigma _ { k k _ { j } } ^ { ( j ) } - \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } } { n } \bar { w } _ { k _ { j } j } + \frac { \Sigma _ { k k } ^ { ( j ) } - \Sigma _ { k k _ { j } } ^ { ( j ) } } { n } \bar { w } _ { k j } > 0
$$

whenever $\bar { w } _ { k _ { i } j } + \bar { w } _ { k j } > 0 .$ , excluding two interior gates in the same context. For $q \ne k$ and $0 < \bar { w } _ { q j } ^ { \circ } < 1$ , stationarity requires

$$
\bar { w } _ { k j } ^ { \circ } = 1 \quad \Longrightarrow \quad 0 = S _ { q j } ( \bar { w } ^ { \circ } ) = \left( \sigma _ { \mu } ^ { 2 } + \frac { { \Sigma } _ { q q } ^ { ( j ) } } { n } \right) \bar { w } _ { q j } ^ { \circ } + \frac { { \Sigma } _ { q k } ^ { ( j ) } } { n } > 0 ,
$$

$$
\bar { w } _ { k j } ^ { \circ } = 0 \Longrightarrow \bar { w } _ { q j } ^ { \circ } = \frac { n \sigma _ { \mu } ^ { 2 } } { n \sigma _ { \mu } ^ { 2 } + \Sigma _ { q q } ^ { ( j ) } } .
$$

The four corners have $\bar { w } _ { q j } ^ { \prime } = 0$ , completing the classification. The unique optimum follows from convexity, strict convexity on the selected coordinates, and

$$
\begin{array} { r l r } & { \displaystyle \frac { \partial \mathcal { L } ^ { \mathrm { t r u n c a t e d } } } { \partial \bar { w } _ { q j } } = \frac { 1 } { 2 } S _ { q j } , } & { S _ { k _ { j } j } ( \bar { w } ^ { * } ) = 0 , } \\ & { \displaystyle S _ { k j } ( \bar { w } ^ { * } ) = \frac { \bar { w } _ { k _ { j } j } ^ { * } } { n } \big ( \Sigma _ { k k _ { j } } ^ { ( j ) } - \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } \big ) > 0 } & { ( k \ne k _ { j } ) . } \end{array}
$$

## D.3.2 PROOF OF LEMMA B.8

ProofofLemma B.8. Whenever $\bar { w } _ { q j } \in [ 1 / 4 , 3 / 4 ]$ , we have $\bar { w } _ { q j } ^ { \prime } \ge 3 / 1 6$ , and therefore

$$
\begin{array} { r l r } {  { \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { b i a s } } \| _ { 2 } ^ { 2 } = \frac { \sigma _ { \mu } ^ { 4 } } { 4 } \sum _ { q = 1 } ^ { 2 } \| \sum _ { j } ( \bar { w } _ { 1 j } + \bar { w } _ { 2 j } - 1 ) \bar { w } _ { q j } ^ { \prime } \bar { \mathbf { z } } _ { j } \| _ { 2 } ^ { 2 } } } \\ & { } & { \geq \frac { \sigma _ { \mu } ^ { 4 } \lambda _ { \operatorname* { m i n } } ( Z ) } { 4 } \sum _ { q , j } ( \bar { w } _ { 1 j } + \bar { w } _ { 2 j } - 1 ) ^ { 2 } ( \bar { w } _ { q j } ^ { \prime } ) ^ { 2 } } \\ & { } & { \geq 2 \alpha \mathcal { L } ^ { \mathrm { b i a s } } . } \end{array}
$$

Consequently, locally the flow satisfies a PL inequality. Moreover,

$$
- \frac { d } { d t } \sqrt { \mathcal { L } ^ { \mathrm { b i a s } } } = \frac { \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { b i a s } } \| _ { 2 } ^ { 2 } } { 2 \sqrt { \mathcal { L } ^ { \mathrm { b i a s } } } } \geq \sqrt { \frac { \alpha } { 2 } } \| \dot { \theta } \| _ { 2 }
$$

Thus,

$$
\| \theta ( 0 ) - \theta ( t ) \| _ { 2 } \leq \int _ { 0 } ^ { t } \| { \dot { \theta } } ( s ) \| _ { 2 } d s \leq \sqrt { \frac { 2 \mathcal { L } ^ { \mathrm { b i a s } } ( \theta ( 0 ) ) } { \alpha } } .
$$

Since the map $\theta \mapsto \bar { w }$ is globally $\sqrt { \lambda _ { \operatorname* { m a x } } ( Z ) } / 4 \mathrm { - } \mathrm { I }$ Lipschitz, we obtain

$$
\int _ { 0 } ^ { t } \| \dot { \bar { w } } ( s ) \| _ { \mathrm { F } } d s \leq \frac { \sqrt { \lambda _ { \operatorname* { m a x } } ( Z ) } } { 4 } \sqrt { \frac { 2 \mathcal { L } ^ { \mathrm { b i a s } } ( \theta ( 0 ) ) } { \alpha } } .
$$

Since ma $\mathrm { x } _ { q , j } \vert \bar { w } _ { q j } ( 0 ) - 1 / 2 \vert \leq \delta$ , we have $\mathcal { L } ^ { \mathrm { b i a s } } ( \theta ( 0 ) ) \leq 2 \sigma _ { \mu } ^ { 2 } \delta ^ { 2 }$ . Thus we obtain a travel bound on w¯ of

$$
\lVert \bar { w } ( 0 ) - \bar { w } ( t ) \rVert _ { \mathrm { F } } \leq C \delta .
$$

With $( 1 + C ) \delta < 1 / 4 \ /$ , a first-exit argument shows that the flow remains in $\bar { w } \in [ 1 / 4 , 3 / 4 ] ^ { 2 \times 2 }$ for all time. Moreover,

$$
\frac { d } { d t } \mathcal { L } ^ { \mathrm { { b i a s } } } = - \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { { b i a s } } } \| _ { 2 } ^ { 2 } \leq - 2 \alpha \mathcal { L } ^ { \mathrm { { b i a s } } } ,
$$

which gives the asserted exponential bound on the sum residuals.

The truncated gradient differs from the bias gradient by $\mathcal { O } ( n ^ { - 1 } )$ at fixed λ. The same bound holds for the full population gradient on bounded sets of θ, by the loss and gradient expansions. The PL inequality and $\| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { b i a s } } \| _ { 2 } \leq C \sqrt { \mathcal { L } ^ { \mathrm { b i a s } } }$ give

$$
\frac { d } { d t } \sqrt { \mathcal { L } ^ { \mathrm { b i a s } } } \leq - \alpha \sqrt { \mathcal { L } ^ { \mathrm { b i a s } } } + \frac { C } { n } , \sqrt { \mathcal { L } ^ { \mathrm { b i a s } } ( \theta ( t ) ) } \leq e ^ { - \alpha t } \sqrt { \mathcal { L } ^ { \mathrm { b i a s } } ( \theta ( 0 ) ) } + \frac { C } { \alpha n } .
$$

At zero loss the inequality is interpreted using the upper right derivative. Integrating $\begin{array} { r l } { \| \dot { \theta } \| _ { 2 } } & { { } \leq } \end{array}$ $C \sqrt { \mathcal { L } ^ { \mathrm { b i a s } } } + C / n$ now bounds the gate travel by $C \delta + C ( 1 + t ) / n$ . Choose $t _ { \mathrm { c r i t } } = 2 \alpha ^ { - 1 }$ log n and then n sufficiently large. The same first-exit argument keeps the gates in $[ 1 / 4 , 3 / 4 ]$ up $\mathrm { t o } t _ { \mathrm { c r i t } }$ , and the displayed bound gives the claimed $\mathcal { O } ( n ^ { - 1 } )$ residual. For the full population flow, the same travel bound keeps θ in a fixed bounded set up to $t _ { \mathrm { c r i t } }$

For the empirical flow, apply Lemma D.4 with $\beta = \operatorname* { m i n } \{ \epsilon , 1 \} / 4$ and failure probabilities of order $1 / n .$ , conditional on the family labels. Hoeffding’s inequality bounds the deviations of the family proportions from $1 / 2$ by $\widetilde { \mathcal { O } } ( m ^ { - 1 / 2 } )$ . On bounded sets of $\theta ,$ the conditional gradient expansion bounds each family’s population gradient by $C \operatorname* { m a x } _ { j } | \bar { w } _ { 1 j } + \bar { w } _ { 2 j } - 1 | + C / n$ . Combining these bounds gives, with probability at least $1 - 1 / n$ , uniformly on these sets,

$$
\begin{array} { r } { \| \nabla _ { \theta } ( \hat { \mathcal { L } } _ { m } - \mathcal { L } ) \| _ { 2 } \leq \widetilde { \mathcal { O } } ( n ^ { - 1 / 2 - \epsilon / 2 } ) \operatorname* { m a x } _ { i } | \bar { w } _ { 1 j } + \bar { w } _ { 2 j } - 1 | + \widetilde { \mathcal { O } } ( n ^ { - 1 - \epsilon / 4 } ) . } \end{array}\tag{17}
$$

For fixed $\sigma _ { \mu } ^ { 2 } > 0$ , the first term is $o ( 1 ) \sqrt { \mathcal { L } ^ { \mathrm { b i a s } } }$ and the second is $o ( n ^ { - 1 } )$ . Up to first exit from the bounded parameter set and $[ 1 / 4 , 3 / 4 ] ^ { 2 \times 2 }$ , the preceding PL inequality therefore gives

$$
\frac { d } { d t } \sqrt { \mathcal { L } ^ { \mathrm { b i a s } } } \leq - \frac { \alpha } { 2 } \sqrt { \mathcal { L } ^ { \mathrm { b i a s } } } + \frac { C } { n } , \qquad \sqrt { \mathcal { L } ^ { \mathrm { b i a s } } ( \theta ( t ) ) } \leq e ^ { - \alpha t / 2 } \sqrt { \mathcal { L } ^ { \mathrm { b i a s } } ( \theta ( 0 ) ) } + \frac { 2 C } { \alpha n } .
$$

The same integration of $\lVert \dot { \boldsymbol { \theta } } \rVert _ { 2 }$ bounds the travel by $C \delta + C ( 1 + t ) / n$ and prevents exit up to $t _ { \mathrm { c r i t } } = 2 \alpha ^ { - 1 }$ log n. This proves the empirical residual and parameter bounds. □

## D.3.3 PROOF OF LEMMA B.9

ProofofLemma B.9. First, we check the assumptions of Lemma D.8. By Lemma F.5, almost surely over ω we have $Z \succ 0 , Z _ { \mathrm { G A } } \geq 0$ and $Z _ { \mathrm { G A } } <$ min $\{ Z _ { \mathrm { G G } } , Z _ { \mathrm { A A } } \}$ , with $\lambda _ { \mathrm { m i n } } ( Z )$ bounded below independently of n.

Set $s = ( t - t _ { \mathrm { c r i t } } ) / n$ and

$$
r _ { j } = n \sigma _ { \mu } ^ { 2 } ( \bar { w } _ { 1 j } + \bar { w } _ { 2 j } - 1 ) .
$$

The truncated flow satisfies

$$
\frac { 1 } { n \sigma _ { \mu } ^ { 2 } } \frac { d r _ { k } } { d s } = - \frac { 1 } { 2 } \bar { w } _ { 1 k } ^ { \prime } \sum _ { j } Z _ { j k } \bar { w } _ { 1 j } ^ { \prime } \left( 2 r _ { j } + \sum _ { i = 1 } ^ { 2 } \bar { w } _ { i j } ( \Sigma _ { 1 i } ^ { ( j ) } + \Sigma _ { 2 i } ^ { ( j ) } ) \right) + \mathcal { O } \left( \frac { 1 } { n \sigma _ { \mu } ^ { 2 } } \right) ,
$$

$$
\frac { d } { d s } \left[ \frac { \sigma ^ { - 1 } ( \bar { w } _ { 1 k } ) - \sigma ^ { - 1 } ( \bar { w } _ { 2 k } ) } { 2 } \right] = - \frac { 1 } { 4 } \sum _ { j } Z _ { j k } \bar { w } _ { 1 j } ^ { \prime } n ( S _ { 1 j } - S _ { 2 j } ) ( \bar { w } ) + { \mathcal O } \left( \frac { 1 } { n \sigma _ { \mu } ^ { 2 } } \right) ,
$$

where $S _ { q j }$ is defined in Lemma B.7. On bounded coordinate sets, the vector fields are bounded through two derivatives, and the remainder bounds hold through two derivatives.

Initialize the reduced flow in Lemma D.8 by

$$
\bar { w } _ { 1 j } ^ { \mathrm { r e d u c e d } } ( 0 ) = \sigma \left( \frac { \sigma ^ { - 1 } ( \bar { w } _ { 1 j } ( t _ { \mathrm { c r i t } } ) ) - \sigma ^ { - 1 } ( \bar { w } _ { 2 j } ( t _ { \mathrm { c r i t } } ) ) } { 2 } \right) , \qquad \bar { w } _ { 2 j } ^ { \mathrm { r e d u c e d } } ( 0 ) = 1 - \bar { w } _ { 1 j } ^ { \mathrm { r e d u c e d } } ( 0 ) .
$$

Then

$$
\begin{array} { r l } & { \displaystyle \frac { d } { d s } \left( n \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { \mathrm { r e d u c e d } } ) \right) = - \frac { 1 } { 8 } \sum _ { j , k } \bar { w } _ { 1 j } ^ { \mathrm { r e d u c e d } \prime } n ( S _ { 1 j } - S _ { 2 j } ) ( \bar { w } ^ { \mathrm { r e d u c e d } } ) } \\ & { \qquad \times \sum _ { j k } \bar { w } _ { 1 k } ^ { \mathrm { r e d u c e d } \prime } n ( S _ { 1 k } - S _ { 2 k } ) ( \bar { w } ^ { \mathrm { r e d u c e d } } ) } \\ & { \qquad \leq - \frac { \lambda _ { \operatorname* { m i n } } ( Z ) } { 8 } \sum _ { j } ( \bar { w } _ { 1 j } ^ { \mathrm { r e d u c e d } \prime } n ( S _ { 1 j } - S _ { 2 j } ) ( \bar { w } ^ { \mathrm { r e d u c e d } } ) ) ^ { 2 } \leq 0 . } \end{array}
$$

The sum is integrable and uniformly continuous, hence tends to zero. Since $n ( S _ { \mathrm { 1 G } } - S _ { \mathrm { 2 G } } ) < 0 <$ $n ( S _ { \mathrm { 1 A } } - S _ { \mathrm { 2 A } } )$ , connectedness of the accumulation set gives convergence to a corner. Lemma $_ { \mathrm { ~ D . 8 . } }$ also with indices interchanged, excludes the two corners selecting the same expert. Loss monotonicity excludes the remaining corner, which selects the wrong expert in both contexts. Indeed, on the critical manifold

$$
{ \mathcal E } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ) = \frac { 1 } { 4 n } \sum _ { j } q _ { j } ( \bar { w } _ { 1 j } ) , \qquad q _ { j } ( \pi ) = \pi ^ { 2 } \Sigma _ { 1 1 } ^ { ( j ) } + 2 \pi ( 1 - \pi ) \Sigma _ { 1 2 } ^ { ( j ) } + ( 1 - \pi ) ^ { 2 } \Sigma _ { 2 2 } ^ { ( j ) } ,
$$

and $q _ { j } ^ { \prime \prime } = 2 ( 1 , - 1 ) \Sigma ^ { ( j ) } ( 1 , - 1 ) ^ { \intercal } \geq 0$ because $\pmb { \Sigma } ^ { ( j ) }$ is a covariance matrix. Each $q _ { j }$ is therefore convex on [0, 1] and attains its maximum at an endpoint. As $q _ { j } ( 1 ) = \Sigma _ { 1 1 } ^ { ( j ) }$ and $q _ { j } ( 0 ) = \Sigma _ { 2 2 } ^ { ( j ) }$ , which the strict hierarchy separates, the maximum is attained at the endpoint selecting the wrong expert in family $j ,$ , and strictly so over $\bar { w } _ { 1 j } \in [ c , 1 - c ]$ for any $c > 0$ . Summing over $j ,$ , the corner selecting the wrong expert in both contexts is the strict maximizer of $\mathcal { L } ^ { \mathrm { t r u n c a t e d } }$ on the manifold. Since $\bar { w } ( t _ { \mathrm { c r i t } } ) \in [ 1 / \bar { 4 } , 3 / 4 ] ^ { 2 \times 2 }$ gives bounded logits, $\bar { w } ^ { \mathrm { r e d u c e d } } ( 0 )$ lies in $[ c , 1 - c ] ^ { 2 }$ for some absolute $c > 0$ so the nonincreasing reduced flow cannot converge to that corner. Thus $\bar { w } ^ { \mathrm { r e d u c e d } } ( s ) \to I _ { 2 }$ , and compactness, continuous dependence, and loss monotonicity give common $S , \kappa > 0$ with

$$
n \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { \mathrm { r e d u c e d } } ( S ) ) \leq \operatorname* { m i n } _ { \bar { w } ^ { \mathrm { o } } \operatorname* { i n } ( 1 5 ) } n \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { \mathrm { o } } ) - 4 \kappa .
$$

Bounded logit derivatives give $\bar { w } _ { 1 j } ^ { \mathrm { r e d u c e d } } ( s ) \in [ c , 1 - c ] \mathrm { o n } [ 0 , S ]$ , uniformly for some $0 < c < 1 / 2$ Following Theorem G.2, we set

$$
\begin{array} { c } { \displaystyle { y = \left( \frac { \sigma ^ { - 1 } ( \bar { w } _ { 1 j } ) - \sigma ^ { - 1 } ( \bar { w } _ { 2 j } ) } { 2 } \right) _ { j \in \{ \mathrm { G } , \mathrm { A } \} } , \qquad \varepsilon = \frac { 1 } { n \sigma _ { \mu } ^ { 2 } } , } } \\ { \displaystyle { y ^ { 0 } ( s ) = \left( \sigma ^ { - 1 } ( \bar { w } _ { 1 j } ^ { \mathrm { r e d u c e d } } ( s ) ) \right) _ { j \in \{ \mathrm { G } , \mathrm { A } \} } . } } \end{array}
$$

On $\bar { w } _ { 2 j } = 1 - \bar { w } _ { 1 j }$ , the limiting normal equilibrium and stability bound are

$$
\begin{array} { r } { r _ { j } ^ { \star } ( y ) = - \displaystyle \frac { 1 } { 2 } \sum _ { i = 1 } ^ { 2 } \bar { w } _ { i j } ( \Sigma _ { 1 i } ^ { ( j ) } + \Sigma _ { 2 i } ^ { ( j ) } ) , } \\ { \lambda _ { \operatorname* { m a x } } \left( \left[ - \bar { w } _ { 1 k } ^ { \prime } Z _ { j k } \bar { w } _ { 1 j } ^ { \prime } \right] _ { k , j \in \{ \mathrm { G } , \mathrm { A } \} } \right) \le - c ^ { 2 } ( 1 - c ) ^ { 2 } \lambda _ { \operatorname* { m i n } } ( Z ) < 0 . } \end{array}
$$

The affine normal equation gives, up to first exit from a bounded interior neighborhood,

$$
\left\| \left( r _ { j } ( s ) + \frac { 1 } { 2 } \sum _ { i = 1 } ^ { 2 } \bar { w } _ { i j } ( t _ { \mathrm { c r i t } } + n s ) ( \Sigma _ { 1 i } ^ { ( j ) } + \Sigma _ { 2 i } ^ { ( j ) } ) \right) _ { j \in \{ \mathbf { G } , \mathbf { A } \} } \right\| _ { 2 } \leq C e ^ { - \kappa _ { 0 } n \sigma _ { \mu } ^ { 2 } s } + \frac { C } { n \sigma _ { \mu } ^ { 2 } } .
$$

This bounds r uniformly up to first exit. The limiting slow equation is independent of $r ,$ so its difference from the actual slow equation is uniformly $\bar { \mathcal { O } } ( \varepsilon )$ on this region. Since $y ( 0 ) = y ^ { 0 } ( 0 )$ , the Lipschitz bound and Gronwall’s inequality give, up to first exit,

$$
\begin{array} { c } { \displaystyle \| y ( s ) - y ^ { 0 } ( s ) \| _ { 2 } \leq C \displaystyle \int _ { 0 } ^ { s } \| y ( u ) - y ^ { 0 } ( u ) \| _ { 2 } d u + C \varepsilon s , } \\ { \displaystyle \displaystyle \operatorname* { s u p } _ { 0 \leq s \leq S } \| y ( s ) - y ^ { 0 } ( s ) \| _ { 2 } \leq C _ { S } \varepsilon . } \end{array}
$$

Since $\bar { w } _ { 1 j } + \bar { w } _ { 2 j } - 1 = \varepsilon r _ { j }$ , smooth inversion of the coordinates on the interior region gives

$$
\operatorname* { s u p } _ { 0 \leq s \leq S } \| \bar { w } ( t _ { \mathrm { c i t } } + n s ) - \bar { w } ^ { \mathrm { r e d u c e d } } ( s ) \| _ { \mathrm { F } } \leq \frac { C _ { S } } { n \sigma _ { \mu } ^ { 2 } } , \qquad \operatorname* { s u p } _ { 0 \leq s \leq S } \| r ( s ) \| _ { 2 } \leq C _ { S } .
$$

These bounds prevent exit before S, and

$$
\begin{array} { c } { { n \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t _ { \mathrm { c r i t } } + n S ) ) = n \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { \mathrm { r e d u c e d } } ( S ) ) + \mathcal { O } \left( \displaystyle \frac { 1 } { n \sigma _ { \mu } ^ { 2 } } \right) } } \\ { { < \displaystyle \operatorname* { m i n } _ { \bar { w } ^ { \mathrm { o i n } } ( 1 5 ) } n \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { \circ } ) - \kappa } } \end{array}
$$

for sufficiently large n.

For the full population flow, the loss and gradient expansions give, on bounded sets of $\theta ,$

$$
\begin{array} { r l } & { | \mathcal { L } - \mathcal { L } ^ { \mathrm { t r u n c a t e d } } | + \| \nabla _ { \theta } ( \mathcal { L } - \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ) \| _ { 2 } \leq C \left( \displaystyle \frac { \sigma _ { \mu } ^ { 2 } } { n } + n ^ { - 3 / 2 } \right) , } \\ & { \quad \quad \quad \quad \left\| \displaystyle \frac { d \theta _ { q } ^ { \perp } } { d s } \right\| _ { 2 } \leq C ( \sigma _ { \mu } ^ { 2 } + n ^ { - 1 / 2 } ) , } \end{array}
$$

uniformly in n and $0 < \sigma _ { \mu } ^ { 2 } \le 1$ , by Lemma F.6. By Lemma B.5, bounded logits and perpendicular components give bounded θ. The coordinate errors are $\mathcal { O } ( \sigma _ { \mu } ^ { 2 } + n ^ { - 1 / 2 } )$ , so normal stability and Gronwall’s inequality give, up to first exit,

$$
\operatorname* { s u p } _ { 0 \leq s \leq S } \| \bar { w } ( t _ { \mathrm { c r i t } } + n s ) - \bar { w } ^ { \mathrm { r e d u c e d } } ( s ) \| _ { \mathrm { F } } \leq C _ { S } \left( \sigma _ { \mu } ^ { 2 } + n ^ { - 1 / 2 } + \frac { 1 } { n \sigma _ { \mu } ^ { 2 } } \right) , \qquad \operatorname* { s u p } _ { 0 \leq s \leq S } \| r ( s ) \| _ { 2 } \leq C _ { S } .
$$

For sufficiently small $\sigma _ { \mu } ^ { 2 }$ and then sufficiently large $n ,$ these bounds prevent exit before $S ,$ and

$$
\begin{array} { r l } & { \operatorname* { m a x } \big \{ n \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t _ { \mathrm { c r i t } } + n S ) ) , ~ n \mathcal { L } ( \theta ( t _ { \mathrm { c r i t } } + n S ) ) \big \} } \\ & { \quad \leq n \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { \mathrm { r e d u c e d } } ( S ) ) + C _ { S } \left( \sigma _ { \mu } ^ { 2 } + n ^ { - 1 / 2 } + \frac { 1 } { n \sigma _ { \mu } ^ { 2 } } \right) } \\ & { \quad < \underset { \bar { w } ^ { \mathrm { o } } \mathrm { i n } ( 1 5 ) } { \operatorname* { m i n } } n \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { \circ } ) - \kappa . } \end{array}
$$

For the empirical flow, use the uniform concentration event in (17). Up to first exit from the same bounded region in $( r , y , \theta ^ { \perp } )$ , bounded r gives max<sub>j</sub> $| \bar { w } _ { 1 j } + \bar { w } _ { 2 j } - 1 | \overset { \cdot } { = } \mathcal { O } ( n ^ { - 1 } )$ for fixed $\sigma _ { \mu } ^ { 2 } > 0$ and hence

$$
\begin{array} { r } { n \| \nabla _ { \theta } ( \hat { \mathcal { L } } _ { m } - \mathcal { L } ) \| _ { 2 } = o ( 1 ) . } \end{array}
$$

In the rescaled equations this adds $o ( 1 )$ to the right-hand sides of the normal equation, the slow equation, and the equation for $\theta _ { q } ^ { \perp }$ . The normal stability estimate and Gronwall’s inequality for the slow equation consequently give

$$
\begin{array} { r l } { \displaystyle \operatorname* { s u p } _ { 0 \leq s \leq S } \| \bar { w } ( t _ { \mathrm { c r i t } } + n s ) - \bar { w } ^ { \mathrm { r e d u c e d } } ( s ) \| _ { \mathrm { F } } \leq C _ { S } \left( \sigma _ { \mu } ^ { 2 } + n ^ { - 1 / 2 } + \frac { 1 } { n \sigma _ { \mu } ^ { 2 } } \right) + o ( 1 ) , } & { } \\ { \displaystyle \operatorname* { s u p } _ { 0 \leq s \leq S } \| r ( s ) \| _ { 2 } \leq C _ { S } + o ( 1 ) . } & { } \end{array}
$$

Also, $\begin{array} { r } { \operatorname* { s u p } _ { 0 \leq s \leq S } \| \theta _ { q } ^ { \perp } ( t _ { \mathrm { c r i t } } + n s ) - \theta _ { q } ^ { \perp } ( t _ { \mathrm { c r i t } } ) \| _ { 2 } \leq C _ { S } \sigma _ { \mu } ^ { 2 } + o ( 1 ) } \end{array}$ . Choose the bounds on $r$ and $\theta ^ { \perp }$ strictly larger than these estimates and the logit bounds strictly larger than those of the reduced flow. For sufficiently small fixed $\sigma _ { \mu } ^ { 2 }$ and then sufficiently large n, the estimates prevent exit before $S .$ The preceding loss comparison acquires only an additional $o ( 1 )$ term after multiplication by n, so the strict loss margin is preserved. Bounded logits and perpendicular components give the stated parameter bound by Lemma B.5. □

Lemma D.8 (Exit from a neighborhood of a suboptimal stationary point). Let $Z _ { i j } = \langle \bar { \bf z } _ { i } , \bar { \bf z } _ { j } \rangle$ and assume

$$
Z _ { \mathrm { G A } } < \operatorname* { m i n } \left\{ Z _ { \mathrm { G G } } , Z _ { \mathrm { A A } } \right\} .
$$

In the time variable $s = ( t - t _ { c r i t } ) / n$ , consider the reducedflow on the critical manifold $\bar { w } _ { 1 j } + \bar { w } _ { 2 j } = 1$ given by

$$
\dot { \bar { w } } _ { 1 q } = - \frac { 1 } { 4 } \bar { w } _ { 1 q } ^ { \prime } \sum _ { j } Z _ { j q } \bar { w } _ { 1 j } ^ { \prime } n ( S _ { 1 j } - S _ { 2 j } ) \ d ( \bar { w } ) ,
$$

where $S _ { q j }$ is defined in Lemma $B . 7 ,$ so that

$$
n ( S _ { 1 j } - S _ { 2 j } ) ( \bar { w } ) = ( \Sigma _ { 1 1 } ^ { ( j ) } - \Sigma _ { 1 2 } ^ { ( j ) } ) \bar { w } _ { 1 j } + ( \Sigma _ { 1 2 } ^ { ( j ) } - \Sigma _ { 2 2 } ^ { ( j ) } ) \bar { w } _ { 2 j } .
$$

Suppose the expert hierarchy of Definition B.1 holds and consider the suboptimal stationary point $\bar { w } _ { \mathrm { 1 G } } ^ { \circ } = \bar { w } _ { \mathrm { 1 A } } ^ { \circ } = 1 , \bar { w } _ { \mathrm { 2 G } } ^ { \circ } = \bar { w } _ { \mathrm { 2 A } } ^ { \circ } = 0 .$ . The corner selecting the other expert in both contexts is covered by exchanging the expert indices. There exist $\delta \in ( 0 , 1 / 2 ) , r _ { 0 } \in ( 0 , 1 ) , p > 0$ and $\gamma > 0 ,$ , depending only on $Z$ and $\{ \pmb { \Sigma } ^ { ( j ) } \} _ { j \in \{ \mathrm { G } , \mathrm { A } \} }$ , such that thefollowing holds. Define

$$
R _ { \delta } = \{ 0 < \bar { w } _ { \mathrm { 2 G } } , \bar { w } _ { \mathrm { 2 A } } < \delta \}
$$

$$
W ( \bar { w } ) = \frac { 1 } { \bar { w } _ { \mathrm { 2 A } } } \operatorname* { m a x } \left\{ 1 , \left( \frac { \bar { w } _ { \mathrm { 2 G } } } { r _ { 0 } \bar { w } _ { \mathrm { 2 A } } } \right) ^ { p } \right\} .
$$

For every initialization $\bar { w } ( 0 ) \in R _ { \delta }$ , the first exit time satisfies

$$
\operatorname* { i n f } \{ s \geq 0 : \bar { w } ( s ) \notin R _ { \delta } \} \leq \frac { W ( \bar { w } ( 0 ) ) } { \gamma } < \infty .
$$

Proof. Note that $\bar { w } _ { 1 q } ^ { \prime } = \bar { w } _ { 2 q } ^ { \prime }$ . Also

$$
\begin{array} { r } { n ( S _ { 1 \mathrm { G } } - S _ { 2 \mathrm { G } } ) ( \bar { w } ) \leq 0 , \qquad n ( S _ { 1 \mathrm { A } } - S _ { 2 \mathrm { A } } ) ( \bar { w } ^ { \circ } ) = \Sigma _ { 1 1 } ^ { ( \mathrm { A } ) } - \Sigma _ { 1 2 } ^ { ( \mathrm { A } ) } > 0 . } \end{array}
$$

Thus, for sufficiently small $\delta > 0$

$$
n ( S _ { \mathrm { 1 A } } - S _ { \mathrm { 2 A } } ) ( \bar { w } ) \geq \frac 1 2 n ( S _ { \mathrm { 1 A } } - S _ { \mathrm { 2 A } } ) ( \bar { w } ^ { \circ } ) > 0
$$

for all $\bar { w } \in R _ { \delta }$ . Choose $p > 0$ such that

$$
\begin{array} { r } { p Z _ { \mathrm { G G } } - ( p + 1 ) Z _ { \mathrm { G A } } > 0 , } \\ { ( p + 1 ) Z _ { \mathrm { A A } } - p Z _ { \mathrm { G A } } > 0 . } \end{array}
$$

Such a p exists: since $Z _ { \mathrm { A A } } > 0$ and $Z _ { \mathrm { G A } } < Z _ { \mathrm { A A } }$ , the second inequality holds for every $p > 0 ,$ and since $Z _ { \mathrm { G A } } < Z _ { \mathrm { G G } }$ , the first holds for every $p >$ m $\mathrm { a x } \{ 0 , Z _ { \mathrm { G A } } \} / ( Z _ { \mathrm { G G } } - Z _ { \mathrm { G A } } )$ . Choose δ sufficiently small that

$$
\begin{array} { r } { p ( 1 - \bar { w } _ { \mathrm { 2 G } } ) Z _ { \mathrm { G G } } - ( p + 1 ) ( 1 - \bar { w } _ { \mathrm { 2 A } } ) Z _ { \mathrm { G A } } > 0 , } \\ { ( p + 1 ) ( 1 - \bar { w } _ { \mathrm { 2 A } } ) Z _ { \mathrm { A A } } - p ( 1 - \bar { w } _ { \mathrm { 2 G } } ) Z _ { \mathrm { G A } } > 0 , } \end{array}
$$

uniformly on $R _ { \delta }$ .

We show $\dot { W } \leq - \gamma < 0$ for $\bar { w } \in R _ { \delta }$

If $\bar { w } _ { \mathrm { 2 G } } \leq r _ { 0 } \bar { w } _ { \mathrm { 2 A } }$ , then $W ( \bar { w } ) = 1 / \bar { w } _ { \mathrm { 2 A } }$ and

$$
\begin{array} { r l } & { \dot { W } = - \frac { \dot { \overline { { w } } } _ { \mathrm { 2 A } } } { \dot { \overline { { w } } } _ { \mathrm { 2 A } } ^ { 2 } } } \\ & { \quad = - \frac { ( 1 - \bar { w } _ { 2 \mathrm { A } } ) ^ { 2 } } { 4 } \left[ Z _ { \mathrm { G A } } \frac { \bar { w } _ { 1 \mathrm { G } } ^ { \prime } } { \bar { w } _ { 1 \mathrm { A } } ^ { \prime } } n ( S _ { 1 \mathrm { G } } - S _ { 2 \mathrm { G } } ) ( \bar { w } ) + Z _ { \mathrm { A A } } n ( S _ { 1 \mathrm { A } } - S _ { 2 \mathrm { A } } ) ( \bar { w } ) \right] . } \end{array}
$$

Moreover,

$$
\frac { \bar { w } _ { \mathrm { 1 G } } ^ { \prime } } { \bar { w } _ { \mathrm { 1 A } } ^ { \prime } } = \frac { \bar { w } _ { \mathrm { 2 G } } ( 1 - \bar { w } _ { \mathrm { 2 G } } ) } { \bar { w } _ { \mathrm { 2 A } } ( 1 - \bar { w } _ { \mathrm { 2 A } } ) } \leq \frac { r _ { 0 } } { 1 - \delta } .
$$

Since $n ( S _ { \mathrm { 1 G } } - S _ { \mathrm { 2 G } } )$ is bounded, choose $r _ { 0 } \in ( 0 , 1 )$ such that

$$
\operatorname* { m a x } \{ Z _ { \mathrm { G A } } , 0 \} \frac { r _ { 0 } } { 1 - \delta } \operatorname* { s u p } _ { \bar { w } \in \overline { { R _ { \delta } } } } \left| n ( S _ { \mathrm { 1 G } } - S _ { \mathrm { 2 G } } ) ( \bar { w } ) \right| \leq \frac { 1 } { 4 } Z _ { \mathrm { A A } } n ( S _ { \mathrm { 1 A } } - S _ { \mathrm { 2 A } } ) ( \bar { w } ^ { \circ } ) .
$$

Then

$$
\dot { W } \leq - \frac { ( 1 - \delta ) ^ { 2 } Z _ { \mathrm { A A } } n ( S _ { \mathrm { 1 A } } - S _ { \mathrm { 2 A } } ) ( \bar { w } ^ { \circ } ) } { 1 6 } < 0 .\tag{18}
$$

Suppose next that $\bar { w } _ { \mathrm { 2 G } } \geq r _ { 0 } \bar { w } _ { \mathrm { 2 A } }$ . For the branch $W = \bar { w } _ { \mathrm { 2 G } } ^ { p } / ( r _ { 0 } ^ { p } \bar { w } _ { \mathrm { 2 A } } ^ { p + 1 } )$ , the uniform bounds above give a constant $c > 0$ , independent of w¯ $\in R _ { \delta }$ , such that

$$
\begin{array} { r l } {  { 4 \frac { \hat { W } } { W } = 4 ( \frac { \hat { w } _ { 2 G } ^ { \prime } } { \hat { w } _ { 2 G } } - ( p + 1 ) \frac { \hat { w } _ { 2 A } ^ { \prime } } { \hat { w } _ { 2 G } } ) } } \\ & { = p \frac { \hat { w } _ { 1 B } ^ { \prime } } { \hat { w } _ { 2 G } } \sum _ { J \in \Omega } \bar { w } _ { 1 , J } ^ { \prime } n ( S _ { 1 , j } - S _ { 2 } ) ( \bar { w } ) - ( p + 1 ) \frac { \hat { w } _ { 1 B } ^ { \prime } } { \hat { w } _ { 2 A } } \sum _ { j _ { 3 } } \sum _ { \beta _ { 4 } \hat { N } _ { 4 } ^ { \prime } , n } ( S _ { 1 j } - S _ { 2 j } ) ( \bar { w } ) } \\ & { = p ( 1 - \hat { w } _ { 2 G } ) [ Z _ { G G } \hat { w } _ { 1 , \bar { \beta } } ^ { \prime } n ( S _ { 1 G } - S _ { 2 G } ) ( \bar { w } ) + Z _ { G A } \hat { w } _ { 1 , n } ^ { \prime } n ( S _ { 1 A } - S _ { 2 A } ) ( \bar { w } ) ] } \\ & { \quad - ( p + 1 ) ( 1 - \hat { w } _ { 2 \bar { \alpha } } ) [ Z _ { G \bar { A } } \bar { w } _ { 1 , \bar { \beta } } ^ { \prime } n ( S _ { 1 G } - S _ { 2 G } ) ( \bar { w } ) + Z _ { G \bar { A } } \bar { w } _ { 1 , n } ^ { \prime } n ( S _ { 1 A } - S _ { 2 A } ) ( \bar { w } ) ] } \\ & { = [ p ( 1 - \hat { w } _ { 2 G } ) ] _ { G \bar { G } \bar { G } } - ( p + 1 ) ( 1 - \hat { w } _ { 2 \bar { \alpha } } ) Z _ { G \bar { A } } ] \sigma _ { 1 \bar { \beta } } ( S _ { 1 G } - S _ { 2 G } ) ( \bar { w } ) } \\ &  \quad + [ p ( 1 \end{array}
$$

Thus

$$
\begin{array} { r l } & { \dot { W } \le - c \bar { w } _ { \mathrm { 1 A } } ^ { \prime } W } \\ & { \quad \quad = - c ( 1 - \bar { w } _ { \mathrm { 2 A } } ) \left( \frac { \bar { w } _ { \mathrm { 2 G } } } { r _ { 0 } \bar { w } _ { \mathrm { 2 A } } } \right) ^ { p } } \\ & { \quad \quad \le - c ( 1 - \delta ) < 0 . } \end{array}
$$

Both branch derivatives satisfy these bounds at $\bar { w } _ { \mathrm { 2 G } } = r _ { 0 } \bar { w } _ { \mathrm { 2 A } }$ as well. Since W is locally Lipschitz, $\dot { W } \le - \gamma$ almost everywhere along the trajectory for a common $\gamma > 0$ . Apply Lemma G.1 with lower bound $C = 0$ □

Lemma D.9 (Gradient bound away from stationary points). Fix λ, suppose the expert hierarchy of Definition B.1 holds and $Z \succ 0 ,$ , and let $\sigma _ { \mu } ^ { 2 } , c , \rho > 0 .$ . Use S from Lemma $B . 7$ and define

$$
\begin{array} { r l } & { R _ { c , \rho } = \left\{ \bar { w } \in [ 0 , 1 ] ^ { 2 \times 2 } : \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ) - \displaystyle \operatorname* { m i n } _ { \bar { w } ^ { 0 } \in \mathcal { S } } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { \circ } ) \leq \frac { c } { n } , \quad \displaystyle \operatorname* { m i n } _ { \bar { w } ^ { 0 } \in \mathcal { S } } \| \bar { w } - \bar { w } ^ { \circ } \| _ { \mathrm { F } } \geq \rho \right\} , } \\ & { \qquad C _ { \Sigma } = \displaystyle \operatorname* { m a x } _ { j \in \{ \mathsf { G } , \Delta \} } \left\{ \frac { 1 } { \Sigma _ { k , j k _ { j } } ^ { ( j ) } } , \frac { 1 } { \Sigma _ { q k _ { j } } ^ { ( j ) } - \Sigma _ { k , j k _ { j } } ^ { ( j ) } } , \frac { 1 } { \Sigma _ { q q } ^ { ( j ) } - \Sigma _ { q k _ { j } } ^ { ( j ) } } \right\} . } \end{array}
$$

If

$$
n \sigma _ { \mu } ^ { 2 } \geq \operatorname* { m a x } \left\{ 1 6 c + 4 \sum _ { j } \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } , \operatorname* { m a x } _ { i , k , j } \Sigma _ { i k } ^ { ( j ) } \right\} ,
$$

then every $\bar { w } \in R _ { c , \rho }$ satisfies

$$
\| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ) \| _ { 2 } \geq \frac { \sqrt { \lambda _ { \operatorname* { m i n } } ( Z ) } } { 1 6 0 C _ { \Sigma } } \frac { \rho } { n } .
$$

Proof. The loss bound and min $\bar { w } ^ { \circ } \in { \mathcal { S } } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { \circ } ) \leq \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( I _ { 2 } )$ give

$$
\left| \bar { w } _ { 1 j } + \bar { w } _ { 2 j } - 1 \right| ^ { 2 } \leq \frac { 4 c + \sum _ { \ell } \Sigma _ { \ell \ell } ^ { ( \ell ) } } { n \sigma _ { \mu } ^ { 2 } } \leq \frac { 1 } { 4 } .
$$

For each j, choose q with $\bar { w } _ { q j } \geq \bar { w } _ { k j } , k \neq q , \mathrm { s o } \bar { w } _ { q j } \geq 1 / 4$ and $\bar { w } _ { k j } \leq 3 / 4$ . The hierarchy gives $| S _ { q j } - S _ { k j } | \geq 1 / ( 2 \bar { C } _ { \Sigma } n ) . \mathrm { I f } \stackrel {  } { 1 } - \bar { w } _ { q j } \geq \bar { w } _ { k j } / 2 .$ , then

$$
\frac { \bar { w } _ { k j } } { 2 C _ { \Sigma } n } \leq \bar { w } _ { k j } | S _ { q j } - S _ { k j } | \leq 1 2 \operatorname* { m a x } _ { i } | \bar { w } _ { i j } ^ { \prime } S _ { i j } | .
$$

Otherwise $S _ { k j } \geq \Sigma _ { k q } ^ { ( j ) } \bar { w } _ { q j } / n \geq 1 / ( 4 C _ { \Sigma } n )$ , so $\bar { w } _ { k j } ^ { \prime } S _ { k j } \ge \bar { w } _ { k j } / ( 1 6 C _ { \Sigma } n )$ . Thus in both cases

$$
\bar { w } _ { k j } \leq 2 4 C _ { \Sigma } n \operatorname* { m a x } _ { i } | \bar { w } _ { i j } ^ { \prime } S _ { i j } | .
$$

As a polynomial in $\bar { w } _ { q j } , ( 1 - \bar { w } _ { q j } ) S _ { q j }$ has roots

$$
1 , \qquad \frac { n \sigma _ { \mu } ^ { 2 } - ( n \sigma _ { \mu } ^ { 2 } + \Sigma _ { q k } ^ { ( j ) } ) \bar { w } _ { k j } } { n \sigma _ { \mu } ^ { 2 } + \Sigma _ { q q } ^ { ( j ) } } ,
$$

separated by at least $\Sigma _ { q q } ^ { ( j ) } / ( n \sigma _ { \mu } ^ { 2 } + \Sigma _ { q q } ^ { ( j ) } )$ . Using $\bar { w } _ { q j } \ge 1 / 4$ and $( n \sigma _ { \mu } ^ { 2 } + \Sigma _ { q k } ^ { ( j ) } ) / ( n \sigma _ { \mu } ^ { 2 } + \Sigma _ { q q } ^ { ( j ) } ) \leq 2$ gives

$$
\operatorname* { m i n } \left\{ 1 - \bar { w } _ { q j } , \bigg | \bar { w } _ { q j } - \frac { n \sigma _ { \mu } ^ { 2 } } { n \sigma _ { \mu } ^ { 2 } + \Sigma _ { q q } ^ { ( j ) } } \bigg | \right\} \leq 8 C _ { \Sigma } n | \bar { w } _ { q j } ^ { \prime } S _ { q j } | + 2 \bar { w } _ { k j } .
$$

Setting $\bar { w } _ { k j } = 0$ and choosing the nearer of these two values in each family produces a stationary point, hence

$$
\rho \leq \operatorname* { m i n } _ { \bar { w } ^ { 0 } \in S } \| \bar { w } - \bar { w } ^ { 0 } \| _ { \mathrm { F } } \leq 8 0 C _ { \Sigma } n \left( \sum _ { i , j } ( \bar { w } _ { i j } ^ { \prime } S _ { i j } ) ^ { 2 } \right) ^ { 1 / 2 } \leq \frac { 1 6 0 C _ { \Sigma } n } { \sqrt { \lambda _ { \operatorname* { m i n } } ( Z ) } } \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 } .
$$

Lemma D.10 (Population gradient perturbation). Assume the initial loss and covariance conditions of Lemma $B . I O ,$ , with its constant $\kappa ,$ at a point $\theta _ { 0 } .$ . Fix $B , c > 0$ independently of n and $\sigma _ { \mu } ^ { 2 } .$ , and assume $\lVert \theta _ { 0 } \rVert _ { 2 } \leq B$ and L<sup>truncated</sup> $( \bar { w } ( \theta _ { 0 } ) ) - \mathcal { L } ^ { 1 }$ truncated $( \bar { w } ^ { * } ) \leq c / n$ . Here $\bar { w } ^ { * }$ denotes the optimum, and $\mathscr { L } ^ { \mathrm { t r u n c a i e d } } :$ is composed with $\bar { w } ( \theta )$ when differentiating. For a constant $C _ { 0 }$ and $\rho = C _ { 0 } \bar { / } ( n \sigma _ { \mu } ^ { 2 } )$ set

$$
E ( \theta ) = { \mathcal { L } } ^ { \mathrm { t r u n c a t e d } } ( { \bar { w } } ( \theta ) ) - { \mathcal { L } } ^ { \mathrm { t r u n c a t e d } } ( { \bar { w } } ^ { * } ) + { \frac { 1 } { n ^ { 2 } \sigma _ { \mu } ^ { 2 } } } , \quad \quad E _ { 0 } = { \frac { c + \kappa } { n } } + { \frac { 1 } { n ^ { 2 } \sigma _ { \mu } ^ { 2 } } } ,
$$

and let D be the closed set of θ satisfying

$$
\begin{array} { r l } & { \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( \theta ) ) \leq \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( \theta _ { 0 } ) ) + \frac { \kappa } { 2 n } , \qquad \displaystyle \operatorname* { m i n } _ { \bar { w } ^ { \upsilon } \in \mathcal { S } } \| \bar { w } ( \theta ) - \bar { w } ^ { \circ } \| _ { \mathrm { F } } \geq \rho , } \\ & { \qquad \| \theta \| _ { 2 } \leq B + 1 + 4 C _ { 0 } \log \frac { E _ { 0 } } { E ( \theta ) } . } \end{array}
$$

There is such a $C _ { 0 } ,$ , independent ofn and $\sigma _ { \mu } ^ { 2 } ,$ for which thefollowing holdfor sufficiently smallfixed $\sigma _ { \mu } ^ { 2 } > 0$ and then sufficiently large n.

(a) D is contained in the ball of radius $r = B + 1 + 4 C _ { 0 } \log ( 1 + ( c + \kappa ) n \sigma _ { \mu } ^ { 2 } ) = \mathcal { O } ( \log n ) ,$ and uniformly over $\theta \in \mathcal { D }$

$$
\begin{array} { r l r } {  { \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 } \geq \frac { E ( \theta ) } { C _ { 0 } } , } } \\ & { } & { \displaystyle \sum _ { q , j } \bar { w } _ { q j } ^ { \prime } \leq C _ { 0 } n \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 } , } \\ & { } & { \| \nabla _ { \theta } ( \mathcal { L } - \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ) \| _ { 2 } \leq [ C _ { 0 } \sigma _ { \mu } ^ { 2 } n E ( \theta ) ( 1 + \| \theta \| _ { 2 } ) ^ { 2 } + \widetilde { O } ( n ^ { - 1 } ) ] \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 } } \\ & { } & { \leq \frac { 1 } { 4 } \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 } . } \end{array}
$$

(b) Let $\theta ( t ) , t \geq t _ { c r i t }$ , be absolutely continuous with $\theta ( t _ { c r i t } ) = \theta _ { 0 }$ and

$$
\begin{array} { c } { \displaystyle \dot { \theta } = - \nabla _ { \theta } \mathcal { L } ( \theta ) - \zeta ( t ) , } \\ { \displaystyle \| \zeta ( t ) \| _ { 2 } \leq \operatorname* { m i n } \left\{ \frac { 1 } { 4 } , \frac { 1 } { \log n } \right\} \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \theta ( t ) ) \| _ { 2 } \quad \ w h e n e v e r \theta ( t ) \in \mathcal { D } , } \end{array}
$$

Let $t _ { \rho }$ be the first time $t \geq t _ { c r i t }$ with min<sub>w¯</sub>◦<sub>∈S</sub> $\| \bar { w } ( \theta ( t ) ) - \bar { w } ^ { \circ } \| _ { \mathrm { F } } \leq \rho .$ . Then $t _ { \rho } \leq t _ { c r i t } +$ $2 C _ { 0 } ^ { 2 } \dot { n } ^ { 2 } \sigma _ { \mu } ^ { 2 } , \theta ( t ) \in \mathscr { D } f o r t _ { c r i t } \le t < t _ { \rho } ,$ and for $t _ { c r i t } \leq t \leq t _ { \rho } ,$

$$
\frac { d } { d t } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( \theta ( t ) ) ) \leq - \frac { 1 } { 2 } \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 } ^ { 2 } , \qquad \| \dot { \theta } \| _ { 2 } \leq \frac { 3 } { 2 } \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 } ,
$$

$$
\| \theta ( t ) \| _ { 2 } \leq B + 3 C _ { 0 } \log \frac { E ( \theta _ { 0 } ) } { E ( \theta ( t ) ) } \leq B + 3 C _ { 0 } \log ( 1 + c n \sigma _ { \mu } ^ { 2 } ) ,
$$

$$
\| \theta _ { q } ^ { \perp } ( t ) - \theta _ { q } ^ { \perp } ( t _ { c r i t } ) \| _ { 2 } \leq \mathcal { O } ( \sigma _ { \mu } ^ { 2 } ) + \widetilde { \mathcal { O } } ( n ^ { - 1 } ) + \frac { 2 C _ { 0 } } { \log n } \log ( 1 + c n \sigma _ { \mu } ^ { 2 } ) .
$$

The full population flow is the case $\zeta \equiv 0 .$

Proof. Part (a). On D, the truncated loss is $\mathcal { O } ( n ^ { - 1 } )$ , lies at least $\kappa / ( 2 n )$ below the loss of every point in (15), and has excess at most $( c + \kappa ) / n$ . In particular, $1 / ( n ^ { 2 } \sigma _ { \mu } ^ { 2 } ) \leq E ( \theta ) \leq E _ { 0 }$ , which gives the radius r. The excess-loss bound (20) of Lemma D.11 gives

$$
\operatorname* { m a x } _ { j } \left[ \sum _ { q } \bar { w } _ { q j } ^ { \prime } + \left| \sum _ { q } \bar { w } _ { q j } - 1 \right| \right] \lesssim n [ \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ) - \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { * } ) ] + \frac { 1 } { n \sigma _ { \mu } ^ { 2 } } = n E ( \theta ) .
$$

Minimizing the quadratic loss over the other gates shows that the loss gap below each wrong selection bounds $\bar { w } _ { k _ { j } j }$ and $1 - \bar { w } _ { q j }$ away from zero for $q \neq k _ { j }$ . The same two-case calculation as in Lemma D.9 then gives

$$
\frac { \bar { w } _ { q j } } { n } + \sigma _ { \mu } ^ { 2 } \left( \sum _ { i } \bar { w } _ { i j } - 1 \right) ^ { 2 } \lesssim \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 } + \frac { 1 } { n ^ { 2 } \sigma _ { \mu } ^ { 2 } } .
$$

Expanding the quadratic loss and applying Lemma D.9 with $\rho = C _ { 0 } / ( n \sigma _ { \mu } ^ { 2 } )$ proves the first estimate for sufficiently large $C _ { 0 }$ . Combined with the first display, it gives the second estimate after enlarging $C _ { 0 }$ . These two estimates depend only on w¯, so $C _ { 0 }$ is fixed before $\sigma _ { \mu } ^ { 2 }$ is chosen.

Taylor-expand the conditional gradient through cubic order in the centered empirical moments. Their signed cubic moments and fourth absolute moments are $\mathcal { O } ( n ^ { - 2 } )$ , so retaining the sigmoid derivatives gives

$$
\| \nabla _ { \theta } ( \mathcal { L } - \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ) \| _ { 2 } \lesssim \sum _ { q , j } \bar { w } _ { q j } ^ { \prime } \left[ \frac { \sigma _ { \mu } ^ { 2 } ( 1 + \| \theta \| _ { 2 } ) ^ { 2 } } { n } \left( \sum _ { i } \bar { w } _ { i j } ^ { \prime } + \left| \sum _ { i } \bar { w } _ { i j } - 1 \right| \right) + \frac { ( 1 + \| \theta \| _ { 2 } ) ^ { 4 } } { n ^ { 2 } } \right] .
$$

This bound is uniform for $\| \theta \| _ { 2 } \leq r$ and $0 < \sigma _ { u } ^ { 2 } \le 1$ , using the moment bounds and truncation from Lemma D.4. The first display and the second estimate turn it into the first inequality of the third estimate. On $\mathcal { D } , 1 + \| \theta \| _ { 2 } \overset { - } { \le } 2 + B + 4 C _ { 0 } \log ( E _ { 0 } / E ( \theta ) )$ ), so with $u = E ( \theta ) / \bar { E _ { 0 } } \in ( \bar { 0 } , 1 ]$ and $n E _ { 0 } = \mathcal { O } ( 1 )$

$$
C _ { 0 } \sigma _ { \mu } ^ { 2 } n E ( \theta ) ( 1 + \| \theta \| _ { 2 } ) ^ { 2 } \leq C _ { 0 } \sigma _ { \mu } ^ { 2 } n E _ { 0 } \operatorname* { s u p } _ { 0 < u \leq 1 } u \left( 2 + B + 4 C _ { 0 } \log \frac { 1 } { u } \right) ^ { 2 } = \mathcal { O } ( \sigma _ { \mu } ^ { 2 } ) .
$$

Choose $\sigma _ { \mu } ^ { 2 }$ small and then n large to make the bracket at most $1 / 4$

Part (b). Write $\dot { \theta } = - \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } - p$ with $p = \nabla _ { \theta } ( \mathcal { L } - \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ) + \zeta$ . While $\theta ( t ) \in \mathcal { D }$ , part (a) gives $\begin{array} { r } { \| p \| _ { 2 } \leq \frac { 1 } { 2 } \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 } . } \end{array}$ , hence the displayed descent and speed bounds, and, by the first estimate,

$$
\frac { d } { d t } \log E ( \theta ( t ) ) \leq - \frac { \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 } ^ { 2 } } { 2 E ( \theta ( t ) ) } \leq - \frac { \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 } } { 2 C _ { 0 } } .
$$

Integrating,

$$
\int _ { t _ { \mathrm { c e l l } } } ^ { t } \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 } d s \leq 2 C _ { 0 } \log \frac { E ( \theta _ { 0 } ) } { E ( \theta ( t ) ) } , \qquad \int _ { t _ { \mathrm { c e l l } } } ^ { t } \| \dot { \theta } ( s ) \| _ { 2 } d s \leq 3 C _ { 0 } \log \frac { E ( \theta _ { 0 } ) } { E ( \theta ( t ) ) } .\tag{19}
$$

We first show that the trajectory cannot leave D before $t _ { \rho } .$ Let $T$ be the supremum of times $t \leq t _ { \rho }$ with $\theta ( [ t _ { \mathrm { c r i t } } , t ] ) \subset \mathcal { D }$ , and suppose $T < t _ { \rho } .$ . Since D is closed, (19) holds up to $T ,$ , so

$$
\mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( \theta ( T ) ) ) \leq \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( \theta _ { 0 } ) ) < \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( \theta _ { 0 } ) ) + \frac { \kappa } { 2 n } ,
$$

$$
\| \theta ( T ) \| _ { 2 } \leq B + 3 C _ { 0 } \log \frac { E ( \theta _ { 0 } ) } { E ( \theta ( T ) ) } < B + 1 + 4 C _ { 0 } \log \frac { E _ { 0 } } { E ( \theta ( T ) ) } ,
$$

using $E ( \theta _ { 0 } ) \leq E _ { 0 }$ , while the distance to $s$ exceeds $\rho$ because $T < t _ { \rho } . \mathrm { A l l }$ l three conditions defining D hold strictly at $\theta ( T )$ , so by continuity they hold on a neighborhood of T, contradicting the definition of T.

For finite entry, set $W = - 1 / E . \mathrm { O n } \left[ t _ { \mathrm { c r i t } } , t _ { \rho } \right)$

$$
\frac { d } { d t } W ( \theta ( t ) ) = \frac { \dot { E } } { E ^ { 2 } } \leq - \frac { \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 } ^ { 2 } } { 2 E ^ { 2 } } \leq - \frac { 1 } { 2 C _ { 0 } ^ { 2 } } , \qquad - n ^ { 2 } \sigma _ { \mu } ^ { 2 } \leq W < 0 ,
$$

so, as in Lemma G.1, $t _ { \rho } \leq t _ { \mathrm { c r i t } } + 2 C _ { 0 } ^ { 2 } n ^ { 2 } \sigma _ { \mu } ^ { 2 }$ . The parameter bound follows from $( 1 9 ) , \| \theta _ { 0 } \| _ { 2 } \leq B _ { \mathrm { \ell } }$ and $n ^ { 2 } \sigma _ { \mu } ^ { 2 } E ( \theta _ { 0 } ) \leq 1 + c n \sigma _ { \mu } ^ { 2 } \leq n ^ { 2 } \sigma _ { \mu } ^ { 2 } E _ { 0 }$

Since the truncated gradient has zero perpendicular component, $\| \dot { \theta } _ { q } ^ { \perp } \| _ { 2 } \leq \| \nabla _ { \theta } ( \mathcal { L } - \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ) \| _ { 2 } +$ $\| \zeta \| _ { 2 }$ . Along the trajectory, $E$ decreases and $\| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 } d s \leq - 2 C _ { 0 } d E / E$ . Substituting the third estimate of part (a) and the parameter bound of $\mathcal { D }$

$$
\begin{array} { r l } { \displaystyle \int _ { t _ { \mathrm { c e l t } } } ^ { t } \| \nabla _ { \theta } ( \mathcal { L } - \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ) \| _ { 2 } d s \leq 2 C _ { 0 } ^ { 2 } \sigma _ { \mu } ^ { 2 } n E _ { 0 } \int _ { 0 } ^ { 1 } \bigg ( 2 + B + 4 C _ { 0 } \log \frac { 1 } { u } \bigg ) ^ { 2 } d u } & { } \\ { \displaystyle + \widetilde O ( n ^ { - 1 } ) \log ( 1 + c n \sigma _ { \mu } ^ { 2 } ) } & { } \\ { \displaystyle = O ( \sigma _ { \mu } ^ { 2 } ) + \widetilde O ( n ^ { - 1 } ) . } \end{array}
$$

Finally, $\begin{array} { r } { \int _ { t _ { \mathrm { c r i t } } } ^ { t } \| \zeta \| _ { 2 } d s \leq \frac { 2 C _ { 0 } } { \log n } \log ( 1 + c n \sigma _ { \mu } ^ { 2 } ) } \end{array}$ by (19).

## D.3.4 PROOF OF LEMMA B.10

Proof. Use S from Lemma B.7 and $C _ { \Sigma }$ from Lemma D.9. Until entry within distance $1 / n$ of $s ,$

$$
\begin{array} { r l } & { \quad \quad \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 } \geq \frac { \sqrt { \lambda _ { \operatorname* { m i n } } ( Z ) } } { 1 6 0 C _ { \Sigma } n ^ { 2 } } , } \\ & { \frac { d } { d t } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t ) ) = - \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 } ^ { 2 } \leq - \frac { \lambda _ { \operatorname* { m i n } } ( Z ) } { ( 1 6 0 C _ { \Sigma } ) ^ { 2 } n ^ { 4 } } . } \end{array}
$$

Since $\mathcal { L } ^ { \mathrm { t r u n c a t e d } } \geq 0$ , entry occurs by

$$
t _ { \theta } \leq t _ { \mathrm { c r i t } } + \frac { ( 1 6 0 C _ { \Sigma } ) ^ { 2 } n ^ { 4 } } { \lambda _ { \operatorname* { m i n } } ( Z ) } \mathcal L ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t _ { \mathrm { c r i t } } ) ) = t _ { \mathrm { c r i t } } + \mathcal O ( n ^ { 3 } ) .
$$

For $\rho = \mathcal { O } ( n ^ { - 1 } )$ , ρ-neighborhoods of stationary points with both gates 0 or 1 in a context have

$$
\mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ) \geq \frac { \sigma _ { \mu } ^ { 2 } } { 4 } - \mathcal { O } ( n ^ { - 1 } ) > \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t _ { \mathrm { c r i t } } ) ) .
$$

For every $\bar { w } ^ { \circ }$ in (15) and every $\bar { w }$ with $\| { \bar { w } } - { \bar { w } } ^ { \circ } \| _ { \mathrm { F } } = { \mathcal { O } } ( n ^ { - 1 } )$ , the quadratic expansion gives

$$
\begin{array} { r l } & { \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ) \geq \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { \circ } ) - \mathcal { O } ( n ^ { - 2 } ) } \\ & { ~ > \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t _ { \mathrm { c r i t } } ) ) . } \end{array}
$$

Thus, by Lemma B.7, loss descent permits entry only near stationary points with

$$
\bar { w } _ { \cdot j } ^ { \circ } \in \{ e _ { k _ { j } } , \bar { w } _ { \cdot j } ^ { * } \} , \qquad \| e _ { k _ { j } } - \bar { w } _ { \cdot j } ^ { * } \| _ { 2 } = \frac { \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } } { n \sigma _ { \mu } ^ { 2 } + \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } } .
$$

Adding the entry radius $1 / n$ bounds $\| \bar { w } _ { \cdot j } ( t _ { \theta } ) - e _ { k _ { j } } \| _ { 2 }$ as claimed.

For the full flow, take B and c as the assumed bounds on $\| \theta ( t _ { \mathrm { c r i t } } ) \| _ { 2 }$ and $n \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t _ { \mathrm { c r i t } } ) )$ , and apply part (b) of Lemma D.10 with $\theta _ { 0 } = \theta ( t _ { \mathrm { c r i t } } )$ and $\zeta \equiv 0 .$ Set $t _ { \theta } \stackrel { . . . } { = } t _ { \rho } \leq t _ { \mathrm { c r i t } } + 2 C _ { 0 } ^ { 2 } \stackrel { . . } { n } ^ { 2 } \sigma _ { \mu } ^ { 2 }$ . It gives L<sup>truncated</sup>( ¯w(t<sub>θ</sub>)) ≤ L<sup>truncated</sup> $( \bar { w } ( t _ { \mathrm { c r i t } } ) )$ and $\| \theta ( t ) \| _ { 2 } \leq B + 3 C _ { 0 } \log ( 1 + c n \sigma _ { \mu } ^ { 2 } ) = \mathcal { O } ( \log n )$ for $t _ { \mathrm { c r i t } } \leq t \leq t _ { \theta }$ . Since $\rho = \mathcal { O } ( n ^ { - 1 } )$ , the exclusion above applies to a point of $s$ within distance $\rho$ of $\bar { w } ( t _ { \theta } )$ , and the triangle inequality gives the full-flow gate bound. □

## D.3.5 PROOF OF LEMMA B.11

Proof. Let D and $r = \mathcal { O } ( \log n )$ be as in Lemma D.10 with $\theta _ { 0 } = \theta ( t _ { \mathrm { c r i t } } )$ . The radius r depends only on $B , c , \kappa ,$ , and $C _ { 0 } ,$ so the concentration events below do not depend on the data-dependent initial point. Write $m ^ { ( j ) } = \# \{ b : \tau _ { b } = j \}$ and $\begin{array} { r } { \mathcal { L } ^ { ( j ) } = \frac { 1 } { 2 } \mathbb { E } _ { j } [ ( \widehat { \mu } - \mu ) ^ { 2 } ] } \end{array}$ , with $\hat { \mathcal { L } } _ { m ^ { ( j ) } } ^ { ( j ) }$ its empirical average over the tasks of family j, so that

$$
\mathcal { L } = \frac { 1 } { 2 } \sum _ { j } \mathcal { L } ^ { ( j ) } , \qquad \hat { \mathcal { L } } _ { m } = \sum _ { j } \frac { m ^ { ( j ) } } { m } \hat { \mathcal { L } } _ { m ^ { ( j ) } } ^ { ( j ) } .
$$

Concentration. Hoeffding’s inequality gives, with failure probability at most $1 / ( 3 n )$

$$
\operatorname* { m a x } _ { j } \left| \frac { m ^ { ( j ) } } { m } - \frac { 1 } { 2 } \right| = \widetilde { \mathcal { O } } ( m ^ { - 1 / 2 } ) ,
$$

so $m ^ { ( j ) } \geq m / 4$ for large n. Conditioning on the family labels, apply Lemma D.4 and Lemma D.6 with radius r and $\beta = \mathrm { m i n } \{ \epsilon , 1 \} / 4$ to the $\it { m } ^ { ( j ) }$ tasks of each family, each with failure probability $1 / ( 6 n )$ . The total failure probability is at most $1 / n$ , and all bounds below hold on this event, uniformly over the ball of radius r.

Relative gradient error. On D, since $\pmb { \Sigma } ^ { ( j ) } \succeq 0$

$$
\operatorname* { m a x } _ { j } \left. \sum _ { k } \bar { w } _ { k j } - 1 \right. ^ { 2 } \leq \frac { 4 } { \sigma _ { \mu } ^ { 2 } } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ) \leq \frac { 4 } { \sigma _ { \mu } ^ { 2 } } \left( \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t _ { \mathrm { c r i t } } ) ) + \frac { \kappa } { 2 n } \right) = \mathcal { O } ( n ^ { - 1 } ) ,
$$

so the prefactor in Lemma D.4 is at most $2 n ^ { - 1 / 2 + \beta }$ . Summing its coordinatewise bound over the fixed number of coordinates and weighting by $m ^ { ( j ) } / m$

$$
\left\| \sum _ { j } \frac { m ^ { ( j ) } } { m } \nabla _ { \theta } \left( \hat { \mathcal { L } } _ { m ^ { ( j ) } } ^ { ( j ) } - \mathcal { L } ^ { ( j ) } \right) \right\| _ { 2 } \leq \widetilde { \mathcal { O } } \left( \frac { n ^ { - 1 / 2 + \beta } } { \sqrt { m } } \right) \sum _ { q , j } \bar { w } _ { q j } ^ { \prime } .
$$

The truncation bounds in the proof of Lemma D.4 also give

$$
\left\| \sum _ { j } \left( \frac { m ^ { ( j ) } } { m } - \frac { 1 } { 2 } \right) \nabla _ { \theta } \mathcal { L } ^ { ( j ) } \right\| _ { 2 } \leq \widetilde { \mathcal { O } } \left( \frac { n ^ { - 1 / 2 + \beta } } { \sqrt { m } } \right) \sum _ { q , j } \bar { w } _ { q j } ^ { \prime } .
$$

By the second estimate of Lemma D.10(a), uniformly over $\theta \in \mathcal { D }$

$$
\frac { \| \nabla _ { \theta } ( \hat { \mathcal { L } } _ { m } - \mathcal { L } ) \| _ { 2 } } { \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 } } \leq C _ { 0 } n \widetilde { \mathcal { O } } \left( \frac { n ^ { - 1 / 2 + \beta } } { \sqrt { m } } \right) = \widetilde { \mathcal { O } } \left( \frac { n ^ { 1 / 2 + \beta } } { \sqrt { m } } \right) \leq \operatorname* { m i n } \left\{ \frac { 1 } { 4 } , \frac { 1 } { \log n } \right\} .
$$

For the last inequality, write the preceding bound as $C ( \log n ) ^ { a } n ^ { 1 / 2 + \beta } / \sqrt { m }$ with C and a fixed. Since $\beta = \mathrm { m i n } \{ \epsilon , 1 \} / 4$ and $m ~ \geq ~ n ^ { 1 + \epsilon }$ , it is at most $\bar { C } ( \log n ) ^ { a } n ^ { - \operatorname* { m i n } \{ \epsilon , 1 \} / 4 }$ , hence at most min $\{ 1 / 4 , 1 / \log { n } \}$ for sufficiently large n.

Entry, parameter bound, and gate accuracy. The empirical flow is $\dot { \theta } = - \nabla _ { \theta } \mathcal { L } - \zeta ( t )$ with $\zeta ( t ) =$ $\nabla _ { \theta } ( \hat { \mathcal { L } } _ { m } - \mathcal { L } ) ( \theta ( t ) )$ , which satisfies $\| \zeta ( t ) \| _ { 2 } \leq \operatorname* { m i n } \{ 1 / 4 , 1 / \log n \} \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 }$ whenever $\theta ( t ) \in$ D. Part (b) of Lemma D.10 applies. Combined with part (a), it gives, before entry,

$$
\begin{array} { r l r } & { } & { \| \nabla _ { \theta } ( \hat { \mathcal { L } } _ { m } - \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ) \| _ { 2 } \leq \displaystyle \frac { 1 } { 2 } \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 } , } \\ & { } & { \displaystyle \frac { d } { d t } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \leq - \frac { 1 } { 2 } \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 } ^ { 2 } , } \\ & { } & { \displaystyle \| \dot { \theta } \| _ { 2 } \leq \displaystyle \frac { 3 } { 2 } \| \nabla _ { \theta } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } \| _ { 2 } . } \end{array}
$$

Set $t _ { \theta } = t _ { \rho } \leq t _ { \mathrm { c r i t } } + 2 C _ { 0 } ^ { 2 } n ^ { 2 } \sigma _ { \mu } ^ { 2 }$ . Part (b) gives loss descent and the stated parameter bound on $[ t _ { \mathrm { c r i t } } , t _ { \theta } ]$ and

$$
\| \theta _ { q } ^ { \perp } ( t ) \| _ { 2 } \leq B + \mathcal { O } ( \sigma _ { \mu } ^ { 2 } ) + \widetilde { \mathcal { O } } ( n ^ { - 1 } ) + \frac { 2 C _ { 0 } } { \log n } \log ( 1 + c n \sigma _ { \mu } ^ { 2 } ) = \mathcal { O } ( 1 ) .
$$

Since $\rho = \mathcal { O } ( n ^ { - 1 } )$ and L<sup>truncated</sup> $( \bar { w } ( t _ { \theta } ) ) \leq \mathcal { L } ^ { 1 }$ truncated $( \bar { w } ( t _ { \mathrm { c r i t } } ) )$ , the exclusion in the proof of Lemma B.10 applies to a point of $s$ within distance $\rho$ of $\bar { w } ( t _ { \theta } )$ . Its columns are $e _ { k _ { j } } ~ \mathrm { o r } ~ \bar { w } _ { \cdot j } ^ { \ast }$ , and the triangle inequality gives the gate bound. Since $\| e _ { k _ { j } } - \bar { w } _ { \cdot j } ^ { * } \| _ { 2 } = \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } / ( n \sigma _ { \mu } ^ { 2 } + \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } )$

$$
\| \bar { w } _ { \cdot j } ( t _ { \theta } ) - e _ { k _ { j } } \| _ { 1 } \leq \sqrt { 2 } \left( \rho + \frac { 2 \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } } { n \sigma _ { \mu } ^ { 2 } + \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } } \right) = \mathcal { O } ( n ^ { - 1 } ) .
$$

Risk. By Lemma F.6, $\mathbb { E } _ { j } [ ( g _ { k _ { i } } - \mu ) ^ { 2 } ] = \mathcal { O } ( n ^ { - 1 } )$ . For the 1-sub-Gaussian centered observations, $\mathbb { E } _ { j } [ \operatorname* { m a x } _ { \ell \in [ n ] } | X _ { \ell } - \mu | ^ { 4 } ] = { \\mathcal { O } } ( \log ^ { 2 } n )$ . Apply Lemma 3.5 with $~ i ~ = ~ k _ { j }$ and ma $\natural _ { k } \left. \theta _ { k } ( t _ { \theta } ) \right. _ { 2 } =$ ${ \mathcal { O } } ( \log n )$

$$
\sqrt { \mathbb { E } _ { j } [ ( \widehat { \mu } - \mu ) ^ { 2 } ] } \leq \mathcal { O } ( n ^ { - 1 / 2 } ) + \mathcal { O } ( n ^ { - 1 } ) \left( \sigma _ { \mu } ^ { 2 } + \mathcal { O } ( \log n ) \right) ^ { 1 / 2 } = \mathcal { O } ( n ^ { - 1 / 2 } ) .
$$

Hence $\mathcal { L } ^ { ( j ) } ( \theta ( t _ { \theta } ) ) = \mathcal { O } ( n ^ { - 1 } )$ for each $j ,$ , and $\mathcal { L } ( \boldsymbol { \theta } ( t _ { \theta } ) ) = \mathcal { O } ( n ^ { - 1 } )$ . Moreover, $\begin{array} { r } { \big | \sum _ { k } \bar { w } _ { k j } ( t _ { \theta } ) - 1 \big | \le } \end{array}$ $\| \bar { w } . _ { j } ( t _ { \theta } ) - e _ { k _ { j } } \| _ { 1 } = \mathcal { O } ( n ^ { - 1 } )$ with a fixed implied constant, and $\| \theta ( t _ { \theta } ) \| _ { 2 } \leq r$ . Hence Lemma D.6 and $m ^ { ( j ) } \geq m / 4$ give

$$
\left| \hat { \mathcal { L } } _ { m ^ { ( j ) } } ^ { ( j ) } ( \theta ( t _ { \theta } ) ) - \mathcal { L } ^ { ( j ) } ( \theta ( t _ { \theta } ) ) \right| = \widetilde { \mathcal { O } } \left( \frac { n ^ { - 1 + \beta } } { \sqrt { m } } + \frac { n ^ { - 1 + 2 \beta } } { m } \right) = \mathcal { O } ( n ^ { - 3 / 2 } ) ,
$$

so $\begin{array} { r } { \hat { \mathcal { L } } _ { m } ( \theta ( t _ { \theta } ) ) \leq \operatorname* { m a x } _ { j } \hat { \mathcal { L } } _ { m ^ { ( j ) } } ^ { ( j ) } ( \theta ( t _ { \theta } ) ) = \mathcal { O } ( n ^ { - 1 } ) } \end{array}$ . All concentration bounds are uniform over the ball of radius r, so they apply at the data-dependent time $t _ { \theta }$ □

Lemma D.11 (Loss barrier near the optimal stationary point). Fix λ and let $\sigma _ { \mu } ^ { 2 } > 0 .$ . Suppose the expert hierarchy of Definition B.1 holds and use Lemma $B . 7$ and the truncated loss from Definition B.6. For each family j, choose the positive threshold

$$
\delta _ { j } = \frac { 1 } { 1 6 n ^ { 2 } } \operatorname* { m i n } \left\{ \sigma _ { \mu } ^ { 2 } \big ( \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } \big ) ^ { 2 } , ~ \bar { w } _ { k _ { j } j } ^ { * } \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } \operatorname* { m i n } _ { k \neq k _ { j } } \big ( \Sigma _ { k k _ { j } } ^ { ( j ) } - \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } \big ) \right\} ,
$$

and let $\delta = \mathrm { m i n } _ { j \in \{ \mathrm { G } , \mathrm { A } \} } \delta _ { j }$

Define the region

$$
N _ { \delta } = \left\{ \bar { w } \in [ 0 , 1 ] ^ { 2 \times 2 } : \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ) - \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { * } ) \leq \delta \right\} .
$$

Every point in $N _ { \delta }$ satisfies,for each $j ,$

$$
\| \bar { w } . _ { j } - \bar { w } _ { \cdot j } ^ { * } \| _ { 2 } \leq \frac { \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } } { n } .
$$

Moreover, for every $\bar { w } \in [ 0 , 1 ] ^ { 2 \times 2 }$ and all sufficiently large $n ,$

$$
\operatorname* { m a x } _ { j } \left[ \sum _ { q } \bar { w } _ { q j } ^ { \prime } + \left| \sum _ { q } \bar { w } _ { q j } - 1 \right| \right] \lesssim n \big [ \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ) - \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { * } ) \big ] + \frac { 1 } { n \sigma _ { \mu } ^ { 2 } } ,\tag{20}
$$

with an implied constant depending only on $\pmb { \Sigma } ^ { ( \mathrm { G } ) }$ and $\pmb { \Sigma } ^ { \mathrm { ( A ) } }$ . The truncated gradientflow

$$
\dot { \bar { w } } _ { q k } = - \frac { 1 } { 2 } \bar { w } _ { q k } ^ { \prime } \sum _ { j } Z _ { j k } \bar { w } _ { q j } ^ { \prime } S _ { q j } ( \bar { w } ) , \qquad Z _ { j k } = \langle \bar { \bf z } _ { j } , \bar { \bf z } _ { k } \rangle ,
$$

cannot exit $N _ { \delta }$ once it enters it.

For the full population gradient flow $\dot { \theta } = - \nabla _ { \theta } \mathcal { L }$ atfixed λ, the same region cannot be exited after time $t _ { c r i t }$ provided

$$
\mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t _ { c r i t } ) ) - \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { * } ) \leq \delta / 4
$$

and,for every θ whose gates w¯ lie in $N _ { \delta }$

$$
\left| \mathcal { L } ( \theta ) - \mathcal { L } ( \theta ( t _ { c r i t } ) ) - \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ) + \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t _ { c r i t } ) ) \right| \leq \delta / 4 .
$$

The same statement holdsfor the empirical gradientflow with L replaced by $\hat { \mathcal { L } } _ { m }$ in both theflow and this bound.

Proof. Since $\mathcal { L } ^ { \mathrm { t r u n c a t e d } }$ is quadratic and $S _ { k _ { j } j } ( \bar { w } ^ { * } ) = 0$ , its expansion at $\bar { w } ^ { * }$ is exact:

$$
\begin{array} { r l } { { 2 } \big ( \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ) - \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { * } ) \big ) = \displaystyle \sum _ { j } \Bigg [ \frac { \overline { { w } } _ { k j } ^ { * } } { n } \displaystyle \sum _ { k \neq k _ { j } } \big ( \Sigma _ { k k _ { j } } ^ { ( j ) } - \Sigma _ { k j k _ { j } } ^ { ( j ) } \big ) \bar { w } _ { k j } } & { } \\ { + \frac { \sigma _ { \mu } ^ { 2 } } { 2 } \left( \displaystyle \sum _ { k = 1 } ^ { 2 } ( \bar { w } _ { k j } - \bar { w } _ { k j } ^ { * } ) \right) ^ { 2 } } & { } \\ { + \frac { 1 } { 2 n } ( \bar { w } _ { \cdot j } - \bar { w } _ { \cdot j } ^ { * } ) ^ { \top } \Sigma ^ { ( j ) } ( \bar { w } _ { \cdot j } - \bar { w } _ { \cdot j } ^ { * } ) \Bigg ] . } \end{array}
$$

Every term on the right is nonnegative by the expert hierarchy and positive semidefiniteness of the covariance matrices.

We first deduce (20). Bound each group of terms in the expansion by $2 [ \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ) ~ -$ $\mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { * } ) ]$ . Since $\bar { w } _ { k _ { i } j } ^ { * } ~ \ge ~ 1 / 2$ for large n and the hierarchy gaps are positive constants, the first group gives $\begin{array} { r } { \sum _ { k \ne k _ { i } } \bar { w } _ { k j } \lesssim n [ \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ) - \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { * } ) ] } \end{array}$ . The second gives

$$
\begin{array} { r l r } & { } & { \displaystyle \left. \sum _ { k } ( \bar { w } _ { k j } - \bar { w } _ { k j } ^ { * } ) \right. \leq 2 \sqrt { [ { \mathcal L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ) - { \mathcal L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { * } ) ] / \sigma _ { \mu } ^ { 2 } } } \\ & { } & { \quad \quad = 2 \sqrt { n [ { \mathcal L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ) - { \mathcal L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { * } ) ] \cdot \frac { 1 } { n \sigma _ { \mu } ^ { 2 } } } } \\ & { } & { \quad \leq n [ { \mathcal L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ) - { \mathcal L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { * } ) ] + \frac { 1 } { n \sigma _ { \mu } ^ { 2 } } } \end{array}
$$

by the arithmetic–geometric mean inequality. As $\begin{array} { r } { \sum _ { k } \bar { w } _ { k j } ^ { * } = \bar { w } _ { k _ { j } j } ^ { * } } \end{array}$ and $1 - \bar { w } _ { k _ { j } j } ^ { * } = \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } / ( n \sigma _ { \mu } ^ { 2 } +$ $\Sigma _ { k _ { i } k _ { i } } ^ { ( j ) } ) \lesssim ( n \sigma _ { \mu } ^ { 2 } ) ^ { - 1 }$ , the triangle inequality bounds $| \Sigma _ { k } \bar { w } _ { k j } - 1 |$ by the right-hand side of (20). Combining the same two bounds controls $| \bar { w } _ { k _ { j } j } - \bar { w } _ { k _ { i } j } ^ { * } |$ and hence $1 - \bar { w } _ { k _ { j } j } , \mathrm { s o } \bar { w } _ { k _ { j } j } ^ { \prime } \le 1 - \bar { w } _ { k _ { j } j }$ and $\bar { w } _ { q j } ^ { \prime } \le \bar { w } _ { q j }$ for $q \neq k _ { j }$ are bounded by the same quantity. This proves (20).

For each family j, the bound $\delta \leq \delta _ { j }$ gives, on $N _ { \delta }$

$$
\sum _ { k \neq k _ { j } } \bar { w } _ { k j } \leq \frac { 2 n \delta } { \bar { w } _ { k _ { j } j } ^ { * } \operatorname* { m i n } _ { k \neq k _ { j } } \bigl ( \Sigma _ { k k _ { j } } ^ { ( j ) } - \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } \bigr ) } \leq \frac { \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } } { 8 n } ,
$$

and

$$
\left| \sum _ { k = 1 } ^ { 2 } ( \bar { w } _ { k j } - \bar { w } _ { k j } ^ { * } ) \right| \leq \sqrt { \frac { 4 \delta } { \sigma _ { \mu } ^ { 2 } } } \leq \frac { \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } } { 2 n } .
$$

Consequently,

$$
\begin{array} { r l r } {  { \| \bar { w } _ { \cdot j } - \bar { w } _ { \cdot j } ^ { * } \| _ { 2 } \leq | \bar { w } _ { k j } \jmath - \bar { w } _ { k j } ^ { * } | + \sum _ { k \neq k _ { j } } \bar { w } _ { k j } } } \\ & { } & { \leq  \displaystyle \sum _ { k = 1 } ^ { 2 } ( \bar { w } _ { k j } - \bar { w } _ { k j } ^ { * } ) | + 2 \sum _ { k \neq k _ { j } } \bar { w } _ { k j } } \\ & { } & { \leq \sqrt { \frac { 4 \delta } { \sigma _ { \mu } ^ { 2 } } } + \frac { 4 n \delta } { \bar { w } _ { k j } ^ { * } \operatorname* { m i n } _ { k \neq k _ { j } } \big ( \Sigma _ { k k _ { j } } ^ { ( j ) } - \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } \big ) } \leq \frac { 3 \Sigma _ { k _ { j } k _ { j } } ^ { ( j ) } } { 4 n } . } \end{array}
$$

The truncated flow preserves $[ 0 , 1 ] ^ { 2 \times 2 }$ because $\bar { w } _ { q k } ^ { \prime } = 0$ on each coordinate boundary. Moreover, $\frac { \partial \mathcal { L } ^ { \mathrm { t r u n c a t e d } } } { \partial \bar { w } _ { q j } } = S _ { q j } / 2$ , so

$$
\frac { d } { d t } \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t ) ) = - \frac { 1 } { 4 } \sum _ { q = 1 } ^ { 2 } \left\| \sum _ { j } S _ { q j } ( \bar { w } ) \bar { w } _ { q j } ^ { \prime } \bar { \mathbf { z } } _ { j } \right\| _ { 2 } ^ { 2 } \leq 0 .
$$

Since λ is fixed, $\mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { * } )$ and $\delta$ are constant. Hence the truncated flow cannot exit $N _ { \delta }$

For the full flow, a first exit would require a time t with $\mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t ) ) - \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ^ { * } ) = \delta$ . The assumed bound would then imply

$$
\mathcal { L } ( \theta ( t ) ) - \mathcal { L } ( \theta ( t _ { \mathrm { c r i t } } ) ) \geq \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t ) ) - \mathcal { L } ^ { \mathrm { t r u n c a t e d } } ( \bar { w } ( t _ { \mathrm { c r i t } } ) ) - \delta / 4 \geq \delta / 2 > 0 ,
$$

contradicting $d \mathcal { L } / d t = - \| \nabla _ { \theta } \mathcal { L } \| _ { 2 } ^ { 2 } \leq 0$ . Replacing L by $\hat { \mathcal { L } } _ { m }$ gives the empirical statement. □

## D.3.6 PROOF OF COROLLARY B.12

Proof of Corollary B.12. Since $g _ { 1 }$ is the sample mean at $\lambda _ { 1 } = 0$ , integrating the Gaussian gradient expansion in Lemma F.1 gives, uniformly for $| \lambda _ { 1 } | \le 2$

$$
\mathbb { E } _ { \mathrm { G } } [ ( g _ { 1 } - \mu ) ^ { 2 } ] = \frac { \cosh ( \lambda _ { 1 } ^ { 2 } ) + \lambda _ { 1 } ^ { 2 } \sinh ( \lambda _ { 1 } ^ { 2 } ) } { n } + \mathcal { O } ( n ^ { - 2 } ) .
$$

At $t _ { \lambda }$ , Lemma F.1, Lemma F.2, and Lemma F.3 give

$$
\mathbb { E } _ { \mathbb { G } } [ ( g _ { 1 } - \mu ) ^ { 2 } ] = \frac { 1 + o ( 1 ) } { n } , \qquad \mathbb { E } _ { \mathbb { A } } [ ( g _ { 2 } - \mu ) ^ { 2 } ] = \left\{ \mathcal { O } ( ( n | \lambda _ { 2 } | ) ^ { - 1 } ) , \quad \mathrm { U n i f o r m } , \atop \mathcal { O } ( \sigma _ { n } ^ { 2 } / n ) ,  \right.
$$

since $| \lambda _ { 1 } ( t _ { \lambda } ) | = o ( 1 )$ and $e ^ { - 4 \sqrt { 1 - \sigma _ { n } ^ { 2 } } | \lambda _ { 2 } | } = \sigma _ { n } ^ { 2 }$ for the mixture. The sample-maximum fourth moment is $\mathcal { O } ( 1 )$ for Uniform data and ${ \mathcal { O } } ( ( \log n ) ^ { \frac { \cdot } { 2 } \cdot }$ ) for Gaussian and Gaussian-mixture data. Apply Lemma 3.5 with $i = 1$ for $j = \mathrm { G }$ and $i = 2 \operatorname { f o r } j = \mathrm { A }$ , and $\| \bar { w } . _ { j } - e _ { i } \| _ { 1 } = \mathcal { O } ( n ^ { - 1 } )$ . Its proof gives $\lVert \widehat { \mu } - g _ { 1 } \rVert _ { L ^ { 2 } ( \mathbb { P } _ { \mathrm { G } } ) } = \mathcal { O } ( \sqrt { \log n } / n ) = o ( n ^ { - 1 / 2 } )$ , so the triangle inequality in both directions preserves the Gaussian leading constant. In the Gaussian-mixture case, this discrepancy is of lower order than the expert risk, since

$$
\frac { n ^ { - 2 } \log n } { \sigma _ { n } ^ { 2 } / n } \leq n ^ { - \epsilon } \log n = o ( 1 ) .
$$

For the bounds at fixed $\mu ,$ the translation identities and the gate bound give

$$
\begin{array} { r l r } & { } & { \exp ( { \mathbf { \sum } _ { i = 1 } ^ { n } { \rho _ { i } \mathbf { R } _ { i } ^ { \rho } } } + { \mathbf { \sum } _ { j = 1 } ^ { n } { \rho _ { i } \mathbf { S } _ { j } } } ) , \quad \mathrm { ~ o s s : ~ } \operatorname { \Re } ( { \mathbf { X } _ { i } ^ { \rho } \times ( \rho _ { i } \mathbf { X } ) - \rho _ { i } \mathbf { X } _ { j } } ) } \\ & { } & { \exp ( { \mathbf { X } _ { i } ^ { \rho } \times ( \rho _ { i } \mathbf { X } ) - \rho _ { i } \mathbf { X } _ { j } } ) , } \\ & { } & { \exp ( { \mathbf { \sum } _ { i = 1 } ^ { n } { \rho _ { i } \mathbf { X } _ { j } - \rho _ { i } \mathbf { X } _ { j } } } ) \Bigg \} \int _ { 0 } ^ { \infty } { \exp ( { \mathbf { X } _ { i } ^ { \rho } \times ( \rho _ { i } \mathbf { X } _ { j } - \rho _ { i } \mathbf { X } _ { j } ) } ) } \cdot } \\ & { } & { \exp ( { \mathbf { \sum } _ { i = 1 } ^ { n } { \rho _ { i } \mathbf { X } _ { j } - \rho _ { i } \mathbf { X } _ { j } } } ) \Bigg \mathrm { \Bigg \downarrow } \langle { \mathbf { S } _ { \mathbf { E } _ { i } } \times \mathbf { X } _ { j } - \rho _ { i } \mathbf { Z } _ { j } ^ { \rho } } } \\ & { } & { \exp ( { \mathbf { \rho } _ { i } \mathbf { X } _ { i } \cdot \mathbf { X } _ { j } } ) - \sigma _ { i } \exp ( { \mathbf { \rho } _ { i } \mathbf { X } _ { j } - \rho _ { i } \mathbf { X } _ { j } } ) } \\ & { } & { \exp ( { \mathbf { X } _ { i } ^ { \rho } \times ( \rho _ { i } \mathbf { X } _ { j } ) } ) } \\ & { } &  \exp (  \mathbf { X } _ { i } ^  \rho  \end{array}
$$

## D.3.7 PROOF OF COROLLARY 3.2

ProofofCorollary 3.2. The population gates $\bar { w } _ { \cdot j }$ depend on the data only through $\bar { \mathbf { z } } _ { j } .$ , which does not depend on the context length. Hence max $\{ \| \bar { w } _ { \cdot \mathrm { G } } - e _ { 1 } \| _ { 1 } , \| \bar { w } _ { \cdot \mathrm { A } } - e _ { 2 } \| _ { 1 } \} = \mathcal { O } ( n ^ { - 1 } )$ and ma $\mathbf { x } _ { q } \parallel \theta _ { q } \parallel _ { 2 } =$ $\bar { \mathcal { O } ( \log n ) } = \mathcal { O } ( \log n ^ { \prime } )$ at the trained parameters, and Lemma $3 . 5$ applies with $n ^ { \prime }$ in place of n. The hypotheses of Lemma F.2 hold at length $n ^ { \prime }$ , since $| \lambda _ { 2 } ( t _ { \lambda } ) | = n ^ { 1 - \epsilon } \overset {  } { \leq } n ^ { \prime 1 - \epsilon }$ , and those of Lemma F.3 hold since $n ^ { \prime - 1 + \epsilon } \leq n ^ { - 1 + \epsilon } \leq \sigma _ { n } ^ { 2 }$ . The proof of Corollary B.12 with $n ^ { \prime }$ test observations then gives the claim, with sample-maximum fourth moment $\mathcal { O } ( \mathrm { 1 } )$ for Uniform data and $\ O ( ( \log n ^ { \prime } ) ^ { 2 } )$ otherwise. 口

## E PROOF OF THE SOFTMAX MOE PART OF LEMMA 3.3

For the proof, write $\vartheta = \theta _ { 1 } - \theta _ { 2 }$ . The softmax MoE weights in (3) satisfy $w _ { 1 } ( { \mathbf { X } } ) = \sigma ( \langle { \mathbf { z } } ( { \mathbf { X } } ) , \vartheta \rangle )$ and $w _ { 2 } ( \mathbf { X } ) = 1 - w _ { 1 } ( \mathbf { X } )$ , with the same identities at $\bar { \mathbf { z } } _ { j }$ . Writing the loss as a function of ϑ, the original population flow satisfies $\dot { \vartheta } = - 2 \nabla _ { \vartheta } \mathcal { L }$ and $\begin{array} { r } { \frac { d } { d t } ( \theta _ { 1 } + \theta _ { 2 } ) = 0 . } \end{array}$ . The same identities hold for the empirical loss and for $\mathcal { L } ^ { \mathrm { s m } }$ . Since λ is fixed in the first stage, $| \lambda _ { k } ( 0 ) | \le 1$ and $0 < | \lambda _ { 1 } ( 0 ) | < | \lambda _ { 2 } ( 0 ) |$ make expert 1 better in family G and expert 2 better in family A by Lemma B.2, Lemma B.3 and Lemma B.4 for sufficiently large n, as in Section B.3. The strict covariance gaps do not depend on n for the Gaussian and uniform families; for the Gaussian mixture, (37) converges as $\sigma _ { n }  0$ to $\mathrm { s e c h } ^ { 2 } ( \lambda _ { i } ) \mathrm { s e c h } ^ { 2 } ( \lambda _ { k } )$ , whose gaps are positive, so they are bounded below independently of n.

Proof of the softmax MoE part of Lemma 3.3. The gate and parameter bounds follow from the empirical part of Lemma E.2 with $\dot { B } = 1$ . Since $\theta _ { 1 } + \theta _ { 2 }$ is constant and bounded at initialization, the bound on ϑ gives max ${ \mathfrak { a } } \| \theta _ { k } ( t ) \| _ { 2 } = { \mathcal { O } } ( \log n )$ throughout training of the router. □

Lemma E.1 (Taylor expansion of the loss). Under the assumptions of Lemma 3.3, for every fixed $C _ { 0 } > 0$ and all sufficiently large n,

$$
| { \mathcal { L } } ( \vartheta ) - { \mathcal { L } } ^ { \mathrm { s m } } ( \vartheta ) | \leq \frac { C ( 1 + \| \vartheta \| _ { 2 } ) ^ { 4 } } { n ^ { 3 / 2 } } ,\tag{21}
$$

$$
\| \nabla _ { \vartheta } \mathcal { L } - \nabla _ { \vartheta } \mathcal { L } ^ { \mathrm { s m } } \| _ { 2 } \leq \frac { C ( 1 + \| \vartheta \| _ { 2 } ) ^ { 4 } } { n ^ { 3 / 2 } } \sum _ { j } \bar { w } _ { 1 j } ^ { \prime } ,\tag{22}
$$

uniformly on $\| \vartheta \| _ { 2 } \leq C _ { 0 }$ log $n ,$ , where $\mathcal { L } ^ { \mathrm { s m } }$ is defined in (8).

Proof. $w _ { 1 } + w _ { 2 } = 1$ gives

$$
{ \widehat { \mu } } - \mu = \sum _ { i = 1 } ^ { 2 } w _ { i } ( g _ { i } - \mu ) .
$$

Write $\Delta \mathbf { z } = \mathbf { z } - \bar { \mathbf { z } } _ { i }$ as in Appendix D. The empirical-moment expansions in Lemma F.6 and Lemma F.1, together with the layer-normalization argument in Appendix D, give

$$
\begin{array} { r l } & { \quad \mathbb { E } _ { j } [ ( g _ { i } - \mu ) ( g _ { k } - \mu ) ] = \frac { \Sigma _ { i k } ^ { ( j ) } } { n } + { \mathcal O } ( n ^ { - 3 / 2 } ) , } \\ & { \quad \quad \quad \quad \quad \mathbb { E } _ { j } [ \| g _ { i } - \mu | ^ { 4 } ] = { \mathcal O } ( n ^ { - 2 } ) , \quad \quad \mathbb { E } _ { j } [ \| \Delta \mathbf { z } \| _ { 2 } ^ { 8 } ] = { \mathcal O } ( n ^ { - 4 } ) , } \\ & { \quad \quad \quad \| \mathbb { E } _ { j } [ ( g _ { i } - \mu ) ( g _ { k } - \mu ) \Delta \mathbf { z } ] \| _ { 2 } \leq \left( \mathbb { E } _ { j } [ \| g _ { i } - \mu | ^ { 4 } ] \mathbb { E } _ { j } [ \| g _ { k } - \mu | ^ { 4 } ] \right) ^ { 1 / 4 } \mathbb { E } _ { j } [ \| \Delta \mathbf { z } \| _ { 2 } ^ { 2 } ] ^ { 1 / 2 } } \\ & { \quad \quad \quad \quad \quad \quad = { \mathcal O } ( n ^ { - 3 / 2 } ) . } \end{array}
$$

The exponential-moment bound in the proof of Lemma 3.5 and Hölder’s inequality give

$$
\begin{array} { r l } & { \quad \mathbb { E } _ { j } \Big [ e ^ { 8 | \langle \vartheta , \Delta \mathbf { z } \rangle | } \Big ] \leq C , } \\ & { \quad \mathbb { E } _ { j } \Big [ | g _ { i } - \mu | | g _ { k } - \mu | \| \Delta \mathbf { z } \| _ { 2 } ^ { 2 } e ^ { 2 | \langle \vartheta , \Delta \mathbf { z } \rangle | } \Big ] \leq \big ( \mathbb { E } _ { j } [ \| g _ { i } - \mu | ^ { 4 } ] \mathbb { E } _ { j } [ | g _ { k } - \mu | ^ { 4 } ] \big ) ^ { 1 / 4 } } \\ & { \quad \quad \quad \quad \quad \cdot \left( \mathbb { E } _ { j } [ \| \Delta \mathbf { z } \| _ { 2 } ^ { 8 } ] \mathbb { E } _ { j } [ e ^ { 8 | \langle \vartheta , \Delta \mathbf { z } \rangle | } ] \right) ^ { 1 / 4 } = \mathcal { O } ( n ^ { - 2 } ) . } \end{array}
$$

Taylor expansion of the gate factors at $\bar { \mathbf { z } } _ { j }$ gives

$$
\begin{array} { r l } & { \quad w _ { i } w _ { k } = \bar { w } _ { i j } \bar { w } _ { k j } + \nabla _ { \mathbf { z } } ( w _ { i } w _ { k } ) | _ { \overline { { \mathbf { z } } } _ { j } } ^ { \top } \Delta \mathbf { z } + \mathcal { O } \big ( ( 1 + \| \vartheta \| _ { 2 } ) ^ { 2 } \| \Delta \mathbf { z } \| _ { 2 } ^ { 2 } \big ) , } \\ & { \quad w _ { i } w _ { 1 } ^ { \prime } \mathbf { z } = \bar { w } _ { i j } \bar { w } _ { 1 j } ^ { \prime } \bar { \mathbf { z } } _ { j } + \bar { w } _ { i j } \bar { w } _ { 1 j } ^ { \prime } \Delta \mathbf { z } + \bar { \mathbf { z } } _ { j } \ \nabla _ { \mathbf { z } } ( w _ { i } w _ { 1 } ^ { \prime } ) | _ { \overline { { \mathbf { z } } } _ { j } } ^ { \top } \Delta \mathbf { z } } \\ & { \qquad + \bar { w } _ { 1 j } ^ { \prime } \mathcal { O } \Big ( ( 1 + \| \vartheta \| _ { 2 } ) ^ { 2 } e ^ { 2 | \langle \vartheta , \Delta \mathbf { z } \rangle | } \| \Delta \mathbf { z } \| _ { 2 } ^ { 2 } \Big ) , } \end{array}
$$

where

$$
\begin{array} { r } { \Big \| \nabla _ { \mathbf { z } } ( w _ { i } w _ { k } ) \vert _ { \bar { \mathbf { z } } _ { j } } \Big \| _ { 2 } \leq C \| \vartheta \| _ { 2 } , \qquad \Big \| \nabla _ { \mathbf { z } } ( w _ { i } w _ { 1 } ^ { \prime } ) \vert _ { \bar { \mathbf { z } } _ { j } } \Big \| _ { 2 } \leq C \bar { w } _ { 1 j } ^ { \prime } \| \vartheta \| _ { 2 } . } \end{array}
$$

Substituting these expansions yields

$$
\begin{array} { r l } & { \nabla _ { \vartheta } \mathcal { L } = \displaystyle \frac { 1 } { 2 } \sum _ { j } \mathbb { E } _ { j } \left[ \left( \displaystyle \sum _ { i = 1 } ^ { 2 } w _ { i } ( g _ { i } - \mu ) \right) ( g _ { 1 } - g _ { 2 } ) w _ { 1 } ^ { \prime } \mathbf { z } \right] } \\ & { \quad \quad = \displaystyle \frac { 1 } { 2 n } \sum _ { j } \bar { w } _ { 1 j } ^ { \prime } \bar { \mathbf { z } } _ { j } \displaystyle \sum _ { i = 1 } ^ { 2 } \bar { w } _ { i j } ( \Sigma _ { i 1 } ^ { ( j ) } - \Sigma _ { i 2 } ^ { ( j ) } ) + \mathcal { O } \left( \frac { ( 1 + \| \vartheta \| _ { 2 } ) ^ { 4 } } { n ^ { 3 / 2 } } \right) \sum _ { j } \bar { w } _ { 1 j } ^ { \prime } } \\ & { \quad \quad = \nabla _ { \vartheta } \mathcal { L } ^ { \mathrm { s m } } ( \vartheta ) + \mathcal { O } \left( \frac { ( 1 + \| \vartheta \| _ { 2 } ) ^ { 4 } } { n ^ { 3 / 2 } } \right) \sum _ { j } \bar { w } _ { 1 j } ^ { \prime } , } \end{array}\tag{23}
$$

$$
\begin{array} { l } { \displaystyle \mathcal { L } ( \vartheta ) = \frac { 1 } { 4 } \sum _ { j } \mathbb { E } _ { j } \left[ \left( \sum _ { i = 1 } ^ { 2 } w _ { i } ( g _ { i } - \mu ) \right) ^ { 2 } \right] } \\ { \displaystyle = \frac { 1 } { 4 n } \sum _ { j } \sum _ { i , k } \bar { w } _ { i j } \bar { w } _ { k j } \Sigma _ { i k } ^ { ( j ) } + \mathcal { O } \left( \frac { ( 1 + \| \vartheta \| _ { 2 } ) ^ { 4 } } { n ^ { 3 / 2 } } \right) } \\ { \displaystyle = \mathcal { L } ^ { \mathrm { s m } } ( \vartheta ) + \mathcal { O } \left( \frac { ( 1 + \| \vartheta \| _ { 2 } ) ^ { 4 } } { n ^ { 3 / 2 } } \right) . } \end{array}\tag{24}
$$

Lemma E.2 (Convergence of the softmax router). Under the assumptions ofLemma $3 . 3 ,$ consider the populationflow $\dot { \vartheta } = - 2 \nabla _ { \vartheta } \mathcal { L }$ with the expert parameters fixed. For every fixed $B > 0$ and all sufficiently large n, there is afinite time $t _ { \theta } = \stackrel { \cdot } { \mathcal { O } } ( n ^ { \mathbf { \hat { B } } + 1 } )$ such that

$$
\operatorname* { m a x } \{ \| \bar { w } _ { \cdot \mathrm { G } } ( t _ { \theta } ) - e _ { 1 } \| _ { 1 } , \| \bar { w } _ { \cdot \mathrm { A } } ( t _ { \theta } ) - e _ { 2 } \| _ { 1 } \} = \mathcal { O } ( n ^ { - B } ) , \qquad \operatorname* { s u p } _ { 0 \leq t \leq t _ { \theta } } \| \vartheta ( t ) \| _ { 2 } = \mathcal { O } ( \log n ) .
$$

The reducedflow $\dot { \vartheta } = - 2 \nabla _ { \vartheta } \mathcal { L } ^ { \mathrm { s m } }$ satisfies $\bar { w } ( t ) \to I _ { 2 }$ as $t \to \infty$ . Under the sample-size conditions of Lemma 3.3, the same finite-time bounds hold for $\dot { \vartheta } = - 2 \nabla _ { \vartheta } \hat { \mathcal { L } } _ { m }$ with probability at least $1 - 1 / n$

Proof. In time $s = t / n ,$ , the reduced gate flow is the flow of Lemma D.8, with time multiplied by four. Its hypotheses hold since $Z _ { \mathrm { G G } } = Z _ { \mathrm { A A } } = 1$ and $0 \leq Z _ { \mathrm { G A } } < 1$ by Lemma F.5, which also bounds $\lambda _ { \mathrm { m i n } } ( Z )$ below by a positive constant independent of n. The reduced convergence argument there, using avoidance and loss monotonicity, gives $\bar { w } ^ { \mathrm { r e d u c e d } } ( s ) \to I _ { 2 }$ and fixed $S , \bar { \kappa } > 0$ such that

$$
n \mathcal { L } ^ { \mathrm { s m } } ( \vartheta ^ { \mathrm { r e d u c e d } } ( S ) ) \leq \operatorname* { m i n } _ { \bar { w } ^ { \mathrm { s u b o p t i m a l } } } n \mathcal { L } ^ { \mathrm { s m } } ( \bar { w } ^ { \mathrm { s u b o p t i m a l } } ) - 3 \kappa ,\tag{25}
$$

where the minimum is over the three incorrect corners and $\vartheta ^ { \mathrm { r e d u c e d } } ( 0 ) = \vartheta ( 0 )$

The rescaled vector field is uniformly Lipschitz on a bounded neighborhood of $\vartheta ^ { \mathrm { r e d u c e d } } ( [ 0 , S ] )$ , so Lemma E.1 and Gronwall’s inequality give

$$
\operatorname* { s u p } _ { 0 \leq s \leq S } \| \vartheta ( n s ) - \vartheta ^ { \mathrm { r e d u c e d } } ( s ) \| _ { 2 } \leq { \frac { C _ { S } } { \sqrt { n } } } .\tag{26}
$$

This prevents exit from that neighborhood for sufficiently large n, gives $\| \vartheta ( n S ) \| _ { 2 } = \mathcal { O } ( 1 )$ , and transfers (25) to

$$
n \mathcal L ^ { \mathrm { s m } } ( \vartheta ( n S ) ) \leq \operatorname* { m i n } _ { \bar { w } ^ { \mathrm { s u b o p t i m a l } } } n \mathcal L ^ { \mathrm { s m } } ( \bar { w } ^ { \mathrm { s u b o p t i m a l } } ) - 2 \kappa .\tag{27}
$$

On this sublevel set, $\bar { w } _ { \mathrm { 1 G } }$ and $\bar { w } _ { \mathrm { 2 A } }$ are bounded below by a positive constant, since the minimum on each face $\bar { w } _ { 1 \mathrm { G } } = 0$ and $\bar { w } _ { \mathrm { 2 A } } = 0$ is attained at an incorrect corner. The strict hierarchy and $Z \succ 0$ therefore give

$$
\| \nabla _ { \boldsymbol { \vartheta } } \mathcal { L } ^ { \mathrm { s m } } \| _ { 2 } \asymp \frac { 1 } { n } \sum _ { j } \bar { w } _ { 1 j } ^ { \prime } \asymp \mathcal { L } ^ { \mathrm { s m } } - \mathcal { L } ^ { \mathrm { s m } } ( I _ { 2 } ) ,\tag{28}
$$

$$
\operatorname* { m a x } \{ \| \bar { w } _ { \cdot \mathrm { G } } - e _ { 1 } \| _ { 1 } , \| \bar { w } _ { \cdot \mathrm { A } } - e _ { 2 } \| _ { 1 } \} \le C n \left[ \mathcal { L } ^ { \mathrm { s m } } - \mathcal { L } ^ { \mathrm { s m } } ( I _ { 2 } ) \right] .\tag{29}
$$

By Lemma E.1, while $\| \vartheta \| _ { 2 } \leq C _ { 0 }$ log n,

$$
\frac { \| \nabla _ { \vartheta } ( \mathcal { L } - \mathcal { L } ^ { \mathrm { s m } } ) \| _ { 2 } } { \| \nabla _ { \vartheta } \mathcal { L } ^ { \mathrm { s m } } \| _ { 2 } } \leq \frac { C ( 1 + \| \vartheta \| _ { 2 } ) ^ { 4 } } { \sqrt { n } } \leq \frac { 1 } { 2 }\tag{30}
$$

for sufficiently large n, hence

$$
\begin{array} { r l r } & { \displaystyle \frac { d } { d t } { \mathcal L } ^ { \mathrm { s m } } \leq - \| \nabla _ { \vartheta } { \mathcal L } ^ { \mathrm { s m } } \| _ { 2 } ^ { 2 } \leq - c \left[ { \mathcal L } ^ { \mathrm { s m } } - { \mathcal L } ^ { \mathrm { s m } } ( I _ { 2 } ) \right] ^ { 2 } , } & \\ & { \displaystyle \| \dot { \vartheta } \| _ { 2 } \leq 3 \| \nabla _ { \vartheta } { \mathcal L } ^ { \mathrm { s m } } \| _ { 2 } . } & \end{array}\tag{31}
$$

These bounds preserve (27) and, by the parameter-travel calculation in Lemma D.10, give

$$
\| \vartheta ( t ) - \vartheta ( n S ) \| _ { 2 } \leq C \log \frac { \mathcal { L } ^ { \mathrm { s m } } ( \vartheta ( n S ) ) - \mathcal { L } ^ { \mathrm { s m } } ( I _ { 2 } ) } { \mathcal { L } ^ { \mathrm { s m } } ( \vartheta ( t ) ) - \mathcal { L } ^ { \mathrm { s m } } ( I _ { 2 } ) } \leq C B \log n + C\tag{32}
$$

until the excess loss first reaches $n ^ { - B - 1 }$ . Choosing $C _ { 0 }$ larger than this coefficient prevents exit, and integrating (31) gives attainment by $t _ { \theta } = n S + \bar { \mathcal { O } } ( n ^ { B + 1 } )$ ; if the target already holds at $n S .$ , take $t _ { \theta } = n S$ . The gate bound follows from (29).

For empirical flow, apply Lemma D.4 with $\beta = \mathrm { m i n } \{ \epsilon , 1 \} / 4$ to the sigmoid parameters $( \vartheta , - \vartheta )$ and subtract the two gradient blocks. Since $\bar { w } _ { 1 j } + \bar { w } _ { 2 j } = 1$ , the normalization-error term vanishes.

Conditioning on the family counts as in Section D.3.5, and using the classwise bound $\| \nabla _ { \vartheta } \mathcal { L } ^ { ( j ) } \| _ { 2 } \le$ $C \bar { w } _ { 1 j } ^ { \prime } / n$ from (23), gives, with probability at least $1 - 1 / n$ , uniformly on $\| \vartheta \| _ { 2 } \le C _ { 0 } \log n$

$$
\| \nabla _ { \vartheta } ( \hat { \mathcal { L } } _ { m } - \mathcal { L } ) \| _ { 2 } \leq \widetilde { \mathcal { O } } \bigg ( \frac { n ^ { - 1 / 2 + \beta } } { \sqrt { m } } \bigg ) \sum _ { j } \bar { w } _ { 1 j } ^ { \prime } .
$$

On [0, S], (26) consequently holds with right-hand side $\mathcal { O } ( n ^ { - 1 / 2 } ) + \widetilde { \mathcal { O } } ( n ^ { 1 / 2 + \beta } / \sqrt { m } ) = o ( 1 )$ , as in Section D.3.5, which preserves (27). On that sublevel set, (28) gives

$$
\frac { \| \nabla _ { \vartheta } ( \hat { \mathcal { L } } _ { m } - \mathcal { L } ^ { \mathrm { s m } } ) \| _ { 2 } } { \| \nabla _ { \vartheta } \mathcal { L } ^ { \mathrm { s m } } \| _ { 2 } } \leq \frac { C ( 1 + \| \vartheta \| _ { 2 } ) ^ { 4 } } { \sqrt { n } } + \widetilde O \bigg ( \frac { n ^ { 1 / 2 + \beta } } { \sqrt { m } } \bigg ) \leq \frac { 1 } { 2 } .
$$

Thus (31) and (32) also hold for $\dot { \vartheta } = - 2 \nabla _ { \vartheta } \hat { \mathcal { L } } _ { m } ,$ , giving the same stopping-time and parameter bounds. □

## F ATTENTION-EXPERT DYNAMICS

## F.1 PROOF OF LEMMA 3.4

ProofofLemma 3.4. Since $g _ { k }$ is even in $\lambda _ { k }$ , it suffices to consider $\lambda _ { 2 } ( t _ { \theta } ) > 0$ . For sufficiently large $n ,$ the uniform event in Lemma F.4 has probability at least $1 - \eta$ . On this event, the vector field points inward at $\lambda _ { 1 } = \pm 2$ and $\lambda _ { 2 }$ increases, so the flow stays in the stated ranges of $\lambda _ { 1 } , \lambda _ { 2 }$ until the first target time. Throughout this interval,

$$
\begin{array} { r } { \dot { \lambda } _ { 2 } \geq \left\{ \begin{array} { l l } { 1 / ( C n \lambda _ { 2 } ^ { 2 } ) , } & { \mathrm { U n i f o r m , } } \\ { e ^ { - 4 \sqrt { 1 - \sigma _ { n } ^ { 2 } } \lambda _ { 2 } } / ( C n ) , } & { \mathrm { G a u s s i a n m i x t u r e . } } \end{array} \right. } \end{array}
$$

Integration of $\lambda _ { 2 } ^ { 2 } d \lambda _ { 2 }$ or $e ^ { 4 \sqrt { 1 - \sigma _ { n } ^ { 2 } } \lambda _ { 2 } } d \lambda _ { 2 }$ , respectively, gives the asserted finite times. The initial loss bound in Lemma F.4 and $d \hat { \mathcal { L } } _ { m } / d t = - \| \dot { \lambda } \| _ { 2 } ^ { 2 }$ give

$$
| \lambda _ { 2 } ( t _ { \lambda } ) - \lambda _ { 2 } ( t _ { \theta } ) | ^ { 2 } \leq ( t _ { \lambda } - t _ { \theta } ) \int _ { t _ { \theta } } ^ { t _ { \lambda } } \| { \dot { \lambda } } ( t ) \| _ { 2 } ^ { 2 } d t \leq { \frac { C ( t _ { \lambda } - t _ { \theta } ) } { n } } .
$$

Both target values diverge, so $( t _ { \lambda } - t _ { \theta } ) / n \to \infty$ . For every fixed $\delta \in ( 0 , 1 )$ , the uniform $\lambda _ { 1 }$ gradient expansion in Lemma F.4 gives $\mathrm { s g n } ( \lambda _ { 1 } ) \partial _ { \lambda _ { 1 } } \hat { \mathcal { L } } _ { m } \geq \delta ^ { 3 } / n$ whenever $\delta \leq | \lambda _ { 1 } | \leq 2$ , for sufficiently large $n .$ Thus $| \lambda _ { 1 } |$ enters $[ 0 , \delta ]$ ] within $2 n / \delta ^ { 3 }$ time after $t _ { \theta }$ and cannot leave before $t _ { \lambda }$ . Since $\delta$ is arbitrary, $| \lambda _ { 1 } ( t _ { \lambda } ) | = \overset { \cdot } { o } ( 1 )$ . □

## F.2 EXPERT-PARAMETER GRADIENTS

Throughout this subsection the data follow Definition 2.1, and in the Gaussian-mixture case we assume

$$
n ^ { - 1 + \epsilon } \leq \sigma _ { n } ^ { 2 } , \qquad \sigma _ { n } ^ { 2 } = o ( 1 )
$$

for a constant $\epsilon > 0$ independent of $n .$ . Fix $C < \infty , p \geq 2$ and, for Uniform data, $\epsilon \in \mathsf { \Gamma } ( 0 , 1 )$ independently of n. In the next three lemmas, the covariance $\Sigma _ { g }$ is evaluated under the indicated distribution. Constants depend on $\epsilon , C , p ,$ and the gradient inequalities hold for sufficiently large n.

Lemma F.1 (Expert moments and normal gradients). For $q \in [ 2 ]$ and $j \in \{ \mathrm { G } , \mathrm { A } \}$ , uniformly for $| \lambda _ { q } | \le 2$

$$
\| g _ { q } - \mu \| _ { L ^ { p } ( \mathbb { P } _ { j } ) } + \| \frac { \partial g _ { q } } { \partial \lambda _ { q } } \| _ { L ^ { p } ( \mathbb { P } _ { j } ) } \leq C _ { p } n ^ { - 1 / 2 } .
$$

Uniformly for $| \lambda _ { 1 } | \le 2 ,$

$$
\mathbb { E } _ { \mathrm { G } } [ ( g _ { 1 } - \mu ) \frac { \partial g _ { 1 } } { \partial \lambda _ { 1 } } ] = \frac { 1 } { 2 n } \frac { d } { d \lambda _ { 1 } } \Sigma _ { g } ( \lambda _ { 1 } , \lambda _ { 1 } ) + \mathcal { O } ( n ^ { - 2 } ) .
$$

At $\lambda _ { 1 } = \pm 2 ,$ , this gives

$$
\operatorname { s g n } ( \lambda _ { 1 } ) \mathbb { E } _ { \mathrm { G } } [ ( g _ { 1 } - \mu ) \frac { \partial g _ { 1 } } { \partial \lambda _ { 1 } } ] \geq C ^ { - 1 } / n .
$$

Proof. With $\xi _ { i } = X _ { i } - \mu$ from Definition 2.1 and the population moment ratios from Lemma F.7, the linear terms for $h ( \mathbf { X } , \lambda ) - \mu - m _ { j } ( \lambda )$ and $\frac { \partial h ( \mathbf { X } , \lambda ) } { \partial \lambda } - \overline { { { m } } } _ { j } ^ { \prime } ( \lambda )$ are the sample averages of

$$
\frac { e ^ { \lambda \xi _ { i } } } { M _ { Y } ( \lambda ) } ( \xi _ { i } - m _ { j } ( \lambda ) ) , \qquad \frac { e ^ { \lambda \xi _ { i } } } { M _ { Y } ( \lambda ) } \big ( ( \xi _ { i } - m _ { j } ( \lambda ) ) ^ { 2 } - m _ { j } ^ { \prime } ( \lambda ) \big ) ,\tag{33}
$$

respectively. The second expression is the derivative of the first with respect to λ. Pairing $\mathfrak { a t } \pm \lambda _ { q }$ cancels the population means, since $m _ { j } ( - \lambda ) = - m _ { j } ( \lambda )$ and $m _ { j } ^ { \prime } ( - \lambda ) = \bar { m } _ { j } ^ { \prime } ( \lambda )$ . The covariance of the paired linear terms is therefore $( 2 n ) ^ { - 1 } d \Sigma _ { g } ( \lambda _ { q } , \lambda _ { q } ) / d \lambda _ { q }$ . For $| \lambda _ { q } | \le 2$ , the moment summands have uniformly bounded fixed-order moments and population denominators bounded below. Expansion of the ratios gives the stated $L ^ { p }$ bounds in both contexts. Outside a fixed neighborhood of the population moments, higher moment bounds and the sample maximum control the contribution.

For Gaussian data, let

$$
V _ { i } = \left( \frac { e ^ { \nu \xi _ { i } } } { C _ { \nu } } , \frac { \xi _ { i } e ^ { \nu \xi _ { i } } } { C _ { \nu } } , \frac { \xi _ { i } ^ { 2 } e ^ { \nu \xi _ { i } } } { C _ { \nu } } \right) _ { \nu = \lambda _ { q } , - \lambda _ { q } } , \qquad \mathbf { V } _ { n } = \frac { 1 } { n } \sum _ { i } V _ { i } , \quad \mu _ { V } = \mathbb { E } _ { j } [ V _ { i } ] ,
$$

where $C _ { \nu } = e ^ { \nu ^ { 2 } / 2 }$ . The normalized population denominators are 1, and all fixed-order moments of $V _ { i }$ are uniformly bounded. Write $f ( V )$ for the product $( g _ { q } - \mu ) \frac { \partial g _ { q } } { \partial \lambda _ { a } }$ expressed in these moments and $\Delta \mathbf { V } _ { n } = \mathbf { V } _ { n } - \mu _ { V }$ . Since $f ( \mu _ { V } ) = 0$ and $\nabla f ( \mu _ { V } ) = 0$ , Taylor expansion gives

$$
f ( \mathbf { V } _ { n } ) = \frac { 1 } { 2 } \Delta \mathbf { V } _ { n } ^ { \top } \nabla ^ { 2 } f ( \mu _ { V } ) \Delta \mathbf { V } _ { n } + \frac { 1 } { 6 } \nabla ^ { 3 } f ( \mu _ { V } ) \Delta \mathbf { V } _ { n } ^ { \otimes 3 } + R _ { 4 , n } .
$$

Independence and centering give

$$
\begin{array} { r l } & { \mathbb { E } _ { j } \left[ \frac { 1 } { 2 } \Delta \mathbf { V } _ { n } ^ { \top } \nabla ^ { 2 } f ( \mu _ { V } ) \Delta \mathbf { V } _ { n } \right] = \displaystyle \frac { 1 } { 2 n } \frac { d } { d \lambda _ { q } } \Sigma _ { g } ( \lambda _ { q } , \lambda _ { q } ) , } \\ & { \quad \quad \quad \quad \mathbb { E } _ { j } [ \Delta \mathbf { V } _ { n } ^ { \otimes 3 } ] = \mathcal { O } ( n ^ { - 2 } ) , \qquad \mathbb { E } _ { j } [ \| \Delta \mathbf { V } _ { n } \| _ { 2 } ^ { 4 } ] = \mathcal { O } ( n ^ { - 2 } ) . } \end{array}
$$

Bounded derivatives near $\mu _ { V }$ give $\mathbb { E } _ { j } [ | R _ { 4 , n } | ] = \mathcal { O } ( n ^ { - 2 } )$ there. Outside that neighborhood, use higher moments and

$$
| g _ { q } - \mu | \leq \operatorname* { m a x } _ { i } | \xi _ { i } | , \qquad | \frac { \partial g _ { q } } { \partial \lambda _ { q } } | \leq C \operatorname* { m a x } _ { i } | \xi _ { i } | ^ { 2 } .
$$

For Gaussian data, the covariance formula gives

$$
\frac { 1 } { 2 n } \frac { d } { d \lambda _ { 1 } } \Sigma _ { g } ( \lambda _ { 1 } , \lambda _ { 1 } ) = \frac { \lambda _ { 1 } } { n } \left[ 2 \sinh ( \lambda _ { 1 } ^ { 2 } ) + \lambda _ { 1 } ^ { 2 } \cosh ( \lambda _ { 1 } ^ { 2 } ) \right] .
$$

The claimed sign at $\lambda _ { 1 } = \pm 2$ follows.

Lemma F.2 (Uniform expert moments and gradients). Uniformlyfor $C ^ { - 1 } \leq \lambda _ { 2 } \leq n ^ { 1 - \epsilon }$

$$
\| g _ { 2 } - \mu \| _ { L ^ { p } ( \mathbb { P } _ { \mathrm { U } } ) } \leq \frac { C _ { p } } { \sqrt { n \lambda _ { 2 } } } ,
$$

$$
\| \frac { \partial g _ { 2 } } { \partial \lambda _ { 2 } } \| _ { L ^ { p } ( \mathbb { P } _ { \mathrm { U } } ) } \le \frac { C _ { p } } { \sqrt { n \lambda _ { 2 } ^ { 3 } } } ,
$$

$$
\mathbb { E } _ { \mathbb { U } } [ \left( g _ { 2 } - \mu \right) \frac { \partial g _ { 2 } } { \partial \lambda _ { 2 } } ] = \frac { 1 } { 2 n } \frac { d } { d \lambda _ { 2 } } \Sigma _ { g } ( \lambda _ { 2 } , \lambda _ { 2 } ) + \mathcal { O } \left( \frac { 1 } { n ^ { 3 / 2 } \lambda _ { 2 } ^ { 3 / 2 } } \right) \leq - \frac { 1 } { C n \lambda _ { 2 } ^ { 2 } } .
$$

Proof. With $\xi _ { i } = X _ { i } - \mu$ as in the proof of Lemma $\mathrm { F . 1 }$

$$
\mathbb { E } _ { \mathrm { U } } [ e ^ { \lambda _ { 2 } ( \xi _ { i } - \sqrt { 3 } ) } ] = \frac { 1 - e ^ { - 2 \sqrt { 3 } \lambda _ { 2 } } } { 2 \sqrt { 3 } \lambda _ { 2 } } = \Theta ( \lambda _ { 2 } ^ { - 1 } ) ,
$$

$$
\begin{array} { r } { \left\| \frac { \frac { 1 } { n } \sum _ { i } e ^ { \lambda _ { 2 } ( \xi _ { i } - \sqrt { 3 } ) } } { \mathbb { E } _ { \mathrm { U } } \left[ e ^ { \lambda _ { 2 } ( \xi _ { i } - \sqrt { 3 } ) } \right] } - 1 \right\| _ { L ^ { p } ( \mathbb { P } _ { \mathrm { U } } ) } \le C _ { p } \sqrt { \lambda _ { 2 } / n } . } \end{array}
$$

The linear terms for the centered ratio and its derivative, given by (33), have $L ^ { p }$ norms $O ( ( n \lambda _ { 2 } ) ^ { - 1 / 2 } )$ and $O ( ( n \lambda _ { 2 } ^ { 3 } ) ^ { - 1 / 2 } )$ , respectively. These follow by integrating the bounded functions $e ^ { - \lambda _ { 2 } u }$ times powers of u over $0 \leq u \leq 2 { \sqrt { 3 } } .$ , and applying the moment bounds for independent averages. Dividing by one plus the relative denominator fluctuation leaves remainders $\mathcal { O } ( n ^ { - 1 } )$ and $\mathcal { O } ( ( n \lambda _ { 2 } ) ^ { - 1 } )$ respectively. For the derivative, the additional squared linear numerator has the latter order as well. The denominator lies outside a fixed relative neighborhood of its expectation with probability at most $C \exp ( - n / ( C \lambda _ { 2 } ) )$ ), by Bernstein’s inequality. Since $n / \lambda _ { 2 } \geq n ^ { \epsilon }$ , this event contributes less than any inverse power of n to the stated moments. The product remainders are bounded by

$$
C \left( \frac { 1 } { n \sqrt { n \lambda _ { 2 } ^ { 3 } } } + \frac { 1 } { n \lambda _ { 2 } \sqrt { n \lambda _ { 2 } } } + \frac { 1 } { n ^ { 2 } \lambda _ { 2 } } \right) \le \frac { C } { n ^ { 3 / 2 } \lambda _ { 2 } ^ { 3 / 2 } } .
$$

The Uniform covariance and its derivative in Lemma F.11 satisfy

$$
- \frac { d } { d \lambda _ { 2 } } \Sigma _ { g } ( \lambda _ { 2 } , \lambda _ { 2 } ) = \frac { \sqrt { 3 } } { 4 \lambda _ { 2 } ^ { 2 } } + \mathcal { O } \left( \lambda _ { 2 } ^ { 2 } e ^ { - 2 \sqrt { 3 } \lambda _ { 2 } } \right) \qquad ( \lambda _ { 2 } \to \infty ) .
$$

Positivity and continuity on $[ C ^ { - 1 } , \infty )$ therefore give

$$
\operatorname* { i n f } _ { \lambda _ { 2 } \geq C ^ { - 1 } } \left( - \lambda _ { 2 } ^ { 2 } \frac { d } { d \lambda _ { 2 } } \Sigma _ { g } ( \lambda _ { 2 } , \lambda _ { 2 } ) \right) > 0 , \qquad \frac { n ^ { - 3 / 2 } \lambda _ { 2 } ^ { - 3 / 2 } } { 1 / ( n \lambda _ { 2 } ^ { 2 } ) } = \sqrt { \lambda _ { 2 } / n } = o ( 1 ) .
$$

Lemma F.3 (Gaussian-mixture expert moments and gradients). Assume $n ^ { - 1 + \epsilon } \leq \sigma _ { n } ^ { 2 }$ and $\sigma _ { n } ^ { 2 } = o ( 1 )$ for a fixed $\epsilon > 0 .$ . Uniformly for $C ^ { - 1 } \leq \lambda _ { 2 } \leq \log ( 1 / \sigma _ { n } ^ { 2 } ) / ( 4 \sqrt { 1 - \sigma _ { n } ^ { 2 } } ) ,$

$$
\mathbb { E } _ { \mathrm { M } } [ ( g _ { 2 } - \mu ) ^ { 2 } ] = \frac { \sum _ { g } ( \lambda _ { 2 } , \lambda _ { 2 } ) } { n } + \mathcal { O } ( n ^ { - 2 } ) ,
$$

$$
\mathbb { E } _ { \mathrm { M } } [ ( g _ { 2 } - \mu ) \frac { \partial g _ { 2 } } { \partial \lambda _ { 2 } } ] = \frac { 1 } { 2 n } \frac { d } { d \lambda _ { 2 } } \Sigma _ { g } ( \lambda _ { 2 } , \lambda _ { 2 } ) + \mathcal { O } ( n ^ { - 2 } ) \leq - \frac { e ^ { - 4 \sqrt { 1 - \sigma _ { n } ^ { 2 } } \lambda _ { 2 } } } { C n } ,
$$

with

$$
\| g _ { 2 } - \mu \| _ { L ^ { p } ( \mathbb { P } _ { \mathrm { M } } ) } + \| \frac { \partial g _ { 2 } } { \partial \lambda _ { 2 } } \| _ { L ^ { p } ( \mathbb { P } _ { \mathrm { M } } ) } \leq C _ { p } ( 1 + \lambda _ { 2 } ) ^ { C } \left( \frac { \sigma _ { n } + e ^ { - 2 \sqrt { 1 - \sigma _ { n } ^ { 2 } } \lambda _ { 2 } } } { \sqrt { n } } + \frac { 1 } { n } \right) .
$$

Proof. Use V<sub>i</sub> from the proof of Lemma F.1, with $j = \mathrm { M }$ and $q = 2$ and $C _ { \nu } = e ^ { \sqrt { 1 - \sigma _ { n } ^ { 2 } } | \nu | + \sigma _ { n } ^ { 2 } \nu ^ { 2 } / 2 }$ For $\xi _ { i } = \sqrt { 1 - \sigma _ { n } ^ { 2 } } S _ { i } + \sigma _ { n } G _ { i } .$

$$
\mathbb { E } _ { \mathrm { M } } \left[ \frac { e ^ { \nu \xi _ { i } } } { C _ { \nu } } \right] = e ^ { - \sqrt { 1 - \sigma _ { n } ^ { 2 } } \vert \nu \vert } \cosh ( \sqrt { 1 - \sigma _ { n } ^ { 2 } } \nu ) \in [ 1 / 2 , 1 ] ,
$$

$$
\mathbb { E } _ { \mathbb { M } } \left[ \left( \frac { e ^ { \nu \xi _ { i } } } { C _ { \nu } } \right) ^ { p } \right] = e ^ { - p \sqrt { 1 - \sigma _ { n } ^ { 2 } } | \nu | } \cosh ( p \sqrt { 1 - \sigma _ { n } ^ { 2 } } \nu ) e ^ { ( p ^ { 2 } - p ) \sigma _ { n } ^ { 2 } \nu ^ { 2 } / 2 } \leq e ^ { ( p ^ { 2 } - p ) \sigma _ { n } ^ { 2 } \nu ^ { 2 } / 2 } .
$$

Including the polynomial factors by Gaussian tilting preserves boundedness because $\sigma _ { n } \lambda _ { 2 } =$ $\mathcal { O } ( \sigma _ { n } \log ( 1 / \sigma _ { n } ^ { 2 } ) ) = o ( 1 )$ . The Taylor expansion in the proof of Lemma F.1, applied to $( g _ { 2 } - \mu ) { \frac { \partial g _ { 2 } } { \partial \lambda _ { 2 } } }$ and $( g _ { 2 } - \mu ) ^ { 2 }$ , gives the two covariance expansions.

For the Gaussian mixture, (37) gives

$$
\begin{array} { c } { { \Sigma _ { g } ( \lambda _ { 2 } , \lambda _ { 2 } ) = \sigma _ { n } ^ { 2 } + ( 1 - \sigma _ { n } ^ { 2 } ) \mathrm { s e c h } ^ { 4 } ( \sqrt { 1 - \sigma _ { n } ^ { 2 } } \lambda _ { 2 } ) + { \mathcal O } \big ( \sigma _ { n } ^ { 4 } ( 1 + \lambda _ { 2 } ) ^ { 4 } \big ) , } } \\ { { { \frac { d } { d \lambda _ { 2 } } } \Sigma _ { g } ( \lambda _ { 2 } , \lambda _ { 2 } ) = - 4 ( 1 - \sigma _ { n } ^ { 2 } ) ^ { 3 / 2 } \mathrm { s e c h } ^ { 4 } ( \sqrt { 1 - \sigma _ { n } ^ { 2 } } \lambda _ { 2 } ) \mathrm { t a n h } ( \sqrt { 1 - \sigma _ { n } ^ { 2 } } \lambda _ { 2 } ) + { \mathcal O } \big ( \sigma _ { n } ^ { 4 } ( 1 + \lambda _ { 2 } ) ^ { 5 } \big ) . } } \end{array}
$$

Since $e ^ { - 4 \sqrt { 1 - \sigma _ { n } ^ { 2 } } \lambda _ { 2 } } \geq \sigma _ { n } ^ { 2 }$

$$
\begin{array} { r l r } {  { \frac { \sigma _ { n } ^ { 4 } ( 1 + \lambda _ { 2 } ) ^ { 5 } } { e ^ { - 4 \sqrt { 1 - \sigma _ { n } ^ { 2 } } \lambda _ { 2 } } } \le \sigma _ { n } ^ { 2 } ( 1 + \lambda _ { 2 } ) ^ { 5 } } } \\ & { } & { \le C \sigma _ { n } ^ { 2 } ( 1 + \log ( 1 / \sigma _ { n } ^ { 2 } ) ) ^ { 5 } = o ( 1 ) , } \end{array}
$$

$$
\frac { n ^ { - 2 } } { e ^ { - 4 \sqrt { 1 - \sigma _ { n } ^ { 2 } } \lambda _ { 2 } } / n } \le ( n \sigma _ { n } ^ { 2 } ) ^ { - 1 } \le n ^ { - \epsilon } = o ( 1 ) .
$$

For the refined moment bounds, substitute $\xi _ { i } = \sqrt { 1 - \sigma _ { n } ^ { 2 } } S _ { i } + \sigma _ { n } G _ { i }$ in the linear terms of (33). At $\sigma _ { n } = 0$ their paired values are bounded by $C e ^ { - 2 \sqrt { 1 - \sigma _ { n } ^ { 2 } } \lambda _ { 2 } }$ . The difference for $\sigma _ { n } > 0$ has $L ^ { p }$ norm at most $C _ { p } ( \bar { 1 + \lambda _ { 2 } } ) ^ { C } \sigma _ { n }$ , by Gaussian moments and $\sigma _ { n } \lambda _ { 2 } = o ( 1 )$ . Averaging the centered linear terms gives the factor $n ^ { - 1 / 2 }$ , and the ratio remainders are $\mathcal { O } ( n ^ { - 1 } )$ in $L ^ { p }$ □

Lemma F.4 (Empirical gradients with fixed gating parameters). Assume the data model $o f D e f i n i -$ tion 2.1 andfix $\bar { C } < \infty$ and $\eta \in ( 0 , 1 )$ independently ofn. For Uniform data, also fix $\epsilon > 0 .$ . For Gaussian-mixture data, $\hbar x \epsilon > 0$ and assume n $- 1 + \epsilon \le \sigma _ { n } ^ { 2 }$ and $\sigma _ { n } ^ { 2 } = { \overset { . } { o } } ( 1 )$ . Consider $( \theta , \lambda )$ satisfying

$$
\operatorname* { m a x } \{ \| \bar { w } . _ { \mathrm { G } } - e _ { 1 } \| _ { 1 } , \| \bar { w } . _ { \mathrm { A } } - e _ { 2 } \| _ { 1 } \} \le C / n , \qquad \operatorname* { m a x } _ { a } \| \theta _ { q } \| _ { 2 } \le C \log n , \qquad | \lambda _ { 1 } | \le 2 ,
$$

with $C ^ { - 1 } \leq \lambda _ { 2 } \leq n ^ { 1 - \epsilon }$ for Uniform data and $C ^ { - 1 } \leq \lambda _ { 2 } \leq \log ( 1 / \sigma _ { n } ^ { 2 } ) / ( 4 \sqrt { 1 - \sigma _ { n } ^ { 2 } } )$ for the mixture. For $m = \widetilde \Omega ( n ^ { 1 + \epsilon } )$ independent training tasks and sufficiently large n, with probability at least $1 - \eta ,$ uniformly over this class,

$$
\frac { \partial \hat { \mathcal { L } } _ { m } } { \partial \lambda _ { 2 } } \leq \left\{ \begin{array} { l l } { - 1 / ( C n \lambda _ { 2 } ^ { 2 } ) , } & { U n i f o r m , } \\ { - e ^ { - 4 \sqrt { 1 - \sigma _ { n } ^ { 2 } } \lambda _ { 2 } } / ( C n ) , } & { G a u s s i a n m i x t u r e . } \end{array} \right.
$$

On the same event, uniformly over all (θ, λ) satisfying the bounds above,

$$
\frac { \partial \hat { \mathcal { L } } _ { m } } { \partial \lambda _ { 1 } } = \frac { \lambda _ { 1 } } { 2 n } \left[ 2 \sinh ( \lambda _ { 1 } ^ { 2 } ) + \lambda _ { 1 } ^ { 2 } \cosh ( \lambda _ { 1 } ^ { 2 } ) \right] + \mathcal { O } ( n ^ { - 3 / 2 + o ( 1 ) } ) .
$$

On the same event, $\hat { \mathcal { L } } _ { m } ( \theta , \lambda ) \leq C / n$ whenever also $\lambda _ { 2 } \leq 1$ . Constants may depend on $\epsilon , C , d ,$ and $\omega .$

Proof. For every fixed $p \geq 2$ , the feature moment bounds and the sigmoid inequalities in the proof of Lemma 3.5 give

$$
\begin{array} { r l } { \| 1 - w _ { 1 } \| _ { L ^ { p } ( \mathbb { P } _ { \mathrm { G } } ) } + \| w _ { 2 } \| _ { L ^ { p } ( \mathbb { P } _ { \mathrm { G } } ) } + \| 1 - w _ { 2 } \| _ { L ^ { p } ( \mathbb { P } _ { \mathrm { A } } ) } + \| w _ { 1 } \| _ { L ^ { p } ( \mathbb { P } _ { \mathrm { A } } ) } \leq C _ { p } / n , } & { } \\ { \| w _ { k } - \bar { w } _ { k j } \| _ { L ^ { p } ( \mathbb { P } _ { j } ) } \leq C _ { p } \log n / n ^ { 3 / 2 } . } & { } \end{array}
$$

For $q , k \in [ 2 ]$ , the fixed ω give

$$
\frac { \partial w _ { k } ( \mathbf { X } ) } { \partial \lambda _ { q } } = w _ { k } ( \mathbf { X } ) ( 1 - w _ { k } ( \mathbf { X } ) ) \left. \theta _ { k } , \frac { \partial \mathbf { z } ( \mathbf { X } ) } { \partial \lambda _ { q } } \right. = 0 ,
$$

$$
\frac { \partial h ( { \bf X } , \lambda ) } { \partial \lambda } = \frac { \sum _ { i } X _ { i } ^ { 2 } e ^ { \lambda X _ { i } } } { \sum _ { i } e ^ { \lambda X _ { i } } } - h ( { \bf X } , \lambda ) ^ { 2 } ,
$$

$$
\frac { \partial g _ { q } ( \mathbf { X } ) } { \partial \lambda _ { q } } = \frac { 1 } { 2 } \left( \left. \frac { \partial h ( \mathbf { X } , \lambda ) } { \partial \lambda } \right| _ { \lambda = \lambda _ { q } } - \left. \frac { \partial h ( \mathbf { X } , \lambda ) } { \partial \lambda } \right| _ { \lambda = - \lambda _ { q } } \right) ,
$$

$$
\frac { \partial \widehat { \mu } ( \mathbf { X } ) } { \partial \lambda _ { q } } = w _ { q } ( \mathbf { X } ) \frac { \partial g _ { q } ( \mathbf { X } ) } { \partial \lambda _ { q } } .
$$

Consequently,

$$
\begin{array} { l } { \displaystyle \frac { \partial } { \partial \lambda _ { q } } \frac { 1 } { 2 } ( \widehat { \mu } - \mu ) ^ { 2 } = ( \widehat { \mu } - \mu ) w _ { q } \frac { \partial g _ { q } } { \partial \lambda _ { q } } } \\ { \displaystyle = \left[ \sum _ { k } w _ { k } ( g _ { k } - \mu ) + \mu \left( \sum _ { k } w _ { k } - 1 \right) \right] w _ { q } \frac { \partial g _ { q } } { \partial \lambda _ { q } } , } \end{array}
$$

$$
\frac { \partial \hat { \mathcal { L } } _ { m } } { \partial \lambda _ { q } } = \frac { 1 } { m } \sum _ { b = 1 } ^ { m } ( \widehat { \mu } ( \mathbf { X } _ { b } ) - \mu _ { b } ) w _ { q } ( \mathbf { X } _ { b } ) \frac { \partial g _ { q } ( \mathbf { X } _ { b } ) } { \partial \lambda _ { q } } .
$$

Since the gates and centered experts are independent of $\mu .$

$$
\mathbb { E } _ { j } \left[ \mu \left( \sum _ { k } w _ { k } - 1 \right) w _ { q } \frac { \partial g _ { q } } { \partial \lambda _ { q } } \right] = \mathbb { E } [ \mu ] \mathbb { E } _ { j } \left[ \left( \sum _ { k } w _ { k } - 1 \right) w _ { q } \frac { \partial g _ { q } } { \partial \lambda _ { q } } \right] = 0 ,
$$

$$
\frac { \partial \mathcal { L } } { \partial \lambda _ { q } } = \frac { 1 } { 2 } \sum _ { j } \sum _ { k } \mathbb { E } _ { j } \left[ w _ { q } w _ { k } ( g _ { k } - \mu ) \frac { \partial g _ { q } } { \partial \lambda _ { q } } \right] .
$$

For $\lambda > 0$ and $u \geq 0$

$$
\begin{array} { r l } { \displaystyle } & { \displaystyle \sum _ { i \colon \operatorname* { m a x } _ { \ell } X _ { \ell } - X _ { i } \geq u } \frac { e ^ { \lambda X _ { i } } } { \sum _ { \ell } e ^ { \lambda X _ { \ell } } } \leq \operatorname* { m i n } \{ 1 , n e ^ { - \lambda u } \} , } \\ { \displaystyle } & { \displaystyle 0 \leq \frac { \partial h ( { \mathbf X } , \lambda ) } { \partial \lambda } = \sum _ { i } \frac { e ^ { \lambda X _ { i } } } { \sum _ { \ell } e ^ { \lambda X _ { \ell } } } ( X _ { i } - h ( { \mathbf X } , \lambda ) ) ^ { 2 } } \\ & { \displaystyle \leq \sum _ { i } \frac { e ^ { \lambda X _ { i } } } { \sum _ { \ell } e ^ { \lambda X _ { \ell } } } ( \operatorname* { m a x } _ { \ell } x _ { \ell } - X _ { i } ) ^ { 2 } } \\ & { \displaystyle \leq \int _ { 0 } ^ { \infty } 2 u \operatorname* { m i n } \{ 1 , n e ^ { - \lambda u } \} d u \leq \frac { C ( 1 + \log n ) ^ { 2 } } { \lambda ^ { 2 } } . } \end{array}
$$

Applying the same bound $\mathbf { t o } - \mathbf { X }$ gives

$$
\left| \frac { \partial g _ { q } ( \mathbf { X } ) } { \partial \lambda _ { q } } \right| \leq \frac { C ( 1 + \log n ) ^ { 2 } } { \lambda _ { q } ^ { 2 } } .
$$

For Uniform data, the gate moment bounds, Hölder’s inequality, and Lemma F.2 give

$$
\begin{array} { r l } & { \left| \mathbb { E } _ { \mathrm { U } } \left[ \displaystyle \frac { \partial } { \partial \lambda _ { 2 } } \frac { 1 } { 2 } ( \widehat { \mu } - \mu ) ^ { 2 } \right] - \mathbb { E } _ { \mathrm { U } } \left[ ( g _ { 2 } - \mu ) \displaystyle \frac { \partial g _ { 2 } } { \partial \lambda _ { 2 } } \right] \right| } \\ & { \quad = \left| \mathbb { E } _ { \mathrm { U } } \left[ \left( ( w _ { 2 } ^ { 2 } - 1 ) ( g _ { 2 } - \mu ) + w _ { 1 } w _ { 2 } ( g _ { 1 } - \mu ) \right) \displaystyle \frac { \partial g _ { 2 } } { \partial \lambda _ { 2 } } \right] \right| } \\ & { \quad \le \displaystyle \frac { C } { n } \left( \displaystyle \frac { 1 } { \sqrt { n \lambda _ { 2 } } } \displaystyle \frac { 1 } { \sqrt { n \lambda _ { 2 } ^ { 3 } } } + \displaystyle \frac { 1 } { \sqrt { n } } \displaystyle \frac { 1 } { \sqrt { n \lambda _ { 2 } ^ { 3 } } } \right) = C \left( \displaystyle \frac { 1 } { n ^ { 2 } \lambda _ { 2 } ^ { 2 } } + \displaystyle \frac { 1 } { n ^ { 2 } \lambda _ { 2 } ^ { 3 / 2 } } \right) . } \end{array}
$$

In family G, the same bounds give

$$
\left| \sum _ { k } \mathbb { E } _ { \mathrm { G } } \left[ w _ { 2 } w _ { k } ( g _ { k } - \mu ) \frac { \partial g _ { 2 } } { \partial \lambda _ { 2 } } \right] \right| \leq \frac { C ( 1 + \log n ) ^ { C } } { n ^ { 3 / 2 } \lambda _ { 2 } ^ { 2 } } .
$$

Consequently,

$$
\frac { \partial \mathcal { L } } { \partial \lambda _ { 2 } } \le - \frac { 1 } { C n \lambda _ { 2 } ^ { 2 } } .
$$

For the Gaussian mixture, Lemma F.3 gives

$$
\left| \mathbb { E } _ { \mathrm { M } } \left[ \left( ( w _ { 2 } ^ { 2 } - 1 ) ( g _ { 2 } - \mu ) + w _ { 1 } w _ { 2 } ( g _ { 1 } - \mu ) \right) \frac { \partial g _ { 2 } } { \partial \lambda _ { 2 } } \right] \right| \leq \frac { C ( 1 + \log n ) ^ { C } } { n ^ { 2 } } .
$$

In family G, symmetry gives

$$
\mathbb { E } _ { \mathrm { G } } [ g _ { 1 } - \mu ] = \mathbb { E } _ { \mathrm { G } } \left[ \frac { \partial g _ { 2 } } { \partial \lambda _ { 2 } } \right] = 0 .
$$

Integration by parts and Cauchy–Schwarz give

$$
\left| \mathbb { E } _ { \mathrm { G } } \left[ ( g _ { 1 } - \mu ) \frac { \partial g _ { 2 } } { \partial \lambda _ { 2 } } \right] \right| \leq \left\| \operatorname* { m a x } _ { i } \left| \frac { \partial g _ { 1 } } { \partial X _ { i } } \right| \right\| _ { L ^ { 2 } ( \mathbb { P } _ { \mathrm { G } } ) } \left\| \sum _ { i } \left| \frac { \partial ^ { 2 } g _ { 2 } } { \partial X _ { i } \partial \lambda _ { 2 } } \right| \right\| _ { L ^ { 2 } ( \mathbb { P } _ { \mathrm { G } } ) } .
$$

Differentiating

$$
{ \frac { \partial h ( \mathbf { X } , \lambda ) } { \partial X _ { i } } } = { \frac { e ^ { \lambda X _ { i } } } { \sum _ { \ell } e ^ { \lambda X _ { \ell } } } } { \big ( } 1 + \lambda ( X _ { i } - h ( \mathbf { X } , \lambda ) ) { \big ) }
$$

gives

$$
\left\| \operatorname* { m a x } _ { i } \left| \frac { \partial g _ { 1 } } { \partial X _ { i } } \right| \right\| _ { L ^ { 2 } ( \mathbb { P } _ { \mathrm { G } } ) } \leq \frac { 1 } { n } \left\| C ( 1 + \operatorname* { m a x } _ { i } | X _ { i } - \mu | ) e ^ { C \operatorname* { m a x } _ { i } | X _ { i } - \mu | } \right\| _ { L ^ { 2 } ( \mathbb { P } _ { \mathrm { G } } ) } = n ^ { - 1 + o ( 1 ) } ,
$$

$$
\left\| \sum _ { i } \left| { \frac { \partial ^ { 2 } g _ { 2 } } { \partial X _ { i } \partial \lambda _ { 2 } } } \right| \right\| _ { L ^ { 2 } ( \mathbb { P } _ { \mathbf { G } } ) } \leq C ( 1 + \lambda _ { 2 } ) \left\| ( 1 + \operatorname* { m a x } _ { i } | X _ { i } - \mu | ) ^ { 2 } \right\| _ { L ^ { 2 } ( \mathbb { P } _ { \mathbf { G } } ) } \leq C ( 1 + \log n ) ^ { 2 } .
$$

Using the gate moment bounds,

$$
\begin{array} { r l } & { \left| \displaystyle \sum _ { k } \mathbb { E } _ { \mathbb { G } } \left[ w _ { 2 } w _ { k } ( g _ { k } - \mu ) \frac { \partial g _ { 2 } } { \partial \lambda _ { 2 } } \right] \right| \leq \displaystyle \frac { C } { n } \left| \mathbb { E } _ { \mathbb { G } } \left[ ( g _ { 1 } - \mu ) \frac { \partial g _ { 2 } } { \partial \lambda _ { 2 } } \right] \right| + \frac { C \log n } { n ^ { 3 / 2 } } n ^ { - 1 / 2 + o ( 1 ) } + n ^ { - 2 + o ( 1 ) } } \\ & { \qquad \leq n ^ { - 2 + o ( 1 ) } . } \end{array}
$$

It follows that

$$
\frac { \partial \mathcal { L } } { \partial \lambda _ { 2 } } \leq - \frac { e ^ { - 4 \sqrt { 1 - \sigma _ { n } ^ { 2 } } \lambda _ { 2 } } } { C n } + n ^ { - 2 + o ( 1 ) } \leq - \frac { e ^ { - 4 \sqrt { 1 - \sigma _ { n } ^ { 2 } } \lambda _ { 2 } } } { 2 C n } .
$$

The same estimates, including the mean term, give

$$
\left\| \frac { \partial } { \partial \lambda _ { 2 } } \frac { 1 } { 2 } ( \widehat { \mu } - \mu ) ^ { 2 } \right\| _ { L ^ { p } ( \mathbb { P } _ { j } ) } \leq \left\{ \begin{array} { l l } { \displaystyle \frac { C _ { p } ( 1 + \log n ) ^ { C } } { n \lambda _ { 2 } ^ { 2 } } , } & { \mathrm { U n i f o r m } , j \in \{ \mathrm { G } , \mathrm { U } \} , } \\ { \displaystyle n ^ { - 1 + o ( 1 ) } ( \sigma _ { n } ^ { 2 } + e ^ { - 4 \sqrt { 1 - \sigma _ { n } ^ { 2 } } \lambda _ { 2 } } ) , } & { \mathrm { G a u s s i a n ~ m i x t u r e } , j = \mathrm { M } , } \\ { \displaystyle n ^ { - 3 / 2 + o ( 1 ) } , } & { \mathrm { G a u s s i a n ~ m i x t u r e } , j = \mathrm { G } . } \end{array} \right.
$$

For $m = \widetilde { \Omega } ( n ^ { 1 + \epsilon } )$ independent tasks and sufficiently large $n ,$ the moment inequality for centered independent sums and the preceding bounds give, for every fixed $\beta > 0$ , uniformly on any fixed polynomial-size net in $( \theta , \lambda )$ , with probability at least $1 - \eta / 4$

$$
\left| { \frac { \partial { \hat { \mathcal { L } } } _ { m } } { \partial \lambda _ { 2 } } } - { \frac { \partial { \mathcal { L } } } { \partial \lambda _ { 2 } } } \right| \leq { \frac { 1 } { \sqrt { m } } } \left\{ { \frac { n ^ { \beta } } { n \lambda _ { 2 } ^ { 2 } } } , \right. \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \mathrm { U n i f o r m } , \quad \quad \quad
$$

For a net with at most $n ^ { C }$ points, choose $p > 2 ( C + 1 ) / \beta$ . Markov’s inequality and a union bound give

P (the preceding bound fails at some net point $) \leq C _ { p } n ^ { C } n ^ { - p \beta / 2 } \leq C _ { p } / n$

Since $\| \mathbf { z } \| _ { 2 } = 1$ and z is independent of λ,

$$
\operatorname* { s u p } _ { \theta , \lambda } \left\| \nabla _ { ( \theta , \lambda ) } \left[ ( { \widehat { \mu } } - \mu ) w _ { q } { \frac { \partial g _ { q } } { \partial \lambda _ { q } } } \right] \right\| _ { 2 } \leq C ( 1 + | \mu | + \operatorname* { m a x } _ { i } | X _ { i } - \mu | ) ^ { 4 } .
$$

The expectation of the right-hand side is $\mathcal { O } ( ( 1 + \log n ) ^ { 2 } )$ , so Markov’s inequality gives, for sufficiently large n, with probability at least $1 - \eta / 4$

$$
\frac { 1 } { m } \sum _ { b = 1 } ^ { m } ( 1 + | \mu _ { b } | + \operatorname* { m a x } _ { \ell \in [ n ] } | \mathbf { X } _ { b \ell } - \mu _ { b } | ) ^ { 4 } \leq n .
$$

$( \theta , \lambda )$ has $2 d + 2$ coordinates with ranges of polynomial length. A net of sufficiently fine inversepolynomial mesh therefore extends the gradient inequalities to the full stated domain. Choose $\bar { \beta } < \operatorname* { m i n } \{ 1 / 4 , \epsilon / 2 \}$ . Relative to the respective population gradient lower bounds, the empirical errors are at most

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \displaystyle C n ^ { - 1 / 2 - \epsilon / 2 + \beta } , } & { \mathrm { U n i f o r m , } } \\ { \displaystyle C \left( n ^ { - 1 / 2 - \epsilon / 2 + \beta } + n ^ { - 3 \epsilon / 2 + \beta } \right) , } & { \mathrm { G a u s s i a n m i x t u r e , } } \end{array} \right. } \end{array}
$$

which tend to zero.

Uniformly for $| \lambda _ { 1 } | \leq 2 ,$ , the gate moment bounds, Lemma F.1, and $| g _ { 2 } - \mu | \leq \operatorname* { m a x } _ { i } | X _ { i } - \mu |$ give

$$
\left\| ( \widehat { \mu } - \mu ) w _ { 1 } \frac { \partial g _ { 1 } } { \partial \lambda _ { 1 } } \right\| _ { L ^ { p } ( \mathbb { P } _ { j } ) } \leq C _ { p } / n .
$$

Hölder’s inequality bounds the population error, while Markov’s inequality and a union bound on the parameter net above bound the empirical error at the net points. Extending between net points using the derivative bound above gives, for sufficiently large n, with probability at least $1 - \eta / 4$ , uniformly over all $( \theta , \lambda )$ satisfying the bounds in Lemma $\mathrm { F } . 4$

$$
\left| \frac { \partial \mathcal { L } } { \partial \lambda _ { 1 } } - \frac { 1 } { 2 } \mathbb { E } _ { \mathrm { G } } \left[ ( g _ { 1 } - \mu ) \frac { \partial g _ { 1 } } { \partial \lambda _ { 1 } } \right] \right| + \left| \frac { \partial \hat { \mathcal { L } } _ { m } } { \partial \lambda _ { 1 } } - \frac { \partial \mathcal { L } } { \partial \lambda _ { 1 } } \right| \leq n ^ { - 3 / 2 + o ( 1 ) } .
$$

The Gaussian gradient formula in Lemma F.1 proves the expansion. When also $\lambda _ { 2 } \leq 1$ , the boundedparameter expert moments and the gate moment bounds give $\| ( \widehat \mu - \mu ) ^ { 2 } \| _ { L ^ { p } ( \mathbb { P } _ { j } ) } \leq C _ { p } / n$ . Applying the same net argument to the loss gives ${ \hat { \mathcal { L } } } _ { m } \leq C / n$ uniformly over these parameters, for sufficiently large n, with probability at least $1 - \eta / 4$ . A union bound gives all conclusions jointly with probability at least $1 - \eta$ □

## F.3 GEOMETRY OF z

Lemma F.5 (Geometry of z). Let d $\geq 3$ and sample ω as in Definition 2.2. Recall $m _ { j } , { \bar { \phi } } _ { j }$ , and $\bar { \mathbf { z } } _ { j }$ from Section 3.1, with $m _ { j } ( 0 ) = 0$ by continuity, and let $Z _ { j k } = \langle \bar { \bf z } _ { j } , \bar { \bf z } _ { k } \rangle$ . Explicitly,

$$
\begin{array} { l l } { { m _ { \mathrm { G } } ( a ) = a , } } & { { } } \\ { { m _ { \mathrm { A } } ( a ) = \sqrt { 3 } \coth ( \sqrt { 3 } a ) - 1 / a } } & { { \qquad ( u n i f o r m a l t e r n a t i v e ) , } } \\ { { m _ { \mathrm { A } } ( a ) = \sigma _ { n } ^ { 2 } a + \sqrt { 1 - \sigma _ { n } ^ { 2 } } \operatorname { t a n h } \left( \sqrt { 1 - \sigma _ { n } ^ { 2 } } a \right) \qquad } } & { { \mathrm { ( } G a u s s i a n - m i x t u r e ~ a l t e r n a t i v e ) . } } \end{array}
$$

Then, almost surely over ω,

$$
\| P _ { d } \bar { \phi } _ { j } \| _ { 2 } > 0 , \qquad Z _ { \mathrm { G G } } = Z _ { \mathrm { A A } } = 1 , \qquad 0 < Z _ { \mathrm { G A } } < 1 .
$$

In particular $Z \succ 0 , Z _ { \mathrm { G A } } \geq 0 ,$ , and $Z _ { \mathrm { G A } } < \operatorname* { m i n } \left\{ Z _ { \mathrm { G G } } , Z _ { \mathrm { A A } } \right\}$ . For the uniform alternative these quantities do not depend on n. For the Gaussian-mixture alternative they depend on n only through $\sigma _ { n } ,$ , and $i f \sigma _ { n } \to 0$ then, almost surely, $Z _ { \mathrm { G A } }$ converges to a limit strictly below one. Consequently, $\lambda _ { \mathrm { m i n } } ( Z )$ is bounded below by a positive constant independent of n, for all sufficiently large n.

Proof. Let $M _ { j } ( a ) = \mathbb { E } _ { j } [ e ^ { a ( X - \mu ) } ]$ , which is finite for every a because the centered observations are 1-sub-Gaussian. Then $\bar { m } _ { j } = ( \log \bar { M } _ { j } ) ^ { \prime }$ and $m _ { i } ^ { \prime } ( a ) = \operatorname { V a r } ( Y _ { a } ) > 0$ , where $Y _ { a }$ denotes the centered observation tilted by a. Hence, each $m _ { j }$ is strictly increasing on R. The displayed formulas are the cumulant generating functions of $\mathcal { N } ( 0 , 1 )$ , of $\mathrm { U n i f } [ - \sqrt { 3 } , \sqrt { 3 } ]$ , and of the mixture in Lemma F.14, differentiated once.

The $\omega _ { r }$ are almost surely distinct, so strict monotonicity makes the entries of $\bar { \phi } _ { j }$ distinct. In particular, they are not all equal and $P _ { d } \bar { \phi } _ { j } \neq 0$ . This proves the first claim, makes $\bar { \mathbf { z } } _ { j }$ well defined, and gives $Z _ { j j } = \| \bar { \bf z } _ { j } \| _ { 2 } ^ { 2 } = 1$

For the lower bound, write $\begin{array} { r } { \overline { { m _ { j } } } = d ^ { - 1 } \sum _ { r } m _ { j } ( \omega _ { r } ) } \end{array}$ and use the identity

$$
\begin{array} { l } { { \displaystyle \left. P _ { d } \bar { \phi } _ { \mathrm { G } } , P _ { d } \bar { \phi } _ { \mathrm { A } } \right. = \sum _ { r = 1 } ^ { d } \left( m _ { \mathrm { G } } ( \omega _ { r } ) - \overline { { m _ { \mathrm { G } } } } \right) \left( m _ { \mathrm { A } } ( \omega _ { r } ) - \overline { { m _ { \mathrm { A } } } } \right) } \ ~ } \\ { { \displaystyle ~ = \frac { 1 } { 2 d } \sum _ { r , r ^ { \prime } = 1 } ^ { d } \left( m _ { \mathrm { G } } ( \omega _ { r } ) - m _ { \mathrm { G } } ( \omega _ { r ^ { \prime } } ) \right) \left( m _ { \mathrm { A } } ( \omega _ { r } ) - m _ { \mathrm { A } } ( \omega _ { r ^ { \prime } } ) \right) } . } \end{array}
$$

Both $m _ { \mathrm { G } }$ and $m _ { \mathrm { A } }$ are strictly increasing, so every summand with $\omega _ { r } \neq \omega _ { r ^ { \prime } }$ is strictly positive and the rest vanish. Since the $\omega _ { r }$ are almost surely distinct and $d \geq 2$ , the sum is strictly positive, whence $Z _ { \mathrm { G A } } > 0$

For the upper bound, Cauchy–Schwarz gives $Z _ { \mathrm { G A } } \leq 1$ with equality only if $P _ { d } \bar { \phi } _ { \mathrm { A } }$ is a nonnegative multiple of $P _ { d } \bar { \phi } _ { \mathrm { G } }$ , that is, only if there exist $\alpha , \beta \in \mathbb { R }$ with

$$
m _ { \mathrm { A } } ( \omega _ { r } ) = \alpha \omega _ { r } + \beta \qquad { \mathrm { f o r } } { \mathrm { e v e r y } } r \in [ d ] ,
$$

using $m _ { \mathrm { G } } ( a ) = a$ . In both cases $m _ { \mathrm { A } }$ is real analytic on R and is not affine there: for the uniform alternative $m _ { \mathrm { U J } } ^ { \prime } ( a ) = a ^ { - 2 } - 3 \mathrm { c s c h } ^ { 2 } ( \sqrt { 3 } a )$ is nonconstant, and for the mixture $m _ { \mathrm { M } } ^ { \prime \prime } ( a ) = - 2 ( 1 -$ $\sigma _ { n } ^ { 2 } ) ^ { 3 / 2 } \mathrm { s e c h } ^ { 2 } ( \sqrt { 1 - \sigma _ { n } ^ { 2 } } a )$ tanh ${ \sqrt { 1 - \sigma _ { n } ^ { 2 } } } a )$ vanishes only at $a = 0$ . Condition on $( \omega _ { 1 } , \omega _ { 2 } )$ with $\omega _ { 1 } \neq \omega _ { 2 }$ , which determines α and $\beta .$ The remaining requirement is that $\omega _ { 3 }$ be a zero of $a \mapsto$ $m _ { \mathrm { A } } ( a ) - \alpha a - \beta _ { \mathrm { \Omega } }$ , a real-analytic function that is not identically zero and therefore has at most countably many zeros in R. $\operatorname { A s } d \geq 3$ and $\omega _ { 3 }$ is independent of $( \omega _ { 1 } , \omega _ { 2 } )$ with a density, this event has probability zero. Hence $Z _ { \mathrm { G A } } < 1$ almost surely, and $\lambda _ { \operatorname* { m i n } } ( Z ) = 1 - Z _ { \mathrm { G A } } > 0$

Only $m _ { \mathrm { A } }$ depends on $n _ { : }$ , and only through $\sigma _ { n }$ in the Gaussian-mixture case. If $\sigma _ { n } \  \ 0$ then $m _ { \mathrm { M } } ( \omega _ { r } ) \to \operatorname { t a n h } ( \omega _ { r } )$ for each of the finitely many sampled coordinates $r \in [ d ]$ , so $Z _ { \mathrm { G A } }$ converges to the value obtained from tanh, which is strictly below one by the same argument because tanh is real analytic and not affine. Hence $Z _ { \mathrm { G A } }$ is bounded away from one, and $\bar { \lambda _ { \operatorname* { m i n } } } ( Z )$ away from zero, for all sufficiently large n, with constants depending only on ω. □

## F.4 EXPERT COVARIANCE

Lemma F.6 (Expert covariance approximation). Here, Σ is the covariance obtained from the multivariate delta method. Y is sub-Gaussian.

$$
\mathbb { E } \left[ n \left( \frac { \sum _ { i = 1 } ^ { n } \exp ( \lambda _ { k } Y _ { i } ) Y _ { i } } { \sum _ { i = 1 } ^ { n } \exp ( \lambda _ { k } Y _ { i } ) } - \frac { \mathbb { E } \left[ \exp ( \lambda _ { k } Y ) Y \right] } { \mathbb { E } \left[ \exp ( \lambda _ { k } Y ) \right] } \right) ^ { 2 } - \Sigma _ { k k } \right] = \mathcal { O } \left( \frac { 1 } { n } \right)\tag{34}
$$

Proof. Set

$$
a = \frac { \mathbb { E } [ e ^ { \lambda _ { k } Y } Y ] } { \mathbb { E } [ e ^ { \lambda _ { k } Y } ] } , \qquad B _ { i } = e ^ { \lambda _ { k } Y _ { i } } , \qquad A _ { i } = B _ { i } ( Y _ { i } - a ) , \qquad B ^ { * } = \mathbb { E } [ B _ { i } ] ,
$$

and write $A _ { n } = n ^ { - 1 } \sum _ { i } A _ { i }$ and $\begin{array} { r } { B _ { n } = n ^ { - 1 } \sum _ { i } B _ { i } } \end{array}$ . Then $\mathbb { E } [ A _ { i } ] = 0$ and the expression inside the expectation in (34) is $n ( A _ { n } / B _ { n } ) ^ { 2 }$ . Taylor’s theorem gives

$$
B _ { n } ^ { - 2 } = ( B ^ { * } ) ^ { - 2 } - 2 ( B ^ { * } ) ^ { - 3 } ( B _ { n } - B ^ { * } ) + 3 \tilde { B } ^ { - 4 } ( B _ { n } - B ^ { * } ) ^ { 2 } ,\tag{35}
$$

where $\tilde { B }$ lies between $B _ { n }$ and $B ^ { * }$ . The constant term in (35) contributes

$$
{ \frac { n \mathbb { E } [ A _ { n } ^ { 2 } ] } { ( B ^ { * } ) ^ { 2 } } } = { \frac { \mathrm { V a r } ( e ^ { \lambda _ { k } Y } ( Y - a ) ) } { \mathbb { E } [ e ^ { \lambda _ { k } Y } ] ^ { 2 } } } = \Sigma _ { k k } .
$$

For the linear term, independence and centering leave only coincident indices, so

$$
- \frac { 2 n } { ( B ^ { * } ) ^ { 3 } } \mathbb { E } \big [ A _ { n } ^ { 2 } ( B _ { n } - B ^ { * } ) \big ] = - \frac { 2 } { n ( B ^ { * } ) ^ { 3 } } \mathbb { E } \big [ A _ { 1 } ^ { 2 } ( B _ { 1 } - B ^ { * } ) \big ] = \mathcal { O } ( n ^ { - 1 } ) .
$$

It remains to bound the quadratic remainder. Because $\tilde { B }$ lies between two positive numbers, $\tilde { B } ^ { - 8 } \leq$ $B _ { n } ^ { - 8 } + ( B ^ { * } ) ^ { - 8 }$ . Jensen’s inequality gives $B _ { n } \geq e ^ { \lambda _ { k } \bar { Y } _ { n } }$ , and sub-Gaussianity therefore implies

$$
\begin{array} { r } { \mathbb { E } [ \tilde { B } ^ { - 8 } ] \leq \mathbb { E } [ e ^ { - 8 \lambda _ { k } \bar { Y } _ { n } } ] + ( B ^ { * } ) ^ { - 8 } = \mathcal { O } ( 1 ) . } \end{array}
$$

The variables $A _ { i }$ and $B _ { i }$ have moments of every fixed order, and the standard moment bound for centered i.i.d. averages yields

$$
\begin{array} { r } { \mathbb { E } [ A _ { n } ^ { 8 } ] = \mathcal { O } ( n ^ { - 4 } ) , \qquad \mathbb { E } [ ( B _ { n } - B ^ { * } ) ^ { 8 } ] = \mathcal { O } ( n ^ { - 4 } ) . } \end{array}
$$

Hölder’s inequality with exponents $2 , 4 ,$ 4 now gives

$$
\begin{array} { r l } & { 3 n \mathbb { E } \Big [ \tilde { B } ^ { - 4 } A _ { n } ^ { 2 } ( B _ { n } - B ^ { * } ) ^ { 2 } \Big ] \leq 3 n \mathbb { E } [ \tilde { B } ^ { - 8 } ] ^ { 1 / 2 } \mathbb { E } [ A _ { n } ^ { 8 } ] ^ { 1 / 4 } \mathbb { E } [ ( B _ { n } - B ^ { * } ) ^ { 8 } ] ^ { 1 / 4 } } \\ & { \qquad = { \cal O } ( n ^ { - 1 } ) . } \end{array}
$$

Combining the three terms proves (34).

## F.4.1 EXPERT HIERARCHY

The goal of this section is to verify the expert hierarchy of Definition B.1 for Gaussian, uniform and Gaussian-mixture data.

We denote the moment generating function as

$$
M _ { Y } ( \lambda ) = \operatorname { \mathbb { E } } \left[ \exp \left( \lambda Y \right) \right] .
$$

Lemma F.7 (Asymptotic normality). This lemma concerns the full stacked vector $( \lambda _ { 1 } , - \lambda _ { 1 } , \lambda _ { 2 } , - \lambda _ { 2 } , \overset { \cdot } { \omega } _ { 1 } , \overset { \cdot } { \dots } , \omega _ { d } ) \in \mathbb { R } ^ { 4 + d }$ . For readability, index this stacked vector by $\lambda _ { 1 } , \ldots , \lambda _ { 4 + d }$ within this lemma only. Let $Y _ { 1 } , \ldots , Y _ { n } \stackrel { \mathrm { i . i . d . } } { \sim } Q _ { j }$ . Recall the one-sided attention output h

from (1). Write $\mathbf { h } ( \mathbf { Y } ) \ = \ ( h ( \mathbf { Y } , \lambda _ { 1 } ) , \dots , h ( \mathbf { Y } , \lambda _ { 4 + d } ) ) ^ { \top }$ for these stacked head outputs. For $\lambda = ( \lambda _ { 1 } , \ldots , \lambda _ { 4 + d } )$

$$
\sqrt { n } \left( \mathbf { h } ( \mathbf { Y } ) - m _ { j } ( \boldsymbol { \lambda } ) \right) \stackrel { d } { \to } \mathcal { N } \left( 0 , \boldsymbol { \Sigma } _ { h } \right) ,
$$

where $m _ { j } ( \lambda )$ is the vector with components $\begin{array} { r } { m _ { j } ( \lambda _ { k } ) = \frac { M _ { Y } ^ { \prime } ( \lambda _ { k } ) } { M _ { Y } ( \lambda _ { k } ) } } \end{array}$ . The covariance matrix $\Sigma _ { h }$ is given by:

$$
\Sigma _ { h } ( \lambda _ { k } , \lambda _ { l } ) = \frac { M _ { Y } ^ { \prime \prime } ( \lambda _ { k } + \lambda _ { l } ) - ( m _ { j } ( \lambda _ { k } ) + m _ { j } ( \lambda _ { l } ) ) M _ { Y } ^ { \prime } ( \lambda _ { k } + \lambda _ { l } ) + m _ { j } ( \lambda _ { k } ) m _ { j } ( \lambda _ { l } ) M _ { Y } ( \lambda _ { k } + \lambda _ { l } ) } { M _ { Y } ( \lambda _ { k } ) M _ { Y } ( \lambda _ { l } ) } .
$$

Proof. By the law of large numbers, as $n  \infty ,$ , the sample averages converge to expectations:

$$
h ( \mathbf { Y } , \lambda ) \ { \overset { a . s . } { \longrightarrow } } \ { \frac { \mathbb { E } [ Y e ^ { \lambda Y } ] } { \mathbb { E } [ e ^ { \lambda Y } ] } } = { \frac { M _ { Y } ^ { \prime } ( \lambda ) } { M _ { Y } ( \lambda ) } } = m _ { j } ( \lambda ) .
$$

To find the asymptotic variance, we use the multivariate delta method. Let $V _ { i } ( \lambda ) = { \binom { Y _ { i } e ^ { \lambda Y _ { i } } } { e ^ { \lambda Y _ { i } } } }$ . By the central limit theorem, $\sqrt { n } ( \bar { V } - \mathbb { E } [ V ] ) \stackrel { d } { \to } { \mathcal { N } } ( 0 , \Sigma _ { V } )$ . Define the mapping $\textstyle \phi ( u , v ) = { \frac { u } { v } }$ . The gradient is:

$$
\nabla \phi ( u , v ) = \left( { 1 / v \atop - u / v ^ { 2 } } \right) .
$$

Evaluated at the expectations $( u , v ) = ( M _ { Y } ^ { \prime } ( \lambda ) , M _ { Y } ( \lambda ) )$ ), the gradient is:

$$
\nabla \phi = \frac { 1 } { M _ { Y } ( \lambda ) } \left( { 1 \atop - m _ { j } ( \lambda ) } \right) .
$$

The linear approximation (influence function) for the estimator at $\lambda$ is:

$$
\mathrm { I F } _ { i } ( \lambda ) = \nabla \boldsymbol \phi ^ { \top } ( V _ { i } ( \lambda ) - \mathbb E [ V _ { i } ( \lambda ) ] ) = \frac { e ^ { \lambda Y _ { i } } } { M _ { Y } ( \lambda ) } ( Y _ { i } - m _ { j } ( \lambda ) ) .
$$

The asymptotic covariance between the estimators at $\lambda _ { k }$ and $\lambda _ { l }$ is the covariance of their respective influence functions:

$$
\begin{array} { r l } & { \Sigma _ { h } ( \lambda _ { k } , \lambda _ { l } ) = \operatorname { C o v } ( \operatorname { I F } ( \lambda _ { k } ) , \operatorname { I F } ( \lambda _ { l } ) ) = \operatorname { \mathbb { E } } [ \operatorname { I F } ( \lambda _ { k } ) \operatorname { I F } ( \lambda _ { l } ) ] } \\ & { \quad \quad \quad = \operatorname { \mathbb { E } } \left[ \left( \frac { e ^ { \lambda _ { k } Y } } { M _ { Y } ( \lambda _ { k } ) } ( Y - m _ { j } ( \lambda _ { k } ) ) \right) \left( \frac { e ^ { \lambda _ { l } Y } } { M _ { Y } ( \lambda _ { l } ) } ( Y - m _ { j } ( \lambda _ { l } ) ) \right) \right] } \end{array}
$$

Expanding yields

$$
\begin{array} { l } { \Sigma _ { h } ( \lambda _ { k } , \lambda _ { l } ) = \displaystyle \frac { 1 } { M _ { Y } ( \lambda _ { k } ) M _ { Y } ( \lambda _ { l } ) } \mathbb { E } \left[ e ^ { ( \lambda _ { k } + \lambda _ { l } ) Y } ( Y - m _ { j } ( \lambda _ { k } ) ) ( Y - m _ { j } ( \lambda _ { l } ) ) \right] } \\ { \displaystyle \ = \frac { 1 } { M _ { Y } ( \lambda _ { k } ) M _ { Y } ( \lambda _ { l } ) } \mathbb { E } \left[ e ^ { ( \lambda _ { k } + \lambda _ { l } ) Y } ( Y ^ { 2 } - Y ( m _ { j } ( \lambda _ { k } ) + m _ { j } ( \lambda _ { l } ) ) + m _ { j } ( \lambda _ { k } ) m _ { j } ( \lambda _ { l } ) ) \right] } \end{array}
$$

Using the property that $\mathbb { E } [ Y ^ { k } e ^ { t Y } ] = M _ { Y } ^ { ( k ) } ( t )$ , we obtain the final expression.

## F.4.2 GAUSSIAN EXPERT COVARIANCE

Lemma F.8 (Gaussian distribution). $I f Y \sim { \mathcal { N } } ( 0 , 1 )$ , then $M _ { Y } ( t ) = e ^ { t ^ { 2 } / 2 } ,$

$$
\begin{array} { c } { { m _ { \mathrm { G } } ( \lambda ) = \lambda , } } \\ { { \sum _ { h } ( \lambda _ { k } , \lambda _ { l } ) = ( 1 + \lambda _ { k } \lambda _ { l } ) e ^ { \lambda _ { k } \lambda _ { l } } . } } \end{array}
$$

Proof. Let $Y \sim \mathcal { N } ( 0 , 1 )$ . The moment generating function is $M _ { Y } ( t ) = e ^ { t ^ { 2 } / 2 }$ . The first two derivatives are:

$$
\begin{array} { l } { { M _ { Y } ^ { \prime } ( t ) = t e ^ { t ^ { 2 } / 2 } = t M _ { Y } ( t ) , } } \\ { { M _ { Y } ^ { \prime \prime } ( t ) = ( 1 + t ^ { 2 } ) e ^ { t ^ { 2 } / 2 } = ( 1 + t ^ { 2 } ) M _ { Y } ( t ) . } } \end{array}
$$

The mean function is $m _ { \mathrm { G } } ( \lambda ) = M _ { Y } ^ { \prime } ( \lambda ) / M _ { Y } ( \lambda ) = \lambda$ . We apply the general covariance formula from Lemma F.7:

$$
\Sigma _ { h } ( \lambda _ { k } , \lambda _ { l } ) = \frac { M _ { Y } ^ { \prime \prime } ( \lambda _ { k } + \lambda _ { l } ) - ( m _ { \mathrm { G } } ( \lambda _ { k } ) + m _ { \mathrm { G } } ( \lambda _ { l } ) ) M _ { Y } ^ { \prime } ( \lambda _ { k } + \lambda _ { l } ) + m _ { \mathrm { G } } ( \lambda _ { k } ) m _ { \mathrm { G } } ( \lambda _ { l } ) M _ { Y } ( \lambda _ { k } + \lambda _ { l } ) } { M _ { Y } ( \lambda _ { k } ) M _ { Y } ( \lambda _ { l } ) } .
$$

$$
\begin{array} { r l } & { \mathrm { S u b s t i t u t i n g ~ t h e ~ d e r i v a t i v e s ~ a n d ~ } m _ { \mathrm { G } } ( \lambda ) = \lambda ; } \\ & { \qquad M _ { Y } ^ { \prime \prime } ( \lambda _ { k } + \lambda _ { l } ) - ( m _ { \mathrm { G } } ( \lambda _ { k } ) + m _ { \mathrm { G } } ( \lambda _ { l } ) ) M _ { Y } ^ { \prime } ( \lambda _ { k } + \lambda _ { l } ) + m _ { \mathrm { G } } ( \lambda _ { k } ) m _ { \mathrm { G } } ( \lambda _ { l } ) M _ { Y } ( \lambda _ { k } + \lambda _ { l } ) } \\ & { \qquad = ( 1 + ( \lambda _ { k } + \lambda _ { l } ) ^ { 2 } ) M _ { Y } ( \lambda _ { k } + \lambda _ { l } ) - ( \lambda _ { k } + \lambda _ { l } ) ^ { 2 } M _ { Y } ( \lambda _ { k } + \lambda _ { l } ) + \lambda _ { k } \lambda _ { l } M _ { Y } ( \lambda _ { k } + \lambda _ { l } ) } \\ & { \qquad = M _ { Y } ( \lambda _ { k } + \lambda _ { l } ) \left[ 1 + ( \lambda _ { k } + \lambda _ { l } ) ^ { 2 } - ( \lambda _ { k } + \lambda _ { l } ) ^ { 2 } + \lambda _ { k } \lambda _ { l } \right] } \\ & { \qquad = M _ { Y } ( \lambda _ { k } + \lambda _ { l } ) ( 1 + \lambda _ { k } \lambda _ { l } ) . } \end{array}
$$

Dividing by the denominator $M _ { Y } ( \lambda _ { k } ) M _ { Y } ( \lambda _ { l } ) = e ^ { \lambda _ { k } ^ { 2 } / 2 } e ^ { \lambda _ { l } ^ { 2 } / 2 } .$

$$
\begin{array} { l } { { \Sigma _ { h } ( \lambda _ { k } , \lambda _ { l } ) = ( 1 + \lambda _ { k } \lambda _ { l } ) { \frac { e ^ { ( \lambda _ { k } + \lambda _ { l } ) ^ { 2 } / 2 } } { e ^ { ( \lambda _ { k } ^ { 2 } + \lambda _ { l } ^ { 2 } ) / 2 } } } } } \\ { { \ = ( 1 + \lambda _ { k } \lambda _ { l } ) \exp \left( { \frac { \lambda _ { k } ^ { 2 } + 2 \lambda _ { k } \lambda _ { l } + \lambda _ { l } ^ { 2 } - \lambda _ { k } ^ { 2 } - \lambda _ { l } ^ { 2 } } { 2 } } \right) } } \\ { { \ = ( 1 + \lambda _ { k } \lambda _ { l } ) e ^ { \lambda _ { k } \lambda _ { l } } . } } \end{array}
$$

Definition F.9 (Symmetrized expert covariance). Recall $\begin{array} { r } { g _ { k } ( { \mathbf X } ) = \frac { 1 } { 2 } \left( h ( { \mathbf X } , \lambda _ { k } ) + h ( { \mathbf X } , - \lambda _ { k } ) \right) } \end{array}$ . For $a , b \in \mathbb { R } .$ , the asymptotic covariance of $h ( \cdot , a )$ and $h ( \cdot , - a ) ^ { \dagger }$ ’s symmetrization against the analogous pair at b is denoted $\textstyle { \dot { \sum _ { q } } } ( a , b )$ . Since each $Q _ { j }$ is symmetric about the origin, it follows from the linearity of the influence functions that:

$$
\Sigma _ { g } ( a , b ) = \frac { 1 } { 2 } \left( \Sigma _ { h } ( a , b ) + \Sigma _ { h } ( a , - b ) \right)\tag{36}
$$

where $\Sigma _ { h } ( \cdot , \cdot )$ is the one-sided covariance defined in Lemma F.7.

## F.4.3 UNIFORM EXPERT COVARIANCE

Lemma F.10 (Uniform distribution). $I f Y \sim \mathrm { U n i f } [ - { \sqrt { 3 } } , { \sqrt { 3 } } ] ,$ , then

$$
m _ { \mathrm { U } } ( \lambda ) = \sqrt { 3 } \coth ( \sqrt { 3 } \lambda ) - \frac { 1 } { \lambda } ,
$$

and

$$
\begin{array} { c } { { \Sigma _ { h } ( \lambda _ { k } , \lambda _ { l } ) = \displaystyle { \frac { 2 \sqrt { 3 } \lambda _ { k } \lambda _ { l } \{ \coth ( \sqrt { 3 } \lambda _ { k } ) + \coth ( \sqrt { 3 } \lambda _ { l } ) \} } { ( \lambda _ { k } + \lambda _ { l } ) ^ { 3 } } } } } \\ { { - \displaystyle { \frac { 3 \{ \lambda _ { k } ^ { 2 } \cosh ^ { 2 } ( \sqrt { 3 } \lambda _ { k } ) + \lambda _ { l } ^ { 2 } \operatorname { c s c h } ^ { 2 } ( \sqrt { 3 } \lambda _ { l } ) \} } { ( \lambda _ { k } + \lambda _ { l } ) ^ { 2 } } } . } } \end{array}
$$

Proof. Put $a = { \sqrt { 3 } } .$ . The moment generating function and its tilted mean are

$$
M _ { Y } ( \lambda ) = \frac { \sinh ( a \lambda ) } { a \lambda } , \qquad m _ { \mathrm { U } } ( \lambda ) = \frac { M _ { Y } ^ { \prime } ( \lambda ) } { M _ { Y } ( \lambda ) } = a \coth ( a \lambda ) - \frac { 1 } { \lambda } ,
$$

with $m _ { \mathrm { U } } ^ { \prime } ( \lambda ) = - a ^ { 2 } \cosh ^ { 2 } ( a \lambda ) + \lambda ^ { - 2 }$ . The general covariance identity from Lemma F.7 can be written

$$
\begin{array} { l } { \Sigma _ { h } ( \lambda _ { k } , \lambda _ { l } ) = \frac { M _ { Y } ( \lambda _ { k } + \lambda _ { l } ) } { M _ { Y } ( \lambda _ { k } ) M _ { Y } ( \lambda _ { l } ) } \Big [ m _ { \mathrm { U } } ^ { \prime } ( \lambda _ { k } + \lambda _ { l } ) } \\ { \quad \quad \ + \ \{ m _ { \mathrm { U } } ( \lambda _ { k } + \lambda _ { l } ) - m _ { \mathrm { U } } ( \lambda _ { k } ) \} \{ m _ { \mathrm { U } } ( \lambda _ { k } + \lambda _ { l } ) - m _ { \mathrm { U } } ( \lambda _ { l } ) \} \Big ] . } \end{array}
$$

Substitute the displayed formulas for $M _ { Y } , m _ { \mathrm { U } } , m _ { \mathrm { U } } ^ { \prime }$ and use

$$
\frac { \sinh ( a ( \lambda _ { k } + \lambda _ { l } ) ) } { \sinh ( a \lambda _ { k } ) \sinh ( a \lambda _ { l } ) } = \coth ( a \lambda _ { k } ) + \coth ( a \lambda _ { l } ) .
$$

Collecting the rational and hyperbolic terms gives exactly the stated expression.

Lemma F.11 (Monotonicity for uniform distribution). Let $Y \sim \mathrm { U n i f } [ - { \sqrt { 3 } } , { \sqrt { 3 } } ]$ . Then $\Sigma _ { g } ( \lambda , \lambda )$ is strictly decreasing in |λ|.

Proof. For $t = \sqrt { 3 } \lambda > 0$ , Lemma F.10 and symmetry give

$$
\Sigma _ { g } ( t / \sqrt { 3 } , t / \sqrt { 3 } ) = \frac { 3 } { 4 } \left( \frac { \coth t } { t } - \csc ^ { 2 } t \right) + \frac { 1 } { 2 } t ^ { 2 } \operatorname { c s c h } ^ { 2 } t \left[ 1 - 3 \left( \coth t - \frac { 1 } { t } \right) ^ { 2 } \right] .
$$

$$
\begin{array} { r l r } {  { - 6 4 t ^ { 2 } \sinh ^ { 5 } t \frac { d } { d t } \Sigma _ { g } ( t / \sqrt { 3 } , t / \sqrt { 3 } ) } } \\ & { } & { = 3 \cosh ( 5 t ) - ( 9 + 1 2 0 t ^ { 2 } + 3 2 t ^ { 4 } ) \cosh ( 3 t ) } \\ & { } & { \quad + ( 6 + 1 2 0 t ^ { 2 } - 3 5 2 t ^ { 4 } ) \cosh t + ( 1 2 t + 1 2 8 t ^ { 3 } ) \sinh ( 3 t ) } \\ & { } & { \quad + ( - 3 6 t + 3 8 4 t ^ { 3 } ) \sinh t = \displaystyle \sum _ { m \ge 0 } A _ { m } t ^ { m } . } \end{array}
$$

$$
A _ { 0 } = A _ { 2 } = A _ { 4 } = A _ { 6 } = 0 , \qquad A _ { 2 k + 1 } = 0 \quad ( k \geq 0 ) ,
$$

$$
\begin{array} { r } { \small { ( \begin{array} { l l l l l l } { m } & { \middle | } & { 8 } & { 1 0 } & { 1 2 } & { 1 4 } & { 1 8 } & { 1 8 } \\ { A _ { m } } & { | } & { \frac { 2 5 6 } { 5 } } & { \frac { 3 7 1 2 } { 3 1 5 } } & { \frac { 6 3 6 8 } { 4 7 2 5 } } & { \frac { 3 9 5 2 } { 3 1 1 8 5 } } & { \frac { 5 3 9 6 0 6 } { 4 2 5 6 7 5 2 5 } } & { \frac { 1 9 7 9 } { 1 6 5 8 4 7 5 } } \end{array} ) } } \end{array}
$$

For even $m \geq 2 0 ,$ , dropping positive terms and bounding falling factorials by powers gives

$$
\frac { m ! A _ { m } } { 5 ^ { m } } \geq 3 - \left( \frac { 3 } { 5 } \right) ^ { m } \left( 9 + \frac { 4 0 } { 3 } m ^ { 2 } + \frac { 3 2 } { 8 1 } m ^ { 4 } \right) - \frac { 3 5 2 m ^ { 4 } + 3 6 m } { 5 ^ { m } } .
$$

Each subtracted term decreases for $m \geq 2 0$ , since its ratio at consecutive integers is at most

$$
{ \frac { 3 } { 5 } } \left( { \frac { m + 1 } { m } } \right) ^ { 4 } \leq { \frac { 3 } { 5 } } \left( { \frac { 2 1 } { 2 0 } } \right) ^ { 4 } = { \frac { 5 8 3 4 4 3 } { 8 0 0 0 0 0 } } < 1 .
$$

Thus

$$
\frac { m ! A _ { m } } { 5 ^ { m } } \geq 3 - \frac { 2 3 9 0 2 6 8 3 2 3 7 2 3 2 9 } { 9 5 3 6 7 4 3 1 6 4 0 6 2 5 } > \frac { 1 } { 3 } , \qquad \frac { d } { d t } \Sigma _ { g } ( t / \sqrt { 3 } , t / \sqrt { 3 } ) < 0 \quad ( t > 0 ) .
$$

Evenness and continuity at zero give the claim.

Lemma F.12 (Symmetrized cross-covariance for uniform distribution). Let $Y \sim \mathrm { U n i f } [ - { \sqrt { 3 } } , { \sqrt { 3 } } ]$ and $\lambda _ { 1 } , \lambda _ { 2 } \in \mathbb { R }$ . Then, with the right-hand side interpreted by continuity when $\lambda _ { 1 } \lambda _ { 2 } = 0 \stackrel { . } { o r } \lambda _ { 1 } = \dot { \pm } \lambda _ { 2 }$

$$
\begin{array} { c } { { \Sigma _ { g } ( \lambda _ { 1 } , \lambda _ { 2 } ) = \sqrt { 3 } \lambda _ { 1 } \lambda _ { 2 } \left[ \displaystyle \frac { \coth ( \sqrt { 3 } \lambda _ { 1 } ) + \coth ( \sqrt { 3 } \lambda _ { 2 } ) } { ( \lambda _ { 1 } + \lambda _ { 2 } ) ^ { 3 } } - \displaystyle \frac { \coth ( \sqrt { 3 } \lambda _ { 1 } ) - \coth ( \sqrt { 3 } \lambda _ { 2 } ) } { ( \lambda _ { 1 } - \lambda _ { 2 } ) ^ { 3 } } \right] } } \\ { { - \displaystyle \frac { 3 } { 2 } \left[ \lambda _ { 1 } ^ { 2 } \operatorname { c s c h } ^ { 2 } ( \sqrt { 3 } \lambda _ { 1 } ) + \lambda _ { 2 } ^ { 2 } \operatorname { c s c h } ^ { 2 } ( \sqrt { 3 } \lambda _ { 2 } ) \right] } } \\ { { \times \left[ \displaystyle \frac { 1 } { ( \lambda _ { 1 } + \lambda _ { 2 } ) ^ { 2 } } + \displaystyle \frac { 1 } { ( \lambda _ { 1 } - \lambda _ { 2 } ) ^ { 2 } } \right] . } } \end{array}
$$

Proof. By symmetry, $\Sigma _ { g } ( \lambda _ { 1 } , \lambda _ { 2 } ) = { \textstyle { \frac { 1 } { 2 } } } \{ \Sigma _ { h } ( \lambda _ { 1 } , \lambda _ { 2 } ) + \Sigma _ { h } ( \lambda _ { 1 } , - \lambda _ { 2 } ) \}$ . Substitute the formula in Lemma F.10 and use cot $\operatorname { \rho } _ { 1 } ( - x ) = - \operatorname { c o t h } ( x )$ and $\operatorname { c s c h } ^ { 2 } ( - x ) = \operatorname { c s c h } ^ { 2 } ( x )$ . Collecting the common terms gives the displayed expression. Continuity of the covariance formula in Lemma F.7 gives the stated extensions. For $\lambda _ { 1 } = \pm \lambda _ { 2 }$ , evenness gives the self-variance $\Sigma _ { g } ( \lambda _ { 1 } , \lambda _ { 1 } )$ □

Proof. $\Sigma _ { g }$ is even in each argument, so it suffices to treat $0 \leq \lambda _ { 1 } < \lambda _ { 2 } \leq 1$ . Write

$$
M _ { Y } ( \lambda ) = \frac { \sinh ( \sqrt { 3 } \lambda ) } { \sqrt { 3 } \lambda } , \qquad \chi _ { \lambda } ( x ) = M _ { Y } ( \lambda ) x \cosh ( \lambda x ) - M _ { Y } ^ { \prime } ( \lambda ) \sinh ( \lambda x ) ,
$$

with continuous extensions at zero. For $u > 0 ,$

$$
\frac { d } { d u } ( u \coth u ) = \frac { \sinh ( 2 u ) - 2 u } { 2 \sinh ^ { 2 } u } > 0 ,
$$

$$
\begin{array} { c } { { \displaystyle \frac { 2 u } 3 - \frac { d } { d u } ( u \coth u ) = \frac { 2 u \cosh ( 2 u ) + 4 u - 3 \sinh ( 2 u ) } { 6 \sinh ^ { 2 } u } } } \\ { { = \displaystyle \frac { 1 } { 6 \sinh ^ { 2 } u } \sum _ { k \geq 2 } \frac { 2 ^ { 2 k + 1 } ( 2 k - 2 ) } { ( 2 k + 1 ) ! } u ^ { 2 k + 1 } > 0 . } } \end{array}
$$

Hence

$$
\operatorname* { l i m } _ { t \downarrow 0 } t \coth t = 1 , \qquad 0 < s \coth s - t \coth t < { \frac { s ^ { 2 } - t ^ { 2 } } { 3 } } \quad ( 0 \leq t < s ) .
$$

For $0 < x < { \sqrt { 3 } }$ and $0 < \lambda \leq 1$ ，

$$
\begin{array} { c } { { \displaystyle \frac { \chi _ { \lambda } ( x ) } { M _ { Y } ( \lambda ) ^ { 2 } } = \displaystyle \frac { \sinh ( \lambda x ) } { \lambda M _ { Y } ( \lambda ) } \left[ 1 - \sqrt { 3 } \lambda \coth ( \sqrt { 3 } \lambda ) + \lambda x \coth ( \lambda x ) \right] } } \\ { { \displaystyle \qquad \mathrm > \displaystyle \frac { \sinh ( \lambda x ) } { \lambda M _ { Y } ( \lambda ) } ( 1 - \lambda ^ { 2 } ) \geq 0 , } } \\ { { \displaystyle \frac { \partial } { \partial \lambda } \frac { \chi _ { \lambda } ( x ) } { M _ { Y } ( \lambda ) ^ { 2 } } = \displaystyle \frac { \sinh ( \lambda x ) } { \lambda ^ { 2 } M _ { Y } ( \lambda ) } \left[ \lambda ^ { 2 } ( x ^ { 2 } - 3 ) + 2 \{ \sqrt { 3 } \lambda \coth ( \sqrt { 3 } \lambda ) - 1 \} \right. } } \\ { { \displaystyle \qquad \left. \times \{ \sqrt { 3 } \lambda \coth ( \sqrt { 3 } \lambda ) - \lambda x \coth ( \lambda x ) \} \right] } } \\ { { \displaystyle \qquad \leq - \displaystyle \frac { ( 3 - x ^ { 2 } ) \sinh ( \lambda x ) } { M _ { Y } ( \lambda ) } \left( 1 - \frac { 2 \lambda ^ { 2 } } { 3 } \right) < 0 . } } \end{array}
$$

Using $\chi _ { 0 } ( x ) = x$ , oddness in x, and Lemma F.7,

$$
\begin{array} { l } { { \displaystyle \Sigma _ { g } ( \lambda _ { 2 } , \lambda _ { 2 } ) = \frac { 1 } { \sqrt { 3 } } \int _ { 0 } ^ { \sqrt { 3 } } \frac { \chi _ { \lambda _ { 2 } } ( x ) ^ { 2 } } { M _ { Y } ( \lambda _ { 2 } ) ^ { 4 } } d x } } \\ { { \displaystyle < \frac { 1 } { \sqrt { 3 } } \int _ { 0 } ^ { \sqrt { 3 } } \frac { \chi _ { \lambda _ { 2 } } ( x ) \chi _ { \lambda _ { 1 } } ( x ) } { M _ { Y } ( \lambda _ { 2 } ) ^ { 2 } M _ { Y } ( \lambda _ { 1 } ) ^ { 2 } } d x = \Sigma _ { g } ( \lambda _ { 1 } , \lambda _ { 2 } ) } } \\ { { \displaystyle < \frac { 1 } { \sqrt { 3 } } \int _ { 0 } ^ { \sqrt { 3 } } \frac { \chi _ { \lambda _ { 1 } } ( x ) ^ { 2 } } { M _ { Y } ( \lambda _ { 1 } ) ^ { 4 } } d x = \Sigma _ { g } ( \lambda _ { 1 } , \lambda _ { 1 } ) . } } \end{array}
$$

Taking $\lambda _ { 1 } = \lambda _ { 2 }$ above also gives $\begin{array} { r } { \frac { d } { d \lambda } \Sigma _ { g } ( \lambda , \lambda ) < 0 \mathrm { o n } ( 0 , 1 ) } \end{array}$ . Lemma F.11 is needed for the larger λ reached in the second stage, where the hierarchy itself fails and no bounded-range argument applies.

## F.4.4 GAUSSIAN-MIXTURE EXPERT COVARIANCE

We now give the analogous proof for the Gaussian mixture in Definition 2.1. Write $a _ { n } = \sqrt { 1 - \sigma _ { n } ^ { 2 } }$ The centered observation has the representation $Y \overset { D } { = } a _ { n } S + \sigma _ { n } G$ , where $S$ is Rademacher, $G$ is standard Gaussian, and $S , G$ are independent. Thus Y has mean zero and variance one.

Lemma F.14 (Mixture moments).

$$
\begin{array} { r l } & { M _ { Y } ( t ) = e ^ { \sigma _ { n } ^ { 2 } t ^ { 2 } / 2 } \cosh ( a _ { n } t ) , } \\ & { m _ { \mathrm { M } } ( t ) = \sigma _ { n } ^ { 2 } t + a _ { n } \operatorname { t a n h } ( a _ { n } t ) , } \\ & { m _ { \mathrm { M } } ^ { \prime } ( t ) = \sigma _ { n } ^ { 2 } + a _ { n } ^ { 2 } \operatorname { s e c h } ^ { 2 } ( a _ { n } t ) . } \end{array}
$$

Moreover, Y is 1-sub-Gaussian.

Proof. Independence of S and G gives

$$
M _ { Y } ( t ) = \mathbb { E } [ e ^ { t a _ { n } S } ] \mathbb { E } [ e ^ { t \sigma _ { n } G } ] = \cosh ( a _ { n } t ) e ^ { \sigma _ { n } ^ { 2 } t ^ { 2 } / 2 } .
$$

Differentiating log $M _ { Y }$ twice gives $m _ { \mathrm { M } } ( t ) = M _ { Y } ^ { \prime } ( t ) / M _ { Y } ( t )$ and $m _ { \mathrm { M } } ^ { \prime } ( t ) = M _ { Y } ^ { \prime \prime } ( t ) / M _ { Y } ( t ) -$ $m _ { \mathrm { M } } ( t ) ^ { 2 }$ . Finally, cosh $\phantom { } _ { 1 } ( a _ { n } t ) \leq e ^ { a _ { n } ^ { 2 } t ^ { 2 } / 2 }$ gives $M _ { Y } ( t ) \leq e ^ { t ^ { 2 } / 2 }$ □

Lemma F.15 (Gaussian mixture expert covariance).

$$
\begin{array} { r l r } {  { \Sigma _ { h } ( \lambda _ { k } , \lambda _ { l } ) = \frac { M _ { Y } ( \lambda _ { k } + \lambda _ { l } ) } { M _ { Y } ( \lambda _ { k } ) M _ { Y } ( \lambda _ { l } ) } \Big [ m _ { \mathrm { M } } ^ { \prime } ( \lambda _ { k } + \lambda _ { l } ) } } \\ & { } & { + ( m _ { \mathrm { M } } ( \lambda _ { k } + \lambda _ { l } ) - m _ { \mathrm { M } } ( \lambda _ { k } ) ) ( m _ { \mathrm { M } } ( \lambda _ { k } + \lambda _ { l } ) - m _ { \mathrm { M } } ( \lambda _ { l } ) ) \Big ] . } \end{array}
$$

Writing $\begin{array} { r } { \Sigma _ { g } ( \lambda _ { k } , \lambda _ { l } ) = \frac { 1 } { 2 } \{ \Sigma _ { h } ( \lambda _ { k } , \lambda _ { l } ) + \Sigma _ { h } ( \lambda _ { k } , - \lambda _ { l } ) \} } \end{array}$

$$
\begin{array} { l } { { \Sigma _ { g } ( \lambda _ { k } , \lambda _ { l } ) = \cosh ( \sigma _ { n } ^ { 2 } \lambda _ { k } \lambda _ { l } ) \Bigl [ a _ { n } ^ { 2 } \operatorname { s e c h } ^ { 2 } ( a _ { n } \lambda _ { k } ) \operatorname { s e c h } ^ { 2 } ( a _ { n } \lambda _ { l } ) + \sigma _ { n } ^ { 2 } } } \\ { { + \sigma _ { n } ^ { 4 } \lambda _ { k } \lambda _ { l } \operatorname { t a n h } ( a _ { n } \lambda _ { k } ) \operatorname { t a n h } ( a _ { n } \lambda _ { l } ) \Bigl ] } } \\ { { + \sinh ( \sigma _ { n } ^ { 2 } \lambda _ { k } \lambda _ { l } ) \Bigl [ \sigma _ { n } ^ { 2 } \operatorname { t a n h } ( a _ { n } \lambda _ { k } ) \operatorname { t a n h } ( a _ { n } \lambda _ { l } ) + \sigma _ { n } ^ { 4 } \lambda _ { k } \lambda _ { l } } } \\ { { + \sigma _ { n } ^ { 2 } a _ { n } \{ \lambda _ { k } \operatorname { s e c h } ^ { 2 } ( a _ { n } \lambda _ { k } ) \operatorname { t a n h } ( a _ { n } \lambda _ { l } ) + \lambda _ { l } \operatorname { s e c h } ^ { 2 } ( a _ { n } \lambda _ { l } ) \operatorname { t a n h } ( a _ { n } \lambda _ { k } ) \} \Bigl ] . } } \end{array}\tag{37}
$$

In particular, $\Sigma _ { g } ( \lambda _ { k } , 0 ) = \sigma _ { n } ^ { 2 } + a _ { n } ^ { 2 } \operatorname { s e c h } ^ { 2 } ( a _ { n } \lambda _ { k } ) a n d \Sigma _ { g } ( 0 , 0 ) = 1 .$

Proof. The formula for $\Sigma _ { h }$ is Lemma F.7 rewritten using Lemma ${ \mathrm { F . l 4 } } ,$ exactly as in the uniform case. Exponentially tilting $a _ { n } S + \sigma _ { n } G$ by $\lambda _ { k } + \lambda _ { l }$ shifts the Gaussian mean by $\sigma _ { n } ^ { 2 } ( \lambda _ { k } + \lambda _ { l } )$ ) and tilts the Rademacher probabilities. For $S \in \{ - 1 , 1 \}$

$$
\frac { e ^ { a _ { n } \lambda _ { k } S } } { \cosh ( a _ { n } \lambda _ { k } ) } \big ( a _ { n } S - a _ { n } \operatorname { t a n h } ( a _ { n } \lambda _ { k } ) \big ) = a _ { n } \operatorname { s e c h } ^ { 2 } ( a _ { n } \lambda _ { k } ) S .
$$

Substitution gives

$$
\begin{array} { l } { { \Sigma _ { h } ( \lambda _ { k } , \lambda _ { l } ) = e ^ { \sigma _ { n } ^ { 2 } \lambda _ { k } \lambda _ { l } } \Big [ a _ { n } ^ { 2 } \operatorname { s e c h } ^ { 2 } ( a _ { n } \lambda _ { k } ) \operatorname { s e c h } ^ { 2 } ( a _ { n } \lambda _ { l } ) } } \\ { { { } + ( \sigma _ { n } ^ { 2 } + \sigma _ { n } ^ { 4 } \lambda _ { k } \lambda _ { l } ) \{ 1 + \operatorname { t a n h } ( a _ { n } \lambda _ { k } ) \operatorname { t a n h } ( a _ { n } \lambda _ { l } ) \} } } \\ { { { } + \sigma _ { n } ^ { 2 } a _ { n } \{ \lambda _ { k } \operatorname { s e c h } ^ { 2 } ( a _ { n } \lambda _ { k } ) \operatorname { t a n h } ( a _ { n } \lambda _ { l } ) + \lambda _ { l } \operatorname { s e c h } ^ { 2 } ( a _ { n } \lambda _ { l } ) \operatorname { t a n h } ( a _ { n } \lambda _ { k } ) \} \Big ] . } } \end{array}
$$

Averaging this expression with its value at $- \lambda _ { l }$ and using the definitions of cosh and sinh gives (37). □

Lemma F.16 (Expert hierarchy for the Gaussian mixture). For every fixed $0 < \delta < 1$ there exists $\sigma _ { \delta } > 0$ such that,for all $0 < \sigma _ { n } < \sigma _ { \delta } < 1 / \sqrt { 2 }$ and

$$
0 \leq | \lambda _ { 1 } | < | \lambda _ { 2 } | \leq L _ { \operatorname* { m a x } } = ( 1 - \delta ) \log ( 1 / \sigma _ { n } ) ,
$$

we have

$$
\Sigma _ { g } ( \lambda _ { 2 } , \lambda _ { 2 } ) < \Sigma _ { g } ( \lambda _ { 2 } , \lambda _ { 1 } ) < \Sigma _ { g } ( \lambda _ { 1 } , \lambda _ { 1 } ) .
$$

Quantitatively,

$$
\operatorname* { m i n } \{ \Sigma _ { g } ( \lambda _ { 2 } , \lambda _ { 1 } ) - \Sigma _ { g } ( \lambda _ { 2 } , \lambda _ { 2 } ) , \Sigma _ { g } ( \lambda _ { 1 } , \lambda _ { 1 } ) - \Sigma _ { g } ( \lambda _ { 2 } , \lambda _ { 1 } ) \} \geq \frac { e ^ { - 4 L _ { \operatorname* { m a x } } } } { 8 ( 1 + L _ { \operatorname* { m a x } } ) } ( \lambda _ { 2 } ^ { 2 } - \lambda _ { 1 } ^ { 2 } ) .
$$

Thus the expert hierarchy of Definition B.1 holds throughout $[ - L _ { \mathrm { m a x } } , L _ { \mathrm { m a x } } ] ,$ , and $L _ { \mathrm { m a x } } \to \infty a s$ $\sigma _ { n } \downarrow 0 f o r$ any fixed δ.

Proof. Write $u = \lambda _ { 2 } ^ { 2 } , v = \lambda _ { 1 } ^ { 2 }$ , and separate

$$
\Sigma _ { g } ( \sqrt { u } , \sqrt { v } ) = a _ { n } ^ { 2 } \operatorname { s e c h } ^ { 2 } ( a _ { n } \sqrt { u } ) \operatorname { s e c h } ^ { 2 } ( a _ { n } \sqrt { v } ) + \sigma _ { n } ^ { 2 } + R _ { \sigma _ { n } } ( u , v ) .
$$

For $L \geq 1$ and $0 \leq u , v \leq L ^ { 2 }$ , expanding (37), using cosh $t - 1 \leq t ^ { 2 } e ^ { t } / 2$ and $1 \leq \sinh ( t ) / t \leq e ^ { t }$ and bounding the resulting derivatives gives a universal C such that

$$
| \partial _ { u } R _ { \sigma _ { n } } ( u , v ) | + | \partial _ { v } R _ { \sigma _ { n } } ( u , v ) | \leq C \sigma _ { n } ^ { 4 } ( 1 + L ) ^ { 8 } e ^ { \sigma _ { n } ^ { 2 } L ^ { 2 } } .
$$

For the dominant term, $a _ { n } ^ { 4 } \geq 1 / 4 .$ , sech $t \geq e ^ { - t }$ , and $\operatorname { t a n h } ( t ) / t \geq 1 / ( 1 + t )$ imply

$$
\begin{array} { l } { \displaystyle - \frac { \partial } { \partial u } \bigl \{ a _ { n } ^ { 2 } \mathrm { s e c h } ^ { 2 } ( a _ { n } \sqrt { u } ) \mathrm { s e c h } ^ { 2 } ( a _ { n } \sqrt { v } ) \bigr \} } \\ { \displaystyle \ } \\ { \displaystyle = a _ { n } ^ { 4 } \frac { \mathrm { t a n h } \bigl ( a _ { n } \sqrt { u } \bigr ) } { a _ { n } \sqrt { u } } \mathrm { s e c h } ^ { 2 } ( a _ { n } \sqrt { u } ) \mathrm { s e c h } ^ { 2 } ( a _ { n } \sqrt { v } ) } \\ { \displaystyle \ \geq \frac { e ^ { - 4 L } } { 4 ( 1 + L ) } . } \end{array}
$$

Hence $\partial _ { u } \Sigma _ { g } ( \sqrt { u } , \sqrt { v } ) \leq - e ^ { - 4 L } / \{ 8 ( 1 + L ) \}$ , and symmetrically in $v ,$ whenever

$$
8 C \sigma _ { n } ^ { 4 } ( 1 + L ) ^ { 9 } e ^ { 4 L + \sigma _ { n } ^ { 2 } L ^ { 2 } } \leq 1 .
$$

This condition holds at $L = L _ { \operatorname* { m a x } } = ( 1 - \delta ) \log ( 1 / \sigma _ { n } )$ for sufficiently small $\sigma _ { n }$ , since $\sigma _ { n } ^ { 4 } e ^ { 4 L _ { \mathrm { m a x } } } =$ $\sigma _ { n } ^ { 4 \delta } \  \ 0$ . Integrating the derivative bounds in one coordinate between $\lambda _ { 1 } ^ { 2 }$ and $\lambda _ { 2 } ^ { 2 }$ gives both quantitative gaps and hence the strict hierarchy. □

## F.5 CONCENTRATION BOUNDS

## F.5.1 SOFTMAX CONCENTRATION

Lemma F.17 (Concentration of softmax). Let $Y _ { 1 } , \dots , Y _ { n }$ be independent copies ofa mean-zero 1- sub-Gaussian variable $Y , a n d \# x \lambda \in \mathbb { R }$ and an integer $k \geq 1$ . There are constants $\hat { C } ( \lambda ) , C _ { 2 } ( \lambda ) > 0$ depending only on $\lambda , k ,$ such that, $f o r 0 < M \leq 1$ and $R \geq 1$ with $\mathbb { P } ( | Y | > R ) \le \dot { M } ^ { 2 } C _ { 2 } ( \lambda )$

$$
\mathbb { P } \left( \left| \frac { \sum _ { i = 1 } ^ { n } e ^ { \lambda Y _ { i } } Y _ { i } ^ { k } } { \sum _ { i = 1 } ^ { n } e ^ { \lambda Y _ { i } } } - \frac { \mathbb { E } [ Y ^ { k } e ^ { \lambda Y } ] } { \mathbb { E } [ e ^ { \lambda Y } ] } \right| \geq M C ( \lambda ) \right) \leq 2 n e ^ { - R ^ { 2 } / 2 } + 4 \exp \left( - \frac { n M ^ { 2 } } { 3 2 R ^ { 2 k } e ^ { 2 R | \lambda | } } \right) .
$$

Proof. Set $A _ { i } = Y _ { i } ^ { k } e ^ { \lambda Y _ { i } } \mathbf { 1 } _ { \{ | Y _ { i } | \leq R \} }$ and $B _ { i } = e ^ { \lambda Y _ { i } } \mathbf { 1 } _ { \{ | Y _ { i } | \leq R \} }$ , and write $\begin{array} { r } { A _ { n } = n ^ { - 1 } \sum _ { i } A _ { i } , B _ { n } = } \end{array}$ $\begin{array} { r } { n ^ { - 1 } \sum _ { i } B _ { i } , A ^ { * } = \dot { \mathbb { E } } [ Y ^ { k } e ^ { \lambda \dot { Y } } ] , } \end{array}$ and $B ^ { * } = \mathbb { E } [ e ^ { \lambda Y } ] \geq 1$ . The $A _ { i }$ are independent, as are the $B _ { i } ,$ and satisfy $| { \bf \bar { \cal A } } _ { i } | \le { \cal R } ^ { k } e ^ { { \cal R } | \lambda | }$ and $0 \leq B _ { i } \leq e ^ { R | \lambda | }$ . The second moments of $Y ^ { k } e ^ { \lambda Y }$ and $e ^ { \lambda Y }$ are bounded in terms of $\lambda , k$ by the sub-Gaussian assumption. Cauchy–Schwarz and a sufficiently small $C _ { 2 } ( \lambda )$ give $| \mathbb { E } A _ { n } - A ^ { * } | , | \dot { \mathbb { E } } B _ { n } - B ^ { * } | \le M / 4$ . Hoeffding’s inequality therefore gives $| A _ { n } - { \dot { A } } ^ { * } | , | B _ { n } - { \dot { B } } ^ { * } | \leq$ $M / 2 ,$ , except with probability at most 4 $\mathsf { l e x p } ( \bar { - } n M ^ { 2 } / ( 3 2 \bar { R } ^ { 2 k } e ^ { 2 R | \lambda | } ) )$ . On this event, $B _ { n } \geq 1 / 2$ , so

$$
\left| \frac { A _ { n } } { B _ { n } } - \frac { A ^ { * } } { B ^ { * } } \right| \leq \frac { | A _ { n } - A ^ { * } | } { B _ { n } } + \frac { | A ^ { * } | | B _ { n } - B ^ { * } | } { B _ { n } B ^ { * } } \leq M ( 1 + | A ^ { * } | ) .\tag{38}
$$

The sub-Gaussian moment bound allows $C ( \lambda ) > 1 + | A ^ { * } |$ to be chosen depending only on $\lambda , k$ Finally, $\mathbb { P } ( \operatorname* { m a x } _ { i } | Y _ { i } | > R ) \le 2 n e ^ { - R ^ { 2 } / 2 }$ , and outside this event the truncated and original ratios agree. □

## G ADDITIONAL RESULTS

Lemma G.1 (Lyapunov exit time). Consider ${ \dot { x } } = f ( x )$ and a region R. Suppose W is locally Lipschitz, $W ( x ) > C$ on R, and $\begin{array} { r } { \frac { d } { d t } W ( x ( t ) ) \leq - \gamma < 0 } \end{array}$ almost everywhere while $x ( t ) \in R$ . If $x ( 0 ) \in R$ and the solution exists up to exit, then its first exit time is at most

$$
t ^ { * } = \frac { W ( x ( 0 ) ) - C } { \gamma } .
$$

Moreover, $W ( x ( t ) ) \leq W ( x ( 0 ) )$ until exit.

Proof. While the trajectory remains in $R ,$

$$
C < W ( x ( t ) ) \leq W ( x ( 0 ) ) - \gamma t .
$$

At $t = t ^ { * }$ these inequalities contradict each other.

Theorem G.2 (Tikhonov’s theorem). Consider

$$
\varepsilon { \dot { r } } = f ( r , y ) , \qquad { \dot { y } } = g ( r , y ) , \qquad ( r ( 0 ) , y ( 0 ) ) = ( r _ { 0 } , y _ { 0 } ) .
$$

Let $\mathcal { N } _ { 0 }$ be open and $r ^ { \star }$ be a twice continuously differentiablefunction on $\mathcal { N } _ { 0 }$ with $f ( r ^ { \star } ( y ) , y ) = 0$ and graph contained in an open set ${ \mathcal { N } } .$ Assume that, for every $( r , y ) \in \mathcal { N }$ with $y \in { \mathcal { N } } _ { 0 } ,$ , the line segment from $( r ^ { \star } ( y ) , y )$ to $( r , y )$ lies in ${ \mathcal { N } } .$ . Assume that $f , g$ and their derivatives through order two are uniformly bounded in ${ \dot { \mathcal { N } } } ,$ , and that all eigenvalues of $\bar { \cdot } \partial _ { r } f ( r ^ { \star } ( y ) , y )$ have real parts at most $- a _ { 0 } < 0 ,$ , uniformly for $y \in \mathcal { N } _ { 0 }$ . Let

$$
\dot { y } ^ { 0 } = g ( r ^ { \star } ( y ^ { 0 } ) , y ^ { 0 } ) , \qquad y ^ { 0 } ( 0 ) = y _ { 0 } ,
$$

and let $L _ { g } > 0$ be a Lipschitz constant $f o r y \mapsto g ( r ^ { \star } ( y ) , y )$ on ${ \mathcal { N } } _ { 0 } .$ . There exist $\varepsilon _ { 0 } , c _ { 0 } , \ldots , c _ { 4 } , \kappa _ { 0 } > 0$ such that, for $0 < \varepsilon < \varepsilon _ { 0 }$ and $( r _ { 0 } , y _ { 0 } ) \in \mathcal { N }$ with $\| r _ { 0 } - r ^ { \star } ( y _ { 0 } ) \| _ { 2 } < c _ { 0 }$

$$
\| r ( s ) - r ^ { \star } ( y ( s ) ) \| _ { 2 } \le c _ { 1 } \varepsilon + c _ { 2 } \| r _ { 0 } - r ^ { \star } ( y _ { 0 } ) \| _ { 2 } e ^ { - \kappa _ { 0 } s / \varepsilon } ,
$$

$$
\begin{array} { r } { \| y ( s ) - y ^ { 0 } ( s ) \| _ { 2 } \le c _ { 3 } \varepsilon e ^ { L _ { g } s } + c _ { 4 } \| r _ { 0 } - r ^ { \star } ( y _ { 0 } ) \| _ { 2 } e ^ { - \kappa _ { 0 } s / \varepsilon } , } \end{array}
$$

for all $s \geq 0$ such that $( r ( u ) , y ( u ) ) \in \mathcal { N } a n d y ( u ) , y ^ { 0 } ( u ) \in \mathcal { N } _ { 0 } f o r 0 \le u \le s .$

Proof. See Berglund (2001, Theorem 4.1.2).

## H LOWER BOUND FOR THE GAUSSIAN MIXTURE

## H.1 GAUSSIAN AND UNIFORM BENCHMARKS

Proposition H.1 (Gaussian and uniform benchmarks). For the variance-one Gaussian location family, $R _ { { \mathrm G } , n } ^ { \star } = n ^ { - 1 }$ , attained by the sample mean. For the uniform location family, $R _ { \mathrm { U } , n } ^ { \star } \asymp n ^ { - 2 }$ and the sample mid-range has risk

$$
\mathbb { E } _ { \mu , \mathrm { U } } \left[ \left( \frac { X _ { ( 1 ) } + X _ { ( n ) } } { 2 } - \mu \right) ^ { 2 } \right] = \frac { 6 } { ( n + 1 ) ( n + 2 ) } .
$$

The sample mean instead has risk $n ^ { - 1 }$ under eitherfamily.

Proof. See (Lehmann & Casella, 1998, Theorem 3.5) or (Shao, 2003, Section 2).

## H.2 GAUSSIAN-MIXTURE BENCHMARK

Lemma H.2 (Fisher information and lower bounds for the location). For known $\sigma _ { n } \in ( 0 , 1 )$ , the observations have density

$$
f _ { \mu } ( x ) = \frac { 1 } { \sqrt { 2 \pi } \sigma _ { n } } \exp \left( - \frac { ( x - \mu ) ^ { 2 } + 1 - \sigma _ { n } ^ { 2 } } { 2 \sigma _ { n } ^ { 2 } } \right) \cosh \left( \frac { \sqrt { 1 - \sigma _ { n } ^ { 2 } } ( x - \mu ) } { \sigma _ { n } ^ { 2 } } \right) ,\tag{39}
$$

and the Fisher information in one observation is

$$
I _ { \sigma _ { n } } = \frac { 1 } { \sigma _ { n } ^ { 2 } } - \frac { 1 - \sigma _ { n } ^ { 2 } } { \sigma _ { n } ^ { 4 } } \mathbb { E } \left[ \mathrm { s e c h } ^ { 2 } \left( \frac { \sqrt { 1 - \sigma _ { n } ^ { 2 } } ( X - \mu ) } { \sigma _ { n } ^ { 2 } } \right) \mid \mu \right] .\tag{40}
$$

As $\sigma _ { n } \downarrow 0 ,$

$$
I _ { \sigma _ { n } } = \frac { 1 } { \sigma _ { n } ^ { 2 } } \left[ 1 - \sqrt { \frac { \pi } { 2 } } \frac { \sqrt { 1 - \sigma _ { n } ^ { 2 } } } { \sigma _ { n } } e ^ { - ( 1 - \sigma _ { n } ^ { 2 } ) / ( 2 \sigma _ { n } ^ { 2 } ) } \left( 1 + \mathcal { O } ( \sigma _ { n } ^ { 2 } ) \right) \right] .\tag{41}
$$

By the Cramér–Rao inequality, every unbiased estimator of µ based on n i.i.d. observations satisfies

$$
\begin{array} { r l r } {  { \operatorname { V a r } _ { \mu } ( \widehat { \mu } ) \geq \frac { 1 } { n I _ { \sigma _ { n } } } \sim \frac { \sigma _ { n } ^ { 2 } } { n } \qquad } } & { { } a s \sigma _ { n } \downarrow 0 . } \\ & { } & { \qquad \quad R _ { \mathrm { M } , n } ^ { \star } \geq \frac { \sigma _ { n } ^ { 2 } } { n } . \qquad } \end{array}\tag{42}
$$

Proof. Differentiating (39) with respect to µ gives

$$
\begin{array} { c l l } & { \displaystyle \frac { \partial } { \partial \mu } \log f _ { \mu } ( x ) = \frac { x - \mu - \sqrt { 1 - \sigma _ { n } ^ { 2 } } \operatorname { t a n h } ( \sqrt { 1 - \sigma _ { n } ^ { 2 } } ( x - \mu ) / \sigma _ { n } ^ { 2 } ) } { \sigma _ { n } ^ { 2 } } , } \\ & { \displaystyle \frac { \partial ^ { 2 } } { \partial \mu ^ { 2 } } \log f _ { \mu } ( x ) = - \frac { 1 } { \sigma _ { n } ^ { 2 } } + \frac { 1 - \sigma _ { n } ^ { 2 } } { \sigma _ { n } ^ { 4 } } \operatorname { s e c h } ^ { 2 } \left( \frac { \sqrt { 1 - \sigma _ { n } ^ { 2 } } ( x - \mu ) } { \sigma _ { n } ^ { 2 } } \right) . } \end{array}
$$

Thus $\begin{array} { r } { I _ { \sigma _ { n } } = - \mathbb { E } [ \frac { \partial ^ { 2 } } { \partial \mu ^ { 2 } } } \end{array}$ log $f _ { \mu } ( X ) \mid \mu ]$ , proving (40).

Substituting $u = \sqrt { 1 - \sigma _ { n } ^ { 2 } } ( x - \mu ) / \sigma _ { n } ^ { 2 }$ gives

$$
\begin{array} { r l } & { \mathbb { E } \left[ \mathrm { s e c h } ^ { 2 } \Bigg ( \frac { \sqrt { 1 - \sigma _ { n } ^ { 2 } } ( X - \mu ) } { \sigma _ { n } ^ { 2 } } \Bigg ) \mid \mu \right] = \frac { \sigma _ { n } } { \sqrt { 1 - \sigma _ { n } ^ { 2 } } \sqrt { 2 \pi } } e ^ { - ( 1 - \sigma _ { n } ^ { 2 } ) / ( 2 \sigma _ { n } ^ { 2 } ) } \int _ { \mathbb { R } } e ^ { - \sigma _ { n } ^ { 2 } u ^ { 2 } / ( 2 ( 1 - \sigma _ { n } ^ { 2 } ) ) } \mathrm { s e c h } u d u } \\ & { \qquad = \sqrt { \frac { \pi } { 2 } } \frac { \sigma _ { n } } { \sqrt { 1 - \sigma _ { n } ^ { 2 } } } e ^ { - ( 1 - \sigma _ { n } ^ { 2 } ) / ( 2 \sigma _ { n } ^ { 2 } ) } \left( 1 + \mathcal { O } ( \sigma _ { n } ^ { 2 } ) \right) . } \end{array}
$$

The last equality follows from $\int _ { \mathbb { R } }$ sech u $\iota d u = \pi , \int _ { \mathbb { R } } u ^ { 2 }$ sech u du $< \infty ,$ and $0 \leq 1 - e ^ { - t } \leq t$ for $t \geq 0$ . Thus, we obtain (41).

For the last bound, reveal the mixture component labels and subtract the corresponding offsets, leaving n independent $\textstyle { \mathcal { N } } ( \mu , \sigma _ { n } ^ { 2 } )$ observations conditional on $\mu .$ Revealing the labels cannot increase the location-minimax risk:

$$
R _ { \mathrm { M } , n } ^ { \star } \geq \operatorname* { i n f } _ { \widetilde { \mu } } \operatorname* { s u p } _ { \mu \in \mathbb { R } } \mathbb { E } _ { { \mathbf { X } } \sim \mathcal { N } ( \mu , \sigma _ { n } ^ { 2 } ) ^ { \otimes n } } [ ( \widetilde { \mu } ( { \mathbf { X } } ) - \mu ) ^ { 2 } ] = \frac { \sigma _ { n } ^ { 2 } } { n } .
$$

See Proposition H.1.

## I ADDITIONAL EXPERIMENTS

Joint-training setup. We provide further training details for Figures 2 and 3. The joint-training experiments below use the same setup unless stated otherwise. We train the architecture of Definition 2.2 with both softmax MoE and GLU gates, updating θ and λ jointly while keeping ω fixed. We use $m = 8 0 0 0$ fixed training tasks with contexts of length $n = 1 5 0$ from Definition 2.1, with $\sigma _ { \mu } ^ { 2 } = 0 . 5$ $K = 2$ , and $d = 1 0$ . Each gradient-descent step uses all m training tasks, with constant step sizes 20 for θ and 100 for λ, for 600,000 steps. We initialize $\theta _ { k } ( 0 ) \sim \mathcal { N } ( \breve { 0 } , 0 . 2 5 ^ { 2 } I _ { d } )$ and sample the fixed gating queries $\omega _ { r } \sim \mathcal { N } ( 0 , 1 )$ ) independently. We draw the initial expert parameters independently and uniformly from [0, 1.9], rejecting draws with separation below 0.6 and ordering them by magnitude. Each joint-training experiment uses ten independent runs and 200 validation tasks per family in each run. In the joint-training plots, experts are relabeled so that expert 1 is assigned to Gaussian contexts at the end of training. Curves show medians, and shaded regions span the minimum and maximum across runs.

## I.1 JOINT TRAINING, GAUSSIAN-MIXTURE ALTERNATIVE

Figures 4 and 5 repeat Figures 2 and 3 with the Gaussian-mixture alternative and $\sigma _ { n } ^ { 2 } = 0 . 0 5$ . The gates move toward $\bar { w } \approx I _ { 2 }$ and $| \lambda _ { 1 } |$ decreases to zero, as with the uniform alternative. Unlike with the uniform alternative, $| \lambda _ { 2 } |$ settles near 2, above the stopping target $\log ( 1 / \sigma _ { n } ^ { 2 } ) / ( 4 \sqrt { 1 - \sigma _ { n } ^ { 2 } } ) \approx 0 . 7 7$ of Lemma 3.4. On Gaussian-mixture contexts, both architectures improve on the sample mean but remain above the component-mean error and the lower bound of Lemma H.2. The component mean assigns observations to components according to their position relative to the sample mean, subtracts the corresponding offset $\pm \sqrt { 1 - \sigma _ { n } ^ { 2 } }$ , and averages the resulting values.

![](images/9013e624587074c07507f4682e448be23ddcfe19486e8fc99768fbdcd1d4b44d.jpg)

![](images/dee17960d5c8265a8522c0945a1bd3747fd4e7a1555b9472d7d82bc0e1e57c39.jpg)

![](images/e0008c8d18601e81ed2b7d9cee85fa1f78659be3b3e1a66e11ab3a46676682e2.jpg)  
Figure 4: Softmax MoE, joint training with the Gaussian-mixture alternative, as in Figure 2. The dashed line in the middle panel marks log $( 1 / \sigma _ { n } ^ { 2 } ) / ( 4 \sqrt { 1 - \sigma _ { n } ^ { 2 } } )$ .

![](images/73ef930a4033279ec6488419b7e878d3bc19e2dbf241fdd8ec32e368983f4de8.jpg)

![](images/e9ad18f6d2e70ffe3b9c057b4c3224834a1f74947e29ccbefd0947e93d243784.jpg)

![](images/c54c41c28da2cd1394e9794f4251679ed3f5f50546226aef3117ec7920238f9a.jpg)  
Figure 5: GLU, joint training with the Gaussian-mixture alternative, as in Figure 3. The dashed line in the middle panel marks log $( 1 / \sigma _ { n } ^ { 2 } ) / ( 4 \sqrt { 1 - \sigma _ { n } ^ { 2 } } )$

## I.2 JOINT TRAINING: GATING DYNAMICS

Figures 6 and 7 show the GLU gates and normalization error during joint training. After an initial transient, the normalization error falls to about $1 0 ^ { - 2 }$ before substantial family-specific selection develops. The gates then move toward $\bar { w } \approx I _ { 2 }$ , with less complete selection and greater variation across runs for the uniform alternative.

![](images/8ad94775ba8b92d77dd53ff750a580dc9e599fd0d63bbb7a52d1a04da4ac5de1.jpg)

![](images/e89caa52e661ef92f175d701e95231ffcfc07b9080be81ddb5de86936ebfd328.jpg)  
Figure 6: GLU, joint training with the uniform alternative: population gates $\bar { w } _ { k j }$ (left) and normalization error $| \bar { w } _ { 1 j } + \bar { w } _ { 2 j } - 1 |$ (right). Curves show medians, and shading spans all ten runs.

![](images/1a813a78a880244aa2c9d1e4f30819bccfd40fa4aae50ee40177eed9575e8041.jpg)

![](images/c211ff96b20bdc3d571a3715b708f117435ee484340fbbbd66d027f96f5da45b.jpg)  
Figure 7: GLU, joint training with the Gaussian-mixture alternative, with panels and conventions as in Figure 6.

## I.3 STAGEWISE TRAINING

We train the GLU with Definition 2.3, using the uniform alternative with $\sigma _ { \mu } ^ { 2 } = 0 . 0 5$ and $n = 5 0 0$ . The values of $m , K , d ,$ the initialization, and the numbers of runs and validation tasks are as in the jointtraining setup above. In the first stage, we minimize $\hat { \mathcal { L } } _ { m }$ over θ using 20,000,000 gradient-descent steps with step size 100. In the second stage, we minimize $\hat { \mathcal { L } } _ { m }$ over λ using 300,000 gradient-descent steps with step size 1000.

First stage: critical manifold. Figure 8 shows $( \bar { w } _ { 1 j } , \bar { w } _ { 2 j } )$ during the first stage for Gaussian $( j = \mathrm { G } )$ and uniform $( j = \mathrm { U } )$ contexts. The population gates first approach $\bar { w } _ { 1 j } + \bar { w } _ { 2 j } \approx 1$ , before substantial specialization toward $e _ { 1 }$ and $e _ { 2 }$ for Gaussian and uniform contexts, respectively.

![](images/4ca49b2de73fc2740ae3bf703ed1bfadeb7fa8b921be1f5742550e86b05f18ae.jpg)

![](images/6c8f225cc978765a80b31a2cfeb523d369a21070971cf9ed0b6ecb856b32cd40.jpg)  
Figure 8: First stage with the uniform alternative: $( \bar { w } _ { 1 j } , \bar { w } _ { 2 j } )$ for Gaussian (left) and uniform (right) contexts across runs, colored by iteration. The dashed line is the critical manifold $\bar { w } _ { 1 j } + \bar { w } _ { 2 j } = 1$

Per-expert accuracy. Figure 9 shows $\mathbb { E } _ { j } [ ( g _ { k } - \mu ) ^ { 2 } ]$ for both experts and $\mathbb { E } _ { j } [ ( \widehat { \mu } - \mu ) ^ { 2 } ]$ on validation tasks. In the first stage, λ is fixed, so the expert errors are constant and only the gating changes the error of ${ \widehat { \mu } } .$ . On Gaussian contexts, expert 1 and $\widehat { \mu }$ stay near the sample-mean error. On uniform contexts, the errors of expert 2 and $\widehat { \mu }$ decrease further in the second stage toward the sample mid-range error. Although expert 2 becomes worse on Gaussian contexts during this stage, the error of $\widehat { \mu }$ remains approximately unchanged since $\bar { w } _ { \mathrm { 2 G } } \approx 0$ after the first stage. The Gaussian-mixture results are in Section I.4.

![](images/edbdb8c6208ee5618b037726b4e1dd921500280d489a94f809eaab45944a1218.jpg)

![](images/dde9e9b953ff9d9c4fd206c50579166eafb7952b18429360dccee236554d2093.jpg)  
Figure 9: Uniform alternative: validation error of each expert $g _ { k }$ and of $\widehat { \mu }$ on Gaussian (left) and uniform (right) contexts, with the sample mean and sample mid-range as references. The vertical line marks $t _ { \theta }$

## I.4 STAGEWISE TRAINING, GAUSSIAN-MIXTURE ALTERNATIVE

We use the same Gaussian-mixture alternative as in the joint-training experiments, with the stagewise setup of Section I.3. Figure 10 repeats the uniform errors for comparison with Figure 12. In both cases, training the experts in the second stage reduces the alternative-family error while leaving the Gaussian error approximately unchanged. For the mixture, the error remains above the componentmean error and the lower bound of Lemma H.2. Figure 11 shows the first-stage gates moving toward the appropriate expert for each family, with a residual normalization error as in Figure 8.

![](images/d0bfd7b7b3d488833f73c09ed84720ccc1bca2064e5467a0f94146de567fd7b4.jpg)

![](images/471bd075ea09ba7796dc3864428ebbecd51dacbecfce3b8e4c35f8e6c30c22e0.jpg)  
Figure 10: Uniform alternative: validation error of each expert $g _ { k }$ and of $\widehat { \mu }$ on Gaussian (left) and uniform (right) contexts, repeated for comparison with Figure 12. The vertical line marks t<sub>θ</sub>.

![](images/56e2a23859321a0273c9207abf1a79e04fd563fcd578c08c02d91b7b37eb4aa8.jpg)

![](images/dffac9c6be828a6d0e309a1df0c783806a75d49bd973cb3d39053d03b291bcca.jpg)  
Figure 11: First stage with the Gaussian-mixture alternative: $( \bar { w } _ { 1 j } , \bar { w } _ { 2 j } )$ for Gaussian (left) and Gaussian-mixture (right) contexts across runs, colored by iteration. The dashed line is the critical manifold $\bar { w } _ { 1 j } + \bar { w } _ { 2 j } = 1$

![](images/af065346adb1417e9277ad900688fc4ce2126851dd260d429596df47d5b53b70.jpg)

![](images/132ff5ef904df60326eff9efcdbdba4a27879075135a9ba516ddf567b0e21e58.jpg)  
Figure 12: Gaussian-mixture alternative: validation error of each expert $g _ { k }$ and of $\widehat { \mu }$ on Gaussian (left) and Gaussian-mixture (right) contexts. The vertical line marks $t _ { \theta }$

## I.5 RELAXED ARCHITECTURE

We train ten attention parameters, whose attention outputs serve both as gating features and as inputs to the experts. The attention outputs are not centered or normalized. Each expert and each gate logit is a learned linear function of the ten attention outputs. We use the uniform alternative and train with SGD with $n = 2 0 0 , m = 8 , 0 0 0$ , learning rate 10, $K = 1 0$ , and $\sigma _ { \mu } ^ { 2 } = 0 . 1$ , and train all parameters jointly for 15,000 epochs. Each experiment uses five independent runs and 2,000 held-out validation tasks. Figure 13 shows end-to-end training of this less constrained architecture. The Gaussian validation error stays near the sample-mean error, while the uniform validation error falls well below it toward the sample mid-range error.

![](images/079c35cec9cd94346440d0bdfaadcbba93177bbaa13023efcc4364eb5c0863f8.jpg)

![](images/dc984b955e6c8ca14ae72fc4ee8a0a4991b7c0be46d71a90aa02b05d63cbc6bb.jpg)

![](images/364e88db27117be9475b5d29759a3166ab62cab67d94217c5d576e4db33d955e.jpg)  
Figure 13: Less constrained architecture, end-to-end training with the uniform alternative. Left: training loss. Middle and right: validation error on Gaussian and uniform contexts, with the sample mean and sample mid-range as references.

## J ROLE OF LAYERNORM

LayerNorm enforces the condition $Z _ { \mathrm { G A } } < \operatorname* { m i n } \left\{ Z _ { \mathrm { G G } } , Z _ { \mathrm { A A } } \right\}$ used in Lemma D.8. In our experiments, convergence is also observed without LayerNorm. Figure 14 illustrates the reduced flow studied in that lemma with and without LayerNorm. For the convergence argument, it would suffice to establish convergence to the optimal assignment from a region containing the reduced initial states at $t _ { \mathrm { c r i t } }$ , with a margin for the approximation errors. Since the gates are initialized near $1 / 2 ,$ a neighborhood of $( \bar { w } _ { 1 \mathrm { G } } , \bar { w } _ { 1 \mathrm { A } } ) = \bar { ( 1 / 2 , 1 / 2 ) }$ containing these states could suffice. The stream plot suggests convergence from such a central region without LayerNorm. Under the strict expert hierarchy and $Z _ { \mathrm { G A } } > Z _ { \mathrm { A A } } > 0$ , the reduced flow nevertheless has a region converging to the suboptimal corner $( 0 , 0 )$ : sufficiently close to this corner, when $\bar { w } _ { \mathrm { 1 G } } / \bar { w } _ { \mathrm { 1 A } }$ is sufficiently small, both coordinates and their ratio decrease, keeping the trajectory in this region and forcing both coordinates to zero. This does not preclude convergence from the central region.

![](images/17eb78f1dbc9d0dc1dd1b43aab4b7118bdc834fab362429187d67af9f7c1232a.jpg)

![](images/33d26047773a8320d949654d223f2d0377d30ed3ad21a09c3615a005f9dc0514.jpg)  
Figure 14: Stream plots of the reduced flow studied in Lemma D.8. Left: population features with LayerNorm. Right: uncentered population features at $\mu = 0$ without LayerNorm.

## K EXTENDED RELATED WORK

In-context learning of statistical procedures. Prior-data fitted networks (PFNs) train transformers on synthetic tasks to approximate Bayesian prediction in context (Müller et al., 2024). Their adaptation to tabular data has produced models that perform strongly on standardized benchmarks, particularly for small and medium datasets (Hollmann et al., 2023; 2025; Erickson et al., 2025; Ye et al., 2025). Related pretrained models estimate heterogeneous treatment effects, survival curves, mutual information, and Poisson means (Balazadeh et al., 2025; Qi et al., 2026; Peyrard & Cho, 2025; Teh et al., 2025), and amortized simulation-based inference likewise maps simulated datasets to parameters (Cranmer et al., 2020). Most of these systems target prediction at a query point. These works demonstrate that pretraining can produce effective predictors and parameter estimators. Our focus is on how gradient-based pretraining learns an estimator with rates that depend on the unknown family.

Statistical theory of pretrained predictors. Constructive and empirical analyses study which algorithms trained transformers implement in context (Akyürek et al., 2023; Chen et al., 2024b; Li et al., 2024). Bai et al. (2023) construct transformers that select among base algorithms, either by testing summary statistics of the context or by validating candidates on a held-out split, with guarantees for the empirical risk minimizer. Kim et al. (2024) show that a deep network followed by linear attention attains minimax rates for nonparametric regression over a known function class, and argue that representations fixed after pretraining limit adaptivity. Wakayama & Suzuki (2025) and Ma et al. (2025) analyze pretrained transformers as approximations to Bayesian posterior predictors (see also Nagler, 2023). In particular, Ma et al. (2025) use posterior contraction to obtain rates that adapt to unknown smoothness and effective dimension. Cannella et al. (2026) establish near-optimal regret for an idealized pretrained Bayes estimator in Poisson empirical Bayes, with concurrent guarantees for nonparametric Bayes estimators in the Gaussian case (Ignatiadis & Kankanala, 2026). These results show that the empirical risk minimizer, or an idealized Bayes predictor, can be adaptive. They do not follow the training trajectory, and they treat the adaptive map as a black box. In our model, adaptation is carried by an identifiable internal parameter, the attention scale of each expert, which moves the estimator across a family of estimators.

Training dynamics of attention. A complementary line analyzes how gradient-based training shapes attention on in-context tasks, including linear attention for linear regression (Zhang et al., 2024a;b), softmax attention (Huang et al., 2023), mean-field dynamics (Kim & Suzuki, 2024), and learning of low-dimensional structure (Oko et al., 2024). Closest to our analysis, Chen et al. (2024a) show that gradient flow on multi-head softmax attention allocates each head to one task of a multi-task linear regression, after warm-up and emergence phases. There, the task handled by each head is a fixed coordinate block determined by the task structure and the initialization. Here, the family varies across contexts and must be inferred from each sample, so specialization must be paired with learned routing. They also find that concentrated attention is suboptimal for noisy linear tasks, whereas the irregular uniform family calls for attention concentrated on the extremes. For the GLU, our gating dynamics separate into fast normalization and slow selection, which we analyze with Tikhonov’s theorem (Berglund, 2001).

Adaptive estimation. In statistics, an estimator is adaptive over a collection of models if, for each model, it attains the rate achievable when that model is known (Lepskii, 1991; Donoho & Johnstone, 1995; Tsybakov, 2009). Location estimation provides a classical parametric instance: for symmetric densities with finite Fisher information, the location can be estimated asymptotically as efficiently without knowing the shape as with it (Stein, 1956; Beran, 1974; Stone, 1975; Bickel, 1982; Bickel et al., 1993). Differences in rate between location families instead arise from irregularity: the uniform family has parameter-dependent support and admits the faster rate $n ^ { - 2 }$ (Ibragimov & Has’minskii, 1981). Classical adaptive location estimators estimate the score function and apply a one-step correction, whereas the procedure learned here selects a point in a family of exponentially tilted averages.

Set architectures, gating, and mixtures of experts. Deep Sets (Zaheer et al., 2017) and Set Transformers (Lee et al., 2019) provide architectures for functions of unordered data, and our model is a small Set Transformer specialized to scalar estimation. Its second layer is either a softmax mixture of experts (Jacobs et al., 1991; Shazeer et al., 2017) or a gated linear unit (Dauphin et al., 2017; Shazeer, 2020). Comparing the two isolates the effect of enforcing a statistical constraint, namely normalization of the expert weights and hence exact translation equivariance, versus learning it. Bu et al. (2025) analyze how gradient-based training learns to retrieve task concepts from context. They show that training on question answer data enables this behavior, whereas training on their in-context demonstration distributions can fail. Their results concern factual recall, while we study how learned routing and expert specialization yield family-specific estimation rates.