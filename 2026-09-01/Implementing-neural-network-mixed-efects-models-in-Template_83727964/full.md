# Implementing neural network mixed-efects models in Template Model Builder (TMB)

Nan Zheng<sup>∗1</sup>, Hoi Yiu Cheung<sup>1</sup>, Vibhu Sharma<sup>1</sup>, James T. Thorson<sup>2</sup>, and Noel G. Cadigan<sup>3</sup>

<sup>1</sup>Department of Mathematics and Statistics, Memorial University of Newfoundland, Canada

<sup>2</sup>Resource Ecology and Fisheries Management, Alaska Fisheries Science Center, National Marine Fisheries Service, National Oceanic and Atmospheric Administration, USA

<sup>3</sup>Centre for Fisheries Ecosystems Research, Fisheries and Marine Institute of Memorial University of Newfoundland, Canada

September 1, 2026

## Abstract

Neural network mixed-efects models (NMMs) have gained traction by combining the strong representation and predictive power of artificial neural networks with the capacity of mixed-efects modeling to capture complex correlation structures. However, existing estimation approaches rely heavily on manual derivations of objective functions and gradients, which inherently forces simplifying approximations and severely constrains the complexity and accuracy of NMMs. In this work, we introduce a general framework for implementing NMMs using Template Model Builder (TMB). By leveraging automatic diferentiation and Laplace approximation, TMB requires users to specify only the negative joint log-likelihood and any regularization terms. The framework automatically integrates out random efects and evaluates the marginal objective function alongside its exact gradients, eliminating the need for manual derivations or ad hoc approximations. We demonstrate the eficiency, flexibility, and statistical performance of TMB-based NMMs across two numerical examples, including an application to monotonic NMMs. Reproducible code is provided to facilitate broader adoption.

Keywords: neural network mixed-efects model; TMB; monotonic neural network; feedforward neural network; Laplace approximation; fish maturation model

## 1 Introduction

The development of technologies for implementing deep neural networks, that is, neural networks with multiple hidden layers, has significantly advanced the fields of artificial intelligence and machine learning (LeCun et al., 2015). Owing to their strong representational and predictive capabilities, neural networks have been increasingly adopted for analyzing conventional statistical datasets. Compared with modern deep learning applications, these datasets are typically much smaller, making relatively shallow network architectures a practical and commonly adopted choice (Hornik et al., 1989; Hastie et al., 2009; Goodfellow et al., 2016).

A standard feedforward neural network does not explicitly encode structural priors, such as spatial, temporal, or contextual dependencies, within its architecture. As a result, it must learn these relationships solely from the training data, leading to a hypothesis space that is often substantially larger than necessary for structured learning tasks (Bengio, 2009; Goodfellow et al., 2016). Subsequent developments have addressed this limitation by incorporating domain-specific correlation structures into network architectures. For example, convolutional neural networks exploit translation invariance and spatial hierarchies in image data, recurrent neural networks (RNNs) capture temporal dependencies in sequential data, and transformers model long-range contextual relationships in natural language (see, e.g., Chollet et al., 2022). Although these architectures substantially improve learning eficiency by embedding appropriate structural assumptions, they typically remain data-intensive and often require training data with specific structural characteristics that are not satisfied in many statistical applications. For example, RNNs are generally designed for regularly sampled sequential data and are most naturally applied to collections of sequences with comparable temporal structure, assumptions that are frequently violated by longitudinal datasets with irregular observation times or unequal sequence lengths (Mandel et al., 2023). In contrast, statistical mixed-efects models provide considerably greater flexibility for accommodating unbalanced longitudinal data, irregular observation schedules, and complex sampling designs while remaining efective with relatively small sample sizes. Motivated by the complementary strengths of these two paradigms, neural network mixed-efects models (NMMs) have emerged as a natural extension that combines the representational power of neural networks with the flexibility and interpretability of statistical mixed-efects models (Tandon et al., 2006; Tran et al., 2017; Mandel et al., 2023).

A NMM can be generally formulated as $\pmb { \mu } = g \{ f _ { N } ( \mathbf { X } ) + \mathbf { Z } ^ { \top } \mathbf { b } \}$ , where $f _ { N } ( { \mathbf { X } } )$ represents the output of a neural network with input X, Z is the design matrix associated with the random efects (REs) b, $\pmb { \mu }$ denotes the vector of parameters governing the observational model $( \mathrm { e . g . }$ , the mean vector), and $g ( \cdot )$ is an elementwise transformation analogous to the inverse link function in generalized linear mixed-efects models (GLMMs). Throughout this paper, boldface notation is used for vectors and matrices. The REs b are typically excluded from the neural network $f _ { N } ( { \mathbf { X } } )$ , as their inclusion may lead to prohibitive computational costs. Together with the observational model $f ( \mathbf { D } | \pmb { \mu } )$ for responses D, the RE prior $f ( \mathbf { b } )$ and the weight regularization term $f _ { \mathrm { r e g } } ( \mathbf { w } )$ , the joint objective function is

$$
f ( { \bf D } , { \bf b } ) = f ( { \bf D } | \mu ) f ( { \bf b } ) f _ { \mathrm { r e g } } ( { \bf w } ) .\tag{1}
$$

The weight regularization term penalizes excessively large neural network weights, thereby reducing overfitting and improving the generalization ability of the model (see, e.g., Krogh and Hertz, 1991). Before applying optimization algorithms, such as stochastic gradient descent, in deep learning, the REs b must be appropriately accommodated to obtain the objective function and its gradients. To this end, Tran et al. (2017) perform estimation in a Bayesian framework using variational approximation, whereas Mandel et al. (2023) replace the conditional likelihood function $f ( \mathbf { D } | \pmb { \mu } )$ with a quasi-likelihood function and integrate out b using the Laplace approximation to obtain the marginal objective function. In this work, we adopt the latter approach because of its computational simplicity and the availability of eficient modeling packages, such as the Template Model Builder (TMB;

Kristensen et al., 2016) within R (R Core Team, 2024), which automatically integrates out the REs from the joint likelihood and provides the marginal likelihood together with its gradients. By leveraging automatic diferentiation alongside sparse matrix techniques and parallel computing, TMB enables rapid likelihood-based inference even in high-dimensional settings, and ofers a highly competitive alternative to Bayesian methods in terms of speed, scalability, and model complexity (Thygesen et al., 2017).

Without leveraging modern computational frameworks such as TMB, Mandel et al. (2023) relied on manual derivations of the marginal objective function and its corresponding gradient to enable numerical optimization. This approach introduces several key limitations. First, as discussed previously, the quasi-likelihood function, rather than the full likelihood, must be used to facilitate manual derivation. Second, the approximation requires omission of a term involving the determinant of the covariance matrix of the REs and the Hessian matrix of the joint objective function with respect to the REs. Neglecting this term may reduce the accuracy and eficiency of inference for certain model parameters. Third, the analytical complexity increases substantially as the number of neural network layers grows, limiting the scalability of the approach (to two layers in Mandel et al., 2023) and making the derivation and implementation increasingly dificult and error-prone. Notably, Mandel et al. (2023) omitted the derivative of the RE estimators with respect to the model parameters when diferentiating the objective function (see, e.g., Eq. 7 in Kristensen et al., 2016), an omission that may lead to suboptimal optimization or less precise parameter estimates.

To address these limitations, we propose and demonstrate the use of TMB for implementing NMMs. Rather than using a quasi-likelihood approximation, we directly provide the joint objective function (1) to TMB. TMB then automatically and eficiently computes the Laplace approximation to the marginal objective function, together with its corresponding gradients, without omitting any terms. Consequently, the proposed approach improves computational eficiency, inferential accuracy, ease of implementation, and scalability relative to the manual derivation approach.

The remainder of this paper is organized as follows. Section 2 details the proposed NMM framework and its implementation using TMB. Section 3 replicates the simulation studies from Mandel et al. (2023), demonstrating the computational eficiency and superior accuracy of our approach. In Section 4, we apply both the NMM and GLMM to an American plaice maturity dataset, illustrating the ease of implementing monotonic neural networks within this framework. Finally, Section 5 ofers concluding remarks and directions for future research.

## 2 Methods

![](images/2479aa2f12e9d7e5106fbaddb4cca2f89b30105de66165d3d65fc212c47dc19c.jpg)  
Figure 1: The architecture of a multi-layer NMM.

Our model architecture is displayed in Fig. 1 and closely follows the feed-forward network framework of Mandel et al. (2023). Let $\mathbf { x } _ { i } ~ \in ~ \mathbb { R } ^ { p }$ denote the input vector for the ith observation and $\mu _ { i } \in \mathbb { R }$ be its corresponding univariate output. For a network with L hidden layers, where layer $l \in \{ 1 , \ldots , L \}$ consists of $d _ { l }$ units, each layer sequentially applies an afine transformation followed by an element-wise activation function $f _ { a } ( \cdot )$ (see, e.g., Chollet et al., 2022). Specifically, for the ith observation, the output of the first hidden layer, $\pmb { \alpha } _ { i } ^ { ( 1 ) } \in \mathbb { R } ^ { d _ { 1 } }$ , is expressed as

$$
\begin{array} { r } { \pmb { \alpha } _ { i } ^ { ( 1 ) } = f _ { a } \{ \mathbf { w } ^ { ( 1 ) } \mathbf { x } _ { i } + \pmb { \delta } ^ { ( 1 ) } \} , } \end{array}\tag{2}
$$

where $\mathbf { w } ^ { ( 1 ) } \in \mathbb { R } ^ { d _ { 1 } \times p }$ and $\pmb { \delta } ^ { ( 1 ) } \in \mathbb { R } ^ { d _ { 1 } }$ represent the weight matrix and bias vector, respectively. For each hidden layer $l \in \{ 2 , \ldots , L \}$ , the activation output is given by

$$
\begin{array} { r } { \pmb { \alpha } _ { i } ^ { ( l ) } = f _ { a } \{ \mathbf { w } ^ { ( l ) } \pmb { \alpha } _ { i } ^ { ( l - 1 ) } + \pmb { \delta } ^ { ( l ) } \} , } \end{array}\tag{3}
$$

with parameters $\mathbf { w } ^ { ( l ) } \in \mathbb { R } ^ { d _ { l } \times d _ { l - 1 } }$ and $\pmb { \delta } ^ { ( l ) } \in \mathbb { R } ^ { d _ { l } }$ . Finally, the univariate output of the neural network is related to the observational parameter $\mu _ { i }$ via

$$
\mu _ { i } ( \mathbf { b } _ { i } ) = g \{ \mathbf { w } \alpha _ { i } ^ { ( L ) } + \delta + \mathbf { Z } _ { i } ^ { \top } \mathbf { b } _ { i } \} ,\tag{4}
$$

where $\mathbf { w } \in \mathbb { R } ^ { 1 \times d _ { L } } , \delta \in \mathbb { R }$ , and $\mathbf { Z } _ { i }$ and $\mathbf { b } _ { i }$ denote the design matrix and RE vector associated with the ith observation, respectively. The REs $\mathbf { b } _ { i }$ are assumed to follow a multivariate normal (MVN) distribution, inducing dependence among the components of the ith observation. Dependence across observations can be introduced either by sharing one or more RE components among observations or by specifying a correlated MVN distribution for the REs $\mathbf { b } _ { i }$ across observations. In Mandel et al. (2023), $\mu _ { i }$ denotes the conditional mean of the response under an exponential-family distribution given $\mathbf { b } _ { i } .$ , which enables the use of a quasi-likelihood approximation to the true conditional likelihood. In contrast, our framework is not restricted to exponential-family distributions. The observational distribution can be specified more generally, and $\mu _ { i }$ need not represent the conditional mean; rather, it may denote any governing parameter of the observational distribution.

A comparison can also be made to the GLMM framework (Breslow and Clayton, 1993), in which the conditional mean is linked to the linear predictor via

$$
\mu _ { i } = g \{ \mathbf { x } _ { i } ^ { \top } { \boldsymbol { \beta } } + { \boldsymbol { \beta } } _ { 0 } + \mathbf { Z } _ { i } ^ { \top } \mathbf { b } _ { i } \} ,\tag{5}
$$

where $\beta$ represents the vector of fixed efects, $\mu _ { i }$ denotes the conditional mean of the ith observation given the REs, and $g ( \cdot )$ is the inverse link function. The NMM defined in Eqs. (2)–(4) replaces the fixed-efects portion of the linear predictor with the output of a feed-forward neural network. Because neural networks ofer significantly greater expressiveness than linear predictors, the NMM is expected to yield better representational and predictive capabilities compared to the standard GLMM.

Let there be n observations, and let $\mathbf { B } = ( \mathbf { b } _ { 1 } ^ { \top } , \ldots , \mathbf { b } _ { n } ^ { \top } ) ^ { \top }$ denote the collection of all REs. The marginal likelihood l for NMM defined by (2)–(4) is expressed as

$$
l ( \pmb \theta | \mathbf D _ { 1 } , \dots , \mathbf D _ { n } ) = \int \left[ \prod _ { i = 1 } ^ { n } f \{ \mathbf D _ { i } | \mu _ { i } ( \mathbf b _ { i } ) \} \right] f ( \mathbf B ) d \mathbf B ,\tag{6}
$$

where $\mathbf { D } _ { i }$ denotes the ith observation, $f \{ { \bf D } _ { i } \mid \mu _ { i } ( { \bf b } _ { i } ) \}$ is its conditional distribution given the REs, and $f ( \mathbf { B } )$ is the prior (marginal) distribution of B. The vector $\pmb \theta$ collects all model parameters, including the neural network weights and biases, observational model parameters, and prior distribution parameters. To evaluate the integral over the REs in Eq. (6), we employ TMB. TMB eficiently approximates the marginal likelihood via the Laplace approximation, scaling to tens of thousands of REs by exploiting the sparsity in their joint distribution with the observations.

The model parameters are estimated by minimizing the objective function

$$
- \log \{ l ( \pmb \theta | \mathbf D _ { 1 } , \dots , \mathbf D _ { n } ) \} + \lambda ( \mathbf W ^ { \top } \mathbf W + \pmb \Delta ^ { \top } \pmb \Delta ) ,\tag{7}
$$

where W and $\Delta$ denote the vectors of all neural network weights and biases, respectively, and $\lambda > 0$ is a regularization hyperparameter. The second term introduces $L _ { 2 }$ regularization, which penalizes larger parameter magnitudes to mitigate overfitting and enhance generalization. Following Mandel et al. (2023), we employ an $L _ { 2 }$ penalty to prioritize predictive accuracy over feature selection (Tibshirani, 1996; Goodfellow et al., 2016) while ensuring diferentiability.

## 3 Simulation study

We assess the proposed TMB implementation of the NMM by reproducing the nonlinear simulation experiment of Mandel et al. (2023), considering both a continuous and a binary response. In each case the network is provided only with the raw predictors and must recover a nonlinear signal from them, and predictive performance is assessed by leaveone-out cross-validation on the final observation. To facilitate direct comparison with the results of Mandel et al. (2023), we adopt the notation used in their simulation studies throughout this section.

## 3.1 Simulation design

For each of $m = 1 0 0$ subjects, the number of observations $n _ { i }$ is drawn from a Poisson $( 6 ) + 2$ distribution, ensuring at least one training and one validation observation per subject. For each observation we generate six independent binary predictors $\mathbf { x } _ { i j } = ( x _ { i j 1 } , \dots , x _ { i j 6 } ) ^ { \top }$ , each from a Bernoulli(0.5) distribution, and form three nonlinear features

$$
Z _ { i j 1 } = \mathbb { 1 } { \big ( } x _ { i j 1 } + x _ { i j 2 } = 1 { \big ) } , \quad Z _ { i j 2 } = \mathbb { 1 } { \big ( } x _ { i j 3 } + x _ { i j 4 } \neq 1 { \big ) } , \quad Z _ { i j 3 } = \mathbb { 1 } { \big ( } x _ { i j 5 } + x _ { i j 6 } = 1 { \big ) } ,\tag{8}
$$

which encode exclusive-or (XOR) type interactions that a linear predictor cannot represent. Writing $\mathbf { Z } _ { i j } = ( Z _ { i j 1 } , Z _ { i j 2 } , Z _ { i j 3 } ) ^ { \top }$ , the continuous response is generated as

$$
y _ { i j } = \mathbf { Z } _ { i j } ^ { \top } \beta + b _ { i } + \epsilon _ { i j } , \qquad \beta \sim \mathrm { M V N } ( 2 \cdot { \bf 1 } , { \bf I } ) ,\tag{9}
$$

and the binary response as

$$
\mathrm { l o g i t } \{ \mathrm { P } ( y _ { i j } = 1 ) \} = \mathbf { Z } _ { i j } ^ { \top } \boldsymbol { \beta } + b _ { i } + \epsilon _ { i j } , \qquad \boldsymbol { \beta } \sim \mathrm { M V N } ( 3 \cdot { \bf 1 } , { \bf I } ) ,\tag{10}
$$

where the subject-level random intercept is $b _ { i } \sim N ( 0 , \tau )$ and $\epsilon _ { i j } \sim N ( 0 , 0 . 0 5 ^ { 2 } )$ , and $\beta$ is drawn once per replicate. The variance component τ governs the strength of withinsubject correlation and is swept over {0, 0.1, 0.2, 0.3, 0.4, 0.5} for the continuous response and {0, 4, 8, 12, 16, 20} for the binary response, following Mandel et al. (2023). Crucially, the model is fit using only the raw predictors ${ \bf x } _ { i j } ;$ the features $\mathbf { Z } _ { i j }$ are never supplied, so the network must recover the XOR structure of (8) on its own.

Predictive accuracy is evaluated by leave-one-out cross-validation: the final observation of each subject is withheld, the model is fit to the remaining observations, and the withheld observation is predicted as $\hat { \mu } _ { i j } = g \{ \hat { f } _ { N } ( \mathbf x _ { i j } ) + \hat { b } _ { i } \}$ , combining the fitted network output with the subject’s estimated random intercept. Performance is summarized by the mean squared prediction error (MSPE) for the continuous response and by the area under the receiver operating characteristic curve (AUC) for the binary response, averaged across the 100 withheld observations. Each configuration is replicated 500 times.

## 3.2 Model and estimation

We implement the two-hidden-layer network of Mandel et al. (2023), denoted GNMM-2, with $d _ { 1 } = 1 0$ units in the first hidden layer and $d _ { 2 } = 5$ in the second, a six-dimensional input, and a scalar output, as in Eqs. (2)–(4). The activation function $f _ { a }$ is the logistic sigmoid, and the output transformation g is the identity for the continuous response and the inverse logit for the binary response. A single scalar random intercept enters the output layer for each subject $( \mathbf { Z } _ { i } \equiv 1$ in Eq. (4)), with prior $b _ { i } \sim N ( 0 , \tau )$ . The observational distribution is Gaussian with an estimated residual variance for the continuous response and Bernoulli for the binary response, as specified in Eqs. (9) and (10), respectively.

Estimation follows the objective (7) with regularization parameter $\lambda = 0 . 0 0 5$ , matching the value used for the two-hidden-layer network in Mandel et al. (2023). The random intercepts are integrated out by TMB via the Laplace approximation, and the resulting marginal objective is optimized using nlminb function in $\operatorname { R } ;$ standard deviations are parameterized on the log scale to enforce positivity, and λ is held fixed. In contrast to the manual quasi-likelihood derivation of Mandel et al. (2023), the TMB implementation requires only a specification of the joint objective (1), automatically deriving the marginal

objective and its gradients.

## 3.3 Results

Figures 2 and 3 present the predictive performance of the TMB implementation of GNMM-2 as a function of $\tau ,$ overlaid on the corresponding GNMM-2 results reported by Mandel et al. (2023). For the continuous response, the TMB MSPE is essentially flat at approximately 0.003 across all values of $\tau ,$ closely tracking the irreducible noise floor $\sigma _ { \epsilon } ^ { 2 } = 0 . 0 0 2 5 $ : the fitted network recovers the XOR signal almost exactly and the random intercept absorbs the within-subject correlation, leaving only observation noise. For the binary response, the TMB AUC rises from approximately 0.89 at $\tau = 0$ to 0.91 at $\tau = 2 0$ , increasing with $\tau$ because a larger random-intercept variance renders the subject-specific estimate $\hat { b } _ { i }$ more informative for the withheld observation.

In both settings the TMB implementation attains markedly better predictive performance than the GNMM-2 results reported by Mandel et al. (2023), whose MSPE is approximately 0.5 and whose AUC ranges from approximately 0.78 to 0.88.

![](images/652269d03e7c470d1f14c16f57853df336788d2274259fbe36dd150cdc94f4a3.jpg)

![](images/14e0817649e47efc7bb7210719d8a04cdd94767b591c21b22d99986849ca58be.jpg)  
Figure 2: Mean squared prediction error (MSPE) of the TMB implementation of GNMM-2 for the continuous response (solid, blue), as a function of the random-intercept variance $\tau _ { : }$ overlaid on the GNMM-2 results reported by Mandel et al. (2023) (grey dashed, read from their Figure 2). Both panels show the same two curves for the nonlinear data-generating mechanism and a first hidden layer of 10 units. Panel (a) uses the full y-axis range [0, 4] of Mandel et al. (2023): on this scale the TMB curve is pinned to the horizontal axis, clearly far below the reported GNMM-2 curve. Panel (b) shows the same data on a restricted $y -$ axis [0, 0.6], which makes the diference explicit: the TMB MSPE $( \approx 0 . 0 0 3 )$ lies just above the axis, well below the reported GNMM-2 MSPE (≈ 0.45–0.55).

![](images/97af31d2f052c44bfccf9565fdbfbfdd9d0a9aa9cb8e7fd34d6c8988d49b6f91.jpg)  
Figure 3: Area under the ROC curve (AUC) of the TMB implementation of GNMM-2 for the binary response (solid), as a function of the random-intercept variance τ , overlaid on the GNMM-2 results reported by Mandel et al. (2023) (grey dashed, read from their Figure 3). Both correspond to the nonlinear data-generating mechanism and a first hidden layer of 10 units; the axes match the scale of Mandel et al. (2023).

## 4 Real data example: maturity data for 3LNO American plaice

To illustrate the implementation of the NMM using real data, we fitted the model to maturity data for female American plaice (Hippoglossoides platessoides) collected during annual surveys conducted by Fisheries and Oceans Canada (DFO) in the Northwest Atlantic Fisheries Organization (NAFO) Divisions 3L, 3N, and 3O (3LNO) from 1978 to 2015. Fish born in the same year (i.e., belonging to the same cohort) are generally exposed to similar environmental conditions and other shared influences throughout their life history. Therefore, cohort (birth year), rather than survey year, was used as the temporal index. In total, the dataset comprises 58 cohorts spanning birth years from 1958 to 2015. We further partition the 58 cohorts into cohort segments (cohort seg) of eight consecutive cohorts each. Specifically, the first eight cohorts are assigned cohort seg 1, the next eight cohorts are assigned cohort seg 2, and so forth. Under this partitioning, the final cohort segment contains only three cohorts. To avoid such a small segment, we merge it with the penultimate cohort segment (cohort seg 7). Consequently, the data are divided into seven cohort segments, with the first six segments each containing eight consecutive cohorts and the seventh segment containing 11 consecutive cohorts.

The NMM consisted of two hidden layers, each containing twenty nodes. Input for the ith fish was an eight-dimensional vector comprising age as the first component, followed by a seven-dimensional one-hot encoded representation of the cohort segment (cohort seg number), where the active category was assigned a value of 1 and all remaining categories were set to 0. The activation function $f _ { a }$ in Fig. 1 is the logistic sigmoid. The output yields the conditional maturation probability for the ith fish given the REs, specifying Eq. (4) as

$$
\mathrm { P } _ { i } ( a _ { i } | \pmb \theta , b _ { c _ { i } } ) = \mathrm { i n v \mathrm { _ - } l o g i t } \{ \mathbf w \pmb \alpha _ { i } ^ { ( 2 ) } + \delta + b _ { c _ { i } } \} ,\tag{11}
$$

where $a _ { i }$ and $c _ { i }$ denote respectively the age and cohort index for the ith individual, inv logit(·) represents the inverse logit transformation function, and cohort-specific REs $( b _ { 1 } , \ldots , b _ { 5 8 } )$ are modeled using a first-order autoregressive [AR(1)] process across the 58 cohorts. Maturity was recorded as a binary response variable $( d _ { i } )$ , where 1 indicates a mature fish and 0 indicates an immature fish. Accordingly, the observational model $f \{ \mathbf { D } _ { i } | \mu _ { i } ( \mathbf { b } _ { i } ) \}$ in Eq. (6) takes the Bernoulli form: $f \{ d _ { i } | \mathrm { P } _ { i } \} = \{ \mathrm { P } _ { i } \} ^ { d _ { i } } \{ 1 - \mathrm { P } _ { i } \} ^ { 1 - d _ { i } }$ . The objective function is defined by Eqs. (6)-(7).

The maturation probability in Eq. (11) is constrained to be a non-decreasing function of age. This monotonicity constraint is imposed by generalizing the Lagrange multiplier method using the Karush-Kuhn-Tucker (KKT) conditions (see, e.g., Karush, 1939; Chen and Ye, 2022). Specifically, let

$$
h ( \pmb \theta ) = \operatorname* { s u p } _ { a , c } \left\{ - \frac { \partial \mathrm { P } ( a | \pmb \theta , b _ { c } ) } { \partial a } \right\} ,\tag{12}
$$

denote the minimum derivative of the maturity probability with respect to age a across all the ages and cohort segments (c). The generalized Lagrangian function is then formulated as

$$
\begin{array} { r } { \mathcal { L } ( \pmb { \theta } , \eta ) = - \log \lbrace l ( \pmb { \theta } | d _ { 1 } , \dots , d _ { n } ) \rbrace + \lambda ( \mathbf { W } ^ { \top } \mathbf { W } + \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } ^ { \top } \mathbf { \Delta } \mathbf { \Delta } ) + \eta h ( \pmb { \theta } ) , } \end{array}\tag{13}
$$

where $\eta \geq 0$ is the KKT multiplier corresponding to the inequality constraint. According to the KKT conditions, the optimization procedure begins by setting $\eta = 0$ to find the initial minimizer $\hat { \pmb { \theta } }$ of the generalized Lagrangian function (13) with respect to $\pmb { \theta } .$ The algorithm then evaluates whether the monotonicity constraint $h ( \hat { \pmb { \theta } } ) \leq 0$ is satisfied. If the condition is violated, $\eta$ is increased by a small increment and $\hat { \pmb { \theta } }$ is re-estimated; this process is repeated iteratively until the condition $h ( \hat { \pmb { \theta } } ) \leq 0$ holds. TMB eficiently computes the generalized Lagrangian function (13) and its gradient with respect to $\pmb \theta$ for optimization via nlminb function in R.

As a baseline for comparison, we consider a GLMM, in which the conditional maturation

probability for the ith fish, given the REs, is modeled as:

$$
\mathrm { P } _ { i } ( a _ { i } | \beta _ { 0 } , \beta _ { s _ { c _ { i } } } , b _ { c _ { i } } ) = \mathrm { i n v \mathrm { \_ l o g i t } } ( \beta _ { 0 } + \beta _ { s _ { c _ { i } } } a _ { i } + b _ { c _ { i } } ) ,\tag{14}
$$

where $\boldsymbol { s } _ { c _ { i } }$ is the cohort seg number for cohort $c _ { i } .$ , and $\beta _ { 0 }$ and $\beta _ { s _ { c _ { i } } }$ ’s are model parameters to estimate. The principal distinction between the NMM (11) and the GLMM (14) lies in the age efect: while the latter incorporates a strictly linear predictor with respect to age, the former accommodates flexible, nonlinear functional relationships.

We used the 3L data (31,427 observations) as the training set, the 3N data (23,392 observations) as the validation set, and the 3O data (24,104 observations) as the test set. We evaluated model performance on the validation and test sets using the average negative loglikelihood, referred to as the binary cross-entropy (BCE) in the machine learning literature. For instance, the BCE for the validation set, denoted by $\mathrm { B C E } _ { \mathrm { v a l i d } }$ , is defined as

$$
\mathrm { B C E } _ { \mathrm { v a l i d } } = - \frac { 1 } { N _ { \mathrm { v a l i d } } } \sum _ { i = 1 } ^ { N _ { \mathrm { v a l i d } } } \left[ d _ { i } \log \{ \mathrm { P } i ( a _ { i } \ | \ \hat { \varphi } , \hat { b } _ { c _ { i } } ) \} + ( 1 - d _ { i } ) \log \{ 1 - \mathrm { P } i ( a _ { i } \ | \ \hat { \varphi } , \hat { b } _ { c _ { i } } ) \} \right] ,\tag{15}
$$

where $i = 1 , \dots , N _ { \mathrm { v a l i d } }$ indexes the validation observations, and $\hat { \varphi }$ and $\hat { b } _ { c _ { i } }$ represent the estimated model parameters and REs, respectively.

We implemented both the NMM and GLMM using TMB, which eficiently evaluates the marginal objective function and its gradient. These quantities were then supplied to nlminb() for optimization of the marginal objective function. During optimization, the models were fitted to the training data, while $\mathrm { B C E } _ { \mathrm { v a l i d } }$ was recorded at each optimization step. We identified the point at which $\mathrm { B C E } _ { \mathrm { v a l i d } }$ ceased to improve as the point of best generalization. The corresponding parameter and RE estimates were subsequently taken as the final estimates. Prior to this final fit, we performed a grid search over the regularization hyperparameter λ, and at each grid point, the KKT multiplier η was determined using the algorithm described previously. This procedure yielded $\lambda = 0 . 0 0 1$ and $\eta = 1 . 7$

Based on the optimized parameter and RE estimates, we evaluated the BCE on the test set, $\mathrm { B C E } _ { \mathrm { t e s t } } .$ to compare the predictive performance of the NMM and GLMM. The NMM achieved a slightly lower $\mathrm { B C E _ { t e s t } }$ of 0.249, compared with 0.250 for the GLMM.

TMB automatically predicts the REs using their posterior modes and predicts functions of the REs and fixed parameters using plug-in predictors. Figure 4 compares the predicted age-specific maturation probabilities for the 50th cohort between the NMM and the GLMM. While the GLMM yields predictions that are symmetric around the inflection point, the NMM accommodates mild asymmetry to better capture the underlying data.

![](images/c372b062f7592a382efb36bb88f58946fd778576c9a5ffe66defd05b59739330.jpg)  
Figure 4: Predicted probabilities of maturation by age for the 50th cohort, using the NMM and GLMM.

## 5 Conclusion

Due to its enormous implementation and computation complexity, eficient software and hardware tools are always one of the central problem in artificial intelligence. In this work, we propose to implement neural network mixed-efects models (NMMs) in statistical modeling using Template Model Builder (TMB). In this approach, users need to specify only the negative joint log-likelihood and any regularization terms. The framework automatically integrates out random efects and evaluates the marginal objective function alongside its exact gradients, eliminating the need for manual derivations or ad hoc approximations.

Replicating the simulation studies of Mandel et al. (2023), the results in Section 3 highlight several advantages of our proposed methodology. First, adopting a full likelihood formulation instead of a quasi-likelihood approach increases statistical eficiency. Second, by fully automating the derivation process, our method eliminates human error and avoids the heuristic omissions or approximations typically needed for manual solutions. As a result, our approach ofers enhanced estimation accuracy, improved implementation eficiency, and superior scalability toward complex NMMs.

The empirical results in Section 4 demonstrate the simplicity of implementing constrained NMMs within the TMB framework. Specifically, TMB enables the eficient computation of the monotonicity-constrained objective function and its gradients for the downstream optimization algorithm. Furthermore, even without extensive architecture tuning, this application underscores the representational capacity of NMMs in modeling intricate, nonlinear input–output dynamics.

In the current implementation, feedforward neural networks are coded manually; however, extending this manual approach to convolutional, recurrent, or transformer archi tectures is impractical. Immediate future work will focus on integrating Keras (see, e.g., Chapter 3 of Chollet et al., 2022), an R interface to TensorFlow, with RTMB (Kristensen, 2024), which provides native R automatic diferentiation for TMB without C++ compilation. Synthesizing these tools will allow us to construct sophisticated deep learning mixed-efects models. A secondary direction involves extending these models to incorporate complex spatiotemporal structures via random efects. Additionally, migrating our implementation of NMMs from a strictly CPU-bound environment to a hybrid CPU/GPU workflow will optimize computational performance and support the increased overhead of these advanced architectures.

## Codes availability

The TMB code supporting the findings of this study is publicly available at https://github.com/ZhengNan18/Implementing-neural-network-mixed-efects-models-in-Template-Model-Builder-TMB-

## Acknowledgments

Research funding to NZ was provided by the Natural Sciences and Engineering Research Council of Canada [RGPIN-2024-04746]. Research funding to NC was provided by 1) the Ocean Choice International Industry Research Chair program at the Marine Institute of Memorial University of Newfoundland, 2) the Ocean Frontier Institute, through an award from the Canada First Research Excellence Fund, and 3) the Natural Sciences and Engineering Research Council of Canada [RGPIN-2016-04307].

## References

Bengio, Y. (2009). Learning deep architectures for ai. Foundations and Trends® in Finance 2 (1), 1–127.

Breslow, N. and D. Clayton (1993). Approximate inference in generalized linear mixed models. Journal of the American statistical Association 88 (421), 9–25.

Chen, D. and W. Ye (2022). Monotonic neural additive models: Pursuing regulated machine learning models for credit scoring. In Proceedings of the third ACM international conference on AI in finance, pp. 70–78.

Chollet, F., T. Kalinowski, and J. J. Allaire (2022). Deep learning with R. Simon and Schuster.

Goodfellow, I., Y. Bengio, and A. Courville (2016). Deep Learning. MIT Press.

Hastie, T., R. Tibshirani, J. Friedman, et al. (2009). The elements of statistical learning.

Hornik, K., M. Stinchcombe, and H. White (1989). Multilayer feedforward networks are universal approximators. Neural networks 2(5), 359–366.

Karush, W. (1939). Minima of functions of several variables with inequalities as side constraints. M. Sc. Dissertation. Dept. of Mathematics, Univ. of Chicago.

Kristensen, K. (2024). RTMB: R Bindings for ’TMB’. R package version 1.5.

Kristensen, K., A. Nielsen, C. W. Berg, H. Skaug, and B. Bell (2016). Tmb: automatic diferentiation and laplace approximation. Journal of Statistical Software 70 (5), 1–21.

Krogh, A. and J. Hertz (1991). A simple weight decay can improve generalization. Advances in neural information processing systems 4.

LeCun, Y., Y. Bengio, and G. Hinton (2015). Deep learning. nature 521 (7553), 436–444.

Mandel, F., R. P. Ghosh, and I. Barnett (2023). Neural networks for clustered and longitudinal data using mixed efects models. Biometrics 79 (2), 711–721.

R Core Team (2024). R: A Language and Environment for Statistical Computing. Vienna, Austria: R Foundation for Statistical Computing.

Tandon, R., S. Adak, and J. A. Kaye (2006). Neural networks for longitudinal studies in alzheimer’s disease. Artificial intelligence in medicine 36 (3), 245–255.

Thygesen, U. H., C. M. Albertsen, C. W. Berg, K. Kristensen, and A. Nielsen (2017). Validation of ecological state space models using the laplace approximation. Environmental and Ecological Statistics 24 (2), 317–339.

Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society Series B: Statistical Methodology 58 (1), 267–288.

Tran, M.-N., N. Nguyen, D. Nott, and R. Kohn (2017). Random efects models with deep neural network basis functions: Methodology and computation. Technical report.