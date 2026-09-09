# Flexible Spectral-Normalized Neural Gaussian Process for Dynamic Aperture Prediction

Yousra El-Bachir\* Swiss Data Science Center, ETH Zurich & EPFL

Davide di Croce EPFL, Lausanne; CERN, Geneva

Carlo Emilio Montanari CERN, Geneva

Ekaterina Krymova† Swiss Data Science Center, ETH Zurich & EPFL

Frederik Van der Veken CERN, Geneva

Massimo Giovannozzi CERN, Geneva

Tatiana Pieloni EPFL, Lausanne; CERN, Geneva

September 9, 2026

## Abstract

We address the challenge of scalable uncertainty quantification in large-scale scientific applications, where complex state-of-the-art machine learning methods are often computationally infeasible. Our primary contribution is a simple yet effective empirical Bayes method for automatically tuning the hyperparameters of a flexible, heteroscedastic Spectral-normalized Neural Gaussian Process. This approach retains the expressiveness and uncertainty-awareness of semi-Bayesian neural models while significantly reducing the computational burden by integrating hyperparameter learning directly into the training loop. We demonstrate the practical impact of our method on the task of estimating the dynamic aperture in circular particle accelerators, a fundamental problem in high-energy physics colliders and storage rings using simulation data from the case of the Large Hadron Collider at CERN. Traditional approaches to DA estimation require extensive particle-tracking simulations, which are prohibitively time-consuming and resource-intensive. Our results show that the proposed method achieves competitive predictive performance and well-calibrated uncertainty estimates at much lower computational cost than state-of-theart approaches. We stress that, beyond this application, the proposed empirical Bayes framework offers a general solution for training heteroscedastic neural models in situations where manual hyperparameter tuning is impractical. Accordingly, we anticipate that this framework can be applied to other domains that encounter comparable computational limitations.

## 1 Introduction

Uncertainty in predictive modeling is traditionally categorized into two fundamentally different types: aleatoric and epistemic. Aleatoric uncertainty arises from inherent noise or randomness in the data, such as measurement errors, and is considered irreducible. In contrast, epistemic uncertainty reflects a lack of knowledge due to limited observed data and can, in principle, be reduced by increasing the training set or using more complex models. Accurate disentanglement of these types of uncertainty is particularly important in active learning and safety-critical applications such as health care or autonomous driving [Esteva et al., 2017, Huang and Chen, 2020]. However, recent machine learning literature challenges the assumption that aleatoric and epistemic uncertainties are cleanly separable, showing that they often overlap and interact in practice. Moreover, widely used uncertainty measures produce inconsistent or conflicting results, and current uncertainty quantification methods are poorly calibrated under distributional changes, further motivating the need to develop more robust and coherent approaches and measures; see Smith et al. [2024], Wimmer et al. [2023], Bengs et al. [2023], Schweighofer et al. [2023], Kotelevskii and Panov [2024], Postels et al. [2022].

Despite these efforts, reliably disentangling epistemic and aleatoric uncertainty, as well as developing scalable well-calibrated methods, remain open problems. Therefore, in this paper, we take a more pragmatic approach and introduce an efficient method to estimate the total predictive uncertainty directly within the model training process.

Two of the most commonly used state-of-the-art methods for uncertainty quantification are deep ensembles (DE) [Lakshminarayanan et al., 2017] and deep deterministic uncertainty (DDU) [Mukhoti et al., 2023]. For the classification task, DE involves training multiple deep learning models independently with different random initializations and averaging their predictions. The variance across the ensemble is used to estimate the uncertainty. Although DE improves accuracy, it may still produce overconfident predictions in regions far from the training data, a limitation inherited from the individual base models. Moreover training multiple deep learning models can be computationally prohibitive in real-world applications, including scientific ones, making DE impractical in resource-constrained settings. In contrast, DDU estimates predictive uncertainty from a single forward pass of a deterministic neural network trained with residual connections and spectral normalization to mitigate feature collapse. At inference time, DDU estimates the uncertainty from a feature-space density estimator constructed based on Gaussian Discriminant Analysis, fitted separately to each output class. This makes DDU computationally efficient compared to DE, while requiring minimal architectural modifications. A more advanced state-of-the-art method is the Spectral-normalized Neural Gaussian Process (SNGP) [Liu et al., 2023], along with its heteroscedastic extension [Fortuin et al., 2022]. This semi-Bayesian approach combines the expressive predictive capabilities of deep neural networks with the principled uncertainty estimation of Gaussian Processes (GPs), and produces a better calibrated uncertainty under distributional shifts compared to several competing methods [Postels et al., 2022]. By appending a GP layer to a residual network with spectrally normalized weights, SNGP becomes sensitive to the distance between training and test points, which is particularly advantageous for out-of-distribution detection and helps mitigate the overconfidence observed in DE. However, SNGP's reliance on manual hyperparameter tuning makes it less computationally attractive in practice. We propose to improve the flexibility of SNGP and integrate its hyperparameter tuning directly into the training process, thereby improving usability without compromising performance. We motivate the development of this method through a critical problem in the design of circular particle accelerators for high-energy physics, namely, the prediction of dynamic aperture (DA) and possibly particle loss rates; see, e.g. Schenk et al. [2021], Giovannozzi et al. [2021], Van der Veken et al. [2021, 2022], based on the simulation data from the CERN Large Hadron Collider (LHC) Brüning et al. [2004].

The DA is a fundamental concept in accelerator physics, representing the extent of the region of phase space in which particle trajectories remain bounded over a predefined number of revolutions around a circular accelerator. Particles starting outside of this region are ultimately lost, resulting in beam degradation and reduced operational efficiency, making the DA a key indicator of long-term beam stability. Therefore, accurate prediction of the DA is critical for understanding nonlinear beam dynamics and for optimizing both the performance and the safe operation of modern colliders and storage rings; see, e.g. Todesco and Giovannozzi [1996], Giovannozzi et al. [1997, 1998], Giovannozzi [2012, 2026]. For the LHC, determining the DA traditionally involves scanning a high-dimensional parameter space through particle-tracking simulations, often guided by expert accelerator physics knowledge and heuristic strategies. This process is both computationally intensive and time-consuming, making DA studies challenging. In this paper, in contrast to direct DA prediction Montanari et al. [2025], Di Croce et al. [2024], we introduce an efficient and automated approach to predicting DA by distinguishing the stable region from the rest of the phase space. Moreover, we produce reliable uncertainty estimates that can guide tracking efforts toward regions where predictions are uncertain.

Particle-tracking simulations for the LHC, such as SixTrack Maria et al. [2019] with MAD-X¹, enable the identification of three distinct spatial regions within a particle accelerator. Particles that complete the maximum number of observable revolutions define the stable region of the phase space. In contrast, the unstable region corresponds to the coordinates where the particles are rapidly lost, failing to complete a significant number of revolutions. Between these two extremes lies a chaotic boundary region, where particles survive a substantial—but submaximal—number of revolutions before being lost. Figure 1 illustrates the DA, corresponding to the stable region, for three randomly selected accelerator configurations. Each configuration is defined by a unique combination of the values of the input control parameters used in the simulator. For simplicity, the chaotic boundary and the unstable region are merged, as the primary focus is on accurately predicting the stable region. These examples already highlight several challenges associated with modeling the DA. First, the stable region has a complex and nonlinear structure that cannot be captured by simple parametric models, motivating the use of data-driven approaches. Second, the inherently noisy and ambiguous structure of the chaotic boundary makes the classification problem difficult. Third, the dataset represents simulations for millions of particles given hundreds of control variables, making the modeling task computationally intensive. The objective is therefore to develop a flexible, data-driven classification model for predicting the DA, designed for efficient training with minimal dependence on hyperparameter tuning Crucially, the model's uncertainty estimates should be well-calibrated: they should be high only in regions where predictions are likely to be unreliable.

![](images/3f362f7f3bdd957356c89c01692e4560f6254792171f530f528ceab0d9d4241b.jpg)

![](images/c270980c84c2b121ce688bd109e555738a43d00b02065023659e4d77bb8e0242.jpg)

![](images/c9719621be687db8fa4fef07bea69ea5f7d56217e388df82cafb7ea49652e2a8.jpg)

![](images/782fa0098c5198cb1a191b5bc20eba7dcf39285dded8dd8dd6aaaf7b48061df8.jpg)  
Figure 1: Stability regions derived from particle-tracking simulations of the LHC under three different accelerator configurations, with particles initialized at varying radial positions and angular orientations.

The paper is organized as follows. Section 2 extends the standard SNGP architecture to accommodate more flexible settings and describes the residual network architecture tailored to the tabular data used in our study. Section 3 introduces an empirical Bayes framework that enables automatic integration of hyperparameter tuning into the training process. Section 4 presents the empirical results, and Section 5 closes the article with a discussion of the findings and potential future research directions.

## 2 The flexible auto-hetSNGP model

The heteroscedastic Spectral-normalized Neural Gaussian Process version (hetSNGP) introduced by Fortuin et al. [2022] extends the original SNGP framework of Liu et al. [2023] by incorporating the heteroscedastic model of Collier et al. [2021] using a latent variable drawn from an approximate Gaussian Process, as follows. Let $\{ ( \pmb { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ be n independent pairs of input vectors $\pmb { x } _ { i } \in \mathbb { R } ^ { d }$ and outputs $y _ { i } \in \{ 1 , \ldots , K \}$ . Each label $y _ { i }$ is modeled by the softmax distribution over a latent random vector $\pmb { u } ( \pmb { x } _ { i } ) = \{ u _ { 1 } ( \pmb { x } _ { i } ) , \dots , u _ { K } ( \pmb { x } _ { i } ) \} \in \mathbb { R } ^ { 1 \times K }$

$$
P \{ y _ { i } = c \mid \pmb { u } ( \pmb { x } _ { i } ) \} = \mathrm { S o f t m a x } _ { \tau } \{ \pmb { u } ( \pmb { x } _ { i } ) \} = \frac { \exp \{ u _ { c } ( \pmb { x } _ { i } ) / \tau \} } { \sum _ { k = 1 } ^ { K } \exp \{ u _ { k } ( \pmb { x } _ { i } ) / \tau \} }\tag{1}
$$

for $c = 1 , \ldots , K$ , where $\tau$ is a temperature parameter. Each component $u _ { c } ( { \pmb x } _ { i } )$ is modeled with a mean-variance decomposition

$$
\begin{array} { r l r } { u _ { c } ( { \pmb x } _ { i } ) } & { { } = } & { \mu _ { c } ( { \pmb x } _ { i } ) + \epsilon _ { c } ( { \pmb x } _ { i } ) , } \end{array}\tag{2}
$$

where $\mu _ { c }$ denotes the predictive mean, and $\epsilon _ { c }$ represents a zero-mean stochastic class-specific deviation capturing heteroscedastic uncertainty. The predictive mean is modeled using Bayesian linear regression,

$$
\begin{array} { r l r } { \mu _ { c } ( { \pmb x } _ { i } ) } & { = } & { \Phi ( { \pmb x } _ { i } ) \beta _ { c } , \quad \beta _ { c } \sim \mathcal { N } ( \mathbf { 0 } , I ) , } \end{array}\tag{3}
$$

where $\beta _ { c } \in \mathbb { R } ^ { D _ { L } \times 1 }$ is a class-specific weight vector, and $\Phi ( { \pmb x } _ { i } ) \in \mathbb { R } ^ { 1 \times D _ { L } }$ is a random feature embedding constructed as follows. We first adapt the ResNet-like architecture that the hetSNGP model was originally trained on to our tabular data problem. The input $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ is first passed through a residual neural network illustrated in Figure 2, which comprises a sequence of residual blocks, each consisting of a basic block paired with an associated shortcut block. The shortcut block has a single linear layer and serves two key purposes: it ensures dimensional compatibility between the input and the output of the main path, and it preserves feature identity, which facilitates stable gradient flow during training. Each basic block is implemented as a multilayer perceptron, where each linear layer has spectrally normalized (SN) weights, followed by batch normalization, a ReLU activation, and dropout. Spectral normalization is applied to preserve the Lipschitz continuity of the forward pass and to maintain geometric consistency in the feature space.

![](images/a3dc2de6c3e2f5c3dff9d90691b0557e4d13d928124e9e852a43d47395762d3b.jpg)  
Figure 2: Residual network with skip connections and Spectral Normalization (SN).

The resulting hidden representation $\pmb { h } ( \pmb { x } _ { i } ) \in \mathbb { R } ^ { D _ { L - 1 } \times 1 }$ is then passed through a GP output layer approximated by Random Fourier Features (RFF) [Rahimi and Recht, 2007]

$$
\begin{array} { c c l } { \Phi ( { \pmb x } _ { i } ) } & { = } & { \sqrt { 2 / D _ { L } } \cos \{ W { \pmb h } ( { \pmb x } _ { i } ) + { \pmb b } \} ^ { T } , } \end{array}
$$

with $\pmb { W } \in \mathbb { R } ^ { D _ { L } \times D _ { L - 1 } }$ and $\pmb { b } \in \mathbb { R } ^ { D _ { L } \times 1 }$ drawn from

$$
\begin{array} { r l r } { W _ { i j } } & { { } \sim } & { \mathcal { N } ( 0 , 1 ) , \quad b _ { i } \sim \mathcal { U } ( 0 , 2 \pi ) , } \end{array}
$$

and kept fixed during training. This RFF-based layer approximates a stationary GP kernel and ensures distance-awareness of the network output, which improves sensitivity to out-of-distribution samples, while the Bayesian formulation in 3 provides a principled way to encode prior uncertainty over the mean function.

The stochastic component $\epsilon _ { c }$ in $2$ models heteroscedasticity through the following combination of classspecific and shared noise terms

$$
\begin{array} { r c l } { \epsilon _ { c } ( \pmb { x } _ { i } ) } & { = } & { d _ { c } ( \pmb { x } _ { i } ) \kappa _ { c } + \pmb { v } _ { c } ( \pmb { x } _ { i } ) \gamma , } \end{array}\tag{4}
$$

$$
\kappa _ { c } \sim \mathcal { N } ( 0 , 1 ) , \gamma \sim \mathcal { N } ( \mathbf { 0 } , I ) ,\tag{5}
$$

where $\kappa _ { c }$ is a class-specific noise term, $\gamma \in \mathbb { R } ^ { R \times 1 }$ is a global latent noise vector of dimension $R \ll K$ , and $d _ { c } ( { \pmb x } _ { i } ) \in \mathbb { R }$ and ${ \pmb v } _ { c } ( { \pmb x } _ { i } ) \in \mathbb { R } ^ { 1 \times R }$ are input-dependent coefficients obtained as linear projections of the hidden representation $\pmb { h } ( \pmb { x } _ { i } )$ 2

$$
\begin{array} { r c l } { { { \pmb d } ( { \pmb x } _ { i } ) } } & { { = } } & { { { \pmb W } _ { d } { \pmb h } ( { \pmb x } _ { i } ) + { \pmb b } _ { d } , } } \end{array}\tag{6}
$$

$$
\begin{array} { r c l } { { \cal V } ( { \bf x } _ { i } ) } & { = } & { \mathrm { R e s h a p e } \{ { \cal W } _ { V } { \cal h } ( { \bf x } _ { i } ) + b _ { V } , K , R \} , } \end{array}\tag{7}
$$

where $\pmb { d } ( \pmb { x } _ { i } ) \in \mathbb { R } ^ { 1 \times K }$ and $V ( \pmb { x } _ { i } ) \in \mathbb { R } ^ { K \times R }$ have learnable parameters $\pmb { W } _ { d } \in \mathbb { R } ^ { K \times D _ { L - 1 } } , \pmb { b } _ { d } \in \mathbb { R } ^ { K \times 1 }$ ， $W _ { V } \in$ $\mathbb { R } ^ { K R \times D _ { L - 1 } }$ and $\pmb { b } _ { V } \in \mathbb { R } ^ { K R \times 1 }$ The condition $R \ll K$ imposes a low-rank structure, and the inclusion of $\pmb { d } ( \pmb { x } _ { i } )$ ensures that the covariance matrix of the latent variable ${ \pmb u } ( { \pmb x } _ { i } )$ remains strictly positive definite. The structured noise formulation in 4 therefore enables the hetSNGP model to capture both independent class-specific noise through the term $d _ { c } ( { \pmb x } _ { i } ) \kappa _ { c }$ and correlations across classes by sharing the global noise $\gamma .$

The hetSNGP model in 1 introduces a global temperature parameter $\tau ,$ which in the original article [Fortuin et al., 2022] was kept fixed during training and prediction, showing that its value did not significantly affect the quality of uncertainty calibration in an ablation study in the ImageNet dataset. In general, the temperature parameter can be tuned post hoc to control the calibration of predictive uncertainty [Guo et al.. $2 0 1 7 ]$ , at the cost of additional computational overhead that we propose to relieve as follows. On setting $\tilde { u } _ { c } ( { \bf x } _ { i } ) = u _ { c } ( { \bf x } _ { i } ) / \tau$ and rewriting the temperature-scaled softmax in 1, we get

$$
\begin{array} { r c l } { P \big \{ y _ { i } = c \mid \boldsymbol { u } ( \boldsymbol { x } _ { i } ) \big \} } & { = } & { \mathrm { S o f t m a x } \{ \tilde { u } ( \boldsymbol { x } _ { i } ) \} , } \\ { \tilde { u } _ { c } ( \boldsymbol { x } _ { i } ) } & { = } & { \Phi ( \boldsymbol { x } _ { i } ) \tilde { \beta } _ { c } + d _ { c } ( \boldsymbol { x } _ { i } ) \tilde { \kappa } _ { c } + v _ { c } ( \boldsymbol { x } _ { i } ) \tilde { \gamma } , } \\ { \tilde { \beta } _ { c } } & { \sim } & { \mathcal { N } ( \mathbf { 0 } , 1 / \tau ^ { 2 } I ) , \quad \tilde { \kappa } _ { c } \sim \mathcal { N } ( 0 , 1 / \tau ^ { 2 } ) , \quad \tilde { \gamma } \sim \mathcal { N } ( 0 , 1 / \tau ^ { 2 } I ) , } \end{array}\tag{8}
$$

where the factor $1 / \tau ^ { 2 }$ in $\tilde { \kappa } _ { c }$ and in $\tilde { \gamma }$ can be absorbed into the learnable parameters $W _ { d } , b _ { d } , W _ { V }$ and $\pmb { b } _ { V }$ in 6-7. The explicit scaling in 8 therefore becomes redundant relative to the reduced form in $5 ,$ but this simplification is only valid when $d ( { \pmb x } _ { i } )$ and $V ( \pmb { x } _ { i } )$ are linear parameterizations as in 6–7. However, we maintain the scaling in the definitions of $\tilde { \kappa } _ { c }$ and $\tilde { \gamma }$ throughout the remainder of the paper to allow seamless extension to settings where explicit control over the variance is essential and to highlight the generality of the proposed optimization framework. We also improve the original hetSNGP model's flexibility by learning class-dependent variance terms through the following reformulation

$$
\begin{array} { r l r } { P \{ y _ { i } = c \mid { \bf u } ( { \bf x } _ { i } ) \} } & { { } = } & { \mathrm { S o f t m a x } _ { \tau = 1 } \{ { \bf u } ( { \bf x } _ { i } ) \} , } \end{array}\tag{9}
$$

$$
\begin{array} { r l r } { u _ { c } ( { \pmb x } _ { i } ) } & { = } & { \Phi ( { \pmb x } _ { i } ) \beta _ { c } + d _ { c } ( { \pmb x } _ { i } ) \kappa _ { c } + { \pmb v } _ { c } ( { \pmb x } _ { i } ) \gamma , } \end{array}\tag{10}
$$

$$
\begin{array} { r c l } { \displaystyle \beta _ { c } } & { \sim } & { \mathcal { N } ( \mathbf { 0 } , \sigma _ { b _ { c } } ^ { 2 } I ) , \quad \kappa _ { c } \sim \mathcal { N } ( 0 , \sigma _ { \kappa _ { c } } ^ { 2 } ) , \quad \gamma _ { j } \sim \mathcal { N } ( 0 , \sigma _ { \gamma _ { j } } ^ { 2 } ) , } \end{array}\tag{11}
$$

and we tune additionally introduced parameters by adopting an empirical Bayes approach that integrates hyperparameter estimation directly into the training process, thereby enhancing scalability. By introducing class-specific variance for $\beta _ { c } ,$ we additionally enable the model to automatically learn the appropriate levels of regularization for each class. This flexibility allows the model to account for class-specific signal strength and potential class imbalance, improving predictive performance and uncertainty calibration. We now describe the optimization procedure.

## 3 Empirical Bayes optimization

Let $\beta = ( \beta _ { 1 } , \dots , \beta _ { K } )$ and $\pmb { \kappa } = ( \kappa _ { 1 } , \ldots , \kappa _ { K } )$ denote the model weights, and let $\pmb { \tau } _ { b } = \{ \log ( \sigma _ { b _ { 1 } } ^ { 2 } ) , \dots , \log ( \sigma _ { b _ { K } } ^ { 2 } ) \}$ $\tau _ { \kappa } = \{ \log ( \sigma _ { \kappa _ { 1 } } ^ { 2 } ) , \dots , \log ( \sigma _ { \kappa _ { K } } ^ { 2 } ) \}$ and $\tau _ { \gamma } = \{ \log ( \sigma _ { \gamma _ { 1 } } ^ { 2 } ) , \dots , \log ( \sigma _ { \gamma _ { R } } ^ { 2 } ) \}$ denote the corresponding log-variances. Learning logarithms instead of original variances mitigates potential numerical instabilities by performing the optimization in an unconstrained parameter space. The optimal variances are then recovered by exponentiation of the learned log-variance values. We divide the model parameters into two groups, a vector of deterministic variables denoted by $\psi = \left( \tau _ { b } , \tau _ { \kappa } , \tau _ { \gamma } , h , W _ { d } , b _ { d } , W _ { V } , b _ { V } \right)$ , which we learn during the training phase, and a vector of stochastic variables denoted by $\pmb \theta = ( \beta , \kappa , \gamma )$ , which we use for posterior sampling during prediction. We train the auto-hetSNGP model in 9–11 by maximizing the log-marginal likelihood with respect to the deterministic parameters to obtain the optimal estimate $\begin{array} { r } { \dot { \psi } , } \end{array}$ which we hold fixed at inference time. We then compute the predictions by sampling from the posterior distribution of θ given $\textbf {  { y } }$ and fixed $\hat { \psi }$ using the Laplace method. Overall, the empirical Bayes method for the flexible model amounts to a double optimization, which we now develop.

## 3.1 Double maximization

The marginal likelihood based loss function for $\psi$ We denote the prior distribution of $\pmb { \theta }$ for a fixed $\psi$ by

$$
p ( \pmb { \theta } ; \pmb { \psi } ) = p ( \pmb { \beta } ; \pmb { \sigma } _ { b } ^ { 2 } ) \ p ( \pmb { \kappa } ; \pmb { \sigma } _ { \kappa } ^ { 2 } ) \ p ( \pmb { \gamma } ; \pmb { \sigma } _ { \gamma } ^ { 2 } ) ,
$$

as a product of individual priors in 11. The marginal likelihood is hence

$$
\begin{array} { r c l } { \displaystyle p ( \pmb { y } ; \pmb { x } , \psi ) } & { = } & { \displaystyle \int p \left( \pmb { y } \mid \pmb { \theta } ; \pmb { x } , \psi \right) p ( \pmb { \theta } ; \pmb { \psi } ) \mathrm d \pmb { \theta } } \\ & { = } & { \displaystyle \mathrm E _ { \pmb { \theta } \sim p ( \pmb { \theta } ; \pmb { \psi } ) } \left[ \exp \left\{ \sum _ { i = 1 } ^ { n } \log p \left( y _ { i } \mid \pmb { \theta } ; \pmb { x } _ { i } , \psi \right) \right\} \right] } \\ & { \approx } & { \displaystyle \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \exp \left[ \sum _ { i = 1 } ^ { n } \log \mathrm { S o f t m a x } \left\{ \pmb { u } _ { y _ { i } } ^ { ( s ) } ( \pmb { x } _ { i } ) \right\} \right] , } \end{array}
$$

where the approximation follows from S Monte-Carlo samples, each of which is generated by

$$
\begin{array} { r l r } { { \pmb u } _ { y _ { i } } ^ { ( s ) } ( { \pmb x } _ { i } ) } & { = } & { \Phi ( { \pmb x } _ { i } ) \beta _ { y _ { i } } ^ { ( s ) } + d _ { y _ { i } } ( { \pmb x } _ { i } ) \kappa _ { y _ { i } } ^ { ( s ) } + { \pmb v } _ { y _ { i } } ( { \pmb x } _ { i } ) \gamma ^ { ( s ) } , } \end{array}
$$

with samples drawn from the respective priors

$$
\begin{array} { r l r } { \boldsymbol \beta _ { y _ { i } } ^ { ( s ) } } & { \sim } & { \mathcal { N } ( \mathbf { 0 } , \sigma _ { b _ { y _ { i } } } ^ { 2 } I ) , \quad \kappa _ { y _ { i } } ^ { ( s ) } \sim \mathcal { N } ( 0 , \sigma _ { \kappa _ { y _ { i } } } ^ { 2 } ) , \quad \gamma _ { j } ^ { ( s ) } \sim \mathcal { N } ( 0 , \sigma _ { \gamma _ { j } } ^ { 2 } ) . } \end{array}
$$

At training time, the optimal deterministic parameters $\hat { \psi }$ are obtained by minimizing $- \log p ( \pmb { y } ; \pmb { x } , \psi )$ using stochastic gradient descent or one of its variants. At inference time, we need to sample from the posterior distribution $p ( \pmb \theta \mid \pmb y ; \pmb x , \hat { \psi } )$ , which we approximate by a Gaussian distribution whose expectation and covariance matrix are learned from the maximum penalized likelihood estimator as we shall now see.

The penalized likelihood based loss function for θ. We denote the log-penalized likelihood of $\pmb \theta$ for fixed $\hat { \psi }$ by

$$
\ell _ { \mathrm { P } } ( \pmb \theta ; \pmb y , \pmb x , \hat { \psi } ) = \log \{ p ( \pmb y \mid \pmb \theta ; \pmb x , \hat { \psi } ) ~ p ( \pmb \theta ; \hat { \psi } ) \} .
$$

Using the hierarchical model in 9–11, we obtain

$$
\begin{array} { r l r } { \ell _ { \mathrm { P } } ( \pmb \theta ; \pmb y , \pmb x , \hat { \psi } ) } & { \equiv } & { \displaystyle \sum _ { i = 1 } ^ { n } \log \mathrm { S o f t m a x } \left\{ \pmb u _ { y _ { i } } ( \pmb x _ { i } ) \right\} - \frac { 1 } { 2 } \left\{ \sum _ { c = 1 } ^ { K } \left( \frac { 1 } { \hat { \sigma } _ { b _ { c } } ^ { 2 } } \left\| \pmb \beta _ { c } \right\| _ { 2 } ^ { 2 } + \frac { 1 } { \hat { \sigma } _ { c _ { c } } ^ { 2 } } \kappa _ { c } ^ { 2 } \right) + \sum _ { r = 1 } ^ { R } \frac { 1 } { \hat { \sigma } _ { \gamma _ { r } } ^ { 2 } } \gamma _ { r } ^ { 2 } \right\} , } \end{array}\tag{12}
$$

where

$$
\begin{array} { r l r } { { \pmb u } _ { y _ { i } } ( { \pmb x } _ { i } ) } & { = } & { \hat { \Phi } ( { \pmb x } _ { i } ) { \pmb \beta } _ { y _ { i } } + \hat { d } _ { y _ { i } } ( { \pmb x } _ { i } ) \kappa _ { y _ { i } } + \hat { \pmb v } _ { y _ { i } } ( { \pmb x } _ { i } ) \gamma , } \end{array}
$$

and the weights are now considered learnable deterministic parameters. Let ${ H } _ { \mathrm { P } } ( \pmb \theta ; \pmb y , \pmb x , \hat { \psi } )$ denote the hessian of the negative log-penalized likelihood, and let $\begin{array} { r } { \hat { \pmb { \theta } } _ { \hat { \pmb { \psi } } } = \arg \operatorname* { m a x } _ { \pmb { \theta } } \ell _ { \mathrm { P } } ( \pmb { \theta } ; \pmb { y } , \pmb { x } , \hat { \psi } ) } \end{array}$ denote the maximizer. The

Laplace approximation to the posterior likelihood leads to

$$
\begin{array} { l l l } { \displaystyle p ( \theta \mid y ; \pmb { x } , \hat { \psi } ) } & { = } & { \displaystyle \frac { p ( y \mid \theta ; \pmb { x } , \hat { \psi } ) ~ p ( \theta ; \hat { \psi } ) } { p ( y ; \pmb { x } , \hat { \psi } ) } } \\ & { \propto } & { \displaystyle \exp \left\{ \ell _ { \mathrm { P } } ( \pmb { \theta } ; \pmb { y } , \pmb { x } , \hat { \psi } ) \right\} } \\ & { \approx } & { \mathcal { N } \left\{ \hat { \theta } _ { \hat { \psi } } ; H _ { \mathrm { P } } ^ { - 1 } ( \hat { \theta } _ { \hat { \psi } } ; \pmb { y } , \pmb { x } , \hat { \psi } ) \right\} . } \end{array}\tag{13}
$$

Once the log-penalized likelihood in 12 is maximized to obtain $\hat { \theta } _ { \hat { \psi } }$ , sampling θ from its posterior distribution involves evaluating ${ \pmb H } _ { \mathrm { P } } ( { \pmb \theta } ; { \pmb y } , { \pmb x } , \hat { \psi } )$ at $\theta \ : = \ : \hat { \theta } _ { \hat { \psi } }$ , and drawing samples from the corresponding Gaussian distribution 13.

The dissociation between the deterministic ψ and the stochastic θ significantly reduces computational overhead, as the expensive training of the neural network is performed only once at the minimization of the negative log-marginal likelihood. Similarly, the optimization of the log-penalized likelihood on the training set is performed only once at inference time for evaluating predictive accuracy. Crucially, the input-dependent terms $\Phi ( { \boldsymbol { \mathbf { x } } } ) , d ( { \boldsymbol { \mathbf { \mathit { x } } } } )$ and ${ \pmb v } ( { \pmb x } )$ of the latent variable ${ \pmb u } ( { \pmb x } )$ are computed from a single forward pass through the neural network and are shared accross the evaluations of both the marginal and penalized likelihoods. We now detail the minimization of the negative log-penalized likelihood in 12 using the Newton-Raphson algorithm, which is feasible in our setting given the low dimensionality of θ.

Stable implementation of the Newton-Raphson minimizer for the log-penalized likelihood. We develop a robust optimization algorithm using preconditioning to improve numerical stability and convergence even in cases where the Hessian matrix may be ill-conditioned.

Let $G _ { \mathrm { P } } ( \boldsymbol { \theta } ; \hat { \boldsymbol { \psi } } )$ and $H _ { \mathrm { P } } ( \theta ; \hat { \psi } )$ denote the gradient vector and Hessian matrix of the negative log-penalized likelihood, respectively. For notational simplicity, we omit the explicit dependence on y and x. Given an intermediate update $\dot { \pmb \theta ^ { ( k ) } }$ , one iteration of the minimization of $- \ell _ { \mathrm { P } }$ proceeds as follows:

1. compute a robust Newton-Raphson step that guarantees a stable descent direction:

(a) extract a preconditioner $P = | H _ { i i } | ^ { - 1 / 2 }$ from the diagonal of $H _ { \mathrm { P } }$ , and form the preconditioned Hessian $\tilde { H } _ { \mathrm { P } } = P H _ { \mathrm { P } } P ;$

(b) enforce positive definiteness of $\tilde { \cal H } _ { \mathrm { P } }$ by performing an eigendecomposition $\tilde { \cal H } _ { \mathrm { P } } = { \cal V } \Lambda { \cal V } ^ { T }$ , and replacing any non-positive or near-zero eigenvalues in Λ with the smallest positive eigenvalue;

(c) compute the Newton-Raphson update direction $\Delta ^ { ( k ) }$ as

$$
\begin{array} { l l l } { { { \bf \Delta } { \bf \Delta } { \bf \Delta } } ^ { ( k ) } } & { { = } } & { { { \cal H } _ { \mathrm { P } } ^ { - 1 } ( { \pmb \theta } ^ { ( k ) } ; \hat { \psi } ) { \cal G } _ { \mathrm { P } } ( { \pmb \theta } ^ { ( k ) } ; \hat { \psi } ) } } \\ { { } } & { { = } } & { { { \cal P } V \Lambda ^ { - 1 } V ^ { T } { \cal P } { \cal G } _ { \mathrm { P } } ( { \pmb \theta } ^ { ( k ) } ; \hat { \psi } ) , } } \end{array}\tag{14}
$$

where the matrix-vector product on the right hand-side of 14 is efficiently evaluated from right to left;

2. compute a candidate

$$
\begin{array} { r l r } { \pmb { \theta } ^ { \ast } } & { { } = } & { \pmb { \theta } ^ { ( k ) } - \delta \pmb { \Delta } ^ { ( k ) } , } \end{array}\tag{15}
$$

with initial learning rate $\delta = 1$ 2

3. tune the learning rate by repeatedly halving δ and evaluating 15 until $- \ell _ { \mathrm { P } } \big ( \pmb { \theta } ^ { * } ; \hat { \psi } \big ) < - \ell _ { \mathrm { P } } \big ( \pmb { \theta } ^ { ( k ) } ; \hat { \psi } \big )$ 2

4. updates $\pmb \theta ^ { ( k ) }$ with $\pmb { \theta } ^ { * }$

At convergence, the last update $\pmb \theta ^ { ( k ) }$ is the optimal $\hat { \theta } _ { \hat { \psi } }$

## 3.2 Posterior predictive inference with Monte-Carlo sampling

Given $\hat { \psi }$ and the corresponding maximum a posteriori estimator $\hat { \theta } _ { \hat { \psi } }$ obtained during training, the predictive probability that a new test point $\tilde { \mathbf { x } } _ { i }$ belongs to class $c = 1 , \ldots , K$ is

$$
\begin{array} { r c l } { p ( \tilde { y } _ { i } = c \mid \tilde { \boldsymbol { x } } _ { i } ; \boldsymbol { y } , \boldsymbol { x } ) } & { = } & { \mathrm { E } _ { \tilde { \theta } \sim p ( \boldsymbol { \theta } \mid \boldsymbol { y } ; \boldsymbol { x } , \hat { \psi } ) } \left\{ p ( \tilde { y } _ { i } = c \mid \tilde { \boldsymbol { \theta } } ; \tilde { \mathbf { x } } _ { i } ) \right\} } \\ & { \approx } & { \displaystyle \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \mathrm { S o f t m a x } \left\{ \tilde { u } _ { c } ^ { ( s ) } ( \tilde { \mathbf { x } } _ { i } ) \right\} , } \end{array}
$$

where each of the Monte-Carlo samples is generated by

$$
\begin{array} { r l r } { \tilde { { \pmb u } } _ { c } ^ { ( s ) } ( \tilde { { \pmb x } } _ { i } ) } & { = } & { \hat { \Phi } ( \tilde { \pmb x } _ { i } ) \tilde { \pmb \beta } _ { c } ^ { ( s ) } + \hat { d } _ { c } ( \tilde { \pmb x } _ { i } ) \tilde { \kappa } _ { c } ^ { ( s ) } + \hat { \pmb v } _ { c } ( \tilde { \pmb x } _ { i } ) \tilde { \gamma } ^ { ( s ) } , } \end{array}
$$

using the eigencomposition $H _ { \mathrm { P } } ( \hat { \pmb { \theta } } _ { \hat { \psi } } ; \hat { \pmb { \psi } } ) = \pmb { Q } \pmb { L } \pmb { Q } ^ { T }$ and $\tilde { \pmb { \theta } } ^ { ( s ) } = \hat { \pmb { \theta } } _ { \hat { w } } + \pmb { Q } \pmb { L } ^ { - 1 / 2 } \pmb { Q } ^ { T } \pmb { w } ^ { ( s ) }$ , where ${ \pmb w } ^ { ( s ) } \sim \mathcal { N } ( { \bf 0 } , I )$ We then predict to the class with the highest probability

$$
\begin{array} { r l r } { y _ { i } ^ { * } } & { { } = } & { \arg \operatorname* { m a x } _ { c } p ( \tilde { y } _ { i } = c \mid \tilde { \mathbfit { x } } _ { i } ; \boldsymbol { y } , \boldsymbol { x } ) . } \end{array}
$$

Overall, the empirical Bayes approach to training the hetSNGP model enhances flexibility while enabling data-driven estimation of hyperparameters, requiring only the spectral-normalization factor and network architecture to be tuned, similar to the minimal tuning required by the competitive baseline DDU. We now assess the performance of our approach on large-scale simulated data from LHC. The synthetic two circles toy dataset, comparing auto-hetSNGP with hetSNGP can be found in Appendix B.

## 4 Results for LHC simulated tracking data

The training set consists of 1'065'273 particles, including 475'362 in the stable region and 589'911 in the unstable region. Each particle is described by 157 features that represent

• A seed index, specifying which of the sixty realizations of the nonlinear magnetic field errors assigned to the magnets of the accelerator model is used;

• 8 parameters describing the phase-coordinates (in terms of radius and angle of polar coordinates), the beam index (specifying whether the clockwise or counter-clockwise beam is used), the value of the linear betatron tunes, the value of the linear chromaticities, the strength of the Landau octupole magnets;

• 10 MAD-X optimal parameters,

• 72 beam optics parameters.

• 7 parameters describing the dependence of the betatron frequency with amplitude, which are related to the nonlinear dynamics

The test set comprises 131'593 particles, with 57'918 in the stable region and 73'675 in the unstable region, so the stable and unstable classes comprise 44% and 55% of the data, respectively. We trained a fully deterministic residual neural network and the auto-hetSNGP model introduced in Section 2, and compared their uncertainty estimates across three different methods: Monte Carlo Dropout [Gal and Ghahramani, 2016], DDU and the empirical Bayes introduced in Section 3. DE was infeasible due to computational constraints. We distributed the training across 16 compute nodes for 250 epochs using mini-batches of size 512. We used an initial learning rate of 0.0001 combined with a cosine annealing scheduler with warm restarts every 100 iterations. In the auto-hetSNGP model, we set the number of Monte Carlo samples to $S = 2 0 4 8$ and used $K = R = 2$ . Note that in practice, we found $R = K$ sufficient for this binary classification task, despite the low-rank assumption $R \ll K$ that motivates the general formulation.

Table 1 summarizes the characteristics of the best performing architectures. Since Monte Carlo Dropout is activated only at inference time, we used the same underlying model that was trained for DDU. Nevertheless. we duplicated its specifications for clarity and ease of reference. The three methods used the residual neural network illustrated in Figure 2, which consists of two linear layers per basic block and a single linear layer per shortcut block. The number of units per layer and the number of basic blocks vary between configurations. Specifically, both MC Dropout and DDU used a network with seven basic blocks. The number of units per layer decreases progressively across blocks: both layers of the first block have 256 units, those of the second block have 128 units, and so on, with the seventh block containing 4 units. Similarly, auto-hetSNGP consists of six basic blocks, each with two linear layers whose units are specified per block. Although both architectures exhibit comparable model complexity, auto-hetSNGP converges in significantly fewer training epochs.

Table 1: Distinguishing specifications of the best performing residual models. The “units per layer" column encodes two pieces of information: the length of the list is the number of basic blocks, each comprising two linear layers, while the values in the list specify the number of units in each layer. The auto-hetSNGP represents the approach proposed in this paper.
<table><tr><td>Model</td><td>Best epoch</td><td>Dropout rate</td><td>SN factor</td><td>Units per layer</td></tr><tr><td>MC Dropout</td><td>114</td><td>0.1</td><td>2.0</td><td>[256, 128, 64, 32, 16, 8, 4]</td></tr><tr><td>DDU</td><td>114</td><td>0.1</td><td>2.0</td><td>[256, 128, 64, 32, 16, 8, 4]</td></tr><tr><td>auto-hetSNGP</td><td>39</td><td>0.1</td><td>1.1</td><td>[256, 128, 64, 32, 16, 8]</td></tr></table>

Table 2 shows a performance comparison of the best models evaluated on the test set, using the Expected Calibration Error (ECE) [Naeini et al., 2015, Guo et al., 2017] to measure uncertainty calibration. Let TP, TN, FP, and FN denote the number of true positives, true negatives, false positives, and false negatives, respectively, positive being the stable region and negative being the unstable one. The metrics used for evaluating accuracy are defined as

$$
{ \begin{array} { r c l l } { { \mathrm { S e n s i t i v i t y } } } & { = } & { { \displaystyle { \frac { \mathrm { T P } } { \mathrm { T P } + { \mathrm { F N } } } } } , } & { { \mathrm { P r e c i s i o n } } = { \frac { \mathrm { T P } } { \mathrm { T P } + { \mathrm { F P } } } } , } & { { \mathrm { S p e c i f i c i t y } } = { \frac { \mathrm { T N } } { \mathrm { T N } + { \mathrm { F P } } } } , } \\ { { } } & { } & { } & { } \\ { F _ { \mathrm { 1 - s c o r e } } } & { = } & { 2 \cdot { \displaystyle { \frac { \mathrm { P r e c i s i o n } \cdot { \mathrm { S e n s i t i v i t y } } } { \mathrm { P r e c i s i o n } + { \mathrm { S e n s i t i v i t y } } } } } . } \end{array} }
$$

The ECE measure is a weighted average over equally spaced bins $B _ { 1 } , \ldots , B _ { M }$ of the data:

$$
\begin{array} { r c l } { { E C E } } & { { = } } & { { \displaystyle \sum _ { m = 1 } ^ { M } \frac { | B _ { m } | } { n } | \mathrm { a c c } ( B _ { m } ) - \mathrm { c o n f } ( B _ { m } ) | , } } \\ { { \mathrm { a c c } ( B _ { m } ) } } & { { = } } & { { \displaystyle \frac { 1 } { | B _ { m } | } \sum _ { i \in B _ { m } } 1 \left\{ y _ { i } ^ { * } = y _ { i } \right\} , \quad \mathrm { c o n f } ( B _ { m } ) = \displaystyle \frac { 1 } { | B _ { m } | } \sum _ { i \in B _ { m } } p ( y _ { i } ^ { * } \mid \tilde { x } _ { i } ; y , x ) . } } \end{array}
$$

The MC Dropout results were obtained by averaging the softmax outputs over 5 stochastic forward passes.

Table 2: Performance of the best models evaluated on the test set. The auto-hetSNGP represents the approach proposed in this paper.
<table><tr><td rowspan="2">Model</td><td colspan="7">Accuracy  $[ \% ]$ </td><td rowspan="2">Uncertainty</td></tr><tr><td>Stable</td><td>Unstable</td><td>Overall</td><td>Sensitivity</td><td>Precision</td><td>Specificity</td><td> $F _ { \mathrm { 1 ^ { - S C O r e } } }$  ECE</td></tr><tr><td>MC Dropout</td><td>94.98</td><td>97.06</td><td>96.14</td><td>94.98</td><td>96.21</td><td>97.06</td><td>95.59</td><td>0.020</td></tr><tr><td>DDU</td><td>96.83</td><td>97.14</td><td>97.00</td><td>96.83</td><td>96.38</td><td>97.14</td><td>96.60</td><td>0.016</td></tr><tr><td>auto-hetSNGP</td><td>96.34</td><td>97.19</td><td>96.82</td><td>96.34</td><td>96.42</td><td>97.19</td><td>96.38</td><td>0.019</td></tr></table>

Overall, DDU and auto-hetSNGP achieve comparable results on the test set, marginally outperforming MC Dropout. However, DDU's uncertainty estimates become less interpretable on the three test configurations originally shown in Figure 1. In Appendix A The corresponding uncertainty visualizations are presented in Figures 3–5, where the high uncertainty is depicted in red and the low uncertainty in yellow. For both MC Dropout and auto-hetSNGP, we use the predictive variance, rescaled by a factor of 0.25 to be within the interval (0, 1), as a proxy for uncertainty

The ideal behavior for uncertainty estimation is that the model exhibits high uncertainty when making incorrect predictions and low uncertainty when the predictions are correct. In particular, the decision boundary. where class overlap and ambiguity are expected, should correspond to regions of high uncertainty, while the interior of well-separated classes should exhibit low uncertainty. However, DDU displays counterintuitive behavior: for configurations 1186 and 29876, it shows higher uncertainty in the stable region than in the unstable region, and its uncertainty estimates are unreliable for configuration 29472. In contrast, autohetSNGP produces more consistent and interpretable uncertainty estimates that align more closely with the expected behavior. Interestingly, these three configurations highlight the trade-off between robustness and flexibility that is often required in real-world applications. Although DDU achieves slightly better overall performance on the full test set, auto-hetSNGP outperforms it on localized test configurations, demonstrating that increased model flexibility can provide a tangible advantage in challenging scenarios. This difference stems from the underlying modeling assumptions: DDU fits a single, shared Gaussian distribution for each class, whereas auto-hetSNGP explicitly incorporates class-dependent noise, allowing it to better capture heteroscedastic uncertainty across regions of the input space. However, the overall poor performance of the MC dropout suggests that the model may be overparameterized, with important units being deactivated at inference time due to the stochastic nature of the dropout. We note that, due to computational constraints the results reflect a single training run per method; variance between random seeds was not assessed.

## 5 Discussion

We considered the computational and modeling challenges associated with uncertainty quantification in large-scale, simulation-driven scientific problems. Specifically, we proposed a flexible hetSNGP model trained with an empirical Bayesian approach that integrates hyperparameter optimization directly into the training loop. This enables efficient and robust learning without requiring costly validation or manual tuning, two common bottlenecks in Bayesian neural models. Our approach extends the SNGP framework by incorporating class-dependent heteroscedasticity with learnable variances, allowing the model to represent both predictive uncertainty and structured noise in a principled manner. We demonstrated the practical benefits of this architecture on the task of DA prediction in circular particle accelerators, where accurate uncertainty estimation is critical for guiding expensive tracking simulations. Compared to existing methods, our approach achieves competitive predictive performance while offering well-calibrated uncertainty estimates and training efficiency similar to the simplest state-of-the-art method. In particular, it is more computationally tractable than DE and exhibits a more consistent uncertainty behavior than DDU on configuration-specific test cases, where standard methods tend to misrepresent class boundaries and overestimate confidence. Our technical contribution is the separation of the model's parameters into deterministic components ψ and stochastic components θ, with distinct functions during training and inference. By restricting optimization to the deterministic parameters ψ of the neural network, we accelerate the training process. The stochastic parameters θ, which govern the predictive uncertainty of the model, are then sampled at inference time using the Laplace approximation centered on their mode. This separation enables scalable and memory-efficient training, while still allowing for a semi-Bayesian treatment of uncertainty at test time. The low dimensionality of θ makes posterior inference tractable via a preconditioned Newton-Raphson procedure, further contributing to the method's computational efficiency.

Our framework is broadly applicable beyond accelerator physics. Many domains, such as climate science, biological modeling, and engineering design, suffer from similar computational constraints and demand high-fidelity uncertainty estimates for decision-making. The empirical Bayes methodology we propose generalizes to these contexts, offering a path forward for scalable uncertainty-aware modeling under resource limitations. Future work could explore replacing the Laplace approximation with more complex posterior inference techniques. Finally, coupling our method with acquisition functions in active learning or Bayesian optimization pipelines may further reduce the need for expensive simulations by focusing computational effort on uncertain regions of parameter space.

## Acknowledgments

This work is funded by the Swiss Data Science Center project grant C20-10.

## References

Viktor Bengs, Eyke Hüllermeier, and Willem Waegeman. On second-order scoring rules for epistemic uncertainty quantification. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 2078–2091. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/bengs23a.html.

Oliver S. Brüning, Paul Collier, Philippe Lebrun, Steven Myers, Ranko Ostojic, John Poole, and Paul Proudlock. LHC Design Report. CERN Yellow Rep. Monogr. CERN, Geneva, 2004. doi: 10.5170/ CERN-2004-003-V-1.

Mark Collier, Basil Mustafa, Efi Kokiopoulou, Rodolphe Jenatton, and Jesse Berent. Correlated inputdependent label noise in large-scale image classification. pages 1551–1560, 06 2021. doi: 10.1109/CVPR46437. 2021.00160.

Davide Di Croce, Massimo Giovannozzi, Ekaterina Krymova, Tatiana Pieloni, Stefano Redaelli, Mike Seidel, Rogelio Tomás, and Frederik F. Van der Veken. Optimizing dynamic aperture studies with active learning Journal of Instrumentation, 19(04):P04004, apr 2024. doi: 10.1088/1748-0221/19/04/P04004. URL https://doi.org/10.1088/1748-0221/19/04/P04004.

Andre Esteva, Brett Kuprel, Roberto A. Novoa, Justin Ko, Susan M. Swetter, Helen M. Blau, and Sebastian Thrun. Dermatologist-level classification of skin cancer with deep neural networks. Nature, 542:115–118, 2017.

Vincent Fortuin, Mark Collier, Florian Wenzel, James Urquhart Allingham, Jeremiah Zhe Liu, Dustin Tran, Balaji Lakshminarayanan, Jesse Berent, Rodolphe Jenatton, and Effrosyni Kokiopoulou. Deep classifiers with label noise modeling and distance awareness. Transactions on Machine Learning Research, 2022. ISSN 2835-8856. URL https://openreview.net/forum?id=Id7hTt78FV.

Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In Maria Florina Balcan and Kilian Q. Weinberger, editors, Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 1050–1059, New York, New York, USA, 20–22 Jun 2016. PMLR. URL https://proceedings.mlr. press/v48/gal16.html.

Massimo Giovannozzi. A proposed scaling law for intensity evolution in hadron storage rings based on dynamic aperture variation with time. Phys. Rev. Spec. Top. Accel Beams, 15:024001, 2012. doi: 10.1103/ PhysRevSTAB.15.024001.

Massimo Giovannozzi. Single-particle nonlinear beam dynamics. In B. Foster, editor, Oxford Research Encyclopedia of Physics. Oxford University Press, 2026. ISBN 9780197851753. doi: 10.1093/acrefore/ 9780190871994.013.139. URL https://doi.org/10.1093/acrefore/9780190871994.013.139.

Massimo Giovannozzi, Walter Scandale, and Ezio Todesco. Prediction of long-term stability in large hadron colliders. Part. Accel., 56:195, 1997.

Massimo Giovannozzi, Walter Scandale, and Ezio Todesco. Dynamic aperture extrapolation in presence of tune modulation. Phys. Rev. E, 57:3432, 1998. doi: 10.1103/PhysRevE.57.3432.

Massimo Giovannozzi, Ewen H. Maclean, Carlo Emilio Montanari, Gianluca Valentino, and Frederik F. Van der Veken. Machine learning applied to the analysis of nonlinear beam dynamics simulations for the CERN large hadron collider and its luminosity upgrade. Information, 12(2), 2021. ISSN 2078-2489. doi: 10.3390/info12020053. URL https://www.mdpi.com/2078-2489/12/2/53.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In International conference on machine learning, pages 1321–1330. PMLR, 2017.

Yu Huang and Yue Chen. Autonomous driving with deep learning: A survey of state-of-art technologies. CoRR, abs/2006.06091, 2020. URL https://arxiv.org/abs/2006.06091.

Nikita Kotelevskii and Maxim Panov. Predictive uncertainties based on proper scoring rules. In ICML 2024 Workshop on Structured Probabilistic Inference & Generative Modeling, 2024. URL https://openreview. net/forum?id=M7SUiOqzz0.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper\_files/paper/ 2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf.

Jeremiah Zhe Liu, Shreyas Padhy, Jie Ren, Zi Lin, Yeming Wen, Ghassen Jerfel, Zachary Nado, Jasper Snoek, Dustin Tran, and Balaji Lakshminarayanan. A simple approach to improve single-model deep uncertainty via distance-awareness. Journal of Machine Learning Research, 24(42):1–63, 2023. URL http://jmlr.org/papers/v24/22-0479.html.

Riccardo De Maria et al. SixTrack Version 5: Status and New Developments. In Proc. IPAC'19, pages 3200–3203. JACoW Publishing, Geneva, Switzerland, 2019. doi: 10.18429/JACoW-IPAC2019-WEPTS043. URL http://accelconf.web.cern.ch/ipac2019/papers/WEPTS043.pdf.

Carlo Emilio Montanari, Robert B. Appleby, Davide Di Croce, Massimo Giovannozzi, Tatiana Pieloni, Stefano Redaelli, and Frederik F. Van der Veken. Machine learning techniques for uncertainty estimation in dynamic aperture prediction. Computers, 14(7), 2025. ISSN 2073-431X. doi: 10.3390/computers14070287. URL https://www.mdpi.com/2073-431X/14/7/287.

Jishnu Mukhoti, Andreas Kirsch, Joost van Amersfoort, Philip H.S. Torr, and Yarin Gal. Deep deterministic uncertainty: A new simple baseline. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24384–24394, June 2023.

Mahdi Pakdaman Naeini, Gregory F. Cooper, and Milos Hauskrecht. Obtaining well calibrated probabilities using bayesian binning. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, AAAI'15, page 2901–2907. AAAI Press, 2015. ISBN 0262511290.

Janis Postels, Mattia Segu, Tao Sun, Luc Van Gool, Fisher Yu, and Federico Tombari. On the practicality of deterministic epistemic uncertainty. International Conference on Machine Learning, 2022.

Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. In J. Platt, D. Koller, Y. Singer, and S. Roweis, editors, Advances in Neural Information Processing Systems, volume 20. Curran Associates, Inc., 2007. URL https://proceedings.neurips.cc/paper\_files/paper/2007/file/ 013a006f03dbc5392effeb8f18fda755-Paper.pdf.

Michael Schenk, Loic Coyle, Massimo Giovannozzi, Ekaterina Krymova, Alessio Mereghetti, Guillaume Obozinski, and Tatiana Pieloni. Modeling Particle Stability Plots for Accelerator Optimization Using Adaptive Sampling. In Proc. IPAC'21, pages 1923–1926. JACoW Publishing, Geneva, Switzerland, 2021. doi: 10.18429/JACoW-IPAC2021-TUPAB216. URL https://jacow.org/ipac2021/papers/TUPAB216.pdf.

Kajetan Schweighofer, Lukas Aichberger, Mykyta Ielanskyi, and Sepp Hochreiter. Introducing an improved information-theoretic measure of predictive uncertainty. In NeurIPS 2023 workshop: Information-Theoretic Principles in Cognitive Systems, 2023. URL https://openreview.net/forum?id=6K8c90L2mM.

Freddie Bickford Smith, Jannik Kossen, Eleanor Trollope, Mark van der Wilk, Adam Foster, and Tom Rainforth. Rethinking aleatoric and epistemic uncertainty. In NeurIPS 2024 Workshop on Bayesian Decision-making and Uncertainty, 2024. URL https://openreview.net/forum?id=WIjgbXd2zK.

Ezio Todesco and Massimo Giovannozzi. Dynamic aperture estimates and phase-space distortions in nonlinear betatron motion. Phys. Rev. E, 53:4067–4076, 4 1996. doi: 10.1103/PhysRevE.53.4067. URL https://1ink.aps.org/doi/10.1103/PhysRevE.53.4067.

Frederik Van der Veken, Runa Akbari, Michiel Bogaert, Elena Fol, Massimo Giovannozzi, Amy Lowyck, Carlo Emilio Montanari, and Wietse Van Goethem. Determination of the Phase-Space Stability Border with Machine Learning Techniques. JACoW IPAC, 2022:183–186, 2022. doi: 10.18429/ JACoW-IPAC2022-MOPOST047. URL https://cds.cern.ch/record/2845745.

Frederik F Van der Veken, Massimo Giovannozzi, Ewen H Maclean, Carlo Emilio Montanari, and Gianluca Valentino. Using Machine Learning to Improve Dynamic Aperture Estimates. JACo W IPAC, 2021:134–137, 2021. doi: 10.18429/JACoW-IPAC2021-MOPAB028. URL https://cds.cern.ch/record/2804876.

Lisa Wimmer, Yusuf Sale, Paul Hofman, Bernd Bischl, and Eyke Hüllermeier. Quantifying aleatoric and epistemic uncertainty in machine learning: Are conditional entropy and mutual information appropriate measures? In Robin J. Evans and Ilya Shpitser, editors, Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence, volume 216 of Proceedings of Machine Learning Research, pages 2282–2292. PMLR, 31 Jul–04 Aug 2023. URL https://proceedings.mlr.press/v216/wimmer23a.html

## A Performance on different configurations

We now present different configurations corresponding to low, medium, and high DA. Intuitively, the boundary between the stable and unstable regions should correspond to higher uncertainty estimates, while the complementary regions are expected to exhibit lower uncertainty. For the predicted boundary, a wellperforming method is expected to demonstrate both a low number of prediction errors around the DA region and a qualitatively reasonable concentration of uncertainty around the predicted boundary

## A.1 Comparison for configuration 1186. Medium DA.

In Figure 3, the plots on the left show that all methods perform reasonably well, although MC Dropout overestimates the DA for larger angles. Based on the plots on the right, the uncertainty values qualitatively correspond to the expected behavior for MC Dropout and auto-hetSNGP, but not for DDU, for which the uncertainty is high in the stable-particle region.

## A.2 Comparison for configuration 29472. Low DA.

In Figure 4, based on the plots on the left, DDU and auto-hetSNGP demonstrate good overall accuracy, whereas MC Dropout significantly overestimates the DA. Based on the plots on the right, MC Dropout uncertainty estimates are high, as expected, around the incorrectly predicted class boundary. DDU demonstrates unexpectedly high uncertainty in the unstable region, while auto-hetSNGP reasonably highlights the uncertain boundary.

## A.3 Comparison for configuration 29876. High DA.

In Figure 5, as in the previous configuration, the plots on the left show that DDU and auto-hetSNGP demonstrate good overall accuracy, whereas MC Dropout significantly underestimates DA. Based on the plots on the right, high MC Dropout uncertainty estimates are concentrated around the incorrectly predicted class boundary. DDU demonstrates unexpectedly high uncertainty in the stable region, while auto-hetSNGP reasonably highlights the uncertain boundary.

Predicted dynamic aperture for configuration 1186 Accuracy: 92.75%  
![](images/a5e7adff83085541806f7cabca8466774c8bf10d03eee9b42388320c24c6c67f.jpg)

Uncertainty for configuration 1186  
![](images/17ef76f3be9d554761489c42ec7e57d6242ab1d91f5f217d50dcc00da7a7dfd8.jpg)  
a) Performance of MC Dropout.

Predicted dynamic aperture for configuration 1186 Accuracy: 95.94%  
![](images/5c86ea2d6f25dda48af560acbfd89cf64e3819c5c33ca2e00d5fce93c0c71bef.jpg)

Uncertainty for configuration 1186  
![](images/1ac6cc2c42c7f6fd5f2c41c968d03b5146162991725245017bc6573e89d191c4.jpg)  
b) Performance of DDU.

Predicted dynamic aperture for configuration 1186 Accuracy: 95.88%  
![](images/553f52a21dc561487ffdae73e96e209faedd976919752b4db418c18e3f212b98.jpg)

Uncertainty for configuration 1186  
![](images/aa88499424c85045686f030e206680678048885f1bf1ef2a75fe3ac68e238fd1.jpg)  
c) Performance of auto-hetSNGP.  
Figure 3: Comparison for configuration 1186. On the left: predicted stability of the particles. If the model prediction was not correct, the true label is depicted with darker color. On the right: uncertainty for the prediction.

Predicted dynamic aperture for configuration 29472 Accuracy: 75.16%  
![](images/928b9f84754f93c2ec23ee4e81c85b2604e1c04493ce6cd95da34fd82b86626f.jpg)

Uncertainty for configuration 29472  
![](images/51621121439cf08536a57817b51e0f850b4b4d1d2326a43b2ec858fa2f7227da.jpg)  
a) Performance of MC Dropout.

Predicted dynamic aperture for configuration 29472 Accuracy: 98.59%  
![](images/dabe6c1eb7471323b2a00df5ce3b437807eeffbc429ca8a050e86e81509981a3.jpg)

Uncertainty for configuration 29472  
![](images/05a34cd3455be76db56f662b332fddeb0c39913e32a05b2a1c7f944064153a1e.jpg)  
b) Performance of DDU.

Predicted dynamic aperture for configuration 29472 Accuracy: 98.78%  
![](images/707e65d1fafe041d4b3229febfbb97af0ad528e2c81738718bcfaf2a8aeb5f63.jpg)

Uncertainty for configuration 29472  
![](images/146509a65ae7daa2be9339a65f77d945225e3b921850f1a42e26e7661d33b009.jpg)  
c) Performance of auto-hetSNGP.  
Figure 4: Comparison for configuration 29472. On the left: predicted stability of the particles. If the model prediction was not correct, the true label is depicted with darker color. On the right: uncertainty for the prediction.

![](images/9d80536ab2c9fd0652c513339dde9420fe92e195da1fef2be0d95d26fb6d21dd.jpg)

![](images/d4befb3aed55494b2666e21570cd35aa14d871642f61324a3f3e35bfa7e081d3.jpg)  
a) Performance of MC Dropout.

![](images/07844f6196cd81cbba21df2f4365b91dee0275e00d8cf935c0d59138d82e47dc.jpg)

![](images/e90d54020f1b6e1db62bd6fd829606efb7a49f4504a67d790b1b391fd14ca935.jpg)  
b) Performance of DDU.

Predicted dynamic aperture for configuration 29876 Accuracy: 99.38%  
![](images/a77c17491fea0c61bc118c6aa68a729c9a6abb2efb65f2989b88eca752ecc964.jpg)  
c) Performance of auto-hetSNGP.

Uncertainty for configuration 29876  
![](images/e1bad05070dd49bc4c928db1df4bf3ba42df70d58bb79f44ed1c4f2f4794dbae.jpg)  
Figure 5: Comparison for configuration 29876.

## B 2D example

Two circles data. We generate a synthetic binary classification dataset inspired by concentric circles with instance-dependent label noise. Let $\mathcal { D } _ { \mathrm { i n n e r } }$ and $\mathcal { D } _ { \mathrm { o u t e r } }$ denote uniform distributions along the circumference of an inner and an outer circle, respectively, each perturbed by isotropic Gaussian noise with variance $\sigma _ { \mathrm { g e o m } } ^ { 2 } = 0 . 1$ . We sample the inputs $\pmb { x } \in \mathbb { R } ^ { 2 }$ and the class label as follows

$$
\begin{array} { r l r } { y _ { \mathrm { t r u e } } } & { = } & { \left\{ \begin{array} { l l } { 0 , } & { x \sim \mathcal { D } _ { \mathrm { i n n e r } } , } \\ { 1 , } & { x \sim \mathcal { D } _ { \mathrm { o u t e r } } . } \end{array} \right. } \end{array}
$$

We flip each label with a probability that depends on the orientation of x, $\pmb { w } = ( 0 , 1 ) ^ { \top }$ say, such that

$$
p _ { \mathrm { H i p } } ( \pmb { x } ) = a \left( \frac { { \pmb w } ^ { \top } { \pmb x } } { \| { \pmb w } \| \| { \pmb x } \| } + 1 \right) ,\tag{16}
$$

where $a \in ( 0 , 1 )$ . The observed new label y has then probability mass function $P ( \tilde { y } = y _ { \mathrm { t r u e } } \mid x ) = 1 - p _ { \mathrm { f l i p } } ( \pmb { x } )$ with corresponding posterior

$$
\begin{array} { r c l } { P ( \tilde { y } = 1 \mid x ) } & { = } & { P ( \tilde { y } = y _ { \mathrm { t r u e } } \mid x ) P ( y _ { \mathrm { t r u e } } = 1 \mid x ) + P ( \tilde { y } \ne y _ { \mathrm { t r u e } } \mid x ) P ( y _ { \mathrm { t r u e } } = 0 \mid x ) } \\ & { = } & { \{ 1 - p _ { \mathrm { f i p } } ( \pmb { x } ) \} P ( y _ { \mathrm { t r u e } } = 1 \mid x ) + p _ { \mathrm { f i p } } ( \pmb { x } ) \{ 1 - P ( y _ { \mathrm { t r u e } } = 1 \mid x ) \} . } \end{array}
$$

Figure 6 illustrates an example of such simulations for $a \in \{ 0 . 0 0 0 1 , 0 . 0 1 , 0 . 1 , 0 . 3 \}$ in 16. We now define the uncertainty measures used for evaluating the performance of hetSNGP, auto-hetSNGP, DE with 10 members, and DDU. Let $u ( { \pmb x } )$ be the second-to-last layer representation. We assess DE and SNGP-based models using the predictive variance of the logits

$$
\hat { U } ^ { \mathrm { v a r } } ( { \pmb x } ) = \mathrm { V a r } ( u ( { \pmb x } ) ) ,\tag{17}
$$

where the expectation is taken with respect to the noise in posterior distribution for SNGP-type methods, or, for DE, with respect to the randomness that defines ensemble members. For the DDU we use the logarithm of feature density estimated by DDU

$$
\hat { U } ^ { \mathrm { D D U } } ( { \pmb x } ) = - \log p _ { \mathrm { D D U } } \left\{ { \pmb u } ( { \pmb x } ) \right\} .\tag{18}
$$

We randomly generated a training set of 1000 points and a test set of 500 points, and illustrated the results for uncertainty quantification in Figure 7, where the lower uncertainty regions are highlighed with the blue color. All the models consisted of 4 ResNet blocks with width 128, dropout probability was set to 0.1. While the models exhibit comparable performance, auto-hetSNGP shows a marginally higher accuracy, see Table 3. Note that the DDU and DE baselines were not extensively tuned on this synthetic benchmark, so the accuracy gap relative to the (auto-)hetSNGP variants should not be over-interpreted as a general performance ranking.

Table 3: Test accuracy (%) across methods and flip scales.
<table><tr><td>Flip scale a</td><td>auto-hetSNGP</td><td>hetSNGP</td><td>DDU</td><td>DE</td></tr><tr><td>0.0001</td><td>92.0</td><td>90.0</td><td>75.6</td><td>74.0</td></tr><tr><td>0.01</td><td>91.2</td><td>90.0</td><td>74.0</td><td>74.4</td></tr><tr><td>0.1</td><td>86.4</td><td>80.0</td><td>67.6</td><td>68.4</td></tr><tr><td>0.3</td><td>62.8</td><td>63.6</td><td>57.2</td><td>58.8</td></tr></table>

![](images/3e30c2fe6ed259bef60a95fb76c6f4c72cef5fb940fed898c85a045db312a092.jpg)  
Figure 6: Data with different noise level contamination.

![](images/9d628cf1b06be73ddd9cfe88be02e388d0f9604e0389ec81999448d74199cf03.jpg)  
Figure 7: Data density, equation 17 variance of predictive logits based uncertainty estimates for hetSNGP methods and DE, DDU feature-density based uncertainty equation 18.