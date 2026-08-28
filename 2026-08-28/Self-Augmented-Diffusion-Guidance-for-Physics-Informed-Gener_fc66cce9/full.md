# Self-Augmented Diffusion Guidance for Physics-Informed Generation

Akira Osaka School of Engineering The University of Tokyo

Naoya Takeishi School of Engineering The University of Tokyo

Takehisa Yairi School of Engineering The University of Tokyo

akr-osaka@g.ecc.u-tokyo.ac.jp

ntake@g.ecc.u-tokyo.ac.jp

yairi@g.ecc.u-tokyo.ac.jp

## Abstract

Diffusion models can be used to generate spatiotemporal signals of physical phenomena, such as time-series images of fluid dynamics. However, a major limitation of standard diffusion models is that they do not incorporate constraints derived from the underlying physical laws. Consequently, generated samples may appear visually plausible while deviating substantially from the true dynamics. In this study, we propose a simple yet effective physics-informed approach based on diffusion guidance with self-generated data augmentation. The proposed method learns the data distribution conditioned on the degree of deviation from the physically correct dynamics and generates samples by explicitly setting the deviation condition to be zero. The method decouples the evaluation of the governing equations from the diffusion model training and sampling processes, avoiding the need to solve the governing equations at every iteration of the denoising process. This design makes the method applicable to problems requiring computationally expensive numerical simulations and enables faster sample generation. Experimental results demonstrate that the proposed model not only significantly reduces the deviations compared with standard diffusion models but also achieves further reductions when combined with existing physics-constrained diffusion methods.

## 1 Introduction

Diffusion models provide a powerful framework for learning probability distributions and generating realistic samples. Their applications extend beyond general image generation to physical phenomena in fields such as fluid dynamics (Shu et al., 2023), oceanography (Xu & Li, 2025), and atmospheric science (Wang et al., 2025). However, when applied to physical problems, standard diffusion models, including score-based models (Song & Ermon, 2019; Song et al., 2021b) and denoising models (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2021a), face a critical limitation that they do not explicitly account for the underlying physical laws. Consequently, they may generate samples that appear visually plausible but are physically inconsistent, reducing their validity and practical utility.

Several previous studies have addressed this issue by developing diffusion models that enforce adherence to governing physical laws. The Physics-Informed Diffusion Model (PIDM) (Bastek et al., 2025) reduces discrepancies between generated samples and the underlying physical constraints by incorporating the residuals of the governing equations into the training loss. PIDM imposes these constraints only during training, leaving the sampling process unchanged from that of standard diffusion models. Among the methods that incorporate physical constraints during sampling, CoCoGen (Jacobsen et al., 2025) modifies the sampling procedure so that denoising proceeds in a direction that reduces the constraint violation. Similarly, Physics-Guided Diffusion (PG Diffusion) (Shu et al., 2023) integrates residual information into the model architecture using classifier-free guidance. Regarding PG Diffusion, Bastek et al. (2025) point out that although the method incorporates gradient information into the model, it does not explicitly enforce residual minimization, resulting in insufficient physical consistency. This observation highlights the need for methods that explicitly impose a zero-residual condition during sampling.

Table 1: Comparison of assumptions required by previous physics-informed generative methods and ours.
<table><tr><td>Method</td><td>Constraint Evaluation during Training</td><td>Constraint Evaluation during Sampling</td><td>Gradient of Constraint</td><td>Projection/ Correction</td></tr><tr><td>PG Diffusion (Shu et al., 2023)</td><td>√</td><td>√</td><td>√</td><td></td></tr><tr><td>PIDM (Bastek et al., 2025)</td><td>√</td><td></td><td>√</td><td></td></tr><tr><td>CoCoGen (Jacobsen et al., 2025)</td><td></td><td>√</td><td>√</td><td></td></tr><tr><td>ECI (Cheng et al., 2025)</td><td></td><td></td><td></td><td>√</td></tr><tr><td>Proposed</td><td></td><td></td><td></td><td></td></tr></table>

Note: ECI is based on flow matching, whereas PG Diffusion, PIDM, CoCoGen, and ours are based on diffusion models.

Another critical challenge is computational cost. Existing methods (Bastek et al., 2025; Jacobsen et al., 2025; Shu et al., 2023) require the evaluation and differentiation of the constraints during either training or sampling. In some problems, such as the generation of time-series data in fluid dynamics, snapshots are generated at spatial and temporal resolutions coarser than those required for stable and accurate numerical differentiation. Such nature of data makes it inapplicable to evaluate the residuals of the partial differential equations (PDEs) as a mean to impose physics constraints expressed as PDEs. In such cases, instead, the degrees of constraint satisfaction should be evaluated by measuring the deviations between generated samples and physically consistent time-series samples obtained by numerically integrating the governing equations with smaller internal steps. Evaluating such integration-based constraints and their derivatives are typically more computationally expensive than diffusion model training or sampling, especially in high-dimensional settings, and can result in prohibitively long runtimes. To broaden the applicability of physics-informed generative models, it is important to develop a method that separates constraint evaluation from diffusion training and sampling while still incorporating physical information into the model.

In this study, we propose a simple yet effective physics-aware diffusion guidance method based on the classifier-free guidance framework. Our method is based on a self-augmentation strategy in which a base diffusion model is used to generate a kind of negative samples, while the original dataset is treated as positive samples. The diffusion model is then guided to generate samples that are closer to the positive samples and further from the negative. Furthermore, for problems requiring computationally expensive numerical simulations, our approach decouples constraint evaluation from diffusion model training and sampling. It eliminates the need for calculating gradients of the constraint functions during either training or sampling

Unlike PG Diffusion (Shu et al., 2023), which also employs classifier-free guidance, our method explicitly enforces physical consistency by sampling from a diffusion model conditioned on zero constraint violation. The Extrapolation-Correction-Interpolation (ECI) sampling method (Cheng et al., 2025), based on flow matching (Lipman et al., 2023), also achieves gradient-free generation by taking advantage of its deterministic flow structure. However, it assumes the availability of an appropriate correction algorithm that projects the samples to satisfy the constraints. Our approach further removes this requirement and is applicable as long as the constraints can be evaluated somehow. Table 1 compares the requirements of previous approaches with those of our method. Our method does not rely on any of the assumptions required by the existing methods.

We evaluate our method on fluid dynamics tasks and demonstrate that it significantly reduces deviations from the underlying physical laws. Our approach can also be readily combined with existing physics-informed diffusion methods in a plug-and-play manner. We show that combining it with previous approaches such as PIDM and CoCoGen yields further improvements in physical consistency.

## 2 Related work

As diffusion models and flow matching have become increasingly popular for generative modeling, constrained generation has attracted growing attention in recent years. One of the early studies in this area is Fishman et al. (2023), which investigated diffusion models defined on constrained domains. Subsequent studies have explored several directions, such as constraint-aware diffusion training (Khalafi et al., 2024), gradient-guided sampling for constraint satisfaction (Huang et al., 2024b), and projection onto constrained regions during sampling (Christopher et al., 2024). In flow matching, Li et al. (2026b) utilized its differential-equation structure to formulate constrained generation as an optimal control problem.

Generative modeling under physical constraints is a major application area of constrained generation. A common approach in physics-informed diffusion models and flow matching is to incorporate physical knowledge into the sampling process. Yuan et al. (2023) proposed PhysDiff, which directly modifies intermediate samples during denoising via imitation learning to generate physically plausible human motions. In the context of flow matching, Cheng et al. (2025), Utkarsh et al. (2025), and Christopher et al. (2026) developed methods that extrapolate intermediate states to the terminal time, project or refine the resulting terminalstate estimates to satisfy the constraints, and then propagate the corrections back to the current sampling steps. These methods exploit the deterministic nature of flow matching and apply the constraint operations to estimated terminal states rather than directly to intermediate states. For diffusion models, Li et al. (2026a) also introduced projections toward feasible regions during denoising. Similarly, Blanke et al. (2026) integrated projection steps into Langevin sampling. Diffusion posterior sampling (Chung et al., 2023; Yao et al., 2025), which is designed to solve inverse problems, has also been applied to physics-constrained generation. Huang et al. (2024a) and Gallon et al. (2026) proposed methods that correct intermediate samples along the gradient of the residual. Peng et al. (2026) injected the physics gradient into the noise-prediction steps. Similarly, CoCoGen (Jacobsen et al., 2025) modifies the sampling process so that the denoising steps move in a direction that reduces the residual:

$$
x _ { i - 1 } = \mathrm { S o l v e r } ( x _ { i } ) - \gamma \nabla _ { x _ { i } } \| r ( x _ { i } ) \| _ { 2 } ^ { 2 } ,\tag{1}
$$

where $x _ { i }$ is an intermediate state, $r ( x )$ is the residual of the governing equation, and $\gamma$ is a hyperparameter.

Another major approach is to introduce constraints during training to reduce deviations from physically consistent dynamics. In the context of flow matching, Tauberschmidt et al. (2026) incorporated the Adjoint Matching framework (Domingo-Enrich et al., 2025) to fine-tune the flow for lower deviations using an additional control term. For diffusion models, PIDM (Bastek et al., 2025) and related training-based approaches (Wang et al., 2025; Zeng et al., 2026; Bai et al., 2026; Ismayilzada et al., 2026) incorporate physics-based constraints such as physics residuals into the training objective. For example, when training a model to estimate the added noise $\epsilon ,$ a simplified form of the PIDM objective can be written as

$$
\| \epsilon _ { \theta } ( x _ { t } , t ) - \epsilon \| ^ { 2 } + \lambda \| r ( \mathbb { E } [ x _ { 0 } \mid x _ { t } ] ) \| ^ { 2 } ,\tag{2}
$$

where $\epsilon _ { \theta } ( x _ { t } , t )$ denotes the function that estimates the noise, $r ( x )$ denotes the residual, and λ is a hyperparameter (Bastek et al., 2025). Related extensions of PIDM have also been explored in settings such as knowledge distillation (Zhang et al., 2026) and generation without full knowledge of the underlying dynamics (Xu & Li, 2025; Tan et al., 2026).

Classifier-free guidance (Ho & Salimans, 2022), which provides a method for learning a conditional distribution $p ( x \mid c )$ , has also been applied to physics-informed generation. Although it was originally introduced for general image generation, PG Diffusion (Shu et al., 2023) uses classifier-free guidance to integrate physical information into the model architecture when reconstructing high-resolution images from low-fidelity inputs. By using the gradient of the physical residual as conditioning information within the classifier-free guidance framework, they incorporate residual information into the model and enables physics-aware training and sampling. However, as noted by Bastek et al. (2025), PG Diffusion does not explicitly guide the sampling process toward a zero-residual condition.

Although these approaches enable physics-aware generation, they require constraint evaluation during either training or sampling. In some problem settings, such as generating snapshot time-series of 3D fluid flow, the physical simulations required for constraint evaluation are computationally expensive and can substantially increase training and sampling times. Motivated by this limitation, our approach separates physical simulations required for constraint evaluation from both diffusion training and sampling, avoiding a substantial increase in computational cost even in such cases.

## 3 Background

## 3.1 Diffusion models

Diffusion models (Ho et al., 2020) are generative models that learn the reverse-time dynamics of a diffusion process, allowing data to be generated from Gaussian noise. Let $t \in [ 0 , T ]$ denote a time step and $x _ { t } \sim p _ { t } ( x )$ be an intermediate noisy sample. The forward diffusion process is defined as

$$
q ( x _ { t } \mid x _ { t - 1 } ) = { \cal N } ( x _ { t } ; \sqrt { 1 - \beta _ { t } } x _ { t - 1 } , \beta _ { t } I ) ,\tag{3}
$$

where $\beta _ { t }$ denotes the noise variance added at time t. Given Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , I )$ , a noisy sample at time t can be obtained as follows:

$$
x _ { t } ( x _ { 0 } , \epsilon ) = \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , \bar { \alpha } _ { t } = \prod _ { s = 1 } ^ { t } ( 1 - \beta _ { s } ) .\tag{4}
$$

The corresponding reverse process, parameterized by a set of learnable parameters $\theta ,$ can be written as

$$
\begin{array} { r } { p _ { \theta } ( x _ { t - 1 } \mid x _ { t } ) = \mathcal { N } ( x _ { t - 1 } ; \mu _ { \theta } ( x _ { t } , t ) , \sigma _ { t } ^ { 2 } I ) , } \end{array}\tag{5}
$$

where $\sigma _ { t } ^ { 2 }$ is a fixed variance. The diffusion model is trained by optimizing θ so that the learned reverse process approximates the data distribution.

According to Ho et al. (2020), the parameter set θ can be trained by predicting the added noise ε from the noisy sample $x _ { t } \mathrm { : }$

$$
\operatorname* { m i n } _ { \theta } \mathbb { E } _ { t , x _ { 0 } , \epsilon } \| \epsilon - \epsilon _ { \theta } ( \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , t ) \| ^ { 2 } ,\tag{6}
$$

where the function $\boldsymbol { \epsilon } _ { \boldsymbol { \theta } } ( \boldsymbol { x } , t )$ predicts the noise added to x.

After training, samples can be generated by starting from Gaussian noise at $t = T$ and iteratively applying the following reverse process until $t = 0$

$$
x _ { t - 1 } = \frac { 1 } { \sqrt { 1 - \beta _ { t } } } \left( x _ { t } - \frac { \beta _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \epsilon _ { \theta } ( x _ { t } , t ) \right) + \sigma _ { t } z ,\tag{7}
$$

where $z \sim \mathcal { N } ( 0 , I )$

## 3.2 Diffusion guidance

Dhariwal & Nichol (2021) proposed classifier guidance, which uses the gradient of a classifier to guide a diffusion model toward samples satisfying a condition c. The conditional noise-estimation model takes the condition c as an additional input. The model parameters are optimized as

$$
\operatorname* { m i n } _ { \theta } \quad \left\| \epsilon _ { \theta } ( x _ { t } , t , c ) - \epsilon \right\| ^ { 2 } .\tag{8}
$$

During sampling, the estimated noise $\epsilon _ { \theta }$ is modified using a classifier $p _ { \phi } ( c \mid x _ { t } )$ trained on noisy samples (Ho & Salimans, 2022):

$$
\tilde { \epsilon } _ { \theta } ( x _ { t } , t , c ) = \epsilon _ { \theta } ( x _ { t } , t , c ) - w \sigma _ { t } \nabla _ { x _ { t } } \log p _ { \phi } ( c | x _ { t } ) ,\tag{9}
$$

where w is a guidance scale, and $\sigma _ { t }$ is the standard deviation of the noise at time t. As discussed in Ho & Salimans (2022), Eq. 9 effectively modifies the probability distribution as

$$
\tilde { p } _ { \boldsymbol { \theta } } ( x \mid c ) \propto p _ { \boldsymbol { \theta } } ( x \mid c ) p _ { \boldsymbol { \phi } } ( c \mid x ) ^ { \boldsymbol { w } } ,\tag{10}
$$

which increases the likelihood of samples that the classifier associates with condition c and guides the sampling process accordingly.

Although classifier guidance improves sample quality, it requires training an additional classifier and complicates the overall procedure. To address this issue, Ho & Salimans (2022) proposed classifier-free guidance, which achieves a similar effect without an explicit classifier. In this approach, the same model is trained with and without condition $^ { c , }$ and sampling is guided by modifying the noise estimate $\epsilon _ { \theta }$ as

$$
\tilde { \epsilon } _ { \theta } ( x _ { t } , t , c ) = ( 1 + w ) \epsilon _ { \theta } ( x _ { t } , t , c ) - w \epsilon _ { \theta } ( x _ { t } , t ) ,\tag{11}
$$

which can be interpreted as using an implicit classifier:

$$
p ^ { \mathrm { i } } ( c \mid x ) \propto \frac { p ( x \mid c ) } { p ( x ) } .\tag{12}
$$

## 4 Method

We propose a physics-informed diffusion framework with data augmentation. The main idea of the proposed approach is to train the guidance model using original data, which serve as physically consistent positive samples, and augmented data, which serve as negative samples with some deviations from the underlying physics. The sampling process is then guided toward the positive samples.

The augmented data consist of visually plausible but physically inconsistent negative samples generated by an ordinary diffusion model before the guidance model is trained. As this initial diffusion model generates the augmented data that are subsequently used to train the guidance model, we refer to this process as self-augmentation. An overview of the proposed framework is shown in Figure 1.

Let $\mathcal { D } _ { \mathrm { o } } = \{ x _ { \mathrm { o } } \}$ denote the original dataset, where each sample satisfies the governing equation $E ( x ) = 0$ . The objective is to generate samples that are as close as possible to satisfying $E ( x ) = 0$ . We evaluate deviations from the correct dynamics either by calculating the PDE residuals $\| E ( x ) \|$ or by measuring the deviations between generated time-series samples and numerically integrated reference samples. For simplicity, we use the term residual to refer to both types of deviation throughout this study and define a residual function $r ( x )$ to quantify them.

## 4.1 Standard diffusion training for data augmentation and constraint evaluation

First, we train a diffusion model without guidance on the original dataset $\mathcal { D } _ { \mathrm { o } }$ to obtain an approximate distribution $p _ { \theta } ( x )$ . Any diffusion algorithm, such as DDPM (Ho et al., 2020), can be used for this purpose. After training, we sample from the model to construct an augmentation dataset:

$$
{ \mathcal { D } } _ { \mathrm { a } } = \{ x _ { \mathrm { a } } \sim p _ { \theta } ( x ) \} .\tag{13}
$$

Subsequently, we evaluate residuals with $r ( x )$ . The original samples $x _ { \mathrm { { o } } }$ are assumed to satisfy the governing equation, such that $r ( x _ { \mathrm { o } } ) = 0$ . In contrast, the generated samples $x _ { \mathrm { a } }$ generally exhibit nonzero residuals because the model is trained without physical constraints, resulting in $r ( x _ { \mathrm { a } } ) > 0$ Thus, we treat $x _ { \mathrm { o } }$ as positive samples satisfying the physical law, and $x _ { \mathrm { a } }$ as negative samples violating it.

We then construct an augmented dataset by combining the original and generated samples with their corresponding residuals:

$$
{ \mathcal { D } } ^ { \prime } = \{ ( x , r ) \} = \{ ( x _ { \mathrm { o } } , r ( x _ { \mathrm { o } } ) ) \} \cup \{ ( x _ { \mathrm { a } } , r ( x _ { \mathrm { a } } ) ) \} .\tag{14}
$$

## 4.2 Physics-informed classifier-free guidance

Finally, we train a classifier-free guidance model (Ho & Salimans, 2022) conditioned on the residual r using the dataset $\mathcal { D } ^ { \prime }$ , thereby obtaining an approximate distribution $p _ { \psi } ( x \mid r )$ . Based on the general conditional training objective in $\operatorname { E q . 8 } ,$ we use the physical residual r as the condition c and define the training objective as follows:

$$
\operatorname* { m i n } _ { \psi } \quad \| \epsilon _ { \psi } ( x _ { t } , t , r ) - \epsilon \| ^ { 2 } .\tag{15}
$$

![](images/2cc01acecb75c36a2abfee8ac7ef4b96d328b215145b556954660d6e1f820714.jpg)  
Figure 1: Overview of the proposed guidance approach.

As in standard classifier-free guidance, the condition r is randomly replaced with a null condition during training, allowing the same model to learn both conditional and unconditional noise estimates.

After training, we generate samples from $p _ { \psi } ( x \mid 0 )$ by setting the residual condition to $r = 0$ , which is expected to guide the model toward samples with lower physical residuals. Following Eq. 11, the noise estimate is modified as

$$
\tilde { \epsilon } _ { \psi } ( x _ { t } , t , 0 ) = ( 1 + w ) \epsilon _ { \psi } ( x _ { t } , t , 0 ) - w \epsilon _ { \psi } ( x _ { t } , t ) .\tag{16}
$$

In the proposed method, any physical simulation required for residual evaluation is performed independently of diffusion training and sampling. During the classifier-free guidance step, the model uses only the precomputed residuals. Algorithm 1 presents the overall procedure of the proposed method.

Algorithm 1 Physics-informed classifier-free guidance   
Require: $r ( x )$ : function to evaluate residuals, $\mathcal { D } _ { \mathrm { o } } \colon$ original dataset, LDm: diffusion loss, $\mathcal { L } _ { \mathrm { C F G } }$ : classifier-free   
guidance loss   
1: Diffusion training and sampling   
2: $\theta \gets \arg \operatorname* { m i n } _ { \theta } \mathcal { L } _ { \mathrm { D M } } ( \theta ; \mathcal { D } _ { \mathrm { o } } )$   
3: $\begin{array} { r } { \mathcal { D } _ { \mathrm { a } } = \{ x _ { \mathrm { a } } \} , \quad x _ { \mathrm { a } } \sim p _ { \theta } ( x ) } \end{array}$   
4: Residual calculation and data augmentation   
$\mathfrak { s } \colon \mathcal { D } ^ { \prime }  \{ ( x _ { \mathrm { o } } , 0 ) \mid x _ { \mathrm { o } } \in \mathcal { D } _ { \mathrm { o } } \} \cup \{ ( x _ { \mathrm { a } } , r ( x _ { \mathrm { a } } ) ) \mid x _ { \mathrm { a } } \in \mathcal { D } _ { \mathrm { a } } \}$   
6: Guidance model training and sampling   
7: $\psi  \arg \operatorname* { m i n } _ { \psi } \mathcal { L } _ { \mathrm { C F G } } ( \psi ; \mathcal { D } ^ { \prime } )$   
8: $x ^ { * } \sim p _ { \psi } ( { \cdot } \mid 0 )$

## 4.3 Extensibility

While the proposed algorithm can improve physical consistency on its own, it can also be readily extended to achieve further performance improvements.

First, the proposed algorithm can be applied iteratively by using samples generated by the guidance model as additional negative samples. Specifically, these newly generated samples $x _ { \mathrm { a } } ^ { \prime } \in \mathcal { D } _ { \mathrm { a } } ^ { \prime }$ are added to the existing set of negative samples, and the guidance model is retrained on the expanded dataset ${ \mathcal { D } } ^ { \prime \prime } = { \mathcal { D } } ^ { \prime } \cup \{ ( x _ { \mathrm { a } } ^ { \prime } , r ( x _ { \mathrm { a } } ^ { \prime } ) ) \}$ Samples are then generated again by conditioning on zero residual. Repeating this procedure allows the model to progressively generate samples with lower residuals.

Second, the proposed method can be readily combined with existing approaches that incorporate physical constraints during training (e.g., PIDM (Bastek et al., 2025)) or during sampling (e.g., CoCoGen (Jacobsen et al., 2025)). The proposed method differs from ordinary diffusion models in four main aspects, (i) the generation of negative samples, (ii) the inclusion of a simulation step, (iii) the introduction of the conditional variable r into the noise estimation model $\epsilon _ { \psi }$ in Eq. 15, (iv) the guidance mechanism in Eq. 16. These modifications do not fundamentally alter the structure of the diffusion model, which allows the proposed method to be applied to a wide range of existing diffusion-based approaches in plug-and-play manner.

## 5 Experiments

We demonstrate physics-guided generation using the proposed method on Darcy flow and time-series fluid data.

## 5.1 Darcy flow

## 5.1.1 Problem setting

In this section, we compare the proposed method with existing methods on the Darcy flow problem, which is commonly used as a benchmark in previous studies.

The Darcy flow equations describe fluid flow through porous media. Let $K ( x )$ denote the permeability, $p ( x )$ the pressure field, u(x) the velocity field, and $f ( x )$ a source function representing where fluid enters and exits the domain. As introduced in the related studies (Bastek et al., 2025; Jacobsen et al., 2025), the governing equations are given by:

$$
\begin{array} { r l } & { \displaystyle \boldsymbol { u } ( \boldsymbol { x } ) = - K ( \boldsymbol { x } ) \nabla p ( \boldsymbol { x } ) , \quad \boldsymbol { x } \in \Omega , } \\ & { \displaystyle \nabla \cdot \boldsymbol { u } ( \boldsymbol { x } ) = f ( \boldsymbol { x } ) , \quad \boldsymbol { x } \in \Omega , } \\ & { \displaystyle \boldsymbol { u } ( \boldsymbol { x } ) \cdot \hat { \boldsymbol { n } } ( \boldsymbol { x } ) = 0 , \quad \boldsymbol { x } \in \partial \Omega , } \\ & { \displaystyle \int _ { \Omega } p ( \boldsymbol { x } ) d \boldsymbol { x } = 0 , } \end{array}\tag{17}
$$

where $\hat { n } ( x )$ denotes the outward normal vector on the boundary.

In this experiment, the purpose of the diffusion model is to generate samples of the pressure field $p ( x )$ and the permeability field $K ( x )$ while maintaining consistency with the static relations defined in $\operatorname { E q . }$ 17. The residual is defined as PDE residual of the governing equation.

We use the Darcy flow dataset released by Bastek et al. (2025). The dataset consists of 10000 training samples and 1000 validation samples, each with a resolution of $6 4 \times 6 4$ The number of negative samples generated in the first step of our method is 10000, matching the size of the training set.

We compare the proposed method with a standard DDPM without physical constraints and evaluate the reduction in residuals achieved by the proposed method. We also evaluate the extensibility of our approach by applying a second round of guidance. We further combine the proposed method with other physics-informed diffusion approaches, including PIDM (Bastek et al., 2025), which incorporates physical constraints during training, and CoCoGen (Jacobsen et al., 2025), which incorporates them during sampling. In addition, we compare our method with PG Diffusion (Shu et al., 2023), another approach using classifier-free guidance. In each setting, we repeated the experiments 8 times using different random seeds. More detailed settings are provided in Appendix A.

## 5.1.2 Results

Figure 2 compares samples generated by the standard diffusion model and the proposed guidance model The comparison visually demonstrates that our classifier-free guidance approach reduces the residuals.

Figure 3 shows the distribution of the mean residuals over 8 trials with different random seeds. In each trial, 1000 samples were generated to calculate the mean. The results quantitatively demonstrate that the proposed method reduces the residual relative to the baseline and that repeated application of the proposed method further reduces it. A similar trend is observed in Figure 4, which shows the residual distributions within individual trials. The distributions shift toward zero after applying the proposed method, with an additional shift observed after the second application.

![](images/60e98b3c5d486865c1f5756dc60822d3659ee1628df712011a413b01c3114757.jpg)

![](images/4847e457ac489060f81840f3d5713d1e55c4cb1961f361d6b56592a04f3fa25e.jpg)

![](images/b8f434b4aad1f598ccb82690fb48f5e9b92112fa637435e7b4ca1723f35940eb.jpg)

![](images/7e1b8a8baf99f231f6144200a7975da4cbe354cfc165b2ba6a6d80ac0fc74544.jpg)

(a) Standard diffusion model.  
![](images/02164414c42dba155b9da46679cf1120c10d0eff005d5688360845ea73d09faf.jpg)  
(b) Proposed guidance model.  
Figure 2: Comparison of samples and residuals generated by the standard diffusion model and the proposed guidance model in the Darcy flow experiment. The residual represents the PDE residual of the Darcy flow equation.

![](images/79f37bd3ae4b1e4f64e1899fa4696b9f1a0e0c41c63e6b280c8dda09a4a6e401.jpg)

![](images/a9bfe65c4dc3c3db2899218633a47e27c43b5e424e17a5af50c10a92fbf1b1a4.jpg)  
Figure 3: Distribution of the mean residuals across independent runs under different conditions in the Darcy flow experiment. The red dashed lines show the mean values. The notation 1x and 2x indicate that the proposed method is applied once and twice, respectively

Figure 3 also demonstrates that the proposed method achieves lower residuals than PG Diffusion. We hypothesize that this improvement is attributable to the sampling strategy of the proposed method, which uses zero residual as an explicit target condition. Furthermore, combining the proposed method with both PIDM and CoCoGen achieves the best overall performance, outperforming the PIDM + CoCoGen baseline. This result highlights a key advantage of our approach, which is compatible with existing physics-informed diffusion methods for further improvements in physical consistency. Additional experimental results are provided in Appendix B.

![](images/512588f892db250017046d294135f0ef5ba7e7da18ecea8807e2698cf8dfc947.jpg)

![](images/5e15162fa9ecad00ea4dfa789312a2a0ea836d51a5a191ff444116c9b3bfca3b.jpg)  
Figure 4: Residual distributions of 1000 generated samples in 2 different trials in the Darcy flow experiment.

## 5.2 Time-series generation of 2D fluid dynamics

## 5.2.1 Problem setting

The second experiment considers the generation of time-series snapshots of 2D fluid dynamics.

Let u denote the velocity field, p the pressure, f the external forcing, Re the Reynolds number, and ρ the density. We consider 2D incompressible flows as introduced in (Kochkov et al., 2021), described by the following Navier-Stokes equations:

$$
\frac { \partial \boldsymbol { u } } { \partial t } = - \nabla \cdot ( \boldsymbol { u } \otimes \boldsymbol { u } ) + \frac { 1 } { R e } \nabla ^ { 2 } \boldsymbol { u } - \frac { 1 } { \rho } \nabla p + \boldsymbol { f } ,\tag{18}
$$

$$
\nabla \cdot u = 0 ,\tag{19}
$$

where ⊗ denotes the tensor product. The vorticity associated with the velocity field in Eqs. 18 and 19 is defined as

$$
\omega = \frac { \partial u _ { y } } { \partial x } - \frac { \partial u _ { x } } { \partial y } .\tag{20}
$$

In this experiment, we consider the generation of 2D vorticity fields in the absence of external forcing (i.e., $f = 0 ;$ corresponding to decaying flow). We prepared 10000 training samples and 1000 validation samples, each consisting of 4 snapshots showing the evolution over 3 seconds at a resolution of $6 4 \times 6 4$ . The number of negative samples generated in the first step of our method is 10000, matching the size of the training set.

To calculate the residuals, we use the generated snapshot at $t = 0$ as the initial condition and numerically integrate Eq. 18 and 19 to obtain a physically consistent reference trajectory. The residuals are defined as the deviations between the generated samples and the corresponding reference snapshots. For numerical stability, the integration timestep ∆t is set to 0.05, which is smaller than the snapshot interval of 1.0. As the snapshots are generated at a coarser temporal resolution than that adopted for stable numerical integration. directly calculated PDE residuals at the snapshot interval are not suitable for evaluating physical consistency.

As in the Darcy flow experiment, we compare the proposed method with a standard DDPM, PG Diffusion (Shu et al., 2023), PIDM (Bastek et al., 2025), and CoCoGen (Jacobsen et al., 2025) baselines. In each setting, we repeated the experiments 8 times using different random seeds. More detailed settings are provided in Appendix C.

## 5.2.2 Results

Figure 5 compares samples generated by the standard diffusion model and the proposed guidance model.   
The comparison visually demonstrates that our classifier-free guidance approach reduces the residuals.

Figure 6 shows the distribution of the mean residuals over 8 trials with different random seeds. In each trial, 1000 samples were generated to calculate the mean. The trends are similar to those observed in the Darcy flow experiment, and the proposed guidance models outperform the standard diffusion and PG Diffusion baselines. Although the difference is not clearly visible in the figure, the average mean residual decreases from $1 . 1 9 \times 1 0 ^ { - 2 }$ for Proposed Guidance applied once (1x) to $1 . 1 1 \times 1 0 ^ { - 2 }$ for Proposed Guidance applied twice (2x). In addition, the largest mean residual among the trials is lower for Proposed Guidance (2x), demonstrating the effectiveness of repeated applications of the proposed method. This trend is also reflected in Figure 7, which shows the residual distribution within individual trials. Furthermore, combining the proposed model with PIDM and CoCoGen achieves the best performance and substantially improves from the PIDM + CoCoGen baseline. Additional experimental results are provided in Appendix D.

![](images/6c26b7c3c4e2b2b9704abb7ecb140e11943b37f03c8fca0f2703d3c12f94912a.jpg)

![](images/1cd59703504acf4a0ceae9eefed8f7e4989922479116497b65af85ad8c02dd0b.jpg)

(a) Standard diffusion model.  
![](images/f6ec76aaa18dad8206ff8696a863bb0b320c0c51806b3752ba4182ad8ce8faf7.jpg)

![](images/fe1fb43bffa857e6a45aca7489134c30f5dc67f2ee6997c436132142505f7407.jpg)  
(b) Proposed guidance model.  
Figure 5: Comparison of samples and residuals generated by the standard diffusion model and the proposed guidance model in the time-series fluid experiment. The residual represents the deviation between generated samples and the corresponding snapshots obtained by numerically integrating the Navier-Stokes equations from the same initial states.

## 5.2.3 Comparison of computational cost

We compare the computational cost among different sampling algorithms. We measure the time to generate 1000 samples using PG Diffusion (Shu et al., 2023), CoCoGen (Jacobsen et al., 2025), and our proposed diffusion guidance approach. PG Diffusion and CoCoGen repeatedly require gradient calculations during the sampling process, while the proposed approach does not. The experimental configurations used for these measurements correspond to PG Diffusion, $\mathrm { P I D M } + \mathrm { C o C o G e n }$ , and Proposed Guidance (1x) in Section 5.2.2.

![](images/c4c9d74ceb19f47413e4b7520ca6a05f24c975408bdbe1c98409661ccca9ff30.jpg)

![](images/81e6a110c9be3f55a8cd27d31a4d75e3b20d664df68cc88b86b0991d7fb138de.jpg)  
(1x)

Figure 6: Distribution of the mean residuals across independent runs under different conditions in the timeseries fluid experiment. The red dashed lines show the mean values. The notation 1x and 2x indicate that the proposed method is applied once and twice, respectively.  
![](images/ac9c527f9031836bf8af463641a597e2606e9709dcc8a326122ebcc0935495a8.jpg)

![](images/bcc0f51d4495e34366d210cf198de665bd8beee978c376c6fcc082d4e2c13c09.jpg)  
Figure 7: Residual distributions of 1000 generated samples in 2 different trials in the time-series fluid experiment.

Table 2 reports the average sampling time over 8 trials for each method. The results show that the proposed approach substantially reduces the sampling time compared with gradient-based PG Diffusion and CoCoGen. This reduction demonstrates the computational advantage of our simulation-free sampling strategy, which avoids repeated evaluation of physical residuals and their gradients during sampling.

We expect this advantage to become more effective for computationally demanding problems such as 3D fluid dynamics. In such problems, a single numerical simulation can require hours or days, making simulation at every sampling iteration computationally prohibitive. Our approach instead uses a model trained in advance on precomputed residuals and incorporates information about the underlying physics without performing simulations during generation, in a manner analogous to amortized inference (Cranmer et al., 2020; Zammit-Mangion et al., 2025)

## 5.2.4 Data augmentation with noise

In our approach, we first train a standard diffusion model to obtain negative samples. However, negative samples can also be constructed by adding noise to the original data and then evaluating the residuals of the noisy samples, without using diffusion training or sampling. This provides a simpler variant of our approach. Here, we conduct experiments to examine whether the first-stage diffusion training can be replaced by noise-based data augmentation

Table 2: Comparison of the wall-clock time required to generate 1000 samples in the time-series fluid experiment. Results are reported as the mean ± standard deviation across trials.
<table><tr><td>Sampling method</td><td>Mean Sampling Time ± Std. [s]</td></tr><tr><td>PG Diffusion</td><td> $2 6 5 . 9 \pm 7 . 4$ </td></tr><tr><td>CoCoGen</td><td> $1 3 5 . 6 \pm 4 . 4$ </td></tr><tr><td>Classifier-free guidance-based (proposed)</td><td> ${ \bf 6 6 . 8 \pm 0 . 8 }$ </td></tr></table>

Table 3: Comparison of mean residuals between models augmented with Gaussian noise and with diffusiongenerated samples. Results across trials are reported as the mean ± standard deviation and the median with the interquartile range shown in parentheses.
<table><tr><td>Experimental condition</td><td>Mean Residual ± Std.</td><td>Median Residual (IQR)</td></tr><tr><td>Standard Diffusion (baseline)</td><td> $4 . 9 7 \times 1 0 ^ { - 2 } \pm 2 . 9 0 \times 1 0 ^ { - 2 }$ </td><td> $4 . 0 2 \times 1 0 ^ { - 2 }$   $( 3 . 9 8 \times 1 0 ^ { - 2 } )$ </td></tr><tr><td>Noise Augmentation (guidance)</td><td> $1 . 2 8 \times 1 0 ^ { - 2 } \pm 7 . 9 2 \times 1 0 ^ { - 4 }$ </td><td> $1 . 3 1 \times 1 0 ^ { - 2 } \ ( 8 . 9 5 \times 1 0 ^ { - 4 } )$ </td></tr><tr><td>Diffusion-Based Augmentation (guidance)</td><td> $\mathbf { 1 . 1 9 \times 1 0 ^ { - 2 } \pm 5 . 9 6 \times 1 0 ^ { - 3 } }$ </td><td> $\mathbf { 8 . 6 4 \times 1 0 ^ { - 3 } }$   $( \mathbf { 5 . 8 4 \times 1 0 ^ { - 3 } } )$ </td></tr><tr><td>PIDM + CoCoGen (baseline)</td><td> $6 . 9 0 \times 1 0 ^ { - 3 } \pm 2 . 2 3 \times 1 0 ^ { - 3 }$ </td><td> $6 . 4 5 \times 1 0 ^ { - 3 }$   $( 2 . 4 2 \times 1 0 ^ { - 3 } )$ </td></tr><tr><td>Noise Augmentation (guidance)</td><td> $1 . 0 7 \times { 1 0 ^ { - 3 } } \pm 9 . 8 9 \times { 1 0 ^ { - 5 } }$ </td><td> $1 . 0 8 \times 1 0 ^ { - 3 }$   $( 1 . 6 4 \times 1 0 ^ { - 4 } )$ </td></tr><tr><td>Diffusion-Based Augmentation (guidance)</td><td> $\mathbf { 8 . 4 3 \times 1 0 ^ { - 4 } } \pm \mathbf { 4 . 6 4 } \times \mathbf { 1 0 ^ { - 4 } }$ </td><td> $\mathbf { 7 . 1 1 \times 1 0 ^ { - 4 } }$   $( \mathbf { 7 . 1 5 \times 1 0 ^ { - 4 } } )$ </td></tr></table>

We use the same time-series 2D fluid dataset and consider two experimental settings, with the standard diffusion model and PIDM + CoCoGen serving as the respective baselines. For each setting, we generate negative samples by adding Gaussian noise to the original data. We adjust the noise variance so that the mean residual of the resulting negative samples matches that of the corresponding baseline. For the standard diffusion setting, the mean residuals are $4 . 9 7 \times 1 0 ^ { - 2 }$ for the baseline and $4 . 9 3 \times 1 0 ^ { - 2 }$ for the noiseaugmented negative samples. For the PIDM + CoCoGen setting, the corresponding values are $6 . 9 0 \times 1 0 ^ { - 3 }$ and $6 . 8 3 \times 1 0 ^ { - 3 }$ , respectively.

We train the classifier-free guidance model using the noise-augmented dataset. The model parameters and training hyperparameters are kept the same as those used in the diffusion-based augmentation experiments, and we conduct 8 trials with different random seeds. Table 3 compares the mean residuals calculated over 1000 generated samples. Although noise augmentation also reduces the residuals, the diffusion-based augmentation adopted in our method results in lower mean residuals. These results indicate that augmentation with random noise provides a potential alternative, particularly when training the first-stage diffusion model is prohibitively expensive. Nevertheless, when computationally feasible, first-stage diffusion training is preferable because employing visually plausible negative samples tends to provide better physical consistency

## 6 Conclusion

In this study, we proposed a method to enhance the physical consistency of diffusion models using classifierfree guidance with self-generated data augmentation. The proposed model learns a distribution conditioned on residuals and guides the generation process toward samples with lower deviations from physical laws by setting the residual condition to zero. In addition, our approach decouples physics simulation from diffusion training and sampling, avoiding repeated and time-consuming constraint evaluation as well as gradient calculations of the constraints.

Experimental results demonstrate that the proposed method reduces residuals compared with standard diffusion models such as DDPM, and further improves physical consistency when combined with existing physics-constrained training and sampling approaches. Moreover, our empirical results show that multiple application of the proposed method yields additional improvements in physical consistency without requiring evaluation of the governing equations during diffusion training or sampling. This property broadens the applicability of physics-aware diffusion models to computationally demanding problems.

The proposed framework is applicable to a wide range of constrained generation tasks, provided that consistency with a given set of rules can be evaluated. Such rules are not limited to the physical laws considered in this study. For example, the method could be applied to generating control inputs for robotic systems under motion constraints or generating robot trajectories that comply with operational rules. Exploring such extensions remains an important direction for future work.

## Acknowledgments

This work was supported by JST PRESTO JPMJPR24T6, JSPS KAKENHI JP25H01454, JP26K02968, and JST SPRING JPMJSP2108.

## References

Yang Bai, George Eskandar, Ziyuan Liu, and Gitta Kutyniok. Physics-informed video diffusion for shallow water equations. In Proceedings of the 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing, pp. 13242–13246, 2026.

Jan-Hendrik Bastek, WaiChing Sun, and Dennis M. Kochmann. Physics-informed diffusion models. In Proceedings of the 13th International Conference on Learning Representations, 2025

Matthieu Blanke, Yongquan Qu, Sara Shamekh, and Pierre Gentine. Strictly constrained generative modeling via split augmented langevin sampling. In Proceedings of the 14th International Conference on Learning Representations, 2026.

Shuhao Cao, Francesco Brarda, Ruipeng Li, and Yuanzhe Xi. Spectral-Refiner: Accurate fine-tuning of spatiotemporal fourier neural operator for turbulent flows. In Proceedings of the 13th International Conference on Learning Representations, 2025.

Chaoran Cheng, Boran Han, Danielle C. Maddix, Abdul Fatir Ansari, Andrew Stuart, Michael W. Mahoney, and Bernie Wang. Gradient-free generation for hard-constrained systems. In Proceedings of the 13th International Conference on Learning Representations, 2025.

Jacob K. Christopher, Stephen Baek, and Ferdinando Fioretto. Constrained synthesis with projected diffusion models. In Advances in Neural Information Processing Systems, volume 37, pp. 89307–89333, 2024.

Jacob K. Christopher, James E. Warner, and Ferdinando Fioretto. Constraint-aware flow matching: Decision aligned end-to-end training for constrained sampling. arXiv preprint arXiv:2605.12754, 2026.

Hyungjin Chung, Jeongsol Kim, Michael T. McCann, Marc L. Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In Proceedings of the 11th International Conference on Learning Representations, 2023.

Kyle Cranmer, Johann Brehmer, and Gilles Louppe. The frontier of simulation-based inference. Proceedings of the National Academy of Sciences, 117(48):30055–30062, 2020.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat GANs on image synthesis. In Advances in Neural Information Processing Systems, volume 34, pp. 8780–8794, 2021.

Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, and Ricky T. Q. Chen. Adjoint matching: Finetuning flow and diffusion generative models with memoryless stochastic optimal control. In Proceedings of the 13th International Conference on Learning Representations, 2025.

Gideon Dresdner, Dmitrii Kochkov, Peter Christian Norgaard, Leonardo Zepeda-Núñez, Jamie A. Smith, Michael P. Brenner, and Stephan Hoyer. Learning to correct spectral methods for simulating turbulent flows. Transactions on Machine Learning Research, 2023.

Nic Fishman, Leo Klarner, Valentin De Bortoli, Emile Mathieu, and Michael John Hutchinson. Diffusion models for constrained domains. Transactions on Machine Learning Research, 2023.

Davide Gallon, Philippe von Wurstemberger, Patrick Cheridito, and Arnulf Jentzen. Physics-informed diffusion models in spectral space. In Proceedings of the 43rd International Conference on Machine Learning, 2026.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pp. 6840–6851, 2020.

Jiahe Huang, Guandao Yang, Zichen Wang, and Jeong Joon Park. DiffusionPDE: Generative PDE-solving under partial observation. In Advances in Neural Information Processing Systems, volume 37, pp. 130291– 130323, 2024a.

William Huang, Yifeng Jiang, Tom Van Wouwe, and C. Karen Liu. Constrained diffusion with trust sampling In Advances in Neural Information Processing Systems, volume 37, pp. 93849–93873, 2024b.

Elkhan Ismayilzada, Yufei Zhang, and Zijun Cui. PAD-Hand: Physics-aware diffusion for hand motion recovery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 28358–28368, 2026.

Christian Jacobsen, Yilin Zhuang, and Karthik Duraisamy. CoCoGen: Physically consistent and conditioned score-based generative models for forward and inverse problems. SIAM Journal on Scientific Computing, 47(2):C399–C425, 2025.

Shervin Khalafi, Dongsheng Ding, and Alejandro Ribeiro. Constrained diffusion models via dual training. In Advances in Neural Information Processing Systems, volume 37, pp. 26543–26576, 2024.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of the 3rd International Conference on Learning Representations, 2015.

Dmitrii Kochkov, Jamie A. Smith, Ayya Alieva, Qing Wang, Michael P. Brenner, and Stephan Hoyer. Machine learning-accelerated computational fluid dynamics. Proceedings of the National Academy of Sciences, 118(21):e2101784118, 2021.

Tianyi Li, Michele Buzzicotti, Fabio Bonaccorso, and Luca Biferale. Physics-constrained diffusion model for synthesis of 3d turbulent data. arXiv preprint arXiv:2603.12834, 2026a.

Zeyang Li, Kaveh Alim, and Navid Azizan. HardFlow: Hard-constrained sampling for flow-matching models via trajectory optimization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026b.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In Proceedings of the 11th International Conference on Learning Representations, 2023.

Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In Proceedings of the 38th International Conference on Machine Learning, pp. 8162–8171, 2021.

Danyang Peng, Yang Chen, Yunlong Zhou, and Xiao-Tong Yuan. LoPhyDA: Low-rank tensor and physics gradient guided diffusion for atmospheric data assimilation. In Proceedings of the 43rd International Conference on Machine Learning, 2026.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional networks for biomedical image segmentation. In Proceedings of the 18th International Conference on Medical Image Computing and Computer-Assisted Intervention, volume 9351 of Lecture Notes in Computer Science, pp. 234–241, 2015.

Dule Shu, Zijie Li, and Amir Barati Farimani. A physics-informed diffusion model for high-fidelity flow field reconstruction. Journal of Computational Physics, 478:111972, 2023.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Proceedings of the 32nd International Conference on Machine Learning, pp. 2256–2265, 2015.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In Proceedings of the 9th International Conference on Learning Representations, 2021a.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In Advances in Neural Information Processing Systems, volume 32, pp. 11918–11930, 2019.

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In Proceedings of the 9th International Conference on Learning Representations, 2021b.

Kaiyuan Tan, Kendra Givens, Peilun Li, and Thomas Beckers. PHDME: Physics-informed diffusion models without explicit governing equations. arXiv preprint arXiv:2601.21234, 2026.

Jan Tauberschmidt, Sophie Fellenz, Sebastian Josef Vollmer, and Andrew B. Duncan. Physics-constrained fine-tuning of flow-matching models for generation and inverse problems. In Proceedings of the 14th International Conference on Learning Representations, 2026.

Utkarsh Utkarsh, Pengfei Cai, Alan Edelman, Rafael Gomez-Bombarelli, and Christopher Vincent Rackauckas. Physics-constrained flow matching: Sampling generative models with hard constraints. In Advances in Neural Information Processing Systems, volume 38, pp. 160217–160252, 2025.

Hao Wang, Jindong Han, Wei Fan, Weijia Zhang, and Hao Liu. PhyDA: Physics-guided diffusion models for data assimilation in atmospheric systems. arXiv preprint arXiv:2505.12882, 2025.

Qianxun Xu and Zuchuan Li. Partial physics informed diffusion model for ocean chlorophyll concentration reconstruction. In Advances in Neural Information Processing Systems, volume 38, pp. 156490–156507, 2025.

Jiachen Yao, Abbas Mammadov, Julius Berner, Gavin Kerrigan, Jong Chul Ye, Kamyar Azizzadenesheli, and Animashree Anandkumar. Guided diffusion sampling on function spaces with applications to PDEs. In Advances in Neural Information Processing Systems, volume 38, pp. 127057–127094, 2025.

Ye Yuan, Jiaming Song, Umar Iqbal, Arash Vahdat, and Jan Kautz. PhysDiff: Physics-guided human motion diffusion model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15964–15975, 2023.

Andrew Zammit-Mangion, Matthew Sainsbury-Dale, and Raphaël Huser. Neural methods for amortized inference. Annual Review of Statistics and Its Application, 12:311–335, 2025.

Tianyi Zeng, Tianyi Wang, Jiaru Zhang, Zimo Zeng, Feiyang Zhang, Yiming Xu, Sikai Chen, Junfeng Jiao, Christian Claudel, and Xinbo Chen. PILD: Physics-informed learning via diffusion. arXiv preprint arXiv:2601.21284, 2026.

Yi Zhang, Peng Wang, and Difan Zou. Physics-informed distillation of diffusion models for PDE-constrained generation. In Forty-third International Conference on Machine Learning, 2026.

## A Details of the Darcy flow experiment

## A.1 Dataset

The Darcy flow dataset released by Bastek et al. (2025) consists of 10000 training samples and 1000 validation samples. Each sample contains the pressure field p and permeability field K, both at a resolution of $6 4 \times 6 4$ We concatenate p and K along the channel dimension so that each sample is represented as $\boldsymbol { x } \in \mathbb { R } ^ { 2 \times 6 4 \times 6 4 }$ Figure 8 shows examples from the dataset.

![](images/413546cec09127b77e1b9e92bfa3f35fdcbc655e96e912baa1eaafa66ffe8bc0.jpg)  
Figure 8: Examples from the Darcy flow dataset.

## A.2 Implementation details

The proposed method was implemented based on the code provided for Denoising Diffusion Implicit Models (DDIM) (Song et al., 2021a). All experiments were conducted on an NVIDIA GH200 GPU for both training and sampling.

The pressure and permeability channels were independently normalized to [-1, 1] using the minimum and maximum values computed from the corresponding training set. Before residual evaluation, generated samples were transformed back to their original physical scales.

Following Song et al. (2021a), we use a U-Net architecture (Ronneberger et al., 2015) to estimate the added noise. The model parameters are optimized using Adam (Kingma & Ba, 2015). The residual r is input to the model as a condition vector $c = [ r , 1 ] \in \mathbb { R } ^ { 2 }$ , where the second component distinguishes a valid condition from the null condition used for classifier-free guidance. The condition vector is embedded with an encoder neural network, and added to residual blocks of the U-Net. During classifier-free guidance training, the condition is replaced with the null condition $c _ { \delta } = [ - 1 , 0 ]$ with probability 0.1. During sampling, the zero-residual condition is represented by $c = [ 0 , 1 ]$

Table 4 summarizes the model and experimental settings. The same model configuration is used for both the first-stage diffusion model and the subsequent guidance model.

## B Additional results of the Darcy flow experiment

## B.1 Residual distributions of individual trials

Figure 4 in Section 5.1 shows the residual distributions for 2 of the 8 trials, while Figure 9 presents those for the remaining 6 trials. Each histogram is computed from 1000 samples generated in a single independent trial with a different random seed. In all trials, the distributions shift toward lower residual values when the proposed guidance is applied. The distributions shift further toward zero when the proposed method is applied twice by adding the samples generated by the first guidance model to the existing training data and retraining the guidance model. These results are consistent with the findings reported in Section 5.1.

Table 4: Experimental settings for the Darcy flow experiments.
<table><tr><td>Parameter</td><td>Setting</td></tr><tr><td>Model</td><td></td></tr><tr><td>Architecture</td><td>U-Net</td></tr><tr><td>Base channels</td><td>64</td></tr><tr><td>Channel multipliers</td><td>(1, 2, 4, 8)</td></tr><tr><td>Residual blocks</td><td>2 per resolution</td></tr><tr><td>Attention</td><td>16 × 16 resolution</td></tr><tr><td>Dropout</td><td>0.1</td></tr><tr><td>EMA decay</td><td>0.9999</td></tr><tr><td>Diffusion</td><td></td></tr><tr><td>Number of diffusion steps</td><td>1000</td></tr><tr><td>Variance schedule</td><td>Linear  $( 1 \times 1 0 ^ { - 4 }  2 \times 1 0 ^ { - 2 } )$ </td></tr><tr><td>Training</td><td></td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td> $1 . 0 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Epochs</td><td>500</td></tr><tr><td>Weight decay</td><td>0.0</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Classifier-free dropout</td><td>0.1</td></tr><tr><td>Sampling</td><td></td></tr><tr><td>Sampling steps</td><td>100</td></tr><tr><td>DDIM stochasticity parameter η</td><td>1.0</td></tr><tr><td>Guidance scale w</td><td>2.0</td></tr></table>

## B.2 Generated samples

Figure 10 shows samples generated under different experimental conditions. The residual represents the PDE residual of the Darcy flow equation.

It illustrates how the proposed method and other physics-informed diffusion methods reduce the residuals. It also demonstrates the proposed method combined with PIDM and CoCoGen achieves the largest reduction.

## B.3 Novelty check

We compare the generated samples with the training data to assess whether the reduction in residuals could be attributed to memorization of training samples that satisfy the governing equations. This analysis examines whether the model can generate novel, physically consistent samples rather than simply reproducing the training data.

In Figure 11, we randomly select four samples generated by PIDM + CoCoGen with the proposed guidance, which achieved the lowest residuals among the settings evaluated in Section 5.1. For each generated sample, we show the three closest training samples in terms of MSE. The generated samples differ from their nearest training samples, suggesting that the model does not simply reproduce the training data and can generate novel physics-aware samples.

## C Details of the time-series 2D fluid experiment

## C.1 Dataset

We construct a dataset of time-series vorticity snapshots ω using torch-cfd (Cao et al., 2025), a PyTorchbased CFD library built on JAX-CFD (Kochkov et al., 2021; Dresdner et al., 2023). The viscosity is set to

![](images/db490f21904efd17e9e82411ff77d6cc38e45f8b57a8e71337f51eae64f5204f.jpg)

![](images/81223726ce9306bc86da352d14bc771013a4e5cb245f22f83046aee4349f9d45.jpg)

![](images/c19296a3133e208da9f3efb9edcefb564c1347e692ad79f03b43ce4716544194.jpg)

![](images/e5384a2a152aa24b99229ec6245dd1e3ac4d4c9def9a48de0b7ead6e8b7b3899.jpg)

![](images/5589c44ef829cabab9f35e0def83a9982247bbf53787ed059553ec6e0c26fdd2.jpg)

![](images/f8c815ba1479ab0be324759153980ee30df025712ecbf7d23dedbe9e2f9ad7e5.jpg)  
Figure 9: Residual distributions over 1000 generated samples for 6 independent trials in the Darcy flow experiment. Each histogram corresponds to a different random seed.

$1 . 5 \times 1 0 ^ { - 2 }$ , the maximum initial velocity to 0.4, and the computational domain to $[ 0 , 5 ] \times [ 0 , 5 ]$ . No external forcing is applied. The initial velocity fields are generated by applying a spectral filter to Gaussian noise. The governing equations are integrated with an internal time step of 0.05. Each sample is represented as $\boldsymbol { x } \in \mathbb { R } ^ { 4 \times 6 4 \times 6 4 }$ , consisting of 4 snapshots at a resolution of $6 4 \times 6 4$ . The snapshots span 3 seconds and are recorded at 1 second intervals. We generate 10000 training samples and 1000 validation samples. Examples are shown in Figure 12.

## C.2 Implementation details

The proposed method was implemented based on the code provided for DDIM (Song et al., 2021a). All experiments were conducted on an NVIDIA GH200 GPU for both training and sampling.

The four vorticity channels were jointly normalized to $[ - 1 , 1 ]$ using the global minimum and maximum values computed from the corresponding training set. Before residual evaluation, generated samples were transformed back to their original physical scales.

Following Song et al. (2021a), we use a U-Net architecture (Ronneberger et al., 2015) to estimate the added noise. The model parameters are optimized using Adam (Kingma & Ba, 2015). We use the cosine noise

![](images/2137f5330f998662e5ebcd241042c00e75994b8d1eb4cdfea9b4e2cc14844785.jpg)

![](images/7238f06004eb15d6c9248cecbf6ce86f2046ff71518e6b63fac3b3b39508dcc5.jpg)

![](images/fb4bb14fe21fd7b7836134a29d86c86d80b5461e2ca3360d427ccd941bf51efc.jpg)

![](images/8ff21c142301bace5c395fc4dec5f8f0c66552104d1636819315bc429be7ad05.jpg)

![](images/34064aad78eaddeddd08c65479b59974467a135337dfedf5a7896a4a054db217.jpg)  
(a) Standard diffusion model.

![](images/412ed36ed0259900547a7657fc305f73e1fe4eef147053eb52e5d08fab80d132.jpg)

![](images/0905cc7c77e3dd40c2f7a536ef66c607ca4da0f359bd029208844fe3bedc40af.jpg)

![](images/6f4d699ffe6f4e8719011179f9e6e293b117246fff594d4faba9ac4a9868952c.jpg)

![](images/e2f9ba944081c3a3cb10eb34bc4d0528d1d0b4d89b306495becdb0d5c1a216d4.jpg)

(b) PG Diffusion.  
![](images/5b6d5c7cbbf099ca90464e465cda45bd4ccd2499db30805c2967097545f44142.jpg)

![](images/93c3625344e1b363965a5944304e78f4ed561202781fddd51cd4525cbb8231ec.jpg)

(c) Proposed guidance model applied once.  
![](images/239d3c247a5bd7463e9e9b071e27f88c89a4f6028d8c02cd01e50a479d96777f.jpg)

![](images/5847d86faed6b5a94068baceb5bad34ac01a072b3bee94ad80f3de10cf8d96ef.jpg)

(d) Proposed guidance model applied twice.  
![](images/bc66b733ca43d53fa297294125c6fd22f007f49a78fcadf59506abea2558fd5b.jpg)  
(e) PIDM + CoCoGen.  
(f) PIDM + CoCoGen with the proposed guidance applied once.  
Figure 10: Comparison of samples and residuals generated under different conditions in the Darcy flow experiment.

schedule proposed by Nichol & Dhariwal (2021). The fluid residual r is rescaled to $r _ { \mathrm { r e s c a l e d } } \in [ 0 , 1 ]$ . The transformed residual is input to the model as a condition vector $\mathbf { c } = [ r _ { \mathrm { r e s c a l e d } } , 1 ] \in \mathbb { R } ^ { 2 }$ , where the second component distinguishes a valid condition from the null condition used for classifier-free guidance. The

![](images/4d34efccc4c4d6c5f0a7ec40aa2606ce612d70822ed00bef619924bdd5f14f47.jpg)  
Figure 11: Samples generated using PIDM + CoCoGen with the proposed guidance and their three nearest neighbors in the training dataset for the Darcy flow experiment.

condition vector is embedded with an encoder neural network, and added to residual blocks of the $\mathrm { U - N e t }$ During classifier-free guidance training, the condition is replaced with the null condition $\mathbf { c } _ { \mathcal { O } } = [ - 1 , 0 ]$ with probability 0.2. During sampling, the zero-residual target is represented by $\mathbf { c } = [ 0 , 1 ]$

![](images/be7763312f8a66f71e9235b7841c4ea627921086e33a3a84c4fa2451bb266668.jpg)  
Figure 12: Examples of time-series 2D fluid data.

Table 5 summarizes the model and experimental settings. The same model configuration is used for both the first-stage diffusion model and the subsequent guidance model.

## D Additional results of the time-series 2D fluid experiment

## D.1 Residual distributions of individual trials

Figure 7 in Section 5.2 shows the residual distributions for 2 of the 8 trials, while Figure 13 presents those for the remaining 6 trials. Each histogram is computed from 1000 samples generated in a single independent trial with a different random seed. In all trials, the distributions shift toward lower residual values when the proposed guidance is applied. The results also show a trend toward further reductions in the residuals when the proposed method is applied twice. These results are consistent with the findings reported in Section 5.2.

Table 5: Experimental settings for the time-series 2D fluid experiments.
<table><tr><td>Parameter</td><td>Setting</td></tr><tr><td>Model</td><td></td></tr><tr><td>Architecture</td><td>U-Net</td></tr><tr><td>Base channels</td><td>64</td></tr><tr><td>Channel multipliers</td><td>(1, 2, 2, 4)</td></tr><tr><td>Residual blocks</td><td>2 per resolution</td></tr><tr><td>Attention</td><td>16 × 16 resolution</td></tr><tr><td>Dropout</td><td>0.2</td></tr><tr><td>EMA decay</td><td>0.9999</td></tr><tr><td>Diffusion</td><td></td></tr><tr><td>Number of diffusion steps</td><td>1000</td></tr><tr><td>Variance schedule</td><td>Cosine (offset: 0.008)</td></tr><tr><td>Training</td><td></td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td> $1 . 0 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Batch size</td><td>32</td></tr><tr><td>Epochs</td><td>200</td></tr><tr><td>Weight decay</td><td> $1 . 0 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Classifier-free dropout</td><td>0.2</td></tr><tr><td>Sampling</td><td></td></tr><tr><td>Sampling steps</td><td>100</td></tr><tr><td>DDIM stochasticity parameter η</td><td>1.0</td></tr><tr><td>Guidance scale w</td><td>2.0</td></tr></table>

## D.2 Generated samples

Figures 14 and 15 show samples generated under different conditions. The residual represents the deviation between generated samples and the corresponding snapshots obtained by numerically integrating the Navier Stokes equations from the same initial states.

They illustrate that the proposed method reduces the residuals compared with the baselines and that its combination with PIDM and CoCoGen achieves the lowest residuals among the adopted settings.

## D.3 Novelty check

As in the Darcy flow experiment, we compare the generated samples with the training dataset to assess whether the reduction in residuals could be attributed to memorization of training samples that satisfy the governing equations. This analysis examines whether the model can generate novel, physically consistent samples rather than simply reproducing the training data.

In Figure 16, we randomly select four samples generated by PIDM + CoCoGen with the proposed guidance, which achieved the lowest residuals among the settings evaluated in Section 5.2. Following the procedure used for the Darcy flow experiment in Figure 11, we show the three closest training samples for each generated sample in terms of MSE. The generated samples differ from their nearest training samples, suggesting that the model does not simply reproduce the training data and can generate novel physics-aware samples.

![](images/7e62a289160cb15681de60b72e5e1cafc9f83f5a5ffc1e896d17543e87440213.jpg)

![](images/c406374df455ea351c5003f4604453575222c7dfb57120e3e384b8d33b0a49d7.jpg)

![](images/468048dacb76449940585d6d9d4f2febac2d6b176505ffa40b30236af0f64909.jpg)

![](images/d6e6371bac4c22060e37c3cfda4e5cfb8f97a0f1d5423109aca029f298a96eba.jpg)

![](images/aef31106e2e24c95d9a34c2276cbe1622d1bf2744aba109a8cf573cfb5eb567b.jpg)

![](images/3af61fc38ebb4a7511174a3b8a37edadde29d78f6d520fa274b6e311e4ed3466.jpg)  
Figure 13: Residual distributions over 1000 generated samples for 6 independent trials in the time-series 2D fluid experiment. Each histogram corresponds to a different random seed.

![](images/f2f319e206c7df47c3790081f4d6b3fd707effb1ef637764fffc382b9fc311cf.jpg)

![](images/7c872a21a011f4e9b192c073a1accbde3e7d46d740ff07d50f9dfb8fd04c5628.jpg)

(a) Standard diffusion model.  
![](images/186baad1af5c0e17295c6db84e0ce2a4a08ff19d9c76ee33a8ab6704a55a2d18.jpg)

![](images/e2ae711e88806b9aee1ad687629f41975616e3c7394ca1568e0408e8d3cfc834.jpg)

(b) Proposed guidance model applied once.  
![](images/2d10e9473f700027c88bcec5d65d92e3fe1eb30a6baf294c8f8ecb18bf07fd03.jpg)

![](images/2c0a2b3b6f9b4f1399f6288cac95dc1852ff9b500b0412f7f3d5dad71ab6dae3.jpg)  
(c) Proposed guidance model applied twice.  
Figure 14: Comparison of samples and residuals generated by the standard diffusion baseline and the proposed guidance models in the time-series 2D fluid experiment.

![](images/ecd43507e9f3e504c326242f02c82aa86292f65658f2a8e168e0a37b326570df.jpg)

![](images/e565e06bb7d6d3a1b175665bb94233da706d3643490d5ce81c935310bb81fc77.jpg)  
(a) PG Diffusion.

![](images/ced53fd70aaa98a87fccf9cb69f840f796b1801be540465b01402f62a63c8ea3.jpg)

![](images/598e9e9db287e1bd01278865b89077905979e60622bef395889416807ea061b8.jpg)  
(b) PIDM + CoCoGen.

![](images/f610e38a59e428e0c725f11159cb35cf6c927ab3c75097057d535bdf62ac636e.jpg)

![](images/16237e75de43ff9bc5c125f71319c986ebf291df10e9482d739cd9d41b96b140.jpg)  
(c) PIDM + CoCoGen with the proposed guidance applied once.  
Figure 15: Comparison of samples and residuals generated by the physics-informed diffusion baselines and their combination with the proposed guidance in the time-series 2D fluid experiment.

![](images/30d3d6c13a612a1eb7f23bde0008cb3845205f6132cddb41617e33614582cc31.jpg)

![](images/10d8943cdf6f6a4b78952dc23e611a31495448c6ddb4c2c63485e54489956227.jpg)  
Figure 16: Samples generated using PIDM + CoCoGen with the proposed guidance and their three nearest neighbors in the training dataset for the time-series 2D fluid experiment