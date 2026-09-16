# ONLINE GRADIENT COMPUTATION FOR WARPING GAUSSIAN PROCESS TRANSFORMATIONS

Emilio Ruiz-Moreno<sup>1,2</sup>, Konstantinos Slavakis<sup>3</sup>, and Baltasar Beferull-Lozano<sup>1,2</sup>

## ABSTRACT

Warped Gaussian processes (GPs) handle non-Gaussian observations by mapping them into a latent standard GP via a parametric transformation called warping. Existing streaming variants, however, either optimize the warping parameters periodically or sacrifice analytical tractability for a higher model capacity. To bridge this gap, we show that the gradient of the instantaneous negative log-likelihood of a warped GP admits an exact recursive computation. Based on this result, we propose a novel online method for warped GPs that jointly updates the latent GP moments and optimizes the warping parameters.

Index Terms— Gaussian process, warping transformation, online learning

## 1 Introduction

Gaussian processes (GPs) provide a non-parametric Bayesian framework for probabilistic function approximation [1], proving especially useful when the true underlying function is analytically unknown or expensive to query [2, 3]. By placing a prior directly over the space of functions of interest, GPs offer a principled approach to uncertainty quantification, making them particularly attractive for applications where predictive confidence is as critical as point accuracy; for instance, in robotics [4] or geostatistics [5, 6]. Crucially, the mathematical tractability of GPs allows for exact Bayesian inference, yielding closed-form predictive distributions when paired with Gaussian function observation likelihoods.

Despite their widespread success, standard GPs are fundamentally limited by their core assumption that the function observations are adequately modeled by a joint Gaussian distribution. Indeed, this assumption often breaks down in practice, as real-world observations may exhibit heavy tails, distinct skewness, or physical constraints (such as strict positivity). Consequently, applying standard GPs to such non-Gaussian observations can produce highly inaccurate predictions and poorly calibrated uncertainty bounds.

Motivated by these limitations, warped GPs [7] extend standard GPs by applying a parametric transformation—termed warping—to non-Gaussian observations, mapping them into latent targets where standard GP assumptions hold. By jointly estimating latent GP moments and warping parameters, warped GPs accommodate complex observation likelihoods while retaining the analytical tractability of standard GPs.

However, the implementation of warped GPs in memoryconstrained or real-time tasks remains a significant challenge. This is because, to the best of our knowledge, there is no online method to jointly update the latent GP moments and optimize the warping parameters of warped GPs. The methods arguably closest to this work in the literature either optimize the warping parameters periodically [8] or trade analytical tractability for higher model capacity [9].

In this paper, we show that the gradient of the negative log-likelihood (NLL) of a warped GP can be computed recursively in a exact manner. Based on this, we propose a novel online method for warped GPs that jointly updates the latent GP moments and optimizes the warping parameters.

## 2 Background

Consider the problem of inferring an unknown, real-valued function f over an arbitrary d-dimensional input space $\mathcal { X } \subseteq$ $\mathbb { R } ^ { d }$ , based on prior beliefs about its function space and a set of n (possibly corrupted) function observations $y _ { 1 } , \ldots , y _ { n }$ at corresponding input locations $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { n } .$

Such a problem can be addressed analytically, provided certain assumptions hold, as we detail next.

## 2.1 Gaussian processes

A GP model typically assumes the function f is a realization of a zero-mean GP prior and the function observations are corrupted by white Gaussian noise. That is,

$$
f ( \pmb { x } ) \sim \mathcal { G P } \left( 0 , \kappa ( \pmb { x } , \pmb { x } ^ { \prime } ) \right) ,\tag{1a}
$$

$$
y _ { i } = f ( \pmb { x } _ { i } ) + \epsilon _ { i } ,\tag{1b}
$$

where the kernel $\kappa : \mathcal { X } \times \mathcal { X } $ R is the covariance function of the GP prior, and each $\epsilon _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ is an observation noise term with standard deviation σ.

Thanks to Gaussianity and the closure properties of GPs [1], the posterior distribution over functions conditioned

![](images/58ddcc67d9e40ea4ad063fed689042107f36e28fbe3f7c4b861d73fab2374274.jpg)  
Fig. 1. GP prior (top) and GP posterior (bottom), i.e., the GP prior conditioned on observations (indicated by + markers). The solid lines represent sample functions drawn from the GP. The shaded area covers ±1.96 standard deviations from the mean. Example adapted from [1, Ch. 2].

on a set of n observations collected in the observation vector $\pmb { y } _ { n } = [ y _ { 1 } , \dots , y _ { n } ] ^ { \top } \in \mathbb { R } ^ { n }$ is also Gaussian. Specifically,

$$
f ( \pmb { x } ) | \pmb { y } _ { n } \sim \mathcal { N } \left( m _ { n } ( \pmb { x } ) , v _ { n } ( \pmb { x } ) \right) ,\tag{2}
$$

with closed-form posterior mean and variance

$$
m _ { n } ( { \pmb x } ) = { \pmb k } _ { n } ( { \pmb x } ) ^ { \top } { \pmb \alpha } _ { n } ,\tag{3a}
$$

$$
v _ { n } ( { \pmb x } ) = \kappa ( { \pmb x } , { \pmb x } ) - { \pmb k } _ { n } ( { \pmb x } ) ^ { \top } \Omega _ { n } { \pmb k } _ { n } ( { \pmb x } ) ,\tag{3b}
$$

respectively, where

$$
\pmb { \alpha } _ { n } = \pmb { \Omega } _ { n } \pmb { y } _ { n } \in \mathbb { R } ^ { n } ,\tag{4a}
$$

$$
\Omega _ { n } = \left( K _ { n } + \sigma ^ { 2 } I _ { n } \right) ^ { - 1 } \in \mathbb { S } _ { \ge 0 } ^ { n } , ,\tag{4b}
$$

with $\pmb { k } _ { n } ( \pmb { x } ) \in \mathbb { R } ^ { n }$ and $K _ { n } \in \mathbb { S } _ { > 0 } ^ { n }$ constructed as $[ { \pmb k } _ { n } ( { \pmb x } ) ] _ { i } =$ $\kappa ( \pmb { x } _ { i } , \pmb { x } )$ and $[ { \cal K } _ { n } ] _ { i , j } = \kappa ( { \pmb x } _ { i } , \bar { { \pmb x } _ { j } } )$ , for all $i , j \in \{ 1 , \ldots , n \}$

To aid conceptual understanding, an illustration of the prior (1a) and posterior (2) of a GP is provided in Fig. 1.

## 2.1.1 Recursive Gaussian process updates

Direct computation of the terms $\alpha _ { n }$ and $\Omega _ { n }$ for the posterior moments in (3) scales poorly with the number of observations $n ,$ mostly due to the matrix inversion on the right-hand side of (4b). Fortunately, one can still compute these terms exactly, but at a reduced computational load, by exploiting the

following recursive structure

$$
\begin{array} { r } { \pmb { \alpha } _ { n } = \left[ \pmb { \alpha } _ { n - 1 } \right] - \frac { y _ { n } - m _ { n - 1 } ( \pmb { x } _ { n } ) } { s _ { n } } \left[ \pmb { \omega } _ { n } \right] , } \end{array}\tag{5a}
$$

$$
\pmb { \Omega } _ { n } = \left[ \pmb { \Omega } _ { n - 1 } \quad \mathbf { 0 } _ { n - 1 } \right] + \frac { 1 } { s _ { n } } \left[ \pmb { \omega } _ { n } \right] \left[ \pmb { \omega } _ { n } ^ { \top } \quad - 1 \right] ,\tag{5b}
$$

where $s _ { n } = v _ { n - 1 } ( \pmb { x } _ { n } ) + \sigma ^ { 2 } \in \mathbb { R }$ , and $\omega _ { n } = \Omega _ { n - 1 } k _ { n - 1 } ( { \pmb x } _ { n } ) \in$ $\mathbb { R } ^ { n - 1 }$ . Thus, the GP posterior (2) can be updated recursively as each new observation $y _ { n }$ and associated input location ${ \pmb x } _ { n }$ become available, simply by maintaining $\alpha _ { n }$ and $\Omega _ { n }$ as state variables.

## 2.1.2 Sparse Gaussian process updates

Indeed, the recursive updates in Sec. 2.1.1 effectively avoid the computational complexity bottleneck of an $n \times n$ matrix inversion. However, computing the posterior moments in (3) still requires evaluating an $n \times 1$ vector dot product for the mean and an $n \times n$ quadratic form for the variance. That is, the computational and memory requirements scale as ${ \mathcal { O } } ( n )$ for the mean, and $\mathcal { O } ( n ^ { 2 } )$ for the variance.

This unbounded growth with the number of observations $n ,$ is known in the literature as the “curse of kernelization” [10, 11] and is typically addressed via kernel approximation methods such as Nystrom [12] and random Fourier features¨ [13, 14], or via dictionary-based sparsification methods including forgetting mechanisms [15] and the approximate linear dependency (ALD) technique [16, 17].

In this work, we rely on the ALD technique, as it naturally integrates with the recursive updates in Sec. 2.1.1 (see Sec. 4 for more details).

## 2.2 Warped Gaussian processes

Warped GP models assume that observations follow a GP model only after applying a monotonic, parametric, and differentiable transformation referred to as warping [7]. Specifically, given a warping transformation g of r parameters $\theta \in$ $\Theta \subseteq \mathbb { R } ^ { r }$ , the observation model in (1b) becomes

$$
z _ { i } : = g ( y _ { i } ; \pmb \theta ) = f ( \pmb x _ { i } ) + \epsilon _ { i } ,\tag{6}
$$

where each $z _ { i } \in \mathbb { R }$ serves as a latent target. Accordingly, the term $\mathbf { \alpha } _ { \alpha } ,$ defined in (4a), becomes ${ \pmb { \alpha } } _ { n } = \pmb { \Omega } _ { n } { \pmb z } _ { n } \in \mathbb { R } ^ { n }$ , with $z _ { n } = [ z _ { 1 } , \ldots , z _ { n } ] ^ { \top } \in \mathbb { R } ^ { n }$

Warped GPs can be seen as a generalization of GPs. In fact, they are typically non-Gaussian and even asymmetric in the observation space.

By invoking the change of variables formula [18, Ch. 2], the joint probability density $p$ of any n observations ${ \bf { \nabla } } \pmb { y } _ { n }$ can be expressed in terms of the corresponding density $q$ of latent targets $z _ { n }$ , and the warping transformation g and warping parameters θ. Specifically,

$$
p ( \pmb { y } _ { n } | \pmb { \theta } ) = q ( \pmb { g } ( \pmb { y } _ { n } ; \pmb { \theta } ) ) \prod _ { i = 1 } ^ { n } \frac { \partial g ( \pmb { y } _ { i } ; \pmb { \theta } ) } { \partial \pmb { y } _ { i } } ,\tag{7}
$$

Algorithm 1 Proposed online method for warped GPs.   
1: % tilde superscripts distinguish sparsified variables from   
their exact counterparts.   
2: Choose $g , \Theta , \kappa , \{ \sigma _ { n } \}$ , the ALD sparsification threshold   
$\nu ,$ and the projected optimizer $\Pi _ { \Theta } .$   
3: Initialize $\begin{array} { r } { \tilde { \pmb { \theta } } _ { 0 } \overset {  } { \in } \Theta , \tilde { \pmb { K } } _ { 0 } ^ { - 1 } = 1 / \kappa ( \pmb { x } _ { 1 } , \pmb { x } _ { 1 } ) , \tilde { \pmb { \alpha } } _ { 0 } = \tilde { \pmb { C } } _ { 0 } = 0 , } \end{array}$   
$\tilde { B } _ { 0 } = \mathbf { 0 } _ { r } ,$ and the dictionary $\mathcal { D } _ { 0 } = \{ \pmb { x } _ { 1 } \}$   
4: for $n = 1 , 2 , \ldots$ do   
5: Observe ${ \pmb x } _ { n }$ and $y _ { n }$   
6: Compute $\hat { \pmb { a } } _ { n } = \tilde { \pmb { K } } _ { n - 1 } ^ { - 1 } \tilde { \pmb { k } } _ { n - 1 } ( { \pmb x } _ { t } )$ , and   
7: $\delta _ { n } = \kappa ( { \pmb x } _ { n } , { \pmb x } _ { n } ) - \tilde { \pmb k } _ { n - 1 } ( { \pmb x } _ { n } ) ^ { \top } \hat { { \pmb a } } _ { n }$   
8: Warp $z _ { n } = g ( y _ { n } ; \pmb \theta _ { n - 1 } )$ and $\dot { z } _ { n } = \nabla _ { \pmb { \theta } } g ( y _ { n } ; \pmb { \theta } _ { n - 1 } )$   
9: Compute $\tilde { e } _ { n _ { \bullet } } = z _ { n _ { \bullet } } - \tilde { k } _ { n - 1 } ( { \pmb x } _ { n } ) ^ { \top } \tilde { \alpha } _ { n - 1 } ,$ , and   
10: $\tilde { \pmb { b } } _ { n } = \dot { \pmb { z } } _ { n } - \tilde { \pmb { B } } _ { n - 1 } \tilde { \pmb { k } } _ { n - 1 } ( { \pmb { x } } _ { n } )$   
11: if $\delta _ { n } > \nu$ then   
12: Update $\begin{array} { r } { \tilde { \bf K } _ { n } ^ { - 1 } = \frac { 1 } { \delta _ { n } } \left[ \begin{array} { c c } { \delta _ { n } \tilde { \bf K } _ { n - 1 } ^ { - 1 } + \hat { \bf a } _ { n } \hat { \bf a } _ { n } ^ { \top } } & { - \hat { \bf a } _ { n } } \\ { - \hat { \bf a } _ { n } ^ { \top } } & { 1 } \end{array} \right] } \end{array}$   
13: Compute $\tilde { \mathbf { c } } _ { n } = \left[ \begin{array} { c } { - \tilde { C } _ { n - 1 } \tilde { \mathbf { k } } _ { n - 1 } ( \mathbf { x } _ { n } ) } \\ { 1 } \end{array} \right]$ , and   
14: $\tilde { s } _ { n } = \kappa ( { \pmb x } _ { n } , { \pmb x } _ { n } ) + \sigma _ { n } ^ { 2 } - \tilde { \pmb k } _ { n - 1 } ( { \pmb x } _ { n } ) ^ { \top } \tilde { \pmb C } _ { n - 1 } \tilde { \pmb k } _ { n - 1 } ( { \pmb x } _ { n } )$   
15: Reshape $\tilde { C } _ { n - 1 } = \left[ \begin{array} { c c } { { \tilde { C } _ { n - 1 } } } & { { { \bf 0 } _ { D _ { n } - 1 } } } \\ { { { \bf 0 } _ { D _ { n } - 1 } ^ { \top } } } & { { 0 } } \end{array} \right] ,$   
16: $\tilde { B } _ { n - 1 } = [ \tilde { B } _ { n - 1 } , \mathbf { 0 } _ { r } ] ,$ , and $\tilde { \alpha } _ { n - 1 } = \left[ \begin{array} { c } { { \tilde { \alpha } _ { n - 1 } } } \\ { { 0 } } \end{array} \right]$   
17: $\mathbf { U p d a t e } \mathcal { D } _ { n } = \mathcal { D } _ { n - 1 } \cup \left\{ \pmb { x } _ { n } \right\}$   
18: else   
19: Update $\tilde { K } _ { n } = \tilde { K } _ { n - 1 }$   
20: Compute $\tilde { c } _ { n } = \hat { a } _ { n } - \tilde { C } _ { n - 1 } \tilde { k } _ { n - 1 } ( \pmb { x } _ { n } ) ,$ , and ${ \tilde { s } } _ { n } =$   
21: $\tilde { \pmb { k } } _ { n - 1 } ( { \pmb x } _ { n } ) ^ { \top } \hat { \pmb { a } } _ { n } + \sigma _ { n } ^ { 2 } - \tilde { \pmb { k } } _ { n - 1 } ( { \pmb x } _ { n } ) ^ { \top } \tilde { \pmb { C } } _ { n - 1 } \tilde { \pmb { k } } _ { n - 1 } ( { \pmb x } _ { n } )$   
22: Update $\mathcal { D } _ { n } = \mathcal { D } _ { n - 1 }$   
23: end if   
24: Update $\begin{array} { r } { \tilde { \alpha } _ { n } = \tilde { \alpha } _ { n - 1 } + \frac { \tilde { e } _ { n } } { \tilde { s } _ { n } } \tilde { c } _ { n } , \tilde { B } _ { n } = \tilde { B } _ { n - 1 } + \frac { 1 } { \tilde { s } _ { n } } \tilde { b } _ { n } \tilde { c } _ { n } ^ { \top } , } \end{array}$   
25: and $\begin{array} { r } { \tilde { C } _ { n } = \tilde { C } _ { n - 1 } + \frac { 1 } { \tilde { s } _ { n } } \tilde { c } _ { n } \tilde { c } _ { n } ^ { \top } } \end{array}$   
26: Compute $\begin{array} { r } { \nabla _ { \pmb { \theta } } \tilde { \ell } _ { n } ( \pmb { \theta } _ { n - 1 } ) = \frac { \tilde { e } _ { n } } { \tilde { s } _ { n } } \tilde { \pmb { b } } _ { n } - \nabla _ { \pmb { \theta } } \log ( \frac { \partial z _ { n } } { \partial y _ { n } } ) } \end{array}$   
27: Update $\pmb { \theta } _ { n } = \Pi _ { \Theta } ( \pmb { \theta } _ { n - 1 } , \nabla \pmb { \theta } \tilde { \ell } _ { n } ( \pmb { \theta } _ { n - 1 } ) )$   
28: end for   
29: Return $\mathcal { D } _ { n } , \tilde { \alpha } _ { n } , \tilde { C } _ { n } ,$ , and $\pmb \theta _ { n }$

where g applies the warping transformation g elementwise, and the joint probability of latent targets follows

$$
q ( z ) = \left( ( 2 \pi ) ^ { n } \left| \Omega _ { n } ^ { - 1 } \right| \right) ^ { - \frac { 1 } { 2 } } \exp \left( - \frac { 1 } { 2 } z _ { n } ^ { \top } \Omega _ { n } z _ { n } \right)\tag{8}
$$

by construction from (6).

One of the greatest strengths of warped GPs is that the warping parameters θ can be estimated directly from the observations ${ \bf { \nabla } } \pmb { y } _ { n }$ without requiring explicit knowledge of their joint probability density p. As long as the latent density q and the warping transformation $g$ are defined, (7) provides a tractable likelihood that can be optimized.

## 3 Recursive gradient computation

The n-observation joint NLL

$$
L _ { n } ( \pmb { \theta } ) = - \log p ( \pmb { y } _ { n } | \pmb { \theta } )\tag{9}
$$

measures the discrepancy between the observations ${ \bf { \nabla } } \pmb { y } _ { n }$ and the joint probability density $p$ (implicitly) described by the warping parameters $\pmb \theta .$ Using the multiplication rule of probability [19], one can readily get that

$$
L _ { n } ( \pmb \theta ) = L _ { n - 1 } ( \pmb \theta ) + \ell _ { n } ( \pmb \theta ) ,\tag{10}
$$

where the instantaneous NLL

$$
\ell _ { n } ( \pmb { \theta } ) = - \log p ( y _ { n } | \pmb { y } _ { n - 1 } , \pmb { \theta } ) ,\tag{11}
$$

quantifies how unexpected the last observation $y _ { n }$ was given the previous observations $\scriptstyle { \mathbf { y } } _ { n - 1 }$

In many applications, the set of observations $y _ { 1 } , \ldots , y _ { n }$ is too large to fit in memory, or is streamed, rendering standard batch processing intractable. As a result, estimating the warping parameters by directly minimizing the NLL in (9) with respect to θ may not be possible in practice. On the other hand, by exploiting the additive decomposition of the NLL in (10), one can naturally transition from a batch to an online estimation framework. Specifically, the warping parameter estimate can be iteratively refined utilizing the gradient of the instantaneous NLL, e.g., by a first-order method [20].

Although evaluating every instantaneous NLL $\ell _ { n } ( \pmb \theta )$ depends on the previous observations ${ \mathbf { } } y _ { n - 1 } .$ , Theorem 3.1 demonstrates that its gradient can be computed recursively.

Theorem 3.1 (Recursive gradient computation). The gradient of the instantaneous NLL in (11) with respect to the warping parameters θ can be computed as

$$
\nabla _ { \pmb { \theta } } \ell _ { n } ( \pmb { \theta } ) = \frac { e _ { n } } { s _ { n } } \pmb { b } _ { n } - \nabla _ { \pmb { \theta } } \log \left( \frac { \partial z _ { n } } { \partial y _ { n } } \right) ,\tag{12}
$$

where the terms $e _ { n } \in \mathbb { R }$ and $b _ { n } \in \mathbb { R } ^ { r }$ correspond to

$$
e _ { n } = z _ { n } - m _ { n - 1 } ( \mathbf { x } _ { n } ) , a n d\tag{13a}
$$

$$
\pmb { b } _ { n } = \dot { z } _ { n } - \pmb { B } _ { n - 1 } \pmb { k } _ { n - 1 } ( \pmb { x } _ { n } ) ,\tag{13b}
$$

with $\dot { z } _ { n } = \nabla _ { \pmb { \theta } } g ( y _ { n } ; \pmb { \theta } ) \in \mathbb { R } ^ { r }$ , and the term $\boldsymbol { B } _ { n } \in \mathbb { R } ^ { r \times n }$ acts as a state variable updated according to

$$
\mathbf { } B _ { n } = \left[ B _ { n - 1 } \quad \mathbf { 0 } _ { r } \right] - { \frac { { \dot { z } } _ { n } - B _ { n - 1 } k _ { n - 1 } ( { \pmb x } _ { n } ) } { s _ { n } } } \left[ { \pmb \omega } _ { n } ^ { \top } \quad - 1 \right] .\tag{14}
$$

Proof. See the supplementary material S1.

As a result, the gradient of the instantaneous NLL $\nabla _ { \pmb { \theta } } \ell _ { n } ( \pmb { \theta } )$ can be updated recursively as each new observation $y _ { n }$ and associated input location ${ \pmb x } _ { n }$ become available by maintaining $\alpha _ { n } , \Omega _ { n }$ (as in Sec. 2.1.1), and $B _ { n }$ as state variables.

![](images/9d03ec4beb6d72fadf48113491423dbde291d6e917ba72f386120e9c7bf001e9.jpg)  
(a) True process, warped GP, and distribution slices (gray dashed line)

![](images/473ad187457876a5480bc524cd3c7cf23cd014426a66d5e2ad930cb1efb1b29a.jpg)  
(b) Forward and inverse warping transformation  
Fig. 2. In Fig. 2a, the observations are marked with gray dots. The dotted lines show the true generating distribution $p ( \pmb { y } )$ and the solid lines show the warped GP prediction $p ( \pmb { y } | \pmb \theta )$ . The triplets of lines represent the median, along with the 2.5th and 97.5th percentiles in each case. The cross-section shows the probability densities at $\textstyle x = - { \frac { \pi } { 1 2 } }$ ; that is, $\begin{array} { r } { p ( y | x = - \frac { \pi } { 1 2 } ) } \end{array}$ and $\begin{array} { r } { p ( y | \pmb { \theta } , x = - \frac { \pi } { 1 2 } ) } \end{array}$ . Fig. 2b shows the forward-inverse pass of the estimated warping transformation as well as its inversion error.

## 4 Proposed online method

Algorithm 1 outlines the proposed online method for warped GPs, which jointly updates the latent GP moments and optimizes the warping parameters. It integrates the recursive GP updates from Sec. 2.1.1, the recursive gradient computation of the instantaneous NLL from Sec. 3, and the ALD sparsification technique introduced in Sec. 2.1.2. As a result, the update rule (12) yields an approximate gradient.

The per-step computational and memory complexity is $\mathcal { O } ( D _ { n } ^ { 2 } + r D _ { n } )$ . We refer the reader to the supplementary material S2 and S3 for further details.

## 5 Revisiting the 1D regression task

To evaluate our proposed method, we adapt the simple 1D regression task introduced in [7, Sec. 4] to an online setting. This task is specifically designed to generate non-Gaussian observations with sharp transitions, causing standard GPs to fail; therefore, relying entirely on the modeling capabilities of the warped GP.

Experimental setup. The input domain is the real subset $\mathcal { X } \ : = \ : [ - \pi , \pi ] \subset \mathbb { R }$ . The observational data are generated from a sinusoidal signal corrupted by white Gaussian noise followed by a cubic root transformation. That is, every nth observation is generated as $y _ { n } = ( \sin ( x _ { n } ) + \varepsilon _ { n } ) ^ { \frac { 1 } { 3 } }$ with $\varepsilon _ { n } \sim \mathcal { N } ( 0 , \frac { 1 } { 3 ^ { 2 } } )$ . Then, we generate 1000 realizations of 101 uniformly spaced observations each. That is, a total of 101000 observations, input location pairs that are fed sequentially to the warped GP model.

Model configuration. The choice of warping transformation follows that of the revisited task. That $\operatorname { i s } , g ( y ; \pmb { \theta } ) = y +$ $\textstyle \sum _ { i = 1 } ^ { t } \theta _ { 1 , i }$ tanh $\left( \theta _ { 2 , i } ( y + \theta _ { 3 , i } ) \right)$ , where $\pmb { \theta } = [ \pmb { \theta } _ { 1 } ^ { \top } , \pmb { \theta } _ { 2 } ^ { \top } , \pmb { \theta } _ { 3 } ^ { \top } ] ^ { \top } \in$ Θ, and $\begin{array} { r } { \Theta = \{ \pmb \theta \in \mathbb { R } ^ { 3 t } : \pmb \theta _ { 1 } , \pmb \theta _ { 2 } \succeq \mathbf 0 _ { t } , \mathbf 1 _ { t } ^ { \top } \pmb \theta _ { 1 } > 0 } \end{array}$ , and $\mathbf { 1 } _ { t } ^ { \top } \pmb { \theta } _ { 2 } >$ 0}, to ensure monotonicity. We use $t ~ = ~ 1 0$ terms, which leads to a total of $r \ = \ 3 0$ warping parameters. Similarly, the choice of the covariance function remains the same, i.e., $\kappa ( x , x ^ { \prime } ) ~ = ~ k _ { a } \exp ( - { \textstyle \frac { 1 } { 2 k _ { \scriptscriptstyle m } ^ { 2 } } } ( x - x ^ { \prime } ) ^ { 2 } )$ , with $k _ { a } \ = \ 2 .$ , and $\begin{array} { r } { k _ { w } = 2 \cdot \frac { 2 \pi } { 1 0 1 } \approx 0 . 1 2 4 } \end{array}$ . Lastly, the noise standard deviation of the latent observation model is set to $\sigma = 3$

Implementation details. We use the Adam<sup>1</sup> optimizer [22], followed by a projection onto the feasible set Θ. We set the ALD threshold to $\nu = 0 . 1$

Results. The results of the experiment are summarized in Fig. 2. Fig. 2a shows that the predictions of the warped GP closely align with the true generating distribution. On the other hand, Fig. 2b shows that the learned warped transformation closely resembles the true generating cubic transformations in the region of interest where most observations y lie, $\mathrm { i . e . , ~ } - 1 \lesssim y \lesssim 1$ . It also shows that its inversion is numerically stable. The (ALD sparsification) dictionary ends up with 68 atoms. Finally, the warped GP achieves an empirical NLL of −0.28 nats per observation. This is a 0.76 nat improvement over a standard GP under an identical model configuration except for a well-calibrated noise variance.

## 6 Conclusion and future work

We presented a recursive gradient evaluation for the instantaneous NLL of warped GPs, requiring only one extra state variable over standard recursive implementations. Building on this result, we introduced an online algorithm for jointly updating latent GP moments and optimizing warping parameters, demonstrating its performance on a warped GP benchmark. Future work includes online learning of the kernel parameters (or directly a suitable kernel [23]) and the observation model noise variance; evaluations on broader benchmarks are underway.

## 7 References

[1] Carl Edward Rasmussen, “Gaussian processes in machine learning,” in Summer school on machine learning, pp. 63–71. Springer, 2003.

[2] Laura P Swiler, Mamikon Gulian, Ari L Frankel, Cosmin Safta, and John D Jakeman, “A survey of constrained gaussian process regression: Approaches and implementation challenges,” Journal of Machine Learning for Modeling and Computing, vol. 1, no. 2, 2020.

[3] Emilio Ruiz-Moreno and Baltasar Beferull-Lozano, “Doubly truncated mode kriging,” in 2025 IEEE Statistical Signal Processing Workshop (SSP). IEEE, 2025, pp. 306–310.

[4] Marc Deisenroth and Carl E Rasmussen, “Pilco: A modelbased and data-efficient approach to policy search,” in Proceedings of the 28th International Conference on machine learning (ICML-11), 2011, pp. 465–472.

[5] Peter J Diggle, Jonathan A Tawn, and Rana A Moyeed, “Model-based geostatistics,” Journal of the Royal Statistical Society Series C: Applied Statistics, vol. 47, no. 3, pp. 299– 350, 1998.

[6] Maura Dewey, Laura Wilcox, Bjørn Samset, and Annica Ekman, “Deep-aerogp: deep kernel learning for projecting the regional climate response to anthropogenic aerosol emission changes,” Tech. Rep., Copernicus Meetings, 2026.

[7] Edward Snelson, Zoubin Ghahramani, and Carl Rasmussen, “Warped gaussian processes,” Advances in neural information processing systems, vol. 16, 2003.

[8] Peng Kou, Feng Gao, and Xiaohong Guan, “Sparse online warped gaussian process for wind power probabilistic forecasting,” Applied energy, vol. 108, pp. 410–428, 2013.

[9] Thang Bui, Daniel Hernandez-Lobato, Jose Hernandez-´ Lobato, Yingzhen Li, and Richard Turner, “Deep gaussian processes for regression using approximate expectation propagation,” in International conference on machine learning. PMLR, 2016, pp. 1472–1481.

[10] Zhuang Wang, Koby Crammer, and Slobodan Vucetic, “Breaking the curse of kernelization: Budgeted stochastic gradient descent for large-scale svm training,” The Journal ofMachine Learning Research, vol. 13, no. 1, pp. 3103–3131, 2012.

[11] Daniele Calandriello, Alessandro Lazaric, and Michal Valko, “Efficient second-order online kernel learning with adaptive embedding,” Advances in Neural Information Processing Systems, vol. 30, 2017.

[12] Christopher Williams and Matthias Seeger, “Using the nystrom¨ method to speed up kernel machines,” Advances in neural information processing systems, vol. 13, 2000.

[13] Ali Rahimi and Benjamin Recht, “Random features for largescale kernel machines,” Advances in neural information processing systems, vol. 20, 2007.

[14] Rohan T Money, Joshin P Krishnan, and Baltasar Beferull-Lozano, “Sparse online learning with kernels using random features for estimating nonlinear dynamic graphs,” IEEE Transactions on Signal Processing, vol. 71, pp. 2027–2042, 2023.

[15] Konstantinos Slavakis, Pantelis Bouboulis, and Sergios Theodoridis, “Online learning in reproducing kernel hilbert spaces,” in Academic Press Library in Signal Processing, vol. 1, pp. 883–987. Elsevier, 2014.

[16] Yaakov Engel, Shie Mannor, and Ron Meir, “The kernel recursive least-squares algorithm,” IEEE Transactions on signal processing, vol. 52, no. 8, pp. 2275–2285, 2004.

[17] Yaakov Engel, Algorithms and representations for reinforcement learning, Hebrew University of Jerusalem Jerusalem, 2005.

[18] George Casella and Roger Berger, Statistical inference, CRC press, 2024.

[19] Sheldon M Ross, Sheldon M Ross, Sheldon M Ross, Sheldon M Ross, and Etats-Unis Mathematicien,´ A first course in probability, vol. 8, Pearson London, 2014.

[20] Amir Beck, First-order methods in optimization, SIAM, 2017.

[21] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al., “Pytorch: An imperative style, high-performance deep learning library,” Advances in neural information processing systems, vol. 32, 2019.

[22] Diederik P Kingma and Jimmy Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.

[23] Emilio Ruiz-Moreno and Baltasar Beferull-Lozano, “An online multiple kernel parallelizable learning scheme,” IEEE Signal Processing Letters, vol. 31, pp. 121–125, 2023.

[24] Roger A Horn and Charles R Johnson, Matrix analysis, Cambridge university press, 2012.

# Supplementary material for “Online Gradient Computation for Warping Gaussian Process Transformations”

## S1 Proof of Theorem 3.1

The NLL in (9) corresponds to

$$
\begin{array} { r l } & { L _ { n } ( \pmb \theta ) = - \log p ( \pmb y _ { n } | \pmb \theta ) } \\ & { \quad \quad = - \log q ( \pmb g ( \pmb y _ { n } ; \pmb \theta ) ) - \log \displaystyle \prod _ { i = 1 } ^ { n } \frac { \partial g ( \pmb y _ { i } ; \pmb \theta ) } { \partial \pmb y _ { i } } } \\ & { \quad \quad = \displaystyle \frac { 1 } { 2 } ( 2 \pi ) ^ { n } \left| \pmb { \Omega } _ { n } ^ { - 1 } \right| + \frac { 1 } { 2 } g ( \pmb y _ { n } ; \pmb \theta ) ^ { \top } \pmb { \Omega } _ { n } g ( \pmb y _ { n } ; \pmb \theta ) - \displaystyle \sum _ { i = 1 } ^ { n } \log \left( \frac { \partial g ( \pmb y _ { i } ; \pmb \theta ) } { \partial \pmb y _ { i } } \right) . } \end{array}\tag{s1a}
$$

(s1b)

(s1c)

Its gradient with respect to the warping parameters θ is thus,

$$
\begin{array} { l } { \displaystyle \nabla _ { \theta } L _ { n } ( \theta ) = \underbrace { \nabla _ { \theta } \frac { 1 } { 2 } ( 2 \pi ) ^ { p - 1 } \widehat { \Omega _ { n } ^ { - 1 } } _ { 1 } } _ { = \Theta } + \frac { 1 } { 2 } \nabla _ { \theta } ^ { \top } g ( y _ { n } ; \theta ) \Omega _ { n } g ( y _ { n } ; \theta ) - \displaystyle \sum _ { i = 1 } ^ { n } \nabla _ { \theta } \log \left( \frac { \partial g ( y _ { i } ; \theta ) } { \partial y _ { i } } \right) } \\ { \displaystyle = \dot { Z } _ { n } \Omega _ { n } z _ { n } - \displaystyle \sum _ { i = 1 } ^ { n } \nabla _ { \theta } \log \left( \frac { \partial z _ { i } } { \partial y _ { i } } \right) , } \end{array}\tag{s2a}
$$

(s2b)

where

$$
z _ { n } = g ( y _ { n } ; \pmb \theta ) = \left[ \begin{array} { l } { z _ { 1 } } \\ { \vdots } \\ { z _ { n } } \end{array} \right] \in \mathbb { R } ^ { n } ,\tag{s3a}
$$

$$
\dot { z } _ { i } = \nabla _ { \pmb { \theta } } g ( y _ { i } ; \pmb { \theta } ) \in \mathbb { R } ^ { r } , \mathrm { ~ f o r ~ a l l ~ } i \in \{ 1 , 2 , \dots , n \} , \mathrm { ~ a n d ~ }\tag{s3b}
$$

$$
\dot { Z } _ { n } = \nabla _ { \theta } ^ { \top } g ( y _ { n } ; \theta ) = \left[ \dot { z } _ { 1 } \quad \dot { z } _ { 2 } \quad \ldots \quad \dot { z } _ { n } \right] = \left[ \dot { Z } _ { n - 1 } \quad \dot { z } _ { n } \right] \in \mathbb { R } ^ { r \times n } .\tag{s3c}
$$

The first term in (s2b) can be expanded by using the recursive representation of $\pmb { \Omega } _ { n }$ described in (5b) as

$$
\begin{array} { r l } { \dot { Z } _ { n } \Omega _ { n } z _ { n } = [ \dot { Z } _ { n - 1 }  } & { { } \dot { z } _ { n } ] ( [ \begin{array} { c c } { \Omega _ { n - 1 } } & { \mathbf { 0 } _ { n - 1 } } \\ { \mathbf { 0 } _ { n - 1 } ^ { \top } } & { 0 } \end{array} ] + \frac { 1 } { s _ { n } } [ \begin{array} { c c } { \omega _ { n } } \\ { - 1 } \end{array} ] [ \omega _ { n } ^ { \top } } & { - 1 ] ) [ \begin{array} { c } { z _ { n - 1 } } \\ { z _ { n } } \end{array} ] } \end{array}\tag{s4a}
$$

$$
= \dot { \bar { Z } } _ { n - 1 } \Omega _ { n - 1 } z _ { n - 1 } + \frac { 1 } { s _ { n } } \left( \dot { \bar { Z } } _ { n - 1 } \omega _ { n } - \dot { z } _ { n } \right) \left( \omega _ { n } ^ { \top } z _ { n - 1 } - z _ { n } \right)\tag{s4b}
$$

$$
= \dot { Z } _ { n - 1 } \Omega _ { n - 1 } z _ { n - 1 } + \frac { e _ { n } } { s _ { n } } b _ { n } ,\tag{s4c}
$$

where

$$
e _ { n } = z _ { n } - \omega _ { n } ^ { \top } z _ { n - 1 } \in \mathbb { R } , ~ \mathrm { a n d }\tag{s5a}
$$

$$
\pmb { b } _ { n } = \dot { z } _ { n } - \dot { \pmb { Z } } _ { n - 1 } \pmb { \omega } _ { n } \in \mathbb { R } ^ { r } .\tag{s5b}
$$

Note that $e _ { n }$ already admits a recursive update rule since it is expressed in terms of the current latent target $z _ { n }$ and its corresponding location ${ \pmb x } _ { n }$ , and the previous state terms $\scriptstyle \alpha _ { n - 1 }$ and $\Omega _ { n - 1 }$ . Explicitly,

$$
e _ { n } = z _ { n } - k _ { n - 1 } ( { \pmb x } _ { n } ) ^ { \top } \pmb \Omega _ { n - 1 } z _ { n - 1 }\tag{s6a}
$$

$$
= z _ { n } - k _ { n - 1 } \big ( \pmb { x } _ { n } \big ) ^ { \top } \pmb { \alpha } _ { n - 1 }\tag{s6b}
$$

$$
= z _ { n } - m _ { n - 1 } ( \mathbf { x } _ { n } ) .\tag{s6c}
$$

Regarding $b _ { n }$ , we maintain an additional state term $\pmb { B } _ { n } = \dot { \pmb { Z } } _ { n } \pmb { \Omega } _ { n } \in \mathbb { R } ^ { r \times n }$ to enable recursive updates. Specifically,

$$
B _ { n } = \left[ \dot { \bar { Z } } _ { n - 1 } \quad \dot { z } _ { n } \right] \left( \left[ \begin{array} { c c } { { \Omega _ { n - 1 } } } & { { \mathbf { 0 } _ { n - 1 } } } \\ { { \mathbf { 0 } _ { n - 1 } ^ { \top } } } & { { 0 } } \end{array} \right] + \frac { 1 } { s _ { n } } \left[ \begin{array} { c c } { { \omega _ { n } } } \\ { { - 1 } } \end{array} \right] \left[ \omega _ { n } ^ { \top }  &  { - 1 } \right] \right)\tag{s7a}
$$

$$
= \left[ \dot { \pmb { Z } } _ { n - 1 } \pmb { \Omega } _ { n - 1 } \quad \pmb { 0 } _ { r } \right] + \frac { 1 } { s _ { n } } \left( \dot { \pmb { Z } } _ { n - 1 } \pmb { \omega } _ { n } - \dot { \pmb { z } } _ { n } \right) \left[ \pmb { \omega } _ { n } ^ { \top } \quad - 1 \right]\tag{s7b}
$$

$$
= \left[ { \pmb B } _ { n - 1 } \quad { \bf 0 } _ { r } \right] - \frac { 1 } { s _ { n } } \left( \dot { z } _ { n } - { \pmb B } _ { n - 1 } { \pmb k } _ { n - 1 } ( { \pmb x } _ { n } ) \right) \left[ { \pmb \omega } _ { n } ^ { \top } \quad - 1 \right] ,\tag{s7c}
$$

and accordingly,

$$
\pmb { b } _ { n } = \dot { z } _ { n } - \dot { Z } _ { n - 1 } \Omega _ { n - 1 } \pmb { k } _ { n - 1 } ( \pmb { x } _ { n } )
$$

$$
= { \dot { z } } _ { n } - B _ { n - 1 } \pmb { k } _ { n - 1 } ( \pmb { x } _ { n } ) .\tag{s8a}
$$

(s8b)

Substituting (s4c) in (s2b) we get

$$
{ \nabla _ { \theta } L _ { n } ( \theta ) = \dot { Z } _ { n - 1 } \Omega _ { n - 1 } z _ { n - 1 } + \frac { { e _ { n } } } { { s _ { n } } } b _ { n } - \sum _ { i = 1 } ^ { n } \nabla _ { \theta } \log \left( \frac { { \partial z _ { i } } } { { \partial { y _ { i } } } } \right) }\tag{s9a}
$$

$$
= { \dot { Z } } _ { n - 1 } \Omega _ { n - 1 } z _ { n - 1 } - \sum _ { i = 1 } ^ { n - 1 } \nabla _ { \theta } \log \left( { \frac { \partial z _ { i } } { \partial y _ { i } } } \right) + { \frac { e _ { n } } { s _ { n } } } \pmb { b } _ { n } - \nabla _ { \theta } \log \left( { \frac { \partial z _ { n } } { \partial y _ { n } } } \right)\tag{s9b}
$$

$$
= \nabla _ { \pmb { \theta } } L _ { n - 1 } ( \pmb { \theta } ) + \frac { e _ { n } } { s _ { n } } \pmb { b } _ { n } - \nabla _ { \pmb { \theta } } \log \left( \frac { \partial z _ { n } } { \partial y _ { n } } \right) .\tag{s9c}
$$

Finally, by taking the gradient with respect to the warping parameters $\pmb { \theta }$ in both sides of (10) and identifying terms with (s9) we obtain

$$
\nabla _ { \pmb \theta } \ell _ { n } ( \pmb \theta ) = \frac { e _ { n } } { s _ { n } } \pmb b _ { n } - \nabla _ { \pmb \theta } \log \left( \frac { \partial z _ { n } } { \partial y _ { n } } \right) .\tag{s10}
$$

## S2 ALD sparsification

Consider a feature mapping $\phi : \mathcal { X } \to \mathcal { F }$ such that $\kappa ( { \pmb x } , { \pmb x } ^ { \prime } ) = \langle \phi ( { \pmb x } ) , \phi ( { \pmb x } ^ { \prime } ) \rangle _ { \mathcal { F } }$ for all $\mathbf { { x } } , \mathbf { { x } } ^ { \prime } \in { \mathcal { X } } ,$ , an ongoing trajectory of input locations $\pmb { x } _ { 1 } , \pmb { x } _ { 2 } , \dots , \pmb { x } _ { n - 1 }$ , and a dictionary (sparse set) $\mathcal { D } _ { n - 1 } \subseteq \mathcal { X }$ of representative input locations collected across that trajectory.

Every time a new input location ${ \bf { x } } _ { n }$ is presented, we check whether its feature representation $\phi ( { \pmb x } _ { n } )$ is approximately linearly dependent on the feature representation of the previously collected input locations $\tilde { { \pmb { x } } } _ { 1 } , \tilde { { \pmb { x } } } _ { 2 } , \dots , \tilde { { \pmb { x } } } _ { D _ { n - 1 } }$ in $\mathcal { D } _ { n - 1 }$ where $D _ { n } = | \mathcal { D } _ { n } |$ . That is, we check whether the squared distance $\delta _ { n }$ between $\phi ( { \pmb x } _ { n } )$ and span $\{ \phi ( \tilde { \pmb { x } } _ { 1 } ) , \phi ( \tilde { \pmb { x } } _ { 2 } ) , \dots , \phi ( \tilde { \pmb { x } } _ { D _ { n - 1 } } ) \}$ is less than or equal to a user-defined threshold $\nu \in \mathbb { R } _ { > 0 }$ . If it is, the dictionary remains the same, i.e., $\mathcal { D } _ { n } = \mathcal { D } _ { n - 1 }$ . If not, we update the dictionary to include the new input location, i.e., ${ \mathcal { D } } _ { n } = { \mathcal { D } } _ { n - 1 } \cup \{ { \pmb x } _ { n } \}$

Explicitly,

$$
\delta _ { n } = \operatorname* { m i n } _ { \substack { a \in \mathbb { R } ^ { D _ { n - 1 } } } } \left\| \phi ( { \pmb x } _ { n } ) - \sum _ { i = 1 } ^ { D _ { n - 1 } } a _ { i } \phi ( \tilde { { \pmb x } } _ { i } ) \right\| _ { \mathcal { F } } ^ { 2 }\tag{s11a}
$$

$$
= \operatorname* { m i n } _ { a \in \mathbb { R } ^ { D _ { n - 1 } } } \left. \phi ( \pmb { x } _ { n } ) - \sum _ { i = 1 } ^ { D _ { n - 1 } } a _ { i } \phi ( \tilde { \pmb { x } } _ { i } ) , \phi ( \pmb { x } _ { n } ) - \sum _ { j = 1 } ^ { D _ { n - 1 } } a _ { j } \phi ( \tilde { \pmb { x } } _ { j } ) \right. _ { \pmb { \mathscr { F } } }\tag{s11b}
$$

$$
= \operatorname* { m i n } _ { \substack { a \in \mathbb { R } ^ { D _ { n - 1 } } } } \langle \phi ( { \pmb x } _ { n } ) , \phi ( { \pmb x } _ { n } ) \rangle _ { \mathcal { F } } - 2 \sum _ { i = 1 } ^ { D _ { n - 1 } } a _ { i } \langle \phi ( { \pmb x } _ { n } ) , \phi ( { \pmb x } _ { i } ) \rangle _ { \mathcal { F } } + \sum _ { i , j = 1 } ^ { D _ { n - 1 } } a _ { i } a _ { j } \langle \phi ( { \pmb x } _ { i } ) , \phi ( { \pmb x } _ { j } ) \rangle _ { \mathcal { F } }\tag{s11c}
$$

$$
= \operatorname* { m i n } _ { a \in \mathbb { R } ^ { D _ { n - 1 } } } \kappa ( { \pmb x } _ { n } , { \pmb x } _ { n } ) - 2 \sum _ { i = 1 } ^ { D _ { n - 1 } } a _ { i } \kappa ( { \pmb x } _ { n } , { \tilde { \pmb x } } _ { i } ) + \sum _ { i , j = 1 } ^ { D _ { n - 1 } } a _ { i } a _ { j } \kappa ( { \tilde { \pmb x } } _ { i } , { \tilde { \pmb x } } _ { j } )\tag{s11d}
$$

$$
= \operatorname* { m i n } _ { \substack { a \in \mathbb { R } ^ { D _ { n - 1 } } } } \kappa ( { \pmb x } _ { n } , { \pmb x } _ { n } ) - 2 \tilde { { \pmb k } } _ { n - 1 } ( { \pmb x } _ { n } ) ^ { \top } { \pmb a } + { \pmb a } ^ { \top } \tilde { { \pmb K } } _ { n - 1 } { \pmb a } ,\tag{s11e}
$$

where $\tilde { K } _ { n } \in \mathbb { R } ^ { D _ { n } \times D _ { n } }$ and $\pmb { \tilde { k } } _ { n } ( \pmb { x } ) \in \mathbb { R } ^ { D _ { n } }$ are constructed as $[ \tilde { K } _ { n } ] _ { i , j } = \kappa ( \tilde { \pmb { x } } _ { i } , \tilde { \pmb { x } } _ { j } )$ and $[ \tilde { \pmb { k } } _ { n } ( { \pmb x } ) ] _ { i } = \kappa ( \tilde { \pmb { x } } _ { i } , { \pmb x } )$ for all $\tilde { \pmb { x } } _ { i } , \tilde { \pmb { x } } _ { j } \in \mathcal { D } _ { n }$ and $\pmb { x } \in \mathcal { X }$ . Thus, finding $\delta _ { n }$ consists of solving a convex quadratic optimization problem with a closed-form solution

$$
\delta _ { n } = \kappa ( \pmb { x } _ { n } , \pmb { x } _ { n } ) - \tilde { \pmb { k } } _ { n - 1 } ( \pmb { x } _ { n } ) ^ { \top } \hat { \pmb { a } } _ { n } , \mathrm { ~ w h e r e ~ }\tag{s12a}
$$

$$
\hat { \pmb { a } } _ { n } = \tilde { \pmb { K } } _ { n - 1 } ^ { - 1 } \tilde { \pmb { k } } _ { n - 1 } ( { \pmb x } _ { n } ) .\tag{s12b}
$$

From here, if ${ \pmb x } _ { n }$ is included in $\mathcal { D } _ { n }$ (meaning that $\delta _ { n } > \nu )$ , we set the approximation coefficients as $\pmb { a } _ { n } = [ 0 , \ldots , 0 , 1 ] ^ { \top } \in \{ 0 \} ^ { D _ { n } - 1 } \times$ $\{ 1 \} \subset \mathbb { R } ^ { D _ { n } }$ since $\phi ( { \pmb x } _ { n } )$ can be exactly represented by itself. If not (meaning that $\delta _ { n } \leq \nu )$ , we set $\mathbf { a } _ { n } = \hat { \mathbf { a } } _ { n }$ with $\hat { \pmb { a } } _ { n }$ as in (s12b).

Let us construct the matrix $\Phi _ { n } = [ \phi ( \pmb { x } _ { 1 } ) , \phi ( \pmb { x } _ { 2 } ) , . . . , \phi ( \pmb { x } _ { n } ) ]$ of size dim $( { \mathcal { F } } ) \times n$ . By the sparsification procedure, we know that every ith feature $\phi ( { \pmb x } _ { i } )$ is approximately linearly dependent on the features of the input locations in $\begin{array} { r } { \mathcal { D } _ { i } , \mathrm { i . e . , } \phi ( { \pmb x } _ { i } ) \simeq \sum _ { i = 1 } ^ { D _ { i } } [ { \pmb a } _ { i } ] _ { j } \phi ( \tilde { \pmb x } _ { j } ) } \end{array}$ . Based on this, we can approximate $\Phi _ { n } \simeq \tilde { \Phi } _ { n } A _ { n } ^ { \top }$ where $\tilde { \Phi } _ { n } = [ \phi ( \tilde { \pmb { x } } _ { 1 } ) , \phi ( \tilde { \pmb { x } } _ { 2 } ) , \dots , \phi ( \tilde { \pmb { x } } _ { D _ { n } } ) ]$ is a matrix of size dim $( \mathcal { F } ) \times D _ { n }$ , and A is a matrix of size $n \times D _ { n }$ for which each ith row is constructed by padding zeros to $\mathbf { \alpha } _  \mathbf { \alpha } \mathbf { \alpha } _ { \mathbf { \alpha } } \mathbf { \alpha } _  \mathbf { \alpha } \mathbf { \alpha } _ { \mathbf { \beta } \mathbf { \alpha } _ { \mathbf { \alpha } \mathbf { \beta } \mathbf { \alpha } _ { \mathbf { \beta } \mathbf { \alpha } _ { \lambda } \mathbf { \alpha } _ { \lambda } \mathbf { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } } } }$ until dimension $D _ { n }$ . On the other hand, we know that $K _ { n } = \Phi _ { n } ^ { \top } \Phi _ { n }$ by construction. As a result, we can approximate

$$
K _ { n } = \Phi _ { n } ^ { \top } \Phi _ { n }
$$

$$
\simeq A _ { n } \tilde { \Phi } _ { n } ^ { \top } \tilde { \Phi } _ { n } \pmb { A } _ { n } ^ { \top }\tag{s13a}
$$

$$
\mathbf { \Phi } = { \pmb { A } } _ { n } { \tilde { \pmb { K } } } _ { n } { \pmb { A } } _ { n } ^ { \top } .\tag{s13b}
$$

(s13c)

Similarly,

$$
\pmb { k } _ { n } ( \pmb { x } ) = \left[ \phi ( \pmb { x } _ { 1 } ) ^ { \top } \phi ( \pmb { x } ) \quad \phi ( \pmb { x } _ { 2 } ) ^ { \top } \phi ( \pmb { x } ) \quad \cdots \quad \phi ( \pmb { x } _ { n } ) ^ { \top } \phi ( \pmb { x } ) \right] = \pmb { \Phi } _ { n } ^ { \top } \phi ( \pmb { x } )\tag{s14a}
$$

$$
\simeq \left( \tilde { \Phi } _ { n } \pmb { A } _ { n } ^ { \top } \right) ^ { \top } \phi ( \pmb { x } ) = \pmb { A } _ { n } \tilde { \Phi } _ { n } ^ { \top } \phi ( \pmb { x } )\tag{s14b}
$$

$$
\begin{array} { r l } { { \bf \Pi } } & { { } = { \cal A } _ { n } \tilde { \pmb { k } } _ { n } ( { \pmb x } ) . } \end{array}\tag{s14c}
$$

By extension, plugging (s13) and (s14) into (3) yields the following approximation of the posterior moments

$$
m _ { n } ( { \pmb x } ) \simeq \tilde { m } _ { n } ( { \pmb x } ) = \tilde { k } _ { n } ( { \pmb x } ) ^ { \top } \tilde { \pmb { \alpha } } _ { n } , \ \mathrm { a n d }\tag{s15a}
$$

$$
v _ { n } ( { \pmb x } ) \simeq \tilde { v } _ { n } ( { \pmb x } ) = \kappa ( { \pmb x } , { \pmb x } ) - \tilde { \pmb k } _ { n } ( { \pmb x } ) ^ { \top } \tilde { C } _ { n } \tilde { \pmb k } _ { n } ( { \pmb x } ) ,\tag{s15b}
$$

where

$$
\tilde { \pmb { \alpha } } _ { n } = \pmb { A } _ { n } ^ { \top } \tilde { \pmb { \Omega } } _ { n } \pmb { y } _ { n } \in \mathbb { R } ^ { D _ { n } } ,\tag{s16a}
$$

$$
\tilde { \Omega } _ { n } = \left( A _ { n } \tilde { K } _ { n } \pmb { A } _ { n } ^ { \top } + \pmb { \Sigma } _ { n } \right) ^ { - 1 } \in \mathbb { R } ^ { n \times n } , \ \mathrm { a n d }
$$

$$
\begin{array} { r } { \tilde { \cal C } _ { n } = { \cal A } _ { n } ^ { \top } \tilde { \Omega } _ { n } { \cal A } _ { n } \in \mathbb { R } ^ { D _ { n } \times D _ { n } } . } \end{array}\tag{s16b}
$$

(s16c)

## S3 ALD sparse and recursive Gaussian process updates

The goal is updating the approximated posterior moments in (s15) from the current observation y , input location $y _ { n }$ ${ \pmb x } _ { n } .$ , and the previous state variables $\tilde { \alpha } _ { n - 1 }$ and $\tilde { C } _ { n - 1 }$ . Recall from Sec. S2 that every nth input location ${ \pmb x } _ { n }$ may either be left out of the dictionary, in which case $\mathcal { D } _ { n } = \mathcal { D } _ { n - 1 }$ , or added to it, in which case ${ \mathcal { D } } _ { n } = { \mathcal { D } } _ { n - 1 } \cup \left\{ { \pmb x } _ { n } \right\}$ . Depending on the case, the recursive update will vary accordingly.

## S3.1 Case $\mathcal { D } _ { n } = \mathcal { D } _ { n - 1 }$ (meaning that $\delta _ { n } \leq \nu )$

Since the dictionary remains unchanged

$$
\tilde { \pmb { K } } _ { n } = \tilde { \pmb { K } } _ { n - 1 } \in \mathbb { R } ^ { D _ { n } \times D _ { n } } ,\tag{s17a}
$$

$$
\pmb { A } _ { n } = \left[ \pmb { A } _ { n - 1 } \right] \in \mathbb { R } ^ { n \times D _ { n } } , \ \mathrm { a n d }\tag{s17b}
$$

$$
\mathbf { { a } } _ { n } = \tilde { K } _ { n - 1 } ^ { - 1 } \tilde { k } _ { n - 1 } \big ( \mathbf { { x } } _ { n } \big ) .\tag{s17c}
$$

Then,

$$
\tilde { \Omega } _ { n } ^ { - 1 } = \left[ \begin{array} { c c c } { { { \bf A } _ { n - 1 } } } \\ { { { \bf a } _ { n } ^ { \top } } } \end{array} \right] \tilde { K } _ { n - 1 } \left[ \begin{array} { c c c } { { \bf A } _ { n - 1 } ^ { \top } } & { { \bf a } _ { n } } \end{array} \right] + \left[ \begin{array} { c c c } { { \sigma ^ { 2 } I _ { n - 1 } } } & { { { \bf 0 } _ { n - 1 } } } \\ { { { \bf 0 } _ { n - 1 } } } & { { \sigma ^ { 2 } } } \end{array} \right]\tag{s18a}
$$

$$
\begin{array} { r l } { \mathbf { \Sigma } } & { = \left[ \begin{array} { l l } { \pmb { A } _ { n - 1 } \tilde { \pmb { K } } _ { n - 1 } \pmb { A } _ { n - 1 } ^ { \top } } & { \pmb { A } _ { n - 1 } \tilde { \pmb { K } } _ { n - 1 } \pmb { a } _ { n } } \\ { \pmb { a } _ { n } ^ { \top } \tilde { \pmb { K } } _ { n - 1 } \pmb { A } _ { n - 1 } ^ { \top } } & { \pmb { a } _ { n } ^ { \top } \tilde { \pmb { K } } _ { n - 1 } \pmb { a } _ { n } } \end{array} \right] + \left[ \begin{array} { l l } { \sigma ^ { 2 } \pmb { I } _ { n - 1 } } & { \pmb { 0 } _ { n - 1 } } \\ { \pmb { 0 } _ { n - 1 } } & { \sigma ^ { 2 } } \end{array} \right] } \end{array}\tag{s18b}
$$

$$
\begin{array} { r l } & { = \left[ \tilde { \Omega } _ { n } ^ { - 1 } \qquad A _ { n - 1 } \tilde { K } _ { n - 1 } { \pmb a } _ { n } \right] . } \\ & { = \left[ \left( { \pmb A } _ { n - 1 } \tilde { K } _ { n - 1 } { \pmb a } _ { n } \right) ^ { \top } \quad { \pmb a } _ { n } ^ { \top } \tilde { K } _ { n - 1 } { \pmb a } _ { n } + \sigma ^ { 2 } \right] . } \end{array}\tag{s18c}
$$

Notice that,

$$
A _ { n - 1 } \tilde { K } _ { n - 1 } { \pmb a } _ { n } = { \pmb A } _ { n - 1 } \tilde { K } _ { n - 1 } \tilde { K } _ { n - 1 } ^ { - 1 } \tilde { \pmb k } _ { n - 1 } ( { \pmb x } _ { n } )
$$

$$
= \pmb { A } _ { n - 1 } \tilde { \pmb { k } } _ { n - 1 } ( \pmb { x } _ { n } ) , \mathrm { ~ a n d }\tag{s19a}
$$

$$
\begin{array} { r } { \pmb { a } _ { n } ^ { \top } \tilde { \pmb { K } } _ { n - 1 } \pmb { a } _ { n } = \tilde { \pmb { k } } _ { n - 1 } ( \pmb { x } _ { n } ) ^ { \top } \tilde { \pmb { K } } _ { n - 1 } ^ { - 1 } \tilde { \pmb { K } } _ { n - 1 } \pmb { a } _ { n } } \end{array}\tag{s19b}
$$

$$
\begin{array} { r l } { { \bf \Pi } } & { { } = \tilde { \pmb { k } } _ { n - 1 } ( { \pmb x } _ { n } ) ^ { \top } { \pmb a } _ { n } . } \end{array}\tag{s19c}
$$

(s19d)

Therefore,

$$
\tilde { \Omega } _ { n } ^ { - 1 } = \left[ \begin{array} { c c } { { \tilde { \Omega } _ { n - 1 } ^ { - 1 } } } & { { A _ { n - 1 } \tilde { \pmb { k } } _ { n - 1 } ( \pmb { x } _ { n } ) } } \\ { { \left( \pmb { A } _ { n - 1 } \tilde { \pmb { k } } _ { n - 1 } ( \pmb { x } _ { n } ) \right) ^ { \top } } } & { { \tilde { \pmb { k } } _ { n - 1 } ( \pmb { x } _ { n } ) ^ { \top } \pmb { a } _ { n } + \sigma ^ { 2 } } } \end{array} \right] ,\tag{s20}
$$

and applying the partitioned (symmetric and positive semi-definite) matrix inverse formula [24, Ch. 0.7] we get

$$
\tilde { \pmb { \Omega } } _ { n } = \left[ \begin{array} { c c } { \tilde { \pmb { \Omega } } _ { n - 1 } } & { { \bf 0 } _ { n - 1 } } \\ { { \bf 0 } _ { n - 1 } ^ { \top } } & { 0 } \end{array} \right] + \frac { 1 } { \tilde { s } _ { n } } \left[ \begin{array} { c c } { \tilde { \omega } _ { n } } \\ { - 1 } \end{array} \right] \left[ \tilde { \pmb { \omega } } _ { n } ^ { \top }  &  - 1 \right] ,\tag{s21}
$$

where

$$
\tilde { \pmb { \omega } } _ { n } = \tilde { \pmb { \Omega } } _ { n - 1 } \pmb { A } _ { n - 1 } \tilde { \pmb { k } } _ { n - 1 } ( { \pmb x } _ { n } ) \in \mathbb { R } ^ { n - 1 } , \ \mathrm { a n d }
$$

$$
\begin{array} { r } { \tilde { s } _ { n } = \tilde { k } _ { n - 1 } ( { \pmb x } _ { n } ) ^ { \top } { \pmb a } _ { n } + \sigma ^ { 2 } - \tilde { \omega } _ { n } ^ { \top } { \pmb A } _ { n - 1 } \tilde { \cal k } _ { n - 1 } ( { \pmb x } _ { n } ) \in \mathbb { R } . } \end{array}\tag{s22a}
$$

(s22b)

Now,

$$
\tilde { C } _ { n } = \left[ A _ { n - 1 } ^ { \top } \quad \mathbf { a } _ { n } \right] \left( \left[ \tilde { \Omega } _ { n - 1 } \quad \mathbf { 0 } _ { n - 1 } \right] + \frac { 1 } { \tilde { s } _ { n } } \left[ \tilde { \omega } _ { n } \right] \left[ \tilde { \omega } _ { n } ^ { \top } \quad - 1 \right] \right) \left[ \begin{array} { c c } { A _ { n - 1 } } \\ { a _ { n } ^ { \top } } \end{array} \right]\tag{s23a}
$$

$$
= { \cal A } _ { n - 1 } ^ { \top } \tilde { \Omega } _ { n - 1 } { \cal A } _ { n - 1 } + \frac { 1 } { \tilde { s } _ { n } } \left( { \cal A } _ { n - 1 } ^ { \top } \tilde { \omega } _ { n } - { \bf a } _ { n } \right) \left( \tilde { \omega } _ { n } ^ { \top } { \cal A } _ { n - 1 } - { \bf a } _ { n } ^ { \top } \right)\tag{s23b}
$$

$$
\mathbf { \tilde { \Sigma } } = \tilde { C } _ { n - 1 } + \frac { 1 } { \tilde { s } _ { n } } \tilde { { \pmb { c } } } _ { n } \tilde { { \pmb { c } } } _ { n } ^ { \top } ,\tag{s23c}
$$

where

$$
\tilde { \pmb { c } } _ { n } = \pmb { a } _ { n } - \pmb { A } _ { n - 1 } ^ { \top } \tilde { \pmb { w } } _ { n } \in \mathbb { R } ^ { D _ { n } } .\tag{s24}
$$

In the same way,

$$
\tilde { \alpha } _ { n } = \left[ A _ { n - 1 } ^ { \top } \quad a _ { n } \right] \left( \left[ \tilde { \Omega } _ { n - 1 } \quad \mathbf { 0 } _ { n - 1 } \right] + \frac { 1 } { \tilde { s } _ { n } } \left[ \tilde { \omega } _ { n } \right] \left[ \tilde { \omega } _ { n } ^ { \top } \quad - 1 \right] \right) \left[ \begin{array} { c } { y _ { n - 1 } } \\ { y _ { n } } \end{array} \right]\tag{s25a}
$$

$$
= { \cal A } _ { n - 1 } ^ { \top } \tilde { \Omega } _ { n - 1 } y _ { n - 1 } + \frac { 1 } { \tilde { s } _ { n } } \left( { \cal A } _ { n - 1 } ^ { \top } \tilde { \omega } _ { n } - { \pmb a } _ { n } \right) \left( \tilde { \omega } _ { n } ^ { \top } { \pmb y } _ { n - 1 } - y _ { n } \right)\tag{s25b}
$$

$$
\mathbf { \Psi } = \tilde { \alpha } _ { n - 1 } + \frac { \tilde { e } _ { n } } { \tilde { s } _ { n } } \tilde { \mathbf { c } } _ { n } ,\tag{s25c}
$$

where

$$
\tilde { e } _ { n } = y _ { n } - \tilde { \omega } _ { n } ^ { \top } y _ { n - 1 } \in \mathbb { R } .\tag{s26}
$$

Finally,

$$
\tilde { \pmb { c } } _ { n } = \pmb { a } _ { n } - \pmb { A } _ { n - 1 } ^ { \top } \tilde { \pmb { \Omega } } _ { n - 1 } \pmb { A } _ { n - 1 } \tilde { \pmb { k } } _ { n - 1 } ( \pmb { x } _ { n } )\tag{s27a}
$$

$$
\mathbf { \Psi } = \mathbf { a } _ { n } - \tilde { C } _ { n - 1 } \tilde { \mathbf { k } } _ { n - 1 } \big ( \mathbf { x } _ { n } \big ) ,\tag{s27b}
$$

$$
\tilde { e } _ { n } = y _ { n } - \tilde { k } _ { n - 1 } \big ( { \pmb x } _ { n } \big ) ^ { \top } { \pmb A } _ { n - 1 } ^ { \top } \tilde { \pmb \Omega } _ { n - 1 } { \pmb y } _ { n - 1 }\tag{s27c}
$$

$$
= y _ { n } - \tilde { k } _ { n - 1 } ( { \pmb x } _ { n } ) ^ { \top } \tilde { { \pmb \alpha } } _ { n - 1 } , \ \mathrm { a n d }\tag{s27d}
$$

$$
\tilde { s } _ { n } = \tilde { k } _ { n - 1 } ( { \pmb x } _ { n } ) ^ { \top } { \pmb a } _ { n } + \sigma ^ { 2 } - \tilde { k } _ { n - 1 } ( { \pmb x } _ { n } ) { \pmb A } _ { n - 1 } ^ { \top } \tilde { \Omega } _ { n - 1 } { \pmb A } _ { n - 1 } \tilde { k } _ { n - 1 } ( { \pmb x } _ { n } )\tag{s27e}
$$

$$
= \tilde { \pmb { k } } _ { n - 1 } ( { \pmb x } _ { n } ) ^ { \top } { \pmb a } _ { n } + \sigma ^ { 2 } - \tilde { \pmb { k } } _ { n - 1 } ( { \pmb x } _ { n } ) ^ { \top } \tilde { C } _ { n - 1 } \tilde { \pmb { k } } _ { n - 1 } ( { \pmb x } _ { n } ) .\tag{s27f}
$$

S3.2 Case $\mathcal { D } _ { n } = \mathcal { D } _ { n - 1 } \cup \{ { \pmb x } _ { n } \}$ (meaning that $\delta _ { n } > \nu )$

In this case,

$$
\begin{array} { r } { \tilde { \mathbf { K } } _ { n } = \left[ \begin{array} { c c } { \tilde { \mathbf { K } } _ { n - 1 } } & { \tilde { \mathbf { k } } _ { n - 1 } ( \pmb { x } _ { n } ) } \\ { \tilde { \mathbf { k } } _ { n - 1 } ( \pmb { x } _ { n } ) ^ { \top } } & { \kappa ( \pmb { x } _ { n } , \pmb { x } _ { n } ) } \end{array} \right] \in \mathbb { R } ^ { D _ { n } \times D _ { n } } , } \end{array}\tag{s28a}
$$

$$
\mathbf { \delta A } _ { n } = \left[ { \frac { \left[ A _ { n - 1 } \mathbf { \delta } \mathbf { 0 } _ { n - 1 } \right] } { { \pmb a } _ { n } } } \right] \in \mathbb { R } ^ { n \times D _ { n } } , \ \mathrm { a n d }\tag{s28b}
$$

$$
\begin{array} { r } { \pmb { a } _ { n } = \left[ 0 \quad \cdot \cdot \cdot \quad 0 \quad 1 \right] \in \{ 0 \} ^ { D _ { n } - 1 } \times \{ 1 \} \subset \mathbb { R } ^ { D _ { n } } . } \end{array}\tag{s28c}
$$

Then,

$$
\tilde { \Omega } _ { n } ^ { - 1 } = [ \begin{array} { c c } { [ A _ { n - 1 } } & { \mathbf { 0 } _ { n - 1 } ] } \\ { \qquad \mathbf { \Phi } _ { \mathbf { { a } } _ { n } } } \end{array} ] [ \begin{array} { c c } { \tilde { K } _ { n - 1 } } & { \tilde { k } _ { n - 1 } ( \mathbf { x } _ { n } ) } \\ { \tilde { k } _ { n - 1 } ( \mathbf { x } _ { n } ) ^ { \top } } & { \kappa ( \mathbf { x } _ { n } , \mathbf { x } _ { n } ) } \end{array} ] [ \begin{array} { c c } { [ A _ { n - 1 } ^ { \top } ] } & { } \\ { \mathbf { 0 } _ { n - 1 } ^ { \top } } \end{array} ] \quad \mathbf { { a } } _ { n } ] + [ \begin{array} { c c } { \sigma ^ { 2 } I _ { n - 1 } } & { \mathbf { 0 } _ { n - 1 } } \\ { \mathbf { 0 } _ { n - 1 } ^ { \top } } & { \sigma ^ { 2 } } \end{array} ]\tag{s29a}
$$

$$
= \left[ \begin{array} { c c } { \pmb { A } _ { n - 1 } \tilde { \pmb { K } } _ { n - 1 } } & { \pmb { A } _ { n - 1 } \tilde { \pmb { k } } _ { n - 1 } ( \pmb { x } _ { n } ) } \\ { \tilde { \pmb { k } } _ { n - 1 } ( \pmb { x } _ { n } ) ^ { \top } } & { \kappa ( \pmb { x } _ { n } , \pmb { x } _ { n } ) } \end{array} \right] \left[ \left[ \begin{array} { c c } { \pmb { A } _ { n - 1 } ^ { \top } } \\ { \pmb { 0 } _ { n - 1 } ^ { \top } } \end{array} \right]  &  \pmb { a } _ { n } \right] + \left[ \begin{array} { c c } { \sigma ^ { 2 } \pmb { I } _ { n - 1 } } & { \pmb { 0 } _ { n - 1 } } \\ { \pmb { 0 } _ { n - 1 } ^ { \top } } & { \sigma ^ { 2 } } \end{array} \right]\tag{s29b}
$$

$$
= \left[ \begin{array} { c c } { A _ { n - 1 } \tilde { K } _ { n - 1 } A _ { n - 1 } ^ { \top } + \sigma ^ { 2 } I _ { n - 1 } } & { A _ { n - 1 } \tilde { k } _ { n - 1 } ( { \pmb x } _ { n } ) } \\ { \tilde { k } _ { n - 1 } ( { \pmb x } _ { n } ) ^ { \top } { \pmb A } _ { n - 1 } ^ { \top } } & { \kappa ( { \pmb x } _ { n } , { \pmb x } _ { n } ) + \sigma ^ { 2 } } \end{array} \right]\tag{s29c}
$$

$$
\mathbf { \Sigma } = \left[ \begin{array} { c c } { \tilde { \Omega } _ { n - 1 } ^ { - 1 } } & { A _ { n - 1 } \tilde { \pmb { k } } _ { n - 1 } ( \pmb { x } _ { n } ) } \\ { \left( \mathbf { A } _ { n - 1 } \tilde { \pmb { k } } _ { n - 1 } ( \pmb { x } _ { n } ) \right) ^ { \top } } & { \kappa ( \pmb { x } _ { n } , \pmb { x } _ { n } ) + \sigma ^ { 2 } } \end{array} \right] ,\tag{s29d}
$$

and applying the partitioned (symmetric and positive semi-definite) matrix inverse formula [24, Ch. 0.7] we get

$$
\tilde { \pmb { \Omega } } _ { n } = \left[ \begin{array} { c c } { \tilde { \pmb { \Omega } } _ { n - 1 } } & { { \bf 0 } _ { n - 1 } } \\ { { \bf 0 } _ { n - 1 } ^ { \top } } & { 0 } \end{array} \right] + \frac { 1 } { \tilde { s } _ { n } } \left[ \begin{array} { c c } { \tilde { \omega } _ { n } } \\ { - 1 } \end{array} \right] \left[ \tilde { \pmb { \omega } } _ { n } ^ { \top }  &  - 1 \right] ,\tag{s30}
$$

where

$$
\tilde { \pmb { \omega } } _ { n } = \tilde { \pmb { \Omega } } _ { n - 1 } \pmb { A } _ { n - 1 } \tilde { \pmb { k } } _ { n - 1 } ( { \pmb x } _ { n } ) \in \mathbb { R } ^ { n - 1 } , \ \mathrm { a n d }\tag{s31a}
$$

$$
\begin{array} { r } { \tilde { s } _ { n } = \kappa ( \pmb { x } _ { n } , \pmb { x } _ { n } ) + \sigma ^ { 2 } - \tilde { \omega } _ { n } ^ { \top } \pmb { A } _ { n - 1 } \tilde { \pmb { k } } _ { n - 1 } ( \pmb { x } _ { n } ) \in \mathbb { R } . } \end{array}\tag{s31b}
$$

From here, we are able to expand

$$
\begin{array} { r l } { \tilde { C } _ { n } = \left[ \left[ \begin{array} { l } { \pmb { A } _ { n - 1 } ^ { \top } } \\ { \mathbf { 0 } _ { n - 1 } ^ { \top } } \end{array} \right] } & { { } \mathbf { { \alpha } } \mathbf { a } _ { n } \right] \left( \left[ \begin{array} { l l } { \tilde { \Omega } _ { n - 1 } } & { \mathbf { 0 } _ { n - 1 } } \\ { \mathbf { 0 } _ { n - 1 } ^ { \top } } & { 0 } \end{array} \right] + \frac { 1 } { \tilde { s } _ { n } } \left[ \begin{array} { l l } { \tilde { \omega } _ { n } } \\ { - 1 } \end{array} \right] \left[ \tilde { \omega } _ { n } ^ { \top } } & { - 1 \right] \right) \left[ \begin{array} { l l } { \left[ \pmb { A } _ { n - 1 } } & { \mathbf { 0 } _ { n - 1 } \right] } \\ { \qquad \mathbf { { \alpha } } \mathbf { a } _ { n } } \end{array} \right] } \end{array}\tag{s32a}
$$

$$
= \left[ \mathbf { A } _ { n - 1 } ^ { \top } \right] \tilde { \Omega } _ { n - 1 } \left[ A _ { n - 1 } \quad \mathbf { 0 } _ { n - 1 } \right] + \frac { 1 } { \tilde { s } _ { n } } \left( \left[ \mathbf { A } _ { n - 1 } ^ { \top } \right] \tilde { \omega } _ { n } - a _ { n } \right) \left( \tilde { \omega } _ { n } ^ { \top } \left[ A _ { n - 1 } \quad \mathbf { 0 } _ { n - 1 } \right] - a _ { n } ^ { \top } \right)\tag{s32b}
$$

$$
\mathbf { \Sigma } = \left[ \begin{array} { c c } { \tilde { C } _ { n - 1 } } & { \mathbf { 0 } _ { D _ { n } - 1 } } \\ { \mathbf { 0 } _ { D _ { n } - 1 } ^ { \top } } & { 0 } \end{array} \right] + \frac { 1 } { \tilde { s } _ { n } } \tilde { c } _ { n } \tilde { c } _ { n } ^ { \top }\tag{s32c}
$$

where

$$
\tilde { \pmb { c } } _ { n } = \pmb { a } _ { n } - \left[ \pmb { A } _ { n - 1 } ^ { \top } \right] \tilde { \pmb { w } } _ { n } \in \mathbb { R } ^ { D _ { n } } .\tag{s33}
$$

Similarly,

$$
\tilde { \alpha } _ { n } = [ \begin{array} { c c } { [ \mathbf { A } _ { n - 1 } ^ { \top } ] } & { } \\ { \mathbf { 0 } _ { n - 1 } ^ { \top } } \end{array} ] ~ \mathbf { \alpha } _ { \boldsymbol { a } _ { n } } ] ( [ \begin{array} { c c } { \tilde { \Omega } _ { n - 1 } } & { \mathbf { 0 } _ { n - 1 } } \\ { \mathbf { 0 } _ { n - 1 } ^ { \top } } & { 0 } \end{array} ] + \frac { 1 } { \tilde { s } _ { n } } [ \begin{array} { c c } { \tilde { \omega } _ { n } } \\ { - 1 } \end{array} ] ~ [ \begin{array} { c c } { \tilde { \omega } _ { n } ^ { \top } } & { - 1 } \end{array} ] ) [ \begin{array} { c } { y _ { n - 1 } } \\ { y _ { n } } \end{array} ]\tag{s34a}
$$

$$
= \left[ \mathbf { 0 } _ { n - 1 } ^ { \mathsf { T } } \right] \tilde { \Omega } _ { n - 1 } y _ { n - 1 } + \frac { 1 } { \tilde { s } _ { n } } \left( \left[ \mathbf { 0 } _ { n - 1 } ^ { \mathsf { T } } \right] \tilde { \omega } _ { n } - \pmb { a } _ { n } \right) \left( \tilde { \omega } _ { n } ^ { \top } \pmb { y } _ { n - 1 } - y _ { n } \right)\tag{s34b}
$$

$$
\mathbf { \Sigma } = \left[ \tilde { \alpha } _ { n - 1 } \right] + \frac { \tilde { e } _ { n } } { \tilde { s } _ { n } } \tilde { \mathbf { c } } _ { n } ,\tag{s34c}
$$

where

$$
\tilde { e } _ { n } = y _ { n } - \tilde { \omega } _ { n } ^ { \top } y _ { n - 1 } \in \mathbb { R } .\tag{s35}
$$

Finally,

$$
\tilde { \pmb { c } } _ { n } = \pmb { a } _ { n } - \left[ \begin{array} { c } { \pmb { A } _ { n - 1 } ^ { \top } \tilde { \pmb { \Omega } } _ { n - 1 } \pmb { A } _ { n - 1 } \tilde { \pmb { k } } _ { n - 1 } \big ( \pmb { x } _ { n } \big ) } \\ { 0 } \end{array} \right]\tag{s36a}
$$

$$
\mathbf { \Sigma } = \left[ \begin{array} { c } { - \tilde { C } _ { n - 1 } \tilde { \pmb { k } } _ { n - 1 } ( \pmb { x } _ { n } ) } \\ { 1 } \end{array} \right] ,\tag{s36b}
$$

$$
\tilde { e } _ { n } = y _ { n } - \tilde { k } _ { n - 1 } \big ( { \pmb x } _ { n } \big ) ^ { \top } { \pmb A } _ { n - 1 } ^ { \top } \tilde { \pmb \Omega } _ { n - 1 } { \pmb y } _ { n - 1 }\tag{s36c}
$$

$$
= y _ { n } - { \tilde { k } } _ { n - 1 } ( \mathbf x _ { n } ) ^ { \top } { \tilde { \alpha } } _ { n - 1 } , ~ \mathrm { a n d }\tag{s36d}
$$

$$
\tilde { s } _ { n } = \kappa ( { \pmb x } _ { n } , { \pmb x } _ { n } ) + \sigma ^ { 2 } - \tilde { { \pmb k } } _ { n - 1 } ( { \pmb x } _ { n } ) ^ { \top } { \pmb A } _ { n - 1 } ^ { \top } \tilde { \pmb \Omega } _ { n - 1 } { \pmb A } _ { n - 1 } \tilde { \pmb k } _ { n - 1 } ( { \pmb x } _ { n } )\tag{s36e}
$$

$$
= \kappa ( { \pmb x } _ { n } , { \pmb x } _ { n } ) + \sigma ^ { 2 } - \tilde { { \pmb k } } _ { n - 1 } ( { \pmb x } _ { n } ) ^ { \top } \tilde { C } _ { n - 1 } \tilde { { \pmb k } } _ { n - 1 } ( { \pmb x } _ { n } ) .\tag{s36f}
$$

It is worth noting that by applying the partitioned matrix inverse formula [24, Ch. 0.7] on (s28a) one can get

$$
\tilde { \bf K } _ { n - 1 } ^ { - 1 } = \frac { 1 } { \delta _ { n } } \left[ \begin{array} { c c } { \delta _ { n } \tilde { \cal K } _ { n - 1 } ^ { - 1 } + \hat { a } _ { n } \hat { a } _ { n } ^ { \top } } & { - \hat { a } _ { n } } \\ { - \hat { a } _ { n } ^ { \top } } & { 1 } \end{array} \right] ,\tag{s37}
$$

with $\delta _ { n }$ and $\hat { \pmb { a } } _ { r }$ <sub>n</sub> as in (s12).

## S3.3 Comments on the computational and memory complexity

The per-step computational and memory complexity of Algorithm 1 is $\mathcal { O } ( D _ { n } ^ { 2 } + r D _ { n } )$ , where $D _ { n }$ is the ALD sparsification dictionary size, and r is the number of warping parameters.

The following operations dominate this complexity:

• Evaluating the kernel vector $\tilde { \pmb { k } } _ { n - 1 } ( { \pmb x } _ { n } ) \mathrm { i s } \mathcal { O } ( D _ { n } )$ , since it evaluates the kernel between ${ \pmb x } _ { n }$ and each of the $D _ { n }$ dictionary elements.

• Computing the ALD projection coefficients $\hat { \pmb { a } } _ { n } = \tilde { \pmb { K } } _ { n - 1 } ^ { - 1 } \tilde { \pmb { k } } _ { n - 1 } \big ( { \pmb x } _ { n } \big ) \mathrm { i s } \mathcal { O } ( D _ { n } ^ { 2 } )$ since $\tilde { K } _ { n } ^ { - 1 }$ is updated via a rank-1 block inversion.

• Evaluating the warping transformation used in Sec. $5 g ( y _ { n } ; \pmb { \theta } )$ (and its gradient) is $\mathcal { O } ( r )$ due to basic arithmetic and tanh operations.

• Computing the matrix-vector products $\tilde { B } _ { n - 1 } \tilde { k } _ { n - 1 } \big ( \pmb { x } _ { n } \big )$ and $\tilde { C } _ { n - 1 } \tilde { k } _ { n - 1 } ( \pmb { x } _ { n } ) \mathrm { i s } \mathcal { O } ( r D _ { n } )$ and $\mathcal { O } ( D _ { n } ^ { 2 } )$ , respectively, due to the matrix and vector dimensions.

• Finally, updating $\tilde { C } _ { n }$ and $\tilde { B } _ { n }$ is $\mathcal { O } ( D _ { n } ^ { 2 } )$ and $\mathcal { O } ( r D _ { n } )$ , respectively, due to the rank-1 outer products $\tilde { \pmb { c } } _ { n } \tilde { \pmb { c } } _ { n } ^ { \top }$ and $\tilde { \pmb { b } } _ { n } \tilde { \pmb { c } } _ { n } ^ { \top }$ in their update rules.