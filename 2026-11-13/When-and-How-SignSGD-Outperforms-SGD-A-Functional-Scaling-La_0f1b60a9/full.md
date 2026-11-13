# When and How SignSGD Outperforms SGD: A Functional Scaling Law Perspective

Zilin Wang1,\* Binghui Li1,\* Jia-Nan Wang1,\* Lean Wang1,\* Jinbo Wang1,\* Lei Wu1,2,† 1Peking University 2AI for Science Institute, Beijing

(Accepted at NeurIPS 2026)

## Abstract

We study how the sign operator interacts with gradient noise to shape the learning dynamics and scaling behavior of sign-based optimizers. In power-law kernel regression with diagonal features, we establish a functional scaling law (FSL) for SignSGD under general learning rate schedules. The FSL describes the learning dynamics through two processes: signal learning and noise forgetting. The key mechanism is noise-induced preconditioning: gradient noise smooths the sign operator, inducing an effective preconditioned SGD. Noise-curvature alignment enables this preconditioning to transform the effective Hessian from H to H1/2. The resulting dynamics exhibit faster signal learning but slower noise forgetting than SGD. This tradeoff yields a regime-dependent comparison with SGD: (i) SignSGD achieves improved data efficiency in the hard-task regime; (ii) both optimizers achieve the same data efficiency in the easy-task regime, but SignSGD maintains this efficiency at larger batch sizes, requiring fewer optimization steps for a fixed data budget. Power-law kernel regression experiments support the proposed mechanism and predicted scaling behavior. LLM pre-training experiments further suggest that the SignSGD FSL better captures the batch-size dependence of loss dynamics for sign-based optimizers than the original SGD FSL.

## 1 Introduction

Sign-based optimizers play an increasingly prominent role in modern large-scale training. SignSGD (Bernstein et al., 2018) and Lion (Chen et al., 2023) apply the sign operator coordinate-wise, while Muon (Jordan et al., 2024; Liu et al., 2025) uses the matrix sign operator. Related sign-like behavior also arises in RMSprop (Hinton et al., 2012) and Adam (Kingma & Ba, 2015; Loshchilov & Hutter, 2019) through coordinate-wise second-moment normalization (Balles & Hennig, 2018). Together, these methods motivate understanding the sign operator as a basic mechanism in modern optimization.

Existing theoretical analyses of these sign-based optimizers have largely focused on deterministic full-batch dynamics (Kunstner et al., 2023, 2024; Zhang et al., 2024; Kunstner & Bach, 2025; Li et al., 2026b) or convergence guarantees for convex and nonconvex objectives under general smoothness assumptions (Reddi et al., 2018; Chen et al., 2019; Défossez et al., 2022; Sun et al., 2023; Shen et al., 2025). These analyses establish important optimization properties, but do not fully explain how the sign operator interacts with gradient noise or adequately quantify how this interaction shapes loss dynamics and scaling behavior.

![](images/c6e6386b28e25bc287b8dcfc3fc7bd2020cb80d6b45de7d2a5537dc4c36e6db9.jpg)

![](images/58e27c77e98e16d6e35a7baa983b1924f2d498e283129f9c821f60d3d2e2401c.jpg)

![](images/9a6a1e88493bd807768d4158330015583d242d22539521da9a271903d02ab922.jpg)  
Figure 1: SignSGD FSL accurately captures risk dynamics and predicts scaling behavior. (Left) FSL predictions track SignSGD risk trajectories under cosine, WSD, and cyclic LR schedules. (Middle) In the hard-task regime, SignSGD achieves a larger data-scaling exponent than SGD, $\gamma _ { \mathrm { s g n } } > \gamma _ { \mathrm { s g d } }$ in $\mathcal E ( D ) \approx D ^ { - \gamma }$ , indicating improved data efficiency. Dashed lines denote power-law fits. (Right) In the easy-task regime, SignSGD improves step efficiency through a larger CBS, requiring fewer optimization steps at a fixed data budget while preserving small-batch performance. Risk is normalized by each optimizer's $B = 1$ baseline. Marked CBS estimates use a 20% risk-increase tolerance.

In this work, we study SignSGD in power-law kernel (PLK) regression with diagonal features. We extend the functional scaling law (FSL) framework of Li et al. (2025) to characterize how sign operator reshapes signal learning and noise forgetting under general learning rate (LR) schedules. Our analysis shows that SignSGD acts as an effective preconditioned SGD method, with the preconditioner determined by the structure of gradient noise. Diagonal features and noise-curvature alignment are important in this square-root spectral preconditioning, accelerating signal learning in low-curvature directions. The accompanying change in update noise, however, leads to slower noise forgetting. The FSL captures these effects by expressing the risk at step k as

$$
\mathbb { E } [ \mathcal { E } ( \pmb { \mathscr { E } } ( \pmb { \theta } _ { k } ) ] \approx S ( t _ { k } ) + \sum _ { j = 0 } ^ { k - 1 } K ( t _ { k } - t _ { j } ) \eta _ { j } ^ { 2 } .
$$

Here $\eta _ { j }$ is the learning rate at step $j ,$ and $\begin{array} { r } { t _ { k } = \sqrt { B } \sum _ { j = 0 } ^ { k - 1 } \eta _ { j } } \end{array}$ is the intrinsic time of SignSGD with batch size B. Signal-learning function $S ( t _ { k } )$ describes how the target signal is learned, and noise-forgetting kernel $K ( t _ { k } - t _ { j } )$ measures the remaining effect of noise injected at step $j .$ $S ( \cdot )$ and $K ( \cdot )$ follow the power laws: $S ( t ) \asymp t ^ { - 2 s } , K ( t ) \asymp t ^ { - ( 2 - 2 / q ) }$ , where $s > 0$ and $q > 1$ are task-dependent parameters. Figure 1 (left) shows that this FSL captures the risk dynamics of SignSGD. Leveraging this FSL framework, we furthermore identify two advantages of SignSGD over SGD:

• Improved data efficiency (Section 4.1). In the hard-task regime, SignSGD achieves a faster data-scaling rate than SGD, as illustrated in Figure 1 (middle).

• Improved step efficiency (Section 4.2). In the easy-task regime, SignSGD supports a larger critical batch size (CBS) than SGD while preserving optimal data efficiency, requiring fewer optimization steps for a fixed data budget, as illustrated in Figure 1 (right).

Empirical validation. PLK regression experiments (Section 5.1) corroborate this mechanism and the resulting scaling predictions. In large language model (LLM) pre-training (Section 5.2), the SignSGD FSL captures the dependence of sign-based optimizers' loss trajectories on batch size more accurately than the original SGD FSL in Li et al. (2025).

Related work. Xiao et al. (2025) and Monzio Compagnoni et al. (2025) develop SDE descriptions of signSGD and analyze how the sign operator modifies the drift and noise, revealing effects on preconditioning, risk dynamics, and convergence. More directly related to scaling laws, Kim et al. (2026) study signSGD in PLK regression, primarily in the noiseless setting. Our work focuses on the label-noise-dominated regime, establishing an explicit FSL that characterizes signal learning and noise forgetting under general LR schedules, which makes the batch-size dependence explicit and enables comparisons of data and step efficiency between signSGD and SGD. Paquette et al. (2026) study Muon and SignSGD in matrix-valued least squares and show that these sign-based optimizers can provide square-root spectral preconditioning effects. Additional related work is discussed in Appendix A.

## 2 Preliminaries

Notation. We use ≈ to denote equivalence up to a constant factor, and $\lesssim ( \mathrm { r e s p . ~ } \gtrsim )$ to indicate inequality up to a constant factor. For two non-negative functions $f , g ,$ we write $f ( x ) \lesssim g ( x )$ (resp. $f ( x ) \gtrsim g ( x ) )$ if there exists a constant $C > 0$ such that $f ( x ) \leqslant C g ( x )$ (resp. $g ( x ) \leqslant C f ( x ) )$ for all x under consideration. We write $f ( x ) \stackrel { } { \sim } g ( x )$ if both $f ( x ) \lesssim g ( x )$ and $f ( x ) \gtrsim g ( x )$ hold.

## 2.1 Problem setup

Power-law kernel (PLK) regression with diagonalized features. Consider a kernel regression task in which labels are generated by $y = \langle \phi ( \mathbf { x } ) , \pmb \theta ^ { * } \rangle + \epsilon .$ where data input $\mathbf { x } \sim \mathcal { D }$ and label noise $\epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ are independent. Here $\phi : \mathcal { X }  \mathbb { R } ^ { d }$ is the feature mapping and $\pmb { \theta } ^ { * } \in \mathbb { R } ^ { d }$ is the unknown target parameter. We consider the infinite-dimensional setting, i.e., $d = \infty$ We assume Gaussian features $\phi ( \mathbf { x } ) \sim \mathcal { N } ( 0 , \mathbf { H } )$ , where $\mathbf { H } = \mathbb { E } _ { \mathbf { x } \sim \mathcal { D } } [ \phi ( \mathbf { x } ) \phi ( \mathbf { x } ) ^ { \top } ]$ is the feature covariance. The prediction model $f ( \mathbf { x } ; \pmb \theta ) = \langle \phi ( \mathbf { x } ) , \pmb \theta \rangle$ is trained by minimizing the population risk $\mathcal { R } ( \pmb { \theta } ) = \mathbb { E } _ { \mathbf { x } , y } [ \ell ( \pmb { \theta } ; \mathbf { x } , y ) ]$ under the squared loss $\ell ( \pmb \theta ; \mathbf x , y ) = { \frac { 1 } { \sqrt { \pi } } } ( f ( \mathbf { x } ; \pmb \theta ) - y ) ^ { 2 }$ . The population risk decomposes as $\mathcal { R } ( \pmb { \theta } ) = \mathcal { E } ( \pmb { \theta } ) + \sigma ^ { 2 } / 2$ , where $\begin{array} { r } { \mathcal { E } ( \pmb { \theta } ) = \frac { 1 } { 2 } ( \pmb { \theta } - \pmb { \theta } ^ { * } ) ^ { \top } \mathbf { H } ( \pmb { \theta } - \pmb { \theta } ^ { * } ) } \end{array}$ is the excess risk.

Assumption 2.1 (Constant label noise). Assume $\sigma \stackrel { } { \sim } 1$

Assumption 2.2 (Diagonal features). Assume $\mathbf { H } = \mathrm { d i a g } ( \lambda _ { 1 } , \cdot \cdot \cdot , \lambda _ { d } )$

Assumption 2.3 (Power-law structures). For some $q > 1$ and $s > 0 , \ \lambda _ { j } \ \approx \ j ^ { - q } , \lambda _ { j } | \theta _ { j } ^ { * } | ^ { 2 } \ \stackrel { \_ } { \sim } \ $ $j ^ { - ( 1 + s q ) }$

Assumption 2.3 specifies the power-law source and capacity structure of the regression. Here q is the spectral-decay exponent, which controls the decay rate of the feature spectrum, and s is the source exponent, which describes how the target energy is distributed across spectral directions.

One-pass SignSGD. At each training step $k = 0 , \ldots , N - 1$ , we draw a fresh mini-batch $\mathcal { B } _ { k } = \{ ( \mathbf { x } _ { k , i } , y _ { k , i } ) \} _ { i = 1 } ^ { B }$ of size B, with all samples drawn i.i.d. from the data distribution. Starting from $\pmb \theta _ { 0 } = \mathbf 0$ , one-pass SignSGD performs the updates

$$
\mathbf { g } _ { k } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \nabla _ { \pmb { \theta } } \ell ( \pmb { \theta } _ { k } ; \mathbf { x } _ { k , i } , y _ { k , i } ) , \qquad \pmb { \theta } _ { k + 1 } = \pmb { \theta } _ { k } - \eta _ { k } \operatorname { s g n } ( \mathbf { g } _ { k } ) ,\tag{1}
$$

where $\operatorname { s g n } ( { \mathord { \cdot } } )$ acts coordinate-wise and $\{ \eta _ { k } \} _ { k = 0 } ^ { N - 1 }$ is the LR schedule.

## 2.2 An SDE modeling for SignSGD

We extend the intrinsic-time SDE developed for SGD by Li et al. (2025) to SignSGD.

Modeling gradient noise. Write $\mathbf { g } _ { k } = \pmb { \mu } _ { k } + \pmb { \xi } _ { k }$ where $\pmb { \mu } _ { k } : = \nabla \mathcal { R } ( \pmb { \theta } _ { k } )$ is the population gradient and ${ \boldsymbol { \xi } } _ { k }$ is the gradient noise satisfying $\mathbb { E } [ \pmb { \xi } _ { k } ] = \mathbf { 0 }$ . We model $\pmb { \xi } _ { k } \sim \mathcal { N } ( \mathbf { 0 } , \sigma _ { B } ^ { 2 } \pmb { \Sigma } )$ , where Σ is a single-sample covariance and $\sigma _ { B } = 1 / \sqrt { B }$ characterizes the dependence on batch size. An extension beyond Gaussian noise is discussed in Appendix C.5.

Smoothed sign and update noise. For batch size B, define the scaled smoothed sign operator

$$
\mathbf { m } _ { B } ( \mathbf { x } ) : = \sigma _ { B } \mathbb { E } _ { \pmb { \xi } \sim \mathcal { N } ( \mathbf { 0 } , \pmb { \Sigma } ) } [ \mathrm { s g n } ( \mathbf { x } + \sigma _ { B } \pmb { \xi } ) ]
$$

Since $\operatorname { s g n } ( a \mathbf { x } ) = \operatorname { s g n } ( \mathbf { x } )$ for $a > 0$ , the Gaussian noise model gives $\mathbb { E } [ \mathrm { s g n } ( \mathbf { g } _ { k } ) ] = \mathbf { m } _ { B } ( \pmb { \mu } _ { k } ) / \sigma _ { B }$ Define the centered sign noise ${ \bf n } _ { k } : = \mathrm { s g n } ( { \bf g } _ { k } ) - { \bf m } _ { B } ( \pmb { \mu } _ { k } ) / \sigma _ { B }$ and its covariance $\mathbf { Q } _ { B } ( \pmb \theta _ { k } ) : =$ $\mathbb { E } [ \mathbf { n } _ { k } \mathbf { n } _ { k } ^ { \top } ]$ . The update then becomes $\pmb { \theta } _ { k + 1 } - \pmb { \theta } _ { k } = - \eta _ { k } \big [ \mathbf { m } _ { B } ( \pmb { \mu } _ { k } ) / \sigma _ { B } + \mathbf { n } _ { k } \big ]$ . Its conditional mean and covariance are $- \eta _ { k } \mathbf { m } _ { B } ( { \pmb \mu } _ { k } ) / \sigma _ { B }$ and $\eta _ { k } ^ { 2 } \mathbf { Q } _ { B } ( \theta _ { k } )$ , respectively, which specify the drift and diffusion covariance of the effective continuum model.

Intrinsic-time dynamics. To analyze the effects of the LR schedule and batch size, we introduce two key quantities: intrinsic time and noise intensity. Let z denote continuous training steps and $\eta ( z )$ a continuous LR schedule satisfying $\eta ( k ) = \eta _ { k }$ . We define the intrinsic time of SignSGD as $\begin{array} { r } { t = \Gamma _ { B } ( z ) : = \int _ { 0 } ^ { z } \frac { \eta ( u ) } { \sigma _ { B } } } \end{array}$ du, which is the continuous analogue of $\begin{array} { r } { \sigma _ { B } ^ { - 1 } \sum _ { j = 0 } ^ { k - 1 } \eta _ { j } } \end{array}$ The corresponding noise intensity is defined by $n _ { B } ( t ) : = \sigma _ { B } \eta ( \Gamma _ { B } ^ { - 1 } ( t ) )$ , which measures the noise strength injected per unit intrinsic time. With these two quantities, we show that SignSGD can be modeled by the SDE

$$
\mathrm { d } \pmb { \vartheta } _ { t } = - \mathbf { m } _ { B } \big ( \nabla \mathcal { R } ( \pmb { \vartheta } _ { t } ) \big ) \mathrm { d } t + \sqrt { n _ { B } ( t ) } \mathbf { Q } _ { B } ^ { 1 / 2 } \big ( \pmb { \vartheta } _ { t } \big ) \mathrm { d } \mathbf { W } _ { t } ,\tag{2}
$$

where $\vartheta _ { t }$ denotes the continuum parameter process with $\vartheta _ { 0 } = \mathbf { 0 } .$ and $\mathbf { W } _ { t }$ is a standard Brownian motion. The function $n _ { B } ( t )$ controls the noise injection per unit intrinsic time, with instantaneous diffusion covariance $n _ { B } ( t ) \mathbf { Q } _ { B } ( \pmb { \vartheta } _ { t } )$ . Monzio Compagnoni et al. (2025) derive a related SDE for constant learning rates. Our formulation incorporates a learning rate schedule and the intrinsic-time transformation $t = \Gamma _ { B } ( z )$ . Appendix C.3 provides the derivation and the explicit coordinate-wise dynamics for the PLK regression model.

Remark 2.4 (Comparison with SGD). The intrinsic time of SGD is $\begin{array} { r } { t _ { \mathrm { s g d } } ( z ) = \int _ { 0 } ^ { z } \eta ( u ) } \end{array}$ du (Li et al., 2025, 2026c). For SignSGD, we include an additional factor $1 / \sigma _ { B }$ to account for the batch-size dependence of its effective drift. Section 6 explains the mechanism underlying this choice.

## Assumption 2.5 (Curvature-aligned noise). Assume $\Sigma  \mathbf { H }$

Assumption 2.5 states that the noise variance along each direction scales with the corresponding curvature, $\mathrm { i . e . , } \mathbf { v } ^ { \top } \pmb { \Sigma } \mathbf { v } \mp \mathbf { v } ^ { \top }$ Hv for all v, with comparison constants independent of v. Such noise-curvature alignment has been studied theoretically and observed empirically in deep learning (Wang & Wu, 2026). In our PLK regression setting, the exact single-sample gradient-noise covariance at $\pmb { \theta } _ { k }$ is comparable to $( \mathcal { E } ( \pmb { \theta } _ { k } ) + \sigma ^ { 2 } ) \mathbf { H }$ (see Appendix C.1). In the label-noise-dominated regime $\mathcal { E } ( \pmb \theta _ { k } ) \lesssim \sigma ^ { 2 } \approx 1$ , this covariance is therefore comparable to H, motivating the assumption in our Gaussian noise model.

## 3 Functional scaling law for SignSGD

Theorem 3.1 (FSL for SignSGD). Suppose Assumptions ${ \it 2 . 1 , } \ { \it 2 . 2 , } \ { \it 2 . 3 , }$ and 2.5 hold. Consider the SDE (2) with $\begin{array} { r } { \sqrt { B } \operatorname* { s u p } _ { z } \eta ( z ) \lesssim 1 } \end{array}$ . For $t \gtrsim B ^ { \operatorname* { m a x } \{ 1 / 2 , 1 / ( 2 s + 2 / q ) \} }$ , it holds that

$$
\mathbb { E } [ \mathcal { E } ( \pmb { \vartheta } _ { t } ) ] \approx S ( t ) + \int _ { 0 } ^ { t } K ( t - \tau ) n _ { B } ( \tau ) ~ \mathrm { d } \tau ,\tag{3}
$$

where

$$
S ( t ) \asymp t ^ { - 2 s } , \qquad K ( t ) \asymp t ^ { - ( 2 - 2 / q ) } .\tag{4}
$$

Here the hidden constants are independent of B and the LR schedule.

The proof is provided in Appendix D. Section 6 explains the underlying mechanism and derives the FSL from the effective preconditioned dynamics. The FSL separates the remaining signal error $S ( t )$ from the accumulated effect of update noise. Here $n _ { B } ( \tau )$ is the noise injection intensity per unit intrinsic time at time $\tau _ { \mathrm { { ; } } }$ while $K ( t - \tau )$ measures the residual risk from that injection after elapsed time $t - \tau$ . The LR schedule enters through both the intrinsic time and $n _ { B }$ , allowing the FSL to describe the risk trajectory under different schedules.

Li et al. (2025) establish an SGD FSL with the same convolution form as (3). For SGD, the intrinsic time, noise intensity and FSL components are

$$
t = \Gamma _ { \mathrm { s g d } } ( z ) : = \int _ { 0 } ^ { z } \eta ( u ) ~ \mathrm { d } u , \quad n _ { B } ( t ) = \frac { \eta ( \Gamma _ { \mathrm { s g d } } ^ { - 1 } ( t ) ) } { B } , \quad S ( t ) \asymp t ^ { - s } , \quad K ( t ) \asymp t ^ { - ( 2 - 1 / q ) } .
$$

Signal learning and noise forgetting. In their respective intrinsic times, SignSGD has a faster signal decay, $t ^ { - 2 s }$ compared with $t ^ { - s }$ for SGD, but a slower forgetting kernel, $t ^ { - ( 2 - 2 / q ) }$ compared with $t ^ { - ( 2 - 1 / q ) }$ . Thus, SignSGD learns the target signal faster while retaining the effects of past noise for longer.

The effect of batch size. For the same training steps and LR schedule, the intrinsic times satisfy $t _ { \mathrm { s g n } } = \sqrt { B } t _ { \mathrm { s g d } }$ . Unlike SGD, whose intrinsic time is independent of B, SignSGD has an essentially longer training time as B increases. A larger batch also allows more forgetting of earlier noise injections. Moreover, the noise intensity scales as $1 / \sqrt { B }$ for SignSGD and $1 / B$ for SGD.
<table><tr><td>Optimizer</td><td></td><td>Intrinsic time Noise intensity S(t)</td><td></td><td> $K ( t )$ </td></tr><tr><td>SGD (Li et al., 2025)</td><td> $\textstyle \int _ { 0 } ^ { z } \eta ( \tau ) \ \mathrm { d } \tau$ </td><td> $1 / B$ </td><td> $t ^ { - s }$ </td><td> $t ^ { - ( 2 - 1 / q ) }$ </td></tr><tr><td>SignSGD (ours)</td><td> $\begin{array} { r } { \sqrt { B } \int _ { 0 } ^ { z } \eta ( \tau ) \mathrm { d } \tau } \end{array}$ </td><td> $1 / \sqrt { B }$ </td><td> $t ^ { - 2 s }$ </td><td> $t ^ { - ( 2 - 2 / q ) }$ </td></tr></table>

## 4 When and how does SignSGD outperform SGD?

The FSL enables the comparison of scaling efficiency between SignSGD and SGD. For a data budget $D = N B ,$ we characterize the achievable excess risk and determine how increasing the batch size B can reduce the number of training steps N without sacrificing data efficiency. Following Li et al. (2026c), we consider the power-decay LR schedule

$$
\eta ( z ) = \eta _ { \mathrm { p e a k } } \left( 1 - \frac { z } { N } \right) ^ { p } , \qquad 0 \leqslant z \leqslant N ,\tag{5}
$$

where $\eta _ { \mathrm { p e a k } } > 0$ and $p > 0$ denote the peak learning rate and decay exponent, respectively. This schedule family attains the optimal data-scaling rates characterized below.

## 4.1 Data efficiency

We first ask: what is the optimal final-step excess risk that SignSGD can achieve by tuning the LR schedule at a fixed data budget D? We consider $B \ = \ 1$ , with the case of $B > 1$ deferred to Section 4.2. For the power-decay schedule (5), the final expected excess risk is $\mathcal { E } _ { D } : = \mathbb { E } [ \mathcal { E } ( \pmb { \vartheta } _ { \Gamma _ { 1 } ( D ) } ) ]$ , where $\begin{array} { r } { \Gamma _ { 1 } ( D ) = \int _ { 0 } ^ { D } \eta ( \tau ) } \end{array}$ dτ. The following theorem specifies choices of $\eta _ { \mathrm { p e a k } }$ and $p$ and the resulting data-scaling rates.

Theorem 4.1 (Data scaling of SignSGD). Under the assumptions of Theorem 3.1, let $B = 1$ and use the power-decay schedule (5). Then

$\begin{array} { r } { \mathrm { ~ I f ~ } s \geqslant \frac { 1 } { 2 } - \frac { 1 } { q } } \end{array}$ , then choosing $p > { \frac { q } { 2 } } - 1$ and $\eta _ { \mathrm { p e a k } } \approx D ^ { - \frac { 2 s - 1 + 2 / q } { 2 s + 2 / q } } \ \mathrm { ~ J ~ }$ vields $\mathcal { E } _ { D } \approx D ^ { - \frac { s q } { s q + 1 } }$

• If $\textstyle 0 < s < { \frac { 1 } { 2 } } - { \frac { 1 } { q } }$ , then choosing $\begin{array} { r } { p > \frac { 2 s } { 1 - 2 s } } \end{array}$ and $\eta _ { \mathrm { p e a k } } \approx 1$ yields $\mathcal { E } _ { D } \approx D ^ { - 2 s }$

Its proof is given in Appendix E.2. We show that the rates achieved in Theorem 4.1 are optimal among all admissible LR schedules (see details in Appendix E.4). Thus, the optimal data-scaling rate of SignSGD can be written as $\hat { D } ^ { - \operatorname* { m i n } \left\{ 2 s , s q / ( s q + 1 ) \right\} }$ . For comparison, the optimal SGD rate under its FSL is $D ^ { - \operatorname* { m i n } \{ s , s q / ( s q + 1 ) \} }$ (Li et al., 2026c) (Appendix E.3). Thus, these data-scaling rates partition the $( s , q )$ space into three regimes: easy, hard, and super hard. The phase diagram on the right summarizes these regimes, with boundaries at $s = 1 \mathrm { - } 1 / q$ and $s = 1 / 2 \mathrm { - } 1 / q$ The corresponding SGD and SignSGD rates are $D ^ { - }  { \gamma _ { \mathrm { s g d } } }$ and $D ^ { - }$ , where

![](images/2eb4c40828af0bcb4971eb053da99c419974d8ce564b73022025aa3f03b60b16.jpg)

$$
( \gamma _ { \mathrm { s g d } } , \gamma _ { \mathrm { s g n } } ) = \left\{ \begin{array} { l l } { ( \frac { s q } { s q + 1 } , \frac { s q } { s q + 1 } ) , } & { s \geqslant 1 - \frac { 1 } { q } , } \\ { ( s , \frac { s q } { s q + 1 } ) , } & { \frac { 1 } { 2 } - \frac { 1 } { q } \leqslant s < 1 - \frac { 1 } { q } , } \\ { ( s , 2 s ) , } & { 0 < s < \frac { 1 } { 2 } - \frac { 1 } { q } . } \end{array} \right.
$$

Remark 4.2 (Connection to statistical lower bounds.). Classical results give the minimax rate $D ^ { - s q / ( s q + 1 ) }$ for the regression problem with the same spectral-decay and source conditions (Caponnetto & De Vito, 2007; Steinwart et al., 2009). This rate is established in the discrete setting and it matches the optimal rate in Theorem 4.1 under our continuous SDE model when $\begin{array} { r } { s \geqslant \frac { 1 } { 2 } - \frac { 1 } { q } } \end{array}$

## 4.2 Step efficiency

At a fixed data budget $D = N B$ , increasing the batch size reduces the number of training steps. We therefore ask two questions: $( i )$ Can we increase the batch size while preserving each optimizer's small-batch risk? (ii) If so, how large can the batch size be? The critical batch size (CBS) characterizes the largest such batch size, with the LR schedule allowed to be retuned.

Definition 4.3 (Critical batch size). Let $\mathcal { A } _ { D , B }$ denote the admissible LR schedules for a given optimizer, and define $\begin{array} { r } { \mathcal { E } _ { \star } ( D , B ) : = \operatorname* { i n f } _ { \eta \in \mathcal { A } _ { D , B } } \mathbb { E } \big [ \mathcal { E } \big ( \vartheta _ { \Gamma _ { B } ( D / B ) } \big ) \big ] } \end{array}$ . For a fixed constant $\delta > 0$ we define

$$
B _ { \mathrm { c r i t } } ( D ) : = \operatorname* { s u p } \left\{ B \in \left\{ 1 , \ldots , D \right\} : \mathcal { E } _ { \star } ( D , B ) \leqslant \left( 1 + \delta \right) \mathcal { E } _ { \star } ( D , 1 ) \right\} , \qquad N _ { \mathrm { c r i t } } ( D ) : = \frac { D } { B _ { \mathrm { c r i t } } ( D ) } .
$$

Here $\Gamma _ { B }$ denotes the optimizer's intrinsic-time map for the chosen schedule η, and $N _ { \mathrm { c r i t } } ( D )$ is the minimum number of steps needed to preserve the optimal small-batch risk within the factor $1 + \delta .$

For SGD, admissibility requires $\eta _ { \mathrm { p e a k } } \lesssim 1$ . For SignSGD, it requires $\eta _ { \mathrm { p e a k } } \sqrt { B } \lesssim 1$ . We consider $0 < s < 3 / 2 - 1 / q$ , where the SignSGD training-time condition is compatible with the batch-size ranges below. The boundary $s = 3 / 2 - 1 / q$ and the super-easy regime $s > 3 / 2 - 1 / q$ are discussed in Appendix F. For notational convenience, we write $\textstyle \alpha = { \dot { s } } + { \dot { \frac { 1 } { q } } }$

Theorem 4.4 (Critical batch size). Suppose that the assumptions of Theorem 3.1 hold and that $0 < s < 3 / 2 - 1 / q$ . Then the critical batch sizes satisfy

$$
B _ { \mathrm { c r i t } } ^ { \mathrm { s g d } } ( D ) \approx \operatorname* { m a x } \left\{ 1 , D ^ { 1 - \frac { 1 } { \alpha } } \right\} , \qquad B _ { \mathrm { c r i t } } ^ { \mathrm { s g n } } ( D ) \stackrel { } { \sim } \operatorname* { m a x } \left\{ 1 , D ^ { 1 - \frac { 1 } { 2 \alpha } } \right\} .\tag{6}
$$

Peak LR scaling. The batch-size ranges in Theorem 4.4 are attained by the power-decay schedule (5) with $N = D / B$ , any fixed $p > q - 1$ , and peak learning rates

$$
\eta _ { \mathrm { p e a k } } ^ { \mathrm { s g d } } \ = \operatorname* { m i n } \left\{ B D ^ { - 1 + \frac { 1 } { \alpha } } , 1 \right\} , \qquad \eta _ { \mathrm { p e a k } } ^ { \mathrm { s g n } } \ = \operatorname* { m i n } \left\{ \sqrt { B } D ^ { - 1 + \frac { 1 } { 2 \alpha } } , B ^ { - 1 / 2 } \right\} .
$$

In the easy-task regime, the linear scaling rule for SGD and the square-root scaling rule for SignSGD preserve the small-batch risk as B increases. Their admissibility constraints, $\bar { \eta } _ { \mathrm { p e a k } } ^ { \mathrm { s g d } } \lesssim 1$ and $\sqrt { B } \eta _ { \mathrm { p e a k } } ^ { \mathrm { s g n } } \lesssim 1$ , determine the batch-size ranges over which these rules apply.

Efficiency gains across task regimes. The figure on the right compares the CBS exponents b in $B _ { \mathrm { c r i t } } ( D ) \lsim D ^ { b }$ . In the easy-task regime, $1 - 1 / q \leqslant s < 3 / 2 - 1 / q$ , both optimizers attain excess risk of order $D ^ { - s / \alpha }$ . At their respective CBSs, SGD requires $N _ { c r \mathrm { i } \mathrm { t } } ^ { \mathrm { s g d } } \approx D ^ { 1 / \alpha }$ updates, whereas SignSGD requires only $N _ { \mathrm { c r i t } } ^ { \mathrm { s g n } } \approx D ^ { 1 / ( 2 \alpha ) }$ . Thus, SignSGD uses a factor $D ^ { 1 / ( 2 \alpha ) }$ fewer sequential updates while maintaining the same data-scaling rate. For hard tasks $\textstyle { \frac { 1 } { 2 } } - { \frac { 1 } { q } } < s < 1 - { \frac { 1 } { q } }$ , SignSGD achieves both a faster

![](images/48d64949bedd4ed715124d582af9314c669222582d9801e883d497f8c15aa9c5.jpg)

data-scaling rate and a CBS that grows with D, whereas SGD has $B _ { \mathrm { c r i t } } ^ { \mathrm { s g d } } ( D ) \approx 1$ . These CBSs preserve each optimizer's small-batch risk, so SignSGD improves both data and step efficiency. For super-hard tasks, $\begin{array} { r } { \mathrm { ~ \normalfont ~ { ~ 3 ~ } ~ } \leqslant \frac { 1 } { 2 } - \frac { 1 } { q } } \end{array}$ , both CBSs are of constant order, requiring $N \gtrsim D$ updates to preserve their respective small-batch risks. SignSGD's advantage in this regime therefore lies in data efficiency rather than step efficiency.

## 5 Experiments

We evaluate the predictive performance of FSL in PLK regression and language model pre-training. The FSL for SignSGD motivates the parameterized form

$$
L _ { \Theta } ( k ) = L _ { \infty } + S _ { k } + N _ { k } , \qquad S _ { k } = C _ { S } ( A _ { S } + t _ { k } ) ^ { - \alpha } , \qquad N _ { k } = C _ { K } \sum _ { j = 0 } ^ { k - 1 } ( A _ { K } + t _ { k } - t _ { j } ) ^ { - \nu } \eta _ { j } ^ { 2 } ,\tag{7}
$$

where $L _ { \infty }$ is the irreducible loss, $S _ { k }$ is the signal term, and $N _ { k }$ is the noise term. The trainable parameters are $\Theta = ( L _ { \infty } , C _ { S } , A _ { S } , \alpha , C _ { K } , A _ { K } , \nu )$ , and $\begin{array} { r } { t _ { k } = \sqrt { B } \sum _ { j = 0 } ^ { k - 1 } \eta _ { j } } \end{array}$ is the intrinsic time associated with the LR schedule $( \eta _ { 0 } , \dots , \eta _ { N - 1 } )$ . Further details are provided in Appendix H.1.

## 5.1 PLK regression

FSL accurately captures risk dynamics. Although our FSL is derived from the continuous-time model (35), Figure 1 (left) shows that the resulting FSL also accurately describes discrete-time training. For PLK regression with $( s , q ) = ( 0 . 3 , 4 )$ , we show accurate fitting and prediction of SignSGD risk curves across cosine, WSD, and cyclic schedules.

Signal-noise decomposition and intrinsic-time collapse. Using the experiments in Figure 1 (left), we decompose the risk trajectories under the three LR schedules into the noise term $N _ { k }$ and signal term $S _ { k }$ , shown in the left and middle panels of Figure 2, respectively. Although the signal trajectories differ in training steps, they collapse onto a common curve when plotted against intrinsic time in the right panel. The FSL thus separates schedule-dependent noise effects from a shared signal-learning profile governed by intrinsic time.

![](images/4d47e704198d019f187b9f99de5bf8af0a92b6a8b779c860939b1968ec2e136c.jpg)

![](images/2d285e871628c7574ecf28e7a4a1f8616899b6fa9ad6817bd838eb78a0c74c8a.jpg)

![](images/b79317d58e991a43f75f40430b82b16fc55abf47273cafcbb37ba28961778ae2.jpg)  
Figure 2: FSL decomposition and intrinsic-time collapse for SignSGD. We use the same configurations and shared FSL as in Figure 1 (left). Solid curves correspond to the cosine and WSD schedules used for fitting; dashed curves show predictions for the held-out cyclic schedule. (Left) Noise term $N _ { k }$ versus training steps. (Middle) Signal term $S _ { k }$ versus training steps. (Right) The signal terms collapse onto a common curve when plotted against intrinsic time.

Data scaling and critical batch size. We validate the data-scaling and batch-size predictions in Sections 4.1 and 4.2. On the hard task with $( s , q ) \ : = \ : ( 0 . 3 , 4 )$ , we tune the peak learning rate for each optimizer and data budget and compare the best final excess risk. Figure 1 (middle) shows that SignSGD achieves a faster data-scaling rate than SGD, consistent with Theorem 4.1. On the easy task with $( s , q ) \ = \ ( 1 , 4 )$ , we sweep batch sizes at each data budget

![](images/d0f2f77d1c384010495c33446d5ca3d8c29a3c4d15b80f5236a7cfcbe8b2db53.jpg)

and tune the peak learning rate separately for every optimizer and batch size. We estimate the CBS as the largest batch size that approximately preserves each optimizer's small-batch risk. The right panel shows that SignSGD has the largest CBS, followed by Adam and SGD, throughout the tested range. The power-law fits further indicate that the CBS grows faster with the data budget for SignSGD and Adam than for SGD. These findings support the predicted step efficiency of SignSGD and its heuristic extension to Adam. Similar trends appear in language-model training: Muon exhibits greater large-batch robustness than Adam, which in turn outperforms SGD (Marek et al., 2025; Srećković et al., 2025; Thérien et al., 2026). Since Muon applies matrix sign operator to gradient updates, these observations suggest a broader connection between sign-based normalization and batch-size efficiency.

## 5.2 Language model pre-training

We train a Llama-style language model (Touvron et al., 2023) with approximately 0.1B parameters on 10B tokens, using a realistic token-to-parameter ratio of approximately 100. We evaluate the FSL for SignSGD and Adam under different LR schedules, including cosine, WSD (Hu et al., 2024), and the 8-1-1 schedule (DeepSeek-AI et al., 2024). Additional experimental details are provided in Appendix H.

FSL accurately captures loss dynamics across optimizers. We evaluate the predictive power of FSL across different LR schedules and optimizers. Specifically, we extend the Adam experiments of Li et al. (2025) to SignSGD. For each optimizer, we fit the FSL parameters using the loss curves of cosine and 8-1-1 schedules, and then use the fitted FSL to predict the loss curves under the WSD schedule. The results for SignSGD and Adam are reported in Figure 3 (a). We observe that FSL accurately captures the loss curves for both optimizers and yields accurate predictions for the held-out schedule.

![](images/e8584efd917864bdf06a220459ff4114a4e07610287324d5cb02aed731118536.jpg)

![](images/e9950e6484c5f81d9aed26c495f50118a91dd93e0c6f1395ed3a2be88d6a4e43.jpg)  
(a) Cross-schedule prediction

![](images/ea571dd4ef5adf4e3701c381a123d83f3ff8c9a2dd7f0db4f7621c51b6ff34fc.jpg)

(b) Batch-size extrapolation  
![](images/26ede0e9ed96d98b2e94d133d741f84b0daacb14f36490da788799747370dddc.jpg)  
Figure 3: FSL captures loss dynamics and batch-size dependence in practical language model pre-training. (a) Cross-schedule loss prediction for SignSGD (left) and Adam (right). For each optimizer, we fit the FSL parameters Θ to the loss curves under the cosine and 8-1-1 schedules and predict the loss curve under the held-out WSD schedule. (b) Batch-size extrapolation for SignSGD (left) and Adam (right). With a constant learning rate, we fit Θ to the loss curves at $B = 6 4$ and $B = 1 2 8$ and predict the loss curve at $B = 2 5 6$ Solid and dashed FSL curves indicate fits and predictions, respectively.

FSL accurately captures batch-size dependence and enables extrapolation. A key difference between the FSL in (7) and the SGD-based FSL of Li et al. (2025) lies in the batch-size scaling. We test this batch-size dependence in practical training. We train with batch sizes $B \in \{ 6 4 , 1 2 8 , 2 5 6 \}$ and a constant learning rate. For each optimizer, we fit the FSL using the curves for $B = 6 4$ and $B = 1 2 8$ , and then predict the held-out curve for $B = 2 5 6$ As shown in Figure 3 (b), the FSL jointly fits the two batch sizes and extrapolates well to the larger batch size for both SignSGD and Adam. An additional experiment with cosine schedule for Adam is reported in Appendix H.3, Figure 12 (left).

## 6 Why does SignSGD outperform SGD?

## 6.1 Smoothed sign and one-dimensional dynamics

Smoothed sign. We first examine the drift and diffusion in (2) in one dimension. Recall that $\sigma _ { B } ~ = ~ 1 / \sqrt { B }$ . Let $a \ > \ 0$ denote the single-sample gradient-noise standard deviation, so the mini-batch noise standard deviation is $\sigma _ { B } a$ The scaled smoothed sign is $m _ { B } ( x ) = $ $\sigma _ { B } \mathbb { E } _ { \xi \sim \mathcal { N } ( 0 , a ^ { 2 } ) } [ \mathrm { s g n } ( x + \sigma _ { B } \xi ) ]$ . Writing $c = { \sqrt { 2 / \pi } }$ , its two limiting responses are

$$
m _ { B } ( x ) \approx \left\{ \begin{array} { l l } { c x / a , } & { | x | / ( \sigma _ { B } a ) \ll 1 , } \\ { \sigma _ { B } \operatorname { s g n } ( x ) , } & { | x | / ( \sigma _ { B } a ) \gg 1 . } \end{array} \right.\tag{8}
$$

The variance of the corresponding sign update is $Q _ { B } ( x ) = 1 - m _ { B } ( x ) ^ { 2 } / \sigma _ { B } ^ { 2 }$ Thus, the noise-dominated region has approximately linear drift and $Q _ { B } ( x ) ~ \approx ~ 1$ , whereas the signal-dominated region has saturated drift and $Q _ { B } ( x ) \approx 0$

One-dimensional dynamics. For a coordinate with curvature λ, let $\delta _ { t } = \vartheta _ { t } - \theta ^ { * }$ , so the population gradient is $\lambda \delta _ { t }$ . Applying (8) to $x = \lambda \delta$ reveals two dynamical regimes. For $\lambda | \delta | / ( \sigma _ { B } a ) \gg 1$ , the response saturates at $\sigma _ { B } \ : \mathrm { s g n } ( \delta )$ and $Q _ { B } ( \lambda \delta ) \approx 0 .$ giving approximately deterministic motion toward zero. For $\lambda | \delta | / ( \sigma _ { B } a ) \ll 1$ the response is linear, $m _ { B } ( \lambda \delta ) \approx c \lambda \delta / a .$ and $Q _ { B } ( \lambda \delta ) \approx$ 1. The local dynamics therefore have the drift of SGD preconditioned by $c / a$ , with approximately unit sign-noise variance. The crossover occurs at $| \delta | \ \stackrel { } { \sim }$ $\sigma _ { B } a / \lambda .$ Under noise-curvature alignment Under noise-curvature alignment $a ^ { 2 } \approx \lambda$ , th

![](images/c7138b736bf87d7d58a1e42ad6fcad83becf594f1089900aafa541b5cbb1aa34.jpg)

, this scale becomes $\sigma _ { B } / \sqrt { \lambda } \ : = \ : ( \lambda B ) ^ { - 1 / 2 }$ 2 as illustrated on the right for $\delta \geqslant 0$

These two limits describe the saturated dynamics during burn-in and the local linear dynamics underlying the scaling regime, respectively. Substituting them into (2) gives

$$
\begin{array} { r } { \mathrm { d } \delta _ { t } \approx \left\{ { \begin{array} { l l l } { - \sigma _ { B } \mathrm { { s g n } } ( \delta _ { t } ) \ \mathrm { { d } } t , } & { \frac { \lambda | \delta _ { t } | } { \sigma _ { B } a } \gg 1 } & { ( \mathrm { { b u r n } - i n \ p h a s e } ) , } \\ { - c \frac { \lambda } { a } \delta _ { t } \mathrm { { d } } t + \sqrt { n _ { B } ( t ) } \ \mathrm { { d } } W _ { t } , } & { \frac { \lambda | \delta _ { t } | } { \sigma _ { B } a } \ll 1 } & { ( \mathrm { { l o c a l } \ p h a s e } ) . } \end{array} } \right. } \end{array}\tag{9}
$$

From burn-in to local phase. We refer to the initial transient before entering the linear region as burn-in. The first branch of (9) describes its saturated part. Under zero initialization, this approximation gives $| \delta _ { t } | \approx | \theta ^ { * } | - \sigma _ { B } t$ until the transition region is reached. The source condition implies $| \theta _ { j } ^ { * } | \stackrel { } { \sim } j ^ { ( q - 1 - s q ) / 2 }$ . For easy tasks, $s + 1 / q \geqslant 1$ , the initial coordinate errors are uniformly bounded, giving a saturated burn-in time of at most order $\sigma _ { B } ^ { - 1 } = \sqrt { B }$ . For hard tasks, $s + 1 / q < 1$ , these errors can grow with the coordinate index, and controlling burn-in across the relevant spectral modes requires $t \gtrsim \sigma _ { B } ^ { - 1 / ( s + 1 / q ) } = B ^ { 1 / ( 2 s + 2 / q ) }$ . Together, these motivate the training-time condition $t \gtrsim B ^ { \operatorname* { m a x } \{ 1 / 2 , 1 / ( 2 s + 2 / q ) \} }$ in Theorem 3.1. This local analysis provides intuition for the scaling regime. The rigorous argument in Appendix D uses a Lyapunov function to establish global moment bounds for the nonlinear dynamics, allowing noise-driven excursions outside the local linear region.

## 6.2 Noise-induced preconditioning

From coordinate-wise normalization to preconditioning. The local linear response in the preceding subsection extends coordinate-wise to the multidimensional dynamics. Let $\delta _ { t } = $ $\vartheta _ { t } - \theta ^ { * }$ and $\mathbf { D } = \mathrm { d i a g } ( \Sigma _ { 1 1 } , \Sigma _ { 2 2 } , . . . )$ . Linearizing the smoothed sign around the origin gives m ${ \bf \nabla } _ { \cdot B } ( { \bf H } \delta _ { t } ) \approx c { \bf D } ^ { - 1 / 2 } { \bf H } \delta _ { t }$ . Thus, the local drift takes the form $- \mathbf { P } \nabla \mathcal { R } ( \vartheta _ { t } )$ , where $\mathbf { P } : = c \mathbf { D } ^ { - 1 / 2 }$ This is the drift of preconditioned SGD: the gradient is normalized by its single-sample noise scale on each coordinate. Such normalization does not by itself guarantee acceleration; its effectiveness depends on the relation between the noise scales and the curvature.

Structural conditions for effective preconditioning. Noise-curvature alignment and diagonal features play important roles in SignSGD's scaling advantage. The former ensures that $\Sigma _ { j j } \ : \approx \ : H _ { j j }$ , so the preconditioner compensates for coordinate-wise curvature. Diagonal features further ensure that these coordinates coincide with the eigendirections of H. Together, they give $a _ { j } : = \sqrt { \Sigma _ { j j } } \approx \sqrt { \lambda _ { j } }$ and effective decay rates $\begin{array} { r } { \rho _ { j } : = \frac { c \lambda _ { j } } { a _ { j } } \overset { } { \sim } \sqrt { \lambda _ { j } } } \end{array}$ . The effective spectrum therefore decays as $j ^ { - q / 2 }$ , rather than $j ^ { - q }$ , reducing the disparity between fast and slow spectral directions. Without noise-curvature alignment, noise normalization need not compensate for curvature. Without diagonal features, coordinate-wise preconditioning need not produce a square-root transformation of the Hessian spectrum.

Effective preconditioned SGD. Under our diagonal Gaussian noise model, the sign-noise coordinates are conditionally independent, and their variances approach one in the local regime. Hence $\mathbf { Q } _ { B } \approx \mathbf { I } .$ yielding

$$
\mathrm { d } \delta _ { t } \approx - \mathbf { P H } \delta _ { t } \mathrm { ~ d } t + \sqrt { n _ { B } ( t ) } \ \mathrm { d } \mathbf { W } _ { t } , \qquad \mathbf { P } = c \mathbf { D } ^ { - 1 / 2 } .\tag{10}
$$

This has the form of preconditioned SGD with normalized gradient noise. Under noise-curvature alignment and diagonal features, $\mathbf { P H } = \mathrm { d i a g } ( \rho _ { j } )$ with $\rho _ { j } \lesssim \sqrt { \lambda _ { j } }$ . The local dynamics therefore combine square-root spectral decay rates with approximately isotropic noise injection. These two changes determine the signal-learning and noise-forgetting profiles derived next.

Empirical evidence for the structural conditions. Figure 4 illustrates the role of these two structural ingredients at a fixed data budget. With matched total gradient noise energy and separately tuned learning rates, SignSGD attains a substantially lower final risk in the setting with diagonal features and curvature-aligned noise, while rotations (breaking diagonal structure) or isotropic noise (breaking noise-curvature alignment) reduce this advantage.

![](images/e04e7d04712570e1b5156e3aed39c3409cf9733f251f86910aa39e5bae7cb1f9.jpg)

![](images/c55571235e00ba1c987cc5c9845a5346dbf457ce0fca87b8b250a8d79ab49058.jpg)

![](images/69a7d46afac78037bc4d880643a5aa84591eaf30bced4a58e16234e80edb4194.jpg)  
Figure 4: Structural conditions for the SignSGD advantage. (Left) Risk dynamics of SGD and SignSGD with curvature-aligned noise, isotropic-noise controls, and a rotated feature covariance. (Middle) We vary noise-curvature alignment by setting the noise covariance to $\Sigma \ \propto$ Hγ with $\gamma \in \ [ 0 , 1 ]$ . SignSGD shows an advantage only when $\gamma$ is close to 1. (Right) We vary feature diagonality by rotating the feature covariance, with $\rho$ controlling the rotation strength. As $\rho$ increases, the covariance becomes less diagonal and the SignSGD advantage weakens. All experiments use optimally tuned learning rates.

## 6.3 Deriving the functional scaling law

Spectral representation. The effective dynamics (10) have diagonal drift $\mathbf { P H } = \mathrm { d i a g } ( \rho _ { j } )$ where $\rho _ { j } = c \lambda _ { j } / a _ { j } \stackrel { } { \sim } \sqrt { \lambda _ { j } }$ . For this linear model initialized at $\delta _ { \mathrm { 0 } } = - \theta ^ { \ast }$ , the expected excess risk decomposes as

$$
\mathbb { E } [ \mathcal { E } ( \pmb { \vartheta } _ { t } ) ] \approx S ( t ) + \int _ { 0 } ^ { t } K ( t - \tau ) n _ { B } ( \tau ) ~ \mathrm { d } \tau ,\tag{11}
$$

where

$$
S ( t ) = \frac { 1 } { 2 } \sum _ { j } \lambda _ { j } | \theta _ { j } ^ { * } | ^ { 2 } e ^ { - 2 \rho _ { j } t } , \qquad K ( t ) = \frac { 1 } { 2 } \sum _ { j } \lambda _ { j } e ^ { - 2 \rho _ { j } t } .
$$

The signal profile measures the remaining initial error, while the forgetting kernel measures the residual risk from a unit noise injection after elapsed intrinsic time t. Both are mixtures of exponential decays with rates $\rho _ { j }$ , but their spectral weights differ. These expressions describe the effective linear model; they do not assume that the original dynamics are linear throughout burn-in.

From exponential modes to power laws. Each spectral mode has characteristic time $\tau _ { j } : = \rho _ { j } ^ { - 1 } \stackrel { } { \sim } j ^ { q / 2 }$ . At time t, modes with $\tau _ { j } \ll t$ have largely decayed, whereas those with $\tau _ { j } \gg t$ retain substantial contributions. Therefore, we introduce the moving frontier $j _ { t }$ to characterize which modes are learned at time t. It is determined by $\tau _ { j _ { t } } \approx t ,$ and thus satisfies $j _ { t } \approx t ^ { 2 / q }$ . To estimate the signal profile, split the spectrum at this cutoff:

$$
2 S ( t ) = \underbrace { \sum _ { j \leqslant j _ { t } } \lambda _ { j } | \theta _ { j } ^ { * } | ^ { 2 } e ^ { - 2 \rho _ { j } t } } _ { \mathrm { d e c a y i n g ~ m o d e s } } + \underbrace { \sum _ { j > j _ { t } } \lambda _ { j } | \theta _ { j } ^ { * } | ^ { 2 } e ^ { - 2 \rho _ { j } t } } _ { \mathrm { r e m a i n i n g ~ s p e c t r a l ~ t a i l } } .\tag{12}
$$

For $j > j _ { t } ,$ we have $\rho _ { j } t \lesssim 1$ , so the exponential factor is bounded above and below by positive constants. The tail contribution is therefore $\begin{array} { r } { \gtrsim \sum _ { j > j _ { t } } j ^ { - ( 1 + s q ) } \gtrsim j _ { t } ^ { - s q } } \end{array}$ . For $j \leqslant j _ { t }$ , the bound $e ^ { - u } \lesssim u ^ { - p }$ , with any fixed $p > 2 s$ gives

$$
\sum _ { j \leqslant j _ { t } } \lambda _ { j } | \theta _ { j } ^ { * } | ^ { 2 } e ^ { - 2 \rho _ { j } t } \lesssim t ^ { - p } \sum _ { j \leqslant j _ { t } } j ^ { - 1 - s q + p q / 2 } \asymp t ^ { - p } j _ { t } ^ { p q / 2 - s q } \asymp j _ { t } ^ { - s q } .
$$

Thus, the scaling behavior is determined by ${ j _ { t } ^ { - s q } }$ . Applying the same argument to K yields

$$
S ( t ) \approx j _ { t } ^ { - s q } \pi ^ { } \pi t ^ { - 2 s } , \qquad K ( t ) \approx j _ { t } ^ { 1 - q } \pi ^ { } t ^ { - ( 2 - 2 / q ) }\tag{13}
$$

for sufficiently large t. Substitution into (11) recovers the FSL structure. Appendix D controls the nonlinear dynamics to establish the FSL of Theorem 3.1.

Faster signal learning, slower noise forgetting. The same spectral-cutoff argument explains the difference between SGD and SignSGD. In their respective intrinsic times, SGD has decay rates of order $\lambda _ { j }$ and cutoff $j _ { t } \approx t ^ { 1 / q }$ , whereas SignSGD has rates of order $\sqrt { \lambda _ { j } }$ and cutoff $j _ { t } \approx t ^ { 2 / q }$ . The faster advance through the spectrum changes the signal-learning function from $t ^ { - s } \ \mathrm {  t o } \ t ^ { - 2 s }$ . Applying the same argument to the forgetting kernels gives decay rates $t ^ { - ( 2 - 1 / q ) }$ for SGD and $\bar { t } ^ { - ( 2 - 2 / q ) }$ for SignSGD, showing that SignSGD forgets noise more slowly. Thus, the preconditioning effect accelerates signal learning while slows the noise forgetting.

The effect of batch size. The comparison above uses each method's own intrinsic time. For the same LR schedule $\eta ( z )$ , SGD uses $\begin{array} { r } { t _ { \mathrm { s g d } } ( z ) = \int _ { 0 } ^ { z } \eta ( \tau ) } \end{array}$ dτ, whereas SignSGD uses $t _ { \mathrm { s g n } } ( z ) =$ $\sqrt { B } t _ { \mathrm { s g d } } ( z )$ . Consequently, their signal-learning functions expressed in training steps satisfy $S _ { \mathrm { s g n } } ( z ) \stackrel { - } { \sim } ( \sqrt { B } t _ { \mathrm { s g d } } ( z ) ) ^ { - 2 s }$ and $S _ { \mathrm { s g d } } ( z ) \stackrel { } { \sim } t _ { \mathrm { s g d } } ( z ) ^ { - s }$ . The additional $\sqrt { B }$ factor is from the local response $\mathbb { E } [ \mathrm { s g n } ( g _ { j } ) ]$ ≈ $c \sqrt { B } \lambda _ { j } \delta _ { j } / a _ { j } \mathrm { ; }$ a larger batch size reduces gradient noise and increases the mean sign response in the local regime. Batch size also enters the noise injection intensity through $n _ { B } ( t ) = \eta ( \Gamma _ { B } ^ { - 1 } ( t ) ) / \sqrt { B }$ Together, the intrinsic-time transformation and the noise injection intensity determine how batch size affects signal learning and noise accumulation in the FSL.

## 7 Concluding remarks

Our results show that the advantages of SignSGD depend on both task difficulty and the structure of gradient noise. The FSL makes this dependence explicit by tracking the decay of the initial error and the persistence of injected noise. Its mechanistic interpretation links the local response of the sign operator to noise-dependent preconditioning, whose effectiveness relies on diagonal features and noise-curvature alignment. The resulting balance between faster signal learning and slower noise forgetting distinguishes two benefits: better use of data on hard tasks, and fewer sequential updates at comparable data efficiency in the covered easy-task regime. Experiments corroborate these conclusions in kernel regression and show that the

SignSGD FSL better describes the effect of batch size on sign-based optimizer trajectories in LLM pre-training than the original SGD FSL.

Our results motivate two directions for future work:

• Extensions to RMSprop and Adam. Our derivations predict the same FSL form and batch-size dependence, with supporting numerical evidence (Appendix G and Appendix H.2). A rigorous treatment would need to account for momentum, evolving second-moment estimates, and their coupling with gradient noise.

• From coordinate-wise sign to matrix sign. Extending the analysis to the matrix sign operator underlying Muon would clarify how matrix structure and gradient noise jointly determine effective preconditioning, signal learning, noise forgetting, and batch-size dependence.

## Acknowledgement

Lei Wu is supported in part by the National Natural Science Foundation of China (NSFC12522120, NSFC92470122, and NSFC12288101). Binghui Li is supported by the Elite Ph.D. Program in Applied Mathematics at Peking University.

## References

Alexander Atanasov, Jacob A. Zavatone-Veth, and Cengiz Pehlevan. Scaling and renormalization in high-dimensional regression. Journal of Statistical Mechanics: Theory and Experiment, 2026(4):043404, 2026. doi: 10.1088/1742-5468/ae4bba. (cited on page 19)

Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon Lee, and Utkarsh Sharma. Explaining neural scaling laws. Proceedings of the National Academy of Sciences, 121(27):e2311878121, 2024. (cited on page 19)

Lukas Balles and Philipp Hennig. Dissecting Adam: The sign, magnitude and variance of stochastic gradients. In International Conference on Machine Learning, volume 80, pp. 404–413. PMLR, 2018. (cited on page 1)

Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Animashree Anandkumar. signSGD: Compressed optimisation for non-convex problems. In International Conference on Machine Learning, volume 80, pp. 560–569. PMLR, 2018. (cited on pages 1 and 19)

Blake Bordelon, Alexander Atanasov, and Cengiz Pehlevan. A dynamical model of neural scaling laws. In International Conference on Machine Learning, pp. 4345–4382. PMLR, 2024. (cited on page 19)

Blake Bordelon, Alexander Atanasov, and Cengiz Pehlevan. How feature learning can improve neural scaling laws. Journal of Statistical Mechanics: Theory and Experiment, 2025(8): 084002, 2025. (cited on page 19)

Andrea Caponnetto and Ernesto De Vito. Optimal rates for the regularized least-squares algorithm. Foundations of Computational Mathematics, 7:331–368, 2007. (cited on page 6)

Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, and Quoc V. Le. Symbolic discovery of optimization algorithms. Advances in Neural Information Processing Systems, 36: 49205–49233, 2023. (cited on page 1)

Xiangyi Chen, Sijia Liu, Ruoyu Sun, and Mingyi Hong. On the convergence of a class of Adam-type algorithms for non-convex optimization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=H1x-x309tm. (cited on page 1)

Xiaoran Chen, Binghui Li, Liwei Wang, and Pinyan Lu. When and how does Muon precondition? The role of structured gradient noise. arXiv preprint arXiv:2610.31728, 2026. (cited on page 19)

DeepSeek-AI, Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, Huazuo Gao, Kaige Gao, Wenjun Gao, Ruiqi Ge, Kang Guan, Daya Guo, Jianzhong Guo, Guangbo Hao, Zhewen Hao, Ying He, Wenjie Hu, Panpan Huang, Erhang Li, Guowei Li, Jiashi Li, Yao Li, Y. K. Li, Wenfeng Liang, Fangyun Lin, A. X. Liu, Bo Liu, Wen Liu, Xiaodong Liu, Xin Liu, Yiyuan Liu, Haoyu Lu, Shanghao Lu, Fuli Luo, Shirong Ma, Xiaotao Nie, Tian Pei, Yishi Piao, Junjie Qiu, Hui Qu, Tongzheng Ren, Zehui Ren, Chong Ruan, Zhangli Sha, Zhihong Shao, Junxiao Song, Xuecheng Su, Jingxiang Sun, Yaofeng Sun, Minghui Tang, Bingxuan Wang, Peiyi Wang, Shiyu Wang, Yaohui Wang, Yongji Wang, Tong Wu, Y. Wu, Xin Xie, Zhenda Xie, Ziwei Xie, Yiliang Xiong, Hanwei Xu, R. X. Xu, Yanhong Xu, Dejian Yang, Yuxiang You, Shuiping Yu, Xingkai Yu, B. Zhang, Haowei Zhang, Lecong Zhang, Liyue Zhang, Mingchuan Zhang, Minghua Zhang, Wentao Zhang, Yichao Zhang, Chenggang Zhao, Yao Zhao, Shangyan Zhou, Shunfeng Zhou, Qihao Zhu, and Yuheng Zou. DeepSeek LLM: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954, 2024. (cited on pages 8 and 51)

Alexandre Défossez, Léon Bottou, Francis Bach, and Nicolas Usunier. A simple convergence proof of Adam and Adagrad. Transactions on Machine Learning Research, 2022. URL https://openreview.net/forum?id=ZPQhzTSWA7. (cited on page 1)

Shihong Ding, Haihan Zhang, Hanzhen Zhao, and Cong Fang. Scaling law for stochastic gradient descent in quadratically parameterized linear regression. arXiv preprint arXiv:2502.09106, 2025. (cited on page 19)

Elvis Dohmatob, Yunzhen Feng, Pu Yang, François Charton, and Julia Kempe. A tale of tails: Model collapse as a change of scaling laws. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 11165–11197. PMLR, 2024. (cited on page 19)

Geoffrey Hinton, Nitish Srivastava, and Kevin Swersky. Neural networks for machine learning, lecture 6a: Overview of mini-batch gradient descent. Coursera lecture slides, 2012. URL https://www.cs.toronto.edu/\~hinton/coursera/lecture6/lec6.pdf. (cited on page 1)

Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, and Others. MiniCPM: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024. (cited on pages 8, 48, and 51)

Ayush Jain, Andrea Montanari, and Eren Sasoglu. Scaling laws for learning with real and surrogate data. Advances in Neural Information Processing Systems, 37:110246–110289, 2024. (cited on page 19)

Wei Jiang, Sifan Yang, Wenhao Yang, and Lijun Zhang. Efficient sign-based optimization: Accelerating convergence via variance reduction. Advances in Neural Information Processing Systems, 37:33891–33932, 2024. (cited on page 19)

Wei Jiang, Dingzhi Yu, Sifan Yang, Wenhao Yang, and Lijun Zhang. Improved analysis for sign-based methods with momentum updates. arXiv preprint arXiv:2507.12091v1, 2025. (cited on page 19)

Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https://kellerjordan.github.io/posts/muon/. (cited on page 1)

Jihwan Kim, Dogyoon Song, and Chulhee Yun. Scaling laws of SignSGD in linear regression: When does it outperform SGD? In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=zVwRP0ikrx. (cited on pages 3 and 19)

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015. (cited on page 1)

Frederik Kunstner and Francis Bach. Scaling laws for gradient descent and sign descent for linear bigram models under Zipf's law. In Advances in Neural Information Processing Systems, volume 38, pp. 160282–160314, 2025. (cited on pages 1 and 19)

Frederik Kunstner, Jacques Chen, J. Wilder Lavington, and Mark Schmidt. Noise is not the main factor behind the gap between SGD and Adam on transformers, but sign descent might be. In International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=a65YK0cqH8g. (cited on page 1)

Frederik Kunstner, Alan Milligan, Robin Yadav, Mark Schmidt, and Alberto Bietti. Heavy-tailed class imbalance and why Adam outperforms gradient descent on language models. Advances in Neural Information Processing Systems, 37:30106–30148, 2024. (cited on page 1)

Binghui Li, Fengling Chen, Zixun Huang, Lean Wang, and Lei Wu. Functional scaling laws in kernel regression: Loss dynamics and learning rate schedules. In Advances in Neural Information Processing Systems, volume 38, pp. 112258–112316, 2025. (cited on pages 2, 3, 4, 5, 9, 19, 24, and 39)

Binghui Li, Jianan Wang, Jinbo Wang, Lean Wang, Zilin Wang, and Lei Wu. Scaling-law analysis of SignSGD: From feature-space linear regression to LLM pre-training. In SciForDL, 2nd edition, 2026a. URL https://openreview.net/forum?id=KQfV97S12I. (cited on page 19)

Binghui Li, Kaifei Wang, Han Zhong, Pinyan Lu, and Liwei Wang. Muon in associative memory learning: Training dynamics and scaling laws. arXiv preprint arXiv:2602.05725, 2026b. (cited on pages 1 and 19)

Binghui Li, Zilin Wang, Fengling Chen, Shiyang Zhao, Ruiheng Zheng, and Lei Wu. Optimal learning rate schedules under functional scaling laws: Power decay and warmup-stable-decay. arXiv preprint arXiv:2602.06797, 2026c. (cited on pages 4, 5, 6, 19, 37, 39, and 48)

Licong Lin, Jingfeng Wu, Sham M. Kakade, Peter L. Bartlett, and Jason D. Lee. Scaling laws in linear regression: Compute, parameters, and data. In Advances in Neural Information Processing Systems, volume 37, pp. 60556–60606, 2024. (cited on page 19)

Licong Lin, Jingfeng Wu, and Peter L. Bartlett. Improved scaling laws in linear regression via data reuse. In Advances in Neural Information Processing Systems, volume 38, pp. 100719–100765, 2025. (cited on page 19)

Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang, Jianzhou Wang, Mengnan Dong, Zheng Zhang, Yongsheng Kang, Hao Zhang, Xinran Xu, Yutao Zhang, Yuxin Wu, Xinyu Zhou, and Zhilin Yang. Muon is scalable for LLM training. arXiv preprint arXiv:2502.16982, 2025. (cited on page 1)

Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In International Conference on Learning Representations, 2017. URL https://openreview. net/forum?id=Skq89Scxx. (cited on pages 47 and 51)

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id= Bkg6RiCqY7. (cited on pages 1 and 51)

Sadhika Malladi, Kaifeng Lyu, Abhishek Panigrahi, and Sanjeev Arora. On the SDEs and scaling rules for adaptive gradient algorithms. Advances in Neural Information Processing Systems, 35:7697–7711, 2022. (cited on page 19)

Alexander Maloney, Daniel A Roberts, and James Sully. A solvable model of neural scaling laws. arXiv preprint arXiv:2210.16859, 2022. (cited on page 19)

Martin Marek, Sanae Lotfi, Aditya Somasundaram, Andrew Gordon Wilson, and Micah Goldblum. Small batch size training for language models: When vanilla SGD works, and why gradient accumulation is wasteful. In Advances in Neural Information Processing Systems, volume 38, pp. 165262–165287, 2025. (cited on page 8)

Eric Michaud, Ziming Liu, Uzay Girit, and Max Tegmark. The quantization model of neural scaling. Advances in Neural Information Processing Systems, 36:28699–28722, 2023. (cited on page 19)

Enea Monzio Compagnoni, Tianlin Liu, Rustem Islamov, Frank Norbert Proske, Antonio Orvieto, and Aurelien Lucchi. Adaptive methods through the lens of SDEs: Theoretical insights on the role of noise. In The Thirteenth International Conference on Learning Representations, 2025. (cited on pages 3, 4, and 19)

Yoonsoo Nam, Nayara Fonseca, Seok Hyeong Lee, Chris Mingard, and Ard A. Louis. An exactly solvable model for emergence and scaling laws in the multitask sparse parity problem. In Advances in Neural Information Processing Systems, volume 37, pp. 39632–39693, 2024. (cited on page 19)

Bernt Øksendal. Stochastic Differential Equations: An Introduction with Applications. Universitext. Springer, Berlin, Heidelberg, 6 edition, 2003. doi: 10.1007/978-3-642-14394-6. (cited on pages 19 and 21)

Elliot Paquette, Courtney Paquette, Lechao Xiao, and Jeffrey Pennington. 4+3 phases of compute-optimal neural scaling laws. In Advances in Neural Information Processing Systems, volume 37, pp. 16459–16537, 2024. (cited on page 19)

Elliot Paquette, Noah Marshall, Lucas Benigni, Guangyuan Wang, Atish Agarwala, and Courtney Paquette. Phases of Muon: When Muon eclipses SignSGD. arXiv preprint arXiv:2605.09552, 2026. URL https://arxiv.org/abs/2605.09552. (cited on page 3)

Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar. On the convergence of Adam and beyond. In International Conference on Learning Representations, 2018. URL https://openreview. net/forum?id=ryQu7f-RZ. (cited on page 1)

Utkarsh Sharma and Jared Kaplan. Scaling laws from the data manifold dimension. Journal of Machine Learning Research, 23(9):1–34, 2022. (cited on page 19)

Wei Shen, Ruichuan Huang, Minhui Huang, Cong Shen, and Jiawei Zhang. On the convergence analysis of Muon. arXiv preprint arXiv:2505.23737, 2025. URL https://arxiv.org/abs/ 2505.23737. (cited on page 1)

Leslie N Smith. Cyclical learning rates for training neural networks. In 2017 IEEE Winter Conference on Applications of Computer Vision (WACV), pp. 464–472. IEEE, 2017. doi: 10.1109/WACV.2017.58. (cited on page 48)

Teodora Srećković, Jonas Geiping, and Antonio Orvieto. Is your batch size the problem? Revisiting the Adam-SGD gap in language modeling. arXiv preprint arXiv:2506.12543, 2025. URL https://arxiv.org/abs/2506.12543. (cited on page 8)

Ingo Steinwart, Don R Hush, and Clint Scovel. Optimal rates for regularized least squares regression. In COLT, pp. 79–93, 2009. (cited on page 6)

Tao Sun, Qingsong Wang, Dongsheng Li, and Bao Wang. Momentum ensures convergence of signSGD under weaker assumptions. In International Conference on Machine Learning, pp. 33077–33099. PMLR, 2023. (cited on pages 1 and 19)

Benjamin Thérien, Xiaolong Huang, Aaron Defazio, Irina Rish, and Eugene Belilovsky MuLoCo: Muon is a practical inner optimizer for DiLoCo. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pp. 120419–120462. PMLR, 2026. URL https://proceedings.mlr. press/v306/therien26a.html. (cited on page 8)

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. (cited on page 8)

Jia-Nan Wang, Zixun Huang, Kairui Li, Zilin Wang, and Lei Wu. Momentum in large-batch training: Polyak enlarges the critical batch size, Nesterov improves data efficiency. arXiv preprint arXiv:2609.02728, 2026a. (cited on page 44)

Jinbo Wang, Binghui Li, Zhanpeng Zhou, Mingze Wang, Yuxuan Sun, Jiaqi Zhang, Xunliang Cai, and Lei Wu. Fast catch-up, late switching: Optimal batch size scheduling via functional scaling laws. In The Fourteenth International Conference on Learning Representations, 2026b. URL https://openreview.net/forum?id=PXWgzUkVwo. (cited on page 19)

Mingze Wang and Lei Wu. The noise geometry of stochastic gradient descent. Transactions on Machine Learning Research, 2026. URL https://openreview.net/forum?id=F6wmIilZXG. (cited on pages 4 and 22)

Alexander Wei, Wei Hu, and Jacob Steinhardt. More than a toy: Random matrix models predict how real-world neural representations generalize. In International Conference on Machine Learning, pp. 23549–23588. PMLR, 2022. (cited on page 19)

Lei Wu, Mingze Wang, and Weijie Su. The alignment property of SGD noise and how it helps select flat minima: A stability analysis. Advances in Neural Information Processing Systems, 35:4680–4693, 2022. (cited on page 22)

Ke Liang Xiao, Noah Marshall, Atish Agarwala, and Elliot Paquette. Exact risk curves of signSGD in high-dimensions: Quantifying preconditioning and noise-compression effects. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 68391–68439. PMLR, 2025. (cited on pages 3 and 19)

Shuo Xie, Mohamad Amin Mohamadi, and Zhiyuan Li. Adam Exploits l∞-Geometry of Loss Landscape via Coordinate-wise Adaptivity. In The Thirteenth International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2410.08198. (cited on page 50)

Tingkai Yan, Haodong Wen, Binghui Li, Kairong Luo, Wenguang Chen, and Kaifeng Lyu. Larger datasets can be repeated more: A theoretical analysis of multi-epoch scaling in linear regression. In International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=0CXjpAxHUE. (cited on page 19)

Dechen Zhang, Xuan Tang, Yingyu Liang, and Difan Zou. Scaling laws for precision in high-dimensional linear regression. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pp. 158654–158750. PMLR, 2026. URL https://proceedings.mlr.press/v306/zhang26fq. html. (cited on page 19)

Yushun Zhang, Congliang Chen, Tian Ding, Ziniu Li, Ruoyu Sun, and Zhi-Quan Luo. Why transformers need Adam: A Hessian perspective. Advances in Neural Information Processing Systems, 37:131786–131823, 2024. (cited on page 1)

## Appendix

## Table of Contents

A Additional related work 19   
B Background on SDEs 19   
B.1 The generator for expectation estimates 19   
B.2 Time-change formula for SDEs 21   
C The SDE modeling for SGD and SignSGD 22   
C.1 The structure of gradient noise 22   
C.2 The SDE modeling for SGD. 24   
C.3 The SDE modeling for SignSGD 25   
C.4 Properties of the smoothed sign function 26   
C.5 An extension to non-Gaussian gradient noise 27   
D FSL of SignSGD (Proof of Theorem 3.1) 28   
D.1 A technical lemma: power-law spectral sums 28   
D.2 Time rescaling and normalized modal dynamics 29   
D.3 The generator and a global lower bound 30   
D.4 A Foster-Lyapunov upper bound 31   
D.5 Control of the burn-in process 33   
D.6 Completion of the proof 35   
D.7 Discussion: Initialization and relaxation across spectral modes 36   
E Data scaling analysis 37   
E.1 Power decay schedules 37   
E.2 Data scaling for SignSGD (Proof of Theorem 4.1) 37   
E.3 Data scaling of SGD 39   
E.4 Optimality of power decay schedules in data scaling 40   
F Batch-size scaling analysis 41   
G Heuristic extension to RMSprop and Adam 42   
G.1 Second moments and coordinate curvatures 42   
G.2 The effective RMSprop dynamics 43   
G.3 The FSL of RMSprop 43   
G.4 The FSL of Adam 44   
H Experimental details and additional results 45   
H.1 Fitting and prediction using FSL 45   
H.2 PLK regression experiments 47   
H.3 Language model experiments 51

## A Additional related work

Theoretical analysis of SignSGD. Existing studies have primarily focused on convergence guarantees of SignSGD (Bernstein et al., 2018; Sun et al., 2023; Jiang et al., 2024, 2025; Chen et al., 2026). Another line of work derives continuous-time stochastic differential equations (SDEs) for modeling SignSGD dynamics (Malladi et al., 2022; Xiao et al., 2025; Monzio Compagnoni et al., 2025). We build on this continuous-time approach, and use it to derive a trajectory-level functional scaling law for SignSGD.

Theoretical studies of scaling laws. A growing literature seeks theoretical explanations for scaling laws (Sharma & Kaplan, 2022; Maloney et al., 2022; Wei et al., 2022; Jain et al., 2024; Michaud et al., 2023; Nam et al., 2024; Atanasov et al., 2026; Dohmatob et al., 2024; Bahri et al., 2024; Bordelon et al., 2024; Lin et al., 2024; Paquette et al., 2024; Bordelon et al., 2025; Li et al., 2025; Kunstner & Bach, 2025; Ding et al., 2025; Yan et al., 2026; Lin et al., 2025; Wang et al., 2026b; Li et al., 2026c,b; Zhang et al., 2026; Li et al., 2026a), but predominantly analyzes vanilla SGD. In contrast, we study SignSGD, which is closer to optimizers used in large-scale practice, and show that it exhibits qualitatively different scaling behavior. Another related work is Kunstner & Bach (2025), which studies full-batch sign gradient descent (signGD) in a linear bigram model and identifies scaling properties of deterministic sign dynamics. We instead focus on the stochastic regime, where gradient noise plays a central role and interacts nontrivially with sign normalization to shape the scaling behavior.

Comparison with Xiao et al. (2025). Xiao et al. (2025) derive a high-dimensional SDE and a deterministic risk ODE for SignSGD, quantifying diagonal preconditioning and noise reshaping. Their formulation allows time-dependent learning rates, and they also discuss a minibatch extension. Their main high-dimensional approximation theorem assumes covariance eigenvalues bounded above and away from zero uniformly in dimension. We instead study a diagonal power-law spectrum with eigenvalues tending to zero, focusing on the label-noise-dominated SDE model. In this setting, we obtain an explicit FSL with power-law signal-learning and noise-forgetting profiles under general LR schedules. This yields optimal data-scaling rates over all admissible schedules and comparisons of critical batch sizes.

Comparison with Kim et al. (2026). The most closely related work is Kim et al. (2026), which also provides a scaling-law analysis of SignSGD in PLK regression. Our work differs in three main aspects. First, we focus on the label-noise-dominated regime and establish an explicit FSL for general LR schedules in our diagonal SDE model, whereas their general trajectory formulation is implicit and depends on the unknown risk. Second, we characterize the batch-size dependence through the intrinsic clock $\begin{array} { r } { t \ = \ \int _ { 0 } ^ { z } \frac { \eta ( u ) } { \sigma _ { B } } } \end{array}$ du, whereas their theory is restricted to B = 1. Third, we derive the optimal data-scaling rate of SignSGD over all admissible LR schedules and obtain a complete phase diagram comparing the rates of SignSGD and SGD under optimally tuned LR schedules. In contrast, their comparison is restricted to specific schedules.

## B Background on SDEs

This appendix reviews two standard tools for analyzing Itô stochastic differential equations (SDEs): the generator for expectation estimates and a time-change formula. These tools are used in the SDE modeling of Appendix C and in the proof in Appendix D.

## B.1 The generator for expectation estimates

We briefly recall how the generator of an Itô SDE can be used to derive expectation and moment estimates. For the details of this method, one can refer to Sec.7 of Øksendal (2003). We now state a more general version (time-inhomogeneous case) and derive the corresponding results here.

Consider the following SDE

$$
\mathrm { d } \mathbf { Y } _ { t } = \mathbf { b } ( t , \mathbf { Y } _ { t } ) \mathrm { ~ d } t + \mathbf { G } ( t , \mathbf { Y } _ { t } ) \mathrm { ~ d } \mathbf { W } _ { t } , \qquad \mathbf { Y } _ { t } \in \mathbb { R } ^ { n } ,\tag{14}
$$

where $\mathbf { W } _ { t }$ is a d-dimensional Brownian motion, $\mathbf { b } ( t , \mathbf { y } ) \in \mathbb { R } ^ { n }$ , and $\mathbf { G } ( t , \mathbf { y } ) \in \mathbb { R } ^ { n \times d }$

We first recall the time-homogeneous case, in which b and G do not depend explicitly on t. Let $\mathbf { Y } _ { t } ^ { \mathbf { y } }$ denote the solution initialized at $\mathbf { Y } _ { 0 } ^ { \mathbf { y } } = \mathbf { y }$ . The infinitesimal generator is defined by

$$
( \mathcal { L } f ) ( \mathbf { y } ) : = \operatorname* { l i m } _ { h \downarrow 0 } \frac { \mathbb { E } [ f ( \mathbf { Y } _ { h } ^ { \mathbf { y } } ) ] - f ( \mathbf { y } ) } { h } ,\tag{15}
$$

whenever the limit exists. For $f \in C ^ { 2 } ( \mathbb { R } ^ { n } )$ , Itô's formula gives the explicit representation

$$
( \mathcal { L } f ) ( \mathbf { y } ) = \mathbf { b } ( \mathbf { y } ) ^ { \top } \nabla f ( \mathbf { y } ) + \frac { 1 } { 2 } \operatorname { t r } \Big ( \mathbf { G } ( \mathbf { y } ) \mathbf { G } ( \mathbf { y } ) ^ { \top } \nabla ^ { 2 } f ( \mathbf { y } ) \Big ) .\tag{16}
$$

For the time-inhomogeneous SDE (14), we define the instantaneous generator at time t by

$$
( \mathcal { L } _ { t } f ) ( \mathbf { y } ) : = \mathbf { b } ( t , \mathbf { y } ) ^ { \top } \nabla f ( \mathbf { y } ) + \frac { 1 } { 2 } \operatorname { t r } \Big ( \mathbf { G } ( t , \mathbf { y } ) \mathbf { G } ( t , \mathbf { y } ) ^ { \top } \nabla ^ { 2 } f ( \mathbf { y } ) \Big ) .\tag{17}
$$

Under the standard Markov regularity conditions, this operator agrees with the infinitesimal conditional expectation

$$
( \ L _ { t } f ) ( \mathbf { y } ) = \operatorname* { l i m } _ { h \downarrow 0 } { \frac { \mathbb { E } [ f ( \mathbf { Y } _ { t + h } ) \mid \mathbf { Y } _ { t } = \mathbf { y } ] - f ( \mathbf { y } ) } { h } } ,\tag{18}
$$

whenever the limit exists. Thus, $\mathcal { L } _ { t } f$ describes the instantaneous rate of change of $f ( \mathbf { Y } _ { t } )$ after the stochastic fluctuation has been averaged out.

Applying Itô's formula to (14) gives

$$
f ( \mathbf { Y } _ { t } ) = f ( \mathbf { Y } _ { 0 } ) + \int _ { 0 } ^ { t } ( \mathcal { L } _ { s } f ) ( \mathbf { Y } _ { s } ) \mathrm { ~ d } s + \int _ { 0 } ^ { t } \nabla f ( \mathbf { Y } _ { s } ) ^ { \top } \mathbf { G } ( s , \mathbf { Y } _ { s } ) \mathrm { ~ d } \mathbf { W } _ { s } .\tag{19}
$$

Under suitable integrability conditions, the stochastic integral has zero expectation. Therefore, the generator $\mathcal { L } _ { t } f$ determines the evolution of expectation $\mathbb { E } [ f ( \mathbf { Y } _ { t } ) ]$

The following lemma states Dynkin's formula and the upper and lower expectation bounds obtained from generator inequalities. Its proof combines $\mathrm { I t o } ^ { \prime } \mathrm { s }$ formula with an integrating factor and also applies to time-inhomogeneous coefficients.

Lemma B.1 (Dynkin's formula and expectation bounds). Let $\mathbf { Y } _ { t }$ satisfy (14), and let $V \in$ $C ^ { 2 } ( \mathbb { R } ^ { n } ; [ 0 , \infty ) )$ . Assume that $\mathrm { s u p } _ { t \leqslant T } V ( \mathbf { Y } _ { t } )$ and $\begin{array} { r } { \int _ { 0 } ^ { T } | ( \mathcal { L } _ { t } V ) ( \mathbf { Y } _ { t } ) | } \end{array}$ dt are integrable. Then

$$
\mathbb { E } [ V ( \mathbf { Y } _ { t } ) ] = \mathbb { E } [ V ( \mathbf { Y } _ { 0 } ) ] + \int _ { 0 } ^ { t } \mathbb { E } [ ( \mathcal { L } _ { s } V ) ( \mathbf { Y } _ { s } ) ] ~ \mathrm { d } s .\tag{20}
$$

Suppose, in addition, that there exist $c _ { V } > 0$ and a nonnegative function $r : [ 0 , T ] \times \mathbb { R } ^ { n } \to [ 0 , \infty )$ such that

$$
( \mathcal { L } _ { t } V ) ( \mathbf { y } ) \leqslant - c _ { V } V ( \mathbf { y } ) + r ( t , \mathbf { y } )\tag{21}
$$

for every $t \in [ 0 , T ]$ and $\mathbf { y } \in \mathbb { R } ^ { n }$ . Then

$$
\mathbb { E } [ V ( \mathbf { Y } _ { t } ) ] \leqslant e ^ { - c _ { V } t } \mathbb { E } [ V ( \mathbf { Y } _ { 0 } ) ] + \int _ { 0 } ^ { t } e ^ { - c _ { V } ( t - s ) } \mathbb { E } [ r ( s , \mathbf { Y } _ { s } ) ] { \mathrm { ~ d } } s .\tag{22}
$$

If instead the generator satisfies

$$
( \mathcal { L } _ { t } V ) ( \mathbf { y } ) \geqslant - c _ { V } V ( \mathbf { y } ) + r ( t , \mathbf { y } )\tag{23}
$$

for every $t \in [ 0 , T ]$ and $\mathbf { y } \in \mathbb { R } ^ { n }$ , then

$$
\mathbb { E } [ V ( \mathbf { Y } _ { t } ) ] \geqslant e ^ { - c _ { V } t } \mathbb { E } [ V ( \mathbf { Y } _ { 0 } ) ] + \int _ { 0 } ^ { t } e ^ { - c _ { V } ( t - s ) } \mathbb { E } [ r ( s , \mathbf { Y } _ { s } ) ] { \mathrm { ~ d } } s .\tag{24}
$$

Proof. For $R > 0$ , define the stopping time

$$
\rho _ { R } : = \operatorname* { i n f } \left\{ t \geqslant 0 : \int _ { 0 } ^ { t } \left\| \nabla V ( \mathbf { Y } _ { s } ) ^ { \top } \mathbf { G } ( s , \mathbf { Y } _ { s } ) \right\| ^ { 2 } \mathrm { d } s \geqslant R \right\} .
$$

Applying Itô's formula to $V ( \mathbf { Y } _ { t \wedge \rho _ { R } } )$ gives

$$
\begin{array} { r l r } {  { V ( \mathbf { Y } _ { t \wedge \rho _ { R } } ) = V ( \mathbf { Y } _ { 0 } ) + \int _ { 0 } ^ { t \wedge \rho _ { R } } ( \mathscr { L } _ { s } V ) ( \mathbf { Y } _ { s } ) \ \mathrm { d } s } } \\ & { } & { + \int _ { 0 } ^ { t \wedge \rho _ { R } } \nabla V ( \mathbf { Y } _ { s } ) ^ { \top } \mathbf { G } ( s , \mathbf { Y } _ { s } ) \ \mathrm { d } \mathbf { W } _ { s } . } \end{array}
$$

The stopped stochastic integral is a martingale and therefore has zero expectation. Hence,

$$
\mathbb { E } [ V ( \mathbf { Y } _ { t \wedge \rho _ { R } } ) ] = \mathbb { E } [ V ( \mathbf { Y } _ { 0 } ) ] + \mathbb { E } \left[ \int _ { 0 } ^ { t \wedge \rho _ { R } } ( \mathcal { L } _ { s } V ) ( \mathbf { Y } _ { s } ) \ \mathrm { d } s \right] .
$$

Letting $R \to \infty$ under the stated integrability assumptions and applying Fubini's theorem gives (20). If we further assume (21), differentiating (20) gives, for almost every t,

$$
\frac { \mathrm { d } } { \mathrm { d } t } \mathbb { E } [ V ( \mathbf { Y } _ { t } ) ] \leqslant - c _ { V } \mathbb { E } [ V ( \mathbf { Y } _ { t } ) ] + \mathbb { E } [ r ( t , \mathbf { Y } _ { t } ) ] .
$$

Multiplying by $e ^ { c _ { V } t }$ and integrating from 0 to t proves (22). Under (23), the same argument with both inequalities reversed proves (24). □

To obtain a moment bound, we choose a test function V, establish a pointwise inequality for $\mathcal { L } _ { t } V$ , and apply the corresponding expectation bound in Lemma B.1. This strategy is used in Appendix D. Specifically, Corollary D.3 combines the quadratic drift estimate in Lemma D.2 with the lower bound (24). Corollary D.6 combines the Lyapunov drift estimate in Lemma D.5 with the upper bound (22).

## B.2 Time-change formula for SDEs

We next recall how an SDE transforms under a deterministic time change. This follows from the standard time-change formula for Itô integrals; see Sec.8.5 of Øksendal (2003). To make the presentation self-contained, we state and prove the version needed here.

Lemma B.2 (Deterministic time change for SDEs). Let $\Gamma : [ 0 , Z ]  [ 0 , T ]$ be a continuously differentiable bijection satisfying $\Gamma ( 0 ) = 0$ and $\Gamma ^ { \prime } ( z ) ~ > ~ 0 ~ f o r ~ a l l ~ z ~ \in ~ [ 0 , Z ]$ . Suppose that $( \tilde { \mathbf { Y } } _ { z } ) _ { z \in [ 0 , Z ] }$ satises

$$
\mathrm { d } \widetilde { \mathbf { Y } } _ { z } = \mathbf { a } ( z , \widetilde { \mathbf { Y } } _ { z } ) \mathrm { ~ d } z + \mathbf { C } ( z , \widetilde { \mathbf { Y } } _ { z } ) \mathrm { ~ d } \mathbf { W } _ { z } ,\tag{25}
$$

where $\mathbf { W } _ { z }$ is a d-dimensional Brownian motion. Define

$$
\overline { { \mathbf { Y } } } _ { t } : = \widetilde { \mathbf { Y } } _ { \Gamma ^ { - 1 } ( t ) } , \qquad t \in [ 0 , T ] .
$$

Then there exists a d-dimensional Brownian motion $\overline { { \mathbf { W } } } _ { t }$ such that

$$
\begin{array} { r } { \mathrm { d } \overline { \mathbf { Y } } _ { t } = \big ( \Gamma ^ { - 1 } \big ) ^ { \prime } ( t ) \mathbf { a } \big ( \Gamma ^ { - 1 } ( t ) , \overline { \mathbf { Y } } _ { t } \big ) \mathrm { ~ d } t + \sqrt { \big ( \Gamma ^ { - 1 } \big ) ^ { \prime } ( t ) } \mathbf { C } \big ( \Gamma ^ { - 1 } ( t ) , \overline { \mathbf { Y } } _ { t } \big ) \mathrm { ~ d } \overline { \mathbf { W } } _ { t } . } \end{array}\tag{26}
$$

Proof. Write (25) in integral form. For $t \in [ 0 , T ]$

$$
\overline { { \mathbf { Y } } } _ { t } = \widetilde { \mathbf { Y } } _ { 0 } + \int _ { 0 } ^ { \Gamma ^ { - 1 } ( t ) } \mathbf { a } ( u , \widetilde { \mathbf { Y } } _ { u } ) \mathrm { d } u + \int _ { 0 } ^ { \Gamma ^ { - 1 } ( t ) } \mathbf { C } ( u , \widetilde { \mathbf { Y } } _ { u } ) \mathrm { d } \mathbf { W } _ { u } .\tag{27}
$$

For the drift term, using the change of variables $u = \Gamma ^ { - 1 } ( s )$ , we have

$$
\int _ { 0 } ^ { \Gamma ^ { - 1 } ( t ) } \mathbf { a } ( u , \widetilde { \mathbf { Y } } _ { u } ) \mathrm { d } u = \int _ { 0 } ^ { t } ( \Gamma ^ { - 1 } ) ^ { \prime } ( s ) \mathbf { a } ( \Gamma ^ { - 1 } ( s ) , \overline { { \mathbf { Y } } } _ { s } ) \mathrm { d } s .\tag{28}
$$

To make the change of the stochastic integral explicit, define

$$
\overline { { \mathbf { W } } } _ { t } : = \int _ { 0 } ^ { \Gamma ^ { - 1 } ( t ) } \sqrt { \Gamma ^ { \prime } ( u ) } ~ \mathrm { d } \mathbf { W } _ { u } .
$$

We claim that $\overline { { \mathbf { W } } }$ is a standard Brownian motion in the new time variable t. Indeed, for $0 \leqslant s < t .$

$$
\overline { { \mathbf { W } } } _ { t } - \overline { { \mathbf { W } } } _ { s } = \int _ { \Gamma ^ { - 1 } ( s ) } ^ { \Gamma ^ { - 1 } ( t ) } \sqrt { \Gamma ^ { \prime } ( u ) } ~ \mathrm { d } \mathbf { W } _ { u } .
$$

This increment is Gaussian with mean zero and covariance

$$
\int _ { \Gamma ^ { - 1 } ( s ) } ^ { \Gamma ^ { - 1 } ( t ) } \Gamma ^ { \prime } ( u ) \mathrm { d } u \mathbf { I } _ { d } = \left( \Gamma ( \Gamma ^ { - 1 } ( t ) ) - \Gamma ( \Gamma ^ { - 1 } ( s ) ) \right) \mathbf { I } _ { d } = ( t - s ) \mathbf { I } _ { d } .
$$

Moreover, increments over disjoint intervals in t are independent because they are stochastic integrals over disjoint intervals in the original Brownian time u. Thus,

$$
\overline { { \mathbf { W } } } _ { t } - \overline { { \mathbf { W } } } _ { s } \sim \mathcal { N } ( 0 , ( t - s ) \mathbf { I } _ { d } )
$$

and has independent increments relative to the time-changed filtration. Therefore, W is a standard d-dimensional Brownian motion in the new time variable.

Using $\begin{array} { r } { ( \Gamma ^ { - 1 } ) ^ { \prime } ( t ) = \frac { 1 } { \Gamma ^ { \prime } ( \Gamma ^ { - 1 } ( t ) ) } } \end{array}$ , the deterministic time-change formula for Itô integrals gives

$$
\int _ { 0 } ^ { \Gamma ^ { - 1 } ( t ) } \mathbf { C } ( u , \widetilde { \mathbf { Y } } _ { u } ) ~ \mathrm { d } \mathbf { W } _ { u } = \int _ { 0 } ^ { t } \sqrt { ( \Gamma ^ { - 1 } ) ^ { \prime } ( s ) } \mathbf { C } ( \Gamma ^ { - 1 } ( s ) , \overline { { \mathbf { Y } } } _ { s } ) ~ \mathrm { d } \overline { { \mathbf { W } } } _ { s } .\tag{29}
$$

Combining (27), (28), and (29), we obtain

$$
\overline { { \mathbf { Y } } } _ { t } = \overline { { \mathbf { Y } } } _ { 0 } + \int _ { 0 } ^ { t } ( \Gamma ^ { - 1 } ) ^ { \prime } ( s ) \mathbf { a } ( \Gamma ^ { - 1 } ( s ) , \overline { { \mathbf { Y } } } _ { s } ) \mathrm { d } s + \int _ { 0 } ^ { t } \sqrt { ( \Gamma ^ { - 1 } ) ^ { \prime } ( s ) } \mathbf { C } ( \Gamma ^ { - 1 } ( s ) , \overline { { \mathbf { Y } } } _ { s } ) \mathrm { d } \overline { { \mathbf { W } } } _ { s } .
$$

This is precisely the integral form of (26).

## C The SDE modeling for SGD and SignSGD

Throughout this appendix, z denotes physical time measured in training steps, and $\eta ( z )$ is a continuous learning rate schedule satisfying $\eta ( k ) = \eta _ { k }$

## C.1 The structure of gradient noise

We first characterize the covariance structure of the gradient noise. The Hessian-aligned structure of SGD noise has been observed and analyzed in prior works on SGD noise geometry (Wu et al., 2022; Wang & Wu, 2026). For completeness, we provide a self-contained derivation in our kernel regression model.

We use the batch noise scale $\sigma _ { B } = 1 / \sqrt { B }$ from Section 2. Let B be a fresh mini-batch of B i.i.d. samples and define its gradient by

$$
\mathbf { g } ( \pmb \theta ) = \frac { 1 } { B } \sum _ { ( \mathbf { x } , y ) \in B } \nabla _ { \pmb \theta } \ell ( \pmb \theta ; \mathbf { x } , y ) .
$$

Define the gradient noise by $\pmb { \xi } ( \pmb { \theta } ) : = \mathbf { g } ( \pmb { \theta } ) - \nabla \mathcal { R } ( \pmb { \theta } )$ . Then $\mathbb { E } _ { B } [ \pmb { \xi } ( \pmb { \theta } ) ] = \mathbf { 0 }$ . At step k, evaluating ${ \bf g } ( \pmb \theta )$ and $\xi ( \theta )$ at $\pmb { \theta } _ { k }$ using the fresh batch $\boldsymbol { B } _ { k }$ gives gk and ${ \boldsymbol { \xi } } _ { k }$ , respectively. The following theorem shows that the covariance of this noise is aligned with the Hessian.

Theorem C.1 (Gradient-noise covariance). Consider the kernel regression model in Section 2. The mini-batch and single-sample gradient-noise covariances satisfy

$$
\mathbb { E } _ { \boldsymbol { B } } [ \pmb { \xi } ( \pmb { \theta } ) \pmb { \xi } ( \pmb { \theta } ) ^ { \top } ] = \sigma _ { B } ^ { 2 } \pmb { \Sigma } ( \pmb { \theta } ) , \qquad \pmb { \Sigma } ( \pmb { \theta } ) \mp \big ( \mathcal { E } ( \pmb { \theta } ) + \sigma ^ { 2 } \big ) \mathbf { H } .
$$

Here and below, for two positive semidefinite matrices A and $\mathbf { B } ,$ we write $\mathbf { A }  \mathbf { B }$ if their quadratic forms are equivalent up to universal constants, i.e., there exist constants $c _ { 1 } , c _ { 2 } > 0 ;$ independent of $\mathbf { v } ,$ such that $c _ { 1 } \mathbf { v } ^ { \top } \mathbf { B } \mathbf { v } \leqslant \mathbf { v } ^ { \top } \mathbf { A } \mathbf { v } \leqslant c _ { 2 } \mathbf { v } ^ { \top } \mathbf { B } \mathbf { v }$

Therefore, for any unit direction $\mathbf { v } _ { \mathrm { : } }$

$$
\begin{array} { r } { \mathbb { E } _ { \boldsymbol { B } } [ \vert \boldsymbol { \xi } ( \boldsymbol { \theta } ) ^ { \top } { \bf v } \vert ^ { 2 } ] = \sigma _ { B } ^ { 2 } { \bf v } ^ { \top } \boldsymbol { \Sigma } ( \boldsymbol { \theta } ) { \bf v }  \sigma _ { B } ^ { 2 } \big ( \boldsymbol { \mathcal { E } } ( \boldsymbol { \theta } ) + \boldsymbol { \sigma } ^ { 2 } \big ) { \bf v } ^ { \top } { \bf H } { \bf v } . } \end{array}
$$

Since $\mathbf { v } ^ { \top } \mathbf { H } \mathbf { v }$ is the curvature along $\mathbf { v } ,$ the gradient noise variance in any direction is proportional to the curvature in that direction (up to constants). In the label-noise-dominated regime $\mathcal { E } ( \pmb { \theta } ) \lesssim$ $\sigma ^ { 2 }$ , this simplifies to

$$
\begin{array} { r } { \pmb { \Sigma } ( \pmb { \theta } ) \approx \mathbf { H } , \qquad \mathbb { E } _ { \pmb { \mathcal { B } } } [ | \pmb { \xi } ( \pmb { \theta } ) ^ { \top } \mathbf { v } | ^ { 2 } ] \approx \sigma _ { B } ^ { 2 } \mathbf { v } ^ { \top } \mathbf { H } \mathbf { v } . } \end{array}
$$

This is due to our assumption $\sigma ^ { 2 } \lesssim 1$ . Thus, in coordinate $j ,$ the mini-batch noise has standard deviation of order $\sigma _ { B } \sqrt { \lambda _ { j } }$ . This coordinate-wise noise scale motivates the normalized noise model used below for the SignSGD scaling analysis.

Proof of Theorem C.1. For a mini-batch of size $B ,$ write $\begin{array} { r } { \pmb { \xi } ( \pmb { \theta } ) = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \zeta _ { i } ( \pmb { \theta } ) } \end{array}$ , where each $\zeta _ { i } ( \pmb \theta )$ is the gradient noise over a single data point. Then

$$
\mathbb { E } [ \xi ( \theta ) \xi ( \theta ) ^ { \top } ] = \frac { 1 } { B ^ { 2 } } \sum _ { i , j = 1 } ^ { B } \mathbb { E } [ \zeta _ { i } ( \theta ) \zeta _ { j } ( \theta ) ^ { \top } ] = \frac { 1 } { B ^ { 2 } } \sum _ { i = 1 } ^ { B } \mathbb { E } [ \zeta _ { i } ( \theta ) \zeta _ { i } ( \theta ) ^ { \top } ] = \frac { 1 } { B } \pmb { \Sigma } ( \theta ) = \sigma _ { B } ^ { 2 } \pmb { \Sigma } ( \theta ) .
$$

It remains to compute $\Sigma ( \theta )$ . Consider a single data point $\left( \mathbf { x } , y \right)$ and let $\mathbf { g } _ { \mathbf { x } , y } ( \pmb { \theta } )$ be its gradient. Let $\pmb { \delta } = \pmb { \theta } - \pmb { \theta } ^ { * } , \pmb { \phi } = \pmb { \phi } ( \mathbf { x } )$ . Since $y = \left. \phi , \theta ^ { * } \right. + \epsilon .$ we have

$$
\mathbf { g } _ { \mathbf { x } , y } ( \pmb { \theta } ) = \nabla _ { \pmb { \theta } } \left( \frac { 1 } { 2 } ( f ( \mathbf { x } ; \pmb { \theta } ) - y ) ^ { 2 } \right) = \big ( \langle \pmb { \phi } , \pmb { \theta } \rangle - y \big ) \pmb { \phi } = \big ( \langle \pmb { \phi } , \pmb { \delta } \rangle - \epsilon \big ) \pmb { \phi } .
$$

The noise € and feature $\phi$ are independent and satisfy $\mathbb { E } [ \boldsymbol { \epsilon } ] = 0 , \mathbb { E } [ \boldsymbol { \phi } \boldsymbol { \phi } ^ { \intercal } ] = \mathbf { H }$ . Taking expectations over $\left( \mathbf { x } , y \right)$ , we obtain

$$
\nabla \mathcal { R } ( \pmb \theta ) = \mathbb { E } _ { \mathbf { x } , y } [ \mathbf { g } _ { \mathbf { x } , y } ( \pmb \theta ) ] = \mathbf { H } \delta .
$$

Thus, the single-sample gradient noise is

$$
\zeta ( \pmb \theta ) = \mathbf { g } _ { \mathbf { x } , y } ( \pmb \theta ) - \nabla \mathcal { R } ( \pmb \theta ) = ( \phi \phi ^ { \top } - \mathbf H ) \pmb \delta - \epsilon \pmb \phi .
$$

We next compute its covariance:

$$
\pmb { \Sigma } ( \pmb { \theta } ) = \mathbb { E } _ { \mathbf { x } , y } [ \zeta ( \pmb { \theta } ) \zeta ( \pmb { \theta } ) ^ { \top } ] = \mathbb { E } \left[ ( \pmb { \phi } \pmb { \phi } ^ { \top } - \mathbf { H } ) \delta \pmb { \delta } ^ { \top } ( \pmb { \phi } \pmb { \phi } ^ { \top } - \mathbf { H } ) \right] + \sigma ^ { 2 } \mathbb { E } [ \phi \pmb { \phi } ^ { \top } ] ,
$$

where the cross terms vanish because ε is independent of $\phi$ and $\mathbb { E } [ \epsilon ] = 0$ . The second term equals $\sigma ^ { 2 } \mathbf { H }$ . For the first term, we have

$$
\mathbb { E } \left[ ( \phi \phi ^ { \top } - \mathbf { H } ) \delta \delta ^ { \top } ( \phi \phi ^ { \top } - \mathbf { H } ) \right] = \mathbb { E } \left[ ( \delta ^ { \top } \phi ) ^ { 2 } \phi \phi ^ { \top } \right] - \mathbf { H } \delta \delta ^ { \top } \mathbf { H } = ( \delta ^ { \top } \mathbf { H } \delta ) \mathbf { H } + \mathbf { H } \delta \delta ^ { \top } \mathbf { H } ,
$$

where we use the Gaussian fourth-moment identity 1 $\mathbb { E } \left[ ( \pmb { \delta } ^ { \top } \phi ) ^ { 2 } \phi \phi ^ { \top } \right] = ( \pmb { \delta } ^ { \top } \mathbf { H } \pmb { \delta } ) \mathbf { H } + 2 \mathbf { H } \pmb { \delta } \pmb { \delta } ^ { \top } \mathbf { H }$ Thus, we obtain the noise covariance of single-sample gradient

$$
\pmb { \Sigma } ( \pmb { \theta } ) = \big ( \sigma ^ { 2 } + \delta ^ { \top } \mathbf { H } \delta \big ) \mathbf { H } + \mathbf { H } \delta \delta ^ { \top } \mathbf { H } .
$$

We claim that, uniformly over all $\mathbf { v } \in \mathbb { R } ^ { d }$

$$
\mathbf { v } ^ { \top } \Sigma ( \pmb { \theta } ) \mathbf { v } \approx ( \mathbf { \mathcal { E } } ( \pmb { \theta } ) + \sigma ^ { 2 } ) \mathbf { v } ^ { \top } \mathbf { H } \mathbf { v } .
$$

Indeed, using the covariance formula above,

$$
\mathbf { v } ^ { \top } \Sigma ( \pmb { \theta } ) \mathbf { v } = \left( \sigma ^ { 2 } + \delta ^ { \top } \mathbf { H } \delta \right) \mathbf { v } ^ { \top } \mathbf { H } \mathbf { v } + ( \delta ^ { \top } \mathbf { H } \mathbf { v } ) ^ { 2 } .
$$

The lower bound follows immediately by dropping the nonnegative second term: ${ \bf v } ^ { \top } \pmb { \Sigma } ( \pmb { \theta } ) { \bf v } \geqslant$ $\begin{array} { r } { \big ( \sigma ^ { 2 } + \delta ^ { \top } \mathbf H \delta \big ) \mathbf v ^ { \top } \mathbf H \mathbf v } \end{array}$ . For the upper bound, the Cauchy-Schwarz inequality gives $( \delta ^ { \top } \mathbf { H } \mathbf { v } ) ^ { 2 } \leqslant$ $\dot { ( \delta ^ { \top } } \mathbf { H } \delta ) ( \mathbf { v } ^ { \top } \dot { \mathbf { H } } \mathbf { v } )$ . Therefore,

$$
\mathbf { v } ^ { \top } \Sigma ( \pmb { \theta } ) \mathbf { v } \leqslant \big ( \sigma ^ { 2 } + 2 \delta ^ { \top } \mathbf { H } \delta \big ) \mathbf { v } ^ { \top } \mathbf { H } \mathbf { v } .
$$

Using $\begin{array} { r } { \mathcal { E } ( \pmb { \theta } ) = \frac { 1 } { 2 } \pmb { \delta } ^ { \top } } \end{array}$ Hδ and combining this with the lower bound, we obtain

$$
\big ( \mathcal E ( \pmb \theta ) + \sigma ^ { 2 } \big ) \mathbf v ^ { \top } \mathbf H \mathbf v \leqslant \mathbf v ^ { \top } \pmb \Sigma ( \pmb \theta ) \mathbf v \leqslant 4 \big ( \mathcal E ( \pmb \theta ) + \sigma ^ { 2 } \big ) \mathbf v ^ { \top } \mathbf H \mathbf v .
$$

This proves

$$
\Sigma ( \pmb \theta ) \approx ( \mathcal E ( \pmb \theta ) + \sigma ^ { 2 } ) \mathbf { H } .
$$

Remark C.2 (Beyond Gaussian features). Gaussianity is not needed for curvature alignment in the label-noise-dominated regime. Suppose the hypercontractivity condition $\mathbb { E } [ ( \mathbf { v } ^ { \top } \phi ) ^ { 4 } ] \leqslant$ $C ( \mathbf { v } ^ { \top } \mathbf { H } \mathbf { v } ) ^ { 2 }$ holds for every $\textbf { v } \in \mathbb { R } ^ { d }$ , with a constant C independent of v and d. A similar argument gives

$$
\sigma _ { B } ^ { 2 } \sigma ^ { 2 } \mathbf { H } \lesssim \sigma _ { B } ^ { 2 } \pmb { \Sigma } ( \pmb { \theta } ) \lesssim \sigma _ { B } ^ { 2 } \big ( \sigma ^ { 2 } + \pmb { \mathcal { E } } ( \pmb { \theta } ) \big ) \mathbf { H } .
$$

This leads to the same result $\sigma _ { B } ^ { 2 } \mathbf { \Sigma } \Sigma ( \pmb { \theta } ) \approx \sigma _ { B } ^ { 2 } \mathbf { H }$ in the label-noise-dominated regime $\mathcal { E } ( \pmb { \theta } ) \lesssim \sigma ^ { 2 } \approx$ 1.

## C.2 The SDE modeling for SGD

We briefly review the SDE modeling for SGD in Li et al. (2025). The SGD update is

$$
\pmb { \theta } _ { k + 1 } = \pmb { \theta } _ { k } - \eta _ { k } \mathbf { g } _ { k } .\tag{30}
$$

We decompose the mini-batch gradient as $\mathbf { g } _ { k } = \nabla \mathcal { R } ( \pmb { \theta } _ { k } ) + \pmb { \xi } _ { k }$ . By Theorem C.1, its conditional moments satisfy

$$
\mathbb { E } [ \pmb { \xi } _ { k } \ | \ \pmb { \theta } _ { k } ] = \mathbf { 0 } , \qquad \mathbb { E } [ \pmb { \xi } _ { k } \pmb { \xi } _ { k } ^ { \top } \ | \ \pmb { \theta } _ { k } ] = \sigma _ { B } ^ { 2 } \pmb { \Sigma } ( \pmb { \theta } _ { k } ) .
$$

Here $\Sigma ( \theta _ { k } )$ denotes the covariance of a single-sample stochastic gradient.

Therefore, one SGD step can be written as $\pmb { \theta } _ { k + 1 } - \pmb { \theta } _ { k } = - \eta _ { k } \nabla \mathcal { R } ( \pmb { \theta } _ { k } ) - \eta _ { k } \pmb { \xi } _ { k }$ , consisting of a drift term $- \eta _ { k } \nabla \mathcal { R } ( \theta _ { k } )$ and a fluctuation term with covariance $\eta _ { k } ^ { 2 } \pmb { \Sigma } ( \pmb { \theta } _ { k } ) / B$ . This motivates the physical-time SDE:

$$
\mathrm { d } \widetilde { \pmb { \theta } } _ { z } = - \eta ( z ) \nabla \mathcal { R } ( \widetilde { \pmb { \theta } } _ { z } ) \ \mathrm { d } z + \frac { \eta ( z ) } { \sqrt { B } } \pmb { \Sigma } ( \widetilde { \pmb { \theta } } _ { z } ) ^ { 1 / 2 } \ \mathrm { d } \mathbf { W } _ { z } ,\tag{31}
$$

where $\eta ( z )$ is the continuous-time LRS satisfying $\eta ( k ) = \eta _ { k }$ for $k \in \mathbb N$ , and z denotes the physical time, which is a continuous analogue of the training step. To account for the influence of LRS, we define the intrinsic time $\begin{array} { r } { t = \Gamma _ { \mathrm { s g d } } ( z ) : = \int _ { 0 } ^ { z } \eta ( u ) } \end{array}$ du, which can be viewed as the continuous-time analogue of $\textstyle \sum _ { j = 0 } ^ { k - 1 } \eta _ { j }$ . Let $\vartheta _ { t } = \widetilde { \pmb { \theta } } _ { \Gamma _ { \mathrm { s g d } } ^ { - 1 } ( t ) }$ Since $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } \Gamma _ { \mathrm { s g d } } ^ { - 1 } ( t ) = \frac { 1 } { \eta ( \Gamma _ { \mathrm { s g d } } ^ { - 1 } ( t ) ) } } \end{array}$ , applying the time-change formula in Lemma B.2 with $\begin{array} { r } { { \bf a } ( z , \pmb { \theta } ) = - \eta ( z ) \nabla \mathcal { R } ( \pmb { \theta } ) , { \bf C } ( z , \pmb { \theta } ) = \frac { \eta ( z ) } { \sqrt { B } } \pmb { \Sigma } ( \pmb { \theta } ) ^ { 1 / 2 } } \end{array}$ yields the intrinsic-time SDE

$$
\mathrm { d } \pmb { \vartheta } _ { t } = - \nabla \mathcal { R } ( \pmb { \vartheta } _ { t } ) \mathrm { d } t + \sqrt { \frac { \varphi _ { \mathrm { s g d } } ( t ) } { B } } \pmb { \Sigma } ( \pmb { \vartheta } _ { t } ) ^ { 1 / 2 } \mathrm { d } \mathbf { W } _ { t } ,\tag{32}
$$

where $\varphi _ { \mathrm { s g d } } ( t ) : = \eta ( \Gamma _ { \mathrm { s g d } } ^ { - 1 } ( t ) )$ is the learning rate schedule in intrinsic time. By the time change, we decouple the effect of the learning rate from the drift term.

Here $t = t _ { \mathrm { s g d } } ( z ) = \Gamma _ { \mathrm { s g d } } ( z )$ , and the noise injection intensity is $n _ { \mathrm { s g d } } ( t ) = \varphi _ { \mathrm { s g d } } ( t ) / B$

## C.3 The SDE modeling for SignSGD

We now derive the SDE modeling for SignSGD. The SignSGD update is

$$
\begin{array} { r } { \pmb { \theta } _ { k + 1 } = \pmb { \theta } _ { k } - \eta _ { k } \mathrm { s g n } ( \mathbf { g } _ { k } ) . } \end{array}\tag{33}
$$

As in Section 2, write $\mathbf { g } _ { k } = \pmb { \mu } _ { k } + \pmb { \xi } _ { k }$ , where $\pmb { \mu } _ { k } = \nabla \mathcal { R } ( \pmb { \theta } _ { k } )$ , and use the fixed-covariance Gaussian model $\pmb { \xi } _ { k } \sim \mathcal { N } ( \mathbf { 0 } , \sigma _ { B } ^ { 2 } \pmb { \Sigma } )$ . Recall the scaled smoothed sign operator for batch size B:

$$
\begin{array} { r } { \mathbf { m } _ { B } ( \mathbf { x } ) : = \sigma _ { B } \mathbb { E } _ { \pmb { \xi } \sim \mathcal { N } ( \mathbf { 0 } , \pmb { \Sigma } ) } \big [ \mathrm { s g n } ( \mathbf { x } + \sigma _ { B } \pmb { \xi } ) \big ] . } \end{array}
$$

Then

$$
\mathbb { E } [ \mathrm { s g n } ( \mathbf { g } _ { k } ) \mid \pmb { \theta } _ { k } ] = \mathbf { m } _ { B } ( \pmb { \mu } _ { k } ) / \sigma _ { B } .
$$

Define the centered sign noise and its covariance by

$$
\mathbf { n } _ { k } : = \mathrm { s g n } ( \mathbf { g } _ { k } ) - \mathbf { m } _ { B } ( \pmb { \mu } _ { k } ) / \sigma _ { B } , \qquad \mathbf { Q } _ { B } ( \pmb { \theta } _ { k } ) : = \mathbb { E } [ \mathbf { n } _ { k } \mathbf { n } _ { k } ^ { \top } \mathbf { \sigma } \ | \mathbf { \theta } _ { k } ] .
$$

Thus the update reads

$$
\pmb { \theta } _ { k + 1 } - \pmb { \theta } _ { k } = - \frac { \eta _ { k } } { \sigma _ { B } } \mathbf { m } _ { B } ( \nabla \mathcal { R } ( \pmb { \theta } _ { k } ) ) - \eta _ { k } \mathbf { n } _ { k } .
$$

Matching its conditional mean and covariance gives the physical-time model

$$
\mathrm { d } \widetilde { \pmb { \theta } } _ { z } = - \frac { \eta ( z ) } { \sigma _ { B } } \mathbf { m } _ { B } ( \nabla \mathcal { R } ( \widetilde { \pmb { \theta } } _ { z } ) ) \mathrm { d } z + \eta ( z ) \mathbf { Q } _ { B } ( \widetilde { \pmb { \theta } } _ { z } ) ^ { 1 / 2 } \mathrm { d } \mathbf { W } _ { z } .\tag{34}
$$

Define the SignSGD intrinsic clock, its intrinsic-time learning rate, and its noise injection intensity by

$$
\begin{array} { r l } & { \displaystyle t = \Gamma _ { B } ( z ) : = \int _ { 0 } ^ { z } \frac { \eta ( u ) } { \sigma _ { B } } \mathrm { d } u , } \\ & { \displaystyle \varphi ( t ) = \varphi _ { \mathrm { s g n } } ( t ) : = \eta ( \Gamma _ { B } ^ { - 1 } ( t ) ) , \qquad n _ { B } ( t ) : = \sigma _ { B } \varphi ( t ) . } \end{array}
$$

Set $\vartheta _ { t } = \widetilde { \pmb { \theta } } _ { \Gamma _ { B } ^ { - 1 } ( t ) }$ . Applying Lemma B.2 with

$$
\mathbf { a } ( z , \pmb { \theta } ) = - \frac { \eta ( z ) } { \sigma _ { B } } \mathbf { m } _ { B } ( \nabla \mathcal { R } ( \pmb { \theta } ) ) , \qquad \mathbf { C } ( z , \pmb { \theta } ) = \eta ( z ) \mathbf { Q } _ { B } ( \pmb { \theta } ) ^ { 1 / 2 } ,
$$

and $( \Gamma _ { B } ^ { - 1 } ) ^ { \prime } ( t ) = \sigma _ { B } / \varphi ( t )$ gives

$$
\mathrm { d } \pmb { \vartheta } _ { t } = - \mathbf { m } _ { B } ( \nabla \mathcal { R } ( \pmb { \vartheta } _ { t } ) ) \mathrm { ~ } \mathrm { d } t + \sqrt { n _ { B } ( t ) } \mathbf { Q } _ { B } ( \pmb { \vartheta } _ { t } ) ^ { 1 / 2 } \mathrm { ~ } \mathrm { d } \mathbf { W } _ { t } .\tag{35}
$$

This is the same SDE as (2). Its instantaneous diffusion covariance is $n _ { B } ( t ) \mathbf { Q } _ { B } ( \pmb { \vartheta } _ { t } )$ , and $\mathbb { E } [ \mathcal { E } ( \pmb { \vartheta } _ { t } ) ]$ denotes the expected excess risk. At zeros of the learning rate the dynamics are understood by continuity. On any interval where $\eta = 0$ , both the parameter process and its clock are constant.

Under Assumption 2.5, the marginal standard deviations satisfy $a _ { j } : = \sqrt { \Sigma _ { j j } } \ : \approx \sqrt { \lambda _ { j } }$ . We use these marginals to derive the coordinate equations for Appendix D.

Lemma C.3 (Smoothed sign under Gaussian noise). Let $Z \sim { \mathcal { N } } ( 0 , 1 )$ , let $\tau > 0$ , and define the smoothed sign function $m ( u ) : = 2 \Phi ( u ) - 1$ . Then, for any $\mu \in \mathbb { R }$

$$
\mathbb { E } [ \mathrm { s g n } ( \mu + \tau Z ) ] = m \left( \frac { \mu } { \tau } \right) , \qquad \mathrm { V a r } ( \mathrm { s g n } ( \mu + \tau Z ) ) = 1 - m \left( \frac { \mu } { \tau } \right) ^ { 2 } .
$$

Proof. Since $\operatorname { s g n } ( \mu + \tau Z ) = 1$ on the event $Z > - \mu / \tau$ and equals -1 on its complement, we have

$$
\mathbb { E } [ \mathrm { s g n } ( \mu + \tau Z ) ] = \mathbb { P } \left( Z > - \frac { \mu } { \tau } \right) - \mathbb { P } \left( Z \leqslant - \frac { \mu } { \tau } \right) = 2 \Phi \left( \frac { \mu } { \tau } \right) - 1 = m \left( \frac { \mu } { \tau } \right) .
$$

Moreover, sgn $( \mu + \tau Z ) ^ { 2 } = 1$ almost surely, so

$$
\operatorname { V a r } ( \operatorname { s g n } ( \mu + \tau Z ) ) = \mathbb { E } [ \operatorname { s g n } ( \mu + \tau Z ) ^ { 2 } ] - \mathbb { E } [ \operatorname { s g n } ( \mu + \tau Z ) ] ^ { 2 } = 1 - m \left( { \frac { \mu } { \tau } } \right) ^ { 2 } .
$$

Applying Lemma C.3 coordinate-wise to the drift and sign-noise covariance in (35) yields the following explicit dynamics.

Theorem C.4 (Coordinate-wise dynamics). Suppose $\mathbf { H } = \mathrm { d i a g } ( \lambda _ { j } )$ and $\pmb { \xi } \sim \mathcal { N } ( \mathbf { 0 } , \sigma _ { B } ^ { 2 } \pmb { \Sigma } )$ with $\Sigma  \mathbf { H }$ . Set $k _ { j } = \lambda _ { j } / ( \sigma _ { B } a _ { j } )$ . For each coordinate j, we have

$$
[ \mathbf { m } _ { B } ( \nabla \mathcal { R } ( \pmb { \theta } ) ) ] _ { j } = \sigma _ { B } m \big ( k _ { j } ( \theta _ { j } - \theta _ { j } ^ { * } ) \big ) , \qquad [ \mathbf { Q } _ { B } ( \pmb { \theta } ) ] _ { j j } = 1 - m \big ( k _ { j } ( \theta _ { j } - \theta _ { j } ^ { * } ) \big ) ^ { 2 } .
$$

Accordingly, the j-th coordinate of (35) satisfies

$$
\mathrm { d } X _ { j } ( t ) = - \sigma _ { B } m ( k _ { j } X _ { j } ( t ) ) \mathrm { d } t + \sqrt { n _ { B } ( t ) \left[ 1 - m ( k _ { j } X _ { j } ( t ) ) ^ { 2 } \right] } \ \mathrm { d } \widetilde { W } _ { j , t } ,\tag{36}
$$

where $X _ { j } ( t ) = ( \vartheta _ { t } ) _ { j } - \theta _ { i } ^ { * }$ and $X _ { j } ( 0 ) = - \theta _ { j } ^ { * }$ under the initialization $\vartheta _ { 0 } = \mathbf { 0 }$

Proof. The j-th gradient coordinate has mean $\lambda _ { j } ( \theta _ { j } - \theta _ { j } ^ { * } )$ and noise standard deviation $\sigma _ { B } a _ { j }$ Lemma C.3 therefore gives its mean sign $m ( k _ { j } ( \theta _ { j } - \theta _ { j } ^ { * } ) )$ and variance $1 - m ( k _ { j } ( \theta _ { j } - \theta _ { j } ^ { * } ) ) ^ { 2 }$ Multiplying the mean by $\sigma _ { B }$ gives the scaled drift. The j-th coordinate martingale in (35) has instantaneous variance $n _ { B } ( t ) [ \mathbf { Q } _ { B } ( \pmb { \vartheta } _ { t } ) ] _ { j j }$ , so it admits the scalar Brownian representation in (36). The Brownian motions for different coordinates may be correlated. □

## C.4 Properties of the smoothed sign function

We collect the properties of the smoothed sign function $m ( z ) ~ = ~ 2 \Phi ( z ) - 1$ introduced in Appendix C.3. These estimates will be used in Appendix D to obtain lower and upper bounds for the modal second moments. Throughout the following analysis, we write

$$
c : = m ^ { \prime } ( 0 ) = { \sqrt { 2 / \pi } } .
$$

Lemma C.5 (Properties of the smoothed sign function). The smoothed sign function m has the following properties.

1. The function m is smooth, odd, and strictly increasing, with $| m ( z ) | < 1$ for any $z \in \mathbb { R }$

2. The following bounds hold globally:

$$
0 \leqslant z m ( z ) \leqslant c z ^ { 2 } , \qquad m ( z ) ^ { 2 } \leqslant c ^ { 2 } z ^ { 2 } , \qquad z \in \mathbb { R } .\tag{37}
$$

3. Near the origin, we have

$$
m ( z ) = c z + O ( z ^ { 3 } ) , ~ z  0 ,
$$

$$
z m ( z ) \geqslant \frac { c } { 2 } z ^ { 2 } , | z | \leqslant 1 .\tag{38}
$$

4. In the tails,

$$
m ( z ) \operatorname { s g n } ( z ) \to 1 , \qquad 1 - m ( z ) ^ { 2 } \to 0 , \qquad | z | \to \infty .\tag{39}
$$

Proof. The definition of m gives

$$
m ( z ) = c \int _ { 0 } ^ { z } e ^ { - u ^ { 2 } / 2 } ~ \mathrm { d } u , \qquad m ^ { \prime } ( z ) = c e ^ { - z ^ { 2 } / 2 } .
$$

Thus m is smooth, odd, and strictly increasing. Since $0 < \Phi ( z ) < 1$ for any finite $z ,$ we also have $| m ( z ) | < 1$ . The integral representation gives $| m ( z ) | \leqslant c | z |$ , and $m ( z )$ has the same sign as z. These observations prove (37).

Expanding the integrand at zero gives $m ( z ) = c z + O ( z ^ { 3 } )$ . For $| z | \leqslant 1$ , the same integral representation yields

$$
z m ( z ) = c | z | \int _ { 0 } ^ { | z | } e ^ { - u ^ { 2 } / 2 } \mathrm { ~ d } u \geqslant c e ^ { - 1 / 2 } z ^ { 2 } \geqslant \frac { c } { 2 } z ^ { 2 } ,
$$

which proves (38). Finally, $\Phi ( z ) \to 1 { \mathrm { ~ a s ~ } } z \to \infty$ and $\Phi ( z ) \to 0$ as $z  - \infty$ , giving (39). □

## C.5 An extension to non-Gaussian gradient noise

This extension is supplementary and may be skipped when reading the main proof in Appendix D, which uses the Gaussian SDE from Appendix C.3. We show that the original mini-batch noise also yields the relevant smoothed-sign properties for integer $B \geqslant 2$

Mini-batch gradient and density at zero. Fix $\delta \ : = \ : \theta \ : - \ : \theta ^ { * }$ and a coordinate $j$ . For independent samples from the Gaussian regression model,

$$
g _ { B , j } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \phi _ { i j } \bigl ( \phi _ { i } ^ { \top } \pmb \delta - \epsilon _ { i } \bigr ) = \lambda _ { j } \delta _ { j } + \xi _ { j } .
$$

Set

$$
\sigma _ { \mathrm { r e s } , j } ^ { 2 } = \sigma ^ { 2 } + \sum _ { \ell \neq j } \lambda _ { \ell } \delta _ { \ell } ^ { 2 } , \qquad R _ { B } = { \frac { 1 } { B \lambda _ { j } } } \sum _ { i = 1 } ^ { B } \phi _ { i j } ^ { 2 } \sim { \frac { \chi _ { B } ^ { 2 } } { B } } .
$$

The residual $\begin{array} { r } { \sum _ { \ell \neq j } \phi _ { i \ell } \delta _ { \ell } - \epsilon _ { i } \mathrm { i s } \mathcal { N } ( 0 , \sigma _ { \mathrm { r e s } , j } ^ { 2 } ) } \end{array}$ and independent of $\phi _ { i j }$ . Conditioning on the coordinate features gives

$$
g _ { B , j } \overset { \mathrm { d } } { = } \lambda _ { j } \delta _ { j } R _ { B } + \sigma _ { \mathrm { r e s } , j } \sqrt { \frac { \lambda _ { j } } { B } } \sqrt { R _ { B } } Z , \qquad Z \perp R _ { B } .
$$

Here $Z \sim \mathcal { N } ( 0 , 1 )$ . At zero coordinate error, $\delta _ { j } = 0 _ { : }$ the noise is a symmetric Gaussian scale mixture with ${ \widehat { \sigma } } _ { j } = \sigma _ { \mathrm { r e s } , j } { \sqrt { \lambda _ { j } / B } }$ . Its density $f _ { j }$ satisfies

$$
f _ { j } ( 0 ) = \frac { \mathbb { E } [ R _ { B } ^ { - 1 / 2 } ] } { \sqrt { 2 \pi } \widehat { \sigma } _ { j } } = \frac { \sqrt { B } } { 2 \sqrt { \pi } \widehat { \sigma } _ { j } } \frac { \Gamma ( ( B - 1 ) / 2 ) } { \Gamma ( B / 2 ) } \approx \frac { 1 } { \widehat { \sigma } _ { j } } .
$$

For $B \geqslant 3 , \mathbb { E } [ R _ { B } ^ { - 1 } ] = B / ( B - 2 ) \leqslant 3$ gives $1 \leqslant \mathbb { E } [ R _ { B } ^ { - 1 / 2 } ] \leqslant { \sqrt { 3 } }$ For $B = 2$ , the mixture has the Laplace density

$$
f _ { j } ( x ) = \frac { 1 } { \sqrt { 2 } \widehat { \sigma } _ { j } } \exp \left( - \frac { \sqrt { 2 } \left| x \right| } { \widehat { \sigma } _ { j } } \right) .
$$

Thus the density is continuous, finite, and strictly positive at zero, with $f _ { j } ( 0 ) \lesssim \widehat { \sigma } _ { j } ^ { - 1 }$ uniformly over $B \geqslant 2$

Smoothed sign and its role in the proof. To account for the dependence of the noise on $\delta ,$ compute the mean sign directly. With $m ( u ) = 2 \Phi ( u ) - 1$ , conditioning gives

$$
\mathbb { E } [ \mathrm { s g n } ( g _ { B , j } ) \mid \delta ] = \widetilde { m } _ { B } \left( \frac { \sqrt { B \lambda _ { j } } \delta _ { j } } { \sigma _ { \mathrm { r e s } , j } } \right) , \qquad \widetilde { m } _ { B } ( u ) : = \mathbb { E } [ m ( u \sqrt { R _ { B } } ) ] .\tag{40}
$$

Integrating against the $\chi _ { B } ^ { 2 }$ density yields

$$
\widetilde { m } _ { B } ^ { \prime } ( u ) = \frac { 2 } { \sqrt { \pi B } } \frac { \Gamma ( ( B + 1 ) / 2 ) } { \Gamma ( B / 2 ) } \left( 1 + \frac { u ^ { 2 } } { B } \right) ^ { - ( B + 1 ) / 2 } .
$$

Hence $\widetilde { m } _ { B } ^ { \prime } ( 0 ) \ = 1$ , with a uniform upper bound on the derivative and a positive lower bound on $| u | \leqslant 1$ . For $B \geqslant 3 , \tilde { m } _ { B } ( u ) = \mathbb { E } [ \mathrm { s g n } ( u + Z / \sqrt { R _ { B } } ) ]$ and $\mathbb { E } [ Z ^ { 2 } / R _ { B } ] \leqslant 3$ , so Chebyshev's inequality gives $1 - \widetilde { m } _ { B } ( u ) ^ { 2 } \leqslant 1 2 / u ^ { 2 }$ for $| u | \geqslant 1$ . For $B = 2$ , direct integration gives

$$
\widetilde { m } _ { 2 } ( u ) = \frac { u } { \sqrt { u ^ { 2 } + 2 } } , \qquad 1 - \widetilde { m } _ { 2 } ( u ) ^ { 2 } = \frac { 2 } { u ^ { 2 } + 2 } ,
$$

which yields the same tail bound. These estimates provide the local linear response, global linear upper bound, and restoring drift used in Appendix D, uniformly over $B \geqslant 2$

If $\sigma ^ { 2 } \approx 1$ and $\delta ^ { \top } \mathbf { H } \delta \ \lesssim \ 1$ , then $\sigma _ { \mathrm { r e s } , j } ~ \lesssim ~ 1$ uniformly in $j .$ Consequently, the response in (40) has the same coordinate-wise scale $\sqrt { B \lambda _ { j } }$ as the Gaussian model of Appendix C.3, with comparable local response and restoring drift bounds uniformly over $B \ \geqslant \ 2$ .These estimates support the Gaussian modeling choice. Extending the full FSL proof to the original mini-batch noise would additionally require controlling the state-dependent residual scales along the trajectory.

## D FSL of SignSGD (Proof of Theorem 3.1)

Throughout this appendix, $\begin{array} { r } { t = \Gamma _ { B } ( z ) = \sqrt { B } \int _ { 0 } ^ { z } \eta ( u ) } \end{array}$ du denotes SignSGD intrinsic time. We work under the assumptions and the learning-rate and training-time conditions of Theorem 3.1, using the Gaussian model with $\pmb { \Sigma } \approx \mathbf { H }$ specified in Appendix C.3.

We use $\varphi ( t ) = \varphi _ { \mathrm { s g n } } ( t ) = \eta ( \Gamma _ { B } ^ { - 1 } ( t ) )$ and $n _ { B } ( t ) = \varphi ( t ) / \sqrt { B }$ , consistently with Section 2. The scalar $m ( u ) = 2 \Phi ( u ) - 1$ is the unscaled Gaussian sign response; the scaled vector drift in the main text is $\mathbf { m } _ { B }$

We prove the functional scaling law by obtaining upper and lower bounds for every modal second moment and then summing over the spectrum. The lower bound follows directly from the exact second-moment equation. For the upper bound, a Foster-Lyapunov estimate controls the dynamics, while the training-time requirement ensures burn-in control for modes with large normalized initial values. Throughout the proof, constants may depend on $q , s ,$ the fixed bound $C _ { 0 }$ in $\sqrt { B } \eta _ { \mathrm { p e a k } } \leqslant C _ { 0 }$ , and the comparison constants in Assumptions 2.3 and 2.5, but not on B, t or the particular learning rate schedule $\varphi .$

## D.1 A technical lemma: power-law spectral sums

The modal estimates below reduce the signal learning and noise accumulation terms to sums of exponentials over the spectrum. The following lemma converts both sums into power laws and will be used in Appendix D.6 to complete the proof of Theorem 3.1.

Lemma D.1 (Power-law spectral sums). Let $r > 1$ and $\theta > 0$ , and suppose $b _ { j } \ \approx \ j ^ { - r }$ and $\rho _ { j } \stackrel { } { \sim } j ^ { - \theta }$ . There is a constant $C \geqslant 1$ such that, for every $Q \geqslant 0$

$$
C ^ { - 1 } ( 1 + Q ) ^ { - ( r - 1 ) / \theta } \leqslant \sum _ { j = 1 } ^ { \infty } b _ { j } e ^ { - Q \rho _ { j } } \leqslant C ( 1 + Q ) ^ { - ( r - 1 ) / \theta } .\tag{41}
$$

Proof. It suffices to consider $b _ { j } = j ^ { - r }$ and $\rho _ { j } = j ^ { - \theta }$ . First consider $0 \leqslant Q \leqslant 1$ . Then the $j = 1$ term gives a positive constant lower bound, while

$$
\sum _ { j = 1 } ^ { \infty } j ^ { - r } e ^ { - Q j ^ { - \theta } } \leqslant \sum _ { j = 1 } ^ { \infty } j ^ { - r } < \infty .
$$

Also $( 1 + Q ) ^ { - ( r - 1 ) / \theta } \stackrel { \_ } { \sim } 1$ on this interval.

Now consider $Q > 1$ . Let $N = Q ^ { 1 / \theta }$ . For the upper bound, split the sum at $N$ The tail satisfies

$$
\sum _ { j > N } j ^ { - r } e ^ { - Q j ^ { - \theta } } \leqslant \sum _ { j > N } j ^ { - r } \leqslant C N ^ { 1 - r } .
$$

For the part $j \leqslant N$ , divide the indices into dyadic blocks on which $j$ is comparable to $2 ^ { - k } N$ The contribution of the k-th block is bounded by

$$
C N ^ { 1 - r } 2 ^ { k ( r - 1 ) } e ^ { - 2 ^ { k \theta } } .
$$

The sum over $k \geqslant 0$ is finite, so the contribution of $j \leqslant N$ is also bounded by $C N ^ { 1 - r } ~ { = } ~$ $C Q ^ { - ( r - 1 ) / \theta }$

For the lower bound, consider the integer interval $\lceil N \rceil \leqslant j \leqslant \lfloor 3 N \rfloor$ , which contains at least N indices. On this interval, $e ^ { - Q j ^ { - \theta } } \geqslant e ^ { - 1 }$ and $j ^ { - r } \geqslant ( 3 N ) ^ { - r }$ . Therefore,

$$
\sum _ { j = 1 } ^ { \infty } j ^ { - r } e ^ { - Q j ^ { - \theta } } \geqslant e ^ { - 1 } N ( 3 N ) ^ { - r } \gtrsim Q ^ { - ( r - 1 ) / \theta } .
$$

Since $Q \geqslant 1$ , this is equivalent to $( 1 + Q ) ^ { - ( r - 1 ) / \theta }$

## D.2 Time rescaling and normalized modal dynamics

Let $a _ { j } = \sqrt { \Sigma _ { j j } }$ be the single-sample noise standard deviation, as in Section 6. With $k _ { j } =$ $\sqrt { B } \lambda _ { j } / a _ { j }$ , Theorem C.4 gives

$$
\mathrm { d } X _ { j } ( t ) = - \frac { 1 } { \sqrt { B } } m ( k _ { j } X _ { j } ( t ) ) \mathrm { d } t + \sqrt { n _ { B } ( t ) \left[ 1 - m ( k _ { j } X _ { j } ( t ) ) ^ { 2 } \right] } \ \mathrm { d } \widetilde { W } _ { j , t } ,\tag{42}
$$

where $X _ { j } ( t ) = ( \vartheta _ { t } ) _ { j } - \theta _ { j } ^ { * } , X _ { j } ( 0 ) = - \theta _ { i } ^ { * }$ , and $m ( z ) = 2 \Phi ( z ) - 1$ is the smoothed sign function. The initial signal energy in mode $\dot { j }$ is

$$
w _ { j } : = \lambda _ { j } X _ { j } ( 0 ) ^ { 2 } = \lambda _ { j } | \theta _ { j } ^ { * } | ^ { 2 } .
$$

By Assumption 2.3, we have $\lambda _ { j } \approx j ^ { - q }$ and $w _ { j } \lesssim j ^ { - 1 - q s }$ . Tonelli's theorem gives

$$
\mathbb { E } [ { \mathcal { E } } ( \pmb { \vartheta } _ { t } ) ] = \frac { 1 } { 2 } \sum _ { j = 1 } ^ { \infty } \lambda _ { j } \mathbb { E } [ X _ { j } ( t ) ^ { 2 } ] .\tag{43}
$$

The Brownian motions in different coordinates may be correlated. Since H is diagonal, (43) requires only the marginal second moments, and the scalar drift and diffusion coefficients in (42) depend only on $X _ { j } ( t )$

At intrinsic time $t ,$ the elapsed time for mode $j$ in rescaled units is

$$
{ \widehat { t } } _ { j } = { \frac { k _ { j } t } { \sqrt { B } } } = { \frac { \lambda _ { j } } { a _ { j } } } t .
$$

Writing $\tau$ for the time variable in these units, define

$$
Z _ { j } ( \tau ) : = k _ { j } X _ { j } \left( \frac { \sqrt { B } \tau } { k _ { j } } \right) .\tag{44}
$$

Then $Z _ { j } ( 0 ) ^ { 2 } = B ( \lambda _ { j } / a _ { j } ^ { 2 } ) w _ { j }$ . Applying Lemma B.2 with the time change $\tau = k _ { j } t / \sqrt { B }$ , and then using the rescaling (44), we obtain

$$
\mathrm { d } Z _ { j } ( \tau ) = - m ( Z _ { j } ( \tau ) ) \ \mathrm { d } \tau + \varepsilon _ { j } ( \tau ) \sqrt { 1 - m ( Z _ { j } ( \tau ) ) ^ { 2 } } \ \mathrm { d } \widetilde { W } _ { j , \tau } ,\tag{45}
$$

where $\widetilde { W } _ { j , \tau }$ is a standard Brownian motion in the rescaled time τ, and

$$
\varepsilon _ { j } ( \tau ) ^ { 2 } : = k _ { j } \varphi \left( \frac { \sqrt { B } \tau } { k _ { j } } \right) = \sqrt { B } k _ { j } n _ { B } \left( \frac { \sqrt { B } \tau } { k _ { j } } \right) .\tag{46}
$$

Choose fixed constants $\lambda _ { + } , w _ { + } , h _ { - } , h _ { + } > 0$ such that

$$
\lambda _ { j } \leqslant \lambda _ { + } j ^ { - q } , \qquad w _ { j } \leqslant w _ { + } j ^ { - 1 - q s } , \qquad h _ { - } \lambda _ { j } \leqslant a _ { j } ^ { 2 } \leqslant h _ { + } \lambda _ { j } .
$$

By the learning-rate condition in Theorem 3.1, fix $C _ { 0 } > 0$ such that $\sqrt { B } \| \varphi \| _ { \infty } = \sqrt { B } \eta _ { \mathrm { p e a k } } \leqslant C _ { 0 }$ This gives the uniform normalized noise bound

$$
\varepsilon _ { j } ( \tau ) ^ { 2 } \leqslant k _ { j } \| \varphi \| _ { \infty } \leqslant C _ { 0 } \sqrt { \lambda _ { + } / h _ { - } } = : \varepsilon _ { \operatorname* { m a x } } ^ { 2 } .\tag{47}
$$

Here we used $k _ { j } / \sqrt { B } = \lambda _ { j } / a _ { j } \leqslant \sqrt { \lambda _ { + } / h _ { - } } .$

Returning to the original risk weights in (43) gives

$$
\mathbb { E } [ \mathcal { E } ( \pmb { \vartheta } _ { t } ) ] = \frac { 1 } { 2 } \sum _ { j = 1 } ^ { \infty } \frac { \lambda _ { j } } { k _ { j } ^ { 2 } } \mathbb { E } [ Z _ { j } ( \widehat { t } _ { j } ) ^ { 2 } ] , \qquad \frac { \lambda _ { j } } { k _ { j } ^ { 2 } } = \frac { a _ { j } ^ { 2 } } { B \lambda _ { j } } .\tag{48}
$$

## D.3 The generator and a global lower bound

For $f \in C ^ { 2 } ( \mathbb { R } )$ , we can write the instantaneous generator of the SDE (45) as

$$
( \mathcal { L } _ { j , \tau } f ) ( z ) = - m ( z ) f ^ { \prime } ( z ) + \frac { \varepsilon _ { j } ( \tau ) ^ { 2 } } { 2 } \left( 1 - m ( z ) ^ { 2 } \right) f ^ { \prime \prime } ( z ) .\tag{49}
$$

Note that both $j$ and τ enter the instantaneous generator only through the scalar $\varepsilon _ { j } ( \tau )$ . We therefore introduce the following operator, which provides a common representation of all the instantaneous generators:

$$
( L _ { \varepsilon } f ) ( z ) : = - m ( z ) f ^ { \prime } ( z ) + \frac { \varepsilon ^ { 2 } } { 2 } \left( 1 - m ( z ) ^ { 2 } \right) f ^ { \prime \prime } ( z ) .\tag{50}
$$

We have

$$
\begin{array} { r } { \mathcal { L } _ { j , \tau } = L _ { \varepsilon _ { j } ( \tau ) } . } \end{array}
$$

After the time rescaling, the $\mathrm { d r i f t } - m ( z )$ is identical for every mode, while all dependence on the mode $j$ and on the learning-rate schedule is encoded in the single scalar $\varepsilon _ { j } ( \tau )$ . It therefore suffices to establish drift estimates for a single generator $L _ { \varepsilon }$ , rather than treating each mode separately.

We use the properties of m collected in Appendix C.4, with $c = m ^ { \prime } ( 0 )$ as defined there. The local expansion (38) identifies c as the strength of the local linear drift, which enters both the lower and upper moment bounds. This local behavior motivates the quadratic test function $V ( z ) = z ^ { 2 }$ . Since the drift saturates in the tails and the process can leave the neighborhood of the origin, the global upper bound uses a Lyapunov function that is quadratic near the origin and exponential outside a bounded interval.

Lemma D.2 (Quadratic drift lower bound). For $V ( z ) = z ^ { 2 }$ , set

$$
\omega : = 2 c + c ^ { 2 } \varepsilon _ { \mathrm { m a x } } ^ { 2 } .\tag{51}
$$

Then, for any $z \in \mathbb { R }$ and $0 \leqslant \varepsilon ^ { 2 } \leqslant \varepsilon _ { \mathrm { m a x } } ^ { 2 }$ ，

$$
L _ { \varepsilon } V ( z ) \geqslant - \omega V ( z ) + \varepsilon ^ { 2 } .\tag{52}
$$

Proof. For the test function $V ( z ) = z ^ { 2 }$ , the global bounds (37) in Lemma C.5 give

$$
\begin{array} { c } { { L _ { \varepsilon } V ( z ) = - 2 z m ( z ) + \varepsilon ^ { 2 } \left( 1 - m ( z ) ^ { 2 } \right) } } \\ { { \geqslant - \left( 2 c + c ^ { 2 } \varepsilon ^ { 2 } \right) z ^ { 2 } + \varepsilon ^ { 2 } } } \\ { { \geqslant - \omega V ( z ) + \varepsilon ^ { 2 } . } } \end{array}
$$

Applying the lower-bound conclusion of Lemma B.1 to this drift estimate gives the following second-moment bound.

Corollary D.3 (Global second-moment lower bound). Let $Z$ solve

$$
\mathrm { d } Z ( \tau ) = - m ( Z ( \tau ) ) \mathrm { d } \tau + \varepsilon ( \tau ) \sqrt { 1 - m ( Z ( \tau ) ) ^ { 2 } } \mathrm { d } W _ { \tau } , \qquad Z ( 0 ) = z _ { 0 } ,
$$

where $0 \leqslant \varepsilon ( \tau ) ^ { 2 } \leqslant \varepsilon _ { \mathrm { m a x } } ^ { 2 }$ . With $\omega$ as in (51), for any $\tau \geqslant 0$

$$
\mathbb { E } [ Z ( \tau ) ^ { 2 } ] \geqslant z _ { 0 } ^ { 2 } e ^ { - \omega \tau } + \int _ { 0 } ^ { \tau } e ^ { - \omega ( \tau - r ) } \varepsilon ( r ) ^ { 2 } { \mathrm { ~ d } } r .\tag{53}
$$

Proof. Apply (24) in Lemma B.1 with $V ( z ) = z ^ { 2 } , c _ { V } = \omega$ , and $r ( \tau , z ) = \varepsilon ( \tau ) ^ { 2 }$ , using Lemma D.2. □

## D.4 A Foster-Lyapunov upper bound

We first construct a Lyapunov function, then use it to prove the second-moment upper bound.

Lemma D.4 (Auxiliary Lyapunov function). For any $R \geqslant 1$ , there exists a nonnegative even function $\Psi _ { R } \in C ^ { 2 } ( \mathbb { R } )$ with the following forms in the three regions:

1. For $| z | \leqslant { \frac { 1 } { 2 } }$ , the function is quadratic:

$$
\Psi _ { R } ( z ) = z ^ { 2 } .\tag{54}
$$

2. For $\begin{array} { r } { \frac { 1 } { 2 } < | z | < R } \end{array}$ , the function smoothly interpolates between the quadratic and exponential pieces and satisfies

$$
z ^ { 2 } \leqslant \Psi _ { R } ( z ) \leqslant e ^ { | z | } .
$$

3. $F o r \left| z \right| \geqslant R$ , the function is exponential:

$$
\Psi _ { R } ( z ) = e ^ { | z | } .
$$

In particular, the following bounds hold globally:

$$
z ^ { 2 } \leqslant \Psi _ { R } ( z ) \leqslant 4 z ^ { 2 } e ^ { | z | } , \qquad z \in \mathbb { R } .\tag{55}
$$

Proof. For $0 \leqslant y \leqslant 1$ , let

$$
h ( y ) = 1 0 y ^ { 3 } - 1 5 y ^ { 4 } + 6 y ^ { 5 } .
$$

Its first two derivatives are

$$
h ^ { \prime } ( y ) = 3 0 y ^ { 2 } ( 1 - y ) ^ { 2 } , \qquad h ^ { \prime \prime } ( y ) = 6 0 y ( 1 - y ) ( 1 - 2 y ) .
$$

In particular, we have $h ( 0 ) = 0 , h ( 1 ) = 1$ , and $h ^ { \prime } ( 0 ) = h ^ { \prime } ( 1 ) = h ^ { \prime \prime } ( 0 ) = h ^ { \prime \prime } ( 1 ) = 0$ .Since $h ^ { \prime } ( y ) \geqslant 0$ we also have $0 \leqslant h ( y ) \leqslant 1$ on [0, 1].

For $r \geqslant 0$ define

$$
\psi _ { R } ( r ) = \left\{ \begin{array} { l l } { r ^ { 2 } , } & { 0 \leqslant r \leqslant \frac { 1 } { 2 } , } \\ { 0 \leqslant r ( 1 - h ( y ) ) r ^ { 2 } + h ( y ) e ^ { r } , } & { y = \displaystyle \frac { r - \frac { 1 } { 2 } } { R - \frac { 1 } { 2 } } , } \\ { e ^ { r } , } & { r \geqslant R , } \end{array} \right. \quad \Psi _ { R } ( z ) : = \psi _ { R } ( | z | ) .
$$

The definition gives the required quadratic and exponential forms. At $\begin{array} { r } { r = { \frac { 1 } { 2 } } } \end{array}$ and $r = R ,$ the endpoint conditions on $h , h ^ { \prime } , h ^ { \prime \prime }$ make the values and the first two derivatives of the adjacent pieces agree. Thus $\psi _ { R }$ is $C ^ { 2 }$ across both boundaries. Since $\Psi _ { R } ( z ) ~ = ~ z ^ { 2 }$ near zero, the

even extension $\Psi _ { R } ( z ) = \psi _ { R } ( | z | )$ belongs to $C ^ { 2 } ( \mathbb { R } )$ . It is nonnegative because each piece is nonnegative.

On the connecting interval, $\psi _ { R } ( r )$ is a convex combination of $r ^ { 2 }$ and $e ^ { r }$ . Since $e ^ { r } \geqslant r ^ { 2 }$ for $r \geqslant 0$ , we have

$$
r ^ { 2 } \leqslant \psi _ { R } ( r ) \leqslant e ^ { r } \leqslant e ^ { R } .
$$

Finally, the construction gives $\Psi _ { R } ( z ) \geqslant z ^ { 2 }$ for any $z \in \mathbb { R }$ . For $| z | \geqslant { \frac { 1 } { 2 } }$ , the connecting and tail regions satisfy

$$
\Psi _ { R } ( z ) \leqslant e ^ { | z | } \leqslant 4 z ^ { 2 } e ^ { | z | } ;
$$

for $| z | \leqslant { \frac { 1 } { 2 } } , \Psi _ { R } ( z ) = z ^ { 2 } \leqslant 4 z ^ { 2 } e ^ { | z | }$ . This proves (55).

Lemma D.5 (Foster-Lyapunov drift estimate). Let $\varepsilon _ { \mathrm { m a x } } ^ { 2 }$ be the fixed noise bound in (47). There exist a Lyapunov function V of the form constructed in Lemma $D . 4$ and constants $c _ { L } > 0 , C \geqslant 1$ such that

$$
L _ { \varepsilon } \mathcal { V } ( z ) \leqslant - c _ { L } \mathcal { V } ( z ) + C \varepsilon ^ { 2 }
$$

for any $z \in \mathbb { R }$ and $0 \leqslant \varepsilon ^ { 2 } \leqslant \varepsilon _ { \mathrm { m a x } } ^ { 2 }$

Proof. By the tail limits (39) in Lemma C.5, we may choose $R \geqslant 1$ sufficiently large for the bound $\varepsilon _ { \mathrm { m a x } } ^ { 2 }$ in (47) so that

$$
m ( z ) \ : \mathrm { s g n } ( z ) \geqslant \frac { 3 } { 4 } , \qquad \frac { \varepsilon _ { \mathrm { m a x } } ^ { 2 } } { 2 } \left( 1 - m ( z ) ^ { 2 } \right) \leqslant \frac { 1 } { 4 } , \qquad | z | \geqslant R .
$$

Set $ { \boldsymbol { \nu } } = \Psi _ { R }$ , using the function constructed in Lemma D.4. We estimate its generator in the local, tail, and connecting regions.

For $| z | \leqslant { \frac { 1 } { 2 } }$ , we have $\mathcal { V } ( z ) = z ^ { 2 }$ , SO

$$
L _ { \varepsilon } z ^ { 2 } = - 2 z m ( z ) + \varepsilon ^ { 2 } ( 1 - m ( z ) ^ { 2 } ) .
$$

Using the local lower bound (38) in Lemma C.5 and $1 - m ( z ) ^ { 2 } \leqslant 1$ , we obtain

$$
L _ { \varepsilon } z ^ { 2 } \leqslant - c z ^ { 2 } + \varepsilon ^ { 2 } .
$$

Thus the desired drift inequality holds in the local region.

On the tail region $| z | \geqslant R$ we have $\mathcal { V } ( z ) = e ^ { | z | }$ . Hence

$$
\mathcal { V } ^ { \prime } ( z ) = \mathrm { s g n } ( z ) \mathcal { V } ( z ) , \qquad \mathcal { V } ^ { \prime \prime } ( z ) = \mathcal { V } ( z ) .
$$

Therefore, for any $\varepsilon ^ { 2 } \leqslant \varepsilon _ { \mathrm { m a x } } ^ { 2 }$

$$
L _ { \varepsilon } \mathcal { V } ( z ) = \mathcal { V } ( z ) \left[ - m ( z ) \operatorname { s g n } ( z ) + \frac { \varepsilon ^ { 2 } } { 2 } ( 1 - m ( z ) ^ { 2 } ) \right] \leqslant \mathcal { V } ( z ) \left[ - \frac { 3 } { 4 } + \frac { 1 } { 4 } \right] = - \frac { 1 } { 2 } \mathcal { V } ( z ) .
$$

On the connecting region ${ \frac { 1 } { 2 } } \leqslant | z | \leqslant R$ write $r = | \boldsymbol { z } |$ and $y = ( r - \textstyle { \frac { 1 } { 2 } } ) / ( R - \textstyle { \frac { 1 } { 2 } } )$ . Differentiating the interpolation in Lemma D.4 gives

$$
\mathcal V ^ { \prime } ( z ) \mathrm { s g n } ( z ) = ( 1 - h ( y ) ) 2 r + h ( y ) e ^ { r } + \frac { h ^ { \prime } ( y ) } { R - \frac { 1 } { 2 } } ( e ^ { r } - r ^ { 2 } ) \geqslant 1 ,
$$

since $2 r \geqslant 1 , e ^ { r } \geqslant 1$ , and $h ^ { \prime } ( y ) \geqslant 0$ . Smoothness on this compact region gives $| \mathcal { V } ^ { \prime \prime } ( z ) | \leqslant b _ { R }$ for some finite $b _ { R } ~ > ~ 0$ By the oddness and strict monotonicity in Lemma C.5, we have $m ( z ) \ : \mathrm { s g n } ( z ) \geqslant m _ { * } : = m ( \frac { 1 } { 2 } ) > 0$ on this region, so

$$
\begin{array} { c l c r } { { \displaystyle { \cal L } _ { \varepsilon } \mathcal { V } ( z ) = - m ( z ) \mathcal { V } ^ { \prime } ( z ) + \frac { \varepsilon ^ { 2 } } { 2 } ( 1 - m ( z ) ^ { 2 } ) \mathcal { V } ^ { \prime \prime } ( z ) } } \\ { { \leqslant - m _ { * } + \frac { b _ { R } } { 2 } \varepsilon ^ { 2 } . } } \end{array}
$$

Choose

$$
c _ { L } : = \operatorname* { m i n } \left\{ c , \frac { 1 } { 2 } , \frac { m _ { * } } { 2 e ^ { R } } \right\} > 0 , \qquad C : = \operatorname* { m a x } \left\{ 1 , \frac { b _ { R } } { 2 } \right\} .
$$

Because $\mathcal { V } ( z ) \leqslant e ^ { R }$ on the connecting region, we have

$$
\begin{array} { r } { - m _ { * } \leqslant - c _ { L } \mathcal { V } ( z ) . } \end{array}
$$

Together with the local and tail estimates, these choices give

$$
L _ { \varepsilon } \mathcal { V } ( z ) \leqslant - c _ { L } \mathcal { V } ( z ) + C \varepsilon ^ { 2 }
$$

for any $z \in \mathbb { R }$ and $0 \leqslant \varepsilon ^ { 2 } \leqslant \varepsilon _ { \mathrm { m a x } } ^ { 2 }$

Corollary D.6 (Second-moment upper bound). Let $ { \boldsymbol \nu } ,  { \boldsymbol c } _ { L }$ , and C be as in Lemma D.5. If Z solves

$$
\mathrm { d } Z ( \tau ) = - m ( Z ( \tau ) ) \mathrm { d } \tau + \varepsilon ( \tau ) \sqrt { 1 - m ( Z ( \tau ) ) ^ { 2 } } \mathrm { d } W _ { \tau } ,
$$

with deterministic initial value $Z ( 0 )$ and $\begin{array} { r } { \operatorname* { s u p } _ { \tau \geqslant 0 } \varepsilon ( \tau ) ^ { 2 } \leqslant \varepsilon _ { \mathrm { m a x } } ^ { 2 } } \end{array}$ , then

$$
\mathbb { E } [ Z ( \tau ) ^ { 2 } ] \leqslant C e ^ { - c _ { L } \tau } \mathcal { V } ( Z ( 0 ) ) + C \int _ { 0 } ^ { \tau } e ^ { - c _ { L } ( \tau - r ) } \varepsilon ( r ) ^ { 2 } \mathrm { d } r .
$$

Proof. The bounded drift and diffusion in the scalar SDE give exponential moments of every order, uniformly on each finite time interval. This ensures integrability in Dynkin's formula for V and its derivatives. Using the drift estimate in Lemma D.5, the upper-bound conclusion (22) of Lemma B.1 with $V = \mathcal { V } , \thinspace c _ { V } = c _ { L }$ , and $r ( \tau , z ) = C \varepsilon ( \tau ) ^ { 2 }$ gives

$$
\mathbb { E } [ \mathcal { V } ( Z ( \tau ) ) ] \leqslant e ^ { - c _ { L } \tau } \mathcal { V } ( Z ( 0 ) ) + C \int _ { 0 } ^ { \tau } e ^ { - c _ { L } ( \tau - r ) } \varepsilon ( r ) ^ { 2 } \mathrm { d } r .
$$

The bound $z ^ { 2 } \leqslant \mathcal { V } ( z )$ in (55) and $C \geqslant 1$ give the claimed second-moment upper bound. □

## D.5 Control of the burn-in process

The following argument controls the burn-in phase for modes with large normalized initial values. Recall that $| Z _ { j } ( 0 ) | = k _ { j } | \theta _ { j } ^ { * } |$ . A large $| Z _ { j } ( 0 ) |$ places the mode in the saturated region at initialization. Therefore, we require the intrinsic time t to be sufficiently large relative to $\sqrt { B } \left| \theta _ { j } ^ { * } \right|$ to go through this phase.

We compare the signal learning terms in Corollary D.3 and Corollary D.6, which involve $Z _ { j } ( 0 ) ^ { 2 }$ and $\mathcal { V } ( Z _ { j } ( 0 ) )$ , respectively. For modes with $\begin{array} { r } { | Z _ { j } ( 0 ) | \leqslant \frac { 1 } { 2 } } \end{array}$ , the identity $\mathcal { V } ( Z _ { j } ( 0 ) ) = Z _ { j } ( 0 ) ^ { 2 }$ gives a matching initialization in both bounds. For modes with large $| Z _ { j } ( 0 )$ , the factor $\nu ( Z _ { j } ( 0 ) )$ cannot be bounded by $Z _ { j } ( 0 ) ^ { 2 }$ up to a uniform constant, so we need to control the effect of burn-in. Each modal signal learning term decays exponentially in time, allowing a sufficiently large t to suppress this extra factor. For these modes, the global envelope (55) gives

$$
\mathcal { V } ( Z _ { j } ( 0 ) ) \leqslant 4 Z _ { j } ( 0 ) ^ { 2 } e ^ { | Z _ { j } ( 0 ) | } .
$$

The next lemma uses (57) to absorb $e ^ { | Z _ { j } ( 0 ) | }$ into the exponential decay. To separate the modes according to their initial values, set $\zeta : = 1 + q s$ and define

$$
J _ { B } : = \operatorname* { m a x } \left\{ 1 , \left\lceil ( 4 w _ { + } B / h _ { - } ) ^ { 1 / \zeta } \right\rceil \right\} .\tag{56}
$$

The cutoff satisfies $J _ { B } \approx B ^ { 1 / \zeta } = B ^ { 1 / ( 1 + q s ) }$ . Its purpose is to separate two cases in the signal learning bound. For $j \geqslant J _ { B }$ , the bounds on $w _ { j }$ and $\lambda _ { j } / a _ { j } ^ { 2 }$ give

$$
\begin{array} { r } { Z _ { j } ( 0 ) ^ { 2 } = B ( \lambda _ { j } / a _ { j } ^ { 2 } ) w _ { j } \leqslant \frac { 1 } { 4 } , \qquad \mathcal { V } ( Z _ { j } ( 0 ) ) = Z _ { j } ( 0 ) ^ { 2 } . } \end{array}
$$

These modes already have a matching initialization in both bounds. For $j < J _ { B }$ , the initial value may grow with B. The conditions in (57) ensure that t exceeds the burn-in scale for these modes by a sufficient constant factor, allowing the extra factor to be absorbed into the exponential decay in t.

Lemma D.7 (Burn-in control for small indices). Suppose

$$
t \geqslant C _ { 1 } \sqrt { B } , \qquad t \geqslant C _ { 3 } B ^ { 1 / ( 2 s + 2 / q ) } ,\tag{57}
$$

where $C _ { 1 } , C _ { 3 } > 0$ are suficiently large constants depending only on the xed model parameters. Then, for any integer $1 \leqslant j < J _ { B }$ 2

$$
e ^ { - c _ { L } k _ { j } t / \sqrt { B } } \mathcal { V } ( Z _ { j } ( 0 ) ) \leqslant 4 Z _ { j } ( 0 ) ^ { 2 } e ^ { - c _ { L } k _ { j } t / ( 2 \sqrt { B } ) } .\tag{58}
$$

Proof. Fix an integer $1 \leqslant j < J _ { B }$ . Applying the global envelope (55) to the signal learning term gives

$$
e ^ { - c _ { L } k _ { j } t / \sqrt { B } } \mathcal { V } ( Z _ { j } ( 0 ) ) \leqslant 4 Z _ { j } ( 0 ) ^ { 2 } e ^ { | Z _ { j } ( 0 ) | - c _ { L } k _ { j } t / \sqrt { B } } .
$$

Our goal is to bound this expression by $4 Z _ { j } ( 0 ) ^ { 2 } e ^ { - c _ { L } k _ { j } t / ( 2 \sqrt { B } ) }$ . It therefore suffices to show

$$
| Z _ { j } ( 0 ) | \leqslant \frac { c _ { L } k _ { j } t } { 2 \sqrt { B } } .
$$

Since $| Z _ { j } ( 0 ) | = k _ { j } | \theta _ { j } ^ { * } |$ and $k _ { j } > 0$ , this condition reduces to

$$
t \geqslant \frac { 2 \sqrt { B } | \theta _ { j } ^ { * } | } { c _ { L } } .\tag{59}
$$

The target (59) compares the observation time with the burn-in scale $\sqrt { B } \left| \theta _ { j } ^ { * } \right|$ . We next estimate this scale for $j < J _ { B }$ . The power-law assumptions give

$$
| \theta _ { j } ^ { * } | = \sqrt { \frac { w _ { j } } { \lambda _ { j } } } \lesssim j ^ { ( q - \zeta ) / 2 } .
$$

The sign of $q - \zeta$ determines whether this bound grows with j. If $\zeta \geqslant q$ , it is bounded by a constant $C _ { \theta }$ independent of B. Taking $C _ { 1 } \geqslant 2 C _ { \theta } / c _ { L }$ , the first condition in (57) gives

$$
t \geqslant C _ { 1 } \sqrt { B } \geqslant \frac { 2 C _ { \theta } \sqrt { B } } { c _ { L } } \geqslant \frac { 2 \sqrt { B } | \theta _ { j } ^ { * } | } { c _ { L } } .
$$

$\zeta < q$ the bound grows with $j ,$ so we use $j < J _ { B }$ and $J _ { B } \gtrsim B ^ { 1 / \zeta }$ to obtain

$$
| \theta _ { j } ^ { * } | \leqslant C _ { \theta } B ^ { ( q - \zeta ) / ( 2 \zeta ) } .
$$

Thus we need t to dominate $\sqrt { B }$ times this polynomial in B. This is precisely what the second condition in (57) ensures: choosing $C _ { 3 } \geqslant 2 C _ { \theta } / c _ { L }$ gives

$$
t \geqslant C _ { 3 } B ^ { q / ( 2 \zeta ) } \geqslant \frac { 2 C _ { \theta } \sqrt { B } } { c _ { L } } B ^ { ( q - \zeta ) / ( 2 \zeta ) } \geqslant \frac { 2 \sqrt { B } | \theta _ { j } ^ { * } | } { c _ { L } } .
$$

Thus, (59) holds in both cases, which proves the claim.

Combining this control of the burn-in process with the normalized second-moment estimates gives the following bounds, with signal learning terms that depend quadratically on $Z _ { j } ( 0 )$ on both sides.

Proposition D.8 (Uniform two-sided modal bounds). Under the condition (57), there is a constant $C \geqslant 1 ,$ , independent of B,t and the learning rate schedule, such that, for any $j \geqslant 1$ and $\widehat { t } _ { j } = k _ { j } t / \sqrt { B }$

$$
\mathbb { E } [ Z _ { j } ( \widehat { t _ { j } } ) ^ { 2 } ] \geqslant Z _ { j } ( 0 ) ^ { 2 } e ^ { - \omega \widehat { t _ { j } } } + \int _ { 0 } ^ { \widehat { t _ { j } } } e ^ { - \omega ( \widehat { t _ { j } } - r ) } \varepsilon _ { j } ( r ) ^ { 2 } \mathrm { d } r ,\tag{60}
$$

and

$$
\mathbb { E } [ Z _ { j } ( \widehat { t _ { j } } ) ^ { 2 } ] \leqslant C \left[ Z _ { j } ( 0 ) ^ { 2 } e ^ { - c _ { L } \widehat { t _ { j } } / 2 } + \int _ { 0 } ^ { \widehat { t _ { j } } } e ^ { - c _ { L } ( \widehat { t _ { j } } - r ) / 2 } \varepsilon _ { j } ( r ) ^ { 2 } \ \mathrm { d } r \right] .\tag{61}
$$

Here $\omega$ and $c _ { L }$ are as in Lemma D.2 and Lemma D.5, respectively.

Proof. The lower bound follows directly from Corollary D.3. For the upper bound, apply Corollary D.6 to $Z _ { j }$ . When $j ~ < ~ J _ { B }$ , Lemma D.7 bounds the signal learning term by $Z _ { j } ( 0 ) ^ { 2 } e ^ { - c _ { L } \widehat { t } _ { j } / 2 }$ up to a constant factor. When $j \geqslant J _ { B }$ , the identity $\mathcal { V } ( Z _ { j } ( 0 ) ) = Z _ { j } ( 0 ) ^ { 2 }$ in (54) gives the same bound. Since $\varepsilon _ { j } ( r ) ^ { 2 } \geqslant 0$ , replacing $e ^ { - c _ { L } ( \widehat { t } _ { j } - r ) } \mathrm { ~ b y ~ } e ^ { - c _ { L } ( \widehat { t } _ { j } - r ) / 2 }$ bounds the noise integral from above. These estimates give (61) □

## D.6 Completion of the proof

Recall from (48) that

$$
\mathbb { E } [ \mathcal { E } ( \pmb { \vartheta } _ { t } ) ] = \frac { 1 } { 2 } \sum _ { j = 1 } ^ { \infty } \frac { \lambda _ { j } } { k _ { j } ^ { 2 } } \mathbb { E } [ Z _ { j } ( \widehat { t _ { j } } ) ^ { 2 } ] .
$$

We next prove the upper bound for the expected excess risk. The lower bound follows by the same argument. The key step is to sum the exponentially decaying modal terms and apply Lemma D.1 to obtain the power laws. Choose $C _ { 1 } , C _ { 3 }$ as in Lemma D.7. The training-time condition in Theorem 3.1 then implies (57), so Proposition D.8 applies.

For the signal term, the exact risk weight gives $( \lambda _ { j } / k _ { j } ^ { 2 } ) Z _ { j } ( 0 ) ^ { 2 } = w _ { j }$ . For the noise term, substitute (46) and change variables $r = k _ { j } u / \sqrt { B }$ . The factors $k _ { j }$ cancel, giving

$$
\frac { \lambda _ { j } } { k _ { j } ^ { 2 } } \int _ { 0 } ^ { \widehat { t _ { j } } } e ^ { - c _ { L } ( \widehat { t _ { j } } - r ) / 2 } \varepsilon _ { j } ( r ) ^ { 2 } \mathrm { d } r = \lambda _ { j } \int _ { 0 } ^ { t } e ^ { - c _ { L } k _ { j } ( t - u ) / ( 2 \sqrt { B } ) } n _ { B } ( u ) \mathrm { d } u .
$$

Substituting these identities into (61) and summing over $j ,$ we obtain

$$
\mathbb { E } [ \mathcal { E } ( \vartheta _ { t } ) ] \lesssim \sum _ { j = 1 } ^ { \infty } w _ { j } e ^ { - c _ { L } k _ { j } t / ( 2 \sqrt { B } ) } + \int _ { 0 } ^ { t } \left( \sum _ { j = 1 } ^ { \infty } \lambda _ { j } e ^ { - c _ { L } k _ { j } ( t - u ) / ( 2 \sqrt { B } ) } \right) n _ { B } ( u ) \mathrm { d } u .\tag{62}
$$

Since $w _ { j } \stackrel { } { \sim } j ^ { - 1 - q s }$ and $k _ { j } / \sqrt { B } = \lambda _ { j } / a _ { j } \approx \sqrt { \lambda _ { j } } \approx j ^ { - q / 2 }$ , Lemma D.1 with weights $w _ { j }$ , rates $k _ { j } / \sqrt { B }$ , and $Q = c _ { L } t / 2$ gives

$$
\sum _ { j = 1 } ^ { \infty } w _ { j } e ^ { - c _ { L } k _ { j } t / ( 2 \sqrt { B } ) } \stackrel { } { \sim } ( 1 + c _ { L } t / 2 ) ^ { - 2 s } \stackrel { } { \sim } ( 1 + t ) ^ { - 2 s } .\tag{63}
$$

Here in the last step we used the fact that $c _ { L }$ is a fixed positive constant. For the noise accumulation term, apply the same lemma with $b _ { j } = \lambda _ { j }$ , the same rates $k _ { j } / \sqrt { B } , r = q , \theta = q / 2$ and $Q = c _ { L } ( t - u ) / 2$ . This gives

$$
\sum _ { j = 1 } ^ { \infty } \lambda _ { j } e ^ { - c _ { L } k _ { j } ( t - u ) / ( 2 \sqrt { B } ) } \gtrsim ( 1 + t - u ) ^ { - ( 2 - 2 / q ) } , \qquad 0 \leqslant u \leqslant t .\tag{64}
$$

The noise accumulation estimate also holds at $u = t$ , where $Q = 0$ . Substituting (63) and (64) into (62) gives the required upper bound.

For the lower bound, apply the same steps to (60), taking $Q = \omega t$ and $Q = \omega ( t - u )$ in Lemma D.1. Since ω is also a fixed positive constant, the same signal and noise weights $w _ { j }$ and $\lambda _ { j }$ give the same power laws. Together, the two bounds prove

$$
\mathbb { E } [ \mathcal { E } ( \vartheta _ { t } ) ] \stackrel { } { \sim } ( 1 + t ) ^ { - 2 s } + \int _ { 0 } ^ { t } ( 1 + t - r ) ^ { - ( 2 - 2 / q ) } n _ { B } ( r ) \mathrm { d } r .
$$

Since $n _ { B } ( r ) = \sigma _ { B } \varphi ( r )$ , this establishes the FSL (3) with the bounded-time profiles

$$
S ( t ) \approx ( 1 + t ) ^ { - 2 s } , ~ K ( t ) \approx ( 1 + t ) ^ { - ( 2 - 2 / q ) } .
$$

These profiles have the large-time decay rates stated in Theorem 3.1, while the shift by 1 keeps the forgetting kernel finite at zero lag. All comparison constants are independent of B, t and the particular schedule $\varphi .$ This completes the proof of Theorem 3.1.

## D.7 Discussion: Initialization and relaxation across spectral modes

The initial cutoff in (56) and the characteristic frontier from Section 6.3 satisfy

$$
J _ { B } \approx B ^ { 1 / ( 1 + q s ) } , \qquad j _ { t } \approx t ^ { 2 / q } .
$$

Here $J _ { B }$ separates modes by their initial normalized error, with $| Z _ { j } ( 0 ) | \leqslant 1 / 2$ for $j \geqslant J _ { B }$ . The frontier $j _ { t }$ compares the observation time with the characteristic relaxation time:

$$
\tau _ { j } \approx { \frac { \sqrt { B } } { k _ { j } } } \approx j ^ { q / 2 } , \qquad \tau _ { j _ { t } } \approx t .
$$

Thus, $J _ { B }$ is fixed at a given batch size, while $j _ { t }$ moves to larger indices as training proceeds. The bound $t \gtrsim B ^ { 1 / ( 2 s + 2 / q ) }$ in (57) gives $J _ { B } \lesssim j _ { t }$ . Burn-in control also uses $t \gtrsim \sqrt { B }$ , which is the stronger bound when $s + 1 / q > 1$

For the ordering $J _ { B } < j _ { t }$ shown in Figure 5, the three regions have the following meanings.

• Burn-in-time dominated $\left( j < J _ { B } \right)$ . These modes may start with a large normalized error. When $| Z _ { j } ( 0 ) | \gg 1$ , the saturated drift gives an initial burn-in scale of order $\sqrt { B } \left| \theta _ { j } ^ { * } \right|$ , which can exceed $\tau _ { j }$ . Lemma D.7 controls the effect of this initial transient under the training-time requirement.

• Effectively linear modes $( J _ { B } \leqslant j < j _ { t } )$ . These modes start with $| Z _ { j } ( 0 ) | \leqslant 1 / 2$ and have $\tau _ { j } \lesssim t .$ Their initial signal decays substantially when $j \ll j _ { t }$ , as reflected in the exponential terms of the modal bounds.

• Unlearned tail $( j > j _ { t } )$ . These modes have $\tau _ { j } \gtrsim t ,$ so modes with $j \gg j _ { t }$ retain most of their initial signal. Summing over this tail $\mathrm { g i }$ ves a signal contribution of order $j _ { t } ^ { - q s } \lesssim t ^ { - 2 s }$

These regions describe initial conditions and relaxation time scales. The “unlearned" label refers to the initial signal, while update noise contributes through the convolution term in the FSL.

![](images/f91020d33c9ba6e36df222e132dd0a0d95e20012e6e2bad25d8f94e0a09365f3.jpg)  
Figure 5: Regime separation for different modes when $J _ { B } < j _ { t } .$ The left region identifies modes that may require initial burn-in, controlled by Lemma D.7. The middle region contains modes with small initial normalized errors and relaxation times of order t or shorter. The right region contains slower modes whose initial signal remains partly unlearned. The boundaries indicate characteristic scales.

## E Data scaling analysis

In this appendix, we analyze the data scaling behavior of SignSGD and SGD. Throughout, we fix the batch size at $B = 1$ , so the number of iterations equals the data budget, $N = D / B = D$ where D denotes the total number of samples.

At $B = 1$ , the two intrinsic clocks coincide: $\begin{array} { r } { \Gamma ( z ) : = \Gamma _ { 1 } ( z ) = \Gamma _ { \mathrm { s g d } } ( z ) = \int _ { 0 } ^ { z } \eta ( u ) } \end{array}$ du. We use Γ in the main-text data-scaling analysis. Similarly, $\varphi = \varphi _ { \mathrm { s g n } } = \varphi _ { \mathrm { s g d } }$ and $n _ { 1 } ( t ) = \varphi ( t )$ . The final SignSGD risk is $\mathcal { E } _ { D } ^ { \mathrm { s g n } } = \mathbb { E } [ \mathcal { E } ( \pmb { \vartheta } _ { \Gamma ( D ) } ) ]$

## E.1 Power decay schedules

The goal of the data scaling analysis is to characterize the best rate achievable by SignSGD or SGD through an appropriate choice of the learning rate schedule. To this end, we introduce the power decay schedule (Li et al., 2026c) and derive the rate it achieves in the following analysis. In Appendix E.4 we show that this schedule indeed gives the optimal rate. For $k \in \{ 0 , 1 , \ldots , N - 1 \}$ , the schedule is defined as

$$
\eta _ { k } = \eta _ { \mathrm { p e a k } } \left( 1 - \frac { k } { N } \right) ^ { p } .\tag{65}
$$

It is determined by two hyperparameters: the peak learning rate $\eta _ { \mathrm { p e a k } }$ and the decay rate $p > 0$ In the continuous modeling, the power decay schedule takes the form

$$
\eta ( z ) = \eta _ { \mathrm { p e a k } } \left( 1 - \frac { z } { N } \right) ^ { p } ,
$$

where $z \in [ 0 , N ]$ is the continuous training step. The intrinsic time is given by

$$
t = \Gamma ( z ) = \int _ { 0 } ^ { z } \eta _ { \mathrm { p e a k } } \left( 1 - \frac { u } { N } \right) ^ { p } \mathrm { d } u = \frac { \eta _ { \mathrm { p e a k } } N } { p + 1 } \left[ 1 - \left( 1 - \frac { z } { N } \right) ^ { p + 1 } \right] .
$$

Thus, the total intrinsic time of the entire training process is

$$
T _ { D } = \Gamma ( D ) = \frac { \eta _ { \mathrm { p e a k } } D } { p + 1 } \approx \eta _ { \mathrm { p e a k } } D .
$$

The learning rate schedule can be expressed in intrinsic time as

$$
\varphi ( t ) = \eta ( \Gamma ^ { - 1 } ( t ) ) = \eta _ { \mathrm { p e a k } } \left( 1 - \frac { t } { T _ { D } } \right) ^ { \frac { p } { p + 1 } } .\tag{66}
$$

Intuition behind the power decay schedule. We now provide an intuition for the choice of power decay. In our FSL, the loss is determined by two competing terms: the signal term, which prefers a large total intrinsic time, and the noise term, which prefers sufficiently fast decay near the end of training so that the accumulated noise can be forgotten. Since both SGD and SignSGD have power-law forgetting kernels (see Section 3), a power decay tail is the natural schedule shape for matching the kernel.

## E.2 Data scaling for SignSGD (Proof of Theorem 4.1)

We first prove the data scaling result for SignSGD. Let $\mathcal { E } _ { D } ^ { \mathrm { s g n } } : = \mathbb { E } [ \mathcal { E } ( \vartheta _ { T _ { D } } ) ]$ denote the final-step excess risk of SignSGD where $T _ { D } = \Gamma ( D )$ is the total intrinsic time. The SignSGD FSL from Theorem 3.1 gives 1

$$
\mathcal { E } _ { D } ^ { \mathrm { s g n } } \gtrsim T _ { D } ^ { - 2 s } + \int _ { 0 } ^ { T _ { D } } ( 1 + T _ { D } - t ) ^ { - ( 2 - 2 / q ) } \varphi ( t ) \ \mathrm { d } t .\tag{67}
$$

The following lemma simplifies the FSL under the power decay schedule.

Lemma E.1 (SignSGD FSL under power decay schedules). Under the assumptions of Theorem 3.1, let $B = 1$ and use the power-decay schedule (66). Let $\alpha _ { \mathrm { s g n } } = \operatorname* { m i n } \{ \frac { q } { 2 } , p + 1 \}$ where p is the power decay exponent. Then the inal excess risk satisfies

$$
\mathcal { E } _ { D } ^ { \mathrm { s g n } } \asymp ( \eta _ { \mathrm { p e a k } } D ) ^ { - 2 s } + \eta _ { \mathrm { p e a k } } ( \eta _ { \mathrm { p e a k } } D ) ^ { - ( 1 - \frac { 1 } { \alpha _ { \mathrm { s g n } } } ) }\tag{68}
$$

up to a logarithmic factor when $\textstyle p + 1 = { \frac { q } { 2 } }$

Proof. For the power decay schedule, the total intrinsic time satisfies $T _ { D } \approx \eta _ { \mathrm { p e a k } } D$ . By the SignSGD FSL, it remains to estimate the noise term $\begin{array} { r } { \int _ { 0 } ^ { T _ { D } } ( 1 + T _ { D } - t ) ^ { - ( 2 - 2 / q ) } \varphi ( t ) } \end{array}$ dt. Write $u = T _ { D } - t$ . Then, under the power decay schedule (66),

$$
\varphi ( T _ { D } - u ) = \eta _ { \mathrm { p e a k } } \left( \frac { u } { T _ { D } } \right) ^ { \frac { p } { p + 1 } } .
$$

Hence the noise term is

$$
\eta _ { \mathrm { p e a k } } T _ { D } ^ { - { \frac { p } { p + 1 } } } \int _ { 0 } ^ { T _ { D } } ( 1 + u ) ^ { - ( 2 - 2 / q ) } u ^ { \frac { p } { p + 1 } } \mathrm { d } u .
$$

Let $\textstyle a = { \frac { p } { p + 1 } }$ . Since $( 1 + u ) ^ { - ( 2 - 2 / q ) } u ^ { a } \approx u ^ { a - ( 2 - 2 / q ) }$ for large u, we have

$$
\int _ { 0 } ^ { T _ { D } } ( 1 + u ) ^ { - ( 2 - 2 / q ) } u ^ { a } \mathrm { d } u \asymp \left\{ \begin{array} { l l } { 1 , } & { a - ( 2 - 2 / q ) < - 1 , } \\ { \log T _ { D } , } & { a - ( 2 - 2 / q ) = - 1 , } \\ { T _ { D } ^ { a - ( 2 - 2 / q ) + 1 } , } & { a - ( 2 - 2 / q ) > - 1 . } \end{array} \right.
$$

The boundary condition $a - ( 2 - 2 / q ) = - 1$ is equivalent to $\textstyle p + 1 = { \frac { q } { 2 } } $ . Therefore, the noise term satisfies

$$
\int _ { 0 } ^ { T _ { D } } ( 1 + T _ { D } - t ) ^ { - ( 2 - 2 / q ) } \varphi ( t ) \ \mathrm { d } t \stackrel { } { \sim } \eta _ { \mathrm { p e a k } } T _ { D } ^ { - \left( 1 - \frac { 1 } { \alpha _ { \mathrm { s g n } } } \right) }
$$

up to a logarithmic factor at the boundary, where $\alpha _ { \mathrm { s g n } } = \operatorname* { m i n } \{ \frac { q } { 2 } , p + 1 \}$ . Hence we have

$$
\begin{array} { r } { \mathcal { E } _ { D } ^ { \mathrm { s g n } } \asymp ( \eta _ { \mathrm { p e a k } } D ) ^ { - 2 s } + \eta _ { \mathrm { p e a k } } ( \eta _ { \mathrm { p e a k } } D ) ^ { - \left( 1 - \frac { 1 } { \alpha _ { \mathrm { s g n } } } \right) } . } \end{array}
$$

Optimizing (68) over the peak learning rate gives the data-scaling result. The following theorem is a restatement of Theorem 4.1.

Theorem E.2 (Data scaling for SignSGD). Under the assumptions of Theorem 3.1, let $B = 1$ and use the power-decay schedule (66). Then the following rates hold.

$\begin{array} { r } { I f s \geqslant \frac { 1 } { 2 } - \frac { 1 } { q } . } \end{array}$ , then choosing $p > { \frac { q } { 2 } } - 1 , \eta _ { \mathrm { p e a k } } \approx D ^ { - \frac { 2 s - 1 + 2 / q } { 2 s + 2 / q } }$ yields

$$
\mathcal { E } _ { D } ^ { \mathrm { s g n } } \approx D ^ { - \frac { s q } { s q + 1 } } .
$$

$\begin{array} { r } { I f 0 < s < \frac { 1 } { 2 } - \frac { 1 } { q } } \end{array}$ , then choosing $\begin{array} { r } { p > \frac { 2 s } { 1 - 2 s } , \eta _ { \mathrm { p e a k } } \asymp 1 } \end{array}$ yields

$$
\mathcal { E } _ { D } ^ { \mathrm { s g n } } \approx D ^ { - 2 s } .
$$

Proof. First consider $s \geqslant \frac { 1 } { 2 } - \frac { 1 } { q }$ . Choosing $p > { \frac { q } { 2 } } - 1$ gives $\begin{array} { r } { \alpha _ { \mathrm { s g n } } = \frac { q } { 2 } } \end{array}$ . Then, applying Lemma E.1, we have

$$
\mathcal { E } _ { D } ^ { \mathrm { s g n } } \gtrsim ( \eta _ { \mathrm { p e a k } } D ) ^ { - 2 s } + \eta _ { \mathrm { p e a k } } ( \eta _ { \mathrm { p e a k } } D ) ^ { - ( 1 - 2 / q ) } .
$$

Balancing the two terms yields

$$
\eta _ { \mathrm { p e a k } } \approx D ^ { - \frac { 2 s - 1 + 2 / q } { 2 s + 2 / q } } .
$$

Substituting this into the loss formula gives

$$
\mathcal { E } _ { D } ^ { \mathrm { s g n } } \approx D ^ { - \frac { s q } { s q + 1 } } .
$$

If $\textstyle 0 < s < { \frac { 1 } { 2 } } - { \frac { 1 } { q } }$ , the choice of $\eta _ { \mathrm { p e a k } }$ above grows with D, which violates the stability condition $\eta _ { \mathrm { p e a k } } \lesssim 1$ . We therefore take $\eta _ { \mathrm { p e a k } } \approx 1$ . Choosing $\begin{array} { r } { p > \frac { 2 s } { 1 - 2 s } } \end{array}$ gives $\begin{array} { r } { \alpha _ { \mathrm { s g n } } = \operatorname* { m i n } \{ \frac { q } { 2 } , p + 1 \} > \frac { 1 } { 1 - 2 s } } \end{array}$ The noise term is $o ( D ^ { - 2 s } )$ , including the logarithmic factor when $p + 1 = q / 2$ . Hence

$$
\mathcal { E } _ { D } ^ { \mathrm { s g n } } \approx D ^ { - 2 s } .
$$

## E.3 Data scaling of SGD

We now review the data scaling result for SGD. The SGD FSL yields the following achievable rates for power decay (Li et al., 2026c). Proposition E.5 supplies the matching lower bounds for all admissible schedules.

For SGD, let $\vartheta _ { t }$ denote the process in (32) and $\mathcal { E } _ { D } ^ { \mathrm { s g d } } : = \mathbb { E } [ \mathcal { E } ( \vartheta _ { T _ { D } } ) ]$ its final expected excess risk at $T _ { D } = \Gamma ( D )$ . The SGD FSL from Li et al. (2025) gives

$$
\mathcal { E } _ { D } ^ { \mathrm { s g d } } \gtrsim T _ { D } ^ { - s } + \int _ { 0 } ^ { T _ { D } } ( 1 + T _ { D } - t ) ^ { - ( 2 - 1 / q ) } \varphi ( t ) \ \mathrm { d } t .\tag{69}
$$

The following lemma summarizes the SGD FSL under power decay schedules.

Lemma E.3 (SGD FSL under power decay schedules; Li et al. (2026c)). Consider the SGD FSL (69) with the power decay schedule (66). Let $\alpha _ { \mathrm { s g d } } = \operatorname* { m i n } \{ q , p + 1 \}$ , where p is the power decay exponent. Then the final excess risk satisfies

$$
\mathcal { E } _ { D } ^ { \mathrm { s g d } } \asymp ( \eta _ { \mathrm { p e a k } } D ) ^ { - s } + \eta _ { \mathrm { p e a k } } ( \eta _ { \mathrm { p e a k } } D ) ^ { - \left( 1 - \frac { 1 } { \alpha _ { \mathrm { s g d } } } \right) }\tag{70}
$$

up to a logarithmic factor when $p + 1 = q$

Proof. The proof is the same as that of Lemma E.1, with the SignSGD signal exponent 2s replaced by the SGD signal exponent s, and the effective capacity exponent $q / 2$ replaced by $q .$ Therefore, with $\alpha _ { \mathrm { s g d } } = \operatorname* { m i n } \{ q , p + 1 \}$ , the claimed result follows from the same calculation.

Optimizing (70) over the peak learning rate gives the data-scaling result:

Theorem E.4 (Data scaling for SGD; Li et al. (2026c)). Consider the intrinsic time SDE for SGD (32) with the power decay schedule (66). Then we have:

$\begin{array} { r } { I f s \geqslant 1 - \frac { 1 } { q } } \end{array}$ , then choosing $p > q - 1 , \eta _ { \mathrm { p e a k } } \approx D ^ { - \frac { s - 1 + 1 / q } { s + 1 / q } }$ yields

$$
\mathcal { E } _ { D } ^ { \mathrm { s g d } } \approx D ^ { - \frac { s q } { s q + 1 } } .
$$

$\begin{array} { r } { I f 0 < s < 1 - \frac { 1 } { q } } \end{array}$ , then choosing $\begin{array} { r } { p > \frac { s } { 1 - s } , \eta _ { \mathrm { p e a k } } \asymp 1 } \end{array}$ yields

$$
\mathcal { E } _ { D } ^ { \mathrm { s g d } } \approx D ^ { - s } .
$$

Proof. First consider $\begin{array} { r } { s \geqslant 1 - \frac { 1 } { q } } \end{array}$ Choosing $p > q - 1$ gives $\alpha _ { \mathrm { s g d } } = q$ , then applying Lemma E.3, we have

$$
\mathcal { E } _ { D } ^ { \mathrm { s g d } } \Xi ^ { - } ( \eta _ { \mathrm { p e a k } } D ) ^ { - s } + \eta _ { \mathrm { p e a k } } ( \eta _ { \mathrm { p e a k } } D ) ^ { - ( 1 - 1 / q ) } .
$$

Balancing the signal and noise terms gives $\eta _ { \mathrm { p e a k } } \approx D ^ { - \frac { s - 1 + 1 / q } { s + 1 / q } }$ and the minimax rate $D ^ { - s q / ( s q + 1 ) }$ Now consider $\begin{array} { r } { 0 < s < 1 - \frac { 1 } { q } . } \end{array}$ , the choice of $\eta _ { \mathrm { p e a k } }$ above grows with $D ,$ which violates the stability condition $\eta _ { \mathrm { p e a k } } \lesssim 1 .$ We therefore take $\eta _ { \mathrm { p e a k } } \ : \approx \ : 1$ . Choosing $p > { \frac { s } { 1 - \varepsilon } }$ gives $\alpha _ { \mathrm { s g d } } = \mathrm { m i n } \{ q , p + 1 \} > \frac { 1 } { 1 - s }$ . The noise term is $o ( D ^ { - s } )$ , including the logarithmic factor when $p + 1 = q$ . Hence

$$
\mathcal { E } _ { D } ^ { \mathrm { s g d } } \approx D ^ { - s } .
$$

## E.4 Optimality of power decay schedules in data scaling

We compare all admissible learning rate schedules at the same sample budget D. A lower bound on accumulated noise limits the benefit of increasing the total intrinsic time, while the peak learning rate bounds the time available for learning the signal.

Proposition E.5 (Optimal rates over bounded learning rate schedules). Consider the SignSGD and SGD models of Appendix $E . 2$ and Appendix $E . 3$ with $B \ = \ 1$ Let $A _ { D }$ consist of all nonnegative schedules on $[ 0 , D ]$ bounded by a fixed admissible peak $C _ { 0 } \ > \ 0$ , independent of D. Then

$$
\operatorname* { i n f } _ { \eta \in A _ { D } } \mathcal { E } _ { D } ^ { \mathrm { s g n } } \gtrsim D ^ { - \operatorname* { m i n } \{ 2 s , s q / ( s q + 1 ) \} } , \qquad \operatorname* { i n f } _ { \eta \in A _ { D } } \mathcal { E } _ { D } ^ { \mathrm { s g d } } \stackrel { } { \sim } D ^ { - \operatorname* { m i n } \{ s , s q / ( s q + 1 ) \} } .
$$

Proof. Fix $\eta \in { \mathcal { A } } _ { D }$ and write $T = \Gamma ( D )$ . For SignSGD, summing the global moment lower bound in Corollary D.3 by Lemma D.1, as in Appendix D.6, bounds the risk below by the sum of the signal and noise terms for every $T \geqslant 0$ . For SGD the corresponding lower bound follows from (69) when $T \geqslant 1$ . For $T < 1$ , the first coordinate of the mean SGD error retains the factor $e ^ { - \lambda _ { 1 } T }$ from (32), so its risk is bounded below by a positive constant. The SignSGD signal lower bound gives the same conclusion. It suffices to consider $T \geqslant 1$

For a forgetting kernel $( 1 ~ + ~ u ) ^ { - \beta }$ , write the noise contribution in training steps. Cauchy-Schwarz on $[ 0 , D ]$ gives

$$
\begin{array} { r l r } & { } & { \displaystyle \int _ { 0 } ^ { D } ( 1 + T - \Gamma ( z ) ) ^ { - \beta } \eta ( z ) ^ { 2 } \mathrm { d } z \geqslant \frac { 1 } { D } \left[ \displaystyle \int _ { 0 } ^ { D } ( 1 + T - \Gamma ( z ) ) ^ { - \beta / 2 } \eta ( z ) \mathrm { d } z \right] ^ { 2 } } \\ & { } & { \quad \quad \quad = \frac { 1 } { D } \left[ \displaystyle \int _ { 0 } ^ { T } ( 1 + u ) ^ { - \beta / 2 } \mathrm { d } u \right] ^ { 2 } . } \end{array}
$$

The equality uses $\Gamma ^ { \prime } ( z ) = \eta ( z )$ almost everywhere and also holds on schedules with zero learning rates, including entire intervals. With $\beta = 2 - 2 / q$ for SignSGD and $\beta = 2 - 1 / q$ for SGD, adding the signal lower bounds yields

$$
\mathcal { E } _ { D } ^ { \mathrm { s g n } } \gtrsim T ^ { - 2 s } + \frac { T ^ { 2 / q } } { D } , \qquad \mathcal { E } _ { D } ^ { \mathrm { s g d } } \gtrsim T ^ { - s } + \frac { T ^ { 1 / q } } { D } .
$$

The two terms balance at $T = D ^ { 1 / ( 2 s + 2 / q ) }$ for SignSGD and at $T = D ^ { 1 / ( s + 1 / q ) }$ for SGD, giving a lower bound of order $D ^ { - s q / ( s q + 1 ) }$ in both cases. The cap gives $T \leqslant C _ { 0 } D$ , so the signal terms also bound the respective risks below by constant multiples of $D ^ { - 2 s }$ and $D ^ { - s }$ . Taking the larger of the two lower bounds for each optimizer gives the asserted exponents. The power decay schedules in Theorem E.2 and Theorem E.4, with the peak prefactors chosen to satisfy the cap, attain these rates. □

When SignSGD attains $D ^ { - s q / ( s q + 1 ) }$ , its exponent agrees with the statistical minimax benchmark discussed in Section 4.1. The proposition establishes optimality over schedules within the continuous models.

## F Batch-size scaling analysis

We prove Theorem 4.4 by comparing the intrinsic time needed to preserve the optimal small-batch risk with the time available at batch size B. This gives one argument for all task regimes in the theorem. Write $\alpha = s + 1 / q$ and $N = D / B$

Proof. The small-batch target. Let $T = \Gamma _ { \mathrm { s g d } } ( N )$ and $U = \Gamma _ { B } ( N )$ denote the final intrinsic times of SGD and SignSGD. The data-scaling analysis in Appendix E identifies the optimal $B = 1$ time scales

$$
T _ { * } = \operatorname* { m i n } \{ D , D ^ { 1 / \alpha } \} , \qquad U _ { * } = \operatorname* { m i n } \{ D , D ^ { 1 / ( 2 \alpha ) } \} .
$$

By Proposition E.5, the corresponding optimal risks satisfy

$$
\begin{array} { r } { \mathcal E _ { \star } ^ { \mathrm { s g d } } ( D , 1 ) \approx T _ { * } ^ { - s } , \qquad \mathcal E _ { \star } ^ { \mathrm { s g n } } ( D , 1 ) \approx U _ { * } ^ { - 2 s } . } \end{array}
$$

Thus the question is how large B can be while still reaching these time scales.

The time available at batch size B. Since there are only $N = D / B$ updates, the peak learning rate bounds $\eta _ { \mathrm { p e a k } } ^ { \mathrm { s g d } } \lesssim 1$ and $\sqrt { B } \eta _ { \mathrm { p e a k } } ^ { \mathrm { s g n } } \lesssim 1$ imply

$$
T \lesssim \frac { D } { B } , \qquad U \lesssim \frac { D } { B } .
$$

The signal lower bounds for SGD and Corollary D.3 for SignSGD hold for every schedule, including before burn-in. Consequently, for $1 \leqslant B \leqslant D$ 2

$$
\mathcal { E } _ { \star } ^ { \mathrm { s g d } } ( D , B ) \gtrsim \left( \frac { B } { D } \right) ^ { s } , \qquad \mathcal { E } _ { \star } ^ { \mathrm { s g n } } ( D , B ) \gtrsim \left( \frac { B } { D } \right) ^ { 2 s } .
$$

Comparing with the small-batch risks above, preserving a fixed risk ratio requires, respectively,

$$
B \lesssim \frac { D } { T _ { * } } = \operatorname* { m a x } \{ 1 , D ^ { 1 - 1 / \alpha } \} , \qquad B \lesssim \frac { D } { U _ { * } } = \operatorname* { m a x } \{ 1 , D ^ { 1 - 1 / ( 2 \alpha ) } \} .
$$

These are the claimed CBS upper bounds.

Reaching the target times. Use the power-decay schedule with fixed $p > q - 1$ and the peak learning rates stated after Theorem 4.4. Within the respective batch-size ranges above, they give $T \gtrsim T _ { * }$ and $U \gtrsim U _ { * }$ . For SignSGD, we also need the training-time conditions

$$
U \gtrsim \sqrt { B } , \qquad U \gtrsim B ^ { 1 / ( 2 \alpha ) } .
$$

If $\alpha \leqslant 1 / 2$ , the claimed batch range is bounded and $U \approx D ,$ so both conditions hold for large D. If $1 / 2 < \alpha < 3 / 2$ , substituting $B \lesssim D ^ { 1 - 1 / ( 2 \alpha ) }$ and $U \stackrel { \cdot } { \sim } D ^ { 1 / ( 2 \alpha ) }$ verifies both conditions. The power-decay estimates from Appendix E therefore give

$$
\begin{array} { l } { \displaystyle \mathcal { E } ^ { \mathrm { s g d } } ( D , B ; \eta ) \lesssim T _ { * } ^ { - s } + \frac { T _ { * } ^ { 1 / q } } { D } \stackrel { } { \sim } T _ { * } ^ { - s } , } \\ { \displaystyle \mathcal { E } ^ { \mathrm { s g n } } ( D , B ; \eta ) \lesssim U _ { * } ^ { - 2 s } + \frac { U _ { * } ^ { 2 / q } } { D } \stackrel { } { \sim } U _ { * } ^ { - 2 s } , } \end{array}
$$

where $\mathcal { E } ^ { \mathrm { s g d } } ( D , B ; \eta )$ and $\mathcal { E } ^ { \mathrm { s g n } } ( D , B ; \eta )$ denote the final risks under these schedules. Hence both methods retain their respective small-batch risk throughout the claimed ranges. If the claimed CBS is of constant order, the matching lower bound already follows from $B _ { \mathrm { c r i t } } ( D ) \geqslant 1$

The fixed comparison factor. Choose common constants $0 < c _ { - } \leqslant C _ { + }$ such that each optimizer's $B = 1$ optimal risk is at least $c _ { - }$ times its rate above, and the constructed schedules

have risk at most $C _ { + }$ times that rate. These constants do not depend on D or B. For each optimizer, the constructed schedule therefore satisfies

$$
\frac { \mathcal { E } ( D , B ; \eta ) } { \mathcal { E } _ { \star } ( D , 1 ) } \leqslant \frac { C _ { + } } { c _ { - } }
$$

throughout its claimed batch-size range. Fix the comparison factor $1 + \delta$ in Definition 4.3 sufficiently large to absorb this ratio, independently of D and B. The constructed schedules then give the CBS lower bounds for this fixed comparison factor. Together with the upper bounds, this proves (6). □

The super-easy regime. Along the balanced SignSGD schedule, $U \approx D ^ { 1 / ( 2 \alpha ) }$ . The burn-in requirement $U \gtrsim \sqrt { B }$ gives $B \lesssim D ^ { 1 / \alpha }$ , while the peak learning rate cap gives $B \lesssim D ^ { 1 - 1 / ( 2 \alpha ) }$ . At $s = 3 / 2 - 1 / q$ both bounds have order $D ^ { 2 / 3 }$ , and the constants determine which one is reached first. For $s > 3 / 2 - 1 / q .$ the burn-in condition restricts this schedule at a smaller batch scale. Beyond that scale the two-sided FSL for the balanced schedule no longer gives the optimal-risk comparison in Theorem 4.4.

## G Heuristic extension to RMSprop and Adam

This appendix develops a heuristic extension of the SignSGD analysis to RMSprop and Adam. We work in the diagonal model and approximate the second-moment normalizer by the coordinate-wise gradient-noise variance after burn-in. For Adam, we additionally require a separation between its momentum memory and the coordinate relaxation time. These assumptions yield effective dynamics with the same FSL exponents and batch-size dependence as SignSGD. These results are supported by our experiments involving Adam and RMSprop in Appendix H.2.

Ignoring the € in the denominator and bias correction, the Adam update is

$$
\begin{array} { r } { \mathbf { m } _ { k + 1 } = \beta _ { 1 } \mathbf { m } _ { k } + ( 1 - \beta _ { 1 } ) \mathbf { g } _ { k } , \ \mathbf { v } _ { k + 1 } = \beta _ { 2 } \mathbf { v } _ { k } + ( 1 - \beta _ { 2 } ) \mathbf { g } _ { k } ^ { 2 } , \quad \theta _ { k + 1 } = \theta _ { k } - \eta _ { k } \mathbf { m } _ { k + 1 } / \sqrt { \mathbf { v } _ { k + 1 } } , } \end{array}
$$

where all operations are coordinate-wise, with $\mathbf { m } _ { 0 } = \mathbf { v } _ { 0 } = \mathbf { 0 }$ . RMSprop corresponds to the case $\beta _ { 1 } = 0$ . To expose the main mechanism, we first focus on RMSprop and then discuss Adam as a momentum-filtered variant.

## G.1 Second moments and coordinate curvatures

Consider one coordinate with curvature λ and write the stochastic gradient as $g _ { k } = \mu _ { k } + \xi _ { k }$ with $\mu _ { k }$ being the full-batch gradient. By Assumption 2.5, in the label-noise-dominated regime, $\mathbb { E } [ \xi _ { k } ] = 0$ and $\mathbb { E } [ \xi _ { k } ^ { 2 } ] \gtrsim \sigma _ { B } ^ { 2 } \lambda$ , where $\sigma _ { B } = 1 / \sqrt { B }$ is the batch noise scale from Section 2. For the following heuristic scaling calculation, we suppress fixed noise-curvature comparison constants and use $\sigma _ { B } ^ { 2 } \lambda$ as a representative variance. In the noise-dominated regime: $\mu _ { k } ^ { 2 } \ll \mathbb { E } [ \xi _ { k } ^ { 2 } ]$ , we have $\begin{array} { r } { v _ { k + 1 } = ( 1 - \beta _ { 2 } ) \sum _ { j = 0 } ^ { k } \beta _ { 2 } ^ { k - j } g _ { j } ^ { 2 } \approx ( 1 - \beta _ { 2 } ) \sum _ { j = 0 } ^ { k } \beta _ { 2 } ^ { k - j } \xi _ { j } ^ { 2 } } \end{array}$ . Approximating the noise across steps by independent draws with fixed variance, for $k ( 1 - \beta _ { 2 } ) \gg 1$

$$
\begin{array} { r l } & { \mathbb { E } [ v _ { k + 1 } ] \approx ( 1 - \beta _ { 2 } ^ { k + 1 } ) \sigma _ { B } ^ { 2 } \lambda \approx \sigma _ { B } ^ { 2 } \lambda , } \\ & { \mathrm { V a r } [ v _ { k + 1 } ] \approx \displaystyle \frac { ( 1 - \beta _ { 2 } ) ( 1 - \beta _ { 2 } ^ { 2 ( k + 1 ) } ) } { 1 + \beta _ { 2 } } \mathrm { V a r } [ \xi ^ { 2 } ] \approx ( 1 - \beta _ { 2 } ) \sigma _ { B } ^ { 4 } \lambda ^ { 2 } , } \end{array}
$$

where we use $\mathbb { E } [ \xi ^ { 4 } ] \lesssim ( \mathbb { E } [ \xi ^ { 2 } ] ) ^ { 2 }$ for Gaussian noise. Hence, $v _ { k + 1 } \approx \sigma _ { B } ^ { 2 } \lambda \left[ 1 + O _ { \mathbb { P } } \left( \sqrt { 1 - \beta _ { 2 } } \right) \right]$ When $\beta _ { 2 }$ is close to 1, the EMA self-averages and for $k ( 1 - \beta _ { 2 } ) \gg 1$

$$
v _ { k + 1 } \approx \sigma _ { B } ^ { 2 } \lambda .
$$

Thus, in the noise-dominated regime, RMSprop and Adam estimate a coordinate-wise gradient-noise variance proportional to curvature.

## G.2 The effective RMSprop dynamics

Let $u _ { k } = \theta _ { k } - \theta ^ { * }$ be the deviation along the coordinate with curvature λ. Using the representative normalization $\sqrt { v _ { k + 1 } } \approx \sigma _ { B } \sqrt { \lambda }$ , the RMSprop update in the noise-dominated regime becomes

$$
u _ { k + 1 } \approx u _ { k } - \eta _ { k } \frac { \lambda u _ { k } + \xi _ { k } } { \sigma _ { B } \sqrt { \lambda } } = u _ { k } - \eta _ { k } \left( \sigma _ { B } ^ { - 1 } \sqrt { \lambda } u _ { k } + \zeta _ { k } \right) ,\tag{71}
$$

where $\zeta _ { k } = \xi _ { k } / ( \sigma _ { B } \sqrt { \lambda } )$ is the normalized noise satisfying $\mathbb { E } [ \zeta _ { k } ^ { 2 } ] \asymp 1$ . In the continuous model, let $u _ { j } ( t )$ be the j-th coordinate deviation at intrinsic time $\begin{array} { r } { \dot { t } = \Gamma _ { B } ( z ) = \sigma _ { B } ^ { - 1 } \int _ { 0 } ^ { z } \eta ( v ) } \end{array}$ dv. We use $\varphi ( t ) = \varphi _ { \mathrm { s g n } } ( t ) = \eta ( \Gamma _ { B } ^ { - 1 } ( t ) )$ and $n _ { B } ( t ) = \varphi ( t ) / \sqrt { B }$ , as in the main text. With the fixed normalized noise-variance factor also suppressed, the representative effective dynamics become

$$
\mathrm { d } u _ { j } ( t ) = - \sqrt { \lambda _ { j } } u _ { j } ( t ) \mathrm { d } t + \sqrt { \frac { \varphi ( t ) } { \sqrt { B } } } \mathrm { d } W _ { j } ( t ) .\tag{72}
$$

Therefore, the second moment of the j-th coordinate is given by

$$
\mathbb { E } [ u _ { j } ( t ) ^ { 2 } ] = u _ { j } ( 0 ) ^ { 2 } e ^ { - 2 \sqrt { \lambda _ { j } } t } + \frac { 1 } { \sqrt { B } } \int _ { 0 } ^ { t } e ^ { - 2 \sqrt { \lambda _ { j } } ( t - t ^ { \prime } ) } \varphi ( t ^ { \prime } ) \mathrm { d } t ^ { \prime } .\tag{73}
$$

Remark G.1 (Characteristic time). The j-th coordinate dynamics has characteristic time $\tau _ { j } \sim$ $\lambda _ { j } ^ { - 1 / 2 }$ in intrinsic time. In contrast, the characteristic time of SGD is $\lambda _ { j } ^ { - 1 }$ . This means the coordinate-wise normalization introduces acceleration by compressing the characteristic time from $\lambda ^ { - 1 }$ to $\lambda ^ { - 1 / 2 }$ , which is curvature adaptive. Batch size accelerates the intrinsic clock through the factor $\sqrt { B }$ in its definition.

## G.3 The FSL of RMSprop

We sum the coordinate-wise dynamics over the infinite power-law spectrum to identify the signal-learning and noise-forgetting profiles.

Multiplying (73) by $\lambda _ { j }$ and summing over coordinates gives the dynamics of total loss $\frac { 1 } { 2 } \mathbb { E } [ \mathbf { u } _ { t } ^ { \top } \mathbf { H } \mathbf { u } _ { t } ]$

$$
\frac { 1 } { 2 } \mathbb { E } [ \mathbf { u } _ { t } ^ { \top } \mathbf { H } \mathbf { u } _ { t } ] \approx \sum _ { j } \lambda _ { j } u _ { j } ( 0 ) ^ { 2 } e ^ { - 2 \sqrt { \lambda _ { j } } t } + \frac { 1 } { \sqrt { B } } \int _ { 0 } ^ { t } \sum _ { j } \lambda _ { j } e ^ { - 2 \sqrt { \lambda _ { j } } ( t - t ^ { \prime } ) } \varphi ( t ^ { \prime } ) \mathrm { d } t ^ { \prime } .\tag{74}
$$

Hence, the loss dynamics of RMSprop admit the FSL with noise injection intensity $n _ { B } ( t ) = $ $\varphi ( t ) / \sqrt { B }$ and the signal learning and forgetting kernel given by

$$
S _ { \mathrm { r m s } } ( t ) = \sum _ { j } \lambda _ { j } u _ { j } ( 0 ) ^ { 2 } e ^ { - 2 \sqrt { \lambda _ { j } } t } , \quad K _ { \mathrm { r m s } } ( t ) = \sum _ { j } \lambda _ { j } e ^ { - 2 \sqrt { \lambda _ { j } } t } .
$$

The origin of power-law scaling. Under the power-law assumption (Assumption 2.3): $\lambda _ { j } \approx$ $j ^ { - q }$ and $w _ { j } : = \lambda _ { j } u _ { j } ( 0 ) ^ { 2 } \lesssim j ^ { - ( 1 + s q ) }$ , we have

$$
S _ { \mathrm { r m s } } ( t ) \stackrel { = } \sim ( 1 + t ) ^ { - 2 s } , \qquad K _ { \mathrm { r m s } } ( t ) \stackrel { = } \sim ( 1 + t ) ^ { - ( 2 - 2 / q ) } .
$$

We now provide a physical-style derivation of why they follow the above power-law decay. The coordinate with curvature $\lambda _ { j }$ has characteristic time $\tau _ { j } \approx \lambda _ { j } ^ { - 1 / 2 }$ . Take the signal-learning term as an example, $\begin{array} { r } { S ( t ) = \sum _ { j } w _ { j } e ^ { - 2 \sqrt { \lambda _ { j } } t } } \end{array}$ . At time $t ,$ the modes with $\tau _ { j } \lesssim t$ have essentially been learned, while the modes with $\tau _ { j } \gtrsim t$ are still largely unlearned. Since $\lambda _ { j } \approx j ^ { - q }$ , the moving frontier $j _ { t }$ is determined by $\tau _ { j _ { t } } \asymp t ,$ yielding $j _ { t } \approx t ^ { 2 / q }$ , as in Section 6.3. For $t \gtrsim 1$ 2

$$
S _ { \mathrm { r m s } } ( t ) = \sum _ { j \leqslant j _ { t } } w _ { j } \underbrace { e ^ { - 2 \sqrt { \lambda _ { j } } t } } _ { \approx 0 } + \sum _ { j > j _ { t } } w _ { j } \underbrace { e ^ { - 2 \sqrt { \lambda _ { j } } t } } _ { \approx 1 } \approx \sum _ { j > j _ { t } } w _ { j } \approx ( j _ { t } ) ^ { - s q } \approx t ^ { - 2 s } .\tag{75}
$$

The power-law scaling of the forgetting kernel follows from the same moving-cutoff argument.

Remark G.2 (Tail dominates scaling law). The above derivation suggests a simple picture of learning in power-law data. The expected excess risk is a sum over spectral modes, and each mode has its own characteristic time $\tau _ { j }$ . At time $t ,$ the moving frontier $j _ { t }$ is determined by $\tau _ { j _ { t } } \ \approx \ t$ . Modes with $\tau _ { j } \ \lesssim \ t$ have already been learned and contribute little to the remaining signal error, whereas modes with $\tau _ { j } \gtrsim t$ are still largely unlearned. Thus the signal-learning term is essentially the tail mass of the unlearned modes. Analogously, the power-law decay of $S ( t )$ therefore comes from how fast this moving frontier advances through the spectrum. In summary,

![](images/01202cacb52128dd6be56514060462d2a233eee8f82c9513a6f26b1cffd17fff.jpg)  
Figure 6: Tail-sum scaling

$$
S ( t ) \approx \sum _ { j \gtrsim j _ { t } } w _ { j } , \qquad K ( t ) \approx \sum _ { j \gtrsim j _ { t } } \lambda _ { j } , \qquad \mathrm { w i t h } \ \tau _ { j _ { t } } \stackrel { _ { \ast } } { \sim } t .\tag{76}
$$

Figure 6 demonstrates that the above moving-tail approximation indeed yields the right scaling for both signal-learning and noise-forgetting terms

## G.4 The FSL of Adam

Adam differs from RMSprop only through the first-moment accumulator in the numerator. For a complementary analysis of how momentum affects data scaling and the critical batch size in one-pass kernel regression, see Wang et al. (2026a). Under the same denominator approximation $v _ { k + 1 } \approx \sigma _ { B } ^ { 2 } \lambda$ , the first-moment EMA has an effective memory window $M _ { 1 } \stackrel { \textstyle = } { \sim } ( 1 - \beta _ { 1 } ) ^ { - 1 }$ steps. After burn-in, $k \gg M _ { 1 }$ , its expansion is

$$
m _ { k + 1 } = ( 1 - \beta _ { 1 } ) \sum _ { j = 0 } ^ { k } \beta _ { 1 } ^ { j } \mu _ { k - j } + ( 1 - \beta _ { 1 } ) \sum _ { j = 0 } ^ { k } \beta _ { 1 } ^ { j } \xi _ { k - j } .
$$

Assume that the learning rate varies little across this window and that $\sqrt { B } \eta _ { k } M _ { 1 } \ll \tau _ { \lambda }$ , where $\tau _ { \lambda } \ \approx \ \lambda ^ { - 1 / 2 }$ is the intrinsic relaxation time. Recall from (71) that $\zeta _ { k } = \xi _ { k } / ( \sigma _ { B } \sqrt { \lambda } )$ is the normalized gradient noise. Its first-moment EMA, denoted by $\zeta _ { k } ,$ gives the noise contribution to the normalized Adam update. The drift contribution is then approximately quasi-static, giving

$$
u _ { k + 1 } \approx u _ { k } - \eta _ { k } \left( \sigma _ { B } ^ { - 1 } \sqrt { \lambda } u _ { k } + \widetilde { \zeta } _ { k } \right) , \qquad \widetilde { \zeta } _ { k } : = ( 1 - \beta _ { 1 } ) \sum _ { j = 0 } ^ { k } \beta _ { 1 } ^ { j } \zeta _ { k - j } .\tag{77}
$$

Assume that the normalized noises $\zeta _ { k }$ are independent and identically distributed, with mean zero and variance $\sigma _ { \zeta } ^ { 2 } : = \mathrm { V a r } [ \zeta _ { k } ] \asymp 1$ . In the stationary approximation after burn-in,

$$
\mathrm { V a r } [ \widetilde { \zeta } _ { k } ] \approx ( 1 - \beta _ { 1 } ) ^ { 2 } \sum _ { j = 0 } ^ { \infty } \beta _ { 1 } ^ { 2 j } \sigma _ { \zeta } ^ { 2 } = \frac { 1 - \beta _ { 1 } } { 1 + \beta _ { 1 } } \sigma _ { \zeta } ^ { 2 } , \quad \mathrm { C o v } ( \widetilde { \zeta } _ { k } , \widetilde { \zeta } _ { k + h } ) \approx \beta _ { 1 } ^ { h } \mathrm { V a r } [ \widetilde { \zeta } _ { k } ] , \quad h \geqslant 0 .
$$

The first equation gives the single-step variance reduction. The second equation shows that different steps remain positively correlated because their moving averages share past noise. For a block of $L \gg M _ { 1 }$ steps, the variance of the accumulated noise therefore satisfies

$$
\frac { 1 } { L } \operatorname { V a r } \left[ \sum _ { r = 0 } ^ { L - 1 } \widetilde { \zeta } _ { k + r } \right] \approx \sigma _ { \zeta } ^ { 2 } \frac { 1 - \beta _ { 1 } } { 1 + \beta _ { 1 } } \left[ 1 + 2 \sum _ { h = 1 } ^ { L - 1 } \left( 1 - \frac { h } { L } \right) \beta _ { 1 } ^ { h } \right] \approx \sigma _ { \zeta } ^ { 2 } \frac { 1 - \beta _ { 1 } } { 1 + \beta _ { 1 } } \left( 1 + \frac { 2 \beta _ { 1 } } { 1 - \beta _ { 1 } } \right) = \sigma _ { \zeta } ^ { 2 } .
$$

Momentum spreads the noise of each step across its memory window. For $L \gg M _ { 1 }$ , these contributions accumulate within the block, giving the same leading noise variance $L \sigma _ { \zeta } ^ { 2 }$ as the unfiltered noise despite smoother individual updates.

Now choose a block with $M _ { 1 } \ll L \ll \tau _ { \lambda } / ( \sqrt { B } \eta _ { k } )$ , and assume that the learning rate remains approximately constant over this block. The intrinsic time elapsed over this block and the accumulated parameter-noise variance are

$$
\Delta t \approx \sqrt { B } L \eta _ { k } , \qquad \mathrm { V a r } \left[ \sum _ { r = 0 } ^ { L - 1 } \eta _ { k + r } \widetilde { \zeta } _ { k + r } \right] \approx L \eta _ { k } ^ { 2 } \sigma _ { \zeta } ^ { 2 } \approx \frac { \sigma _ { \zeta } ^ { 2 } \eta _ { k } } { \sqrt { B } } \Delta t .
$$

Together with the drift in (77), this yields the effective intrinsic-time SDE

$$
\mathrm { d } u ( t ) = - \sqrt { \lambda } u ( t ) \ \mathrm { d } t + \sqrt { \frac { \sigma _ { \zeta } ^ { 2 } \varphi ( t ) } { \sqrt { B } } } \ \mathrm { d } W _ { t } .
$$

This is the same form as the RMSprop model (72), with $\sigma _ { \zeta } ^ { 2 } \lesssim 1$ absorbed into the constants. Therefore, following the same argument as for RMSprop, we obtain the FSL for Adam with noise injection intensity $n ( t ) \stackrel { } { \sim } n _ { B } ( t ) = \varphi ( t ) / \sqrt { B }$ and

$$
S _ { \mathrm { a d a m } } ( t ) = ( 1 + t ) ^ { - 2 s } , \qquad K _ { \mathrm { a d a m } } ( t ) = ( 1 + t ) ^ { - ( 2 - 2 / q ) } .\tag{78}
$$

## H Experimental details and additional results

## H.1 Fitting and prediction using FSL

Although derived from a kernel regression model, the FSL framework is expected to model loss dynamics in more general training scenarios, as the decomposition into “irreducible, signal, and noise" components carries a broader physical interpretation. Inspired by the functional form of (3), we heuristically propose a parametrized ansatz:

$$
\begin{array} { r l } & { L _ { \Theta } ( t ; \varphi , B ) = \underbrace { L _ { \infty } } _ { \mathrm { i r r e d u c i b l e ~ l o s s } } + \underbrace { C _ { S } ( A _ { S } + t ) ^ { - \alpha } } _ { \mathrm { s i g n a l - l e a r n i n g ~ t e r m } } } \\ & { \quad + \underbrace { \frac { C _ { K } } { \sqrt { B } } \int _ { 0 } ^ { t } \bigl ( A _ { K } + t - r \bigr ) ^ { - \nu } \varphi ( r ) \ \mathrm { d } r } _ { \mathrm { n o i s e - f o r g e t t i n g ~ t e r m } } , } \end{array}\tag{79}
$$

Here $\begin{array} { r } { t = \Gamma _ { B } ( z ) = \sqrt { B } \int _ { 0 } ^ { z } \eta ( u ) } \end{array}$ du denotes intrinsic time, B is the batch size, $\varphi ( t ) = \eta ( \Gamma _ { B } ^ { - 1 } ( t ) )$ is the learning rate schedule in intrinsic time, and $\Theta = ( L _ { \infty } , C _ { S } , A _ { S } , \alpha , C _ { K } , A _ { K } , \nu )$ denotes fitting parameters. Thus $\varphi = \varphi _ { \mathrm { s g n } }$ and $n _ { B } ( t ) = \varphi ( t ) / \sqrt { B }$ . The noise-forgetting term in (79) is equivalently CK $\begin{array} { r } { \int _ { 0 } ^ { t } ( A _ { K } + t - r ) ^ { - \nu } n _ { B } ( r ) } \end{array}$ dr. We use this base form for synthetic experiments and extend its learning-rate dependence for language model training as described in Appendix H.3

Discrete FSL. To obtain the left-endpoint form used in the main text, define

$$
t _ { k } = \sqrt { B } \sum _ { j = 0 } ^ { k - 1 } \eta _ { j } , \qquad I _ { j } = [ t _ { j } , t _ { j + 1 } ) ,
$$

$$
\Delta t _ { j } = t _ { j + 1 } - t _ { j } = \sqrt { B } \eta _ { j } .
$$

On each nonempty interval $I _ { j }$ , take $\varphi ( r ) = \eta _ { j }$ . The left-endpoint rectangle rule gives

$$
\begin{array} { r c l } { \displaystyle \frac { 1 } { \sqrt { B } } \int _ { 0 } ^ { t _ { k } } K ( t _ { k } - r ) \varphi ( r ) \ \mathrm { d } r \approx \frac { 1 } { \sqrt { B } } \sum _ { j = 0 } ^ { k - 1 } K ( t _ { k } - t _ { j } ) \varphi ( t _ { j } ) \Delta t _ { j } } \\ { \displaystyle = \sum _ { j = 0 } ^ { k - 1 } K ( t _ { k } - t _ { j } ) \eta _ { j } ^ { 2 } . } \end{array}
$$

Steps with $\eta _ { j } = 0$ contribute zero. With $K ( u ) = ( A _ { K } + u ) ^ { - \nu }$ , this yields the parameterized form in (7).

Numerical convention used in the reported fits. We discretize the noise integral on the cumulative-learning-rate grid

$$
\tau _ { k } = \sum _ { j = 0 } ^ { k - 1 } \eta _ { j } , \qquad \tau _ { 0 } = 0 , \qquad \Delta \tau _ { j } = \tau _ { j + 1 } - \tau _ { j } = \eta _ { j } .
$$

Here k is the number of completed updates, so $\ell _ { k }$ is the loss at $\pmb { \theta } _ { k }$ and $t _ { k } = \sqrt { B } \tau _ { k }$ is the same intrinsic clock as in the main text. The first stored loss corresponds to $k = 1$ . Thus the signal term is $C _ { S } ( A _ { S } + \sqrt { B } \tau _ { k } ) ^ { - \alpha }$

To discretize the noise integral in (79), set $r = \sqrt { B } v$ and write $\psi ( v ) = \varphi ( { \sqrt { B } } v )$ for the learning rate expressed in cumulative-learning-rate time. Then the noise term at $t _ { k }$ is

$$
\frac { C _ { K } } { \sqrt { B } } \int _ { 0 } ^ { t _ { k } } K ( t _ { k } - r ) \varphi ( r ) \mathrm { d } r = C _ { K } \int _ { 0 } ^ { \tau _ { k } } K \Big ( \sqrt { B } ( \tau _ { k } - v ) \Big ) \psi ( v ) \mathrm { d } v ,
$$

where $K ( u ) = ( A _ { K } + u ) ^ { - \nu }$ . On each interval $I _ { j } = ( \tau _ { j } , \tau _ { j + 1 } ]$ , take $\psi ( v ) = \eta _ { j }$ . The right-endpoint rectangle rule gives

$$
\begin{array} { r } { \displaystyle \int _ { 0 } ^ { \tau _ { k } } K \Big ( \sqrt { B } ( \tau _ { k } - v ) \Big ) \psi ( v ) \ \mathrm { d } v \approx \displaystyle \sum _ { j = 0 } ^ { k - 1 } K \Big ( \sqrt { B } ( \tau _ { k } - \tau _ { j + 1 } ) \Big ) \ \eta _ { j } \Delta \tau _ { j } } \\ { \displaystyle = \displaystyle \sum _ { j = 0 } ^ { k - 1 } K \Big ( \sqrt { B } ( \tau _ { k } - \tau _ { j + 1 } ) \Big ) \ \eta _ { j } ^ { 2 } . } \end{array}
$$

Steps with $\eta _ { j } = 0$ have zero interval length and contribute zero. The resulting predictor is

$$
\begin{array} { l } { { \displaystyle \widehat { L } _ { \Theta } ( k ; \eta , B ) = L _ { \infty } + C _ { S } ( A _ { S } + \sqrt { B } \tau _ { k } ) ^ { - \alpha } } } \\ { { \displaystyle \qquad + C _ { K } \sum _ { j = 0 } ^ { k - 1 } \bigl ( A _ { K } + \sqrt { B } ( \tau _ { k } - \tau _ { j + 1 } ) \bigr ) ^ { - \nu } \eta _ { j } ^ { 2 } . } } \end{array}\tag{80}
$$

The discretization step is $\Delta \tau _ { j } = \eta _ { j }$ . Batch size enters the signal and forgetting arguments through $\sqrt { B }$ . Equation (80) records the right-endpoint convention used for the reported fits and predictions. The main-text expression $( 7 )$ uses left endpoints. Both discretize the continuous FSL (79), but they need not agree at finite step sizes.

Fitting the discrete FSL. For a loss trajectory with learning rate schedule $\eta \quad =$ $( \eta _ { 0 } , \dots , \eta _ { N - 1 } )$ , batch size B, and losses $( \ell _ { 1 } , \ldots , \ell _ { N } )$ , the parameters $\Theta$ are estimated by minimizing the mean squared error (MSE):

$$
\operatorname* { m i n } _ { \Theta } \ \frac { 1 } { | \mathcal { T } | } \sum _ { k \in \mathcal { T } } \bigl ( \widehat { L } _ { \Theta } ( k ; \pmb { \eta } , B ) - \ell _ { k } \bigr ) ^ { 2 } ,\tag{81}
$$

adam\_optimizer = torch.optim.Adam( lbfgs\_optimizer = torch.optim.LBFGS(   
model.parameters(), model.parameters(),   
1r=0.05, 1r=0.1,   
weight\_decay=1e-6, max\_iter=50,   
betas=(0.9,0.95), tolerance\_grad=1e-20,   
eps=1e-8, tolerance\_change=1e-20,   
) history\_size=100,   
line\_search\_fn="strong\_wolfe",   
)  
Figure 7: Optimizer configurations used in the two-stage FSL fitting procedure. (Left) Stage 1: Adam. (Right) Stage 2: L-BFGS.

where $\mathcal { T } = \{ N _ { \mathrm { s k i p } } + 1 , \hdots , N \}$ denotes the set of step indices used for fitting (the first $N _ { \mathrm { s k i p } }$ steps are excluded). We optimize Θ using a two-stage procedure: 2000 steps of Adam followed by 30 steps of L-BFGS. The optimizer configurations are shown in Figure 7

In some experiments, we fit multiple loss trajectories simultaneously. The fitting objective is the average MSE:

$$
\operatorname* { m i n } _ { \Theta } \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \frac { 1 } { | \mathcal { T } ^ { ( m ) } | } \sum _ { k \in \mathcal { I } ^ { ( m ) } } \big ( \widehat { L } _ { \Theta } ( k ; \pmb { \eta } ^ { ( m ) } , B ^ { ( m ) } ) - \ell _ { k } ^ { ( m ) } \big ) ^ { 2 } ,
$$

where $m \in \{ 1 , \cdots , M \}$ denotes the index of the loss trajectory. Each trajectory uses its own cumulative-learning-rate grid $\begin{array} { r } { \tau _ { k } ^ { \left( m \right) } = \sum _ { j = 0 } ^ { k - 1 } \eta _ { j } ^ { \left( m \right) } } \end{array}$ . Its intrinsic time is $\sqrt { B ^ { ( m ) } } \tau _ { k } ^ { ( m ) }$

After fitting the parameters Θ, we use the discrete FSL (80) to predict loss trajectories for held-out learning rate schedules or batch sizes.

## H.2 PLK regression experiments

The loss-curve and data-scaling experiments in Section 5.1 use $( s , q ) = ( 0 . 3 , 4 )$ , which satisfies

$$
\frac { 1 } { 2 } - \frac { 1 } { q } \leqslant s < 1 - \frac { 1 } { q } .
$$

As discussed in Section 4.1, this is the regime where SignSGD attains the statistically optimal data scaling rate, whereas SGD does not.

In all experiments, we set dimension $d = 5 1 2 .$ , eigenvalues $\lambda _ { j } = j ^ { - q }$ , and target magnitudes $| \theta _ { j } ^ { * } | = j ^ { - \frac { 1 } { 2 } ( s q - q + 1 ) }$ for $j = 1 , \ldots , d .$ We fix the noise variance $\sigma ^ { 2 } = 1 . 0$ , and initialize at $\pmb \theta _ { 0 } = \mathbf 0$ Unless otherwise stated, the default peak learning rate and batch size are $\eta _ { \mathrm { p e a k } } = 0 . 1$ and $B = 1$ 2 respectively. We report the population excess risk after k updates,

$$
\ell _ { k } = \mathcal { E } ( \pmb { \theta } _ { k } ) = \frac { 1 } { 2 } \sum _ { j = 1 } ^ { d } \lambda _ { j } ( \theta _ { k , j } - \theta _ { j } ^ { * } ) ^ { 2 } .
$$

Across the synthetic experiments, we use SGD, SignSGD, RMSprop and Adam; the optimizer subset depends on the experiment. For RMSprop, we set $\beta _ { 2 } = 0 . 9$ and $\epsilon = 1 0 ^ { - 8 }$ For Adam, we set $( \beta _ { 1 } , \beta _ { 2 } ) \ : = \ : ( 0 . 9 , 0 . 9 5 )$ and $\epsilon = 1 0 ^ { - 8 }$ . For the loss-curve and data-scaling experiments, we report the excess risk averaged over 200 independent runs unless otherwise stated.

## H.2.1 Risk dynamics across learning rate schedules

Fitting and prediction for different schedules. We provide experimental details for risk-curve fitting and prediction. For each optimizer, we fit the FSL on cosine (Loshchilov & Hutter, 2017) and WSD (Hu et al., 2024) schedules and evaluate its predictive performance on the held-out cyclic schedule (Smith, 2017). See Figure 8 for a visualization of these schedules. For SignSGD in Figure 1 (left), we set $( s , q , d ) = ( 0 . 3 , 4 , 5 1 2 )$ $B = 1$ , and $\sigma ^ { 2 } = 1$ , initialize at zero, and train for 10,000 updates with peak learning rate 0.1. For each schedule, we report the unsmoothed population excess risk averaged over 200 independent runs. We fit six shared parameters in (80) and keep them fixed for the held-out cyclic prediction. We use the fitting procedure of Appendix H.1, consisting of 2000 Adam steps followed by 30 L-BFGS steps, with $L _ { \infty } = 0$ for excess risk and $N _ { \mathrm { s k i p } } = 5 0 0$ unless specified otherwise.

![](images/18e926859e96d2e5f13d28d5d149d1b428a8587a6cdaa52437abf56d366ade32.jpg)  
Figure 8: Learning rate schedules in synthetic experiments.

For Adam in Figure 9 (left), we set $N _ { \mathrm { s k i p } } = 1 0 0 0$ and fit piecewise-linear interpolants of the risk curves at zero-based stored indices 1000 through 9999.

Decomposition and intrinsic-time collapse across schedules. Figure 2 uses the same experimental settings and fitted FSL parameters as Figure 1 (left). Figure 2 (a) shows the signal-learning and noise-forgetting terms for the three LR schedules. Figure 2 (b) plots the signal-learning terms against intrinsic time $t _ { k } = \sqrt { B } \tau _ { k }$ , showing their collapse across schedules.

## H.2.2 Data scaling and critical batch size

Data scaling. We provide details for the SGD/SignSGD data-scaling comparison in Figure 1 (middle) and the adaptive-optimizer comparison in Figure 10 (left). Across these comparisons, we run SGD, SignSGD, RMSprop, and Adam using the power decay schedule (Li et al., 2026c):

$$
\eta _ { k } = \eta _ { \mathrm { p e a k } } \left( 1 - \frac { k } { N } \right) ^ { p } , \qquad k = 0 , 1 , \ldots , N - 1 .
$$

Here N is the number of updates and $D ~ = ~ B N$ . We choose $p \ = \ 4 .$ which satisfies the power-decay conditions in Theorem 4.1 for SignSGD and in Appendix E.3 for SGD at the experimental parameters. We use the same schedule family for RMSprop and Adam to compare their data-scaling behavior.

All runs use batch size $\textit { B } = \textit { 1 }$ , with $\eta _ { \mathrm { p e a k } }$ tuned at each budget. We use $D \in$ {1000, 2000, 4000, 8000, 16000, 32000}, evaluate SignSGD on 64 independent runs after tuning, and compute SGD risk by exact second-moment recursion. The four-optimizer comparison in Figure 10 (left) uses the five budgets from 2000 to 32000, gradient mean $2 \mathbf { H } ( \theta - \theta ^ { * } )$ , the same noise covariance, and 200 evaluation runs per setting.

Critical batch size. Figure 1 (right) compares SGD and SignSGD. Figure 10 (right) additionally reports Adam and RMSprop. For these batch-size experiments, we use the common PLK setup with $( s , q ) = ( 1 , 4 )$ . We use the budgets and schedule

$$
\begin{array} { r l } & { D \in \{ 2 0 0 0 , 4 0 0 0 , 8 0 0 0 , 1 6 0 0 0 , 3 2 0 0 0 , 6 4 0 0 0 , 1 2 8 0 0 0 \} , \qquad N = \lceil D / B \rceil , } \\ & { \eta _ { k } = \eta _ { \mathrm { p e a k } } ( 1 - k / N ) ^ { 4 } , \qquad k = 0 , \dots , N - 1 . } \end{array}
$$

The final batch consumes the remaining samples, so every run uses exactly D samples. For every optimizer and $( D , B )$ , we tune $\eta _ { \mathrm { p e a k } }$ within (0, 1]. RMSprop uses decay 0.9 and Adam uses $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 )$ with bias correction, both with $\epsilon = 1 0 ^ { - 8 }$ and no weight decay. Each risk is normalized by the same optimizer's independently evaluated, optimized $B = 1$ risk at the same D. The CBS is the last tested batch in the passing prefix whose normalized risk is at most 1.2, ending at the first tested failure even if later batches pass again. We refine the batch grid adaptively near this first crossing. SGD risks are computed by an exact second-moment recursion. For the normalized optimizers, learning-rate selection and final risk evaluation use independent random streams, and stratified Monte Carlo includes all 512 coordinates. The dashed curves in the CBS plot in Section 5.1 are ordinary least-squares fits in log-log coordinates over all displayed data budgets.

Batch-size experiment. Figure $1 \ \mathrm { ( r i g h t ) }$ uses the diagonal Gaussian-gradient model with $D = 3 2 , 0 0 0 , \ ( s , q , d ) = ( 1 , 4 , 5 1 2 )$ , unit noise, and zero initialization. At every batch size, we run $N = D / B$ updates with power decay $p = 4$ and retune the peak learning rate up to one. SignSGD risks are averaged over 512 runs independent of tuning, while SGD risks are computed exactly. We take the empirical CBS as the last tested batch before the mean risk exceeds 1.2 times its own $B = 1$ value. This gives 2000 for SignSGD and 16 for SGD. Adding the SignSGD cap $\eta _ { \mathrm { p e a k } } \leqslant B ^ { - 1 / 2 }$ gives 256 versus 16. These finite-grid results illustrate the batch-size advantage. The common-cap experiment does not directly test the theoretical CBS exponent.

## H.2.3 Additional results for Adam and RMSprop

Figure 9 shows that the FSL captures risk dynamics for both Adam and RMSprop, which also exhibit data-scaling and batch-size advantages over SGD (Figure 10). In Figure 9 (right), the held-out cyclic prediction has a root mean squared error of $1 . 9 0 \times 1 0 ^ { - 3 }$ against the mean excess risk over 200 runs. On the same stored entries (zero-based indices 500,..., 9999), the fitting errors are $1 . 6 0 \times 1 0 ^ { - 3 }$ for cosine and $2 . 4 6 \times 1 0 ^ { - 3 }$ for WSD.

![](images/9b72ef2c3d8ac5e050c398d81eac1f406bbe632d60a764bbed3faf091ef0b8c4.jpg)  
Figure 9: FSL predictions for Adam and RMSprop. Risk dynamics for Adam and RMSprop, respectively. Light lines show the mean excess risk over 200 runs. Dark solid lines are FSL fits for cosine and WSD, and dashed lines predict the held-out cyclic schedule

![](images/21079913f88058e81489a5a8034b9ffb2213ece505582db136d12f931c9f1996.jpg)  
Figure 10: Data scaling and critical batch size. (Left) Final excess risk versus data budget for SGD, SignSGD, RMSprop and Adam. (Right) Estimated CBS versus data budget for the same optimizers. Dashed lines in both panels denote power-law fits. The CBS uses a 20% risk-increase tolerance relative to each optimizer's B = 1 baseline.

## H.2.4 Structural-condition experiments

Figure 4 uses quadratic objectives with additive Gaussian gradient noise and the spectral and target-magnitude parameterization above. The target signs are fixed as in the rotation construction below. We set $( s , q , d ) = ( 0 . 3 , 4 , 5 1 2 )$ , initialize at zero, and use $B = 1$ and $D =$ 32,000. All conditions have the same total gradient noise energy. Each optimizer and setting uses a separately tuned peak learning rate in $[ 1 0 ^ { - 5 } , 1 ]$ and the schedule $\eta _ { k } = \eta _ { \mathrm { p e a k } } ( 1 - k / N ) ^ { 4 }$ Each setting and rotation path is evaluated on 64 held-out stochastic runs. The left panel shows mean risk trajectories, with isotropic-noise controls (iso.) and one fixed rotated geometry (rot.).

For the middle panel, the Hessian remains diagonal and the noise covariance varies as

$$
\pmb { \Sigma } _ { \gamma } = \frac { \mathrm { t r } ( \mathbf { H } ) } { \mathrm { t r } ( \mathbf { H } \gamma ) } \mathbf { H } ^ { \gamma } , \qquad \gamma \in \{ 0 , 0 . 1 , \dots , 1 \} .
$$

The endpoints give isotropic and curvature-aligned noise, respectively. The right panel uses the joint rotations described in the rotation construction below, preserving the Hessian spectrum and target energy in each eigenmode. Each positive rotation strength averages three fixed rotation paths, with one shared unrotated control. Both panels report the ratio of the mean final SGD excess risk to the mean final SignSGD excess risk, so values above one favor SignSGD.

Rotation construction. We use the spectral and target parameterization above, with $( s , q , d ) = ( 0 . 3 , 4 , 5 1 2 ) , \lambda _ { j } = j ^ { - q } .$ and $| \theta _ { j } ^ { * } | \stackrel { - } { = } j ^ { - \frac 1 2 ( s q - q + 1 ) }$ . The target has independent random signs sampled once and then fixed. Following the matrix-exponential rotation construction of Xie et al. (2025), we draw a matrix M with independent standard Gaussian entries and set

$$
\begin{array} { r l } & { \mathbf { A } = \mathbf { M } - \mathbf { M } ^ { \top } , \qquad \mathbf { R } _ { \rho } = \exp ( \rho \mathbf { A } ) , } \\ & { \mathbf { H } _ { 0 } = \mathrm { d i a g } ( \lambda _ { 1 } , \dots , \lambda _ { d } ) , \qquad \mathbf { H } _ { \rho } = \mathbf { R } _ { \rho } ^ { \top } \mathbf { H } _ { 0 } \mathbf { R } _ { \rho } , \qquad \theta _ { \rho } ^ { * } = \mathbf { R } _ { \rho } ^ { \top } \pmb { \theta } ^ { * } . } \end{array}
$$

No normalization is applied to A. We use three independent generator matrices with seeds 21161024, 21161025, and 21161026, each held fixed across all ρ. A pilot experiment sets the six positive rotation strengths $\rho \in \{ 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } \}$ . We also evaluate the exactly diagonal control $\rho = 0$ using the same implementation and Gaussian seed protocol.

Training and evaluation. For each fixed $\rho ,$ we initialize $\pmb \theta _ { 0 } = \mathbf 0$ and set $B = \sigma ^ { 2 } = 1$ . The gradient oracle and population excess risk are

$$
\begin{array} { r } { \mathbf { g } _ { k } = \mathbf { H } _ { \rho } ( \pmb { \theta } _ { k } - \pmb { \theta } _ { \rho } ^ { * } ) + \pmb { \xi } _ { k } , \qquad \pmb { \xi } _ { k } = \mathbf { R } _ { \rho } ^ { \top } \mathbf { H } _ { 0 } ^ { 1 / 2 } \mathbf { z } _ { k } , \qquad \mathbf { z } _ { k } \overset { \mathrm { i i d } } { \sim } \mathcal { N } ( \mathbf { 0 } , I ) , } \\ { \ell _ { k } = \xi ( \pmb { \theta } _ { k } ) = \frac { 1 } { 2 } ( \pmb { \theta } _ { k } - \pmb { \theta } _ { \rho } ^ { * } ) ^ { \top } \mathbf { H } _ { \rho } ( \pmb { \theta } _ { k } - \pmb { \theta } _ { \rho } ^ { * } ) . \qquad } \end{array}
$$

Thus $\xi _ { k } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { H } _ { \rho } )$ . This additive Gaussian oracle isolates the noise geometry used in the mechanism analysis, rather than simulating the full state-dependent noise of stochastic regression gradients.

## H.3 Language model experiments

Training configuration. We train a 0.1B-parameter LLaMA-style Transformer and track the validation loss throughout training. Unless otherwise specified, each run uses 10B tokens with sequence length 2048, batch size 128, peak learning rate $5 \times 1 0 ^ { - 4 } .$ and gradient clipping threshold 1.0.

We train with SignSGD and Adam. For Adam, we set $( \beta _ { 1 } , \beta _ { 2 } ) \ = \ ( 0 . 9 , 0 . 9 5 )$ and $\epsilon =$ $1 0 ^ { - 8 }$ . The schedule-comparison runs use decoupled weight decay with $\lambda ~ = ~ 0 . 1$ for both optimizers (Loshchilov & Hutter, 2019), giving the update

$$
\pmb { \theta } _ { k + 1 } = ( 1 - \eta _ { k } \lambda ) \pmb { \theta } _ { k } - \eta _ { k } \mathbf { u } _ { k } ,
$$

where $\mathbf { u } _ { k }$ is the optimizer-specific update at step k.

Learning rate schedules. Figure 11 visualizes the learning rate schedules in our experiments: cosine (Loshchilov & Hutter, 2017), WSD (Hu et al., 2024), and 8-1-1 (DeepSeek-AI et al., 2024). All schedules incorporate a 2000-step linear warmup from 0 to the peak learning rate $\eta _ { \mathrm { p e a k } }$ . Following warmup, both WSD and 8-1-1 maintain a constant learning rate of $\eta _ { \mathrm { p e a k } }$ until step 0.8N (where N denotes the total training steps), after which WSD linearly decays to zero by the end of training, while the 8-1-1 schedule adopts a piecewise constant profile: dropping to $\eta _ { \mathrm { p e a k } } / \sqrt { 1 0 }$ at step 0.8N, and then to $\eta _ { \mathrm { p e a k } } / 1 0$ at step 0.9N.

![](images/79d75e0b0748a4306d6e960cad29f52ec31193894cff0fbabe89f50c9f5d3a9d.jpg)  
Figure 11: Learning rate schedules for language model training. All schedules use a warmup of 2000 steps and the same peak learning rate.

Details of FSL fitting. We use the fitting procedure described in Appendix H.1 with $N _ { \mathrm { s k i p } } =$ 2000.

Loss curve fitting and prediction for different schedules. For Figure 3 (a), we fit the FSL to the cosine and 8-1-1 loss curves and predict the held-out WSD curve. In the fitting objective (81), we use a 100-step moving average of the loss,

$$
\ell _ { k } ^ { \mathrm { s m o o t h } } = \frac { 1 } { 2 H } \sum _ { j = k - H } ^ { k + H - 1 } \widetilde { \ell } _ { j } , \qquad H = 5 0 ,
$$

where $\widetilde { \ell } _ { j } = \ell _ { \operatorname* { m i n } \{ N , \operatorname* { m a x } \{ 1 , j \} \} }$ extends the first and last observed losses beyond the endpoints.

Joint fitting across batch sizes under a cosine schedule. We next consider a nonconstant learning rate. Figure 12 (left) reports joint fitting of Adam cosine-schedule runs across all three batch sizes $B \in \{ 6 4 , 1 2 8 , 2 5 6 \}$ , with peak learning rate $5 \times 1 0 ^ { - 4 }$ . The FSL closely tracks all three curves, indicating that the same square-root batch-size clock remains effective in this setting. Since all three curves are included in this fit, this experiment tests joint batch-size consistency rather than held-out batch-size extrapolation.

Extrapolation of batch sizes and data size. In Figure 3 (b), we demonstrate that FSL can extrapolate to held-out batch sizes. We now test the simultaneous extrapolation of both batch size and data size. The batch-size comparisons at peak learning rate $5 \times 1 0 ^ { - 4 }$ use the first 80% of the 10B-token runs (about 8B tokens) for $B \in \{ 6 4 , 1 2 8 , 2 5 6 \}$ . SignSGD uses a constant rate throughout. The Adam curves cover the warmup and plateau of the WSD schedule up to the start of decay. We fit a single FSL model using only

• the initial 50% of the $B = 6 4 \mathrm { \ c u r v e } .$

• the initial 50% of the B = 128 curve.

We then evaluate the predictive performance on the remaining parts of the two curves and on the entire held-out $B = 2 5 6$ curve. As shown in Figure 12 (right), FSL predicts both the continuation of the fitted batch-size curves and the fully held-out $B = 2 5 6$ curve, with small deviations over the longest continuation horizons.

![](images/5f089645994aa66229aa54a37cce2023998f1c344eedcf92023d080f2b111f86.jpg)

![](images/17902dec6c41df6477626326c4d5491d33c93c0ee17338f641a1222b89690976.jpg)  
Figure 12: Joint batch-size fitting and extrapolation for Adam. (Left) Under a cosine schedule with peak learning rate $5 \times 1 0 ^ { - 4 }$ , the parameters Θ are jointly fitted to all three loss curves for $B \in \{ 6 4 , 1 2 8 , 2 5 6 \}$ . (Right) Under a constant learning rate $\eta = 5 \times 1 0 ^ { - 4 }$ , FSL is fitted using the first half of the $B = 6 4$ and $B = 1 2 8$ trajectories, then predicts their remaining parts and the entire held-out $B = 2 5 6$ trajectory. Solid portions denote FSL fits, and dashed portions denote predictions on held-out horizons.

Robustness to a larger learning rate. We extend the Adam experiments in Figure 3 (b) and Figure 12 (right) by increasing the learning rate from $5 \times 1 0 ^ { - 4 }$ to 10−3. The model, data, batch sizes, and training horizon are kept unchanged. As shown in Figure 13, FSL continues to capture the loss dynamics under this larger learning rate.

![](images/b5d30a5ac4d3bb80f71e3e1a3449c69e3d7a9864968124cb4cc175aed63f96ed.jpg)

![](images/f847a811e2327307ec4b595190b752484eb6403aa905e40b61792b7617a822d6.jpg)  
Figure 13: Fitting and extrapolation under a larger learning rate. We repeat the Adam experiments in Figure 3 (b) and Figure 12 (right) with the learning rate increased from $5 \times 1 0 ^ { - 4 }$ to $1 0 ^ { - 3 }$ . Solid portions denote FSL fitting results, and dashed portions represent FSL predictions on held-out horizons. (Left) Joint fitting across batch sizes. (Right) Fitting and extrapolation to held-out batch sizes and training horizons. FSL remains accurate for joint fitting and the held-out batch size, while the longer-horizon continuation predictions show modest deviations.