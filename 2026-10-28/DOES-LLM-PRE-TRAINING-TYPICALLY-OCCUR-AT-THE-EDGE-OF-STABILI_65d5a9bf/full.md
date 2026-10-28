# DOES LLM PRE-TRAINING TYPICALLY OCCUR AT THE EDGE OF STABILITY?

Yuhang Cai<sup>\*†</sup> willcai@berkeley.edu

Haofeng Huang<sup>\*‡</sup> huanghf22@mails.tsinghua.edu.cn

Yiyuan Ma<sup>§</sup> mayiyuan.unicorn@bytedance.com

Haodong Wen<sup>\*‡</sup> whd25@mails.tsinghua.edu.cn

Deyi Liu<sup>§</sup> deyiliu@bytedance.com

Kaifeng Lyu<sup>‡</sup> klyu@mail.tsinghua.edu.cn

## ABSTRACT

Quadratic approximations provide a common lens for studying neural network optimization, but recent evidence challenges their predictive validity. In full-batch gradient descent with learning rate (LR) η, Cohen et al. (2021) observed the Edge of Stability (EoS), where the largest Hessian eigenvalue concentrates near $2 / \eta .$ , seemingly at odds with classical stability conditions predicted by quadratic approximations. In this work, we revisit the fidelity of quadratic approximations as a model of neural network training dynamics, with a particular focus on their failure modes in LLM training. We first highlight a distinct failure mode of the quadratic approximation, where persistent negative curvature causes the quadratic dynamics to diverge regardless of the LR, even though the real dynamics remain stable. We term this phenomenon the Edge ofConvexity (EoC). After removing the effect of this LR-independent instability, we then propose a generalized definition of Edge ofStability (EoS) for stochastic training with adaptive optimizers, based on whether increasing the LR induces significant divergence in the quadratic dynamics while leaving the real dynamics largely unchanged. Across different LLM pretraining runs with various model sizes up to 787M, we find that (1) EoC is always observed across LLM pretraining, and (2) EoS is also prevalent but not universal: it disappears when the LR becomes sufficiently small (e.g., after decay) or when the batch size falls below a critical threshold that has a positive correlation with the critical batch size. Together, these findings characterize when and how quadratic approximations fail and provide a foundation for future work on understanding training dynamics.

## 1 INTRODUCTION

In optimization theory, the dynamics of gradient-based methods are commonly analyzed through a local quadratic approximation of the loss function, given by the second-order Taylor expansion around the current iterate:

$$
\mathscr { L } ( \boldsymbol { w } _ { t + 1 } ) \approx \mathscr { L } ( \boldsymbol { w } _ { t } ) + ( \boldsymbol { w } _ { t + 1 } - \boldsymbol { w } _ { t } ) ^ { \top } \nabla \mathscr { L } ( \boldsymbol { w } _ { t } ) + \frac { 1 } { 2 } ( \boldsymbol { w } _ { t + 1 } - \boldsymbol { w } _ { t } ) ^ { \top } \nabla ^ { 2 } \mathscr { L } ( \boldsymbol { w } _ { t } ) ( \boldsymbol { w } _ { t + 1 } - \boldsymbol { w } _ { t } ) .\tag{1}
$$

This approximation yields the well-known descent lemma for gradient descent, which guarantees a decrease in loss when the LR η satisfies $\eta < 2 / \lambda _ { \mathrm { m a x } }$ . Here, $\lambda _ { \mathrm { m a x } }$ is the maximum eigenvalue of the Hessian, which serves as a measure of the local curvature of the loss function. A standard analysis pipeline in optimization theory then proceeds by bounding quantities that appear in the lemma and deriving guarantees on the loss convergence.

However, in deep learning, the quadratic approximation often fails to accurately capture the true training dynamics. Notably, Cohen et al. (2021) show that full-batch gradient descent (GD) typically operates at the Edge ofStability (EoS) when training neural networks. For a fixed LR $\eta ,$ the maximum Hessian eigenvalue $\lambda _ { \mathrm { m a x } }$ tends to increase during training and eventually oscillates around $2 / \eta .$ thus $\eta \approx 2 / \lambda _ { \mathrm { m a x } }$ . Under the quadratic approximation, this places the dynamics exactly at the stability boundary of gradient descent, since the quadratic model predicts loss divergence once $\eta > 2 / \lambda _ { \mathrm { m a x } }$ Consequently, the descent lemma can no longer guarantee a persistent decrease of the loss. Yet in practice, the loss continues to decrease over long time scales. Similar EoS behavior has also been observed for full-batch adaptive optimizers such as Adam and RMSProp (Cohen et al., 2022), when curvature is measured using an appropriately preconditioned Hessian.

![](images/e99278b040914f00a04339ade60bc683c6cb02e0ec8e3d16f71d0b46f9858550.jpg)  
(a) Illustration of Edge of Convexity (EoC).

![](images/d5f4aba486a232229f7f0766e52f0ccc2635a2b2a033b1fc7469081784794f31.jpg)  
(b) Illustration of Edge of Stability (EoS).  
Figure 1: LLM pretraining on FineWeb-10B exhibits two distinct failure modes of the quadratic approximation. (a) EoC phenomenon: regardless of the LR choice, the training dynamics under quadratic approximation diverge due to persistent negative curvature, while the true dynamics remain stable. (b) EoS phenomenon: after compensating for negative curvature, the quadratic loss remains as stable as the true loss for 0.8× and 1.0× the training LR, but increasing the LR slightly to 1.2× causes the quadratic loss to diverge. This shows that the current training LR is close to a critical stability threshold. See Sections 5.1 and 5.2 for details.

The EoS phenomenon therefore exposes a fundamental limitation of existing optimization analyses for deep learning, as many such analyses critically rely on the descent lemma and the underlying quadratic approximation. This discrepancy has motivated a line of work that seeks to develop tools to understand training dynamics without critically relying on the quadratic approximation (Arora et al., 2022; Damian et al., 2022; Cohen et al., 2024).

Despite these advances, existing empirical observations and theoretical analyses remain limited in several important aspects: (i) Most theoretical definitions of EoS focus on full-batch training, whereas stochastic training is more common in practice; (ii) Most studies focus on failures of the quadratic approximation caused by large positive Hessian eigenvalues, even though negative Hessian eigenvalues can also induce a similar gap between theory and practice, as we will elaborate in this work; (iii) Due to the prohibitive computational cost of computing and analyzing the Hessian, direct measurements of EoS are largely limited to relatively small-scale settings such as CIFAR-10 and ImageNet, leaving open whether the same failure mode persists in large-scale training.

Our Contributions. We present a systematic study of the failure modes of the quadratic approximation in stochastic, adaptive, and large-scale training, with a particular focus on LLM pretraining.

• We first highlight an LR-independent failure mode of the quadratic approximation, where persistent negative curvature during training causes the quadratic dynamics to diverge regardless of the LR, even though the real dynamics remain stable. We term this phenomenon the Edge of Convexity (EoC). See Section 5.1.

• We isolate and remove the effect of EoC, and then study the Edge ofStability (EoS) as an LRdependent failure mode. We propose a generalized definition of EoS for stochastic training with adaptive optimizers, based on whether increasing the LR induces significant divergence in the quadratic dynamics while leaving the real dynamics largely unchanged. For full-batch GD, our definition recovers the original EoS definition. See Sections 5.2 and 5.3.

• We design efficient methods to detect both EoC and EoS at scale with theoretical guarantees (Appendices A.2 and A.3). Our results on LLM pretraining suggest that (1) EoC consistently arises; (2) EoS is prevalent but not universal, and disappears when the LR becomes sufficiently small (e.g., after decay) or when the batch size falls below a critical threshold that scales linearly with the critical batch size (Sections 6.2 to 6.4).

![](images/d884423cf2697ce9a982a61a8e2ec2d5d7aa9ef8336116efaafe7fce17444462.jpg)  
(a) GPT-small (104M).

![](images/999adb7ef18e506d079e6c63d0e4927a23a9c7736e5890db6fefdb9880754541.jpg)  
(b) GPT-medium (344M).

![](images/d73d69f51a10fdc9963541dfa54eef8be723986dc360cbcd923a26b54457b0ee.jpg)  
(c) GPT-large (787M).  
Figure 2: EoS and EoC appear across model sizes. At most checkpoints, the bottom heatmaps show a sharp transition from bounded displacement to explosion near LR factor 1.0, indicating that the training LR is at the critical threshold of EoS. In the middle $\mathrm { r o w } , - \lambda _ { \mathrm { m i n } }$ has a smaller magnitude than $\lambda _ { \operatorname* { m a x } } .$ , corresponding to saddleness below the 0.1 reference line in the top plots and consistent with EoC. Each heatmap column is a checkpoint, and the vertical axis gives the simulated LR divided by the training LR. See Section 6.2 for the results and Appendix B.1 for training and detection settings.

## 2 RELATED WORK

Edge of Stability. Early studies connected sharpness and stability to optimizer behavior in neural network training: Wu et al. (2018) analyzed SGD minima selection through dynamical stability using sharpness and non-uniformity, and Xing et al. (2018) observed SGD “bouncing” between valley walls during training. Jastrz˛ebski et al. (2018) then showed that SGD visits increasingly sharp regions up to a maximum set by LR and batch size, and that the step is often too large along the sharpest directions. Cohen et al. (2021) later demonstrated that full-batch gradient descent typically operates at an Edge of Stability (EoS), where the largest Hessian eigenvalue oscillates near $2 / \eta$ while loss decreases non-monotonically. Cohen et al. (2022) extended this phenomenon to adaptive methods via saturation of a preconditioned stability threshold. Finally, Arora et al. (2022); Li et al. (2022); Damian et al. (2022); Lyu et al. (2022) provide theoretical explanations for EoS in different training regimes, but a unified understanding of this phenomenon is still lacking.

Training Stability of Neural Networks Series works studied the training stability of constant LR SGD in quadratic losses via a linear system analysis of the second moment of the parameters (Wu et al., 2018; Ma & Ying, 2021; Granziol et al., 2022; Wu et al., 2022; Mulayoff & Michaeli, 2024). They obtained tight conditions in the form of $\| \mathbb { E } [ ( I - \eta H _ { t } ) ^ { \otimes 2 } ] \| \le 1$ , which are valid instability criteria for a particular Lyapunov function. However, in the training neural networks with (S)GD, the unstable convergence phenomenon was observed (Wu et al., 2018; Xing et al., 2018; Jastrz˛ebski et al., 2018), formally identified and termed Edge of Stability (EoS) (Cohen et al., 2021). Subsequently, several studies sought to explain this unstable convergence (Ma et al., 2022; Ahn et al., 2022; Wu et al., 2023; Arora et al., 2022; Damian et al., 2022; Cohen et al., 2024). Among them, Cohen et al. (2024) gave a general characterization of the oscillatory dynamics of deep learning in the EoS regime via a time-averaged differential equation termed central flow. However, the effectiveness of the quadratic approximation was never carefully studied in the large-scale neural network training, hence it is not clear when and how the quadratic approximation fails or not in the realistic regime.

Hessian Spectrum of Neural Networks. Most studies on the Hessian spectrum of neural networks reported that the Hessian spectra of neural networks consist of a “bulk” together with a few large “outliers” in magnitude (LeCun et al., 2002; Dauphin et al., 2014; Sagun et al., 2016; 2017; Ghorbani et al., 2019; Chaudhari et al., 2019; Yao et al., 2020). Among them, Sagun et al. (2016; 2017) mentioned that there are still negative eigenvalues during training, but their magnitude is much smaller than the large outliers. More recent works reported and explained a near-block diagonal structure in the Hessian spectrum of CNNs and Transformers (Zhang et al., 2024; Dong et al., 2025). However, one thing that has long been overlooked is the characterization of the negative eigenvalues in the population Hessian during large-scale training. In this work, we present detailed empirical findings of the negative eigenvalues across different setups in large-scale neural network training.

Comparison with Andreyev & Beneventano (2024). The recent work (Andreyev & Beneventano, 2024) is among the most relevant to this work, which defines Edge of Stochastic Stability for SGD via the batch sharpness, where Batch Sharpness hovers around $2 / \eta$ . Compared to Andreyev & Beneventano (2024), our work (1) In the full-batch setting, our definition for EoS can be reduced to the original definition of EoS by Cohen et al. (2021; 2022), while theirs cannot; (2) We also identify another failure mode of the quadratic approximation called Edge of Convexity (EoC); (3) We provide scalable EoS and EoC detection methods with theoretical guarantees; (4) We conduct experiments on large-scale LLM pretraining with AdamW, while most of their experiments are limited to ResNets on CIFAR-10 and SVHN with vanilla SGD.

## 3 PRELIMINARY

Notation. Throughout, $\| \cdot \|$ denotes the spectral norm for matrices and the $\ell _ { 2 }$ norm for vectors. We write $\lambda _ { \operatorname* { m a x } } ( A )$ and $\lambda _ { \operatorname* { m i n } } ( A )$ for the largest and smallest eigenvalues of a symmetric matrix A, and $\rho ( A )$ for the spectral radius of a square matrix A. We write vec(·) for vectorization and $\otimes$ for the Kronecker product. We use $\mathbb { S } _ { d }$ for the set of $d \times d$ symmetric matrices and $\mathbb { S } _ { d } ^ { + }$ for the subset of positive semidefinite matrices. For $S \succeq 0$ , we write $\| \pmb { u } \| _ { S } = \sqrt { \pmb { u } ^ { \top } S \pmb { u } }$ . We use [n] to denote $\{ 1 , 2 , \ldots , n \}$ . We write $\nabla f ( \pmb { w } )$ and $\nabla ^ { 2 } f ( \pmb { w } )$ for the gradient and Hessian of $f$ at w.

Problem Setup. We study a neural network $f ( \pmb { x } ; \pmb { w } ) : \mathbb { R } ^ { d + s }  \mathbb { R } ^ { q }$ , where x $\in \mathbb { R } ^ { d }$ is the input and w $\in \ \mathbb { R } ^ { s }$ is the trainable parameter. Given a dataset $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ and a per sample loss function $\ell ,$ define the population loss and batch loss as $\begin{array} { r } { \mathcal { L } ( \pmb { w } ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \bar { \ell } \big ( f ( \pmb { x } _ { i } ; \pmb { w } ) , y _ { i } \big ) } \end{array}$ and $\begin{array} { r } { \mathcal { L } _ { B } ( \pmb { w } ) : = \frac { 1 } { B } \sum _ { i \in B } \ell \big ( f ( \pmb { x } _ { i } ; \pmb { w } ) , y _ { i } \big ) } \end{array}$ , where $B \subseteq [ n ]$ is a batch of size B.

Full-Batch Edge of Stability. We recall the full-batch Edge of Stability (EoS) introduced in Cohen et al. (2021). Consider full-batch gradient descent (GD) with LR η as $\pmb { w } _ { t + 1 } = \pmb { w } _ { t } - \eta \nabla \mathcal { L } ( \pmb { w } _ { t } )$ . Let $\pmb { H } _ { t } : = \nabla ^ { 2 } \mathcal { L } ( \pmb { w } _ { t } )$ ), and write $\lambda _ { \operatorname* { m a x } } ( H _ { t } )$ for the maximal eigenvalue of $\pmb { H } _ { t }$ . Classical local analysis says that, if the loss around w<sub>t</sub> is well described by a positive semidefinite quadratic function, then GD is stable only when $\eta \lambda _ { \operatorname* { m a x } } ( \pmb { H } _ { t } ) \leq 2$ . Equivalently, the threshold $2 / \eta$ is the largest curvature without causing instability.

Cohen et al. (2021) found that neural network training does not usually stay well below this threshold. During full batch training, $\lambda _ { \operatorname* { m a x } } ( H _ { t } )$ often increases until it reaches a value near $2 / \eta$ . It then remains near this value for a long part of training. The loss is not monotone at every step, but it continues to decrease over longer time scales. They call this regime the Edge ofStability (EoS).

To relate this observation to our discussion on the fidelity of quadratic approximation, we fix a checkpoint ${ \pmb w } _ { t _ { 0 } }$ and consider

$$
Q _ { t _ { 0 } } ( \pmb { w } ) = \mathcal { L } ( \pmb { w } _ { t _ { 0 } } ) + ( \pmb { w } - \pmb { w } _ { t _ { 0 } } ) ^ { \top } \nabla \mathcal { L } ( \pmb { w } _ { t _ { 0 } } ) + \frac { 1 } { 2 } ( \pmb { w } - \pmb { w } _ { t _ { 0 } } ) ^ { \top } \pmb { H } _ { t _ { 0 } } ( \pmb { w } - \pmb { w } _ { t _ { 0 } } ) .\tag{2}
$$

Suppose that this quadratic has a stationary point ${ \pmb w } _ { \star }$ such that $\nabla \mathcal { L } ( \pmb { w } _ { t _ { 0 } } ) + \pmb { H } _ { t _ { 0 } } ( \pmb { w } _ { \star } - \pmb { w } _ { t _ { 0 } } ) = 0$ Gradient descent on $Q _ { t _ { 0 } }$ then satisfies

$$
\pmb { w } _ { t + 1 } - \pmb { w } _ { \star } = ( \pmb { I } - \eta \pmb { H } _ { t _ { 0 } } ) ( \pmb { w } _ { t } - \pmb { w } _ { \star } ) .
$$

If $H _ { t _ { 0 } } \succeq 0$ , this system is stable when $\rho ( I - \eta H _ { t _ { 0 } } ) \leq 1$ , or equivalently, $\eta \lambda _ { \operatorname* { m a x } } ( H _ { t _ { 0 } } ) \leq 2$ . Once $\eta \lambda _ { \mathrm { m a x } } ( { H } _ { t _ { 0 } } ) > 2 , w _ { t }$ oscillates with growing magnitude along a leading eigenvector of ${ \cal { H } } _ { t _ { 0 } }$

Based on this perspective, the observation that $\lambda _ { \mathrm { m a x } }$ concentrates near $2 / \eta$ throughout training is striking. It places training dynamics precisely at the boundary of stability predicted by the quadratic model, rather than safely within it. The EoS phenomenon therefore exposes a fundamental mismatch between the predictions of quadratic approximation and the actual behavior of neural network training, which is the starting point for our study of EoS in stochastic training.

## 4 STABILITY OF STOCHASTIC GRADIENT DESCENT ON QUADRATICS

The full-batch EoS reviewed above is insightful but incomplete for modern neural networks since it relies on two simplifying conditions. First, the Hessian is assumed to be positive semidefinite. If it has a negative eigenvalue, then the local quadratic dynamics contain an exploding direction for every positive LR, so the usual positive curvature stability threshold is no longer the only source of instability. Second, the analysis is deterministic and applies to full-batch gradient descent. It does not capture the additional randomness introduced by mini-batch sampling.

We then study these two effects on a stochastic quadratic objective. We consider a stochastic variant of Equation (2) to approximate the training loss. Specifically, at step t, let $\begin{array} { r } { f _ { t } ( \pmb { x } ) = \frac { 1 } { 2 } \pmb { x } ^ { \top } \pmb { A } _ { t } \pmb { x } + \pmb { b } _ { t } ^ { \top } } \end{array}$ x+c<sub>t</sub>, where $\pmb { A } _ { t }$ is symmetric, $\mathbb { E } [ A _ { t } ] = A$ , and $\mathbb { E } [ \hat { b } _ { t } ] = b$ . We assume that $\{ ( A _ { t } , \tilde { \boldsymbol { b } } _ { t } ) \} _ { t \geq 0 }$ are sampled independently from a fixed distribution, have finite second moments, and are independent of the initialization. SGD with LR η gives

$$
\pmb { x } _ { t + 1 } = \pmb { x } _ { t } - \eta \nabla f _ { t } ( \pmb { x } _ { t } ) = ( \pmb { I } - \eta \pmb { A } _ { t } ) \pmb { x } _ { t } - \eta \pmb { b } _ { t } .\tag{3}
$$

Let $M _ { t } : = I - \eta A _ { i }$ . Suppose that the population quadratic has a stationary point $\mathbf { x } _ { \star }$ satisfying $A x _ { \star } + b = 0$ . And we define $\epsilon _ { t } : = x _ { t } - x _ { \star } , r _ { t } : = A _ { t } x _ { \star } + b _ { t }$ . Then $\mathbb { E } [ r _ { t } ] = \dot { 0 }$ , the error $\epsilon _ { t }$ follows

$$
\epsilon _ { t + 1 } = ( I - \eta A _ { t } ) ( x _ { t } - x _ { \star } ) - \eta A _ { t } x _ { \star } - \eta b _ { t } = M _ { t } \epsilon _ { t } - \eta r _ { t } .\tag{4}
$$

We study the mean-square stability of the above stochastic quadratic dynamics.

Definition 4.1. For the stochastic dynamics in Equation (4), we say that the system is mean-square stable if, for every initialization $\epsilon _ { 0 } \doteq \mathbb { R } ^ { d }$ the resulting trajectory satisfies

$$
\operatorname* { s u p } _ { t \geq 0 } \mathbb { E } \left[ \left\| \epsilon _ { t } \right\| ^ { 2 } \right] < \infty .\tag{5}
$$

Why Mean-Square Stability. For a stochastic system, the mean trajectory does not describe the size of a typical perturbation. Positive and negative deviations can cancel in expectation while their squared norm grows. We therefore study $\mathbb { E } [ \| \epsilon _ { t } \| ^ { 2 } ]$ . This is a standard stability measure in many analyses of stochastic linear systems induced by SGD (Wu et al., 2018; Ma & Ying, 2021; Wu et al., 2022; Mulayoff & Michaeli, 2024). Our definition requires this quantity to remain bounded over time, allowing persistent fluctuations without requiring individual trajectories to converge. We discuss the relation to almost-sure convergence in Appendix C.2.

The next proposition gives the moment dynamics and a sufficient condition for mean-square stability. Proposition 4.2. For eachfixed $\eta ,$ there exist a matrix S and a vector $b _ { \mathrm { q u a d } }$ that do not depend on t such that

$$
\mathbb { E } [ \epsilon _ { t + 1 } ] = ( I - \eta A ) \mathbb { E } [ \epsilon _ { t } ] ,\tag{6}
$$

$$
\mathsf { v e c } \big ( \mathbb { E } [ \epsilon _ { t + 1 } \epsilon _ { t + 1 } ^ { \top } ] \big ) = \mathbb { E } [ M _ { t } \otimes M _ { t } ] \mathsf { v e c } \big ( \mathbb { E } [ \epsilon _ { t } \epsilon _ { t } ^ { \top } ] \big ) + S \mathbb { E } [ \epsilon _ { t } ] + b _ { \mathrm { q u a d } } .\tag{7}
$$

For every deterministic initialization, the affine system (4) has bounded second moments if

$$
\rho ( \mathbb { E } [ M _ { t } \otimes M _ { t } ] ) < 1 .
$$

The proof is given in Appendix D.3. The above proposition implies that stochastic quadratics have two different instability modes. The first is caused by the negative curvature of the population quadratic and is independent of the LR threshold. The second is the usual positive-curvature instability, but in the stochastic setting its threshold is controlled by a second-moment operator rather than by $\lambda _ { \mathrm { m a x } } ( A )$ alone. The following arguments make the above ideas clear.

## 4.1 NEGATIVE CURVATURE DIVERGENCE

Suppose that $\lambda _ { \operatorname* { m i n } } ( A ) < 0$ , and let $\mathbf { \delta } \mathbf { u } _ { \mathrm { m i n } }$ be a corresponding unit eigenvector. Then the negative curvature makes the stochastic quadratic dynamics unstable for every positive LR. Define the mean error in this direction by $a _ { t } : = \bar { \mathbf { u } } _ { \operatorname* { m i n } } ^ { \top } \mathbb { E } [ \epsilon _ { t } ]$ . Using the mean dynamics $\dot { \mathbb E } [ \epsilon _ { t + 1 } ] = ( I - \eta A ) \mathbb E [ \epsilon _ { t } ]$ , we have

$$
a _ { t + 1 } = \pmb { u } _ { \mathrm { m i n } } ^ { \top } ( \pmb { I } - \eta \pmb { A } ) \mathbb { E } [ \epsilon _ { t } ] = \left( 1 - \eta \lambda _ { \mathrm { m i n } } ( \pmb { A } ) \right) a _ { t } = \left( 1 + \eta | \lambda _ { \mathrm { m i n } } ( \pmb { A } ) | \right) a _ { t } .
$$

Therefore, we have that $a _ { t } = ( 1 + \eta | \lambda _ { \operatorname* { m i n } } ( A ) | ) ^ { t } a _ { 0 }$ . For any initialization with $a _ { 0 } \neq 0$ , this component grows without bound for every $\eta > 0$ . Moreover,

$$
\begin{array} { r } { \mathbb { E } \left[ \| \boldsymbol { \epsilon } _ { t } \| ^ { 2 } \right] \geq \| \mathbb { E } [ \boldsymbol { \epsilon } _ { t } ] \| ^ { 2 } \geq \left| \boldsymbol { u } _ { \mathrm { m i n } } ^ { \top } \mathbb { E } [ \boldsymbol { \epsilon } _ { t } ] \right| ^ { 2 } = | \boldsymbol { a } _ { t } | ^ { 2 } . } \end{array}
$$

Hence the second moment is also unbounded. This instability does not disappear when the LR is reduced. A smaller LR only slows down the escape along the negative curvature direction, while the quadratic model remains unstable for every positive LR.

## 4.2 POSITIVE CURVATURE DIVERGENCE AND CRITICAL LR

Next, we show that even if we factor out those directions causing negative divergence and focus on the locally convex subspace, the instability can also arise from mini-batch noise and admits a critical LR threshold. The most direct construction is to keep only the directions of positive curvature, discarding the eigenvectors of A with nonpositive eigenvalues. On the remaining subspace the population quadratic is strongly convex.

Let $\Pi ^ { + }$ be the orthogonal projection onto the eigenspace of A with positive eigenvalues. We apply this fixed projection at every step and define $M _ { t } ^ { \mp } : = \dot { \Pi } ^ { + } M _ { t } \Pi ^ { + }$ . The projected error dynamics are

$$
\epsilon _ { t + 1 } ^ { + } = M _ { t } ^ { + } \epsilon _ { t } ^ { + } - \eta \Pi ^ { + } \boldsymbol { r } _ { t } .
$$

These $\mathrm { d y }$ namics keep the positive curvature part of the original stochastic quadratic while removing the negative curvature mode. We then define the critical LR for SGD using the mean-square stability of this projected system.

Definition 4.3. For the stochastic quadratic dynamics in Equation (3), let $\epsilon _ { t } ^ { + }$ follow the projected error dynamics above from $\epsilon _ { 0 } ^ { + } : = \Pi ^ { + } \epsilon _ { 0 }$ . Define the critical LR on the positive curvature subspace by

$$
\eta _ { \star } : = \operatorname* { s u p } \left. \eta > 0 : \operatorname* { s u p } _ { t \geq 0 } \mathbb { E } \left[ \| \epsilon _ { t } ^ { + } \| ^ { 2 } \right] < \infty \mathrm { ~ f o r ~ e v e r y ~ } \epsilon _ { 0 } \in \mathbb { R } ^ { d } \right. .\tag{8}
$$

In the full batch case, $\mathbf { } A _ { t } \equiv A$ and $b _ { t } \equiv b , \mathrm { s o } r _ { t } = A x _ { \star } + b = 0$ by stationarity of $\mathbf { x } _ { \star }$ . Since A is symmetric, we can write the error in the eigenvector basis of A. Projecting onto the positive curvature subspace simply means ignoring all coordinates associated with nonpositive eigenvalues. Along a remaining direction with eigenvalue $\lambda > 0$ , the update multiplies the error by $1 - \eta \lambda ,$ so the error stays bounded when $| 1 - \eta \bar { \lambda } | \leq 1$ and decays when $| 1 - \eta \lambda | < 1$ . Therefore, when $\lambda _ { \operatorname* { m a x } } ( A ) > 0$ all remaining directions have bounded error exactly when

$$
0 < \eta \leq \frac { 2 } { \lambda _ { \operatorname* { m a x } } ( A ) } , \qquad \eta _ { \star } = \frac { 2 } { \lambda _ { \operatorname* { m a x } } ( A ) } .
$$

The definition therefore recovers the full batch threshold in Section 3.

For the stochastic case, the stability depends on more than the eigenvalues of A. For the projected transition,

$$
\mathbb { E } [ M _ { t } ^ { + } \otimes M _ { t } ^ { + } ] = \mathbb { E } [ M _ { t } ^ { + } ] \otimes \mathbb { E } [ M _ { t } ^ { + } ] + \operatorname { C o v } ( M _ { t } ^ { + } ) ,\tag{9}
$$

where

$$
\operatorname { C o v } ( M _ { t } ^ { + } ) : = \mathbb { E } [ M _ { t } ^ { + } \otimes M _ { t } ^ { + } ] - \mathbb { E } [ M _ { t } ^ { + } ] \otimes \mathbb { E } [ M _ { t } ^ { + } ] .
$$

The covariance term records variation in the Hessian across batches. It can make the second moment unstable even when the mean transition is stable, and it can move the stability boundary away from $2 / \lambda _ { \mathrm { m a x } } ( A )$ . This is what is missing from the full-batch EoS picture: stochastic EoS is not determined by $\lambda _ { \mathrm { m a x } }$ alone, and before it can even be defined cleanly, the negative-curvature instability must be separated out. Further stability analysis, including the treatment of zero curvature directions, is given in Appendix C.2.

## 5 STABILITY OF STOCHASTIC GRADIENT DESCENT ON NEURAL NETWORKS

We now apply the stochastic quadratic analysis in Section 4 to neural network training with SGD. Fix a checkpoint ${ \pmb w } _ { t _ { 0 } }$ . For a batch $B _ { t } .$ , we consider the quadratic surrogate

$$
Q _ { \mathcal { B } _ { t } } ( \boldsymbol { w } ) = \mathcal { L } _ { \mathcal { B } _ { t } } ( \boldsymbol { w } _ { t _ { 0 } } ) + ( \boldsymbol { w } - \boldsymbol { w } _ { t _ { 0 } } ) ^ { \top } g _ { \mathcal { B } _ { t } } + \frac { 1 } { 2 } ( \boldsymbol { w } - \boldsymbol { w } _ { t _ { 0 } } ) ^ { \top } H _ { \mathcal { B } _ { t } } ( \boldsymbol { w } - \boldsymbol { w } _ { t _ { 0 } } ) ,\tag{10}
$$

where $g _ { \boldsymbol { B } _ { t } } : = \nabla \mathcal { L } _ { B _ { t } } ( \boldsymbol { w } _ { t _ { 0 } } )$ and $\pmb { H } _ { \mathcal { B } _ { t } } : = \nabla ^ { 2 } \mathcal { L } _ { \mathcal { B } _ { t } } ( \pmb { w } _ { t _ { 0 } } )$ . SGD with LR η on this quadratic is given by

$$
\begin{array} { r } { \pmb { w } _ { t + 1 } = \pmb { w } _ { t } - \eta \nabla Q _ { \mathcal { B } _ { t } } ( \pmb { w } _ { t } ) . } \end{array}\tag{11}
$$

As shown in Section 4, applying SGD to the local quadratic surrogate gives a stochastic linear system. Therefore, the two instability mechanisms identified for stochastic quadratics also arise in the local quadratic surrogate of neural network (NN) training in Equation (11).

First, we recall that $H _ { t _ { 0 } } : = \nabla ^ { 2 } \mathcal { L } ( \pmb { w } _ { t _ { 0 } } ) = \mathbb { E } _ { \pmb { { B } } _ { t } } [ \pmb { H } _ { \pmb { { B } } _ { t } } ]$ is the population Hessian at step $t _ { 0 }$ . If $\lambda _ { \operatorname* { m i n } } ( { H } _ { t _ { 0 } } ) < 0$ , the population quadratic $Q _ { t } ( w ) \stackrel {  } { = } \mathbb { E } [ \stackrel {  } { Q } _ { B _ { t } } ( w ) ]$ is unbounded below as $t \to \infty ,$ , and its mean dynamics have an expanding direction for every positive LR. This is the negative curvature failure studied in Section 4.1.

Second, we remove the directions of nonpositive curvature by projecting every local update. Let $\Pi _ { t _ { 0 } } ^ { + }$ project onto the positive eigenspace of $H _ { t _ { 0 } } ,$ , and let $\epsilon _ { t } ^ { + }$ denote the displacement of this projected run from the checkpoint. The update is

$$
\epsilon _ { t + 1 } ^ { + } = \Pi _ { t _ { 0 } } ^ { + } \big [ ( I - \eta H _ { \mathcal { B } _ { t } } ) \epsilon _ { t } ^ { + } - \eta \nabla \mathcal { L } _ { \mathcal { B } _ { t } } ( w _ { t _ { 0 } } ) \big ] , \qquad \epsilon _ { 0 } ^ { + } \in \mathrm { r a n g e } ( \Pi _ { t _ { 0 } } ^ { + } ) .
$$

As in Definition $4 . 3 , \eta _ { \star } ( w _ { t _ { 0 } } )$ is the supremum of positive LRs for which sup<sub>t</sub> $\mathbb { E } \| \epsilon _ { t } ^ { + } \| ^ { 2 } < \infty$ for every deterministic initial displacement in this subspace.

For NN training, once we replace the actual loss with $Q _ { B _ { t } } ( \pmb { w } )$ , it is typically observed that the linear system formed by quadratic approximation diverges quickly, while the true dynamics of the NN remain stable (Figure 1a, Figure 1b). So a natural question is: which regime leads to the failure of quadratic approximation in NN training, negative or positive curvature divergence? We identify: NN training sits on an edge where either divergence mechanism is just at the threshold of breaking the local quadratic approximation.

## 5.1 EDGE OF CONVEXITY

Negative curvature makes the quadratic model unstable. Negative Hessian eigenvalues in NNs have been widely reported in prior works (Sagun et al., 2016; 2017; Alain et al., 2019). In this paper, we show that they can make the quadratic model diverge while NN training remains stable (Figure 1a). $\mathrm { I f } \lambda _ { \operatorname* { m i n } } ( H _ { t _ { 0 } } ) < \mathrm { \dot { 0 } }$ , SGD on the quadratic model in Equation (10) has an exploding direction for every $\eta > 0$ (Section 4.1). A smaller LR slows this explosion but cannot prevent it. We first measure the size of the negative curvature relative to the positive curvature at the same checkpoint.

Definition 5.1 (Saddleness). Suppose that $\lambda _ { \operatorname* { m a x } } ( H _ { t _ { 0 } } ) > 0$ . We define the saddleness at ${ \pmb w } _ { t _ { 0 } }$ by

$$
\nu ( { w _ { t } } _ { 0 } ) : = \frac { \operatorname* { m a x } \{ 0 , - \lambda _ { \operatorname* { m i n } } ( H _ { t _ { 0 } } ) \} } { \lambda _ { \operatorname* { m a x } } ( H _ { t _ { 0 } } ) } .\tag{12}
$$

This ratio measures the negative curvature relative to the largest positive curvature. For any symmetric matrix H, $\begin{array} { r } { \operatorname* { i n f } _ { H _ { + } \succeq 0 } \| H - H _ { + } \| = \operatorname* { m a x } \{ 0 , - \lambda _ { \operatorname* { m i n } } ( H ) \} } \end{array}$ }. Thus the numerator in Equation (12) is the spectral norm distance from the Hessian to the positive semidefinite cone. Dividing by $\lambda _ { \mathrm { m a x } }$ removes the overall scale of the Hessian. When ν is small, the Hessian is close to positive semidefinite relative to its largest positive curvature. It is still nonconvex whenever $\lambda _ { \operatorname* { m i n } } < 0$

For the saddleness defined above, one observation is that for LLM pre-training, after the warm-up phase, $\nu \leq 0 . 1$ , which leads to the first edge area of LLM pre-training.

Edge of Convexity (EoC) at ${ \pmb w } _ { t _ { 0 } }$ . At a checkpoint $\mathbf { \boldsymbol { w } } _ { t _ { 0 } } .$ , training is at EoC when $\lambda _ { \operatorname* { m i n } } ( { H } _ { t _ { 0 } } ) < 0 ,$ but saddleness $\nu ( \boldsymbol { w } _ { t _ { 0 } } )$ itself is small.

EoC therefore describes a particular failure of the quadratic approximation. The Hessian is close to positive semidefinite in relative scale, but its negative curvature is enough to make the quadratic model unstable for every positive LR.

Measuring population negative curvature during LLM pretraining. Early studies on small image and synthetic datasets reported negative Hessian eigenvalues much smaller in magnitude than the largest positive eigenvalue (Sagun et al., 2016; 2017; Alain et al., 2019). In this paper, we extend these measurements to the population Hessian throughout LLM pretraining (Figure 2). Doing so requires $\lambda _ { \operatorname* { m i n } } ( \pmb { H } _ { t _ { 0 } } )$ The usual approach at scale is to run Lanczos on one mini-batch Hessian (Ghorbani et al., 2019; Yao et al., 2020; Zhang et al., 2024). This gives $\lambda _ { \operatorname* { m i n } } ( \pmb { H } _ { B } )$ . And since $\lambda _ { \mathrm { m i n } } ( \cdot )$ is concave, Jensen’s inequality gives $\mathbb { E } [ \lambda _ { \operatorname* { m i n } } ( H _ { B } ) ] \leq \lambda _ { \operatorname* { m i n } } ( \bar { H _ { t _ { 0 } } } )$ , so the mini-batch value is biased below the population value. This downward bias can be substantial in practice. To this end, we propose a stochastic power iteration for the estimate of $\lambda _ { \operatorname* { m i n } } ( \pmb { H } _ { t _ { 0 } } )$ , which further gives the systematic detection of EoC. The complete EoC detection procedure is given in Appendix $\bar { \mathrm { A } . 2 } .$ , along with the theoretical guarantee in Appendix C.3.

## 5.2 EDGE OF STABILITY UNDER STOCHASTIC TRAINING

Even after removing the effect of EoC (i.e., restricting to the positive-curvature subspace in Definition 4.3), the quadratic approximation can still diverge. This divergence is caused by positive curvature when the LR exceeds the critical threshold $\eta _ { \star } ( \pmb { w } _ { t _ { 0 } } )$ . More subtly, neural network training often operates near this boundary: a tiny positive increase in the LR can already trigger divergence.

```perl
Stochastic Edge of Stability (EoS) at ${ \pmb w } _ { t _ { 0 } }$ . Training is at EoS when the current LR is close to
the stochastic critical LR on the positive-curvature subspace, $\eta \approx \eta _ { \star } ( w _ { t _ { 0 } } )$
```

The mean curvature is positive definite on this subspace, so the LRs with bounded second moments have a single critical endpoint (Appendix C.2). In the full-batch case at a stationary point, this critical LR is $2 / \bar { \lambda _ { \mathrm { m a x } } } ( { H _ { t _ { 0 } } } )$ whenever $\lambda _ { \operatorname* { m a x } } ( H _ { t _ { 0 } } ) > 0$ , as computed after Definition 4.3, which recovers the full-batch EoS condition of Cohen et al. (2021). See Section 2 for comparison with previous work on stochastic EoS.

As discussed in Section 4, for the stochastic case, even after projection, mean-square stability is governed by the second-moment operator $\mathbb { E } [ M _ { t } ^ { + } \otimes M _ { t } ^ { + } ] \stackrel { ^ { \cdot } } { = } \bar { \mathbb { E } } [ M _ { t } ^ { + } ] \otimes \mathbb { E } [ M _ { t } ^ { + } ] + \mathrm { C o v } ( \bar { M _ { t } ^ { + } } )$ The covariance term can shift the critical LR away from $2 / \lambda _ { \mathrm { m a x } } .$ Moreover, directly solving $\rho \big ( \mathbb { E } [ M _ { t } ^ { + } \otimes M _ { t } ^ { + } ] \big ) < 1$ is infeasible at neural network scale, since $M _ { t } ^ { + } \otimes M _ { t } ^ { + }$ is a fourth-order object and its entries are affected by noisy mini-batch Hessian samples.

Thus, the definitions above reveal two practical obstacles. First, EoC and EoS live on different curvature subspaces, but explicitly separating these subspaces is infeasible for large NNs, especially for LLM pretraining. Second, the stochastic EoS threshold is not only determined by $\lambda _ { \mathrm { m a x } }$ alone but also correlated with the stochastic noise in training, so we cannot detect it by simply checking whether $\eta \lambda _ { \mathrm { m a x } } \approx 2$

EoS detection. We use a local simulation to detect EoS in pretraining. First, to compensate for negative curvature, we add $\frac { \lambda _ { \mathsf { P } } } { 2 } \parallel \pmb { w } - \pmb { w } _ { t _ { 0 } } \parallel ^ { 2 }$ to the local quadratic surrogate (10), with $\lambda _ { \mathsf { P } } =$ max $\{ 0 , - \widehat { \lambda } _ { \mathrm { m i n } } \}$ , where $\widehat { \lambda } _ { \mathrm { m i n } }$ is the estimate of the minimum eigenvalue from Appendix $_ { \mathrm { A } . 2 }$ . We then simulate SGD on this loss from the checkpoint for a fixed number of steps over a grid of LRs around the training LR $\eta ,$ using the same batch sequence for every run. The estimate $\widehat { \eta } _ { \star }$ is the smallest LR in the grid for which the displacement norm $\lVert \boldsymbol { w } - \boldsymbol { w } _ { t _ { 0 } } \rVert$ reaches a prescribed cutoff. We declare EoS when $\widehat { \eta } _ { \star }$ is close to η. The complete procedure is given in Appendix A.3 and theoretical analysis is in Appendix C.

Illustration on CIFAR-10. Figure 7 applies both detectors to ResNet-20 and ResNet-32 trained with SGD. Each column is a checkpoint, each row an LR factor, and color encodes the final displacement norm of the local simulation. Across most checkpoints in the middle phase of training, the bounded to-exploding boundary lies at LR factor 1.0 (white dashed line). The training LR is close to $\eta _ { \star }$ , and SGD training operates at the EoS for a large part of training. Section 6 applies the same protocol to LLM pretraining with AdamW, with definitions and detectors extended to adaptive methods below.

## 5.3 ADAPTIVE GRADIENT METHODS AT THE EDGE OF CONVEXITY AND EDGE OF STABILITY

So far, the discussion has focused on SGD. Modern LLMs are usually trained with adaptive methods such as AdamW (Loshchilov & Hutter, 2017). We now consider a general class of adaptive gradient methods (Duchi et al., 2011; Li et al., 2025) as

$$
\begin{array} { r l } & { \pmb { m } _ { t + 1 } = \beta _ { 1 } \pmb { m } _ { t } + ( 1 - \beta _ { 1 } ) \pmb { g } _ { \mathcal { B } _ { t } } ( \pmb { w } _ { t } ) , } \\ & { \pmb { P } _ { t + 1 } = \Psi ( \pmb { P } _ { t } , \pmb { g } _ { \mathcal { B } _ { t } } ( \pmb { w } _ { t } ) , \pmb { w } _ { t } ) , } \\ & { \pmb { w } _ { t + 1 } = \pmb { w } _ { t } - \eta \pmb { P } _ { t + 1 } ^ { - 1 } \pmb { m } _ { t + 1 } - \eta \lambda _ { \mathrm { W D } } \pmb { w } _ { t } , } \end{array}\tag{AGM}
$$

where $0 ~ \le ~ \beta _ { 1 } ~ < ~ 1$ $\mathbf { \nabla } m _ { t }$ is the momentum, $P _ { t } ~ \succ ~ 0$ is the preconditioner matrix, $\Psi ( \cdot )$ is the preconditioner update rule, and $\lambda _ { \mathrm { W D } }$ is the decoupled weight decay coefficient. AdamW, momentum SGD, and SGD are special cases of this update.

At checkpoint ${ \pmb w } _ { t _ { 0 } }$ , we freeze $P : = P _ { t _ { 0 } }$ and use the local quadratic surrogate (10). The effective curvature and its saddleness are $A : = P ^ { - 1 / 2 } H _ { t _ { 0 } } P ^ { - 1 / 2 } + \lambda _ { \mathrm { W D } } I , \nu : = \operatorname* { m a x } \{ 0 , - \bar { \mu } _ { \operatorname* { m i n } } \} / \bar { \mu } _ { \operatorname* { m a x } } ,$ where $\bar { \mu } _ { \mathrm { m i n } } = \lambda _ { \mathrm { m i n } } ( A )$ and $\bar { \mu } _ { \mathrm { m a x } } = \lambda _ { \mathrm { m a x } } ( A ) > 0$ . Negative effective curvature gives an expanding mean direction for every positive LR, even with momentum (Lemma C.5).

![](images/b32258bd42c8469395d661561a8690d9de546ed59696ebdd23f9bb498aa313d6.jpg)  
(a) Cosine schedule.

![](images/bd70dd05686013dec69a2e0092e15e00d5325be0efba60f31e7efe9b46931cf5.jpg)  
(b) Multistep schedule I.

![](images/11bc365bcd0f4a7af97fc87258e8265dedd3d93c4fd8d93648fdf03e66924e70.jpg)  
(c) Multistep schedule II.  
Figure 3: Training may leave the EoS regime after LR decay. In (a) and (b), the transition from bounded displacement to explosion moves to larger LR factors during cosine decay and after the abrupt multistep LR drop, respectively. After the larger LR drop in (c), the local simulations remain bounded at the training LR for the rest of the observed run, where training enters the stable regime and does not come back to EoS. See Section 6.4 and Appendices B.4 and B.5 for further discussion.

For EoS, we project both momentum and displacement onto the strictly positive eigenspace of A in preconditioned coordinates at every local update. As for SGD, the critical LR is the supremum of LRs for which these projected dynamics are mean-square stable. Proposition A.3 shows that this threshold is positive and finite, and gives an equivalent spectral characterization.

Edge of Convexity (EoC) for AGMs. At checkpoint ${ \pmb w } _ { t _ { 0 } }$ , AGM training is at EoC when $\bar { \mu } _ { \operatorname* { m i n } } < 0$ and ν is small.

Edge of Stability (EoS) for AGMs. At checkpoint ${ \pmb w } _ { t _ { 0 } }$ , AGM training is at EoS when its current LR is close to the unique critical LR of the projected local dynamics, $\eta \approx \eta _ { \star }$

The local model and the definitions of AGM saddleness and critical LR $\eta _ { \star }$ are given in Appendix A.1.   
The detailed discussion of detecting EoC and EoS for AGMs can be found in Appendix A.

## 6 EXPERIMENTS

We apply our detection methods to study EoS and EoC during language model pretraining. Our experiments examine how broadly these phenomena occur and how EoS depends on the training configurations like batch size and learning rate schedule.

![](images/e0a159e5580545a6b84f99da784b4c1f604cae4f9f7c4a9742ef6b0370f1d155.jpg)  
(a) EoS batch size threshold vs. critical batch size.

![](images/69c366bf57c1ef8dc5a833300683891e4e64a54dd5effc1f1950dbcff6c88dc8.jpg)  
(b) Critical LR factor $q$ vs. batch size.  
Figure 4: The EoS batch size (EBZ) and the critical batch size (CBZ) show a positive correlation across datasets. (a) Larger EBZ generally corresponds to larger CBZ, and the dashed line is a linear fit. (b) The critical LR factor $q = \widehat { \eta } _ { \star } / \eta$ , averaged over checkpoints, decreases as batch size increases. Stars mark the interpolated crossings of $q = 1 . 1$ , which define EBZ. See Section 6.3 and Appendix B.3 for the experimental protocol and threshold estimation.

![](images/b90d48e38ecbee5d02046dc2f20a63830c3614ed1b5baa4dcecaaa1a0bc865d9.jpg)  
(a) Mean critical LR factor.

![](images/7b289f5f2ff139b055137b4fa9401e872f839b91c54b55e003fa932e3be7e602.jpg)  
(b) Final eval loss.  
Figure 5: Training may go beyond EoS and enter an extremely unstable regime with large LR. (a) As the peak LR increases, the critical LR factor $q = \widehat { \eta } _ { \star } / \eta ,$ averaged over the tested checkpoints, falls from near the reference line at 1.0 to well below it. Values near 1.0 indicate EoS, while smaller values mean that the training LR exceeds the estimated critical LR. (b) These large peak LRs also yield higher final evaluation loss. See Sections 6.2 and 6.4 for further discussion.

## 6.1 EXPERIMENTAL SETUP

We pretrain GPT-style models at four scales ranging from 43M to 787M. Unless otherwise specified, runs use FineWeb (Penedo et al., 2024), AdamW $( \mathrm { W D } = 0 . 1 , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5 )$ , sequence length 1024, and a Warm–Stable–Decay (WSD) LR schedule (Hu et al., 2024) with 4% warmup steps and 20% decay steps. Full details are provided in Appendix B.

## 6.2 EOS AND EOC ARE WIDELY OBSERVED

We evaluate EoS and EoC at over 3,000 checkpoints from more than 100 pretraining runs spanning different model sizes, peak LRs, batch sizes, and shapes of LR schedules. Both phenomena are widely observed across our experiments (Figures 2, 3, 5, 8 and 9). At many checkpoints, the estimated critical LR is close to the training LR, while the estimated minimum eigenvalue is negative and saddleness is below 0.1, consistent with EoS and EoC, respectively. Additional results and discussion are given in Appendices B.2 to B.5. We next examine how these phenomena depend on the training configuration. Specifically, we focus on two key training hyperparameters: batch size and the LR schedule. Note that EoC remains widely observed as these hyperparameters vary, so the following analysis focuses primarily on how they affect EoS.

## 6.3 BATCH SIZE DEPENDENCE OF EOS

For the dependence of batch size, we first show that EoS emerges above a threshold of batch size, which is referred to as the EoS Batch Size (EBZ). We then demonstrate an interesting correlation between EBZ and the critical batch size (CBZ) across different datasets.

EoS emerges above the EBZ threshold. Figure 9 shows that EoS is observed over a larger fraction of training at $B = 5 1 2$ and $B = 1 0 2 4$ than at $B = 2 5 6$ . For the two larger batch sizes, the transition from bounded displacement to explosion stays near LR factor 1.0 over much of the stable phase, indicating that the training LR is close to the estimated critical LR. Below this batch size threshold, the compensated local simulations at checkpoints after the early phase of training remain bounded at every tested LR factor. These runs use GPT-extra-small on FineWeb with WSD schedules and adjust the LR, weight decay, and $\beta _ { 2 }$ according to the batch size scaling rule in Appendix B.3. Further, across all five datasets, Figure 4b shows that the critical LR factor, averaged over the tested checkpoints of each run, decreases as batch size increases. We estimate EBZ as the batch size at which the fitted mean critical LR factor first decreases to 1.1 as batch size increases. This corresponds to the crossing at 1.1 marked in the figure.

EBZ and CBZ show a positive correlation across datasets. In Figure 4a, each point represents one dataset, and the points follow an overall upward linear trend. FineWeb has the largest EBZ and CBZ, while OpenWebMath has the smallest values of both. Here, CBZ is estimated from the same training runs as EBZ and denotes the largest batch size whose final evaluation loss is within 0.3% of the best loss in that dataset’s sweep. Thus, datasets with a larger threshold for EoS also tend to maintain near optimal final evaluation loss at larger batch sizes. Full sweep and estimation details are given in Appendix B.3.

## 6.4 LR DEPENDENCE OF EOS

We next show that LR decay can cause training to leave EoS, and that training may not return to the EoS regime after a sufficiently large LR decay. We also show that training at excessively large peak LRs can enter an extremely unstable regime beyond EoS.

Training can enter the stable regime after LR decay. Before LR decay, training often operates at EoS, with the training LR close to the estimated critical LR (Figures 3, 11 and 12). After the abrupt LR drop in Figure 3b, the transition from bounded displacement to explosion moves to larger LR factors. The estimated critical LR is now higher than the reduced training LR, and the compensated local simulation stays bounded at the training LR over the tested horizon. As training continues, the estimated critical LR can fall back toward the current training LR and EoS can return, as in Figure 11d. With a sufficiently large LR reduction, however, the estimated critical LR stays well above the training LR for the rest of the observed run (Figures 3c, 11c and 12d). The compensated local simulations then remain bounded at the training LR, and EoS does not return. The effects of the drop magnitude and decay duration are discussed in Appendices B.4 and B.5.

Training can enter an extremely unstable regime with large LR. Training with peak LRs substantially above the empirically optimal range can yield higher loss and enter an extremely unstable regime beyond EoS (Figure 5). In this regime, the compensated local quadratic dynamics still allow a critical LR, but it lies far below the training LR. For example, at peak LR 0.04, the estimated critical LR is only about one tenth of the training LR on average, and the compensated local simulations explode at the training LR. See Appendix B.2 for further discussion and Figures 14 to 16 for detailed diagnostics across model sizes and peak LRs.

## 7 CONCLUSION

This paper extends the definition of EoS to the LLM training under a unified framework, together with another clearly clarified phenomenon EoC, accounting for the two failure modes of the quadratic approximation. Several directions remain open for future study. (i) Our experiments mainly focus on SGD and AdamW. It is interesting to see how EoS would behave when training with other optimizers like Muon. (ii) Fully understanding EoS and EoC also calls for solid theoretical analyses of the realistic neural network training dynamics.

## REFERENCES

Kwangjun Ahn, Jingzhao Zhang, and Suvrit Sra. Understanding the unstable convergence of gradient descent. In International conference on machine learning, pp. 247–257. PMLR, 2022.

Guillaume Alain, Nicolas Le Roux, and Pierre-Antoine Manzagol. Negative eigenvalues of the hessian in deep neural networks. arXiv preprint arXiv:1902.02366, 2019.

Arseniy Andreyev and Pierfrancesco Beneventano. Edge of stochastic stability: Revisiting the edge of stability for sgd. arXiv preprint arXiv:2412.20553, 2024.

Sanjeev Arora, Zhiyuan Li, and Abhishek Panigrahi. Understanding gradient descent on the edge of stability in deep learning. In International Conference on Machine Learning, pp. 948–1024. PMLR, 2022.

Kazuoki Azuma. Weighted sums of certain dependent random variables. Tôhoku Mathematical Journal, 19(3):357–367, 1967.

Abraham Berman and Robert J. Plemmons. Nonnegative Matrices in the Mathematical Sciences, volume 9 of Classics in Applied Mathematics. Society for Industrial and Applied Mathematics, Philadelphia, 1994. doi: 10.1137/1.9781611971262. URL https://doi.org/10.1137/1. 9781611971262.

Stéphane Boucheron, Gábor Lugosi, and Pascal Massart. Concentration Inequalities: A Nonasymptotic Theory ofIndependence. Oxford University Press, 2013.

Philippe Bougerol et al. Products of random matrices with applications to Schrödinger operators. Springer Science & Business Media, 2012.

Pratik Chaudhari, Anna Choromanska, Stefano Soatto, Yann LeCun, Carlo Baldassi, Christian Borgs, Jennifer Chayes, Levent Sagun, and Riccardo Zecchina. Entropy-sgd: Biasing gradient descent into wide valleys. Journal ofStatistical Mechanics: Theory and Experiment, 2019(12):124018, 2019.

Jeremy M Cohen, Simran Kaur, Yuanzhi Li, J Zico Kolter, and Ameet Talwalkar. Gradient descent on neural networks typically occurs at the edge of stability. arXiv preprint arXiv:2103.00065, 2021.

Jeremy M Cohen, Behrooz Ghorbani, Shankar Krishnan, Naman Agarwal, Sourabh Medapati, Michal Badura, Daniel Suo, David Cardoze, Zachary Nado, George E Dahl, et al. Adaptive gradient methods at the edge of stability. arXiv preprint arXiv:2207.14484, 2022.

Jeremy M Cohen, Alex Damian, Ameet Talwalkar, J Zico Kolter, and Jason D Lee. Understanding optimization in deep learning with central flows. arXiv preprint arXiv:2410.24206, 2024.

Alex Damian, Eshaan Nichani, and Jason D Lee. Self-stabilization: The implicit bias of gradient descent at the edge of stability. arXiv preprint arXiv:2209.15594, 2022.

Yann N Dauphin, Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Surya Ganguli, and Yoshua Bengio. Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. Advances in neural information processing systems, 27, 2014.

Zhaorui Dong, Yushun Zhang, Jianfeng Yao, and Ruoyu Sun. Towards quantifying the hessian structure of neural networks. arXiv preprint arXiv:2505.02809, 2025.

John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal ofMachine Learning Research, 12:2121–2159, 2011.

Rick Durrett. Probability: Theory and Examples. Cambridge University Press, 5th edition, 2019.

Hillel Furstenberg and Harry Kesten. Products of random matrices. The Annals ofMathematical Statistics, 31(2):457–469, 1960. doi: 10.1214/aoms/1177705909. URL https://doi.org/ 10.1214/aoms/1177705909.

Behrooz Ghorbani, Shankar Krishnan, and Ying Xiao. An investigation into neural net optimization via hessian eigenvalue density. In International Conference on Machine Learning, pp. 2232–2241. PMLR, 2019.

Aaron Gokaslan and Vanya Cohen. Openwebtext corpus. https://skylion007. github.io/OpenWebTextCorpus/, 2019. URL https://skylion007.github. io/OpenWebTextCorpus/.

Diego Granziol, Stefan Zohren, and Stephen Roberts. Learning rates as a function of batch size: A random matrix theory approach to neural network training. Journal of Machine Learning Research, 23(173):1–65, 2022.

Alex Henry, Prudhvi Raj Dachapally, Shubham Shantaram Pawar, and Yuxuan Chen. Query-key normalization for transformers. In Trevor Cohn, Yulan He, and Yang Liu (eds.), Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, Online, November 2020. Association for Computational Linguistics.

Wassily Hoeffding. Probability inequalities for sums of bounded random variables. Journal ofthe American Statistical Association, 58(301):13–30, 1963.

Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, Xinrong Zhang, Zheng Leng Thai, Kaihuo Zhang, Chongyi Wang, Yuan Yao, Chenyang Zhao, Jie Zhou, Jie Cai, Zhongwu Zhai, Ning Ding, Chao Jia, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm: Unveiling the potential of small language models with scalable training strategies, 2024.

Stanisław Jastrz˛ebski, Zachary Kenton, Nicolas Ballas, Asja Fischer, Yoshua Bengio, and Amos Storkey. On the relation between the sharpest directions of dnn loss and the sgd step length. arXiv preprint arXiv:1807.05031, 2018.

Eliahu Ibrahim Jury. Theory and Application of the z-Transform Method. John Wiley & Sons, New York, 1964.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

Yann LeCun, Léon Bottou, Genevieve B Orr, and Klaus-Robert Müller. Efficient backprop. In Neural networks: Tricks of the trade, pp. 9–50. Springer, 2002.

Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Yitzhak Gadre, et al. Datacomp-lm: In search of the next generation of training sets for language models. arXiv preprint arXiv:2406.11794, 2024. URL https://arxiv.org/abs/2406.11794.

Xinghan Li, Haodong Wen, and Kaifeng Lyu. Adam reduces a unique form of sharpness: Theoretical insights near the minimizer manifold. arXiv preprint arXiv:2511.02773, 2025.

Zhouzi Li, Zixuan Wang, and Jian Li. Analyzing sharpness along gd trajectory: Progressive sharpening and edge of stability. arXiv preprint arXiv:2207.12678, 2022.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Kaifeng Lyu, Zhiyuan Li, and Sanjeev Arora. Understanding the generalization benefit of normalization layers: Sharpness reduction. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 34689–34708. Curran Associates, Inc., 2022. doi: 10.52202/ 068431-2514. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/dffd1c523512e557f4e75e8309049213-Paper-Conference.pdf.

Chao Ma and Lexing Ying. On linear stability of sgd and input-smoothness of neural networks. Advances in Neural Information Processing Systems, 34:16805–16817, 2021.

Chao Ma, Daniel Kunin, Lei Wu, and Lexing Ying. Beyond the quadratic approximation: The multiscale structure of neural network loss landscapes. arXiv preprint arXiv:2204.11326, 2022.

Sadhika Malladi, Kaifeng Lyu, Abhishek Panigrahi, and Sanjeev Arora. On the sdes and scaling rules for adaptive gradient algorithms. Advances in Neural Information Processing Systems, 35: 7697–7711, 2022.

Martin Marek, Sanae Lotfi, Aditya Somasundaram, Andrew Gordon Wilson, and Micah Goldblum. Small batch size training for language models: When vanilla sgd works, and why gradient accumulation is wasteful, 2025. URL https://arxiv.org/abs/2507.07101.

Rotem Mulayoff and Tomer Michaeli. Exact mean square linear stability analysis for sgd. In The Thirty Seventh Annual Conference on Learning Theory, pp. 3915–3969. PMLR, 2024.

Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. Openwebmath: An open dataset of high-quality mathematical web text. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2310.06786.

Guilherme Penedo, Hynek Kydlícek, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colinˇ Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale. arXiv preprint arXiv:2406.17557, 2024. URL https://arxiv.org/ abs/2406.17557.

Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Gated attention for large language models: Non-linearity, sparsity, and attention-sink-free. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020. URL https://www.jmlr.org/papers/v21/20-074.html.

Levent Sagun, Leon Bottou, and Yann LeCun. Eigenvalues of the hessian in deep learning: Singularity and beyond. arXiv preprint arXiv:1611.07476, 2016.

Levent Sagun, Utku Evci, V Ugur Guney, Yann Dauphin, and Leon Bottou. Empirical analysis of the hessian of over-parametrized neural networks. arXiv preprint arXiv:1706.04454, 2017.

James S. Vandergraft. Spectral properties of matrices which have invariant cones. SIAM Journal on Applied Mathematics, 16(6):1208–1222, 1968. doi: 10.1137/0116101. URL https://doi. org/10.1137/0116101.

Jean Ville. Étude critique de la notion de collectif. Gauthier-Villars, Paris, 1939.

Jingfeng Wu, Vladimir Braverman, and Jason D Lee. Implicit bias of gradient descent for logistic regression at the edge of stability. Advances in Neural Information Processing Systems, 36: 74229–74256, 2023.

Lei Wu, Chao Ma, and Weinan E. How SGD selects the global minima in over-parameterized learning: A dynamical stability perspective. In Advances in Neural Information Processing Systems, 2018.

Lei Wu, Mingze Wang, and Weijie J. Su. The alignment property of SGD noise and how it helps select flat minima: A stability analysis. In Advances in Neural Information Processing Systems, 2022.

Chen Xing, Devansh Arpit, Christos Tsirigotis, and Yoshua Bengio. A walk with sgd. arXiv preprint arXiv:1802.08770, 2018.

Zhewei Yao, Amir Gholami, Kurt Keutzer, and Michael W Mahoney. Pyhessian: Neural networks through the lens of the hessian. In 2020 IEEE international conference on big data (Big data), pp. 581–590. IEEE, 2020.

Yushun Zhang, Congliang Chen, Tian Ding, Ziniu Li, Ruoyu Sun, and Zhiquan Luo. Why transformers need adam: A hessian perspective. Advances in neural information processing systems, 37: 131786–131823, 2024.

## CONTENTS

1 Introduction 1   
2. Related Work 3   
Preliminary 4   
Stability of Stochastic Gradient Descent on Quadratics 4   
4.1 Negative Curvature Divergence . 5   
4.2 Positive Curvature Divergence and Critical LR 6   
5 Stability of Stochastic Gradient Descent on Neural Networks 6   
5.1 Edge of Convexity 7   
5.2 Edge of Stability under Stochastic Training 8   
5.3 Adaptive Gradient Methods at the Edge of Convexity and Edge of Stability 8   
Experiments 9   
6.1 Experimental Setup 10   
6.2 EoS and EoC Are Widely Observed 10   
6.3 Batch size dependence of EoS 10   
6.4 LR Dependence of EoS 11   
Conclusion 11   
A Detecting EoC and EoS for Adaptive Gradient Methods 17   
A.1 Adaptive Gradient Methods 17   
A.2 EoC Detection 19   
A.3 EoS Detection 22   
A.4 Cost of a Hessian-Vector Product . 23   
B Experimental Details 24   
B.1 Models, Training, and Detection Settings 24   
B.2 Additional Results . 25   
B.3 Batch Size Threshold Experiments 26   
B.4 Multistep LR Drop Ablations 26   
B.5 Linear LR Decay Ablations 26   
B.6 Experiments Across Model Scales 26   
C Theoretical Analysis 35   
C.1 Setting and the Local Linear System 35   
C.2 Stability of the Local Dynamics 37   
C.3 Stochastic Power Iteration for EoC 39   
C.4 Compensated Local Simulation for EoS 42   
D Proofs for the Local Linear System and the Stability Results 44   
D.1 Proof of Proposition C.2 45   
D.2 Proof of Proposition C.3 45   
D.3 Proof of Corollary C.4 47   
D.4 Proofs of Lemma C.5 and Corollary C.6 47   
D.5 Proof of Corollary C.8 48   
D.6 Proof of the AGM Critical LR Characterization 50   
E Proof of the Power-Iteration Bound 51   
E.1 The Bisection Bound 51   
E.2 Proof of the Explosion Bound 52   
E.3 Proof of the No-Explosion Bound 54   
E.4 Auxiliary lemmas 56   
F Proof of the Compensated Finite-Horizon Threshold Bound 57   
F.1 Stability Criterion 57   
F.2 Proof of the Threshold Bound 60   
F.3 Compensation Error 61   
F.4 Finite-Horizon Error 71   
F.4.1 Bounding the Moment Radius 71   
F.4.2 Bounding the Inverse LR 74   
F.4.3 Choosing the Cutoff 76   
F.5 Proofs of the Stability Criterion . 78   
F.5.1 Proof of the Exact Criterion 78   
F.5.2 Proof of the Quantitative Bound 83   
F.5.3 Proof of the AGM Critical LR 87   
F.6 Auxiliary Results 88

## A DETECTING EOC AND EOS FOR ADAPTIVE GRADIENT METHODS

Here we provide detailed detection methods of EoC and EoS for AGMs. We first define the local AGM model, its saddleness, and its critical LR in Appendix A.1. Appendices A.2 and A.3 then describe the EoC and EoS detection procedures and their guarantees. Both procedures use Hessianvector products without forming the Hessian. Appendix A.4 gives their computational cost. The complete theoretical statements and proofs are in Appendices C to F.

## A.1 ADAPTIVE GRADIENT METHODS

We consider adaptive gradient methods (AGMs) with a momentum vector $\mathbf { \nabla } m _ { t }$ and a positive definite preconditioner $P _ { t }$ . For the mini-batch loss $\mathcal { L } _ { B _ { t } }$ , the updates are

$$
\begin{array} { r l } & { \pmb { m } _ { t + 1 } = \beta _ { 1 } \pmb { m } _ { t } + ( 1 - \beta _ { 1 } ) \nabla \mathcal { L } _ { \mathcal { B } _ { t } } ( \pmb { w } _ { t } ) , } \\ & { \pmb { P } _ { t + 1 } = \Psi \big ( P _ { t } , \nabla \mathcal { L } _ { \mathcal { B } _ { t } } ( \pmb { w } _ { t } ) , \pmb { w } _ { t } \big ) , } \\ & { \pmb { w } _ { t + 1 } = \pmb { w } _ { t } - \eta \pmb { P } _ { t + 1 } ^ { - 1 } \pmb { m } _ { t + 1 } - \eta \lambda _ { \mathrm { W D } } \pmb { w } _ { t } . } \end{array}
$$

Here $\eta > 0$ is the LR, $0 \leq \beta _ { 1 } < 1$ is the momentum coefficient, and $\lambda _ { \mathrm { W D } } \geq 0$ is the decoupled weight decay coefficient. The preconditioner update Ψ preserves positive definiteness.

The local quadratic model. Fix a checkpoint ${ \pmb w } _ { t _ { 0 } }$ . For each mini-batch $B _ { t }$ , we approximate its loss $\mathcal { L } _ { B _ { t } }$ by the second-order Taylor expansion around this checkpoint. Write $\epsilon = w - w _ { t _ { 0 } }$ for the displacement, and define the gradient and Hessian at the checkpoint by

$$
\pmb { g } _ { \pmb { \mathscr { B } } _ { t } } : = \nabla \mathcal { L } _ { \mathcal { B } _ { t } } ( \pmb { w } _ { t _ { 0 } } ) , \qquad \pmb { H } _ { \mathcal { B } _ { t } } : = \nabla ^ { 2 } \mathcal { L } _ { \mathcal { B } _ { t } } ( \pmb { w } _ { t _ { 0 } } ) .
$$

The resulting quadratic surrogate is

$$
\begin{array} { r } { Q _ { \mathcal { B } _ { t } } ( \pmb { w } _ { t _ { 0 } } + \epsilon ) : = \mathcal { L } _ { \mathcal { B } _ { t } } ( \pmb { w } _ { t _ { 0 } } ) + \pmb { g } _ { \mathcal { B } _ { t } } ^ { \top } \epsilon + \frac { 1 } { 2 } \epsilon ^ { \top } \pmb { H } _ { \mathcal { B } _ { t } } \epsilon . } \end{array}\tag{13}
$$

This surrogate matches the loss, gradient, and Hessian at the checkpoint and omits terms of higher order in the displacement. At each local step, we sample a fresh batch and use its surrogate in place of the neural network loss. The coefficients $\mathbf { \pmb { g } } _ { B _ { t } }$ and $H _ { B _ { t } }$ are always evaluated at ${ \pmb w } _ { t _ { 0 } }$

Local AGM dynamics. We freeze the preconditioner at $P : = P _ { t { 0 } } \succ 0$ and let the momentum evolve under the surrogate gradients. Let $\epsilon _ { t }$ be the displacement after t local steps. The surrogate gradient at this displacement is $g _ { B _ { t } } + H _ { B _ { t } } \epsilon _ { t }$ . Keeping $\eta , \beta _ { 1 }$ and $\lambda _ { \mathrm { W D } }$ fixed, the local updates become

$$
\begin{array} { r } { \begin{array} { r l } & { \pmb { m } _ { t + 1 } = \beta _ { 1 } \pmb { m } _ { t } + ( 1 - \beta _ { 1 } ) ( \pmb { g } _ { \mathcal { B } _ { t } } + \pmb { H } _ { \mathcal { B } _ { t } } \pmb { \epsilon } _ { t } ) , } \\ & { \pmb { \epsilon } _ { t + 1 } = \pmb { \epsilon } _ { t } - \eta \pmb { P } ^ { - 1 } \pmb { m } _ { t + 1 } - \eta \lambda _ { \mathrm { W D } } ( \pmb { w } _ { t _ { 0 } } + \pmb { \epsilon } _ { t } ) . } \end{array} } \end{array}\tag{14}
$$

The simulation starts at ${ \bf { \epsilon } } \epsilon _ { 0 } = { \bf { 0 } }$ with the checkpoint momentum. Weight decay acts on the full parameter $\boldsymbol { w } _ { t _ { 0 } } + \boldsymbol { \epsilon } _ { t }$ . The choice $\beta _ { 1 } = 0 , P = I$ , and $\lambda _ { \mathrm { W D } } = 0$ recovers SGD.

In preconditioned coordinates, we stack the momentum and displacement as

$$
z _ { t } : = ( \eta P ^ { - 1 / 2 } m _ { t } , P ^ { 1 / 2 } \epsilon _ { t } ) ^ { \top } .
$$

Then (14) has the affine form

$$
z _ { t + 1 } = M _ { t } ( \eta ) z _ { t } + \mathbf { q } _ { t } .\tag{15}
$$

Write $\pmb { D } _ { \pmb { \mathrm { B } } _ { t } } = \pmb { P } ^ { - 1 / 2 } \pmb { H } _ { \pmb { B } _ { t } } \pmb { P } ^ { - 1 / 2 }$ for the preconditioned mini-batch Hessian. The transition matrix and forcing are

$$
\begin{array} { r } { M _ { t } ( \eta ) = \left[ \begin{array} { c c } { \beta _ { 1 } I } & { ( 1 - \beta _ { 1 } ) \eta D _ { B _ { t } } } \\ { - \beta _ { 1 } I } & { ( 1 - \eta \lambda _ { \mathrm { W D } } ) I - ( 1 - \beta _ { 1 } ) \eta D _ { B _ { t } } } \end{array} \right] , } \\ { q _ { t } = \left[ \begin{array} { c c } { ( 1 - \beta _ { 1 } ) \eta P ^ { - 1 / 2 } g _ { \mathcal { B } _ { t } } } \\ { - ( 1 - \beta _ { 1 } ) \eta P ^ { - 1 / 2 } g _ { \mathcal { B } _ { t } } - \eta \lambda _ { \mathrm { W D } } P ^ { 1 / 2 } w _ { t _ { 0 } } } \end{array} \right] . } \end{array}
$$

Two trajectories driven by the same batches have the same forcing $\pmb q _ { t }$ . Their difference therefore evolves by multiplication with $M _ { t } ( \eta )$ . We use this homogeneous evolution to define stability of local perturbations.

Effective curvature and saddleness. Let $\pmb { H } _ { t _ { 0 } } : = \nabla ^ { 2 } \mathcal { L } ( \pmb { w } _ { t _ { 0 } } )$ be the population Hessian. The preconditioned Hessians and effective curvatures are

$$
\begin{array} { r l r } & { } & { D _ { t _ { 0 } } : = P ^ { - 1 / 2 } H _ { t _ { 0 } } P ^ { - 1 / 2 } , \quad \quad A : = D _ { t _ { 0 } } + \lambda _ { \mathrm { W D } } { \cal I } , } \\ & { } & { D _ { \mathcal { B } _ { t } } : = P ^ { - 1 / 2 } H _ { \mathcal { B } _ { t } } P ^ { - 1 / 2 } , \quad A _ { \mathcal { B } _ { t } } : = D _ { \mathcal { B } _ { t } } + \lambda _ { \mathrm { W D } } { \cal I } . } \end{array}\tag{16}
$$

If $\lambda _ { \operatorname* { m i n } } ( A ) < 0$ , some local perturbations grow in mean at every positive LR, even with momentum (Lemma C.5). We measure the size of this negative effective curvature relative to the largest positive effective curvature.

Definition A.1 (Saddleness for AGMs). Suppose $\lambda _ { \operatorname* { m a x } } ( A ) > 0$ . The AGM saddleness at the checkpoint is

$$
\nu : = \frac { \operatorname* { m a x } \{ 0 , - \lambda _ { \operatorname* { m i n } } ( A ) \} } { \lambda _ { \operatorname* { m a x } } ( A ) } .\tag{17}
$$

The preconditioner and weight decay are included through A. AGM training is at EoC when $\lambda _ { \operatorname* { m i n } } ( A ) < 0$ and this ratio is small. For SGD without weight decay, $\boldsymbol { A } = \boldsymbol { H } _ { t _ { 0 } }$ , so this definition reduces to Definition 5.1.

Critical LR. To isolate instability caused by the LR, we remove the directions of nonpositive effective curvature. Let $\Pi ^ { + }$ project onto the strictly positive eigenspace of A, assumed to be nonempty. We project both momentum and displacement in preconditioned coordinates at every update. The effective curvature after the projection is therefore positive definite on this subspace. Write $M _ { t } ^ { + } ( \eta )$ and $\pmb q _ { t } ^ { + }$ for the transition and forcing in coordinates on this subspace. The projected local system is

$$
z _ { t + 1 } ^ { + } = M _ { t } ^ { + } ( \eta ) z _ { t } ^ { + } + { \mathbf { } } q _ { t } ^ { + } .\tag{18}
$$

Following Definition 4.1, we call this system mean-square stable if

$$
\operatorname* { s u p } _ { t \geq 0 } \mathbb { E } \| z _ { t } ^ { + } \| ^ { 2 } < \infty \qquad \mathrm { f o r e v e r y d e t e r m i n i s t i c i n i t i a l s t a t e } z _ { 0 } ^ { + } .
$$

The initial state includes both momentum and displacement. This definition allows persistent fluctuations and does not require the second moment to tend to zero.

Definition A.2 (Critical LR for AGMs). Keep the checkpoint, $P , \beta _ { 1 } , \lambda _ { \mathrm { W D } }$ , and the batch distribution fixed as the LR varies. The AGM critical LR is

$$
\eta _ { \star } : = \operatorname* { s u p } \left\{ \eta > 0 : { \mathrm { t h e ~ p r o j e c t e d ~ s y s t e m ~ ( 1 8 ) ~ i s ~ m e a n { \mathrm { - } } s q u a r e ~ s t a b l e } } \right\} .\tag{19}
$$

The next proposition gives an equivalent way to determine this critical LR from the homogeneous transition. Its second-moment matrix is

$$
\begin{array} { r } { K ^ { + } ( \eta ) : = \mathbb { E } [ M _ { t } ^ { + } ( \eta ) \otimes M _ { t } ^ { + } ( \eta ) ] . } \end{array}\tag{20}
$$

We assume that the pairs $\left( { H } _ { B _ { t } } , { g } _ { B _ { t } } \right)$ are i.i.d. and independent of the initial state. Both entries have finite second moments, each Hessian is symmetric, and their means are the population derivatives at the checkpoint. These are the mini-batch assumptions of Assumption C.1.

Proposition A.3 (Spectral characterization of the AGM critical LR). Under Assumption $C . l ,$ fix $P \succ 0 , 0 \leq \beta _ { 1 } < 1$ and $\lambda _ { \mathrm { W D } } \geq 0$ . Suppose the strictly positive eigenspace of A is nonempty. The critical LR in Definition A.2 satisfies

$$
0 < \eta _ { \star } = \operatorname* { s u p } \bigl \{ \eta > 0 : \rho ( K ^ { + } ( \eta ) ) < 1 \bigr \} < \infty .\tag{21}
$$

Moreover,

$$
\rho \big ( \mathbf { { K } } ^ { + } ( \eta ) \big ) \left\{ \begin{array} { l l } { < 1 , } & { 0 < \eta < \eta _ { \star } , } \\ { = 1 , } & { \eta = \eta _ { \star } , } \\ { > 1 , } & { \eta > \eta _ { \star } . } \end{array} \right.\tag{22}
$$

Below $\eta _ { \star }$ , the projected affine system is mean-square stable. Above $\eta _ { \star }$ , some deterministic initialization has an unbounded second moment. $\ A \eta _ { \star }$ , boundedness can depend on the forcing. Thus the proposition equates the two critical LRs without identifying boundedness with strict decay at the boundary. The proof is in Appendix D.6. AGM training is at EoS when its training LR is close to $\eta _ { \star }$

The stability notion and the projection are those of the main text. For SGD without weight decay, this critical LR is the one of Definition 4.3 applied to the local quadratic. The next two subsections describe how we estimate the effective curvature and the critical LR of this local model.

## A.2 EOC DETECTION

EoC at ${ \pmb w } _ { t _ { 0 } }$ means that A has a negative eigenvalue while the saddleness $\nu$ is small. Detecting it requires $\lambda _ { \operatorname* { m a x } } ( A )$ and $\lambda _ { \mathrm { m i n } } ( A )$ We estimate the maximum eigenvalue by Lanczos on a large batch. In our measurements, the top eigenpair varies little across mini-batches, whereas the bottom eigenvectors are nearly orthogonal (Figure 6). Lanczos on one mini-batch estimates $\lambda _ { \operatorname* { m i n } } ( A _ { B _ { t } } )$ whose expectation is at most $\lambda _ { \mathrm { m i n } } ( A )$ by Jensen’s inequality. We therefore estimate $\lambda _ { \mathrm { m i n } } ( A )$ by a stochastic power iteration method that draws a fresh mini-batch at every step.

![](images/95d7269001580ad6bf585ff26d856f97daf6866289426a154dd810e91bfa3a6c.jpg)

![](images/f8bdea8c356f35aa26940216f54450a5070f7738941de3d81269a7f90c41d954.jpg)  
Figure 6: Mini-batch Lanczos estimates are stable for the top eigenpair but unstable for the bottom eigenpair. We run Lanczos on ten independently sampled mini-batches at the same checkpoint for the Adam-preconditioned Hessian. The maximum eigenvectors have high pairwise absolute cosine and similar eigenvalues across batches. In contrast, the minimum eigenvectors are nearly orthogonal across batches and the corresponding $\lambda _ { \mathrm { m i n } }$ values vary substantially, showing that per-batch Lanczos estimates are dominated by batch-specific negative-curvature directions.

Stochastic power iteration with bisection. Algorithm 1 runs the shifted iteration on the mini-batch effective curvatures,

$$
\pmb { x } _ { t + 1 } = \big ( \pmb { I } - \eta _ { \mathrm { p } } ( \pmb { A } _ { \mathcal { B } _ { t } } + \lambda \pmb { I } ) \big ) \pmb { x } _ { t } , \qquad t = 0 , \ldots , T _ { \mathrm { p } } - 1 ,
$$

with a fresh mini-batch at every step, each $\mathbf { } A _ { B _ { t } } \mathbf { x } _ { t }$ formed from one Hessian-vector product. Since the batches are independent, the mean iterate follows the population matrix, $\mathbb { E } [ { \pmb { x } } _ { t + 1 } ] \stackrel { \cdot } { = } ( { \pmb { I } } - { \eta } _ { \mathrm { p } } ( { \pmb { A } } +$ $\lambda I ) ) \mathbb { E } [ \mathbf { x } _ { t } ]$ . If $\lambda _ { \operatorname* { m i n } } ( A ) + \lambda < 0$ , this matrix has an eigenvalue larger than one. When the initial vector has nonzero overlap with a bottom eigenvector, the component of $\mathbb { E } [ \mathbf { \boldsymbol { x } } _ { t } ]$ along that eigenvector grows geometrically. If $\bar { \lambda } _ { \operatorname* { m i n } } ( A ) + \lambda \geq 0$ and $\eta _ { \mathrm { p } }$ is small enough, $\mathbb { E } [ \mathbf { \boldsymbol { x } } _ { t } ]$ stays bounded. A query at shift λ is therefore declared to explode when $\| \boldsymbol { x } _ { t } \|$ exceeds the cutoff B at some step $t \leq T _ { \mathrm { p } }$ , a relative cutoff since $\| \pmb { x } _ { 0 } \| = 1$ . Bisection starts from a given search interval $[ 0 , \Lambda ]$ and moves its lower endpoint to an exploding shift and its upper endpoint to a non-exploding shift. After N rounds the output $\widehat { \lambda }$ is the final upper endpoint, and the estimate of the minimum eigenvalue is $\widehat { \lambda } _ { \mathrm { m i n } } = - \widehat { \lambda }$

Three choices matter in practice: the initial vector, the stepsize, and the search interval. The initial vector $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ is the same unit vector for every query. For nonzero checkpoint momentum, we use ${ { m } _ { t _ { 0 } } } / { \| { \boldsymbol { m } } _ { t _ { 0 } } \| }$ . The stepsize $\eta _ { \mathrm { p } }$ must satisfy $\dot { \eta _ { \mathrm { p } } } ( \lambda _ { \mathrm { m a x } } ( A ) + \Lambda ) \leq 1$ over the whole search interval. This makes the mean transition positive semidefinite for every queried shift and keeps $\| I - \eta _ { \mathrm { p } } ( A + \lambda I ) \| \le 1$ for every $\lambda \in [ \lambda _ { \star } , \bar { \Lambda } ]$ , where $\lambda _ { \star } : = \operatorname* { m a x } \{ 0 , - \lambda _ { \operatorname* { m i n } } ( \mathbf { \bar { A } } ) \}$ . We set it from the Lanczos estimate $\lambda _ { \operatorname* { m a x } } .$ , which Algorithm 2 computes first on ${ \bar { A } } ,$ , the effective curvature averaged over a fixed set of mini-batches. This estimate does not certify the population stepsize condition. The search interval $\left[ 0 , \Lambda \right]$ is an input to the algorithm and must contain $\lambda _ { \star }$ . The guarantee takes it as given, assumes $\lambda _ { \operatorname* { m i n } } ( \mathbf { \bar { A } } ) \overset { . } { < } 0$ , and includes the bisection resolution $\Lambda 2 ^ { - N }$ . We use $B \geq 2$ . The algorithm also runs when $\lambda _ { \operatorname* { m i n } } ( A ) \geq 0$ , but the guarantee below does not cover that case. Each query uses at most $T _ { \mathrm { p } }$ Hessian-vector products. Algorithms 2 and 3 pass $T _ { \mathrm { p } }$ to Algorithm 1, and Algorithm 3 writes T for its own simulation horizon. The horizons used in the experiments are listed in Appendix B.1.

Algorithm 1 Minimum eigenvalue estimation by stochastic power iteration and bisection   
Require: Symmetric samples $\pmb { A } _ { B _ { t } }$ , i.i.d. with mean A; inner stepsize $\eta _ { \mathrm { p } } ;$ horizon $T _ { \mathrm { p } } ;$ bisection   
rounds ${ \dot { N } } ;$ cutoff $B \geq 2 ;$ a given unit vector ${ \pmb x } _ { 0 } ;$ and a search interval $[ 0 , \Lambda ]$ containing $\lambda _ { \star } =$   
max $\{ 0 , - \lambda _ { \operatorname* { m i n } } ( A ) \}$ with $\eta _ { \mathrm { p } } \overline { { ( \lambda _ { \mathrm { m a x } } ( A ) + \Lambda ) } } \le 1 .$   
Ensure: Estimated shift λb and the estimate $\widehat { \lambda } _ { \mathrm { m i n } } = - \widehat { \lambda }$ of the minimum eigenvalue.   
1: Subroutine STABLE(λ):   
2: Set ${ \pmb x }  { \pmb x } _ { 0 } ;$ use fresh independent samples for this query.   
3: for $t = 0 , 1 , \ldots , T _ { \mathrm { p } } - 1$ do   
4: Draw a fresh mini-batch $B _ { t }$ and form $\pmb { A } _ { B _ { t } }$   
5: $\begin{array} { r } { \pmb { x }  \pmb { x } - \eta _ { \mathrm { p } } ( \pmb { A } \ B _ { t } + \lambda \pmb { I } ) \pmb { x } . } \end{array}$   
6: $\mathbf { i f } \parallel \mathbf { x } \parallel > B$ then   
7: return false {explode}   
8: end if   
9: end for   
10: return true {stable}   
11: Main procedure:   
12: $\lambda _ { \mathrm { l o } }  \mathrm { \bar { 0 } } , \quad \lambda _ { \mathrm { h i } }  \Lambda .$ . {start from the given search interval}   
13: for $k = 1 , 2 , \ldots , N$ do   
14: $\lambda _ { \mathrm { m i d } }  ( \lambda _ { \mathrm { l o } } + \lambda _ { \mathrm { h i } } ) / 2 .$   
15: if STABLE $( \lambda _ { \mathrm { m i d } } ) = \mathrm { t }$ rue then   
16: $\lambda _ { \mathrm { h i } }  \lambda _ { \mathrm { m i d } }$ {stable}   
17: else   
18: $\lambda _ { \mathrm { l o } }  \lambda _ { \mathrm { m i d } }$ {explode}   
19: end if   
20: end for   
21: $\widehat { \lambda }  \lambda _ { \mathrm { h i } } , ~ \widehat { \lambda } _ { \mathrm { m i n } }  - \widehat { \lambda } .$   
22: return $\hat { \lambda } , \ \hat { \lambda } _ { \mathrm { { m i n } } } .$

Algorithm 2 EoC detection for AGMs   
Require: Checkpoint $\mathbf { \Delta } _ { \mathbf { \mathscr { w } } _ { t _ { 0 } } ; }$ frozen preconditioner $\begin{array} { r } { P = P _ { t _ { 0 } } \succ 0 ; } \end{array}$ weight decay $\lambda _ { \mathrm { W D } } \geq 0 ;$ tolerance   
$\nu _ { \mathrm { t o l } } > 0 ;$ momentum state $\begin{array} { r } { m _ { t _ { 0 } } ; } \end{array}$ Algorithm 1 parameters $( \eta _ { \mathrm { p } } , T _ { \mathrm { p } } , N , B , \dot { \Lambda } ) ;$ Lanczos iterations   
$T _ { \mathrm { L Z } } ;$ averaged operator A<sup>¯</sup>, the mean of $A _ { \boldsymbol { B } }$ over a fixed set of mini-batches.   
Ensure: EO $\mathsf { C } \in \{ \mathsf { t r u e } ,$ false}; estimates $\widehat { \lambda } _ { \mathrm { m i n } } , \widehat { \lambda } _ { \mathrm { m a x } } ,$ and ${ \widehat { \nu } } .$   
1: Define symmetric stochastic samples $\begin{array} { r } { A _ { \mathcal { B } _ { t } } : = P ^ { - 1 / 2 } H _ { \mathcal { B } _ { t } } P ^ { - 1 / 2 } + \lambda _ { \mathrm { W D } } I . } \end{array}$   
2: $\widehat { \lambda } _ { \operatorname* { m a x } } \gets \mathrm { I }$ ANCZOSMAXEIG(A<sup>¯</sup>; T<sub>LZ</sub>). {η<sub>p</sub> is chosen from $\widehat { \lambda } _ { \mathrm { m a x } } ,$ see the text}   
3: Set $\pmb { x } _ { 0 } \gets { \pmb { m } } _ { t _ { 0 } } / \lVert { \pmb { m } } _ { t _ { 0 } } \rVert .$   
4: $( \widehat { \lambda } , \widehat { \lambda } _ { \operatorname* { m i n } } ) \gets \mathbf { A }$ lgorithm $1 ( A _ { \mathit { B } _ { t } } ; \eta _ { \mathrm { p } } , T _ { \mathrm { p } } , N , B , \Lambda , { \pmb x } _ { 0 } )$   
5: $\widehat { \nu } \gets \frac { \operatorname* { m a x } \{ 0 , - \widehat { \lambda } _ { \operatorname* { m i n } } \} } { \widehat { \nu } }$   
λb<sub>max</sub>   
6: EOC $:  ( \widehat { \lambda } _ { \mathrm { m i n } } < 0 ) \ \land \ ( \widehat { \nu } \leq \nu _ { \mathrm { t o l } } )$   
7: return EOC, $\widehat { \lambda } _ { \mathrm { m i n } } , \widehat { \lambda } _ { \mathrm { m a x } } , \widehat { \nu } .$

EoC criterion. Algorithm 2 combines the two eigenvalue estimates into the saddleness estimate $\widehat { \nu } = \operatorname* { m a x } \{ 0 , - \widehat { \lambda } _ { \operatorname* { m i n } } \} / \widehat { \lambda } _ { \operatorname* { m a x } } .$ , which estimates the AGM saddleness ν of Definition A.1. We declare EoC when $\widehat { \lambda } _ { \operatorname* { m i n } } < 0$ and $\widehat \nu \leq \nu _ { \mathrm { t o l } }$ . This ratio requires $\widehat { \lambda } _ { \mathrm { m a x } } > 0$ . The experiments use $\nu _ { \mathrm { t o l } } = 0 . 1$ The same estimate sets the compensation coefficient $\lambda _ { \mathsf { P } } = \operatorname* { m a x } \{ 0 , - { \widehat { \lambda } } _ { \operatorname* { m i n } } \}$ for EoS detection.

Guarantee. The following theorem separates the error due to curvature noise, the finite horizon, and the bisection resolution.

Theorem A.4 (Informal; accuracy of the stochastic power iteration). Let $\lambda _ { \operatorname* { m i n } } ( A ) < 0 ,$ and let the mini-batch effective curvatures $\dot { \pmb { A } _ { B _ { t } } } = \pmb { A } + \pmb { \Delta } _ { t }$ be i.i.d. symmetric with $\mathbb { E } [ \Delta _ { t } ] { \stackrel { . } { = } } \mathbf { 0 }$ and $\| \Delta _ { t } \| \leq \sigma$ almost surely. Let the unit initialization $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ satisfy $| \pmb { u } _ { \mathrm { m i n } } ^ { \top } \pmb { x } _ { 0 } | \geq \varepsilon > 0$ for a unit eigenvector $\mathbf { \delta } \mathbf { u } _ { \mathrm { m i n } }$ of A with eigenvalue $\lambda _ { \operatorname* { m i n } } ( A )$ , and let the search-interval and stepsize conditions of Theorem C.12 hold. Forfixed confidence $1 - \delta _ { \mathrm { { i } } }$ , Algorithm 1 with horizon $T _ { \mathrm { p } } ,$ , cutoff B and N bisection rounds outputs, with probability at least $1 - \delta$

$$
\big | \widehat { \lambda } _ { \mathrm { m i n } } - \lambda _ { \mathrm { m i n } } ( \pmb { A } ) \big | \leq O \big ( \eta _ { \mathrm { p } } \log ( 2 N ) \big ) + O \Big ( \frac { 1 } { \eta _ { \mathrm { p } } T _ { \mathrm { p } } } \Big ) + \Lambda 2 ^ { - N } ,
$$

where the hidden constants depend on the noise bound $\sigma ,$ cutoff B, initial overlap ε, and failure probability δ.

The first term comes from curvature noise, the second from running each query for only $T _ { \mathrm { p } }$ steps, and the third from the number of bisection rounds. For large $T _ { \mathrm { p } } ,$ , choosing $\eta _ { \mathrm { p } }$ of order $[ T _ { \mathrm { p } } \log ( 2 N ) ] ^ { - 1 / 2 }$ balances the first two terms and gives $O ( \sqrt { \log ( 2 N ) / T _ { \mathrm { p } } } )$ , plus the bisection error. Increasing the horizon at a fixed stepsize reduces the finite-horizon term but does not remove the noise term from this bound. The theorem does not verify the search-interval condition or the overlap. Both are inputs, and the theorem is silent when $\lambda _ { \operatorname* { m i n } } ( \dot { A } ) \geq 0$ . The formal statement is Theorem C.12 in Appendix C.3, and the proof is in Appendix E.

```latex
Algorithm 3 EoS detection for AGMs
Require: Checkpoint $\mathbf { \Delta } _ { \mathbf { \mathscr { w } } _ { t _ { 0 } } ; }$ momentum state $\mathbf { \nabla } m _ { t _ { 0 } } ;$ frozen preconditioner $\boldsymbol { P } = \boldsymbol { P } _ { t _ { 0 } } \succ 0 ;$ weight
decay $\lambda _ { \mathrm { W D } } \bar { \geq } 0 ;$ training LR η; momentum parameter $\bar { \beta _ { 1 } } \in [ 0 , 1 )$ ; horizon T; cutoff $M ;$ random
seed s; increasing LR factors $\{ \alpha _ { k } \} _ { k = 1 } ^ { K } ;$ EoS tolerance $\tau _ { \mathrm { { e o s } } } > 0 ;$ parameters $( \eta _ { \mathrm { p } } , T _ { \mathrm { p } } , N , B , \Lambda , \pmb { x } _ { 0 } )$
for Algorithm 1, with $\pmb { x } _ { 0 } = { m _ { t } } _ { 0 } / \lVert \pmb { m } _ { t _ { 0 } } \rVert$
Ensure: Estimated critical LR $\widehat { \eta } _ { \star }$ , critical LR ratio $q ( \pmb { w } _ { t _ { 0 } } ) = \widehat { \eta } _ { \star } / \eta .$ , and EOS flag.
1: (Step 1: estimate the compensation coefficient.) Define symmetric stochastic samples
$\begin{array} { r } { \pmb { A } _ { \mathcal { B } _ { t } } : = \pmb { P } ^ { - 1 / 2 } \pmb { H } _ { \mathcal { B } _ { t } } \pmb { P } ^ { - 1 / 2 } + \lambda _ { \mathrm { W D } } \pmb { I } . } \end{array}$ . Run Algorithm 1 on $\pmb { A } _ { B _ { t } }$ to obtain $\widehat { \lambda } _ { \mathrm { m i n } }$ . Set $\lambda _ { \mathsf { P } } \gets$
max $\{ 0 , - \widehat { \lambda } _ { \mathrm { m i n } } \}$
2: Subroutine EXPLODEDETECT(α):
3: Set the random seed to $s ; \epsilon \gets \mathbf { 0 } ; m \gets m _ { t _ { 0 } } ; R \gets 0 .$
4: for $t = 0 , 1 , \ldots , T - 1$ do
5: Sample mini-batch $B _ { t } .$
6: $g \gets g _ { \mathcal { B } _ { t } } + H _ { \mathcal { B } _ { t } } \epsilon + \lambda _ { \mathsf { P } } P \epsilon .$
7: $\pmb { m }  \beta _ { 1 } \pmb { m } + ( \hat { 1 } - \beta _ { 1 } ) \pmb { g } .$
8: $\epsilon  \epsilon - \alpha \eta P ^ { - 1 } m - \alpha \eta \lambda _ { \mathrm { W D } } ( \epsilon + w _ { t _ { 0 } } )$
9: $R \gets \operatorname* { m a x } \{ R , \| \epsilon \| _ { P } \}$
10: if R ≥ M then
11: return true {explode}
12: end if
13: end for
14: return $\mathtt { f a l s e }$ {stable}
15: (Step 2: search over $\{ \alpha _ { k } \} . )$ Find k<sup>⋆</sup> := min{k : EXPLODEDETEC $\Gamma ( \alpha _ { k } ) = \mathtt { t r u e } \}$
16: if no such $k ^ { \star }$ exists then
17: Record the censored bounds $\widehat { \eta } _ { \star } > \alpha _ { K } \eta$ and $q > \alpha _ { K } ;$ set ${ \mathrm { E o S } }  { \mathrm { f a l } } { \mathrm { s e } } .$
18: return $\widehat { \eta } _ { \star } , q ,$ EOS.
19: end if
20: $\widehat { \eta } _ { \star } \gets \alpha _ { k ^ { \star } } \eta .$
21: $q ( w _ { t _ { 0 } } ) \gets \alpha _ { k ^ { \star } }$
22: $\mathrm { E o S }  ( q ( \pmb { w } _ { t _ { 0 } } ) \in [ 1 - \tau _ { \mathrm { e o s } } , 1 + \tau _ { \mathrm { e o s } } ] )$
23: return $\widehat { \eta } _ { \star } , q ,$ EOS.
```

## A.3 EOS DETECTION

The AGM critical LR $\eta _ { \star }$ is the threshold for bounded second moments on the strictly positive eigenspace of A (Definition $\mathrm { A } . 2 )$ . Forming this projection is impractical at LLM scale, and running the local dynamics without it is misleading, because negative effective curvature produces an expanding direction at every positive LR (Lemma C.5). We therefore replace the projection by compensation: the penalty $\frac { \lambda _ { \mathsf { P } } } { 2 } \left\| \epsilon \right\| _ { P } ^ { 2 }$ , with compensation coefficient $\lambda _ { \mathsf { P } } = \operatorname* { m a x } \{ 0 , - { \widehat { \lambda } } _ { \operatorname* { m i n } } \}$ from Appendix $\mathrm { A . 2 } ,$ , shifts the effective curvature from A to $A + \lambda _ { \mathsf { P } } I$ . We then locate the stability boundary of this compensated model by simulation.

Compensated local loss. Let $\epsilon : = w - w _ { t _ { 0 } }$ be the displacement from the checkpoint. With $\lambda _ { \mathsf { P } }$ from Appendix A.2, the compensated local loss is the quadratic surrogate (13) with the penalty measured in the norm of the frozen preconditioner,

$$
\begin{array} { r } { G _ { t } ( \epsilon ) : = \mathcal { L } _ { \mathcal { B } _ { t } } ( \boldsymbol { w } _ { t _ { 0 } } ) + \epsilon ^ { \top } g _ { \mathcal { B } _ { t } } + \frac { 1 } { 2 } \epsilon ^ { \top } H _ { \mathcal { B } _ { t } } \epsilon + \frac { \lambda _ { P } } { 2 } \| \epsilon \| _ { P } ^ { 2 } , \qquad \| \epsilon \| _ { P } ^ { 2 } = \epsilon ^ { \top } P \epsilon . } \end{array}
$$

The penalty shifts the mini-batch effective curvature from $\pmb { A } _ { B _ { t } }$ to $A _ { B _ { t } } + \lambda _ { \mathsf { P } } I$ in preconditioned coordinates. It enters the gradient used in the momentum update, while the decoupled weight decay keeps its coefficient λ<sub>WD</sub>. Appendix C.1 writes the resulting dynamics as a stochastic linear system.

Local simulation. Algorithm 3 first sets $\lambda _ { \mathsf { P } }$ from the power-iteration estimate and then runs the local AGM dynamics (14) with the compensated gradient $\mathsf { \bar { g } } p _ { \mathsf { t } } + ( \pmb { H } _ { B _ { t } } + \lambda _ { \mathsf { P } } \pmb { P } ) \epsilon _ { t }$ . The preconditioner is frozen at ${ \dot { \mathbf { P } } } .$ The momentum starts from the checkpoint value $\mathbf { \nabla } m _ { t _ { 0 } }$ and is updated inside the simulation. The displacement starts from ${ \epsilon _ { 0 } } = { \bf 0 }$ . One trajectory is run for each LR factor $\alpha _ { k }$ in an increasing grid $\alpha _ { 1 } < \cdots < \alpha _ { K }$ around the training LR $\eta ,$ and all trajectories use the same mini-batch sequence, fixed through the random seed s. A trajectory is marked as exploding when $\| \epsilon _ { t } \| _ { P }$ reaches the cutoff M at some step $t \leq T$ . The estimate $\widehat { \eta } _ { \star }$ is the smallest LR in the grid at which the trajectory explodes, and $q = \widehat { \eta } _ { \star } / \eta$ is the critical LR ratio reported in the experiments. We declare EoS when $q \in [ 1 - \tau _ { \mathrm { e o s } } , 1 + \tau _ { \mathrm { e o s } } ]$ , with $\tau _ { \mathrm { e o s } } = 0 . 1$ 1 in the experiments $( { \mathrm { F i g u r e } } 4 )$ . If no factor in the grid explodes, the run is recorded as censored, $q > \alpha _ { K }$ , and the checkpoint is not counted as being at EoS. We hold the training LR $\eta$ fixed during the simulation because the simulation horizon $T$ is short relative to the training schedule. This approximates schedules such as WSD or cosine whenever their LR changes little over the simulation window.

The expected detector. We analyze an idealized version, the expected detector. It runs the same stochastic dynamics but replaces the single trajectory by $\mathbb { E } \Vert \epsilon _ { t } \Vert ^ { 2 }$ , since $\eta _ { \star }$ is a mean-square threshold (Appendix C.2). Fix a search window $\check { \tau }$ around $\eta ,$ the horizon $T$ and a cutoff $M > 1$ . Run the same compensated dynamics from every unit displacement $\| \epsilon _ { 0 } \| = 1$ with the checkpoint momentum, and take the expectation over the mini-batches. The expected detector returns

$$
\bar { \eta } _ { \star } ( M , T ) = \operatorname* { s u p } \Bigl \{ \eta ^ { \prime } \in \mathbb { Z } \colon \operatorname* { m a x } _ { 0 \le t \le T } \mathbb { E } \| \epsilon _ { t } \| ^ { 2 } \le M ^ { 2 } \mathrm { f o r } \mathrm { e v e r y } \| \epsilon _ { 0 } \| = 1 \Bigr \} .
$$

The detector accepts $\eta ^ { \prime }$ if the condition in the braces holds, so $\bar { \eta } _ { \star } ( M , T )$ is the supremum of the accepted LRs (Appendix C.4). We keep $\widehat { \eta } _ { \star }$ for the output of the algorithm. The implemented detector differs from it in three ways. First, the expected detector takes the supremum of accepted LRs in a continuous window, while the algorithm returns the first exploding LR in an increasing grid. Finite-horizon acceptance need not be monotone in the LR, so these thresholds can differ. Second, the expected detector starts from unit displacements, so $M > 1$ is a relative cutoff, while the algorithm starts from zero and uses $M$ as an absolute bound on $\| \epsilon _ { t } \| _ { P }$ . Third, the expected detector takes the expectation of $\| \epsilon _ { t } \| ^ { 2 }$ over the mini-batch sequence, while the algorithm runs one trajectory. The algorithm uses the preconditioner norm, in which the compensation is measured; the analysis of the expected detector is unchanged under this norm (Remark F.10).

Guarantee. Let $\eta _ { \star } ^ { \mathsf { P } }$ , the compensated critical LR, be the critical LR of the compensated loss without projection (Definition C.13). The output of the expected detector differs from $\eta _ { \star }$ because $\eta _ { \star } ^ { \mathsf { P } } \neq \eta _ { \star }$ and because the simulation stops after $T$ steps. The following theorem bounds these two errors separately in inverse LR. It assumes a fixed model with $A + \lambda _ { \mathsf { P } } I ^ { } \succ 0$ (Assumption C.14), $\eta ^ { \prime } \lambda _ { \mathrm { W D } } \leq \bar { 1 }$ on the search window, and that the expected detector accepts $\eta _ { \star } ^ { \mathsf { P } }$ (Assumption C.15).

Theorem A.5 (Informal; compensated finite-horizon threshold). Let $\bar { \eta } _ { \star } ( M , T )$ be the output of the expected detector, and let $v \geq 0$ satisfy $\mathbb { E } ( \Delta _ { t } ^ { + } ) ^ { 2 } \preceq v \pmb { A } ^ { + }$ and E $\mathbf { \Delta } _ { \cdot } \Delta _ { t } ^ { 2 } \preceq v ( A + \lambda _ { \mathsf { P } } I )$ , where $\Delta _ { t } = A _ { B _ { t } } - A$ and the superscript + denotes the block on the positive eigenspace of A. Under the assumptions in Appendix $C . 4 ,$

$$
\left| \frac { 1 } { \bar { \eta } _ { \star } ( M , T ) } - \frac { 1 } { \eta _ { \star } } \right| \leq \underbrace { \frac { 1 - \beta _ { 1 } } { 2 ( 1 + \beta _ { 1 } ) } \lambda _ { \mathrm { P } } } _ { c o m p e n s a t i o n } + \underbrace { \frac { v } { 2 } } _ { H e s s i a n n o i s e } + O \Bigl ( \frac { 1 + \log M } { T } \Bigr ) .
$$

The $O ( \cdot )$ is for a fixed model and search window as $( 1 + \log M ) / T \to 0$ , over choices of M and T that satisfy Assumption C.15.

The compensation term is linear in $\lambda _ { \mathsf { P } } .$ , which equals $\widehat { \nu } \widehat { \lambda } _ { \mathrm { m a x } }$ for the detector. The noise level v need not be small. For a fixed cutoff M, the finite-horizon term decreases as $1 / T$ . Corollary F.8 proves this simplified bound. Theorem C.16 gives a bound of a different form, which compares the noise of the two systems directly instead of bounding each by $v .$ Small saddleness alone need not make the total error small. The theorem applies to the expectation over mini-batches from unit initial displacements, not to the single trajectory from zero displacement that the experiments run. Since the compensation coefficient λ<sub>P</sub> is estimated, it does not by itself guarantee $\bar { A ^ { + } } + \lambda _ { \mathsf { P } } I \succ 0 .$ . Proposition F.11 gives a sufficient cutoff for accepting the compensated critical LR. The complete assumptions and the full theorem are in Appendix C.4. The simplified bound and its proof are in Appendix F.3.

## A.4 COST OF A HESSIAN-VECTOR PRODUCT

Curvature enters both procedures only through Hessian-vector products (HVPs), which we compute with PyTorch’s autograd by differentiating a gradient-vector inner product,

$$
\begin{array} { r } { g ( \pmb { w } ) = \nabla _ { \pmb { w } } \mathcal { L } ( \pmb { w } ) , \qquad \mathrm { H V P } ( \pmb { v } ) = \nabla _ { \pmb { w } } \big ( \pmb { g } ( \pmb { w } ) ^ { \top } \pmb { v } \big ) = \nabla _ { \pmb { w } } ^ { 2 } \mathcal { L } ( \pmb { w } ) \pmb { v } . } \end{array}\tag{23}
$$

We estimate the cost for a linear layer, since matrix multiplications dominate the attention and MLP projections in our models.

Matrix-multiplication count for a linear layer. Consider a linear layer

$$
Y = X W ,\tag{24}
$$

where $\ b { X } \in \mathbb { R } ^ { n \times d _ { \mathrm { i n } } }$ holds a batch of n inputs, $W \in \mathbb { R } ^ { d _ { \mathrm { i n } } \times d _ { \mathrm { o u t } } }$ , and $\pmb { Y } \in \mathbb { R } ^ { n \times d _ { \mathrm { o u t } } }$ . Let $G = \partial \mathcal { L } / \partial Y$ be the upstream gradient. The forward pass uses one general matrix multiplication (GEMM),

$$
Y = X W \quad \Rightarrow \quad { \mathrm { 1 G E M M } } .\tag{25}
$$

The first-order backward pass computes gradients with respect to X and $W$

$$
\frac { \partial \mathcal { L } } { \partial X } = G W ^ { \top } , \qquad \frac { \partial \mathcal { L } } { \partial W } = X ^ { \top } G ,\tag{26}
$$

which costs two GEMMs. Thus, a forward and backward pass through a linear layer costs approximately three GEMMs, ignoring elementwise operations.

The autograd procedure above computes an HVP by backpropagating through the gradient computation in a second reverse-mode pass. For layers whose cost is dominated by matrix multiplication, this incurs overhead comparable to an additional forward and backward pass through the same linear operators, plus extra GEMMs from differentiating the backward equations. Including the required second-order terms gives approximately

$$
\begin{array} { r } { \underbrace { 1 } _ { \mathrm { f o r w a r d } } + \underbrace { 2 } _ { \mathrm { b a c k w a r d } } + \underbrace { 2 } _ { \mathrm { d i f f e r e n t i a t e ~ b a c k w a r d ~ ( w , r , ~ } W ) } + \underbrace { 4 } _ { \mathrm { p r o p a g a t e d i r e c t i o n a l ~ a d j o i n t s } } \approx \underbrace { 9 \mathrm { G E M M s } } _ { \mathrm { d i f f e r e n t i a t e ~ b a c k w a r d ~ ( w , r , t ~ } W ) } . } \end{array}\tag{27}
$$

This is about $3 \times$ the cost of one gradient evaluation (9 versus 3 GEMMs per linear layer), or 9× a forward pass when GEMMs dominate. Since transformer blocks contain many such linear layers (QKV projections, the attention output projection, and the two MLP projections), this constant-factor overhead provides a practical estimate of HVP cost for our models.

## B EXPERIMENTAL DETAILS

This appendix gives the models, training and detection settings behind Section $^ { 6 , }$ the protocol of the batch size threshold experiment, and the figures referenced from the main text but not shown there.

## B.1 MODELS, TRAINING, AND DETECTION SETTINGS

Model architectures. Table 1 lists the models used in our experiments.

Table 1: Model architectures used in our experiments.
<table><tr><td>Model</td><td>#Layers</td><td>Hidden Dim</td><td>#Heads</td><td>Head Dim</td><td>Params</td></tr><tr><td>GPT-extra-small</td><td>4</td><td>768</td><td>12</td><td>64</td><td>43M</td></tr><tr><td>GPT-small</td><td>12</td><td>768</td><td>12</td><td>64</td><td>104M</td></tr><tr><td>GPT-medium</td><td>24</td><td>1,024</td><td>16</td><td>64</td><td>344M</td></tr><tr><td>GPT-large</td><td>36</td><td>1,280</td><td>20</td><td>64</td><td>787M</td></tr></table>

For each dataset, we train a byte-level BPE tokenizer with a vocabulary of 8,192 tokens and use it for all model sizes. We use a small vocabulary instead of the standard 50,257-token GPT-2 tokenizer because rarely trained tokens distort the preconditioned curvature. Under Adam, the second-moment estimate of an embedding row grows with how often its token is trained on, so a token that is seldom seen has $\sqrt { \hat { v } }$ close to the floor set by ϵ. When such a token appears in the batch used for the Hessian-vector products, its row is amplified by roughly $1 / \sqrt { \hat { v } }$ and can dominate the top eigenvalues. A smaller vocabulary leaves fewer such tokens. With 8,192 tokens, at most 72 tokens are rare by the criterion below at the checkpoints we used to build the filter.

Even with the smaller vocabulary, we remove the remaining rare tokens from the data used for EoS and EoC detection. All detection batches come from a single training shard that no run is trained on. For a run with batch size B at checkpoint step K, we call a token rare if it has not appeared in the first K training steps, if it has appeared less than once per step on average, or if it has not appeared in the last 20 steps. The padding token is always treated as rare. We take the union of the rare sets over the stable-phase checkpoints of all three model sizes (72 tokens) and drop every sequence of the shard whose inputs or targets contain a rare token. We also drop every sequence in which a token that appears fewer than 100 times per training step occurs more than 32 times, because such a burst over-represents that token’s row in a single batch. The filtered shard keeps 92,060 of 97,656 sequences (94.3%), and the same shard is used for every model size, learning rate and checkpoint.

![](images/f7bef659a9ae6b422c65b2ad4c04891ecf977c388eb1e033730786bb16c2be17.jpg)

![](images/bea686faa3d182246c8504a8165976f761044f5cfd9770c4745314266e3d873b.jpg)  
(a) ResNet20 Model  
(b) ResNet32 Model  
Figure 7: EoS and EoC appears on model trained with SGD. We train ResNet20 and ResNet32 with MSE loss and SGD, using batch size 1024 and weight decay $5 \times 1 0 ^ { - 4 }$ for 50 epochs. Panels follow Figure 2, using the raw Hessian and epoch on the horizontal axis.

For model architectures, we apply QK-normalization (Henry et al., 2020) and attention gating (Qiu et al., 2025) for stability. All models are trained with a sequence length of 1024.

We run most experiments on GPT-extra-small. Unless otherwise specified, we use 100 stochastic power iterations and 50 compensated local simulation steps for GPT-extra-small. For the larger models (GPT-small/medium/large), we use 50 power iterations and 25 local simulation steps. In all local simulations, we mark a run as exploding when the displacement norm reaches the cutoff $M = 1 0 ^ { 5 }$

## B.2 ADDITIONAL RESULTS

EoS across the pretraining runs. The bounded-to-exploding transition of the compensated local simulation appears throughout our AdamW pretraining runs. Across models from 43M to 787M parameters, Figures 2 and 8 show many checkpoints at which the local simulation changes sharply from bounded trajectories to explosion near LR factor 1. Figures 3 and 8 show that the location and duration of this behavior depend on the LR schedule and the peak LR. Figures 4 and 9 show that it also depends on the batch size. Taken together, these experiments show that EoS is common in our AdamW pretraining runs, and absent at the smallest batch size and extreme peak LRs. Within the LR range tested around the training LR, the simulations show one transition at most checkpoints. Under the assumptions of Theorem C.7, strict homogeneous stability holds on a single LR interval whose endpoint is the AGM critical LR (Proposition A.3).

Robustness to the compensation coefficient. Figure 10 shows one checkpoint with three compensation coefficients. The sharp transition is absent without compensation, appears at the estimated value, and remains visible at the larger value 600.

## B.3 BATCH SIZE THRESHOLD EXPERIMENTS

This subsection gives the protocol behind Section 6.3. We study five datasets (FineWeb (Penedo et al., 2024), OpenWebMath (Paster et al., 2024), OpenWebText (Gokaslan & Cohen, 2019), C4 (Raffel et al., 2020), and DCLM (Li et al., 2024)) using GPT-extra-small. We use AdamW and the WSD schedule throughout, with baseline weight decay $\lambda _ { \mathrm { W D } } = 0 . 1$

To vary the batch size, we follow the square-root scaling rule (Malladi et al., 2022). When increasing the batch size from B to kB, we scale the LR and the weight decay, keeping the remaining hyperparameters fixed,

$$
\eta  \sqrt { k } \eta , \qquad \lambda _ { \mathrm { W D } }  \sqrt { k } \lambda _ { \mathrm { W D } } ,
$$

so that the timescale $B / ( \eta \lambda _ { \mathrm { W D } } )$ is the same for all batch sizes. We also adjust the second-moment decay coefficient $\beta _ { 2 }$ following Marek et al. (2025):

$$
\beta _ { 2 } \gets \beta _ { 2 } ^ { k } .
$$

CBZ estimation. For each dataset, we sweep the batch size under this scaling rule and record the final evaluation loss of each run. We estimate the CBZ discussed in Section 6.3 as the largest batch size whose final evaluation loss stays within 0.3% of the best loss over the sweep.

EBZ estimation. For each batch size B, we estimate the critical LR ratio $q = \widehat { \eta } _ { \star } / \eta$ defined in Section 6.3, averaged over checkpoints. Runs whose local simulations stay bounded over the whole LR-factor grid are recorded as censored at the top of the grid. We then fit the averaged ratio as a function of B. The EBZ of Section 6.3 is the smallest B at which the fitted curve reaches 1.1.

Figure 4a compares EBZ to CBZ. Across the five datasets, EBZ and CBZ are strongly positively correlated (Pearson correlation 0.95). EBZ is below CBZ for C4 and FineWeb and above it for OpenWebMath, DCLM, and OpenWebText, so neither threshold is uniformly smaller.

## B.4 MULTISTEP LR DROP ABLATIONS

Abrupt LR drops. We evaluate multistep schedule variants in Figure 11. Before the drop, all six runs are at EoS $( 0 . 9 \leq q \leq 1 . 1 )$ at most checkpoints, 8 of the 9 checkpoints before the drop in panels a–c and 12 to 13 of the 14 in panels d–f. After the drop, the transition moves to larger LR factors, and EoS returns only when the LR after the drop stays large. With the peak LR fixed at 0.002, no checkpoint after the drop is at EoS for any of the drop ratios 0.6, 0.45, and 0.3 (panels a–c). The median q after the drop is 1.3 for the ratio 0.6 and 1.4 for 0.45. For the ratio 0.3, no LR factor in the grid explodes at 35 of the 41 checkpoints after the drop. With the drop ratio fixed at 0.7 (panels d–f), EoS returns at 26 of the 36 checkpoints after the drop for the peak LR 0.008, at 7 for 0.004, and at none for 0.0005.

## B.5 LINEAR LR DECAY ABLATIONS

We evaluate different linear-decay configurations in Figure 12. As in the multistep ablations, the decay ratio affects whether and when EoS re-emerges during the decay phase. The transition time also depends on the decay length. Longer decay schedules give a slower transition, with the transition time roughly proportional to the number of decay steps in our settings.

Figure 12 shows how EoC changes during LR decay. As the LR decays, $- \widehat { \lambda } _ { \mathrm { m i n } }$ falls from about 850 in the stable phase to between 56 and 540 once the LR is below half its peak, and the saddleness falls from about 0.09 to 0.03 or less. EoC therefore weakens in the small-LR regime of these runs.

## B.6 EXPERIMENTS ACROSS MODEL SCALES

We train GPT-small, GPT-medium, and GPT-large (Table 1) on FineWeb for 20 tokens per parameter with batch sizes 512, 1,024, and 1,536, AdamW $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$ , weight decay 0.1), and a WSD schedule with 4% warmup and a linear decay to zero over the last 20% of steps. For each size, we sweep eight peak LRs from $\mathrm { 3 . 1 2 5 \times 1 0 ^ { - 4 } }$ to 0.04, doubling at each step. We run EoS and EoC detection at ten checkpoints in the stable phase of each run, with compensated local simulations at LR factors from 0.6 to 1.4. For the runs with large peak LRs, where no LR factor from 0.6 stays bounded

![](images/5125d2c6e2dc0bc83ad0c72bac0cef235f77dfdce56c42ad6468887b805204a1.jpg)

![](images/ee881008ee256297fb06855b1a34edc40cac979dec1029cfccd5255abedc7064.jpg)

![](images/0bd92b6e7c7fafdc06e66f36682b6b960bb2685cb16c92dd030ee62c241e4f9d.jpg)

![](images/08db73aacf36adc8cdf290bfac44af2ef3aef9a5cc63de465801e43de0b3e7ba.jpg)

![](images/a1995b69f190173d80f3b0cccb413291bc17ebbc7f87af5026ebafcbbfafd02a.jpg)

![](images/8ec2947b85b4fad243e41a6f4f92cce82d80358cd2cb2fb9b831ffca9f8fa92f.jpg)

![](images/f8f8e1e86b53b2a0c7d85c5df0981057d9c59fb430c04227cfebb00e4ff786d1.jpg)  
(a) WSD, LR = 0.0005.

![](images/857e270a35451b6473049e8b058e3f7af28587475c2ca89755a1891122141042.jpg)  
(b) WSD, LR = 0.004.

![](images/1b0e1e0b862f383c0bdecba38fb99a20d2ca22732aaa4ca92aa0a3e4c3a08334.jpg)  
(c) WSD, LR = 0.032.

Figure 8: EoS and EoC measurements under different LRs. We train GPT-extra-small (43M) on FineWeb-10B using AdamW with the WSD schedule and vary the LR. All runs use 400 linear warmup steps, 7600 stable steps, and 2000 linear decay steps. We identify 0.004 as the optimal peak LR via a separate sweep.

![](images/fbef4709fd6f48f4fc38bae2a3fa0daade1704f9b1fb3843c0a0d1dc626583a1.jpg)

![](images/c8b9872fd9863bf33b81d5c8289044da9ea5f27388b7261beeae4937c9994d82.jpg)

![](images/1341af6e6cf7f5a74bf411101f756b3d2a208ef999d97b4878c489194f31f568.jpg)

![](images/2c86ecd9e6ba8cf2620e64c87e73a121ce82c9fc341cf499061c1478bf7fdac6.jpg)

![](images/146a6faacc1e4b69b0fefd9917f7199a2f1a0c5feaef1f59b1e5787468c44aaf.jpg)

![](images/b46a94a778364a5998d6ab32c79fad5076e66bfd99d6f34832efd8c3006c91f2.jpg)

![](images/e9b98f91d19dbaad8b730497615c3d19d4c8cf1625e946c87a047d93a0666d6c.jpg)  
(a) WSD, batch size = 256.

![](images/119c24d15329580db2cfb91577b038fb9d2c91efe017aeac3f164872a66c7f87.jpg)  
(b) WSD, batch size = 512.

![](images/cd327ccde5e08e3a3dc62d5a331339cbfc76bb1cdddbfe492d1ae7f6943a225b.jpg)  
(c) WSD, batch size = 1024.  
Figure 9: EoS emerges only above a batch size threshold. We train GPT-extra-small (43M) on FineWeb using AdamW with the WSD schedule at global batch sizes 256, 512, and 1,024 with the same number of training tokens. Starting from LR 0.002, weight decay 0.1, and $\beta _ { 2 } = 0 . 9 5$ at batch size 512, we scale the LR, the weight decay, and $\beta _ { 2 }$ with the square-root rule of Appendix B.3. Each panel follows the reporting protocol of Figure 2. (a) At batch size 256, the transition is near factor 1.0 only at the first checkpoints, and from step 3,600 on no LR factor in the grid explodes. (b, c) At 512 and 1,024, the transition stays close to factor 1.0 for most of training.

![](images/b4f7aebfdc395db19071dce7c6fcc24a450fd6be0d14e377890a670b944ca71c.jpg)  
(a) No compensation $( \lambda _ { \mathsf { P } } = 0 )$

![](images/088db9eb42943e8aaad85d3578c754d1480383fc23c164fb83890f814ca96547.jpg)

(b) Estimated compensation (λ<sub>P</sub> ≈ 408).  
![](images/0477f130d1bf0ea65bbc68f15c02f9b4420962c312ca76b57307ff0024ca9261.jpg)  
(c) Compensation $\lambda _ { \mathsf { P } } = 6 0 0 .$

Figure 10: Effects of different compensation coefficients. We train GPT-extra-small (43M) with AdamW on FineWeb-10B, run EoS detection with different compensation coefficients λ , and report the loss at local step 25. (a) $\lambda _ { \mathsf { P } } = 0$ (no compensation). (b) $\lambda _ { \mathsf { P } } \approx 4 0 8$ (estimated). (c) $\lambda _ { \mathsf { P } } = 6 0 0$

![](images/e5a8ca76cb01e158581136b6b2d25ec9698dd93fc443685391c957156c7d4b18.jpg)

![](images/91660f104ffac1264b9bec780a474842e4e14d455ae023f1ac3d1fb4b8ed8479.jpg)

![](images/85fcdd18eb11456ada878322ca36527900ab2a80703d07fb5b8afa6dd49f8f47.jpg)

![](images/68ac1023c16d4a25658b0a41d805234d1c57a9ebf2c08ef0ad253ff5daa381eb.jpg)

![](images/35835eba9924222c92be9199b17abc3ac3add07e5fc2494a04e0e3d9804b7965.jpg)

![](images/675c0ab3f6861d390c2185900330d8665b7c44167dd2394de6db02d8e8b77da2.jpg)

![](images/10ec17f16eb9b1d7a10dc38ecec0c4cdb9b684f097c6f7f0ba3224ef2b7b5055.jpg)  
(a) Multistep, drop ratio = 0.6.

![](images/80ffe7f56df08069a673dbb2ac52c7d2e8e39de9366c99bd7659e93355164b16.jpg)  
(b) Multistep, drop ratio = 0.45.

![](images/1cc84f068830a11b0a9bd77b2614751c8d265511a38b79b79d379f382a37488e.jpg)  
(c) Multistep, drop ratio = 0.3.

![](images/6bae8febba6b5d6787f707fb05146ac957ce92a08c7aac0977fd65c99e2548e8.jpg)

![](images/81f283eac60f9e050bdea9d6990198c435543a4bf061b5fde7ca2f7378e72b60.jpg)

![](images/1022d1e21422bd7b154a079997a6feff1916db0f52aa936d2218a9bfa7f2adb6.jpg)

![](images/562ef78fc6b59dab7ac6ba6abc0a0e52c99fcb540a6c1a1648195dfcf7901224.jpg)

![](images/53b92b7efc9d135a1c6603472b426bdb0839f02a5db52739986ca2df1f8114a8.jpg)

![](images/d588f46b3e4b15fb315fcf2f8193a8ec698d2790507cce2168ffd8a192dadd37.jpg)

![](images/13e5ec2386d5b8fad8c296ce82fac3c3988b9f38e7bc2c1ad691e6270dbf4d5e.jpg)  
(d) Multistep, peak $\mathrm { L R } = 0 . 0 0 8 .$

![](images/b33fd2230d99d3f68346efa85b66c1385b4232f1980edfbeb5eea14d70066501.jpg)  
(e) Multistep, peak $\mathrm { L R } = 0 . 0 0 4 .$

![](images/10bac2aa11a666b9c6153963c7282252592ad2e7fb4bcc7d566d029bde10cc3a.jpg)  
(f) Multistep, peak $\mathrm { L R } = 0 . 0 0 0 5$

Figure 11: Ablations on the multistep LR drop. We train GPT-extra-small (43M) and vary the peak LR and the drop ratio of the multistep schedule. (a)–(c) The peak LR is fixed at 0.002 and the drop ratio varies over 0.6, 0.45, and 0.3, with the drop at step 4000. (d)–(f) The drop ratio is fixed at 0.7 and the peak LR varies over 0.008, 0.004, and 0.0005, with the drop at step 3000.

![](images/5dfa26c3394ced94f70d1d298553a1b5db6506c4100c3461928b4fbee93d004d.jpg)

![](images/c1ad35cc66bda895b32408d55ad19d57aeee2df25cb0d143c6de0553e3678252.jpg)

![](images/85bb03241fb26342db618fbf25feeb9eefba02ed43087730a215d48e2f09b9eb.jpg)

![](images/ed2202d8ce235e87a68cfab675bceecc1aa781c2d6f6b1700bb197f703e1de2e.jpg)

![](images/c3d88fbc03db45c1469f1374a749640401ceaf35bdd3b7ced967bb030eee478d.jpg)

![](images/d5cdec8d746b9fb165bce7210d53e73b1e08006e8fe1137f6c7efd348f5b21bc.jpg)

![](images/a4b2316dc27180054798fdae44c72097c9798e0b6900f2f807fa9c191ccd80ab.jpg)  
(a) Decay length = 3000.

![](images/b87022204d2408142f8731b52810eecff3ae3c8ac94a112a7a1fc69a877c8189.jpg)  
(b) Decay length = 5000.

![](images/38e2aac1ef030bdaef0af6d86d38f369805086f8f07ef242578c3b17788bac6d.jpg)  
(c) Decay length = 8000.

![](images/0516c0c5c0b45444d88028e4bb7599483b321169e11b937399d5ac82395ff9f9.jpg)

![](images/63b9498cd808421c2ae06aa9b3841a4b7f73bd5453c3d4c9825d9ce4ea2a9d3d.jpg)

![](images/7800fcf5c62475e11912d783faf493efc6b891c0c9482f61bb7ae81adfd7d949.jpg)

![](images/12c29d739fa808b7972d843f749a3b485196c17c1c6dc2c00ce78fa4cc813807.jpg)

![](images/5dd2f540e051ecf3d41ac7abf0e11c70f9269dda9bf66557c07d51b017e74329.jpg)

![](images/a3048f8357ba72d06aad4c81cdabcb573306090add4e7ef93ec5858b5d1257a0.jpg)

![](images/ae988e8cbb069d990ecc43e82b7f6a06436f42c7978fbda4f64b77d8c094c5b0.jpg)

(d) Decay ratio = 0.2.  
![](images/21476f270370613b13afeccb6289b3afb57e78ff6d78c8a23a9e466df7c7cf51.jpg)  
(e) Decay ratio = 0.4.

![](images/c6fe3334e66b40759d2e49eef829b47c00cda66e91241ca7fbb650db1db30838.jpg)  
(f) Decay ratio = 0.6.

Figure 12: Ablations for the linear decay phase. We train GPT-extra-small (43M) and vary the decay length and decay ratio of the linear decay phase. (a)–(c) The decay ratio is fixed at 0.1, with decay lengths of 3000, 5000, and 8000 steps. (d)–(f) The decay length is fixed at 6000 steps, with decay ratios of 0.2, 0.4, and 0.6.

![](images/2b8d38708fd69f018ed621c72c7c1009da420a1fe176cfd4b9108b5a155faa62.jpg)  
Figure 13: Training loss of GPT-large. We compare training loss versus steps at the empirically optimal peak LR (0.0025) and the largest tested peak LR (0.04). Both runs are trained for 20 tokens per parameter.

at many checkpoints, we re-estimate λ at LR factor 0.1 and add LR factors 0.2 to 0.5. Figures 14 to 16 show all 24 runs, and Figure 5 summarizes them. Figure 2 uses one additional run per size at peak LR $1 . 2 5 \times 1 0 ^ { - 3 }$ , with checkpoints saved throughout training.

At the three smallest peak LRs, 17, 18, and 19 of the 30 checkpoints of each size are at EoS $( 0 . 9 \leq q \leq 1 . 1 )$ , and 26, 27, and 26 satisfy the EoC criterion. From peak LR 0.01 on, only 12 of the 90 checkpoints are at EoS, and at 38 of them no LR factor in the grid stays bounded.

![](images/4e2985821d7aff0162367d039d5e19d7f07af3c64d8ed626e0f4204c2a1d0f06.jpg)  
Figure 14: GPT-small training stays near EoS up to peak LR 0.01 and goes beyond EoS at larger peak LRs. Up to peak LR 0.01, the critical LR factor $q = \widehat { \eta } _ { \star } / \eta$ , the smallest LR factor whose simulation explodes, lies between 0.9 and 1.2 at 51 of the 60 checkpoints, and 35 of these are at EoS $( 0 . 9 \leq q \leq 1 . 1 )$ . Over the same peak LRs, 44 of the 60 checkpoints satisfy the EoC criterion. For peak LRs from 0.02 on, the training LR exceeds the estimated critical LR $( q < 0 . 9 )$ at 19 of the 20 checkpoints, and at 5 of the 10 checkpoints of peak LR 0.04 no LR factor in the grid stays bounded. In these two runs, the rows below LR factor 0.6, the $- \lambda _ { \operatorname* { m i n } }$ curve, and the saddleness use the compensation coefficient λ estimated at LR factor 0.1, while the rows from 0.6 up use the one estimated at 0.6. Because the two coefficients differ, a bounded factor 0.6 sits above exploding lower factors at one checkpoint of peak LR 0.02. Each panel is one peak LR, each heatmap column is one of ten checkpoints in the stable phase, and the black line marks the largest LR factor that stays bounded. Saddleness above 0.7 runs off the top of the axis, and blank heatmap cells were not simulated. See Figure 2 for the panel format and Appendix B.6 for the setup.

![](images/e70565e41826695e232bd4a93496ad2d267e38a7f80b722af903f64a111f7f45.jpg)  
Figure 15: GPT-medium training stays near EoS up to peak $\mathbf { L R } 5 \times 1 0 ^ { - 3 }$ and goes beyond EoS at larger peak LRs. Up to peak LR $5 \times 1 0 ^ { - 3 }$ , the critical LR factor $q = \widehat { \eta } _ { \star } / \eta .$ , the smallest LR factor whose simulation explodes, lies between 0.9 and 1.2 at 43 of the 50 checkpoints, and 31 of these are at EoS $( 0 . 9 \leq q \leq 1 . 1 )$ . Over the same peak LRs, 38 of the 50 checkpoints satisfy the EoC criterion. For peak LRs from 0.01 on, the training LR exceeds the estimated critical LR $( q < 0 . 9 )$ at 26 of the 30 checkpoints, and at 7 of the 10 checkpoints of peak LR 0.04 no LR factor in the grid stays bounded. In these three runs, the rows below LR factor 0.6, the $- \lambda _ { \mathrm { m i n } }$ curve, and the saddleness use the compensation coefficient $\lambda _ { \mathsf { P } }$ estimated at LR factor 0.1, while the rows from 0.6 up use the one estimated at 0.6. Each panel is one peak LR, each heatmap column is one of ten checkpoints in the stable phase, and the black line marks the largest LR factor that stays bounded. Saddleness above 0.7 runs off the top of the axis, and blank heatmap cells were not simulated. See Figure 2 for the panel format and Appendix B.6 for the setup.

![](images/b08d93ba7d907ba1eb58cc8870b3672335f2e1ee8793596d88eb54e13500ab8c.jpg)  
Figure 16: GPT-large training stays near EoS up to peak LR $1 . 2 5 \times 1 0 ^ { - 3 }$ and goes beyond EoS at larger peak LRs. Up to peak $\mathrm { L R 1 . 2 5 \times 1 0 ^ { - 3 } }$ , the critical LR factor $q = \widehat { \eta } _ { \star } / \eta$ , the smallest LR factor whose simulation explodes, lies between 0.9 and 1.2 at 29 of the 30 checkpoints, and 19 of these are at EoS $( 0 . 9 \leq q \leq 1 . 1 )$ . Over the same peak LRs, 26 of the 30 checkpoints satisfy the EoC criterion. For peak LRs from $2 . 5 \times 1 0 ^ { - 3 }$ on, the training LR exceeds the estimated critical LR $( q < 0 . 9 )$ at 44 of the 50 checkpoints, and at every checkpoint of peak LR 0.04 no LR factor in the grid stays bounded. In these five runs, the rows below LR factor 0.6, the $- \lambda _ { \operatorname* { m i n } }$ curve, and the saddleness use the compensation coefficient $\lambda _ { \mathsf { P } }$ estimated at LR factor 0.1, while the rows from 0.6 up use the one estimated at 0.6. Because the two coefficients differ, a bounded factor 0.6 sits above exploding lower factors at two checkpoints of peak $\mathrm { L R 2 . 5 \times 1 0 ^ { - 3 } }$ and one of 0.01. Each panel is one peak LR, each heatmap column is one of ten checkpoints in the stable phase, and the black line marks the largest LR factor that stays bounded. Saddleness above 0.7 runs off the top of the axis, and blank heatmap cells were not simulated. See Figure 2 for the panel format and Appendix B.6 for the setup.

![](images/1c9b11ce6f717c05d3b1d56e7ee756d515e19de41efb31887e1060e09390a8ea.jpg)

![](images/6bf2a22707b129c63d344c96bd29b7bcbae878941c7dab3c8d1302741677a310.jpg)

![](images/641e5d66bd6f161d14df56ed9c8cdc4447a9d1e745c14b11f8ac5b7e0f77fbf3.jpg)

![](images/e16d506a503a40df4dd59cbad1e9a0da697ca81f5cbe7d18625dae9565a2ae2d.jpg)

![](images/1f0b9a4ad5fb578fed341b65eb542e32af434255aa47c0909d11534576b69caf.jpg)

![](images/ffa4c5587864c2a862133672e07b46ca1f7303ebf0608a5ec5088c3c2af26175.jpg)

![](images/6fdff0ce723aa72acfd0d6ad48b32c02d9a066cafd9e5b6577f1faab8dcfbb3f.jpg)

![](images/6ed17f8a8e3dbf6353798534a5238bab70f8671c340d5af57fb673a0e412f659.jpg)

![](images/c137a14c9b34c2f6ca072c5e418ec5f642bb30f3e5f5f7b77de779e84f3d6d9b.jpg)

![](images/2af9ede5fdce7d0242de7baf72545e4942255673bfdf1dbb53e540f9f898a941.jpg)

![](images/8941ab2592e463e2cdeb9096fbd82f6777c70b9c83c9db218cbd79a0e54fb213.jpg)  
Figure 17: Summary of a representative training run. We show training dynamics for GPT-extrasmall trained on FineWeb-10B using AdamW with LR 0.004, weight decay 0.1, and the WSD LR schedule.

## C THEORETICAL ANALYSIS

This appendix studies optimization dynamics on quadratic approximations of the loss around a fixed training checkpoint, with the preconditioner held fixed. We describe how the mean and second moment of the state (momentum and parameter displacement) evolve from step to step, and present stability thresholds and error bounds for the EoC and EoS detectors. Proofs are in Appendices D to F.

## C.1 SETTING AND THE LOCAL LINEAR SYSTEM

The local system. We consider adaptive gradient methods (AGMs), including Adam (Kingma & Ba, 2015) and AdamW (Loshchilov & Hutter, 2017), with the updates of (AGM), recalled here,

$$
\pmb { m } _ { t + 1 } = \beta _ { 1 } \cdot \pmb { m } _ { t } + ( 1 - \beta _ { 1 } ) \cdot \nabla \mathcal { L } _ { \mathcal { B } _ { t } } ( \pmb { w } _ { t } )\tag{28}
$$

$$
P _ { t + 1 } = \Psi ( P _ { t } , \nabla \mathcal { L } _ { \mathcal { B } _ { t } } ( { \pmb w } _ { t } ) , { \pmb w } _ { t } )\tag{29}
$$

$$
\pmb { w } _ { t + 1 } = \pmb { w } _ { t } - \eta \pmb { P } _ { t + 1 } ^ { - 1 } \pmb { m } _ { t + 1 } - \eta \lambda _ { \mathrm { W D } } \pmb { w } _ { t } ,\tag{30}
$$

where ${ m } _ { t } \in \mathbb { R } ^ { d }$ is the momentum and Ψ maps a positive definite preconditioner and two vectors in $\mathbb { R } ^ { d }$ to a positive definite preconditioner. This class includes SGD with and without momentum. We assume that the preconditioner varies slowly along the training trajectory and freeze it over the short window of a local simulation at a checkpoint ${ \pmb w } _ { t _ { 0 } } .$ writing $P : = P _ { t _ { 0 } }$ . This is an idealization. Adam’s second-moment decay coefficient $\beta _ { 2 }$ close to one motivates this approximation, but freezing $P _ { t }$ still requires its change over the simulation window to be small.

Recall the mini-batch quadratic surrogate (13) from Appendix A.1:

$$
\begin{array} { r l } & { \boldsymbol { Q } _ { \mathcal { B } _ { t } } ( \boldsymbol { w } ) : = \mathcal { L } _ { \mathcal { B } _ { t } } ( \boldsymbol { w } _ { t _ { 0 } } ) + ( \boldsymbol { w } - \boldsymbol { w } _ { t _ { 0 } } ) ^ { \top } \nabla \mathcal { L } _ { \mathcal { B } _ { t } } ( \boldsymbol { w } _ { t _ { 0 } } ) } \\ & { \phantom { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } + \frac { 1 } { 2 } ( \boldsymbol { w } - \boldsymbol { w } _ { t _ { 0 } } ) ^ { \top } \nabla ^ { 2 } \mathcal { L } _ { \mathcal { B } _ { t } } ( \boldsymbol { w } _ { t _ { 0 } } ) ( \boldsymbol { w } - \boldsymbol { w } _ { t _ { 0 } } ) , } \end{array}\tag{31}
$$

where $B _ { t }$ is the mini-batch sampled at local step t. SGD with LR η on this surrogate gives

$$
\pmb { w } _ { t _ { 0 } + t + 1 } = \pmb { w } _ { t _ { 0 } + t } - \eta \nabla Q _ { \mathcal { B } _ { t } } ( \pmb { w } _ { t _ { 0 } + t } ) .\tag{32}
$$

Write $\epsilon _ { t } : = w _ { t _ { 0 } + t } - w _ { t _ { 0 } }$ for the displacement after t local steps, as in the local AGM model of Appendix A.1. For the local simulation of an AGM, the momentum and parameter updates (28) and (30) are run on the surrogate (31), with the gradient $\nabla \mathcal { L } _ { B _ { t } } ( \pmb { w } _ { t } )$ replaced by $\nabla Q _ { B _ { t } } ( \pmb { w } _ { t _ { 0 } + t } ) =$ $g _ { B _ { t } } + H _ { B _ { t } } \epsilon _ { t }$ and the preconditioner fixed at P. Here $\pmb { g } _ { \boldsymbol { B } _ { t } } = \nabla \mathcal { L } _ { B _ { t } } ( \pmb { w } _ { t _ { 0 } } )$ and $\pmb { H } _ { B _ { t } } = \dot { \nabla } ^ { 2 } \mathcal { L } _ { B _ { t } } ( \pmb { w } _ { t _ { 0 } } )$ are the mini-batch gradient and Hessian at the checkpoint, as in Section 5. The local dynamics are

$$
\begin{array} { r l } & { \pmb { m } _ { t + 1 } = \beta _ { 1 } \pmb { m } _ { t } + ( 1 - \beta _ { 1 } ) ( \pmb { g } _ { \mathcal { B } _ { t } } + \pmb { H } _ { \mathcal { B } _ { t } } \pmb { \epsilon } _ { t } ) , } \\ & { \pmb { \epsilon } _ { t + 1 } = \pmb { \epsilon } _ { t } - \eta \pmb { P } ^ { - 1 } \pmb { m } _ { t + 1 } - \eta \lambda _ { \mathrm { W D } } ( \pmb { \epsilon } _ { t } + \pmb { w } _ { t _ { 0 } } ) . } \end{array}\tag{33}
$$

Compensation. Compensation adds a quadratic penalty to the local surrogate, measured in the norm of the frozen preconditioner. As in Appendix A.3, we write $G _ { t }$ for this loss:

$$
\begin{array} { r } { G _ { t } ( \epsilon ) : = \mathcal L _ { \mathcal B _ { t } } ( \pmb w _ { t _ { 0 } } ) + \epsilon ^ { \top } \nabla \mathcal L _ { \mathcal B _ { t } } ( \pmb w _ { t _ { 0 } } ) + \frac { 1 } { 2 } \epsilon ^ { \top } \nabla ^ { 2 } \mathcal L _ { \mathcal B _ { t } } ( \pmb w _ { t _ { 0 } } ) \epsilon + \frac { \lambda _ { \mathcal P } } { 2 } \| \epsilon \| _ { \pmb P } ^ { 2 } . } \end{array}
$$

Here $\| \epsilon \| _ { P } ^ { 2 } = \epsilon ^ { \top } P \epsilon$ , with $_ { P }$ fixed throughout the local simulation. For SGD, $P = I$ , so the penalty is $\frac { \lambda _ { \mathsf { P } } } { 2 } \| \epsilon \| ^ { 2 }$ . Its gradient is $\nabla G _ { t } ( \epsilon ) = g _ { \mathcal { B } _ { t } } + ( H _ { \mathcal { B } _ { t } } + \lambda _ { \mathsf { P } } \boldsymbol { P } ) \epsilon$ , so the penalty replaces the mini-batch Hessian $H _ { B _ { t } }$ by $H _ { B _ { t } } + \lambda _ { \mathsf { P } } P$ in (33), which in preconditioned coordinates is a shift of the curvature by λ<sub>P</sub>I (Proposition C.2). This is the same shift that the Euclidean penalty in Section 5.2 applies to the raw Hessian. The decoupled weight decay keeps its coefficient λ , and $\lambda _ { \mathsf { P } } = 0$ recovers the uncompensated surrogate.

Assumption C.1. The pairs $( H _ { B _ { t } } , g _ { B _ { t } } ) , t \ge 0$ , are i.i.d., independent of the initial state $( m _ { 0 } , \epsilon _ { 0 } )$ each $H _ { B _ { t } }$ is symmetric, and both have finite second moments. Their means are the population derivatives at the checkpoint,

$$
\mathbb { E } [ \pmb { g } _ { \pmb { \mathscr { B } } _ { t } } ] = \nabla \mathcal { L } ( \pmb { w } _ { t _ { 0 } } ) , \qquad \mathbb { E } [ \pmb { H } _ { \pmb { \mathscr { B } } _ { t } } ] = \nabla ^ { 2 } \mathcal { L } ( \pmb { w } _ { t _ { 0 } } ) .
$$

At each local step, we sample a fresh mini-batch independently from the same distribution. The expected batch Hessian and gradient equal their population counterparts at the fixed checkpoint. Together with the finite second moment assumption, this allows us to derive recursions for the mean and second moment of the local iterates.

The augmented linear system. Centering the displacement at a reference point and changing to preconditioned coordinates turns (33) into the stochastic linear system (15) of Appendix A.1. The following proposition gives its matrices in terms of the preconditioned mini-batch Hessian $D _ { B _ { t } } : = P ^ { - 1 / 2 } \bar { { \cal H } } _ { B _ { t } } \bar { P } ^ { - 1 / 2 }$ defined there.

Proposition C.2. Let $\pmb { x } _ { \star } \in \mathbb { R } ^ { d }$ be anyfixed vector. We express the local iterates relative to ${ \pmb w } _ { t _ { 0 } } + { \pmb x } ,$ and define

$$
z _ { t } = \left[ \begin{array} { c c } { \eta P ^ { - 1 / 2 } m _ { t } } \\ { P ^ { 1 / 2 } ( \epsilon _ { t } - x _ { \star } ) } \end{array} \right] , \qquad h _ { t } = ( 1 - \beta _ { 1 } ) \eta P ^ { - 1 / 2 } ( g _ { B _ { t } } + H _ { B _ { t } } x _ { \star } ) .
$$

Under the local dynamics (33), $z _ { t + 1 } = M _ { t } ( \eta ) z _ { t } + \pmb q _ { t }$ , where

$$
M _ { t } ( \eta ) = \left[ \begin{array} { c c } { \beta _ { 1 } I } & { ( 1 - \beta _ { 1 } ) \eta D _ { B _ { t } } } \\ { - \beta _ { 1 } I } & { ( 1 - \eta \lambda _ { \mathrm { W D } } ) I - ( 1 - \beta _ { 1 } ) \eta D _ { B _ { t } } } \end{array} \right] , \quad q _ { t } = \left[ - \hbar _ { t } - \eta \lambda _ { \mathrm { W D } } P ^ { 1 / 2 } ( { \bf x } _ { \star } + { \bf w } _ { t _ { 0 } } ) \right] .\tag{34}
$$

With compensation, the same holds with $D _ { B _ { t } }$ replaced by $D _ { B _ { t } } + \lambda _ { \mathsf { P } } I$ in $M _ { t } ( \eta )$ and $H _ { B _ { t } }$ replaced by $H _ { B _ { t } } + \lambda _ { \mathsf { P } } P$ in $h _ { t } .$ . The transition matrix becomes

$$
\widetilde { M } _ { t } ( \eta ) = \left[ \begin{array} { c c } { \beta _ { 1 } I } & { ( 1 - \beta _ { 1 } ) \eta ( D _ { \mathcal { B } _ { t } } + \lambda _ { \mathsf { P } } I ) } \\ { - \beta _ { 1 } I } & { ( 1 - \eta \lambda _ { \mathrm { W D } } ) I - ( 1 - \beta _ { 1 } ) \eta ( D _ { \mathcal { B } _ { t } } + \lambda _ { \mathsf { P } } I ) } \end{array} \right] .\tag{35}
$$

Under Assumption C.1, $( M _ { t } ( \eta ) , q _ { t } )$ is afunction of $B _ { t }$ alone and is independent of $z _ { t } .$

The choice of $\scriptstyle { \mathbf { { \vec { x } } } } _ { \star }$ changes the additive term $\pmb q _ { t }$ but leaves $M _ { t } ( \eta )$ unchanged. For two trajectories using the same batches, $\pmb q _ { t }$ cancels when we subtract their updates, so $M _ { t } ( \eta )$ determines how their difference evolves. Setting $\mathbf { \pmb { x } } _ { \star } = \mathbf { 0 }$ recovers the local system in (15). Compensation adds $\lambda _ { \mathsf { P } } I$ to the preconditioned batch Hessian in the momentum update, while the weight decay update remains unchanged. The proof is given in Appendix D.1.

Moment recursions and mean square stability. Proposition C.2 expresses the local AGM updates $\mathrm { a s } ~ z _ { t + 1 } = M _ { t } ( \eta ) z _ { t } + { q _ { t } }$ . We now use this representation to study whether $\mathbb { E } [ \| z _ { t } \| ^ { 2 } ]$ remains bounded. For a fixed $\operatorname { L R } \eta ,$ , we write $M _ { t } = M _ { t } ( \eta )$ . The following proposition gives the recursions for the mean and second moment, together with a sufficient condition for mean square stability.

Proposition C.3 (Moment recursion of a stochastic linear system). Let $z _ { t + 1 } = M _ { t } z _ { t } + q _ { t }$ , where $z _ { \mathrm { 0 } }$ hasfinite second moment and the pairs $( M _ { t } , q _ { t } ) , t \ge 0 ,$ , are $i . i . d .$ , have finite second moments, and are independent of $z _ { \mathrm { 0 } } .$ . Let $M : = \mathbb { E } [ M _ { t } ] , z _ { t } ^ { ( 1 ) } : = \mathbb { E } [ z _ { t } ]$ and $\boldsymbol { z } _ { t } ^ { ( 2 ) } : = \mathsf { v e c } ( \mathbb { E } [ \boldsymbol { z } _ { t } \boldsymbol { z } _ { t } ^ { \top } ] )$ , and set

$$
S : = \mathbb { E } [ \pmb { q } _ { t } \otimes \pmb { M } _ { t } + \pmb { M } _ { t } \otimes \pmb { q } _ { t } ] , \quad b _ { \sf q u a d } : = \mathsf { v e c } ( \mathbb { E } [ \pmb { q } _ { t } \pmb { q } _ { t } ^ { \top } ] ) .
$$

Then

$$
\left[ \boldsymbol { z } _ { t + 1 } ^ { ( 1 ) } \right] = \left[ \begin{array} { c c } { \boldsymbol { M } } & { \boldsymbol { 0 } } \\ { \boldsymbol { S } } & { \mathbb { E } [ \boldsymbol { M } _ { t } \otimes \boldsymbol { M } _ { t } ] } \end{array} \right] \left[ \begin{array} { c } { \boldsymbol { z } _ { t } ^ { ( 1 ) } } \\ { \boldsymbol { z } _ { t } ^ { ( 2 ) } } \end{array} \right] + \left[ \begin{array} { c } { \mathbb { E } [ \boldsymbol { q } _ { t } ] } \\ { \boldsymbol { b } _ { \mathrm { q u a d } } } \end{array} \right] ,\tag{36}
$$

and the spectral radius of the block matrix is max $\{ \rho ( M ) , \rho ( \mathbb { E } [ M _ { t } \otimes M _ { t } ] ) \} . \ I f \rho ( \mathbb { E } [ M _ { t } \otimes M _ { t } ] ) < 1$ both moments converge to finite limits, so $\mathrm { s u p } _ { t \geq 0 } \bar { \mathbb { E } } \| z _ { t } \| ^ { 2 } < \bar { \infty } .$ . The transition $M _ { t }$ and the forcing $\pmb q _ { t }$ may be correlated.

Under Assumption C.1, the augmented system of Proposition C.2 is of this form with $M _ { t } = M _ { t } ( \eta )$ and the compensated system with $M _ { t } = \widetilde { M } _ { t } ( \eta )$ . The two diagonal blocks govern the mean and the second moment. The second-moment transition splits as

$$
\mathbb { E } [ M _ { t } \otimes M _ { t } ] = M \otimes M + \mathbb { E } [ ( M _ { t } - M ) \otimes ( M _ { t } - M ) ] ,
$$

since $\otimes$ is bilinear and $\mathbb { E } [ M _ { t } - M ] = \mathbf { 0 }$ . For the augmented system this is the AGM analogue of the decomposition

$$
\mathbb { E } [ M _ { t } ^ { + } \otimes M _ { t } ^ { + } ] = \mathbb { E } [ M _ { t } ^ { + } ] \otimes \mathbb { E } [ M _ { t } ^ { + } ] + \operatorname { C o v } ( M _ { t } ^ { + } )
$$

in Section 5. The mean term is the full-batch transition, and the centered term records the variation of the mini-batch Hessian. The limiting moments of the forced system need not be zero. The proof is in Appendix D.2.

We call the homogeneous system strictly mean-square stable if $\mathbb { E } \Vert \boldsymbol { z } _ { t } \Vert ^ { 2 }  0$ for every deterministic $z _ { \mathrm { 0 } }$ . By the second-moment characterization lemma (Lemma F.14) this holds if and only if $\rho ( \mathbb { E } [ M _ { t } ( \eta ) ^ { } \otimes M _ { t } ( \eta ) ] ) < 1$ . For the same state space, strict homogeneous stability implies bounded moments for the affine system under Proposition C.3. Comparisons of their critical LRs must also project onto the same subspace, as we explain in Appendix C.2.

For SGD the displacement block of the augmented system evolves on its own, because the momentum is the current gradient. Applying Proposition C.3 to that block gives the moment recursion of Proposition 4.2 for the surrogate of a checkpoint. The stability statements follow in Appendix C.2. Corollary C.4 (Moment recursion for SGD). Let Assumption C.1 hold, let $\beta _ { 1 } = 0 , P = I$ and $\lambda _ { \mathrm { W D } } = 0 ,$ , and suppose there exists a w<sub>⋆</sub> such that

$$
\begin{array} { r } { \nabla \mathcal { L } ( \pmb { w } _ { t _ { 0 } } ) + \nabla ^ { 2 } \mathcal { L } ( \pmb { w } _ { t _ { 0 } } ) ( \pmb { w } _ { \star } - \pmb { w } _ { t _ { 0 } } ) = 0 . } \end{array}\tag{37}
$$

Under the SGD update (32), thefirst and second moments

$$
\begin{array} { r l } & { \epsilon _ { t } ^ { ( 1 ) } : = \mathbb { E } \bigl [ { \pmb w } _ { t _ { 0 } + t } - { \pmb w } _ { \star } \bigr ] , } \\ & { \epsilon _ { t } ^ { ( 2 ) } : = \mathrm { v e c } \Bigl ( \mathbb { E } [ ( { \pmb w } _ { t _ { 0 } + t } - { \pmb w } _ { \star } ) ( { \pmb w } _ { t _ { 0 } + t } - { \pmb w } _ { \star } ) ^ { \top } ] \Bigr ) } \end{array}
$$

satisfy

$$
\begin{array} { r } { \left[ \epsilon _ { t + 1 } ^ { ( 1 ) } \right] = \left[ \mathbb { E } [ M _ { t } ] \begin{array} { c c } { \mathbf { 0 } } \\ { S } & { \mathbb { E } [ M _ { t } \otimes M _ { t } ] } \end{array} \right] \left[ \epsilon _ { t } ^ { ( 1 ) } \right] + \left[ \mathbf { 0 } _ { \mathrm { q u a d } } \right] , } \end{array}\tag{38}
$$

where

$$
\begin{array} { r l r } { M _ { t } : = I - \eta H _ { B _ { t } } , } & { \quad } & { \quad r _ { t } : = g _ { \mathcal { B } _ { t } } + H _ { \mathcal { B } _ { t } } ( w _ { \star } - w _ { t _ { 0 } } ) , } \\ { S : = - \eta \mathbb { E } [ M _ { t } \otimes r _ { t } + r _ { t } \otimes M _ { t } ] , } & { \quad } & { \quad b _ { \mathrm { q u a d } } : = \eta ^ { 2 } \mathsf { v e c } \big ( \mathbb { E } [ r _ { t } r _ { t } ^ { \top } ] \big ) , } \end{array}
$$

and $\mathbb { E } [ r _ { t } ] = \mathbf { 0 }$

The corollary gives closed recursions for the mean and the second moment, driven by the constant terms $b _ { \mathrm { q u a d } }$ and S. It makes no stability claim by itself. Applying the strict spectral condition of Proposition C.3 gives the affine boundedness assertion in Proposition 4.2. The stationarity condition (37) removes the forcing from the mean recursion. It holds for some w<sub>⋆</sub> whenever $\nabla \mathcal { L } ( \boldsymbol { w } _ { t _ { 0 } } )$ lies in the range of $\nabla ^ { 2 } \mathcal { L } ( \pmb { w } _ { t _ { 0 } } )$ , in particular whenever the Hessian is invertible. The proof is in Appendix D.3.

## C.2 STABILITY OF THE LOCAL DYNAMICS

Negative effective curvature makes the local dynamics unstable at every positive LR. On the strictly positive subspace, the LRs for strict homogeneous stability form a single interval.

Negative curvature. The first instability mode does not depend on the LR. Write $D = \mathbb { E } [ D _ { B _ { t } } ]$ for the population preconditioned Hessian $( D _ { t _ { 0 } }$ in Appendix A.1).

Lemma C.5 (Negative effective curvature). Let Assumption C.1 hold. If the effective curvature $A = D + \lambda _ { \mathrm { W D } } I$ has a negative eigenvalue, then the homogeneous AGM dynamics $z _ { t + 1 } = M _ { t } ( \eta ) z _ { t }$ ofProposition C.2 are not mean-square stablefor any $\eta > 0$

Along an eigendirection of D with negative effective curvature, the mean dynamics reduce to a $2 \times 2$ system with a real eigenvalue above one for every $\eta > 0$ . Momentum changes the rate of growth, not its existence. A smaller LR slows the escape and does not prevent it. For SGD the rate is explicit. Write $\pmb { H } _ { t _ { 0 } } = \nabla ^ { 2 } \mathcal { L } ( \pmb { w } _ { t _ { 0 } } )$ for the population Hessian at the checkpoint.

Corollary C.6 (Negative-curvature divergence). Under the assumptions of Corollary C.4, suppose $H _ { t _ { 0 } }$ has a unit eigenvector u with eigenvalue $\lambda < 0$ . The population quadratic is unbounded below. For every $\eta > 0$

$$
\begin{array} { r } { \pmb { u } ^ { \top } \mathbb { E } [ \pmb { w } _ { t _ { 0 } + t } - \pmb { w } _ { \star } ] = ( 1 - \eta \lambda ) ^ { t } \pmb { u } ^ { \top } \mathbb { E } [ \pmb { w } _ { t _ { 0 } } - \pmb { w } _ { \star } ] . } \end{array}
$$

Thus the mean diverges whenever the initial mean has a nonzero component along u. The homogeneous system is not mean-square stable.

This is the negative-curvature failure of Section 4.1, now for the surrogate of a checkpoint. The proofs of the lemma and the corollary are in Appendix D.4.

Critical LR. To isolate the instability induced by the LR, we restrict the dynamics to the subspace of positive effective curvature, as in Definition $\mathrm { A } . 2 . \mathrm { L e t } \Pi ^ { + }$ be the orthogonal projection onto the strictly positive eigenspace of the symmetric effective mean $D + \lambda _ { \mathrm { W D } } \bar { I } .$ . All projected matrices below act on $\mathrm { r a n g e } ( \mathrm { H } ^ { + } )$ . Both the momentum and the displacement are restricted to this subspace, which we assume is nonempty. Directions of zero effective curvature are dropped because their mean transition has an eigenvalue equal to one, so no LR would be strictly mean-square stable if they were kept. Let $M _ { t } ^ { + } ( \eta )$ denote the restricted transition and define

$$
\begin{array} { r } { K ^ { + } ( \eta ) : = \mathbb { E } \left[ M _ { t } ^ { + } ( \eta ) \otimes M _ { t } ^ { + } ( \eta ) \right] . } \end{array}\tag{39}
$$

The strictly mean-square stable LR set of the projected homogeneous system is

$$
S _ { \mathrm { A G M } } ( w _ { t _ { 0 } } ) : = \{ \eta > 0 : \rho ( K ^ { + } ( \eta ) ) < 1 \} .\tag{40}
$$

On the range of $\Pi ^ { + }$ , the effective mean is positive definite by construction. The next theorem shows that under Assumption C.1 this set is the interval $( 0 , \eta _ { \star } )$ for a unique finite $\eta _ { \star } ,$ even when $\beta _ { 1 } > 0$ Momentum makes the transition nonsymmetric but does not create a second stable interval.

Theorem C.7 (Unique stability threshold, restated from Corollary F.4). Let Assumption C.1 hold. Fix the preconditioner $\begin{array} { r } { \dot { \pmb { P } } \succ 0 , 0 \leq \beta _ { 1 } < 1 , \lambda _ { \mathrm { W D } } \geq 0 , } \end{array}$ , and the batch distribution. Suppose $\Pi ^ { + } \neq 0$ and

$$
u ^ { \top } { \big ( } P ^ { - 1 / 2 } H _ { t _ { 0 } } P ^ { - 1 / 2 } + \lambda _ { \mathrm { W D } } I { \big ) } u > 0 \qquad { \mathrm { ~ f ~ o r ~ } } e \nu e r y \ u \in \mathrm { r a n g e } ( \Pi ^ { + } ) \setminus \{ 0 \} .
$$

Then there exists a unique $\eta _ { \star } \in ( 0 , \infty )$ such that

$$
\rho \big ( \mathbf { { K } } ^ { + } ( \eta ) \big ) \left\{ \begin{array} { l l } { < 1 , } & { 0 < \eta < \eta _ { \star } , } \\ { = 1 , } & { \eta = \eta _ { \star } , } \\ { > 1 , } & { \eta > \eta _ { \star } . } \end{array} \right.
$$

Consequently, the projected homogeneous system $z _ { t + 1 } ^ { + } = M _ { t } ^ { + } ( \eta ) z _ { t } ^ { + }$ satisfies

lim $\mathbb { E } \big [ \| z _ { t } ^ { + } \| ^ { 2 } \big ] = 0$ for every deterministic initial state $z _ { 0 } ^ { + }$ t→∞

if and only $i f 0 < \eta < \eta _ { \star }$ ⋆

By Proposition A.3, this threshold equals the AGM critical LR defined by bounded affine moments in Definition A.2. Appendix C.4 bounds the finite-horizon error of the expected detector. The same statement holds for the compensated system of Proposition $\mathrm { C } . 2$ whenever $D + ( \lambda _ { \mathrm { W D } } + \lambda _ { \mathsf { P } } ) \pmb { I } \succ 0$ since it is the same system with a shifted effective mean. The exact stability criterion is stated in Appendix F.1. The proof of the AGM threshold statement is in Appendix F.5.3.

Above the critical LR, the second moment of some initial state grows without bound.

Corollary C.8 (Divergence above the critical LR). Under the assumptions of Theorem $C . 7 ,$ for $e v e r y \eta > \eta ,$ <sub>⋆</sub> there is a deterministic initial state $z _ { \mathrm { 0 } }$ ofthe projected homogeneous systemfor which $\mathbb { E } \| z _ { t } \| ^ { 2 } \to + \infty$

The proof is in Appendix D.5. It uses only that the second-moment spectral radius exceeds one for $\eta > \eta _ { \star }$

For SGD, $\beta _ { 1 } = 0 , P = I$ and $\lambda _ { \mathrm { W D } } = 0 .$ , the projected system reduces to the displacement restricted to $\mathrm { r a n g e } ( \Pi _ { t _ { 0 } } ^ { + } )$ , where $\Pi _ { t _ { 0 } } ^ { + }$ is the spectral projector of $H _ { t _ { 0 } }$ onto its strictly positive eigenspace,

$$
\begin{array} { r l r } & { } & { M _ { t } ^ { + } ( \eta ) = I - \eta \Pi _ { t _ { 0 } } ^ { + } H _ { \mathcal { B } _ { t } } \Pi _ { t _ { 0 } } ^ { + } , \qquad \pmb { x } _ { t + 1 } = M _ { t } ^ { + } ( \eta ) \pmb { x } _ { t } , } \\ & { } & { \eta _ { \star } = \operatorname* { s u p } \{ \eta > 0 : \rho ( \mathbb { E } [ M _ { t } ^ { + } ( \eta ) \otimes M _ { t } ^ { + } ( \eta ) ] ) < 1 \} . } \end{array}
$$

In words, $\eta _ { \star }$ is the largest LR below which every perturbation in the convex subspace decays in mean square, and Corollary C.8 says that above it some perturbation grows without bound. This is the positive-curvature failure of Section 4. The stability notion in Definition A.2 is the bounded affine notion of the main text. We project after every update, so a batch Hessian cannot move mass from the discarded directions back into the range of $\Pi _ { t _ { 0 } } ^ { + }$ . This is the projection of Definition 4.3, so for SGD the two critical LRs coincide. Proposition $\mathrm { A } . 3$ justifies computing this threshold from strict homogeneous stability. Below the threshold, strict decay implies bounded affine moments. Above it, differences of affine trajectories driven by the same batches rule out boundedness for every start. The proof is in Appendix D.6. The two stable sets can differ at the endpoint, but their suprema agree.

Zero-curvature directions must be dropped. For example, take SGD with $\pmb { H } _ { B _ { t } } = \mathrm { d i a g } ( 1 , 0 )$ and residual gradient $\pmb { r } _ { t } = ( 0 , \xi _ { t } ) ^ { \top }$ , where the $\xi _ { t }$ are independent uniform signs. From zero initial error, the second coordinate satisfies

$$
\varepsilon _ { t , 2 } = - \eta \sum _ { s = 0 } ^ { t - 1 } \xi _ { s } , \qquad \mathbb { E } [ \varepsilon _ { t , 2 } ^ { 2 } ] = t \eta ^ { 2 } .
$$

If the zero-curvature coordinate were kept, no positive LR would give bounded second moments. The projection onto the positive eigenspace keeps only the first coordinate, and its critical LR is 2.

Mean-square and almost-sure stability. Consider the homogeneous recursion $\boldsymbol { \epsilon } _ { t + 1 } = \boldsymbol { M } _ { t } \boldsymbol { \epsilon } _ { t }$ with i.i.d. $M _ { t }$ of finite second moment. Recall the strict notion of Appendix C.1. The recursion is strictly mean-square stable if $\mathbb { E } [ \lVert \epsilon _ { t } \rVert ^ { 2 } ]  0$ for every deterministic initial state, and this holds if and only if the second-moment map below has spectral radius less than one. The main-text definition (Definition 4.1) asks only for a bounded second moment. The AGM critical LR in Definition A.2 uses this same notion. Proposition ${ \mathrm { A } } . 3$ shows that its threshold agrees with the strict homogeneous threshold on the same positive-mean subspace. Boundedness at the endpoint may depend on affine forcing. The spectral analysis below and in Appendix F uses strict homogeneous stability. Exponential mean-square decay requires constants $K < \infty$ and $0 < \varrho < 1$ such that

$$
\begin{array} { r } { \mathbb { E } [ \| \epsilon _ { t } \| ^ { 2 } ] \leq K \varrho ^ { t } \| \epsilon _ { 0 } \| ^ { 2 } . } \end{array}
$$

Since $M _ { t }$ is independent of $\epsilon _ { t } .$ the second moment $\mathbb { E } [ \epsilon _ { t } \epsilon _ { t } ^ { \top } ]$ evolves as $\mathbb { E } [ \\\\\\\\\\\epsilon _ { t + 1 } \ \epsilon _ { t + 1 } ^ { \top } ] = \mathcal { T } ( \mathbb { E } [ \ \ \epsilon _ { t } \epsilon _ { t } ^ { \top } ] )$ where

$$
\mathcal { T } ( X ) = \mathbb { E } [ M _ { t } X M _ { t } ^ { \top } ] .
$$

Its matrix representation is the $\mathbb { E } [ M _ { t } \otimes M _ { t } ]$ of Proposition C.3, and strict mean-square stability is the spectral condition $\rho ( \mathcal T ) < \bar { 1 }$ . This characterization makes mean-square stability useful for analyzing SGD (Wu et al., 2018; Ma & Ying, 2021; Wu et al., 2022; Mulayoff & Michaeli, 2024). Almost-sure stability requires $\| \epsilon _ { t } \|  0$ with probability one. Its exponential version is governed by the top Lyapunov exponent of the random matrix product,

$$
\operatorname * { l i m } _ { t \to \infty } \operatorname * { s u p } _ { t } { \frac { 1 } { t } } \log \left\| M _ { t - 1 } \cdot \cdot \cdot M _ { 0 } \right\| < 0 \qquad \mathrm { a l m o s t ~ s u r e l y } ,
$$

under the usual integrability conditions (Furstenberg & Kesten, 1960; Bougerol et al., 2012).

The two notions are not equivalent. Exponential mean-square decay implies almost-sure exponential stability by Markov’s inequality and the Borel–Cantelli lemma. The converse is false because rare expanding trajectories can dominate the second moment without changing the behavior of a typical sample path. For example, consider the scalar recursion $\varepsilon _ { t + 1 } = a _ { t } \varepsilon _ { t }$ , where the $a _ { t }$ are i.i.d. and

$$
a _ { t } = { \left\{ \begin{array} { l l } { 1 / 2 , } & { { \mathrm { w i t h ~ p r o b a b i l i t y ~ } } 0 . 9 9 , } \\ { 1 0 , } & { { \mathrm { w i t h ~ p r o b a b i l i t y ~ } } 0 . 0 1 . } \end{array} \right. }
$$

Here

$$
\begin{array} { r l } & { \mathbb { E } [ \log a _ { t } ] = 0 . 9 9 \log \frac { 1 } { 2 } + 0 . 0 1 \log 1 0 \approx - 0 . 6 6 < 0 , } \\ & { \quad \mathbb { E } [ a _ { t } ^ { 2 } ] = 0 . 9 9 \cdot \frac { 1 } { 4 } + 0 . 0 1 \cdot 1 0 0 = 1 . 2 4 7 5 > 1 . } \end{array}
$$

Since log $\begin{array} { r } { | \varepsilon _ { t } | = \log | \varepsilon _ { 0 } | + \sum _ { s = 0 } ^ { t - 1 } \log a _ { s } , } \end{array}$ the strong law of large numbers gives $\begin{array} { r } { \frac { 1 } { t } \log | \varepsilon _ { t } | \  \ } \end{array}$ $\mathbb { E } [ \log a _ { t } ] ~ < ~ 0$ almost surely, so $| \varepsilon _ { t } |  0$ almost surely, whereas $\mathbb { E } [ | \varepsilon _ { t } | ^ { 2 } ] = \mathbb { E } [ a _ { t } ^ { 2 } ] ^ { \acute { t } } | \varepsilon _ { 0 } | ^ { 2 }$ by independence, so $\mathbb { E } [ | \varepsilon _ { t } | ^ { 2 } ]$ grows exponentially. The second moment can therefore reveal growth that an almost-sure criterion does not detect. We use mean-square stability because it controls the expected perturbation energy and admits a closed second-moment recursion. The experiments use a single-trajectory diagnostic, whose threshold need not coincide with this population threshold.

## C.3 STOCHASTIC POWER ITERATION FOR EOC

We state the accuracy of the stochastic power iteration of Appendix A.2. The theorem separates the error due to the finite horizon, the curvature noise, and the bisection resolution.

The system. We use the effective curvature of Appendix C.2, writing the population preconditioned Hessian as $D _ { t _ { 0 } }$ , the D of that section:

$$
\begin{array} { r l } & { \begin{array} { c } { A : = D _ { t _ { 0 } } + \lambda _ { \mathrm { W D } } I , } \\ { \Delta _ { t } : = A _ { \mathcal { B } _ { t } } - A . } \end{array} } \end{array} \qquad \begin{array} { c }  \begin{array} { c } { A _ { \mathcal { B } _ { t } } : = P ^ { - 1 / 2 } H _ { \mathcal { B } _ { t } } P ^ { - 1 / 2 } + \lambda _ { \mathrm { W D } } I , } \\ { \Delta _ { t } : = A _ { \mathcal { B } _ { t } } - A . } \end{array} \end{array}
$$

Thus the deterministic weight-decay term is included in both curvatures and cancels from the noise $\Delta _ { t }$ . For SGD without weight decay, $\boldsymbol { A } = \boldsymbol { H } _ { t _ { 0 } } ,$ so the same procedure estimates the minimum eigenvalue of the raw population Hessian. Let $\mathbf { \delta } \mathbf { u } _ { \mathrm { m i n } }$ be a unit eigenvector of A for its smallest eigenvalue. The shift to estimate is

$$
\lambda _ { \star } : = \operatorname* { m a x } \{ 0 , - \lambda _ { \operatorname* { m i n } } ( A ) \} ,\tag{41}
$$

the smallest nonnegative shift making $A + \lambda I$ positive semidefinite.

A query at shift $\lambda \in [ 0 , \Lambda ]$ starts from the same fixed unit vector $\scriptstyle { \pmb x } _ { 0 }$ and uses the inner stepsize $\eta _ { \mathrm { p } }$ for $T _ { \mathrm { p } }$ steps:

$$
\begin{array} { r l } & { \mathbf { \boldsymbol { x } } _ { t + 1 } = \left[ \mathbf { \boldsymbol { I } } - \eta _ { \mathrm { p } } ( \mathbf { \boldsymbol { A } } _ { B _ { t } } + \lambda \boldsymbol { I } ) \right] \mathbf { \boldsymbol { x } } _ { t } } \\ & { \qquad = \left[ M ( \lambda ) - \eta _ { \mathrm { p } } \mathbf { \boldsymbol { \Delta } } _ { t } \right] \mathbf { \boldsymbol { x } } _ { t } , \qquad t = 0 , \ldots , T _ { \mathrm { p } } - 1 , } \\ & { M ( \lambda ) : = \mathbf { \boldsymbol { I } } - \eta _ { \mathrm { p } } ( \mathbf { \boldsymbol { A } } + \lambda \boldsymbol { I } ) . } \end{array}\tag{42}
$$

Here $\eta _ { \mathrm { p } }$ is the power-iteration stepsize, distinct from the training LR $\eta ,$ and $T _ { \mathrm { p } }$ is its horizon, distinct from the local-simulation horizon T. A query explodes if $\operatorname* { m a x } _ { 0 \leq t \leq T _ { \mathrm { p } } } \left\| { \pmb x } _ { t } \right\| > B$ , where $B > 1$ is the cutoff. Starting from $[ 0 , \Lambda ] , \mathsf { A }$ lgorithm 1 queries the midpoint of the current search interval. An exploding query replaces the lower endpoint and a non-exploding query replaces the upper endpoint. After N queries the output $\widehat { \lambda }$ is the upper endpoint, and $\widehat { \lambda } _ { \mathrm { m i n } } = - \widehat { \lambda }$

Within each query, let $\mathcal { F } _ { t }$ contain all observations of earlier queries and the current observations $\Delta _ { 0 } , \ldots , \Delta _ { t - 1 }$ . The queried shift and $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ are therefore $\mathcal { F } _ { t }$ -measurable. All vector norms below are Euclidean and matrix norms are spectral.

Assumption C.9 (Bounded curvature noise). For each query and each step $t ,$ the symmetric noise $\Delta _ { t }$ is $\mathcal { F } _ { t + 1 }$ -measurable and satisfies

$$
\begin{array} { r } { \mathbb { E } \big [ \Delta _ { t } ~ | ~ \mathcal { F } _ { t } \big ] = \mathbf { 0 } , \qquad \big \| \Delta _ { t } \big \| \leq \sigma \quad \mathrm { a l m o s t ~ s u r e l y } , } \end{array}\tag{43}
$$

for a constant $\sigma > 0$

Fresh independent mini-batches, as in Appendix C.1, give the conditional mean-zero property. The almost-sure norm bound is an additional assumption used by the concentration inequalities in the proof.

Assumption C.10 (Search interval and initial overlap). The population curvature satisfies $\lambda _ { \operatorname* { m i n } } ( A ) <$ 0 and $\lambda _ { \star } \leq \Lambda$ . The fixed initialization satisfies

$$
\| \pmb { x } _ { 0 } \| = 1 , \qquad | \pmb { u } _ { \mathrm { m i n } } ^ { \top } \pmb { x } _ { 0 } | \geq \varepsilon , \qquad 0 < \varepsilon \leq 1 .
$$

The search interval must contain the target. The overlap supplies a nonzero initial component in the direction that grows below $\lambda _ { \star }$ . Initializing with the checkpoint momentum does not by itself ensure this condition.

Assumption C.11 (Power-iteration stepsize). The inner stepsize satisfies

$$
\eta _ { \mathrm { p } } > 0 , \qquad \eta _ { \mathrm { p } } \bigl ( \lambda _ { \mathrm { m a x } } ( A ) + \Lambda \bigr ) \leq 1 .\tag{44}
$$

This makes the mean transition $M ( \lambda )$ positive semidefinite for every queried shift. Above $\lambda _ { \star }$ its norm is $1 - \eta _ { \mathrm { p } } ( \lambda - \lambda _ { \star } )$ , which controls the probability of a false explosion. The explosion argument below $\lambda _ { \star }$ does not need this stepsize condition.

The following result applies to any fixed finite horizon. It imposes no upper bound on that horizon. Theorem C.12 (Accuracy of the stochastic power iteration). Let Assumptions C.9 to C.11 hold. Fix a cutoff $B > 1$ , positive integers N and $T _ { \mathrm { p } } ,$ , and $0 < \delta < 1$ . Define

$$
g _ { - } : = \frac { 4 \eta _ { \mathrm { p } } \sigma ^ { 2 } B ^ { 2 } } { \varepsilon ^ { 2 } } \log \frac { N } { \delta } , \qquad g _ { + } : = 2 \eta _ { \mathrm { p } } \sigma ^ { 2 } \operatorname * { m a x } \left\{ 1 , \frac { \log ( N / \delta ) } { \log B } \right\} ,
$$

$$
e _ { \mathrm { p } } : = \frac { ( 2 B / \varepsilon ) ^ { 1 / T _ { \mathrm { p } } } - 1 } { \eta _ { \mathrm { p } } } .\tag{45}
$$

Then the bisection output satisfies, with probability at least $1 - \delta ,$

$$
\lambda _ { \star } - \operatorname* { m a x } \{ g _ { - } , e _ { \mathrm { p } } \} \leq \widehat { \lambda } \leq \lambda _ { \star } + g _ { + } + \frac { \Lambda } { 2 ^ { N } } .\tag{46}
$$

The finite-horizon term $e _ { \mathrm { p } }$ obeys $e _ { \mathrm { p } } \leq g _ { - } i f$ and only if

$$
T _ { \mathrm { p } } \geq \left\lceil \frac { \log ( 2 B / \varepsilon ) } { \log ( 1 + \eta _ { \mathrm { p } } g _ { - } ) } \right\rceil .\tag{47}
$$

Interpretation of the bound. The lower error is the larger of the noise term $g _ { - }$ and the finitehorizon term $e _ { \mathrm { p } }$ . For fixed $B , \varepsilon ,$ , and $\eta _ { \mathrm { p } }$

$$
e _ { \mathrm { p } } = { \frac { \log ( 2 B / \varepsilon ) } { \eta _ { \mathrm { p } } T _ { \mathrm { p } } } } + O ( T _ { \mathrm { p } } ^ { - 2 } ) \qquad { \mathrm { a s ~ } } T _ { \mathrm { p } } \to \infty .
$$

This is the margin needed for the initial component of size ε to grow past B within the allotted steps. Above $\lambda _ { \star }$ , the error consists of the noise term $g _ { + }$ and the bisection resolution $\Lambda { 2 ^ { - N } }$ . The displayed noise terms do not decrease with the horizon when the stepsize, cutoff, overlap, confidence, and number of queries are fixed. A longer horizon therefore reduces the finite-horizon term but does not make this bound vanish at a fixed stepsize. The theorem assumes the search-interval condition and the overlap and does not verify them. It also assumes that the population minimum is negative. It bounds the estimation error under that assumption and does not certify the sign of the population minimum from the sign of the returned estimate. Its probability guarantee is for each fixed choice of $T _ { \mathrm { p } }$ . The proof is in $\mathbf { A } _ { \mathbf { l } }$ ppendix E.

## C.4 COMPENSATED LOCAL SIMULATION FOR EOS

Our goal is to estimate the critical LR $\eta _ { \star }$ used to define EoS. The detector runs a compensated local model for only $T$ steps. Compensation changes the system being tested, whose critical LR we denote by $\eta _ { \star } ^ { \mathsf { P } }$ , and a finite simulation can accept LRs above this threshold. Theorem C.16 bounds the resulting error of the expected detector by accounting for these two effects. We first specify the systems and the detector, then introduce the quantities needed to bound each error.

Curvature at the checkpoint. We continue with the checkpoint, frozen preconditioner and local model of Appendix C.1. Recall the effective curvature and its mini-batch version from Appendix C.3,

$$
\begin{array} { r } { A : = D _ { t _ { 0 } } + \lambda _ { \mathrm { W D } } I , \qquad A _ { B _ { t } } : = P ^ { - 1 / 2 } H _ { B _ { t } } P ^ { - 1 / 2 } + \lambda _ { \mathrm { W D } } I , \qquad D _ { t _ { 0 } } : = P ^ { - 1 / 2 } H _ { t _ { 0 } } P ^ { - 1 / 2 } , } \end{array}\tag{48}
$$

and $\Delta _ { t } : = A _ { B _ { t } } - A$ is the deviation of the mini-batch effective curvature from its mean. It is symmetric with E $\ \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \Delta \mathbf { \Psi } = 0$ . Let $\Pi ^ { + }$ be the orthogonal projector onto the strictly positive eigenspace of A, of dimension $p \geq 1$ , and choose orthogonal coordinates adapted to it, so that

$$
{ \cal A } = \mathrm { d i a g } ( A ^ { + } , A ^ { - } ) , \qquad A ^ { + } \succ 0 , \qquad A ^ { - } \preceq 0 ,
$$

where $A ^ { + }$ is the $p \times p$ block on the positive eigenspace and $A ^ { - }$ the block on the discarded directions, absent when $p = d .$ For a symmetric d × d matrix M written in these coordinates, $M ^ { + }$ denotes its $p \times p$ block on the positive eigenspace and $M ^ { - }$ its discarded block, so that $\Pi ^ { + } M \Pi ^ { + } = \mathrm { d i a g } ( M ^ { + } , 0 )$ is $M ^ { + }$ padded with zeros. In particular $\Delta _ { t } ^ { + }$ is the block of the noise on the positive eigenspace, of the same size as $A ^ { + }$ , and $\Delta _ { t } ^ { - \perp } : = \Delta _ { t } - \dot { \Pi } ^ { + } \Delta _ { t } \Pi ^ { + }$ collects the discarded and cross blocks. We use $L : = \lambda _ { \operatorname* { m a x } } ( A ) = \lambda _ { \operatorname* { m a x } } ( A ^ { \dagger } )$ throughout. Thus the compensated mean has largest eigenvalue $L + \lambda _ { \mathsf { P } }$ . The compensation shifts the effective curvature to $\begin{array} { r } { \bar { A _ { \sf P } } : = \pmb { A } + \lambda _ { \sf P } \pmb { I } . } \end{array}$

The compensated local simulation. The detector runs the local AGM update of Appendix C.1 on the compensated loss $G _ { t }$ . We retain $\epsilon _ { t }$ for the displacement from the checkpoint, so the simulated parameter is $\boldsymbol { w } _ { t _ { 0 } } + \boldsymbol { \epsilon } _ { t }$ . A run at LR $\eta > 0$ from a prescribed displacement $\epsilon _ { \mathrm { 0 } }$ and the checkpoint momentum $m _ { 0 } = m _ { t _ { 0 } }$ follows

$$
\begin{array} { r } { \pmb { m } _ { t + 1 } = \beta _ { 1 } \pmb { m } _ { t } + ( 1 - \beta _ { 1 } ) \big [ \pmb { g } _ { \mathcal { B } _ { t } } + ( \pmb { H } _ { \mathcal { B } _ { t } } + \lambda _ { \mathrm { P } } \pmb { P } ) \pmb { \epsilon } _ { t } \big ] , } \\ { \pmb { \epsilon } _ { t + 1 } = ( 1 - \eta \lambda _ { \mathrm { W D } } ) \pmb { \epsilon } _ { t } - \eta \pmb { P } ^ { - 1 } \pmb { m } _ { t + 1 } - \eta \lambda _ { \mathrm { W D } } \pmb { w } _ { t _ { 0 } } . } \end{array}\tag{49}
$$

The two critical LRs. The homogeneous differences of two runs driven by the same mini-batches satisfy a momentum recurrence. In preconditioned position coordinates this recurrence is

$$
\begin{array} { r } { x _ { t + 1 } = \big [ ( 1 + b ) \pmb { I } - \alpha ( \pmb { C } + \Delta _ { t } ) \big ] x _ { t } - b x _ { t - 1 } , \qquad b : = \beta _ { 1 } ( 1 - \eta \lambda _ { \mathrm { W D } } ) , \quad \alpha : = \eta ( 1 - \beta _ { 1 } ) . } \end{array}
$$

For the compensated system, the mean and noise are $( C , \Delta _ { t } ) = \left( A _ { \mathsf { P } } , \Delta _ { t } \right)$ . For the projected system, they are $( A ^ { + } , \Delta _ { t } ^ { + } )$ , on the p-dimensional positive eigenspace. The derivation and relation to the augmented state of Appendix C.1 are given in Appendix F.1.

Definition C.13 (Critical LRs). For either homogeneous system, strict mean-square stability means that $\mathbb { E } \| \mathbf { \boldsymbol { x } } _ { t } \| ^ { 2 } + \mathbb { E } \| \mathbf { \boldsymbol { x } } _ { t - 1 } \| ^ { 2 } \to 0$ for every deterministic initial pair. The projected critical LR $\eta _ { \star }$ <sub>⋆</sub> and the compensated critical LR $\eta _ { \star } ^ { \mathsf { P } }$ are the respective suprema of the strictly stable LRs.

By Proposition $\mathrm { A } . 3 , \eta _ { \star }$ is the projected critical LR defined by bounded affine moments in Definition $_ { \mathrm { A } . 2 }$ . The same argument identifies $\eta _ { \star } ^ { \mathsf { P } }$ with the bounded affine threshold of the compensated system when $A _ { \mathsf { P } } \succ 0$ . By Corollary F.4, the strictly stable rates form the intervals $( 0 , \eta _ { \star } )$ and $( 0 , \eta _ { \star } ^ { \mathsf { P } } )$ , respectively. Their second-moment maps are denoted $\mathcal { T } _ { \eta }$ and $\mathcal { T } _ { \eta } ^ { \mathsf { P } }$ , as defined in (64).

The detector. Fix a search window $\mathcal { T } = [ \eta - w , \eta + w ] \subset ( 0 , \infty )$ around the training LR η, an integer horizon $T \geq 1$ and a cutoff $M > 1$ . When $\lambda _ { \mathrm { W D } } > 0 .$ , assume $\mathcal { T } \subset ( 0 , 1 / \lambda _ { \mathrm { W D } } ]$ , so each candidate $\eta ^ { \prime } \in \mathcal { I }$ has effective momentum $b = \beta _ { 1 } ( 1 - \eta ^ { \prime } \lambda _ { \mathrm { W D } } ) \in [ 0 , \beta _ { 1 } ]$ . The expected detector, the idealized version of Algorithm 3 analyzed here, returns

$$
\bar { \eta } _ { \star } ( M , T ) : = \operatorname* { s u p } \Bigl \{ \eta ^ { \prime } \in \mathcal { T } : \ \operatorname* { m a x } _ { 0 \leq t \leq T } \mathbb { E } \left. \epsilon _ { t } ( \eta ^ { \prime } ) \right. ^ { 2 } \leq M ^ { 2 } \mathrm { ~ f o r ~ e v e r y ~ } \left. \epsilon _ { 0 } \right. = 1 \Bigr \} ,\tag{50}
$$

where $\epsilon _ { t } ( \eta ^ { \prime } )$ follows (49) with prescribed momentum $m _ { 0 } = m _ { t _ { 0 } }$ , and the expectation is over minibatches. Differences between this expected detector and its implementation are listed in Appendix A.3. The detector accepts $\eta ^ { \prime } \in \mathcal { T } \mathrm { i f } \eta ^ { \prime }$ satisfies the condition inside the braces in (50), so $\bar { \eta } _ { \star } ( \bar { M } , T )$ is the supremum of the accepted LRs.

Assumption C.14 (Frozen compensated model). Assumption C.1 holds. Fix $0 \leq \beta _ { 1 } < 1 , \lambda _ { \mathrm { W D } } \geq 0$ and $\lambda _ { \mathsf { P } } > 0$ . The compensation satisfies $A _ { \mathsf { P } } = A + \lambda _ { \mathsf { P } } \pmb { I } \succ 0$ . All parameters and the batch law remain fixed while $\eta$ varies.

The preconditioner and batch law are frozen at the checkpoint. The Hessian and gradient from the same mini-batch may be correlated, and no small-noise assumption is imposed. Positive definiteness of $A _ { \mathsf { P } }$ requires compensation to remove all negative and zero curvature directions. If the compensation is estimated from data, the theorem applies after fixing that estimate only when the resulting mean is positive definite and the simulation batches still satisfy Assumption C.1. In particular, after the checkpoint and compensation are fixed, those batches must remain i.i.d. and independent of the prescribed initial state. The estimated compensation coefficient alone does not verify positive definiteness.

Assumption C.15 (Acceptance at the compensated critical LR). The compensated critical LR $\eta _ { \star } ^ { \mathsf { P } }$ belongs to the window I and is accepted by the detector (50): max $) \leq t \leq T \left. \epsilon _ { t } ( \eta _ { \star } ^ { \mathsf { P } } ) \right. ^ { 2 } \leq M ^ { 2 }$ for every $\| \epsilon _ { 0 } \| = 1$ , with the prescribed initial momentum.

The boundary condition $\rho ( \mathcal { T } _ { \eta _ { \star } ^ { \mathsf { P } } } ^ { \mathsf { P } } ) = 1$ does not imply acceptance. Transient growth, checkpoint momentum, and affine forcing can exceed the cutoff. Proposition F.11 gives an explicit sufficient condition on M. Acceptance of $\eta _ { \star } ^ { \mathsf { P } }$ is used for the upper bound below. The lower bound only needs $\eta _ { \star } ^ { \mathsf { P } } \in \mathcal { I }$ and a nonempty accepted set.

The two sources of error. The detector output $\bar { \eta } _ { \star } ( M , T )$ estimates the threshold $\eta _ { \star } ^ { \mathsf { P } }$ of the compensated system. To compare it with the target $\eta _ { \star }$ , we separate the error in inverse LR as

$$
\frac { 1 } { \bar { \eta } _ { \star } ( M , T ) } - \frac { 1 } { \eta _ { \star } } = \underbrace { \left( \frac { 1 } { \bar { \eta } _ { \star } ( M , T ) } - \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } \right) } _ { \mathrm { f i n i t e ~ s i m u l a t i o n } } + \underbrace { \left( \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } - \frac { 1 } { \eta _ { \star } } \right) } _ { \mathrm { c o m p e n s a t i o n } } .
$$

The first term comes from observing only $T$ steps. The second comes from replacing projection onto the positive eigenspace with compensation, and remains even if the compensated threshold is known exactly. We compare inverse LRs because $( 1 - b ) / \alpha = 1 / \eta + \beta _ { 1 } \lambda _ { \mathrm { W D } } / ( 1 - \beta _ { 1 } )$ in the stability criterion below.

The quantity that determines stability. Unqualified vector norms are Euclidean, matrix norms are spectral, and $\| \cdot \| _ { F }$ is the Frobenius norm. For a positive definite mean $C$ with its corresponding centered noise, and $- 1 < b < 1$ , let

$$
c _ { b } : = \frac { 1 - b } { 1 + b } , \qquad k _ { b } : = \frac { b } { 1 - b ^ { 2 } } .
$$

For a symmetric test matrix, $[ C , X ] : = C X - X C$ is the commutator. Define

$$
q _ { C , b } ( X ) : = \frac { \mathbb { E } \operatorname { t r } ( X \Delta _ { t } X \Delta _ { t } ) + c _ { b } \operatorname { t r } ( X C X C ) - k _ { b } \left\| [ C , X ] \right\| _ { F } ^ { 2 } } { 2 \operatorname { t r } ( C X ^ { 2 } ) } , \qquad 0 \neq X \succeq 0 ,\tag{51}
$$

$$
r _ { C } ( \boldsymbol { b } ) : = \operatorname* { m a x } _ { \boldsymbol { x } \succeq 0 , \ \| \boldsymbol { X } \| _ { F } = 1 } q _ { C , \boldsymbol { b } } ( \boldsymbol { X } ) .\tag{52}
$$

The noise is $\Delta _ { t } ^ { + }$ when $C = A ^ { + }$ and $\Delta _ { \imath }$ <sub>t</sub> when $C = A _ { \mathsf { P } }$ . The maximum is attained on the compact set of positive semidefinite matrices of unit Frobenius norm. The exact criterion in Theorem F.1 shows that the corresponding recurrence is strictly mean-square stable precisely when $( 1 - b ) / \alpha > r _ { C } ( b )$

Curvatures in the error bound. The maximizers of the stability quotient determine which curvatures enter the comparison. Set $b _ { \star } : = \beta _ { 1 } ( 1 - \eta _ { \star } \lambda _ { \mathrm { W D } } )$ and $b _ { \star } ^ { \mathsf { P } } : = \dot { \beta } _ { 1 } \dot { ( } 1 - \eta _ { \star } ^ { \mathsf { P } } \lambda _ { \mathrm { W D } } )$ , and choose

$$
X _ { \star } \in \underset { X \succeq 0 , \parallel X \parallel _ { F } = 1 } { \arg \operatorname* { m a x } } q _ { A ^ { + } , b _ { \star } } ( X ) , \quad \mu _ { \star } : = \mathrm { t r } ( A ^ { + } X _ { \star } ^ { 2 } ) ,
$$

$$
\pmb { X } _ { \star } ^ { \mathsf { P } } \in \operatorname * { a r g m a x } _ { \pmb { X } \succeq 0 , \| \pmb { X } \| _ { F } = 1 } q _ { A _ { \mathsf { P } } , b _ { \star } ^ { \mathsf { P } } } ( \pmb { X } ) , \quad \mu _ { \star } ^ { \mathsf { P } } : = \mathrm { t r } \big ( \pmb { A } _ { \mathsf { P } } ( \pmb { X } _ { \star } ^ { \mathsf { P } } ) ^ { 2 } \big ) .\tag{53}
$$

The first maximization is over $p \times p$ matrices and the second over d $\times d$ matrices. Since $\operatorname { t r } ( X _ { \star } ^ { 2 } ) =$ $\mathrm { t r } ( ( X _ { \star } ^ { \mathsf { P } } ) ^ { 2 } ) = 1$ , these are weighted averages of curvature, with

$$
\lambda _ { \operatorname* { m i n } } ( A ^ { + } ) \leq \mu _ { \star } \leq L , \qquad \lambda _ { \operatorname* { m i n } } ( A _ { \mathsf { P } } ) \leq \mu _ { \star } ^ { \mathsf { P } } \leq L + \lambda _ { \mathsf { P } } .
$$

In particular, $\mu _ { \star }$ need not equal the largest curvature L. Noise in directions with smaller mean curvature can also limit stability.

Compensation error. Compensation shifts the curvature on the positive eigenspace and retains the directions removed by projection. We account for these effects separately. The shift is controlled by the function

$$
\Phi ( s ) : = \frac { c _ { \star } \lambda _ { \sf P } } { 2 } - \frac { \lambda _ { \sf P } } { s + \lambda _ { \sf P } } \Big ( r _ { A ^ { + } } \big ( b _ { \star } \big ) - \frac { c _ { \star } s } { 2 } \Big ) , \qquad 0 \leq s \leq L ,\tag{54}
$$

where $c _ { \star } : = ( 1 - b _ { \star } ) / ( 1 + b _ { \star } )$ . This function is increasing and satisfies $\Phi ( L ) \leq c _ { \star } \lambda _ { \mathsf { P } } / 2$ . The second term can exceed the first, so $\Phi ( s )$ need not be positive.

The compensated system also retains noise in the discarded directions and between the two subspaces. We compare its noise contribution with that of the projected system through

$$
\delta _ { \mathrm { c o v } } : = \operatorname* { m a x } _ { X \geq 0 , \ \| X \| _ { F } = 1 } \frac { \mathbb { E } \operatorname { t r } ( X \mathbf { \Delta } _ { t } X \mathbf { \Delta } \mathbf { \Delta } X _ { t } ) - \mathbb { E } \operatorname { t r } ( X ^ { + } \mathbf { \Delta } _ { t } ^ { + } X ^ { + } \mathbf { \Delta } \mathbf { \Delta } X _ { t } ^ { + } ) } { 2 \operatorname { t r } ( A _ { \mathbb { P } } X ^ { 2 } ) } \geq 0 ,\tag{55}
$$

where $X ^ { + }$ is the block on the positive eigenspace. This quantity is finite and vanishes when the discarded and cross blocks of the noise vanish. Lemma F.7 bounds it using their second moments. Together, Φ and $\delta _ { \mathrm { c o v } }$ bound the difference between $1 / \eta _ { \star } ^ { \mathsf { P } }$ and $1 / \eta _ { \star }$ (Proposition F.5).

Finite simulation error. An LR above $\eta _ { \star } ^ { \mathsf { P } }$ can satisfy the expected detector’s displacement bound over only $T$ steps. Under Assumption C.14, every accepted LR $\eta ^ { \prime }$ satisfies $\dot { \rho } ( \mathcal { T } _ { \eta ^ { \prime } } ^ { \mathsf { P } } ) ~ \leq ~ \rho _ { T }$ by Lemma F.9, where

$$
\rho _ { T } : = \left[ \frac { M ( 1 + \beta _ { 1 } ) } { 1 - \beta _ { 1 } } \right] ^ { 2 / T } .\tag{56}
$$

This bound approaches one as $( 1 + \log M ) / T \to 0$ , limiting the instability that can go undetected. Proposition F.6 converts it into a bound on inverse LR. With $\alpha _ { \star } ^ { \mathsf { P } } : = ( 1 - \check { \beta } _ { 1 } ) \eta _ { \star } ^ { \mathsf { P } }$ , the resulting error term is

$$
H _ { T } : = ( \rho _ { T } - 1 ) \left[ \frac { 2 b _ { \star } ^ { \mathrm { P } } } { ( \rho _ { T } + b _ { \star } ^ { \mathrm { P } } ) \alpha _ { \star } ^ { \mathrm { P } } } + \frac { ( 1 - b _ { \star } ^ { \mathrm { P } } / \rho _ { T } ) ^ { 2 } } { 2 \mu _ { \star } ^ { \mathrm { P } } ( \alpha _ { \star } ^ { \mathrm { P } } ) ^ { 2 } } \right] .\tag{57}
$$

Combining the compensation and finite simulation bounds gives the following result.

Theorem C.16 (Accuracy of the expected EoS detector). Under Assumptions C.14 and C.15, the expected detector satisfies

$$
\operatorname* { m i n } \{ \Phi ( \mu _ { \star } ) , 0 \} - H _ { T } \ \leq \ \frac { 1 } { \bar { \eta } _ { \star } ( M , T ) } - \frac { 1 } { \eta _ { \star } } \ \leq \ \operatorname* { m a x } \{ \Phi ( L ) + \delta _ { \mathrm { c o v } } , 0 \} .\tag{58}
$$

Forfixed model parameters, $H _ { T } = O ( ( 1 + \log M ) / T ) a s ( 1 + \log M ) / T \to 0 . \ I f \beta _ { 1 } \lambda _ { \mathrm { W D } } = 0 ,$ , the minimum and maximum with zero can be omittedfrom (58).

Acceptance of $\eta _ { \star } ^ { \mathsf { P } }$ ensures $\bar { \eta } _ { \star } ( M , T ) \geq \eta _ { \star } ^ { \mathsf { P } }$ , so the finite simulation error lies in $[ - H _ { T } , 0 ]$ . This explains why $H _ { T }$ appears only in the lower bound. Increasing $T$ reduces this contribution when $( 1 + \log M ) { \dot { / } } T \to { \dot { 0 } }$ and Assumption C.15 continues to hold, but does not remove the compensation error. Proposition F.11 shows how a cutoff growing polynomially with $T$ can ensure acceptance of $\eta _ { \star } ^ { \mathsf { P } }$

Weight decay makes the effective momentum depend on the LR. The minimum and maximum with zero account for this dependence when comparing the thresholds. Without momentum, (57) reduces to $H _ { T } = ( \rho _ { T } - 1 ) / ( 2 \dot { \mu } _ { \star } ^ { \sf P } ( \eta _ { \star } ^ { \sf P } ) ^ { 2 } )$ . The proof of Theorem C.16 is in Appendix F. Corollary F.8 gives the simpler compensation and noise bound used in Theorem A.5.

## D PROOFS FOR THE LOCAL LINEAR SYSTEM AND THE STABILITY RESULTS

This appendix proves the statements of Appendices C.1 and C.2 in the order they are stated. Appendix D.1 derives the augmented linear system (Proposition C.2) and its compensated form. Appendix D.2 derives the moment recursion of a stochastic linear system (Proposition C.3). Appendix D.3 derives the moment recursion for SGD (Corollary C.4) from the two propositions. Appendix D.4 proves the two negative-curvature results (Lemma C.5 and Corollary C.6) and Appendix D.5 the divergence above the critical LR (Corollary C.8). Theorem C.7 is restated from Corollary F.4 and proved in Appendix F.5.3. Appendix D.6 proves that bounded affine stability gives the same critical LR as the spectral criterion (Proposition A.3). Throughout, $\rho$ denotes the spectral radius and ∥ · ∥ the spectral norm. We use gray text to indicate the identity, definition, or assumption used in each step of the derivation.

## D.1 PROOF OF PROPOSITION C.2

ProofofProposition C.2. The idea is to center the displacement at $\mathbf { x } _ { \star } .$ , multiply each of the two updates in (33) by the power of $_ { r }$ that appears in the corresponding block of $z _ { t } .$ , and identify the blocks of $M _ { t } ( \eta )$ and $\pmb q _ { t } .$ . Here $\scriptstyle { \mathbf { 2 } } \neq$ <sub>⋆</sub> is the fixed reference displacement in the proposition and need not be a stationary point. Write $\bar { \boldsymbol { \epsilon } } _ { t } : = \boldsymbol { \epsilon } _ { t } - \boldsymbol { x } ,$ for the centered displacement, so that $z _ { t } =$ $\left( \eta { \cal P } ^ { - 1 / 2 } m _ { t } , { \cal P } ^ { 1 / 2 } \bar { \epsilon } _ { t } \right)$ . Recall the forcing in the momentum block,

$$
\pmb { h } _ { t } = ( 1 - \beta _ { 1 } ) \eta \pmb { P } ^ { - 1 / 2 } ( \pmb { g } _ { \mathcal { B } _ { t } } + \pmb { H } _ { \mathcal { B } _ { t } } \pmb { x } _ { \star } ) .
$$

Recall that $\pmb { D } _ { \pmb { \mathrm { B } } _ { t } } = \pmb { P } ^ { - 1 / 2 } \pmb { H } _ { \pmb { B } _ { t } } \pmb { P } ^ { - 1 / 2 }$ is the preconditioned mini-batch Hessian, as defined before Proposition C.2.

Step 1. The momentum block. Multiplying the momentum update in (33) by $\eta P ^ { - 1 / 2 }$ and substituting $\boldsymbol { \epsilon } _ { t } = \bar { \boldsymbol { \epsilon } } _ { t } + \boldsymbol { x }$ gives the first block of $z _ { t + 1 }$ ，

$$
\begin{array} { r l r } { \eta P ^ { - 1 / 2 } m _ { t + 1 } = \beta _ { 1 } \eta P ^ { - 1 / 2 } m _ { t } + ( 1 - \beta _ { 1 } ) \eta P ^ { - 1 / 2 } ( g _ { B _ { t } } + H _ { B _ { t } } \epsilon _ { t } ) } & { \mathrm { ~ m o m e n t u m ~ u p d a t e } } \\ { = \beta _ { 1 } \eta P ^ { - 1 / 2 } m _ { t } + ( 1 - \beta _ { 1 } ) \eta P ^ { - 1 / 2 } H _ { B _ { t } } \bar { \epsilon } _ { t } } & \\ { + ( 1 - \beta _ { 1 } ) \eta P ^ { - 1 / 2 } ( g _ { B _ { t } } + H _ { B _ { t } } x _ { * } ) } & { \epsilon _ { t } = \bar { \epsilon } _ { t } + x _ { * } } \\ { = \beta _ { 1 } \eta P ^ { - 1 / 2 } m _ { t } + ( 1 - \beta _ { 1 } ) \eta P ^ { - 1 / 2 } H _ { B _ { t } } P ^ { - 1 / 2 } P ^ { 1 / 2 } \bar { \epsilon } _ { t } } & \\ { + ( 1 - \beta _ { 1 } ) \eta P ^ { - 1 / 2 } ( g _ { B _ { t } } + H _ { B _ { t } } x _ { * } ) } & { P ^ { - 1 / 2 } P ^ { 1 / 2 } = I } \\ { } & { } & \\ { = \beta _ { 1 } \eta P ^ { - 1 / 2 } m _ { t } + ( 1 - \beta _ { 1 } ) \eta D _ { B _ { t } } P ^ { 1 / 2 } \bar { \epsilon } _ { t } + h _ { t } . } & { \mathrm { d e f i n i t i o n s ~ o f } \ : D _ { B _ { t } } , h _ { t } } \end{array}
$$

Step 2. The displacement block. Subtracting x<sub>⋆</sub> from both sides of the displacement update and multiplying by $P ^ { 1 / 2 }$ gives the second block,

$$
\begin{array} { r l r } { P ^ { 1 / 2 } \bar { \epsilon } _ { t + 1 } = P ^ { 1 / 2 } \bar { \epsilon } _ { t } - \eta P ^ { - 1 / 2 } m _ { t + 1 } - \eta \lambda _ { \mathrm { W D } } P ^ { 1 / 2 } \big ( \bar { \epsilon } _ { t } + x _ { \star } + w _ { t _ { 0 } } \big ) } & { \epsilon _ { t } = \bar { \epsilon } _ { t } + x _ { \star } } \\ { = \left( 1 - \eta \lambda _ { \mathrm { W D } } \right) P ^ { 1 / 2 } \bar { \epsilon } _ { t } - \eta P ^ { - 1 / 2 } m _ { t + 1 } - \eta \lambda _ { \mathrm { W D } } P ^ { 1 / 2 } \big ( x _ { \star } + w _ { t _ { 0 } } \big ) } & { \mathrm { c o l l e c t ~ } \bar { \epsilon } _ { t } \mathrm { ~ t e r m s } } \\ { = - \beta _ { 1 } \eta P ^ { - 1 / 2 } m _ { t } + \left( \left( 1 - \eta \lambda _ { \mathrm { W D } } \right) I - \left( 1 - \beta _ { 1 } \right) \eta D _ { B _ { t } } \right) P ^ { 1 / 2 } \bar { \epsilon } _ { t } } & \\ { - h _ { t } - \eta \lambda _ { \mathrm { W D } } P ^ { 1 / 2 } \big ( x _ { \star } + w _ { t _ { 0 } } \big ) . } & { \mathrm { i n s e r t ~ S t e p ~ 1 } } \end{array}
$$

In the state ${ \boldsymbol { z } } _ { t }$ , the two identities read $z _ { t + 1 } = M _ { t } ( \eta ) z _ { t } + \pmb q _ { t }$ with $M _ { t } ( \eta )$ and $\pmb q _ { t }$ as in (34).

Step 3. Compensation. The gradient of the compensated loss $G _ { t }$ of Appendix C.1 is

$$
\begin{array} { r l } { \nabla G _ { t } ( \epsilon ) = \nabla \mathcal { L } _ { B _ { t } } ( w _ { t _ { 0 } } ) + \nabla ^ { 2 } \mathcal { L } _ { B _ { t } } ( w _ { t _ { 0 } } ) \epsilon + \lambda _ { \mathrm { P } } P \epsilon } & { \nabla \frac { \lambda _ { \mathrm { P } } } { 2 } \epsilon ^ { \top } P \epsilon = \lambda _ { \mathrm { P } } P \epsilon , \mathrm { a s } P \mathrm { i s } \mathrm { s y m m e t r i c } } \\ { = g _ { B _ { t } } + ( H _ { B _ { t } } + \lambda _ { \mathrm { P } } P ) \epsilon , } & { \mathrm { d e f i n i t i o n s o f } g _ { B _ { t } } \mathrm { a n d } H _ { B _ { t } } } \end{array}
$$

so the penalty replaces the mini-batch Hessian $H _ { B _ { t } }$ by $H _ { B _ { t } } + \lambda _ { \mathsf { P } } P$ in the momentum update. In preconditioned coordinates,

$$
\begin{array} { r l } & { P ^ { - 1 / 2 } ( H _ { \mathsf { B } _ { t } } + \lambda \mathsf { p } P ) P ^ { - 1 / 2 } = P ^ { - 1 / 2 } H _ { \mathsf { B } _ { t } } P ^ { - 1 / 2 } + \lambda \mathsf { p } P ^ { - 1 / 2 } P P ^ { - 1 / 2 } } \\ & { \qquad = D _ { \mathsf { B } _ { t } } + \lambda \mathsf { p } I . } \end{array}
$$

Repeating Steps 1 and 2 with $H _ { B _ { t } } + \lambda _ { \mathsf { P } } P$ in place of $H _ { B _ { t } }$ , and hence $D _ { B _ { t } } + \lambda _ { \mathsf { P } } I$ in place of $\pmb { D } _ { \pmb { B } _ { t } }$ gives the augmented system with the transition matrix $\breve { M } _ { t } ( \eta )$ of (35) and with $h _ { t }$ formed from $H _ { B _ { t } } + \lambda _ { \mathsf { P } } P .$ . The weight decay term is not touched by the penalty, so its coefficient remains $\lambda _ { \mathrm { W D } }$

Step 4. Independence. Under Assumption $\mathbf { C } . 1 , \left( M _ { t } ( \eta ) , \pmb { q } _ { t } \right)$ is a function of $B _ { t }$ alone, while ${ \boldsymbol { z } } _ { t }$ depends only on the initial state and on $\boldsymbol { B } _ { 0 } , \ldots , \boldsymbol { B } _ { t - 1 }$ , so $M _ { t } ( \eta )$ and $\pmb q _ { t }$ are independent of ${ \boldsymbol { z } } _ { t }$ . This completes the proof of Proposition C.2. □

## D.2 PROOF OF PROPOSITION C.3

ProofofProposition C.3. We first derive the closed recursions for the mean and the second moment. A strict spectral bound then makes this finite-dimensional affine recursion converge, including when

the current transition and forcing are correlated. Since ${ \boldsymbol { z } } _ { t }$ is a function of $z _ { \mathrm { 0 } }$ and of $( M _ { s } , q _ { s } ) _ { s < t }$ , the pair $( M _ { t } , q _ { t } )$ is independent of ${ \boldsymbol { z } } _ { t } .$ . Finite second moments propagate by induction:

$$
\begin{array} { r l r } & { \mathbb { E } \| z _ { t + 1 } \| ^ { 2 } \leq 2 \mathbb { E } \| M _ { t } z _ { t } \| ^ { 2 } + 2 \mathbb { E } \| q _ { t } \| ^ { 2 } } & { \| a + b \| ^ { 2 } \leq 2 \| a \| ^ { 2 } + 2 \| b \| ^ { 2 } } \\ & { \leq 2 \mathbb { E } \| M _ { t } \| ^ { 2 } \mathbb { E } \| z _ { t } \| ^ { 2 } + 2 \mathbb { E } \| q _ { t } \| ^ { 2 } < \infty . } & { \mathrm { i n d e p e n d e n c e ~ a n d ~ t h e ~ i n d u c t i o n ~ h y p o t h e s i s } } \end{array}
$$

The initial moment is finite by assumption. The Cauchy–Schwarz inequality also gives $\mathbb { E } [ \| M _ { t } \| \| q _ { t } \| ] < \infty .$ , so the cross terms below are well defined without independence between $M _ { t }$ and $\pmb q _ { t }$ .

Step 1. The mean. Recall that $M = \mathbb { E } [ M _ { t } ]$ and $\pmb { z } _ { t } ^ { ( 1 ) } = \mathbb { E } [ \pmb { z } _ { t } ]$ . Then

$$
\begin{array} { r l r l } { z _ { t + 1 } ^ { ( 1 ) } = \mathbb { E } [ M _ { t } z _ { t } ] + \mathbb { E } [ \pmb { q } _ { t } ] } & { \quad } & & { \mathrm { e x p e c t a t i o n ~ o f ~ } z _ { t + 1 } = M _ { t } z _ { t } + q _ { t } } \\ & { \quad \quad = M z _ { t } ^ { ( 1 ) } + \mathbb { E } [ \pmb { q } _ { t } ] . } & { \quad } & & { M _ { t } \mathrm { ~ i s ~ i n d e p e n d e n t ~ o f ~ } z _ { t } } \end{array}
$$

Step 2. The second moment. Recall that $\boldsymbol { z } _ { t } ^ { ( 2 ) } = \mathsf { v e c } ( \mathbb { E } [ \boldsymbol { z } _ { t } \boldsymbol { z } _ { t } ^ { \top } ] ) , \boldsymbol { S } = \mathbb { E } [ \boldsymbol { q } _ { t } \otimes \boldsymbol { M } _ { t } + \boldsymbol { M } _ { t } \otimes \boldsymbol { q } _ { t } ]$ and $\begin{array} { r } { \pmb { b } _ { \mathrm { q u a d } } = \mathsf { v e c } ( \mathbb { E } [ \pmb { q } _ { t } \pmb { q } _ { t } ^ { \top } ] ) } \end{array}$ . Expanding the outer product of $z _ { t + 1 } = M _ { t } z _ { t } + q _ { t }$ with itself gives

$$
z _ { t + 1 } z _ { t + 1 } ^ { \top } = { M _ { t } } z _ { t } z _ { t } ^ { \top } { M _ { t } ^ { \top } } + { M _ { t } } z _ { t } q _ { t } ^ { \top } + { q _ { t } } z _ { t } ^ { \top } { M _ { t } ^ { \top } } + { q _ { t } } q _ { t } ^ { \top } .
$$

Vectorize term by term using vec $( X Y Z ) = ( Z ^ { \top } \otimes X ) { \mathsf { v e c } } ( Y )$ . In particular,

$$
\begin{array} { r } { \mathsf { v e c } ( M _ { t } z _ { t } \pmb { q } _ { t } ^ { \top } ) = ( \pmb { q } _ { t } \otimes M _ { t } ) z _ { t } , } \\ { \mathsf { v e c } ( \pmb { q } _ { t } \pmb { z } _ { t } ^ { \top } M _ { t } ^ { \top } ) = ( M _ { t } \otimes \pmb { q } _ { t } ) z _ { t } . } \end{array}
$$

Taking expectations gives

$$
\begin{array} { r l r l } & { z _ { t + 1 } ^ { ( 2 ) } = \mathbb E \big [ ( M _ { t } \otimes M _ { t } ) \mathsf { v e c } ( z _ { t } z _ { t } ^ { \top } ) \big ] } \\ & { \qquad + \mathbb E \big [ ( \pmb { q } _ { t } \otimes M _ { t } + M _ { t } \otimes \pmb { q } _ { t } ) z _ { t } \big ] } \\ & { \qquad + \mathsf { v e c } ( \mathbb E [ \pmb { q } _ { t } \pmb { q } _ { t } ^ { \top } ] ) } & & { \qquad \mathrm { v e c t o r i z e ~ t e r m ~ b y ~ t e r m } } \\ & { = \mathbb E [ M _ { t } \otimes M _ { t } ] z _ { t } ^ { ( 2 ) } + S z _ { t } ^ { ( 1 ) } + b _ { \mathrm { q u a d } } . } & & { \qquad z _ { t } \mathrm { i n d e p e n d e n t ~ o f } \left( M _ { t } , q _ { t } \right) } \end{array}
$$

Combining the two moment recursions gives the block form (36).

Step 3. The spectral radius. For the block lower triangular matrix of the proposition and every scalar z,

$$
\begin{array} { r l } &  \operatorname* { d e t } \left[ \begin{array} { c c } { z I - M } & { \mathbf { 0 } } \\ { - S } & { z I - \mathbb { E } \big [ M _ { t } \otimes M _ { t } \big ] \right] } \\ & { \qquad = \operatorname* { d e t } ( z I - M ) \operatorname* { d e t } ( z I - \mathbb { E } \big [ M _ { t } \otimes M _ { t } \big ] ) , \quad \mathrm { d e t e r m i n a n t ~ o f ~ a ~ b l o c k ~ l o w e r ~ t r i a n g u l a r ~ m a t r i x } } \end{array} \end{array}
$$

so the eigenvalues of the block matrix are those of M together with those of $\mathbb { E } [ M _ { t } \otimes M _ { t } ]$ ], and its spectral radius is max $\left\{ \rho ( M ) , \rho ( \mathbb { E } [ M _ { t } \otimes M _ { t } ] ) \right\}$ }.

Step 4. Bounded affine moments under the strict condition. Suppose $\rho ( \mathbb { E } [ M _ { t } \otimes M _ { t } ] ) < 1$ . By the mean-radius inequality in the second-moment characterization (Lemma F.14),

$$
\rho ( M ) ^ { 2 } \leq \rho ( \mathbb { E } [ M _ { t } \otimes M _ { t } ] ) < 1 .
$$

Let B be the block matrix in (36), let $\mathbf { c } : = ( \mathbb { E } [ \pmb { q } _ { t } ] , \pmb { b } _ { \mathrm { q u a d } } ) ^ { \top }$ be its constant forcing, and write $\mathbf { u } _ { t } : = ( z _ { t } ^ { ( 1 ) } , z _ { t } ^ { ( 2 ) } ) ^ { \top }$ . Step 3 gives $\rho ( \mathbf { B } ) < 1$ . Iterating the moment recursion yields

$$
\begin{array} { r l } { \mathbf { u } _ { t } = \mathbf { B } ^ { t } \mathbf { u } _ { 0 } + \displaystyle \sum _ { j = 0 } ^ { t - 1 } \mathbf { B } ^ { j } \mathbf { c } \qquad } & { \mathrm { i t e r a t e ~ } \mathbf { u } _ { t + 1 } = \mathbf { B } \mathbf { u } _ { t } + \mathbf { c } } \\ { \longrightarrow ( I - \mathbf { B } ) ^ { - 1 } \mathbf { c } . \qquad } & { \mathbf { B } ^ { t } \to 0 \mathrm { ~ a n d ~ t h e ~ N e u m a n n ~ s e r i e s , ~ } \rho ( \mathbf { B } ) < 1 } \end{array}
$$

The initial vector $\mathbf { u } _ { 0 }$ is finite by the second-moment assumption. The trace of the second-moment matrix is $\mathbb { E } \Vert z _ { t } \Vert ^ { 2 }$ , so convergence of its entries implies convergence, and hence boundedness, of this trace. This proves the affine stability assertion for every initial state of finite second moment independent of the batch sequence. The cross block S already includes any correlation between the current transition and forcing. This completes the proof of Proposition C.3. □

## D.3 PROOF OF COROLLARY C.4

Proof of Corollary C.4. We center the SGD recurrence at the population stationary point, which removes its mean forcing.

Step 1. The displacement block closes. Take $\beta _ { 1 } = 0 , P = I$ and $\lambda _ { \mathrm { W D } } = 0$ . Write $\pmb { x } _ { \star } : = \pmb { w } _ { \star } - \pmb { w } _ { t _ { 0 } }$ and $\bar { \epsilon } _ { t } : = \epsilon _ { t } - \pmb { x } _ { \star } = \pmb { w } _ { t _ { 0 } + t } - \pmb { w }$ <sub>⋆</sub>. The gradient of the surrogate is $g _ { B _ { t } } + H _ { B _ { t } } \epsilon _ { t }$ , so centering its SGD update gives

$$
\begin{array} { r l r l } & { \bar { \epsilon } _ { t + 1 } = \bar { \epsilon } _ { t } - \eta \big ( g _ { \mathcal { B } _ { t } } + H _ { \mathcal { B } _ { t } } ( \bar { \epsilon } _ { t } + x _ { \star } ) \big ) } & & { \quad \epsilon _ { t } = \bar { \epsilon } _ { t } + x _ { \star } } \\ & { \quad \quad = ( I - \eta H _ { \mathcal { B } _ { t } } ) \bar { \epsilon } _ { t } - \eta \big ( g _ { \mathcal { B } _ { t } } + H _ { \mathcal { B } _ { t } } x _ { \star } \big ) } & & { \quad \mathrm { c o l l e c t ~ t h e ~ c e n t e r e d ~ d i s p l a c e m e n t } } \\ & { \quad \quad = M _ { t } \bar { \epsilon } _ { t } - \eta r _ { t } . } & & { \quad \mathrm { d e f i n i t i o n s ~ o f ~ } M _ { t } \mathrm { ~ a n d ~ } r _ { t } } \end{array}
$$

So the centered displacement is itself a stochastic linear system with transition M and forcing $- \eta \mathbf { r } _ { t }$ Under Assumption C.1 the pairs $( M _ { t } , - \eta \pmb { r } _ { t } )$ are i.i.d. functions of $\left( { H } _ { B _ { t } } , { g } _ { B _ { t } } \right)$ , have finite second moments, and are independent of $\bar { \epsilon } _ { 0 } = - \pmb { x } _ { \star }$

Step 2. Apply Proposition C.3. Apply the proposition to $\bar { \boldsymbol { \epsilon } } _ { t + 1 } = M _ { t } \bar { \boldsymbol { \epsilon } } _ { t } + \left( - \eta \boldsymbol { r } _ { t } \right)$ . Its mean forcing vanishes,

$$
\begin{array} { r l } { \mathbb { E } [ - \eta r _ { t } ] = - \eta \big ( \mathbb { E } [ g _ { \mathcal { B } _ { t } } ] + \mathbb { E } [ H _ { \mathcal { B } _ { t } } ] x _ { \star } \big ) } & { x _ { \star } \mathrm { ~ i s ~ d e t e r m i n i s t i c } } \\ { = - \eta \big ( \nabla \mathcal { L } ( w _ { t _ { 0 } } ) + \nabla ^ { 2 } \mathcal { L } ( w _ { t _ { 0 } } ) ( w _ { \star } - w _ { t _ { 0 } } ) \big ) } & { \mathrm { u n b i a s e d ~ m i n i - b a t c h ~ g r a d i e n t ~ a n d ~ H e s s i a n } } \\ { = 0 , } & { \mathrm { b y ~ ( 3 7 ) } } \end{array}
$$

and its two constants are those of the corollary,

$$
\begin{array} { r l } { \mathbb { E } [ ( - \eta r _ { t } ) \otimes M _ { t } + M _ { t } \otimes ( - \eta r _ { t } ) ] = - \eta \mathbb { E } [ r _ { t } \otimes M _ { t } + M _ { t } \otimes r _ { t } ] = S , } & { \mathrm { ~ \otimes ~ i s ~ b i l i n e a r } } \\ { \mathrm { v e c } ( \mathbb { E } [ ( \eta r _ { t } ) ( \eta r _ { t } ) ^ { \top } ] ) = \eta ^ { 2 } \mathrm { v e c } ( \mathbb { E } [ r _ { t } r _ { t } ^ { \top } ] ) = b _ { \mathrm { q u a d . } } } & { \mathrm { ~ d e f i n i t i o n ~ o f ~ } b _ { \mathrm { q u a d . } } } \end{array}
$$

The two mean identities used above are part of Assumption C.1. The moments $\mathbb { E } [ \bar { \epsilon } _ { t } ]$ and $\mathsf { v e c } ( \mathbb { E } [ \bar { \epsilon } _ { t } \bar { \epsilon } _ { t } ^ { \top } ] )$ are the $\epsilon _ { t } ^ { ( 1 ) }$ and $\epsilon _ { t } ^ { ( 2 ) }$ of the corollary. Thus (36) is (38), with zero forcing in the mean block. For the generic quadratic of Section 4, apply the same proposition with $M _ { t } = I - \eta A _ { t }$ and $\pmb q _ { t } = - \eta ( \pmb A _ { t } \pmb x _ { \star } + \pmb b _ { t } )$ . Its assumptions follow from the stated iid finite-moment model, and $\mathbb { E } [ q _ { t } ] =$ $- \eta ( A \pmb { x } _ { \star } + \pmb { b } ) = 0$ . This proves the moment equations of Proposition 4.2. If its second-moment spectral radius is less than one, Step 4 of the generic proof gives finite limiting moments for every deterministic initialization, completing that proposition’s affine boundedness claim. This completes the proof of Corollary C.4. □

## D.4 PROOFS OF LEMMA C.5 AND COROLLARY C.6

ProofofLemma C.5. The idea is that the mean AGM dynamics decouple along the eigenvectors of D into $2 \times 2$ systems, and that the one along a negative effective curvature has a real eigenvalue larger than one. Let $\mu _ { 1 } , \ldots , \mu _ { d }$ be the eigenvalues of the population preconditioned Hessian D. The corresponding eigenvalues of the effective curvature $\pmb { A } = \mathbf { \bar { D } } + \lambda _ { \mathrm { W D } } \pmb { I }$ are $\bar { \mu } _ { i } : = \mu _ { i } + \lambda _ { \mathrm { W D } }$ . Since (34) is affine in $\pmb { D } _ { \pmb { B } _ { t } }$ , the mean transition $\mathbb { E } [ M _ { t } ( \eta ) ]$ is obtained by replacing ${ D } _ { { B } _ { t } }$ with its mean $D ,$ so its blocks are $\beta _ { 1 } I , ( 1 - \beta _ { 1 } ) \eta D , - \beta _ { 1 } I$ and $( 1 - \dot { \eta } \lambda _ { \mathrm { W D } } ) { \cal I } - ( 1 - \dot { \beta } _ { 1 } ) \bar { \eta } { \cal D }$ . Every block is a polynomial in D, so this transition is block diagonal in an eigenbasis of D. Along an eigendirection with eigenvalue $\mu _ { i }$ , the mean AGM dynamics reduce to a two-dimensional linear system with transition matrix $M _ { \mu _ { i } } ( \eta )$ , the $2 \times 2$ block of $\mathbb { E } [ M _ { t } ( \eta ) ]$ along that eigendirection, obtained by replacing D with $\mu _ { i }$ in each block:

$$
\begin{array} { r l } { M _ { \mu _ { i } } ( \eta ) = \left[ \begin{array} { c c } { \beta _ { 1 } } & { ( 1 - \beta _ { 1 } ) \eta \mu _ { i } } \\ { - \beta _ { 1 } } & { 1 - \eta \lambda _ { \mathrm { W D } } - ( 1 - \beta _ { 1 } ) \eta \mu _ { i } } \end{array} \right] . } \end{array}
$$

Its characteristic polynomial evaluated at 1 is

$$
\begin{array} { r l } { \operatorname* { d e t } ( I - M _ { \mu _ { i } } ( \eta ) ) = \operatorname* { d e t } \bigg [ \stackrel { - \beta _ { 1 } } { \beta _ { 1 } } ~ \stackrel { - ( 1 - \beta _ { 1 } ) \eta \mu _ { i } } { \eta \lambda _ { \mathrm { W D } } } + ( 1 - \beta _ { 1 } ) \eta \mu _ { i } \bigg ] } \\ { = ( 1 - \beta _ { 1 } ) \big ( \eta \lambda _ { \mathrm { W D } } + ( 1 - \beta _ { 1 } ) \eta \mu _ { i } \big ) + \beta _ { 1 } ( 1 - \beta _ { 1 } ) \eta \mu _ { i } } & { 2 \times 2 \mathrm { ~ d e t e r m i n a n t } } \\ { = ( 1 - \beta _ { 1 } ) \eta \lambda _ { \mathrm { W D } } + ( 1 - \beta _ { 1 } ) \eta \mu _ { i } \big ( ( 1 - \beta _ { 1 } ) + \beta _ { 1 } \big ) } & { \mathrm { c o l l e c t ~ t h e ~ } \mu _ { i } \mathrm { ~ t e r m s } } \\ { = ( 1 - \beta _ { 1 } ) \eta \bar { \mu } _ { i } . } & { \bar { \mu } _ { i } = \mu _ { i } + \lambda _ { \mathrm { W D } } } \end{array}
$$

If $\bar { \mu } _ { i } < 0$ , this quantity is negative for every $\eta > 0$ . The characteristic polynomial $\lambda \mapsto \operatorname* { d e t } ( \lambda I -$ $M _ { \mu _ { i } } ( \eta ) )$ is monic of degree two, so it is positive for all large $\lambda .$ . Since it is negative at $\lambda = 1$ , the intermediate value theorem gives a real root larger than one, that is, an eigenvalue of $M _ { \mu _ { i } } ( \eta )$ larger than one. The homogeneous mean dynamics therefore have an expanding mode. Choose $z _ { \mathrm { 0 } }$ to be a nonzero right eigenvector of the mean transition for this real eigenvalue $r > 1$ . Independence across steps gives $\mathbb { E } [ z _ { t } ] = r ^ { t } z _ { 0 }$ , and Jensen’s inequality yields

$$
\begin{array} { r } { \mathbb { E } \| z _ { t } \| ^ { 2 } \geq \| \mathbb { E } [ z _ { t } ] \| ^ { 2 } = r ^ { 2 t } \| z _ { 0 } \| ^ { 2 } \longrightarrow \infty . } \end{array}
$$

Thus the homogeneous system is not mean-square stable. This completes the proof of Lemma C.5.

ProofofCorollary C.6. The idea is that the mean recursion is the full-batch recursion, along whose negative eigendirection the mean grows geometrically, and that the second moment dominates the squared mean. Recall from Corollary C.4 that $\epsilon _ { t } ^ { ( 1 ) } = \mathbb { E } [ { \pmb w } _ { t _ { 0 } + t } - { \pmb w } _ { \star } ]$ is the mean error and $M _ { t } \stackrel { - } { = } I - \eta { \cal H } _ { B _ { t } }$ is the mini-batch transition matrix, so that the mean recursion in (38) reads $\epsilon _ { t + 1 } ^ { ( 1 ) } = \mathbb { E } [ M _ { t } ] \epsilon _ { t } ^ { ( 1 ) }$ . Recall also that $\pmb { H } _ { t _ { 0 } } = \nabla ^ { 2 } \mathcal { L } ( \pmb { w } _ { t _ { 0 } } )$ is the population Hessian at the checkpoint (Appendix C.2), so that $\mathbb { E } [ M _ { t } ] = I - \eta M _ { t _ { 0 } }$ for every t. If $\bar { \pmb { H } } _ { t _ { 0 } } \bar { \ \pmb { u } } =$ λu with $\lambda < 0 \mathrm { \ a n d \ } \| \pmb { u } \| ^ { - } = 1$ then

$$
\begin{array} { r l r l } & { \boldsymbol { u } ^ { \top } \mathbb { E } [ w _ { t _ { 0 } + t } - w _ { \star } ] = \boldsymbol { u } ^ { \top } \mathbb { E } [ M _ { t } ] ^ { t } \mathbb { E } [ w _ { t _ { 0 } } - w _ { \star } ] } & & { \mathrm { i t e r a t e ~ t h e ~ m e a n ~ r e c u r s i o n ~ f o r ~ } t \mathrm { ~ s t e p s } } \\ & { \quad \quad \quad = \boldsymbol { u } ^ { \top } ( I - \eta H _ { t _ { 0 } } ) ^ { t } \mathbb { E } [ w _ { t _ { 0 } } - w _ { \star } ] } & & { \mathbb { E } [ M _ { t } ] = I - \eta H _ { t _ { 0 } } } \\ & { \quad \quad \quad = ( 1 - \eta \lambda ) ^ { t } \boldsymbol { u } ^ { \top } \mathbb { E } [ w _ { t _ { 0 } } - w _ { \star } ] . } & & { \boldsymbol { u } ^ { \top } H _ { t _ { 0 } } = \lambda \boldsymbol { u } ^ { \top } , \mathrm { a s } H _ { t _ { 0 } } \mathrm { ~ i s ~ s y m m e t r i c } } \end{array}
$$

Because $1 - \eta \lambda > 1$ , a nonzero initial component along u grows geometrically. The second moment then diverges for this initialization:

$$
\begin{array} { r l r l } & { \mathbb { E } \| w _ { t _ { 0 } + t } - w _ { \star } \| ^ { 2 } \geq \| \mathbb { E } [ w _ { t _ { 0 } + t } - w _ { \star } ] \| ^ { 2 } } & & { \mathrm { b y ~ J e n s e n ' s ~ i n e q u a l i t y } } \\ & { \geq \big ( u ^ { \top } \mathbb { E } [ w _ { t _ { 0 } + t } - w _ { \star } ] \big ) ^ { 2 } } & & { \mathrm { b y ~ t h e ~ C a u c h y - S c h w a r z ~ i n e q u a l i t y } } \\ & { = ( 1 - \eta \lambda ) ^ { 2 t } \big ( u ^ { \top } \mathbb { E } [ w _ { t _ { 0 } } - w _ { \star } ] \big ) ^ { 2 } } & & { \mathrm { b y ~ t h e ~ d i s p l a y ~ a b o v e } } \\ & { \to \infty . } & & { 1 - \eta \lambda > 1 } \end{array}
$$

The same computation with $\mathbf { \Lambda } _ { \mathbf { r } _ { t } } \equiv \mathbf { \Lambda } _ { \mathbf { 0 } }$ shows that the homogeneous system $\bar { \boldsymbol { \epsilon } } _ { t + 1 } = M _ { t } \bar { \boldsymbol { \epsilon } } _ { t }$ is not mean-square stable. The population quadratic evaluated at $\pmb { w } _ { \star } + a \pmb { u }$ is a quadratic polynomial in a whose $a ^ { 2 }$ coefficient is $\begin{array} { r } { \frac { 1 } { 2 } \mathbf { \dot { \boldsymbol { u } } } ^ { \top } \mathbfcal { H } _ { t _ { 0 } } \mathbf { \boldsymbol { u } } = \mathbf { \dot { \boldsymbol { \lambda } } } / 2 < 0 . } \end{array}$ , so it is unbounded below. This completes the proof of Corollary C.6. □

## D.5 PROOF OF COROLLARY C.8

ProofofCorollary C.8. We use a PSD eigenmatrix of the adjoint second-moment map to identify an initialization with geometric second-moment growth. Recall from Appendix C.2 that $M _ { t } ^ { + } ( \dot { \eta } )$ is the transition of the projected homogeneous system, acting on the momentum and displacement restricted to the positive eigenspace, and that $\breve { K } ^ { + } ( \eta ) = \breve { \mathbb { E } } [ M _ { t } ^ { + } ( \eta ) \otimes M _ { t } ^ { + } ( \eta ) ]$ ]. By the unique stability threshold (Theorem C.7), $\rho ( K ^ { + } ( \eta ) ) > 1$ for every $\eta > \eta _ { \star }$ . Write this radius as $r > 1$ and define the second-moment map on symmetric matrices by

$$
\mathcal { T } ( X ) : = \mathbb { E } [ M _ { t } ^ { + } ( \eta ) X M _ { t } ^ { + } ( \eta ) ^ { \top } ] .
$$

Restriction to symmetric matrices preserves the spectral radius, so $\rho ( \mathcal T ) = r$ (see the discussion after Lemma F.14 in Appendix F.6). The adjoint map $\mathcal { T } ^ { * } ( X ) = \mathbb { E } [ M _ { t } ^ { + } ( \eta ) ^ { \top } X M _ { t } ^ { + } ( \eta ) ]$ ] preserves PSD matrices and has the same spectral radius as $\tau _ { \ast }$ , so by the Perron–Frobenius theorem for cone-preserving maps (Lemma F.16) it has a nonzero PSD eigenmatrix W with ${ \mathcal { T } } ^ { * } ( W ) = r W$ Choose a deterministic initial state $z _ { \mathrm { 0 } }$ of the projected system such that $z _ { 0 } ^ { \top } W z _ { 0 } > 0$ , for instance an eigenvector of W with positive eigenvalue. Its second moment $\begin{array} { r } { \pmb { X } _ { t } = \mathbb { E } [ \pmb { z } _ { t } \pmb { z } _ { t } ^ { \top } ] } \end{array}$ satisfies $X _ { t } =$ $\mathcal { T } ^ { t } ( z _ { 0 } \ z _ { 0 } ^ { \top } )$ , by the recursion $\mathbb { E } [ z _ { t + 1 } \breve { z _ { t + 1 } } ] = \mathcal { T } ( \mathbb { E } [ z _ { t } { z _ { t } ^ { \top } } ] )$ of the second-moment characterization

(Lemma F.14). Hence

$$
{ \begin{array} { r l r l } { \mathbb { E } \| z _ { t } \| ^ { 2 } = \operatorname { t r } ( X _ { t } ) } & & { \| z _ { t } \| ^ { 2 } = \operatorname { t r } ( z _ { t } z _ { t } ^ { \top } ) \cdot \mathbb { E } { \mathrm { a n d ~ t r ~ c o m m u t c } } } \\ & { \geq { \frac { \operatorname { t r } ( W X _ { t } ) } { \| W \| } } } & & { X _ { t } \succeq 0 { \mathrm { ~ a n d ~ } } 0 \leq W \leq \| W \| I } \\ & { = { \frac { \operatorname { t r } ( W ^ { T } ( z _ { 0 } z _ { 0 } ^ { \top } ) ) } { \| W \| } } } & & { X _ { t } = \mathcal { T } ^ { t } ( z _ { 0 } z _ { 0 } ^ { \top } ) } \\ & { = { \frac { \operatorname { t r } \left( \left( { T ^ { - } } \right) ^ { t } \left( W \right) z _ { 0 } z _ { 0 } ^ { \top } \right) } { \| W \| } } } & & { { \mathrm { ~ d e f i n i t i o n ~ o f ~ t h e ~ a d j o i n t , ~ a p p l i e d ~ t i m e s } } } \\ & { = { \frac { \operatorname { r } ^ { t } \operatorname { t r } ( W z _ { 0 } z _ { 0 } ^ { \top } ) } { \| W \| } } } & & { { T ^ { * } ( W ) = r W } } \\ & { = { \frac { r ^ { t } z _ { 0 } ^ { \top } W z _ { 0 } } { \| W \| } } } & & { { \mathrm { ~ c y c l i c i t y ~ o f ~ t h e ~ r a c e } } ; r > 1 { \mathrm { ~ a n d ~ } } z _ { 0 } ^ { \top } W z _ { 0 } > 0 } \end{array} }
$$

For $\mathrm { S G D } , \beta _ { 1 } = 0 , P = I$ and $\lambda _ { \mathrm { W D } } = 0$ , the projected state reduces to the restricted displacement $\pmb { x } _ { t } \in \mathrm { r a n g e } ( \Pi _ { t _ { 0 } } ^ { + } )$ with transition $M _ { t } ^ { + } ( \eta ) = I - \eta \Pi _ { t _ { 0 } } ^ { + } H _ { B _ { t } } \Pi _ { t _ { 0 } } ^ { + }$ , and the same argument gives the exploding initial displacement of the SGD case in Appendix C.2. This completes the proof of Corollary C.8. □

## D.6 PROOF OF THE AGM CRITICAL LR CHARACTERIZATION

Proof of Proposition A.3. We show that every LR below the spectral threshold has bounded affine moments, while no LR above it does so for every initialization. This identifies the supremum without making an assumption about the endpoint. Recall the projected affine system (18),

$$
z _ { t + 1 } ^ { + } = M _ { t } ^ { + } ( \eta ) z _ { t } ^ { + } + q _ { t } ^ { + } , \qquad K ^ { + } ( \eta ) = \mathbb { E } [ M _ { t } ^ { + } ( \eta ) \otimes M _ { t } ^ { + } ( \eta ) ] .
$$

The pairs $( M _ { t } ^ { + } ( \eta ) , q _ { t } ^ { + } )$ are i.i.d. with finite second moments by Assumption C.1. The transition and forcing within a pair may be correlated. The effective curvature A is positive definite on the subspace onto which $\Pi ^ { + }$ projects. By Corollary F.4 and the change of state in Appendix $\mathrm { F . } 6 , \rho ( K ^ { + } ( \eta ) )$ is less than, equal to, or greater than one below, at, or above a threshold $\eta _ { \mathrm { c r i t } } \in ( 0 , \infty )$

Step 1. LRs below the spectral threshold. If $0 < \eta < \eta _ { \mathrm { c r i t } }$ , then $\rho ( K ^ { + } ( \eta ) ) < 1$ . The affine moment recursion in Proposition C.3 therefore gives

$$
\operatorname* { s u p } _ { t \geq 0 } \mathbb { E } \| z _ { t } ^ { + } \| ^ { 2 } < \infty \qquad \mathrm { f o r ~ e v e r y ~ d e t e r m i n i s t i c ~ } z _ { 0 } ^ { + } .
$$

Thus every such LR belongs to the stable set in Definition A.2.

Step 2. LRs above the spectral threshold. Let $\eta > \eta _ { \mathrm { c r i t } }$ , so that $\rho ( K ^ { + } ( \eta ) ) > 1$ . The Perron– Frobenius argument in Appendix D.5 then gives a deterministic initial vector u whose homogeneous second moment diverges. Run two affine trajectories, $z _ { t } ^ { + }$ and $\widehat { z } _ { t } ^ { + }$ , with the same pairs $( M _ { t } ^ { + } \overline { { ( \eta ) } } , q _ { t } ^ { + } )$ and initial states $z _ { 0 } ^ { + } = u$ and $\widehat { z } _ { 0 } ^ { + } = 0$ . Their difference $\pmb { d } _ { t } : = \pmb { z } _ { t } ^ { + } - \widehat { \pmb { z } } _ { t } ^ { + }$ satisfies

$$
\begin{array} { r l r l } & { d _ { t + 1 } = \left( M _ { t } ^ { + } ( \eta ) z _ { t } ^ { + } + q _ { t } ^ { + } \right) - \left( M _ { t } ^ { + } ( \eta ) \widehat { z } _ { t } ^ { + } + q _ { t } ^ { + } \right) \quad } & & { \mathrm { u s e ~ t h e ~ s a m e ~ b a t c h e s ~ a n d ~ f o r c i n g } } \\ & { \quad \quad = M _ { t } ^ { + } ( \eta ) d _ { t } , \quad } & & { \mathrm { t h e ~ f o r c i n g ~ c a n c e l s ~ p a t h w i s e } } \end{array}
$$

If the affine system were bounded in mean square for every deterministic start, then

$$
\operatorname* { s u p } _ { t \geq 0 } \mathbb { E } \| d _ { t } \| ^ { 2 } \leq 2 \operatorname* { s u p } _ { t \geq 0 } \mathbb { E } \| z _ { t } ^ { + } \| ^ { 2 } + 2 \operatorname* { s u p } _ { t \geq 0 } \mathbb { E } \| \widehat { z } _ { t } ^ { + } \| ^ { 2 } \qquad \| a - b \| ^ { 2 } \leq 2 \| a \| ^ { 2 } + 2 \| b \| ^ { 2 }
$$

boundedness for the two chosen starts

contradicting the choice of u. Thus no $\eta > \eta _ { \mathrm { c r i t } }$ belongs to the stable set. Together, the two steps give

$$
( 0 , \eta _ { \mathrm { c r i t } } ) \subseteq \{ \eta > 0 : ( 1 8 ) \mathrm { i s ~ m e a n \mathrm { - } s q u a r e ~ s t a b l e } \} \subseteq ( 0 , \eta _ { \mathrm { c r i t } } ] .
$$

The middle set has supremum $\eta _ { \star }$ by Definition A.2, so $\eta _ { \star } = \eta _ { \mathrm { c r i t } }$ . The spectral characterization and the three cases in (22) now follow from Corollary F.4. At the endpoint, forcing can change boundedness without changing this supremum. For example, $x _ { t + 1 } = - x _ { t }$ has bounded squared iterates, whereas $x _ { t + 1 } = - x _ { t } + \xi _ { t }$ with independent uniform signs $\xi _ { t }$ satisfies

$$
\begin{array} { r l } & { \mathbb { E } [ x _ { t + 1 } ^ { 2 } ] = \mathbb { E } [ x _ { t } ^ { 2 } ] - 2 \mathbb { E } [ x _ { t } \xi _ { t } ] + \mathbb { E } [ \xi _ { t } ^ { 2 } ] } \\ & { \qquad = \mathbb { E } [ x _ { t } ^ { 2 } ] + 1 = x _ { 0 } ^ { 2 } + t + 1 . } \end{array}
$$

The second line uses independence, $\mathbb { E } [ \xi _ { t } ] = 0$ and $\xi _ { t } ^ { 2 } = 1$ , then iterates the recursion. Both have second-moment spectral radius one. This completes the proof of Proposition A.3. □

## E PROOF OF THE POWER-ITERATION BOUND

## E.1 THE BISECTION BOUND

We prove Theorem C.12 by bounding the two possible query errors, then tracking the search interval during bisection. Below the target shift, we show that the bottom-eigenvector component grows past the cutoff with high probability. Above the target, we bound the probability of ever crossing the cutoff.

Recall that $A = D _ { t _ { 0 } } + \lambda _ { \mathrm { W D } } I$ is the effective curvature and $\Delta _ { t } = A _ { B _ { t } } - A$ is its centered mini-batch noise. A query at shift λ, with inner stepsize $\eta _ { \mathrm { p } }$ and norm cutoff B, follows

$$
\begin{array} { r } { \pmb { x } _ { t + 1 } = \big [ \pmb { M } ( \lambda ) - \eta _ { \mathrm { p } } \pmb { \Delta } _ { t } \big ] \pmb { x } _ { t } , \qquad \pmb { M } ( \lambda ) = \pmb { I } - \eta _ { \mathrm { p } } ( \pmb { A } + \lambda \pmb { I } ) , \qquad \| \pmb { x } _ { 0 } \| = 1 . } \end{array}
$$

Under Assumption $\mathrm { C } . 1 0 , \lambda _ { \star } = - \lambda _ { \operatorname* { m i n } } ( A )$ . For this proof, define the largest eigenvalue of the mean transition and the first exit time by

$$
\begin{array} { r l } & { \rho ( \boldsymbol { \lambda } ) : = 1 + \eta _ { \mathrm { p } } ( \lambda _ { \star } - \boldsymbol { \lambda } ) , } \\ & { \quad \quad \tau : = \operatorname* { i n f } \{ t \ge 0 : \| \pmb { x } _ { t } \| > B \} , \qquad \operatorname* { i n f } \mathcal { O } = \infty . } \end{array}
$$

A query of horizon $T _ { \mathrm { p } }$ explodes precisely when $\tau \leq T _ { \mathrm { p } }$ . The filtration $\mathcal { F } _ { t }$ includes all previous queries and the first t observations of the current query, as defined in Appendix C.3. The query bounds below hold conditionally on the previous queries, because Assumption C.9 gives conditional centering and the same noise bound under this conditioning.

The next two lemmas control the query errors below and above $\lambda _ { \star }$ . Their proofs are given in Appendices E.2 and E.3. In the first lemma, write $\rho = \rho ( \lambda )$ .

Lemma E.1 (Explosion below $\lambda _ { \star } )$ . Let Assumptions C.9 and C.10 hold, let $\eta _ { \mathrm { p } } > 0 , B > 1$ $\lambda \in [ 0 , \lambda _ { \star } )$ , and let $T _ { \mathrm { p } } \geq 1$ be an integer with $B \rho ^ { - T _ { \mathrm { p } } } < \varepsilon$ . Then

$$
\mathbb { P } \{ \tau > T _ { \mathrm { p } } \} \le \exp \Biggl ( - \frac { ( \varepsilon - B \rho ^ { - T _ { \mathrm { p } } } ) ^ { 2 } ( \rho ^ { 2 } - 1 ) } { 2 \eta _ { \mathrm { p } } ^ { 2 } \sigma ^ { 2 } B ^ { 2 } } \Biggr ) .\tag{59}
$$

In particular, let $g > 0$ and let $e _ { \mathrm { p } }$ be the finite-horizon term of (45). $I f \lambda \leq \lambda _ { \star } - \operatorname* { m a x } \{ g , e _ { \mathrm { p } } \}$ , then

$$
\mathbb { P } \{ \tau > T _ { \mathrm { p } } \} \le \exp \Bigl ( - \frac { \varepsilon ^ { 2 } g } { 4 \eta _ { \mathrm { p } } \sigma ^ { 2 } B ^ { 2 } } \Bigr ) ,\tag{60}
$$

and for $g = g .$ <sub>−</sub> the right-hand side equals $\delta / N$

Lemma E.2 (No explosion above $\lambda _ { \star } )$ . Let Assumptions C.9 to $C . I I$ hold, let $B > 1$ , and fix $\lambda \in [ \lambda _ { \star } , \Lambda ]$ . With $a : = \eta _ { \mathrm { p } } ( \lambda - \lambda _ { \star } )$ and $h : = \eta _ { \mathrm { p } } \sigma _ { \mathrm { i } }$ , suppose $a > h ^ { 2 }$ . Then

$$
\begin{array} { r } { \mathbb { P } \big \{ \exists t \ge 0 : \| { \pmb x } _ { t } \| > B \big \} \le B ^ { - ( a - h ^ { 2 } ) / h ^ { 2 } } . } \end{array}
$$

$H a \geq 2 h ^ { 2 }$ , the right-hand side is at most $B ^ { - ( \lambda - \lambda _ { \star } ) / ( 2 \eta _ { \mathrm { p } } \sigma ^ { 2 } ) }$ . In particular, when $\lambda \ge \lambda _ { \star } + g _ { + }$ , with g<sub>+</sub> from (45), the probability is at most $\delta / N$

ProofofTheorem C.12. Recall that $g _ { - }$ and $g _ { + }$ in (45) are the noise margins below and above $\lambda _ { \star }$ respectively. The finite horizon contributes $e _ { \mathrm { p } } = ( ( 2 B / \varepsilon ) ^ { 1 / T _ { \mathrm { p } } } - 1 ) / \eta _ { \mathrm { p } }$ . Fix an integer $T _ { \mathrm { p } } \geq 1$

## Step 1. Control the queries outside the error margins.

By the gap form (60) of Lemma E.1 with $g = g _ { - } , \mathtt { a }$ query at $\lambda \leq \lambda _ { \star } - \operatorname* { m a x } \{ g _ { - } , e _ { \mathrm { p } } \}$ explodes with probability at least $1 - \delta / N$ . By Lemma E.2, a query at $\lambda \ge \lambda _ { \star } + g _ { + }$ does not explode with probability at least $1 - \delta / N$

For the kth query, let $\mathcal { G } _ { k - 1 }$ contain all observations of the preceding queries, and let $E _ { k }$ be the event that this query returns the wrong outcome in either of these two regions. Conditionally on $\mathcal { G } _ { k - 1 }$ , the queried shift is fixed and Assumption C.9 still holds. The two regions are disjoint, so the query can incur only one of the two errors. Therefore

$$
\mathbb { P } ( E _ { k } \mid { \mathcal { G } } _ { k - 1 } ) \leq { \frac { \delta } { N } } \qquad { \mathrm { a l m o s t ~ s u r e l y } } .
$$

Taking expectations and then a union bound gives

$$
\begin{array} { r l } { \mathbb { P } \Big ( \displaystyle \bigcup _ { k = 1 } ^ { N } E _ { k } \Big ) \leq \displaystyle \sum _ { k = 1 } ^ { N } \mathbb { P } ( E _ { k } ) } & { \qquad \mathrm { u n i o n ~ b o u n d } } \\ { = \displaystyle \sum _ { k = 1 } ^ { N } \mathbb { E } \big [ \mathbb { P } ( E _ { k } \mid \mathcal { G } _ { k - 1 } ) \big ] } & { \qquad \mathrm { t o w e r ~ p r o p e r t y } } \\ { \leq \displaystyle \sum _ { k = 1 } ^ { N } \frac { \delta } { N } = \delta . } & { \qquad \mathrm { t h e ~ c o n d i t i o n a l ~ b o u n d } } \end{array}
$$

Thus, with probability at least $1 - \delta ,$ , every query in either region has the stated outcome. Queries strictly between the two regions may return either outcome.

## Step 2. The search interval.

On this event, an exploding query has $\lambda < \lambda _ { \star } + g _ { + }$ and a non-exploding query has $\lambda > \lambda _ { \star } -$ $\operatorname* { m a x } \{ g _ { - } , e _ { \mathrm { p } } \}$ . Every lower endpoint of the search interval is either 0 or an exploding query, hence at most $\lambda _ { \star } + g _ { + }$ . Every upper endpoint is either Λ, which is at least $\lambda _ { \star }$ by Assumption C.10, or a non-exploding query, hence at least $\lambda _ { \star } - \operatorname* { m a x } \{ g _ { - } , e _ { \mathrm { p } } \}$ . After N halvings the search interval has length $\Lambda / 2 ^ { N }$ , so

$$
\lambda _ { \star } - \operatorname* { m a x } \{ g _ { - } , e _ { \mathrm { p } } \} \leq \widehat { \lambda } = \lambda _ { \mathrm { l o } } + \frac { \Lambda } { 2 ^ { N } } \leq \lambda _ { \star } + g _ { + } + \frac { \Lambda } { 2 ^ { N } } ,
$$

which is (46).

## Step 3. The crossover.

By the definition of $e _ { \mathrm { p } } ,$

$$
\begin{array} { r l } { \displaystyle e _ { \mathrm { p } } \leq g _ { - } \iff } & { \iff ( 2 B / \varepsilon ) ^ { 1 / T _ { \mathrm { p } } } \leq 1 + \eta _ { \mathrm { p } } g _ { - } } \\ { \iff } & { \frac { \log ( 2 B / \varepsilon ) } { T _ { \mathrm { p } } } \leq \log ( 1 + \eta _ { \mathrm { p } } g _ { - } ) } \\ { \iff } & { T _ { \mathrm { p } } \geq \frac { \log ( 2 B / \varepsilon ) } { \log ( 1 + \eta _ { \mathrm { p } } g _ { - } ) } . } \end{array}
$$

The second equivalence takes logarithms of positive quantities. Since $T _ { \mathrm { p } }$ is an integer, this is equivalent to the ceiling condition in (47). This completes the proof of Theorem C.12. □

## E.2 PROOF OF THE EXPLOSION BOUND

Fix a query at a shift $\lambda \in [ 0 , \lambda _ { \star } )$ . Recall that $\rho = \rho ( \lambda ) = 1 + \eta _ { \mathrm { p } } ( \lambda _ { \star } - \lambda ) > 1$ is the largest eigenvalue of $M ( \lambda )$ , attained at $\mathbf { \delta } \mathbf { u } _ { \mathrm { m i n } } ,$ , so that $\begin{array} { r } { \pmb { u } _ { \mathrm { m i n } } ^ { \top } \pmb { M } ( \lambda ) = \rho \pmb { u } _ { \mathrm { m i n } } ^ { \top } . } \end{array}$ , and that τ is the first time at which $\| \boldsymbol { x } _ { t } \|$ exceeds the cutoff B. Choose the sign of $\mathbf { \delta } \mathbf { u } _ { \mathrm { m i n } }$ so that ${ \pmb u } _ { \mathrm { m i n } } ^ { \mathrm { ~ ! ~ } } { \pmb x } _ { 0 } \ge \varepsilon$ , which Assumption C.10 allows.

ProofofLemma E.1. The idea is to divide the bottom-eigenvector component by its mean growth factor $\rho ^ { t }$ and stop it at τ. The resulting martingale has increment bounds whose squares form a summable geometric sequence. The Azuma–Hoeffding inequality then bounds the probability that this component falls as far as the event $\tau > T _ { \mathrm { p } }$ would require.

## Step 1. A stopped martingale with summable increment bounds.

Set $Z _ { t } : = \rho ^ { - t } \pmb { u } _ { \mathrm { m i n } } ^ { \top } \pmb { x } _ { t }$ . By the iteration (42),

$$
\begin{array} { r l } & { Z _ { t + 1 } - Z _ { t } = \rho ^ { - ( t + 1 ) } \pmb { u } _ { \operatorname* { m i n } } ^ { \top } \big ( \pmb { M } ( \lambda ) - \eta _ { \mathrm { p } } \pmb { \Delta } _ { t } \big ) \pmb { x } _ { t } - \rho ^ { - t } \pmb { u } _ { \operatorname* { m i n } } ^ { \top } \pmb { x } _ { t } } \\ & { \qquad = \big ( \rho ^ { - ( t + 1 ) } \rho - \rho ^ { - t } \big ) \pmb { u } _ { \operatorname* { m i n } } ^ { \top } \pmb { x } _ { t } - \eta _ { \mathrm { p } } \rho ^ { - ( t + 1 ) } \pmb { u } _ { \operatorname* { m i n } } ^ { \top } \pmb { \Delta } _ { t } \pmb { x } _ { t } } \\ & { \qquad = - \eta _ { \mathrm { p } } \rho ^ { - ( t + 1 ) } \pmb { u } _ { \operatorname* { m i n } } ^ { \top } \pmb { \Delta } _ { t } \pmb { x } _ { t } . } \end{array}
$$

The second equality uses $\pmb { u } _ { \mathrm { m i n } } ^ { \top } \pmb { M } ( \lambda ) = \rho \pmb { u } _ { \mathrm { m i n } } ^ { \top }$ . Since $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ is $\mathcal { F } _ { t }$ -measurable, the conditional increment is

$$
\begin{array} { r l r l } & { \mathbb { E } [ Z _ { t + 1 } - Z _ { t } \mid \mathcal { F } _ { t } ] = - \eta _ { \mathrm { p } } \rho ^ { - ( t + 1 ) } \pmb { u } _ { \mathrm { m i n } } ^ { \top } \mathbb { E } [ \Delta _ { t } \mid \mathcal { F } _ { t } ] \pmb { x } _ { t } \quad } & & { \mathrm { c o n d i t i o n a l ~ m e a s u r a b i l i t y } } \\ & { \quad \quad = 0 . } & & { \mathrm { A s s u m p t i o n } \mathbb { C } . 9 } \end{array}
$$

The bounded noise makes the state, and hence $Z _ { t }$ , integrable at each finite time. Thus $( Z _ { t } ) _ { t \geq 0 }$ is a martingale with $Z _ { 0 } = { \pmb u } _ { \mathrm { m i n } } ^ { \top } { \pmb x } _ { 0 } \ge \varepsilon$

We stop it at $\tau .$ . The event $\smash { \{ t < \tau \} } = \{ \operatorname* { m a x } _ { s \leq t } \| \pmb { x } _ { s } \| \leq B \}$ is $\mathcal { F } _ { t }$ -measurable. The stopped increments therefore satisfy

$$
\begin{array} { r l } & { \boldsymbol Z _ { ( t + 1 ) \wedge \tau } - \boldsymbol Z _ { t \wedge \tau } = \mathbf 1 _ { \{ t < \tau \} } ( \boldsymbol Z _ { t + 1 } - \boldsymbol Z _ { t } ) , } \\ & { \mathbb E [ \boldsymbol Z _ { ( t + 1 ) \wedge \tau } - \boldsymbol Z _ { t \wedge \tau } \mid \mathcal F _ { t } ] = \mathbf 1 _ { \{ t < \tau \} } \mathbb E [ \boldsymbol Z _ { t + 1 } - \boldsymbol Z _ { t } \mid \mathcal F _ { t } ] } \\ & { \qquad = 0 . } \end{array}
$$

Thus the stopped process is again a martingale. On $\{ t < \tau \}$ we have $\| \pmb { x } _ { t } \| \le B$ , so

$$
\begin{array} { r l r l } { \left| Z _ { ( t + 1 ) \wedge \tau } - Z _ { t \wedge \tau } \right| = \mathbf { 1 } _ { \left\{ t < \tau \right\} } \eta _ { \mathrm { p } } \rho ^ { - ( t + 1 ) } | u _ { \mathrm { m i n } } ^ { \top } \Delta _ { t } x _ { t } | } & & { } \\ { \leq \mathbf { 1 } _ { \left\{ t < \tau \right\} } \eta _ { \mathrm { p } } \rho ^ { - ( t + 1 ) } \left\| u _ { \mathrm { m i n } } \right\| \left\| \Delta _ { t } \right\| \left\| x _ { t } \right\| } & & { \mathrm { C a u c h y - S c h w a r z } } \\ { \leq \eta _ { \mathrm { p } } \sigma B \rho ^ { - ( t + 1 ) } . } & & { \left\| x _ { t } \right\| \leq B \mathrm { ~ b e f o r e ~ } \tau } \end{array}
$$

Since $\rho > 1$ , extending the series of positive terms and summing it gives

$$
\begin{array} { r l } { \displaystyle \sum _ { t = 0 } ^ { T _ { \mathrm { p } } - 1 } \left( \eta _ { \mathrm { p } } \sigma B \rho ^ { - ( t + 1 ) } \right) ^ { 2 } = \eta _ { \mathrm { p } } ^ { 2 } \sigma ^ { 2 } B ^ { 2 } \displaystyle \sum _ { t = 1 } ^ { T _ { \mathrm { p } } } \rho ^ { - 2 t } } & { \qquad \mathrm { r e i n d e x } } \\ { \displaystyle \le \eta _ { \mathrm { p } } ^ { 2 } \sigma ^ { 2 } B ^ { 2 } \displaystyle \sum _ { t = 1 } ^ { \infty } \rho ^ { - 2 t } } & { \qquad \mathrm { n o m e g a t i v e ~ s u m m a n d s } } \\ { \displaystyle = \eta _ { \mathrm { p } } ^ { 2 } \sigma ^ { 2 } B ^ { 2 } \cdot \frac { \rho ^ { - 2 } } { 1 - \rho ^ { - 2 } } } & { \qquad \mathrm { g e o m e t r i c ~ s e r i e s , ~ } \rho > 1 } \\ { \displaystyle = \frac { \eta _ { \mathrm { p } } ^ { 2 } \sigma ^ { 2 } B ^ { 2 } } { \rho ^ { 2 } - 1 } . } & { \qquad \mathrm { m u l t i p l y ~ b y ~ } \rho ^ { 2 } / \rho ^ { 2 } } \end{array}
$$

## Step 2. The Azuma–Hoeffding bound.

Suppose $\tau > T _ { \mathrm { p } }$ . Then $T _ { \mathrm { p } } \wedge \tau = T _ { \mathrm { p } }$ and $\| \pmb { x } _ { T _ { \mathrm { p } } } \| \le B _ { \mathrm { \ell } }$ , so by the Cauchy–Schwarz inequality

$$
\begin{array} { r l } & { Z _ { T _ { \mathrm { p } } \wedge \tau } = Z _ { T _ { \mathrm { p } } } = \rho ^ { - T _ { \mathrm { p } } } \pmb { u } _ { \mathrm { m i n } } ^ { \top } \pmb { x } _ { T _ { \mathrm { p } } } } \\ & { \qquad \leq \rho ^ { - T _ { \mathrm { p } } } \lVert \pmb { x } _ { T _ { \mathrm { p } } } \rVert } \\ & { \qquad \leq B \rho ^ { - T _ { \mathrm { p } } } . } \end{array}
$$

Since $Z _ { 0 } \geq \varepsilon ,$ , it follows that

$$
Z _ { T _ { \mathrm { p } } \wedge \tau } - Z _ { 0 } \leq B \rho ^ { - T _ { \mathrm { p } } } - \varepsilon < 0 .
$$

By the Azuma–Hoeffding inequality (Lemma E.5), applied to the martingale $( Z _ { t \wedge \tau } )$ with the increment bounds above and with $\dot { r ^ { \mathrm { ~ } } } = \varepsilon ^ { \mathrm { ~ } } - B \rho ^ { - T _ { \mathrm { P } } } > 0$

$$
\begin{array} { r l } & { \mathbb { P } \{ \tau > T _ { \mathrm { p } } \} \le \mathbb { P } \{ Z _ { T _ { \mathrm { p } } \wedge \tau } - Z _ { 0 } \le - ( \varepsilon - B \rho ^ { - T _ { \mathrm { p } } } ) \} } \\ & { \qquad \le \exp \biggl ( - \frac { ( \varepsilon - B \rho ^ { - T _ { \mathrm { p } } } ) ^ { 2 } } { 2 \eta _ { \mathrm { p } } ^ { 2 } \sigma ^ { 2 } B ^ { 2 } / ( \rho ^ { 2 } - 1 ) } \biggr ) , } \end{array}
$$

which is (59).

## Step 3. The gap form.

Suppose $\lambda \leq \lambda _ { \star } - \operatorname* { m a x } \{ g , e _ { \mathrm { p } } \}$ . Recall that $e _ { \mathrm { p } } = ( ( 2 B / \varepsilon ) ^ { 1 / T _ { \mathrm { p } } } - 1 ) / \eta _ { \mathrm { p } } , \mathrm { s o } 1 + \eta _ { \mathrm { p } } e _ { \mathrm { p } } = ( 2 B / \varepsilon ) ^ { 1 / T _ { \mathrm { p } } } .$ Then

$$
\begin{array} { c } { \rho = 1 + \eta _ { \mathrm { p } } ( \lambda _ { \star } - \lambda ) \geq 1 + \eta _ { \mathrm { p } } \operatorname* { m a x } \{ g , e _ { \mathrm { p } } \} , } \\ { B \rho ^ { - T _ { \mathrm { p } } } \leq B ( 1 + \eta _ { \mathrm { p } } e _ { \mathrm { p } } ) ^ { - T _ { \mathrm { p } } } } \\ { = B \cdot \displaystyle \frac { \varepsilon } { 2 B } = \displaystyle \frac { \varepsilon } { 2 } , } \\ { \rho ^ { 2 } - 1 = ( \rho - 1 ) ( \rho + 1 ) \geq 2 ( \rho - 1 ) \geq 2 \eta _ { \mathrm { p } } g . } \end{array}
$$

The last line uses $\rho \geq 1$ and $\rho \ge 1 + \eta _ { \mathrm { p } } g$ . Substituting these two bounds into the exponent of (59),

$$
\begin{array} { r } { \frac { ( \varepsilon - B \rho ^ { - T _ { \mathrm { p } } } ) ^ { 2 } ( \rho ^ { 2 } - 1 ) } { 2 \eta _ { \mathrm { p } } ^ { 2 } \sigma ^ { 2 } B ^ { 2 } } \geq \frac { ( \varepsilon / 2 ) ^ { 2 } \cdot 2 \eta _ { \mathrm { p } } g } { 2 \eta _ { \mathrm { p } } ^ { 2 } \sigma ^ { 2 } B ^ { 2 } } } \\ { = \frac { \varepsilon ^ { 2 } g } { 4 \eta _ { \mathrm { p } } \sigma ^ { 2 } B ^ { 2 } } , } \end{array}
$$

which gives (60). Recall $g _ { - } = 4 \eta _ { \mathrm { p } } \sigma ^ { 2 } B ^ { 2 } \log ( N / \delta ) / \varepsilon ^ { 2 }$ . Substituting this value gives

$$
\begin{array} { r l r } {  { \frac { \varepsilon ^ { 2 } g _ { - } } { 4 \eta _ { \mathrm { p } } \sigma ^ { 2 } B ^ { 2 } } = \frac { \varepsilon ^ { 2 } } { 4 \eta _ { \mathrm { p } } \sigma ^ { 2 } B ^ { 2 } } \frac { 4 \eta _ { \mathrm { p } } \sigma ^ { 2 } B ^ { 2 } } { \varepsilon ^ { 2 } } \log \frac { N } { \delta } } } \\ & { } & \\ & { } & { = \log \frac { N } { \delta } , } \\ & { } & { \mathbb { P } \{ \tau > T _ { \mathrm { p } } \} \le \exp ( - \log \frac { N } { \delta } ) = \frac { \delta } { N } . } \end{array}
$$

This completes the proof of Lemma E.1.

The horizon enters (59) through $B \rho ^ { - T _ { \mathrm { p } } }$ . The definition of $e _ { \mathrm { p } }$ ensures that this quantity is at most $\varepsilon / 2$

## E.3 PROOF OF THE NO-EXPLOSION BOUND

Fix a query at $\lambda \in [ \lambda _ { \star } , \Lambda ]$ . In the adaptive application, we condition on $\mathcal { F } _ { 0 }$ , which contains the previous queries, so this shift is fixed throughout the proof. Recall that the update is $\pmb { x } _ { t + 1 } =$ $[ M ( \lambda ) - \eta _ { \mathrm { p } } \Delta _ { t } ] \pmb { x } _ { t }$ , where $M ( \lambda ) = I - \eta _ { \mathrm { p } } ( \mathbf { \bar { A } } + \lambda I )$ and $\Delta _ { t } = A _ { B _ { t } } - A$ has conditional mean zero and norm at most σ. Define the contraction margin and the noise bound per step by

$$
a : = \eta _ { \mathrm { p } } ( \lambda - \lambda _ { \star } ) , \qquad h : = \eta _ { \mathrm { p } } \sigma .
$$

The margin lies in [0, 1]. Indeed, since $\lambda _ { \star } = - \lambda _ { \operatorname* { m i n } } ( A )$

$$
\begin{array} { r l r } {  { 0 \leq a = \eta _ { \mathrm { p } } \bigl ( \lambda + \lambda _ { \operatorname* { m i n } } ( A ) \bigr ) } } \\ & { } & { \leq \eta _ { \mathrm { p } } \bigl ( \Lambda + \lambda _ { \operatorname* { m a x } } ( A ) \bigr ) } \\ & { } & { \leq 1 . } \end{array}
$$

Assumption C.11

The same assumption bounds the eigenvalues of the mean transition:

$$
0 \leq 1 - \eta _ { \mathrm { p } } \big ( \lambda _ { \operatorname* { m a x } } ( A ) + \Lambda \big ) \leq \lambda _ { i } \big ( M ( \lambda ) \big ) \leq 1 - \eta _ { \mathrm { p } } \big ( \lambda _ { \operatorname* { m i n } } ( A ) + \lambda \big ) = 1 - a .
$$

The upper endpoint is attained at $\mathbf { \delta } \mathbf { u } _ { \mathrm { m i n } } .$ , so

$$
\| M ( \lambda ) \| = 1 - a .\tag{61}
$$

We use this contraction to control the chance that curvature noise pushes the trajectory past the cutoff.

ProofofLemma E.2. We choose a positive power of the norm whose conditional expectation cannot increase. Hoeffding’s lemma controls the curvature noise in one step, and Ville’s maximal inequality then controls the probability of ever crossing the cutoff

## Step 1. Expand the one-step change in the norm.

For $\mathbf { \boldsymbol { x } } _ { t } \neq \mathbf { \boldsymbol { 0 } }$ , set $\hat { \pmb x } _ { t } : = { \pmb x } _ { t } / \| { \pmb x } _ { t } \|$ . Define

$$
\xi _ { t } : = - 2 \eta _ { \mathrm { p } } \langle M ( \lambda ) \hat { \pmb x } _ { t } , \pmb \Delta _ { t } \hat { \pmb x } _ { t } \rangle .
$$

If $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } } = \mathbf { 0 }$ , set both $\hat { \mathbf { x } } _ { t }$ and $\xi _ { t }$ to zero. Then $\pmb { x } _ { t + 1 } = \mathbf { 0 }$ and the norm bounds below remain valid. For $\mathbf { \boldsymbol { x } } _ { t } \neq \mathbf { \boldsymbol { 0 } }$ , expanding the update gives

$$
\begin{array} { r l r l } & { \frac { \Vert x _ { t + 1 } \Vert ^ { 2 } } { \Vert x _ { t } \Vert ^ { 2 } } = \Vert M ( \lambda ) \hat { x } _ { t } - \eta _ { \mathrm { p } } \Delta _ { t } \hat { x } _ { t } \Vert ^ { 2 } } & & { \quad \mathrm { t h e ~ u p d a t e } } \\ & { \quad = \Vert M ( \lambda ) \hat { x } _ { t } \Vert ^ { 2 } - 2 \eta _ { \mathrm { p } } \langle M ( \lambda ) \hat { x } _ { t } , \Delta _ { t } \hat { x } _ { t } \rangle } \\ & { \quad \quad \quad + \eta _ { \mathrm { p } } ^ { 2 } \Vert \Delta _ { t } \hat { x } _ { t } \Vert ^ { 2 } } & & { \quad \mathrm { e x p a n d ~ t h e ~ s q u a r e } } \\ & { \quad = \Vert M ( \lambda ) \hat { x } _ { t } \Vert ^ { 2 } + \xi _ { t } + \eta _ { \mathrm { p } } ^ { 2 } \Vert \Delta _ { t } \hat { x } _ { t } \Vert ^ { 2 } } & & { \quad \mathrm { t h e ~ d e f i n i t i o n ~ o f ~ } \xi _ { t } } \\ & { \quad \le \left( 1 - a \right) ^ { 2 } + \xi _ { t } + h ^ { 2 } } & & { \quad \mathrm { ( 6 ) ~ a n d ~ } \Vert \Delta _ { t } \Vert \le \sigma } \\ & { \quad \quad = 1 - 2 a + a ^ { 2 } + \xi _ { t } + h ^ { 2 } } & & { \quad \mathrm { e x p a n d ~ } ( 1 - a ) ^ { 2 } } \\ & { \quad \quad \le 1 + \xi _ { t } - ( a - h ^ { 2 } ) } & & { \quad \mathrm { ~ 0 \le ~ a \le 1 ~ } } \\ & { \quad \quad \le \mathrm { e x p } { \left( \xi _ { t } - ( a - h ^ { 2 } ) \right) } } & & { \quad 1 + y \le e ^ { y } } \end{array}
$$

The conditional mean of $\xi _ { t }$ is zero. The queried shift and $\hat { \mathbf { x } } _ { t }$ are $\mathcal { F } _ { t }$ -measurable, so Assumption C.9 gives

$$
\begin{array} { r l } { \mathbb { E } [ \xi _ { t } \mid \mathcal { F } _ { t } ] = - 2 \eta _ { \mathrm { p } } \mathbb { E } \big [ \langle M ( \lambda ) \hat { { \boldsymbol x } } _ { t } , \Delta _ { t } \hat { { \boldsymbol x } } _ { t } \rangle \mid \mathcal { F } _ { t } \big ] } & { } \\ { = - 2 \eta _ { \mathrm { p } } \langle M ( \lambda ) \hat { { \boldsymbol x } } _ { t } , \mathbb { E } [ \Delta _ { t } \mid \mathcal { F } _ { t } ] \hat { { \boldsymbol x } } _ { t } \rangle } & { \qquad \mathrm { c o n d i t i o n a l ~ m e a s u r a b i l i t y } } \\ { = 0 . \qquad } & { \mathrm { A s s u m p t i o n ~ } \mathrm { C . 9 } } \end{array}
$$

Its magnitude is bounded by 2h:

$$
\begin{array} { r l r l } & { | \xi _ { t } | \leq 2 \eta _ { \mathrm { p } } \| M ( \lambda ) \hat { x } _ { t } \| \| \Delta _ { t } \hat { x } _ { t } \| } & { \qquad } & { \mathrm { C a u c h y - S c h w a r z } } \\ & { \qquad \leq 2 \eta _ { \mathrm { p } } ( 1 - a ) \sigma } & { \qquad } & { \| \hat { x } _ { t } \| \leq 1 } \\ & { \qquad \leq 2 \eta _ { \mathrm { p } } \sigma = 2 h . } & { \qquad } & { 0 \leq a \leq 1 } \end{array}
$$

## Step 2. Construct a nonnegative supermartingale.

Since $a > h ^ { 2 }$ and $h = \eta _ { \mathrm { p } } \sigma > 0$ , define

$$
p : = \frac { a - h ^ { 2 } } { 2 h ^ { 2 } } > 0 , \qquad Y _ { t } : = \| \pmb { x } _ { t } \| ^ { 2 p } .
$$

The preceding norm-ratio bound implies

$$
Y _ { t + 1 } \leq Y _ { t } \exp \bigl ( p \xi _ { t } - p ( a - h ^ { 2 } ) \bigr ) .
$$

When $\mathbf { \boldsymbol { x } } _ { t } = \mathbf { \boldsymbol { 0 } }$ , the next state is also zero, so this inequality holds with both sides zero without dividing by $\| \boldsymbol { x } _ { t } \| . \mathrm { B y }$ Hoeffding’s lemma (Lemma E.3), applied conditionally to $\xi _ { t }$ with $| \xi _ { t } | \leq 2 h$

$$
\begin{array} { r l r l } { \mathbb { E } [ Y _ { t + 1 } \mid \mathcal { F } _ { t } ] \leq \mathbb { E } \big [ Y _ { t } e ^ { p \xi _ { t } - p ( a - h ^ { 2 } ) } \mid \mathcal { F } _ { t } \big ] } & { \quad } & & { \mathrm { t h e ~ o n e - s t e p ~ b o u n d } } \\ { = Y _ { t } e ^ { - p ( a - h ^ { 2 } ) } \mathbb { E } [ e ^ { p \xi _ { t } } \mid \mathcal { F } _ { t } ] } & { \quad } & & { Y _ { t } \mathrm { i s } \mathcal { F } _ { t } \mathrm { - m e a s u r a b l e } } \\ { \leq Y _ { t } \exp \bigg ( - p ( a - h ^ { 2 } ) + \frac { p ^ { 2 } ( 2 h ) ^ { 2 } } { 2 } \bigg ) } & { \quad } & & { \mathrm { H o e f f d i n g ' s ~ l e m m a ~ } } \\ { = Y _ { t } \exp \bigg ( - \frac { ( a - h ^ { 2 } ) ^ { 2 } } { 2 h ^ { 2 } } + \frac { ( a - h ^ { 2 } ) ^ { 2 } } { 2 h ^ { 2 } } \bigg ) } & { \quad } & & { \mathrm { t h e ~ d e f i n i t i o n ~ o f } p } \\ { = Y _ { t } . } & { } \end{array}\tag{62}
$$

Because $Y _ { 0 } ~ = ~ 1$ , induction gives $\mathbb { E } [ Y _ { t } ] \ \leq \ 1$ for every finite t. Thus $( Y _ { t } ) _ { t \geq 0 }$ is an integrable nonnegative supermartingale.

## Step 3. Bound the probability of crossing the cutoff.

Since $p > 0$ , crossing B in norm is the same as crossing $B ^ { 2 p }$ for $Y _ { t }$ . Ville’s maximal inequality (Lemma E.4) gives

$$
\begin{array} { r l r l } & { \mathbb { P } \big \{ \exists t \geq 0 : \| \pmb { x } _ { t } \| > B \big \} = \mathbb { P } \big \{ \exists t \geq 0 : Y _ { t } > B ^ { 2 p } \big \} } \\ & { \qquad \leq \mathbb { P } \Big \{ \underset { t \geq 0 } { \operatorname* { s u p } } Y _ { t } \geq B ^ { 2 p } \Big \} } \\ & { \qquad \leq \frac { \mathbb { E } [ Y _ { 0 } ] } { B ^ { 2 p } } } & & { \mathrm { ~ V i l e ' s ~ m a x i m a l ~ i n e q u a l i t y } } \\ & { \qquad = B ^ { - 2 p } } & & { Y _ { 0 } = \| \alpha _ { 0 } \| ^ { 2 p } = 1 } \\ & { \qquad = B ^ { - ( a - h ^ { 2 } ) / h ^ { 2 } } . } & & { \mathrm { t h e ~ d e f i n i t i o n ~ o f ~ } p } \end{array}
$$

If $a \geq 2 h ^ { 2 }$ , then

$$
\begin{array} { l } { \displaystyle \frac { a - h ^ { 2 } } { h ^ { 2 } } \geq \frac { a } { 2 h ^ { 2 } } } \\ { \displaystyle = \frac { \eta _ { \mathrm { p } } \left( \lambda - \lambda _ { \star } \right) } { 2 \eta _ { \mathrm { p } } ^ { 2 } \sigma ^ { 2 } } = \frac { \lambda - \lambda _ { \star } } { 2 \eta _ { \mathrm { p } } \sigma ^ { 2 } } . } \end{array}
$$

Since $B > 1$ , the probability bound is therefore at most $B ^ { - ( \lambda - \lambda _ { \star } ) / ( 2 \eta _ { \mathrm { p } } \sigma ^ { 2 } ) }$ . Finally, recall $g _ { + } =$ $2 \eta _ { \mathrm { p } } \sigma ^ { 2 }$ max{1, log(N/δ)/ log B}. If $\lambda - \lambda _ { \star } \geq g _ { + }$ , its two defining lower bounds give

$$
\begin{array} { c } { a = \eta _ { \mathrm { p } } ( \lambda - \lambda _ { \star } ) \geq 2 \eta _ { \mathrm { p } } ^ { 2 } \sigma ^ { 2 } = 2 h ^ { 2 } , } \\ { \displaystyle \frac { \lambda - \lambda _ { \star } } { 2 \eta _ { \mathrm { p } } \sigma ^ { 2 } } \geq \frac { \log ( N / \delta ) } { \log B } . } \end{array}
$$

Consequently,

$$
\begin{array} { r l } & { B ^ { - ( \lambda - \lambda _ { \star } ) / ( 2 \eta _ { \mathrm { p } } \sigma ^ { 2 } ) } \le B ^ { - \log ( N / \delta ) / \log B } } \\ & { \qquad = \exp \bigl ( - \log ( N / \delta ) \bigr ) = \frac \delta N . } \end{array}
$$

This completes the proof of Lemma E.2.

## E.4 AUXILIARY LEMMAS

We state the three standard concentration results used above. Throughout, $( \mathcal { F } _ { t } ) _ { t \geq 0 }$ is a filtration.

Lemma E.3 (Hoeffding’s lemma, conditional form; Hoeffding (1963), see also Lemma 2.2 of Boucheron et al. (2013)). Let D be a random variable and G a sigma-algebra such that $\mathbb { E } [ D \mid { \mathcal { G } } ] = 0$ and $| D | \le a$ almost surelyfor a constant $a \geq 0$ . Thenfor every $\bar { \boldsymbol { \theta } } \in \mathbb { R }$

$$
\mathbb { E } \big [ e ^ { \theta D } \mid \mathcal { G } \big ] \leq e ^ { \theta ^ { 2 } a ^ { 2 } / 2 } .
$$

Lemma E.4 (Ville’s maximal inequality for nonnegative supermartingales; Ville (1939), Theorem 4.8.4 of Durrett (2019)). Let $( \bar { Y _ { t } } ) _ { t \geq 0 }$ be a nonnegative supermartingale with respect to $( \mathcal { F } _ { t } ) _ { t \geq 0 }$ Then for every $c > 0$

$$
\mathbb { P } \Big \{ \operatorname* { s u p } _ { t \geq 0 } Y _ { t } \geq c \Big \} \leq \frac { \mathbb { E } [ Y _ { 0 } ] } { c } .
$$

Lemma E.5 (Azuma–Hoeffding inequality; Azuma (1967); Hoeffding (1963)). Let $( D _ { t } ) _ { t \geq 0 }$ be random variables such that $D _ { t } i s \mathcal { F } _ { t + 1 }$ -measurable, $\mathbb { E } [ D _ { t } \mid { \mathcal { F } } _ { t } ] = 0 $ , and $| D _ { t } | \leq a _ { t }$ almost surely for constants $a _ { t } \geq 0$ . Let $\begin{array} { r } { S _ { t } : = \sum _ { s < t } \dot { D } _ { s } } \end{array}$ . For every integer $T _ { \mathrm { p } } \overset { \cdot } { \geq } 1$ with $\textstyle \sum _ { t < T _ { \mathrm { p } } } a _ { t } ^ { 2 } > 0$ and every $r > 0 ,$

$$
\mathbb { P } \{ S _ { T _ { \mathrm { p } } } \leq - r \} \leq \exp \biggl ( - \frac { r ^ { 2 } } { 2 \sum _ { t < T _ { \mathrm { p } } } a _ { t } ^ { 2 } } \biggr ) .
$$

## F PROOF OF THE COMPENSATED FINITE-HORIZON THRESHOLD BOUND

In this appendix, we give the detailed proof of Theorem C.16. The proof compares the projected critical LR, the compensated critical LR, and the finite-horizon detector output. Appendix F.1 gives the stability results used in this comparison. Appendix F.2 then proves the main bound from one compensation bound and one horizon bound. We prove these two bounds in Appendices F.3 and F.4. The proofs of the stability results and the auxiliary matrix facts follow in Appendices F.5 and F.6.

## F.1 STABILITY CRITERION

We first analyze a generic momentum recurrence. Let the mean be a symmetric positive definite $C \in \mathbb { S } _ { n }$ , let the parameters satisfy $- 1 < b < 1$ and $\alpha > 0$ , and let the noise matrices be i.i.d. and symmetric $\Delta _ { t }$ with $\mathbb { E } \pmb { \Delta } _ { t } = 0 , \mathbb { E } \left\| \pmb { \Delta } _ { t } \right\| _ { F } ^ { 2 } < \infty$ , independent of the deterministic initial pair $( { \pmb x } _ { 0 } , { \pmb x } _ { - 1 } ) \colon$

$$
\begin{array} { r } { \pmb { x } _ { t + 1 } = \big [ ( 1 + b ) \pmb { I } - \alpha ( \pmb { C } + \pmb { \Delta } _ { t } ) \big ] \pmb { x } _ { t } - b \pmb { x } _ { t - 1 } . } \end{array}\tag{63}
$$

The history state $\pmb { y } _ { t } : = ( \pmb { x } _ { t } , \pmb { x } _ { t - 1 } ) ^ { \top }$ evolves by $\pmb { y } _ { t + 1 } = \pmb { F } _ { t } \pmb { y } _ { t }$ with

$$
\begin{array} { r } { F _ { t } : = \left( \begin{array} { c c } { ( 1 + b ) I - \alpha ( C + \Delta _ { t } ) } & { - b I } \\ { I } & { 0 } \end{array} \right) , \qquad T ( S ) : = \mathbb { E } [ F _ { t } S F _ { t } ^ { \top } ] , } \end{array}\tag{64}
$$

where $\tau _ { \ast }$ , the second-moment map, sends the second moment of ${ \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \Xi } _ { \mathbf { } } \mathbf { \Lambda } _ { \mathbf { } } \mathbf { \Lambda } _ { \mathbf { } } \textbf { } _ { \mathbf { } } \textbf { } \textbf { } _ { \mathrm { } }$ to that of $\mathbf { \pmb { y } } _ { t + 1 }$ . Strict mean-square stability means that $\mathbb { E } \Vert \pmb { y } _ { t } \Vert ^ { 2 }  \hat { 0 }$ for every deterministic initial pair.

We write $\mathbb { S } _ { n }$ and $\mathbb { S } _ { n } ^ { + }$ for the symmetric and positive semidefinite $n \times n$ matrices, respectively, and $\langle X , Y \rangle = \operatorname { t r } ( X Y )$ for their Frobenius inner product. The largest eigenvalue of the generic mean is $L _ { C } : = \lambda _ { \operatorname* { m a x } } ( C )$ . This distinguishes it from the paper’s $L = \bar { \lambda _ { \mathrm { m a x } } } ( \bar { A } )$

The stability quotient q gives both an exact test for stability and a quantitative bound when the moment spectral radius may exceed one. The proofs are in Appendix F.5.

Recall the quotient and its maximum from (51)–(52):

$$
\begin{array} { r l r } {  { q _ { C , b } ( \pmb X ) = \frac { \mathbb { E } \operatorname { t r } ( \pmb X \Delta _ { t } \pmb X \Delta _ { t } ) + c _ { b } \operatorname { t r } ( \pmb X C X C ) - k _ { b } \| [ \pmb C , \pmb X ] \| _ { F } ^ { 2 } } { 2 \operatorname { t r } ( C \pmb X ^ { 2 } ) } , } } \\ & { } & { r _ { C } ( b ) = \operatorname* { m a x } _ { \pmb X \geq 0 , \ \| \pmb X \| _ { F } = 1 } q _ { C , b } ( \pmb X ) , \qquad c _ { b } = \frac { 1 - b } { 1 + b } , \quad k _ { b } = \frac { b } { 1 - b ^ { 2 } } . } \end{array}
$$

The first result compares the normalized inverse coefficient $( 1 - b ) / \alpha$ with the largest value of this quotient.

Theorem F.1 (Exact criterion). For the momentum recurrence (63) with $C \succ 0 , - 1 < b < 1 , \alpha > 0$ and i.i.d. symmetric centered noise offinite second moment, strict mean-square stability holds ifand only if

$$
\frac { 1 - b } { \alpha } > r _ { C } ( b ) .\tag{65}
$$

Moreover, $r _ { C } ( b ) \geq c _ { b } L _ { C } / 2 > 0$ . For fixed $b ,$ the stable coefficients form the interval $0 < \alpha <$ $( 1 - b ) / r _ { C } ( b )$

Proposition F.2 (A spectral-radius bound controls the stability quotient). For the same recurrence, suppose additionally that $0 \leq b < 1$ . Let $R \geq 1$ satisfy ${ \dot { \rho } } ( { \dot { \mathcal { T } } } ) \leq R .$ . For every $X \succeq 0$ with $\| \bar { \cal X } \| _ { F } = 1 , s e t \mu : = \mathrm { t r } ( { \cal C } { \cal X } ^ { 2 } ) > 0 .$ . Then

$$
q _ { C , b } ( \mathbf { X } ) - \frac { 1 - b } { \alpha } \leq ( R - 1 ) \left[ \frac { 2 b } { ( R + b ) \alpha } + \frac { ( 1 - b / R ) ^ { 2 } } { 2 \mu \alpha ^ { 2 } } \right] .\tag{66}
$$

The test matrix can be chosen freely and need not maximize $q . \mathrm { A t } R = 1$ , the result gives $q _ { C , b } ( \boldsymbol { X } ) \le$ $( 1 - b ) / \alpha$ . For $R > 1$ , it controls the amount by which this inequality can fail. When $b = 0 ;$ , the bound reduces to $( R - 1 ) / ( 2 \mu \alpha ^ { 2 } )$ . In Appendix F.4, the detector supplies $R = \rho _ { T }$ , and the critical test matrix converts the left side into a bound on the inverse LR.

Changing the momentum. First compare two momentum values while keeping the mean, the noise and the test matrix fixed. Recall $c _ { b } = ( 1 - b ) / ( 1 + b )$ and $k _ { b } = b / ( 1 - b ^ { 2 } )$ . For $- 1 < b ^ { \prime } \leq b < 1$ we have $c _ { b ^ { \prime } } \geq c _ { b }$ and $k _ { b ^ { \prime } } \leq k _ { b }$ . The noise term cancels when we subtract the two quotients, so for every nonzero PSD $\boldsymbol { X }$

$$
\begin{array} { l l } { q _ { C , b ^ { \prime } } ( \pmb { X } ) - q _ { C , b } ( \pmb { X } ) = \frac { ( c _ { b ^ { \prime } } - c _ { b } ) \mathrm { t r } ( \pmb { X } C \pmb { X } C ) + ( k _ { b } - k _ { b ^ { \prime } } ) \| [ \pmb { [ \pmb { C } , \pmb { X } ] } \| _ { F } ^ { 2 }  } {  2 \mathrm { t r } ( C \pmb { X } ^ { 2 } )  } } \\ { \geq 0 . } \end{array}\tag{67}
$$

Both coefficients and both quadratic terms in the numerator are nonnegative. Thus $q _ { C , b ^ { \prime } } ( \boldsymbol { X } ) \ge$ $q _ { C , b } ( \boldsymbol { X } )$ for every admissible test matrix X.

Shifting the mean. A scalar shift changes only the mean-dependent terms of the quotient. The following identity will be used in both the quantitative stability bound and the compensation comparison. Its proof is in Appendix F.5.2.

Lemma F.3 (The shift identity). Let $C \succ 0 o n \mathbb { R } ^ { n }$ with noise $\Delta _ { t }$ and momentum $b \in ( - 1 , 1 )$ , and $h \geq 0$ . The shifted mean $\dot { C } + h I$ carries the same noise. For every $X \in \mathbb { S } _ { n } ^ { + }$ with $\| \boldsymbol { X } \| _ { F } = 1$ and $\mu : = \operatorname { t r } ( C X ^ { 2 } )$

$$
q _ { C + h I , b } ( X ) = \frac { \mu q _ { C , b } ( X ) + c _ { b } h ( \mu + h / 2 ) } { \mu + h } .\tag{68}
$$

Application to AGMs. Consider two affine runs $( \epsilon _ { t } ^ { ( 1 ) } , m _ { t } ^ { ( 1 ) } )$ and $( \epsilon _ { t } ^ { ( 2 ) } , m _ { t } ^ { ( 2 ) } )$ driven by the same mini-batches. Define their preconditioned differences by

$$
\ b x _ { t } : = \ b P ^ { 1 / 2 } ( \epsilon _ { t } ^ { ( 1 ) } - \epsilon _ { t } ^ { ( 2 ) } ) , \qquad \ b n _ { t } : = \ b P ^ { - 1 / 2 } ( \ b m _ { t } ^ { ( 1 ) } - \ b m _ { t } ^ { ( 2 ) } ) .
$$

Subtracting the two copies of (49) cancels the common batch gradient and checkpoint drift. Thus

$$
\begin{array} { r l } & { \pmb { n } _ { t + 1 } = \beta _ { 1 } \pmb { n } _ { t } + ( 1 - \beta _ { 1 } ) ( \pmb { A } _ { \mathrm { P } } - \lambda _ { \mathrm { W D } } \pmb { I } + \pmb { \Delta } _ { t } ) \pmb { x } _ { t } , } \\ & { \pmb { x } _ { t + 1 } = ( 1 - \eta \lambda _ { \mathrm { W D } } ) \pmb { x } _ { t } - \eta \pmb { n } _ { t + 1 } . } \end{array}\tag{69}
$$

The curvature term follows from $P ^ { - 1 / 2 } ( { \pmb H } _ { B _ { t } } + \lambda _ { \mathrm { P } } { \pmb P } ) P ^ { - 1 / 2 } = A _ { B _ { t } } - \lambda _ { \mathrm { W D } } { \pmb I } + \lambda _ { \mathrm { P } } { \pmb I }$ . These are the homogeneous difference dynamics of the compensated augmented system in (35). Its state $\left( \eta \pmb { n } _ { t } , \pmb { x } _ { t } \right)$ is written here in the coordinates $\left( { \pmb x } _ { t } , { \pmb n } _ { t } \right)$ . To obtain a recurrence for the positions, set

$$
\alpha : = \eta ( 1 - \beta _ { 1 } ) , \qquad b : = \beta _ { 1 } ( 1 - \eta \lambda _ { \mathrm { W D } } ) .
$$

For $t \geq 1$ , the preceding position update gives $- \eta \pmb { n } _ { t } = \pmb { x } _ { t } - \big ( 1 - \eta \lambda _ { \mathrm { W D } } \big ) \pmb { x } _ { t - 1 }$ . Substituting this into (69),

$$
\begin{array} { r l } & { x _ { t + 1 } = ( 1 - \eta \lambda _ { \mathrm { W D } } ) x _ { t } - \eta \beta _ { 1 } n _ { t } - \alpha ( A _ { \mathrm { P } } - \lambda _ { \mathrm { W D } } I + \Delta _ { t } ) x _ { t } } \\ & { \qquad = ( 1 - \eta \lambda _ { \mathrm { W D } } ) x _ { t } + \beta _ { 1 } \big [ x _ { t } - ( 1 - \eta \lambda _ { \mathrm { W D } } ) x _ { t - 1 } \big ] - \alpha ( A _ { \mathrm { P } } + \Delta _ { t } ) x _ { t } + \alpha \lambda _ { \mathrm { W D } } x _ { t } } \\ & { \qquad = \big [ ( 1 + \beta _ { 1 } - \eta \lambda _ { \mathrm { W D } } + \alpha \lambda _ { \mathrm { W D } } ) I - \alpha ( A _ { \mathrm { P } } + \Delta _ { t } ) \big ] x _ { t } - b x _ { t - 1 } . } \end{array}
$$

The scalar coefficient simplifies because

$$
\begin{array} { r } { 1 + \beta _ { 1 } - \eta \lambda _ { \mathrm { W D } } + \alpha \lambda _ { \mathrm { W D } } = 1 + \beta _ { 1 } - \eta \lambda _ { \mathrm { W D } } + \eta ( 1 - \beta _ { 1 } ) \lambda _ { \mathrm { W D } } = 1 + \beta _ { 1 } ( 1 - \eta \lambda _ { \mathrm { W D } } ) = 1 + b . } \end{array}
$$

Thus the positions satisfy

$$
\begin{array} { r } { \pmb { x } _ { t + 1 } = \big [ ( 1 + b ) \pmb { I } - \alpha ( \pmb { A } _ { \mathrm { P } } + \pmb { \Delta } _ { t } ) \big ] \pmb { x } _ { t } - b \pmb { x } _ { t - 1 } . } \end{array}\tag{70}
$$

The effective curvature already includes weight decay. Its remaining contribution is absorbed into the effective momentum b. The LR enters through α and b. The elimination is valid for every $\eta ,$ including $\eta \lambda _ { \mathrm { W D } } = 1$ , where $b = 0$ and the recurrence has first order.

The generic criterion applies to two positive definite means. The compensated system (70) uses $C =$ $A _ { \mathsf { P } }$ and noise $\Delta _ { t }$ in dimension d. The projected system uses $C = A ^ { \dagger }$ and noise $\Delta _ { t } ^ { + }$ in dimension p, the dimension of the positive eigenspace. It is the homogeneous difference recurrence associated with the projected affine system in Definition A.2. In both systems, $( b , \alpha ) = ( \beta _ { 1 } ( 1 - \eta \lambda _ { \mathrm { W D } } ) , \eta ( 1 - \beta _ { 1 } ) )$ so that

$$
\frac { 1 - b } { \alpha } = \frac { 1 - \beta _ { 1 } + \beta _ { 1 } \eta \lambda _ { \mathrm { W D } } } { \eta ( 1 - \beta _ { 1 } ) } = \frac { 1 } { \eta } + \frac { \beta _ { 1 } \lambda _ { \mathrm { W D } } } { 1 - \beta _ { 1 } } .\tag{71}
$$

The normalized inverse LR $( 1 - b ) / \alpha$ and $1 / \eta$ differ by a constant.

For the two AGM instances, we write $\mathcal { T } _ { \eta }$ for the projected system and $\mathcal { T } _ { \eta } ^ { \mathsf { P } }$ for the compensated system. For either instance, this map has the same spectral radius as the corresponding second-moment operator for the augmented position–momentum state in Appendix C.1: the two states are related by an invertible linear change when $\eta \lambda _ { \mathrm { W D } } \neq 1$ , and both spectral radii depend continuously on η (Appendix F.6).

We now vary the LR along the AGM parameter curve, keeping $C ,$ the noise law, $\beta _ { 1 }$ and $\lambda _ { \mathrm { W D } }$ fixed. Here $\dot { 0 } \le \beta _ { 1 } < 1$ and $\lambda _ { \mathrm { W D } } \geq 0$ . Let $\eta _ { \mathrm { m a x } } : = ( 1 + \bar { \beta _ { 1 } } ) / \bar { ( \beta _ { 1 } \lambda _ { \mathrm { W D } } ) }$ when $\beta _ { 1 } \lambda _ { \mathrm { W D } } > 0$ , and $\eta _ { \mathrm { m a x } } : = \infty$ otherwise. This is the upper endpoint of the range where $b > - 1$ . Define

$$
G _ { C } ( \eta ) : = \frac { 1 } { \eta } + \frac { \beta _ { 1 } \lambda _ { \mathrm { W D } } } { 1 - \beta _ { 1 } } - r _ { C } \bigl ( \beta _ { 1 } ( 1 - \eta \lambda _ { \mathrm { W D } } ) \bigr ) , \qquad 0 < \eta < \eta _ { \mathrm { m a x } } .\tag{72}
$$

The next corollary shows that this function crosses zero once, so the fixed-parameter criterion also gives a unique critical LR along the AGM parameter curve.

Corollary F.4 (AGM critical LR). For an AGM instance of (63) with $C \succ 0 ,$ , the centered i.i.d. symmetric noise offinite second moment, and the parameter curve above, $G _ { C }$ has a unique root $\eta _ { \mathrm { c r i t } }$ The strictly mean-square stable LRs are $( 0 , \eta _ { \mathrm { c r i t } } ) .$ . The second-moment spectral radius equals one at $\eta _ { \mathrm { c r i t } }$ and is strictly greater than onefor every $\eta > \eta _ { \mathrm { c r i t } } .$ . For $0 < \eta _ { 2 } < \eta _ { 1 } < \eta _ { \mathrm { m a x } }$

$$
G _ { C } ( \eta _ { 2 } ) - G _ { C } ( \eta _ { 1 } ) \geq \frac { 1 } { \eta _ { 2 } } - \frac { 1 } { \eta _ { 1 } } .\tag{73}
$$

The corollary gives the projected and compensated thresholds $\eta _ { \star }$ and $\eta _ { \star } ^ { \mathsf { P } }$ from Definition $\mathrm { C } . 1 3 ,$ with critical effective momenta in $( - 1 , 1 )$ . These agree with the corresponding thresholds defined by bounded affine moments by Proposition A.3 and its argument for $A _ { \mathsf { P } } \succ 0$

At the unique root, write $b _ { \mathrm { c r i t } } : = \beta _ { 1 } ( 1 - \eta _ { \mathrm { c r i t } } \lambda _ { \mathrm { W D } } )$ . The normalized inverse rate equals the quotient maximum:

$$
r _ { C } ( b _ { \mathrm { c r i t } } ) = \frac { 1 - b _ { \mathrm { c r i t } } } { ( 1 - \beta _ { 1 } ) \eta _ { \mathrm { c r i t } } } = \frac { 1 } { \eta _ { \mathrm { c r i t } } } + \frac { \beta _ { 1 } \lambda _ { \mathrm { W D } } } { 1 - \beta _ { 1 } } .\tag{74}
$$

The offset $\beta _ { 1 } \lambda _ { \mathrm { W D } } / ( 1 - \beta _ { 1 } )$ is the same for the projected and compensated systems.

Relation to the exact SGD threshold. When $b = 0 ,$ , we have $c _ { 0 } = 1$ and $k _ { 0 } = 0$ . Centering of the noise gives

$$
q _ { C , 0 } ( X ) = \frac { \mathbb { E } \operatorname { t r } \bigl ( X ( C + \Delta _ { t } ) X ( C + \Delta _ { t } ) \bigr ) } { 2 \operatorname { t r } ( C X ^ { 2 } ) } .
$$

Under the sampling and positive-semidefinite sample-Hessian assumptions of Mulayoff & Michaeli (2024, Theorem 5), maximizing this quotient recovers their exact SGD threshold on the positivecurvature subspace. Our criterion extends it to momentum and allows indefinite curvature samples with a positive definite mean. For the normalized momentum update used here, $\alpha = \eta ( 1 - \beta _ { 1 } )$ . Thus, without weight decay,

$$
\eta _ { \mathrm { c r i t } } = \frac { 1 } { r _ { C } ( 0 ) } \quad \mathrm { f o r ~ S G D } , \qquad \eta _ { \mathrm { c r i t } } = \frac { 1 } { r _ { C } ( \beta _ { 1 } ) } \quad \mathrm { w i t h ~ m o m e n t u m } .
$$

Momentum changes the deterministic-curvature coefficient and adds a commutator term, so it does not universally rescale the SGD threshold. When $\beta _ { 1 } \lambda _ { \mathrm { W D } } > 0$ , the threshold is the unique root of $G _ { C }$ in (72).

The noise-free threshold. Set $\ \Delta _ { t } = 0$ while retaining the same mean $C _ { i }$ , whose largest eigenvalue is $L _ { C }$ . The noise term in $q _ { C , b }$ then vanishes. The remaining numerator is at most $c _ { b } L _ { C } \operatorname { t r } ( \bar { C } X ^ { 2 } )$ by (107), so division by its denominator gives $q _ { C , b } ( \boldsymbol { X } ) \leq c _ { b } \bar { L } _ { C } / 2$ for every admissible X. Equality holds at $\pmb { X } = \pmb { u } \pmb { u } ^ { \top }$ for a unit top eigenvector of C. Together with the rank-one lower bound,

$$
r _ { C } ( b ) = \frac { c _ { b } L _ { C } } { 2 } \mathrm { w i t h o u t n o i s e } , \qquad r ( b ) \geq \frac { c _ { b } L _ { C } } { 2 } \mathrm { f o r a r b i t r a r y n o i s e } .\tag{75}
$$

Thus noise cannot increase the critical LR. For the noise-free AGM system, the equality in the stability criterion becomes

$$
\begin{array} { r } { \alpha L _ { C } = 2 ( 1 + b ) , \qquad } \\ { \eta ( 1 - \beta _ { 1 } ) L _ { C } = 2 ( 1 + \beta _ { 1 } ) - 2 \beta _ { 1 } \lambda _ { \mathrm { W D } } \eta . } \end{array}
$$

Solving for the LR gives

$$
\eta _ { \mathrm { d e t } } = \frac { 2 ( 1 + \beta _ { 1 } ) } { ( 1 - \beta _ { 1 } ) L _ { C } + 2 \beta _ { 1 } \lambda _ { \mathrm { W D } } } ,\tag{76}
$$

the noise-free critical LR of this AGM system.

## F.2 PROOF OF THE THRESHOLD BOUND

The detector runs the compensated system, whereas EoS is defined by the projected system. We therefore compare three LRs: the detector output $\bar { \eta } _ { \star } ( M , T )$ , the compensated critical rate $\dot { \eta } _ { \star } ^ { \mathsf { P } }$ , and the projected critical rate $\eta _ { \star }$ . Their inverse difference splits into two terms:

$$
\frac { 1 } { \bar { \eta } _ { \star } ( M , T ) } - \frac { 1 } { \eta _ { \star } } = \underbrace { \left( \frac { 1 } { \bar { \eta } _ { \star } ( M , T ) } - \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } \right) } _ { \mathrm { f i n i t e - h o r i z o n ~ e r r o r } } + \underbrace { \left( \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } - \frac { 1 } { \eta _ { \star } } \right) } _ { \mathrm { c o m p e n s a t i o n ~ e r r o r } } .\tag{77}
$$

The first comes from observing a run for only T steps. The second remains even with an unlimited horizon, because compensation changes the system being tested. The theorem bounds both terms without assuming that the Hessian noise is small.

Proof from the two error bounds. The following propositions control the two terms in (77). They are proved in Appendices F.3 and F.4, respectively. Recall that $\mu _ { \star }$ is the curvature at a test matrix maximizing the projected stability quotient. The function Φ gives the scalar-shift correction, and $\delta _ { \mathrm { c o v } }$ is the maximum normalized noise mismatch between the projected and compensated systems. Their formulas are given in (53), (54), and (55), respectively.

Proposition F.5 (Compensation error). Under Assumption C.14,

$$
\operatorname* { m i n } \{ \Phi ( \mu _ { \star } ) , 0 \} \ \leq \ \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } - \frac { 1 } { \eta _ { \star } } \ \leq \ \operatorname* { m a x } \{ \Phi ( L ) + \delta _ { \mathrm { c o v } } , 0 \} .\tag{78}
$$

$I f \beta _ { 1 } \lambda _ { \mathrm { W D } } = 0 ;$ , the clipping can be omitted: $\Phi ( \mu _ { \star } ) \leq 1 / \eta _ { \star } ^ { \mathsf { P } } - 1 / \eta _ { \star } \leq \Phi ( L ) + \delta _ { \mathrm { c o v } } .$

Proposition F.6 (Finite-horizon error). Under Assumption C.14, let $\eta \in \mathcal { I }$ be accepted by the detector. $I f \eta \overset { - } { \leq } \eta _ { \star } ^ { \mathsf { P } }$ , then $1 / \eta \ge 1 / \eta _ { \star } ^ { \mathsf { P } }$ . Otherwise $b _ { \star } ^ { \mathsf { P } } \geq 0$ and

$$
\frac { 1 } { \eta } \geq \frac { 1 } { \eta _ { \star } ^ { \sf P } } - H _ { T } ,\tag{79}
$$

with H<sub>T</sub> given by (57). In particular, (79) holdsfor every accepted LR whenever $\eta _ { \star } ^ { \mathsf { P } } \in \mathcal { I }$

ProofofTheorem C.16. We bound the two inverse-rate differences in (77) separately and then add them. Recall that $\bar { \eta } _ { \star } ( M , T )$ is the supremum of the accepted rates in the search window. By Assumption $\mathsf { C } . 1 5 , \eta _ { \star } ^ { \mathsf { P } }$ belongs to that window and is accepted, so $\bar { \eta } _ { \star } ( M , T ) \geq \eta _ { \star } ^ { \mathsf { P } }$

By the finite-horizon bound (Proposition F.6), every accepted rate satisfies $1 / \eta \ge 1 / \eta _ { \star } ^ { \mathsf { P } } - H _ { T }$ . We now pass this inequality to the supremum without assuming that the accepted rates form an interval. Every accepted η is at most $\bar { \eta } _ { \star } ( \bar { M } , T )$ , so its reciprocal is at least $1 / \bar { \eta } _ { \star } ( \bar { M } , T )$ . Conversely, by the definition of a supremum, there is a sequence of accepted rates $\eta _ { j }  \bar { \eta } _ { \star } ( M , T )$ . The window is bounded and bounded away from zero, so

$$
\begin{array} { r l r } { \displaystyle \operatorname* { i n f } _ { \eta \in \mathbb { Z } } \frac { 1 } { \eta } \geq \frac { 1 } { \bar { \eta } _ { \star } ( M , T ) } , } & \\ { \displaystyle \operatorname* { i n f } _ { \eta \in \mathrm { \Theta } } \frac { 1 } { \eta } \leq \displaystyle \operatorname* { l i m } _ { j \to \infty } \frac { 1 } { \eta _ { j } } = \frac { 1 } { \bar { \eta } _ { \star } ( M , T ) } . } & \end{array}
$$

The two inequalities identify the infimum. Applying the bound for each accepted rate and then using $\bar { \eta } _ { \star } ( M , T ) \geq \eta _ { \star } ^ { \mathsf { P } }$ gives

$$
\begin{array} { c } { \displaystyle \frac { 1 } { \bar { \eta } _ { \star } ( M , T ) } = \operatorname* { i n f } _ { \eta \in \mathbb { Z } \atop \eta \mathrm { ~ a c c e p t e d } } \frac { 1 } { \eta } \geq \displaystyle \frac { 1 } { \eta _ { \star } ^ { \mathrm { p } } } - H _ { T } , } \\ { \displaystyle - H _ { T } \leq \frac { 1 } { \bar { \eta } _ { \star } ( M , T ) } - \frac { 1 } { \eta _ { \star } ^ { \mathrm { p } } } \leq 0 . } \end{array}
$$

The compensation bound (Proposition F.5) controls the remaining difference. Thus

$$
\begin{array} { r l } & { \qquad \frac { 1 } { \bar { \eta } _ { \star } ( M , T ) } - \frac { 1 } { \eta _ { \star } } = \left( \frac { 1 } { \bar { \eta } _ { \star } ( M , T ) } - \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } \right) + \left( \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } - \frac { 1 } { \eta _ { \star } } \right) , } \\ & { \operatorname* { m i n } \{ \Phi ( \mu _ { \star } ) , 0 \} - H _ { T } \leq \frac { 1 } { \bar { \eta } _ { \star } ( M , T ) } - \frac { 1 } { \eta _ { \star } } \leq \operatorname* { m a x } \{ \Phi ( L ) + \delta _ { \mathrm { c o v } } , 0 \} . } \end{array}
$$

When $\beta _ { 1 } \lambda _ { \mathrm { W D } } = 0$ , the compensation bound has no clipping, so neither does the sum. When $\beta _ { 1 } = 0$ the definitions give $b _ { \star } ^ { \mathsf { P } } = 0$ and $\alpha _ { \star } ^ { \mathsf { P } } = \eta _ { \star } ^ { \mathsf { P } }$ . The first term in the bracket defining $H _ { T }$ vanishes, and the second gives $\bar { H _ { T } } = \hat { ( } \rho _ { T } - 1 ) / ( 2 \hat { \mu } _ { \star } ^ { \sf P } ( \eta _ { \star } ^ { \sf P } ) ^ { 2 } )$ . Finally, with $x : = ( 2 / T ) \log ( M ( 1 + \beta _ { 1 } ) / ( 1 - \beta _ { 1 } ) )$

$$
\rho _ { T } - 1 = e ^ { x } - 1 \leq x e ^ { x } = O { \big ( } ( 1 + \log M ) / T { \big ) } \quad { \mathrm { a s ~ } } ( 1 + \log M ) / T \to 0 .
$$

The search window gives $0 \leq b _ { \star } ^ { \mathsf { P } } < 1$ , and $\rho _ { T } \geq 1$ . Thus the bracket in (57) has the fixed upper bound

$$
\begin{array} { r l } & { \frac { 2 b _ { \star } ^ { \mathsf { P } } } { ( \rho _ { T } + b _ { \star } ^ { \mathsf { P } } ) \alpha _ { \star } ^ { \mathsf { P } } } + \frac { ( 1 - b _ { \star } ^ { \mathsf { P } } / \rho _ { T } ) ^ { 2 } } { 2 \mu _ { \star } ^ { \mathsf { P } } ( \alpha _ { \star } ^ { \mathsf { P } } ) ^ { 2 } } } \\ & { \quad \leq \frac { 2 b _ { \star } ^ { \mathsf { P } } } { ( 1 + b _ { \star } ^ { \mathsf { P } } ) \alpha _ { \star } ^ { \mathsf { P } } } + \frac { 1 } { 2 \mu _ { \star } ^ { \mathsf { P } } ( \alpha _ { \star } ^ { \mathsf { P } } ) ^ { 2 } } . } \end{array}
$$

The denominator in each term is positive and fixed with the model. Multiplying by the preceding bound on $\rho _ { T } - 1$ gives the stated order of $H _ { T }$ and completes the proof of Theorem C.16. □

## F.3 COMPENSATION ERROR

This section proves Proposition F.5 by comparing the two systems at the same momentum and the same test matrix. A scalar shift of the mean gives an exact identity. Adding the discarded directions does not change the threshold if their noise is removed, and restoring that noise costs at most $\delta _ { \mathrm { c o v } }$

Recall that $A = D _ { t _ { 0 } } + \lambda _ { \mathrm { W D } } I$ is the effective curvature and $\Delta _ { t } = A _ { B _ { t } } - A$ is its centered noise. Its block on the positive eigenspace is $A ^ { + } \succ 0$ , its compensated mean is $A _ { \mathsf { P } } = A + \lambda _ { \mathsf { P } } I \succ 0$ , and $L = \lambda _ { \mathrm { m a x } } ( A ^ { + } )$ . The quotient and its maximum from (51)–(52) are

$$
\begin{array} { r l r } {  { q _ { C , b } ( \pmb X ) = \frac { \mathbb { E } \operatorname { t r } ( \pmb X \Delta _ { t } \pmb X \Delta _ { t } ) + c _ { b } \operatorname { t r } ( \pmb X C X C ) - k _ { b } \| [ \pmb C , \pmb X ] \| _ { F } ^ { 2 } } { 2 \operatorname { t r } ( C \pmb X ^ { 2 } ) } , } } \\ & { } & { r _ { C } ( b ) = \operatorname* { m a x } _ { \pmb X \geq 0 , \ \| \pmb X \| _ { F } = 1 } q _ { C , b } ( \pmb X ) , \qquad c _ { b } = \frac { 1 - b } { 1 + b } , \quad k _ { b } = \frac { b } { 1 - b ^ { 2 } } . } \end{array}
$$

The noise in this formula is $\Delta _ { t } ^ { + }$ for the projected mean $A ^ { + }$ and the full $\Delta _ { t }$ for the compensated mean $A _ { \mathsf { P } }$ . The shift identity (Lemma F.3) was established in Appendix F.1 and applies here with the corresponding noise law fixed.

ProofofProposition F.5. We compare first at $b _ { \star }$ , then at the systems’ own critical momenta using (67) and its consequence for r<sub>C</sub> proved in Appendix F.5.3.

Recall that $b _ { \star } = \beta _ { 1 } \big ( 1 - \eta _ { \star } \lambda _ { \mathrm { W D } } \big )$ is the projected critical momentum and $c _ { \star } = ( 1 - b _ { \star } ) / ( 1 + b _ { \star } )$ The matrix $X ,$ <sub>⋆</sub> maximizes ${ { q } _ { A ^ { + } , { { b } _ { \star } } } }$ , with $\| \ b { X } _ { \star } \| _ { F } = 1$ and $\mu _ { \star } = \mathrm { t r } ( A ^ { + } X _ { \star } ^ { 2 } )$ . All threshold functions in Steps 1–2 are evaluated at the same momentum $b _ { \star }$

## Step 1. Shift the mean on the positive eigenspace.

Fix $X \succeq 0$ with $\| \boldsymbol { X } \| _ { F } = 1$ on the positive eigenspace, and write $\mu : = \mathrm { t r } ( A ^ { + } X ^ { 2 } )$ , as in the shift identity. Since $0 \prec A ^ { + } \preceq L I$ and $X ^ { 2 } \succeq 0$

$$
0 < \mu = \operatorname { t r } ( A ^ { + } X ^ { 2 } ) \leq L \operatorname { t r } ( X ^ { 2 } ) = L .
$$

Apply (68) with $C = A ^ { + } , b = b ,$ and $h = \lambda _ { \mathsf { P } }$ . Subtracting $r _ { A ^ { + } } ( b _ { \star } )$ and collecting its coefficients gives

$$
\begin{array} { r l } & { q _ { A ^ { + } + \lambda _ { \mathsf { P } } I , b _ { \star } } ( X ) - r _ { A ^ { + } } ( b _ { \star } ) } \\ & { \quad = \frac { \mu q _ { A ^ { + } , b _ { \star } } ( X ) + c _ { \star } \lambda _ { \mathsf { P } } ( \mu + \lambda _ { \mathsf { P } } / 2 ) } { \mu + \lambda _ { \mathsf { P } } } - r _ { A ^ { + } } ( b _ { \star } ) } \\ & { \quad = \frac { \mu \bigl ( q _ { A ^ { + } , b _ { \star } } ( X ) - r _ { A ^ { + } } ( b _ { \star } ) \bigr ) } { \mu + \lambda _ { \mathsf { P } } } + \frac { \lambda _ { \mathsf { P } } \bigl [ c _ { \star } ( \mu + \lambda _ { \mathsf { P } } / 2 ) - r _ { A ^ { + } } ( b _ { \star } ) \bigr ] } { \mu + \lambda _ { \mathsf { P } } } . } \end{array}
$$

Recall that Φ is the compensation term defined in (54). At the curvature $\mu = \operatorname { t r } ( A ^ { + } X ^ { 2 } ) $ , its definition gives

$$
\Phi ( \mu ) = { \frac { c _ { \star } \lambda _ { \mathsf { P } } } { 2 } } - { \frac { \lambda _ { \mathsf { P } } } { \mu + \lambda _ { \mathsf { P } } } } \left( r _ { A ^ { + } } ( b _ { \star } ) - { \frac { c _ { \star } \mu } { 2 } } \right) , \qquad c _ { \star } = { \frac { 1 - b _ { \star } } { 1 + b _ { \star } } } .
$$

To identify the second term above with $\Phi ( \mu )$ , expand its numerator as follows:

$$
\begin{array} { r l } & { \frac { \lambda _ { \mathsf { P } } \big [ c _ { \star } ( \mu + \lambda _ { \mathsf { P } } / 2 ) - r _ { A ^ { + } } ( b _ { \star } ) \big ] } { \mu + \lambda _ { \mathsf { P } } } } \\ & { \quad = \frac { \lambda _ { \mathsf { P } } \big [ c _ { \star } ( \mu + \lambda _ { \mathsf { P } } ) / 2 + c _ { \star } \mu / 2 - r _ { A ^ { + } } ( b _ { \star } ) \big ] } { \mu + \lambda _ { \mathsf { P } } } } \\ & { \quad = \frac { c _ { \star } \lambda _ { \mathsf { P } } } { 2 } - \frac { \lambda _ { \mathsf { P } } } { \mu + \lambda _ { \mathsf { P } } } \Big ( r _ { A ^ { + } } ( b _ { \star } ) - \frac { c _ { \star } \mu } { 2 } \Big ) } \\ & { \quad = \Phi ( \mu ) . } \end{array}
$$

Consequently,

$$
\begin{array} { l } { \displaystyle q _ { A ^ { + } + \lambda _ { \mathsf { P } } I , b _ { \star } } ( X ) - r _ { A ^ { + } } ( b _ { \star } ) } \\ { \displaystyle \quad = \frac { \mu } { \mu + \lambda _ { \mathsf { P } } } \big ( q _ { A ^ { + } , b _ { \star } } ( X ) - r _ { A ^ { + } } ( b _ { \star } ) \big ) + \Phi ( \mu ) . } \end{array}\tag{80}
$$

For $0 \leq s \leq L$ , the rank-one bound (110) yields

$$
r _ { A ^ { + } } ( b _ { \star } ) - \frac { c _ { \star } s } { 2 } \geq \frac { c _ { \star } ( L - s ) } { 2 } \geq 0 , \qquad \mathrm { h e n c e } \qquad \Phi ( s ) \leq \frac { c _ { \star } \lambda _ { \mathrm { P } } } { 2 } .
$$

To compare $\Phi ( \mu )$ with $\Phi ( L )$ , differentiate its equivalent form $\Phi ( s ) ~ = ~ \lambda _ { \sf P } [ c _ { \star } ( s + \lambda _ { \sf P } / 2 ) ~ -$ $r _ { A ^ { + } } ( b _ { \star } ) ] / ( s + \lambda _ { \mathsf { P } } )$

$$
\begin{array} { l } { \Phi ^ { \prime } ( s ) = \lambda _ { \mathsf { P } } \frac { c _ { \star } ( s + \lambda _ { \mathsf { P } } ) - \big [ c _ { \star } ( s + \lambda _ { \mathsf { P } } / 2 ) - r _ { A ^ { + } } ( b _ { \star } ) \big ] } { ( s + \lambda _ { \mathsf { P } } ) ^ { 2 } } } \\ { = \frac { \lambda _ { \mathsf { P } } \big [ r _ { A ^ { + } } ( b _ { \star } ) + c _ { \star } \lambda _ { \mathsf { P } } / 2 \big ] } { ( s + \lambda _ { \mathsf { P } } ) ^ { 2 } } > 0 . } \end{array}
$$

The numerator is positive because $\lambda _ { \mathsf { P } } > 0 , c _ { \star } > 0$ , and $r _ { A ^ { + } } ( b _ { \star } ) \geq c _ { \star } L / 2 > 0 .$

Since $q _ { A ^ { + } , b _ { \star } } ( X ) \leq r _ { A ^ { + } } ( b _ { \star } )$ , the first term in (80) is nonpositive. Moreover, Φ is increasing and $\mu \leq L$ . Thus, for every admissible X,

$$
\begin{array} { r } { q _ { \pmb { A } ^ { + } + \lambda _ { \mathsf { P } } \pmb { I } , \boldsymbol { b } _ { \star } } ( \pmb { X } ) - r _ { \pmb { A } ^ { + } } ( \pmb { b } _ { \star } ) \leq \Phi ( \mu ) } \\ { \leq \Phi ( \pmb { L } ) . } \end{array}
$$

Writing out the corresponding maximum gives

$$
\begin{array} { r l } & { r _ { A ^ { + } + \lambda _ { \mathsf { P } } I } ( b _ { \star } ) - r _ { A ^ { + } } ( b _ { \star } ) } \\ & { \quad = \underset { X \in \mathbb { S } _ { p } ^ { + } , \parallel X \parallel _ { F } = 1 } { \operatorname* { m a x } } \bigl [ q _ { A ^ { + } + \lambda _ { \mathsf { P } } I , b _ { \star } } ( X ) - r _ { A ^ { + } } ( b _ { \star } ) \bigr ] } \\ & { \quad \le \underset { X \in \mathbb { S } _ { p } ^ { + } , \parallel X \parallel _ { F } = 1 } { \operatorname* { m a x } } \Phi \bigl ( \mathrm { t r } ( A ^ { + } X ^ { 2 } ) \bigr ) } \\ & { \quad \le \Phi ( L ) . } \end{array}
$$

The first equality moves the constant $r _ { A ^ { + } } ( b _ { \star } )$ inside the maximum. Each subsequent inequality uses a bound valid at every matrix in this same set.

For the lower bound, the maximizing matrix $X _ { \star }$ satisfies $q _ { A ^ { + } , b _ { \star } } ( X _ { \star } ) = r _ { A ^ { + } } ( b _ { \star } )$ . Thus

$$
\begin{array} { r l } & { r _ { A ^ { + } + \lambda \mathsf { p } I } ( b _ { \star } ) - r _ { A ^ { + } } ( b _ { \star } ) } \\ & { \quad \ge q _ { A ^ { + } + \lambda \mathsf { p } I , b _ { \star } } ( X _ { \star } ) - r _ { A ^ { + } } ( b _ { \star } ) } \\ & { \quad = \frac { \mu _ { \star } } { \mu _ { \star } + \lambda \mathsf { p } } \big [ q _ { A ^ { + } , b _ { \star } } ( X _ { \star } ) - r _ { A ^ { + } } ( b _ { \star } ) \big ] + \Phi ( \mu _ { \star } ) } \\ & { \quad = \Phi ( \mu _ { \star } ) . } \end{array}
$$

The inequality tests the shifted maximum at $X _ { \star }$ , the next line uses (80), and the last line uses its maximizing property. We have proved

$$
\Phi ( \mu _ { \star } ) \leq r _ { A ^ { + } + \lambda \mathsf { p } } { \cal I } ( b _ { \star } ) - r _ { A ^ { + } } ( b _ { \star } ) \leq \Phi ( L ) .
$$

## Step 2. Restore the discarded directions and their noise.

Fix a matrix $X \in \mathbb { S } _ { p } ^ { + }$ on the positive eigenspace with $\| \boldsymbol { X } \| _ { F } = 1$ . Its embedding $\mathrm { d i a g } ( \pmb { X } , 0 )$ is positive semidefinite and has the same Frobenius norm, since

$$
\begin{array} { r l } & { \mathrm { d i a g } ( \boldsymbol { X } , 0 ) ^ { 2 } = \mathrm { d i a g } ( \boldsymbol { X } ^ { 2 } , 0 ) , } \\ & { \| \mathrm { d i a g } ( \boldsymbol { X } , 0 ) \| _ { F } ^ { 2 } = \mathrm { t r } \big ( \mathrm { d i a g } ( \boldsymbol { X } ^ { 2 } , 0 ) \big ) = \mathrm { t r } ( \boldsymbol { X } ^ { 2 } ) = 1 . } \end{array}
$$

We check the four terms of its quotient. For the noise term, block multiplication gives

$$
\begin{array} { r l } & { \mathrm { d i a g } ( \boldsymbol { X } , 0 ) \Delta _ { t } \mathrm { d i a g } ( \boldsymbol { X } , 0 ) = \mathrm { d i a g } ( \boldsymbol { X } \Delta _ { t } ^ { + } \boldsymbol { X } , 0 ) , } \\ & { \mathrm { t r } \big ( \mathrm { d i a g } ( \boldsymbol { X } , 0 ) \Delta _ { t } \mathrm { d i a g } ( \boldsymbol { X } , 0 ) \Delta _ { t } \big ) = \mathrm { t r } ( \boldsymbol { X } \Delta _ { t } ^ { + } \boldsymbol { X } \Delta _ { t } ^ { + } ) . } \end{array}
$$

The equality holds for each sample, and therefore also after taking expectations. For the mean term and the denominator, the block diagonal form of $A _ { \mathsf { P } }$ gives

$$
\begin{array} { c } { { \mathrm { d i a g } ( { \pmb X } , 0 ) { \pmb A } _ { \mathrm { P } } = \mathrm { d i a g } \big ( { \pmb X } ( { \pmb A } ^ { + } + \lambda { \mathsf { p } } { \pmb I } ) , 0 \big ) , } } \\ { { \mathrm { t r } \big ( \mathrm { d i a g } ( { \pmb X } , 0 ) { \pmb A } _ { \mathrm { P } } \mathrm { d i a g } ( { \pmb X } , 0 ) { \pmb A } _ { \mathrm { P } } \big ) = \mathrm { t r } \big ( { \pmb X } ( { \pmb A } ^ { + } + \lambda _ { \mathrm { P } } { \pmb I } ) { \pmb X } ( { \pmb A } ^ { + } + \lambda _ { \mathrm { P } } { \pmb I } ) \big ) , } } \\ { { \mathrm { t r } \big ( { \pmb A } _ { \mathrm { P } } \mathrm { d i a g } ( { \pmb X } , 0 ) ^ { 2 } \big ) = \mathrm { t r } \big ( ( { \pmb A } ^ { + } + \lambda _ { \mathrm { P } } { \pmb I } ) { \pmb X } ^ { 2 } \big ) . } } \end{array}
$$

The commutator also reduces to the block on the positive eigenspace,

$$
\begin{array} { r } { \left[ { \cal A } _ { \mathrm { P } } , \mathrm { d i a g } ( { \cal X } , 0 ) \right] = \mathrm { d i a g } \big ( [ { \cal A } ^ { + } + \lambda _ { \mathsf { P } } { \cal I } , { \cal X } ] , 0 \big ) , } \\ { \| [ { \cal A } _ { \mathrm { P } } , \mathrm { d i a g } ( { \cal X } , 0 ) ] \| _ { F } ^ { 2 } = \big \| [ { \cal A } ^ { + } + \lambda _ { \mathsf { P } } { \cal I } , { \cal X } ] \big \| _ { F } ^ { 2 } . \qquad } \end{array}
$$

Substituting these equalities into the quotient proves

$$
q _ { A _ { \mathsf { P } } , b _ { \star } } \left( \mathrm { d i a g } ( X , 0 ) \right) = q _ { A ^ { + } + \lambda _ { \mathsf { P } } I , b _ { \star } } ( X ) .\tag{81}
$$

The full maximization includes every embedded test matrix. In particular,

$$
\begin{array} { l } { r _ { A _ { \mathrm { P } } } ( b _ { \star } ) = \displaystyle \operatorname* { m a x } _ { X \in \mathfrak { S } _ { + } ^ { + } } q _ { A _ { \mathrm { P } } , b _ { \star } } ( X ) } \\ { \| X \| _ { F ^ { \infty } } } \\ { \ge \displaystyle \operatorname* { m a x } _ { X \in \mathfrak { S } _ { + } ^ { + } } q _ { A _ { \mathrm { P } } , b _ { \star } } \left( \mathrm { d i a g } ( X , 0 ) \right) } \\ { \| X \| _ { F ^ { \infty } } 1 } \\ { = \displaystyle \operatorname* { m a x } _ { X \in \mathfrak { S } _ { + } ^ { + } } q _ { A ^ { + } + \lambda _ { \mathrm { P } } I , b _ { \star } } ( X ) } \\ { \| X \| _ { F ^ { \infty } } 1 } \\ { = r _ { A ^ { + } + \lambda _ { \mathrm { P } } } I ( b _ { \star } ) . } \end{array}
$$

The inequality restricts the test matrices to those embedded from the positive eigenspace, and the next equality uses (81).

For the upper bound, first remove the discarded and cross blocks of the noise. The resulting samples are

$$
\begin{array} { r } { A _ { \mathsf { P } } + \Pi ^ { + } \Delta _ { t } \Pi ^ { + } = \left( \begin{array} { c c } { \pmb { A } ^ { + } + \lambda _ { \mathsf { P } } \pmb { I } + \pmb { \Delta } _ { t } ^ { + } } & { 0 } \\ { 0 } & { \pmb { A } ^ { - } + \lambda _ { \mathsf { P } } \pmb { I } } \end{array} \right) . } \end{array}
$$

When $p < d ,$ this recurrence is strictly mean-square stable if and only if both block recurrences are stable. Sufficiency follows by adding their expected squared state norms, and necessity follows by initializing only one block. Both block means are positive definite because $A _ { \mathsf { P } } \succ 0$

At the fixed momentum $b _ { \star }$ , let the coefficient $\alpha > 0$ vary. By the exact criterion (Theorem F.1) and the noise-free threshold (75), both blocks are strictly stable if and only if

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { ( 1 - b _ { \star } ) / \alpha > r _ { A ^ { + } + \lambda \mathsf { p } } I ( b _ { \star } ) , } \\ { ( 1 - b _ { \star } ) / \alpha > c _ { \star } \big ( \lambda _ { \operatorname* { m a x } } ( A ^ { - } ) + \lambda _ { \mathsf { P } } \big ) / 2 } \end{array} \right. } \end{array}
$$

These two inequalities are equivalent to

$$
\frac { 1 - b _ { \star } } { \alpha } > \operatorname* { m a x } \left\{ r _ { A ^ { + } + \lambda \mathsf { p } I } ( b _ { \star } ) , \frac { c _ { \star } } { 2 } \big ( \lambda _ { \operatorname* { m a x } } ( A ^ { - } ) + \lambda \mathsf { p } \big ) \right\} .
$$

Thus the maximum on the right is the threshold function of this block diagonal system. Since $\lambda _ { \operatorname* { m a x } } ( A ^ { - } ) \leq 0 < L$ , the rank-one bound (110) gives

$$
\begin{array} { r } { \frac { c _ { \star } } { 2 } \big ( \lambda _ { \operatorname* { m a x } } ( A ^ { - } ) + \lambda _ { \mathsf { P } } \big ) < \frac { c _ { \star } } { 2 } ( L + \lambda _ { \mathsf { P } } ) } \\ { \leq r _ { A ^ { + } + \lambda _ { \mathsf { P } } } \boldsymbol { \mathsf { ( } } b _ { \star } \big ) . } \end{array}
$$

The threshold is therefore $r _ { A ^ { + } + \lambda _ { \mathsf { P } } I } ( b _ { \star } )$ . When $p = d ,$ there is no discarded block and this equality holds directly.

Now fix a full-space test matrix $\ b X \in \mathbb S _ { d } ^ { + }$ with $\| \boldsymbol { X } \| _ { F } = 1$ . Before the omitted noise is restored, its noise term is

$$
\begin{array} { r l } & { \mathbb { E } \operatorname { t r } \bigl ( X ( \Pi ^ { + } \Delta _ { t } \Pi ^ { + } ) X ( \Pi ^ { + } \Delta _ { t } \Pi ^ { + } ) \bigr ) } \\ & { \qquad = \mathbb { E } \operatorname { t r } \bigl ( ( \Pi ^ { + } X \Pi ^ { + } ) \Delta _ { t } ( \Pi ^ { + } X \Pi ^ { + } ) \Delta _ { t } \bigr ) } \\ & { \qquad = \mathbb { E } \operatorname { t r } ( X ^ { + } \Delta _ { t } ^ { + } X ^ { + } \Delta _ { t } ^ { + } ) . } \end{array}
$$

The first equality uses cyclicity of the trace and $( \Pi ^ { + } ) ^ { 2 } = \Pi ^ { + }$ , and the second follows by block multiplication. The mean, commutator and denominator are unchanged. Subtracting the change in the noise term therefore recovers the quotient before restoration:

$$
\begin{array} { r l } & { q _ { A _ { \mathsf { P } } , b _ { \star } } ( X ) - \frac { \mathbb { E } \operatorname { t r } ( X \Delta _ { t } X \Delta _ { t } ) - \mathbb { E } \operatorname { t r } ( X ^ { + } \Delta _ { t } ^ { + } X ^ { + } \Delta _ { t } ^ { + } ) } { 2 \operatorname { t r } ( A _ { \mathsf { P } } X ^ { 2 } ) } } \\ & { \quad \quad = \frac { \mathbb { E } \operatorname { t r } ( X ^ { + } \Delta _ { t } ^ { + } X ^ { + } \Delta _ { t } ^ { + } ) + c _ { \star } \operatorname { t r } ( X A _ { \mathsf { P } } X A _ { \mathsf { P } } ) - k _ { b _ { \star } } \| [ A _ { \mathsf { P } } , X ] \| _ { F } ^ { 2 } } { 2 \operatorname { t r } ( A _ { \mathsf { P } } X ^ { 2 } ) } } \\ & { \quad \le r _ { A ^ { + } + \lambda _ { \mathsf { P } } I } ( b _ { \star } ) . } \end{array}
$$

The denominator is positive because $A _ { \mathsf { P } } \succ 0$ and $X \neq 0 .$ . Recall that $\delta _ { \mathrm { c o v } }$ is the largest normalized change in the noise term over these full-space test matrices:

$$
\delta _ { \mathrm { c o v } } = \operatorname* { m a x } _ { X \in \mathbb { S } _ { d } ^ { + } } \frac { \mathbb { E } \operatorname { t r } ( X \Delta _ { t } X \Delta _ { t } ) - \mathbb { E } \operatorname { t r } ( X ^ { + } \Delta _ { t } ^ { + } X ^ { + } \Delta _ { t } ^ { + } ) } { 2 \operatorname { t r } ( A _ { \mathrm { P } } X ^ { 2 } ) } .
$$

Adding back the subtracted fraction and bounding it by this maximum, as defined in (55), gives

$$
\begin{array} { r l } & { q _ { A _ { \mathrm { P } } , b _ { \star } } ( X ) \leq r _ { A ^ { + } + \lambda \mathrm { p } I } ( b _ { \star } ) + \frac { \mathbb { E } \mathrm { t r } ( X \Delta _ { t } X \Delta _ { t } ) - \mathbb { E } \mathrm { t r } ( X ^ { + } \Delta _ { t } ^ { + } X ^ { + } \Delta _ { t } ^ { + } ) } { 2 \mathrm { t r } ( A _ { \mathrm { P } } X ^ { 2 } ) } } \\ & { \phantom { a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a } } \\ & { \phantom { a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a } \leq r _ { A ^ { + } + \lambda \mathrm { p } I } ( b _ { \star } ) + \delta _ { \mathrm { c o v } } . } \end{array}
$$

This holds for every full-space test matrix. Maximizing both sides,

$$
\begin{array} { r l } { r _ { A _ { \mathrm { P } } } ( b _ { \star } ) = } & { \underset { X \in \mathbb S _ { d } ^ { + } } { \mathrm { m a x } } ~ q _ { A _ { \mathrm { P } } , b _ { \star } } ( X ) } \\ & { \parallel X \parallel _ { F } = 1 } \\ { \le } & { \underset { X \in \mathbb S _ { d } ^ { + } } { \mathrm { m a x } } \left[ r _ { A ^ { + } + \lambda _ { \mathrm { P } } I } ( b _ { \star } ) + \delta _ { \mathrm { c o v } } \right] } \\ & { \parallel X \parallel _ { F } = 1 } \\ & { = r _ { A ^ { + } + \lambda _ { \mathrm { P } } I } ( b _ { \star } ) + \delta _ { \mathrm { c o v } } . } \end{array}\tag{82}
$$

We can now combine the lower bound from the embedded test matrices and the upper bound (82) with Step 1:

$$
\begin{array} { r l } & { r _ { A _ { \mathrm { P } } } ( b _ { \star } ) - r _ { A ^ { + } } ( b _ { \star } ) \geq r _ { A ^ { + } + \lambda _ { \mathrm { P } } I } ( b _ { \star } ) - r _ { A ^ { + } } ( b _ { \star } ) } \\ & { ~ \geq \Phi ( \mu _ { \star } ) , } \end{array}
$$

$$
\begin{array} { r l } & { r _ { A _ { \mathrm { P } } } ( b _ { \star } ) - r _ { A ^ { + } } ( b _ { \star } ) \leq r _ { A ^ { + } + \lambda \mathsf { p } } \iota ( b _ { \star } ) - r _ { A ^ { + } } ( b _ { \star } ) + \delta _ { \mathrm { c o v } } } \\ & { \qquad \leq \Phi ( L ) + \delta _ { \mathrm { c o v } } . } \end{array}
$$

Equivalently,

$$
\Phi ( \mu _ { \star } ) \leq r _ { A _ { \mathrm { P } } } ( b _ { \star } ) - r _ { A ^ { + } } ( b _ { \star } ) \leq \Phi ( L ) + \delta _ { \mathrm { c o v } } .\tag{83}
$$

## Step 3. Compare the critical LRs.

Recall that $b _ { \star } ^ { \mathsf { P } } = \beta _ { 1 } ( 1 - \eta _ { \star } ^ { \mathsf { P } } \lambda _ { \mathrm { W D } } )$ and $b _ { \star } = \beta _ { 1 } ( 1 - \eta _ { \star } \lambda _ { \mathrm { W D } } )$ are the momenta at the two critical LRs. The critical-rate identity (74) expresses each quotient maximum as the inverse critical rate plus the same weight-decay offset:

$$
\begin{array} { r } { r _ { A \mathrm { P } } ( b _ { \star } ^ { \mathrm { P } } ) = \displaystyle \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } + \frac { \beta _ { 1 } \lambda _ { \mathrm { W D } } } { 1 - \beta _ { 1 } } , } \\ { r _ { A ^ { + } } ( b _ { \star } ) = \displaystyle \frac { 1 } { \eta _ { \star } } + \frac { \beta _ { 1 } \lambda _ { \mathrm { W D } } } { 1 - \beta _ { 1 } } . } \end{array}
$$

Subtracting the same offset from both thresholds,

$$
\begin{array} { c } { \displaystyle \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } - \displaystyle \frac { 1 } { \eta _ { \star } } = \left[ r _ { A \mathrm { P } } ( { b } _ { \star } ^ { \mathrm { P } } ) - \displaystyle \frac { \beta _ { 1 } \lambda _ { \mathrm { W D } } } { 1 - \beta _ { 1 } } \right] - \left[ r _ { A ^ { + } } ( b _ { \star } ) - \displaystyle \frac { \beta _ { 1 } \lambda _ { \mathrm { W D } } } { 1 - \beta _ { 1 } } \right] } \\ { = r _ { A \mathrm { P } } ( { b } _ { \star } ^ { \mathrm { P } } ) - r _ { A ^ { + } } ( b _ { \star } ) . } \end{array}
$$

The two critical momenta differ by

$$
\begin{array} { r l } & { b _ { \star } ^ { \mathsf { P } } - b _ { \star } = \beta _ { 1 } ( 1 - \eta _ { \star } ^ { \mathsf { P } } \lambda _ { \mathrm { W D } } ) - \beta _ { 1 } ( 1 - \eta _ { \star } \lambda _ { \mathrm { W D } } ) } \\ & { \qquad = \beta _ { 1 } \lambda _ { \mathrm { W D } } ( \eta _ { \star } - \eta _ { \star } ^ { \mathsf { P } } ) . } \end{array}
$$

Also, both critical LRs are positive, so

$$
\frac { 1 } { \eta _ { \star } ^ { \mathsf { P } } } - \frac { 1 } { \eta _ { \star } } = \frac { \eta _ { \star } - \eta _ { \star } ^ { \mathsf { P } } } { \eta _ { \star } ^ { \mathsf { P } } \eta _ { \star } } .
$$

These two identities determine the signs in the following cases.

If $\eta _ { \star } ^ { \mathsf { P } } \le \eta _ { \star }$ , then $b _ { \star } ^ { \mathsf { P } } \geq b _ { \star }$ and $1 / \eta _ { \star } ^ { \mathsf { P } } - 1 / \eta _ { \star } \geq 0$ . Since $r _ { A _ { \mathsf { P } } }$ is nonincreasing in its momentum argument, $r _ { A _ { \mathsf { P } } } ( b _ { \star } ^ { \mathsf { P } } ) \leq r _ { A _ { \mathsf { P } } } ( b _ { \star } )$ . Combining this with the upper bound in (83),

$$
\begin{array} { r l r } {  { \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } - \frac { 1 } { \eta _ { \star } } = r _ { A \mathrm { P } } ( b _ { \star } ^ { \mathrm { P } } ) - r _ { A ^ { + } } ( b _ { \star } ) } } \\ & { } & { \leq r _ { A \mathrm { P } } ( b _ { \star } ) - r _ { A ^ { + } } ( b _ { \star } ) } \\ & { } & { \leq \Phi ( L ) + \delta _ { \mathrm { c o v } } . } \end{array}
$$

In this case both clipped bounds follow from

$$
\begin{array} { r l } & { \operatorname* { m i n } \{ \Phi ( \mu _ { \star } ) , 0 \} \le 0 \le \displaystyle \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } - \displaystyle \frac { 1 } { \eta _ { \star } } , } \\ & { \qquad \displaystyle \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } - \displaystyle \frac { 1 } { \eta _ { \star } } \le \Phi ( L ) + \delta _ { \mathrm { c o v } } \le \operatorname* { m a x } \{ \Phi ( L ) + \delta _ { \mathrm { c o v } } , 0 \} . } \end{array}
$$

If $\eta _ { \star } ^ { \mathsf { P } } \geq \eta _ { \star }$ , then $b _ { \star } ^ { \mathsf { P } } \leq b _ { \star }$ and $1 / \eta _ { \star } ^ { \mathsf { P } } - 1 / \eta _ { \star } \leq 0$ . Monotonicity now gives $r _ { A _ { \mathsf { P } } } ( b _ { \star } ^ { \mathsf { P } } ) \geq r _ { A _ { \mathsf { P } } } ( b _ { \star } )$ Combining this with the lower bound in (83),

$$
\begin{array} { l } { \displaystyle \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } - \frac { 1 } { \eta _ { \star } } = r _ { A _ { \mathrm { P } } } ( b _ { \star } ^ { \mathrm { P } } ) - r _ { A ^ { + } } ( b _ { \star } ) } \\ { \displaystyle \geq r _ { A _ { \mathrm { P } } } ( b _ { \star } ) - r _ { A ^ { + } } ( b _ { \star } ) } \\ { \displaystyle \geq \Phi ( \mu _ { \star } ) . } \end{array}
$$

The clipped bounds now follow from

$$
\begin{array} { c l l } { \displaystyle \operatorname* { m i n } \{ \Phi ( \mu _ { \star } ) , 0 \} \leq \Phi ( \mu _ { \star } ) \leq \displaystyle \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } - \frac { 1 } { \eta _ { \star } } , } \\ { \displaystyle \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } - \frac { 1 } { \eta _ { \star } } \leq 0 \leq \operatorname* { m a x } \{ \Phi ( L ) + \delta _ { \mathrm { c o v } } , 0 \} . } \end{array}
$$

This proves (78) in both cases.

Finally, if $\beta _ { 1 } \lambda _ { \mathrm { W D } } = 0$ , then

$$
b _ { \star } ^ { \sf P } = \beta _ { 1 } - \beta _ { 1 } \lambda _ { \mathrm { W D } } \eta _ { \star } ^ { \sf P } = \beta _ { 1 } = \beta _ { 1 } - \beta _ { 1 } \lambda _ { \mathrm { W D } } \eta _ { \star } = b _ { \star } .
$$

The critical identities and (83) therefore give

$$
\begin{array} { l } { \displaystyle \Phi ( \mu _ { \star } ) \leq r _ { A _ { \mathrm { P } } } ( b _ { \star } ) - r _ { A ^ { + } } ( b _ { \star } ) } \\ { \displaystyle = r _ { A _ { \mathrm { P } } } ( b _ { \star } ^ { \mathrm { P } } ) - r _ { A ^ { + } } ( b _ { \star } ) } \\ { \displaystyle = \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } - \frac { 1 } { \eta _ { \star } } } \\ { \displaystyle \leq \Phi ( L ) + \delta _ { \mathrm { c o v } } . } \end{array}
$$

Hence no clipping is needed in this case. This completes the proof of Proposition F.5.

A scalar example. Let $\beta _ { 1 } = \lambda _ { \mathrm { W D } } = 0 , d = p = 1 , \pmb { A } = 1 , \mathbb { E } \pmb { \Delta } _ { t } ^ { 2 } = 1$ , and $\lambda _ { \mathsf { P } } = 0 . 0 1$ . For the homogeneous scalar SGD recurrence with mean $a > 0$ and centered noise of variance $\sigma ^ { 2 }$ , the factor by which one step multiplies the second moment is

$$
\begin{array} { r } { \mathbb { E } \big ( 1 - \eta ( a + \Delta _ { t } ) \big ) ^ { 2 } = 1 - 2 \eta a + \eta ^ { 2 } \mathbb { E } ( a + \Delta _ { t } ) ^ { 2 } } \\ { = 1 - 2 \eta a + \eta ^ { 2 } ( a ^ { 2 } + \sigma ^ { 2 } ) . } \end{array}
$$

For $\eta > 0$ , this factor is below one if and only if

$$
- 2 \eta a + \eta ^ { 2 } ( a ^ { 2 } + \sigma ^ { 2 } ) < 0 \quad \Longleftrightarrow \quad \eta < { \frac { 2 a } { a ^ { 2 } + \sigma ^ { 2 } } } .
$$

The inverse critical rate is therefore $( a ^ { 2 } + \sigma ^ { 2 } ) / ( 2 a )$ . In the projected system $a = 1$ and $\sigma ^ { 2 } = 1$ while compensation replaces a by $1 + \lambda _ { \mathsf { P } }$ and leaves the variance unchanged. Hence

$$
\begin{array} { c } { \displaystyle { \frac { 1 } { \eta _ { \star } } = 1 , } } \\ { \displaystyle { \frac { 1 } { \eta _ { \star } ^ { \mathrm { p } } } - \frac { 1 } { \eta _ { \star } } = \frac { ( 1 + \lambda _ { \mathrm { P } } ) ^ { 2 } + 1 } { 2 ( 1 + \lambda _ { \mathrm { P } } ) } - 1 } } \\ { = \displaystyle { \frac { \lambda _ { \mathrm { P } } ^ { 2 } } { 2 ( 1 + \lambda _ { \mathrm { P } } ) } \approx 4 . 9 5 \cdot 1 0 ^ { - 5 } } . } \end{array}
$$

Here $\mu _ { \star } = L = 1$ and $\delta _ { \mathrm { c o v } } = 0$ . Since $b _ { \star } = 0 , c _ { \star } = 1$ and $r _ { A ^ { + } } ( b _ { \star } ) = 1$ , the definition of $\Phi$ gives

$$
\Phi ( 1 ) = { \frac { \lambda _ { \sf P } } { 2 } } - { \frac { \lambda _ { \sf P } } { 1 + \lambda _ { \sf P } } } \left( 1 - { \frac { 1 } { 2 } } \right) = { \frac { \lambda _ { \sf P } ^ { 2 } } { 2 ( 1 + \lambda _ { \sf P } ) } } .
$$

Both compensation endpoints therefore equal the inverse-rate difference. The comparison is exact because the shared noise is compared directly rather than bounded separately in each system.

The SGD case. For SGD without weight decay, $\beta _ { 1 } = \lambda _ { \mathrm { W D } } = 0$ and $P = I$ , so $A = H _ { t _ { 0 } }$ . Here $b _ { \star } ^ { \mathsf { P } } = 0 , \alpha _ { \star } ^ { \mathsf { P } } = \eta _ { \star } ^ { \mathsf { P } }$ and $c _ { \star } = 1$ . The critical identity (74) gives $r _ { A ^ { + } } ( 0 ) = 1 / \eta _ { \star }$ . Substituting these values into the definitions of $\rho _ { T } , H _ { T }$ and Φ gives

$$
\begin{array} { c } { { \rho _ { T } = M ^ { 2 / T } , \qquad H _ { T } = \displaystyle \frac { M ^ { 2 / T } - 1 } { 2 \mu _ { \star } ^ { \mathrm { P } } ( \eta _ { \star } ^ { \mathrm { P } } ) ^ { 2 } } , } } \\ { { \Phi ( s ) = \displaystyle \frac { \lambda _ { \mathrm { P } } } { 2 } - \displaystyle \frac { \lambda _ { \mathrm { P } } } { s + \lambda _ { \mathrm { P } } } \left( \displaystyle \frac { 1 } { \eta _ { \star } } - \displaystyle \frac { s } { 2 } \right) . } } \end{array}
$$

The theorem applies without clipping. The direct LR bound derived in Appendix F.4 is $\bar { \eta } _ { \star } ( M , T ) \leq$ $\eta _ { \star } ^ { \mathsf { P } } + ( M ^ { 2 / T } - 1 ) / ( 2 \mu _ { \star } ^ { \mathsf { P } } )$

The covariance mismatch can be bounded by second moments of the discarded and cross blocks of the noise. Recall that $\Delta _ { t } ^ { \perp } = \Delta _ { t } - \Pi ^ { + } \Delta _ { t } \Pi ^ { + }$ contains precisely those discarded and cross blocks.

Lemma F.7 (A bound on the covariance mismatch). Under Assumption C.14, let $v _ { 0 } , v _ { E } \ge 0$ satisfy $\begin{array} { r } { \mathbb { E } ( \Delta _ { t } ^ { + } ) ^ { 2 } \preceq v _ { 0 } ( A ^ { + } + \lambda _ { \mathsf { P } } I ) } \end{array}$ and $\begin{array} { r } { \mathbb { E } ( \Delta _ { t } ^ { \perp } ) ^ { 2 } \preceq v _ { E } A _ { \mathsf { P } } } \end{array}$ . Then

$$
\delta _ { \mathrm { c o v } } \leq \sqrt { v _ { 0 } v _ { E } } + \frac { v _ { E } } { 2 } .
$$

ProofofLemma F.7. We bound the noise difference that defines $\delta _ { \mathrm { c o v } }$ , keeping its denominator $2 \operatorname { t r } ( \dot { A _ { \mathsf { P } } } X ^ { 2 } )$ unchanged. Within this proof write

$$
\begin{array} { r } { \pmb { \Delta } _ { t } ^ { \circ } : = \Pi ^ { + } \pmb { \Delta } _ { t } \Pi ^ { + } = \mathrm { d i a g } ( \pmb { \Delta } _ { t } ^ { + } , 0 ) , \qquad \pmb { \Delta } _ { t } = \pmb { \Delta } _ { t } ^ { \circ } + \pmb { \Delta } _ { t } ^ { \bot } . } \end{array}
$$

Step 1. Bound the separate noise terms.

The moment bound on $\Delta _ { t } ^ { + }$ in the hypothesis extends to the full space:

$$
\begin{array} { r l } & { \mathbb { E } ( \Delta _ { t } ^ { \circ } ) ^ { 2 } = \mathrm { d i a g } \big ( \mathbb { E } ( \Delta _ { t } ^ { + } ) ^ { 2 } , 0 \big ) } \\ & { ~ \preceq v _ { 0 } \mathrm { d i a g } ( A ^ { + } + \lambda _ { \mathsf { P } } \pmb { I } , 0 ) } \\ & { ~ \preceq v _ { 0 } \pmb { A } _ { \mathsf { P } } . } \end{array}
$$

The last inequality uses the positive definiteness of the discarded block when it is present, and is an equality when $p = d .$ . Fix $0 \not = X \in \mathbb { S } _ { d } ^ { + }$ . Since $\Delta _ { t } ^ { \circ } = \mathrm { d i a g } ( \Delta _ { t } ^ { + } , 0 )$ , block multiplication gives

$$
\mathbb { E } \operatorname { t r } ( X ^ { + } \Delta _ { t } ^ { + } X ^ { + } \Delta _ { t } ^ { + } ) = \mathbb { E } \operatorname { t r } ( X \Delta _ { t } ^ { \circ } X \Delta _ { t } ^ { \circ } ) .
$$

We first estimate the cross term and the discarded-noise term separately. For any symmetric $\Delta .$ symmetry and cyclicity of the trace give

$$
\begin{array} { r l } & { \| \Delta \boldsymbol { X } \| _ { F } ^ { 2 } = \mathrm { t r } \big ( ( \Delta \boldsymbol { X } ) ^ { \top } \Delta \boldsymbol { X } \big ) } \\ & { \qquad = \mathrm { t r } ( \boldsymbol { X } \Delta ^ { 2 } \boldsymbol { X } ) } \\ & { \qquad = \mathrm { t r } ( \boldsymbol { X } ^ { 2 } \Delta ^ { 2 } ) , } \end{array}
$$

$$
\begin{array} { r l } & { \| X \pmb { \Delta } \| _ { F } ^ { 2 } = \mathrm { t r } \big ( ( X \pmb { \Delta } ) ^ { \top } X \pmb { \Delta } \big ) } \\ & { \qquad = \mathrm { t r } ( \pmb { \Delta } X ^ { 2 } \pmb { \Delta } ) } \\ & { \qquad = \mathrm { t r } ( X ^ { 2 } \pmb { \Delta } ^ { 2 } ) . } \end{array}
$$

The two moment bounds therefore imply

$$
\begin{array} { r l r l } { \mathbb { E } \left\| \pmb { \Delta } _ { t } ^ { \circ } \pmb { X } \right\| _ { F } ^ { 2 } = \operatorname { t r } \bigl ( \pmb { X } ^ { 2 } \mathbb { E } ( \pmb { \Delta } _ { t } ^ { \circ } ) ^ { 2 } \bigr ) } & { \qquad } & & { \mathrm { t h e ~ n o r m ~ i d e n t i t y ~ a b o v e } } \\ & { \qquad \leq v _ { 0 } \operatorname { t r } ( \pmb { A } _ { \sf P } \pmb { X } ^ { 2 } ) , } & & { \qquad } & & { \mathrm { t h e ~ m o m e n t ~ b o u n d ~ o n ~ } \pmb { \Delta } _ { t } ^ { + } } \\ { \mathbb { E } \left\| \pmb { X } \pmb { \Delta } _ { t } ^ { \bot } \right\| _ { F } ^ { 2 } = \operatorname { t r } \bigl ( \pmb { X } ^ { 2 } \mathbb { E } ( \pmb { \Delta } _ { t } ^ { \bot } ) ^ { 2 } \bigr ) } & { \qquad } & & { \mathrm { t h e ~ n o r m ~ i d e n t i t y ~ a b o v e } } \\ & { \qquad \leq v _ { E } \operatorname { t r } ( \pmb { A } _ { \sf P } \pmb { X } ^ { 2 } ) . } & & { \qquad } & & { \mathrm { t h e ~ d i s c a r d e d - n o i s e ~ m o m e n t ~ b o u n d } } \end{array}
$$

Both inequalities use the same trace-order fact. For $Q \succeq 0 .$

$$
\mathrm { t r } ( X ^ { 2 } Q ) = \mathrm { t r } ( X Q X ) \geq 0 .
$$

Apply this to the differences between the right and left matrices in the two moment bounds.

For the cross term, the Cauchy–Schwarz inequality in the Frobenius inner product and then in $L ^ { 2 }$ gives

$$
\begin{array} { r l r l } & { \mathbb { E } \operatorname { t r } ( X \Delta _ { t } ^ { \circ } X \Delta _ { t } ^ { \perp } ) } \\ & { \quad = \mathbb { E } \langle \Delta _ { t } ^ { \circ } X , X \Delta _ { t } ^ { \perp } \rangle _ { F } } & & { \qquad \mathrm { s y m m e t r y ~ o f ~ } X \mathrm { ~ a n d ~ } \Delta _ { t } ^ { \circ } } \\ & { \quad \le \mathbb { E } \big [ \big \| \Delta _ { t } ^ { \circ } X \big \| _ { F } \big \| X \Delta _ { t } ^ { \perp } \big \| _ { F } \big ] } & & { \qquad \mathrm { F r o b e n i u s ~ C a u c h y - S c h w a r z } } \\ & { \quad \le \sqrt { \mathbb { E } \big \| \Delta _ { t } ^ { \circ } X \big \| _ { F } ^ { 2 } \mathbb { E } \big \| X \Delta _ { t } ^ { \perp } \big \| _ { F } ^ { 2 } } \qquad } & & { \qquad \mathrm { C a u c h y - S c h w a r z ~ i n ~ } L ^ { 2 } } \\ & { \quad \le \sqrt { v _ { 0 } v _ { E } } \operatorname { t r } ( A _ { \mathsf { P } } X ^ { 2 } ) . } & & { \qquad \mathrm { t h e ~ t w o ~ m o m e n t ~ b o u n d s } } \end{array}
$$

For the discarded-noise term, the two Frobenius norms are equal, so

$$
\begin{array} { r l r l } & { \mathbb { E } \operatorname { t r } ( X \Delta _ { t } ^ { \perp } X \Delta _ { t } ^ { \perp } ) } \\ & { \quad = \mathbb { E } \langle \Delta _ { t } ^ { \perp } X , X \Delta _ { t } ^ { \perp } \rangle _ { F } } \\ & { \quad \le \mathbb { E } \big [ \big \| \Delta _ { t } ^ { \perp } X \big \| _ { F } \left\| X \Delta _ { t } ^ { \perp } \right\| _ { F } \big ] } & { \qquad } & { \mathrm { F r o b e n i u s ~ C a u c h y - S c h w a r z } } \\ & { \quad = \mathbb { E } \left\| X \Delta _ { t } ^ { \perp } \right\| _ { F } ^ { 2 } \qquad } & & { \mathrm { t h e ~ n o r m ~ i d e n t i t y ~ a b o v e } } \\ & { \quad \le v _ { E } \operatorname { t r } ( A _ { \mathrm { P } } X ^ { 2 } ) . } & & { \mathrm { t h e ~ d i s c a r d e d - n o i s e ~ m o m e n t ~ b o u n d } } \end{array}
$$

Step 2. Combine the terms and normalize.

Expand $\Delta _ { t } = \Delta _ { t } ^ { \circ } + \Delta _ { t } ^ { \scriptscriptstyle \perp }$ in the numerator. The two cross terms have the same trace by cyclicity, and the preceding bounds give

$$
\begin{array} { r l } & { \mathbb { E } \operatorname { t r } ( X \Delta _ { t } X \Delta _ { t } ) - \mathbb { E } \operatorname { t r } ( X \Delta _ { t } ^ { \circ } X \Delta _ { t } ^ { \circ } ) } \\ & { \qquad = 2 \mathbb { E } \operatorname { t r } ( X \Delta _ { t } ^ { \circ } X \Delta _ { t } ^ { \perp } ) + \mathbb { E } \operatorname { t r } ( X \Delta _ { t } ^ { \perp } X \Delta _ { t } ^ { \perp } ) } \\ & { \qquad \leq \left( 2 \sqrt { v _ { 0 } v _ { E } } + v _ { E } \right) \operatorname { t r } ( A _ { \mathsf { P } } X ^ { 2 } ) . } \end{array}
$$

For every nonzero PSD X, the denominator $2 \operatorname { t r } ( A _ { \mathsf { P } } X ^ { 2 } )$ is positive. Dividing by it and using the block trace identity above,

$$
\frac { \mathbb { E } \operatorname { t r } ( X \Delta _ { t } X \Delta _ { t } ) - \mathbb { E } \operatorname { t r } ( X ^ { + } \Delta _ { t } ^ { + } X ^ { + } \Delta _ { t } ^ { + } ) } { 2 \operatorname { t r } ( A _ { \mathrm { P } } X ^ { 2 } ) } \leq \frac { 2 \sqrt { v _ { 0 } v _ { E } } + v _ { E } } { 2 } = \sqrt { v _ { 0 } v _ { E } } + \frac { v _ { E } } { 2 } .
$$

The right side is independent of X, so it also bounds the maximum $\delta _ { \mathrm { c o v } }$ . This completes the proof of Lemma F.7. □

Since $A ^ { + } \preceq A ^ { + } + \lambda _ { \mathsf { P } } I$ , one may take $v _ { 0 }$ to be the noise level of the projected system, the smallest v with $\mathbb { E } ( \Delta _ { t } ^ { + } ) ^ { 2 } \preceq v A ^ { + }$ . The lemma controls the noise in the discarded and cross blocks relative to the compensated effective curvature. A bound on the compensation coefficient alone does not control these quantities, as the discarded-noise example below shows.

Dependence on saddleness. The next bound makes the dependence on the compensation coefficient explicit. It bounds the Hessian noise in each system by its second moment relative to the corresponding positive definite mean.

Corollary F.8 (Compensation and saddleness). Under Assumption C.14, let $v \geq 0$ satisfy

$$
\begin{array} { r } { \mathbb { E } ( \Delta _ { t } ^ { + } ) ^ { 2 } \preceq v \pmb { A } ^ { + } , \qquad \mathbb { E } \pmb { \Delta } _ { t } ^ { 2 } \preceq v \pmb { A } _ { \sf P } . } \end{array}\tag{84}
$$

Then the projected and compensated critical LRs satisfy

$$
\left| \frac { 1 } { \eta _ { \star } ^ { \sf P } } - \frac { 1 } { \eta _ { \star } } \right| \leq \frac { 1 - \beta _ { 1 } } { 2 ( 1 + \beta _ { 1 } ) } \lambda _ { \sf P } + \frac { v } { 2 } .\tag{85}
$$

If Assumption $C . I 5$ also holds, with the search window specified in Appendix C.4, then the expected detector satisfies

$$
\left| \frac { 1 } { \bar { \eta } _ { \star } ( M , T ) } - \frac { 1 } { \eta _ { \star } } \right| \leq \frac { 1 - \beta _ { 1 } } { 2 ( 1 + \beta _ { 1 } ) } \lambda _ { \sf P } + \frac { v } { 2 } + H _ { T } ,\tag{86}
$$

where $H _ { T }$ is the finite-horizon term in (57).

A finite v always exists because both means are positive definite and the Hessian noise has finite second moments. The corollary does not assume that v is small. Its second condition uses $A _ { \mathsf { P } } .$ , so v can depend on the compensation coefficient and need not stay bounded as that coefficient decreases.

Recall that the effective saddleness in (17) is

$$
\nu = \frac { \operatorname* { m a x } \{ 0 , - \lambda _ { \operatorname* { m i n } } ( A ) \} } { L } , \qquad L = \lambda _ { \operatorname* { m a x } } ( A ) > 0 .
$$

When this saddleness is positive, choosing $\lambda _ { \mathsf { P } } = c \nu L$ with $c > 1$ makes $A _ { \mathsf { P } }$ positive definite. More generally, whenever a valid compensation satisfies $\lambda _ { \mathsf { P } } \leq c \nu L$ , (85) gives

$$
\left| \frac { 1 } { \eta _ { \star } ^ { \sf P } } - \frac { 1 } { \eta _ { \star } } \right| \leq \frac { c ( 1 - \beta _ { 1 } ) } { 2 ( 1 + \beta _ { 1 } ) } \nu L + \frac { v } { 2 } .
$$

Thus saddleness controls the curvature shift, while v accounts for Hessian noise. If the compensation is chosen from an estimated eigenvalue, an upper bound on $\lambda _ { \mathsf { P } }$ must also retain the estimation error. Small saddleness alone does not control v.

ProofofCorollary F.8. We first bound the change in a critical LR caused by adding noise to a positive definite mean. Applying this bound to the projected and compensated means leaves an explicit curvature shift and at most $v / 2$ from the difference of the two noise contributions.

## Step 1. Bound the noise contribution to the quotient.

Consider the AGM recurrence with a generic mean $C \succ 0$ and symmetric centered noise satisfying E $\mathbf { \Delta } ; \mathbf { \Delta } \Delta _ { t } ^ { 2 } \preceq v C$ . Recall that $L _ { C } = \lambda _ { \operatorname* { m a x } } ( \mathbf { \bar { C } } ) , c _ { b } = ( 1 - b ) / ( 1 + b )$ , and $r _ { C } ( \boldsymbol { b } )$ is the maximum of the stability quotient over nonzero positive semidefinite test matrices. For such a matrix X, the noise numerator is nonnegative because

$$
\mathbb { E } \operatorname { t r } ( X \Delta _ { t } X \Delta _ { t } ) = \mathbb { E } \left. X ^ { 1 / 2 } \Delta _ { t } X ^ { 1 / 2 } \right. _ { F } ^ { 2 } \geq 0 .
$$

For the upper bound, apply the Cauchy–Schwarz inequality in the Frobenius inner product and use the symmetry of X and $\Delta _ { t } \mathrm { : }$

$$
\begin{array} { r l r l } { \mathbb { E } \operatorname { t r } ( X \Delta _ { t } X \Delta _ { t } ) = \mathbb { E } \langle \Delta _ { t } X , X \Delta _ { t } \rangle _ { F } } & { \quad } & & { \mathrm { t h e ~ F r o b e n i u s ~ i n n e r ~ p r o d u c t } } \\ { \leq \mathbb { E } \big [ \| \Delta _ { t } X \| _ { F } \| X \Delta _ { t } \| _ { F } \big ] } & { \quad } & & { \mathrm { C a u c h y - S c h w a r z } } \\ { = \mathbb { E } \mathrm { t r } ( X ^ { 2 } \Delta _ { t } ^ { 2 } ) } & { \quad } & & { \mathrm { t h e ~ t w o ~ n o m m s ~ a r e ~ e q u a l } } \\ { = \mathrm { t r } \big ( X ^ { 2 } \mathbb { E } \Delta _ { t } ^ { 2 } \big ) } & { \quad } & & { \mathrm { l i n e a r i t y ~ o f ~ e x p e c t a t i o n } } \\ { \leq v \mathrm { t r } ( C X ^ { 2 } ) . } & { \quad } & & { \mathbb { E } \Delta _ { t } ^ { 2 } \preceq v C } \end{array}
$$

The quotient divides this numerator by $2 \operatorname { t r } ( C X ^ { 2 } ) > 0$ . Its noise contribution therefore lies between zero and $v / 2$ for every admissible test matrix. The maximum without noise equals $c _ { b } L _ { C } / 2$ by (75). Taking maxima gives, for every $- 1 < b < 1$

$$
\frac { c _ { b } L _ { C } } { 2 } \leq r _ { C } ( b ) \leq \frac { c _ { b } L _ { C } } { 2 } + \frac { v } { 2 } .
$$

## Step 2. Compare with the critical LR without noise.

Let $\eta _ { \mathrm { c r i t } }$ be the critical rate for this noisy recurrence. Recall from (76) that the same mean without noise has critical rate

$$
\eta _ { \mathrm { d e t } } = \frac { 2 ( 1 + \beta _ { 1 } ) } { ( 1 - \beta _ { 1 } ) L _ { C } + 2 \beta _ { 1 } \lambda _ { \mathrm { W D } } } .
$$

By the exact stability criterion (Theorem F.1), adding the nonnegative noise term can only remove strictly stable LRs of the homogeneous recurrence. Hence $\eta _ { \mathrm { c r i t } } \leq \eta _ { \mathrm { d e t } }$ . Their effective momenta are

$$
b _ { \mathrm { c r i t } } : = \beta _ { 1 } ( 1 - \eta _ { \mathrm { c r i t } } \lambda _ { \mathrm { W D } } ) , \qquad b _ { \mathrm { d e t } } : = \beta _ { 1 } ( 1 - \eta _ { \mathrm { d e t } } \lambda _ { \mathrm { W D } } ) .
$$

Both lie in (−1, 1) by Corollary F.4, and

$$
\begin{array} { r l } & { b _ { \mathrm { c r i t } } - b _ { \mathrm { d e t } } = \beta _ { 1 } \lambda _ { \mathrm { W D } } ( \eta _ { \mathrm { d e t } } - \eta _ { \mathrm { c r i t } } ) } \\ & { ~ \geq 0 . } \end{array}
$$

Recall that the critical-rate identity (74) adds the same weight-decay offset to the inverse rate in both systems. Since $r _ { C }$ is nonincreasing in its momentum argument,

$$
\begin{array} { r l r l } { \displaystyle \frac { 1 } { \eta _ { \mathrm { c r i t } } } + \frac { \beta _ { 1 } \lambda _ { \mathrm { W D } } } { 1 - \beta _ { 1 } } = r _ { C } ( b _ { \mathrm { c r i t } } ) } & { } & & { \mathrm { c r i t i c a l - r a t e ~ i d e n t i t y } } \\ { \displaystyle \leq r _ { C } ( b _ { \mathrm { d e t } } ) } & { } & & { \mathrm { m o n o t o n i c i t y ~ i n ~ } b } \\ { \displaystyle \leq \frac { c _ { b _ { \mathrm { d e t } } } L _ { C } } { 2 } + \frac { v } { 2 } } & { } & & { \mathrm { S t e p ~ 1 } } \\ { \displaystyle = \frac { 1 } { \eta _ { \mathrm { d e t } } } + \frac { \beta _ { 1 } \lambda _ { \mathrm { W D } } } { 1 - \beta _ { 1 } } + \frac { v } { 2 } . } & { } & & { \mathrm { b o u n d a r y ~ w i t h o u t ~ n o i s e } } \end{array}
$$

Subtracting the common offset and using $\eta _ { \mathrm { c r i t } } \leq \eta _ { \mathrm { d e t } }$ gives

$$
0 \leq \frac { 1 } { \eta _ { \mathrm { c r i t } } } - \frac { 1 } { \eta _ { \mathrm { d e t } } } \leq \frac { v } { 2 } .
$$

This argument uses monotonicity on the full interval (−1, 1) and therefore also covers negative effective critical momenta.

## Step 3. Apply the bound to the two systems.

The projected system has mean $A ^ { + }$ , noise $\Delta _ { t } ^ { + }$ and largest mean eigenvalue L. The compensated system has mean $A _ { \mathsf { P } }$ , noise $\Delta _ { t }$ and largest mean eigenvalue $L + \lambda _ { \mathsf { P } }$ . The two inequalities in (84) allow us to apply Step 2 with the same v:

$$
\begin{array} { r l } & { 0 \leq \displaystyle \frac { 1 } { \eta _ { \star } } - \frac { ( 1 - \beta _ { 1 } ) L + 2 \beta _ { 1 } \lambda _ { \mathrm { W D } } } { 2 ( 1 + \beta _ { 1 } ) } \leq \frac { v } { 2 } , } \\ & { 0 \leq \displaystyle \frac { 1 } { \eta _ { \star } ^ { \sf P } } - \frac { ( 1 - \beta _ { 1 } ) ( L + \lambda _ { \sf P } ) + 2 \beta _ { 1 } \lambda _ { \mathrm { W D } } } { 2 ( 1 + \beta _ { 1 } ) } \leq \frac { v } { 2 } . } \end{array}
$$

The difference between the two displayed noise contributions lies in $[ - v / 2 , v / 2 ]$ . The deterministic terms differ by

$$
{ \begin{array} { r l r l } & { { \frac { ( 1 - \beta _ { 1 } ) ( L + \lambda _ { \mathrm { P } } ) + 2 \beta _ { 1 } \lambda _ { \mathrm { W D } } } { 2 ( 1 + \beta _ { 1 } ) } } - { \frac { ( 1 - \beta _ { 1 } ) L + 2 \beta _ { 1 } \lambda _ { \mathrm { W D } } } { 2 ( 1 + \beta _ { 1 } ) } } } \\ & { \ = { \frac { ( 1 - \beta _ { 1 } ) ( L + \lambda _ { \mathrm { P } } ) - ( 1 - \beta _ { 1 } ) L } { 2 ( 1 + \beta _ { 1 } ) } } } & & { { \mathrm { c a n c e l ~ t h e ~ w e i g h t - d e c a y ~ t e r m s } } } \\ & { \ = { \frac { 1 - \beta _ { 1 } } { 2 ( 1 + \beta _ { 1 } ) } } \lambda _ { \mathrm { P } } . } \end{array} }
$$

To subtract the two noise contributions, use the upper endpoint of one interval and the lower endpoint of the other:

$$
\begin{array} { r l } & { \frac { 1 } { \eta _ { \star } ^ { \mathsf { P } } } - \frac { 1 } { \eta _ { \star } } - \frac { 1 - \beta _ { 1 } } { 2 ( 1 + \beta _ { 1 } ) } \lambda _ { \sf P } } \\ & { \quad = \bigg [ \frac { 1 } { \eta _ { \star } ^ { \mathsf { P } } } - \frac { ( 1 - \beta _ { 1 } ) ( L + \lambda _ { \sf P } ) + 2 \beta _ { 1 } \lambda _ { \sf W D } } { 2 ( 1 + \beta _ { 1 } ) } \bigg ] } \\ & { \qquad - \left[ \frac { 1 } { \eta _ { \star } } - \frac { ( 1 - \beta _ { 1 } ) L + 2 \beta _ { 1 } \lambda _ { \sf W D } } { 2 ( 1 + \beta _ { 1 } ) } \right] } \\ & { \qquad \leq \frac { v } { 2 } - 0 = \frac { v } { 2 } . } \end{array}
$$

Using the opposite endpoints in the same identity gives

$$
\frac { 1 } { \eta _ { \star } ^ { \mathsf { P } } } - \frac { 1 } { \eta _ { \star } } - \frac { 1 - \beta _ { 1 } } { 2 ( 1 + \beta _ { 1 } ) } \lambda _ { \mathsf { P } } \ge 0 - \frac { v } { 2 } = - \frac { v } { 2 } .
$$

Since the curvature shift is nonnegative, the triangle inequality gives

$$
\begin{array} { r l r } {  { | \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } - \frac { 1 } { \eta _ { \star } } | \leq \frac { 1 - \beta _ { 1 } } { 2 ( 1 + \beta _ { 1 } ) } \lambda _ { \mathrm { P } } + | \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } - \frac { 1 } { \eta _ { \star } } - \frac { 1 - \beta _ { 1 } } { 2 ( 1 + \beta _ { 1 } ) } \lambda _ { \mathrm { P } } | } } \\ & { } & { \leq \frac { 1 - \beta _ { 1 } } { 2 ( 1 + \beta _ { 1 } ) } \lambda _ { \mathrm { P } } + \frac { v } { 2 } . } \end{array}
$$

This proves (85).

For the expected detector, Proposition F.6 bounds the inverse of every accepted rate below by $1 / \eta _ { \star } ^ { \mathsf { P } } - H _ { T } ^ { \mathsf { ^ { * } } }$ . Taking their infimum gives the same lower bound for $1 / \bar { \eta } _ { \star } ( \dot { M } , T )$ . Acceptance of $\eta _ { \star } ^ { \mathsf { P } }$ in Assumption C.15 gives $\bar { \eta } _ { \star } ( M , T ) \bar { \geq } \eta _ { \star } ^ { \mathsf { P } }$ . Together these imply

$$
- H _ { T } \leq \frac { 1 } { \bar { \eta } _ { \star } ( M , T ) } - \frac { 1 } { \eta _ { \star } ^ { \sf p } } \leq 0 .
$$

Split the inverse-rate error at $\eta _ { \star } ^ { \mathsf { P } }$ and apply the triangle inequality:

$$
\begin{array} { r l } & { \left| \frac { 1 } { \bar { \eta } _ { \star } ( M , T ) } - \frac { 1 } { \eta _ { \star } } \right| = \left| \left( \frac { 1 } { \bar { \eta } _ { \star } ( M , T ) } - \frac { 1 } { \eta _ { \star } ^ { \mathrm { p } } } \right) + \left( \frac { 1 } { \eta _ { \star } ^ { \mathrm { p } } } - \frac { 1 } { \eta _ { \star } } \right) \right| } \\ & { \qquad \leq \left| \frac { 1 } { \bar { \eta } _ { \star } ( M , T ) } - \frac { 1 } { \eta _ { \star } ^ { \mathrm { p } } } \right| + \left| \frac { 1 } { \eta _ { \star } ^ { \mathrm { p } } } - \frac { 1 } { \eta _ { \star } } \right| } \\ & { \qquad \leq H _ { T } + \frac { 1 - \beta _ { 1 } } { 2 ( 1 + \beta _ { 1 } ) } \lambda _ { \mathrm { P } } + \frac { v } { 2 } . } \end{array}
$$

This proves (86) and completes the proof of Corollary F.8.

Small saddleness alone is insufficient. For $0 < s \le 1$ , take $P = I _ { \mathrm { 2 } } , \beta _ { 1 } = \lambda _ { \mathrm { W D } } = 0$

$$
\begin{array} { r } { { \cal H } _ { { \cal B } _ { t } } = \mathrm { d i a g } ( 1 , - s + \sqrt { 2 s } \xi _ { t } ) , \qquad \operatorname* { P r } ( \xi _ { t } = \pm 1 ) = \frac { 1 } { 2 } , \qquad \lambda _ { \mathrm { P } } = 2 s . } \end{array}
$$

The saddleness is s. The first coordinate, which spans the positive eigenspace, is noiseless, while the compensated second coordinate has mean s and noise variance 2s. Here $A _ { \mathsf { P } } = \mathrm { d i a g } ( 1 + 2 s , s )$ and $\Delta _ { t } ^ { + } = 0$ . To evaluate the covariance mismatch, recall that $\delta _ { \mathrm { c o v } }$ maximizes the normalized noise difference in (55). For a symmetric $2 \times 2$ test matrix with entries $X _ { i j }$

$$
\begin{array} { r l } & { \mathbb { E } \operatorname { t r } ( X \Delta _ { t } X \Delta _ { t } ) = 2 s X _ { 2 2 } ^ { 2 } , } \\ & { \quad \quad \quad \quad \operatorname { t r } ( A _ { \mathbb { P } } X ^ { 2 } ) = ( 1 + 2 s ) ( X _ { 1 1 } ^ { 2 } + X _ { 1 2 } ^ { 2 } ) + s ( X _ { 1 2 } ^ { 2 } + X _ { 2 2 } ^ { 2 } ) , } \\ & { \quad \quad \quad \frac { \mathbb { E } \operatorname { t r } ( X \Delta _ { t } X \Delta _ { t } ) } { 2 \operatorname { t r } ( A _ { \mathbb { P } } X ^ { 2 } ) } = \frac { s X _ { 2 2 } ^ { 2 } } { ( 1 + 2 s ) ( X _ { 1 1 } ^ { 2 } + X _ { 1 2 } ^ { 2 } ) + s ( X _ { 1 2 } ^ { 2 } + X _ { 2 2 } ^ { 2 } ) } \leq 1 . } \end{array}
$$

Equality holds at the admissible matrix $\pmb { X } = \mathrm { d i a g } ( 0 , 1 )$ , so $\delta _ { \mathrm { c o v } } = 1$ . The scalar threshold above and the definition of Φ now give

$$
\begin{array} { l } { \displaystyle \frac { 1 } { \eta _ { \star } } = \frac { 1 } { 2 } , \qquad \displaystyle \Phi ( 1 ) = s , \qquad \delta _ { \mathrm { c o v } } = 1 , } \\ { \displaystyle \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } = \operatorname* { m a x } \left\{ \frac { 1 + 2 s } { 2 } , 1 + \frac { s } { 2 } \right\} = 1 + \frac { s } { 2 } . } \end{array}
$$

The last equality uses $s \leq 1$ . The inverse-rate difference tends to $1 / 2 \arg s  0$ . The discarded noise prevents the compensation error from vanishing, despite vanishing saddleness.

## F.4 FINITE-HORIZON ERROR

The homogeneous compensated recurrence, with mean $ { \boldsymbol { A } } _ { \sf P } =  { \boldsymbol { A } } + \lambda _ { \sf P }  { \boldsymbol { I } }$ and centered noise $\Delta _ { t } =$ $A _ { B _ { t } } - A$ , is

$$
\begin{array} { c } { { { \pmb x } _ { t + 1 } = \big [ ( 1 + b ) { \pmb I } - \alpha ( { \pmb A } _ { \mathrm { P } } + { \pmb \Delta } _ { t } ) \big ] { \pmb x } _ { t } - b { \pmb x } _ { t - 1 } , } } \\ { { b = \beta _ { 1 } ( 1 - \eta \lambda _ { \mathrm { W D } } ) , \qquad \alpha = \eta ( 1 - \beta _ { 1 } ) , } } \end{array}
$$

Its second-moment map on the history state $\pmb { y } _ { t } = ( \pmb { x } _ { t } , \pmb { x } _ { t - 1 } ) ^ { \top }$ is $\mathcal { T } _ { \eta } ^ { \mathsf { P } }$ . The mean and noise law stay fixed as the candidate LR η varies.

Acceptance means $\mathbb { E } \left\| \epsilon _ { t } \right\| ^ { 2 } \leq M ^ { 2 }$ for every unit start and $0 \leq t \leq T$ in the original affine run. We convert this condition into a moment-radius bound, then apply Proposition F.2. Recall the uniform radius bound from (56),

$$
\rho _ { T } = \left[ \frac { M ( 1 + \beta _ { 1 } ) } { 1 - \beta _ { 1 } } \right] ^ { 2 / T } .
$$

It depends on M and $T ,$ , not on the candidate LR. The last proposition gives a sufficient cutoff for the original affine run to be accepted at $\eta _ { \star } ^ { \mathsf { P } }$

## F.4.1 BOUNDING THE MOMENT RADIUS

Lemma F.9 (The expected-displacement criterion bounds the moment radius). Under Assumption C.14, every $L R \eta \in \mathcal { T }$ accepted by the detector (50) satisfies

$$
\rho ( \mathcal { T } _ { \eta } ^ { \mathsf { P } } ) \leq \left[ \frac { M ( 1 + \beta _ { 1 } \vert 1 - \eta \lambda _ { \mathsf { W D } } \vert ) } { 1 - \beta _ { 1 } } \right] ^ { 2 / T } \leq \rho _ { T } .\tag{87}
$$

ProofofLemma F.9. Fix an accepted LR η and write $\rho : = \rho ( \mathcal { T } _ { \eta } ^ { \mathsf { P } } )$ . If $\rho < 1$ , the claim follows because the proposed upper bound is greater than one. We therefore assume $\rho \geq 1$ . Our goal is to prove

$$
\rho ^ { T / 2 } \leq \frac { M ( 1 + b ) } { 1 - \beta _ { 1 } } , \qquad b = \beta _ { 1 } ( 1 - \eta \lambda _ { \mathrm { W D } } ) .
$$

The detector bounds displacement, whereas $\rho$ describes growth of the full second moment. We connect the two by choosing a nonnegative quadratic form whose expectation grows by the factor $\rho$ at each step. The three steps below remove the affine forcing, construct this quadratic form, and bound its value at time T using the detector.

## Step 1. Bound the homogeneous trajectory.

For any unit vector v, run the original affine simulation (49) twice, from initial displacements v and $- v ,$ , using the same mini-batches and the same checkpoint momentum. Denote their positions and momenta by $( \epsilon _ { t } ^ { + } , m _ { t } ^ { + } )$ and $( \epsilon _ { t } ^ { - } , m _ { t } ^ { - } )$ . Within this proof, set

$$
\epsilon _ { t } : = \frac { \epsilon _ { t } ^ { + } - \epsilon _ { t } ^ { - } } { 2 } , \qquad p _ { t } : = \frac { \eta } { 2 } P ^ { - 1 } ( m _ { t } ^ { + } - m _ { t } ^ { - } ) .
$$

Here ${ \mathbf { } } p _ { t }$ is the momentum difference scaled into displacement units. Thus $\epsilon _ { \mathrm { 0 } } = v$ and $p _ { 0 } = 0$ Subtracting the two momentum updates cancels their common batch gradient. Subtracting the two position updates also cancels the checkpoint weight-decay drift. The resulting equations are

$$
\begin{array} { r l } & { p _ { t + 1 } = \beta _ { 1 } p _ { t } + \alpha ( P ^ { - 1 } H _ { \mathcal B _ { t } } + \lambda _ { \mathsf { P } } I ) \epsilon _ { t } , } \\ & { \epsilon _ { t + 1 } = ( 1 - \eta \lambda _ { \mathrm { W D } } ) \epsilon _ { t } - p _ { t + 1 } , \qquad \alpha = \eta ( 1 - \beta _ { 1 } ) . } \end{array}
$$

Here $H _ { B _ { t } }$ is the raw mini-batch Hessian. Weight decay remains in the separate position factor $1 - \eta \lambda _ { \mathrm { W D } }$ . These are the same homogeneous compensated dynamics as before, expressed in the detector’s original coordinates.

Both affine runs pass the detector because their initial displacements have unit norm. Consequently, for every $0 \leq t \bar { \leq } T$

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left\| \epsilon _ { t } \right\| ^ { 2 } = \frac 1 4 \mathbb { E } \left\| \epsilon _ { t } ^ { + } - \epsilon _ { t } ^ { - } \right\| ^ { 2 } } \\ { \displaystyle \leq \frac 1 2 \mathbb { E } \big ( \big \| \epsilon _ { t } ^ { + } \big \| ^ { 2 } + \big \| \epsilon _ { t } ^ { - } \big \| ^ { 2 } \big ) } \\ { \displaystyle \leq \frac 1 2 ( M ^ { 2 } + M ^ { 2 } ) = M ^ { 2 } . } \end{array}
$$

The second line uses $\left\| \pmb { a } - \pmb { b } \right\| ^ { 2 } \leq 2 \left\| \pmb { a } \right\| ^ { 2 } + 2 \left\| \pmb { b } \right\| ^ { 2 }$ . To control the momentum at time $T ,$ use the preceding position update:

$$
p _ { T } = \big ( 1 - \eta \lambda _ { \mathrm { W D } } \big ) \epsilon _ { T - 1 } - \epsilon _ { T } .
$$

The search window guarantees $0 \leq 1 - \eta \lambda _ { \mathrm { W D } } \leq 1$ , and $T \geq 1$ . Minkowski’s inequality therefore gives

$$
\begin{array} { r l } & { ( \mathbb { E } \left\| p _ { T } \right\| ^ { 2 } ) ^ { 1 / 2 } \leq ( 1 - \eta \lambda _ { \mathrm { W D } } ) ( \mathbb { E } \left\| \epsilon _ { T - 1 } \right\| ^ { 2 } ) ^ { 1 / 2 } + ( \mathbb { E } \left\| \epsilon _ { T } \right\| ^ { 2 } ) ^ { 1 / 2 } } \\ & { \qquad \leq ( 1 - \eta \lambda _ { \mathrm { W D } } ) M + M } \\ & { \qquad = ( 2 - \eta \lambda _ { \mathrm { W D } } ) M . } \end{array}
$$

We have now bounded both components of the homogeneous state, for every unit initial displacement and zero initial momentum.

## Step 2. Measure growth with a quadratic form.

Write $z _ { t } : = ( \epsilon _ { t } , p _ { t } ) ^ { \top }$ . The two updates in Step 1 give

$$
\begin{array} { r l } & { z _ { t + 1 } = \widehat { F } _ { t } z _ { t } , } \\ & { ~ \widehat { F } _ { t } : = \left( ^ { \left( 1 - \eta \lambda _ { \mathrm { W D } } \right) I - \alpha \left( P ^ { - 1 } H _ { B _ { t } } + \lambda _ { \mathrm { P } } I \right) } - \beta _ { 1 } I \right) . } \\ & { ~ \alpha ( P ^ { - 1 } H _ { B _ { t } } + \lambda _ { \mathrm { P } } I ) ~ \beta _ { 1 } I } \end{array}\tag{88}
$$

This state is related to the existing position–momentum state of (69) by $\pmb { x } _ { t } = P ^ { 1 / 2 } \pmb { \epsilon } _ { t }$ and $\mathbf { \delta } _ { \mathbf { n } _ { t } } =$ $\eta ^ { - 1 } P ^ { 1 / 2 } p _ { t }$ . This change is invertible for every $\eta > 0$ . The moment map for $\widehat { F } _ { t }$ thus has spectral radius $\rho ,$ by the coordinate argument in Appendix F.6, which also covers $\eta \lambda _ { \mathrm { W D } } = 1$

We seek a matrix $W \succeq 0 , W \neq 0$ , such that

$$
\mathbb { E } \big [ \widehat { F } _ { t } ^ { \top } W \widehat { F } _ { t } \big ] = \rho W .
$$

This identity has a direct purpose: it makes the expected value of $z _ { t } ^ { \top } \mathbf { \boldsymbol { W } } z _ { t }$ multiply by $\rho$ after one step. Such a matrix exists by the Perron–Frobenius theorem for PSD-preserving maps (Lemma F.16), applied to the adjoint moment map $W \mapsto \mathbb { E } [ \widehat { \pmb { F } } _ { t } ^ { \top } \pmb { W } \widehat { \pmb { F } } _ { t } ]$ . The adjoint has the same spectral radius as the moment map and preserves PSD matrices.

Let $W _ { 1 1 }$ and $W _ { 2 2 }$ be the two diagonal blocks of W, corresponding to displacement and scaled momentum. We next bound their relative size, so that Step 1 can control this quadratic form. Since $W \succeq 0$ , the Euclidean triangle inequality gives, for any vectors ${ \mathbf { } } u , v ,$

$$
\begin{array} { r l } & { \left[ \left( \begin{array} { l } { \pmb { u } } \\ { \pmb { v } } \end{array} \right) ^ { \top } \pmb { W } \left( \begin{array} { l } { \pmb { u } } \\ { \pmb { v } } \end{array} \right) \right] ^ { 1 / 2 } = \left\| \pmb { W } ^ { 1 / 2 } \left( \begin{array} { l } { \pmb { u } } \\ { 0 } \end{array} \right) + \pmb { W } ^ { 1 / 2 } \left( \begin{array} { l } { 0 } \\ { \pmb { v } } \end{array} \right) \right\| } \\ & { \quad \leq ( \pmb { u } ^ { \top } \pmb { W } _ { 1 1 } \pmb { u } ) ^ { 1 / 2 } + ( \pmb { v } ^ { \top } \pmb { W } _ { 2 2 } \pmb { v } ) ^ { 1 / 2 } . } \end{array}
$$

The momentum column of $\widehat { F } _ { t }$ is deterministic: $\widehat { F } _ { t } ( 0 , \pmb { v } ) ^ { \top } = \beta _ { 1 } ( - \pmb { v } , \pmb { v } ) ^ { \top }$ . Testing the eigenmatrix identity on (0, v) therefore gives

$$
\begin{array} { r l } & { \rho \boldsymbol { v } ^ { \top } \boldsymbol { W } _ { 2 2 } \boldsymbol { v } = \left( \begin{array} { l } { 0 } \\ { \boldsymbol { v } } \end{array} \right) ^ { \top } \mathbb { E } [ \widehat { \boldsymbol { F } } _ { t } ^ { \top } \boldsymbol { W } \widehat { \boldsymbol { F } } _ { t } ] \left( \begin{array} { l } { 0 } \\ { \boldsymbol { v } } \end{array} \right) } \\ & { \qquad = \beta _ { 1 } ^ { 2 } \left( \begin{array} { l } { - \boldsymbol { v } } \\ { \boldsymbol { v } } \end{array} \right) ^ { \top } \boldsymbol { W } \left( \begin{array} { l } { - \boldsymbol { v } } \\ { \boldsymbol { v } } \end{array} \right) } \\ & { \qquad \leq \beta _ { 1 } ^ { 2 } \left[ \left( \boldsymbol { v } ^ { \top } \boldsymbol { W } _ { 1 1 } \boldsymbol { v } \right) ^ { 1 / 2 } + \left( \boldsymbol { v } ^ { \top } \boldsymbol { W } _ { 2 2 } \boldsymbol { v } \right) ^ { 1 / 2 } \right] ^ { 2 } . } \end{array}
$$

Taking square roots and moving the $W _ { 2 2 }$ term to the left gives, for every $^ { v , }$

$$
( \sqrt { \rho } - \beta _ { 1 } ) ( \pmb { v } ^ { \top } \pmb { W } _ { 2 2 } \pmb { v } ) ^ { 1 / 2 } \leq \beta _ { 1 } ( \pmb { v } ^ { \top } \pmb { W } _ { 1 1 } \pmb { v } ) ^ { 1 / 2 } .
$$

The displacement block $W _ { 1 1 }$ is nonzero. Otherwise $\sqrt { \rho } - \beta _ { 1 } > 0$ would force $W _ { 2 2 } = 0$ . The triangle inequality above would then give $z ^ { \top } W z = 0$ for every z, contradicting $W \neq 0$

Rescale W so that $\| \mathbfcal { W } _ { 1 1 } \| = 1$ , where the norm is the spectral norm. The eigenmatrix identity is unchanged by this positive scalar rescaling. Dividing the preceding inequality by $\sqrt { \rho } - \beta _ { 1 } > 0$ and maximizing over unit v gives

$$
\begin{array} { r l } & { \sqrt { \| W _ { 2 2 } \| } = \displaystyle \operatorname* { m a x } _ { \| v \| = 1 } ( v ^ { \top } W _ { 2 2 } v ) ^ { 1 / 2 } } \\ & { \qquad \leq \displaystyle \frac { \beta _ { 1 } } { \sqrt { \rho } - \beta _ { 1 } } \displaystyle \operatorname* { m a x } _ { \| v \| = 1 } ( v ^ { \top } W _ { 1 1 } v ) ^ { 1 / 2 } } \\ & { \qquad = \displaystyle \frac { \beta _ { 1 } } { \sqrt { \rho } - \beta _ { 1 } } \leq \frac { \beta _ { 1 } } { 1 - \beta _ { 1 } } . } \end{array}
$$

The last line uses $\| \mathbfcal { W } _ { 1 1 } \| = 1$ and $\rho \geq 1$ . Applying the same triangle inequality to an arbitrary state $z = ( \epsilon , p ) ^ { \top }$ now yields

$$
\begin{array} { r l } & { ( z ^ { \top } W z ) ^ { 1 / 2 } \leq ( \epsilon ^ { \top } W _ { 1 1 } \epsilon ) ^ { 1 / 2 } + ( p ^ { \top } W _ { 2 2 } p ) ^ { 1 / 2 } } \\ & { \qquad \leq \| \epsilon \| + \sqrt { \| W _ { 2 2 } \| } \| p \| } \\ & { \qquad \leq \| \epsilon \| + \frac { \beta _ { 1 } } { 1 - \beta _ { 1 } } \| p \| . } \end{array}\tag{89}
$$

Step 3. Compare exact growth with the detector bounds.

Choose a unit top eigenvector v of $W _ { 1 1 }$ . Our normalization gives ${ \pmb v } ^ { \top } { \pmb W } _ { 1 1 } { \pmb v } = 1$ . Use the homogeneous trajectory from Step 1 with this initial displacement and zero initial momentum. Its initial quadratic form is

$$
z _ { 0 } ^ { \top } W z _ { 0 } = { \binom { v } { 0 } } ^ { \top } W \left( { v } \right) = { v } ^ { \top } W _ { 1 1 } { v } = 1 .
$$

This choice is allowed because acceptance covers every unit initial displacement.

The batch at time t is independent of ${ \boldsymbol { z } } _ { t } .$ . The tower property and the eigenmatrix identity give

$$
\begin{array} { r l } & { \mathbb { E } [ z _ { t + 1 } ^ { \top } W z _ { t + 1 } ] = \mathbb { E } [ z _ { t } ^ { \top } \widehat { F } _ { t } ^ { \top } W \widehat { F } _ { t } z _ { t } ] } \\ & { \phantom { = \mathbb { E } \left[ z _ { t } ^ { \top } \mathbb { E } \left[ \widehat { F } _ { t } ^ { \top } W \widehat { F } _ { t } \mid z _ { t } \right] z _ { t } \right] } } \\ & { = \rho \mathbb { E } [ z _ { t } ^ { \top } W z _ { t } ] . } \end{array}
$$

Iterating from the initial value one gives $\mathbb { E } [ z _ { T } ^ { \top } \pmb { W } z _ { T } ] = \rho ^ { T }$ . On the other hand, (89), Minkowski’s inequality and the bounds from Step 1 imply

$$
\begin{array} { r l } { \rho ^ { T / 2 } = \left( \mathbf { E } \left[ z _ { T } ^ { - \top } W z _ { T } \right] \right) ^ { 1 / 2 } } & { } \\ & { \leq \left[ \mathbb { E } \left( \left\| t e _ { T } \right\| + \frac { \beta _ { 1 } } { 1 - \beta _ { 1 } } \left\| p r \right\| \right) ^ { 2 } \right] ^ { 1 / 2 } } \\ & { \leq ( \mathbb { E } \left\| t e _ { T } \right\| ^ { 2 } ) ^ { 1 / 2 } + \frac { \beta _ { 1 } } { 1 - \beta _ { 1 } } ( \mathbb { E } \left\| p r \right\| ^ { 2 } ) ^ { 1 / 2 } } \\ & { \leq M + \frac { \beta _ { 1 } } { 1 - \beta _ { 1 } } ( 2 - \eta \lambda w \mathbf { D } ) M } \\ & { = \frac { M \left[ 1 - \beta _ { 1 } + 2 \beta _ { 1 } - \beta _ { 1 } \eta \lambda w \mathbf { D } \right] } { 1 - \beta _ { 1 } } } \\ & { = \frac { M ( 1 + b ) } { 1 - \beta _ { 1 } } . } \end{array}
$$

Finally, $b = \beta _ { 1 } ( 1 - \eta \lambda _ { \mathrm { W D } } ) = \beta _ { 1 } | 1 - \eta \lambda _ { \mathrm { W D } } |$ and $b \leq \beta _ { 1 }$ on the search window. Taking the power $2 / T$

$$
\begin{array} { r l } & { \rho \leq \bigg [ \frac { M ( 1 + b ) } { 1 - \beta _ { 1 } } \bigg ] ^ { 2 / T } } \\ & { \quad = \bigg [ \frac { M ( 1 + \beta _ { 1 } | 1 - \eta \lambda _ { \mathrm { W D } } | ) } { 1 - \beta _ { 1 } } \bigg ] ^ { 2 / T } } \\ & { \quad \leq \bigg [ \frac { M ( 1 + \beta _ { 1 } ) } { 1 - \beta _ { 1 } } \bigg ] ^ { 2 / T } = \rho _ { T } . } \end{array}
$$

This completes the proof of Lemma F.9.

Remark F.10 (Fixed weighted displacement norms). For any fixed $W _ { 0 } \succ 0$ , Lemma F.9 holds with the same constant when both the unit initialization and the displacement cutoff use $\| \epsilon \| _ { W _ { 0 } } ^ { 2 } = \epsilon ^ { \top } W _ { 0 } \epsilon \mathrm { : }$ replacing $( \epsilon , p )$ by $( W _ { 0 } ^ { 1 / 2 } \epsilon , W _ { 0 } ^ { 1 / 2 } p )$ preserves the scalar $\beta _ { 1 }$ and $1 - \eta \lambda _ { \mathrm { W D } }$ blocks in the proof and changes the moment map by a similarity. This includes the fixed preconditioner norm used by the implemented detector.

## F.4.2 BOUNDING THE INVERSE LR

Proof of Proposition F.6. We apply the quantitative stability bound from Proposition F.2 to the compensated system. Its critical test matrix then converts the quotient difference into the desired inverse-rate difference.

Fix an accepted LR $\eta \in \mathcal { I }$ . If $\eta \leq \eta _ { \star } ^ { \mathsf { P } }$ , then $1 / \eta \geq 1 / \eta _ { \star } ^ { \mathsf { P } }$ , as claimed. Assume $\eta > \eta _ { \star } ^ { \mathsf { P } }$ from now on. Recall the candidate and critical coefficients,

$$
\begin{array} { r }  b = \beta _ { 1 } ( 1 - \eta \lambda _ { \mathrm { W D } } ) , \alpha = ( 1 - \beta _ { 1 } ) \eta , \ \end{array}
$$

The search window and $\eta > \eta _ { \star } ^ { \mathsf { P } }$ imply

$$
0 \leq b \leq b _ { \star } ^ { \mathsf { P } } \leq \beta _ { 1 } < 1 , \qquad \alpha > \alpha _ { \star } ^ { \mathsf { P } } > 0 .
$$

The matrix $X _ { \star } ^ { \mathsf { P } } \succeq 0$ maximizes the compensated quotient at its critical rate, with $\| X _ { \star } ^ { \mathsf { P } } \| _ { F } = 1$ . Its curvature is $\mu _ { \star } ^ { \mathsf { P } } = \mathrm { t r } ( A _ { \mathsf { P } } ( X _ { \star } ^ { \mathsf { P } } ) ^ { 2 } ) > 0$

## Step 1. Use acceptance to bound the quotient.

Lemma F.9 gives $\rho ( \mathcal { T } _ { \eta } ^ { \mathsf { P } } ) \leq \rho _ { T }$ . Apply Proposition F.2 with the compensated mean $C = A _ { \mathsf { P } }$ , its full noise $\Delta _ { t }$ , radius bound $R = \rho _ { T }$ , and test matrix $X = X _ { \star } ^ { \mathsf { P } }$ . This gives

$$
q _ { A \mathrm { p } , b } ( \mathbf { X _ { \star } ^ { \mathrm { P } } } ) - \frac { 1 - b } { \alpha } \leq ( \rho _ { T } - 1 ) \left[ \frac { 2 b } { ( \rho _ { T } + b ) \alpha } + \frac { ( 1 - b / \rho _ { T } ) ^ { 2 } } { 2 \mu _ { \star } ^ { \mathrm { P } } \alpha ^ { 2 } } \right] .\tag{90}
$$

## Step 2. Convert the quotient into an inverse-rate difference.

At the critical rate, the quotient maximum equals the normalized inverse LR. Recall (74) and the AGM parameter curve:

$$
\begin{array} { r l } & { q _ { A _ { \mathrm { P } } , b _ { \star } ^ { \mathrm { P } } } ( X _ { \star } ^ { \mathrm { P } } ) = \cfrac { 1 - b _ { \star } ^ { \mathrm { P } } } { \alpha _ { \star } ^ { \mathrm { P } } } = \cfrac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } + \cfrac { \beta _ { 1 } \lambda _ { \mathrm { W D } } } { 1 - \beta _ { 1 } } , } \\ & { \qquad \cfrac { 1 - b } { \alpha } = \cfrac { 1 - \beta _ { 1 } + \beta _ { 1 } \eta \lambda _ { \mathrm { W D } } } { ( 1 - \beta _ { 1 } ) \eta } = \cfrac { 1 } { \eta } + \cfrac { \beta _ { 1 } \lambda _ { \mathrm { W D } } } { 1 - \beta _ { 1 } } . } \end{array}
$$

Subtracting cancels the same weight-decay offset:

$$
\begin{array} { l } { { \displaystyle q _ { A \mathrm { P } , b _ { \star } ^ { \mathrm { P } } } ( { \bf X } _ { \star } ^ { \mathrm { P } } ) - \frac { 1 - b } { \alpha } = \left( \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } + \frac { \beta _ { 1 } \lambda _ { \mathrm { W D } } } { 1 - \beta _ { 1 } } \right) - \left( \frac { 1 } { \eta } + \frac { \beta _ { 1 } \lambda _ { \mathrm { W D } } } { 1 - \beta _ { 1 } } \right) } } \\ { { \displaystyle \qquad = \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } - \frac { 1 } { \eta } } . } \end{array}\tag{91}
$$

Because $b \leq b _ { \star } ^ { \mathsf { P } }$ , the pointwise momentum inequality (67) gives $q _ { A _ { \mathsf { P } } , b _ { \star } ^ { \mathsf { P } } } ( X _ { \star } ^ { \mathsf { P } } ) \leq q _ { A _ { \mathsf { P } } , b } ( X _ { \star } ^ { \mathsf { P } } )$ . Hence

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { \eta _ { \star } ^ { \mathsf { P } } } - \frac { 1 } { \eta } = q _ { A \mathsf { P } , b _ { \star } ^ { \mathsf { P } } } ( X _ { \star } ^ { \mathsf { P } } ) - \frac { 1 - b } { \alpha } } \\ & { \qquad \le q _ { A \mathsf { P } , b } ( X _ { \star } ^ { \mathsf { P } } ) - \frac { 1 - b } { \alpha } } \\ & { \qquad \le ( \rho _ { T } - 1 ) \left[ \frac { 2 b } { ( \rho _ { T } + b ) \alpha } + \frac { 1 } { 2 \mu _ { \star } ^ { \mathsf { P } } } \left( \frac { 1 - b / \rho _ { T } } { \alpha } \right) ^ { 2 } \right] . } \end{array}
$$

The last line uses (90). It remains to replace the candidate coefficients by the fixed critical coefficients.

## Step 3. Obtain the uniform bound $H _ { T }$

For the first term, $0 \leq b \leq b _ { \star } ^ { \mathsf { P } }$ gives

$$
\frac { b _ { \star } ^ { \mathsf { P } } } { \rho _ { T } + b _ { \star } ^ { \mathsf { P } } } - \frac { b } { \rho _ { T } + b } = \frac { \rho _ { T } ( b _ { \star } ^ { \mathsf { P } } - b ) } { ( \rho _ { T } + b _ { \star } ^ { \mathsf { P } } ) ( \rho _ { T } + b ) } \geq 0 .
$$

Combining this with $\alpha > \alpha _ { \star } ^ { \sf P } > 0$

$$
\frac { b } { ( \rho _ { T } + b ) \alpha } \leq \frac { b _ { \star } ^ { \mathsf { P } } } { ( \rho _ { T } + b _ { \star } ^ { \mathsf { P } } ) \alpha } \leq \frac { b _ { \star } ^ { \mathsf { P } } } { ( \rho _ { T } + b _ { \star } ^ { \mathsf { P } } ) \alpha _ { \star } ^ { \mathsf { P } } } .
$$

For the second term, we compare the entire ratio $( 1 - b / \rho _ { T } ) / \alpha$ . Substituting the AGM expressions for b and $\alpha .$

$$
\begin{array} { r l } & { \frac { 1 - b / \rho _ { T } } { \alpha } = \frac { 1 - \beta _ { 1 } \left( 1 - \eta \lambda _ { \mathrm { W D } } \right) / \rho _ { T } } { \left( 1 - \beta _ { 1 } \right) \eta } } \\ & { \quad \quad \quad = \frac { 1 } { 1 - \beta _ { 1 } } \left[ \frac { 1 - \beta _ { 1 } / \rho _ { T } } { \eta } + \frac { \beta _ { 1 } \lambda _ { \mathrm { W D } } } { \rho _ { T } } \right] } \\ & { \quad \quad \quad \le \frac { 1 } { 1 - \beta _ { 1 } } \left[ \frac { 1 - \beta _ { 1 } / \rho _ { T } } { \eta _ { \star } ^ { \mathrm { p } } } + \frac { \beta _ { 1 } \lambda _ { \mathrm { W D } } } { \rho _ { T } } \right] } \\ & { \quad \quad \quad \quad = \frac { 1 - \beta _ { 1 } \left( 1 - \eta _ { \star } ^ { \mathrm { p } } \lambda _ { \mathrm { W D } } \right) / \rho _ { T } } { \left( 1 - \beta _ { 1 } \right) \eta _ { \star } ^ { \mathrm { p } } } = \frac { 1 - b _ { \star } ^ { \mathrm { p } } / \rho _ { T } } { \alpha _ { \star } ^ { \mathrm { p } } } . } \end{array}
$$

The inequality uses $\eta > \eta _ { \star } ^ { \mathsf { P } }$ and $1 - \beta _ { 1 } / \rho _ { T } > 0$ . Both ratios are positive, so the inequality remains valid after squaring. Substituting these two comparisons gives exactly the definition of $H _ { T }$ from (57):

$$
\frac { 1 } { \eta _ { \star } ^ { \sf P } } - \frac { 1 } { \eta } \leq ( \rho _ { T } - 1 ) \left[ \frac { 2 b _ { \star } ^ { \sf P } } { ( \rho _ { T } + b _ { \star } ^ { \sf P } ) \alpha _ { \star } ^ { \sf P } } + \frac { ( 1 - b _ { \star } ^ { \sf P } / \rho _ { T } ) ^ { 2 } } { 2 \mu _ { \star } ^ { \sf P } ( \alpha _ { \star } ^ { \sf P } ) ^ { 2 } } \right] = H _ { T } .
$$

For $\beta _ { 1 } = 0 , b = 0$ and $\alpha = \eta$ , so the same quotient bound (90) also gives the direct LR bound:

$$
\begin{array} { r l } & { \frac { 1 } { \eta _ { \star } ^ { \mathrm { P } } } - \frac { 1 } { \eta } \leq \frac { \rho _ { T } - 1 } { 2 \mu _ { \star } ^ { \mathrm { P } } \eta ^ { 2 } } , } \\ & { \eta ^ { 2 } - \eta _ { \star } ^ { \mathrm { P } } \eta \leq \frac { \left( \rho _ { T } - 1 \right) \eta _ { \star } ^ { \mathrm { P } } } { 2 \mu _ { \star } ^ { \mathrm { P } } } , } \\ & { \qquad \eta \leq \frac { \eta _ { \star } ^ { \mathrm { P } } } { 2 } \left[ 1 + \sqrt { 1 + \frac { 2 \left( \rho _ { T } - 1 \right) } { \mu _ { \star } ^ { \mathrm { P } } \eta _ { \star } ^ { \mathrm { P } } } } \right] } \\ & { \qquad \leq \eta _ { \star } ^ { \mathrm { P } } + \frac { \rho _ { T } - 1 } { 2 \mu _ { \star } ^ { \mathrm { P } } } . } \end{array}
$$

The second line multiplies the first by $\eta _ { \star } ^ { \mathsf { P } } \eta ^ { 2 } > 0$ . The third takes the positive root of the resulting quadratic, and the last uses $\sqrt { 1 + x } \leq 1 + x / 2$ for $x \geq 0$ . This completes the proof of Proposition F.6. □

For a scalar system without momentum $( \beta _ { 1 } ~ = ~ 0 )$ , with compensated mean a and $\eta > \eta _ { \star } ^ { \mathsf { P } }$ , the quantitative bound (66) is an equality when R is the actual radius $\rho ( T _ { \eta } ^ { \mathsf { P } } ) = \mathbb { E } ( 1 - \eta ( a + \Delta _ { t } ) ) ^ { \dot { 2 } }$ . The inverse LR bound then loses only the gap between this radius and $\rho _ { T }$ in Lemma F.9, and the final relaxation $1 / \eta \leq 1 / \eta _ { \star } ^ { \mathsf { P } }$ . With momentum, the momentum comparisons can introduce a first-order loss when the noise is small.

## F.4.3 CHOOSING THE CUTOFF

At $\eta _ { \star } ^ { \mathsf { P } } , \rho ( \tau _ { \eta _ { \star } ^ { \mathsf { P } } } ^ { \mathsf { P } } ) = 1$ by Corollary F.4, but this does not ensure bounded affine moments or acceptance at a finite cutoff. For example, without noise, forcing or weight decay, the top eigendirection at $\alpha \lambda _ { \mathrm { m a x } } ( A _ { \mathsf { P } } ) = 2 ( 1 + \beta _ { 1 } )$ obeys $x _ { t + 1 } = - ( 1 + \beta _ { 1 } ) x _ { t } - \beta _ { 1 } x _ { t - 1 }$ . From a unit displacement with zero momentum $( x _ { - 1 } = x _ { 0 } = 1 )$ , its solution is

$$
\begin{array} { l } { { \displaystyle x _ { t } = \frac { 1 + \beta _ { 1 } } { 1 - \beta _ { 1 } } ( - 1 ) ^ { t } - \frac { 2 \beta _ { 1 } } { 1 - \beta _ { 1 } } ( - \beta _ { 1 } ) ^ { t } , } } \\ { { \displaystyle x _ { t } \vert \longrightarrow \frac { 1 + \beta _ { 1 } } { 1 - \beta _ { 1 } } , } } \end{array}
$$

$\mathrm { A t } \beta _ { 1 } = 0 . 9$ , the limiting amplitude is 19, so any smaller cutoff M rejects $\eta _ { \star } ^ { \mathsf { P } }$ for sufficiently large T despite bounded displacement. The affine detector may also fail its cutoff because of batch-gradient forcing, weight-decay drift or checkpoint momentum.

Acceptance at $\eta _ { \star } ^ { \mathsf { P } }$ gives $\bar { \eta } _ { \star } ( M , T ) \geq \eta _ { \star } ^ { \mathsf { P } }$ , which supplies the upper bound in Theorem C.16. The lower bound only needs a nonempty accepted set and $\eta _ { \star } ^ { \mathsf { P } } \in \mathcal { I }$ , which ensures nonnegative effective momentum at $\eta _ { \star } ^ { \mathsf { \bar { P } } }$ . Proposition F.11 gives a sufficient cutoff that grows polynomially with T.

We now fix $\eta = \eta _ { \star } ^ { \mathsf { P } }$ and return to the original affine simulation, with its prescribed checkpoint momentum and forcing. The state below is the state of this single run, not a difference of coupled runs or a rescaled trajectory. Use the original-coordinate state from the proof of Lemma F.9, $z _ { t } = ( \epsilon _ { t } , \eta _ { \star } ^ { \sf P } P ^ { - 1 } m _ { t } ) ^ { \top }$ . With $\alpha = \eta _ { \star } ^ { \mathsf { P } } ( 1 - \bar { \beta _ { 1 } } )$ , it satisfies

$$
\begin{array} { r l } & { z _ { t + 1 } = \widehat { F } _ { t } z _ { t } + { f } _ { t } , } \\ & { ~ f _ { t } : = \left( { - \alpha P ^ { - 1 } g _ { \mathcal { B } _ { t } } - \eta _ { \star } ^ { \mathsf { P } } \lambda _ { \mathrm { W D } } \pmb w _ { t _ { 0 } } } \right) , } \\ & { ~ { \alpha P ^ { - 1 } } g _ { \mathcal { B } _ { t } } } \end{array}\tag{92}
$$

with the homogeneous transition matrix $\widehat { F } _ { t }$ from (88), evaluated at $\eta = \eta _ { \star } ^ { \mathsf { P } }$ . Thus all quantities in this cutoff calculation use the critical LR. Write $\widehat { F } _ { t : s } : = \widehat { F } _ { t - 1 } \cdot \cdot \cdot \widehat { F } _ { s }$ for the propagator from time s to time $t ,$ with $\widehat { F } _ { t : t } = I .$ , and define

$$
\begin{array} { r l } & { K _ { T } : = \underset { 0 \leq t \leq T } { \operatorname* { m a x } } \left. \mathbb { E } [ \widehat { F } _ { t : 0 } ^ { \top } \widehat { F } _ { t : 0 } ] \right. , } \\ & { ~ R _ { 0 } : = \big ( 1 + \left. \eta _ { \star } ^ { \mathsf { P } } P ^ { - 1 } m _ { t _ { 0 } } \right. ^ { 2 } \big ) ^ { 1 / 2 } , } \\ & { ~ \sigma _ { f } : = \big ( \mathbb { E } \left. f _ { t } \right. ^ { 2 } \big ) ^ { 1 / 2 } . } \end{array}\tag{93}
$$

$K _ { T } ~ \ge ~ 1$ is the worst-case second-moment growth of the homogeneous run over the horizon, $\mathbb { E } \left\| \widehat { F } _ { t : 0 } z \right\| ^ { 2 } \leq K _ { T } \left\| z \right\| ^ { 2 }$ for $t \leq T$ . The quantity $R _ { 0 }$ is the norm of the initial state for a unit displacement with the checkpoint momentum. The size of the forcing is

$$
\begin{array} { r } { \sigma _ { f } ^ { 2 } = \mathbb { E } \left\| \alpha P ^ { - 1 } g _ { \mathcal { B } _ { t } } + \eta _ { \star } ^ { \mathrm { P } } \lambda _ { \mathrm { W D } } \pmb { w } _ { t _ { 0 } } \right\| ^ { 2 } + \alpha ^ { 2 } \mathbb { E } \left\| P ^ { - 1 } \pmb { g } _ { \mathcal { B } _ { t } } \right\| ^ { 2 } . } \end{array}
$$

Proposition F.11 (An explicit cutoff for acceptance at the compensated critical LR). Under Assumption C.14, suppose $\eta _ { \star } ^ { \mathsf { P } } \in \mathsf { T } . \ H$

$$
M \geq \sqrt { K _ { T } } \left( R _ { 0 } + T \sigma _ { f } \right) ,\tag{94}
$$

then $\eta _ { \star } ^ { \mathsf { P } }$ is accepted by the detector (50), that is, Assumption C.15 holds.

ProofofProposition F.11. We use $K _ { T } .$ , the maximal homogeneous second-moment growth over the horizon, to control the initial state and each forcing term separately. Recall that $\| z _ { 0 } \| = R _ { 0 }$ for every unit initial displacement with the prescribed checkpoint momentum, and $\vec { \mathfrak { L } } \| f _ { j } \| ^ { 2 } = \sigma _ { f } ^ { 2 }$ . Unrolling the affine recursion (92),

$$
z _ { t } = \widehat { F } _ { t : 0 } z _ { 0 } + \sum _ { j = 0 } ^ { t - 1 } \widehat { F } _ { t : j + 1 } f _ { j } .
$$

The propagator $\widehat { F } _ { t : j + 1 }$ depends on the batches $j + 1 , \dots , t - 1$ and $f _ { j }$ on batch $j ,$ , so the two are independent. By the tower property,

$$
\begin{array} { r l } & { \mathbb { E } \left\| \widehat { F } _ { t : j + 1 } f _ { j } \right\| ^ { 2 } = \mathbb { E } \big [ f _ { j } ^ { \top } \mathbb { E } [ \widehat { F } _ { t : j + 1 } ^ { \top } \widehat { F } _ { t : j + 1 } ] f _ { j } \big ] } \\ & { \phantom { { \sum } } \leq \left\| \mathbb { E } [ \widehat { F } _ { t : j + 1 } ^ { \top } \widehat { F } _ { t : j + 1 } ] \right\| \sigma _ { f } ^ { 2 } } \\ & { \phantom { { \sum } } = \left\| \mathbb { E } [ \widehat { F } _ { t - j - 1 : 0 } ^ { \top } \widehat { F } _ { t - j - 1 : 0 } ] \right\| \sigma _ { f } ^ { 2 } } \\ & { \phantom { { \sum } } \leq K _ { T } \sigma _ { f } ^ { 2 } , } \end{array}
$$

where the equality holds because the batches are i.i.d., so that $\widehat { F } _ { t : j + 1 }$ has the law of $\widehat { F } _ { t - j - 1 : 0 }$ Likewise E $\left\| \widehat { \pmb { F } } _ { t : 0 } \pmb { z } _ { 0 } \right\| ^ { 2 } \leq K _ { T } \left\| \pmb { z } _ { 0 } \right\| ^ { 2 } = K _ { T } R _ { 0 } ^ { 2 }$ for a unit displacement. Minkowski’s inequality in $L ^ { 2 }$ gives, for $t \stackrel { \because } { \leq } T$

$$
\begin{array} { l } { \displaystyle \left( \mathbb { E } \left\| z _ { t } \right\| ^ { 2 } \right) ^ { 1 / 2 } \leq \big ( \mathbb { E } \left\| \widehat { \pmb { F } } _ { t : 0 } z _ { 0 } \right\| ^ { 2 } \big ) ^ { 1 / 2 } + \displaystyle \sum _ { j = 0 } ^ { t - 1 } \big ( \mathbb { E } \left\| \widehat { \pmb { F } } _ { t : j + 1 } { \pmb { f } } _ { j } \right\| ^ { 2 } \big ) ^ { 1 / 2 } } \\ { \leq \sqrt { K _ { T } } \left( R _ { 0 } + t \sigma _ { f } \right) } \\ { \leq M . } \end{array}
$$

Since $\lVert \epsilon _ { t } \rVert \leq \lVert z _ { t } \rVert , \mathbb { E } \left. \epsilon _ { t } \right. ^ { 2 } \leq M ^ { 2 }$ for all $0 \leq t \leq T$ and every unit $\epsilon _ { \mathrm { 0 } } .$ , which is the acceptance of $\eta _ { \star } ^ { \mathsf { P } }$ . This completes the proof of Proposition F.11. □

The condition is not a smallness condition on the noise: $K _ { T } , R _ { 0 }$ and $\sigma _ { f }$ enter with their actual values, and $K _ { T }$ is finite for every $T .$ To control its growth, write

$$
\mathcal { U } ( S ) : = \mathbb { E } [ \widehat { F } _ { t } S \widehat { F } _ { t } ^ { \top } ] , \qquad S \in \mathbb { S } _ { 2 d } ,
$$

for the homogeneous moment map at $\eta _ { \star } ^ { \mathsf { P } }$ . The critical-rate result (Corollary F.4) and the change of coordinates above give $\rho ( \mathcal { U } ) = 1$ . The Jordan decomposition therefore bounds the powers of its adjoint, applied to any fixed matrix, by a polynomial. In particular, for some fixed $C ^ { \prime } > 1$ and integer $\begin{array} { r } { 0 \leq k < \dim \mathbb { S } _ { 2 d } . } \end{array}$

$$
\begin{array} { r } { \left\| ( \mathcal { U } ^ { * } ) ^ { t } ( I _ { 2 d } ) \right\| _ { F } \leq C ^ { \prime } ( 1 + t ) ^ { k } . } \end{array}
$$

Iterating the adjoint map uses the same independence across steps as the forward second-moment recursion. Consequently,

$$
\begin{array} { r l } { \mathbb { E } [ \widehat { F } _ { t : 0 } ^ { \top } \widehat { F } _ { t : 0 } ] = ( \mathcal { U } ^ { * } ) ^ { t } ( I _ { 2 d } ) , } & { \qquad \mathrm { i t e r a t e ~ t h e ~ a d j o i n t ~ m o m e n t ~ m a p } } \\ { \big \| \mathbb { E } [ \widehat { F } _ { t : 0 } ^ { \top } \widehat { F } _ { t : 0 } ] \big \| \leq \big \| ( \mathcal { U } ^ { * } ) ^ { t } ( I _ { 2 d } ) \big \| _ { F } } \\ { \leq C ^ { \prime } ( 1 + t ) ^ { k } , } & { \qquad \mathrm { J o r d a n ~ p o w e r ~ b o u n d } } \\ { K _ { T } \leq C ^ { \prime } ( 1 + T ) ^ { k } . } & { \qquad \mathrm { m a x i m i z e ~ o v e r ~ } 0 \leq t \leq T } \end{array}
$$

The resulting cutoff satisfies

$$
\begin{array} { r l } { M _ { T } : = \sqrt { C ^ { \prime } ( 1 + T ) ^ { k } \left( R _ { 0 } + T \sigma _ { f } \right) } } & { } \\ { \geq \sqrt { K _ { T } } \left( R _ { 0 } + T \sigma _ { f } \right) , } & { \qquad \mathrm { t h e ~ s u f f i c i e n t ~ c u t o f f ~ a b o v e } } \\ { \displaystyle \frac { 1 + \log M _ { T } } { T } = O \bigg ( \frac { \log T } { T } \bigg ) \longrightarrow 0 . } & { \qquad \mathrm { f i x e d ~ m o d e l ~ a n d ~ p o l y n o m i a l ~ c u t o f f } } \end{array}
$$

Thus $M _ { T } > 1$ is admissible at every horizon and yields $H _ { T } = O ( \log T / T )$ for a fixed model. The factor $T \sigma _ { f }$ comes from Minkowski’s inequality and does not assert linear growth of the actual run.

## F.5 PROOFS OF THE STABILITY CRITERION

We prove the exact criterion, its quantitative form, and the AGM critical-rate statement from $\mathbf { A p } \mathbf { \cdot }$ pendix F.1. For the exact and quantitative criteria, we use the generic recurrence

$$
\pmb { x } _ { t + 1 } = \big [ ( 1 + b ) \pmb { I } - \alpha ( \pmb { C } + \Delta _ { t } ) \big ] \pmb { x } _ { t } - b \pmb { x } _ { t - 1 } , \qquad \pmb { C } \succ 0 , \quad - 1 < b < 1 , \quad \alpha > 0 .
$$

The AGM proof also treats the remaining LRs, for which $b \leq - 1$ . Recall that $\mathscr { T } ( S ) = \mathbb { E } [ F _ { t } S F _ { t } ^ { \top } ]$ is its second-moment map, with the history transition $\mathbf { } F _ { t }$ from (64). The quotient $q c , b$ and its maximum $r _ { C } ( \boldsymbol { b } )$ are recalled in Appendix F.1.

## F.5.1 PROOF OF THE EXACT CRITERION

A general random linear recurrence Consider the random linear recurrence

$$
\pmb { y } _ { t + 1 } = \pmb { E } _ { t } \pmb { y } _ { t } , \qquad \pmb { y } _ { t } \in \mathbb { R } ^ { m } ,\tag{95}
$$

where the matrices $\scriptstyle { E _ { t } }$ are i.i.d. with finite second moments and the initial state $\scriptstyle { \mathbf { 3 0 } }$ is deterministic. We seek a criterion for $\mathbb { E }  \boldsymbol { y } _ { t }  ^ { 2 }  0$ for every initial state. Separate the mean matrix from its centered perturbation,

$$
\begin{array} { r } { \begin{array} { c r c l } { F : = \mathbb { E } E _ { t } , } & { N _ { t } : = E _ { t } - F , } \\ { \ T _ { 0 } ( S ) : = F S F ^ { \top } , } & { N ( S ) : = \mathbb { E } [ N _ { t } S N _ { t } ^ { \top } ] . } \end{array} } \end{array}\tag{96}
$$

Both $\mathcal { T } _ { 0 }$ and $\mathcal { N }$ preserve positive semidefinite matrices. The second moment $\Sigma _ { t } : = \mathbb { E } [ { \pmb { y } } _ { t } { \pmb { y } } _ { t } ^ { \top } ]$ evolves as

$$
\begin{array} { r l r } & { \pmb { \Sigma } _ { t + 1 } = \mathbb { E } [ ( \pmb { F } + \pmb { N } _ { t } ) \pmb { \Sigma } _ { t } ( \pmb { F } + \pmb { N } _ { t } ) ^ { \top } ] } & \\ & { \quad \quad = \mathcal { T } _ { 0 } ( \pmb { \Sigma } _ { t } ) + \mathcal { N } ( \pmb { \Sigma } _ { t } ) . } & { \mathbb { E } \pmb { N } _ { t } = 0 } \end{array}\tag{97}
$$

The first equality uses the independence of ${ \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \Xi } _ { \mathbf { } } \mathbf { \Lambda } _ { \mathbf { } } \mathbf { \Lambda } _ { \mathbf { } } \textbf { } _ { \mathbf { } } \textbf { } \textbf { } _ { \mathrm { } }$ and $\scriptstyle { E _ { t } }$ . Write $\mathscr { T } : = \mathscr { T } _ { 0 } + \mathscr { N }$ for this second-moment map.

Suppose first that the mean dynamics are stable, $\rho ( F ) < 1$ . A perturbation with second moment $Q$ injected once produces the later second moments ${ \dot { \mathbf { F } } } ^ { j } { \dot { Q } } ( { \mathbf { \boldsymbol { F } } } ^ { \top } ) ^ { j }$ . Their sum is

$$
{ \mathcal { K } } ( Q ) : = \sum _ { j \geq 0 } F ^ { j } Q ( F ^ { \top } ) ^ { j } , \qquad { \mathcal { K } } = ( { \mathcal { T } } - { \mathcal { T } } _ { 0 } ) ^ { - 1 } .\tag{98}
$$

Here $\mathcal { T }$ is the identity operator on symmetric matrices. The series converges and preserves PSD matrices. Thus $\mathcal { K } ( Q )$ is the cumulative response of the mean dynamics to one injection, and $\kappa { \mathcal N }$ first forms the second moment of a noise term and then accumulates its response. The next proposition characterizes stability through this composition.

Proposition F.12 (Feedback criterion). The recurrence (95) is strictly mean-square stable if and only $i f \rho ( \bar { F } ) < 1 a n d \rho ( \mathcal { K N } ) < 1$

ProofofProposition F.12. We transfer a Lyapunov inequality between $\tau$ and KN. The inverse K accounts for the mean dynamics in this transfer.

## Step 1. Separate the mean dynamics.

Strict mean-square stability implies $\rho ( F ) < 1$ by the second-moment characterization (Lemma F.14). Assume $\rho ( F ) < 1$ from now on, so K exists. Then

$$
\begin{array} { c c } { { ( \mathcal { Z } - \mathcal { T } _ { 0 } ) ( \mathcal { T } - K \mathcal { N } ) = \mathcal { Z } - \mathcal { T } _ { 0 } - \mathcal { N } ~ } } & { { ~ ( \mathcal { Z } - \mathcal { T } _ { 0 } ) \mathcal { K } = \mathcal { Z } } } \\ { { { } } } & { { { } } } \\ { { = \mathcal { Z } - \mathcal { T } . ~ } } & { { ~ \mathcal { T } = \mathcal { T } _ { 0 } + \mathcal { N } } } \end{array}
$$

## Step 2. Necessity.

$\mathrm { I f } \rho ( \mathcal T ) < 1$ , its Neumann series defines

$$
W : = ( \mathcal { T } - \mathcal { T } ) ^ { - 1 } ( I _ { m } ) = \sum _ { j \geq 0 } \mathcal { T } ^ { j } ( I _ { m } ) \succeq I _ { m } .
$$

The factorization in Step 1 gives

$$
\begin{array} { r l } & { W - K \mathcal { N } ( W ) = \mathcal { K } \big ( W - \mathcal { T } ( W ) \big ) } \\ & { \qquad = \mathcal { K } ( I _ { m } ) \succeq I _ { m } . } \end{array}
$$

The first equality multiplies the factorization by $\kappa .$ . The second line uses the Neumann series for W and K. The Lyapunov criterion for PSD-preserving maps (Lemma F.15) therefore gives $\rho ( \mathcal { K N } ) < 1$

## Step 3. Sufficiency.

Conversely, if $\rho ( \mathcal { K N } ) < 1$ , put

$$
W : = ( \mathcal { T } - K \mathcal { N } ) ^ { - 1 } { \boldsymbol { K } } ( { \boldsymbol { I _ { m } } } ) = \sum _ { j \geq 0 } ( K \mathcal { N } ) ^ { j } { \boldsymbol { K } } ( { \boldsymbol { I _ { m } } } ) \succeq I _ { m } .
$$

Now the same factorization gives

$$
\begin{array} { r l r l } { W - { \mathcal T } ( W ) = ( { \mathcal T } - { \mathcal T } _ { 0 } ) \bigl ( W - \mathcal { K } \mathcal { N } ( W ) \bigr ) } & { \qquad } & & { \mathrm { S t e p ~ 1 } } \\ { = ( { \mathcal T } - { \mathcal T } _ { 0 } ) \mathcal { K } ( I _ { m } ) = I _ { m } . } & { \qquad } & & { \mathrm { d e f i n i t i o n ~ o f ~ } W } \end{array}
$$

The Lyapunov criterion gives $\rho ( \mathcal T ) < 1$ , which is strict mean-square stability by Lemma F.14. This completes the proof of Proposition F.12. □

The momentum recurrence We now apply the general criterion to the recurrence from (63). Its mean curvature is $C \succ 0$ , its parameters satisfy $\prime - 1 < b < 1$ and $\alpha > 0 ,$ , and its symmetric noise $\Delta _ { t }$ is i.i.d. with zero mean and finite second moments. The initial pair $( { \pmb x } _ { 0 } , { \pmb x } _ { - 1 } )$ is deterministic. Recall the recurrence and its history state,

$$
\begin{array} { r } { \pmb { x } _ { t + 1 } = \big [ ( 1 + b ) \pmb { I } - \alpha ( \pmb { C } + \Delta _ { t } ) \big ] \pmb { x } _ { t } - b \pmb { x } _ { t - 1 } , \qquad \pmb { y } _ { t } : = ( \pmb { x } _ { t } , \pmb { x } _ { t - 1 } ) ^ { \top } . } \end{array}
$$

Write $E : = ( 1 + b ) { \cal I } - \alpha C$ . The random and mean history matrices, and the injection into the position block, are

$$
\begin{array} { r } { F _ { t } : = \left( \begin{array} { c c } { E - \alpha \mathbf { \Delta } _ { t } } & { - b I } \\ { I } & { 0 } \end{array} \right) , \qquad F : = \mathbb { E } F _ { t } = \left( \begin{array} { c c } { E } & { - b I } \\ { I } & { 0 } \end{array} \right) , \qquad J : = \left( \begin{array} { c } { I } \\ { 0 } \end{array} \right) . } \end{array}\tag{99}
$$

This is the general recurrence with state dimension $m = 2 n$ and random matrix $\mathbf { } E _ { t } = \mathbf { } F _ { t } $ . We use $\mathcal { T } _ { 0 }$ $\mathcal { N }$ and K for this history system from now on. The matrix J embeds a position vector as $( { \pmb x } , 0 ) ^ { \top }$ while $J ^ { \top } S J$ extracts the position block of a history second moment. Let $a _ { i } > 0$ be the eigenvalues of $C .$ Recall that $L _ { C } = \bar { \lambda _ { \operatorname* { m a x } } } ( C )$ . The characteristic polynomials of $\pmb { F }$ along these eigendirections are $x ^ { 2 } - ( 1 + b - \alpha a _ { i } ) x + b .$ The Jury stability test (Lemma F.17) requires

$$
| b | < 1 , \qquad \alpha a _ { i } > 0 , \qquad 2 ( 1 + b ) - \alpha a _ { i } > 0 .
$$

The first two conditions already hold. Thus the mean dynamics are stable if and only if

$$
\alpha L _ { C } < 2 ( 1 + b ) .\tag{100}
$$

The centered perturbation is $N _ { t } = F _ { t } - F = - \alpha J \Delta _ { t } J ^ { \top }$ , so it acts only on the position. Define its second-moment map on position matrices by

$$
\mathcal { V } ( X ) : = \mathbb { E } [ \Delta _ { t } X \Delta _ { t } ] , \qquad X \in \mathbb { S } _ { n } .\tag{101}
$$

Then the full noise map from (96) is

$$
\begin{array} { r } { \mathcal { N } ( S ) = \alpha ^ { 2 } J \mathcal { V } ( J ^ { \top } S J ) J ^ { \top } . } \end{array}\tag{102}
$$

Under the mean stability condition $\alpha L _ { C } < 2 ( 1 + b )$ , K propagates an injection through the mean history matrix $\pmb { F }$ above, as in (98). Since $\mathcal { N }$ uses only the position block, reducing $\bar { \kappa } \mathcal { N }$ requires the position response to a position injection. The following lemma computes that response. Recall $c _ { b } : = ( 1 - b ) / ( 1 + b )$ and $\dot { k } _ { b } : = b / ( \dot { 1 } - b ^ { 2 } )$

Lemma F.13 (Position block of the cumulative response). Assume (100), and define the operator on $\mathbb { S } _ { n }$

$$
\mathcal { D } ( \pmb { X } ) : = \frac { 1 - b } { \alpha } ( \pmb { C } \pmb { X } + \pmb { X } \pmb { C } ) - c _ { b } \pmb { C } \pmb { X } \pmb { C } + k _ { b } [ \pmb { C } , [ \pmb { C } , \pmb { X } ] ] .\tag{103}
$$

Then $\mathcal { D }$ is self-adjoint and positive definite. For every $Q \in \mathbb { S } _ { n }$ ，

$$
J ^ { \top } { \boldsymbol { \mathcal { K } } } ( J Q J ^ { \top } ) J = \alpha ^ { - 2 } { \mathcal { D } } ^ { - 1 } ( Q ) .\tag{104}
$$

Consequently, $\mathcal { D } ^ { - 1 }$ and $\mathcal { D } ^ { - 1 } \mathcal { V }$ preserve PSD matrices, and $\kappa \mathcal { N }$ and $\mathcal { D } ^ { - 1 } \mathcal { V }$ have the same nonzero eigenvalues. In particular, their spectral radii are equal.

Proof of Lemma F.13. We solve the Lyapunov equation for a single position injection by eliminating its delayed and cross blocks. This identifies $\mathcal { D } ,$ while the response series proves that its inverse preserves PSD matrices.

## Step 1. Eliminate the cross and delayed blocks.

Fix $Q \in \mathbb { S } _ { n }$ . Since (100) gives $\rho ( F ) < 1$ , the cumulative response is

$$
\Sigma _ { Q } : = K ( J Q J ^ { \top } ) = \sum _ { j \ge 0 } F ^ { j } J Q J ^ { \top } ( F ^ { \top } ) ^ { j } = { \binom { X } { Y ^ { \top } } } \quad Y \bigg ) .\tag{105}
$$

Here X is the cumulative position response, $W$ the cumulative delayed-position response, and $\mathbf { Y }$ their cross block. These blocks depend linearly on the injection $\mathbf { \bar { Q } }$ . Its Lyapunov equation $\Sigma _ { Q } = J Q J ^ { \top } + F \Sigma _ { Q } F ^ { \top }$ reads

$$
\begin{array} { r l } & { X = Q + E X E - b ( E Y + Y ^ { \top } E ) + b ^ { 2 } W , } \\ & { Y = E X - b Y ^ { \top } , \qquad W = X . } \end{array}
$$

The cross block Y need not be symmetric. Its equation and its transpose give

$$
\pmb { Y } + b \pmb { Y } ^ { \top } = \pmb { E } \pmb { X } , \qquad \pmb { Y } ^ { \top } + b \pmb { Y } = \pmb { X } \pmb { E } ,
$$

$$
( 1 - b ^ { 2 } ) { \cal Y } = E { \cal X } - b { \cal X } E ,
$$

$$
E Y + Y ^ { \top } E = \frac { E ^ { 2 } X + X E ^ { 2 } - 2 b E X E } { 1 - b ^ { 2 } } .
$$

The second line subtracts b times the transposed equation from the first. The last line substitutes the resulting expressions for $\mathbf { Y }$ and $\mathbf { \mathbf { \mathbf { \mathbf { Y } } } ^ { \top } }$ . Substituting these identities and $W = X$ into the first block equation,

$$
\begin{array} { l } { \displaystyle { Q = ( 1 - b ^ { 2 } ) X - E X E + b ( E Y + { \mathbf Y } ^ { \top } E ) } } \\ { \displaystyle { \quad = ( 1 - b ^ { 2 } ) X - \frac { 1 + b ^ { 2 } } { 1 - b ^ { 2 } } E X E + \frac { b } { 1 - b ^ { 2 } } ( E ^ { 2 } X + X E ^ { 2 } ) . } } \end{array}
$$

## Step 2. Identify the position operator.

Expand ${ \pmb E } = ( 1 + b ) { \pmb I } - \alpha { \pmb C }$ in the last display. The coefficients of X and $C X + X C$ simplify, respectively, to

$$
\begin{array} { c } { { ( 1 - b ^ { 2 } ) - \displaystyle \frac { ( 1 + b ) ^ { 2 } ( 1 + b ^ { 2 } - 2 b ) } { 1 - b ^ { 2 } } = ( 1 - b ^ { 2 } ) - ( 1 - b ^ { 2 } ) = 0 , } } \\ { { \displaystyle \frac { \alpha ( 1 + b ) ( 1 + b ^ { 2 } - 2 b ) } { 1 - b ^ { 2 } } = \alpha ( 1 - b ) . } } \end{array}
$$

The quadratic terms are

$$
\begin{array} { r l } & { - \frac { ( 1 + b ^ { 2 } ) \alpha ^ { 2 } } { 1 - b ^ { 2 } } C X C + \frac { b \alpha ^ { 2 } } { 1 - b ^ { 2 } } ( C ^ { 2 } X + X C ^ { 2 } ) } \\ & { \quad = - \frac { ( 1 - b ) ^ { 2 } \alpha ^ { 2 } } { 1 - b ^ { 2 } } C X C + \frac { b \alpha ^ { 2 } } { 1 - b ^ { 2 } } [ C , [ C , X ] ] } \\ & { \quad = - \alpha ^ { 2 } c _ { b } C X C + \alpha ^ { 2 } k _ { b } [ C , [ C , X ] ] . } \end{array}
$$

The second line uses $C ^ { 2 } X + X C ^ { 2 } = [ C , [ C , X ] ] + 2 C X C$ . Combining the linear and quadratic terms gives

$$
Q = \alpha ^ { 2 } { \mathcal { D } } ( X ) .\tag{106}
$$

## Step 3. Prove positive definiteness.

Each term in (103) is self-adjoint under the Frobenius inner product. For every symmetric X, the commutator identity gives

$$
\begin{array} { r l } & { c _ { b } \operatorname { t r } ( \boldsymbol { X } C \boldsymbol { X } C ) - k _ { b } \left\| \left[ \boldsymbol { C } , \boldsymbol { X } \right] \right\| _ { F } ^ { 2 } } \\ & { \quad = c _ { b } \operatorname { t r } ( C ^ { 2 } X ^ { 2 } ) - \left( \displaystyle \frac { c _ { b } } { 2 } + k _ { b } \right) \left\| \left[ \boldsymbol { C } , \boldsymbol { X } \right] \right\| _ { F } ^ { 2 } } \\ & { \quad \le c _ { b } \operatorname { t r } ( C ^ { 2 } X ^ { 2 } ) \le c _ { b } L _ { C } \operatorname { t r } ( C X ^ { 2 } ) . } \end{array}\tag{107}
$$

Here $c _ { b } / 2 + k _ { b } = ( 1 + b ^ { 2 } ) / ( 2 ( 1 - b ^ { 2 } ) ) > 0$ for every $- 1 < b < 1$ , and $C ^ { 2 } \preceq L _ { C } C$ . Therefore, for $X \neq 0 ,$

$$
\langle X , { \mathcal { D } } X \rangle \geq \left( { \frac { 2 ( 1 - b ) } { \alpha } } - c _ { b } L c \right) \operatorname { t r } ( C X ^ { 2 } )
$$

The first inequality follows from (103) and (107). The second uses α $L _ { C } < 2 ( 1 + b )$ and $C \succ 0$ . Thus $\mathcal { D } \succ 0$ . Solving ${ \dot { Q } } = \alpha ^ { 2 } { \mathcal { D } } ( X )$ from (106) gives $X = \alpha ^ { - 2 } { \mathcal { D } } ^ { - 1 } ( Q )$ , the position response in (104). If $Q \succeq 0$ , every summand of the response series (105) is PSD, so $X \succeq 0$ . Hence $\bar { \mathcal { D } } ^ { - 1 }$ preserves PSD matrices. This property follows from the response series, not from positive definiteness of $\mathcal { D }$ as an operator. The composition $\mathcal { D } ^ { - 1 } \mathcal { V }$ also preserves PSD matrices.

## Step 4. Reduce the spectrum to the position block.

The full noise map injects $\alpha ^ { 2 } \mathcal { V } ( J ^ { \top } S J )$ into the position, and the response of a position injection is $\alpha ^ { - 2 } \mathcal { D } ^ { - 1 }$ . Combining these two identities, (102) and (104), gives

$$
\begin{array} { c } { { \displaystyle { \mathcal K N } ( { \pmb S } ) = \alpha ^ { 2 } { \mathcal K } \big ( { \pmb J } { \mathcal V } ( { \pmb J } ^ { \top } { \pmb S } { \pmb J } ) { \pmb J } ^ { \top } \big ) , } } \\ { { { \pmb J } ^ { \top } { \mathcal K N } ( { \pmb S } ) { \pmb J } = \mathcal { D } ^ { - 1 } \mathcal { V } ( { \pmb J } ^ { \top } { \pmb S } { \pmb J } ) . } } \end{array}\tag{108}
$$

I $\mathrm { { f } } \ K \mathcal { N } ( S ) = \lambda S$ with $\lambda \neq 0 ,$ , compress to $X : = J ^ { \top } S J$ . Then

$$
{ \mathcal { D } } ^ { - 1 } { \mathcal { V } } ( X ) = \lambda X .
$$

For $S \neq 0$ this X is nonzero, since otherwise the first line of (108) would give $\lambda S = 0$ . Conversely, suppose ${ \mathcal { D } } ^ { - 1 } { \mathcal { V } } ( X ) = \lambda X$ with $\lambda \neq 0$ and $X \neq 0$ . Set $S : = \overline { { \alpha ^ { 2 } } } \mathcal { K } ( J \mathcal { V } ( X ) J ^ { \top } )$ ). Then

$$
\begin{array} { r l } & { \pmb { J } ^ { \top } \pmb { S } \pmb { J } = \pmb { { \mathcal { D } } } ^ { - 1 } \pmb { \mathcal { V } } ( \pmb { X } ) = \lambda \pmb { X } \neq 0 , } \\ & { \kappa \pmb { \mathcal { N } } ( \pmb { S } ) = \alpha ^ { 2 } \mathcal { K } \big ( \pmb { J } \pmb { \mathcal { V } } ( \lambda \pmb { X } ) \pmb { J } ^ { \top } \big ) = \lambda \pmb { S } . } \end{array}
$$

The first line uses (104). The second uses (108) and linearity. The nonzero eigenvalues coincide, so the spectral radii are equal. This completes the proof of Lemma F.13. □

We can now express stability through the quotient from (51). For the same mean $^ { C , }$ , noise $\Delta _ { t }$ and fixed momentum $b ,$ recall

$$
q _ { C , b } ( X ) = \frac { \mathbb { E } \operatorname { t r } ( X \Delta _ { t } X \Delta _ { t } ) + c _ { b } \operatorname { t r } ( X C X C ) - k _ { b } \left\| [ C , X ] \right\| _ { F } ^ { 2 } } { 2 \operatorname { t r } ( C X ^ { 2 } ) } , \qquad 0 \neq X \succeq 0 .
$$

Here $c _ { b } = ( 1 - b ) / ( 1 + b )$ and $k _ { b } = b / ( 1 - b ^ { 2 } )$ , and $r _ { C } ( \boldsymbol { b } )$ is the maximum over $X \succeq 0$ with $\| \boldsymbol { X } \| _ { F } = 1$ . The first numerator term is $\langle \dot { X } , \dot { \nu } X \rangle$ , the noise contribution. The other two come from the mean dynamics through $\mathcal { D } .$ . Thus

$$
\langle X , ( { \mathcal { D } } - \mathcal { V } ) X \rangle = 2 \operatorname { t r } ( C X ^ { 2 } ) \left( { \frac { 1 - b } { \alpha } } - q _ { C , b } ( X ) \right) .\tag{109}
$$

This identity holds for every nonzero PSD $\boldsymbol { X }$ . It follows from $\langle X , \mathcal { V } X \rangle = \mathbb { E } \operatorname { t r } ( X \Delta _ { t } X \Delta _ { t } )$ and $\langle X , [ C , [ C , X ] ] \rangle = \left\| [ C , X ] \right\| _ { F } ^ { 2 } .$

Proof of Theorem F.1. The general criterion reduces stability to $\mathcal { D } ^ { - 1 } \mathcal { V }$ . A PSD eigenmatrix at its spectral radius then reduces the test to the quotient in (109).

## Step 1. Reduce to the position operator.

For a unit top eigenvector u of $C ,$ test the quotient with ${ \mathbf { } } { \mathbf { } } { \mathbf { } } u { \mathbf { } } u ^ { \top }$ :

$$
r _ { C } ( b ) \geq q _ { C , b } ( { \pmb u } { \pmb u } ^ { \top } ) = \frac { c _ { b } L _ { C } } { 2 } + \frac { \mathbb { E } ( { \pmb u } ^ { \top } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } _ { t } { \pmb u } ) ^ { 2 } } { 2 L _ { C } } \geq \frac { c _ { b } L _ { C } } { 2 } > 0 .\tag{110}
$$

Thus (65) implies $\alpha L _ { C } < 2 ( 1 + b )$ . Strict mean-square stability also implies this mean stability condition. Under it, Proposition F.12 and Lemma F.13 give

$$
\mathrm { s t r i c t ~ m e a n - s q u a r e ~ s t a b i l i t y } \quad \Longleftrightarrow \quad \rho ( \mathcal { D } ^ { - 1 } \mathcal { V } ) < 1 .\tag{111}
$$

## Step 2. Test the quadratic form on PSD matrices.

Write $\rho ~ : = ~ \rho ( \mathcal { D } ^ { - 1 } \mathcal { V } )$ . If $\rho \ > \ 0 .$ , the Perron–Frobenius theorem for PSD-preserving maps (Lemma F.16) gives $W \succeq 0 , W \neq 0$ , such that

$$
{ \mathcal { V } } ( W ) = \rho { \mathcal { D } } ( W ) , \qquad \rho = { \frac { \langle W , \mathcal { V } W \rangle } { \langle W , { \mathcal { D } } W \rangle } } .
$$

Consequently, positivity of $\langle X , ( \mathcal { D } - \mathcal { V } ) X \rangle$ for every nonzero PSD X implies $\rho < 1$ . This implication also holds when $\rho = 0$

Conversely, suppose $\rho < 1$ . We proved in Lemma F.13 that $\mathcal { D }$ is a positive definite self-adjoint operator on $\mathbb { S } _ { n } ,$ so it has a positive square root $\mathcal { D } ^ { 1 / 2 }$ . The operator $\nu$ is self-adjoint because the noise matrices are symmetric. The similarity

$$
{ \mathcal { D } } ^ { 1 / 2 } ( { \mathcal { D } } ^ { - 1 } \mathcal { V } ) { \mathcal { D } } ^ { - 1 / 2 } = { \mathcal { D } } ^ { - 1 / 2 } { \mathcal { V } } { \mathcal { D } } ^ { - 1 / 2 }
$$

therefore transforms $\mathcal { D } ^ { - 1 } \mathcal { V }$ into a self-adjoint operator with the same spectral radius $\rho .$ Its largest eigenvalue is at most $\rho .$ Applying its Rayleigh bound to $\mathcal { D } ^ { 1 / 2 } \boldsymbol { X }$ , for every symmetric $X \neq 0$

$$
\begin{array} { r l r } & { \langle { \pmb X } , \mathcal { V } { \pmb X } \rangle = \left. \mathcal { D } ^ { 1 / 2 } { \pmb X } , ( \mathcal { D } ^ { - 1 / 2 } \mathcal { V } \mathcal { D } ^ { - 1 / 2 } ) \mathcal { D } ^ { 1 / 2 } { \pmb X } \right. } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \mathrm { R a y l e i g h b o u n d } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \beta < 1 \mathrm { a n d } \mathcal { D } \succ 0 } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \end{array}
$$

This proves the equivalence between stability and positivity of the quadratic form on nonzero PSD matrices.

## Step 3. Take the maximum of the quotient.

The identity (109) expresses this quadratic form as the positive factor $2 \operatorname { t r } ( C X ^ { 2 } )$ times $( 1 - b ) / \alpha -$ $q _ { C , b } ( \boldsymbol { X } )$ . Its positivity is therefore equivalent to

$$
q _ { C , b } ( { \boldsymbol { X } } ) < { \frac { 1 - b } { \alpha } } \qquad { \mathrm { f o r ~ e v e r y ~ } } { \boldsymbol { X } } \succeq 0 { \mathrm { ~ w i t h ~ } } \| { \boldsymbol { X } } \| _ { F } = 1 .
$$

The maximum is attained on this compact set, since its denominator $2 \operatorname { t r } ( C X ^ { 2 } )$ is at least $2 \lambda _ { \operatorname* { m i n } } ( C ) > 0$ . The condition is therefore (65). For fixed $b ,$ its right side is independent of $\alpha ,$ giving the stated interval. This completes the proof of Theorem F.1. □

## F.5.2 PROOF OF THE QUANTITATIVE BOUND

The proof rescales the recurrence to make it strictly stable and then applies the exact criterion. We first prove the shift identity (Lemma F.3), which compares the rescaled mean with the original one.

Proof of Lemma $F . 3 .$ . The noise term is unchanged because the two systems use the same $\Delta _ { t } .$ We compute the other terms separately. First,

$$
{ \begin{array} { l } { [ } { C + h I , X ] = ( C + h I ) X - X ( C + h I ) } \\ { \qquad = C X + h X - X C - h X } \\ { \qquad = [ C , X ] . } \end{array} }
$$

the scalar-shift terms cancel

Next, expanding both factors of the shifted mean gives

$$
\begin{array} { r l } & { \mathrm { t r } \big ( X ( C + h I ) X ( C + h I ) \big ) } \\ & { \quad = \mathrm { t r } ( X C X C ) + h \mathrm { t r } ( X C X ) + h \mathrm { t r } ( X ^ { 2 } C ) + h ^ { 2 } \mathrm { t r } ( X ^ { 2 } ) } \\ & { \quad = \mathrm { t r } ( X C X C ) + 2 h \mathrm { t r } ( C X ^ { 2 } ) + h ^ { 2 } \mathrm { t r } ( X ^ { 2 } ) } \\ & { \quad = \mathrm { t r } ( X C X C ) + 2 h \mu + h ^ { 2 } . } \end{array}
$$

The second equality uses cyclicity of the trace. The last substitutes $\mathrm { t r } ( C X ^ { 2 } ) = \mu \mathrm { a n d } \mathrm { t r } ( X ^ { 2 } ) = 1$ The denominator changes according to

$$
\begin{array} { c } { \operatorname { t r } \big ( ( C + h I ) X ^ { 2 } \big ) = \operatorname { t r } ( C X ^ { 2 } ) + h \operatorname { t r } ( X ^ { 2 } ) } \\ { = \mu + h . } \end{array}
$$

The original quotient satisfies

$$
2 \mu q _ { C , b } ( \pmb X ) = \mathbb { E } \operatorname { t r } ( \pmb X \Delta _ { t } \pmb X \Delta _ { t } ) + c _ { b } \operatorname { t r } ( \pmb X C \pmb X C ) - k _ { b } \left\| \left[ \pmb C , \pmb X \right] \right\| _ { F } ^ { 2 } .
$$

Thus the shifted numerator adds only $c _ { b } ( 2 h \mu + h ^ { 2 } )$ to this expression. Using the denominator computed above,

$$
\begin{array} { r } { q _ { C + h I , b } ( \boldsymbol { X } ) = \frac { 2 \mu q _ { C , b } ( \boldsymbol { X } ) + c _ { b } ( 2 h \mu + h ^ { 2 } ) } { 2 ( \mu + h ) } } \\ { = \frac { \mu q _ { C , b } ( \boldsymbol { X } ) + c _ { b } h ( \mu + h / 2 ) } { \mu + h } . } \end{array}
$$

The last equality divides numerator and denominator by 2. This is (68) and completes the proof of Lemma F.3. □

ProofofProposition F.2. Fix a PSD test matrix X with $\| \boldsymbol { X } \| _ { F } = 1$ , and recall $\mu = \operatorname { t r } ( C X ^ { 2 } ) > 0$ We divide the trajectories by a geometric sequence to apply the exact criterion, compare the resulting quotient with $q _ { C , b } ( \boldsymbol { X } )$ , and bound the difference by scalar algebra.

## Step 1. Divide out enough growth to obtain stability.

By assumption, $\rho ( { \mathcal { T } } ) \leq R$ . Choose any real number $s > \sqrt { R } \geq 1$ . Dividing the state at time t by $s ^ { t }$ divides its one-step transition matrix by s and its second-moment map by $s ^ { 2 } .$ . Indeed, for the existing history state $\pmb { y } _ { t } = \overline { { ( \pmb { x } _ { t } , \pmb { x } _ { t - 1 } ) } } ^ { \top }$ and transition matrix $\mathbf { } F _ { t }$

$$
\begin{array} { r l } & { s ^ { - ( t + 1 ) } \pmb { y } _ { t + 1 } = \frac { { \pmb { F } } _ { t } } { s } ( s ^ { - t } \pmb { y } _ { t } ) , } \\ & { \mathbb { E } \left[ \frac { { \pmb { F } } _ { t } } { s } s \frac { { \pmb { F } } _ { t } ^ { \top } } { s } \right] = \frac { 1 } { s ^ { 2 } } \mathbb { E } [ \pmb { F } _ { t } \pmb { S } \pmb { F } _ { t } ^ { \top } ] = \frac { 1 } { s ^ { 2 } } \mathcal { T } ( \pmb { S } ) . } \end{array}
$$

The last identity holds for every symmetric matrix $S .$ Hence the new moment map has spectral radius

$$
\rho \bigg ( \frac { 1 } { s ^ { 2 } } \mathcal { T } \bigg ) = \frac { \rho ( \mathcal { T } ) } { s ^ { 2 } } \leq \frac { R } { s ^ { 2 } } < 1 .
$$

By the second-moment characterization (Lemma F.14), the divided state is strictly mean-square stable.

We now express this division in the position recurrence so that the exact criterion (Theorem F.1) applies. Set $\pmb { x } _ { t } ^ { \prime } : = s ^ { - t } \pmb { x } _ { t }$ . Its history state satisfies

$$
\left( \begin{array} { c c } { { { \bf { x } } _ { t } ^ { \prime } } } \\ { { { \bf { x } } _ { t - 1 } ^ { \prime } } } \end{array} \right) = \left( \begin{array} { c c } { { { \cal I } } } & { { 0 } } \\ { { 0 } } & { { s { \cal I } } } \end{array} \right) ( s ^ { - t } { \bf { y } } _ { t } ) .
$$

This fixed invertible change of coordinates preserves strict mean-square stability. Thus the recurrence for $ { \mathbf { { x } } } _ { t } ^ { \prime }$ is strictly stable for every initial pair. To find that recurrence, substitute $\pmb { x } _ { t } = s ^ { t } \pmb { x } _ { t } ^ { \prime }$ and $\mathbf { { x } } _ { t - 1 } \mathbf { { \dot { = } } } s ^ { t - 1 } \mathbf { { x } } _ { t - 1 } ^ { \prime }$ into the original update:

$$
\begin{array} { l } { { \displaystyle x _ { t + 1 } ^ { \prime } = s ^ { - ( t + 1 ) } \left( \left[ ( 1 + b ) I - \alpha ( C + \Delta _ { t } ) \right] x _ { t } - b x _ { t - 1 } \right) } } \\ { { \displaystyle \quad = s ^ { - ( t + 1 ) } \left( \left[ ( 1 + b ) I - \alpha ( C + \Delta _ { t } ) \right] s ^ { t } x _ { t } ^ { \prime } - b s ^ { t - 1 } x _ { t - 1 } ^ { \prime } \right) } } \\ { { \displaystyle \quad = \left[ \frac { 1 + b } { s } I - \frac { \alpha } { s } ( C + \Delta _ { t } ) \right] x _ { t } ^ { \prime } - \frac { b } { s ^ { 2 } } x _ { t - 1 } ^ { \prime } } . } \end{array}
$$

The new momentum coefficient is $b / s ^ { 2 }$ . The generic recurrence uses the identity coefficient $1 + b / s ^ { 2 }$ whereas the last display has $( 1 + b ) / s$ . We absorb their difference into a scalar shift of the mean. Define

$$
\begin{array} { c } { { u : = \displaystyle \frac { s } { \alpha } \left( 1 + \frac { b } { s ^ { 2 } } - \frac { 1 + b } { s } \right) } } \\ { { { } } } \\ { { = \displaystyle \frac { s ^ { 2 } + b - s ( 1 + b ) } { s \alpha } } } \\ { { { } } } \\ { { { } = \displaystyle \frac { ( s - 1 ) ( s - b ) } { s \alpha } \geq 0 . } } \end{array}
$$

Here both factors in the numerator are positive because $s > 1 > b$ . By construction,

$$
\left( 1 + \frac { b } { s ^ { 2 } } \right) I - \frac { \alpha } { s } ( C + u I + \Delta _ { t } ) = \frac { 1 + b } { s } I - \frac { \alpha } { s } ( C + \Delta _ { t } ) .
$$

Consequently, the divided positions obey

$$
\pmb { x } _ { t + 1 } ^ { \prime } = \left[ \left( 1 + \frac { b } { s ^ { 2 } } \right) \pmb { I } - \frac { \alpha } { s } ( \pmb { C } + u \pmb { I } + \pmb { \Delta } _ { t } ) \right] \pmb { x } _ { t } ^ { \prime } - \frac { b } { s ^ { 2 } } \pmb { x } _ { t - 1 } ^ { \prime } .
$$

This is the generic recurrence with mean $C + u I \succ 0$ , momentum $b / s ^ { 2 } \in [ 0 , 1 )$ , coefficient $\alpha / s > 0$ and the same noise $\Delta _ { t }$ . These parameters satisfy the assumptions of Theorem F.1.

## Step 2. Compare the quotient at the same test matrix.

Keep the matrix X from the statement fixed. The divided recurrence is strictly stable, so the exact criterion (Theorem F.1) gives

$$
\begin{array} { r l } { \cfrac { s - b / s } { \alpha } = \cfrac { 1 - b / s ^ { 2 } } { \alpha / s } } & { } \\ { \phantom { \frac { s - b / s } { \alpha } = } > r _ { C + u I } ( b / s ^ { 2 } ) } & { } \\ { \phantom { \frac { s - b / s } { \alpha } = } } & { \geq q _ { C + u I , b / s ^ { 2 } } ( X ) . } \end{array}\tag{112}
$$

The last inequality holds because $X \succeq 0$ and $\| \boldsymbol { X } \| _ { F } = 1$ make it an admissible test matrix. We compare this last quotient with $q _ { C , b } ( \boldsymbol { X } )$ by undoing the mean shift and comparing the momenta.

First, recall that $\mu = \operatorname { t r } ( C X ^ { 2 } )$ and $\operatorname { t r } ( X ^ { 2 } ) = 1$ , so

$$
\mathrm { t r } \big ( ( C + u I ) X ^ { 2 } \big ) = \mathrm { t r } ( C X ^ { 2 } ) + u \mathrm { t r } ( X ^ { 2 } ) = \mu + u .
$$

The shift identity (Lemma F.3), with the same noise in both quotients, gives

$$
q _ { C + u I , b / s ^ { 2 } } ( \boldsymbol { X } ) = \frac { \mu q _ { C , b / s ^ { 2 } } ( \boldsymbol { X } ) + c _ { b / s ^ { 2 } } u ( \mu + u / 2 ) } { \mu + u } .
$$

Second, $b \geq 0$ and $s > 1$ imply $b / s ^ { 2 } \leq b .$ . The pointwise momentum inequality (67) therefore gives $q _ { C , b / s ^ { 2 } } ( \boldsymbol { X } ) \geq q _ { C , b } ( \boldsymbol { X } )$

Substituting the shift identity into (112), multiplying by $\mu + u > 0$ , and then applying this comparison,

$$
\begin{array} { r l r } {  { \frac { ( s - b / s ) ( \mu + u ) } { \alpha } \geq \mu q _ { C , b / s ^ { 2 } } ( X ) + c _ { b / s ^ { 2 } } u ( \mu + u / 2 ) } } \\ & { } & { \geq \mu q _ { C , b } ( X ) + c _ { b / s ^ { 2 } } u ( \mu + u / 2 ) . } \end{array}\tag{113}
$$

We have weakened the strict first inequality to a non-strict one. The right side now contains the quotient of the original recurrence at the original momentum.

## Step 3. Bound the quotient difference.

Moving all other terms in (113) to the right gives

$$
\begin{array} { r l r } {  { \mu ( q _ { C , b } ( \mathbf { X } ) - \frac { 1 - b } { \alpha } ) \leq \frac { ( s - b / s ) ( \mu + u ) } { \alpha } - \frac { \mu ( 1 - b ) } { \alpha } - c _ { b / s ^ { 2 } } u ( \mu + \frac { u } { 2 } ) } } \\ & { } & { = \mu [ \frac { s - b / s - ( 1 - b ) } { \alpha } - c _ { b / s ^ { 2 } } u ] + u [ \frac { s - b / s } { \alpha } - \frac { c _ { b / s ^ { 2 } } u } { 2 } ] . } \end{array}
$$

The first numerator simplifies as

$$
s - { \frac { b } { s } } - ( 1 - b ) = ( s - 1 ) + b \left( 1 - { \frac { 1 } { s } } \right) = ( s - 1 ) \left( 1 + { \frac { b } { s } } \right) .
$$

Dividing by $\mu > 0 .$ , we obtain

$$
q _ { C , b } ( X ) - \frac { 1 - b } { \alpha } \leq \left[ \frac { ( s - 1 ) ( 1 + b / s ) } { \alpha } - c _ { b / s ^ { 2 } } u \right] + \frac { u } { \mu } \left[ \frac { s - b / s } { \alpha } - \frac { c _ { b / s ^ { 2 } } u } { 2 } \right] .
$$

We simplify these two terms separately. Recall the scalar shift from Step 1 and the definition of $c _ { b } \mathrm { : }$

$$
u = { \frac { ( s - 1 ) ( s - b ) } { s \alpha } } , \qquad c _ { b / s ^ { 2 } } = { \frac { 1 - b / s ^ { 2 } } { 1 + b / s ^ { 2 } } } = { \frac { s ^ { 2 } - b } { s ^ { 2 } + b } } .
$$

For the first term, substituting these expressions and combining denominators gives

$$
\begin{array} { r l } & { \frac { ( s - 1 ) ( 1 + b / s ) } { \alpha } - c _ { b / s ^ { 2 } } u = \frac { s - 1 } { s \alpha } \left[ ( s + b ) - \frac { ( s ^ { 2 } - b ) ( s - b ) } { s ^ { 2 } + b } \right] } \\ & { \qquad = \frac { ( s - 1 ) \left[ ( s + b ) ( s ^ { 2 } + b ) - ( s ^ { 2 } - b ) ( s - b ) \right] } { s \alpha ( s ^ { 2 } + b ) } } \\ & { \qquad = \frac { ( s - 1 ) ( 2 b s ^ { 2 } + 2 b s ) } { s \alpha ( s ^ { 2 } + b ) } } \\ & { \qquad = \frac { 2 b ( s - 1 ) ( s + 1 ) } { \alpha ( s ^ { 2 } + b ) } = \frac { 2 b ( s ^ { 2 } - 1 ) } { ( s ^ { 2 } + b ) \alpha } . } \end{array}
$$

For the second term, first substitute u inside the bracket, then factor its numerator:

$$
\begin{array} { r l } & { \frac { 8 } { \pi } [ \frac { \kappa - b / s } { \alpha } - \frac { \omega _ { 0 } \kappa / \alpha } { 2 } ] } \\ & { = \frac { \mathbf { w } ( s ^ { 2 } - \hat { \omega } ) } { \beta \omega } [ 1 - \frac { ( \frac { \kappa - 1 } { 2 } ) ( s - b ) } { 2 ( s ^ { 2 } + \hat { \omega } ) } ] } \\ & { = \frac { \mathbf { w } ( s ^ { 2 } - \hat { \omega } ) } { 2 ( s ^ { 2 } + \hat { \omega } ) } \frac { [ 2 \kappa ^ { 2 } - \hat { \omega } ] ( s - b ) } { 2 \beta \omega \Delta ( s ^ { 2 } - \hat { \omega } ) } } \\ & { = \frac { \mathbf { w } ( s ^ { 2 } - \hat { \omega } ) [ 2 \kappa ^ { 2 } + ( \hat { \omega } + 1 ) s ] } { 2 \beta \omega \Delta ( s ^ { 2 } + \hat { \omega } ) } } \\ & { = \frac { \mathbf { w } ( s ^ { 2 } - \hat { \omega } ) ( s - \hat { \omega } ) ( s + 1 ) ( s + b ) } { 2 \beta \omega \Delta ( s ^ { 2 } + \hat { \omega } ) } } \\ & { = \frac { \mathbf { w } ( s ^ { 2 } - \hat { \omega } ) ( s - b + 1 ) ( s + b ) } { 2 \beta \omega \Delta ( s ^ { 2 } + \hat { \omega } ) } } \\ & { = \frac { ( s - 1 ) ( s - b ) ( s ^ { 2 } - b ) ( s + 1 ) ( s + b ) } { 2 ( s ^ { 2 } + \hat { \omega } ) ( s ^ { 2 } + \hat { \omega } ) } } \\ &  = \frac { ( s ^ { 2 } - \hat { \omega } ) ( s - b ^ { 2 } + b ) ( s ^ { 2 } - b ) }  2 ( s ^ { 2 } + \hat { \omega } ) ( s ^ { 2 } + \hat { \omega } \end{array}
$$

The penultimate line substitutes the remaining factor u. The last line groups $( s - 1 ) ( s + 1 )$ and $( s - b ) ( s + b )$ . Adding the two terms yields, for every $s > { \sqrt { R } }$

$$
q _ { C , b } ( X ) - \frac { 1 - b } { \alpha } \leq ( s ^ { 2 } - 1 ) \left[ \frac { 2 b } { ( s ^ { 2 } + b ) \alpha } + \frac { ( s ^ { 2 } - b ^ { 2 } ) ( s ^ { 2 } - b ) } { 2 s ^ { 2 } ( s ^ { 2 } + b ) \mu \alpha ^ { 2 } } \right] .
$$

The left side does not depend on s, and the right side is continuous at $s = { \sqrt { R } }$ , with positive denominators. Letting $s \downarrow \sqrt { R }$ therefore gives

$$
q _ { C , b } ( X ) - \frac { 1 - b } { \alpha } \leq ( R - 1 ) \left[ \frac { 2 b } { ( R + b ) \alpha } + \frac { ( R - b ^ { 2 } ) ( R - b ) } { 2 R ( R + b ) \mu \alpha ^ { 2 } } \right] .\tag{114}
$$

We retain this sharper form and now simplify its second coefficient.

## Step 4. Simplify the second coefficient.

The second coefficient in (114) is easier to compare after bounding it by a square. To justify this bound, subtract its two relevant factors:

$$
\begin{array} { c } { \displaystyle \frac { R - b } { R } - \frac { R - b ^ { 2 } } { R + b } = \frac { ( R - b ) ( R + b ) - R ( R - b ^ { 2 } ) } { R ( R + b ) } } \\ { = \frac { R ^ { 2 } - b ^ { 2 } - R ^ { 2 } + R b ^ { 2 } } { R ( R + b ) } } \\ { = \displaystyle \frac { b ^ { 2 } ( R - 1 ) } { R ( R + b ) } \geq 0 . } \end{array}
$$

Multiplying by $( R - b ) / R > 0$ gives

$$
{ \frac { ( R - b ^ { 2 } ) ( R - b ) } { R ( R + b ) } } \leq \left( { \frac { R - b } { R } } \right) ^ { 2 } = \left( 1 - { \frac { b } { R } } \right) ^ { 2 } .
$$

Thus (114) implies

$$
q _ { C , b } ( X ) - \frac { 1 - b } { \alpha } \leq ( R - 1 ) \left[ \frac { 2 b } { ( R + b ) \alpha } + \frac { 1 } { 2 \mu } \left( \frac { 1 - b / R } { \alpha } \right) ^ { 2 } \right] .
$$

This is (66) and completes the proof of Proposition F.2.

## F.5.3 PROOF OF THE AGM CRITICAL LR

Recall from Appendix F.1 that $b = \beta _ { 1 } ( 1 - \eta \lambda _ { \mathrm { W D } } ) , \alpha = \eta ( 1 - \beta _ { 1 } )$ , and

$$
G _ { C } ( \eta ) = \frac { 1 } { \eta } + \frac { \beta _ { 1 } \lambda _ { \mathrm { W D } } } { 1 - \beta _ { 1 } } - r _ { C } \bigl ( \beta _ { 1 } ( 1 - \eta \lambda _ { \mathrm { W D } } ) \bigr ) .
$$

We prove that this function has one zero, which is the endpoint of the stable LRs, and that the moment spectral radius is strictly greater than one above it.

ProofofCorollary F.4. To rule out $\rho ( \mathcal T ) \le 1$ above the root, we rescale a trajectory by $s ^ { - t }$ and let $s \downarrow 1$

## Step 1. Apply the fixed-parameter criterion.

For $\eta < \eta _ { \mathrm { m a x } } ,$ , Theorem F.1 gives strict stability if and only if $G _ { C } ( \eta ) > 0$ . If $\eta _ { \mathrm { m a x } } < \infty$ and $\eta \geq \eta _ { \mathrm { m a x } } .$ , then $b \leq - 1$ . The roots of the mean recurrence along each eigendirection have product $b ,$ so one root has modulus at least one. Strict mean-square stability then fails by the mean-radius inequality (115). $\Delta \mathrm { t } \eta \lambda _ { \mathrm { W D } } = 1$ , the effective momentum is simply $b = 0$ , already covered by the criterion.

## Step 2. Prove monotonicity and continuity.

Recall the pointwise momentum comparison (67): for $- 1 < b ^ { \prime } \leq b < 1$ $q _ { C , b ^ { \prime } } ( \boldsymbol { X } ) \geq q _ { C , b } ( \boldsymbol { X } )$ for every nonzero PSD X. We apply this comparison before taking maxima.

The maximization defining $r _ { C }$ uses the same set $\{ X \subseteq 0 : \| X \| _ { F } = 1 \}$ for every b. Choose a maximizer $X _ { b }$ at $b ,$ which exists by (52). It satisfies $X _ { b } \succeq 0 , \| X _ { b } \| _ { F } ^ { * } = 1$ , and $q _ { C , b } ( \boldsymbol { X } _ { b } ) = r _ { C } ( \boldsymbol { b } )$ Therefore,

$$
\begin{array} { r l r l } { r _ { C } ( b ^ { \prime } ) = \underset { X \succeq 0 , \parallel X \parallel _ { F } = 1 } { \operatorname* { m a x } } q _ { C , b ^ { \prime } } ( X ) \quad \quad } & & { \mathrm { ~ d e f i n i t i o n ~ o f ~ } r _ { C } ( b ^ { \prime } ) } \\ { \ge q _ { C , b ^ { \prime } } ( X _ { b } ) \quad \quad } & & { X _ { b } \mathrm { ~ i s ~ a l s o ~ a d m i s s i b l e ~ a t ~ } b ^ { \prime } } \\ { \ge q _ { C , b } ( X _ { b } ) \quad \quad } & & { \mathrm { ~ t h e ~ p o i n t w i s e ~ i n e q u a l i t y ~ a b o v e } } \\ { = r _ { C } ( b ) . \quad \quad } & & { X _ { b } \mathrm { ~ m a x i m i z e s ~ t h e ~ q u o t i e n t ~ a t ~ } b } \end{array}
$$

Hence $b ^ { \prime } \leq b$ implies $r _ { C } ( b ^ { \prime } ) \geq r _ { C } ( b )$ , which is the claimed nonincreasing dependence on $b .$

Return now to the AGM parameter curve and set $b _ { i } : = \beta _ { 1 } ( 1 - \eta _ { i } \lambda _ { \mathrm { W D } } )$ . Since $\eta _ { 2 } < \eta _ { 1 }$ , we have $b _ { 2 } \geq$ $b _ { 1 }$ and therefore $r _ { C } ( b _ { 1 } ) \geq r _ { C } ( b _ { 2 } )$ . Recall that $G _ { C } ( \eta )$ subtracts $r _ { C } ( \boldsymbol { b } )$ from $1 / \eta + \beta _ { 1 } \lambda _ { \mathrm { W D } } / ( 1 - \beta _ { 1 } )$ The offset $\beta _ { 1 } \lambda _ { \mathrm { W D } } / ( 1 - \beta _ { 1 } )$ is constant along this curve, giving

$$
\begin{array} { l } { \displaystyle { G _ { C } ( \eta _ { 2 } ) - G _ { C } ( \eta _ { 1 } ) = \frac { 1 } { \eta _ { 2 } } - \frac { 1 } { \eta _ { 1 } } + r _ { C } ( b _ { 1 } ) - r _ { C } ( b _ { 2 } ) } } \\ { \displaystyle { \ge \frac { 1 } { \eta _ { 2 } } - \frac { 1 } { \eta _ { 1 } } > 0 . } } \end{array}
$$

This is (73). The quotient is jointly continuous in $b \in ( - 1 , 1 )$ and X on the compact PSD unit sphere, with denominator at least $2 \lambda _ { \mathrm { m i n } } ( C )$ . Its maximum $r _ { C } ( \boldsymbol { b } )$ ), and hence $G _ { C }$ , is continuous.

## Step 3. Check the endpoints.

As $\eta \downarrow 0 , b  \beta _ { 1 } < 1$ and

$$
G _ { C } ( \eta ) = \frac { 1 } { \eta } + \frac { \beta _ { 1 } \lambda _ { \mathrm { W D } } } { 1 - \beta _ { 1 } } - r _ { C } ( b ) \longrightarrow + \infty .
$$

If $\beta _ { 1 } \lambda _ { \mathrm { W D } } > 0$ , then $b \downarrow - 1$ as $\eta \uparrow \eta _ { \mathrm { m a x } }$ . The rank-one lower bound gives

$$
r _ { C } ( b ) \geq \frac { 1 - b } { 1 + b } \frac { L _ { C } } { 2 } \longrightarrow \infty , ~ G _ { C } ( \eta ) \longrightarrow - \infty .
$$

If $\beta _ { 1 } \lambda _ { \mathrm { W D } } = 0$ , then $b = \beta _ { 1 }$ is fixed and

$$
G _ { C } ( \eta ) = \frac { 1 } { \eta } - r _ { C } ( \beta _ { 1 } ) \longrightarrow - r _ { C } ( \beta _ { 1 } ) < 0 ( \eta  \infty ) .
$$

Continuity and strict monotonicity now give one root, with $G _ { C } > 0$ precisely below it.

## Step 4. Locate the second-moment spectral radius.

First fix $\eta \in \left( \eta _ { \mathrm { c r i t } } , \eta _ { \mathrm { m a x } } \right)$ , so that $- 1 < b < 1$ and $G _ { C } ( \eta ) < 0$ . Suppose for a contradiction that its second-moment map satisfies $\rho ( \mathcal { T } ) \le 1$ . For every $s > 1$ , dividing the state at time t by $s ^ { t }$ gives

$$
\rho ( s ^ { - 2 } \mathcal T ) = s ^ { - 2 } \rho ( \mathcal T ) \le s ^ { - 2 } < 1 .
$$

Recall the position calculation in the proof of Proposition F.2. The divided recurrence has coefficient $\alpha / s$ , momentum $b / s ^ { 2 }$ , unchanged noise, and mean $C + u I$ , where

$$
u : = \frac { ( s - 1 ) ( s - b ) } { s \alpha } > 0 , 1 + \frac { b } { s ^ { 2 } } - \frac { \alpha } { s } u = \frac { 1 + b } { s } .
$$

That calculation needs only $- 1 < b < 1 :$ for every $s > 1$ we have $- 1 < b / s ^ { 2 } < 1 , \alpha / s > 0$ , and $u > 0$ because $s > b$ . Thus the shifted mean remains positive definite even when $b < 0 ,$ . The divided recurrence is strictly stable, so the exact criterion gives, for every $X \succeq 0$ with $\| \boldsymbol { X } \| _ { F } = 1$

$$
\begin{array} { l } { { q _ { C + u I , b / s ^ { 2 } } ( x ) \le r _ { C + u I } ( b / s ^ { 2 } ) } } \\ { { < \displaystyle \frac { 1 - b / s ^ { 2 } } { \alpha / s } = \displaystyle \frac { s - b / s } { \alpha } . } } \end{array}
$$

As $s \downarrow 1$ , we have $u \to 0$ and $b / s ^ { 2 } \to b$ . The quotient is continuous in these parameters, with a positive denominator. Hence

$$
\begin{array} { r l } & { q _ { C , b } ( \pmb { X } ) \le \displaystyle \frac { 1 - b } { \alpha } \qquad } & { \mathrm { t a k e ~ t h e ~ l i m i t ~ f o r ~ e a c h ~ f i x e d ~ } X } \\ & { \quad r _ { C } ( b ) = \displaystyle \operatorname* { m a x } _ { \pmb { X } \ge 0 , \| \pmb { X } \| _ { F } = 1 } q _ { C , b } ( \pmb { X } ) \le \displaystyle \frac { 1 - b } { \alpha } , } \\ & { \quad G _ { C } ( \eta ) = \displaystyle \frac { 1 - b } { \alpha } - r _ { C } ( b ) \ge 0 . } \end{array}
$$

This contradicts $G _ { C } ( \eta ) < 0$ . Thus $\rho ( \mathcal T ) > 1$ throughout this interval.

If $\eta _ { \mathrm { m a x } } <$ ∞ and $\eta > \eta _ { \mathrm { m a x } } .$ , then $b < - 1$ . The two roots along any eigendirection of the mean have product $b ,$ so at least one has modulus greater than one. $\mathrm { { A t } } \eta = \eta _ { \mathrm { { m a x } } } .$ we have $b = - 1$ . For an eigenvalue $a _ { i } > 0$ of C, the characteristic polynomial satisfies

$$
p _ { i } ( x ) = x ^ { 2 } + \alpha a _ { i } x - 1 , \qquad p _ { i } ( - 1 ) = - \alpha a _ { i } < 0 , \qquad \operatorname* { l i m } _ { x \to - \infty } p _ { i } ( x ) = + \infty .
$$

It therefore has a root below −1. In both cases, the mean-radius inequality (115) implies $\rho ( \mathcal T ) > 1$

Finally, the entries of the second-moment map are continuous in η, so its spectral radius is continuous. The stable rates approaching $\eta _ { \mathrm { c r i t } }$ from below give $\rho ( \mathcal { T } ) \le 1$ at the critical rate. Since $G _ { C } ( \eta _ { \mathrm { c r i t } } ) = 0$ the exact criterion excludes strict stability there and gives $\rho ( \mathcal T ) \ge 1$ . Thus the radius equals one at the critical rate. This completes the proof of Corollary F.4. □

## F.6 AUXILIARY RESULTS

The proofs use four finite-dimensional lemmas on second-moment maps and on maps that preserve positive semidefinite matrices. We also use the Cauchy–Schwarz and Minkowski inequalities, Jensen’s inequality, the Neumann series for $( s \mathcal { T } - \mathcal { U } ) ^ { - 1 }$ when $| s | > \rho ( \mathcal { U } )$ , the Rayleigh quotient characterization of the largest eigenvalue of a self-adjoint operator, and continuity of polynomial roots in their coefficients. The cutoff argument uses the Jordan decomposition: the powers of a finite-dimensional operator with spectral radius one grow at most polynomially.

We distinguish two notions of positivity. A map $\mathcal { U } : \mathbb { S } _ { n }  \mathbb { S } _ { n }$ preserves PSD matrices if $x \succeq$ 0 implies ${ \bf \bar { \chi } } _ { } ( { \bf \chi } _ { } ) \mathrm { ~ \succeq ~ 0 ~ }$ . A self-adjoint operator D on $\mathbb { S } _ { n }$ is positive definite, written $\mathcal { D } \succ 0 ,$ if $\langle \pmb { X } , \mathcal { D } \pmb { X } \rangle > 0$ for every nonzero symmetric X. Preservation of PSD matrices does not imply positive definiteness as an operator, and the converse fails as well. Second moments are uncentered throughout: $\pmb { \Sigma } _ { t } = \mathbb { E } [ \pmb { y } _ { t } \pmb { y } _ { t } ^ { \top } ]$ , so that tr $\Sigma _ { t } = \mathbb { E } \left. \pmb { y } _ { t } \right. ^ { 2 }$ , and no zero-mean assumption is imposed on the state.

Lemma F.14 (Second-moment characterization). Let $\pmb { y } _ { t + 1 } = \pmb { M } _ { t } \pmb { y } _ { t }$ , where the $M _ { t }$ are $i . i . d .$ with finite second moments and $\scriptstyle { \mathbf { 3 0 } }$ is deterministic. Define $\mathcal { T } ( S ) : = \mathbb { E } [ M _ { t } S M _ { t } ^ { \top } ]$ on $\mathbb { S } _ { n }$ . Then $\mathbb { E } \left\| y _ { t } \right\| ^ { 2 } \to 0 f o r$ every initial state $i f$ and only $i f \rho ( \mathcal { T } ) < 1$ . Strict mean-square stability also implies $\rho ( \mathbb { E } M _ { t } ) < 1$ , and in general

$$
\begin{array} { r } { \rho ( \mathbb { E } M _ { t } ) ^ { 2 } \le \rho ( \mathcal { T } ) . } \end{array}\tag{115}
$$

Proof of Lemma $F . l 4 .$ Write $\Sigma _ { t } : = \mathbb { E } [ { \pmb { y } } _ { t } { \pmb { y } } _ { t } ^ { \top } ]$ . Since ${ \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \mathbf { } } _ { \mathbf { } } \mathbf { \Xi } _ { \mathbf { } } \mathbf { \Lambda } _ { \mathbf { } } \mathbf { \Lambda } _ { \mathbf { } } \textbf { } _ { \mathbf { } } \textbf { } \textbf { } _ { \mathrm { } }$ depends only on the initial state and on $M _ { 0 } , \dots , M _ { t - 1 }$ , it is independent of $M _ { t }$ , so

$$
\begin{array} { r l } & { \pmb { \Sigma } _ { t + 1 } = \mathbb { E } [ \pmb { M } _ { t } \pmb { y } _ { t } \pmb { y } _ { t } ^ { \top } \pmb { M } _ { t } ^ { \top } ] } \\ & { \qquad = \mathbb { E } \big [ \pmb { M } _ { t } \mathbb { E } [ \pmb { y } _ { t } \pmb { y } _ { t } ^ { \top } \mid \pmb { M } _ { t } ] \pmb { M } _ { t } ^ { \top } \big ] } \\ & { \qquad = \mathbb { E } [ \pmb { M } _ { t } \pmb { \Sigma } _ { t } \pmb { M } _ { t } ^ { \top } ] = \mathcal { T } ( \pmb { \Sigma } _ { t } ) , } \end{array}
$$

with $\pmb { \Sigma } _ { 0 } = \pmb { y } _ { 0 } \pmb { y } _ { 0 } ^ { \top }$ . The second equality uses the tower property, conditioning on $M _ { t }$ , and the third uses independence. $\mathrm { I f } \rho ( \mathcal T ) < 1$ , the Jordan form of $\tau$ shows that $\| \mathcal T ^ { t } \|$ is bounded by a polynomial in t times a decaying exponential, so $\mathcal { T } ^ { t } \to 0$ and tr $\Sigma _ { t } \to 0$

Conversely, suppose $\mathbb { E }  \pmb { y } _ { t }  ^ { 2 }  \mathrm { ~ 0 ~ }$ for every $\scriptstyle { \mathbf { 3 0 } }$ . Since $\Sigma _ { t } \ \succeq \ 0$ we have $\| \Sigma _ { t } \| _ { F } \leq \operatorname { t r } \Sigma _ { t }$ , so $\mathcal { T } ^ { t } ( \pmb { y } _ { 0 } \pmb { y } _ { 0 } ^ { \top } ) = \pmb { \Sigma } _ { t }  0$ for every $\mathbf { { \boldsymbol { { y } } } _ { 0 } }$ . Rank-one symmetric matrices span $\mathbb { S } _ { n } \colon$ the diagonal basis elements are $e _ { i } e _ { i } ^ { \phantom { } }$ , and each off-diagonal symmetric basis element is $( e _ { i } ^ { \cdot } + e _ { j } ) ( e _ { i } + e _ { j } ) ^ { \top } - e _ { i } e _ { i } ^ { \top } -$ $e _ { j } e _ { j } ^ { \intercal }$ . Hence $\mathcal T ^ { t } \to 0$ as an operator on the finite-dimensional space $\mathbb { S } _ { n } .$ An eigenvalue of modulus at least one would contradict this convergence, so $\rho ( \mathcal T ) < 1$

For the statements about the mean, independence gives $\mathbb { E } \pmb { y } _ { t + 1 } = ( \mathbb { E } \pmb { M } _ { t } ) \mathbb { E } \pmb { y } _ { t }$ , and the matrices have the same law at each time. Iterating and applying Jensen’s inequality,

$$
\begin{array} { r l r } & { } & { \mathbb { E } { \pmb { y } } _ { t } = ( \mathbb { E } { \pmb { M } } _ { t } ) ^ { t } { \pmb { y } } _ { 0 } , } \\ & { } & { \left\| ( \mathbb { E } { \pmb { M } } _ { t } ) ^ { t } { \pmb { y } } _ { 0 } \right\| ^ { 2 } = \| \mathbb { E } { \pmb { y } } _ { t } \| ^ { 2 } \leq \mathbb { E } \left\| { \pmb { y } } _ { t } \right\| ^ { 2 } = \mathrm { t r } \mathcal { T } ^ { t } ( { \pmb { y } } _ { 0 } { \pmb { y } } _ { 0 } ^ { \top } ) . } \end{array}
$$

Under strict mean-square stability the right side tends to zero for every $\mathbf { { \boldsymbol { { y } } } _ { 0 } } \cdot$ , so $( \mathbb { E } M _ { t } ) ^ { t }  0$ and $\rho ( \mathbb { E } M _ { t } ) < 1$ . To obtain (115), apply this implication to the matrices ${ M } _ { t } / { \sqrt { s } }$ for any real $s > \rho ( \mathcal T )$ Their second-moment map is ${ \mathcal { T } } / s ,$ , so

$$
\begin{array} { r l r l } & { \rho ( \mathcal { T } / s ) = \rho ( \mathcal { T } ) / s < 1 , } & { \qquad } & & { \mathrm { c h o i c e ~ o f ~ } s } \\ & { \rho \bigl ( \mathbb { E } M _ { t } / \sqrt { s } \bigr ) < 1 , } & { \qquad } & & { \mathrm { t h e ~ m e a n \mathrm { - } s t a b i l i t y ~ i m p l i c a t i o n ~ a b o v e } } \\ & { \rho ( \mathbb { E } M _ { t } ) ^ { 2 } < s . } & { \qquad } & & { \mathrm { r e s c a l e ~ t h e ~ m e a n ~ m a t r i x } } \end{array}
$$

Letting $s \downarrow \rho ( \mathcal { T } )$ proves (115), including when $\rho ( \mathcal T ) = 0$ . This completes the proof of Lemma F.14.

Connection to the Kronecker second-moment matrix. The matrix $\mathbb { E } [ M _ { t } \otimes M _ { t } ]$ represents the map $S \mapsto \mathbb { E } [ M _ { t } S M _ { t } ^ { \top } ]$ on all $n \times n$ matrices in the column-stacked basis. For the projected AGM dynamics, this is $K ^ { + } \bar { ( \eta ) }$ in (20). The map $\tau$ in the preceding lemma is the restriction to symmetric matrices.

Write $\Phi _ { t } : = M _ { t - 1 } \cdot \cdot \cdot M _ { 0 }$ . For fixed vectors ${ \mathbf { } } u , v ,$ , the Cauchy–Schwarz inequality gives

$$
\begin{array} { r l } & { \left\| \mathbb { E } [ \Phi _ { t } { \pmb u } { \pmb v } ^ { \top } \Phi _ { t } ^ { \top } ] \right\| _ { F } \leq \mathbb { E } \left\| \Phi _ { t } { \pmb u } { \pmb v } ^ { \top } \Phi _ { t } ^ { \top } \right\| _ { F } } \\ & { \qquad = \mathbb { E } \bigl [ \left\| \Phi _ { t } { \pmb u } \right\| \left\| \Phi _ { t } { \pmb v } \right\| \bigr ] } \\ & { \qquad \leq \bigl ( \mathbb { E } \left\| \Phi _ { t } { \pmb u } \right\| ^ { 2 } \mathbb { E } \left\| \Phi _ { t } { \pmb v } \right\| ^ { 2 } \bigr ) ^ { 1 / 2 } . } \end{array}
$$

If the symmetric second moments decay, the right side tends to zero. Since the matrices ${ \mathbf { } } u v ^ { \top }$ span all matrices, the full moment map also has powers tending to zero. The converse follows by restriction. Replacing $M _ { t }$ by ${ M } _ { t } / { \sqrt { s } }$ divides both moment maps by s. Applying the same argument for every $s > 0$ shows that the two spectral radii are below s for exactly the same s, hence are equal.

Recall that the history state uses $( \pmb { x } _ { t } , \pmb { x } _ { t - 1 } )$ , whereas (69) uses position and momentum $( \pmb { x } _ { t } , \pmb { n } _ { t } )$ The position update gives their linear relation

$$
\binom { x _ { t } } { n _ { t } } = \binom { I } { - \eta ^ { - 1 } I } \quad \eta ^ { - 1 } ( 1 - \eta \lambda _ { \mathrm { W D } } ) I \bigg ) \binom { x _ { t } } { x _ { t - 1 } } .
$$

This change is invertible when $\eta \lambda _ { \mathrm { W D } } \neq 1$ , and an invertible change of state induces a similarity between the moment maps. Their spectral radii are therefore equal there. The entries of both moment maps are polynomials in η, and the spectral radius is continuous in the entries, so the equality extends to $\eta \lambda _ { \mathrm { W D } } = 1$ as well.

Lemma F.15 (A Lyapunov criterion for PSD-preserving maps). $I f u : \mathbb { S } _ { n } \to \mathbb { S } _ { n }$ preserves PSD matrices, then

$$
\rho ( \mathcal { U } ) < 1 \quad \Longleftrightarrow \quad t h e r e e x i s t s \ W \succ 0 w i t h W - \mathcal { U } ( W ) \succ 0 .
$$

ProofofLemma F.15. If $\rho ( \mathcal { U } ) < 1$ , the Neumann series $\begin{array} { r } { W : = \sum _ { j > 0 } \mathcal { U } ^ { j } ( I ) } \end{array}$ converges, satisfies $W \succeq I$ because every summand is PSD, and $W - \mathcal { U } ( W ) = I$ . Conversely, given such a $W _ { : }$ choose $\varepsilon \in ( 0 , 1 )$ with $0 \preceq \mathcal { U } ( W ) \preceq ( 1 - \varepsilon ) W$ . For every symmetric X there is a finite a with $- a W \preceq X \preceq a W$ . Since U preserves the order $\preceq ,$

$$
- a ( 1 - \varepsilon ) ^ { j } W \preceq \mathcal { U } ^ { j } ( X ) \preceq a ( 1 - \varepsilon ) ^ { j } W \qquad ( j \geq 0 ) .
$$

Applying this to a finite basis of $\mathbb { S } _ { n }$ gives $\mathcal { U } ^ { j }  0$ , hence $\rho ( \mathcal { U } ) < 1$ . This completes the proof of Lemma F.15. □

Lemma F.16 (Perron–Frobenius theorem for PSD-preserving maps). Let $\mathcal { U } : \mathbb { S } _ { n } \to \mathbb { S } _ { n }$ preserve PSD matrices, and suppose $R = \rho ( \mathcal { U } ) > 0$ . There is a nonzero ${ \bar { W _ { \mathbf { \lambda } } } } \in \mathbb { S } _ { n } ^ { + }$ with $\mathcal { U } ( \mathbf { W } ) = R \mathbf { W }$

This is the Perron–Frobenius theorem for a linear map that leaves a proper cone invariant, here the cone $\mathbb { S } _ { n } ^ { + }$ of positive semidefinite matrices (Vandergraft, 1968; Berman & Plemmons, 1994, Chapter 1).

Lemma F.17 (Jury stability test for a real quadratic). Both roots of $p ( x ) = x ^ { 2 } - \sigma x + \pi ,$ , with real $\sigma ,$ π, lie strictly inside the unit disk ifand only $i f | \pi | < 1 , 1 - \sigma + \pi > 0$ , and $1 + \sigma + \pi > 0$

This is the Schur–Cohn, or Jury, stability test for a real quadratic (Jury, 1964): the conditions are $| \pi | < 1 , p ( 1 ) > 0 { \mathrm { ~ a n d ~ } } p ( - 1 ) > 0$ , and they are equivalent to both roots lying in the open unit disk.