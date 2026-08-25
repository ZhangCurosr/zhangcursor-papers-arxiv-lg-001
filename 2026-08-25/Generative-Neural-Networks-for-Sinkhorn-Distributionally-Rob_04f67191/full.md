# Generative Neural Networks for Sinkhorn Distributionally Robust Hypothesis Testing

Fenglin Zhang, Teyan Liu, Jie Wang<sup>∗</sup>

August 25, 2026

## Abstract

This paper studies the Sinkhorn distributionally robust hypothesis testing (SDRHT) problem, seeking a robust detector against least-favorable distributions in Sinkhorn discrepancybased ambiguity sets centered at the empirical distributions. Existing approaches solve this problem by solving large-scale conic programs, which are not scalable. To overcome this, we propose a generative framework that learns least-favorable distributions and supports eficient training and end-to-end sampling. For the Sinkhorn discrepancy-based ambiguity sets, we first derive an equivalent conditional-KL-divergence representation with respect to kernel-smoothed reference distributions. This property allows us to prove strong duality for both constrained and unconstrained minimax SDRHT formulations. Based on the closed-form optimal detector and Brenier’s theorem, we reformulate the max-min dual formulation as a maximization problem over convex potentials whose gradients characterize invertible transport maps between kernel-smoothed distributions and their least-favorable counterparts. We eficiently approximate these potentials using Hyper Input Convex Neural Networks (HyCNNs) equipped with stochastic gradient estimators and prove the representation power of HyCNNs and the distributional universality of their induced transport maps. Numerical results show that the proposed method achieves superior accuracy and robustness across diferent sample sizes and dimensions, while avoiding the scalability limitations of classical SDRHT methods.

Keywords: Sinkhorn distributionally robust optimization, robust hypothesis testing, generative model, least-favorable distribution generation, hyper input convex neural network

## 1 Introduction

Hypothesis testing is a fundamental problem in statistics and scientific discovery, aiming to find an optimal detector that, given new data, can discriminate between two competing hypotheses while minimizing decision errors. The Neyman–Pearson lemma [1] yields the optimal likelihoodratio test when the densities of the underlying distributions are known. In practice, these distributions are typically unknown; instead, the decision-maker has only access to their independent and identically distributed (i.i.d.) samples. This setting motivates data-driven hypothesis tests that are both statistically reliable and computationally eficient.

However, data-driven tests trained on finite samples are vulnerable to distributional misspeci fication [2]. On the one hand, for some applications like healthcare [3] and anomaly detection [4], where samples are limited and noisy, reliable estimation of the underlying distribution is challenging due to inevitable measurement and statistical errors. On the other hand, real data may be non-stationary or may undergo distributional shifts, leading to diferences between training and testing data. To improve the reliability of hypothesis testing, especially in high-stakes scenarios, robust methods are necessary to circumvent these challenges [5, 6].

Distributionally robust optimization (DRO) [7] ofers a powerful paradigm for making robust decisions under distributional uncertainty. When applied to hypothesis testing, DRO op timizes the robust detector against the least-favorable distributions (also known as worst-case distributions) within an ambiguity set consisting of plausible distributions, safeguarding against misspecification and enhancing out-of-sample reliability when applied to new, unseen data [8]. The design of the ambiguity set is crucial for balancing computational tractability and robust out-of-sample performance [9]. Popular choices include Huber’s contamination model [10], f - divergence [11, 12], moments [13, 14], kernels [15, 16], or Wasserstein distance [8, 17].

DRO with ambiguity sets constructed using the Sinkhorn discrepancy has recently received increasing attention [18, 19]. Conventional Wasserstein DRO-based hypothesis testing restricts the worst-case distributions to have the same support as that of the empirical distributions [8]. By adding entropic regularization to the Wasserstein distance, the Sinkhorn discrepancy provides a smooth alternative to the Wasserstein distance [20], and the Sinkhorn discrepancy-based ambiguity sets naturally encourage continuous distributional estimates. Thus, unlike the Wasserstein distance-based ambiguity set, if a testing sample is outside the support of the observed data, it is still feasible to leverage the likelihood ratio detector to design a robust test. However, finding the least-favorable distributions in Sinkhorn-based hypothesis testing is an infinite-dimensional optimization problem. Existing approaches approximately solve it using sample average approximation and result in large-scale conic programs [21, 22], whose computational cost grows rapidly with the sample size. It is desirable to develop more tractable approaches for solving the robust hypothesis testing problem with Sinkhorn-based ambiguity sets.

Parallel to these developments in robust statistics, flow-based generative models have shown remarkable success in learning complex distributions as pushforwards of simple reference distributions through invertible and diferentiable maps [23, 24, 25]. This framework is not only adept at generating new samples but is also naturally suited for solving optimization problems over probability space, e.g., generating least-favorable distributions in DRO [26, 27, 28]. By parameterizing a distribution as a flow-based transformation starting from a probability density, one can eficiently optimize an objective over the probability space using gradient-based methods [29, 30].

In this work, we leverage flow-based generative models to solve the Sinkhorn distributionally robust hypothesis testing (SDRHT). Based on the theoretical results of optimal transport, we introduce a generative framework to learn continuous least-favorable distributions within the Sinkhorn-based ambiguity sets by optimizing convex potentials. By parameterizing the convex potentials, our approach converts the original infinite-dimensional optimization into a finitedimensional and tractable problem. Our main contributions are summarized as follows:

![](images/18bd80eda96377d5eaa7c78e9c2dc17b47e3e20d0973bc499aeb8a21ff6ad7b4.jpg)  
Figure 1: Illustration of the training and testing processes of our least-favorable detector.

• Novel Reformulation: We establish a strong duality result for SDRHT and thereby reformulate the original minimax optimization as a max-min problem. As the inner minimized detector can be explicitly derived, this problem is equivalent to a maximum problem over distributions. Leveraging Brenier’s theorem, we further reformulate it as a maximization problem over convex potentials.

• Generative Framework: We parameterize the convex potentials by Hyper Input Convex Neural Networks (HyCNNs) and develop stochastic gradient estimators for the objective. This transforms the infinite-dimensional optimization into a finite-dimensional one, enabling eficient and scalable training and sampling. We also prove the representation power of HyCNNs and the distributional universality of their induced gradient maps.

• Empirical Validation: We validate the performance of our method through extensive numerical studies on both synthetic datasets and real-world data with diferent sample sizes and dimensions, demonstrating its superiority over existing baselines in terms of both accuracy and robustness.

Figure 1 illustrates the training and testing processes of the proposed method using a toy Moons dataset. By exploiting the geometry of the Sinkhorn-based ambiguity set (Proposition 1 and Eq. (4)), the least-favorable distributions $( \mathbb { P } _ { 0 } ^ { \star } , \mathbb { P } _ { 1 } ^ { \star } )$ are generated by first applying kernel smoothing to the empirical distributions $\widehat { \mathbb { P } } _ { 0 }$ and $\widehat { \mathbb { P } } _ { 1 }$ to obtain $\mathbb { P } _ { 0 } ^ { \epsilon }$ and $\mathbb { P } _ { 1 } ^ { \epsilon }$ , and next robustifying them. In the robustification procedure, we learn convex-gradient maps (see Eq. (7) and Algorithm 1) that generate $\mathbb { P } _ { 0 } ^ { \star }$ and $\mathbb { P } _ { 1 } ^ { \star }$ based on the kernel-smoothed reference distributions $\mathbb { P } _ { 0 } ^ { \epsilon } , \mathbb { P } _ { 1 } ^ { \epsilon }$ . The resulting least-favorable distributions induce a nonlinear robust decision boundary with high accuracy.

The remainder of this paper is organized as follows. Section 2 presents the models and properties of SDRHT. Section 3 develops a framework for generating least-favorable distributions. Section 4 shows the representation power of HyCNNs. Section 5 reports the numerical results. Section 6 concludes this paper. Proofs and additional technical notes are provided in the online appendix to this paper.

Notations. For $K \in \mathbb { N } .$ , let $[ K ] : = \{ 1 , \cdots , K \}$ . We write $\mathbb { P } \ll \nu$ if the probability measure P is absolutely continuous with respect to the measure ν. We denote $\mathbb { P } \otimes \mathbb { Q }$ by the product measure of two probability measures P and Q. For a transport map $T : \Omega \to \Omega$ , the pushforward of a probability measure P is denoted as $T _ { \# } \mathbb { P } .$ , such that $T _ { \# } \mathbb { P } ( A ) = \mathbb { P } ( T ^ { - 1 } ( A ) )$ for any measurable set $A \subseteq \Omega$ . For a convex function $\varphi : \mathbb { R } ^ { d }  \mathbb { R } \cup \{ \infty \}$ , denote its convex conjugate by $\varphi ^ { * } ( y ) =$ $\scriptstyle \operatorname* { s u p } _ { \pmb { x } \in \mathbb { R } ^ { d } } \{ \pmb { y } ^ { \top } \pmb { x } - \varphi ( \pmb { x } ) \}$ for any $\boldsymbol { y } \in \mathbb { R } ^ { d }$

## 2 Sinkhorn Distributionally Robust Hypothesis Testing

In this section, we introduce the Sinkhorn distributionally robust hypothesis testing (SDRHT), derive its strong duality property, and obtain its equivalent functional optimization reformulation.

Let $\Omega \subseteq \mathbb { R } ^ { d }$ be a closed sample space. Denote $\mathscr { P } ( \Omega )$ as the set of all probability measures on Ω, and $\mathcal { M } ( \Omega )$ as the set of measures on Ω. Denote by ${ \mathcal P } _ { k } \subset { \mathcal P } \left( \Omega \right)$ the ambiguity set under hypotheses $H _ { k } .$ , where $k \in \mathbb { K } = \{ 0 , 1 \}$ . Given a testing sample $\omega ,$ , the composite hypothesis testing decides which one of the following hypotheses is true:

$$
\begin{array} { r l r } { H _ { 0 } : \omega \sim \mathbb { P } _ { 0 } , \mathbb { P } _ { 0 } \in \mathcal { P } _ { 0 } ; } & { { } } & { H _ { 1 } : \omega \sim \mathbb { P } _ { 1 } , \mathbb { P } _ { 1 } \in \mathcal { P } _ { 1 } . } \end{array}
$$

A hypothesis test (detector) $\phi : \Omega \to \mathbb { R }$ is a measurable function. For any testing sample $\omega ,$ it accepts the null hypothesis $H _ { 0 }$ and rejects alternative hypothesis $H _ { 1 }$ whenever $\phi ( \omega ) \geq 0 .$ , otherwise accepts $H _ { 1 }$ and rejects $H _ { 0 }$ . The exact risk of this detector is defined as the sum of type-I and type-II errors (also known as the false positive rate and the false negative rate, respectively). For this nonconvex objective, we follow the generating-function approach [8, 31] and optimize the following convex surrogate risk instead:

$$
\begin{array} { r } { \Phi ( \phi ; \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) : = \mathbb { E } _ { \omega \sim \mathbb { P } _ { 0 } } [ \ell \circ ( - \phi ) ( \omega ) ] + \mathbb { E } _ { \omega \sim \mathbb { P } _ { 1 } } [ \ell \circ \phi ( \omega ) ] , } \end{array}
$$

where the generating function ℓ is defined as follows.

Definition 1 (Generating Function). A generating function $\ell : \mathbb { R } \to \mathbb { R } _ { + } \cup \{ \infty \}$ is a non-negative, nondecreasing, convex function such that $\ell ( 0 ) = 1$ and $\scriptstyle \operatorname* { l i m } _ { t \to - \infty } \ell ( t ) = 0$ ^

For some common choices of the generating function, Table 1 summarizes the corresponding closed-form optimal detectors $\phi _ { \mathrm { o p t } }$ given distributions $\mathbb { P } _ { 0 }$ and $\mathbb { P } _ { 1 } \ [ 8 , 2 1 ]$ , where $p _ { 0 }$ and $p _ { 1 }$ represent the probability density functions of $\mathbb { P } _ { 0 }$ and $\mathbb { P } _ { 1 }$ . In this paper, we choose the generating function $\ell ( t ) : = \exp ( t )$

Given sets of training samples $\{ \pmb { x } _ { k } ^ { ( 1 ) } , \cdots , \pmb { x } _ { k } ^ { ( n _ { k } ) } \}$ for all $k \in \mathbb { K }$ , denote the corresponding empirical distributions as $\begin{array} { r } { \widehat { \mathbb { P } } _ { k } = \frac { 1 } { n _ { k } } \sum _ { i = 1 } ^ { n _ { k } } \delta _ { x _ { k } ^ { ( i ) } } } \end{array}$ , where $n _ { k }$ denotes the number of samples from $\mathbb { P } _ { k } \in \mathcal { P } _ { k }$ . The following definition introduces the Sinkhorn discrepancy, an entropy-regularized variant of the

Table 1: Common choices of generating function (first column), together with their corresponding optimal detector (second column) and surrogate risk (third column).
<table><tr><td> $\ell ( t )$ </td><td>Optimal  $\phi$ </td><td>Corresponding optimal  $1 - \Phi ( { \mathbb P } _ { 0 } , { \mathbb P } _ { 1 } ) / 2$ </td></tr><tr><td> $\exp ( t )$ </td><td> $\log { \sqrt { p _ { 0 } / p _ { 1 } } }$ </td><td> $H ^ { 2 } ( \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } )$ </td></tr><tr><td> $\log ( 1 + \exp ( t ) ) / \log 2$ </td><td> $\log ( p _ { 0 } / p _ { 1 } )$ </td><td> $\mathrm { J } \mathsf { S } ( \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) / \log 2$ </td></tr><tr><td> $( t + 1 ) ^ { 2 }$ </td><td> $1 - 2 p _ { 0 } / ( p _ { 0 } + p _ { 1 } )$ </td><td> $\chi ^ { 2 } ( { \mathbb P } _ { 0 } , { \mathbb P } _ { 1 } )$ </td></tr><tr><td> $( t + 1 ) _ { + }$ </td><td> $\mathrm { s i g n } ( p _ { 0 } - p _ { 1 } )$ </td><td> $\mathrm { T V } ( \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } )$ </td></tr></table>

$H ^ { 2 } ;$ : Hellinger divergence; JS: Jensen-Shannon divergence; TV: Total variation distance.

Wasserstein distance that quantifies the distance between distributions [18, 21]. Throughout the paper, we use the quadratic transport cost $c ( x , z ) : = \textstyle \frac { 1 } { 2 } \| x - z \| _ { 2 } ^ { 2 }$

Definition 2 (Sinkhorn Discrepancy). Consider any two distributions $\mathbb { P } , \mathbb { Q } \in \mathcal { P } ( \Omega )$ and reference measure $\nu \in \mathcal { M } ( \Omega )$ such that $\mathbb { Q } \ll \nu$ . For regularization parameter $\epsilon > 0$ , the Sinkhorn discrepancy between two distributions P and Q is defined as

$$
\mathcal { S } _ { \epsilon } ( \mathbb { P } , \mathbb { Q } ) = \operatorname* { i n f } _ { \gamma \in \Gamma ( \mathbb { P } , \mathbb { Q } ) } \left\{ \mathbb { E } _ { ( \pmb { x } , z ) \sim \gamma } [ c ( \pmb { x } , z ) ] + \epsilon \cdot \mathcal { D } _ { \mathrm { K L } } ( \gamma | | \mathbb { P } \otimes \nu ) \right\} ,
$$

where $c ( { \pmb x } , z )$ represents the transport cost function, Γ(P, Q) denotes the set of couplings of P and $\mathbb { Q } ,$ respectively, and $\mathcal { D } _ { \mathrm { K L } } ( \gamma \Vert \mathbb { P } \otimes \nu )$ denotes the relative entropy of distribution γ with respect to measure $\mathbb { P } \otimes \nu , i . e . ,$

$$
\mathcal { D } _ { \mathrm { K L } } ( \gamma | | \mathbb { P } \otimes \nu ) = \int _ { \Omega \times \Omega } \log \frac { \mathrm { d } \gamma ( \pmb { x } , z ) } { \mathrm { d } \mathbb { P } ( \pmb { x } ) \mathrm { d } \nu ( z ) } \mathrm { d } \gamma ( \pmb { x } , z ) .\tag{^}
$$

Let the space of detectors be $\mathcal { F }$ . We consider the following SDRHT problem, optimizing a detector to minimize the least-favorable risk in ambiguity sets:

$$
\operatorname* { i n f } _ { \phi \in \mathcal { F } } \operatorname* { s u p } _ { \mathbb { P } _ { 0 } \in \mathbb { B } _ { \rho _ { 0 } } ( \widehat { \mathbb { P } } _ { 0 } ) , \mathbb { P } _ { 1 } \in \mathbb { B } _ { \rho _ { 1 } } ( \widehat { \mathbb { P } } _ { 1 } ) } \Phi ( \phi ; \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) ,\tag{SDRHT}
$$

where the ambiguity sets are constructed by the Sinkhorn discrepancy $S _ { \epsilon } ( \widehat { \mathbb { P } } _ { k } , \mathbb { P } )$ , ofering a flexible, non-parametric framework to characterize the least-favorable distributions

$$
\mathbb { B } _ { \rho _ { k } } ( \widehat { \mathbb { P } } _ { k } ) : = \{ \mathbb { P } : { \cal S } _ { \epsilon } ( \widehat { \mathbb { P } } _ { k } , \mathbb { P } ) \leq \rho _ { k } \} , \quad \forall k \in \mathbb { K } .\tag{1}
$$

We also consider the following soft-constrained counterpart of problem (SDRHT):

$$
\operatorname* { i n f } _ { \phi \in \mathcal { F } } \operatorname* { s u p } _ { \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } \in \mathcal { P } ( \Omega ) } \quad \Phi ( \phi ; \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) - \sum _ { k \in \mathbb { K } } \lambda _ { k } S _ { \epsilon } ( \widehat { \mathbb { P } } _ { k } , \mathbb { P } _ { k } ) ,\tag{soft-SDRHT}
$$

where $\lambda _ { k } > 0$ for all $k \in \mathbb { K }$ are penalty parameters.

## 2.1 Strong Duality

In this subsection, since the optimal detector admits a closed form for fixed distributions, we prove the strong duality of problems (SDRHT) and (soft-SDRHT) by interchanging the minimization over detectors and the maximization over distributions. Note that the functional space $\mathcal { F }$ is infinite-dimensional and naturally convex, but is generally not compact. Hence, the classical Sion’s minimax theorem cannot be used to verify strong duality. Instead, we prove strong duality primarily using the lopsided minimax theorem [7], which does not require $\mathcal { F }$ to be compact.

For the transport cost and Sinkhorn discrepancy, we introduce the following assumptions.

Assumption 1. (I) The transport cost function $c ( { \pmb x } , z )$ is $\widehat { \mathbb { P } } \otimes$ ν-measurable and satisfies $\nu \{ z : 0 \leq$ $c ( { \pmb x } , z ) < \infty \} = 1 f o r \ \widehat { \mathbb { P } } \mathrm { - } a l m o s t \ e v e r y \ x .$

(II) $\mathbb { E } _ { z \sim \nu } \big [ e ^ { - c ( { \pmb x } , z ) / \epsilon } \big ] < \infty$ for ${ \widehat { \mathbb { P } } } .$ almost every x.

(III) The cost function $c ( { \boldsymbol { x } } , z )$ is lower semicontinuous in $( x , z )$

(IV) For every joint distribution $\gamma$ on $\Omega \times \Omega$ with first marginal distribution P, it has a regular conditional distribution $\gamma _ { \pmb { x } } g$ iven the value of the first marginal equals x.

According to [18], Assumptions 1(I) and (II) ensure that both the Sinkhorn discrepancy and the equivalent reformulation in the following Proposition 1 are well-defined. Assumption 1(III) ensures that the Sinkhorn discrepancy is weakly lower semicontinuous. Assumption 1(IV) ensures that the joint distribution $\gamma$ can be written as $\mathrm { d } \gamma ( { \pmb x } , z ) = \mathrm { d } \widehat { \mathbb { P } } ( { \pmb x } ) \mathrm { d } \gamma _ { \pmb x }$ and the law of total expectation holds. Our choice of transport cost $\begin{array} { r } { c ( x , z ) = \frac { 1 } { 2 } \| x - z \| _ { 2 } ^ { 2 } } \end{array}$ naturally satisfies Assumption 1(III).

Define the non-negative loss function

$$
L _ { k } ( \phi , \omega ) : = \ell \Big ( ( - 1 ) ^ { k + 1 } \cdot \phi ( \omega ) \Big ) , \quad \forall k \in \mathbb { K } .\tag{2}
$$

In this paper, we focus on continuous detectors and consider the following assumptions for the loss function.

Assumption 2 (Loss Function). (I) For any $k \in \mathbb { K } , \phi \in \mathcal { F }$ , and $\beta _ { k } > 0$ , the reference distribution $\mathbb { P } _ { k } ^ { \epsilon }$ in Lemma 1 satisfies $\mathbb { E } _ { \omega \sim \mathbb { P } _ { \varepsilon } ^ { \epsilon } } [ \exp ( \beta _ { k } \cdot L _ { k } ( \phi , \omega ) ) ] < \infty$

(II) There exists $\mathbb { Q } \in \mathcal { P } ( \Omega )$ with in $\begin{array} { r } { \mathbf { f } _ { \phi \in \mathcal { F } } \mathbb { E } _ { \omega \sim \mathbb { Q } } \Big [ L _ { k } ( \phi , \omega ) - \lambda _ { k } S _ { \epsilon } ( \widehat { \mathbb { P } } _ { k } , \mathbb { Q } ) \Big ] > - \infty f o r \ a n y \ k \in \mathbb { K } . } \end{array}$

Assumption 2(I) leads to the following Lemma 2. Assumption 2(II) ensures that the optimal value of the dual of problem (soft-SDRHT) is strictly larger than −∞.

The following proposition provides an equivalent conditional-KL-divergence representation of the Sinkhorn-based ambiguity set.

Proposition 1 (Conditional-KL Representation of the Sinkhorn Ambiguity $\mathrm { S e t } )$ . The ambiguity sets based on Sinkhorn discrepancy with $\epsilon > 0$ are equivalent to

$$
\mathbb { B } _ { \bar { \rho } _ { k } } ( \mathbb { P } _ { k } ^ { \epsilon } ) : = \left\{ \mathbb { Q } : \operatorname* { i n f } _ { \gamma \in \Gamma ( \widehat { \mathbb { P } } _ { k } , \mathbb { Q } ) } \mathbb { E } _ { \mathbf { x } \sim \widehat { \mathbb { P } } _ { k } } \Big [ \mathcal { D } _ { \mathrm { K L } } ( \gamma _ { x } | | \mathcal { K } _ { \mathbf { x } , \epsilon } ) \Big ] \leq \bar { \rho } _ { k } / \epsilon \right\} , \quad \forall k \in \mathbb { K } ,\tag{3}
$$

where $\bar { \rho } _ { k } : = \rho _ { k } + \epsilon \mathbb { E } _ { \boldsymbol { x } \sim \mathbb { \widehat { P } } _ { k } } \bigg [ \log \int _ { \mathbb { R } ^ { d } } e ^ { - c ( \boldsymbol { x } , \boldsymbol { z } ) / \epsilon } \mathrm { d } \boldsymbol { z } \bigg ] ,$ probability density $\mathrm { d } \mathbb { P } _ { k } ^ { \epsilon } ( z ) : = \mathbb { E } _ { \pmb { x } \sim \widehat { \mathbb { P } } _ { k } } \left[ \mathrm { d } \mathcal { K } _ { \pmb { x } , \epsilon } ( z ) \right]$

$$
\mathrm { d } K _ { \boldsymbol { x } , \epsilon } ( z ) : = \frac { e ^ { - c ( \boldsymbol { x } , z ) / \epsilon } } { \int _ { \mathbb { R } ^ { d } } e ^ { - c ( \boldsymbol { x } , \boldsymbol { u } ) / \epsilon } \cdot \mathrm { d } \boldsymbol { u } } \cdot \mathrm { d } \boldsymbol { v } ( z ) .
$$

The proof of Proposition 1 is provided in Appendix B.1. Proposition 1 shows the connection between the ambiguity set based on Sinkhorn discrepancy and those based on the KL divergence [12]. Since the reference measure $\nu \in \mathcal { M } ( \Omega ) , \mathcal { K } _ { x , \epsilon }$ is a probability kernel, and its corresponding continuous reference distributions in Eq. (3) are probability measures, i.e.,

$$
\mathbb { P } _ { k } ^ { \epsilon } = \frac { 1 } { n _ { k } } \sum _ { i = 1 } ^ { n _ { k } } { K } _ { { \pmb x } _ { k } ^ { ( i ) } , \epsilon } \in \mathcal { P } ( \Omega ) , \qquad \forall k \in \mathbb { K } .\tag{4}
$$

When $\Omega = \mathbb { R } ^ { d }$ , ν is a Lebesgue measure, and the transport cost is defined as $\begin{array} { r } { \frac { 1 } { 2 } \| \pmb { x } - \pmb { z } \| _ { 2 } ^ { 2 } . } \end{array}$ , distribution $\mathbb { P } _ { k } ^ { \epsilon }$ is a Gaussian mixture because $K _ { \mathbf { x } _ { t } ^ { ( i ) } , \epsilon } = \mathcal { N } ( \mathbf { x } _ { k } ^ { ( i ) } , \epsilon \cdot \pmb { I } _ { d } )$ for each $k \in \mathbb { K }$ and $i \in \left[ n _ { k } \right]$

Based on Assumptions 1 and 2 and Proposition 1, the following strong duality results hold.

Theorem 1 (Strong Duality). For the convex riskfunction Φ and Sinkhorn discrepancy ${ \mathcal { S } } _ { \epsilon } ,$ , the following results hold:

(I) If Assumptions 1 and 2(I) hold, then

$$
\mathrm { O p t i m a l ~ V a l u e ~ o f ~ ( S D R H T ) = \underbrace { m a x } _ { \mathbb { P } _ { 0 } \in \mathbb { B } _ { \rho _ { 0 } } ( \widehat { \mathbb { P } } _ { 0 } ) , \mathbb { P } _ { 1 } \in \mathbb { B } _ { \rho _ { 1 } } ( \widehat { \mathbb { P } } _ { 1 } ) } ~ \operatorname* { i n f } _ { \phi \in \mathcal { F } } ~ \Phi ( \phi ; \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) } ;
$$

(II) If Assumptions 1 and 2 hold, then

$$
\mathrm { O p t i m a l \ V a l u e ~ o f ~ ( s o f t - S D R H T ) = \operatorname* { m a x } _ { \mathbb P _ 0 , \mathbb P _ 1 \leqslant \mathcal \mathcal \mathcal \phi ( \Omega ) } ~ i n f ~ } \quad \Phi ( \boldsymbol \phi ; \mathbb P _ 0 , \mathbb P _ 1 ) - \sum _ { k \in \mathbb K } \lambda _ { k } \mathcal S _ { \epsilon } ( \widehat { \mathbb P } _  k , \mathbb P _ { k } ) ,
$$

where $\lambda _ { k } > 0$ for any $k \in \mathbb { K } .$

The proof of Theorem 1 is based on the lopsided minimax theorem [7, Theorem 5.15], and is provided in Appendix B.2. We prove this by showing that the Sinkhorn-based ambiguity sets are weakly compact and the expected loss is weakly upper semicontinuous.

Theorem 1 allows us to eliminate the detector minimization before optimizing over the leastfavorable distributions. Thus, for generating function $\ell ( t ) = \exp ( t )$ , problem (soft-SDRHT) is equivalent to

$$
\operatorname* { m a x } _ { \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } \in \mathcal { P } ( \Omega ) } \quad \Phi ( \phi _ { \mathrm { o p t } } ; \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) - \sum _ { k \in \mathbb { K } } \lambda _ { k } \mathcal { S } _ { \epsilon } ( \widehat { \mathbb { P } } _ { k } , \mathbb { P } _ { k } ) .\tag{5}
$$

where

$$
\Phi ( \phi _ { \mathrm { o p t } } ; \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) = \sum _ { k \in \mathbb { K } } \mathbb { E } _ { \omega \sim \mathbb { P } _ { k } } \Big [ \exp \Big [ ( - 1 ) ^ { k + 1 } \cdot \frac { 1 } { 2 } \Big ( \log p _ { 0 } ( \omega ) - \log p _ { 1 } ( \omega ) \Big ) \Big ] \Big ] .\tag{6}
$$

For a testing sample $\omega \in \Omega$ , it sufices to evaluate the log-density of least-favorable distributions $\mathbb { P } _ { 0 }$ and $\mathbb { P } _ { 1 }$ at $\omega ,$ and then make a decision by our explicitly optimal detector. Thus, the robust detector retains the likelihood-ratio form, but its densities are those of the least-favorable distributions. Problem (5) nevertheless remains infinite-dimensional because the least-favorable distributions are still optimization variables. In the next subsection, we characterize these distributions using parameterized convex-gradient transport maps.

## 2.2 Reformulation via Convex-Gradient Transport Maps

By setting the transport cost to the quadratic function $\begin{array} { r } { c ( \pmb { x } , z ) = \frac { 1 } { 2 } \| \pmb { x } - z \| _ { 2 } ^ { 2 } . } \end{array}$ , we can leverage the following fundamental result from optimal transport [32] to simplify Problem (5).

Theorem 2 (Brenier’s Theorem). Let $\mu , \nu$ be two finite-second-moment probability measures on $\mathbb { R } ^ { d }$ and assume that $\mu$ is absolutely continuous with respect to the Lebesgue measure. Then there exists a proper convex function $\varphi : \mathbb { R } ^ { d } \to \mathbb { R }$ such that $( \nabla \varphi ) _ { \# } \mu = \nu$ . In addition, $\nabla \varphi$ is the unique solution to

$$
\operatorname* { i n f } _ { T : \ T _ { \# } \mu = \nu } \mathbb { E } _ { \mu } \| \pmb { x } - T ( \pmb { x } ) \| _ { 2 } ^ { 2 } ,
$$

and $\gamma _ { o p t } \sim ( \mathrm { I d } , \nabla \varphi ) _ { \# } \mu$ is the unique solution to

$$
\operatorname* { i n f } _ { \gamma \in \Gamma ( \mu , \nu ) } \mathbb { E } _ { \gamma } \| \pmb { x } - \pmb { y } \| _ { 2 } ^ { 2 } .
$$

Theorem 2 shows that the gradient of a convex potential can characterize the transport map between continuous distributions. Proposition 1 and Theorem 2 enable us to reformulate (5) as an equivalent optimization problem over convex gradient transport maps, i.e.,

$$
\operatorname* { m a x } _ { \varphi _ { 0 } , \varphi _ { 1 } \in \mathcal { F } _ { \mathrm { C V X } } } \left\{ \Phi ( \phi _ { \mathrm { o p t } } ; \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) - \sum _ { k \in \mathbb { K } } \lambda _ { k } \mathcal { S } _ { \epsilon } ( \widehat { \mathbb { P } } _ { k } , \mathbb { P } _ { k } ) : \ \mathbb { P } _ { k } = ( \nabla \varphi _ { k } ) _ { \# } \mathbb { P } _ { k } ^ { \epsilon } \right\} ,\tag{7}
$$

where $\mathcal { F } _ { \mathrm { C V X } }$ is the set of convex functions $\varphi : \mathbb { R } ^ { d }  \mathbb { R } \cup \{ + \infty \}$ and the distribution $\mathbb { P } _ { k } ^ { \epsilon }$ for each $k \in \mathbb { K }$ is defined in Eq. (4). Note that we use continuous distributions $\mathbb { P } _ { k } ^ { \epsilon }$ as source distributions of the transport maps for all $k \in \mathbb { K }$ , thanks to the connection between Sinkhorn discrepancy and KL divergence shown in Proposition 1.

For computation, we restrict the convex function $\varphi _ { k } \in C ^ { 2 } ( \mathbb { R } ^ { d } )$ to be strongly convex, and $\nabla \varphi _ { k }$ to be a global difeomorphism for each $k \in \mathbb { K }$ . In formulation (7), given strongly convex functions $\varphi _ { 0 }$ and $\varphi _ { 1 }$ , the log-density terms in risk function $\Phi ( \phi _ { \mathrm { o p t } } ; \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } )$ can be explicitly computed by the log-density form of the change-of-variable formula [23]. Specifically, for any sample $\omega _ { k } \sim \mathbb { P } _ { k }$ we have

$$
\begin{array} { r } { \log p _ { k } ( \omega _ { k } ) = \log p _ { k } ^ { \epsilon } ( z _ { k } ) - \log \Big | \operatorname* { d e t } \nabla ^ { 2 } \varphi _ { k } ( z _ { k } ) \Big | , \quad \forall k \in \mathbb { K } , } \end{array}\tag{8}
$$

where $z _ { k } = \nabla \varphi _ { k } ^ { - 1 } ( \omega _ { k } )$ , and $p _ { k } ^ { \epsilon }$ represents the probability density function of $\mathbb { P } _ { k } ^ { \epsilon }$ for each $k \in \mathbb { K }$ Based on the Fenchel–Young inequality, the gradient of a diferentiable strongly convex function $\varphi$ is the inverse of the gradient of its convex conjugate, i.e., $\nabla \varphi ^ { - 1 } = \nabla \varphi ^ { * }$ . In this case, we have $z _ { k } = \arg \operatorname* { m a x } _ { z \in \mathbb { R } ^ { d } } \left\{ \omega _ { k } ^ { \top } z - \varphi _ { k } ( z ) \right\}$ . Similarly, to evaluate the log-density of sample $\omega _ { k } \sim \mathbb { P } _ { k }$ at the other distribution $\mathbb { P } _ { | 1 - k | } ^ { \epsilon } ,$ i.e., log $p _ { | 1 - k | } ( \omega _ { k } )$ , we trace $\omega _ { k }$ back to its corresponding source sample $\bar { z } _ { k }$ in $\mathbb { P } _ { | 1 - k | } ^ { \epsilon } \colon$

$$
\bar { z } _ { k } = \nabla \varphi _ { | 1 - k | } ^ { - 1 } ( \omega _ { k } ) = \arg \operatorname* { m a x } _ { z \in \mathbb { R } ^ { d } } \bigg \{ \omega _ { k } ^ { \top } z - \varphi _ { | 1 - k | } ( z ) \bigg \} , \quad \forall k \in \mathbb { K } ,\tag{9}
$$

and then we have

$$
\log p _ { | 1 - k | } ( \omega _ { k } ) = \log p _ { | 1 - k | } ^ { \epsilon } ( \bar { z } _ { k } ) - \log \Big | \operatorname* { d e t } \nabla ^ { 2 } \varphi _ { | 1 - k | } ( \bar { z } _ { k } ) \Big | , \quad \forall k \in \mathbb { K } .\tag{10}
$$

Since the potential $\varphi _ { k }$ in (7) is assumed to be twice diferentiable and strongly convex, its Hessian determinant is strictly positive, i.e., $\left| \operatorname* { d e t } \nabla ^ { 2 } \varphi _ { k } ( z ) \right| = \operatorname* { d e t } \nabla ^ { 2 } \varphi _ { k } ( z )$ , and the log-determinant operators in Eqs. (8) and (10) are well defined. In high-dimensional applications, the exact values of the log-determinant-Hessian operators are computationally expensive, but can be estimated eficiently by the stochastic Lanczos quadrature (SLQ) in [33, 34].

Although formulation (7) exposes the transport structure of the least-favorable distributions, it remains an infinite-dimensional optimization. Moreover, the induced densities, inverse maps, and log-determinant terms are generally unavailable in closed form. However, this formulation ofers a new perspective on generating least-favorable distributions by characterizing the gradients of convex potentials. In the next section, we propose a generative method to solve (7).

## 3 A HyCNN-Based Generative Method

Flow-based generative models represent a target distribution as the pushforward of a simple reference distribution through an invertible, diferentiable map, allowing eficient density evaluation and sample generation [23]. In flow-based DRO, these models can also generate least-favorable distributions when guided by proper loss functions [26]. For problem (7), a natural approach is to parameterize the convex potentials by the Input Convex Neural Network (ICNN) [34, 35]. In this section, we develop a generative framework that parametrizes convex potentials by a variant of ICNN, i.e., Hyper Input Convex Neural Network (HyCNN), extending the classical ICNN with more flexible architectures [36].

## 3.1 The Architecture of HyCNN

In this subsection, we first introduce the HyCNN architecture. Let the number of layers of HyCNN be L, and the output of HyCNN given an input $\pmb { x } \in \mathbb { R } ^ { d }$ be $\mathsf { H y C N N } ( \pmb { x } ) \in \mathbb { R }$ . Figure 2 illustrates the architecture of HyCNN, including the detailed structure of each block. The forward propagation process of HyCNN can be reformulated as follows

$$
\mathsf { H y C N N } ( \boldsymbol { x } ) : = { \boldsymbol { y } } _ { L + 1 } = { \boldsymbol { V } } _ { L } { \boldsymbol { y } } _ { L } + { \boldsymbol { W } } _ { L } { \boldsymbol { x } } + { \boldsymbol { b } } _ { L } ,\tag{11a}
$$

$$
\begin{array} { r } { \pmb { y } _ { l + 1 } : = \bar { \sigma } \Big ( \pmb { V } _ { l } ^ { ( 1 ) } \pmb { y } _ { l } + \pmb { W } _ { l } ^ { ( 1 ) } \pmb { x } + \pmb { b } _ { l } ^ { ( 1 ) } , \pmb { V } _ { l } ^ { ( 2 ) } \pmb { y } _ { l } + \pmb { W } _ { l } ^ { ( 2 ) } \pmb { x } + \pmb { b } _ { l } ^ { ( 2 ) } \Big ) , \quad \forall l \in \{ 0 , \cdots , L - 1 \} . } \end{array}\tag{11b}
$$

where $\{ V , W , b \}$ are trainable parameters, $\mathbf { \nabla } y _ { 0 } : = \mathbf { 0 } ,$ , and $\bar { \sigma } : \mathbb { R } ^ { 2 } $ R is a scalar activation function that is convex and component-wise non-decreasing, applied element-wise to vector inputs.

As shown in Figure 2, in each hidden layer $l \geq 1$ of HyCNN, a “passthrough” layer (red arrow) directly connects the input vector with the hidden units. Since the nonnegative sums of convex functions are also convex and the composition of a convex and convex non-decreasing function is also convex, according to Proposition 2.1 in [36], this construction ensures the convexity of HyCNNs.

Proposition 2 (Convexity of HyCNN [36]). Suppose that $\bar { \boldsymbol { \sigma } } : { \mathbb { R } ^ { 2 } }  { \mathbb { R } }$ is convex and component-wise non-decreasing, and weight matrices $\{ V _ { l } ^ { ( 1 ) } , V _ { l } ^ { ( 2 ) } \} _ { l = 0 } ^ { L - 1 }$ and $V _ { L }$ are component-wise nonnegative. Then, HyCNN(x) in Eq. (11a) is convex in x.

![](images/a29d200c7707260999b45ee2d1dfe6c6da7fa032eb4ecbdf778a5bc445ef622b.jpg)  
Figure 2: The architecture of HyCNN

In this paper, the non-negativity of weight matrices $\{ V _ { l } ^ { ( 1 ) } , V _ { l } ^ { ( 2 ) } \} _ { l = 0 } ^ { L - 1 }$ and $V _ { L }$ is enforced by softplus operators. Note that when we choose $\bar { \sigma } ( a _ { 1 } , a _ { 2 } ) = \sigma ( a _ { 1 } )$ , where $\sigma : \mathbb { R }  \mathbb { R }$ is an activation function that is convex and non-decreasing (e.g., ReLU), the HyCNN reduces to the standard ICNN architecture [35]. Thus, HyCNNs can be interpreted as a generalization of ICNNs. When we choose a maxout unit [37] as our activation function, i.e., $\bar { \sigma } ( a _ { 1 } , a _ { 2 } ) = \operatorname* { m a x } ( a _ { 1 } , a _ { 2 } )$ , the resulting HyCNN is piecewise-afine but is nondiferentiable at ties. Therefore, to ensure the trained HyCNN is diferentiable, we use a smooth log-sum-exp approximation of the maxout activation function with parameter $\tau > 0 { : }$

$$
\begin{array} { r } { \bar { \sigma } _ { \tau } ( a _ { 1 } , a _ { 2 } ) : = \tau \cdot \log \Big ( e ^ { a _ { 1 } / \tau } + e ^ { a _ { 2 } / \tau } \Big ) , } \end{array}\tag{12}
$$

which is referred to as the smooth maxout activation function below.

## 3.2 Approximation for Convex Potentials

In this subsection, we present the procedure for training HyCNNs. Given sample $z \in \Omega$ , denote the two HyCNNs we use for the problem (7) by $\mathsf { H y C N N } _ { 0 } ( z )$ and $\mathsf { H y C N N } _ { 1 } ( z )$ , respectively. Denote the approximated convex function by $\widetilde { \varphi _ { k } }$ for each $k \in \mathbb { K }$ . To ensure the gradient of convex potentials is invertible, as the smooth maxout activation makes ${ \mathrm { H y C N N } } _ { k }$ twice continuously diferentiable, we further add a positive quadratic term to ensure strong convexity, i.e., $\begin{array} { r } { \widetilde { \varphi } _ { k } ( z ) = \mathsf { H y C N N } _ { k } ( z ) + \frac { \alpha } { 2 } \| z \| _ { 2 } ^ { 2 } } \end{array}$ where α is a suficient small positive number, such that the Hessian matrix $H _ { \mathcal { \widetilde { S } } _ { k } } \succeq \alpha I > 0$ . Unlike block-wise progressive methods that train a series of maps in [26], we directly learn one convex-gradient map for each hypothesis, since a composition of convexgradient maps may not be the gradient of a convex potential.

The training process is given by the following Algorithm 1. For the gradient of the objective function, in Step 5, we propose two N-sample estimators to approximate the gradient of the risk function and the Sinkhorn discrepancy, respectively; see Algorithms 3 and 4 in the next subsection for details. As shown, to generate the samples for estimators, we first draw N samples from the continuous reference distribution $\mathbb { P } _ { k } ^ { \epsilon }$ (Step 2), and then map them to the target distribution space by the gradient of trained convex potentials (Step 3). In Step $^ { 6 , }$ the stochastic gradient ascent algorithm is used to update the network parameters.

Algorithm 1 Convex Potentials Approximation based on HyCNNs.   
Require: Number of epochs T, training samples $\{ \pmb { x } _ { k } ^ { ( i ) } \} _ { i = 1 } ^ { n _ { k } } ,$ , reference distributions $\mathbb { P } _ { k , 0 } = \mathbb { P } _ { k } ^ { \epsilon } .$   
Output: Convex potential approximations $\widetilde { \varphi } _ { k , T }$ for all $k \in \mathbb { K } .$   
1: Initialize strongly convex potential $\widetilde { \varphi } _ { k , 0 }$ with parameters $\pmb { \theta } _ { k , 0 } \in \Theta$ for each $k \in \mathbb { K } ;$   
2: for $t = 0 , \cdots , T - 1$ do   
3: Draw N i.i.d. samples $\{ z _ { k } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ from initial distribution $\mathbb { P } _ { k , 0 }$ for each $k \in \mathbb { K } ;$   
4: Generate samples $\{ \omega _ { k , t } ^ { ( i ) } \} _ { i = 1 } ^ { N }  \{ \nabla \widetilde { \varphi } _ { k , t } ( z _ { k } ^ { ( i ) } ) \} _ { i = 1 } ^ { N }$ in target distribution $\mathbb { P } _ { k , t }$ for each $k \in \mathbb { K } ;$   
5: Estimate the gradient of objective function of (7) by Algorithms 3 and 4:   
$\nabla _ { \pmb { \theta } _ { k , t } } F ( \pmb { \theta } _ { 0 , t } , \pmb { \theta } _ { 1 , t } )$ ← Risk Gradient Estimator $( \{ \omega _ { 0 } ^ { ( i ) } \} _ { i = 1 } ^ { N } , \{ \omega _ { 1 } ^ { ( i ) } \} _ { i = 1 } ^ { N } , \widetilde { \varphi } _ { 0 , t } , \widetilde { \varphi } _ { 1 , t } , \pmb { \theta } _ { k , t } )$   
$- \nabla _ { \pmb { \theta } _ { k , t } } \sum _ { k \in \mathbb { K } } \lambda _ { k }$ · Sinkhorn Estimator $( \{ \pmb { x } _ { k } ^ { ( i ) } \} _ { i = 1 } ^ { n _ { k } } , \{ \pmb { \omega } _ { k , t } ^ { ( i ) } \} _ { i = 1 } ^ { N } ) ,$ $\forall k \in \mathbb { K } ;$   
6: Update $\pmb { \theta } _ { k , t + 1 }  \pmb { \theta } _ { k , t } + \eta _ { t } \cdot \nabla _ { \pmb { \theta } _ { k , t } } F ( \pmb { \theta } _ { 0 , t } , \pmb { \theta } _ { 1 , t } )$ , where $\eta _ { t }$ is the learning rate;   
7: end for

## 3.3 Stochastic Gradient Estimation

Next, we develop stochastic estimators required by Algorithm 1 to solve Problem (7). Let $\{ z _ { 0 } ^ { ( i ) } \} _ { i = 1 } ^ { N }$   
and $\{ z _ { 1 } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ be N independent and identically distributed (i.i.d.) samples from reference distribu  
tions $\mathbb { P } _ { 0 } ^ { \epsilon }$ and $\mathbb { P } _ { 1 } ^ { \epsilon }$ (see Eq. 4), respectively. These points are subsequently transformed into samples   
from the target distributions $\mathbb { P } _ { 0 }$ and $\mathbb { P } _ { 1 }$ via transport maps, i.e., $\omega _ { 0 } = \nabla \varphi _ { 0 } ( z _ { 0 } )$ and $\omega _ { 1 } = \nabla \varphi _ { 1 } ( z _ { 1 } )$   
where the underlying distributions satisfy $\mathbb { P } _ { 0 } = [ \nabla \varphi _ { 0 } ] _ { \# } \mathbb { P } _ { 0 } ^ { \epsilon }$ and $\mathbb { P } _ { 1 } = [ \nabla \varphi _ { 1 } ] _ { \# } \mathbb { P } _ { 1 } ^ { \epsilon }$

## 3.3.1 Risk Gradient Estimator

In this part, we present the gradient estimator of the risk function with respect to the parameters of HyCNNs in Algorithm 1. For every $k \in \mathbb { K }$ and $\omega _ { k } \sim \mathbb { P } _ { k }$ , we define the basic element of the risk function as

$$
h _ { k } ( \omega _ { k } , \theta _ { 0 } , \theta _ { 1 } ) : = \exp \bigg [ \frac { 1 } { 2 } ( \log p _ { | 1 - k | } ( \omega _ { k } ) - \log p _ { k } ( \omega _ { k } ) ) \bigg ] , \quad \forall k \in \mathbb { K } .
$$

As an example, given convex functions $\widetilde { \varphi } _ { 0 }$ and $\widetilde { \varphi } _ { 1 }$ with parameters $\theta _ { 0 }$ and $\pmb { \theta } _ { 1 }$ , respectively, the gradients of $h _ { 0 } ( \omega _ { 0 } , \pmb { \theta } _ { 0 } , \pmb { \theta } _ { 1 } )$ with respect to $\pmb { \theta } _ { 0 }$ and $\pmb { \theta } _ { 1 }$ are given by

$$
\frac { \partial h _ { 0 } ( \omega _ { 0 } , \theta _ { 0 } , \theta _ { 1 } ) } { \partial \theta _ { 0 } } = \frac { h _ { 0 } ( \omega _ { 0 } , \theta _ { 0 } , \theta _ { 1 } ) } { 2 } \cdot \bigg [ \bigg ( \frac { \partial \omega _ { 0 } } { \partial \theta _ { 0 } } \bigg ) ^ { \top } [ \nabla ^ { 2 } \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) ] ^ { - 1 } g + \frac { \partial } { \partial \theta _ { 0 } } \log \operatorname* { d e t } \nabla ^ { 2 } \widetilde { \varphi } _ { 0 } ( z _ { 0 } ) \bigg ] ,\tag{13a}
$$

$$
\frac { \partial h _ { 0 } ( \omega _ { 0 } , \theta _ { 0 } , \theta _ { 1 } ) } { \partial \theta _ { 1 } } = - \frac { h _ { 0 } ( \omega _ { 0 } , \theta _ { 0 } , \theta _ { 1 } ) } { 2 } \cdot \bigg [ \bigg ( \frac { \partial \nabla \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) } { \partial \theta _ { 1 } } \bigg ) ^ { \top } [ \nabla ^ { 2 } \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) ] ^ { - 1 } g + \frac { \partial } { \partial \theta _ { 1 } } \log \operatorname* { d e t } \nabla ^ { 2 } \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) \bigg ] ,\tag{13b}
$$

where $\begin{array} { r } { \pmb { g } : = \frac { \partial \log p _ { 1 } ( \omega _ { 0 } ) } { \partial \bar { z } _ { 0 } } = \frac { \partial [ \log p _ { 1 } ^ { \epsilon } ( \bar { z } _ { 0 } ) - \log \operatorname* { d e t } \nabla ^ { 2 } \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) ] } { \partial \bar { z } _ { 0 } } } \end{array}$ , and $\bar { z } _ { k }$ is defined in Eq. (9) for each $k \in \mathbb { K } .$ . For brevity, the detailed gradient derivation can be found in Appendix C.1. Eqs. (13a) and (13a) indicate that directly diferentiating the risk requires backpropagation through inverse maps and the log-determinant-Hessian (LDH), which is computationally expensive. Therefore, in the following, we develop a matrix-free stochastic gradient estimator for the LDH in Algorithm 2, and then present the estimator for the risk function in Algorithm 3.

For the gradient of LDH, we first introduce an O(1)-memory estimator for the gradient of the log-determinant [34]. Specifically, for any invertible matrix H with parameters θ:

$$
\frac { \partial } { \partial \pmb { \theta } } \log \operatorname* { d e t } H = \mathrm { t r } ( H ^ { - 1 } \frac { \partial H } { \partial \pmb { \theta } } ) = \mathbb { E } _ { \pmb { v } } [ \pmb { v } ^ { \top } H ^ { - 1 } \frac { \partial H } { \partial \pmb { \theta } } \pmb { v } ] ,\tag{14}
$$

where the last equality uses the Hutchinson trace estimator [38] with an isotropic random vector (e.g., Rademacher or standard Gaussian) $\pmb { v } \in \mathbb { R } ^ { d }$ satisfying $\mathbb { E } [ \pmb { v } ] = 0$ and $\mathbb { E } [ v v ^ { \top } ] = I .$ . With exact linear solves, the Hutchinson construction ofers an unbiased estimator of the gradient of the logdeterminant. In this problem, H is a symmetric positive definite Hessian matrix. Thus, $v ^ { \top } H ^ { - 1 }$ in Eq. (14) can be eficiently solved by the following quadratic optimization problem

$$
\operatorname* { m i n } _ { \pmb { u } } \quad \frac { 1 } { 2 } \pmb { u } ^ { \top } \pmb { H } \pmb { u } - \pmb { v } ^ { \top } \pmb { u } ,\tag{15}
$$

which has the unique minimizer $\pmb { u } _ { \mathrm { o p t } } = H ^ { - 1 } \pmb { v } = ( \pmb { v } ^ { \top } H ^ { - 1 } ) ^ { \top }$ . Then, the gradient of the LDH in our problem can be estimated as $\begin{array} { r } { \mathbb { E } _ { v } [ \frac { \partial } { \partial \pmb { \theta } } ( \pmb { u } _ { \mathrm { o p t } } ^ { \top } H \pmb { v } ) ] } \end{array}$ . In this paper, we summarize this procedure as Algorithm 2.

```latex
Algorithm 2 GLDH Estimator: Estimator for the Gradient of LDH.
Require: Symmetric positive definite matrix $H ,$ parameter $\theta ,$ and sample size M.
1: Initialize estimator $\pmb { { \cal E } } = \mathbf { 0 }$ with the same dimension as $\pmb \theta ;$
2: for $m = 1 , \cdots , M$ do
3: Sample Rademacher random vector ${ \pmb v } ^ { ( m ) }$ with d dimensions;
4: ${ \mathbf { } } { \mathbf { } } u ^ { ( m ) } \gets$ ConjugateGradient $( H , \pmb { v } ^ { ( m ) } ) ;$
5: $\overline { { { \pmb u } } } ^ { ( m ) } \gets$ StopGradient $\left( \pmb { u } ^ { ( m ) } \right)$
6: $E \gets E + \nabla _ { \pmb { \theta } } \bigg [ \left( \overline { { \pmb { u } } } ^ { ( m ) } \right) ^ { \top } H \pmb { v } ^ { ( m ) } \bigg ] ;$
7: end for
8: Return E/M.
```

In Step 4 of Algorithm 2, the function ConjugateGradient $( H , { \pmb v } ) : = \arg \operatorname* { m i n } _ { \pmb u } \frac { 1 } { 2 } { \pmb u } ^ { \top } H { \pmb u } - { \pmb v } ^ { \top } { \pmb u }$ l represents that we use the conjugate gradient method to solve the unconstrained optimization problem (15). In Step 5, StopGradient(·) treats its argument as a constant during backpropagation. Thus, in Step 6, only $H ( \pmb \theta )$ is diferentiated, and the term $\nabla _ { \pmb { \theta } } [ \left( \overline { { \pmb { u } } } ^ { ( m ) } \right) ^ { \top } H \pmb { v } ^ { ( m ) } ]$ can be eficiently computed by Hessian-vector and vector-Jacobian products.

Then, the gradient of the risk function (6) can be estimated by Algorithm 3. Both functions ConjugateGradient in Step 5 and GLDH Estimator have been defined in Algorithm 2. In Step 6, we compute the value of the log-determinant in $h _ { k } ( \omega _ { k } , \pmb { \theta } _ { 0 } , \pmb { \theta } _ { 1 } )$ exactly when $d \leq 3 2$ and estimate it by the stochastic Lanczos quadrature (SLQ) [33, 34] when $d > 3 2$

Algorithm 3 Risk Gradient Estimator: Gradient Estimator for the Risk Function.   
Require: Generated samples $\{ \omega _ { k } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ and convex potentials $\widetilde { \varphi _ { k } }$ for all $k \in \mathbb { K }$ , parameter $\pmb { \theta } .$   
1: Initialize estimator $\pmb { { \cal E } } = \mathbf { 0 }$ with the same dimension as $\pmb \theta ;$   
2: for $k = 0 , 1$ do   
3: for $i = 1 , \cdots , N$ do   
4: Compute $z _ { k } ^ { ( i ) } = \nabla \widetilde { \varphi } _ { k } ^ { - 1 } ( \omega _ { k } ^ { ( i ) } )$ and $\bar { z } _ { k } ^ { ( i ) } = \nabla \widetilde { \varphi } _ { | 1 - k | } ^ { - 1 } ( \omega _ { k } ^ { ( i ) } )$ by Eq. (9);   
5: Compute $\boldsymbol { u } _ { k } ^ { ( i ) }$ ← ConjugateGradient $\begin{array} { r } { \cdot \Big ( \nabla ^ { 2 } \widetilde { \varphi } _ { | 1 - k | } ( \bar { z } _ { k } ^ { ( i ) } ) , \frac { \partial \log p _ { | 1 - k | } ( \omega _ { k } ^ { ( i ) } ) } { \partial \bar { z } _ { k } ^ { ( i ) } } \Big ) ; } \end{array}$   
6: Update the gradient estimator   
$E \gets E + \frac { h _ { k } ( \omega _ { k } ^ { ( i ) } , \theta _ { 0 } , \theta _ { 1 } ) } { 2 } \cdot \Bigg \{ \Bigg [ \frac { \partial \Big ( \omega _ { k } ^ { ( i ) } - \nabla \widetilde { \varphi } _ { | 1 - k | } ( \bar { z } _ { k } ^ { ( i ) } ) \Big ) } { \partial \theta } \Bigg ] ^ { \top } u _ { k } ^ { ( i ) }$   
+ GLDH Estimato $\cdot \left( \nabla ^ { 2 } \widetilde { \varphi _ { k } } ( z _ { k } ^ { ( i ) } ) , \pmb \theta \right)$ − GLDH Estimator $\left. \left. \nabla ^ { 2 } \widetilde { \varphi } _ { | 1 - k | } ( \bar { z } _ { k } ^ { ( i ) } ) , \theta \right. \right\}$   
7: end for   
8: end for   
9: Return $E / N .$

## 3.3.2 Sinkhorn Discrepancy Estimator

Computing the Sinkhorn discrepancy term $S _ { \epsilon } ( \widehat { \mathbb { P } } _ { k } , \mathbb { P } _ { k } )$ exactly (see Definition 2) in the objective function is challenging. In this part, we present a diferentiable finite-sample estimator for the Sinkhorn discrepancy. The gradient of this estimator can be computed eficiently by backpropagation and is used in Algorithm 1.

We first introduce the following proposition regarding the dual formulation of Sinkhorn discrepancy [32, 39].

Proposition 3. Let P and Q be probability measures with finite second moments. The strong dual formulation of Sinkhorn discrepancy $S _ { \epsilon } ( \mathbb { P } , \mathbb { Q } )$ is given by

$$
\operatorname* { s u p } _ { f \in L ^ { 1 } ( \mathbb { P } ) , g \in L ^ { 1 } ( \mathbb { Q } ) } \quad \left\{ \int f ( x ) \mathrm { d } \mathbb { P } ( x ) + \int g ( z ) \mathrm { d } \mathbb { Q } ( z ) - \epsilon \iint \left[ \exp \left( \frac { f ( x ) + g ( z ) - c ( x , z ) } { \epsilon } \right) - 1 \right] \mathrm { d } \mathbb { P } ( x ) \mathrm { d } \mathbb { Q } ( z ) \right\} ,
$$

Let $f _ { \epsilon }$ and $g _ { \epsilon }$ be the solutions of the dual formulation, which is also known as optimal entropic potentials. Then the transport plan

$$
\gamma _ { \epsilon } ( \mathrm { d } x , \mathrm { d } z ) = \exp \left( \frac { f _ { \epsilon } ( x ) + g _ { \epsilon } ( z ) - c ( x , z ) } { \epsilon } \right) \mathrm { d } \mathbb { P } ( x ) \mathrm { d } \mathbb { Q } ( z )
$$

is the unique solution to the Sinkhorn discrepancy, and the optimal entropic potentials satisfy

$$
\int \exp \bigg ( \frac { f _ { \epsilon } ( x ) + g _ { \epsilon } ( z ) - c ( x , z ) } { \epsilon } \bigg ) \mathrm { d } \mathbb { P } ( x ) = 1 , \quad \forall z \in \Omega ,\tag{16a}
$$

$$
\int \exp \left( \frac { f _ { \epsilon } ( x ) + g _ { \epsilon } ( z ) - c ( x , z ) } { \epsilon } \right) \mathrm { d } \mathbb { Q } ( z ) = 1 , \quad \forall x \in \Omega ,\tag{16b}
$$

for Q-almost every z and P-almost every x, respectively.

Since the target continuous distribution $\mathbb { P } _ { k }$ is computationally intractable, we approximate it as empirical distributions formed by the generated samples, denoted by $\begin{array} { r } { \widetilde { \mathbb { P } } _ { 0 } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \delta _ { \omega _ { \mathrm { 0 } } ^ { ( i ) } } } \end{array}$ and $\begin{array} { r } { \widetilde { \mathbb { P } } _ { 1 } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \delta _ { \omega _ { 1 } ^ { ( i ) } } } \end{array}$ . Therefore, it sufices to compute the value $S _ { \epsilon , ( n _ { k } , N ) } ( \widehat { \mathbb { P } } _ { k } , \widetilde { \mathbb { P } } _ { k } )$ . The optimal entropic potentials between discrete distributions $\widehat { \mathbb { P } } _ { k }$ and $\widetilde { \mathbb { P } } _ { k }$ , denoted by $f _ { \epsilon , k , ( n _ { k } , N ) }$ and $g _ { \epsilon , k , ( n _ { k } , N ) } $ , can be obtained by Sinkhorn’s algorithm [20, 39], leading to the following estimator for the barycentric projection of the optimal entropic coupling:

$$
\begin{array} { r l } & { T _ { \epsilon , k , ( n _ { k } , N ) } ( x ) = \frac { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \omega _ { k } ^ { ( i ) } e ^ { \left( g _ { \epsilon , k , ( n _ { k } , N ) } ( \omega _ { k } ^ { ( i ) } ) - c ( \pmb { x } , \omega _ { k } ^ { ( i ) } ) \right) / \epsilon } } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e ^ { \left( g _ { \epsilon , k , ( n _ { k } , N ) } ( \omega _ { k } ^ { ( i ) } ) - c ( \pmb { x } , \omega _ { k } ^ { ( i ) } ) \right) / \epsilon } } , \quad \forall k \in \mathbb { K } . } \end{array}
$$

However, the discrete dual potential $g _ { \epsilon , k , ( n _ { k } , N ) }$ is initially defined only on the generated support points and is thus not diferentiable for all samples in Ω. To make the optimized empirical Sinkhorn discrepancy diferentiable with respect to the generated sample locations, we extend g to the general continuous sample space for fixed $f \colon$

$$
g _ { \epsilon , k } ^ { c } ( \omega ) = - \epsilon \log \left( \frac { 1 } { n _ { k } } \sum _ { j = 1 } ^ { n _ { k } } \exp \Big ( \frac { f _ { k } ( x _ { k } ^ { ( j ) } ) - c ( x _ { k } ^ { ( j ) } , \omega ) } { \epsilon } \Big ) \right) , \quad \forall \omega \in \Omega .\tag{17}
$$

This extension agrees with the original potential g on the empirical support at convergence while making it diferentiable. Therefore, our diferentiable estimator for the Sinkhorn discrepancy is given by

$$
\widehat { \mathcal { S } } _ { \epsilon , ( n _ { k } , N ) } ( \widehat { \mathbb { P } } _ { k } , \mathbb { P } _ { k } ) : = \mathcal { S } _ { \epsilon , ( n _ { k } , N ) } ( \widehat { \mathbb { P } } _ { k } , \widetilde { \mathbb { P } } _ { k } ) = \frac { 1 } { n _ { k } } \sum _ { j = 1 } ^ { n _ { k } } f _ { \epsilon , k , ( n _ { k } , N ) } ( \boldsymbol { x } _ { k } ^ { ( j ) } ) + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } g _ { \epsilon , k } ^ { c } ( \boldsymbol { \omega } _ { k } ^ { ( i ) } ) ,\tag{18}
$$

and its gradient can be computed by backpropagation (see Appendix C.2).

Therefore, the pseudo-code for estimating the diferentiable Sinkhorn discrepancy is shown in Algorithm 4. Steps 1-6 in Algorithm 4 are essentially Sinkhorn’s algorithm [39], and thus the computational complexity of each iteration is $\mathcal { O } ( n _ { k } \cdot N )$ . This alternating algorithm computes the entropic potentials by iteratively updating f and $g$ while satisfying one of the two optimal marginal conditions in Eq. (16).

## 4 Representation Power of HyCNNs

This section provides theoretical analyses of the representation capacity of HyCNNs. Specifically, we focus on their representation power for both convex functions and their gradients, as well as the distributional universality of their induced transport maps.

```latex
Algorithm 4 Sinkhorn Estimator: Estimator for the Sinkhorn Discrepancy.
Require: Training samples $\{ \pmb { x } _ { k } ^ { ( i ) } \} _ { i = 1 } ^ { n _ { k } }$ from $\widehat { \mathbb { P } } _ { k }$ and generated samples $\{ \boldsymbol { \omega } _ { k } ^ { ( i ) } \} _ { i = 1 } ^ { N }$
1: Initialize $i = 0$ and $f _ { \epsilon , k } ^ { ( 0 ) } = 0$ for each $k ;$
2: while not converged do
3: update $\begin{array} { r } { g _ { \epsilon , k } ^ { ( i + 1 ) } ( \omega _ { k } ) \gets - \epsilon \log \frac { 1 } { n _ { k } } \sum _ { j = 1 } ^ { n _ { k } } \exp \Big ( \frac { f _ { \epsilon , k } ^ { ( i ) } ( x _ { k } ^ { ( j ) } ) - c ( x _ { k } ^ { ( j ) } , \omega _ { k } ) } { \epsilon } \Big ) } \end{array}$ for each $k \in \mathbb { K } ;$
4: update $f _ { \epsilon , k } ^ { ( i + 1 ) } ( { \pmb x } _ { k } )$ ← −ϵ log $\begin{array} { r } { \frac { 1 } { N } \sum _ { j = 1 } ^ { N } } \end{array}$ exp $\left( \frac { g _ { \epsilon , k } ^ { ( i + 1 ) } ( \omega _ { k } ^ { ( j ) } ) - c ( \alpha _ { k } , \omega _ { k } ^ { ( j ) } ) } { \epsilon } \right)$ for each $k \in \mathbb { K } ;$
5: $i \gets i + 1 ;$
6: end while
7: Extend the entropic potential $g _ { \epsilon , k } ^ { ( i + 1 ) } ( \omega _ { k } )$ to continuous space by Eq. (17);
8: Return $\widetilde { S } _ { \epsilon , ( n _ { k } , N ) } ( \widehat { \mathbb { P } } _ { k } , \mathbb { P } _ { k , t } )$ computed by Eq. (18).
```

We refer to HyCNNs equipped with smooth maxout activation functions $\bar { \sigma } _ { \tau }$ (see Eq. (12)) as smooth HyCNNs. We first show the representation power of both standard and smooth HyCNNs. Let L and h denote the number of layers and the width at each hidden layer of HyCNNs, respectively. Since we take the strongly convex coeficient $\alpha  0$ , these results hold in our strongly convex setting.

In the following results, we take the strongly convex coeficient $\alpha \to 0$

Theorem 3 (Representation Power of (Smooth) HyCNNs). For any $L _ { f } – L i p s c h i t z$ convex function $f : \mathbb { R } ^ { d }  \mathbb { R } ,$ compact convex set $\mathbb { D } \subset \mathbb { R } ^ { d }$ , and $\varepsilon > 0$ , assume that D has a diameter $D \in ( 0 , \infty )$ such that $\| \pmb { x } _ { 1 } - \pmb { x } _ { 2 } \| \le D$ for any $\pmb { x } _ { 1 } , \pmb { x } _ { 2 } \in \mathbb { D }$ , and then:

(I) There exists a hyper input convex neural network $\widehat { \mathsf { H y C N N } }$ with maxout activation functions satisfying $\begin{array} { r } { \operatorname* { s u p } _ { x \in \mathbb { D } } | \widehat { \mathsf { H y C N N } } ( x ) - f ( x ) | \leq \varepsilon } \end{array}$ and $L \cdot h = \mathcal { O } ( ( \frac { D L _ { f } } { \varepsilon } ) ^ { d } )$

(II) There exists a smooth HyCNN with a suficiently small parameter $\tau > 0 ,$ , denoted by $\widehat { \mathsf { H y C N N } } ^ { \tau }$ satisfying $\begin{array} { r } { \operatorname* { s u p } _ { { \pmb x } \in \mathbb { D } } \big | \widehat { \sf H y C N N } ^ { \tau } ( { \pmb x } ) - f ( { \pmb x } ) \big | \leq \varepsilon } \end{array}$ and $L \cdot h = \mathcal { O } ( ( \frac { D L _ { f } } { \varepsilon } ) ^ { d } )$

(III) There exists a sequence of smooth HyCNNs, denoted by $\{ { \mathsf { H y C N N } } _ { n } ^ { \tau } \} _ { n = 1 } ^ { \infty }$ , satisfying $\mathsf { H y C N N } _ { n } ^ { \tau } \to f$ uniformly on D, when τ is suficiently small and $L \cdot h = \mathcal { O } ( ( \frac { D L _ { f } } { \varepsilon } ) ^ { d } )$

The proof of Theorem 3 is provided in Appendix B.3. Compared to existing qualitative representation power theorems of ICNNs [34, 40], Theorem 3 extends this result to HyCNNs and provides quantitative guidelines for hyperparameter selection.

Although these properties of (smooth) HyCNNs show that they can uniformly approximate convex potentials on a compact set, convergence of potentials does not generally imply convergence of the gradient fields [34]. Theorem 4 shows that the smooth HyCNNs approximate the gradient of convex functions well.

Theorem 4 (Gradient Convergence of Smooth HyCNNs). Let $f :  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ be a Lipschitz continuous convex function, and $\{ \mathsf { H } \widehat { \mathsf { y C N N } } _ { n } ^ { \tau } \} _ { n = 1 } ^ { \infty }$ be a sequence of smooth HyCNNs with hyperparameters $L _ { n } \cdot h _ { n } =$ $O ( \varepsilon _ { n } ^ { - d } )$ for any $\varepsilon _ { n } > 0 ,$ , where $\begin{array} { r } { \widehat { \sf H y C N N } _ { n } ^ { \tau } : \mathbb { R } ^ { d }  \mathbb { R } . } \end{array}$ . Assume that $\mathsf { H y C N N } _ { n } ^ { \tau } \to f$ . Then

(I) For almost every $\pmb { x } \in \mathbb { R } ^ { d } , \nabla \mathsf { H y C N N } _ { n } ^ { \tau } ( \pmb { x } )  \nabla f ( \pmb { x } )$

(II) If f is Lipschitz smooth, the gradient sequence ${ \widehat { \mathsf { V H y C N N } } } _ { n }$ converges to $\nabla f$ uniformly at a rate of $O \left( { \sqrt { \varepsilon _ { n } } } \right)$

The proof of Theorem 4 is provided in Appendix B.4. Generally, Theorem 4 holds for any diferentiable convex function sequence. Note that the uniform convergence assumption in Theorem 4 can be verified using the representation power of smooth HyCNNs on compact sets (see Theorem 3(III)).

Based on the results above and Brenier’s theorem, the following Theorem 5 shows that smooth HyCNNs are distributionally universal in our problem.

Theorem 5 (Distributional Universality of Smooth HyCNNs). Let $\mu , \nu \in \mathcal { P } ( \Omega )$ be regarded as probability measures on $\mathbb { R } ^ { d }$ supported on Ω. Assume that $\mu$ is absolutely continuous with respect to the d-dimensional Lebesgue measure on $\mathbb { R } ^ { d }$ , and that $\mu$ and ν have finite second moments. Then there exists a sequence of smooth HyCNNs, denoted by $\{ \mathsf { H y C N N } _ { n } ^ { \tau } \} _ { n = 1 } ^ { \infty }$ , such that $( \widehat { \nabla { \mathsf { H y C N N } } _ { n } ^ { \tau } } ) _ { \# } \mu \Rightarrow \nu$ , where ⇒ denotes weak convergence of probability measures on $\mathbb { R } ^ { d }$

The proof of Theorem 5 is provided in Appendix B.5. It demonstrates that smooth HyCNN gradient maps are distributionally universal for approximating target distributions. For a reference distribution $\mathbb { P } ^ { \epsilon } \in \mathcal { P } ( \Omega )$ and any candidate least-favorable distribution $\mathbb { P } ^ { \star } \in \mathcal { P } ( \Omega )$ in our problem, assume that both $\mathbb { P } ^ { \epsilon }$ and $\mathbb { P } ^ { \star }$ have finite second moments, and $\mathbb { P } ^ { \epsilon }$ is absolutely continuous with respect to the Lebesgue measure on $\mathbb { R } ^ { d }$ . Then, the gradient map of smooth HyCNNs is distributionally universal for approximating any candidate least-favorable distribution.

## 5 Experiments

In this section, we conduct experiments to assess the efectiveness of the proposed method on datasets of varying training sample sizes and dimensions. Specifically, we evaluate its performance on small sample sizes using a synthetic Gaussian mixture dataset (Section 5.1) and the MNIST dataset (Section 5.2), and on large sample sizes using the Higgs dataset (Section 5.3). We also provide a qualitative illustration of our method on a high-dimensional human face generation task (Section 5.4). All experiments were conducted on a laptop with an NVIDIA GeForce RTX 4060 GPU using Python. Compared to the existing methods, our approach can handle hypothesis testing problems across diferent sample sizes and dimensions, achieving higher accuracy and robustness. Codes and experimental results are available at https://github.com/Arthas819/SDRHT.

## 5.1 Synthetic Gaussian Mixture Dataset

In this subsection, we consider a synthetic Gaussian mixture dataset based on [17]. Let the samples under the two hypotheses be generated from distributions $0 . 5 \mathcal { N } ( 0 . 4 e , I _ { d } ) + 0 . 5 \mathcal { N } ( - 0 . 4 e , I _ { d } )$ and $0 . 5 \mathcal { N } ( 0 . 4 f , I _ { d } ) + 0 . 5 \mathcal { N } ( - 0 . 4 f , I _ { d } )$ , respectively. Here $e \in \mathbb { R } ^ { d }$ is a vector with all entries equal to 1, and $f \in \mathbb { R } ^ { d }$ is a vector with the first ⌈d/2⌉ entries equal to 1 and the remaining entries equal to −1. Consider a setting with a small number of training samples $n _ { 0 } = n _ { 1 } = 1 0$ , and then test on 1000 new samples from each mixture model. We compare our method against Wasserstein DRO-based hypothesis testing (WDRO) [17], Gaussian Mixture Model (GMM), kernel support vector machine (SVM) with radial basis function (RBF) kernel, and a three-layer neural network (3NN) with hidden layer dimensions (32,32,32). The details of the WDRO method are provided in Appendix D. All hyperparameters, including λ and ϵ in SDRO-H (SDRHT with HyCNN) and SDRO-C (SDRHT with standard ICNN), ambiguity set radii, and kernel bandwidth in WDRO, are determined by cross-validation.

![](images/fc9839d3017fed72d56aa5cb1a4c9f6b5774d88be5cfd1c0ff01f63ce15c1876.jpg)  
(a) d = 4

![](images/1c3438f7842e7cd228b75c38cea81ee3553ee31197e0c46be5a8b42b4467b3c7.jpg)  
(b) d = 10

![](images/2592f9e0b3d770718ca92a55d2cc851c6125937069878fa515d5221f45af06f1.jpg)  
(c) d = 30

![](images/48727bf0cae536afb80ed4b7ce35864b9a00d092d021f24ef75965acc7ac0086.jpg)  
(d) d = 50

![](images/b83f0afb280bcfdcf41ea69b70ebdb3adce6b35d997122526fdf9035886fac9a.jpg)  
(e) d = 70

![](images/2a8bea26d68fbf8e38f117e0285c156623a055aadc1587fe86633c9bf874fc37.jpg)  
(f) d = 100  
Figure 3: Error rate comparison on GMM dataset, averaged over 100 trials. SDRO-H: SDRHT method with HyCNN; SDRO-C: SDRHT method with standard ICNN.

Table 2: Computation time (s) of methods on Gaussian mixture dataset, averaged over 100 trials.
<table><tr><td rowspan="2">d</td><td colspan="5">Computation Time (s)</td></tr><tr><td>SDRO-H</td><td>SDRO-C</td><td>WDRO</td><td>GMM</td><td>SVM 3NN</td></tr><tr><td>4</td><td>1.906</td><td>1.749</td><td>0.283</td><td>0.075</td><td>&lt;0.000 0.052</td></tr><tr><td>10</td><td>1.507</td><td>1.802</td><td>0.275</td><td>0.002 0.001</td><td>0.040</td></tr><tr><td>30</td><td>1.292</td><td>1.900</td><td>0.271</td><td>0.002</td><td>&lt;0.000 0.034</td></tr><tr><td>50</td><td>1.753</td><td>1.897</td><td>0.285</td><td>0.002</td><td>0.001 0.033</td></tr><tr><td>70</td><td>1.614</td><td>1.907</td><td>0.290</td><td>0.002</td><td>&lt;0.000 0.026</td></tr><tr><td>100</td><td>1.725</td><td>1.844</td><td>0.327</td><td>0.002</td><td>&lt;0.000 0.028</td></tr></table>

Figure 3 compares the classification error rate of selected methods over varying dimensions $d \in \{ 4 , 1 0 , 3 0 , 5 0 , 7 0 , 1 0 0 \}$ . The horizontal axis in Figure 3 represents that there are m independent observations in a testing sample, and the decision is made by majority rule [17]. As illustrated, Hy CNN is more suitable for approximating convex potentials in SDRO than standard ICNN (though the performance of SDRO-C remains competitive in Figures 3b-3f), and the SDRO-H method achieves the lowest classification error rate in almost all dimensions. These results demonstrate the superior performance of our method on hypothesis-testing tasks with small sample sizes compared to selected baselines. Table 2 shows the average computation time of all methods. Although the training time of SDRO is higher than that of other baselines, it remains under 2s and remains stable as d increases.

Figure 4 reports the accuracy of SDRO-H and WDRO across diferent hyperparameter combinations when $n _ { 0 } = n _ { 1 } = 1 0$ and $d = 1 0 0$ . As illustrated, the accuracy under the best parameter combination of WDRO is inferior to that of SDRO-H. The result of SDRO-H indicates that one may improve accuracy by moderately increasing the penalty parameter λ and decreasing the en tropic regularization parameter ϵ. This is because a small λ leads to excessive conservatism, and a large ϵ induces stronger smoothing and may dilute the correlation between samples and classification labels. The result of WDRO demonstrates that under an appropriate bandwidth selection, the classification accuracy is not sensitive to λ.

![](images/659deed17704b7c48d2b2ba8f8842c86f8e2628505ebe482708d79d8e2fe5743.jpg)  
(a) SDRO-H

![](images/257eb8e58c977ca08378b1b29994f7212a0293a9bea594fcedb2fe4927fbf5d6.jpg)  
(b) WDRO  
Figure 4: Average accuracy (%) of SDRO-H and WDRO on the GMM dataset with diferent hyperparameters $( n _ { 0 } = n _ { 1 } = 1 0 , d = 1 0 0 )$

To evaluate the adversarial robustness of our model on adversarial examples, we apply adversarial perturbations to the test data using the projected gradient descent (PGD) attack [26, 41]. Specifically, for a testing sample $\omega \sim \mathbb { P } _ { k } .$ , the attacked sample for fixed detector $\phi _ { \mathrm { o p t . } }$ , defined by a point-wise perturbation problem, is given by

$$
\omega _ { \mathrm { a t t a c k e d } } : = \omega + \delta _ { \omega } ^ { * } , \quad \delta _ { \omega } ^ { * } : = \arg \operatorname* { m a x } _ { \| \delta _ { \omega } \| _ { p } \leq \Delta } \quad L _ { k } ( \phi _ { \mathrm { o p t } } , \omega + \delta _ { \omega } ) .\tag{19}
$$

Table 3: Robustness comparison on the GMM dataset after PGD attack, averaged over m $\in [ 1 0 ]$ and 5 trials (accuracy %, $n _ { 0 } = n _ { 1 } = 1 0 , d = 5 0 )$ . Bold represents the highest accuracy in each column.
<table><tr><td rowspan="2">Method</td><td colspan="4"> $\mathbf { P G D } / _ { 2 }$  Attack</td><td colspan="4"> $\mathbf { P G D } { - } L _ { \infty }$  Attack</td></tr><tr><td> $\Delta = 0 . 0 2 5$ </td><td> $\Delta = 0 . 0 5$ </td><td> $\Delta = \mathbf { 0 . 0 7 5 }$ </td><td> $\Delta = 0 . 1$ </td><td> $\Delta = 0 . 0 5$ </td><td> $\Delta = 0 . 1$ </td><td> $\Delta = 0 . 1 5$ </td><td> $\Delta = 0 . 2$ </td></tr><tr><td>SDRO-H</td><td>85.801</td><td>85.769</td><td>85.736</td><td>85.737</td><td>85.467</td><td>85.165</td><td>84.906</td><td>84.811</td></tr><tr><td>SDRO-C</td><td>82.333</td><td>82.347</td><td>82.332</td><td>82.309</td><td>82.175</td><td>82.018</td><td>81.867</td><td>81.774</td></tr><tr><td>WDRO</td><td>69.693</td><td>68.902</td><td>68.108</td><td>67.924</td><td>62.440</td><td>54.711</td><td>48.096</td><td>44.256</td></tr><tr><td>GMM</td><td>73.401</td><td>72.392</td><td>71.367</td><td>71.121</td><td>63.100</td><td>51.579</td><td>41.804</td><td>36.414</td></tr><tr><td>SVM</td><td>64.477</td><td>63.628</td><td>62.774</td><td>62.561</td><td>55.957</td><td>47.171</td><td>40.166</td><td>36.166</td></tr><tr><td>3NN</td><td>58.187</td><td>57.021</td><td>55.906</td><td>55.638</td><td>47.635</td><td>37.184</td><td>29.483</td><td>25.509</td></tr></table>

In the following, we assess robustness using $L _ { 2 }$ and $L _ { \infty }$ norm constraints, i.e., $p = 2$ and $p \to \infty$ in Eq. (19). In Table 3, we compare the classification accuracy of all methods across diferent levels of ∆. As shown in Table 3, the accuracies of the proposed methods remain higher and more stable than those of the baselines as the attack intensity ∆ increases under both $L _ { 2 }$ and $L _ { \infty }$ constraints, and SDRO-H consistently yields the best test accuracy across all values of ∆.

## 5.2 MNIST Handwritten Digits Classification

This subsection considers the MNIST handwritten digits dataset (d=784). In each trial, we randomly select 5 digits as class 0, and the remaining as class 1. We select $n ^ { \prime }$ samples for each digit of each class to form the two training sets, and therefore $n _ { 0 } = n _ { 1 } = 5 n ^ { \prime }$ . The testing data includes 1000 images from both classes. For this dataset, we further include logistic regression (LR) and Flow-based DRO (FDRO) [26] as baselines. The FDRO method, which characterizes least-favorable distributions from a generative perspective, is suitable for high-dimensional robust hypothesis testing and requires training convolutional autoencoders. In the following, the label ‘SDRO’ represents our SDRHT method with HyCNN. All hyperparameters are determined by cross-validation.

![](images/f5ef0555fc957706daeeb2ce33a671e0ed6ac382e0450b9b5b78d17f3e195b26.jpg)  
(a) n<sup>′</sup> = 2, n<sub>0</sub> = n<sub>1</sub> = 10

![](images/ba3565987cf712f063556bf1e6a27708ae8cc945a9c9d74332d003aa948a37c9.jpg)  
(b) $n ^ { \prime } = 5 , n _ { 0 } = n _ { 1 } = 2 5$

![](images/90a774353d218b1b173f704ca2cae0658eb1dc4b38a473c8fdb74daf3206b5fa.jpg)  
(c) $n ^ { \prime } = 1 0 , n _ { 0 } = n _ { 1 } = 5 0$  
Figure 5: Error rate comparison on the MNIST dataset, averaged over 100 trials.

Figure 5 illustrates that the proposed method achieves a lower error rate than existing welltuned DRO methods and common classifiers across diferent training set sizes. Table 4 shows the average computation time of all methods, illustrating that the training time of SDRO is acceptable and stable as $n ^ { \prime }$ increases. To evaluate robustness, we compare all methods under both $L _ { 2 }$ and $L _ { \infty }$ PGD attacks (see Eq. (19)) in Table 5. As shown, SDRO achieves the highest accuracy across all non-zero attack intensities under both norms, indicating the superior robustness of our method. Under the $\mathrm { P G D } { - } L _ { \infty }$ attack, the three DRO-based models demonstrate better robustness than other baselines.

Table 4: Computation time (s) of methods on the MNIST dataset, averaged over 100 trials.
<table><tr><td rowspan="2"> $\mathbf { \boldsymbol { n } } ^ { \prime }$ </td><td colspan="6">Computation Time (s)</td></tr><tr><td>SDRO</td><td>FDRO</td><td>WDRO</td><td>LR</td><td>SVM</td><td>3NN</td></tr><tr><td>2</td><td>1.943</td><td>4.821</td><td>0.022</td><td>0.008</td><td>&lt;0.001</td><td>0.167</td></tr><tr><td>5</td><td>1.477</td><td>8.999</td><td>0.045</td><td>0.011</td><td>0.001</td><td>0.231</td></tr><tr><td>10</td><td>1.465</td><td>17.153</td><td>0.135</td><td>0.018</td><td>0.001</td><td>0.269</td></tr></table>

Table 5: Robustness comparison on MNIST dataset after PGD attack, averaged over $m \in [ 1 0 ]$ and 5 trials (accuracy %, $n _ { 0 } = n _ { 1 } = 2 5 )$ .
<table><tr><td rowspan="2">Method</td><td colspan="4"> $\mathbf { P G D } / _ { 2 }$  Attack</td><td colspan="4"> $\mathbf { P G D } { - } L _ { \infty }$  Attack</td></tr><tr><td> $\Delta = \mathbf { 0 . 0 2 5 }$ </td><td> $\Delta = 0 . 0 5$ </td><td> $\Delta = \mathbf { 0 . 0 7 5 }$ </td><td> $\Delta = 0 . 1$ </td><td> $\Delta = 0 . 0 5$ </td><td> $\Delta = 0 . 1$ </td><td> $\Delta = 0 . 1 5$ </td><td> $\Delta = 0 . 2$ </td></tr><tr><td>SDRO</td><td>95.75</td><td>95.70</td><td>95.67</td><td>95.67</td><td>93.53</td><td>90.60</td><td>87.46</td><td>85.47</td></tr><tr><td>FDRO</td><td>91.04</td><td>90.93</td><td>90.87</td><td>90.87</td><td>86.51</td><td>80.92</td><td>75.28</td><td>71.91</td></tr><tr><td>WDRO</td><td>94.54</td><td>94.47</td><td>94.42</td><td>94.42</td><td>91.18</td><td>85.63</td><td>79.42</td><td>75.56</td></tr><tr><td>LR</td><td>91.15</td><td>90.98</td><td>90.89</td><td>90.89</td><td>84.79</td><td>75.80</td><td>67.27</td><td>62.34</td></tr><tr><td>SVM</td><td>76.67</td><td>76.50</td><td>76.44</td><td>76.44</td><td>72.65</td><td>69.02</td><td>66.21</td><td>64.73</td></tr><tr><td>3NN</td><td>91.39</td><td>91.21</td><td>91.11</td><td>91.11</td><td>82.10</td><td>68.34</td><td>55.00</td><td>48.09</td></tr></table>

We next visualize how the trained convex functions induce adversarial distributional attacks on their opposite classes. In each trial, we select two digits that are easily confused in handwritten form and place them in separate classes. To generate a more intuitive and clear trajectory, in the training phase, we first train a variational autoencoder, including two parts, Encoder and Decoder, to embed all images into a 32-dimensional latent space. Then, the convex potentials are trained on this latent space with a small training sample size. In the generation phase, we randomly select an image $z _ { k }$ from each class $k \in \mathbb { K } ,$ encode it into the latent space, denoted by $\widetilde { \boldsymbol { z } _ { k } } : = \mathsf { E n c o d e r } ( \boldsymbol { z } _ { k } )$ , and then map $\widetilde { z _ { k } }$ by the gradient of the convex potential of the other class, i.e., $\nabla \widetilde { \varphi } _ { | 1 - k | } ( \widetilde { z } _ { k } )$ , and finally draw the attacked image Decoder $\left( \nabla \widetilde { \varphi } _ { | 1 - k | } ( \widetilde { \boldsymbol { z } } _ { k } ) \right)$ . Figure 6 illustrates the trajectory of injecting adversarial attacks on digits in both classes in 7 trials. As illustrated, our adversarial attack naturally achieves conversions of diferent classes of numbers through convex

gradient mapping in a few steps.

![](images/9879c40748bdda4ca117e2103b2df6d4829b1cd5b933c653a33e95672bb47965.jpg)  
(a) Adversarial attack for Class 0 by $\nabla \widetilde { \varphi } _ { 1 }$

![](images/38d849e22bd2b1830b1f4581735499705baa8d2b396a93535bba447370c06b21.jpg)  
(b) Adversarial attack for Class 1 by $\nabla \widetilde { \varphi } _ { 0 }$  
Figure 6: Adversarial attacks on digits by trained convex gradients. The first column displays the original images, while the following six columns show the attacked images from Epochs 1 to 6.

## 5.3 High-energy Physics Higgs Dataset

In this subsection, we evaluate our method on the Higgs dataset, which distinguishes between a signal process where new theoretical Higgs bosons (HIGGS) are produced, and a background process with the identical decay products but distinct kinematics [42]. We consider 21 original features and large training sample sizes.

Table 6 reports the accuracy and computation time of our method against other baselines over varying training sample sizes. As shown, the SDRO method achieves the highest accuracy in a relatively low computation time. Note that Table 6 does not include the result of WDRO, since when the sample size is large, WDRO did not produce an acceptable solution within the 1,800- second time limit. This is because WDRO requires solving a linear program with $\mathcal { O } ( n _ { 0 } n _ { 1 } )$ decision variables (see Appendix D) and becomes computationally prohibitive for large sample sizes.

In Table 7, we show that SDRO remains the highest accuracy over 70% across all non-zero attack intensities under both norms. This result demonstrates that our method can still maintain its accuracy even under significant adversarial perturbations compared to other baselines.

Table 6: Accuracy (%) and computation time (s) of methods on the Higgs dataset, averaged over observations $m \in [ 1 0 ]$ and 10 trials. Bold represents the highest accuracy in each row.
<table><tr><td rowspan="2"> $n _ { 0 } = n _ { 1 }$   $( \times 1 0 ^ { 3 } )$ </td><td colspan="5">Accuracy (%)</td><td colspan="5">Computation Time (s)</td></tr><tr><td>SDRO</td><td>FDRO</td><td>LR</td><td>SVM</td><td>3NN</td><td>SDRO</td><td>FDRO</td><td>LR</td><td>SVM</td><td>3NN</td></tr><tr><td>1</td><td>68.09</td><td>62.07</td><td>59.73</td><td>64.13</td><td>57.92</td><td>1.11</td><td>2.09</td><td>0.02</td><td>0.18</td><td>5.09</td></tr><tr><td>2</td><td>71.32</td><td>65.56</td><td>60.45</td><td>66.67</td><td>58.70</td><td>1.06</td><td>2.06</td><td>0.01</td><td>0.58</td><td>11.06</td></tr><tr><td>3</td><td>72.83</td><td>67.20</td><td>60.68</td><td>67.87</td><td>58.97</td><td>1.13</td><td>2.69</td><td>0.01</td><td>0.74</td><td>9.86</td></tr><tr><td>4</td><td>73.88</td><td>68.03</td><td>60.95</td><td>68.88</td><td>60.17</td><td>1.64</td><td>3.65</td><td>0.02</td><td>1.51</td><td>11.21</td></tr><tr><td>5</td><td>74.62</td><td>68.77</td><td>61.01</td><td>69.56</td><td>61.00</td><td>1.79</td><td>3.56</td><td>0.02</td><td>2.20</td><td>11.16</td></tr><tr><td>6</td><td>75.37</td><td>69.15</td><td>61.07</td><td>70.27</td><td>61.54</td><td>1.87</td><td>3.30</td><td>0.02</td><td>3.01</td><td>13.74</td></tr><tr><td>7</td><td>75.71</td><td>69.21</td><td>61.09</td><td>70.67</td><td>62.86</td><td>2.37</td><td>3.35</td><td>0.03</td><td>4.46</td><td>15.90</td></tr><tr><td>8</td><td>76.25</td><td>69.60</td><td>61.22</td><td>71.13</td><td>63.66</td><td>2.47</td><td>3.30</td><td>0.03</td><td>5.66</td><td>16.71</td></tr><tr><td>9</td><td>76.26</td><td>69.71</td><td>61.18</td><td>71.14</td><td>63.85</td><td>2.54</td><td>3.73</td><td>0.02</td><td>7.33</td><td>18.53</td></tr><tr><td>10</td><td>76.66</td><td>69.91</td><td>61.29</td><td>71.59</td><td>64.89</td><td>2.96</td><td>3.52</td><td>0.03</td><td>8.65</td><td>21.41</td></tr></table>

Note: The WDRO method did not produce an acceptable solution within the 1,800-second time limit.

Table 7: Robustness comparison on Higgs dataset after PGD attack, averaged over $m \in [ 1 0 ]$ and 5 trials (accuracy %, $n _ { 0 } = n _ { 1 } = 2 , 0 0 0 )$
<table><tr><td rowspan="2">Method</td><td colspan="4"> $\mathbf { P G D } / _ { 2 }$  Attack</td><td colspan="4"> $\mathbf { P G D } { - } L _ { \infty }$  Attack</td></tr><tr><td> $\Delta = \mathbf { 0 . 0 2 5 }$ </td><td> $\Delta = 0 . 0 5$ </td><td> $\Delta = \mathbf { 0 . 0 7 5 }$ </td><td> $\Delta = 0 . 1$ </td><td> $\Delta = \mathbf { 0 . 0 } 2 5$ </td><td> $\Delta = 0 . 0 5$ </td><td> $\Delta = \mathbf { 0 . 0 7 5 }$ </td><td> $\Delta = 0 . 1$ </td></tr><tr><td>SDRO</td><td>72.342</td><td>71.998</td><td>71.887</td><td>71.891</td><td>73.352</td><td>72.758</td><td>72.436</td><td>72.284</td></tr><tr><td>FDRO</td><td>66.999</td><td>64.888</td><td>62.736</td><td>60.835</td><td>62.780</td><td>55.396</td><td>48.914</td><td>44.461</td></tr><tr><td>LR</td><td>62.945</td><td>60.913</td><td>59.193</td><td>57.582</td><td>59.708</td><td>54.046</td><td>48.528</td><td>44.412</td></tr><tr><td>SVM</td><td>67.669</td><td>65.849</td><td>64.191</td><td>62.716</td><td>64.524</td><td>58.350</td><td>53.250</td><td>49.614</td></tr><tr><td>3NN</td><td>58.204</td><td>54.669</td><td>50.782</td><td>48.015</td><td>50.264</td><td>38.658</td><td>30.418</td><td>26.266</td></tr></table>

## 5.4 Human Face Generation

In this subsection, similar to [28], we present a qualitative experiment on the human face dataset FFHQ [43] that illustrates how our method generates adversarial-attacked high-dimensional images.

We take the gender attribute of the FFHQ dataset as the binary label in the following image generation experiment. We labeled Class 0 as Female and Class 1 as Male. In the training phase, we first map the original $1 0 2 4 \times 1 0 2 4$ images to a 512-dimensional latent space by a pre-trained adversarial latent autoencoder [44]. Then, we train two HyCNNs by Algorithm 1 on the latent space. In the generation phase, we randomly select three images in each class. These samples are first encoded into a latent space, then mapped by our trained convex gradient, and finally recovered in the original space by the pre-trained decoder.

For all experiments, we set $n _ { 0 } = n _ { 1 } = 5 0 0 , \lambda = 1 , \epsilon = 0 . 1 , T = 3 0 .$ , and each HyCNN has a single hidden layer with 128 units. Figure 7 shows the trajectories of generating the attacked images of all selected samples in the gender classification task. In each subfigure, the first column shows the original image, and the other 12 columns represent the images generated using the convex gradient maps trained for 30 epochs, respectively. As illustrated, the samples we generated underwent meaningful modifications to the original images, such that the generated images gradually acquired visual characteristics associated with the opposite dataset label after a small number of training epochs.

![](images/72306e41cc0d8ba53dc5774e5e9ccaa1149cbfda0edaa2cdc40d1988a44d78b3.jpg)  
(b) Male to Female  
Figure 7: The trajectory of image generation between Female (Class 0) and Male (Class 1). The first column displays the original images, while the following columns show the images generated during 30 epochs.

## 6 Conclusions and Discussions

As a fundamental method in statistics, robust hypothesis testing addresses noisy data, distributional shifts, and measurement errors in real-world, uncertain environments. This paper develops a generative framework for Sinkhorn distributionally robust hypothesis testing (SDRHT), overcoming the non-continuity of the least-favorable distributions of WDRO-based testing and the computational challenges of existing SDRHT methods. By exploiting a conditional-KL represen tation of Sinkhorn ambiguity sets and strong duality, this framework reformulates the infinite dimensional problem as an optimization over parametrized convex potentials. We further estab lish approximation guarantees for the potentials and their gradients, as well as the distributional universality of the induced transport maps. Numerical results demonstrate the superior performance and robustness of our method compared to existing methods on tasks with diferent sample sizes and dimensions.

There are several topics worth investigating for future work. First, it is interesting to extend our method to multi-class hypothesis testing, exploring its closed-form optimal detector and eficient algorithms. Then, since the transport map between distributions is the gradient of a convex potential, directly parameterizing the gradient instead of the potential may yield a better approximation of the transport map. Finally, for distributions with extremely high dimensions, e.g., high-resolution images, it is important to develop methods that eficiently solve transport maps while maintaining the convexity of potentials.

## References

[1] J. Neyman and E. S. Pearson, “On the problem of the most eficient tests of statistical hypotheses,” Philosophical Transactions of the Royal Society of London. Series A, Containing Papers of a Mathematical or Physical Character, vol. 231, no. 694-706, pp. 289–337, 1933.

[2] B. C. Levy, Principles of signal detection and parameter estimation. Springer Science & Business Media, 2008.

[3] P. Schober and T. R. Vetter, “Two-sample unpaired t-tests in medical research,” Anesthesia & Analgesia, vol. 129, no. 4, p. 911, 2019.

[4] V. Chandola, A. Banerjee, and V. Kumar, “Anomaly detection: A survey,” ACM Computing Surveys (CSUR), vol. 41, no. 3, pp. 1–58, 2009.

[5] P. J. Huber, “A robust version of the probability ratio test,” The Annals ofMathematical Statistics, pp. 1753–1758, 1965.

[6] G. Gul and A. M. Zoubir, “Minimax robust hypothesis testing,” ¨ IEEE Transactions on Information Theory, vol. 63, no. 9, pp. 5572–5587, 2017.

[7] D. Kuhn, S. Shafiee, and W. Wiesemann, “Distributionally robust optimization,” Acta Numerica, vol. 34, pp. 579–804, 2025.

[8] R. Gao, L. Xie, Y. Xie, and H. Xu, “Robust hypothesis testing using Wasserstein uncertainty sets,” Advances in Neural Information Processing Systems, vol. 31, 2018.

[9] P. Mohajerin Esfahani and D. Kuhn, “Data-driven distributionally robust optimization using the Wasserstein metric: Performance guarantees and tractable reformulations,” Mathematical Programming, vol. 171, no. 1, pp. 115–166, 2018.

[10] P. J. Huber, Robust statistical procedures. SIAM, 1996.

[11] B. C. Levy, “Robust hypothesis testing with a relative entropy tolerance,” IEEE Transactions on Information Theory, vol. 55, no. 1, pp. 413–421, 2008.

[12] A. Ben-Tal, D. Den Hertog, A. De Waegenaere, B. Melenberg, and G. Rennen, “Robust solutions of optimization problems afected by uncertain probabilities,” Management Science, vol. 59, no. 2, pp. 341–357, 2013.

[13] E. Delage and Y. Ye, “Distributionally robust optimization under moment uncertainty with application to data-driven problems,” Operations Research, vol. 58, no. 3, pp. 595–612, 2010.

[14] W. Wiesemann, D. Kuhn, and M. Sim, “Distributionally robust convex optimization,” Operations Research, vol. 62, no. 6, pp. 1358–1376, 2014.

[15] Z. Sun and S. Zou, “A data-driven approach to robust hypothesis testing using kernel MMD uncertainty sets,” in 2021 IEEE International Symposium on Information Theory (ISIT). IEEE, 2021, pp. 3056–3061.

[16] J.-J. Zhu, W. Jitkrittum, M. Diehl, and B. Scholkopf, “Kernel distributionally robust optimiza-¨ tion: Generalized duality theorem and stochastic approximation,” in International Conference on Artificial Intelligence and Statistics. PMLR, 2021, pp. 280–288.

[17] L. Xie, R. Gao, and Y. Xie, “Robust hypothesis testing with Wasserstein uncertainty sets,” arXiv preprint arXiv:2105.14348, 2021.

[18] J. Wang, R. Gao, and Y. Xie, “Sinkhorn distributionally robust optimization,” Operations Research, 2025.

[19] F. Zhang and J. Wang, “Contextual distributionally robust optimization with causal and continuous structure: An interpretable and tractable approach,” arXiv preprint arXiv:2601.11016, 2026.

[20] G. Peyre and M. Cuturi, ´ Computational optimal transport: With applications to data science. Now Foundations and Trends, 2019.

[21] J. Wang and Y. Xie, “A data-driven approach to robust hypothesis testing using Sinkhorn uncertainty sets,” in 2022 IEEE International Symposium on Information Theory (ISIT). IEEE, 2022, pp. 3315–3320.

[22] J. Wang, R. Gao, and Y. Xie, “Non-convex robust hypothesis testing using Sinkhorn uncertainty sets,” in 2024 IEEE International Symposium on Information Theory (ISIT). IEEE, 2024, pp. 843–848.

[23] C.-H. Lai, Y. Song, D. Kim, Y. Mitsufuji, and S. Ermon, “The principles of difusion models,” arXiv preprint arXiv:2510.21890, 2025.

[24] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” arXiv preprint arXiv:2210.02747, 2022.

[25] T. Dai, D. Simchi-Levi, M. X. Wu, and Y. Xie, “Assured autonomy: How operations research powers and orchestrates generative AI systems,” arXiv preprint arXiv:2512.23978, 2025.

[26] C. Xu, J. Lee, X. Cheng, and Y. Xie, “Flow-based distributionally robust optimization,” IEEE Journal on Selected Areas in Information Theory, 2024.

[27] J. Wang, “Iterative sampling methods for sinkhorn distributionally robust optimization,” arXiv preprint arXiv:2512.12550, 2025.

[28] Z. Xu and J.-J. Zhu, “Gradient flow sampler-based distributionally robust optimization,” arXiv preprint arXiv:2510.25956, 2025.

[29] L. Zhu and Y. Xie, “Distributionally robust optimization via iterative algorithms in continuous probability spaces,” arXiv preprint arXiv:2412.20556, 2024.

[30] X. Cheng, Y. Xie, L. Zhu, and Y. Zhu, “Worst-case generation via minimax optimization in Wasserstein space,” arXiv preprint arXiv:2512.08176, 2025.

[31] A. Nemirovski and A. Shapiro, “Convex approximations of chance-constrained programs,” SIAM Journal on Optimization, vol. 17, no. 4, pp. 969–996, 2007.

[32] S. Chewi, J. Niles-Weed, and P. Rigollet, Statistical optimal transport. Springer, 2025.

[33] S. Ubaru, J. Chen, and Y. Saad, “Fast estimation of tr(f(a)) via stochastic lanczos quadrature,” SIAM Journal on Matrix Analysis and Applications, vol. 38, no. 4, pp. 1075–1099, 2017.

[34] C.-W. Huang, R. T. Chen, C. Tsirigotis, and A. Courville, “Convex potential flows: Universal probability distributions with optimal transport and convex optimization,” arXiv preprint arXiv:2012.05942, 2020.

[35] B. Amos, L. Xu, and J. Z. Kolter, “Input convex neural networks,” in International Conference on Machine Learning. PMLR, 2017, pp. 146–155.

[36] S. Hundrieser, I. Kong, and J. Schmidt-Hieber, “Hyper input convex neural networks for shape constrained learning and optimal transport,” arXiv preprint arXiv:2604.26942, 2026.

[37] I. Goodfellow, D. Warde-Farley, M. Mirza, A. Courville, and Y. Bengio, “Maxout networks,” in International Conference on Machine Learning. PMLR, 2013, pp. 1319–1327.

[38] M. F. Hutchinson, “A stochastic estimator of the trace of the influence matrix for laplacian smoothing splines,” Communications in Statistics-Simulation and Computation, vol. 18, no. 3, pp. 1059–1076, 1989.

[39] A.-A. Pooladian and J. Niles-Weed, “Entropic estimation of optimal transport maps,” arXiv preprint arXiv:2109.12004, 2021.

[40] Y. Chen, Y. Shi, and B. Zhang, “Optimal control via neural networks: A convex approach,” arXiv preprint arXiv:1805.11835, 2018.

[41] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards deep learning models resistant to adversarial attacks,” arXiv preprint arXiv:1706.06083, 2017.

[42] P. Baldi, P. Sadowski, and D. Whiteson, “Searching for exotic particles in high-energy physics with deep learning,” Nature communications, vol. 5, no. 1, p. 4308, 2014.

[43] T. Karras, S. Laine, and T. Aila, “A style-based generator architecture for generative adversarial networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 4401–4410.

[44] S. Pidhorskyi, D. A. Adjeroh, and G. Doretto, “Adversarial latent autoencoders,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 14 104– 14 113.

[45] R. Cescon, A. Martin, and G. Ferrari-Trecate, “Sinkhorn ambiguity sets for distributionally robust control: Convexity, weak compactness, and tractability,” arXiv preprint arXiv:2605.03845, 2026.

[46] S. Shafiee, L. Aolaritei, F. Dorfler, and D. Kuhn, “Nash equilibria, regularization, and compu-¨ tation in optimal transport-based distributionally robust optimization,” Operations Research, 2025.

[47] M. D. Donsker and S. S. Varadhan, “Asymptotic evaluation of certain markov process expectations for large time. iv,” Communications on pure and applied mathematics, vol. 36, no. 2, pp. 183–212, 1983.

[48] A. Shapiro, D. Dentcheva, and A. Ruszczynski, Lectures on stochastic programming: modeling and theory. SIAM, 2021.

## Appendix

## A Summary of Notations

For integer $K \in \mathbb { Z } _ { + } ,$ , define $[ K ] : = \{ 1 , \cdots , K \}$ , which is the set of positive running indices to K. The list of important sets, spaces, and notations used in this paper is shown in Table 8. The list of important functions is shown in Table 9.

Table 8: Notations List
<table><tr><td>Spaces  $\Omega$ </td><td>Description Closed sample space,</td></tr><tr><td> $\mathcal { F }$ </td><td>Functional space, which is not necessarily compact.</td></tr><tr><td>Basic Sets  $\mathbb { K }$ </td><td></td></tr><tr><td> $\mathscr { P } ( \Omega )$ </td><td>Indices set of hypotheses,  $\mathbb { K } = \{ 0 , 1 \}$   $\Omega .$ </td></tr><tr><td> $\mathcal { M } ( \Omega )$ </td><td>The set of all probability distributions on The set of measures on Ω.</td></tr><tr><td> $\Gamma ( \mathbb { P } , \mathbb { Q } )$ </td><td>Joint distribution set of marginal distribution P and</td></tr><tr><td> $\mathcal { P } _ { k }$ </td><td> $\mathbb { Q } .$  The distribution set under hypotheses  $H _ { k } , \mathcal { P } _ { k } \subset \mathcal { P } ( \Omega )$  for each</td></tr><tr><td> $\mathcal { F } _ { \mathrm { C V X } }$ </td><td>Set of convex functions  $\varphi : \mathbb { R } ^ { d }  \mathbb { R } \cup \{ + \infty \}$ </td></tr><tr><td>Parameters</td><td></td></tr><tr><td> $\epsilon$ </td><td>Entropic regularization parameter in Sinkhorn discrepancy,  $\epsilon \geq 0 .$ </td></tr><tr><td> $\rho _ { k }$ </td><td>Radius for Sinkhorn-based ambiguity set, for each  $k \in \mathbb { K } .$ </td></tr><tr><td> $\lambda _ { k }$ </td><td>Penalty parameter in soft-constraint problem (soft-SDRHT),  $\lambda _ { k } \geq 0$  for each</td></tr><tr><td>Distributions</td><td></td></tr><tr><td> $\widehat { \mathbb { P } } _ { k }$ </td><td>Empirical distribution with  $n _ { k }$  samples for each  $k \in \mathbb { K } .$ </td></tr><tr><td> ${ \mathcal K } _ { { \pmb x } _ { k } ^ { ( i ) } , \epsilon }$ </td><td>Kernel extension for any sample point  $i \in [ n _ { k } ]$  and each  $k \in \mathbb { K }$ </td></tr><tr><td> $\mathbb { P } _ { k } ^ { \epsilon }$ </td><td>Mixture of continuous distributions  ${ \mathcal K } _ { { \pmb x } _ { k } ^ { ( i ) } , \epsilon }$  for all  $i \in [ n _ { k } ]$  , see Eq. (4).</td></tr><tr><td> $\widetilde { \mathbb { P } } _ { k }$ </td><td>Discrete approximation of distribution  $\mathbb { P } _ { k } ^ { \epsilon }$  with N samples, for each  $k \in \mathbb { K } .$ </td></tr><tr><td>Discrepancies</td><td></td></tr><tr><td> $S _ { \epsilon } ( \mathbb { P } , \mathbb { Q } )$ </td><td>Sinkhorn discrepancy from distribution P to Q, see Definition 2.</td></tr><tr><td> ${ \mathcal { D } } _ { \mathrm { K L } } ( \mathbb { P } , \mathbb { Q } )$ </td><td>Kullback-Leibler divergence (KL divergence) from distribution P to Q.</td></tr><tr><td>Ambiguity Sets</td><td></td></tr><tr><td> $\mathbb { B } _ { \rho _ { k } } ( \widehat { \mathbb { P } } _ { k } )$ </td><td>Sinkhorn discrepancy-based ambiguity set centered at  $\widehat { \mathbb { P } } _ { k }$  for each  $k \in \mathbb { K } ,$  see Eq. (1).</td></tr><tr><td> $\mathbb { B } _ { \bar { \rho } _ { k } } ( \mathbb { P } _ { k } ^ { \epsilon } )$ </td><td>Equivalent KL-based ambiguity set of  $\mathbb { B } _ { \rho _ { k } } ( \widehat { \mathbb { P } } _ { k } )$  centered at  $\mathbb { P } _ { k } ^ { \epsilon } ,$  see Proposition 1.</td></tr></table>

Table 9: Functions List  
```latex
Functions Description
$c : \Omega \times \Omega  \mathbb { R }$ Transport cost function satisfying Assumption 1, we choose $\begin{array} { r } { c ( \pmb { x } , z ) : = \frac { 1 } { 2 } \| \pmb { x } - \pmb { z } \| _ { 2 } ^ { 2 } . } \end{array}$
$\ell : \mathbb { R } \to \mathbb { R } _ { + } \cup \{ \infty \}$ Generating function (see Definition 1), we choose $\ell ( t ) = \exp ( t )$
$\phi : \Omega \to \mathbb { R }$ Detector for hypothesis testing, which is measurable and upper semicontinuous.
$\phi _ { \mathrm { o p t } } : \Omega \to \mathbb { R }$ Optimal detector for generating function $\begin{array} { r } { \ell ( t ) = \exp ( t ) , \phi _ { \mathrm { o p t } } = \frac { 1 } { 2 } \log ( p _ { 0 } / p _ { 1 } ) . } \end{array}$
$\varphi _ { 1 } , \varphi _ { 2 } : \mathbb { R } ^ { d }  \mathbb { R }$ Convex functions, decision variables in the problem (7).
$\varphi ^ { * } : \mathbb { R } ^ { d }  \mathbb { R } \cup \{ \infty \}$ Convex conjugate of a convex function $\varphi : \mathbb { R } ^ { d }  \mathbb { R } \cup \{ \infty \} .$
$\mathsf { H y C N N } _ { k } : \mathbb { R } ^ { d } \to \mathbb { R }$ Convex function formed by the HyCNN for each $k \in \mathbb { K } .$
$\widetilde { \varphi } _ { k } : \mathbb { R } ^ { d } \to \mathbb { R }$ Strongly convex function, $\begin{array} { r } { \widetilde { \varphi } _ { k } ( z ) = \mathsf { H y C N N } _ { k } ( z ) + \frac { \alpha } { 2 } \| z \| _ { 2 } ^ { 2 } } \end{array}$ for each $k \in \mathbb { K }$ and $\alpha > 0 .$
$L _ { k } : \Omega \to \mathbb { R }$ Loss function of a given detector $\phi$ for sample $\omega ,$ satisfying Assumption 2.
$\Phi : \mathcal { P } ( \Omega ) ^ { 2 }  \mathbb { R }$ Convex surrogate risk of the detector $\phi .$
$g _ { \epsilon , k } ^ { c }$ Continuous extension for the optimal entropic potential $g _ { \epsilon , k , ( n _ { k } , N ) }$ in Algorithm 4.
$S _ { \epsilon , ( n _ { k } , N ) }$ Finite sample estimator for the Sinkhorn discrepancy.
```

## B Technical Analyses and Proofs

## B.1 Proof of Proposition 1

Proof of Proposition 1. By the definition of Sinkhorn discrepancy, for distributions $\mathbb { P } , \mathbb { Q } \in \mathcal { P } ( \Omega )$ , it is equivalent to

$$
\mathcal { S } _ { \epsilon } ( \mathbb { P } , \mathbb { Q } ) = \operatorname* { i n f } _ { \gamma \in \Gamma ( \mathbb { P } , \mathbb { Q } ) } \quad \int _ { \Omega \times \Omega } \Bigl ( c ( x , z ) + \epsilon \cdot \log \frac { \mathrm { d } \gamma ( x , z ) } { \mathrm { d } \mathbb { P } ( x ) \mathrm { d } \nu ( z ) } \Bigr ) \mathrm { d } \gamma ( x , z ) .\tag{20}
$$

Taking the reference distribution as the empirical distribution ${ \widehat { \mathbb { P } } } \in { \mathcal { P } } ( \Omega )$ , Eq. (20) can be further reformulated to

$$
\begin{array} { r l } { \mathcal { S } _ { \epsilon } ( \widehat { \mathbb { P } } , \mathbb { Q } ) = \underset { \gamma \in \Gamma ( \widehat { \mathbb { P } } , \mathbb { Q } ) } { \operatorname* { i n f } } } & { \mathbb { E } _ { x \sim \widehat { \mathbb { P } } } \Big [ \displaystyle \int _ { \Omega } \Big ( c ( x , z ) + \epsilon \log \frac { \mathrm { d } \gamma _ { x x } ( z ) } { \mathrm { d } \nu ( z ) } \Big ) \mathrm { d } z \Big ] } \\ { = \underset { \gamma \in \Gamma ( \widehat { \mathbb { P } } , \mathbb { Q } ) } { \operatorname* { i n f } } } & { \epsilon \cdot \mathbb { E } _ { x \sim \widehat { \mathbb { P } } } \Big [ \displaystyle \int _ { \Omega } \log \frac { \mathrm { d } \gamma _ { x x } ( z ) } { e ^ { - c ( x , z ) } \cdot \mathrm { d } \nu ( z ) } \mathrm { d } z \Big ] } \\ { = \underset { \gamma \in \Gamma ( \widehat { \mathbb { P } } , \mathbb { Q } ) } { \operatorname* { i n f } } } & { \epsilon \cdot \mathbb { E } _ { x \sim \widehat { \mathbb { P } } } \Big [ \displaystyle \int _ { \Omega } \log \frac { \mathrm { d } \gamma _ { x } ( z ) } { \mathrm { d } K _ { x , c } ( z ) } \mathrm { d } z - \log \displaystyle \int _ { \mathbb { R } ^ { d } } e ^ { - c ( x , z ) / \epsilon } \mathrm { d } z \Big ] } \\ { = \underset { \gamma \in \Gamma ( \widehat { \mathbb { P } } , \mathbb { Q } ) } { \operatorname* { i n f } } } & { \epsilon \cdot \mathbb { E } _ { x \sim \widehat { \mathbb { P } } } \Big [ \mathcal { D } _ { \mathrm { K L } } ( \gamma _ { x } | | K _ { x , c } ) \Big ] - \epsilon \cdot \mathbb { E } _ { x \sim \widehat { \mathbb { P } } } \Big [ \log \displaystyle \int _ { \mathbb { R } ^ { d } } e ^ { - c ( x , z ) / \epsilon } \mathrm { d } z \Big ] , } \end{array}\tag{21}
$$

where

$$
\mathrm { d } K _ { \boldsymbol { x } , \epsilon } ( z ) : = \frac { e ^ { - c ( \boldsymbol { x } , z ) / \epsilon } } { \int _ { \mathbb { R } ^ { d } } e ^ { - c ( \boldsymbol { x } , \boldsymbol { u } ) / \epsilon } \cdot \mathrm { d } \boldsymbol { u } } \cdot \mathrm { d } \boldsymbol { v } ( z ) .
$$

Thus, the ambiguity sets in Eq. (1) is equivalent to

$$
\operatorname* { i n f } _ { \gamma \in \Gamma ( \widehat { \mathbb { P } } _ { k } , \mathbb { Q } ) } \quad \epsilon \cdot \mathbb { E } _ { x \sim \widehat { \mathbb { P } } _ { k } } \Big [ \mathcal { D } _ { \mathrm { K L } } ( \gamma _ { x } \| \mathcal { K } _ { x , \epsilon } ) \Big ] \leq \rho _ { k } + \epsilon \cdot \mathbb { E } _ { x \sim \widehat { \mathbb { P } } _ { k } } \Big [ \log \int _ { \mathbb { R } ^ { d } } e ^ { - c ( x , z ) / \epsilon } \mathrm { d } z \Big ] , \quad \forall k \in \mathbb { K } ,
$$

that is,

$$
\mathbb { B } _ { \bar { \rho } _ { k } } ( \mathbb { P } _ { k } ^ { \epsilon } ) : = \{ \mathbb { Q } \in \mathscr { P } ( \Omega ) : \operatorname* { i n f } _ { \gamma \in \Gamma ( \widehat { \mathbb { P } } _ { k } , \mathbb { Q } ) } \mathbb { E } _ { x \sim \widehat { \mathbb { P } } _ { k } } \Big [ \mathcal { D } _ { \mathrm { K L } } ( \gamma _ { x } | | \mathcal { K } _ { x , \epsilon } ) \Big ] \leq \bar { \rho } _ { k } / \epsilon \} , \quad \forall k \in \mathbb { K } ,
$$

where $\bar { \rho } _ { k } : = \rho _ { k } + \epsilon \mathbb { E } _ { \pmb { x } \sim \widehat { \mathbb { P } } _ { k } } \bigg [ \log \int _ { \mathbb { R } ^ { d } } e ^ { - c ( \pmb { x } , \pmb { z } ) / \epsilon } \mathrm { d } \boldsymbol { z } \bigg ]$ and probability density d $\mathbb { P } _ { k } ^ { \epsilon } ( z ) : = \mathbb { E } _ { \pmb { x } \sim \widehat { \mathbb { P } } _ { k } } \left[ { \mathrm { d } } K _ { \pmb { x } , \epsilon } ( z ) \right]$ This completes the proof.

## B.2 Proof of Theorem 1

To prove Theorem 1, we first establish the weak compactness of the Sinkhorn ambiguity sets and the weak upper semicontinuity of the expected loss.

Based on Eq. (3), the following lemma holds.

Lemma 1 (Weak Compactness of the Ambiguity Sets). Under Assumption 1, the ambiguity set defined in (3) is weakly compact in $\mathcal { P } ( \Omega )$ for any closed set $\Omega \subseteq \mathbb { R } ^ { d } , k \in \mathbb { K } ,$ , and $\bar { \rho } _ { k } \ge 0$

The proof of Lemma 1 is provided in the following Appendix B.2.1. While [45] also confirms the weak compactness of Sinkhorn ambiguity sets based on the properties of Wasserstein balls, our proof follows directly from the KL representation in Proposition 1 and holds under more general assumptions.

Lemma 2 (Weak Upper Semicontinuity of the Expected Loss). Under Assumption 2(I), $\mathbb { E } _ { \omega \sim \mathbb { Q } } [ L _ { k } ( \phi , \omega ) ]$ is weakly upper semicontinuous in $\mathbb { Q } \in \mathbb { B } _ { \hat { \rho } _ { k } } ( \mathbb { P } _ { k } ^ { \epsilon } )$ for any $k \in \mathbb { K }$ and $\phi \in { \mathcal { F } }$

The proof of Lemma 2 is provided in the following Appendix B.2.2.

## B.2.1 Proof of Lemma 1

Proof of Lemma 1. To prove $\mathbb { B } _ { \bar { \rho } _ { k } } ( \mathbb { P } _ { k } ^ { \epsilon } )$ is weakly compact, we first show its connection with KLdivergence-based ambiguity set. Since the KL-divergence is a convex function, by Jensen’s inequality, we have

$$
\begin{array} { r l } & { \mathcal { D } _ { \mathrm { K L } } ( \mathbb { Q } | | \mathbb { P } _ { k } ^ { \epsilon } ) \leq \underset { \gamma \in \Gamma ( \widehat { \mathbb { P } } _ { k } , \mathbb { Q } ) } { \operatorname* { i n f } } \mathbb { E } _ { { \boldsymbol { x } } \sim \widehat { \mathbb { P } } _ { k } } \bigg [ \mathcal { D } _ { \mathrm { K L } } ( \gamma _ { \boldsymbol { x } } | | \mathcal { K } _ { { \boldsymbol { x } } , \epsilon } ) \bigg ] } \\ & { \qquad \leq \frac { \rho _ { k } } { \epsilon } + \mathbb { E } _ { { \boldsymbol { x } } \sim \widehat { \mathbb { P } } _ { k } } \bigg [ \log \int _ { \mathbb { R } ^ { d } } e ^ { - c ( { \boldsymbol { x } } , { \boldsymbol { z } } ) / \epsilon } \mathrm { d } { \boldsymbol { z } } \bigg ] , \quad \forall k \in \mathbb { K } , } \end{array}
$$

where probability distribution $\mathbb { Q } \in \mathscr { P } ( \Omega )$ . Let

$$
\mathbb { B } _ { \bar { \rho } _ { k } } ^ { \prime } ( \mathbb { P } _ { k } ^ { \epsilon } ) : = \{ \mathbb { Q } : \mathcal { D } _ { \mathrm { K L } } ( \mathbb { Q } \| \mathbb { P } _ { k } ^ { \epsilon } ) \leq \bar { \rho } _ { k } / \epsilon \} , \quad \forall k \in \mathbb { K } .
$$

According to Proposition 3.12 in [7], the KL-divergence-based ambiguity sets $\mathbb { B } _ { \bar { \rho } _ { k } } ^ { \prime } ( \mathbb { P } _ { k } ^ { \epsilon } )$ for any $k \in \mathbb { K }$ are weakly compact, for any closed set $\Omega \subseteq \mathbb { R } ^ { d }$ and $k \in \mathbb { K }$ , distribution $\mathbb { P } _ { k } ^ { \epsilon } \in \mathcal { P } ( \Omega )$ and $\bar { \rho } _ { k } \ge 0$

Since $\mathbb { B } _ { \bar { \rho } _ { k } } ( \mathbb { P } _ { k } ^ { \epsilon } ) \subseteq \mathbb { B } _ { \bar { \rho } _ { k } } ^ { \prime } ( \mathbb { P } _ { k } ^ { \epsilon } )$ for any $k \in \mathbb { K } ,$ , it sufices to prove that $\mathbb { B } _ { \bar { \rho } _ { k } } ( \mathbb { P } _ { k } ^ { \epsilon } )$ is weakly closed, as weakly closed subsets of weakly compact sets are weakly compact [46, Lemma 1]. Define

$$
\mathrm { d } M _ { k , \epsilon } ( \pmb { x } , \pmb { z } ) : = \mathrm { d } \widehat { \mathbb { P } } _ { k } ( \pmb { x } ) \cdot \mathrm { d } K _ { \pmb { x } , \epsilon } ( \pmb { z } ) = \frac { e ^ { - c ( \pmb { x } , \pmb { z } ) / \epsilon } } { \int _ { \mathbb { R } ^ { d } } e ^ { - c ( \pmb { x } , \pmb { u } ) / \epsilon } \cdot \mathrm { d } \pmb { u } } \cdot \mathrm { d } \widehat { \mathbb { P } } _ { k } ( \pmb { x } ) \cdot \mathrm { d } \pmb { \nu } ( \pmb { z } ) , \quad \forall k \in \mathbb { K } ,
$$

According to Eq. (20), we have

$$
\operatorname* { i n f } _ { \gamma \in \Gamma ( \widehat { \mathbb { P } } _ { k } , \mathbb { Q } ) } \mathbb { E } _ { \mathbf { x } \sim \widehat { \mathbb { P } } _ { k } } \Big [ \mathcal { D } _ { \mathrm { K L } } ( \gamma _ { x } \| \mathcal { K } _ { x , \epsilon } ) \Big ] = \operatorname* { i n f } _ { \gamma \in \Gamma ( \widehat { \mathbb { P } } _ { k } , \mathbb { Q } ) } \mathcal { D } _ { \mathrm { K L } } ( \gamma \| M _ { k , \epsilon } ) , \quad \forall k \in \mathbb { K } .
$$

According to Proposition 3.12 in [7], to prove the Sinkhorn-based ambiguity set is weakly closed, it sufices to prove that function $\begin{array} { r } { f _ { k } ( \mathbb { Q } ) : = \operatorname* { i n f } _ { \gamma \in \Gamma ( \widehat { \mathbb { P } } _ { k } , \mathbb { Q } ) } \mathcal { D } _ { \mathrm { K L } } ( \gamma | | M _ { k , \epsilon } ) } \end{array}$ is weakly lower semicontinuous in Q for each $k \in \mathbb { K }$ , which implies that any sublevel set of $f ( \mathbb { Q } ) , { \mathrm { i . e . , } } \left\{ \mathbb { Q } : f ( \mathbb { Q } ) \leq c \right\}$ for any $c \in \mathbb { R } ,$ , is weakly closed. Let $\{ \mathbb { Q } _ { j } \} _ { j \in \mathbb { N } } \subset \mathcal { P } ( \Omega )$ be any sequence of distributions that converges weakly to $\mathbb { Q } .$ By the definition of infimum, there exists a sequence of couplings $\gamma _ { n } \in \Gamma ( \widehat { \mathbb { P } } _ { k } , \mathbb { Q } _ { n } )$ that converges weakly to $\gamma \in \Gamma ( \widehat { \mathbb { P } } _ { k } , \mathbb { Q } )$ such that

$$
\mathcal { D } _ { \mathrm { K L } } ( \gamma _ { n } \| M _ { k , \epsilon } ) \leq f _ { k } ( \mathbb { Q } _ { n } ) + \frac { 1 } { n } , \quad \forall k \in \mathbb { K } .\tag{22}
$$

Then, for each $k \in \mathbb { K } ,$ , we have

$$
f _ { k } ( \mathbb { Q } ) \leq \mathcal { D } _ { \mathrm { K L } } ( \boldsymbol { \gamma } \| M _ { k , \epsilon } ) \leq \operatorname* { l i m i n f } _ { j \in \mathbb { N } } \ \mathcal { D } _ { \mathrm { K L } } ( \boldsymbol { \gamma } _ { j } \| M _ { k , \epsilon } ) \leq \operatorname* { l i m i n f } _ { j \in \mathbb { N } } \ f _ { k } ( \mathbb { Q } _ { j } ) , \quad \forall k \in \mathbb { K } ,
$$

where the second inequality is due to the weak lower semi-continuity of $\mathcal { D } _ { \mathrm { K L } } ( \gamma \Vert M _ { k , \epsilon } )$ in $\gamma \ [ 7 ,$ Proposition 3.12], and the final inequality is due to Eq. (22). Therefore, for each $k \in \mathbb { K }$ , the function $f _ { k } ( \mathbb { Q } )$ is weakly lower semicontinuous in $\mathbb { Q }$ which implies that the ambiguity set $\mathbb { B } _ { \bar { \rho } _ { k } } ( \mathbb { P } _ { k } ^ { \epsilon } )$ is weakly closed, and thereby weakly compact. This completes the proof. □

## B.2.2 Proof of Lemma 2

Proof of Lemma 2. We prove Lemma 2 in two steps. Specifically, we prove that the loss function $L _ { k }$ is uniformly integrable and $\mathbb { E } _ { \omega \sim \mathbb { Q } } \bigg [ L _ { k } ( \phi , \omega ) \bigg ]$ is weakly upper semicontinuous with respect to the ambiguity set $\mathbb { Q } \in \mathbb { B } _ { \hat { \rho } _ { k } } ( \mathbb { P } _ { k } ^ { \epsilon } )$ for each $k \in \mathbb { K }$ and any fixed $\phi ,$ respectively.

Step 1 of Lemma 2: Based on the Donsker-Varadhan variational representation of the KL divergence [47], for any measurable function $f : \Omega \to \mathbb { R }$ and probability distribution $\mathbb { Q } ,$ we have

$$
\mathbb { E } _ { \omega \sim \mathbb { Q } } [ f ] - \log \mathbb { E } _ { \omega \sim \mathbb { P } _ { k } ^ { \epsilon } } [ e ^ { f } ] \le \mathcal { D } _ { \mathrm { K L } } ( \mathbb { Q } , \mathbb { P } _ { k } ^ { \epsilon } ) \le \operatorname* { i n f } _ { \gamma \in \Gamma ( \widehat { \mathbb { P } } _ { k } , \mathbb { Q } ) } \mathbb { E } _ { x \sim \widehat { \mathbb { P } } _ { k } } \left[ \mathcal { D } _ { \mathrm { K L } } ( \gamma _ { x } \| K _ { x , \epsilon } ) \right] \le \frac { \bar { \rho } _ { k } } { \epsilon } , \quad \forall k \in \mathbb { K } .\tag{23}
$$

According to the Donsker-Varadhan variational representation part in relation (23), under ${ \mathrm { A } } s -$ sumption 2(I), for $\beta _ { k } > 0$ , we have

$$
\mathbb { E } _ { \omega \sim \mathbb { Q } } [ L _ { k } ( \phi , \omega ) ] \le \frac { 1 } { \beta } \Big ( \mathcal { D } _ { \mathrm { K L } } ( \mathbb { Q } , \mathbb { P } _ { k } ^ { \epsilon } ) + \log \mathbb { E } _ { \omega \sim \mathbb { P } _ { k } ^ { \epsilon } } [ e ^ { \beta _ { k } \cdot L _ { k } ( \phi , \omega ) } ] \Big ) < \infty , \quad \forall k \in \mathbb { K } ,\tag{24}
$$

which implies that the inner problem of (SDRHT) is bounded. For a truncation threshold $M > 0$ let $\mathbb { 1 } _ { \{ L _ { k } ( \phi , \omega ) > M \} }$ be an indicator function that equals to 1 if $L _ { k } ( \phi , \omega ) > M$ and 0 otherwise. Based on relation (23), we also have

$$
\mathbb { E } _ { \omega \sim \mathbb Q } [ L _ { k } ( \phi , \omega ) \cdot \mathbb { 1 } _ { \{ L _ { k } ( \phi , \omega ) > M \} } ] \le \frac { 1 } { \beta } \Big ( \mathcal D _ { \mathrm { K L } } ( \mathbb Q , \mathbb P _ { k } ^ { \varepsilon } ) + \log \mathbb E _ { \omega \sim \mathbb P _ { k } ^ { \epsilon } } \Big [ e ^ { \beta _ { k } \cdot L _ { k } ( \phi , \omega ) \cdot \mathbb { 1 } _ { \{ L _ { k } ( \phi , \omega ) > M \} } } \Big ] \Big ) , \quad \forall k \in \mathbb K ,
$$

and it follows that

$$
\operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \hat { \rho } _ { k } } \left( \mathbb { P } _ { k } ^ { e } \right) } \mathbb { E } _ { \omega \sim \mathbb { Q } } [ L _ { k } ( \phi , \omega ) \cdot \mathbb { 1 } _ { \{ L _ { k } ( \phi , \omega ) > M \} } ] \leq \frac { \bar { \rho } _ { k } } { \epsilon \beta } + \frac { 1 } { \beta } \log \mathbb { E } _ { \omega \sim \mathbb { P } _ { k } ^ { e } } \Big [ e ^ { \beta _ { k } \cdot L _ { k } ( \phi , \omega ) \cdot \mathbb { 1 } _ { \{ L _ { k } ( \phi , \omega ) > M \} } } \Big ] , \quad \forall k \in \mathbb { K } .
$$

When $M  \infty$ , we have $\mathbb { 1 } _ { \{ L _ { k } ( \phi , \omega ) > M \} } \to 0$ for $\mathbb { P } _ { k } ^ { \epsilon }$ -almost every $\omega .$ . Moreover, as both $\beta _ { k }$ and $L _ { k } ( \phi , \omega )$ are non-negative, we have

$$
e ^ { \beta _ { k } \cdot L _ { k } ( \phi , \omega ) \mathbb { 1 } _ { \{ L _ { k } ( \phi , \omega ) > M \} } } \leq e ^ { \beta _ { k } \cdot L _ { k } ( \phi , \omega ) } , \quad \forall k \in \mathbb { K } ,
$$

where $e ^ { \beta _ { k } \cdot L _ { k } \left( \phi , \omega \right) }$ is integrable under Assumption 2(I). Hence, by the dominated convergence theorem,

$$
\operatorname* { l i m } _ { M \to \infty } \log \mathbb { E } _ { \omega \sim \mathbb { P } _ { k } ^ { \epsilon } } \bigg [ e ^ { \beta _ { k } \cdot L _ { k } ( \phi , \omega ) \cdot \mathbb { 1 } _ { \{ L _ { k } ( \phi , \omega ) > M \} } } \bigg ] = \log \mathbb { E } _ { \omega \sim \mathbb { P } _ { k } ^ { \epsilon } } \bigg [ e ^ { 0 } \bigg ] = 0 , \quad \forall k \in \mathbb { K } .
$$

Thus, we have

$$
\operatorname* { l i m } _ { M  \infty } \operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \bar { \rho } _ { k } } ( \mathbb { P } _ { k } ^ { \epsilon } ) } \mathbb { E } _ { \omega \sim \mathbb { Q } } \Bigl [ L _ { k } ( \phi , \omega ) \cdot \mathbb { 1 } _ { \{ L _ { k } ( \phi , \omega ) > M \} } \Bigr ] \leq \frac { \bar { \rho } _ { k } } { \epsilon \beta _ { k } } , \quad \forall k \in \mathbb { K } .\tag{25}
$$

Since Assumption 2(I) guarantees that this property holds for any $\beta _ { k } > 0$ , let $\beta _ { k } \to \infty$ , and then we have

$$
\operatorname* { l i m } _ { M \to \infty } \operatorname* { s u p } _ { \mathbb Q \in \mathbb B _ { \tilde { \rho } _ { k } } ( \mathbb P _ { k } ^ { \epsilon } ) } \mathbb E _ { \omega \sim \mathbb Q } \Big [ L _ { k } ( \phi , \omega ) \cdot \mathbb { 1 } _ { \{ L _ { k } ( \phi , \omega ) > M \} } \Big ] \leq \operatorname* { i n f } _ { \beta > 0 } \frac { \bar { \rho } _ { k } } { \epsilon \beta _ { k } } = 0 , \quad \forall k \in \mathbb K ,\tag{26}
$$

which completes the proof of uniform integrability.

Step 2 of Lemma 2: We next prove the weak upper semicontinuity of $\mathbb { E } _ { \omega \sim \mathbb { Q } } [ L _ { k } ( \phi , \omega ) ]$ in $\mathbb { Q } \in$ $\mathbb { B } _ { \bar { \rho } _ { k } } ( \mathbb { P } _ { k } ^ { \epsilon } )$ . We consider a continuous detector $\phi$ and a non-negative, non-decreasing, convex, and continuous generating function ℓ. Then, the loss functions $L _ { 0 }$ and $L _ { 1 }$ are non-negative and upper semicontinuous for all $k \in \mathbb { K } .$ . By Lemma 1, the ambiguity set $\mathbb { B } _ { \bar { \rho } _ { k } } ( \mathbb { P } _ { k } ^ { \epsilon } )$ is weakly closed. For each $k \in \mathbb { K }$ let $\mathbb { Q } _ { j } \in \mathbb { B } _ { \bar { \rho } _ { k } } ( \mathbb { P } _ { k } ^ { \epsilon } )$ for all $j \in \mathbb { N }$ be any sequence of distributions that converges weakly to $\mathbb { Q } \mathrm { , }$ and then we have

$$
\begin{array} { r l } {  { \operatorname* { l i m s u p } _ { j \in N } \mathbb { E } _ { \omega \sim \mathbb { Q } _ { j } } \Big [ L _ { k } ( \phi , \omega ) \Big ] } } \\ & { \leq \operatorname* { l i m s u p } _ { j \in N } \mathbb { E } _ { \omega \sim \mathbb { Q } _ { j } } \Big [ \operatorname* { m i n } \{ L _ { k } ( \phi , \omega ) , M \} \Big ] + \operatorname* { l i m s u p } _ { j \in N } \mathbb { E } _ { \omega \sim \mathbb { Q } _ { j } } \Big [ ( L _ { k } ( \phi , \omega ) - M ) \cdot \mathbb { 1 } _ { \{ L _ { k } ( \phi , \omega ) > M \} } \Big ] } \\ & { \leq \mathbb { E } _ { \omega \sim \mathbb { Q } } \Big [ \operatorname* { m i n } \{ L _ { k } ( \phi , \omega ) , M \} \Big ] + \operatorname* { l i m s u p } _ { j \in N } \mathbb { E } _ { \omega \sim \mathbb { Q } _ { j } } \Big [ ( L _ { k } ( \phi , \omega ) - M ) \cdot \mathbb { 1 } _ { \{ L _ { k } ( \phi , \omega ) > M \} } \Big ] } \\ & { \leq \mathbb { E } _ { \omega \sim \mathbb { Q } } \Big [ \operatorname* { m i n } \{ L _ { k } ( \phi , \omega ) , M \} \Big ] + \operatorname* { s u p } _ { 0 \in \mathbb { P } _ { k } ( \mathbb { P } _ { k } ^ { c } ) } \mathbb { E } _ { \omega \sim \mathbb { Q } } [ L _ { k } ( \phi , \omega ) \cdot \mathbb { 1 } _ { \{ L _ { k } ( \phi , \omega ) > M \} } ] , \quad \forall k \in \mathbb { K } , } \end{array}\tag{27}
$$

where $M > 0$ , the second inequality is due to Proposition 3.3 in [7] for the upper semicontinuous and bounded function min $\{ L _ { k } ( \phi , \omega ) , M \}$ , and the third inequality follows because $\mathbb { Q } _ { j } \in \mathbb { B } _ { \bar { \rho } _ { k } } ( \mathbb { P } _ { k } ^ { \epsilon } )$ for

any $j \in \mathbb { N }$ and $( L _ { k } ( \phi , \omega ) - M ) \cdot \mathbb { 1 } _ { \{ L _ { k } ( \phi , \omega ) > M \} } \leq L _ { k } ( \phi , \omega ) \cdot \mathbb { 1 } _ { \{ L _ { k } ( \phi , \omega ) > M \} }$ . Taking the limit as $M \to \infty$ on both sides of inequality (27), we obtain

$$
\begin{array} { r l } & { \underset { j \in N } { \operatorname* { l i m } } \operatorname* { s u p } \mathbb { E } _ { \omega \sim \mathbb { Q } _ { j } } \Big [ L _ { k } ( \phi , \omega ) \Big ] } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad  \end{array}
$$

where the first equality follows from Eq. (26) and the second equality follows from the monotone convergence theorem. This completes the proof. □

## B.2.3 Formal Proof of Theorem 1

Proof of Theorem 1. For Theorem 1(I), the problem (SDRHT) can be reformulated as

$$
\operatorname* { i n f } _ { \phi \in \mathcal { F } _ { \mathbb { P } _ { k } \in \mathbb { B } _ { \rho _ { k } } ( \widehat { \mathbb { P } } _ { k } ) , \forall k \in \mathbb { K } } } \quad \sum _ { k \in \mathbb { K } } \mathbb { E } _ { \omega \sim \mathbb { P } _ { k } } \Big [ L _ { k } ( \phi , \omega ) \Big ] .
$$

Note that $\mathbb { B } _ { \bar { \rho } _ { k } } ( \mathbb { P } _ { k } ^ { \epsilon } )$ is weakly compact for any $\bar { \rho } _ { k } \ge 0$ thanks to Proposition 1 and Lemma 1, which applies because of Assumption 1. In addition, $\mathbb { E } _ { \omega \sim \mathbb { Q } } \bigg [ L _ { k } ( \phi , \omega ) \bigg ]$ is non-negative and weakly upper semicontinuous in $\mathbb { Q } \in \mathbb { B } _ { \bar { \rho } _ { k } } ( \mathbb { P } _ { k } ^ { \epsilon } )$ for all $k \in \mathbb { K }$ and $\phi \bar { \in \mathcal { F } } .$ , due to Definition 1 and Lemma 2. As $\Phi ( \phi ; \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } )$ is convex in $\phi$ and concave in distributions, Theorem 1(II) in [46], an extension of the lopsided minimax theorem $[ 7 ,$ Theorem 5.15], ensures that

$$
\mathrm { O p t i m a l ~ V a l u e ~ o f ~ ( S D R H T ) } = \operatorname* { m a x } _ { \mathbb { P } _ { 0 } \in \mathbb { B } _ { \rho _ { 0 } } ( \widehat { \mathbb { P } } _ { 0 } ) , \mathbb { P } _ { 1 } \in \mathbb { B } _ { \rho _ { 1 } } ( \widehat { \mathbb { P } } _ { 1 } ) } \operatorname* { i n f } _ { \phi \in \mathcal { F } } \Phi ( \phi ; \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) .
$$

For Theorem 1(II), the problem (soft-SDRHT) can be reformulated as

$$
\operatorname* { i n f } _ { \phi \in \mathscr { F } _ { \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } \in \mathscr { P } ( \Omega ) } } \quad \sum _ { k \in \mathbb { K } } \mathbb { E } _ { \omega \sim \mathbb { P } _ { k } } \Big [ L _ { k } ( \phi , \omega ) \Big ] - \lambda _ { k } { S _ { \epsilon } ( \widehat { \mathbb { P } } _ { k } , \mathbb { P } _ { k } ) } ,
$$

where $\lambda _ { k } > 0$ for any $k \in \mathbb { K }$ . Define

$$
H ( \phi , \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) : = \sum _ { k \in \mathbb { K } } \mathbb { E } _ { \omega \sim \mathbb { P } _ { k } } \Big [ L _ { k } ( \phi , \omega ) \Big ] - \lambda _ { k } \mathcal { S } _ { \epsilon } ( \widehat { \mathbb { P } } _ { k } , \mathbb { P } _ { k } ) .
$$

We next verify the level compactness of $H ( \phi , \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } )$ required for the lopsided minimax theo rem [7, Theorem 5.15]. By Eqs. (21), (23), and (24), for any fixed $\phi .$ , we have

$$
H ( \phi , \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) \leq \sum _ { k \in \mathbb { K } } \bigg [ C _ { k } + \bigg ( \frac { 1 } { \beta _ { k } } - \lambda _ { k } \epsilon \bigg ) \cdot \operatorname* { i n f } _ { \gamma \in \Gamma ( \widehat { \mathbb { P } } _ { k } , \mathbb { P } _ { k } ) } \mathbb { E } _ { x \sim \widehat { \mathbb { P } } _ { k } } [ \mathcal { D } _ { \mathrm { K L } } ( \gamma _ { x } | | \mathcal { K } _ { x , \epsilon } ) ] \bigg ] ,\tag{28}
$$

where $\beta _ { k } > 0$ is finite, $\begin{array} { r } { C _ { k } : = \frac { 1 } { \beta _ { k } } \log \mathbb { E } _ { \omega \sim \mathbb { P } _ { k } ^ { \epsilon } } [ e ^ { \beta _ { k } \cdot L _ { k } ( \phi , \omega ) } ] + \lambda _ { k } \epsilon \cdot \mathbb { E } _ { x \sim \widehat { \mathbb { P } } _ { k } } [ \log \int _ { \mathbb { R } ^ { d } } e ^ { - c ( { x } , { z } ) / \epsilon } \mathrm { d } { z } ] } \end{array}$ for any $k \in \mathbb { K }$ is finite by Assumptions 1(II) and $2 ( \mathrm { I } )$ , and is independent of $\mathbb { P } _ { 0 }$ and $\mathbb { P } _ { 1 }$ .

By choosing $\textstyle \beta _ { k } \in ( \frac { 1 } { \lambda _ { k } \epsilon } , + \infty )$ for each $k \in \mathbb { K } ,$ , when in $\mathrm { f } _ { \gamma \in \Gamma ( \widehat { \mathbb { P } } _ { k } , \mathbb { P } _ { k } ) } \mathbb { E } _ { \pmb { x } \sim \widehat { \mathbb { P } } _ { k } } \Big [ { \mathcal { D } } _ { \mathrm { K L } } ( \gamma _ { \pmb { x } } \| { \mathcal { K } } _ { \pmb { x } , \epsilon } ) \Big ]  \infty$ , we have $H ( \phi , \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) \to - \infty$ . This implies that for any $\alpha \in \mathbb { R }$ , there exist finite radii $r _ { 0 }$ and $r _ { 1 }$ such that the upper-level set of $H ( \phi , \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } )$ satisfies

$$
\{ ( \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) : H ( \phi , \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) \geq \alpha \} \subseteq \mathbb { B } _ { r _ { 0 } } ( \widehat { \mathbb { P } } _ { 0 } ) \times \mathbb { B } _ { r _ { 1 } } ( \widehat { \mathbb { P } } _ { 1 } ) .
$$

By Lemma 1, the set $\mathbb { B } _ { r _ { 0 } } ( \widehat { \mathbb { P } } _ { 0 } ) \times \mathbb { B } _ { r _ { 1 } } ( \widehat { \mathbb { P } } _ { 1 } )$ is weakly compact. For each $k \in \mathbb { K } ,$ the Sinkhorn discrepancy is convex and weakly lower semicontinuous (see the proof of Lemma 1), and $\mathbb { E } _ { \omega \sim \mathbb { P } _ { k } } \bigg [ L _ { k } ( \phi , \omega ) \bigg ]$ is weakly upper semicontinuous in $\mathbb { P } _ { k }$ by Lemma 2. Hence, $H ( \phi , \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } )$ is weakly upper semicontinuous and its upper-level set $\{ ( \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) : H ( \phi , \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) \ge \alpha \}$ is weakly closed. Therefore, for every $\alpha \in \mathbb { R }$ and every fixed $\phi \in { \mathcal { F } }$ , the upper-level set $\{ ( \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) : H ( \phi , \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) \ge \alpha \}$ , a closed subset of a weakly compact set, is also weakly compact. In addition $, - H ( \phi , \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } )$ is convex, weakly lower semicontinuous, and thus weakly closed in $\mathbb { P } _ { 0 }$ and $\mathbb { P } _ { 1 }$

Finally, $H ( \cdot , \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } )$ is convex in $\phi .$ , and Assumption 2(II) ensures that

$$
\begin{array} { r l } { \underset { \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } } { \operatorname* { s u p } } \underset { \phi \in \mathcal { F } } { \operatorname* { i n f } } } & { { } H ( \phi , \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) > - \infty . } \end{array}
$$

Therefore, as all conditions of the lopsided minimax theorem are satisfied, the strong duality of the problem (soft-SDRHT) holds, i.e.,

$$
\mathrm { O p t i m a l \ V a l u e \ o f \ ( s o f t { \cdot } S D R H T ) } = \operatorname* { s u p } _ { \mathbb P _ { 0 } , \mathbb P _ { 1 } \in \mathcal P ( \Omega ) } \operatorname* { i n f } _ { \phi \in \mathcal F } \Phi ( \phi ; \mathbb P _ { 0 } , \mathbb P _ { 1 } ) - \sum _ { k \in \mathbb K } \lambda _ { k } \mathcal S _ { \epsilon } ( \widehat { \mathbb P } _ { k } , \mathbb P _ { k } ) .\tag{29}
$$

Further, since in $\mathrm { f } _ { \phi \in \mathcal { F } } H ( \phi , \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } )$ is weakly upper semicontinuous in $\mathbb { P } _ { 0 }$ and $\mathbb { P } _ { 1 }$ , has a finite upper bound by Eq. (28) under proper $\beta ,$ , and has weakly compact upper-level sets, the supremum in Eq. (29) is attained by Weierstrass’ theorem, i.e.,

$$
\mathrm { O p t i m a l ~ V a l u e ~ o f ~ ( s o f t \mathrm { - } S D R H T ) = \operatorname* { m a x } _ { \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } \in \mathcal { P } ( \Omega ) } \operatorname* { i n f } _ { \phi \in \mathcal { F } } \Phi ( \phi ; \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) - \sum _ { k \in \mathbb { K } } \lambda _ { k } \mathcal { S } _ { \epsilon } ( \widehat { \mathbb { P } } _ { k } , \mathbb { P } _ { k } ) . }
$$

This completes the proof.

## B.3 Proof of Theorem 3

Proof of Theorem 3. As a preparation, when D has a finite diameter D, for any $\upsilon \in ( 0 , 1 )$ , there exists a υ-net $\mathbb { D } _ { \nu } \subset \mathbb { D }$ containing J points such that $\forall \pmb { x } \in \mathbb { D } , \exists \pmb { x } _ { j } \in \mathbb { D } _ { \nu } , \| \pmb { x } - \pmb { x } _ { j } \| \leq \nu$ . According to [48], the size of the υ-net is bounded by $J \le { \cal { O } } ( ( D / \nu ) ^ { d } )$

We prove Theorem 3(I) in two steps. According to Lemma 1 in [40], all continuous Lipschitz convex functions over convex compact sets can be approximated using the maximum of afine functions. Thus, in Step 1, we control the discretization error by adjusting the size J of the υ- net. In Step 2, we prove that a HyCNN with maxout activation functions can exactly represent a maximum of afine functions and determine the quantitative relationship between J and the network hyperparameters (L,h).

Step 1 of Theorem 3(I): For any L<sub>f</sub>-Lipschitz function $f ,$ we approximate it by the maximum of supporting hyperplanes at points in the υ-net. The supporting hyperplane at $( \pmb { x } _ { j } , f ( \pmb { x } _ { j } ) )$ is defined as

$$
\begin{array} { r } { s _ { j } ( { \pmb x } ) : = { \pmb g } _ { j } ^ { \top } ( { \pmb x } - { \pmb x } _ { j } ) + f ( { \pmb x } _ { j } ) . } \end{array}
$$

where $\pmb { g } _ { j } \in \partial f ( \pmb { x } _ { j } )$ . For any $\pmb { x } \in \mathbb { D }$ , let $\boldsymbol { x } _ { j _ { 0 } } \in \mathbb { D } _ { \boldsymbol { \imath } }$ be the nearest point to x in the υ-net. The pointwise error satisfies

$$
\begin{array} { r l } & { \underset { x \in \mathbb { D } } { \operatorname* { s u p } } \bigg | f ( x ) - \underset { j \in [ J ] } { \operatorname* { m a x } } s _ { j } ( x ) \bigg | = \underset { x \in \mathbb { D } } { \operatorname* { s u p } } \bigg ( f ( x ) - \underset { j \in [ J ] } { \operatorname* { m a x } } s _ { j } ( x ) \bigg ) } \\ & { \qquad \leq \underset { x \in \mathbb { D } } { \operatorname* { s u p } } \bigg ( f ( x ) - s _ { j _ { 0 } } ( x ) \bigg ) } \\ & { \qquad = \underset { x \in \mathbb { D } } { \operatorname* { s u p } } \bigg ( f ( x ) - f ( x _ { j _ { 0 } } ) - g _ { j } ^ { \top } ( x - x _ { j _ { 0 } } ) \bigg ) } \\ & { \qquad \leq \underset { x \in \mathbb { D } } { \operatorname* { s u p } } \bigg ( | f ( x ) - f ( x _ { j _ { 0 } } ) | + \| g _ { j } \| \cdot \| x - x _ { j _ { 0 } } \| \bigg ) } \\ & { \qquad \leq 2 L _ { f } \| x - x _ { j _ { 0 } } \| \leq 2 L _ { f } \nu . } \end{array}
$$

Here, the first equality holds since f is convex, implying $s _ { j } ( { \pmb x } ) \le f ( { \pmb x } )$ for all j. The first inequality follows because ma $\mathfrak { c } _ { j \in [ J ] } s _ { j } ( \pmb { x } ) \geq s _ { j _ { 0 } } ( \pmb { x } )$ . The second inequality uses the triangle inequality and Cauchy-Schwarz inequality. The third inequality applies the L -Lipschitz continuity of $f _ { \mathrm { ~ , ~ } }$ which bounds both $| f ( \pmb { x } ) - f ( \pmb { x } _ { j _ { 0 } } ) | \leq L _ { f } \| \pmb { x } - \pmb { x } _ { j _ { 0 } } \|$ and $\| g _ { j } \| \leq L _ { f }$ . The last inequality holds as $\| \pmb { x } - \pmb { x } _ { j _ { 0 } } \| \le \nu$ To ensure $ \begin{array} { r } { \operatorname { : u p } _ { { \pmb x } \in \mathbb { D } } | f ( { \pmb x } ) - \operatorname* { m a x } _ { j \in [ J ] } s _ { j } ( { \pmb x } ) | \leq \varepsilon } \end{array}$ for any $\varepsilon > 0$ , we have $\begin{array} { r } { \nu \leq \operatorname* { m i n } \{ \frac { \varepsilon } { 2 L _ { f } } , 1 \} } \end{array}$ . Taking $\upsilon = \varepsilon / 2 L _ { f }$ By $\begin{array} { r } { J \leq \mathcal { O } ( ( \frac { D } { \nu } ) ^ { d } ) } \end{array}$ , the number of points in the υ-net is given by $J \leq { \cal O } ( ( \frac { D L _ { f } } { \varepsilon } ) ^ { d } )$

Step 2 of Theorem 3(I): Here, we show that the maximum of J afine functions can be represented by a HyCNN with L layers and width h. As an intuitive example, we first consider a maximum of two afine functions and reformulate it as a HyCNN with two layers ${ \bf { \sigma } } \mathbf { { \boldsymbol { y } } } _ { 1 }$ and $\pmb { y } _ { 2 }$ (see Figure 2) by introducing an extra afine term:

$$
\operatorname* { m a x } \left\{ a _ { 1 } ^ { \top } x + b _ { 1 } , a _ { 2 } ^ { \top } x + b _ { 2 } \right\} = \underbrace { ( a _ { 3 } ^ { \top } x + b _ { 3 } ) + \overbrace { \operatorname* { m a x } \left\{ ( a _ { 1 } - a _ { 3 } ) ^ { \top } x + ( b _ { 1 } - b _ { 3 } ) , ( a _ { 2 } - a _ { 3 } ) ^ { \top } x + ( b _ { 2 } - b _ { 3 } ) \right\} } ^ { y _ { 1 } } } _ { u : }
$$

We extend this intuition to a maximum of J afine functions max $\{ { \pmb a } _ { 1 } ^ { \top } { \pmb x } + b _ { 1 } , \cdots , { \pmb a } _ { I } ^ { \top } { \pmb x } + b _ { J } \}$ . We consider a simple case in which the width of each hidden layer is $h = 1$ . In the following, we first construct a HyCNN with maxout activation functions that use both positive and negative weights, representing a maximum over afine functions. Then, all weights can be restricted to be nonnegative by the method in the proof of Theorem 1 in [40] without changing the width and depth of the network. Let $s _ { j } : = \pmb { a } _ { j } ^ { \top } \pmb { x } + b _ { j }$ for any $j \in [ 2 J ]$ , extra afine terms $r _ { j } : = \pmb { c } _ { j } ^ { \top } \pmb { x } + d _ { j }$ for any

$$
j \in [ 2 , \cdots , J ] , s _ { j } ^ { \prime } = s _ { j } - r _ { j }
$$

$$
j \in [ 2 , \cdots , J ]
$$

$$
\widehat { \sigma } : \mathbb { R } ^ { 2 }  \mathbb { R }
$$

$$
\begin{array} { r l } & { \operatorname* { m a x } \{ s _ { 1 } , \cdots , s _ { J } \} } \\ & { = \operatorname* { m a x } \{ \operatorname* { m a x } \{ s _ { 1 } , \cdots , s _ { J - 1 } \} , s _ { J } \} } \\ & { = r _ { J } + \widehat { \sigma } ( \operatorname* { m a x } \{ s _ { 1 } , \cdots , s _ { J - 1 } \} - r _ { J } , s _ { J } ^ { \prime } ) } \\ & { = r _ { J } + \widehat { \sigma } ( \operatorname* { m a x } \{ \operatorname* { m a x } \{ s _ { 1 } , \cdots , s _ { J - 2 } \} , s _ { J - 1 } \} - r _ { J } , s _ { J } ^ { \prime } ) } \\ & { = r _ { J } + \widehat { \sigma } ( r _ { J - 1 } + \widehat { \sigma } ( \operatorname* { m a x } \{ s _ { 1 } , \cdots , s _ { J - 2 } \} - r _ { J - 1 } , s _ { J - 1 } ^ { \prime } ) - r _ { J } , s _ { J } ^ { \prime } ) } \\ & { = \cdots } \\ & { = r _ { J } + \widehat { \sigma } ( r _ { J - 1 } + \widehat { \sigma } ( r _ { J - 2 } + \widehat { \sigma } ( \cdots + r _ { 3 } + \widehat { \sigma } ( r _ { 2 } + \widehat { \sigma } ( s _ { 1 } - r _ { 2 } , s _ { 2 } ^ { \prime } ) - r _ { 3 } , s _ { 3 } ^ { \prime } ) - \cdots ) - r _ { J - 2 } , s _ { J - 1 } ^ { \prime } ) - r _ { J } , s _ { J } ^ { \prime } ) . } \end{array}\tag{30}
$$

The last equation describes a J-layer HyCNN with width $h = 1$ , where the layers are

$$
\begin{array} { r l } & { y _ { 1 } = \widehat { \sigma } ( s _ { 1 } - r _ { 2 } , s _ { 2 } - r _ { 2 } ) , } \\ & { y _ { 2 } = \widehat { \sigma } ( 1 \cdot y _ { 1 } + r _ { 2 } - r _ { 3 } , 0 \cdot y _ { 1 } + s _ { 3 } - r _ { 3 } ) , } \\ & { \quad \cdots } \\ & { y _ { j } = \widehat { \sigma } ( 1 \cdot y _ { j - 1 } + r _ { j } - r _ { j + 1 } , 0 \cdot y _ { j - 1 } + s _ { j + 1 } - r _ { j + 1 } ) , } \\ & { \quad \cdots } \\ & { y _ { J } = r _ { J } + y _ { J - 1 } . } \end{array}
$$

To ensure the coeficients of previous layers in this neural network are nonnegative, following [40], we expand the original input $\pmb { x } \in \mathbb { R } ^ { d }$ in each layer to $\widehat { \pmb { x } } \in \mathbb { R } ^ { 2 d }$ that includes x and −x, i.e., ${ \widehat { \mathbf { x } } } =$ $[ \pmb { x } ^ { \top } , - \pmb { x } ^ { \top } ] ^ { \top }$ . When all coeficients of x are nonnegative, the coeficients of −x are 0. When $h _ { j } < 0$ we set the coeficient of $\widehat { x _ { j + d } } \ \mathrm { t o } \ - h _ { j }$ and that of $x _ { j }$ to 0. Therefore, without loss of generality, we can limit all weights between layers to be nonnegative, thereby ensuring that the neural network is input-convex.

The construction in Eq. (30) represents $\mathrm { m a x } _ { j \in [ J ] } s _ { j } ( { \pmb x } )$ exactly by a HyCNN with width $h = 1$ and depth $L = \mathcal { O } ( J )$ . By Step 1, choosing $h = 1$ and $L = \mathcal { O } ( J )$ gives

$$
L \cdot h = \mathcal { O } ( ( \frac { D L _ { f } } { \varepsilon } ) ^ { d } ) ,
$$

which proves Theorem 3(I).

For Theorem 3(II), we first focus on the property of the smooth maxout activation function. Without loss of generality, suppose $a _ { 1 } \geq a _ { 2 }$ , we have $\bar { \sigma } ( a _ { 1 } , a _ { 2 } ) = a _ { 1 }$ and

$$
\begin{array} { r } { \bar { \sigma } _ { \tau } ( a _ { 1 } , a _ { 2 } ) = \tau \log \left( e ^ { a _ { 1 } / \tau } ( 1 + e ^ { ( a _ { 2 } - a _ { 1 } ) / \tau } ) \right) = a _ { 1 } + \tau \log \left( 1 + e ^ { ( a _ { 2 } - a _ { 1 } ) / \tau } \right) \leq a _ { 1 } + \tau \log 2 , } \end{array}
$$

where $\tau > 0$ and the last inequality is due to $e ^ { ( a _ { 2 } - a _ { 1 } ) / \tau } \in ( 0 , 1 ]$ . This is equivalent to

$$
\begin{array} { r } { \bar { \sigma } ( a _ { 1 } , a _ { 2 } ) < \bar { \sigma } _ { \tau } ( a _ { 1 } , a _ { 2 } ) \leq \bar { \sigma } ( a _ { 1 } , a _ { 2 } ) + \tau \log 2 . } \end{array}
$$

Consequently, for a HyCNN and its smooth version, we have

$$
\operatorname* { s u p } _ { \mathbf { x } \in \mathbb { D } } \big | \widehat { \mathrm { H y C N N } } ^ { \tau } ( \mathbf { x } ) - \widehat { \mathrm { H y C N N } } ( \mathbf { x } ) \big | \leq C \cdot \tau \log 2
$$

where $C > 0$ is a constant depending on the number of maxout activation functions and layers in the network. By Theorem 3(I), for any $\varepsilon > 0$ , there exists a HyCNN such that

$$
\operatorname* { s u p } _ { \pmb { x } \in \mathbb { D } } | \mathrm { H y } \widetilde { \mathrm { C N N } } ( \pmb { x } ) - f ( \pmb { x } ) | \leq \frac { \varepsilon } { 2 } .
$$

By selecting a suficiently small $\begin{array} { r } { \tau , \mathrm { e . g . } , \tau \le \frac { \varepsilon } { 2 C \log 2 } } \end{array}$ , we have

$$
\operatorname* { s u p } _ { x \in \mathbb { D } } \big \lvert \widehat { \mathrm { H y C N } } \mathsf { N } ^ { \tau } ( x ) - f ( x ) \big \rvert \leq \operatorname* { s u p } _ { x \in \mathbb { D } } \big \lvert \widehat { \mathrm { H y C N } } \mathsf { N } ^ { \tau } ( x ) - \mathsf { H y C N } \mathsf { N } ( x ) \big \rvert + \operatorname* { s u p } _ { x \in \mathbb { D } } \big \lvert \widehat { \mathrm { H y C N } } \mathsf { N } ( x ) - f ( x ) \big \rvert \leq \varepsilon .
$$

Note that the smooth HyCNN and standard HyCNN share the same order of network architecture complexity. Hence, the hyperparameters of smooth HyCNN can also be chosen as $L \cdot h = \mathcal { O } ( ( \frac { D L _ { f } } { \varepsilon } ) ^ { d } )$ to guarantee the ε-approximation. This completes the proof of Theorem 3(II).

For Theorem 3(III), we select a sequence of smooth HyCNNs on a compact domain $[ - n , n ] ^ { d }$ with suficiently small τ and each satisfying $L _ { n } \cdot h _ { n } = O ( ( \frac { D \dot { L } _ { f } } { \varepsilon } ) ^ { d } )$ . By setting $\varepsilon _ { n } = 1 / n$ , according to Theorem 3(II), we have $\begin{array} { r } { \operatorname* { s u p } _ { { \pmb x } \in \mathbb { D } } | { \sf H y C N N } _ { { \varepsilon } _ { n } } ^ { \tau } ( { \pmb x } ) - f ( { \pmb x } ) | \leq \frac { 1 } { n } } \end{array}$ where the right-hand-side term converges to 0 as $n \to \infty$ . Hence, we can prove ${ \widehat { \mathsf { H y C N N } } } _ { n } ^ { \tau }$ converges to $f$ uniformly on D. This completes the proof. □

## B.4 Proof of Theorem 4

Proof of Theorem 4. For Theorem 4(I), one can see Theorem 2 in [34] for a detailed proof.

For Theorem 4(II), let the function f be M-Lipschitz smooth, and then we have

$$
f ( { \pmb y } ) - f ( { \pmb x } ) \leq \nabla f ( { \pmb x } ) ^ { \top } ( { \pmb y } - { \pmb x } ) + \frac { M } { 2 } \cdot \| { \pmb y } - { \pmb x } \| ^ { 2 } .\tag{31}
$$

For any $\widehat { \mathsf { H y C N N } } _ { n } ^ { \tau }$ in the sequence $\{ \mathsf { H } \widetilde { \mathsf { y C N N } } _ { n } ^ { \tau } \} _ { n = 1 } ^ { \infty }$ with hyperparameters satisfying $L _ { n } \cdot h _ { n } = \mathcal { O } ( { \varepsilon _ { n } } ^ { - d } )$ for any $\varepsilon _ { n } > 0$ , we have

$$
\begin{array} { r } { \widehat { \mathsf { H y C N N } } _ { n } ^ { \tau } ( \pmb { y } ) - \widehat { \mathsf { H y C N N } } _ { n } ^ { \tau } ( \pmb { x } ) \geq \nabla \widehat { \mathsf { H y C N N } } _ { n } ^ { \tau } ( \pmb { x } ) ^ { \top } ( \pmb { y } - \pmb { x } ) , \quad \forall \pmb { x } , \pmb { y } \in \mathbb { R } ^ { d } . } \end{array}\tag{32}
$$

By Eqs. (31) and (32), we have

$$
\begin{array} { r } { \left( \nabla \widehat { \mathrm { H y C N } } \mathrm { n } _ { n } ^ { \tau } ( x ) - \nabla f ( x ) \right) ^ { \top } ( y - x ) \leq \left( \widehat { \mathrm { H y C N } } \mathrm { n } _ { n } ^ { \tau } ( y ) - f ( y ) \right) - \left( \widehat { \mathrm { H y C N } } \mathrm { n } _ { n } ^ { \tau } ( x ) - f ( x ) \right) + \frac { M } { 2 } \cdot \| y - x \| ^ { 2 } . } \end{array}
$$

Let y be obtained by moving x along the direction $\widehat { \nabla \mathsf { H y C N N } } _ { n } ^ { \tau } ( { \pmb x } ) - \nabla f ( { \pmb x } )$ with a proper step length η, i.e., $\begin{array} { r } { \pmb { y } - \pmb { x } = \eta \frac { \nabla \mathsf { H y C N N } _ { n } ^ { \tau } ( \pmb { x } ) - \nabla f ( \pmb { x } ) } { \lVert \nabla \mathsf { H y C N N } _ { n } ^ { \tau } ( \pmb { x } ) - \nabla f ( \pmb { x } ) \rVert } } \end{array}$ and $\pmb { y } \in \mathbb { D }$ . Then, for the point-wise supremum value of $\widehat { \nabla \mathsf { H y C N N } } _ { n } ^ { \tau } ( { \pmb x } ) - \nabla f ( { \pmb x } )$ on compact set $\mathbb { D } \subset \mathbb { R } ^ { d }$ , we have

$$
\begin{array} { r l } & { \displaystyle \operatorname* { s u p } _ { x \in \mathbb { D } } \Big \| \widehat { \nabla \mathrm { H i y Z N N } } _ { n } ^ { \tau } ( x ) - \nabla f ( x ) \Big \| \leq \displaystyle \frac { 1 } { \eta } \cdot \operatorname* { s u p } _ { x \in \mathbb { D } } \Big ( \mathrm { \mathrm { i } } \widehat { \mathrm { i } \mathrm { y Z C N N } } _ { n } ^ { \tau } ( y ) - f ( y ) - \Big ( \mathrm { \mathrm { i } } \widehat { \mathrm { y Z C N N } } _ { n } ^ { \tau } ( x ) - f ( x ) \Big ) \Big ) + \frac { M } { 2 \eta } \cdot \| y - x \| ^ { 2 } } \\ & { \quad \quad \leq \displaystyle \frac { 2 \varepsilon _ { n } } { \eta } + \frac { M \eta } { 2 } \leq 2 \sqrt { M \varepsilon _ { n } } = \mathcal { O } ( \sqrt { \varepsilon _ { n } } ) , } \end{array}
$$

where the second inequality is due to Theorem 3(II) and $\| \pmb { y } - \pmb { x } \| ^ { 2 } = \eta ^ { 2 } .$ , the last inequality is due to the basic inequality, and the last equality is due to $L _ { n } \cdot h _ { n } = \mathcal { O } ( { \varepsilon _ { n } } ^ { - d } )$ for any $\varepsilon _ { n } > 0$ . Hence, the gradient sequence ${ \widehat { \nabla { \mathsf { H y C N N } } } } _ { n }$ converges to $\nabla f$ uniformly at a rate of $O \left( { \sqrt { \varepsilon _ { n } } } \right)$ when function f is Lipschitz smooth. This completes the proof. □

## B.5 Proof of Theorem 5

Proof of Theorem 5. Since $\mu$ is absolutely continuous and both $\mu$ and ν have finite second moments, Brenier’s theorem yields a finite convex function $f :  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ such that

$$
( \nabla f ) _ { \# } \mu = \nu .
$$

Since every finite convex function on $\mathbb { R } ^ { d }$ is locally Lipschitz, for every $n \in \mathbb { N } , f$ is Lipschitz on a neighborhood of the compact set

$$
\begin{array} { r } { \mathbb { D } _ { n } = [ - n , n ] ^ { d } . } \end{array}
$$

By Theorem 3 (II), there exists a smooth HyCNN $\widehat { \mathsf { H y C N N } } _ { n } ^ { \tau _ { n } }$ satisfying

$$
\operatorname* { s u p } _ { \pmb { x } \in \mathbb { D } _ { n } } \left| \mathrm { H y C N N } _ { n } ^ { \pmb { \tau } _ { n } } ( \pmb { x } ) - f ( \pmb { x } ) \right| \leq \frac { 1 } { n } .
$$

For any fixed compact set $\mathbb { K } \subset \mathbb { R } ^ { d }$ , there exists N such that $\mathbb { K } \subseteq \mathbb { D } _ { n }$ for every $n \geq N$ . Hence,

$$
\operatorname* { s u p } _ { \pmb { x } \in \mathbb { K } } | \pmb { \mathrm { H y C N N } } _ { n } ^ { \tau _ { n } } ( \pmb { x } ) - \pmb { f } ( \pmb { x } ) | \leq \frac { 1 } { n }  0 .
$$

Thus, $\mathsf { H y C N N } _ { n } ^ { \tau _ { n } } \to f$ locally uniformly, and in particular pointwise on $\mathbb { R } ^ { d }$

By Theorem 4 (I),

$$
\widehat { \nabla \mathsf { H y C N N } } _ { n } ^ { \tau _ { n } } ( { \pmb x } ) \to \nabla f ( { \pmb x } )
$$

at almost every $\pmb { x } \in \mathbb { R } ^ { d }$ . Since $\mu$ is absolutely continuous with respect to the Lebesgue measure, this implies the weak convergence of the pushforward measure holds $\mu \cdot$ -almost everywhere, i.e., for $X \sim \mu ,$

$$
\widehat { \nabla \mathsf { H y C N N } } _ { n } ^ { \tau _ { n } } ( X ) \to \nabla f ( X ) \qquad \mathrm { a l m o s t ~ s u r e l y } .
$$

Consequently,

$$
\begin{array} { r } { ( \widehat { \nabla \mathsf { H } \mathsf { y } \mathsf { C N } \mathsf { N } } _ { n } ^ { \tau _ { n } } ) _ { \# } \mu \Rightarrow ( \nabla f ) _ { \# } \mu = \nu , } \end{array}
$$

where ⇒ denotes weak convergence of probability measures on $\mathbb { R } ^ { d }$ . This completes the proof.

## C Technical Note for Training and Testing HyCNN

In this section, we provide details to compute the gradients of the problem (7) in Algorithm 1.

## C.1 Gradient of the Risk Function Estimation

In this subsection, we take the $k = 0$ term in Eq. (6) as an example. For brevity, we omit the superscript (i). Denote the parameters of the neural network ${ \mathsf { H y C N N } } _ { k }$ as $\pmb { \theta } _ { k }$ for each $k \in \mathbb { K } .$ Since our approximated convex function $\widetilde { \varphi _ { k } }$ is strongly convex, we have det $\nabla ^ { 2 } \widetilde { \varphi _ { k } } ( z ) > 0$ for any k and z.

(I) Gradient with respect to $\pmb \theta _ { 0 } .$ For brevity, let $\begin{array} { r } { h ( \pmb { \theta } _ { 0 } , \pmb { \theta } _ { 1 } ) : = \exp \left[ \frac { 1 } { 2 } ( \log p _ { 1 } ( \omega _ { 0 } ) - \log p _ { 0 } ( \omega _ { 0 } ) ) \right] . } \end{array}$ , which is the same as $h _ { 0 } ( \omega _ { 0 } , \pmb { \theta } _ { 0 } , \pmb { \theta } _ { 1 } )$ in Algorithm 3. Then, we have

$$
\frac { \partial h ( \theta _ { 0 } , \theta _ { 1 } ) } { \partial \theta _ { 0 } } = \frac { h ( \theta _ { 0 } , \theta _ { 1 } ) } { 2 } \cdot \bigg [ \frac { \partial \log p _ { 1 } ( \omega _ { 0 } ) } { \partial \theta _ { 0 } } - \frac { \partial \log p _ { 0 } ( \omega _ { 0 } ) } { \partial \theta _ { 0 } } \bigg ] .\tag{33}
$$

For the second item on the right-hand side of Eq. (33), based on Eq. (8), we have

$$
\frac { \partial \log p _ { 0 } ( \omega _ { 0 } ) } { \partial \theta _ { 0 } } = \frac { \partial } { \partial \theta _ { 0 } } \log p _ { 0 } ^ { \epsilon } ( z _ { 0 } ) - \frac { \partial } { \partial \theta _ { 0 } } \log \operatorname* { d e t } \nabla ^ { 2 } \widetilde { \varphi } _ { 0 } ( z _ { 0 } ) = - \frac { \partial } { \partial \theta _ { 0 } } \log \operatorname* { d e t } \nabla ^ { 2 } \widetilde { \varphi } _ { 0 } ( z _ { 0 } ) ,\tag{34}
$$

where $z _ { 0 } = \nabla \varphi _ { 0 } ^ { - 1 } ( \omega _ { 0 } )$ , the first equality is due to the strong convexity of $\widetilde { \varphi } _ { 0 }$ , and the second equality is due to $\begin{array} { r } { \frac { \partial } { \partial \pmb { \theta } _ { 0 } } \log p _ { 0 } ^ { \epsilon } ( z _ { 0 } ) = 0 } \end{array}$ . The log-determinant-Hessian in Eq. (34) can be eficiently computed by Algorithm 2.

For the first item on the right-hand side of Eq. (33), based on the chain rule, we have

$$
\frac { \partial \log p _ { 1 } ( \omega _ { 0 } ) } { \partial \theta _ { 0 } } = \left( \frac { \partial \omega _ { 0 } } { \partial \theta _ { 0 } } \right) ^ { \top } \left( \frac { \partial \bar { z } _ { 0 } } { \partial \omega _ { 0 } } \right) ^ { \top } \frac { \partial \log p _ { 1 } ( \omega _ { 0 } ) } { \partial \bar { z } _ { 0 } } .\tag{35}
$$

Let vector

$$
g : = \frac { \partial \log p _ { 1 } ( \omega _ { 0 } ) } { \partial \bar { z } _ { 0 } } = \frac { \partial \bigg [ \log p _ { 1 } ^ { \epsilon } ( \bar { z } _ { 0 } ) - \log \operatorname* { d e t } \nabla ^ { 2 } \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) \bigg ] } { \partial \bar { z } _ { 0 } } ,
$$

and this derivative can be computed by automatic diferentiation. Let $\bar { z } _ { 0 } = \nabla \widetilde { \varphi } _ { 1 } ^ { - 1 } ( \omega _ { 0 } )$ . As $\begin{array} { r } { \frac { \partial \bar { z } _ { 0 } } { \partial \omega _ { 0 } } = [ \nabla ^ { 2 } \varphi _ { 1 } ( \bar { z } _ { 0 } ) ] ^ { - 1 } } \end{array}$ , Eq.(35) can be reformulated as

$$
\frac { \partial \log p _ { 1 } ( \omega _ { 0 } ) } { \partial \pmb { \theta } _ { 0 } } = \left( \frac { \partial \omega _ { 0 } } { \partial \pmb { \theta } _ { 0 } } \right) ^ { \top } [ \nabla ^ { 2 } \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) ] ^ { - 1 } \pmb { g } ,\tag{36}
$$

where $[ \nabla ^ { 2 } \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) ] ^ { - 1 } \pmb { g }$ can be eficiently solved as the minimizer of a quadratic optimization problem similar to (15) without explicitly computing the inverse of the Hessian matrix. Then, $\left( \frac { \partial \omega _ { 0 } } { \partial \theta _ { 0 } } \right) ^ { \top } [ \nabla ^ { 2 } \widetilde { \varphi _ { 1 } } ( \bar { z } _ { 0 } ) ] ^ { - 1 } \mathbf { { g } }$ can be computed by vector-Jacobian product in an automatic diferentiation framework. In summary, the gradient of $h ( \pmb \theta _ { 0 } , \pmb \theta _ { 1 } )$ with respect to $\pmb { \theta } _ { 0 }$ is given by

$$
\frac { \partial h ( \theta _ { 0 } , \theta _ { 1 } ) } { \partial \theta _ { 0 } } = \frac { h ( \theta _ { 0 } , \theta _ { 1 } ) } { 2 } \cdot \bigg [ \bigg ( \frac { \partial \omega _ { 0 } } { \partial \theta _ { 0 } } \bigg ) ^ { \top } [ \nabla ^ { 2 } \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) ] ^ { - 1 } g + \frac { \partial } { \partial \theta _ { 0 } } \log \operatorname* { d e t } \nabla ^ { 2 } \widetilde { \varphi } _ { 0 } ( z _ { 0 } ) \bigg ] .
$$

## (II) Gradient with respect to $\pmb { \theta } _ { 1 }$

Similar to Eq. (33), we have

$$
\frac { \partial h ( \theta _ { 0 } , \theta _ { 1 } ) } { \partial \theta _ { 1 } } = \frac { h ( \theta _ { 0 } , \theta _ { 1 } ) } { 2 } \cdot \left[ \frac { \partial \log p _ { 1 } ( \omega _ { 0 } ) } { \partial \theta _ { 1 } } - \frac { \partial \log p _ { 0 } ( \omega _ { 0 } ) } { \partial \theta _ { 1 } } \right] = \frac { h ( \theta _ { 0 } , \theta _ { 1 } ) } { 2 } \cdot \frac { \partial \log p _ { 1 } ( \omega _ { 0 } ) } { \partial \theta _ { 1 } } .\tag{37}
$$

Since $\bar { z } _ { 0 } = \nabla \widetilde { \varphi } _ { 1 } ^ { - 1 } ( \omega _ { 0 } )$ , we have

$$
\begin{array} { r l r } {  { \frac { \partial \log p _ { 1 } ( \omega _ { 0 } ) } { \partial \theta _ { 1 } } = \frac { \partial } { \partial \theta _ { 1 } } \Big [ \log p _ { 1 } ^ { \epsilon } ( \bar { z } _ { 0 } ) - \log \operatorname* { d e t } \nabla ^ { 2 } \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) \Big ] } } \\ & { } & { = ( \frac { \partial \bar { z } _ { 0 } } { \partial \theta _ { 1 } } ) ^ { \top } \frac { \partial \log p _ { 1 } ^ { \epsilon } ( \bar { z } _ { 0 } ) - \log \operatorname* { d e t } \nabla ^ { 2 } \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) } { \partial \bar { z } _ { 0 } } - \frac { \partial } { \partial \theta _ { 1 } } \log \operatorname* { d e t } \nabla ^ { 2 } \widetilde { \varphi } _ { 1 } ( z ) \Big | _ { z = \bar { z } _ { 0 } } , } \end{array}\tag{38}
$$

where the second equality is based on the total derivative. To evaluate the Jacobian $\frac { \partial \bar { z } _ { 0 } } { \partial \pmb { \theta } _ { 1 } }$ in the first term, we take the total derivative with respect to $\pmb { \theta } _ { 1 }$ on both sides of $\nabla \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) = \omega _ { 0 } \colon$

$$
\nabla ^ { 2 } \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) \cdot \frac { \partial \bar { z } _ { 0 } } { \partial \pmb { \theta } _ { 1 } } + \frac { \partial \nabla \widetilde { \varphi } _ { 1 } ( z ) } { \partial \pmb { \theta } _ { 1 } } \Big | _ { z = \bar { z } _ { 0 } } = \mathbf { 0 } ,
$$

and it follows that

$$
( \frac { \partial \bar { z } _ { 0 } } { \partial \pmb { \theta } _ { 1 } } ) ^ { \top } = - \bigg [ \nabla ^ { 2 } \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) \bigg ] ^ { - 1 } \frac { \partial \nabla \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) } { \partial \pmb { \theta } _ { 1 } } .
$$

Therefore, Eq. (38) can be reformulated as

$$
\frac { \partial \log p _ { 1 } ( \omega _ { 0 } ) } { \partial \theta _ { 1 } } = - \bigg ( \frac { \partial \nabla \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) } { \partial \theta _ { 1 } } \bigg ) ^ { \top } \bigg [ \nabla ^ { 2 } \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) \bigg ] ^ { - 1 } g - \frac { \partial } { \partial \theta _ { 1 } } \log \operatorname* { d e t } \nabla ^ { 2 } \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) ,
$$

where the first term can be calculated in the same way as Eq. (36) and the second term can be computed by Algorithm 2 as $\bar { z } _ { 0 }$ is a constant here, thanks to the total derivative. In summary, the gradient of $h ( \pmb \theta _ { 0 } , \pmb \theta _ { 1 } )$ with respect to $\pmb { \theta } _ { 1 }$ is given by

$$
\frac { \partial h ( \theta _ { 0 } , \theta _ { 1 } ) } { \partial \theta _ { 1 } } = - \frac { h ( \theta _ { 0 } , \theta _ { 1 } ) } { 2 } \cdot \bigg [ \bigg ( \frac { \partial \nabla \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) } { \partial \theta _ { 1 } } \bigg ) ^ { \top } [ \nabla ^ { 2 } \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) ] ^ { - 1 } g + \frac { \partial } { \partial \theta _ { 1 } } \log \operatorname* { d e t } \nabla ^ { 2 } \widetilde { \varphi } _ { 1 } ( \bar { z } _ { 0 } ) \bigg ] .
$$

## C.2 Gradient of the Sinkhorn Discrepancy Estimation

When the potential f is optimally solved, the total derivative of our Sinkhorn discrepancy estimator (18) with respect to the HyCNN parameters is given by

$$
\frac { \partial \widetilde { S } _ { \epsilon , ( n _ { k } , N ) } ( \widehat { \mathbb { P } } _ { k } , \mathbb { P } _ { k } ) } { \partial \theta _ { k } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { \partial g _ { \epsilon , k } ^ { c } ( \boldsymbol { \omega } _ { k } ^ { ( i ) } ) } { \partial \theta _ { k } } \Big | _ { f = f _ { \epsilon , k , ( n _ { k } , N ) } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \frac { \partial \boldsymbol { \omega } _ { k } ^ { ( i ) } } { \partial \theta _ { k } } \right) ^ { \top } \cdot \frac { \partial g _ { \epsilon , k } ^ { c } ( \boldsymbol { \omega } _ { k } ^ { ( i ) } ) } { \partial \boldsymbol { \omega } _ { k } ^ { ( i ) } } \Big | _ { f = f _ { \epsilon , k , ( n _ { k } , N ) } } .\tag{39}
$$

where the first equality is due to the envelope theorem, $\frac { \partial \omega _ { k } ^ { ( i ) } } { \partial \pmb { \theta } _ { k } }$ is the Jacobian of the generated sample with respect to the HyCNN parameters, and the gradient of $g _ { \epsilon , k } ^ { c }$ with respect to $\boldsymbol { \omega } _ { k } ^ { ( i ) }$ can be explicitly computed based on Eq. (17).

## C.3 Testing

In the testing phase, for a testing sample $\omega \in \Omega$ , we first trace its position in distributions $\mathbb { P } _ { 0 } ^ { \epsilon }$ and $\mathbb { P } _ { 1 } ^ { \epsilon }$ by $\bar { z } _ { 0 } = \nabla \widetilde { \varphi } _ { 0 } ^ { - 1 } ( \omega )$ and $\bar { z } _ { 1 } = \nabla \widetilde { \varphi } _ { 1 } ^ { - 1 } ( \omega )$ , respectively. Then, we evaluate the log-density of least-favorable distributions $\mathbb { P } _ { 0 }$ and $\mathbb { P } _ { 1 }$ at $\omega$ by

$$
\begin{array} { r } { \log p _ { k } ( \omega ) = \log p _ { k } ^ { \epsilon } ( \bar { z } _ { k } ) - \log \operatorname* { d e t } \nabla ^ { 2 } \widetilde { \varphi } _ { k } ( \bar { z } _ { k } ) , \quad \forall k \in \mathbb { K } . } \end{array}
$$

Finally, we make decisions by the selected optimal detector $\begin{array} { r } { \phi _ { \mathrm { o p t } } = \frac { 1 } { 2 } ( \log p _ { 0 } ( \omega ) - \log p _ { 1 } ( \omega ) ) } \end{array}$

## D Wasserstein Distributionally Robust Hypothesis Testing Model

In this section, we provide the model of the WDRO method that is used as a baseline in our experimental part. The WDRO method is the soft-constraint version of the Wasserstein distributionally robust hypothesis testing model proposed by [17] (assuming $n _ { 0 } = n _ { 1 } = n )$ :

$$
\begin{array} { r l } { \underset { p _ { 0 } , p _ { 1 } \in \mathbb { R } _ { + } ^ { n } , \gamma _ { 0 } , \gamma _ { 1 } \in \mathbb { R } _ { + } ^ { n \times n } } { \mathrm { m a x } } } & { \displaystyle \sum _ { i \in [ n ] } \operatorname* { m i n } \{ p _ { 0 } ^ { i } , p _ { 1 } ^ { i } \} - \sum _ { k \in \mathbb { K } } \lambda _ { k } \cdot \displaystyle \sum _ { i \in [ n ] } \sum _ { j \in [ n ] } \gamma _ { k , i , j } \cdot c ( { x _ { k } ^ { ( i ) } , x _ { k } ^ { ( j ) } } ) } \\ { \mathrm { s . t . } } & { \displaystyle \sum _ { j \in [ n ] } \gamma _ { k , i , j } = \widehat { \mathbb { P } } _ { k } ^ { i } \qquad \forall i \in [ n ] , k \in \mathbb { K } , } \\ & { \displaystyle \sum _ { i \in [ n ] } \gamma _ { k , i , j } = p _ { k } ^ { j } , \qquad \forall j \in [ n ] , k \in \mathbb { K } , } \end{array}\tag{40}
$$

where $\lambda _ { k } \ge 0$ is a penalty parameter for each $\boldsymbol { k } \in \mathbb { K } , \gamma _ { k , i , j }$ represents the $i j \cdot$ -th entry of $\gamma _ { k }$ , and the i-th entry of ${ \pmb p } _ { k }$ (respectively, $\widehat { \mathbb { P } } _ { k } )$ is specified by $\pmb { p } _ { k } ^ { i }$ (respectively, $\widehat { \mathbb { P } } _ { k } ^ { i } \big )$ . This model assumes that the samples in the testing set have the same support as the empirical distribution. To extend the resulting least-favorable distribution to the entire space, a kernel smoothing method is introduced. Let $\boldsymbol { p } _ { k } ^ { * }$ be an optimal solution of (40). This method convolves the discrete distribution with a kernel function $G _ { h } : \mathbb { R } ^ { d }  \mathbb { R }$ parameterized by a bandwidth $h > 0$

$$
p _ { k } ^ { h } ( \omega ) = \sum _ { i = 1 } ^ { n } p _ { k } ^ { * } ( \boldsymbol { x } _ { k } ^ { ( i ) } ) \cdot G _ { h } ( \omega - \boldsymbol { x } _ { k } ^ { ( i ) } ) , \quad \forall \omega \in \Omega , k \in \mathbb { K } ,
$$

where we choose the Gaussian kernel function $G _ { h } ( \pmb { x } ) = \exp ( - \| \pmb { x } \| ^ { 2 } / 2 h ^ { 2 } )$ . Note that the number of decision variables of this linear program is $\mathcal { O } ( n _ { 0 } \cdot n _ { 1 } )$ . Thus, this finite-dimensional convex program is not sensitive to dimension $d ,$ but is sensitive to the number of training data points of the empirical distribution.