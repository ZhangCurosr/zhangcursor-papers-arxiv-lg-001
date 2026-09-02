# On the Reliability of Generative Augmentation: A Wasserstein-Based Theoretical and Empirical Study

Chathurika S Abeykoon abeykoonc@rhodes.edu

Mathias Nthiani Muia mnmuia@southalabama.edu

Mallory Goldstein golmd-27@rhodes.edu

## Abstract

Generative data augmentation is widely used to mitigate class imbalance, yet its theoretical efect on downstream generalization remains poorly understood. In this work, we develop a statistical framework for conditional generative augmentation and analyze its impact on classification risk. We formalize augmentation as a distribution-mixing process and show that the resulting risk distortion is controlled by both the augmentation strength and the class-conditional Wasserstein discrepancy between real and generated distributions. We further derive a capacity-dependent generalization bound based on Rademacher complexity, revealing an explicit trade-of between hypothesis complexity, augmentation intensity, and generative fidelity.

Empirically, we evaluate the framework on binary and multiclass imbalanced classification tasks using Conditional GAN and Conditional WGAN-GP augmentation. Across datasets, CWGAN-GP consistently achieves lower Wasserstein discrepancies than CGAN, indicating improved distributional fidelity. However, improved fidelity does not necessarily translate into superior classification performance, with classical oversampling methods often remaining competitive. These findings support the central theoretical prediction that augmentation reliability is governed by distributional approximation error rather than predictive performance alone.

Overall, this work establishes generative augmentation as a distributional perturbation process whose reliability can be quantified through Wasserstein-based measures and supported by finite-sample generalization guarantees. The proposed framework provides a principled foundation for evaluating synthetic data quality beyond classification accuracy alone.

Keywords: Generative Data Augmentation, Imbalanced Classification, Conditional GANs, WGAN-GP, Wasserstein Distance, Optimal Transport, Distributional Fidelity, Generalization Bounds, Class-Conditional Distributions, Synthetic Data Reliability.

## 1 Introduction

Accurate classification of structured data is a central problem in machine learning. In practice, many classification tasks are characterized by limited labeled data and imbalanced class distributions, where certain classes are substantially underrepresented. Such imbalance distorts the empirical distribution used for learning, often leading to biased decision boundaries, reduced minority-class recall, and degraded generalization performance.

From a statistical learning perspective, class imbalance alters the empirical distribution used for risk minimization as discussed by He et al. in [1]. Let $( X , Y ) \sim p _ { \mathrm { d a t a } } ( x , y )$ denote the joint distribution of features and class labels . In finite samples, empirical risk minimization approximates the population risk under the empirical distribution $\hat { p } ( x , y )$ . When ˆp(y) is highly skewed, the learned classifier tends to prioritize majority classes, implicitly minimizing risk under a distorted sampling distribution. This imbalance can lead to biased decision boundaries and poor minority-class recall, even when overall accuracy appears high. Conventional mitigation strategies, including class reweighting, undersampling, and oversampling, partially address this issue but often fail to capture the underlying class-conditional feature distributions in a principled manner.

Synthetic data augmentation provides a promising alternative. Rather than merely replicating minority samples or adjusting loss weights, generative models aim to approximate the conditional distribution $p _ { \mathrm { d a t a } } ( x \mid y )$ and sample new feature vectors from the learned distribution. As a result, synthetic augmentation can enrich the efective support of minority classes and encourage classifiers to learn smoother and more representative decision boundaries. However, applying generative augmentation to structured feature spaces introduces several practical challenges. In contrast to image-based settings, where convolutional inductive biases help preserve spatial coherence and support realistic sample generation, tabular feature vectors typically lack inherent structure. As a result, statistical relationships among variables must be learned directly from often limited data, increasing the dificulty of accurately modeling the underlying distribution. Under these conditions, generative models are more susceptible to training instability, distributional artifacts, noise amplification, and mode collapse, all of which can negatively impact downstream classification performance rather than improve it.[2]

Generative Adversarial Networks (GAN) ofers a flexible framework for learning complex, high-dimensional data distributions through adversarial training [3]. A generator network maps latent noise variables to synthetic samples, while a discriminator (or critic) attempts to distinguish real from generated data. Conditional GAN (CGAN) extends this framework by conditioning both generator and discriminator on class labels, enabling targeted generation of class-specific samples. In settings with class imbalance, conditional generation allows focused augmentation of underrepresented varieties. Nevertheless, classical GAN training based on Jensen–Shannon divergence is known to sufer from vanishing gradients and unstable dynamics when real and generated distributions have limited support overlap.

To address these challenges, Wasserstein GANs (WGANs) reformulate the adversarial objective using the Wasserstein-1 distance, a metric grounded in optimal transport theory that provides informative gradients even when the supports of the real and generated distributions do not overlap [4]. Within this framework, the discriminator is replaced by a Lipschitzconstrained critic trained to approximate the Wasserstein distance between real and synthetic distributions, resulting in a smoother optimization landscape and more stable convergence behavior relative to divergence-based objectives [4, 5]. The gradient penalty formulation (WGAN-GP) further enforces the Lipschitz constraint without restricting model capacity through weight clipping, improving training stability, sample diversity, and robustness to mode collapse across a variety of applications [5, 2]. These theoretical and practical advantages make WGAN-GP particularly well suited for modeling continuous feature distributions and for downstream learning tasks that are sensitive to distributional mismatch.

While generative augmentation has demonstrated empirical success in addressing class imbalance, its theoretical implications for classification performance remain insuficiently understood. Existing studies [6, 2] largely evaluate augmentation strategies through predictive accuracy, with limited attention to how discrepancies between real and generated distributions afect downstream learning. Consequently, many approaches lack formal guarantees regarding the reliability of synthetic data augmentation.

In this work, we develop a theoretical and empirical framework for studying the reliability of conditional generative augmentation in imbalanced classification. We employ CGAN and Conditional WGAN-GP (CWGAN-GP) architectures to learn class-conditional feature distributions and generate synthetic samples for data augmentation. Building on concepts from optimal transport, we derive a Wasserstein-based risk bound that explicitly quantifies how discrepancies between the learned and true data distributions influence classification risk. The framework is evaluated on both binary and multiclass classification tasks exhibiting natural class imbalance.

Experimental results demonstrate that CWGAN-GP consistently achieves lower Wasserstein discrepancies than CGAN, indicating improved distributional fidelity of the generated samples. However, improved fidelity does not necessarily translate into superior classification performance relative to classical oversampling methods. These findings highlight an important distinction between synthetic data fidelity and predictive utility and motivate the need for principled reliability measures beyond classification accuracy alone. Overall, the proposed framework provides both theoretical guarantees and practical tools for evaluating the reliability of generative augmentation in imbalanced learning settings.

## 2 Related Work

## 2.1 Class Imbalance and Data Augmentation

Class imbalance is a longstanding challenge in supervised learning because empirical risk minimization tends to favor majority classes, often resulting in poor minority-class performance [1]. Traditional approaches include class reweighting, undersampling, and oversampling techniques that modify the efective training distribution. Among these methods, the Synthetic Minority Oversampling Technique (SMOTE) generates additional minority-class samples through interpolation between neighboring observations and remains one of the most widely used approaches for imbalanced classification [7]. Although such methods can improve predictive performance, they primarily alter sample frequencies and do not explicitly model the underlying class-conditional distributions.

Recent advances in generative modeling have motivated the use of synthetic data augmentation as an alternative strategy for addressing class imbalance. Rather than replicating or interpolating existing observations, generative models attempt to learn the data-generating process and sample new observations from the learned distribution. This perspective has led to growing interest in generative augmentation across a variety of application domains.

## 2.2 Generative Models and Wasserstein-Based Learning

Generative Adversarial Networks (GANs) introduced a powerful framework for learning complex data distributions through adversarial training [3]. Conditional GANs (CGANs) extend this framework by incorporating class labels into both the generator and discriminator, enabling class-specific sample generation and targeted augmentation of minority classes [8]. Numerous studies have reported improved classification performance through GAN-based augmentation in both image and structured-data settings [6].

Despite their success, classical GANs are known to sufer from training instability, mode collapse, and sensitivity to divergence-based objectives [2]. Wasserstein GANs address these issues by replacing divergence-based training objectives with the Wasserstein-1 distance, a metric grounded in optimal transport theory [4]. The gradient-penalty formulation (WGAN-GP) further improves stability by enforcing the Lipschitz constraint through gradient regularization rather than weight clipping [5]. These developments have made Wasserstein-based generative models particularly attractive for applications where distributional fidelity is important.

## 2.3 Generative Augmentation and Distributional Reliability

Most existing studies evaluate generative augmentation primarily through downstream predictive performance, such as accuracy, F1 score, or recall [6, 9]. While these metrics provide useful information regarding task-specific performance, they ofer limited insight into how closely the generated samples approximate the underlying data-generating distribution. As a result, the relationship between generative fidelity and classification performance remains poorly understood.

Several recent works have advocated the use of distributional metrics for evaluating synthetic data quality, including measures based on optimal transport and Wasserstein distances. However, formal connections between distributional approximation error and downstream classification risk remain relatively underdeveloped. In particular, few studies provide theoretical guarantees describing how discrepancies between real and generated distributions influence the reliability of synthetic augmentation.

The present work addresses this gap by combining conditional generative augmentation with a Wasserstein-based theoretical framework. Rather than evaluating synthetic data solely through predictive performance, we analyze augmentation as a distribution-mixing process and derive explicit risk bounds linking class-conditional Wasserstein approximation error to classification risk. This perspective provides a principled framework for assessing the reliability of synthetic data augmentation in imbalanced learning settings.

## 2.4 Our Contribution

The main contributions of this work are summarized as follows:

• A Wasserstein-Based Framework for Augmentation Reliability. We develop a theoretical framework for analyzing the reliability of generative data augmentation in imbalanced classification. The framework explicitly links class-conditional distributional discrepancies between real and synthetic data to downstream classification risk.

• Risk Bounds for Conditional Generative Augmentation. We derive a Wasserstein-based risk bound showing that augmentation-induced risk distortion is controlled by both the augmentation strength and the class-conditional Wasserstein discrepancy between the true and generated distributions. We further establish capacity-dependent generalization guarantees through a Rademacher complexity analysis.

• Distributional Fidelity Evaluation. We introduce class-wise and weighted Wasserstein fidelity measures that provide interpretable and theoretically motivated criteria for assessing the quality and reliability of synthetic data generated by conditional generative models.

• Empirical Validation on Imbalanced Classification Tasks. We conduct extensive experiments on binary and multiclass datasets exhibiting natural class imbalance. The empirical results demonstrate that CWGAN-GP consistently achieves lower class-conditional Wasserstein discrepancies than CGAN, supporting the theoretical relationship between generative fidelity and augmentation reliability while highlighting the distinction between distributional fidelity and downstream predictive performance.

## 3 Theoretical Framework for Generative Augmentation Reliability

This section develops a statistical learning framework for analyzing conditional generative augmentation in structured feature space. Our goal is to formally characterize how the fidelity of synthetic data generated by a conditional adversarial model influences downstream classification risk and augmentation reliability. Rather than treating augmentation as a heuristic data-expansion procedure, we interpret it as controlled distributional mixing and quantify its impact using tools from optimal transport theory which is an alternative way of measuring the distance between probability distributions.

## 3.1 Learning Under Imbalanced Sampling

Let $( X , Y ) \sim P$ denote the true joint distribution over the feature space $\boldsymbol { \mathcal { X } } \subset \mathbb { R } ^ { d }$ and label set $\mathcal { V } = \{ 1 , \ldots , K \}$ . For a classifier $f : \mathcal { X }  \mathcal { V }$ and loss function $\ell ,$ the population risk is defined as

$$
R _ { P } ( f ) = \mathbb { E } _ { ( x , y ) \sim P } \left[ \ell ( f ( x ) , y ) \right] .\tag{3.1}
$$

Given a training sample $\mathbfcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , learning proceeds by minimizing the empirical risk

$$
{ \widehat { R } } _ { n } ( f ) = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \ell ( f ( x _ { i } ) , y _ { i } ) .\tag{3.2}
$$

Under class imbalance, empirical risk minimization tends to emphasize majority classes while underrepresenting minority-class risk. Generative augmentation seeks to alleviate this issue by learning class-conditional distributions and generating additional minority-class samples.

## 3.2 Conditional Generative Modeling

For each class $y \in \mathcal { V }$ , let

$$
P _ { y } = P ( \cdot \mid Y = y )\tag{3.3}
$$

denote the true class-conditional distribution. A conditional generator is a measurable mapping

$$
G _ { \theta } : \mathcal { Z } \times \mathcal { Y }  \mathcal { X } ,\tag{3.4}
$$

where $z \sim p _ { Z }$ is a latent variable. The generator induces the class-conditional distribution

$$
P _ { G , y } = \left( G _ { \theta } ( \cdot , y ) \right) _ { \# } p _ { Z } ,\tag{3.5}
$$

where $( \cdot ) _ { \# }$ denotes the pushforward measure. Training aims to approximate the true classconditional distributions,

$$
P _ { G , y } \approx P _ { y } , \qquad y \in \mathcal { V } .\tag{3.6}
$$

## 3.3 Wasserstein Approximation Error

To quantify the discrepancy between real and generated distributions, we employ the Wasserstein– 1 distance

$$
W _ { 1 } ( \mu , \nu ) = \operatorname* { i n f } _ { \gamma \in \Pi ( \mu , \nu ) } \mathbb { E } _ { ( x , \tilde { x } ) \sim \gamma } \left[ \left. x - \tilde { x } \right. \right] ,\tag{3.7}
$$

where $\Pi ( \mu , \nu )$ denotes the set of all couplings with marginals $\mu$ and ν. Equivalently, by Kantorovich–Rubinstein duality,

$$
W _ { 1 } ( \mu , \nu ) = \operatorname* { s u p } _ { \| g \| _ { \mathrm { L i p } } \leq 1 } \left| \mathbb { E } _ { \mu } [ g ( x ) ] - \mathbb { E } _ { \nu } [ g ( x ) ] \right| .\tag{3.8}
$$

For each class y, we define the class-conditional approximation error as

$$
\varepsilon _ { y } = W _ { 1 } ( P _ { y } , P _ { G , y } ) .\tag{3.9}
$$

Smaller values of $\varepsilon _ { y }$ indicate closer agreement between the real and generated classconditional distributions and therefore higher generative fidelity.

## 3.4 Synthetic Augmentation as Distribution Mixing

After training, synthetic samples are incorporated into the data through the mixture distribution

$$
P _ { \alpha } = ( 1 - \alpha ) P + \alpha P _ { G } ,\tag{3.10}
$$

where $\alpha \in [ 0 , 1 ]$ denotes the augmentation strength and

$$
P _ { G } ( x , y ) = \pi _ { y } P _ { G , y } ( x ) , \qquad \pi _ { y } = P ( Y = y ) .\tag{3.11}
$$

The corresponding augmented risk is

$$
R _ { P _ { \alpha } } ( f ) = \mathbb { E } _ { ( x , y ) \sim P _ { \alpha } } \left[ \ell ( f ( x ) , y ) \right] .\tag{3.12}
$$

The following theorem establishes how the class-conditional approximation errors $\left\{ \varepsilon _ { y } \right\}$ influence the diference between the original and augmented risks.

## 3.5 Risk Control Under Conditional Wasserstein Approximation

The following theorem establishes a Wasserstein-based bound on the risk distortion induced by synthetic augmentation. It shows that the diference between the original and augmented risks scales linearly with the augmentation strength α and the class-conditional approximation errors $\{ \varepsilon _ { y } \} _ { y = 1 } ^ { K }$ . As a result, higher-fidelity generators yield tighter control of augmentationinduced risk distortion.

Theorem 3.1 (Risk Stability Under Conditional Generative Augmentation). Assume:

1. The loss $\ell ( f ( x ) , y )$ is L<sub>ℓ</sub>-Lipschitz with respect to $x .$

2. The hypothesis class $\mathcal { F }$ is measurable and induces a uniformly bounded loss.

3. For each class $y \in \mathcal { V }$

$$
{ \cal W } _ { 1 } ( P _ { y } , P _ { G , y } ) \le \varepsilon _ { y } .
$$

Then for any classifier $f _ { i }$

$$
| R _ { P _ { \alpha } } ( f ) - R _ { P } ( f ) | \leq \alpha L _ { \ell } \sum _ { y \in \mathcal { V } } \pi _ { y } \varepsilon _ { y } .\tag{3.13}
$$

$I f \varepsilon _ { y } \leq \varepsilon$ uniformly, then

$$
| R _ { P _ { \alpha } } ( f ) - R _ { P } ( f ) | \leq \alpha L _ { \ell } \varepsilon .\tag{3.14}
$$

These assumptions are standard in optimal transport and statistical learning theory. Assumption 1 links Wasserstein discrepancies to diferences in expected loss, Assumption 2 ensures the existence of the relevant expectations, and Assumption 3 quantifies the class-conditional approximation error of the generator.

Proof. Let P denote the true joint distribution and $P _ { G }$ the synthetic joint distribution defined as in Equation 3.10. By linearity of expectation, we decompose the augmented risk (3.12) under mixture distributions.

$$
\begin{array} { r l } & { R _ { P _ { \alpha } } ( f ) = \mathbb { E } _ { ( x , y ) \sim P _ { \alpha } } [ \ell ( f ( x ) , y ) ] } \\ & { \qquad = ( 1 - \alpha ) \mathbb { E } _ { P } [ \ell ( f ( x ) , y ) ] + \alpha \mathbb { E } _ { P _ { G } } [ \ell ( f ( x ) , y ) ] . } \end{array}
$$

Hence,

$$
\begin{array} { r l } & { R _ { P _ { \alpha } } ( f ) - R _ { P } ( f ) = ( 1 - \alpha ) R _ { P } ( f ) + \alpha R _ { P _ { G } } ( f ) - R _ { P } ( f ) } \\ & { \qquad = \alpha \big ( R _ { P _ { G } } ( f ) - R _ { P } ( f ) \big ) . } \end{array}
$$

Taking absolute values,

$$
| R _ { P _ { \alpha } } ( f ) - R _ { P } ( f ) | = \alpha | R _ { P _ { G } } ( f ) - R _ { P } ( f ) | .
$$

Thus it sufices to bound $| R _ { P _ { G } } ( f ) - R _ { P } ( f ) |$ . Next by using the law of total expectation, we do the conditional decomposition as

$$
R _ { P _ { G } } ( f ) = \sum _ { y \in \mathcal { V } } \pi _ { y } \mathbb { E } _ { x \sim P _ { G , y } } [ \ell ( f ( x ) , y ) ] ,
$$

$$
R _ { P } ( f ) = \sum _ { y \in \mathcal { V } } \pi _ { y } \mathbb { E } _ { x \sim P _ { y } } [ \ell ( f ( x ) , y ) ] .
$$

Therefore,

$$
R _ { P _ { G } } ( f ) - R _ { P } ( f ) = \sum _ { y \in \mathcal { Y } } \pi _ { y } \left( \mathbb { E } _ { P _ { G , y } } [ \ell ( f ( x ) , y ) ] - \mathbb { E } _ { P _ { y } } [ \ell ( f ( x ) , y ) ] \right) .\tag{3.15}
$$

Taking absolute values and applying the triangle inequality,

$$
| R _ { P G } ( f ) - R _ { P } ( f ) | \leq \sum _ { y \in \mathcal { V } } \pi _ { y } \left| \mathbb { E } _ { P _ { G , y } } [ \ell ( f ( x ) , y ) ] - \mathbb { E } _ { P _ { y } } [ \ell ( f ( x ) , y ) ] \right| .\tag{3.16}
$$

By assumption, the loss $\ell ( f ( x ) , y )$ is $L _ { \ell ^ { - } } \mathrm { L i p s c h i t z }$ in x. Define $g _ { y } ( x ) : = \ell ( f ( x ) , y )$ . Then

$$
\| g _ { y } \| _ { \mathrm { L i p } } \leq L _ { \ell } .
$$

Let $\begin{array} { r } { \tilde { g } _ { y } ( x ) : = \frac { 1 } { L _ { \ell } } g _ { y } ( x ) } \end{array}$ . Then $\| \tilde { g } _ { y } \| _ { \mathrm { L i p } } \leq 1$ . By Kantorovich–Rubinstein duality,

$$
\big | \mathbb { E } _ { P _ { G , y } } [ \tilde { g } _ { y } ( x ) ] - \mathbb { E } _ { P _ { y } } [ \tilde { g } _ { y } ( x ) ] \big | \le W _ { 1 } ( P _ { y } , P _ { G , y } ) .
$$

Multiplying both sides by $L _ { \ell }$ gives

$$
\begin{array} { r } { \left| \mathbb { E } _ { P _ { G , y } } [ \ell ( f ( x ) , y ) ] - \mathbb { E } _ { P _ { y } } [ \ell ( f ( x ) , y ) ] \right| \le L _ { \ell } W _ { 1 } ( P _ { y } , P _ { G , y } ) . } \end{array}\tag{3.17}
$$

Substituting 3.17 into 3.16,

$$
| R _ { P _ { G } } ( f ) - R _ { P } ( f ) | \le L _ { \ell } \sum _ { y \in \mathcal { Y } } \pi _ { y } W _ { 1 } ( P _ { y } , P _ { G , y } ) .\tag{3.18}
$$

By assumption, $W _ { 1 } ( P _ { y } , P _ { G , y } ) \leq \varepsilon _ { y }$ . Thus,

$$
| R _ { P _ { G } } ( f ) - R _ { P } ( f ) | \leq L _ { \ell } \sum _ { y \in \mathcal { V } } \pi _ { y } \varepsilon _ { y } .\tag{3.19}
$$

Finally,

$$
| R _ { P _ { \alpha } } ( f ) - R _ { P } ( f ) | \leq \alpha L _ { \ell } \sum _ { y \in \mathcal { V } } \pi _ { y } \varepsilon _ { y } .\tag{3.20}
$$

If $\varepsilon _ { y } \leq \varepsilon$ uniformly,

$$
| R _ { P _ { \alpha } } ( f ) - R _ { P } ( f ) | \leq \alpha L _ { \ell } \varepsilon .
$$

This completes the proof.

In practice, the class-conditional Wasserstein discrepancies provide a quantitative measure of generative fidelity. Theorem 3.1 therefore establishes a direct connection between distributional fidelity and augmentation reliability: smaller Wasserstein discrepancies imply tighter control of augmentation-induced risk distortion.

## 3.6 Asymptotic Consistency of Conditional Generative Augmentation

While risk bounds under finite Wasserstein discrepancy quantify the immediate impact of generative augmentation, it is also important to understand the long-term behavior as the generator improves. Specifically, if the generator is able to produce samples that increasingly approximate the true class-conditional distributions, we expect the induced risk distortion to vanish as discussed in [10].

Corollary 3.2 (Consistency Under Vanishing Transport Error). Suppose $\left\{ G _ { n } \right\}$ is a sequence of generators satisfying

$$
\operatorname* { m a x } _ { y \in \mathcal { y } } W _ { 1 } ( P _ { y } , P _ { G _ { n } , y } ) \to 0 .\tag{3.21}
$$

$I f \alpha _ { n }  \alpha \in [ 0 , 1 ]$ , then for any fixed classifier $f ,$

$$
R _ { P _ { \alpha _ { n } } } ( f )  R _ { P } ( f ) .\tag{3.22}
$$

This result establishes that conditional generative augmentation is asymptotically unbiased: as the generator becomes perfectly accurate in the Wasserstein sense, the expected classification risk under the augmented distribution converges to that under the true distribution. In other words, improvements in generator quality translate directly into more reliable augmentation.

Proof . Let $\left\{ G _ { n } \right\}$ be a sequence of generators such that ma $\mathrm { x } _ { y \in \mathcal { V } } W _ { 1 } ( P _ { y } , P _ { G _ { n } , y } )  0$ . From Theorem 3.1,

$$
| R _ { P _ { \alpha _ { n } } } ( f ) - R _ { P } ( f ) | \leq \alpha _ { n } L _ { \ell } \operatorname* { m a x } _ { y } W _ { 1 } ( P _ { y } , P _ { G _ { n } , y } ) .
$$

Since $0 \leq \alpha _ { n } \leq 1$ , the sequence $\left\{ \alpha _ { n } \right\}$ is bounded. Because ma $\mathrm { x } _ { y } W _ { 1 } ( P _ { y } , P _ { G n , y } )  0$ , the product converges to zero. Hence,

$$
R _ { P _ { \alpha _ { n } } } ( f )  R _ { P } ( f ) .
$$

This establishes asymptotic unbiasedness under vanishing Wasserstein approximation error. □

These findings are consistent with prior results in statistical learning theory and optimal transport, which establish that excess risk vanishes as the Wasserstein discrepancy between training and target distributions converges to zero, yielding asymptotic consistency of learning under distributional approximation [11].

## 3.7 Capacity-Dependent Generalization via Rademacher Complexity

The Wasserstein-based bounds developed above quantify the bias introduced by generative augmentation at the population level. In practice, however, classifiers are learned from finite samples and selected from a hypothesis class F. Consequently, generalization performance depends not only on distributional approximation error but also on the capacity of the hypothesis class.

To capture this efect, we employ Rademacher complexity, a standard measure of function class capacity in statistical learning theory. Let $\sigma _ { 1 } , \ldots , \sigma _ { n }$ be independent Rademacher random variables taking values ±1 with equal probability. For the loss-composed class

$$
\ell \circ { \mathcal { F } } = \left\{ ( x , y ) \mapsto \ell ( f ( x ) , y ) : f \in { \mathcal { F } } \right\} ,
$$

the empirical Rademacher complexity is defined as

$$
{ \widehat { \mathfrak { R } } } _ { n } ( \ell \circ { \mathcal { F } } ) = \mathbb { E } _ { \sigma } \left[ \operatorname* { s u p } _ { f \in { \mathcal { F } } } { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \sigma _ { i } \ell ( f ( x _ { i } ) , y _ { i } ) \right] .
$$

Smaller values of $\widehat { \mathfrak { R } } _ { n } ( \ell \circ \mathcal { F } )$ indicate a lower-capacity hypothesis class and stronger generalization guarantees. When training on the augmented distribution 3.10, the generalization error consists of two components:

1. Estimation error, controlled by the Rademacher complexity of ${ \mathcal F } .$

2. Augmentation bias, controlled by the Wasserstein approximation errors $\{ \varepsilon _ { y } \}$

Next we combine these quantities to derive a finite-sample generalization bound that explicitly accounts for hypothesis class capacity, augmentation strength, and generative fidelity.

## 3.8 Capacity-Dependent Excess Risk Under Generative Augmentation

While previous results quantified the bias induced by generative augmentation via classconditional Wasserstein distances, and asymptotic consistency guaranteed vanishing bias for perfect generators, in practice we work with finite samples and a restricted hypothesis class. The generalization ability of the learned classifier therefore depends not only on the fidelity of the synthetic data but also on the capacity of the function class used for learning.

To formalize this, we analyze the empirical risk minimizer trained on a mixture of real and synthetic data. By combining Wasserstein-based bias control with Rademacher complexity bounds that capture hypothesis class capacity, we obtain a high-probability bound on the excess risk. This bound explicitly quantifies how the risk depends on:

1. The augmentation strength α and class-conditional approximation errors $\epsilon _ { y }$ (bias due to the generator).

2. The complexity of the hypothesis class (variance due to finite samples).

3. The sample size N and desired confidence $1 - \delta$

The following theorem formalizes these trade-ofs, showing that with high probability, the empirical minimizer achieves an excess risk that balances the fidelity of the generative augmentation with the expressive power of the classifier.

Theorem 3.3 (Capacity-Dependent Generalization Bound). Let $\begin{array} { r } { \varepsilon = \operatorname* { m a x } _ { y \in \mathcal { V } } W _ { 1 } ( P _ { y } , P _ { G , y } ) } \end{array}$ denote the maximum class-conditional Wasserstein approximation error. Under the assumptions of Theorem 3.1, for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta ,$

$$
R _ { P } ( \widehat { f } _ { \alpha } ) \leq \operatorname* { i n f } _ { f \in \mathcal { F } } R _ { P } ( f ) + 2 \Re _ { n } ( \ell \circ \mathcal { F } ) + 3 B \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } } + \alpha L _ { \ell } \varepsilon .\tag{3.23}
$$

The proof follows from standard Rademacher complexity generalization bounds [12, 13] combined with Theorem 3.1.

Proof. We decompose excess risk into estimation error and augmentation bias. For any $f \in { \mathcal { F } }$

$$
R _ { P } ( \widehat { f } _ { \alpha } ) = R _ { P _ { \alpha } } ( \widehat { f } _ { \alpha } ) + \big ( R _ { P } ( \widehat { f } _ { \alpha } ) - R _ { P _ { \alpha } } ( \widehat { f } _ { \alpha } ) \big ) .
$$

By Theorem 3.1, $| R _ { P } ( f ) - R _ { P _ { \alpha } } ( f ) | \leq \alpha L _ { \ell } \varepsilon$ . Thus we get, $R _ { P } ( \widehat { f } _ { \alpha } ) \leq R _ { P _ { \alpha } } ( \widehat { f } _ { \alpha } ) + \alpha L _ { \ell } \varepsilon$ . Applying the standard Rademacher generalization bound, with probability at least $1 - \delta$

$$
R _ { P _ { \alpha } } ( \widehat { f } _ { \alpha } ) \leq \operatorname* { i n f } _ { f \in \mathcal { F } } R _ { P _ { \alpha } } ( f ) + 2 \Re _ { n } ( \ell \circ \mathcal { F } ) + 3 B \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } } .\tag{3.24}
$$

Again using Theorem 3.1 and considering the infimum over $f \in { \mathcal { F } }$ gives,

$$
\operatorname* { i n f } _ { f \in \mathcal { F } } R _ { P _ { \alpha } } ( f ) \leq \operatorname* { i n f } _ { f \in \mathcal { F } } R _ { P } ( f ) + \alpha L _ { \ell } \varepsilon .\tag{3.25}
$$

Substituting this back into Equation 3.24,

$$
R _ { P } ( \widehat { f } _ { \alpha } ) \leq \operatorname* { i n f } _ { f \in \mathcal { F } } R _ { P } ( f ) + 2 \Re _ { n } ( \ell \circ \mathcal { F } ) + 3 B \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } } + \alpha L _ { \ell } \varepsilon .\tag{3.26}
$$

The bound highlights a fundamental trade-of between estimation error and augmentation bias. While lower Wasserstein discrepancies yield tighter control of augmentation-induced risk distortion, predictive performance also depends on hypothesis class capacity and finitesample efects. Consequently, improvements in generative fidelity do not necessarily translate into proportional gains in classification accuracy.

## 4 Experimental Design

This section describes the datasets, augmentation procedures, evaluation metrics, and experimental settings used to validate the proposed theoretical framework. The experiments are designed to assess both the distributional fidelity of generated samples and their impact on downstream classification performance under class imbalance.

## 4.1 Experimental Objectives

The empirical study evaluates the reliability of conditional generative augmentation by assessing distributional fidelity, classification performance under varying augmentation strengths, diferences between CGAN and CWGAN-GP generated samples, and the relationship between generative fidelity and downstream predictive performance.

## 4.2 Datasets

To evaluate the proposed framework across diferent imbalance settings, experiments were conducted on two benchmark classification datasets exhibiting natural class imbalance.

<table><tr><td>Dataset</td><td>Samples</td><td>Features</td><td>Classes</td><td>Minority Class  $\overline { { \left( \% \right) } }$ </td></tr><tr><td>Credit Card Default</td><td>30,000</td><td>23</td><td>2</td><td>22.1</td></tr><tr><td>Forest Cover Type</td><td>581,012</td><td>54</td><td>7</td><td>0.47</td></tr></table>

Table 1: Summary of datasets used in the empirical evaluation.

The Credit Card Default dataset is a binary classification problem in which the objective is to predict whether a customer will default on the next month’s payment. The dataset contains financial and demographic attributes and exhibits a naturally imbalanced class distribution, making it a widely used benchmark for studying imbalance-aware learning methods.

The Forest Cover Type dataset is a multiclass classification problem consisting of cartographic and environmental variables used to predict forest cover categories. The dataset contains seven classes with substantially diferent class frequencies, providing a challenging large-scale benchmark for evaluating class-conditional generative augmentation in multiclass settings.

Together, these datasets enable evaluation of the proposed framework across both binary and multiclass imbalance scenarios while providing substantially diferent data characteristics and class distributions.

## 4.3 Experimental Setup

Generative Augmentation Models. Two conditional generative models are considered in this study: the CGAN and the CWGAN-GP. Both models are trained to learn class conditional feature distributions from the training data and subsequently generate synthetic samples for data augmentation. The generated samples are then combined with the original training data according to the augmentation strength parameter α to construct the augmented datasets used for downstream classification.

We compare CWGAN-GP against the following baselines:

• Real: Training on the original imbalanced dataset without augmentation.

• ROS: Random Oversampling, replicating minority-class samples until class balance is achieved. While computationally eficient and simple to implement, this method often leads to overfitting due to duplicate data points.

• SMOTE: Synthetic Minority Over-sampling Technique based on nearest-neighbor interpolation.

• CGAN: Conditional GAN-based augmentation without Wasserstein regularization.

Downstream Classifiers. Downstream classification performance is evaluated using two representative classifiers: a Random Forest (RF) with 100 trees and a Multi-Layer Perceptron (MLP) with two hidden layers and ReLU activations. These models were selected to represent two distinct learning paradigms, allowing the efects of synthetic augmentation to be assessed across diferent classifier families.

Augmentation Strength α. To assess the efect of augmentation intensity, experiments were conducted with augmentation strengths α ∈ {0.25, 0.50, 0.75, 1.00}.

Evaluation Metrics and Statistical Testing. Performance is evaluated using Macro-F1 score and Macro Recall, which are appropriate for assessing classification performance under class imbalance. Distributional fidelity is quantified using both the Average Class-Wise Wasserstein Distance and the Weighted Class-Wise Wasserstein Distance between real and generated classconditional distributions.

Implementation Details. CGAN and CWGAN-GP were trained using dataset-specific hyperparameter settings selected through preliminary experimentation. For the Credit Card dataset, a latent dimension of 64 and 500 training epochs were used, whereas the Forest Cover dataset employed a latent dimension of 32 and 300 training epochs. In all experiments, models were optimized using the Adam optimizer with a learning rate of $1 0 ^ { - 4 }$ , and CWGAN-GP was trained using a gradient penalty coeficient of 10. All reported results were averaged over three independent random seeds. Complete hyperparameter settings are provided in Appendix ??.

## 5 Results and Discussion

## 5.1 Distributional Fidelity Analysis

We first evaluate the fidelity of the generated samples using Wasserstein-based measures. Since the theoretical framework developed in Section 3 identifies Wasserstein approximation error as a key determinant of augmentation reliability, these results provide a direct empirical assessment of the proposed risk bounds.

For the Credit Card dataset, fidelity was measured using the Wasserstein distance between the real and generated minority-class distributions. For the Forest Cover dataset, fidelity was assessed using both the Average Class-Wise Wasserstein Distance and the Weighted Class-Wise Wasserstein Distance.

<table><tr><td>Dataset</td><td>Method</td><td>Wasserstein Distance Average Class-Wise W1</td><td></td></tr><tr><td>Credit Card Credit Card</td><td>CGAN CWGAN-GP</td><td> $2 7 4 0 8 . 4 5 \pm 5 6 7 . 1 9$   $3 7 1 4 . 9 9 \pm 5 5 9 . 2 1$ </td><td></td></tr><tr><td>Forest Cover</td><td>CGAN</td><td> $0 . 1 2 3 1 \pm 0 . 0 1 5 8$ </td><td> $0 . 0 6 0 1 \pm 0 . 0 0 7 2$ </td></tr><tr><td>Forest Cover</td><td>CWGAN-GP</td><td> $0 . 1 0 7 2 \pm 0 . 0 1 6 4$ </td><td> $0 . 0 5 4 3 \pm 0 . 0 0 2 7$ </td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

Table 2: Distributional fidelity measured using Wasserstein distances (mean ± standard deviation over three random seeds).

Table 2 shows that CWGAN-GP consistently achieves lower Wasserstein discrepancies than CGAN across both datasets. On the Credit Card dataset, the Wasserstein distance decreases from approximately 27, 408 to 3, 715, while on the Forest Cover dataset CWGAN-GP achieves lower average and weighted class-wise Wasserstein distances. These results indicate that CWGAN-GP produces synthetic samples that more closely approximate the underlying class-conditional distributions.

Because Wasserstein discrepancy appears explicitly in the theoretical risk bounds of Section 3, lower values correspond to tighter control of augmentation-induced risk distortion. Thus, the observed reductions in Wasserstein distance provide empirical support for the proposed framework and suggest that CWGAN-GP generates higher-fidelity synthetic data than CGAN.

For the Credit Card dataset, Wasserstein discrepancy increased with augmentation strength for both generative methods. However, CWGAN-GP consistently maintained substantially lower Wasserstein values than CGAN, ranging from approximately 6,340 to 11,744 compared with 89,924 to 166,518 for CGAN.

The next subsection investigates whether these improvements in distributional fidelity translate into improved downstream classification performance.

## 5.2 Classification Performance Under Augmentation

We next evaluate the impact of generative augmentation on downstream classification performance. Following the experimental protocol described in Section 4, performance is assessed

using Macro-F1 score and Macro Recall, with results averaged over three independent random seeds. These metrics provide complementary perspectives on classification quality under class imbalance, with Macro-F1 measuring overall predictive performance and Macro Recall emphasizing performance on underrepresented classes.
<table><tr><td>Method</td><td>α</td><td>Macro-F1</td><td>Macro Recall</td></tr><tr><td>Real</td><td>0.00</td><td> $0 . 6 7 2 0 \pm 0 . 0 1 0 8$ </td><td> $0 . 6 4 9 5 \pm 0 . 0 1 0 4$ </td></tr><tr><td>ROS</td><td>0.50</td><td> $\mathbf { 0 . 6 8 0 9 \pm 0 . 0 0 8 8 }$ </td><td> $\mathbf { 0 . 6 7 1 4 \pm 0 . 0 1 4 8 }$ </td></tr><tr><td>SMOTE</td><td>0.50</td><td> $0 . 6 7 7 8 \pm 0 . 0 1 3 8$ </td><td> $0 . 6 6 9 9 \pm 0 . 0 0 6 1$ </td></tr><tr><td rowspan="4">CGAN</td><td>0.25</td><td> $0 . 6 7 2 8 \pm 0 . 0 0 9 2$ </td><td> $0 . 6 4 9 4 \pm 0 . 0 0 9 4$ </td></tr><tr><td>0.50</td><td> $0 . 6 7 5 1 \pm 0 . 0 0 4 9$ </td><td> $0 . 6 5 2 0 \pm 0 . 0 0 5 7$ </td></tr><tr><td>0.75</td><td> $0 . 6 6 8 9 \pm 0 . 0 2 0 8$ </td><td> $0 . 6 4 7 1 \pm 0 . 0 1 9 0$ </td></tr><tr><td>1.00</td><td> $0 . 6 7 6 4 \pm 0 . 0 0 8 7$ </td><td> $0 . 6 5 3 7 \pm 0 . 0 0 8 5$ </td></tr><tr><td rowspan="4">CWGAN-GP</td><td>0.25</td><td> $0 . 6 7 2 1 \pm 0 . 0 1 2 6$ </td><td> $0 . 6 4 9 3 \pm 0 . 0 1 2 4$ </td></tr><tr><td>0.50</td><td> $0 . 6 7 4 2 \pm 0 . 0 0 9 4$ </td><td> $0 . 6 5 2 3 \pm 0 . 0 0 8 0$ </td></tr><tr><td>0.75</td><td> $0 . 6 7 4 2 \pm 0 . 0 1 2 8$ </td><td> $0 . 6 5 2 5 \pm 0 . 0 1 2 1$ </td></tr><tr><td>1.00</td><td> $0 . 6 7 1 4 \pm 0 . 0 1 5 2$ </td><td> $0 . 6 4 9 7 \pm 0 . 0 1 3 1$ </td></tr></table>

Table 3: Classification performance on the Credit Card dataset (mean ± standard deviation over three random seeds).

<table><tr><td>Method</td><td>α</td><td>Macro-F1</td><td>Macro Recall</td></tr><tr><td>Real</td><td>0.00</td><td> $0 . 7 7 2 7 \pm 0 . 0 2 0 9$ </td><td> $0 . 7 4 0 7 \pm 0 . 0 0 4 9$ </td></tr><tr><td>ROS</td><td>0.25</td><td> $0 . 7 8 8 7 \pm 0 . 0 3 2 7$ </td><td> $0 . 8 0 5 5 \pm 0 . 0 1 9 3$ </td></tr><tr><td>SMOTE</td><td>0.50</td><td> $\mathbf { 0 . 7 9 0 1 \pm 0 . 0 3 9 7 }$ </td><td> $\mathbf { 0 . 8 1 8 3 \pm 0 . 0 2 3 5 }$ </td></tr><tr><td rowspan="4">CGAN</td><td>0.25</td><td> $0 . 7 5 4 5 \pm 0 . 0 4 3 9$ </td><td> $0 . 7 1 8 5 \pm 0 . 0 3 4 3$ </td></tr><tr><td>0.50</td><td> $0 . 7 5 1 4 \pm 0 . 0 4 4 8$ </td><td> $0 . 7 1 2 1 \pm 0 . 0 3 7 5$ </td></tr><tr><td>0.75</td><td> $0 . 7 4 5 2 \pm 0 . 0 4 6 7$ </td><td> $0 . 7 1 2 8 \pm 0 . 0 3 0 7$ </td></tr><tr><td>1.00</td><td> $0 . 7 4 9 0 \pm 0 . 0 4 7 6$ </td><td> $0 . 7 1 5 1 \pm 0 . 0 3 2 8$ </td></tr><tr><td rowspan="4">CWGAN-GP</td><td>0.25</td><td> $0 . 7 4 1 3 \pm 0 . 0 5 5 3$ </td><td> $0 . 7 0 0 6 \pm 0 . 0 4 7 6$ </td></tr><tr><td>0.50</td><td> $0 . 7 2 9 5 \pm 0 . 0 6 7 7$ </td><td> $0 . 6 9 6 0 \pm 0 . 0 5 3 1$ </td></tr><tr><td>0.75</td><td> $0 . 7 2 7 9 \pm 0 . 0 6 7 9$ </td><td> $0 . 6 8 8 6 \pm 0 . 0 6 2 8$ </td></tr><tr><td>1.00</td><td> $0 . 7 2 3 6 \pm 0 . 0 7 6 4$ </td><td> $0 . 6 8 8 3 \pm 0 . 0 6 3 9$ </td></tr></table>

Table 4: Classification performance on the Forest Cover dataset (mean ± standard deviation over three random seeds).

Tables 3 and 4 summarize the classification results for the Credit Card and Forest Cover datasets, respectively. For ROS and SMOTE, we report the best-performing augmentation level across the same candidate levels used for the generative methods. For CGAN and CWGAN-GP, results are reported separately for each value of α to evaluate sensitivity to augmentation strength. To improve readability, only the Random Forest (RF) and Multi-Layer Perceptron (MLP) classifiers are reported in the main text, while additional classifier results are provided in the Appendix ??.

For the Credit Card dataset, all augmentation methods achieve performance close to the real-data baseline. Random Oversampling obtains the highest Macro-F1 and Macro Recall, followed closely by SMOTE. Among the generative methods, CGAN and CWGAN-GP produce comparable classification performance across augmentation strengths, despite the much lower Wasserstein discrepancy achieved by CWGAN-GP. This suggests that improved distri butional fidelity does not necessarily translate into large predictive gains in the binary setting.

For the Forest Cover dataset, SMOTE achieves the strongest overall performance, with a Macro-F1 of 0.7901 and Macro Recall of 0.8183. Random Oversampling is also highly competitive, while the real-data baseline remains stronger than both generative methods. CGAN outperforms CWGAN-GP in predictive performance across all augmentation levels, even though CWGAN-GP has lower Wasserstein discrepancy. This contrast provides important evidence that distributional fidelity and downstream predictive utility are related but not equivalent.

To assess whether observed performance diferences are statistically significant, paired Wilcoxon signed-rank tests were conducted across the three random seeds. No statistically significant diferences were observed between CGAN and CWGAN-GP in Macro-F1 performance on either dataset $( p > 0 . 0 5 )$ , despite the substantially lower Wasserstein discrepancies achieved by CWGAN-GP.

Taken together, the results highlight an important distinction between distributional fidelity and predictive utility. Although the previous subsection demonstrated that CWGAN-GP consistently achieves lower Wasserstein discrepancies than CGAN, these improvements do not universally translate into superior classification performance. In particular, the Forest Cover experiments show that classical oversampling methods remain highly competitive from a predictive perspective. This observation motivates the more detailed minority-class analysis presented next and provides an important empirical context for interpreting the proposed Wasserstein-based risk bounds.

![](images/53214973766efd241de8bfeab905197907e3488c12303f22be8e5e236167ec36.jpg)  
(a) Credit Card

![](images/ae5041d8d6bca1a4ecb676b4413fc639b4edd637650831dcae70fb894dfaa3c6.jpg)  
(b) Forest Cover  
Figure 1: Macro-F1 as a function of augmentation strength α for the Credit Card and Forest Cover datasets.

Figure 1 provides a visual summary of the efect of augmentation strength on classification performance. For both CGAN and CWGAN-GP, increasing the proportion of synthetic samples generally leads to a gradual reduction in Macro-F1. In contrast, the performance of SMOTE and Random Oversampling remains relatively stable across augmentation strengths. These results suggest that excessive reliance on synthetic data may adversely afect predictive performance despite improvements in distributional fidelity.

## 5.3 Minority-Class Analysis

While aggregate metrics such as Macro-F1 and Macro Recall provide an overall assessment of classification performance, they may obscure behavior on underrepresented classes. To better understand the efect of augmentation on rare categories, we examine class-specific recall for the four least frequent classes in the Forest Cover dataset: Aspen, Cottonwood/Willow, Douglas Fir, and Krummholz.

Since the Credit Card dataset involves only two classes, Macro Recall already provides a direct measure of minority-class detection performance, and a separate class-wise analysis is therefore omitted.
<table><tr><td>Class</td><td>Real</td><td>CGAN</td><td>CWGAN-GP</td><td>SMOTE</td></tr><tr><td>Aspen</td><td> $0 . 3 7 1 \pm 0 . 0 2 0$ </td><td> $0 . 3 7 1 \pm 0 . 0 3 9$ </td><td> $0 . 3 6 9 \pm 0 . 0 2 5$ </td><td> $\mathbf { 0 . 6 4 1 \pm 0 . 0 2 2 }$ </td></tr><tr><td>Cottonwood/Willow</td><td> $0 . 7 1 8 \pm 0 . 0 5 1$ </td><td> $0 . 7 4 2 \pm 0 . 0 4 9$ </td><td> $0 . 7 2 3 \pm 0 . 0 5 7$ </td><td> $\mathbf { 0 . 8 2 2 \pm 0 . 0 3 5 }$ </td></tr><tr><td>Douglas Fir</td><td> $0 . 6 2 5 \pm 0 . 0 2 6$ </td><td> $0 . 6 2 8 \pm 0 . 0 2 9$ </td><td> $0 . 6 3 0 \pm 0 . 0 3 3$ </td><td> $\mathbf { 0 . 7 5 9 \pm 0 . 0 1 0 }$ </td></tr><tr><td>Krummholz</td><td> $0 . 8 0 3 \pm 0 . 0 1 2$ </td><td> $0 . 8 1 1 \pm 0 . 0 1 6$ </td><td> $0 . 8 1 1 \pm 0 . 0 2 1$ </td><td> $\mathbf { 0 . 8 7 8 \pm 0 . 0 1 6 }$ </td></tr></table>

Table 5: Minority-class recall for the Forest Cover dataset (mean ± standard deviation over three random seeds). Results are reported using the best-performing augmentation strength for each method: $\alpha = 0 . 2 5$ for CGAN and CWGAN-GP, $\alpha = 0 . 5 0$ for SMOTE, and $\alpha = 0 . 0 0$ for the real-data baseline.

The results indicate that CGAN and CWGAN-GP provide only modest improvements over the real-data baseline. Small gains are observed for Cottonwood/Willow, Douglas Fir, and Krummholz, while recall for Aspen remains largely unchanged. In contrast, SMOTE consistently achieves the highest recall across all four minority classes, with particularly large improvements for Aspen and Douglas Fir. These improvements reflect SMOTE’s ability to directly increase minority-class representation through interpolation-based oversampling.

![](images/698226e99e886bc46031204765cd0496511015c1cd02d6c9fa85ead2c9e6ca75.jpg)  
Figure 2: Recall of the four least frequent classes in the Forest Cover dataset using the Random Forest classifier. Results are reported for the best-performing augmentation strength for each method $( \alpha = 0 . 2 5$ for CGAN and CWGAN-GP, $\alpha = 0 . 5 0$ for SMOTE, and $\alpha = 0$ for the real-data baseline).

Interestingly, the recall values obtained by CGAN and CWGAN-GP are often very similar despite the substantial diferences in their Wasserstein fidelity reported in Table 2. For example, CWGAN-GP achieves considerably lower Wasserstein discrepancies than CGAN, yet both methods produce nearly identical recall for Douglas Fir and Krummholz. This observation suggests that improvements in distributional fidelity do not necessarily translate into proportional gains in minority-class predictive performance.

Figure 2 provides a visual summary of these results. Although CWGAN-GP achieves substantially lower Wasserstein discrepancies than CGAN, the resulting minority-class recall values remain comparable across most classes. This finding reinforces a central conclusion of the study: distributional fidelity and predictive utility are related but distinct aspects of augmentation quality. While Wasserstein-based generative models produce higher-fidelity synthetic samples, improvements in distributional approximation do not necessarily translate into proportional gains in downstream classification performance.

## 5.4 Empirical Validation of the Theory

The empirical results support the theoretical role of Wasserstein discrepancy in the proposed risk bounds. Table 2 shows that CWGAN-GP yields smaller Wasserstein approximation errors than CGAN on both datasets, implying a smaller augmentation-bias term in Theorem 3.1. The dependence of Wasserstein discrepancy on augmentation strength further supports Theorem 3.1. For the Credit Card dataset, the Wasserstein distance increased from 89,924 to 166,518 under CGAN as α increased from 0.25 to 1.00. CWGAN-GP exhibited the same trend but with substantially smaller discrepancies, ranging from 6,340 to 11,744. Since the theoretical risk distortion term scales as αε, these results suggest that stronger augmentation magnifies the impact of distributional approximation error. However, Tables 3 and 4 show that lower Wasserstein discrepancy does not necessarily produce the highest Macro-F1 or Macro Recall. This behavior is consistent with Theorem 3.3, where predictive performance depends not only on augmentation bias but also on hypothesis class capacity and finite-sample estimation efects.

The empirical findings therefore provide partial validation of the proposed theory. The Wasserstein approximation error successfully explains diferences in distributional fidelity between CGAN and CWGAN-GP and behaves consistently with the theoretical risk bounds. However, predictive performance is also influenced by classifier capacity, finite-sample efects, and the interaction between synthetic and real observations. These findings support the interpretation of Wasserstein discrepancy as a measure of augmentation reliability rather than a direct predictor of classification accuracy.

## 6 Discussion

This study investigated conditional generative augmentation from both theoretical and empirical perspectives. The proposed framework establishes a direct connection between generative fidelity and classification risk through Wasserstein-based approximation error, providing a principled basis for evaluating the reliability of synthetic data augmentation. Unlike many existing studies that assess augmentation methods primarily through predictive performance, the present work emphasizes distributional fidelity as a fundamental property of synthetic data quality.

The empirical results demonstrate that CWGAN-GP consistently achieves lower Wasserstein discrepancies than CGAN across both the Credit Card and Forest Cover datasets. These findings are consistent with the theoretical analysis, which identifies the class-conditional Wasserstein distance as a key quantity governing augmentation-induced risk distortion. From a distributional perspective, the results indicate that Wasserstein-based adversarial training produces synthetic samples that more closely approximate the underlying data-generating process.

At the same time, the experiments reveal that improvements in distributional fidelity do not necessarily translate into superior classification performance. In several settings, classical oversampling methods such as SMOTE achieved higher Macro-F1 scores and minority-class recall than either generative augmentation approach. This observation highlights an important distinction between synthetic data fidelity and predictive utility. While fidelity measures how accurately a generator reproduces the underlying distribution, classification performance depends additionally on factors such as class separability, decision-boundary complexity, sample size, and the inductive bias of the downstream learning algorithm.

An interesting observation is that SMOTE frequently achieved higher classification performance than both CGAN and CWGAN-GP, particularly on the Forest Cover dataset. Because the Forest Cover dataset contains a relatively large number of minority-class observations, interpolation-based oversampling may be suficient to improve decision-boundary estimation without requiring accurate modeling of the full class-conditional distribution. By contrast, generative augmentation seeks to approximate the entire class-conditional distribution, a substantially more dificult task that may not yield proportional gains in predictive performance when the primary challenge is class imbalance rather than distributional sparsity. These results suggest that higher-fidelity synthetic data does not necessarily imply superior classification accuracy and that the efectiveness of augmentation depends on both the data characteristics and the downstream learning objective.

The results further suggest that augmentation strength should be viewed as a datadependent parameter rather than a universally optimal quantity. For the Forest Cover dataset, increasing the proportion of synthetic data generally led to reduced predictive performance, whereas the Credit Card dataset exhibited modest improvements under stronger CWGAN-GP augmentation. These contrasting behaviors indicate that the efectiveness of augmentation depends on the interaction between dataset characteristics, class imbalance structure, and generative fidelity.

Taken together, the findings support the central premise of the proposed framework: generative augmentation should be evaluated not only through predictive metrics but also through distributional reliability. Wasserstein-based fidelity provides a principled measure of how closely synthetic data approximates the underlying class-conditional distributions and therefore ofers a theoretically grounded perspective on augmentation quality. More broadly, the study suggests that distributional fidelity and predictive performance should be regarded as complementary evaluation criteria rather than interchangeable measures of success.

Limitations Several limitations should be acknowledged. First, the empirical evaluation is limited to two structured tabular datasets; additional studies across a broader range of domains would strengthen the generality of the conclusions. Second, the theoretical analysis relies on standard Lipschitz continuity assumptions and Wasserstein-based distributional comparisons, which may not fully capture the behavior of highly complex models in highdimensional settings. Third, only CGAN and CWGAN-GP architectures were considered, while newer generative models such as difusion and transformer-based approaches may exhibit diferent fidelity–performance relationships. Finally, the proposed framework provides guarantees on distributional fidelity and augmentation-induced risk distortion rather than direct guarantees of classification improvement.

## 7 Conclusion

This work developed a statistical framework for analyzing conditional generative augmentation in imbalanced classification. By modeling augmentation as a distribution-mixing process, we derived Wasserstein-based risk bounds that explicitly connect generative fidelity, augmentation strength, and classification risk. The resulting theory provides a principled foundation for understanding how distributional approximation error influences the reliability of synthetic data augmentation.

Empirical evaluations on the Credit Card Default and Forest Cover datasets support the theoretical framework. Across both datasets, CWGAN-GP consistently achieved lower Wasserstein discrepancies than CGAN, indicating closer approximation of the underlying class-conditional distributions. At the same time, the results showed that improved distributional fidelity does not necessarily yield superior classification performance, as classical oversampling methods such as SMOTE often remained competitive or outperformed generative approaches. These findings highlight that distributional reliability and predictive utility are related but distinct aspects of synthetic data quality.

Overall, the proposed framework extends the evaluation of synthetic augmentation beyond predictive accuracy alone by introducing distributional reliability as a theoretically grounded criterion. Rather than viewing augmentation solely as a tool for improving classification performance, this perspective provides a principled approach for assessing the quality and trustworthiness of synthetic data in machine learning applications.

## 8 Future Work

A particularly promising direction is the development of adaptive augmentation strategies in which augmentation strength is selected on a class-by-class basis. The empirical results suggest that a fixed augmentation level may not be optimal across all classes or datasets. Future work will investigate augmentation schemes that incorporate class imbalance, generative fidelity, and structural characteristics of the data to determine class-specific augmentation strengths with accompanying theoretical guarantees.

## Code Availability

The code used to generate all experimental results, figures, and tables will be made publicly available upon acceptance.

## References

[1] H. He and E. A. Garcia, “Learning from imbalanced data,” IEEE Transactions on Knowledge and Data Engineering, vol. 21, no. 9, pp. 1263–1284, 2009.

[2] M. Lucic, K. Kurach, M. Michalski, S. Gelly, and O. Bousquet, “Are gans created equal? a large-scale study,” in Advances in Neural Information Processing Systems, vol. 31, pp. 700–709, 2018.

[3] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” in Advances in Neural Information Processing Systems, vol. 27, pp. 2672–2680, Curran Associates, Inc., 2014.

[4] M. Arjovsky, S. Chintala, and L. Bottou, “Wasserstein generative adversarial networks,” in Proceedings of the 34th International Conference on Machine Learning (D. Precup and Y. W. Teh, eds.), vol. 70 of Proceedings of Machine Learning Research, pp. 214–223, PMLR, 2017.

[5] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville, “Improved training of wasserstein gans,” in Advances in Neural Information Processing Systems, vol. 30, pp. 5767–5777, 2017.

[6] G. Douzas and F. Bacao, “Efective data generation for imbalanced learning using generative adversarial networks,” Information Sciences, vol. 465, pp. 1–20, 2018.

[7] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “Smote: Synthetic minority over-sampling technique,” Journal of Artificial Intelligence Research, vol. 16, pp. 321–357, 2002.

[8] M. Mirza and S. Osindero, “Conditional generative adversarial nets,” in Proceedings of the Workshop on Challenges in Representation Learning, ICML, 2014.

[9] M. Frid-Adar, I. Diamant, E. Klang, M. Amitai, J. Goldberger, and H. Greenspan, “Gan-based synthetic medical image augmentation for increased cnn performance,” Neurocomputing, vol. 321, pp. 321–331, 2018.

[10] S. Ben-David, J. Blitzer, K. Crammer, A. Kulesza, F. Pereira, and J. Vaughan, “A theory of learning from diferent domains,” Machine Learning, vol. 79, no. 1–2, pp. 151–175, 2010.

[11] I. Redko, A. Habrard, M. Sebban, and R. Flamary, “Optimal transport for multi-source domain adaptation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 8, pp. 1853–1865, 2017.

[12] P. L. Bartlett and S. Mendelson, “Rademacher and gaussian complexities: Risk bounds and structural results,” Journal of Machine Learning Research, vol. 3, pp. 463–482, 2002.

[13] M. Mohri, A. Rostamizadeh, and A. Talwalkar, Foundations of Machine Learning. MIT Press, 2 ed., 2018.

## Appendix

## A Classifier-Specific Performance Results

This appendix reports classifier-specific results for the Random Forest (RF) and Multi-Layer Perceptron (MLP) models. While the main text presents aggregated performance summaries, the tables below provide a more detailed view of how individual classifiers respond to diferent augmentation strategies and augmentation strengths.

## A.1 Credit Card Dataset

<table><tr><td>Method</td><td>α</td><td>RF</td><td>MLP</td></tr><tr><td>Real</td><td>0.00</td><td> $0 . 6 7 9 8 \pm 0 . 0 0 5 2$ </td><td> $0 . 6 6 4 2 \pm 0 . 0 0 9 0$ </td></tr><tr><td>CGAN</td><td>0.25 0.50 0.75 1.00</td><td> $0 . 6 8 0 1 \pm 0 . 0 0 4 2$   $0 . 6 7 7 4 \pm 0 . 0 0 4 9$   $0 . 6 7 9 3 \pm 0 . 0 0 3 8$   $0 . 6 7 9 0 \pm 0 . 0 0 6 1$ </td><td> $0 . 6 6 5 5 \pm 0 . 0 0 5 6$   $0 . 6 7 2 8 \pm 0 . 0 0 4 6$   $0 . 6 5 8 4 \pm 0 . 0 2 7 2$   $0 . 6 7 3 8 \pm 0 . 0 1 1 4$ </td></tr><tr><td>CWGAN-GP</td><td>0.25 0.50 0.75 1.00</td><td> $0 . 6 8 0 3 \pm 0 . 0 0 6 0$   $0 . 6 7 9 8 \pm 0 . 0 0 7 4$   $0 . 6 8 0 3 \pm 0 . 0 0 7 3$   $0 . 6 8 3 2 \pm 0 . 0 0 6 2$ </td><td> $0 . 6 6 3 9 \pm 0 . 0 1 2 7$   $0 . 6 6 8 7 \pm 0 . 0 0 8 6$   $0 . 6 6 8 1 \pm 0 . 0 1 5 7$   $0 . 6 5 9 5 \pm 0 . 0 1 1 0$ </td></tr><tr><td>ROS</td><td>0.25 0.50 0.75 1.00</td><td> $0 . 6 7 4 3 \pm 0 . 0 0 5 8$   $0 . 6 8 1 9 \pm 0 . 0 0 5 3$   $0 . 6 8 6 4 \pm 0 . 0 0 6 1$   $0 . 6 8 7 7 \pm 0 . 0 0 5 0$ </td><td> $0 . 6 7 5 0 \pm 0 . 0 0 9 2$   $0 . 6 8 0 0 \pm 0 . 0 1 2 8$   $0 . 6 6 3 0 \pm 0 . 0 0 5 3$   $0 . 6 4 2 2 \pm 0 . 0 1 7 8$ </td></tr><tr><td>SMOTE</td><td>0.25 0.50 0.75 1.00</td><td> $0 . 6 8 5 6 \pm 0 . 0 0 5 3$   $0 . 6 8 9 9 \pm 0 . 0 0 5 9$   $0 . 6 8 8 2 \pm 0 . 0 0 2 1$   $0 . 6 8 7 4 \pm 0 . 0 0 7 9$ </td><td> $0 . 6 6 9 7 \pm 0 . 0 1 1 4$   $0 . 6 6 5 6 \pm 0 . 0 0 0 3$   $0 . 6 5 9 2 \pm 0 . 0 0 3 9$   $0 . 6 4 2 9 \pm 0 . 0 1 1 1$ </td></tr></table>

Table 1: Credit Card dataset: classifier-specific Macro-F1 scores (mean ± standard deviation over three random seeds).

Table 1 presents classifier-specific Macro-F1 scores for the Credit Card dataset. Similar to the Forest Cover results, Random Forest consistently achieves higher performance than MLP across most augmentation settings. The improvements obtained through Random Oversampling and SMOTE are particularly pronounced for RF, suggesting that the classifier benefits from increased minority-class representation. In contrast, the performance diferences between CGAN and CWGAN-GP remain relatively small despite the substantial reductions in Wasserstein discrepancy achieved by CWGAN-GP. This pattern is observed for both RF and MLP, indicating that improvements in generative fidelity do not necessarily produce proportional gains in predictive performance. Importantly, the overall ordering of augmentation methods remains stable across classifiers, providing additional evidence that the conclusions drawn from the aggregated results are not driven by a particular modeling choice.

<table><tr><td>Method</td><td>α</td><td>RF</td><td>MLP</td></tr><tr><td>Real</td><td>0.00</td><td> $0 . 7 9 1 2 \pm 0 . 0 0 3 1$ </td><td> $0 . 7 5 4 2 \pm 0 . 0 0 7 7$   $0 . 7 1 6 1 \pm 0 . 0 1 9 3$ </td></tr><tr><td>CGAN</td><td>0.25 0.50 0.75 1.00</td><td> $0 . 7 9 2 9 \pm 0 . 0 0 4 9$   $0 . 7 9 1 8 \pm 0 . 0 0 5 5$   $0 . 7 8 7 1 \pm 0 . 0 0 8 6$   $0 . 7 9 2 1 \pm 0 . 0 0 4 1$ </td><td> $0 . 7 1 1 0 \pm 0 . 0 0 8 8$   $0 . 7 0 3 3 \pm 0 . 0 1 0 1$   $0 . 7 0 5 9 \pm 0 . 0 0 9 4$ </td></tr><tr><td>CWGAN-GP</td><td>0.25 0.50 0.75 1.00</td><td> $0 . 7 9 1 5 \pm 0 . 0 0 4 1$   $0 . 7 9 0 7 \pm 0 . 0 0 3 1$   $0 . 7 8 7 7 \pm 0 . 0 0 8 7$   $0 . 7 9 2 1 \pm 0 . 0 0 2 0$ </td><td> $0 . 6 9 1 0 \pm 0 . 0 0 6 6$   $0 . 6 6 8 3 \pm 0 . 0 1 3 6$   $0 . 6 6 8 2 \pm 0 . 0 2 7 6$   $0 . 6 5 5 0 \pm 0 . 0 2 2 2$ </td></tr><tr><td>ROS</td><td>0.25 0.50 0.75 1.00</td><td> $0 . 8 1 7 2 \pm 0 . 0 0 9 1$   $0 . 8 1 6 4 \pm 0 . 0 1 1 3$   $0 . 8 1 5 9 \pm 0 . 0 0 8 4$   $0 . 8 1 6 6 \pm 0 . 0 0 6 4$ </td><td> $0 . 7 6 0 1 \pm 0 . 0 1 2 1$   $0 . 7 5 5 1 \pm 0 . 0 0 2 8$   $0 . 7 4 7 7 \pm 0 . 0 0 9 8$   $0 . 7 4 6 7 \pm 0 . 0 1 5 3$ </td></tr><tr><td>SMOTE</td><td>0.25 0.50 0.75 1.00</td><td> $0 . 8 2 1 4 \pm 0 . 0 0 5 1$   $0 . 8 2 5 1 \pm 0 . 0 1 1 8$   $0 . 8 1 9 4 \pm 0 . 0 0 8 6$   $0 . 8 1 7 7 \pm 0 . 0 1 0 1$ </td><td> $0 . 7 5 8 6 \pm 0 . 0 1 6 3$   $0 . 7 5 5 1 \pm 0 . 0 1 1 3$   $0 . 7 5 9 9 \pm 0 . 0 1 3 1$   $0 . 7 4 9 0 \pm 0 . 0 0 8 1$ </td></tr></table>

Table 2: Forest Cover dataset: classifier-specific Macro-F1 scores (mean ± standard deviation over three random seeds).

Table 2 reports classifier-specific Macro-F1 scores for the Forest Cover dataset. Several observations emerge. First, Random Forest consistently outperforms MLP across all augmentation strategies, indicating that tree-based ensemble methods are particularly efective for modeling the complex nonlinear structure of the Forest Cover data. Second, the relative ranking of augmentation methods remains largely unchanged across classifiers. In both RF and MLP, Random Oversampling and SMOTE achieve the strongest performance, while CGAN and CWGAN-GP generally remain below the real-data baseline. Third, although CWGAN-GP achieves substantially lower Wasserstein discrepancies than CGAN, these improvements do not translate into superior classifier performance for either RF or MLP. This finding further supports the central conclusion of the paper that distributional fidelity and predictive utility capture distinct aspects of augmentation quality. Overall, the classifier-specific results demonstrate that the main conclusions reported in the aggregated analysis are robust across diferent learning algorithms.

## B Experimental Hyperparameters

Tables 3 summarize the hyperparameters used for all experiments. Unless otherwise stated, identical classifier settings were used across augmentation methods to ensure fair comparisons.

<table><tr><td>Parameter</td><td>Credit Card</td><td>Forest Cover</td></tr><tr><td>Number of Runs</td><td>3</td><td>3</td></tr><tr><td>Augmentation Strengths (α)</td><td>0.25, 0.50, 0.75, 1.00</td><td>0.25, 0.50, 0.75, 1.00</td></tr><tr><td>Train/Test Split</td><td>Stratified 80% / 20%</td><td>Stratified 80% / 20%</td></tr><tr><td>Latent Dimension</td><td>64</td><td>32</td></tr><tr><td>Batch Size</td><td>256</td><td>64</td></tr><tr><td>Training Epochs</td><td>300</td><td>300</td></tr><tr><td>Optimizer</td><td>Adam</td><td>Adam</td></tr><tr><td>Learning Rate</td><td>10-4</td><td>10-4</td></tr><tr><td colspan="3">CWGAN-GP</td></tr><tr><td>Critic Updates per Generator Step</td><td>3</td><td>5</td></tr><tr><td>Gradient Penalty Coefficient (λGP)</td><td>10</td><td>10</td></tr><tr><td colspan="3">Random Forest</td></tr><tr><td>Number of Trees</td><td>100</td><td>100</td></tr><tr><td>Criterion</td><td>Gini</td><td>Gini</td></tr><tr><td colspan="3">MLP</td></tr><tr><td>Hidden Layers</td><td>(128, 64)</td><td>(128, 64)</td></tr><tr><td>Activation Function</td><td>ReLU</td><td>ReLU</td></tr><tr><td>Maximum Iterations</td><td>2000</td><td>2000</td></tr></table>

Table 3: Hyperparameters used for the Credit Card and Forest Cover experiments.

## Code Availability

The code used to generate all experimental results and figures reported in this study is publicly available at:

https://github.com/yourusername/generative-augmentation-reliability

The repository contains implementations of CGAN, CWGAN-GP, evaluation metrics, and scripts for reproducing all experiments.