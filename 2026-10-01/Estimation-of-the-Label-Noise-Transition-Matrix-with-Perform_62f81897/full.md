# Estimation of the Label-Noise Transition Matrix with Performance Guarantees via Selective Classification

Xabier de Juan<sup>1</sup> Santiago Mazuelas<sup>1,2</sup> Yilun Zhu<sup>3</sup> Clayton Scott<sup>3</sup>

<sup>1</sup>Basque Center of Applied Mathematics (BCAM)

<sup>2</sup>IKERBASQUE-Basque Foundation for Science

<sup>3</sup>Electrical and Computer Engineering, University of Michigan {xdejuan, smazuelas}@bcamath.org {allanzhu, clayscot}@umich.edu

## Abstract

Modern machine learning depends heavily on massive datasets, but obtaining highquality annotations at scale is often expensive. As a result, learning from noisilylabeled data has become common, making accurate estimation of the label-noise transition matrix crucial. However, existing transition matrix estimators rely on the fragile estimation of class-posteriors and do not provide finite-sample performance guarantees. In this work, we propose a novel methodology to estimate the transition matrix based on one-sided selective classification. This approach bypasses class-posterior estimation, provides finite-sample performance guarantees, and leverages flexible learning methods for binary classification. Moreover, we introduce effective algorithms to implement the proposed methodology and provide their refined finite-sample performance bounds.

## 1 Introduction

The success of modern machine learning relies heavily on the availability of massive labeled datasets. However, obtaining high-quality annotations at scale is often cost-prohibitive, leading to the widespread adoption of cost-effective and less accurate labeling procedures [1–3]. Such compromise introduces label noise, where observed labels may differ from the underlying ground truth. The probabilities of these label flips are captured by the label-noise transition matrix.

An accurate estimate of the transition matrix can address many of the problems caused by label noise. For instance, loss correction techniques leverage the transition matrix to recover the Bayesoptimal classifier from noisy samples [4–9]. Beyond standard classification, a reliable estimate of the transition matrix can enable the construction of informative prediction sets for conformal prediction [10], the implementation of fair classification models under biased data [11], and the training of conditional diffusion models with noisy labels [12].

Existing transition matrix estimators heavily rely on the pointwise estimation of the noisy classposterior [7, 8, 13–17]. This requirement is fragile since even a poor estimate at a single point can lead to a large subsequent estimation error for the transition matrix [18]. In addition, estimating the class-posterior is provably hard. For example, any estimator of the class-posterior suffers from the curse of dimensionality [19, 20], and common methods often result in inaccurate class-posterior estimations [21].

While recent works have established consistency for certain estimators of the label-noise transition matrix [15, 16], the literature lacks finite-sample performance guarantees. Approaches in the closely related problem of mixture proportion estimation (MPE) [5, 6, 22–26] offer finite-sample performance guarantees. However, most of these approaches are restricted to binary settings, while others rely on intractable algorithms.

In this work, we introduce a methodology for estimating the label-noise transition matrix based on one-sided selective classification. This approach does not require pointwise class-posterior estimation and provides finite-sample performance guarantees. Moreover, our methodology allows us to estimate the transition matrix by adapting established techniques from selective classification. Specifically, the main contributions in the paper are as follows:

• We frame the estimation of each column of the transition matrix as a one-sided selective classification problem, where we minimize the false discovery rate subject to a minimum coverage constraint.

• We provide finite-sample error bounds for the proposed approach, and prove that our estimator achieves the parametric convergence rate up to a small bias. In particular, we show that our methodology avoids the curse of dimensionality, unlike existing estimators.

• We propose computationally tractable algorithms for our methodology that leverage general methods for binary classification.

• We provide a theoretical analysis of the proposed algorithms, showing that they admit refined finite-sample error bounds analogous to classical generalization bounds for classification.

## 2 Preliminaries

This section defines the label-noise transition matrix and describes existing methods for its estimation.

## 2.1 Label-noise transition matrix

Let $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ be the feature space and let $\mathcal { Y } = \{ 1 , 2 , \dots , | \mathcal { Y } | \}$ be the label set. Instance and cleanlabel pairs $( X , Y ) \in { \mathcal { X } } \times { \mathcal { Y } }$ follow an underlying distribution $\mathrm { p , }$ while their noisy-label counterparts $( X , \widetilde { Y } ) \in \mathcal { X } \times \mathcal { Y }$ are drawn from the distribution ${ \widetilde { \mathrm { p } } } .$ We assume that the distribution of $\widetilde { Y }$ depends on $Y$ , but not on $X , { \mathrm { i . e . , } } X$ and $\widetilde { Y }$ eare independent given $Y .$ e. This assumption is referred to as classedependent label noise and it is standard in the literature (see e.g., [7, 8, 13–18]). It has shown to be effective in practice [16, 18], even in scenarios where label noise may change across instances.

The label-noise transition matrix $\mathbf { T } \in [ 0 , 1 ] ^ { | \mathbb { Y } | \times | \mathbb { Y } | }$ is defined entry-wise as $T _ { i , j } = \mathbb { P } ( \widetilde { Y } = i \mid Y = j )$ for $i , j \in \mathcal { Y }$ , so that $T _ { i , j }$ eis the probability that an instance with true label j is observed as noisy label $i .$ We assume that T is a row-diagonally dominant matrix $( \mathrm { i . e . , } T _ { i , i } > T _ { i , j }$ for all $j \neq i )$ , paralleling common assumptions in the literature [8, 13, 15, 16, 27].

The goal is to estimate the matrix T using only noisy samples drawn from p. Specifically, we assume that we have access to n i.i.d. samples from $\bar { \tilde { \mathrm { p } } } , \{ ( x _ { k } , \tilde { y } _ { k } ) \bar  \} _ { k = 1 } ^ { n } \subseteq \mathcal { X } \times \mathcal { Y }$

## 2.2 Related work

The most common approach to estimate the transition matrix is based on identifying anchor points for each class [7, 8]. This method exploits the fact that clean and noisy class-posterior probabilities are related by the matrix T, that is, $\eta ^ { \mathrm { n o i s y } } ( x ) = \mathbf { T } \eta ( x )$ where $\eta _ { i } ^ { \mathrm { \scriptsize { n o i s y } } } ( x ) = \mathbb { P } ( \widetilde { Y } = i \mid X = x )$ and $\eta _ { i } ( x ) = \mathbb { P } ( \dot { Y } = i \mid X = x )$

The method relies on the anchor-point assumption [7, 8] which states that there are instances with class-posterior probability equal to one for each class. In particular, the anchor-based method exploits the fact that if $x ^ { ( j ) }$ is an anchor point for class $j ,$ , the entries of the transition matrix are given by $T _ { i , j } = \eta _ { i } ^ { \mathrm { n o i s y } } ( x ^ { ( j ) } )$

The anchor-based method [8] first obtains an estimate $\widehat { \eta } ^ { \mathrm { n o i s y } } ( x ) \in [ 0 , 1 ] ^ { | \mathfrak { Y } | }$ of the vector of noisy class-posteriors $\eta ^ { \mathrm { n o i s y } } ( x ) \ \in \ [ 0 , 1 ] ^ { | \mathfrak { Y } | }$ b. Then, for each class $j \in \mathcal Y$ , the algorithm estimates the corresponding column of T through the following procedure.

1. Find a candidate for anchor point of the j-th class by obtaining the instance that maximizes the j-th component of $\widehat { \eta } ^ { \mathrm { n o i s y } } , x ^ { ( j ) } \in \arg \operatorname* { m a x } _ { x \in \mathcal { X } } \widehat { \eta } _ { { j } } ^ { \mathrm { n o i s y } } ( x )$

2. Estimate the $( i , j )$ b-th entry of T as $\widehat { T } _ { i , j } ^ { \mathrm { a n c h o r } } ( x ^ { ( j ) } ) = \widehat { \eta } _ { i } ^ { \mathrm { n o i s y } } ( x ^ { ( j ) } )$

Subsequent methods eliminated the explicit search for anchor points [13–17], yet they still critically rely on an accurate pointwise estimation of the class-posterior. This dependency creates a severe vulnerability since a poorly estimated class-posterior at a single point can produce a large estimation error for the transition matrix [18]. Furthermore, common methods (e.g., neural networks) are known to yield highly unreliable class-posterior estimates (see e.g., [21]). Specifically, if α is the smoothness parameter of the class-posterior and d is the dimension of the feature space, the minimax optimal pointwise estimation error of the class-posterior is $n ^ { - \alpha / ( 2 \alpha + d ) }$ [19, 20], yielding a prohibitively slow convergence rate, even in moderate dimensions. The recent work [18] bypasses the need for class-posterior estimation and instead relies on the minimization of noise-robust losses. However, this approach relies on multiple assumptions that are unlikely to hold in practice. For instance, Assumption 1 in [18] requires that a minimizer of cross-entropy is the only minimizer of the noise-robust loss, a condition that is often not satisfied (see e.g., [28, 29]).

Existing literature does not provide finite-sample performance guarantees and solely consistency has been established for some estimators [15, 16]. Estimators in the closely related field of MPE [5, 6, 22–26] instead estimate the inverse flip-rates $\mathbb { P } ( Y = i \mid \widetilde { Y } = j )$ , and offer attractive finiteesample performance guarantees that decay at the parametric rate of $O ( 1 / \sqrt { n } )$ up to a small bias term. However, the theoretical elegance of these MPE estimators comes at the cost of severe practical limitations. For instance, the estimator in [5, 6] is based on identifying subsets of the feature space using an exhaustive search that relies on VC dimension bounds, which are known to be loose [5]. Alternatively, the estimators in [23, 25] are limited to the binary case.

In this work, we propose to estimate the transition matrix using methods for one-sided selective classification. Standard selective classification (or classification with a reject option) [30–33] allows models to abstain when uncertain. This is a crucial capability when errors are significantly more expensive than abstaining (e.g., in medical applications [34, 35]). Formally, the goal is to learn a standard classifier $f$ and a selection function $h \colon \mathcal { X }  \ \{ \mathtt { A } , \mathtt { R } \}$ [30] that together minimize the probability of misclassification on the accepted samples, $\mathbb { P } ( { \overset { \cdot } { Y } } \neq f ( X ) \mid h ( X ) = \mathtt { A } )$ , subject to a minimum coverage constraint for the selection function $\mathbb { P } ( h ( X ) = \dot { \mathbb { A } } ) \overset { \cdot } { \geq } \dot { \gamma }$ , where $\gamma \in ( 0 , 1 )$

One-sided selective classification [36] simplifies the standard setting by fixing the prediction to a target class $j \in \mathcal { Y } \ ( \mathrm { i . e . , } \ f ( x ) = j , \forall x \in \mathcal { X } )$ . This problem arises in scenarios where it is necessary to isolate a high-purity subset of a single class, for instance to perform high-confidence auditing. The goal then reduces to learning a selection function h that minimizes the false discovery rate $\mathbb { P } ( Y \neq j \mid h ( X ) = \mathtt { A } )$ for the class $y = j$ under the coverage constraint $\mathbb { P } ( h ( X ) = \mathtt { A } ) \ge \gamma$

## 3 Transition matrix estimation via one-sided selective classification

In this section, we present a novel methodology for estimating the transition matrix based on onesided selective classification. For each class $j \in \mathcal Y$ , we estimate the corresponding column of T through the following procedure.

1. Learn a selection function $h ^ { ( j ) } : \mathcal { X }  \{ \mathtt { A } , \mathtt { R } \}$ which minimizes the false discovery rate for the noisy label $\widetilde y = j$ , subject to a minimum coverage constraint.

2. Estimate the $j \cdot$ -th column of T by computing the empirical label probabilities of instances accepted by $h ^ { ( j ) }$

Formally, we aim to solve for each class $j \in \mathcal Y$ the following problem

$$
\begin{array} { r l } { \widetilde { R } _ { \gamma } ^ { ( j ) } = \underset { h } { \mathrm { m i n } } } & { \mathbb { P } ( \widetilde { Y } \neq j \mid h ( X ) = \mathtt { A } ) } \\ { \mathrm { s . t . } } & { \mathbb { P } ( h ( X ) = \mathtt { A } ) \ge \gamma } \end{array}\tag{1}
$$

where $\gamma \in ( 0 , 1 )$ is the minimum coverage constraint and the optimization is over all measurable selection functions $h \colon \mathcal { X }  \{ { \tt A } , { \tt R } \}$ . The smallest possible false discovery rate subject to the minimum coverage constraint is denoted by $\widetilde { R } _ { \gamma } ^ { ( j ) }$

The estimate $\widehat { \bf T }$ is obtained using the selection functions and noisy samples as follows.

Definition 1. Let $h ^ { ( j ) } : \mathcal { X }  \{ \mathtt { A } , \mathtt { R } \}$ be a selection function for the $j \mathrm { - t h }$ label, and let S be a subset of $\{ 1 , 2 , \ldots , n \}$ where $m = | S |$ . For every $i \in \mathcal { Y }$ define

$$
\widehat { T } _ { i , j } ( h ^ { ( j ) } ; S ) = \frac { \sum _ { k \in S } \mathbb { I } \{ h ^ { ( j ) } ( x _ { k } ) = \mathtt { A } , \widetilde { y } _ { k } = i \} } { \# m \big ( h ^ { ( j ) } \big ) }\tag{2}
$$

where $\begin{array} { r } { \# _ { m } \big ( h ^ { ( j ) } \big ) = \sum _ { k \in S } \mathbb { I } \{ h ^ { ( j ) } ( x _ { k } ) = \mathtt { A } \} } \end{array}$ . When the set of indices S is clear from the context we denote the estimator as $\widehat { T } _ { i , j } ( h ^ { ( j ) } )$

The estimator $\widehat { T } _ { i , j } ( h ^ { ( j ) } ; S )$ is well suited to estimating the transition matrix. In particular, we have $\widehat { T } _ { i , j } ( h ^ { ( j ) } ; S ) \approx \bar { T } _ { i , j }$ for a near-optimal solution $h ^ { ( j ) }$ of (1). This holds because $\widehat { T } _ { i , j } ( h ^ { ( j ) } ; S )$ is the bempirical version of $\mathbb { P } ( \widetilde { Y } = i \mid h ^ { ( j ) } ( X ) = \mathtt { A } )$ , which is close to $\mathbb { P } ( \widetilde { Y } = i \mid Y = j )$ for a selection function $h ^ { ( j ) }$ ewith small false discovery rate $\mathbb { P } ( \widetilde { Y } \neq j \ | \ h ^ { ( j ) } ( X ) = \mathtt { A } )$

The next theorem shows that the error of the proposed estimator decreases as

$$
| T _ { i , j } - \widehat { T } _ { i , j } ( h ^ { ( j ) } ) | \lesssim \varepsilon _ { \mathrm { b i a s } } + \frac { 1 } { \sqrt { \gamma n } }
$$

where $\gamma$ is the minimum coverage constraint, and $\varepsilon _ { \mathrm { b i a s } }$ is the bias incurred due to the false discovery rate of the selection function

Theorem 1. Let $\begin{array} { r l r } { h ^ { ( j ) } } & { { } : } & { \mathcal { X } \quad \to \quad \{ { \tt A } , { \tt R } \} } \end{array}$ be a selection function for the $j \mathrm { - t h }$ label with $\mathbb { P } ( h ^ { ( j ) } ( X ) = \mathtt { A } ) \ge \gamma$ , and $\varepsilon _ { \mathrm { o p t } } ( h ^ { ( j ) } )$ be its excess false discovery rate, that is

$$
\varepsilon _ { \mathrm { o p t } } ( h ^ { ( j ) } ) = \mathbb { P } ( \widetilde { Y } \neq j \mid h ^ { ( j ) } ( X ) = \mathtt { A } ) - \widetilde { R } _ { \gamma } ^ { ( j ) } .
$$

If $\widehat { T } _ { i , j } ( h ^ { ( j ) } ; S )$ for $| S | = m$ is the estimator in (2), then

$$
\operatorname* { m a x } _ { i \in \Psi } \vert T _ { i , j } - \widehat { T } _ { i , j } ( h ^ { ( j ) } ; S ) \vert \le C _ { T } \big ( R _ { \gamma } ^ { ( j ) } + \varepsilon _ { \mathrm { o p t } } ( h ^ { ( j ) } ) \big ) + \sqrt { \frac { 2 \log ( | \Psi | / \delta ) } { \# _ { m } ( h ^ { ( j ) } ) } }\tag{3}
$$

holds with probability at least $1 ~ - ~ \delta$ over the draw of the samples indexed by S, where $C _ { T } = 2 / ( T _ { j , j } - \operatorname* { m a x } _ { l \neq j } T _ { j , l } )$ and $R _ { \gamma } ^ { ( j ) } = \operatorname* { m i n } \{ \mathbb { P } ( Y \neq j \mid h ( X ) = \mathtt { A } ) \mathrm { s . t . } \mathbb { P } ( h ( X ) = \mathtt { A } ) \ge \gamma \}$ h Proof. See Appendix A.1. □

The theorem above provides finite-sample performance guarantees for the proposed methodology that show a bias-variance decomposition for the error.

The bias term is given by $C _ { T } \big ( R _ { \gamma } ^ { ( j ) } + \varepsilon _ { \mathrm { o p t } } ( h ^ { ( j ) } ) \big )$ , where the constant $C _ { T }$ acts as a condition number for the transition matrix T. This constant is inversely proportional to the margin $T _ { j , j } - \operatorname* { m a x } _ { l \neq j } T _ { j , l }$ meaning that the estimation problem is better conditioned when the diagonal dominance of T is more pronounced. The bound also incorporates a suboptimality gap, $\varepsilon _ { \mathrm { o p t } }$ , that accounts for the difficulty of finding the optimal selection function in (1) with finite data.

The term $R _ { \gamma } ^ { ( j ) }$ is the smallest false discovery rate on clean data. As discussed at the end of this section, such smallest error is small as long as a relaxed anchor-point assumption is satisfied. In particular, the anchor-point assumption corresponds to the case where $R _ { \gamma } ^ { ( j ) }  0$ as γ tends to 0. Therefore, Theorem 1 also provides performance guarantees for cases where the anchor-point assumption is not satisfied $( \mathrm { i . e . , ~ l i m } _ { \gamma \to 0 } R _ { \gamma } ^ { ( j ) } > 0 )$ . In these cases, the limit value represents an unavoidable positive bias, while the proposed methodology remains effective as long as $R _ { \gamma } ^ { ( j ) }$ takes small values.

The variance term $\sqrt { 2 \log ( | \mathcal { Y } | / \delta ) / \# _ { m } ( h ^ { ( j ) } ) }$ scales at the parametric rate of $O ( 1 / \sqrt { n } )$ . Specifically, the minimum coverage constraint, $\mathbb { P } ( h ^ { ( j ) } ( X ) \ : = \ : \mathtt { A } ) \ : \ge \ : \gamma$ , ensures that the number of accepted samples satisfies $\# _ { m } ( h ^ { ( j ) } ) \gtrsim \gamma n$ with high probability.

The minimum coverage γ controls the trade-off between the bias and the variance terms described above. Decreasing $\gamma$ reduces the value of $R _ { \gamma } ^ { ( j ) }$ , as a selection function that is required to accept fewer instances can provide a smaller false discovery rate. However, this reduction incurs a cost, as rejecting more samples shrinks $\# _ { m } ( h ^ { ( j ) } )$ , thereby increasing the variance. Choosing $\gamma = \omega ( 1 / n )$ implies that both $R _ { \gamma } ^ { ( j ) }$ and $1 / \sqrt { \# _ { m } ( h ^ { ( j ) } ) }$ decrease with the number of samples n. In particular, if $R _ { \gamma } ^ { ( j ) }$ decreases with $\gamma$ as $O ( \gamma ^ { \rho } )$ , taking $\gamma = \Theta ( n ^ { - 1 / ( 1 + 2 \rho ) } )$ results in $R _ { \gamma } ^ { ( j ) } + 1 / \sqrt { \# _ { m } \big ( h ^ { ( j ) } \big ) } = O \big ( n ^ { - 1 / 2 + 1 / ( 2 + 4 \rho ) } \big )$ , which is close to the parametric rate when $\rho$ is large.

Existing theoretical results for MPE methods provide performance guarantees for estimates of the inverse flip-rates $\mathbb { P } ( Y = i \mid { \tilde { Y } } = j )$ that also reveal a bias-variance decomposition, with the variance term scaling as $O ( \mathrm { { 1 } } / \sqrt { n } ) \ \mathrm { { [ } } 2 3 , 2 5 ]$ . However, these results are limited to the binary setting and to specific learning methods. For instance, the estimator in [25] is restricted to the level sets of a fixed scoring function, while [23] is limited to kernel-based methods. In contrast, our approach addresses the estimation of transition matrices offering a more general methodology since it covers the multiclass setting and allows the usage of any method for one-sided selective classification.

To the best of our knowledge, the bounds in Theorem 1 represent the first finite-sample performance guarantees for methods that estimate the label-noise transition matrix. Using the same line of reasoning, the next result provides finite-sample guarantees for the anchor-based estimator.

Proposition 1. Let $\widehat { \eta } ^ { \mathrm { n o i s y } }$ be an estimate of the noisy class-posteriors $\eta ^ { \mathrm { n o i s y } }$ . For every class $j \in \mathcal { Y }$ let $x ^ { ( j ) } \in \arg \operatorname* { m a x } _ { x \in \mathcal { X } } \widehat { \eta } _ { j } ^ { \mathrm { n o i s y } } ( x )$ , and define

$$
\varepsilon _ { \mathrm { o p t } } ^ { \mathrm { a n c h o r } } ( x ^ { ( j ) } ) = \mathbb { P } ( \widetilde { Y } \neq j \mid X = x ^ { ( j ) } ) - \widetilde { R } _ { 0 } ^ { ( j ) }
$$

where $\begin{array} { r c l c r c l } { \widetilde { R } _ { 0 } ^ { ( j ) } } & { = } & { \operatorname* { m i n } _ { x \in { \mathcal X } } \mathbb { P } ( \widetilde { Y } } & { \neq } & { j } & { \vert } & { X } & { = } & { x ) } \end{array}$ Then, the anchor-based estimator $\widehat { T } _ { i , j } ^ { \mathrm { a n c h o r } } ( \boldsymbol { x } ^ { ( j ) } ) = \widehat { \eta } _ { i } ^ { \mathrm { n o i s y } } ( \boldsymbol { x } ^ { ( j ) } )$ e satisfies

$$
\operatorname* { m a x } _ { i \in \mathbb { Y } } | T _ { i , j } - \widehat { T } _ { i , j } ^ { \mathrm { a n c h o r } } ( x ^ { ( j ) } ) | \leq C _ { T } ( R _ { 0 } ^ { ( j ) } + \varepsilon _ { \mathrm { o p t } } ^ { \mathrm { a n c h o r } } ( x ^ { ( j ) } ) ) + \operatorname* { m a x } _ { i \in \mathbb { Y } } | \eta _ { i } ^ { \mathrm { n o i s y } } ( x ^ { ( j ) } ) - \widehat { \eta } _ { i } ^ { \mathrm { n o i s y } } ( x ^ { ( j ) } ) |\tag{4}
$$

where $\begin{array} { r } { R _ { 0 } ^ { ( j ) } = \operatorname* { m i n } _ { x \in \mathcal { X } } \mathbb { P } ( Y \not = j \mid X = x ) } \end{array}$

Proof. See Appendix A.2.

A comparison between Theorem 1 and Proposition 1 indicates that the proposed methodology addresses the main limitation of the anchor-based method. More precisely, Proposition 1 shows that the anchor-based estimator suffers from the curse of dimensionality since it relies on an accurate classposterior estimate. In particular, there are scenarios where the last term of the upper bound in (4) is exactly of the order $n ^ { - \alpha / ( 2 \alpha + d ) }$ [19, 20], which is a very slow rate for real-world high-dimensional datasets.

The value of $R _ { 0 } ^ { ( j ) }$ quantifies deviations from the anchor-point assumption that corresponds to $R _ { 0 } ^ { ( j ) }$ being equal to zero. The value $R _ { \gamma } ^ { ( j ) }$ in Theorem 1 above plays a similar role to $R _ { 0 } ^ { ( j ) }$ but providing more detailed information about the hardness of estimating the transition matrix, as described above. In particular, we have that $R _ { \gamma } ^ { ( j ) }$ approaches $R _ { 0 } ^ { ( j ) }$ when $\gamma \  \ 0 .$ , and the speed with which $R _ { \gamma } ^ { ( j ) }$ decreases with γ determines the best possible rate of decrease for the estimation error, as described above.

The suboptimality term for the anchor-based method $\varepsilon _ { \mathrm { o p t } } ^ { \mathrm { a n c h o r } }$ also depends strongly on the pointwise estimation error of the class-posterior since $x ^ { ( j ) }$ is obtained as the instance with the highest class-posterior estimate. Therefore, there are scenarios where $\varepsilon _ { \mathrm { o p t } } ^ { \mathrm { a n c h o r } } \asymp n ^ { - \alpha / ( 2 \alpha + d ) }$ . In contrast, the suboptimality term for the presented methodology $\varepsilon _ { \mathrm { o p t } }$ is the excess false discovery rate in a problem of selective classification that is small under realistic assumptions. Specifically, the next section presents computationally tractable algorithms that provide selection functions with small suboptimality $\mathrm { g a p } \varepsilon _ { \mathrm { o p t } }$

## 4 Tractable algorithms with finite-sample performance guarantees

The learning methodology from Section 3 can be implemented utilizing general techniques for selective classification (also known as classification with rejection/abstention) [30, 31, 33]. In the following we present two effective approaches and provide their corresponding finite-sample performance guarantees.

## 4.1 Threshold selection approach

A prevalent approach within selective classification, classification with abstention, and classification with generalized metrics, is to rely on the confidence scores of a standard binary classifier [31, 37]. This approach can be adapted to our methodology for estimating the transition matrix as follows. First, split the noisy samples into two disjoint sets. Then, for each class $j \in \mathcal Y$ , the algorithm estimates the corresponding column of T through the following procedure.

1. Learn a binary scoring function $s ^ { ( j ) } : \mathcal { X } $ R to distinguish class $\widetilde { Y } = j$ from all others, using the samples in the first split. For every $\tau \in \mathbb { R }$ e, define a selection function as $h _ { \tau } ^ { ( j ) } ( x ) = \mathtt { A }$ if $s ^ { ( j ) } ( x ) \geq \tau$ , and $h _ { \tau } ^ { ( j ) } ( x ) = \mathtt { R }$ otherwise.

2. Select the threshold τ that minimizes the empirical version of $\mathbb { P } ( \widetilde { Y } \neq j \ | \ h _ { \tau } ^ { ( j ) } ( X ) = \mathtt { A } )$ (i.e., that maximizes $\widehat { T } _ { j , j } ( h _ { \tau } ^ { ( j ) } ) )$ e) while satisfying an empirical minimum coverage constraint, usbing the samples in the second split.

3. Estimate the $( i , j )$ -th entry of T as $\widehat { T } _ { i , j } ( h _ { \widehat { \tau } } ^ { ( j ) } )$ , using the samples in the second split.

This procedure is formalized in Algorithm 1 and theoretically analyzed in Theorem 2.

Algorithm 1 Transition matrix estimation via threshold selection   
Input: $\overline { { \{ ( x _ { k } , \widetilde { y } _ { k } ) \} _ { k = 1 } ^ { n } } }$ and lower bound for the number of accepted samples $\overline { { N _ { + } > 0 } } .$   
1: Split $\{ 1 , 2 , \ldots , n \}$ into two disjoint sets, $S _ { 1 }$ (size $m _ { 1 } )$ and $S _ { 2 }$ (size $m _ { 2 } \geq N _ { + } )$   
2: for $j \in \mathcal { Y }$ do   
3: $T _ { \mathrm { t e m p } } \gets 0$   
4: Learn a binary score function $s ^ { ( j ) } : \mathcal { X }  \mathbb { R }$ to distinguish class ${ \widetilde { Y } } = j$ from all others, using the   
samples indexed by $S _ { 1 }$   
5: $( \tau _ { r } ) _ { r = 1 } ^ { m _ { 2 } }  \mathtt { S o r t } ( \{ s ^ { ( j ) } ( x _ { k } ) \} _ { k \in S _ { 2 } } )$ sort the scores in descending order   
6: for $r = N _ { + } , \ldots , m _ { 2 }$ do   
7: Define $h _ { \tau _ { r } } ^ { ( j ) } ( x ) = \tt A i f \thinspace { } s ^ { ( j ) } ( x ) \geq \tau _ { r }$ , and $h _ { \tau _ { r } } ^ { ( j ) } ( x ) = \mathtt { R }$ otherwise, for every $x \in \mathcal X$   
8: i $\mathrm { ~ f ~ } \widehat { T } _ { j , j } \big ( h _ { \tau _ { r } } ^ { ( j ) } ; S _ { 2 } \big ) \geq T _ { \mathrm { t e m p } }$ then   
9: $\widehat { \tau } \gets \tau _ { r }$   
10: $T _ { \mathrm { t e m p } } \gets \widehat { T } _ { j , j } ( h _ { \widehat { \tau } } ^ { ( j ) } ; S _ { 2 } )$   
11: $\widehat { T } _ { i , j } \gets \widehat { T } _ { i , j } ( h _ { \widehat { \tau } } ^ { ( j ) } ; S _ { 2 } )$ for $i \in \mathcal { Y }$   
12: return $\widehat { \mathbf { T } }$

In the next theorem, we show that the error of Algorithm 1 depends on the ranking excess risk of the learned score function in Step 4.

Theorem 2. Let $\widehat { \bf T }$ be the output of Algorithm 1 and fix $j \in \mathcal Y$ . Let $\gamma = \mathbb { P } ( h _ { \widehat { \tau } } ^ { ( j ) } ( X ) = \mathtt { A } )$ be the bminimum coverage, that is $\gamma = \mathbb { P } ( s ^ { ( j ) } ( X ) \geq \widehat { \tau } )$

If $\widehat { \tau } _ { \eta }$ satisfies $\gamma = \mathbb { P } ( \eta _ { j } ^ { \mathrm { { n o i s y } } } ( X ) \ge \widehat { \tau } _ { \eta } ) = \mathbb { P } ( s ^ { ( j ) } ( X ) \ge \widehat { \tau } )$ , and we take

$$
\mathcal { E } _ { \mathrm { r a n k } } = \mathbb { P } ( \widetilde { Y } \neq j , s ^ { ( j ) } ( X ) \ge \widehat { \tau } , \eta _ { j } ^ { \mathrm { n o i s y } } ( X ) < \widehat { \tau } _ { \eta } ) - \mathbb { P } ( \widetilde { Y } \neq j , s ^ { ( j ) } ( X ) < \widehat { \tau } , \eta _ { j } ^ { \mathrm { n o i s y } } ( X ) \ge \widehat { \tau } _ { \eta } )
$$

then

$$
\operatorname* { m a x } _ { i \in \mathbb { V } } \vert T _ { i , j } - \widehat { T } _ { i , j } \vert \leq C _ { T } \left( R _ { \gamma } ^ { ( j ) } + \frac { \mathcal { E } _ { \mathrm { r a n k } } } { \gamma } \right) + \sqrt { \frac { 1 0 0 0 ( \log ( 8 m _ { 2 } ) + \log ( 4 \vert \mathcal { Y } \vert / \delta ) ) } { \mathcal { H } _ { m _ { 2 } } ( h _ { \widehat { \tau } } ^ { ( j ) } ) } }\tag{5}
$$

holds with probability at least $1 - \delta$ where $\# _ { m _ { 2 } } ( h _ { \widehat { \tau } } ^ { ( j ) } )$ is the number of accepted samples in the second split.

Proof. See Appendix A.3

The theorem above shows that the error of Algorithm 1 is small when the learned score function is close to being order preserving with respect to the noisy class-posterior, $\mathrm { i } . \mathrm { e } . , \ \mathcal { E } _ { \mathrm { r a n k } }$ is close to 0. In particular, if the score is order preserving (i.e., for every pair of instances $x , x ^ { \prime } \in \mathcal { X }$ $s ^ { ( j ) } ( x ) < \overline { { s ^ { ( j ) } } } ( x ^ { \prime } ) \implies \eta _ { j } ^ { \mathrm { \scriptsize { n o i s y } } } ( x ) \leq \eta _ { j } ^ { \mathrm { \scriptsize { n o i s \bar { y } } } } ( x ^ { \prime } ) )$ we have that $\mathcal { E } _ { \mathrm { r a n k } } = 0$ . This follows since in that case $\{ x \in \mathfrak { X } : s ^ { ( j ) } ( x ) < \widehat { \tau } \} \subseteq \{ x \in \mathfrak { X } : \eta _ { j } ^ { \mathrm { n o i s y } } ( x ) < \widehat { \tau } _ { \eta } \}$ , and both sets have the same probability. bNotice also that the existence of $\widehat { \tau } _ { \eta }$ is guaranteed if $\eta _ { j } ^ { \mathrm { n o i s y } } ( X )$ is continuously distributed.

The requirement of a low ranking excess risk ${ \mathcal E } _ { \mathrm { r a n k } }$ is milder than assuming an accurate classposterior estimation since commonly used classifiers (e.g., SVMs or boosted trees) are more effective at producing rankings than precise posterior probabilities [38].

The ranking excess risk ${ \mathcal E } _ { \mathrm { r a n k } }$ decreases as m<sub>1</sub> grows when the score function is learned by empirical risk minimization over a hypothesis class F. Specifically, results for the bipartite ranking problem in [39] can be used to show that the ranking excess risk satisfies

$$
\mathcal { E } _ { \mathrm { r a n k } } \lesssim \mathcal { A } _ { \mathrm { r a n k } } ( \mathcal { F } ) + \sqrt { \frac { \mathsf { V C } ( \mathcal { F } ) \log ( m _ { 1 } ) } { m _ { 1 } } }
$$

where $\mathcal { A } _ { \mathrm { r a n k } } ( \mathcal { F } )$ is the approximation error of the hypothesis class $\mathcal { F }$ for the ranking loss in [39] (difference between the best in class risk and the Bayes risk).

The hyperparameter $N _ { + }$ serves to control the minimum coverage γ, which is of order $N _ { + } / m _ { 2 }$ Specifically, by Hoeffding’s inequality we have that $\gamma \geq N _ { + } / \bar { m _ { 2 } } - O ( 1 / \sqrt { m _ { 2 } } )$ with high probability. In practice, choosing $N _ { + }$ involves a trade-off since it should be large enough to accept a sufficient number of instances (to reduce the variance), while remaining small enough to maintain a low false discovery rate (to minimize the bias). We defer a detailed discussion on the effect of $N _ { + }$ to Appendix B.

In the following we present another algorithm for estimating T that satisfies further refined finitesample guarantees.

## 4.2 Cost-sensitive risk minimization approach

A common approach in selective classification involves penalizing misclassifications and assigning a tunable cost to rejections [33, 40]. This idea can be adapted to our methodology for estimating the transition matrix as follows. First, split the noisy samples into two disjoint sets. Then, for each class $j \in \mathcal { Y }$ , the algorithm estimates the corresponding column of T through the following procedure.

1. Learn a binary classifier $h _ { c } ^ { ( j ) } : \mathcal { X }  \{ \mathtt { A } , \mathtt { R } \}$ where A targets $\widetilde { Y } = j$ and R targets $\widetilde Y \neq j$ , by penalizing the misclassification of the class $\widetilde { Y } = j$ eand assigning a cost $c \in ( 0 , 1 )$ to all instances classified as R. That is, we learn $h _ { c } ^ { ( j ) }$ by minimizing the loss

$$
\ell _ { c } ^ { ( j ) } ( h ( x ) , \widetilde { y } ) = \mathbb { I } \{ \widetilde { y } \neq j , h ( x ) = \mathtt { A } \} + c \mathbb { I } \{ h ( x ) = \mathtt { R } \} .\tag{6}
$$

2. Select the cost c that minimizes the empirical version of $\mathbb { P } ( \widetilde { Y } \neq j \mid h _ { c } ^ { ( j ) } ( X ) = \mathtt { A } )$ (i.e., that maximizes $\widehat { T } _ { j , j } ( h _ { c } ^ { ( j ) } ) )$ e) while satisfying an empirical minimum coverage constraint, using bthe samples in the second split.

3. Estimate the $( i , j )$ -th entry of T as $\widehat { T } _ { i , j } ( h _ { \widehat { c } } ^ { ( j ) } )$ , using the samples in the second split.

This procedure is formalized in Algorithm 2 and theoretically analyzed in Theorem 3, where we prove that the error of Algorithm 2 depends on the (classification) excess risk of the solution to the cost-sensitive binary classification problem in Step 6 of Algorithm 2.

Algorithm 2 Transition matrix estimation via cost-sensitive risk minimization   
Input: $\overline { { \{ ( x _ { k } , \widetilde { y } _ { k } ) \} _ { k = 1 } ^ { n } } }$ , grid spacing ε and lower bound for the number of accepted samples $\overline { { N _ { + } > 0 } }$   
1: Split $\{ 1 , 2 , \ldots , n \}$ into two disjoint sets, $S _ { 1 }$ (size $m _ { 1 } )$ and $S _ { 2 }$ (size $m _ { 2 } \geq N _ { + } )$   
2: for $j \in \mathcal { Y }$ do   
3: $T _ { \mathrm { t e m p } } \gets 0$   
4: for $q = 1 , 2 , \ldots , \lfloor 1 / \varepsilon \rfloor$ do   
5: $c \gets \varepsilon q$   
6: Learn a cost-sensitive binary classifier $h _ { c } ^ { ( j ) }$ with costs given by c as in (6), using the samples   
indexed by $S _ { 1 }$   
7: if $\# _ { m _ { 2 } } ( h _ { c } ^ { ( j ) } ) \geq N _ { + }$ and $\widehat { T } _ { j , j } ( h _ { c } ^ { ( j ) } ; S _ { 2 } ) \geq T _ { \mathrm { t e m p } }$ then   
8: ${ \widehat { c } } \gets c$   
9: $T _ { \mathrm { t e m p } } \gets \widehat { T } _ { j , j } ( h _ { \widehat { c } } ^ { ( j ) } ; S _ { 2 } )$   
10: $\widehat { T } _ { i , j } \gets \widehat { T } _ { i , j } ( h _ { \widehat { c } } ^ { ( j ) } ; S _ { 2 } )$ for $i \in \mathcal { Y }$   
11: return $\widehat { \mathbf { T } }$

Theorem 3. Let $\widehat { \mathbf { T } }$ be the output of Algorithm 2 and fix $j \in \mathcal Y$ . Let $\gamma = \mathbb { P } ( h _ { \widehat { c } } ^ { ( j ) } ( X ) = \mathtt { A } )$ be the bminimum coverage. If $h _ { \widehat { c } } ^ { ( j ) }$ is the classifier corresponding with the output for column j and we take

$$
\mathcal { E } _ { \mathrm { c o s t } } = \mathbb { E } [ \ell _ { \widehat { c } } ^ { ( j ) } ( h _ { \widehat { c } } ^ { ( j ) } ( X ) , \widetilde { Y } ) ] - \operatorname* { i n f } _ { h } \mathbb { E } [ \ell _ { \widehat { c } } ^ { ( j ) } ( h ( X ) , \widetilde { Y } ) ]
$$

then,

$$
\operatorname* { m a x } _ { i \in \mathbb { V } } | T _ { i , j } - \widehat { T } _ { i , j } | \leq C _ { T } \left( R _ { \gamma } ^ { ( j ) } + \frac { \mathcal { E } _ { \mathrm { c o s t } } } { \gamma } \right) + \sqrt { \frac { 2 \log ( | \mathcal { Y } | / ( \varepsilon \delta ) ) } { \# _ { m _ { 2 } } ( h _ { \widehat { c } } ^ { ( j ) } ) } }\tag{7}
$$

holds with probability at least $1 - \delta$ where $\# _ { m _ { 2 } } ( h _ { \widehat { c } } ^ { ( j ) } )$ is the number of accepted samples in the second split.

## Proof. See Appendix A.4.

Theorem 3 shows that the error of Algorithm 2 is small, provided that the binary cost-sensitive classification problem in Step 6 is solved with a small excess risk $\mathcal { E } _ { \mathrm { c o s t } }$ . This requirement is significantly milder than that of a pointwise accurate class-posterior estimate, and the small ranking excess risk required by the thresholding approach in Theorem 2. In particular, achieving a small excess risk corresponds to an accurate classification of instances, whereas the thresholding approach additionally requires correct ranking of instances’ probabilities.

In the next theorem, we provide performance guarantees for Algorithm 2 in terms of the excess risk of a convex surrogate loss and the ψ-transform of the surrogate from [41, Def. 2]. Specifically, let Φ be a classification-calibrated convex surrogate of the 0-1 loss $t \mapsto \mathbb { I } \{ t < 0 \}$ such as the hinge loss $( \Phi ( t ) = \operatorname* { m a x } \{ 0 , 1 - t \} )$ ) or the logistic loss $\mathsf { \bar { ( } } \Phi ( t ) = \log ( 1 + e ^ { - t } ) )$ [41, Def. 1], we define the convex surrogate of $\ell _ { c } ^ { ( j ) }$ as

$$
L _ { c } ^ { ( j ) } ( f ( x ) , \widetilde { y } ) = \mathbb { I } \{ \widetilde { y } \neq j \} \Phi ( - f ( x ) ) + c \Phi ( f ( x ) )
$$

where f is a binary score function.

Theorem 4. Let T be the output of Algorithm 2 assuming that the classifiers learned in Step 6 take the form $h _ { c } ^ { ( j ) } ( x ) = \mathrm { s i g n } ( f _ { c } ^ { ( j ) } ( x ) )$ for $x \in \mathcal { X }$ with $f _ { c } ^ { ( j ) }$ a binary score function. Fix $j \in \mathcal Y$ , let $\gamma = \mathbb { P } ( h _ { \widehat { c } } ^ { ( j ) } ( X ) = \mathtt { A } )$ be the minimum coverage and let

$$
\mathcal { E } _ { \Phi } = \mathbb { E } [ L _ { \widehat { c } } ^ { ( j ) } ( f _ { \widehat { c } } ^ { ( j ) } ( X ) , \widetilde { Y } ) ] - \operatorname* { i n f } _ { f } \mathbb { E } [ L _ { \widehat { c } } ^ { ( j ) } ( f ( X ) , \widetilde { Y } ) ]
$$

be the excess risk of the convex surrogate loss. Then

$$
\operatorname* { m a x } _ { i \in \mathbb { V } } \vert T _ { i , j } - \widehat { T } _ { i , j } \vert \leq C _ { T } \left( R _ { \gamma } ^ { ( j ) } + \frac { 2 \psi _ { \Phi } ^ { - 1 } \left( \mathcal { E } _ { \Phi } / 2 \right) } { \gamma } \right) + \sqrt { \frac { 2 \log ( \vert \mathbb { V } \vert / ( \varepsilon \delta ) ) } { \# _ { m _ { 2 } } ( h _ { \hat { c } } ^ { ( j ) } ) } }\tag{8}
$$

holds with probability at least $1 - \delta$

Proof. See Appendix A.5.

Theorem 4 guarantees that the error of Algorithm 2 remains small as long as the learning technique used in Step 6 successfully minimizes a classification-calibrated convex surrogate loss. Moreover, reducing the excess risk of the convex surrogate loss to zero makes the suboptimality gap $2 \psi _ { \Phi } ^ { - 1 } ( \mathcal { E } _ { \Phi } / 2 ) \big / \gamma$ converge to zero since the ψ<sub>Φ</sub>-transform satisfies $\psi _ { \Phi } ^ { - 1 } ( t ) \to 0$ when $t  0$ as shown in [41, Thm. 1].

In practice, the learning in Step 6 is performed by minimizing the empirical risk of the convex surrogate loss $L _ { c } ^ { ( j ) }$ over a hypothesis class F. By standard generalization bounds [42], we have $\begin{array} { r l r } { \mathcal { E } _ { \Phi } } & { \lesssim } & { \mathcal { A } _ { \mathrm { s u r r o g a t e } } ( \mathfrak { F } ) + { O } ( \sqrt { \mathsf { V C } ( \mathfrak { F } ) \log ( m _ { 1 } / \mathsf { V C } ( \mathfrak { F } ) ) / m _ { 1 } } ) } \end{array}$ where $\begin{array} { r } { \mathcal { A } _ { \mathrm { s u r r o g a t e } } ( \mathcal { F } ) = \operatorname* { i n f } _ { f \in \mathcal { F } } \mathbb { E } [ L _ { \hat { c } } ^ { ( j ) } ( f ( X ) , \widetilde { Y } ) ] - \operatorname* { i n f } _ { f } \mathbb { E } [ L _ { \hat { c } } ^ { ( j ) } ( f ( X ) , \widetilde { Y } ) ] } \end{array}$ is the approximation error of the hypothesis class F for the convex surrogate loss $L _ { \widehat { c } } ^ { ( j ) }$

The choice of the convex surrogate Φ specifies the $\psi _ { \Phi } ^ { - 1 }$ -transform, which determines the convergence rate of the term $\psi _ { \Phi } ^ { - 1 } \left( \mathcal { E } _ { \Phi } / 2 \right)$ . For the hinge loss $( \Phi ( t ) = \operatorname* { m a x } \{ 0 , 1 - t \} )$ , ψ<sub>Φ</sub> is simply the identity function [41]. Therefore, the excess risk bound scales linearly, yielding

$$
\operatorname* { m a x } _ { i \in \Psi } | T _ { i , j } - \widehat { T } _ { i , j } | \lesssim R _ { \gamma } ^ { ( j ) } + \mathcal { A } _ { \mathrm { s u r r o g a t e } } ( \mathcal { F } ) + \sqrt { \frac { \mathsf { V C } ( \mathcal { F } ) \log ( m _ { 1 } / \mathsf { V C } ( \mathcal { F } ) ) } { m _ { 1 } } } + \sqrt { \frac { \log ( | \mathcal { Y } | / \varepsilon ) } { m _ { 2 } } } .
$$

For the logistic loss we have $\psi _ { \Phi } ^ { - 1 } ( t ) \lesssim \sqrt { t } \left[ 4 1 \right]$ , which introduces a square-root dependence on the excess risk resulting in a slightly slower convergence rate.

The results above show that the methodology presented can be implemented using flexible and effective techniques for selective classification. In particular, the theoretical results show that the proposed approach can result in significantly improved error rates that quickly decrease with the number of samples. In the following, we provide numerical results that illustrate such improved performance using real-world datasets.

## 4.3 Experimental results

In what follows, we provide experimental results that illustrate the theoretical results presented above. Specifically, we show that the proposed algorithms achieve significantly lower error rates than methods that rely on pointwise estimates of class-posteriors and that the error rates of the proposed algorithms converge at an order comparable to that of a strong oracle. Furthermore, in Section C.2 we provide additional experimental results on running times and dimensionality’s impact. The code implementing the methods presented and reproducing the experiments can be found at https://github.com/MachineLearningBCAM/label-noise-T-estimation-NeurIPS2026.

Figure 1 shows how the error of different algorithms decreases with the number of samples n. We implement Algorithms 1 and 2, and the anchor-based method from [8] with random forests for the Letter dataset (Figure 1(a)) and neural networks for the MNIST dataset (Figure 1(b)). Both Algorithms 1 and 2 are executed with $N _ { + } = n / 2 0 0$ and $m _ { 1 } = m _ { 2 } = n / 2$ . Moreover, we evaluate the algorithms against a strong oracle that learns T via maximum likelihood estimation (MLE). To act as a strong benchmark reference, the oracle has access to feature representations from a clean-data pre-trained network and initializes the MLE optimization directly at the ground-truth T. Therefore, this oracle has a very small error corresponding to the finite-sample estimation error due to finite n. Further details regarding the implementation and additional experimental results are provided in Appendix C.

In Figure 1 we show that the proposed algorithms achieve significantly lower error rates than existing methods that rely on pointwise estimates of the class-posteriors, in accordance with our derived performance bounds. Moreover, the figure shows that the errors of both the proposed algorithms and the oracle exhibit the same decreasing trend as n increases, whereas the error of the anchorbased method plateaus.

![](images/715594bf825e63ebf81bd7002742f1dc11fe71f4fa6ecf80823ae51493f4a0c9.jpg)  
(a) Letter dataset

![](images/110d38c85cce62b1698a60dba87aa344d325c172b82d080d86f128b480f08411.jpg)  
(b) MNIST dataset  
Figure 1: Mean Absolute Error (MAE) of different estimators as the number of training samples n increases. The errors of the presented algorithms decrease with the number of samples in a similar trend to the error of the oracle, whereas the error of the anchor-based method stagnates.

## 5 Conclusion

The paper presents a new methodology for the estimation of the label-noise transition matrix based on one-sided selective classification. In the proposed approach, the transition matrix is estimated by learning selection functions that minimize the false discovery rate subject to a minimum coverage constraint. We provide finite-sample error bounds for the proposed methodology and show that our approach avoids the curse of dimensionality, unlike existing estimators. Moreover, we propose effective algorithms that can be implemented with any method for binary classification, and provide their corresponding performance guarantees. The presented methodology can help to reduce the impact of label noise on supervised learning methods, making model deployment more cost-efficient by allowing the use of data labeled through inexpensive sources.

Limitations: The proposed methodology is broadly effective but does not scale well with the number of classes, as it requires solving different binary classification problems for each class to estimate the transition matrix. Additionally, like other estimators, the approach may struggle under class imbalance, which can affect the precision of the selection functions for minority classes. These drawbacks are of limited relevance in practice because the sub-problems are naturally parallelizable and can be solved using standard off-the-shelfmethods. Furthermore, as noisy labels are inexpensive to acquire, the assumption of a sufficiently large training sample is typically satisfied in real-world applications, ensuring the approach remains both computationally and statistically viable.

## Acknowledgments

Xabier de Juan and Santiago Mazuelas were supported by project PID2022-137063NB-I00 funded by MCIN/AEI/10.13039/501100011033 and the European Union “NextGenerationEU”/PRTR, BCAM Severo Ochoa accreditation CEX2021-001142-S/MICIN/AEI/10.13039/501100011033 funded by the Ministry of Science and Innovation (Spain), and programs BERC-2022–2025 and ELKARTEK funded by the Basque Government. Yilun Zhu and Clayton Scott were supported in part by the National Science Foundation under award 2551503, and by the Department of Defense, Defense Threat Reduction Agency under award HDTRA1-20-2-0002. Xabier de Juan acknowledges a predoctoral grant from the Basque Government.

## References

[1] Hwanjun Song, Minseok Kim, Dongmin Park, Yooju Shin, and Jae-Gil Lee. Learning from noisy labels with deep neural networks: A survey. IEEE Transactions on Neural Networks and Learning Systems, 34(11):8135–8153, 2023.

[2] Tong Xiao, Tian Xia, Yi Yang, Chang Huang, and Xiaogang Wang. Learning from massive noisy labeled data for image classification. In Conference on Computer Vision and Pattern Recognition, 2015.

[3] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A largescale hierarchical image database. In Conference on Computer Vision and Pattern Recognition, 2009.

[4] Nagarajan Natarajan, Inderjit S Dhillon, Pradeep K Ravikumar, and Ambuj Tewari. Learning with noisy labels. In Advances in Neural Information Processing Systems, 2013.

[5] Clayton Scott. A rate of convergence for mixture proportion estimation, with application to learning from noisy labels. In International Conference on Artificial Intelligence and Statistics, 2015.

[6] Gilles Blanchard, Marek Flaska, Gregory Handy, Sara Pozzi, and Clayton Scott. Classification with asymmetric label noise: Consistency and maximal denoising. Electronic Journal of Statistics, 10(2):2780–2824, 2016.

[7] Tongliang Liu and Dacheng Tao. Classification with noisy labels by importance reweighting. IEEE Transactions on Pattern Analysis and Machine Intelligence, pages 447–461, 2016.

[8] Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon, Richard Nock, and Lizhen Qu. Making deep neural networks robust to label noise: A loss correction approach. In Conference on Computer Vision and Pattern Recognition, 2017.

[9] Lucia Filippozzi, Santiago Mazuelas, and Iñigo Urteaga. Minimax risk classifiers for mislabeled data: a study on patient outcome prediction tasks. In Machine Learning for Healthcare Conference, 2024.

[10] Matteo Sesia, Y. X. Rachel Wang, and Xin Tong. Adaptive conformal classification with noisy labels. Journal of the Royal Statistical Society Series B: Statistical Methodology, 87(3):796– 815, 2024.

[11] Haoran Zhang, Olawale Elijah Salaudeen, and Marzyeh Ghassemi. On group sufficiency under label bias. In Advances in Neural Information Processing Systems, 2025.

[12] Byeonghu Na, Yeongmin Kim, HeeSun Bae, Jung Hyun Lee, Jung Se Kwon, Wanmo Kang, and Il-chul Moon. Label-noise robust diffusion models. In International Conference on Learning Representations, 2024.

[13] Xiaobo Xia, Tongliang Liu, Nannan Wang, Bo Han, Chen Gong, Gang Niu, and Masashi Sugiyama. Are anchor points really indispensable in label-noise learning? In Advances in Neural Information Processing Systems, 2019.

[14] Xiaobo Xia, Tongliang Liu, Bo Han, Nannan Wang, Mingming Gong, Haifeng Liu, Gang Niu, Dacheng Tao, and Masashi Sugiyama. Part-dependent label noise: Towards instancedependent label noise. In Advances in Neural Information Processing Systems, 2020.

[15] Yivan Zhang, Gang Niu, and Masashi Sugiyama. Learning noise transition matrix from only noisy labels via total variation regularization. In International Conference on Machine Learning, 2021.

[16] Xuefeng Li, Tongliang Liu, Bo Han, Gang Niu, and Masashi Sugiyama. Provably end-to-end label-noise learning without anchor points. In International Conference on Machine Learning, 2021.

[17] Mingyuan Zhang, Jane Lee, and Shivani Agarwal. Learning from noisy labels with no change to the training process. In International Conference on Machine Learning, 2021.

[18] Yong Lin, Renjie Pi, Weizhong Zhang, Xiaobo Xia, Jiahui Gao, Xiao Zhou, Tongliang Liu, and Bo Han. A holistic view of label noise transition matrix in deep learning and beyond. In International Conference on Learning Representations, 2023.

[19] László Györfi, Michael Kohler, Adam Krzyzak, and Harro Walk.˙ A Distribution-Free Theory ofNonparametric Regression. Springer, New York, United States, 2002.

[20] Jing Lei. Classification with confidence. Biometrika, 101(4):755–769, 2014.

[21] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In International Conference on Machine Learning, 2017.

[22] Gilles Blanchard and Clayton Scott. Decontamination of mutually contaminated models. In International Conference on Artificial Intelligence and Statistics, 2014.

[23] Harish Ramaswamy, Clayton Scott, and Ambuj Tewari. Mixture proportion estimation via kernel embeddings of distributions. In International Conference on Machine Learning, 2016.

[24] Julian Katz-Samuels, Gilles Blanchard, and Clayton Scott. Decontamination of mutual contamination models. Journal ofMachine Learning Research, 20(41):1–57, 2019.

[25] Saurabh Garg, Yifan Wu, Alexander J Smola, Sivaraman Balakrishnan, and Zachary Lipton. Mixture proportion estimation and PU learning: A modern approach. In Advances in Neural Information Processing Systems, 2021.

[26] Yilun Zhu, Aaron Fjeldsted, Darren Holland, George Landon, Azaree Lintereur, and Clayton Scott. Mixture proportion estimation beyond irreducibility. In International Conference on Machine Learning, 2023.

[27] Yilun Zhu, Jianxin Zhang, Aditya Gangrade, and Clayton Scott. Label noise: Ignorance is bliss. In Advances in Neural Information Processing Systems, 2024.

[28] Zhilu Zhang and Mert Sabuncu. Generalized cross entropy loss for training deep neural networks with noisy labels. In Advances in Neural Information Processing Systems, 2018.

[29] Philip M. Long and Rocco A. Servedio. The perils of being unhinged: On the accuracy of classifiers minimizing a noise-robust convex loss. Neural Computation, 34(6):1488–1499, 2022.

[30] Ran El-Yaniv and Yair Wiener. On the foundations of noise-free selective classification. Journal ofMachine Learning Research, 11(53):1605–1641, 2010.

[31] Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. In Advances in Neural Information Processing Systems, 2017.

[32] Peter L. Bartlett and Marten H. Wegkamp. Classification with a reject option using a hinge loss. Journal ofMachine Learning Research, 9(59):1823–1840, 2008.

[33] Corinna Cortes, Giulia DeSalvo, and Mehryar Mohri. Learning with rejection. In International Conference on Algorithmic Learning Theory, 2016.

[34] Blaise Hanczar and Edward R. Dougherty. Classification with reject option in gene expression data. Bioinformatics, 24(17):1889–1895, 2008.

[35] Akshay Swaminathan, Ivan Lopez, William Wang, Ujwal Srivastava, Edward Tran, Aarohi Bhargava-Shah, Janet Y Wu, Alexander L Ren, Kaitlin Caoili, Brandon Bui, Layth Alkhani, Susan Lee, Nathan Mohit, Noel Seo, Nicholas Macedo, Winson Cheng, Charles Liu, Reena Thomas, Jonathan H Chen, and Olivier Gevaert. Selective prediction for extracting unstructured clinical data. Journal of the American Medical Informatics Association, 31(1):188–197, 2024.

[36] Aditya Gangrade, Anil Kag, and Venkatesh Saligrama. Selective classification via one-sided prediction. In International Conference on Artificial Intelligence and Statistics, 2021.

[37] Oluwasanmi Koyejo, Nagarajan Natarajan, Pradeep Ravikumar, and Inderjit S. Dhillon. Consistent binary classification with generalized performance metrics. In Advances in Neural Information Processing Systems, 2014.

[38] Rich Caruana and Alexandru Niculescu-Mizil. An empirical comparison of supervised learning algorithms. In International Conference on Machine Learning, 2006.

[39] Stéphan Clémençon, Gábor Lugosi, and Nicolas Vayatis. Ranking and empirical minimization of U-statistics. Annals ofStatistics, 36(2):844–874, 2008.

[40] Corinna Cortes, Giulia DeSalvo, and Mehryar Mohri. Theory and algorithms for learning with rejection in binary classification. Annals of Mathematics and Artificial Intelligence, 92(2):277– 315, 2024.

[41] Peter L. Bartlett, Michael I. Jordan, and Jon D. Mcauliffe. Convexity, classification, and risk bounds. Journal ofthe American Statistical Association, 101(473):138–156, 2006.

[42] Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar. Foundations ofMachine Learning. The MIT Press, Cambridge, United States, 2018.

[43] Akshay Balsubramani, Sanjoy Dasgupta, Yoav Freund, and Shay Moran. An adaptive nearest neighbor rule for classification. In Advances in Neural Information Processing Systems, 2019.

[44] Dheeru Dua and Casey Graff. UCI machine learning repository. http://archive.ics.uci.edu, 2017.

[45] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings ofthe IEEE, 86(11):2278–2324, 1998.

[46] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, Computer Science Department, University of Toronto, 2009.

[47] Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. DINOv3. arXiv preprint arXiv:2508.10104, 2025.

## A Proofs

The proofs shown below utilize the following lemma that provides general upper bounds for the estimation error.

Lemma 1. Fix $i , j \in \mathcal { Y }$ and let F be a family of measurable subsets of X. Then, for every $A \in { \mathfrak { F } }$

$$
\begin{array} { r l } & { | T _ { i , j } - \widehat { T } _ { i , j } | \leq C _ { T } \left( R _ { \mathcal { F } } ^ { ( j ) } + \underset { A ^ { \prime } \in \mathcal { F } } { \operatorname* { s u p } } \mathbb { P } ( \widetilde { Y } = j \mid X \in A ^ { \prime } ) - \mathbb { P } ( \widetilde { Y } = j \mid X \in A ) \right) } \\ & { \qquad + \left| \mathbb { P } ( \widetilde { Y } = i \mid X \in A ) - \widehat { T } _ { i , j } \right| } \end{array}
$$

where $C _ { T } = 2 / ( T _ { j , j } - \operatorname* { m a x } _ { l \neq j } T _ { j , l } )$ and $R _ { \mathcal { F } } ^ { ( j ) } = \operatorname* { i n f } _ { A ^ { \prime } \in \mathcal { F } } \mathbb { P } ( Y \neq j \mid X \in A ^ { \prime } )$

Proof. By the triangle inequality we get

$$
| T _ { i , j } - \widehat { T } _ { i , j } | \leq | T _ { i , j } - \mathbb { P } ( \widetilde { Y } = i \mid X \in A ) | + | \mathbb { P } ( \widetilde { Y } = i \mid X \in A ) - \widehat { T } _ { i , j } | .
$$

We apply Hölder’s inequality in the first term so that

$$
\begin{array} { r l } & { | T _ { i , j } - { \mathbb P } ( \widetilde { Y } = i \mid X \in A ) | = \left| T _ { i , j } - \displaystyle \sum _ { l = 1 } ^ { | \mathbb M | } T _ { i , l } { \mathbb P } ( Y = l \mid X \in A ) \right| } \\ & { \qquad = | ( T _ { i , 1 } , T _ { i , 2 } , \ldots , T _ { i , | \mathbb M | } ) ^ { \mathrm T } ( e _ { j } - { \mathbb P } ( Y \mid X \in A ) ) | } \\ & { \qquad \le \| e _ { j } - { \mathbb P } ( Y \mid X \in A ) \| _ { 1 } = 2 ( 1 - { \mathbb P } ( Y = j \mid X \in A ) ) } \end{array}
$$

where $e _ { j }$ denotes the j-th vector from the canonical basis and $\mathbb { P } ( Y \mid X \in A )$ is the vector whose t-th component is $\mathbb { P } ( \boldsymbol { \check { Y } } = t \mid \boldsymbol { X } \in \boldsymbol { A } )$ .

Using the definition of $C _ { T }$ , we have

$$
1 - \mathbb { P } ( Y = j \mid X \in A ) \le \frac { C _ { T } } { 2 } ( T _ { j , j } - \mathbb { P } ( \widetilde { Y } = j \mid X \in A ) ) .\tag{9}
$$

Such a bound can be obtained as follows.

$$
\begin{array} { r l } { \displaystyle \mathbb { P } ( \widetilde { Y } = j \mid X \in A ) = \sum _ { l = 1 } ^ { | \mathbb { Y } | } T _ { j , l } \mathbb { P } ( Y = l \mid X \in A ) } \\ { \displaystyle } & { = T _ { j , j } \mathbb { P } ( Y = j \mid X \in A ) + \sum _ { l = 1 , l \neq j } ^ { | \mathbb { Y } | } T _ { j , l } \mathbb { P } ( Y = l \mid X \in A ) . } \end{array}\tag{10}
$$

Therefore,

$$
\begin{array} { r l } & { \mathbb { P } ( Y = j \mid X \in { \cal A } ) = \frac { \mathbb { P } ( \widetilde { Y } = j \mid X \in { \cal A } ) - \sum _ { l = 1 , l \neq j } ^ { \lfloor \mathcal { Y } \rfloor } T _ { j , l } \mathbb { P } ( Y = l \mid X \in { \cal A } ) } { T _ { j , j } } } \\ & { \qquad \ge \frac { \mathbb { P } ( \widetilde { Y } = j \mid X \in { \cal A } ) - \operatorname* { m a x } _ { l \ne j } \{ T _ { j , l } \} \left( 1 - \mathbb { P } ( Y = j \mid X \in { \cal A } ) \right) } { T _ { j , j } } . } \end{array}
$$

Thus,

$$
\left( 1 - \frac { \operatorname* { m a x } _ { l \neq j } T _ { j , l } } { T _ { j , j } } \right) \mathbb { P } ( Y = j \mid X \in A ) \ge \frac { \mathbb { P } ( \widetilde { Y } = j \mid X \in A ) - \operatorname* { m a x } _ { l \neq j } \{ T _ { j , l } \} } { T _ { j , j } }
$$

whence (9) follows since $\operatorname* { m a x } _ { l \neq j } \{ T _ { j , l } \} / T _ { j , j } < 1$

The right hand side in (9) can be written as

$$
\begin{array} { r l } & { T _ { j , j } - \mathbb { P } ( \widetilde { Y } = j \mid X \in { \cal A } ) = T _ { j , j } - \underset { { \cal A } ^ { \prime } \in \mathcal { F } } { \operatorname* { s u p } } \mathbb { P } ( \widetilde { Y } = j \mid X \in { \cal A } ^ { \prime } ) } \\ & { \qquad + \underset { { \cal A } ^ { \prime } \in \mathcal { F } } { \operatorname* { s u p } } \mathbb { P } ( \widetilde { Y } = j \mid X \in { \cal A } ^ { \prime } ) - \mathbb { P } ( \widetilde { Y } = j \mid X \in { \cal A } ) . } \end{array}
$$

Then, the result is obtained because

$$
T _ { j , j } - \operatorname* { s u p } _ { A ^ { \prime } \in \mathcal { F } } \mathbb { P } ( \widetilde { Y } = j \mid X \in A ^ { \prime } ) \le 1 - \operatorname* { s u p } _ { A ^ { \prime } \in \mathcal { F } } \mathbb { P } ( Y = j \mid X \in A ^ { \prime } ) = R _ { \mathcal { F } } ^ { ( j ) }
$$

since

$$
\operatorname* { s u p } _ { A ^ { \prime } \in { \mathcal { F } } } \mathbb { P } ( \widetilde { Y } = j \mid X \in A ^ { \prime } ) \ge T _ { j , j } \operatorname* { s u p } _ { A ^ { \prime } \in { \mathcal { F } } } \mathbb { P } ( Y = j \mid X \in A ^ { \prime } )
$$

as a consequence of (10), and we also have

$$
T _ { j , j } \underset { A ^ { \prime } \in \mathcal { F } } { \operatorname* { s u p } } \mathbb { P } ( Y = j \mid X \in A ^ { \prime } ) \geq T _ { j , j } + \underset { A ^ { \prime } \in \mathcal { F } } { \operatorname* { s u p } } \mathbb { P } ( Y = j \mid X \in A ^ { \prime } ) - 1
$$

since $( T _ { j , j } - 1 ) { \big ( } \operatorname* { s u p } _ { A ^ { \prime } \in { \mathcal { F } } } \mathbb { P } ( Y = j \mid X \in A ^ { \prime } ) - 1 { \big ) } \geq 0 .$

## A.1 Proof of Theorem 1

The result is a particular case of Lemma 1 taking

$$
\mathcal { F } = \{ A _ { h } \subseteq \mathcal { X } : h \colon \mathcal { X } \to \{ \mathtt { A } , \mathtt { R } \} \mathrm { s . t . } \mathbb { P } ( h ( X ) = \mathtt { A } ) \geq \gamma \}
$$

where $A _ { h } \ = \ \{ x \in \mathfrak { X } \ : \ h ( x ) \ = \ \mathbb { A } \}$ is the set of accepted instances of a selection function $h \colon \mathcal { X }  \{ \mathtt { A } , \mathtt { R } \}$ , and $A = \{ x \in \mathfrak { X } : h ^ { ( j ) } ( x ) = \mathtt { A } \}$ . Therefore, Lemma 1 implies

$$
\begin{array} { r } { \vert T _ { i , j } - \widehat T _ { i , j } \vert \leq C _ { T } \left( R _ { \gamma } ^ { ( j ) } + \varepsilon _ { \mathrm { o p t } } ( h ^ { ( j ) } ) \right) + \vert \mathbb { P } ( \widetilde Y = i \mid X \in A ) - \widehat T _ { i , j } \vert } \end{array}
$$

since $R _ { \gamma } ^ { ( j ) } = R _ { \mathcal { F } } ^ { ( j ) } \mathrm { ~ a n d ~ } \varepsilon _ { \mathrm { o p t } } ( h ^ { ( j ) } ) = \operatorname* { s u p } _ { A _ { h } \in \mathcal { F } } \mathbb { P } ( \widetilde { Y } = j \mid X \in A _ { h } ) - \mathbb { P } ( \widetilde { Y } = j \mid X \in A ) .$

Finally, the desired result follows since for each $i \in \mathcal Y$

$$
\vert \mathbb { P } ( \widetilde { Y } = i \mid h ^ { ( j ) } ( X ) = \mathbb { A } ) - \widehat { T } _ { i , j } ( h ^ { ( j ) } ) \vert \leq \sqrt { \frac { 2 \log ( 1 / \delta ) } { \# m \left( h ^ { ( j ) } \right) } }\tag{11}
$$

with probability at least $1 ~ - ~ \delta$ (over the draw of the samples in S). To obtain (11), note that conditioned on the event that the number of accepted samples is $ { k } \ = \ \# _ { m } ( h ^ { ( j ) } )$ , the random variable $k \widehat { T } _ { i , j } ( h ^ { ( j ) } )$ follows a binomial distribution with parameters k and probability $\mathbb { P } ( \widetilde { Y } = i \mid h ^ { ( j ) } ( X ) = \mathtt { A } )$ . Then, the bound in (11) follows by the Chernoff bound conditionally eon that event, and using the law of total probability varying the values of k.

Finally, taking a union bound over the Y classes we obtain

$$
\operatorname* { m a x } _ { i \in \mathcal { Y } } | \mathbb { P } ( \widetilde { Y } = i \mid h ^ { ( j ) } ( X ) = \mathsf { A } ) - \widehat { T } _ { i , j } ( h ^ { ( j ) } ) | \leq \sqrt { \frac { 2 \log ( | \mathcal { Y } | / \delta ) } { \# _ { m } ( h ^ { ( j ) } ) } }
$$

that holds with probability at least $1 - \delta .$

## A.2 Proof of Proposition 1

The result is proven using Lemma 1 with ${ \mathcal { F } } = \{ \{ x \} : x \in { \mathcal { X } } \}$ and $A = \{ x ^ { ( j ) } \}$ . Therefore,

$$
\vert T _ { i , j } - \widehat { T } _ { i , j } ^ { \mathrm { a n c h o r } } ( x ^ { ( j ) } ) \vert \leq C _ { T } \left( R _ { 0 } ^ { ( j ) } + \varepsilon _ { \mathrm { o p t } } ^ { \mathrm { a n c h o r } } ( x ^ { ( j ) } ) \right) + \vert \eta _ { i } ^ { \mathrm { n o i s y } } ( x ^ { ( j ) } ) - \widehat { \eta } _ { i } ^ { \mathrm { n o i s y } } ( x ^ { ( j ) } ) \vert
$$

$$
\mathrm { s i n c e } R _ { 0 } ^ { ( j ) } = R _ { \mathcal { F } } ^ { ( j ) } \mathrm { ~ a n d ~ } \varepsilon _ { \mathrm { o p t } } ^ { \mathrm { a n c h o r } } ( x ^ { ( j ) } ) = \operatorname* { s u p } _ { A ^ { \prime } \in \mathcal { F } } \mathbb { P } ( \widetilde { Y } = j \mid X \in A ^ { \prime } ) - \mathbb { P } ( \widetilde { Y } = j \mid X \in A ) .
$$

## A.3 Proof of Theorem 2

The result is proven using Lemma 1 with

$$
\mathcal { F } = \{ A _ { h } \subseteq \mathcal { X } : h \colon \mathcal { X } \to \{ \mathtt { A } , \mathtt { R } \} \mathrm { s . t . } \mathbb { P } ( h ( X ) = \mathtt { A } ) \geq \gamma \}
$$

where $A _ { h } \ = \ \{ x \ \in \ \mathfrak { X } \ : \ h ( x ) \ = \ \mathtt { A } \}$ is the set of accepted instances of a selection function $h \colon \mathcal { X }  \{ { \tt A } , { \tt R } \}$ , and $A \ = \ \{ x \in \mathfrak { X } \ : \ h _ { \widehat { \tau } } ^ { ( j ) } ( x ) \ = \ \mathbb { A } \}$ . Moreover, we have $A \ \in \ { \mathcal { F } }$ since $\gamma = \mathbb { P } ( h _ { \widehat { \tau } } ^ { ( j ) } ( X ) = \mathtt { A } )$ by hypothesis. Therefore, Lemma 1 implies

$$
\begin{array} { r } { | T _ { i , j } - \widehat { T } _ { i , j } | \leq C _ { T } \left( R _ { \gamma } ^ { ( j ) } + \varepsilon _ { \mathrm { o p t } } ( h _ { \widehat { \tau } } ^ { ( j ) } ) \right) + | { \mathbb { P } } ( \widetilde { Y } = i \mid X \in A ) - \widehat { T } _ { i , j } | } \end{array}
$$

since $R _ { \gamma } ^ { ( j ) } = R _ { \mathcal { F } } ^ { ( j ) }$ and $\varepsilon _ { \mathrm { o p t } } ( h _ { \widehat { \tau } } ^ { ( j ) } ) = \operatorname* { s u p } _ { A , \ \varepsilon \ \mp } \mathbb { P } ( \widetilde { Y } = j \ | \ X \in A _ { h } ) - \mathbb { P } ( \widetilde { Y } = j \ | \ X \in A )$ . Moreover, since $\widehat { T } _ { i , j }$ is the empirical version of $\mathbb { P } ( \widetilde { Y } = i \mid h _ { \widehat { \tau } } ^ { ( j ) } ( X ) = \mathtt { A } )$ , the uniform convergence of empirical b econditional measures (Theorem 8 in [43]) implies

$$
\bigl | \mathbb { P } ( \widetilde { Y } = i \mid h _ { \widehat { \tau } } ^ { ( j ) } ( X ) = \mathtt { A } ) - \widehat { T } _ { i , j } \bigr | \leq \sqrt { \frac { 1 0 0 0 ( \log ( 8 m _ { 2 } ) + \log ( 4 / \delta ) ) } { \# m _ { 2 } ( h _ { \widehat { \tau } } ^ { ( j ) } ) } }
$$

with probability at least $1 - \delta$ (over the draw of the samples indexed by $S _ { 2 } )$ . Taking a union bound over the Y classes we obtain

$$
\operatorname* { m a x } _ { i \in \mathbb { V } } | \mathbb { P } ( \widetilde { Y } = i \mid h ^ { ( j ) } ( X ) = \mathbb { A } ) - \widehat { T } _ { i , j } ( h ^ { ( j ) } ) | \leq \sqrt { \frac { 1 0 0 0 ( \log ( 8 m _ { 2 } ) + \log ( 4 | \mathbb { V } | / \delta ) ) } { \# m _ { 2 } ( h _ { \widehat { \tau } } ^ { ( j ) } ) } }
$$

with probability at least $1 - \delta$ (over the draw of the samples indexed by $S _ { 2 } )$ .

Then, the result is obtained by bounding the suboptimality gap

$$
\varepsilon _ { \mathrm { o p t } } ( h _ { \widehat { \tau } } ^ { ( j ) } ) = { \mathbb { P } } ( \widetilde { Y } \neq j \mid h _ { \widehat { \tau } } ^ { ( j ) } ( X ) = { \tt A } ) - \operatorname* { m i n } _ { h } \{ { \mathbb { P } } ( \widetilde { Y } \neq j \mid h ( X ) = { \tt A } ) \mathrm { s . t . } { \mathbb { P } } ( h ( X ) = { \tt A } ) \geq \gamma \} .
$$

Note that $\widetilde { R } _ { \gamma } ^ { ( j ) }$ is achieved by thresholding the noisy class-posterior at $\widehat { \tau } _ { \eta }$ . Therefore,

$$
\begin{array} { r l } & { \varepsilon _ { \mathrm { o p t } } ( h _ { \widehat { \tau } } ^ { ( j ) } ) = { \mathbb { P } } ( \widetilde { Y } \neq j \mid s ^ { ( j ) } ( X ) \geq \widehat { \tau } ) - { \mathbb { P } } ( \widetilde { Y } \neq j \mid \eta _ { j } ^ { \mathrm { n o i s y } } ( X ) \geq \widehat { \tau } _ { \eta } ) } \\ & { \qquad = \displaystyle \frac { 1 } { \gamma } ( { \mathbb { P } } ( \widetilde { Y } \neq j , s ^ { ( j ) } ( X ) \geq \widehat { \tau } ) - { \mathbb { P } } ( \widetilde { Y } \neq j , \eta _ { j } ^ { \mathrm { n o i s y } } ( X ) \geq \widehat { \tau } _ { \eta } ) ) . } \end{array}
$$

where the last equality holds since $\mathbb { P } ( h _ { \widehat { \tau } } ^ { ( j ) } ( X ) = \mathtt { A } ) = \gamma = \mathbb { P } ( \eta _ { j } ^ { \mathrm { n o i s y } } ( X ) \ge \widehat { \tau } _ { \eta } )$ . Finally, it is easy to check that

$$
\begin{array} { r l } & { \mathbb { P } ( \widetilde { Y } \neq j , s ^ { ( j ) } ( X ) \geq \widehat { \tau } ) - \mathbb { P } ( \widetilde { Y } \neq j , \eta _ { j } ^ { \mathrm { { \scriptscriptstyle ~ n o i s y } } } ( X ) \geq \widehat { \tau } _ { \eta } ) = \mathbb { P } ( \widetilde { Y } \neq j , s ^ { ( j ) } ( X ) \geq \widehat { \tau } , \eta _ { j } ^ { \mathrm { { \scriptscriptstyle ~ n o i s y } } } ( X ) < \widehat { \tau } _ { \eta } ) } \\ & { \qquad - \mathbb { P } ( \widetilde { Y } \neq j , s ^ { ( j ) } ( X ) < \widehat { \tau } , \eta _ { j } ^ { \mathrm { { \scriptscriptstyle ~ n o i s y } } } ( X ) \geq \widehat { \tau } _ { \eta } ) } \\ & { \qquad = \mathcal { E } _ { \mathrm { r a n k } } . } \end{array}
$$

## A.4 Proof of Theorem 3

The result is proven using Lemma 1. Specifically, let

$$
\mathcal { F } = \{ A _ { h } \subseteq \mathcal { X } : h \colon \mathcal { X } \to \{ \mathtt { A } , \mathtt { R } \} \mathrm { s . t . } \mathbb { P } ( h ( X ) = \mathtt { A } ) \geq \gamma \}
$$

where $A _ { h } \ = \ \{ x \in \mathfrak { X } \ : \ h ( x ) \ = \ \mathbb { A } \}$ is the set of accepted instances of a selection function $h \colon \mathcal { X }  \{ \mathtt { A } , \mathtt { R } \}$ , and let $A \ = \ \{ x \ \in \ \mathfrak { X } \ : \ h _ { \widehat { c } } ^ { ( j ) } ( x ) \ = \ \mathtt { A } \}$ . Moreover, we have $A \ \in \ { \mathcal { F } }$ since $\gamma = \mathbb { P } ( h _ { \widehat { c } } ^ { ( j ) } ( X ) = \mathtt { A } )$ by hypothesis. Therefore, Lemma 1 implies

$$
\bigl | T _ { i , j } - \widehat { T } _ { i , j } \bigr | \leq C _ { T } \left( R _ { \gamma } ^ { ( j ) } + \varepsilon _ { \mathrm { o p t } } \bigl ( h _ { \widehat { c } } ^ { ( j ) } \bigr ) \right) + \bigl | \mathbb { P } \bigl ( \widetilde { Y } = i \mid X \in A \bigr ) - \widehat { T } _ { i , j } \bigr |
$$

$$
R _ { \gamma } ^ { ( j ) } = R _ { \mathcal { F } } ^ { ( j ) } \mathrm { ~ a n d ~ } \varepsilon _ { \mathrm { o p t } } ( h _ { \widehat { c } } ^ { ( j ) } ) = \operatorname* { s u p } _ { A _ { h } \in \mathcal { F } } \mathbb { P } ( \widetilde { Y } = j \mid X \in A _ { h } ) - \mathbb { P } ( \widetilde { Y } = j \mid X \in A ) .
$$

Moreover, by applying the Chernoff bound conditionally, for every c in the grid

$$
\operatorname* { m a x } _ { i \in \mathbb { 3 } } \left. \mathbb { P } ( \widetilde { Y } = i \mid h _ { c } ^ { ( j ) } ( X ) = \mathbb { A } ) - \widehat { T } _ { i , j } ( h _ { c } ^ { ( j ) } ; S _ { 2 } ) \right. \leq \sqrt { \frac { 2 \log ( \left. \mathbb { \mathcal { Y } } \right. / \delta ) } { \# _ { m _ { 2 } } ( h _ { c } ^ { ( j ) } ) } }\tag{12}
$$

holds with probability at least $1 - \delta$ . To obtain (12), note that conditioned on the event that the number of accepted samples is $k = \# m _ { 2 } ( h _ { c } ^ { ( j ) } )$ , the random variable $k \widehat { T } _ { i , j } ( h _ { c } ^ { ( j ) } )$ follows a binomial distribution with parameters k and probability $\mathbb { P } ( \widetilde { Y } = i \ | \ h _ { c } ^ { ( j ) } ( X ) = \mathtt { A } )$ . Then, the bound in (12) efollows by the Chernoff bound conditionally on that event, and using the law of total probability varying the values of k.

Therefore, taking a union bound over the grid of size $\lfloor 1 / \varepsilon \rfloor$ , we obtain for $c = { \widehat { c } }$

$$
\operatorname* { m a x } _ { i \in \mathbb { V } } \vert \mathbb { P } ( \widetilde { Y } = i \mid h _ { \widehat { c } } ^ { ( j ) } ( X ) = \mathbb { A } ) - \widehat { T } _ { i , j } \vert \leq \sqrt { \frac { 2 \log ( \vert \mathbb { Y } \vert / ( \delta \varepsilon ) ) } { \# _ { m _ { 2 } } ( h _ { \widehat { c } } ^ { ( j ) } ) } }
$$

with probability at least $1 - \delta$ (over the draw of the samples indexed by $S _ { 2 } )$ .

Then, the result is obtained by bounding the suboptimality gap

$$
\varepsilon _ { \mathrm { o p t } } ( h _ { \widehat { c } } ^ { ( j ) } ) = { \mathbb { P } } ( \widetilde { Y } \neq j \mid h _ { \widehat { c } } ^ { ( j ) } ( X ) = \mathtt { A } ) - \operatorname* { m i n } _ { h } \{ { \mathbb { P } } ( \widetilde { Y } \neq j \mid h ( X ) = \mathtt { A } ) \} \mathrm { s . t . } { \mathbb { P } } ( h ( X ) = \mathtt { A } ) \geq \gamma \} .
$$

Let $h _ { * } ^ { ( j ) }$ be a solution of the one-sided selective classification problem in (1) such that $\mathbb { P } ( h _ { * } ^ { ( j ) } ( X ) = \mathtt { A } ) = \gamma$ . Then,

$$
\begin{array} { r l } & { \varepsilon _ { \mathrm { o p t } } ( h _ { \hat { c } } ^ { ( j ) } ) = \mathbb { P } ( \widetilde { Y } \neq j \mid h _ { \hat { c } } ^ { ( j ) } ( X ) = \mathbb { A } ) - \mathbb { P } ( \widetilde { Y } \neq j \mid h _ { * } ^ { ( j ) } ( X ) = \mathbb { A } ) } \\ & { \quad \quad \quad \quad = \frac { 1 } { \gamma } ( \mathbb { P } ( \widetilde { Y } \neq j , h _ { \hat { c } } ^ { ( j ) } ( X ) = \mathbb { A } ) - \mathbb { P } ( \widetilde { Y } \neq j , h _ { * } ^ { ( j ) } ( X ) = \mathbb { A } ) ) . } \end{array}
$$

where the last equality holds since $\mathbb { P } ( h _ { \hat { c } } ^ { ( j ) } ( X ) = \mathtt { A } ) = \gamma = \mathbb { P } ( h _ { * } ^ { ( j ) } ( X ) = \mathtt { A } )$ . Finally, a simple calculation yields

$$
\begin{array} { r l } & { \mathbb { P } ( \widetilde { Y } \neq j , h _ { \epsilon } ^ { ( j ) } ( X ) = \mathbb { A } ) - \mathbb { P } ( \widetilde { Y } \neq j , h _ { \epsilon } ^ { ( j ) } ( X ) = \mathbb { A } ) } \\ & { \qquad = \mathbb { P } ( \widetilde { Y } \neq j , h _ { \epsilon } ^ { ( j ) } ( X ) = \mathbb { A } ) - \mathbb { P } ( \widetilde { Y } \neq j , h _ { \epsilon } ^ { ( j ) } ( X ) = \mathbb { A } ) } \\ & { \qquad + \widetilde { c } \mathbb { P } ( \mathbb { P } _ { \epsilon } ^ { ( j ) } ( X ) = \mathbb { R } ) - \mathbb { P } ( h _ { \epsilon } ^ { ( j ) } ( X ) = \mathbb { R } ) ) } \\ & { \qquad - \widetilde { c } ( \mathbb { P } ( h _ { \epsilon } ^ { ( j ) } ( X ) = \mathbb { R } ) - \mathbb { P } ( h _ { \epsilon } ^ { ( j ) } ( X ) = \mathbb { R } ) ) } \\ & { \qquad = \mathbb { E } [ \ell _ { \epsilon } ^ { ( j ) } ( h _ { \epsilon } ^ { ( j ) } ( X ) , \widetilde { Y } ) ] - \mathbb { E } [ \ell _ { \epsilon } ^ { ( j ) } ( h _ { \epsilon } ^ { ( j ) } ( X ) , \widetilde { Y } ) ] } \\ & { \qquad + \widetilde { c } ( \mathbb { P } ( h _ { \epsilon } ^ { ( j ) } ( X ) = \mathbb { A } ) - \mathbb { P } ( h _ { \epsilon } ^ { ( j ) } ( X ) = \mathbb { A } ) ) } \\ & { \qquad \leq \mathbb { E } [ \ell _ { \epsilon } ^ { ( j ) } ( h _ { \epsilon } ^ { ( j ) } ( X ) , \widetilde { Y } ) ] - \operatorname* { i n f } \mathbb { E } [ \ell _ { \epsilon } ^ { ( j ) } ( h ( X ) , \widetilde { Y } ) ] } \\ & { \qquad = \mathcal { E } _ { \mathrm { e x t } } . } \end{array}
$$

where in the last inequality we have used that $\mathbb { P } ( h _ { \hat { c } } ^ { ( j ) } ( X ) = \mathbb { A } ) = \gamma = \mathbb { P } ( h _ { * } ^ { ( j ) } ( X ) = \mathbb { A } )$ and inf $\mathtt { h } \mathbb { E } [ \ell _ { \widehat { c } } ^ { ( j ) } ( h ( X ) , \widetilde { Y } ) ] \le \mathbb { E } [ \ell _ { \widehat { c } } ^ { ( j ) } ( h _ { * } ^ { ( j ) } ( X ) , \widetilde { Y } ) ]$ 口

## A.5 Proof of Theorem 4

The result is a consequence of Theorem 3. Then, the result is obtained by establishing the bound

$$
\begin{array} { r l r } {  { 2 \psi _ { \Phi } ( \frac { \mathbb { E } [ \ell _ { \widehat { c } } ^ { ( j ) } ( h _ { \widehat { c } } ^ { ( j ) } ( X ) , \widetilde { Y } ) ] - \operatorname* { i n f } _ { h } \mathbb { E } [ \ell _ { \widehat { c } } ^ { ( j ) } ( h ( X ) , \widetilde { Y } ) ] } { 2 } ) \leq \mathbb { E } [ L _ { \widehat { c } } ^ { ( j ) } ( f _ { \widehat { c } } ^ { ( j ) } ( X ) , \widetilde { Y } ) ] } } \\ & { } & { - \operatorname* { i n f } _ { f } \mathbb { E } [ L _ { \widehat { c } } ^ { ( j ) } ( f ( X ) , \widetilde { Y } ) ] } \end{array}\tag{13}
$$

which we prove in what follows.

Let $x \in { \mathcal { X } }$ be arbitrary and for notational convenience let $\eta ( x ) = \mathbb { P } ( \widetilde { Y } = j \mid X = x )$ . We define the conditional $L _ { \widehat { c } } ^ { ( j ) }$ -risk as

$$
C _ { \Phi } ( x ; \alpha ) : = \mathbb { E } [ L _ { \hat { c } } ^ { ( j ) } ( \alpha , \widetilde { Y } ) \ | \ X = x ] , \quad \alpha \in \mathbb { R } .
$$

A simple calculation produces

$$
\begin{array} { c } { { C _ { \Phi } ( x ; \alpha ) = ( 1 - \eta ( x ) ) \Phi ( - \alpha ) + \widehat { c } \Phi ( \alpha ) } } \\ { { = S ( x ) \left( ( 1 - \eta ^ { \prime } ( x ) ) \Phi ( - \alpha ) + \eta ^ { \prime } ( x ) \Phi ( \alpha ) \right) } } \end{array}\tag{14}
$$

where $S ( x ) = 1 - \eta ( x ) + \widehat { c }$ and $\eta ^ { \prime } ( x ) = \widehat { c } / S ( x ) \in [ 0 , 1 ]$ . Now define the conditional $\ell _ { \widehat { c } } ^ { ( j ) }$ -risk as

$$
C ( x ; a ) : = \mathbb { E } [ \ell _ { \widehat { c } } ^ { ( j ) } ( a , \widetilde { Y } ) \mid X = x ] , \quad a \in \{ - 1 , + 1 \}
$$

where for notational convenience $\ell _ { \widehat { c } } ^ { ( j ) } ( a , \widetilde { Y } )$ is well defined for $a \in \{ - 1 , 1 \}$ (we identify A with +1 eand R with 1). By [41, Sec. 2.2] we have

$$
\operatorname* { i n f } _ { f } \mathbb { E } [ L _ { \widehat { c } } ^ { ( j ) } ( f ( X ) , \widetilde { Y } ) ] = \mathbb { E } _ { X } \left[ \operatorname* { i n f } _ { \alpha \in \mathbb { R } } C _ { \Phi } ( X ; \alpha ) \right] ,\tag{15}
$$

$$
\operatorname* { i n f } _ { h } \mathbb { E } [ \ell _ { \widehat { c } } ^ { ( j ) } ( h ( X ) , \widetilde { Y } ) ] = \mathbb { E } _ { X } \left[ \operatorname* { i n f } _ { a \in \{ - 1 , + 1 \} } C ( X ; a ) \right] .\tag{16}
$$

Define the excess conditional risks

$$
\begin{array} { r } { \Delta C _ { \Phi } ( x ; \alpha ) : = C _ { \Phi } ( x ; \alpha ) - \underset { \alpha \in \mathbb { R } } { \operatorname* { i n f } } C _ { \Phi } ( x ; \alpha ) , } \\ { \Delta C ( x ; a ) : = C ( x ; a ) - \underset { a \in \{ - 1 , + 1 \} } { \operatorname* { i n f } } C ( x ; a ) . } \end{array}
$$

A simple calculation yields

$$
\Delta C ( x ; a ) = | \eta ( x ) - ( 1 - \widehat { c } ) | \mathbb { I } \{ a \neq \mathrm { s i g n } ( \eta ( x ) - ( 1 - \widehat { c } ) ) \} .\tag{17}
$$

Now we proceed with the proof of (13) for an arbitrary score function $f ^ { ( j ) }$ and its corresponding binary classifier $h ^ { ( j ) } , \mathrm { i . e . , } \bar { h ^ { ( j ) } } ( x ) = \mathrm { s i g n } ( f ^ { ( j ) } ( x ) )$ .

Case 1: sign $( 2 \eta ^ { \prime } ( x ) - 1 ) = \mathrm { s i g n } ( f ^ { ( j ) } ( x ) )$ . We have sign $( 2 \eta ^ { \prime } ( x ) - 1 ) = \mathrm { s i g n } ( \eta ( x ) - ( 1 - \widehat { c } ) )$ , and thus by (17) we get

$$
\Delta C ( x ; h ^ { ( j ) } ( x ) ) = | \eta ( x ) - ( 1 - { \widehat c } ) | \mathbb { I } \{ \operatorname { s i g n } ( f ^ { ( j ) } ( x ) ) \neq \operatorname { s i g n } ( \eta ( x ) - ( 1 - { \widehat c } ) ) \} = 0 .
$$

Since the ψ -transform satisfies $\psi _ { \Phi } ( 0 ) = 0$ bas shown in [41, Lemma 2], the bound

$$
\Delta C _ { \Phi } ( x ; f ^ { ( j ) } ( x ) ) \ge 0 = 2 \psi _ { \Phi } \left( \frac { \Delta C ( x ; h ^ { ( j ) } ( x ) ) } { 2 } \right)
$$

trivially holds.

$$
\begin{array} { r l } & { \mathrm { C a s e ~ 2 } \colon \mathrm { s i g n } ( 2 \eta ^ { \prime } ( x ) - 1 ) \neq \mathrm { s i g n } ( f ^ { ( j ) } ( x ) ) . \mathrm { ~ F o r ~ } t \in [ 0 , 1 ] \mathrm { ~ w e ~ d e f i n e } } \\ & { \qquad H ( t ) = \underset { \alpha \in \mathbb { R } } { \operatorname* { i n f } } \left( ( 1 - t ) \Phi ( - \alpha ) + t \Phi ( \alpha ) \right) , } \\ & { \qquad H ^ { - } ( t ) = \underset { \alpha \operatorname { s } . \mathrm { t } . \alpha ( 2 t - 1 ) \leq 0 } { \operatorname* { i n f } } \left( ( 1 - t ) \Phi ( - \alpha ) + t \Phi ( \alpha ) \right) } \end{array}
$$

following [41, Def. 1]. Then, since $\mathrm { s i g n } ( 2 \eta ^ { \prime } ( x ) - 1 ) \neq \mathrm { s i g n } ( f ^ { ( j ) } ( x ) )$ ), we have

$$
\begin{array} { r l r l } { \Delta C _ { \Phi } ( x ; f ^ { ( \Delta ) } ( x ) ) } & { \ge S ( z ) ( H ^ { - } ( \eta ^ { \prime } ( x ) ) - H ^ { ( \eta ^ { \prime } ( x ) ) } ) } & { ( \mathrm { B y ~ ( 1 4 ) } ) } \\ & { \ge S ( z ) \psi _ { \Phi } ( 2 \eta ^ { \prime } ( x ) - 1 ) } & { ( \mathrm { d e f i n i t i o n o f ~ \psi , ~ [ 4 , 1 , D e f , 2 ] } ) } \\ & { = S ( z ) \psi _ { \Phi } ( 1 2 \psi ( z ) - 1 | ) } & { ( \mathrm { d e f i n i t i o n o f ~ \psi , ~ [ 4 , 1 , L e m m a ~ 2 ] } ) } \\ & { = S ( z ) \psi _ { \Phi } \left( \frac { | \nabla \psi ( z ) - ( 1 - \widetilde \psi \delta \delta \delta \big ] | } { S ( z ) } \right) } & { ( | 2 \eta ^ { \prime } ( x ) - 1 | = \frac { | \psi ( z ) - ( 1 - \widetilde \psi \delta \delta \big ] | } { S ( z ) } ) } \\ & { = S ( z ) \psi _ { \Phi } \left( \frac { \Delta C ( z ; h ^ { ( \delta ) } ( x ) ) ( x ) | } { S ( z ) } \right) } & { ( | \eta ( x ) - ( 1 - \widetilde \psi \delta \big ] = \Delta C ( z ; h ^ { ( \delta ) } ( x ) ) ) } \\ & { = 2 \frac { S ( x ) } { 2 } \psi _ { \Phi } \left( \frac { \Delta C ( z ; h ^ { ( \delta ) } ( x ) ) } { S ( z ) } \right) } & { } \\ & { \ge 2 \psi _ { \Phi } \left( \frac { \Delta C ( z ; h ^ { ( \delta ) } ( x ) ) } { S ( z ) } \frac { S ( z ) } { 2 } \right) } & { ( S ( z ) \le 2 \mathrm { ~ a n d ~ } \forall \xi _ { \Phi } \mathrm { ~ i s ~ c o u n e x } ) } \\ & { = 2 \psi _ { \Phi } \left( \frac { \Delta C ( z ; h ^ { ( \delta ) } ( x ) ) } { S ( z ) } \frac { S ( z ) } { 2 } \right) } &  ( S ( z ) \le 2 \mathrm { ~ a n d ~ } \forall \xi _   \end{array}
$$

Therefore, we have shown that for every $x \in { \mathfrak { X } }$

$$
\Delta C _ { \Phi } ( x ; f ^ { ( j ) } ( x ) ) \ge 2 \psi _ { \Phi } \left( \frac { \Delta C ( x ; h ^ { ( j ) } ( x ) ) } { 2 } \right) .
$$

Finally, since ψ<sub>Φ</sub> is convex, by Jensen’s inequality we obtain

$$
\mathbb { E } _ { X } [ \Delta C _ { \Phi } ( X ; f ^ { ( j ) } ( X ) ) ] \ge \mathbb { E } _ { X } \left[ 2 \psi _ { \Phi } \left( \frac { \Delta C ( X ; h ^ { ( j ) } ( X ) ) } { 2 } \right) \right] \ge 2 \psi _ { \Phi } \left( \frac { \mathbb { E } _ { X } [ \Delta C ( X ; h ^ { ( j ) } ( X ) ) ] } { 2 } \right)
$$

which proves (13) because

$$
\mathbb { E } _ { X } [ \Delta C _ { \Phi } ( X ; f ^ { ( j ) } ( X ) ) ] = \mathbb { E } [ L _ { \widehat { c } } ^ { ( j ) } ( f ^ { ( j ) } ( X ) , \widetilde { Y } ) ] - \operatorname* { i n f } _ { f } \mathbb { E } [ L _ { \widehat { c } } ^ { ( j ) } ( f ( X ) , \widetilde { Y } ) ]
$$

and

$$
\mathbb { E } _ { X } [ \Delta C ( X ; h ^ { ( j ) } ( X ) ) ] = \mathbb { E } [ \ell _ { \widehat { c } } ^ { ( j ) } ( h ^ { ( j ) } ( X ) , \widetilde { Y } ) ] - \operatorname* { i n f } _ { h } \mathbb { E } [ \ell _ { \widehat { c } } ^ { ( j ) } ( h ( X ) , \widetilde { Y } ) ]
$$

hold by (15) and (16), respectively.

## B Analysis of the effect of the number of accepted samples

In this appendix, we provide an analysis on the effect of the hyperparameter $N _ { + }$ for Algorithms 1 and 2. In practice, the optimal choice of the lower bound for the number of accepted samples $N _ { + }$ must balance the need for a sufficiently large number of accepted instances (to reduce the variance of the estimator) with the requirement of maintaining a low false discovery rate (to minimize the bias).

For Algorithm 1, we analyze the impact of $N _ { + }$ by observing the empirical objective function $\widehat { T } _ { j , j } ( h _ { \tau _ { k } } ^ { ( j ) } ; S _ { 2 } )$ as a function of the number of accepted samples $k \in \{ 1 , 2 , \ldots , m _ { 2 } \}$

When plotting $( k , \widehat { T } _ { j , j } ( h _ { \tau _ { k } } ^ { ( j ) } ; S _ { 2 } ) )$ , we typically observe three distinct regimes.

1. When the number of accepted samples is very small, the empirical probability $\widehat { T } _ { j , j } ( h _ { \tau _ { k } } ^ { ( j ) } ; S _ { 2 } )$ fluctuates due to high variance.

2. As the number of accepted samples increases, the variance decreases, and $\widehat { T } _ { j , j } ( h _ { \tau _ { k } } ^ { ( j ) } ; S _ { 2 } )$ bstabilizes at a constant value which is close to the true transition probability $T _ { j , j }$ . Here, the corresponding selection functions accept enough instances to provide a low-variance estimate without accepting samples belonging to other true classes $y \ne j$ , thereby keeping the false discovery rate small.

3. If the number of accepted samples increases beyond the plateau, the corresponding selection functions are forced to accept samples with labels $y \ne j$ . As a consequence, the estimate $\widehat { T } _ { j , j } ( h _ { \tau _ { k } } ^ { ( j ) } ; S _ { 2 } )$ drops sharply from $T _ { j , j }$

Choosing the maximum number of accepted samples inside the plateau guarantees the lowest possible variance for the estimator while ensuring that the bias remains controlled.

We provide experimental results on the CIFAR-10 dataset under different noise scenarios that illustrate the three regimes (see Figure 2), and we provide further implementation details in Appendix C.

In Figure 3 we show that when the number of instances is small, the size of the plateau is smaller and there is more variability, leading to higher estimation errors.

For Algorithm 2, we perform a similar analysis by plotting, for $c ~ \in ~ \{ 0 . 0 5 , 0 . 1 , . . . , 0 . 9 5 \}$ $( c , \widehat { T } _ { j , j } ( h _ { c } ^ { ( j ) } ; S _ { 2 } ) )$ (blue line) and the number of accepted samples is plotted $( c , \# _ { m _ { 2 } } ( h _ { c } ^ { ( j ) } ) )$ (red bline) since the number of accepted samples $\# _ { m _ { 2 } } ( h _ { c } ^ { ( j ) } )$ typically increases monotonically with c. In Figure 4 we plot the corresponding graphs for the Satellite dataset.

![](images/fb87972424a8dd0d00f0100198b2af1672afed13251b2e8f7c4efac56fd15065.jpg)  
(a) CIFAR-10 under uniform noise with $T _ { j , j } = 0 . 5 .$

![](images/bfd6982fbcc69725ac92d66b115b898f017ebf55924bc2e3fdd915a6a50f4d15.jpg)  
(b) CIFAR-10 under flip noise with $T _ { j , j } = 0 . 5 5 .$

Figure 2: The three regimes described above appear on CIFAR-10. Moreover, the value of the plateau is close to $T _ { j , j }$

![](images/108e57a9f97d7f6f3fff01c5df6ba5b76a862f4909c606768120dc6d0bce3bd0.jpg)  
(a) CIFAR-10 using 10,000 samples.

![](images/5ef38d279d9a34b995c7bfe9829808c4a1e6190f3a7e03dc40da983f08ae41e8.jpg)  
(b) CIFAR-10 using 60,000 samples.  
Figure 3: The value of the plateau is close to $T _ { j , j } = 0 . 8$ and the noise is uniform. Moreover, with fewer samples the plateau is shorter and noisier.

![](images/093ae05b0cda32a9041ee7659a14c6b52d97d134a0ee94aa5aa19e6bbf265a1b.jpg)  
Figure 4: The value of the plateau is close to $T _ { j , j } = 0 . 5$ and the noise is uniform.

Again, choosing the highest number of accepted samples inside the plateau ensures a low variance for the estimator while guaranteeing that the bias is small.

The existence and variability of the plateau is determined by the quality of the learned score functions or the binary classifiers in Algorithms 1 and 2, respectively.

## C Implementation details and additional experimental results

In the following we provide further implementation details and describe the datasets used in the experimental results. Then, we complement the results in the main paper by including a table that compares the performance of the algorithms presented in the paper with the anchor-based method.

## C.1 Implementation details

Datasets We evaluate our methodology on four publicly available benchmark datasets from the UCI repository [44] and Kaggle website https://www.kaggle.com. The main characteristics of the datasets used are summarized below.

• Letter: 20,000 samples, 26 classes, 16 features.

• Satellite: 6,435 samples, 6 classes, 36 features.

• MNIST: 70,000 samples, 10 classes, 784 features [45].

• CIFAR-10: 60,000 samples, 10 classes, 3,072 features [46].

Learned binary classifiers For the tabular datasets (Letter and Satellite), we employ Random Forests with 200 maximum splits, and a minimum leaf size of 5. For MNIST, we utilize a multilayer perceptron with two hidden layers of sizes 128 and 64, trained with a regularization parameter of 0.001. For CIFAR-10, we perform logistic regression using features extracted from the final layer of a pre-trained ConvNeXt based vision foundation model [47].

Label noise We generate the noise synthetically, following standard practices in the literature [16, 18]. Specifically, for a noise ratio $p \in ( 0 , 1 )$ , we consider two deterministic noise regimes.

• p-uniform noise: The transition matrix T is defined such that for every column $j \in \mathcal { Y }$

$$
T _ { j , j } = 1 - p , \ T _ { i , j } = \frac { p } { | \mathcal { Y } | - 1 } , \quad \forall i \neq j .
$$

• p-flip noise: $T _ { 1 , 1 } = 1 - p , T _ { 2 , 1 } = p$ and for every $j \in \{ 2 , 3 , \ldots , | \mathcal { Y } | \}$

$$
\begin{array} { r } { T _ { j , j } = 1 - p , T _ { j - 1 , j } = p . } \end{array}
$$

For example, for Figure 1(a) we consider 0.5-uniform noise and for Figure 1(b) 0.2-label flip.

Implementation of the anchor-based method We implement the anchor-based method described in Section 2.2 by utilizing the 97th percentile of the noisy class-posterior rather than the maximum value, as suggested in the original work [8].

Implementation of the proposed algorithms For Algorithms 1 and 2 we split the samples into two sets of equal size $( m _ { 1 } = m _ { 2 } = n / 2 )$

Implementation of the oracle The oracle for the plots in Section 4.3 is implemented as follows. First, a feature representation $\varphi ( x ) \in \mathbb { R } ^ { D }$ is obtained from a network trained on clean labels, with a bias coordinate appended to the representation. Specifically, for the Letter dataset, we use the activations of the final hidden layer of the clean-data MLP (with two hidden layers of sizes 128 and 64), while for MNIST we use the pretrained ConvNeXt representations [47]. Given these features, we parametrize the clean class-posterior as $\widehat { \eta } ( x ; \mu ) = \mathrm { s o f t m a x } ( \varphi ( x ) ^ { \mathrm { T } } \bar { \mu } )$ where $\boldsymbol { \mu } \in \mathbb { R } ^ { D }$ . Then, bwe jointly estimate the classifier parameters and transition matrix by minimizing the negative loglikelihood

$$
\operatorname* { m i n } _ { \mu , \widehat { \mathbf { T } } } - \sum _ { k = 1 } ^ { n } \log \left[ \sum _ { j \in \mathcal { Y } } \widehat { T } _ { \widetilde { y } _ { k } , j } \widehat { \eta } _ { j } ( x _ { k } ; \mu ) \right]
$$

subject to $\widehat { \bf T }$ being column-stochastic. The optimization is initialized with $\mu$ equal to the all-ones vector and $\widehat { \mathbf { T } }$ at the ground-truth transition matrix T.

Number of repetitions The results presented at Section 4.3 (Figure 1) are averaged over 50 independent trials for each sample size n. In Appendix B, we report the mean over 100 repetitions, where the shaded regions denote the standard deviation. Finally, for the results in Appendix C.2, we report the mean and standard deviation using the full datasets over 5 repetitions for Tables 1 and 2, whereas for Table 3 we consider 20 repetitions.

## C.2 Additional experimental results

Transition matrix estimation error Tables 1 and 2 report the Mean Absolute Error (MAE) of the estimated transition matrices under 0.2-uniform noise and 0.45-flip noise, respectively. In Table 3, we introduce noise in a more realistic and challenging way. Specifically, we consider asymmetric label-noise transition matrices with column-wise uniform noise where the noise parameter of each column follows a different value randomly sampled from an $\operatorname { U n i f } ( 0 , 1 / 2 )$ distribution.

We compare our proposed algorithms (Algorithm 1 and Algorithm 2) against the standard anchorbased method across multiple datasets. To evaluate how the choice of the underlying model impacts estimation, we employ both highly expressive classifiers (e.g., Neural Networks and Random Forests) and weaker, linear methods (Logistic Regression). For the high-dimensional CIFAR-10 dataset, we utilize a strong Neural Network architecture.

Table 1: MAE of different estimators under 0.2-uniform noise.
<table><tr><td></td><td rowspan="2">CIFAR-10</td><td colspan="2">MNIST</td><td colspan="2">Letter</td><td colspan="2">Satellite</td></tr><tr><td></td><td>NN</td><td>Logistic reg.</td><td>RF</td><td>Logistic reg.</td><td>RF</td><td>Logistic reg.</td></tr><tr><td>Anchor-based</td><td> $. 0 0 9 \pm . 0 0 1$ </td><td> $. 0 1 0 \pm . 0 0 1$ </td><td> $. 0 1 6 \pm . 0 0 1$ </td><td> $. 0 4 4 \pm . 0 0 0$ </td><td> $. 0 4 7 \pm . 0 0 0$ </td><td> $. 0 2 2 \pm . 0 0 1$ </td><td> $\mathbf { . 0 6 7 \pm { . 0 2 3 } }$ </td></tr><tr><td>Algorithm 1</td><td> $. 0 0 8 \pm . 0 0 1$ </td><td> $. 0 0 8 \pm . 0 0 1$ </td><td> $\mathbf { . 0 0 7 \pm { . 0 0 0 } }$ </td><td> $\mathbf { 0 0 5 } \pm . \mathbf { 0 0 0 }$ </td><td> $\mathbf { 0 2 1 } \pm \mathbf { . 0 0 1 }$ </td><td> $\mathbf { \delta . 0 1 9 \pm . 0 0 2 }$ </td><td> $. 0 6 8 \pm . 0 1 0$ </td></tr><tr><td>Algorithm 2</td><td> $\mathbf { 0 0 6 \pm . 0 0 1 }$ </td><td> $\mathbf { 0 0 4 } \pm . \mathbf { 0 0 0 }$ </td><td> $. 0 1 2 \pm . 0 0 1$ </td><td> $. 0 0 6 \pm . 0 0 0$ </td><td> $. 0 2 9 \pm . 0 0 1$ </td><td> $. 0 2 1 \pm . 0 0 4$ </td><td> $. 0 8 5 \pm . 0 0 3$ </td></tr></table>

Table 2: MAE of different estimators under 0.45-flip noise.
<table><tr><td></td><td>CIFAR-10</td><td colspan="2">MNIST</td><td colspan="2">Letter</td><td colspan="2">Satellite</td></tr><tr><td></td><td></td><td>NN</td><td> ${ \mathrm { L o g i s t i c ~ r e g . } }$ </td><td>RF</td><td> $\operatorname { L o g i s t i c } \operatorname { r e g } .$ </td><td>RF</td><td> ${ \mathrm { L o g i s t i c ~ r e g . } }$ </td></tr><tr><td>Anchor-based</td><td> $. 0 2 9 \pm . 0 1 1$ </td><td> $. 0 1 6 \pm . 0 0 9$ </td><td> $\mathbf { . 0 2 3 \pm . 0 0 3 }$ </td><td> $. 0 4 6 \pm . 0 0 1$ </td><td> $. 0 4 8 \pm . 0 0 2$ </td><td> $\mathbf { . 0 3 9 \pm . 0 1 1 }$ </td><td> $. 0 8 5 \pm . 0 5 6$ </td></tr><tr><td>Algorithm 1</td><td> $. 0 3 6 \pm . 0 0 4$ </td><td> $. 0 3 3 \pm . 0 0 3$ </td><td> $. 0 3 3 \pm . 0 0 4$ </td><td> $\mathbf { 0 1 4 } \pm \mathbf { . 0 0 1 }$ </td><td> $\mathbf { 0 3 1 } \pm . \mathbf { 0 0 0 }$ </td><td> $. 0 4 4 \pm . 0 0 5$ </td><td> $\mathbf { 0 6 6 \pm . 0 0 3 }$ </td></tr><tr><td>Algorithm 2</td><td> $\mathbf { 0 1 5 } \pm . \mathbf { 0 0 5 }$ </td><td> $\mathbf { 0 0 8 } \pm \mathbf { . 0 0 3 }$ </td><td> $. 0 3 7 \pm . 0 0 6$ </td><td> $\mathbf { 0 1 4 } \pm \mathbf { . 0 0 1 }$ </td><td> $. 0 3 9 \pm . 0 0 1$ </td><td> $. 0 4 5 \pm . 0 0 4$ </td><td> $. 0 8 3 \pm . 0 0 3$ </td></tr></table>

Table 3: MAE of different estimators under random asymmetric uniform noise.
<table><tr><td></td><td>CIFAR-10</td><td colspan="2">MNIST</td><td colspan="2">Letter</td><td colspan="2">Satellite</td></tr><tr><td></td><td></td><td>NN</td><td>Logistic reg.</td><td>RF</td><td>Logistic reg.</td><td>RF</td><td>Logistic reg.</td></tr><tr><td>Anchor-based</td><td> $. 0 0 9 \pm . 0 0 1$ </td><td> $. 0 1 1 \pm . 0 0 1$ </td><td> $. 0 1 7 \pm . 0 0 2$ </td><td> $. 0 4 1 \pm . 0 0 1$ </td><td> $. 0 4 5 \pm . 0 0 1$ </td><td> $. 0 2 1 \pm . 0 0 3$ </td><td> $. 1 0 2 \pm . 0 6 5$ </td></tr><tr><td>Algorithm 1</td><td> $. 0 0 9 \pm . 0 0 3$ </td><td> $. 0 0 8 \pm . 0 0 1$ </td><td> $\mathbf { . 0 0 8 \pm . 0 0 1 }$ </td><td> $\mathbf { 0 0 6 \pm . 0 0 1 }$ </td><td> $\mathbf { 0 2 1 } \pm \mathbf { . 0 0 1 }$ </td><td> $\mathbf { . 0 2 0 \pm . 0 0 2 }$ </td><td> $\mathbf { . 0 6 8 \pm . 0 1 0 }$ </td></tr><tr><td>Algorithm 2</td><td> $\mathbf { 0 0 6 \pm . 0 0 1 }$ </td><td> $\mathbf { 0 0 5 } \pm . \mathbf { 0 0 1 }$ </td><td> $. 0 1 1 \pm . 0 0 3$ </td><td> $. 0 0 7 \pm . 0 0 0$ </td><td> $. 0 2 9 \pm . 0 0 1$ </td><td> $. 0 2 1 \pm . 0 0 3$ </td><td> $. 0 8 9 \pm . 0 1 1$ </td></tr></table>

As observed across the three noise regimes, our proposed methods outperform the anchor-based baseline on almost all datasets. As suggested by theoretical results, the anchor-based estimator suffers fundamentally from the difficulty of accurately estimating the class-posterior pointwise. In contrast, our results empirically validate that accurate transition matrix estimation does not require full pointwise class-posterior recovery. Instead, by framing the problem through one-sided selective classification, our methodology only requires learning a sufficiently capable binary classifier.

Furthermore, comparing Table 1 and Table 2 illustrates the practical impact of the theoretical constant $C _ { T }$ derived in our finite-sample bounds. The 0.45-flip noise setting is significantly more challenging than the 0.2-uniform noise setting, which translates to a larger $C _ { T }$ . In accordance with our theory, this increased difficulty causes all evaluated methods to achieve a higher absolute error under flip noise. Nonetheless, our methods, particularly Algorithm 2 when implemented with strong classifiers like NNs on CIFAR-10 and MNIST, demonstrate good performance despite the harsh noise setting.

Table 3 shows that under a more challenging and realistic setting, the different methods present a similar behavior to the scenario from Table 1. Furthermore, in this more demanding setting, on every dataset, at least one of the proposed algorithms outperforms the anchor-based baseline.

Effect of the dimensionality In Figure 5, we show how the error of the different algorithms increases with the number of dimensions d. Specifically, we apply PCA to the ConvNeXt features of CIFAR-10, reducing to dimension d $\in \{ 1 0 \bar { 0 } , 2 5 0 , 5 0 \bar { 0 } , 1 0 0 0 \}$ , and train a logistic regression classifier on the reduced features under uniform 0.2-noise, with the number of training samples fixed at $n = 5 0 0 0$

The error of the anchor-based method grows with d, consistent with the theoretical discussion in the paper. In contrast, the error of Algorithms 1 and 2 remain essentially flat across the tested range of d, in line with Theorem 1’s error bound, which has no explicit dependence on d.

![](images/35b8cb465e9418116dfa2bfecc86f4acc3c2e153532ad80a5a1aee17184a91dd.jpg)  
Figure 5: The presented methods remain insensitive to the dimension, whereas the error of the anchor-based method increases with the dimension.

Running times The computational cost of the presented algorithms increases with the number of classes because the number of learned binary classifiers grows linearly with Y . Nevertheless, this increase is not a major concern because each classifier is trained independently for each class, allowing the learning tasks to be run in parallel.

In Figure 6, we report the median wall-clock training time (in seconds) over 10 executions of each algorithm on the Letter dataset (26 classes) using 26 CPUs.

Algorithm 1’s runtime is comparable to the anchor-based method’s, consistent with each class’s binary problem being solved in parallel using one CPU per class. Algorithm 2 requires an additional grid search over $\lfloor 1 / \bar { \varepsilon } \rfloor = 2 0$ cost values per class $( \varepsilon = 0 . 0 5 )$ , i.e., 520 binary problems in total. With class-level parallelization, Algorithm $2 \mathrm { { ^ { * } s } }$ measured runtime is at most 43 times that of Algorithm 1 across the different sample sizes, somewhat above the naive estimate of 20 from grid size alone, reflecting implementation overheads. Warm-starting across adjacent grid values could further reduce this gap.

![](images/abc0c7859ad82f44c09cd6196b46cf63342e8479f7c9923d777554a56979c096.jpg)  
Figure 6: The proposed algorithms have low computational overhead in practice as an execution requires at most a few minutes to run.