# Rotation-Based Subspace Tracking for Robust Kernel PCA on Streaming Data

Kris Lokere<sup>∗</sup>

Harvard University, Cambridge, MA, USA<sup>†</sup>

klokere@g.harvard.edu

John Fossaceca

George Washington University, Washington, DC, USA

jfossaceca@email.gwu.edu

Abstract—Machine learning models process large amounts of data, and Principal Component Analysis (PCA) is a widely used technique to reduce the dimensionality of the data and extract useful features. In practice, datasets often change over time (data drift) and/or arrive one sample at a time (streaming data), making it infeasible to process the entire dataset at once in batch mode. Real-world data also often contains nonlinear patterns, which traditional PCA cannot extract. Kernel PCA addresses this by implicitly mapping samples into a Reproducing Kernel Hilbert Space (RKHS). Raw data also often contains outliers, which can have an outsized effect on the estimated subspace unless the algorithm is made robust. However, existing online robust kernel PCA algorithms are designed to converge to a subspace that is assumed to be fixed, and gradient-descent-based updates lose their effectiveness at tracking further changes once this initial alignment is achieved. This paper introduces a rotation-based update mechanism, which updates the subspace estimate by rotating it toward each new incoming feature vector in Reproducing Kernel Hilbert Space, rather than relying on gradient descent alone. We present two complementary rotation strategies, and show that the extent of rotation can be moderated by a robust influence function to mitigate the effect of outliers. Through experiments on synthetic streaming data with a known groundtruth subspace, we show that per-sample rotations converge faster than gradient descent alone, demonstrating an effective mechanism for dynamically tracking a nonlinear subspace in streaming data.

Index Terms—kernel PCA, online learning, robust statistics, subspace tracking, streaming data, Reproducing Kernel Hilbert Space

## I. INTRODUCTION

Machine learning models are trained on large amounts of data. When the number of features in the data is very high, processing cost becomes expensive, interpretability suffers, and performance of algorithms can deteriorate because of the ”curse of dimensionality.” Principal Component Analysis (PCA) is a technique to reduce the dimensionality of the data, in a way that lowers processing cost, minimizes information loss, and improves interpretability [1].

In practice, datasets often change over time (data drift) and/or arrive one sample at a time (streaming data). This makes it infeasible to process the entire dataset at once in batch mode. An improvement to batch processing is to calculate the PCA in an online manner, updating the calculation each time that new data is observed, without needing to recompute everything from scratch [2]. Real-world data also often contains nonlinear patterns, which traditional PCA cannot extract. Kernel PCA has been developed to extract nonlinear features from such data by implicitly mapping samples into a Reproducing Kernel Hilbert Space (RKHS) [3], [4].

While online robust kernel PCA algorithms exist, many are designed under the assumption that the underlying distribution is drawn from a static, unchanging subspace, and the focus is on getting the estimate to converge to that fixed subspace. This assumption breaks down for data that drifts over time, and a time-dependent estimation of the subspace is required instead. Existing iterative robust kernel PCA algorithms that update their subspace estimate via gradient descent, such as the method of Huang and Yeh [5], are provably convergent to a batch solution under mild assumptions, but this convergence guarantee applies only when the true subspace is fixed. Once initial convergence is achieved, gradient-based updates lose much of their effectiveness at tracking further changes in the underlying subspace, because the projection of new data onto the current subspace estimate leaves little residual signal to drive further movement.

This paper addresses this limitation by introducing a rotation-based update mechanism for online kernel PCA. Rather than relying solely on gradient descent, the estimated subspace is rotated—using a per-sample rotation matrix constructed in RKHS—toward the direction of each new incoming feature vector. This rotation is an original contribution: to the best of our knowledge, no prior online kernel PCA algorithm uses rotation matrices as an update mechanism for dynamically tracking a nonlinear subspace. We introduce two complementary rotation strategies, and show how the extent of rotation applied at each step can be moderated by a robust influence function, so that the effect of outliers on the estimated subspace remains bounded.

We validate this mechanism through controlled experiments on synthetic streaming data with a known ground-truth subspace. We show that enabling the rotation-based update leads to measurably faster convergence than relying on gradient descent alone, and that the mechanism naturally aligns individual basis vectors of the estimated subspace with the corresponding principal components of the underlying data, ordered by their eigenvalues.

The remainder of this paper is organized as follows. Section

II reviews related work in online, robust, and kernel PCA. Section III formulates the problem of dynamic subspace tracking in RKHS. Section IV presents the rotation-based update mechanism in detail. Section V analyzes its convergence behavior, and Section VI presents experimental results. Section VII discusses limitations and directions for future work, and Section VIII concludes.

## II. BACKGROUND AND RELATED WORK

## A. Online PCA

The standard way to calculate the PCA of a dataset is to consider all available data at once, in a batch calculation with time complexity $\mathcal { O } ( n D )$ and space complexity $\mathcal { O } ( D )$ , where n is the number of samples and D is the dimension of each data sample [6]. This quadratic complexity makes batch PCA infeasible for very large datasets, or for datasets that arrive in a streaming fashion. To overcome this limitation, iterative methods have been developed which process one sample at a time and update an estimate of the principal subspace with each new sample [7]. Such methods, including the Generalized Hebbian Algorithm [8] and the Sequential Karhunen-Loeve (SKL) transform [9], each achieve per-sample time and space complexity of $\mathcal { O } ( D )$ . A limitation of these approaches is that they assume the underlying data distribution is static; to track a distribution that changes over time, a ”forgetting factor” $\alpha < 1$ has been proposed to reduce the weight of previously observed data [10]. However, none of these online PCA methods are kernelized or robust to outliers.

## B. Kernel PCA

Traditional PCA relies on the assumption that the data lies in a linear subspace. Kernel PCA extends this by mapping the input data into a higher-dimensional feature space via a nonlinear function Φ, and performing linear PCA on the mapped features [3]. Rather than explicitly computing Φ, the ”kernel trick” allows all necessary computations to be expressed in terms of a kernel function $k ( \cdot , \cdot )$ , which acts as a similarity measure between data samples [4]. Because the resulting feature space is often infinite-dimensional, it is common in practice to use an empirical kernel map, in which a finite basis U of data samples is used to represent any new sample as a finite-dimensional vector of kernel products [11]. Standard kernel PCA methods are neither online nor robust, since they require calculating the kernel function between every pair of data samples, resulting in a matrix with $n ^ { 2 }$ entries that must be recomputed as new samples arrive.

## C. Robust PCA

Because every sample in a dataset influences the calculation of a PCA, the result is necessarily sensitive to outliers. Robust PCA refers to a family of techniques that modify traditional PCA to reduce this sensitivity. One approach, known as Principal Component Pursuit [12], separates the data matrix into a low-rank component and a sparse outlier component, solved via nuclear-norm minimization. A different approach applies an influence function to each sample, which measures and bounds the possible effect of a single outlier on the estimated principal components [13]. Because the boundedness of the influence function can be achieved by crafting a well-chosen, differentiable robust loss function, this approach is more readily adaptable to an online, kernelized setting than approaches based on the $\ell _ { 1 }$ norm, which are difficult to kernelize.

## D. Combining Online, Robust, and Kernel PCA

Several works combine two of these three capabilities. Online robust PCA methods, such as Recursive Projected Compressive Sensing [14] and its variants, recursively estimate a low-rank subspace and a sparse outlier component from streaming data, but do not extract nonlinear features. Online kernel PCA methods, such as the Kernel Hebbian Algorithm [15] and kernelized versions of the SKL transform [16], extend online PCA into RKHS, but are not robust to outliers. The closest prior work to this paper is the iterative robust kernel PCA algorithm of Huang and Yeh [5], which applies a robust loss function directly within the kernel PCA optimization. It can be proven, under mild assumptions, that this algorithm converges to the same subspace that would be found using batch kernel PCA on the entire dataset. However, this convergence guarantee explicitly assumes that the true underlying subspace is fixed: the algorithm is designed to converge to a static solution, and once initial convergence is achieved, its gradient-based update rule loses much of its effectiveness at tracking further changes in the underlying subspace. This is because, once the projection of a new sample onto the current subspace estimate is small, there is little residual signal left to drive further adjustment, even if the true underlying subspace has since shifted. This limitation motivates the need for a distinct update mechanism, capable of continuing to adjust the subspace estimate even after initial convergence.

## E. Grassmannian Geometry

The subspace Γ estimated by a kernel PCA algorithm can be represented as an orthonormal basis Q of $d$ vectors in an M-dimensional space, where M is the dimension of the (possibly empirical) kernel feature space. Such a subspace is a point on the Grassmann manifold, the set of all d-dimensional subspaces of an M-dimensional vector space, whose geometric structure (including its geodesics and tangent spaces) has been extensively studied in the context of optimization algorithms on matrix manifolds [17]. The distance between two such subspaces, represented by orthonormal bases $Q _ { 1 }$ and $Q _ { 2 } ,$ can be measured by first computing the singular value decomposition of $Q _ { 1 } ^ { T } Q _ { 2 }$ . The resulting singular values are the cosines of the principal angles $\theta _ { i }$ between the two subspaces, and the Grassmann distance is calculated as the root-sum-square of these principal angles.

Prior work has explored subspace tracking directly on the Grassmann manifold in an online setting, using gradient-based updates to incrementally estimate a subspace from streaming, incomplete data [18]. This geometric framing suggests a natural class of update mechanisms for online subspace tracking: rather than adjusting the subspace estimate only via the gradient of a loss function, one can instead move the estimate along a path on the Grassmann manifold, toward the direction indicated by each new incoming sample. This paper adopts a related geometric perspective, but instead uses rotation matrices as the mechanism for moving the estimated subspace along such a path.

## F. Summary of Gaps

Table I summarizes the capabilities of the algorithms discussed above, organized along four dimensions: whether the algorithm is designed to track a dynamically changing subspace, whether it operates in an online, per-sample fashion, whether it is kernelized to extract nonlinear features, and whether it is robust to outliers. As shown, several works combine two or three of these capabilities. The closest prior work, Huang and Yeh [5], combines online, kernel, and robust capabilities, but is not dynamic: it is designed to converge to a fixed subspace, and its update mechanism is not well suited to continued tracking of a subspace that changes gradually over time. This is the specific gap that this paper addresses, by introducing a rotation-based update mechanism, grounded in the Grassmannian geometry described in Section II-E, that enables continued dynamic tracking of the estimated subspace even after initial convergence has been achieved.

## III. PROBLEM FORMULATION

## A. Kernelizing Streaming Data

Consider a stream of data samples $\mathbf { x } _ { t } \in \mathbb { R } ^ { D } \ ( t = 1 , 2 , . . . )$ drawn from some underlying distribution that may change over time. To extract nonlinear features, each sample is mapped into a Hilbert space H via a nonlinear function Φ associated with a kernel function $k ( \cdot , \cdot )$ . Because $\Phi ( { \bf x } _ { t } )$ may be infinitedimensional, we use a reduced-basis empirical kernel map: a finite subset $\mathbf { U } = \{ \mathbf { u } _ { 1 } , \dots , \mathbf { u } _ { M } \} \subset \mathcal { X }$ of the sample space is chosen, and each incoming sample is converted into an $M -$ dimensional feature vector

$$
\mathbf { h } _ { t } = \Phi ( \mathbf { x } _ { t } ) = \left[ k ( \mathbf { u } _ { 1 } , \mathbf { x } _ { t } ) , \dots , k ( \mathbf { u } _ { M } , \mathbf { x } _ { t } ) \right] ^ { T }\tag{1}
$$

These M-dimensional vectors are treated as elements of a finite-dimensional approximation of $\mathcal { H } ,$ , and all subsequent steps of the algorithm operate on them.

## B. Subspace Representation and Projection

We represent the estimated subspace at time t by an $M \times d$ matrix $\mathbf { { { T } } } _ { t }$ , whose $d$ orthonormal columns span the current estimate, together with an estimated mean $\pmb { \mu } _ { t } \in \mathbb { R } ^ { M }$ . Each new feature vector $\mathbf { h } _ { t }$ is decomposed into a component along the subspace and a component orthogonal to it. The loadings along the subspace are

$$
\mathbf { c } _ { t } = ( \mathbf { h } _ { t } - \pmb { \mu } _ { t } ) ^ { T } \mathbf { T } _ { t }\tag{2}
$$

and the (half) squared distance from the centered feature vector to the subspace is

$$
z ( \mathbf { h } _ { t } , \mu _ { t } , \Gamma _ { t } ) = \frac { 1 } { 2 } \left\| \left( \mathbf { I } - \Gamma _ { t } \mathbf { r } _ { t } ^ { T } \right) ( \mathbf { h } _ { t } - \pmb { \mu } _ { t } ) \right\| _ { \mathcal { H } } ^ { 2 }\tag{3}
$$

This distance z serves both as a measure of how well the current subspace explains the new sample, and (as described next) as an input to a robustness mechanism.

## C. Robust Optimization Objective

Conventional kernel PCA seeks the subspace that minimizes the total squared distance of all samples to the subspace:

$$
\mu , \Gamma = \arg \operatorname* { m i n } _ { \Gamma ^ { T } \Gamma = \mathbf { I } } \sum _ { t } z ( \mathbf { h } _ { t } , \mu , \Gamma )\tag{4}
$$

To reduce the influence of outliers, we instead minimize a monotonically increasing, concave, differentiable robust loss function Ψ applied to z:

$$
\pmb { \mu } , \Gamma = \arg \operatorname* { m i n } _ { \Gamma ^ { T } \Gamma = \mathbf { I } } \sum _ { t } \Psi \left( z ( \mathbf { h } _ { t } , \pmb { \mu } , \Gamma ) \right)\tag{5}
$$

The derivative $\dot { \Psi } ( z )$ , which serves as an influence function, acts as a per-sample weight: samples farther from the current subspace estimate are assigned a smaller weight, bounding the influence any single outlier can have on the subspace update. Specific choices for Ψ satisfying these properties are discussed in [19].

## D. Limitation of Gradient-Based Updates

Solving (5) via gradient descent, using Lagrange multipliers to enforce the orthonormality constraint $\begin{array} { r } { \mathbf { \Gamma } \mathbf { \bar { \Gamma } } \mathbf { \bar { \Gamma } } = \mathbf { I } } \end{array}$ , yields an update rule of the form

$$
\boldsymbol { \Gamma } _ { t + 1 } = \boldsymbol { \Gamma } _ { t } - \eta _ { t } \mathbf { g } _ { t } ^ { ( \Gamma ) }\tag{6}
$$

where $\mathbf { g } _ { t } ^ { ( \Gamma ) }$ is the gradient of the Lagrangian with respect to $\mathbf { { \Gamma } } _ { \mathbf { { t } } } ^ { \mathbf { { \Gamma } } }$ , and $\eta _ { t }$ is a learning rate. This gradient-based update is provably convergent to the batch solution when the true underlying subspace is fixed [5]. However, the magnitude of $\mathbf { g } _ { t } ^ { ( \check { \Gamma } ) }$ depends on the residual component of $\left( \mathbf { h } _ { t } - \mu _ { t } \right)$ orthogonal to $\mathbf { { T } } _ { t }$ . Once $\mathbf { { T } } _ { t }$ has converged so that this residual is small for typical samples, the gradient itself becomes small, even if the true underlying subspace has since shifted, and even though new samples continue to carry information about the direction of that shift. This motivates a complementary update mechanism, described in Section IV, that continues to move the subspace estimate based on the direction of new samples rather than relying solely on the magnitude of the residual gradient.

## IV. ROTATION-BASED SUBSPACE UPDATE

## A. Motivation

As discussed in Section III, gradient-based updates to the subspace estimate $\mathbf { { T } } _ { t }$ become ineffective at tracking further changes once the residual component of incoming samples orthogonal to $\mathbf { { { T } } } _ { t }$ becomes small. We address this limitation by introducing a geometrically motivated update mechanism: rather than adjusting $\mathbf { { { T } } } _ { t }$ only in proportion to this residual, we rotate the entire subspace estimate toward the direction of each new incoming feature vector. Because $\mathbf { { { T } } } _ { t }$ is constrained to have orthonormal columns, such a rotation naturally preserves this constraint without requiring an additional projection step, and can be applied even when the residual signal used by gradient-based methods is small.

TABLE I: Algorithm Comparison
<table><tr><td rowspan=1 colspan=1>Algorithm</td><td rowspan=1 colspan=1>Reference</td><td rowspan=1 colspan=1>Dynamic</td><td rowspan=1 colspan=1>Online</td><td rowspan=1 colspan=1>Kernel</td><td rowspan=1 colspan=1>Robust</td></tr><tr><td rowspan=1 colspan=1>Generalized Hebbian Algorithm</td><td rowspan=1 colspan=1>Sanger [8]</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>×</td></tr><tr><td rowspan=1 colspan=1>SKL with forgetting factor</td><td rowspan=1 colspan=1>Ross et al. [10]</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>×</td></tr><tr><td rowspan=1 colspan=1>Principal Component Pursuit</td><td rowspan=1 colspan=1>Candès et al. [12]</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>√</td></tr><tr><td rowspan=1 colspan=1>Kernel Hebbian Algorithm</td><td rowspan=1 colspan=1>Kim et al. [15]</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>×</td></tr><tr><td rowspan=1 colspan=1>Kernel SKL</td><td rowspan=1 colspan=1>Chin &amp; Suter [16]</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>×</td></tr><tr><td rowspan=1 colspan=1>Grassmannian Rank-One Update</td><td rowspan=1 colspan=1>Balzano et al. [18]</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>×</td></tr><tr><td rowspan=1 colspan=1>Iterative Robust Kernel PCA</td><td rowspan=1 colspan=1>Huang &amp; Yeh [5]</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td></tr><tr><td rowspan=1 colspan=1>Rotation-Based Update (this paper)</td><td rowspan=1 colspan=1>_</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td></tr></table>

## B. Rotation Matrix Construction

Given two vectors $\boldsymbol { v } , { \mathbf w } \in \mathbb { R } ^ { M }$ , we construct a rotation matrix that rotates the space spanned by v and w through a fraction α of the angle between them, while leaving the orthogonal complement of that plane unchanged. First, the angle θ between v and w, and the desired rotation angle $\varphi ,$ are given by

$$
\theta = \operatorname { a r c c o s } \left( { \frac { \mathbf { v } ^ { T } \mathbf { w } } { \| \mathbf { v } \| \| \mathbf { w } \| } } \right) , \qquad \varphi = \alpha \theta\tag{7}
$$

A $2 \times 2$ elementary rotation matrix through angle $\varphi$ is embedded into an $M \times M$ block matrix $\mathbf { B } _ { \varphi }$ that leaves the remaining M − 2 dimensions unchanged:

$$
\mathbf { R } _ { \varphi } = \left[ \begin{array} { c c } { \cos \varphi } & { - \sin \varphi } \\ { \sin \varphi } & { \cos \varphi } \end{array} \right] , \qquad \mathbf { B } _ { \varphi } = \left[ \begin{array} { c c } { \mathbf { R } _ { \varphi } } & { \ \mathbf { 0 } } \\ { \mathbf { 0 } } & { \mathbf { I } _ { M - 2 } } \end{array} \right]\tag{8}
$$

To rotate specifically within the plane spanned by v and w, we compute an orthogonal change-of-basis matrix Q via QR factorization of a matrix whose first two columns are v and w, and whose remaining columns are chosen arbitrarily to complete a basis. The final rotation matrix is then

$$
\mathbf { Q } , _ { - } = \operatorname { Q R } \bigl ( [ \mathbf { v } \mathbf { \nabla } \mathbf { w } \mathbf { \nabla } \cdot \cdot \cdot ] \bigr ) , \qquad \mathbf { A } = \mathbf { Q } \mathbf { B } _ { \varphi } \mathbf { Q } ^ { T }\tag{9}
$$

The matrix A rotates the entire M-dimensional space, in the plane spanned by v, w, by an angle that is a specified fraction α of the angle between them, while leaving the orthogonal complement of that plane fixed.

## C. Rotation Type I: Projection-Based Rotation

In the first rotation strategy, we set $\mathbf { w } = \mathbf { h } _ { t } - \pmb { \mu } _ { t }$ and let v be the projection of w onto the current subspace estimate, $\mathbf { v } = \Gamma _ { t } \Gamma _ { t } ^ { T } ( \mathbf { h } _ { t } - \pmb { \mu } _ { t } )$ . The rotation matrix A constructed from v and w as in (9) is then applied to update the subspace:

$$
\mathbf { T } _ { t } \gets \mathbf { A } \mathbf { T } _ { t }\tag{10}
$$

This strategy efficiently rotates the overall subspace to align with the direction of incoming data, and is particularly effective during the initial convergence phase. However, once $\mathbf { { { T } } } _ { t }$ is well aligned with the data, the projection v becomes nearly identical to w, and the rotation (10) loses its effect. Moreover, Type I rotation does not, by itself, encourage the individual columns of Γ<sub>t</sub> to align with the principal directions of the data in order of decreasing eigenvalue.

## D. Rotation Type II: Nearest-Basis-Vector Rotation

To address this limitation, we introduce a second rotation strategy. Rather than rotating from the projection of w onto $\mathbf { { \Gamma } } _ { \mathbf { { t } } } ,$ we instead identify the single column $\gamma _ { i }$ of $\mathbf { { { T } } } _ { t }$ that is closest to w in terms of cosine similarity, and rotate from that column toward w:

$$
\mathbf { v } = \gamma _ { i ^ { * } } , \qquad i ^ { * } = \arg \operatorname* { m a x } _ { i } \left| \frac { \mathbf { w } ^ { T } \boldsymbol { \gamma } _ { i } } { \| \boldsymbol { \gamma } _ { i } \| } \right|\tag{11}
$$

The resulting rotation matrix A, constructed as in (9) using this v and $\begin{array} { r } { \mathbf { w } = \mathbf { h } _ { t } - \pmb { \mu } _ { t } , } \end{array}$ , is applied identically to (10). Because v is a single basis vector rather than the full projection, this rotation continues to have an effect even after the subspace as a whole has converged, and it encourages individual columns of $\mathbf { { T } } _ { t }$ to progressively align with the corresponding principal directions of the underlying data. In practice, we apply both Type I and Type II rotations at each time step, in sequence, combining the fast initial alignment of Type I with the continued refinement enabled by Type II.

## E. Robust Modulation of Rotation Magnitude

To ensure that outliers do not disproportionately affect the subspace estimate, the rotation fraction α in (7) is scaled at each time step by the same influence-function weight used in the gradient-based update:

$$
\alpha _ { t } = \alpha _ { 0 } \dot { \Psi } ( z ( \mathbf { h } _ { t } , \pmb { \mu } _ { t } , \pmb { \Gamma } _ { t } ) )\tag{12}
$$

where $\alpha _ { 0 }$ is a base rotation factor and $\dot { \Psi } ( \cdot )$ is the derivative of the robust loss function introduced in Section III. Samples that lie far from the current subspace estimate (and are therefore more likely to be outliers) are assigned a smaller effective rotation fraction $\alpha _ { t } .$ , bounding their influence on the subspace update in the same manner as for the gradient-based update.

## F. Computationally Optimized Rotations

A direct implementation of (9)–(10) requires explicitly forming the $M \times M$ matrix A, which becomes computationally infeasible for large M, for example, when M corresponds to the flattened dimension of high-resolution video frames. We avoid this by never explicitly constructing A as a dense $M \times M$ matrix. Instead, the rotation is decomposed into its action on the two-dimensional subspace spanned by v and w, and applied directly to the $M \times d$ matrix $\mathbf { { { T } } } _ { t }$ through a sequence of lower-dimensional operations, so that memory and computation scale with $M \times d$ rather than $M ^ { 2 }$

## G. Learning Rate

Both the base rotation factor $\alpha _ { 0 }$ and the gradient-based learning rate $\eta _ { t }$ introduced in Section III control a trade-off between convergence speed and stability. We keep both of these fixed over time, rather than decaying them as $t \to \infty ,$ as is common in classical stochastic approximation schemes. This is a deliberate design choice: a decaying learning rate would cause the algorithm to become unresponsive to further changes in the underlying subspace, which is contrary to the dynamictracking objective of this work. Algorithm 1 summarizes the complete per-sample update procedure described in this section.

Algorithm 1 Rotation-Based Online Robust Kernel PCA   
Require: Kernel basis U, subspace dimension $d ,$ learning rate   
$\eta ,$ base rotation factor $\alpha _ { 0 } ,$ robust loss function Ψ   
1: Initialize $\mu _ { 0 }  0 , \Gamma _ { 0 } $ orthonormal basis (e.g., first d   
standard basis vectors)   
2: for $t = 1 , 2 , \ldots$ do   
3: Receive new sample $\mathbf { x } _ { t }$   
4: $\mathbf { h } _ { t } \gets \Phi ( \mathbf { x } _ { t } )$ $\triangleright$ Kernel map via Eq. (1)   
5: $z _ { t } \gets z ( \mathbf { h } _ { t } , \pmb { \mu } _ { t - 1 } , \pmb { \Gamma } _ { t - 1 } )$ ▷ Distance to subspace,   
Eq. (3)   
6: $w _ { t } \gets \dot { \Psi } ( z _ { t } )$ ▷ Robust influence weight   
7: $/ / \ l$ Gradient-based update   
8: Compute $\mathbf { g } _ { t } ^ { ( \mu ) } , \mathbf { g } _ { t _ { \cdot } } ^ { ( \Gamma ) } ,$ using weight $w _ { t }$   
9: $\pmb { \mu } _ { t }  \pmb { \mu } _ { t - 1 } - \eta \mathbf { g } _ { t } ^ { ( \mu ) }$   
10: $\boldsymbol { \Gamma } _ { t } \gets \boldsymbol { \Gamma } _ { t - 1 } - \eta \mathbf { g } _ { t } ^ { ( \Gamma ) }$   
11: $\Gamma _ { t }  \mathrm { o r t h } ( \Gamma _ { t } )$ ▷ Re-orthonormalize columns   
12: // Rotation-based update   
13: $\mathbf { w }  \mathbf { h } _ { t } - \pmb { \mu } _ { t }$   
14: $\alpha _ { t }  \alpha _ { 0 } w _ { t }$ ▷ Robust rotation factor, Eq. (12)   
15: Type $I \colon \mathbf { v }  \Gamma _ { t } \mathbf { \Gamma } \Gamma _ { t } ^ { T } \mathbf { w }$   
16: A ← RotationMatrix $\left( \mathbf { v } , \mathbf { w } , \alpha _ { t } \right)$ ▷ Eq. (9)   
17: $\mathbf { T } _ { t } \gets \mathbf { A } \mathbf { T } _ { t }$   
18: Type II: i<sup>∗</sup> ← arg max<sub>i</sub> $\lvert \mathbf { w } ^ { T } \gamma _ { i } / \rvert \lvert \gamma _ { i } \rvert \rvert$   
19: $\mathbf { v }  \gamma _ { i ^ { * } }$   
20: A ← RotationMatrix $\left( \mathbf { v } , \mathbf { w } , \alpha _ { t } \right)$   
21: $\mathbf { T } _ { t } \gets \mathbf { A } \mathbf { T } _ { t }$   
22: // Output for current sample   
23: $\mathbf { c } _ { t } \gets ( \mathbf { h } _ { t } - \pmb { \mu } _ { t } ) ^ { T } \mathbf { T } _ { t }$ ▷ Loadings, Eq. (2)   
24: end for

## V. EXPERIMENTAL RESULTS

## A. Experimental Setup

To characterize the convergence behavior of the rotationbased update, we generate synthetic streaming data from a multivariate normal distribution in a D-dimensional space, with mean $\pmb { \mu }$ and covariance matrix $\pmb { \Sigma } = \mathbf { Q } \pmb { \Delta } \mathbf { Q } ^ { T }$ , where the orthonormal columns of $\mathbf { Q }$ represent the true underlying directions of the data, and the eigenvalues $\lambda _ { j }$ on the diagonal of $\pmb { \Delta }$ represent the variance along each corresponding direction:

$$
p ( \mathbf { x } \mid \pmb { \mu } , \pmb { \Sigma } ) = \frac { 1 } { \sqrt { ( 2 \pi ) ^ { D } | \pmb { \Sigma } | } } \exp \left( - \frac { ( \mathbf { x } - \pmb { \mu } ) ^ { T } \pmb { \Sigma } ^ { - 1 } ( \mathbf { x } - \pmb { \mu } ) } { 2 } \right)\tag{13}
$$

Because $\mathbf { Q }$ and $\pmb { \Delta }$ are known by construction, this synthetic setting allows the estimated subspace $\mathbf { { { T } } } _ { t }$ to be directly compared against the ground truth at every time step. We report two performance metrics: the variance-weighted Grassmann distance between the estimated and true subspace, and the cosine similarity between individual estimated eigenvectors and their ground-truth counterparts.

## B. Effect of Learning Rate

We generate data in a $D = \mathrm { 1 0 0 – d i m e n s i o n a l }$ space with a subspace dimension of $d = 1 0$ , with additional Gaussian noise added in all dimensions, and run Algorithm 1 multiple times, each with a different learning rate η. Fig. 1(a) shows the variance-weighted subspace distance as a function of the number of samples processed, for several values of $\eta ;$ Fig. 1(b) shows the corresponding cosine similarity of the first estimated eigenvector. As η increases, the subspace distance converges faster, at the expense of a slightly noisier estimate of the individual eigenvectors. This confirms that η provides the expected trade-off between convergence speed and stability of the estimate.

## C. Rotation Improves Convergence

To isolate the contribution of the rotation-based update, we generate data under the same conditions $( D \ = \ 1 0 0 ,$ $d = 1 0$ , with Gaussian noise), and run Algorithm 1 twice: once with the rotation steps (lines 12–19) disabled, so that the subspace is updated using only the gradient-based mechanism of Section III, and once with the full algorithm, including both Type I and Type II rotations. In both cases, all other hyperparameters, including the learning rate η, are held fixed. Fig. 2 shows the variance-weighted subspace distance as a function of the number of samples processed, with and without rotation enabled. The subspace distance to ground truth converges substantially faster when the rotation-based update is enabled, confirming that rotation provides a convergence benefit beyond what is achievable through gradient descent alone.

## D. Convergence Order by Eigenvalue

To examine how convergence depends on the magnitude of the underlying eigenvalues, we generate data with $D = 1 0 0$ and $d = 1 0 ,$ , setting the eigenvalues along the first five basis vectors to $\lambda \in \ \{ 6 0 , 2 0 , 1 6 , 8 , 4 \}$ . Fig. 3 shows the cosine similarity between each of the five estimated eigenvectors and their corresponding ground-truth directions, as a function of the number of samples processed. The eigenvector associated with the largest eigenvalue converges most quickly, and each subsequently smaller eigenvalue takes correspondingly longer to converge, with the eigenvector of smallest eigenvalue converging last. This ordering is consistent with the geometric intuition underlying Type II rotation (Section IV-D): directions with larger eigenvalues account for a larger share of the variance in the data, so a larger fraction of incoming samples carry a component along those directions, driving faster rotationbased alignment.

![](images/8e910e1a0de066ffe6be6198f041e1f63a6b4c0650f9b1ec833b8ddf677bc78a.jpg)

![](images/f0a46c01bb97590e8c94abf5ceb05419a1d64f67c5639117c590f5ad4aae6382.jpg)  
Fig. 1: (a) Variance-weighted subspace distance and (b) cosine similarity of the first eigenvector, for increasing learning rate η. Larger η yields faster convergence at the cost of increased noise.

![](images/4e67ccac2ed0971b8a08bded2b7ac4c2fe914035c92b27d64f8c617bd14c8ebf.jpg)  
Fig. 2: Variance-weighted subspace distance versus number of samples, with and without the rotation-based update enabled. Enabling rotation yields substantially faster convergence.

## VI. DISCUSSION

The experimental results in Section V demonstrate that the rotation-based update mechanism converges empirically in a manner consistent with its intended design: faster convergence is achieved when rotation is enabled compared to gradient descent alone, the learning rate hyperparameter trades off convergence speed against estimate stability in an interpretable way, and convergence proceeds in an order that reflects the relative importance of each principal direction. Together, these results support the central claim of this paper: that per-sample rotations in Reproducing Kernel Hilbert Space provide an effective mechanism for dynamic subspace tracking, beyond what is achievable through gradient-based updates alone.

![](images/9df9aceb045058282a36e0d29d4f4041601342818522d1ad083fce4a01f91b1f.jpg)  
Fig. 3: Cosine similarity of five estimated eigenvectors to ground truth, versus number of samples, for eigenvalues $\lambda \in \{ 6 0 , 2 0 , 1 6 , 8 , 4 \}$ . Eigenvectors with larger eigenvalues converge first.

## A. Scope of Experimental Validation

The experiments in this paper use synthetic data with known ground truth, in order to isolate and rigorously evaluate the convergence benefit of the rotation-based update mechanism itself. This choice allows the estimated subspace to be compared directly against a known reference at every time step, without the confounding effects of labeling noise or downstream task variance that would be introduced by real-world data. Validating this mechanism within a complete online, robust, kernel PCA pipeline on real-world application data (such as streaming network traffic or video images) is the subject of ongoing work.

## B. Theoretical Limitations

We do not provide a formal convergence theorem for the rotation-based update on the Grassmann manifold. The empirical results in Section V are consistent with convergence under the tested conditions, but they do not establish a convergence bound, nor characterize the trajectory of Γ<sub>t</sub> on the manifold in the way that has been done for gradient-based Grassmannian subspace tracking [18]. Establishing such a theoretical guarantee for the rotation-based update (for example, by characterizing the rotation step as a discrete approximation of a geodesic path on the Grassmann manifold) is a natural and important direction for future work.

## C. Practical Considerations

The rotation-based update introduces two additional hyperparameters beyond those required for the gradient-based update alone: the base rotation factor $\alpha _ { 0 } .$ , and the choice of applying Type I and/or Type II rotation at each step. While we found a fixed, non-decaying value of $\alpha _ { 0 }$ to be effective across the synthetic settings tested, we have not conducted a systematic study of how this hyperparameter should be tuned for datasets with different noise characteristics or different rates of subspace drift. In addition, the computational cost of the rotation-based update, while designed in Section IV-F to scale favorably with the feature dimension M, still requires constructing a QR factorization at each time step. A more detailed computational comparison against the cost of the gradient-based update alone is left for future work.

## VII. CONCLUSION

This paper introduced a rotation-based update mechanism for dynamic subspace tracking in online kernel PCA. Unlike gradient-based updates, which lose effectiveness once the residual component of incoming samples orthogonal to the estimated subspace becomes small, rotation-based updates continue to move the subspace estimate based on the direction of new samples, enabling continued tracking of a subspace that changes over time. We introduced two complementary rotation strategies: one that efficiently aligns the overall subspace during initial convergence, and one that continues to refine the ordering of individual basis vectors according to their corresponding eigenvalues. We also showed how the magnitude of each rotation can be modulated by a robust influence function to bound the effect of outliers. Through controlled experiments on synthetic streaming data with known ground truth, we demonstrated that the rotation-based update converges substantially faster than gradient descent alone, and that this convergence behavior is consistent with the underlying geometric intuition of the mechanism. While a formal convergence theorem on the Grassmann manifold remains an open problem, the empirical results presented here establish rotation-based updates as an effective and efficient mechanism for dynamic subspace tracking, and lay the groundwork for validating this mechanism on a complete online, robust, kernel PCA pipeline on real-world streaming data.

## REFERENCES

[1] I. T. Jolliffe and J. Cadima, “Principal component analysis: A review and recent developments,” Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, vol. 374, no. 2065, p. 20150202, 2016.

[2] S. Dasgupta, S. Kumar, S. Pandey, and P. Sarkar, “Low precision streaming PCA,” Advances in Neural Information Processing Systems, vol. 38, pp. 157961–157996, 2026.

[3] B. Scholkopf, A. Smola, and K.-R. M ¨ uller, “Kernel principal component¨ analysis,” in Artificial Neural Networks — ICANN’97. Springer, 1997, pp. 583–588.

[4] F. Tonin, Q. Tao, P. Patrinos, and J. A. K. Suykens, “Deep kernel principal component analysis for multi-level feature learning,” Neural Networks, vol. 170, pp. 578–595, 2024.

[5] H.-H. Huang and Y.-R. Yeh, “An iterative algorithm for robust kernel principal component analysis,” Neurocomputing, vol. 74, no. 18, pp. 3921–3930, 2011.

[6] C. J. Li, M. Wang, H. Liu, and T. Zhang, “Near-optimal stochastic approximation for online principal component estimation,” Mathematical Programming, vol. 167, no. 1, pp. 75–97, 2018.

[7] E. Oja and J. Karhunen, “On stochastic approximation of the eigenvectors and eigenvalues of the expectation of a random matrix,” Journal of Mathematical Analysis and Applications, vol. 106, no. 1, pp. 69–84, 1985.

[8] T. D. Sanger, “Optimal unsupervised learning in a single-layer linear feedforward neural network,” Neural Networks, vol. 2, no. 6, pp. 459– 473, 1989.

[9] A. Levey and M. Lindenbaum, “Sequential Karhunen-Loeve basis extraction and its application to images,” IEEE Transactions on Image Processing, vol. 9, no. 8, pp. 1371–1374, 2000.

[10] D. A. Ross, J. Lim, R.-S. Lin, and M.-H. Yang, “Incremental learning for robust visual tracking,” International Journal of Computer Vision, vol. 77, no. 1, pp. 125–141, 2008.

[11] C.-M. Vong, C. Chen, and P.-K. Wong, “Empirical kernel map-based multilayer extreme learning machines for representation learning,” Neurocomputing, vol. 310, pp. 265–276, 2018.

[12] E. J. Candes, X. Li, Y. Ma, and J. Wright, “Robust principal component\` analysis?” Journal of the ACM, vol. 58, no. 3, pp. 1–37, 2011.

[13] F. R. Hampel, E. M. Ronchetti, P. J. Rousseeuw, and W. A. Stahel, Robust Statistics: The Approach Based on Influence Functions, 1st ed. Wiley, 2005.

[14] C. Qiu, N. Vaswani, B. Lois, and L. Hogben, “Recursive robust PCA or recursive sparse recovery in large but structured noise,” IEEE Transactions on Information Theory, 2014.

[15] K. I. Kim, M. Franz, and B. Scholkopf, “Iterative kernel principal¨ component analysis for image modeling,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 27, no. 9, pp. 1351–1366, 2005.

[16] T.-J. Chin and D. Suter, “Incremental kernel principal component analysis,” IEEE Transactions on Image Processing, vol. 16, no. 6, pp. 1662–1674, 2007.

[17] P.-A. Absil, R. Mahony, and R. Sepulchre, Optimization Algorithms on Matrix Manifolds. Princeton, NJ: Princeton University Press, 2008.

[18] L. Balzano, R. Nowak, and B. Recht, “Online identification and tracking of subspaces from highly incomplete information,” in Proc. 48th Annual Allerton Conference on Communication, Control, and Computing, 2010, pp. 704–711.

[19] I. Higuchi and S. Eguchi, “The influence function of principal component analysis by self-organizing rule,” Neural Computation, vol. 10, no. 6, pp. 1435–1444, 1998.