# Federated Differentially Private Optimization via Low-Rank Preconditioning

Antti Koskela, Tejas Kulkarni, Elias Raninen Nokia Bell Labs

## Abstract

We consider differentially private (DP) optimization in the federated setting, where a DP estimate of the feature covariance matrix serves as a Hessian approximation for generalized linear models. Existing central DP methods for covariance estimation have communication cost that grows quadratically in the data dimension, while existing baselines with linear communication cost are only locally private and degrade as the number of clients increases. As a remedy, we propose a federated DP-PCA framework that achieves linear communication cost while providing central DP guarantees under secure aggregation. It consists of a DP rank selection step, a DP power method, and a noisy release of the covariance projected onto the selected low-dimensional subspace. We give error bounds for this covariance estimate and for the resulting low-communication preconditioned DP gradient descent. Empirically, our proposed DP optimization method matches a centralized baseline with quadratic communication cost at a fraction of the communication.

## 1 Introduction

Estimating the covariance matrix of high-dimensional data is a fundamental primitive in machine learning. A low-rank approximation of the covariance often provides a useful proxy of the full covariance, can be interpreted through principal component analysis (PCA), and serves as a building block for tasks such as dimensionality reduction and denoising. In many realistic scenarios the data is distributed across several data-owners that cannot or do not want to centralize their data. This motivates federated algorithms that can estimate a global low-rank covariance while providing differential privacy (DP) (Dwork et al., 2006) guarantees for the data owners. At the same time, for practical purposes it would often be crucial to have communication costs that scale linearly rather than quadratically with the feature dimension p.

Covariance estimates can also be used as Hessian approximations in optimization and can accelerate first-order DP optimization methods. Recent private preconditioned methods use DP covariance estimates to approximate second-order information (Mehta et al., 2023; Krouka et al., 2025), reaching the same privacy-utility trade-off as DP-SGD in fewer iterations (see also Ganesh et al., 2023). In federated settings, where each iteration corresponds to a communication round, this directly reduces communication costs. However, obtaining the DP covariance estimate remains the computational and communication bottleneck.

For example, DP-FC (Mehta et al., 2023) relies on so called Analyze Gauss mechanism (Dwork et al., 2014) applied to the full covariance matrix, which in a federated implementation requires $O ( p ^ { 2 } )$ communication. To the best of our knowledge, there is no prior federated method for private second-order optimization that uses a communication-efficient DP low-rank covariance (Hessian proxy) estimate and simultaneously supports secure aggregation end-to-end. We fill this gap

DP-FL via secure summation. While federated learning (FL) (Kairouz et al., 2021b) improves on naive collaborative learning by avoiding direct data transmission, the local updates can still leak information about training data, e.g. through membership inference attacks (Shokri et al., 2017), which DP protects against. Most DP-FL methods follow the same pattern: clients add Gaussian noise to their local messages (e.g., ML model updates) and the server aggregates them (Truex et al., 2019). When the noise is additive and the updates are sums of user contributions, secure aggregation protocols let the clients carry the noise injection themselves, removing the need to trust a central server (Ullah et al., 2023; Kairouz et al., 2021a).

Related Work on DP covariance estimation. The classical Analyze Gauss mechanism (Dwork et al., 2014) adds Gaussian noise directly to the $p \times p$ covariance matrix and post-processes it to obtain a low-rank approximation. The algorithm distributes naturally, but requires $O ( p ^ { 2 } )$ communication, which becomes prohibitive in high feature dimensions p. Exponential-mechanism-based methods (Chaudhuri et al. 2012) and recent centralized refinements (Kim and Jung, 2025; He et al., 2025; Cai et al., 2024), in contrast, do not decompose into local updates compatible with secure aggregation.

Related Work on DP-FL covariance estimation. There are existing works that consider federated algorithms with $O ( p )$ communication (Grammenos et al., 2020; Guo et al., 2024; Li et al., 2024). All of these methods are locally DP: each client calibrates its noise to its local sensitivity and then applies postprocessing (e.g., a truncated SVD), so that post-processed quantities are aggregated rather than sums. This also prevents obtaining central DP guarantees. As a consequence, their privacy-utility trade-off degrades as the number of clients grows. The federated DP power method of Nicolas et al. (2024) (also studied by Balcan et al., 2016; Campbell et al., 2025) does enjoy the benefits of summation, but focuses exclusively on subspace computation. Our contribution is to use this basis step in a complete end-to-end framework that also handles private covariance estimation and rank selection, to empirically show it leads to improvements in DP-FL optimization and to analyze its use as a preconditioner in DP-FL optimization.

## 1.1 Our Contributions

• Communication-efficient federated DP-PCA. We introduce the first federated DP low-rank covariance estimator (subspace, projected covariance and rank) that simultaneously achieves $O ( p k )$ per-client communication per round, where p is the data feature dimension and k rank of the low-rank approximation, and consists of summation operations at every aggregation stage, making it end-to-end compatible with secure aggregation. The method is a three-stage procedure: (i) a DP rank-selection step based on a shared random projection, (ii) a federated DP power method for the leading k left singular vectors, and (iii) an Analyze Gauss mechanism applied to the projected $k \times k$ covariance. We provide an end-to-end error bound for the DP low-rank covariance estimate that separates cleanly into a subspace term and a projected covariance matrix noise term.

• Convergence of low-rank preconditioned DP-FC. We analyze federated DP-FC with the DP low-rank preconditioner on linear regression and show how to extend the analysis to generalized linear models (GLMs). We show that a rank-k preconditioner is approximately a Newton step inside the estimated subspace and a gradient step on its orthogonal complement, and show that there is an effective condition number that governs the convergence and that it is commonly much smaller than the condition number governing the convergence of DP gradient descent (DP-GD). This reduces the number of communication rounds relative to first-order DP-FL by a large explicit factor and gives a closed-form step size.

• Empirical validation. We experimentally show that our DP covariance approximation matches or improves upon the centralized $O ( p ^ { 2 } )$ Analyze Gauss baseline at a fraction of the communication cost, and, unlike LDP-style federated baselines, its accuracy does not degrade as the number of clients grows. When used as a preconditioner for gradients in DP-FL fine-tuning, it improves the accuracycommunication trade-off over both vanilla federated DP-FC (with $O ( p ^ { 2 } )$ communication) and firstorder baselines on benchmark image datasets for moderate privacy budgets.

## 2 The Problem

Assume that each client $j \in [ N ] : = \{ 1 , . . . , N \}$ has a data matrix $\mathbf { X } _ { j } = ( \mathbf { x } _ { 1 } ^ { ( j ) } , \ldots , \mathbf { x } _ { n _ { j } } ^ { ( j ) } ) \in \mathbb { R } ^ { p \times n _ { j } }$ , where p denotes the dimension and $n _ { j }$ the number of samples at client $j .$ The full data matrix is denoted by $\mathbf { X } = ( { \mathbf { X } } _ { 1 } , \ldots , \mathbf { X } _ { N } ) \in \mathbb { R } ^ { p \times n }$ , where $\begin{array} { r } { n = \sum _ { j = 1 } ^ { N } n _ { j } } \end{array}$ . The singular value decomposition (SVD) is defined by $\mathbf { X } =$ $\begin{array} { r } { \mathbf { U } \pmb { \Sigma } \mathbf { W } ^ { \top } = \sum _ { i } \sigma _ { i } \mathbf { u } _ { i } \mathbf { w } _ { i } ^ { \top } } \end{array}$ , where the orthogonal matrices $\mathbf { U } = ( \mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { p } ) \in \mathbb { R } ^ { p \times p }$ and $\mathbf { W } = ( \mathbf { w } _ { 1 } , \ldots , \mathbf { w } _ { n } ) \in$ Rn×n contain the left and right singular vectors, respectively, and $\pmb { \Sigma } \in \mathbb { R } ^ { p \times n }$ contains the singular values $\sigma _ { i }$ $1 \leq i \leq \operatorname* { m i n } \{ n , p \}$ , on its main diagonal.

In principal component analysis (PCA), one seeks a set of k mutually orthogonal vectors called principal components that capture the directions of largest variance of the data matrix. The first k principal components are given by the SVD as the k left singular vectors $( \mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { k } )$ of the data matrix. They also coincide with the principal k eigenvectors of the covariance matrix² $\mathbf { K } : = \mathbf { X } \mathbf { X } ^ { \top } = \mathbf { U } \mathbf { A } \mathbf { U } ^ { \top }$ , where $\pmb { \Lambda } \in \mathbb { R } ^ { p \times p }$ is a diagonal matrix containing the eigenvalues $\lambda _ { i } \geq 0 , 1 \leq i \leq p .$ on its diagonal. Notice that the global covariance matrix can also be written as a sum of the client-wise covariance matrices: $\begin{array} { r } { \mathbf { X } \mathbf { X } ^ { \top } = \sum _ { j = 1 } ^ { N } \mathbf { X } _ { j } \mathbf { X } _ { j } ^ { \top } } \end{array}$

Federated DP-PCA. Our first goal is to develop a differentially private federated algorithm to compute a global k-dimensional subspace by aggregating contributions from the clients and also produce a global rankk covariance matrix estimate. In addition to approximating K accurately with DP guarantees, we want a small communication overhead. Naïve approaches require $O ( p ^ { 2 } )$ communication (since K has $p ^ { 2 }$ elements), however, we would like to achieve $O ( p \cdot k )$ communication, where k is the number of leading singular vectors to be estimated.

Preconditioned federated DP optimization. Suppose in addition that each sample x has a label $y ,$ write D for the full labeled dataset, and consider the regularized empirical risk $\begin{array} { r } { L ( \theta , D ) = \frac { 1 } { n } \sum _ { ( \mathbf { x } , y ) \in D } \ell ( \theta ; \mathbf { x } , y ) + } \end{array}$ $\frac { \lambda } { 2 } \| \theta \| _ { 2 } ^ { 2 } , \lambda > 0$ , where $\ell$ is a generalized linear model (GLM) loss, e.g., a linear classification head trained on output features of a pretrained model. DP-FC (Mehta et al., 2023) is the preconditioned DP gradient descent

$$
\begin{array} { r } { \theta _ { j + 1 } = \theta _ { j } - \eta \widehat { \mathbf { G } } ^ { - 1 } \big ( \nabla L ( \theta _ { j } , D ) + \mathbf { E } _ { j } \big ) , } \end{array}\tag{2.1}
$$

where $\mathbf { E } _ { j }$ is the Gaussian noise added to the sum of clipped per-sample gradients and $\widehat { \mathbf { G } }$ is a DP estimate of $\textstyle { \frac { 1 } { n } } \mathbf { X } \mathbf { X } ^ { \top } + \lambda \mathbf { I }$ . For GLMs the Hessian of $L$ is dominated by this matrix (up to diagonal scaling), so that (2.1) can be seen as an approximate Newton iteration. In particular, in the expression (2.1), both the gradient $\nabla L ( \theta _ { j } , D )$ and the covariance $\mathbf { \mathbf { X X } ^ { \top } }$ are sums of client-wise contributions.

In DP-FC (Mehta et al., 2023), $\widehat { \mathbf { G } }$ is obtained by Analyze Gauss applied to the full $p \times p$ covariance. The naïve federated version considered as a baseline method by Krouka et al. (2025) is straightforward: the clients upload and the server broadcasts $O ( p ^ { 2 } )$ scalars. Our second goal is to remove the communication costs by considering the approximation

$$
\widehat { \mathbf { G } } = \mathbf { V } _ { k } \widetilde { \mathbf { Y } } _ { k } \mathbf { V } _ { k } ^ { \top } + \lambda \mathbf { I } ,
$$

where $( \mathbf { V } _ { k } , \widetilde { \mathbf { Y } } _ { k } ) \in \mathbb { R } ^ { p \times k } \times \mathbb { R } ^ { k \times k }$ is the factored output of a federated DP-PCA algorithm. Then only $O ( p \cdot k )$ scalars per round are communicated, $\widehat { \mathbf { G } } ^ { - 1 }$ is applied in $O ( p k + k ^ { 2 } )$ operations without forming any $p \times p$ matrix, and the preconditioner noise is a $k \times k$ matrix. We require that every stage (rank selection, basis computation, projected covariance estimation, gradient rounds) is a sum of client-wise contributions plus Gaussian noise, so that the whole pipeline is compatible with secure aggregation and its record-level $( \varepsilon , \delta )$ DP guarantee (under the add/remove relation of Section 3) follows from a single composition of Gaussian mechanisms (Lemma 2).

## 3 Preliminaries of Differential Privacy

DP gives quantifiable measures in terms of the parameters $\varepsilon$ and $\delta$ that guarantee indistinguishability of any individual element in the dataset from the output of the algorithm

Neighboring datasets. Throughout, we use the record-level add/remove (unbounded) neighboring relation: two datasets D and $\mathbf { D ^ { \prime } }$ are neighbors, denoted $\mathbf { D } \simeq \mathbf { D } ^ { \prime }$ , if one is obtained from the other by adding or removing a single record $\left( \mathbf { x } , y \right)$ , i.e., a single column of X, held by any one of the clients. All the mechanisms considered can be seen as noisy sums of per-record contributions. Whenever a normalization is needed $\scriptstyle ( \mathbf { e . g . } , { \frac { 1 } { n } } \mathbf { X } ^ { \mathsf { T } }$ or an averaged gradient), the noisy sum is divided by the total number of data elements n, which we treat as public. Remark 4 explains how the guarantees translate to the substitute-one relation.

Definition 1 (Dwork et al., 2006). A randomized algorithm (mechanism) M : $\mathcal { D } \to \mathcal { R }$ is $( \varepsilon , \delta )$ -differentially private if for all neighboring datasets $\mathbf { D } \simeq \mathbf { D } ^ { \prime } , \mathbf { D } , \mathbf { D } ^ { \prime } \in \mathcal { D }$ , and for all measurable sets $S$ in the output space

$\mathcal { R } , S \subset \mathcal { R } ,$ we have

$$
P ( \mathcal { M } ( \mathbf { D } ) \in S ) \leq e ^ { \varepsilon } P ( \mathcal { M } ( \mathbf { D } ^ { \prime } ) \in S ) + \delta .
$$

All the subroutines we consider in this work, as well as their combination, can be seen as adaptive compositions of Gaussian mechanisms. In its simplest form, a Gaussian mechanism M is of the form

$$
\mathcal { M } ( { \bf D } ) = { \bf f } ( { \bf D } ) + { \bf Z } ,
$$

where $\mathbf { f } : \mathcal { D }  \mathbb { R } ^ { p }$ and $\mathbf { Z } \sim { \mathcal { N } } ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } _ { p } )$ , and $\begin{array} { r } { \Delta = \operatorname* { m a x } _ { \mathbf { D } \simeq \mathbf { D } ^ { \prime } } \left. \mathbf { f } ( \mathbf { D } ) - \mathbf { f } ( \mathbf { D } ^ { \prime } ) \right. _ { 2 } } \end{array}$ gives the $L _ { \mathrm { { 2 ^ { - S e n s i t i v i t y } } } }$ . In case of a matrix-valued Gaussian mechanism, we add i.i.d. distributed noise to each element of the matrix and consider the vectorized $L _ { \mathrm { 2 ^ { - S e n s i t i v i t y , } i . e . } }$ , the sensitivity in Frobenius norm, which for a matrix $\mathbf { A } = ( A _ { i j } )$ is defined as $\begin{array} { r } { \| \mathbf { A } \| _ { F } = \sqrt { \operatorname { t r } ( \mathbf { A } ^ { \top } \mathbf { A } ) } = \sqrt { \sum _ { i , j } A _ { i j } ^ { 2 } } } \end{array}$

DP guarantees for a composition of $T$ Gaussian mechanisms, each with varying sensitivity $\Delta _ { i }$ and noise scale $\sigma _ { i } .$ is given by the following result.

Lemma 2. Suppose we have a composition of T Gaussian mechanisms, where the ith mechanism has $L _ { 2 } \cdot$ sensitivity $\Delta _ { i }$ and noise scale parameter $\sigma _ { i } , i \in [ T ]$ . Then, the total composition is $( \varepsilon , \delta ) \ – D P$ for

$$
\delta ( \varepsilon ) = \Phi \left( - \frac { \varepsilon } { \widetilde { \Delta } } + \frac { \widetilde { \Delta } } { 2 } \right) - e ^ { \varepsilon } \Phi \left( - \frac { \varepsilon } { \widetilde { \Delta } } - \frac { \widetilde { \Delta } } { 2 } \right) ,\tag{3.1}
$$

where $\widetilde { \Delta } = \sqrt { \sum _ { i = 1 } ^ { T } \Delta _ { i } ^ { 2 } / \sigma _ { i } ^ { 2 } }$

Remark 3. Equivalently, in Lemma 2, the composition is -Gaussian Differentially Private (GDP) (Dong et al., 2022): the expression (3.1) is the $( \varepsilon , \delta )$ -characterization of $\mu { - } G D P$ with $\mu = \widetilde { \Delta }$

Remark 4 (DP guarantees for substitute-one relation). All ε-values reported in this paper can be converted to the substitute-one (bounded) neighborhood relation, where D' is obtained from D by replacing one record For the matrix releases (rank selection algorithm, power iterations and the projected covariance release), we can use the fact that replacing x by x′ with $\| \mathbf { x } \| _ { 2 } , \| \mathbf { x } ^ { \prime } \| _ { 2 } \leq C$ gives $\big \| \mathbf { x } \mathbf { x } ^ { \top } - \mathbf { x } ^ { \prime } \mathbf { x } ^ { \prime \top } \big \| _ { F } ^ { 2 } = \| \mathbf { x } \| _ { 2 } ^ { 4 } + \| \mathbf { x } ^ { \prime } \| _ { 2 } ^ { 4 } -$ $2 ( \mathbf { x } ^ { \top } \mathbf { x } ^ { \prime } ) ^ { 2 } \leq 2 C ^ { 4 }$ , and that in case of projected covariance release, multiplication by matrices with orthonormal columns does not increase the Frobenius norm. Hence these sensitivities grow by a factor $\sqrt { 2 }$ . For the sums of clipped per-sample gradients, the sensitivity grows by a factor 2. Since Lemma 2 depends on the mechanisms only through the ratios $\Delta _ { i } / \sigma _ { i }$ , the substitute-one (ε,δ)-guarantees are obtained by evaluating (3.1) with the noise scales $\sigma _ { i }$ replaced by $\sigma _ { i } / \sqrt { 2 }$ for the DP-PCA stages and by $\sigma _ { i } / 2$ for the gradient rounds.

## 4 Communication-Efficient DP-PCA

## 4.1 Basis Computation with DP Power Method

We next discuss computing a differentially private k-principal subspace of the data matrix X using the DP Power Iteration (Nicolas et al., 2024). This method simply applies a noisy power iteration and then projects

the matrix onto the set of matrices with orthonormal columns. This means that we first compute

$$
\widetilde { \mathbf { V } } _ { i } = ( \mathbf { X } \mathbf { X } ^ { \top } ) \mathbf { V } _ { i - 1 } + \mathbf { Z } _ { i } ,
$$

where $\mathbf { Z } _ { i } \in \mathbb { R } ^ { p \times k }$ , and each element of $\mathbf { Z } _ { i }$ is independent and zero-mean normally distributed with variance $\sigma _ { 1 } ^ { 2 } .$ where $\sigma _ { 1 } = C ^ { 2 } \sigma$ for a noise scale $\sigma > 0$ and $C > 0$ is a public bound on the 2-norms of the data vectors (columns) of X, enforced by clipping the data as preprocessing. Then, the obtained $\widetilde { \mathbf { V } } _ { i }$ is orthogonalized via the QR-decomposition:

$$
( \mathbf { V } _ { i } , \mathbf { R } _ { i } ) = \mathrm { Q R } ( \widetilde { \mathbf { V } } _ { i } ) .
$$

A distributed version of this method is straightforward: each client j simply computes at iteration i

$$
( \mathbf { X } _ { j } \mathbf { X } _ { j } ^ { \top } ) \mathbf { V } _ { i - 1 } + \mathbf { Z } _ { i , j } ,
$$

where each element of $\mathbf { Z } _ { i , j }$ is independent and normally distributed with variance $\sigma _ { 1 } ^ { 2 } / N$ , and then sends this result to the server, which carries out the QR-orthogonalization. To bound the sensitivity of the Gaussian mechanism at each step, we can use the fact that $\lVert \mathbf { x } \mathbf { x } ^ { \top } \mathbf { V } \rVert _ { F } = \lVert \mathbf { x } \rVert _ { 2 } \lVert \mathbf { V } ^ { \top } \mathbf { x } \rVert _ { 2 } \leq \lVert \mathbf { x } \rVert _ { 2 } ^ { 2 }$ for any $\mathbf { V } \in \mathbb { R } ^ { p \times k }$ with orthonormal columns. Hence, under the add/remove relation, each iteration is a Gaussian mechanism with sensitivity $C ^ { 2 }$ and noise standard deviation $\sigma _ { 1 } = C ^ { 2 } \sigma$

## 4.2 Private Covariance Matrix Estimation

After the basis approximation with DP power method, the local covariance matrices are projected and aggregated privately. Putting together, the core of our proposed algorithm (preceded by the rank selection of Section 5) consists of the following two stages:

First Stage: Carry out the full batch DP power method with T iterations to obtain an orthogonal matrix $\mathbf { V } _ { k } \in \mathbb { R } ^ { p \times k }$

Second Stage: Project the global covariance matrix $\mathbf { X X ^ { \top } }$ onto the subspace spanned by the columns of $\mathbf { V } _ { k } , \mathbf { Y } _ { k } = \mathbf { V } _ { k } ^ { \top } ( \mathbf { X } \mathbf { X } ^ { \top } ) \mathbf { V } _ { k }$ , and release

$$
\widetilde { \mathbf { Y } } _ { k } = \mathbf { Y } _ { k } + \mathbf { Z } ,
$$

where $\mathbf { Z } \in \mathbb { R } ^ { k \times k }$ is a symmetric Gaussian matrix with i.i.d. $\mathcal { N } ( 0 , \sigma _ { 2 } ^ { 2 } )$ upper-triangular entries (including the diagonal), where $\sigma _ { 2 } = C ^ { 2 } \sigma ^ { \prime }$ for a noise scale $\sigma ^ { \prime } > 0$ , following the Analyze Gauss mechanism (Dwork et al., 2014). The scaling with $C ^ { 2 }$ comes from the fact that under the add/remove relation, $\| \mathbf { V } _ { k } ^ { \top } \mathbf { x } \mathbf { x } ^ { \top } \mathbf { V } _ { k } \| _ { F } =$ $\| \mathbf { V } _ { k } ^ { \top } \mathbf { x } \| _ { 2 } ^ { 2 } \leq C ^ { 2 }$ , and the upper-triangular vectorization has $\ell _ { 2 } { \mathrm { - n o r m } }$ at most the Frobenius norm, so this release is a Gaussian mechanism with sensitivity $C ^ { 2 }$ and noise standard deviation $\sigma _ { 2 }$ . Also this step is carried out in a distributed fashion with the sum of local contributions equaling ${ \bf Y } _ { k }$ while off-loading the noise addition to the clients.

The output of these two stages is the factored pair $( \mathbf { V } _ { k } , \widetilde { \mathbf { Y } } _ { k } )$ . Forming the rank-k approximation $\widetilde { \mathbf { K } } _ { k } : =$ $\mathbf { V } _ { k } \widetilde { \mathbf { Y } } _ { k } \mathbf { V } _ { k } ^ { \top }$ can be performed locally, so the server never needs to transmit the full covariance matrix, keeping the communication at $O ( p k )$ per round and $O ( T p k + k ^ { 2 } )$ in total.

Remark 5 (Privacy of the two-stage algorithm). The two stages form an adaptive composition of $T + 1$ Gaussian mechanisms, each with sensitivity $C ^ { 2 }$ : T mechanisms with noise standard deviation $\sigma _ { 1 }$ and one with noise standard deviation $\sigma _ { 2 }$ . By Lemma ${ \mathit { 2 } } ,$ the release $( \mathbf { V } _ { k } , \widetilde { \mathbf { Y } } _ { k } )$ is therefore $\mu { - } G D P$ with

$$
\mu = \sqrt { \frac { T C ^ { 4 } } { \sigma _ { 1 } ^ { 2 } } + \frac { C ^ { 4 } } { \sigma _ { 2 } ^ { 2 } } } = \sqrt { \frac { T } { \sigma ^ { 2 } } + \frac { 1 } { \sigma ^ { \prime 2 } } } ,
$$

and its $( \varepsilon , \delta )$ -guarantees are given by (3.1) with $\widetilde { \Delta } = \mu$ . In particular, the privacy guarantee does not depend on the number of clients N.

## 4.3 Error Bound for Covariance Estimation

Write $\mathbf { K } = \mathbf { X } \mathbf { X } ^ { \top }$ and $\widetilde { \mathbf { K } } _ { k } = \mathbf { V } _ { k } \widetilde { \mathbf { Y } } _ { k } \mathbf { V } _ { k } ^ { \top }$ for the quantities of Section 4. The error $\mathrm { E r r } _ { k } \ = \ { \bf K } - \widetilde { \bf K } _ { k }$ decomposes into two parts: a subspace error, arising from the approximate basis $\mathbf { V } _ { k }$ of the first stage, and a noise error, arising from the symmetric Gaussian perturbation Z in the second stage. The basis error is bounded in Theorem C.4 and is governed by the spectral gap $\lambda _ { k } - \lambda _ { k + 1 }$ of the true covariance. The second error is essentially the spectral norm of a $k \times k$ symmetric Gaussian matrix (Lemma C.2) and it scales linearly in $\sigma _ { 2 } \sqrt { k }$ rather than $\sigma _ { 2 } { \sqrt { p } } .$ Combining the two bounds via the triangle inequality and Lemma C.5 gives the following.

Theorem 6. Let $\mathbf { K } = \mathbf { X } \mathbf { X } ^ { \top } \in \mathbb { R } ^ { p \times p }$ be the true covariance with eigenvalues $\lambda _ { 1 } \geq \cdot \cdot \cdot \geq \lambda _ { p } \geq 0$ , let $C$ be the public bound on $\| \mathbf { x } _ { i } \| _ { 2 }$ , and let $\mathbf { Q } \in \mathbb { R } ^ { p \times k }$ be the orthonormal basis returned by the basis computation stage (Algorithm 1 in the appendix) run for a public number of iterations T with i.i.d. noise entries distributed as $\mathcal N ( 0 , \sigma _ { 1 } ^ { 2 } )$ . Assume the second stage releases $\boldsymbol { \widetilde { \mathbf { Y } } } _ { k } = \boldsymbol { \mathbf { \check { V } } } _ { k } ^ { \top } \boldsymbol { \mathbf { \check { K } } } \boldsymbol { \mathbf { \check { V } } } _ { k } + \boldsymbol { \mathbf { Z } }$ , where $\mathbf { Z } \in \mathbb { R } ^ { k \times k }$ is a symmetric Gaussian matrix whose upper-triangular entries (including the diagonal) are i.i.d. $\mathcal { N } ( 0 , \sigma _ { 2 } ^ { 2 } )$ and whose lower-triangular entries are copied from the upper triangle. Define the total error Err $\mathbf { \Lambda } _ { k } : = \mathbf { K } - \mathbf { V } _ { k } \widetilde { \mathbf { Y } } _ { k } \mathbf { V } _ { k } ^ { \top }$

(i) Let $\varepsilon _ { 1 } + \varepsilon _ { 2 } = \varepsilon$ and $\delta _ { 1 } + \delta _ { 2 } = \delta$ with $0 < \varepsilon _ { i } \leq 2 \log ( 1 / \delta _ { i } ) , i = 1 , 2 ( e . g . , \varepsilon _ { i } = \varepsilon / 2 , \delta _ { i } = \delta / 2 )$ , and let $\sigma _ { 1 } = \varepsilon _ { 1 } ^ { - 1 } C ^ { 2 } \sqrt { 4 T \log ( 1 / \delta _ { 1 } ) }$ and $\sigma _ { 2 } = \varepsilon _ { 2 } ^ { - 1 } C ^ { 2 } \sqrt { 4 \log ( 1 / \delta _ { 2 } ) }$ . Then the release $( \mathbf { V } _ { k } , \tilde { \mathbf { Y } } _ { k } )$ is $( \varepsilon , \delta ) \ – D P$

(ii) Fix a constant $\tau \geq 1$ and assume $T \geq L$ for the number of iterations $\begin{array} { r } { L = O \left( \frac { \lambda _ { k } } { \lambda _ { k } - \lambda _ { k + 1 } } \log p \right) } \end{array}$ required in Theorem C.4. Then, for any $\rho , \nu \in ( 0 , 1 )$ , with probability at least $1 - \nu - \rho - p _ { 0 }$ , where $p _ { 0 } =$ $\tau ^ { - \Omega ( 1 ) } + e ^ { - \Omega ( p ) }$ , we have that

$$
\| \mathrm { E r r } _ { k } \| _ { 2 } \ \leq \ \sqrt { 2 } \big ( \lambda _ { k + 1 } + \lambda _ { 1 } \zeta \big ) \ + \ \sigma _ { 2 } \Big ( 2 \sqrt { k } + 2 \sqrt { \log ( 2 / \rho ) } \Big ) ,
$$

where

$$
\zeta = \operatorname* { m i n } \left\{ 1 , O \left( \frac { \sigma _ { 1 } \sqrt { p } \big ( 1 + \sqrt { \log ( T / \nu ) } \big ) } { \lambda _ { k } - \lambda _ { k + 1 } } \cdot k \right) \right\} .
$$

This result separates into contributions of the two phases, which may guide the privacy budget allocation: one chooses $( \varepsilon _ { 1 } , \varepsilon _ { 2 } )$ subject to $\varepsilon _ { 1 } + \varepsilon _ { 2 } = \varepsilon$ so as to balance the two error sources. A natural approach to minimize the total error is to equate the two terms. This puts a larger share of the privacy budget into $\varepsilon _ { 1 }$ when the spectral gap $\lambda _ { k } - \lambda _ { k + 1 }$ is small.

The two terms also show the dimensional cost of privacy. The first term has the ${ \sqrt { p } } -$ factor from the norm bound on the $p \times k$ Gaussian noise injected at each iteration of noisy power method. The Analyze Gauss baseline applied directly to the $p \times p$ covariance incurs an error of order $O ( \sigma _ { \scriptstyle { \sqrt { p } } } )$ (Dwork et al., 2014). Our bound replaces this by a gap-dependent subspace term and a noise term scaling with $\sqrt { k }$ . Its main benefit is that it is achieved with $O ( T p k + k ^ { 2 } )$ rather than $O ( p ^ { 2 } )$ communication cost in a federated setting

## 5 Private Rank Estimation

We also propose a lightweight federated initialization procedure that produces a DP estimate of an effective rank of the global covariance matrix $\begin{array} { r } { \mathbf { X } \mathbf { X } ^ { \top } = \sum _ { i } \mathbf { X } _ { i } \mathbf { X } _ { i } ^ { \top } } \end{array}$ . First, we compress $\mathbf { \mathbf { X X } ^ { \top } }$ via a shared Gaussian random projection to an $r \times r$ matrix that clients can communicate cheaply, clip the 2-norm of the projected samples, and release a noisy version of this projected matrix. The rank is then chosen by thresholding its eigenvalues.

We assume each client holds local data $\mathbf { X } _ { i } ~ \in ~ \mathbb { R } ^ { p \times n _ { i } }$ whose columns have bounded l2-norm $\| \mathbf { x } \| _ { 2 } ~ \leq$ $C ,$ and that all clients share a common public seed. For a clipping bound $\bar { C } > 0 \mathrm { ( e . g . } \bar { C } = C )$ , let $\mathrm { c l i p } _ { \bar { C } } ( \mathbf { y } ) = \mathbf { y }$ · min $\{ 1 , { \bar { C } } / \| \mathbf { y } \| _ { 2 } \}$ for a vector y. For a matrix argument, clipō is applied column-wise, i.e., $\mathrm { c l i p } _ { \bar { C } } ( { \bf Y } ) = \big ( \mathrm { c l i p } _ { \bar { C } } ( { \bf y } _ { 1 } ) , \dots , \mathrm { c l i p } _ { \bar { C } } ( { \bf y } _ { m } ) \big )$ for $\mathbf { Y } = ( \mathbf { y } _ { 1 } , \ldots , \mathbf { y } _ { m } )$ , so that every column of the result has $\ell _ { 2 } { \mathrm { - n o r m } }$ at most $\bar { C }$ . The procedure is given as follows:

1. The server distributes a public seed to all clients.

2. Each client uses the seed to generate the projection matrix $\mathbf { G } \in \mathbb { R } ^ { p \times r }$ with $G _ { i j } \sim \mathcal { N } ( 0 , 1 / r )$ i.i.d.

3. Each client projects and clips (column-wise) its local data, $\widetilde { \mathbf { X } } _ { i } = \mathrm { d i p } _ { \bar { C } } ( \mathbf { G } ^ { \top } \mathbf { X } _ { i } ) \in \mathbb { R } ^ { r \times n _ { i } }$ , and sends $\widetilde { \mathbf { X } } _ { i } \widetilde { \mathbf { X } } _ { i } ^ { \top } \in \mathbb { R } ^ { r \times \overline { { r } } }$ to the server.

4. Via secure aggregation of the clients’noisy shares (see below), the server obtains $\begin{array} { r } { \sum _ { i } \widetilde { \mathbf { X } } _ { i } \widetilde { \mathbf { X } } _ { i } ^ { \top } + \sigma \mathbf { E } = } \end{array}$ $\widetilde { \mathbf { X } } \widetilde { \mathbf { X } } ^ { \top } + \sigma \mathbf { E } , \sigma > 0$ , where E is symmetric with $\mathrm { i . i . d . } \ N ( 0 , 1 )$ entries on and above the diagonal, and estimates the rank from the eigenvalues of $\widetilde { \mathbf { C } } = \widetilde { \mathbf { X } } \widetilde { \mathbf { X } } ^ { \top } + \widetilde { \sigma } \mathbf { E }$

In case of an honest-but-curious server and secure aggregation with distributed Gaussian noise we obtain the centralized DP guarantees. Each of the N clients independently samples $\mathbf { E } _ { i }$ with the same distribution and adds the share $\textstyle { \frac { \sigma } { \sqrt { N } } } \mathbf { E } _ { i }$ to its local upload. The per-client communication of this initialization stage is only $O ( r ^ { 2 } )$ scalars per round. We next establish the DP guarantee of the released matrix $\widetilde { \mathbf { C } }$

Lemma 7. The DP guarantees of the mechanism

$$
\mathcal { M } ( \mathbf { X } ) = \widetilde { \mathbf { X } } \widetilde { \mathbf { X } } ^ { \top } + \sigma \mathbf { E } , \qquad \widetilde { \mathbf { X } } = \mathrm { c l i p } _ { \bar { C } } ( \mathbf { G } ^ { \top } \mathbf { X } )
$$

are those of the Gaussian mechanism with sensitivity $\bar { C } ^ { 2 }$ and noise level σ. In particular, M is $\mu { - } G D P$ with $\mu = \hat { C } ^ { 2 } / \sigma$

Compared to releasing the unclipped projection $\mathbf { G } ^ { \top } \mathbf { X } \mathbf { X } ^ { \top } \mathbf { G }$ , whose sensitivity can only be bounded with high probability over G, clipping gives a smaller, deterministic sensitivity and a clean GDP privacy guarantee. Since $\lVert \mathbf G ^ { \top } \mathbf { x } \rVert _ { 2 } ^ { 2 }$ concentrates around $\| \mathbf { x } \| _ { 2 } ^ { 2 }$ , the clipping perturbs the projected samples only mildly, and it can only decrease the eigenvalues, as $\tilde { \mathbf { X } } \tilde { \mathbf { X } } ^ { \top } \preceq \mathbf { G } ^ { \top } \mathbf { X } \mathbf { X } ^ { \top } \mathbf { G }$ . The following result shows that in case we select the rank as the number of eigenvalues of $\widetilde { \mathbf { C } }$ exceeding the noise level, then with h.p. the selected rank is sandwiched between the numbers of eigenvalues of the clipped projected covariance $\widetilde { \mathbf { X } } \widetilde { \mathbf { X } } ^ { \top }$ above $2 \sigma b _ { \eta }$ . and above zero; in particular, we do not overestimate the rank

Lemma 8. Let $\rho \in ( 0 , 1 )$ and $b _ { r } ~ = ~ 2 \sqrt { r } + 2 \sqrt { \log ( 2 / \rho ) }$ , and let $\widehat { k } = \big | \{ i : \lambda _ { i } ( \widetilde { \mathbf { C } } ) > \sigma b _ { r } \} \big |$ . Then, with probability at least $1 - \rho _ { ; }$

$$
\left| \{ i : \lambda _ { i } ( \widetilde { \mathbf { X } } \widetilde { \mathbf { X } } ^ { \top } ) > 2 \sigma b _ { r } \} \right| \ \leq \ \widehat { k } \ \leq \ \left| \{ i : \lambda _ { i } ( \widetilde { \mathbf { X } } \widetilde { \mathbf { X } } ^ { \top } ) > 0 \} \right| \ \leq \ \mathrm { r a n k } ( \mathbf { X } \mathbf { X } ^ { \top } ) .
$$

The accuracy of the proposed algorithm is illustrated empirically in Section 7.

## 6 Convergence Analysis

We next analyze when the low-rank estimate $( \mathbf { V } , { \widetilde { \mathbf { Y } } } )$ makes federated DP-FC preferable to federated DP-GD. To this end, we analyze linear regression which reveals the benefits of preconditioning most clearly, and comment on GLMs at the end. In short: for the same privacy budget, rank-k preconditioning reaches the utility of DP-GD in a fraction of the rounds. Thereby, we show that the benefits of private secondorder optimization shown by Ganesh et al. (2023) can be retained with DP low-rank approximation of the covariance. We consider a fixed-point analysis with a weighted norm, similarly to the alternating direction method of multipliers analysis by Cyffers et al. (2023). As in Brown et al. (2024), we write the model state parameter obtained via DP iterations as a closed-form expression with a deterministic optimization term and a weighted sum of injected DP noise.

Setting. Throughout, $\lambda > 0$ is the regularization parameter. The values $\lambda _ { 1 } \geq \cdots \geq \lambda _ { p } \geq 0$ are the eigenvalues of $\begin{array} { r } { \mathbf { C } : = \frac { 1 } { n } \mathbf { K } = \frac { 1 } { n } \mathbf { X } \mathbf { X } ^ { \top } = \mathbf { U } \mathbf { A } \mathbf { U } ^ { \top } } \end{array}$ (in this section, Λ and $\lambda _ { i }$ thus refer to C rather than $\mathbf { K } ) , \mathbf { U } _ { k }$ holds the leading $k < p$ eigenvectors, and $\mathbf { C } _ { \lambda } : = \mathbf { C } + \lambda \mathbf { I }$ . Consider

$$
{ \mathcal { L } } ( \theta ) = { \frac { 1 } { 2 n } } \sum _ { ( x , y ) \in D } ( \theta ^ { \top } x - y ) ^ { 2 } + { \frac { \lambda } { 2 } } \left\| \theta \right\| _ { 2 } ^ { 2 }
$$

with minimizer $\theta _ { * }$ . Since $\mathcal { L }$ is quadratic with constant Hessian $\nabla ^ { 2 } \mathcal { L } = \mathbf { C } _ { \lambda }$ and $\nabla \mathcal { L } ( \theta _ { * } ) = 0$ , we have $\nabla \mathcal { L } ( \theta ) = \mathbf { C } _ { \lambda } ( \theta - \theta _ { * } )$ and

$$
\begin{array} { r } { { \mathcal L } ( \boldsymbol { \theta } ) - { \mathcal L } ( \boldsymbol { \theta } _ { * } ) = \frac { 1 } { 2 } \left\| \boldsymbol { \theta } - \boldsymbol { \theta } _ { * } \right\| _ { { \bf C } _ { \lambda } } ^ { 2 } , } \end{array}
$$

where for a symmetric positive definite (SPD) M we denote $\| \mathbf { v } \| _ { \mathbf { M } } ^ { 2 } : = \mathbf { v } ^ { \top } \mathbf { M } \mathbf { v }$ . For an SPD P we consider

$$
\theta _ { j + 1 } = \theta _ { j } - \eta \mathbf { P } ^ { - 1 } \big ( \nabla \mathcal { L } ( \theta _ { j } ) + \mathcal { E } _ { j } \big ) , \quad \mathcal { E } _ { j } \sim \mathcal { N } ( 0 , \sigma _ { g } ^ { 2 } \mathbf { I } ) .\tag{6.1}
$$

DP-GD iteration is given by $\mathbf { P } = \mathbf { I } .$ DP-FC with our estimate is given by

$$
\mathbf { P } = \widehat { \mathbf { G } } : = \mathbf { V } \widetilde { \mathbf { Y } } \mathbf { V } ^ { \top } + \lambda \mathbf { I } , \qquad \widetilde { \mathbf { Y } } = \mathbf { V } ^ { \top } \mathbf { C } \mathbf { V } + \mathbf { Z } ,\tag{6.2}
$$

where $\mathbf { Z }$ is symmetric with i.i.d. $\mathcal { N } ( 0 , \sigma _ { 2 } ^ { 2 } )$ upper-triangular entries $( \sigma _ { 2 } = C ^ { 2 } \sigma ^ { \prime } / n$ with C the record-norm bound; the factor $1 / n$ is due to the normalization of $\mathbf { C } ,$ cf. Algorithm 3). If $\Delta _ { g }$ is the $L _ { 2 ^ { - } }$ -sensitivity of the clipped averaged gradient (under the add/remove relation, $\Delta _ { g } = c / n$ for per-sample gradients clipped to norm c and normalization by the public $n )$ , then T rounds are $\mu _ { g ^ { - } } \mathrm { G D P }$ with $\mu _ { g } = \sqrt { T } \Delta _ { g } / \sigma _ { g }$ (Lemma 2). The use of the preconditioner in the iteration is post-processing. For simplicity, we assume the gradients stay bounded in 2-norm, i.e., we ignore the clipping bias. We use the two condition numbers

$$
\kappa _ { 0 } : = \frac { \lambda _ { 1 } + \lambda } { \lambda _ { p } + \lambda } , \qquad \kappa _ { k } : = 1 + \frac { \lambda _ { k + 1 } } { \lambda } .\tag{6.3}
$$

For symmetric $\mathbf { C } _ { \lambda }$ and $\mathbf G \succ 0$ , the generalized eigenvalues of $( \mathbf { C } _ { \lambda } , \mathbf { G } )$ are the scalars $\mu$ with $\mathbf { C } _ { \lambda } \mathbf { x } = \mu \mathbf { G } \mathbf { x }$ for some 101 some $\mathbf { x } \neq \mathbf { 0 }$ , i.e., the eigenvalues of , 1.e., une eigenvaiues or $\mathbf { G } ^ { - 1 / 2 } \mathbf { C } _ { \lambda } \mathbf { G } ^ { - 1 / 2 }$

For the oracle preconditioner $\mathbf { G } _ { \star } : = \mathbf { U } _ { k } \mathbf { A } _ { k } \mathbf { U } _ { k } ^ { \top } + \lambda \mathbf { I } \ ( \mathrm { i . e . , ~ } \mathbf { V } = \mathbf { U } _ { k } , \mathbf { Z } = \mathbf { 0 } )$ , the generalized eigenvalues of $( \mathbf { C } _ { \lambda } , \mathbf { G } _ { \star } )$ are 1 for $i \leq k$ and $1 + \lambda _ { i } / \lambda$ for $i > k$ , and hence lie in $[ 1 , \kappa _ { k } ]$ (Lemma D.1, Appendix D). Moreover, whenever $\widetilde { \mathbf { Y } } + \lambda \mathbf { I } _ { k } \succ 0$

$$
\begin{array} { r } { \widehat { \mathbf { G } } ^ { - 1 } = \mathbf { V } \big ( \widetilde { \mathbf { Y } } + \lambda \mathbf { I } _ { k } \big ) ^ { - 1 } \mathbf { V } ^ { \top } + \frac { 1 } { \lambda } \big ( \mathbf { I } - \mathbf { V } \mathbf { V } ^ { \top } \big ) , } \end{array}\tag{6.4}
$$

so a preconditioned step is an approximate Newton step within the span of V and a gradient step with step size $\eta / \lambda$ on its orthogonal complement.

We state the utility bounds in terms of the factor γ, as defined and bounded in the following Lemma.

Lemma 9. Let

$$
\begin{array} { r } { \gamma : = \left. \mathbf { G } _ { \star } ^ { - 1 / 2 } \widehat { \mathbf { G } } \mathbf { G } _ { \star } ^ { - 1 / 2 } - \mathbf { I } \right. _ { 2 } . } \end{array}
$$

Then

$$
( 1 - \gamma ) \mathbf { G } _ { \star } \preccurlyeq \widehat { \mathbf { G } } \preccurlyeq ( 1 + \gamma ) \mathbf { G } _ { \star } .
$$

$I f \gamma < 1$ , the generalized eigenvalues of $( \mathbf { C } _ { \lambda } , \widehat { \mathbf { G } } )$ lie in $\textstyle \left[ { \frac { 1 } { 1 + \gamma } } , { \frac { \kappa _ { k } } { 1 - \gamma } } \right]$ , and $( 1 - \gamma ) \mathrm { t r } ( \widehat { \mathbf { G } } ^ { - 1 } ) \leq \mathrm { t r } ( \mathbf { G } _ { \star } ^ { - 1 } )$ . Moreover, with $\boldsymbol { \zeta } : = \left\| ( \mathbf { I } - \mathbf { V } \mathbf { V } ^ { \top } ) \mathbf { U } _ { k } \right\| _ { 2 } < 1$ (bounded by Theorem $C . 4 ,$ with the eigenvalues $n \lambda _ { i } \mathrm { ~ } o f \mathbf { K } )$

$$
\gamma \leq \frac { 2 \lambda _ { 1 } \zeta + \| \mathbf { Z } \| _ { 2 } } { \lambda } ,
$$

and with probability at least $1 - \rho , \| \mathbf { Z } \| _ { 2 } \leq \sigma _ { 2 } \big ( 2 \sqrt { k } + 2 \sqrt { \log ( 2 / \rho ) } \big )$

We write $\begin{array} { r } { \widehat { \kappa } _ { k } : = \kappa _ { k } \frac { 1 + \gamma } { 1 - \gamma } } \end{array}$ for the effective condition number of the private preconditioner.

Theorem 10 (Utility of preconditioned DP-GD). Let $\mathbf { P } \succ 0$ be independent of $( \mathcal { E } _ { j } )$ , let the generalized eigenvalues of $( \mathbf { C } _ { \lambda } , \mathbf { P } )$ lie in [a, b], and let $0 < \eta \leq 1 / b$ . Then

$$
\mathbb { E } \mathcal { L } ( \theta _ { T } ) - \mathcal { L } ( \theta _ { * } ) \le ( 1 - \eta a ) ^ { 2 T } \bigl ( \mathcal { L } ( \theta _ { 0 } ) - \mathcal { L } ( \theta _ { * } ) \bigr ) + \frac { \eta \sigma _ { g } ^ { 2 } } { 2 } \operatorname { t r } ( \mathbf { P } ^ { - 1 } ) .\tag{6.5}
$$

Take $\eta = 1 / b$ . DP-GD has $a = \lambda _ { p } + \lambda , b = \lambda _ { 1 } + \lambda$ and $\operatorname { t r } ( \mathbf { I } ) = p .$ This gives rate $1 - 1 / \kappa _ { 0 }$ and per-round noise coefficient $p / ( \lambda _ { 1 } + \lambda )$ . By Lemma 9, our preconditioner gives rate $1 - 1 / \widehat { \kappa } _ { k }$ and noise coefficient at most t $\mathrm { r } ( \mathbf G _ { \star } ^ { - 1 } ) / \kappa _ { k } \leq p / ( \lambda + \lambda _ { k + 1 } )$ . The preconditioned method takes larger steps and so injects more noise per round, but it needs proportionally fewer rounds. The next corollary studies these trade-offs and gives an explicit comparison of the low-rank preconditioned DP-GD and DP-GD.

Corollary 11. Let $\gamma < 1$ and let the $T$ gradient rounds be $\mu _ { g } – G D P$ in total, i.e. $\sigma _ { g } ^ { 2 } = T \Delta _ { g } ^ { 2 } / \mu _ { g } ^ { 2 }$ , and assume $R _ { 0 } \geq \Delta _ { g } ^ { 2 } N / ( 4 \mu _ { g } ^ { 2 } )$ for $N \in \{ N _ { \mathrm { G D } } , N _ { k } \}$ . Write $R _ { 0 } : = \mathcal { L } ( \theta _ { 0 } ) - \mathcal { L } ( \theta _ { * } )$ ，

$$
N _ { \mathrm { G D } } : = \frac { p } { \lambda _ { p } + \lambda } , \quad N _ { k } : = \frac { 1 + \gamma } { 1 - \gamma } \Bigl [ \sum _ { i < k } \frac { 1 } { \lambda _ { i } + \lambda } + \frac { p - k } { \lambda } \Bigr ] , \quad \Phi ( N ) : = \frac { \Delta _ { g } ^ { 2 } N } { 4 \mu _ { g } ^ { 2 } } \Bigl ( 3 + \log \frac { 4 \mu _ { g } ^ { 2 } R _ { 0 } } { \Delta _ { g } ^ { 2 } N } \Bigr ) .
$$

(i) With $\eta ~ = ~ 1 / b , ~ i . e . , ~ \eta ~ = ~ 1 / ( \lambda _ { 1 } + \lambda )$ for DP-GD and $\eta ~ = ~ ( 1 - \gamma ) / \kappa _ { k }$ for DP-FC, DP-GD after $\begin{array} { r } { T _ { \mathrm { G D } } = \lceil \frac { \kappa _ { 0 } } { 2 } \log \frac { 4 \mu _ { g } ^ { 2 } R _ { 0 } } { \Delta _ { q } ^ { 2 } N _ { \mathrm { G D } } } \rceil } \end{array}$ rounds, and DP-FC with $\widehat { \mathbf { G } }$ after $\begin{array} { r } { T _ { k } = \lceil \frac { \widehat \kappa _ { k } } { 2 } \log \frac { 4 \mu _ { g } ^ { 2 } R _ { 0 } } { \Delta _ { g } ^ { 2 } N _ { k } } \rceil } \end{array}$ rounds, satisfy E ${ \mathcal { L } } ( \theta _ { T } ) -$ $ { \mathcal { L } } ( \theta _ { * } ) \leq \Phi ( N _ { \mathrm { G D } } ) ^ { \smash { \upsilon } } )$ and $\leq \Phi ( N _ { k } )$ , respectively.

$$
\begin{array} { r } { ( i i ) ~ N _ { k } \leq \frac { 1 + \gamma } { 1 - \gamma } \big ( 1 + \frac { \lambda _ { p } } { \lambda } \big ) N _ { \mathrm { G D } } . ~ F o r ~ \gamma = \lambda _ { p } = 0 , ~ N _ { k } = N _ { \mathrm { G D } } - \sum _ { i \leq k } \frac { \lambda _ { i } } { \lambda ( \lambda _ { i } + \lambda ) } . } \end{array}
$$

When neglecting the log terms, we can read Corollary 11 such that at equal privacy, the low-rank preconditioned method matches the privacy-utility bound of DP-GD up to the factor $\frac { 1 + \gamma } { 1 - \gamma } ( 1 + \lambda _ { p } / \lambda )$ , and slightly improves on it when $\gamma$ and $\lambda _ { p } / \lambda$ are small. It does so in a fraction $\approx \widehat { \kappa } _ { k } / \kappa _ { 0 }$ of the rounds, where

$$
\frac { \kappa _ { 0 } } { \kappa _ { k } } = \frac { \lambda _ { 1 } + \lambda } { \lambda _ { p } + \lambda } \cdot \frac { \lambda } { \lambda + \lambda _ { k + 1 } } .
$$

The gain is therefore essentially in rounds, which is the expensive unit in cross-silo FL (Section 7.3). We remark that this simplified analysis makes three shortcuts. First, "equal privacy" refers to an equal GDPbudget of the gradient descent phase $\mu _ { g } .$ our method and DP-FC also spend the budget $\mu _ { \mathrm { P C A } }$ on the preconditioner, so under a fixed total budget $\mu$ they have $\mu _ { g } ^ { 2 } = \mu ^ { 2 } - \mu _ { \mathrm { P C A } } ^ { 2 } ,$ whereas DP-GD uses $\mu _ { g } = \mu .$ Second, $T _ { k }$ excludes the $T + 1$ rounds used to construct $\widehat { \mathbf { G } }$ . Third, $\gamma$ and $\lambda _ { k + 1 }$ are unknown in practice, so $\eta$ is tuned. The conclusion of gaining similar privacy-utility trade-off as DP-GD in fewer training rounds also matches with the conclusion made by Ganesh et al. (2023) on the benefits of private second-order optimization.

Extension to GLMs. For a convex GLM loss, the mean value theorem gives $\nabla \mathcal { L } ( \theta _ { j } ) = ( \bar { \bf H } _ { j } + \lambda { \bf I } ) ( \theta _ { j } - \theta _ { * } )$ with $\begin{array} { r } { \bar { \bf H } _ { j } = \frac { 1 } { n } \sum \ell ^ { \prime \prime } ( \xi _ { x } ) x x ^ { \top } } \end{array}$ . Suppose that, after rescaling, $\ell ^ { \prime \prime } \in [ c , 1 ]$ along the iterate path, as holds for the logistic loss with bounded logits. Then $c \mathbf { C } + \lambda \mathbf { I } \preccurlyeq \bar { \mathbf { H } } _ { j } + \lambda \mathbf { I } \preccurlyeq \mathbf { C } _ { \lambda }$ , so the generalized eigenvalues of $( \bar { \mathbf { H } } _ { j } + \lambda \mathbf { I } , \widehat { \mathbf { G } } )$ lie in $[ c / ( 1 + \gamma ) , \kappa _ { \boldsymbol { k } } / ( 1 - \gamma ) ]$ . The per-step contraction in the $\widehat { \mathbf { G } } _ { - \mathrm { n o r m } }$ is then $1 - c / \widehat { \kappa } _ { k } ;$ the analogous rate for DP-GD is $1 - c / \kappa _ { 0 }$ . Since $\bar { \mathbf { H } } _ { j }$ varies with $j ,$ the noise term is bounded with the fixed-point argument of Cyffers et al. (2023) in the Ġ-norm (Appendix E). The round comparison of Corollary 11 carries over with $\kappa _ { 0 } , \widehat { \kappa } _ { k }$ replaced by $\kappa _ { 0 } / c , \widehat { \kappa } _ { k } / c$ . The noise term of the GLM bound (Appendix E) is, however, looser than that of Theorem 10 by a factor of order $\widehat { \kappa } _ { k } / c$ , so Corollary 11(ii) does not carry over directly. Finally, we remark that the analysis is for scalar-output GLMs and directly generalizable for the multiclass softmax loss.

## 7 Experiments

We evaluate our method on standard benchmark datasets and compare it against the relevant baselines. In all figures, results are means over 5 independent runs, and error bars denote ±1 empirical standard deviation.

## 7.1 Illustration of Rank Estimation

Figure 1 illustrates our DP rank estimation procedure on MNIST (LeCun et al., 1998) $( p = 7 8 4 )$ at the privacy level $( \varepsilon , \delta ) = ( 2 . 0 , 1 0 ^ { - 5 } )$ , with the noise scale σ calibrated according to Lemma 7. We plot the square roots of the eigenvalues of the noisy projected $r \times r$ matrix $\widetilde { \mathbf { C } }$ for three projection dimensions $r \in \{ 2 0 , 5 0 , 8 0 \}$ 2 together with the true singular values of X for reference. At all three values of $r ,$ the dominant part of the spectrum is recovered accurately.

## 7.2 DP-PCA Comparison

We compare the two-stage algorithm of Section 4 (basis computation followed by the projected covariance release, with k fixed) to the central DP-PCA mechanism Analyze Gauss (Dwork et al., 2014). Figure 2 reports the relative Frobenius error $\left\| \mathbf { K } _ { k } - \widetilde { \mathbf { K } } _ { k } \right\| _ { F } / \left\| \mathbf { K } _ { k } \right\| _ { F }$ , where ${ \bf K } _ { k }$ is the best rank-k approximation of K in 2-norm, as a function of the privacy budget $\varepsilon ,$ for $k \in \{ 1 0 , 2 0 , 4 0 \}$ , on preprocessed UCI HAR features $( p = 5 6 1 )$ . We see that our method improves upon the $O ( p ^ { 2 } )$ baseline at every ε and k, with the largest gains at small ε, while requiring only $O ( T p k + k ^ { 2 } )$ communication. We use $T = 3$ power iterations. Appendix F shows the same behavior on SVHN $( p = 3 0 7 2 )$ , and also compares against the locally private federated method of Grammenos et al. (2020), whose error grows with the number of clients while ours does not depend on it.

MNIST Singular Values Estimation with Random Projections  
![](images/991d065f083070d510e80e84706407cce226a78ebd8d3c021b9f7a2febf62b71.jpg)

Figure 1: Singular values of the MNIST feature matrix $\mathbf { X } \in \mathbb { R } ^ { p \times n }$ versus the square roots of the eigenvalues of the DP estimate $\widetilde { \mathbf { C } }$ for projection dimensions $r \in \{ 2 0 , 5 0 , 8 0 \}$  
![](images/674a92654c0878218095f978064bb75dfa130b08c2a5ef4dec683d41c6f4d1a5.jpg)  
Figure 2: Comparison of our method against the central DP-PCA method of Dwork et al. (2014) on UCI HAR $( p = 5 6 1 )$ with target subspace dimension $k \in \{ 1 0 , 2 0 , 4 0 \}$ . The y-axis shows the relative Frobenius error of the rank-k covariance approximation, the x-axis ε for $\delta = 1 0 ^ { - 5 }$ . The noisy power method communicates $T p k + k ( k + 1 ) / 2$ scalars in total (from 16885 to 68140 for $T = 3 )$ , whereas Analyze Gauss communicates the upper triangle of the covariance, $p ( p + 1 ) / 2 = 1 5 7 6 4 1$

## 7.3 Preconditioned DP Fine-Tuning

We train a linear classification head under DP on frozen features: ResNet-34 features $\left( p = 5 1 2 \right)$ for Fashion-MNIST and ViT-B/16 features $( p = 7 6 8 )$ for CIFAR-10 and CIFAR-100. For the low-rank approximation of the covariance, we use a fixed rank $k = \lceil 1 . 2 m \rceil$ , where m is the number of classes, and $T = 2 ~ \mathrm { D P }$ power iterations. We try to keep both k and $T$ small so that the cost of constructing the preconditioner stays low relative to the gradient rounds. We fix k and do not run the rank-selection step of Section 5; running it would add the term $\bar { C } ^ { 4 } / \sigma ^ { 2 }$ to $\mu ^ { 2 }$ in the composition. Features are extracted with ImageNet 1k-pretrained models obtained from Torchvision (Marcel and Rodriguez, 2010). All methods satisfy the same total $( \varepsilon , \delta )$ DP guarantee; for the preconditioned methods, part of the budget is spent on the covariance estimate. We compare federated DP-FC equipped with our covariance estimate (Our Method) against vanilla federated DP-FC, which differs only in applying Analyze Gauss to the full covariance, and against the DP first-order baselines DP-GD, DP-GD with momentum, and DP-Adam. We omit DP-FedNew (Krouka et al., 2025): as noted by its authors, its record-level sensitivity bound is loose and the method is most competitive under user-level DP, and in preliminary record-level experiments it was not competitive with the second-order baselines above. The main text reports $\varepsilon = 2 ;$ results for $\varepsilon \in \{ 1 , 2 , 4 , 8 \}$ (CIFAR) and $\varepsilon \in \{ 0 . 5 , 1 , 2 , 4 \}$ (FashionMNIST), and for CIFAR-100, are given in Appendix G.

Communication accounting. We report accuracy with respect to two different measures. The number of rounds is the relevant measure whenever a synchronization round is the expensive unit: in crosssilo deployments, each round incurs orchestration and, under secure aggregation, protocol set-up costs. The number of total scalars communicated is a relevant measure when bandwidth is the constraint.

For a linear model on p-dimensional features and with m classes, the model has pm parameters, so each gradient descent round costs pm scalars in each direction for all methods, and thus $T _ { g }$ rounds cost $T _ { g } p m$ The methods differ only in the cost of constructing the preconditioner: the naive federated DP-FC uploads and broadcasts the full $p \times p$ covariance, with $p ^ { 2 }$ scalars in each direction, whereas our proposed method costs T pk scalars in each direction for the T power iterations and $k ^ { 2 }$ for the projected covariance Y. Sending only the upper triangles of the symmetric matrices would reduce these to $p ( p + 1 ) / 2$ and $k ( k + 1 ) / 2 ;$ our plots count the full matrices.

Left sides of Figures 3 and 4 report accuracy as a function of communication rounds. $\mathrm { A t } ~ \varepsilon = 2$ , both preconditioned methods converge faster, and to higher accuracy, than all first-order baselines, showing the benefit of second-order information. Moreover, the low-rank preconditioner obtained by our communicationefficient DP-PCA closely matches the convergence of vanilla DP-FC while avoiding the transmission of the full $p \times p$ covariance.

Right sides of Figures 3 and 4 measure performance against total communication. Here the benefit of the proposed covariance estimate becomes clear: because the one-off cost scales as $O ( T p k + k ^ { 2 } )$ rather than $O ( p ^ { 2 } )$ , our method reaches the final accuracy of vanilla DP-FC at a fraction of its total communication, and achieves the best accuracy-communication trade-off among all methods at this privacy level.

At the smallest budgets (FashionMNIST. $\varepsilon = 0 . 5 ; \mathrm { C I F A R - 1 0 0 , } \varepsilon = 1 ; \mathrm { A p p e n d i x ~ G } )$ , first-order methods are competitive or better, as noise in the estimated subspace outweighs the curvature gain. In this regime, vanilla DP-FC also degrades after about 20 rounds on CIFAR-100, consistent with its $p \times p$ noise making $\widetilde { \mathbf { K } } + \lambda \mathbf { I }$ ill-conditioned; our $k \times k$ noise does not show this.

![](images/3a218a442c4f98dafef65ed8a997d3fb62bba21feeb9b6c1739ef6f3ba334acb.jpg)

![](images/dd3d4ba32b0709a32db7e548adcd25dcecec26ee0b2284b193009bbd068de4e9.jpg)  
Figure 3: CIFAR-10 classification with frozen ViT-B/16 features for $\varepsilon = 2 ~ ( \delta = 1 0 ^ { - 5 } )$ . Left: classification accuracy as a function of the total number of communication rounds; each point corresponds to independent runs with that number of rounds. Our DP low-rank preconditioner achieves accuracy comparable to that of vanilla DP-FC. Right: classification accuracy as a function of total communication (number of transmitted scalars) for $\varepsilon = 2 ~ ( \delta = 1 0 ^ { - 5 } )$ . Our method achieves the best accuracy-communication trade-off across the entire communication range.

## 8 Conclusions

We presented the first federated DP low-rank covariance estimation framework that simultaneously achieves $O ( p k )$ per-round communication and benefits from summation protocols, closing a gap left open by prior work. Our three-stage procedure, DP rank selection via a shared random projection, a federated DP power method for the leading k left singular vectors, and an Analyze Gauss release on the projected covariance decomposes into per-client sums and is therefore end-to-end compatible with secure aggregation. As a preconditioner, the resulting estimate gives an approximate Newton step, and as a consequence our proposed optimizer needs only a fraction of the rounds of DP-GD with similar communication. Empirically, our DP-PCA method matches the centralized $O ( p ^ { 2 } )$ Analyze Gauss baseline at a fraction of the communication cost. In federated DP optimization of GLMs, our optimization method matches vanilla DP-FC in rounds and gives the best accuracy-communication trade-off among all compared methods, except at the smallest privacy budgets.

Extending the framework from the full-batch setting to streaming or random mini-batch sampling is a natural next step, as is integrating covariance preconditioning into more general private training pipelines,

![](images/90e49202b24e4a5193ea713b13ce0c04a1a59745a16789182378df2be564c1c7.jpg)

![](images/e6e795b07ea6e6ac3e7adb9e11ff3e31a3c6d512455bba6f5bec32fc99666f90.jpg)  
Figure 4: FashionMNIST with ResNet-34 features $\left( p = 5 1 2 \right)$ $\varepsilon = 2 ~ ( \delta = 1 0 ^ { - 5 } )$ . Left: accuracy vs. rounds. Right: accuracy vs. total communication. Setup as in Figure 3.

e.g., for multilayer neural networks.

## References

D Anguita, A Ghio, L Oneto, X Parra, and JL Reyes-Ortiz. A public domain dataset for human activity recognition using smartphones. In 21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), pages 437–442. CIACO, 2013.

Maria-Florina Balcan, Simon Shaolei Du, Yining Wang, and Adams Wei Yu. An improved gap-dependency analysis of the noisy power method. In Conference on Learning Theory, pages 284–309. PMLR, 2016.

Borja Balle and Yu-Xiang Wang. Improving the Gaussian mechanism for differential privacy: Analytical calibration and optimal denoising. In International Conference on Machine Learning, pages 394–403. PMLR, 2018.

Gavin Brown, Krishnamurthy (Dj) Dvijotham, Georgina Evans, Daogao Liu, Adam Smith, and Abhradeep Thakurta. Private gradient descent for linear regression: tighter error bounds and instance-specific uncertainty estimation. In Proceedings of the 41st International Conference on Machine Learning, 2024

Mark Bun and Thomas Steinke. Concentrated differential privacy: Simplifications, extensions, and lower bounds. In Theory of Cryptography Conference, pages 635–658. Springer, 2016.

T Tony Cai, Dong Xia, and Mengyue Zha. Optimal differentially private pca and estimation for spiked covariance matrices. arXiv preprint arXiv:2401.03820, 2024.

Andrew Campbell, Anna Scaglione, and Sean Peisert. Decentralized differentially private power method. arXiv preprint arXiv:2507.22849, 2025.

Kamalika Chaudhuri, Anand Sarwate, and Kaushik Sinha. Near-optimal differentially private principal components. Advances in neural information processing systems, 25, 2012.

Edwige Cyffers, Aurélien Bellet, and Debabrota Basu. From noisy fixed-point iterations to private ADMM for centralized and federated learning. In International Conference on Machine Learning, pages 6683–6711. PMLR, 2023.

Kenneth R Davidson and Stanislaw J Szarek. Local operator theory, random matrices and Banach spaces. In Handbook of the geometry of Banach spaces, volume 1, pages 317–366. Elsevier, 2001.

Jinshuo Dong, Aaron Roth, and Weijie J Su. Gaussian differential privacy. Journal of the Royal Statistical Society Series B, 84(1):3–37, 2022.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

Cynthia Dwork, Krishnaram Kenthapadi, Frank McSherry, Ilya Mironov, and Moni Naor. Our Data, Ourselves: Privacy Via Distributed Noise Generation. In Serge Vaudenay, editor, Advances in Cryptology - EUROCRYPT 2006, volume 4004, pages 486–503. Springer Berlin Heidelberg, Berlin, Heidelberg, 2006.

Cynthia Dwork, Kunal Talwar, Abhradeep Thakurta, and Li Zhang. Analyze Gauss: optimal bounds for privacy-preserving principal component analysis. In Proceedings of the forty-sixth annual ACM symposium on Theory of computing, pages 11–20, 2014.

Arun Ganesh, Mahdi Haghifam, Thomas Steinke, and Abhradeep Guha Thakurta. Faster differentially private convex optimization via second-order methods. Advances in Neural Information Processing Systems, 36:79426–79438, 2023.

Sivakanth Gopi, Yin Tat Lee, and Lukas Wutschitz. Numerical composition of differential privacy. Advances in Neural Information Processing Systems, 34:11631–11642, 2021.

Andreas Grammenos, Rodrigo Mendoza Smith, Jon Crowcroft, and Cecilia Mascolo. Federated principal component analysis. Advances in neural information processing systems, 33:6453–6464, 2020.

Xiao Guo, Xiang Li, Xiangyu Chang, Shusen Wang, and Zhihua Zhang. FedPower: privacy-preserving distributed eigenspace estimation. Machine Learning, 113(11):8427–8458, 2024.

Moritz Hardt and Eric Price. The noisy power method: A meta algorithm with applications. Advances in neural information processing systems, 27, 2014.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

Yiyun He, Thomas Strohmer, Roman Vershynin, and Yizhe Zhu. Differentially private low-dimensional synthetic data from high-dimensional datasets. Information and Inference: A Journal of the IMA, 14(1): iaae034, 2025.

Roger A. Horn and Charles R. Johnson. Matrix Analysis. Cambridge University Press, Cambridge, 2 edition, 2012.

Peter Kairouz, Ziyu Liu, and Thomas Steinke. The distributed discrete Gaussian mechanism for federated learning with secure aggregation. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, ICML 2021, volume 139 of Proceedings of Machine Learning Research, pages 5201–5212. PMLR, 2021a.

Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. Advances and open problems in federated learning. Foundations and TrendsR in Machine Learning, 14(1–2):1–210, 2021b.

Minwoo Kim and Sungkyu Jung. Robust and differentially private principal component analysis. Statistical Analysis and Data Mining: An ASA Data Science Journal, 18(6):e70053, 2025.

Alex Krizhevsky. Learning multiple layers of features from tiny images. 2009.

Mounssif Krouka, Antti Koskela, and Tejas D Kulkarni. Communication efficient differentially private federated learning using second order information. Proc. Priv. Enhancing Technol., 2025(1):584–612, 2025.

Jennifer R Kwapisz, Gary M Weiss, and Samuel A Moore. Activity recognition using cell phone accelerometers. ACM SigKDD Explorations Newsletter, 12(2):74–82, 2011.

Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

Jingyang Li, T Tony Cai, Dong Xia, and Anru R Zhang. Federated PCA and estimation for spiked covariance matrices: Optimal rates and efficient algorithm. arXiv preprint arXiv:2411.15660, 2024.

Sébastien Marcel and Yann Rodriguez. Torchvision the machine-vision package of torch. In Proceedings of the 18th ACM international conference on Multimedia, pages 1485–1488, 2010.

Harsh Mehta, Walid Krichene, Abhradeep Guha Thakurta, Alexey Kurakin, and Ashok Cutkosky. Differentially private image classification from features. Transactions on Machine Learning Research, 2023

Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Baolin Wu, and Andrew Y Ng. Reading digits in natural images with unsupervised feature learning. In NIPS workshop on deep learning and unsupervised feature learning, volume 2011, page 7. Granada, 2011.

Julien Nicolas, César Sabater, Mohamed Maouche, Sonia Ben Mokhtar, and Mark Coates. Differentially private and decentralized randomized power method. arXiv preprint arXiv:2411.01931, 2024.

Mark Rudelson and Roman Vershynin. Non-asymptotic theory of random matrices: extreme singular values. Proceedings of the International Congress of Mathematicians, 3:1576–1602, 2010.

Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In 2017 IEEE symposium on security and privacy (SP), pages 3–18. IEEE, 2017.

David M Sommer, Sebastian Meiser, and Esfandiar Mohammadi. Privacy loss classes: The central limit theorem in differential privacy. Proceedings on Privacy Enhancing Technologies, 2019(2):245–269, 2019

Gilbert W Stewart and Ji-guang Sun. Matrix perturbation theory. Academic Press, 1990.

Stacey Truex, Nathalie Baracaldo, Ali Anwar, Thomas Steinke, Heiko Ludwig, Rui Zhang, and Yi Zhou. A hybrid approach to privacy-preserving federated learning. In Proceedings of the 12th ACM workshop on artificial intelligence and security, pages 1–11, 2019.

Enayat Ullah, Christopher A. Choquette-Choo, Peter Kairouz, and Sewoong Oh. Private federated learning with autotuned compression. In International Conference on Machine Learning, ICML 2023, volume 202 of Proceedings of Machine Learning Research, pages 34668–34708. PMLR, 2023.

Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.

Yuqing Zhu, Jinshuo Dong, and Yu-Xiang Wang. Optimal accounting of differential privacy via characteristic function. In International Conference on Artificial Intelligence and Statistics, pages 4782–4817. PMLR, 2022.

## A Proof of Lemma 2

We give a proof of Lemma 2 based on the hockey-stick divergence and on the privacy loss random variable (PRV) framework.

## A.1 Hockey-Stick Divergence, Dominating Pairs, and PRVs

For $\alpha > 0$ , the hockey-stick divergence from a distribution P to a distribution $Q$ is

$$
H _ { \alpha } ( P \parallel Q ) = \int \left[ P ( t ) - \alpha \cdot Q ( t ) \right] _ { + } \mathrm { d } t ,
$$

where $[ t ] _ { + } = \operatorname* { m a x } \{ 0 , t \}$ . Tight $( \varepsilon , \delta ) – \mathrm { D P }$ guarantees can be characterized through this divergence: for $\varepsilon \geq 0$ the smallest δ such that M is (ε, δ)-DP equals

$$
\delta ( \varepsilon ) \ = \ \operatorname* { m a x } _ { D \simeq D ^ { \prime } } H _ { \mathrm { e } ^ { \varepsilon } } \big ( { \mathcal { M } } ( D ) \| { \mathcal { M } } ( D ^ { \prime } ) \big ) ,\tag{A.1}
$$

where the maximum is over neighboring datasets (Zhu et al., 2022).

For some mechanisms, such as the Gaussian mechanism or the Poisson subsampled Gaussian mechanism, we can bound $H _ { \mathrm { e } } \varepsilon$ uniformly by the divergence of a single, dataset-independent pair of distributions, a socalled dominating pair of distributions.

Definition A.1 (Zhu et al. 2022). A pair of distributions $( P , Q )$ dominates M if, for all neighboring datasets $D \simeq D ^ { \prime }$ and all $\alpha > 0$

$$
H _ { \alpha } \bigl ( \mathcal { M } ( D ) \parallel \mathcal { M } ( D ^ { \prime } ) \bigr ) \ \leq \ H _ { \alpha } ( P \parallel Q ) .
$$

Dominating pairs compose multiplicatively under adaptive composition.

Theorem A.2 (Zhu et al. 2022). $I f \left( P _ { i } , Q _ { i } \right)$ dominates $\mathcal { M } _ { i } f o r i = 1 , . . . , T$ , then $\textstyle \left( \prod _ { i } P _ { i } , \prod _ { i } Q _ { i } \right)$ dominates the adaptive composition $\mathcal { M } _ { T } \circ \cdots \circ \mathcal { M } _ { 1 }$

The divergence-based analysis of compositions can be made tractable via privacy loss random variables.

Definition A.3. Let P and Q be probability density functions with $P \ll Q$ . The privacy loss random variable $( P R V ) \ \omega _ { P / Q }$ associated with $( P , Q )$ is

$$
\omega _ { P / Q } = \log \frac { P ( t ) } { Q ( t ) } , \qquad t \sim P .
$$

Theorem A.4 (Gopi et al. 2021). For any $\varepsilon \geq 0$

$$
H _ { \mathrm { e } ^ { \varepsilon } } ( P \parallel Q ) = \underset { s \sim \omega _ { P / Q } } { \mathbb { E } } [ 1 - \mathrm { e } ^ { \varepsilon - s } ] _ { + } .\tag{A.2}
$$

Moreover, the PRV of a product pair $( P _ { 1 } \times \cdot \cdot \cdot \times P _ { T } , Q _ { 1 } \times \cdot \cdot \cdot \times Q _ { T } )$ equals the sum of the per-coordinate PRVs:

$$
\omega _ { \prod _ { i } P _ { i } / \prod _ { i } Q _ { i } } \stackrel { d } { = } \sum _ { i = 1 } ^ { T } \omega _ { P _ { i } / Q _ { i } } ,
$$

where the summands on the right-hand side are independent.

Combining (A.1), Theorem A.2 and Theorem A.4, in order to obtain $( \varepsilon , \delta ) – \mathrm { D P }$ bounds for an adaptive composition it suffices to (i) exhibit a dominating pair for each step, (ii) sum the corresponding PRVs, and (iii) evaluate the expectation in (A.2).

## A.2 The PRV of a Gaussian mechanism

Consider a Gaussian mechanism $\mathcal { M } ( D ) = \mathbf { f } ( D ) + \mathbf { Z }$ with $\mathbf { Z } \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } I _ { d } )$ and L2-sensitivity $\Delta .$ A tightly dominating pair is given by the one-dimensional Gaussians

$$
P = \mathcal { N } ( 0 , \sigma ^ { 2 } ) , \qquad Q = \mathcal { N } ( \Delta , \sigma ^ { 2 } ) ,\tag{A.3}
$$

and a direct computation (see, e.g., Sommer et al., 2019, Lemma 11) shows that the corresponding PRV is itself Gaussian:

$$
\omega _ { P / Q } \sim \mathcal { N } \bigg ( \frac { \Delta ^ { 2 } } { 2 \sigma ^ { 2 } } , \frac { \Delta ^ { 2 } } { \sigma ^ { 2 } } \bigg ) .\tag{A.4}
$$

Plugging (A.4) into (A.2) yields the analytical Gaussian bound of Balle and Wang (2018): for every $\varepsilon \geq 0$

$$
H _ { \mathrm { e } ^ { \varepsilon } } ( P \parallel Q ) = \Phi \left( - \frac { \varepsilon \sigma } { \Delta } + \frac { \Delta } { 2 \sigma } \right) - \mathrm { e } ^ { \varepsilon } \Phi \left( - \frac { \varepsilon \sigma } { \Delta } - \frac { \Delta } { 2 \sigma } \right) ,\tag{A.5}
$$

where Φ denotes the cumulative distribution function of the standard univariate Gaussian.

## A.3 Proof of Lemma 2

We are now ready to prove the lemma.

Proof of Lemma 2. For each $i \in [ T ]$ , let $\mathcal { M } _ { i } ( D ) = \mathbf { f } _ { i } ( D ) + \mathbf { Z } _ { i }$ with $\mathbf Z _ { i } \sim \mathcal N ( 0 , \sigma _ { i } ^ { 2 } I _ { d _ { i } } )$ and L2-sensitivity $\Delta _ { i }$ By (A.3), the one-dimensional Gaussians

$$
P _ { i } = \mathcal { N } ( 0 , \sigma _ { i } ^ { 2 } ) , \qquad Q _ { i } = \mathcal { N } ( \Delta _ { i } , \sigma _ { i } ^ { 2 } )
$$

form a (tightly) dominating pair for $\mathcal { M } _ { i }$ , and (A.4) gives

$$
\omega _ { P _ { i } / Q _ { i } } \sim \mathcal { N } \biggl ( \frac { \Delta _ { i } ^ { 2 } } { 2 \sigma _ { i } ^ { 2 } } , \frac { \Delta _ { i } ^ { 2 } } { \sigma _ { i } ^ { 2 } } \biggr ) .
$$

By Theorem A.2, $\textstyle \left( \prod _ { i } P _ { i } , \prod _ { i } Q _ { i } \right)$ dominates the adaptive composition $\mathcal { M } ^ { ( T ) } = \mathcal { M } _ { T } \circ \cdot \cdot \cdot \circ \mathcal { M } _ { 1 }$ . By Theorem $\mathrm { A . 4 } .$ the PRV of this product pair is the sum of independent Gaussians, hence Gaussian:

$$
\omega : = \sum _ { i = 1 } ^ { T } \omega _ { P _ { i } / Q _ { i } } \sim \mathcal { N } \left( \sum _ { i = 1 } ^ { T } \frac { \Delta _ { i } ^ { 2 } } { 2 \sigma _ { i } ^ { 2 } } , \sum _ { i = 1 } ^ { T } \frac { \Delta _ { i } ^ { 2 } } { \sigma _ { i } ^ { 2 } } \right) = \mathcal { N } \left( \frac { \widetilde { \Delta } ^ { 2 } } { 2 } , \widetilde { \Delta } ^ { 2 } \right) ,
$$

where we used the definition $\begin{array} { r } { \widetilde { \Delta } ^ { 2 } = \sum _ { i = 1 } ^ { T } \Delta _ { i } ^ { 2 } / \sigma _ { i } ^ { 2 } } \end{array}$ . The analytical expression for the privacy profile determined by the PRV $\begin{array} { r } { \mathcal { N } \Big ( \frac { \widetilde { \Delta } ^ { 2 } } { 2 } , \widetilde { \Delta } ^ { 2 } \Big ) } \end{array}$ is given in (Balle and Wang, 2018) from which the claim follows.

The following closed-form calibration, used in Theorems 6 and C.4, is a (conservative) consequence of Lemma 2.

Corollary A.5. Consider an adaptive composition of T Gaussian mechanisms, each with L2-sensitivity ∆ and noise standard deviation $\Delta \sigma$ . Let $\delta \in ( 0 , 1 ) , 0 < \varepsilon \leq 2 \log ( 1 / \delta )$ , and $\sigma = \varepsilon ^ { - 1 } \sqrt { 4 T \log ( 1 / \delta ) }$ . Then the composition is $( \varepsilon , \delta ) \mathopen { } \mathclose \bgroup   D P $

Proof. By (Bun and Steinke, 2016, Prop. 1.6 and Lemma 1.7), the composition is $\rho { \mathrm { - z C D P } }$ with $\rho ~ =$ $T / ( 2 \sigma ^ { 2 } ) = \varepsilon ^ { 2 } / ( 8 L )$ , where $L = \log ( 1 / \delta )$ . By (Bun and Steinke, 2016, Prop. 1.3), it is then $( \varepsilon ^ { \prime } , \delta ) – \mathrm { D P }$ with

$$
\varepsilon ^ { \prime } = \rho + 2 \sqrt { \rho L } = \frac { \varepsilon ^ { 2 } } { 8 L } + \frac { \varepsilon } { \sqrt { 2 } } \leq \varepsilon ,
$$

where the inequality holds for $\varepsilon \le 8 ( 1 - 1 / \sqrt { 2 } ) L$ , and in particular for $\varepsilon \leq 2 L$

## B Proof for Section 5

## B.1 Proof of Lemma 7

Proof. Under the add/remove relation, let $\mathbf { y } \in \mathbb { R } ^ { r }$ denote the clipped projection of the added or removed column, so that $\| \mathbf { y } \| _ { 2 } \leq { \bar { C } }$ . The released matrix $\widetilde { \mathbf { X } } \widetilde { \mathbf { X } } ^ { \top }$ changes by $\mathbf { y } \mathbf { y } ^ { \top }$ , and $\| \mathbf { y } \mathbf { y } ^ { \top } \| _ { F } = \| \mathbf { y } \| _ { 2 } ^ { 2 } \leq \bar { C } ^ { 2 }$ . The upper-triangular vectorisation has $\ell _ { 2 } { \mathrm { - n o r m } }$ at most the Frobenius norm. □

## B.2 Proof of Lemma 8

Proof. By Lemma C.2 with $t = 2 \sqrt { \log ( 2 / \rho ) } , \| \mathbf { E } \| _ { 2 } \leq b _ { \iota }$ . with probability at least $1 - \rho .$ On this event, Weyl's inequality gives $| \lambda _ { i } ( \widetilde { \mathbf { C } } ) - \lambda _ { i } ( \widetilde { \mathbf { X } } \widetilde { \mathbf { X } } ^ { \top } ) | \le \sigma b _ { \eta }$ . for all i. Hence $\lambda _ { i } ( \widetilde { \mathbf { X } } \widetilde { \mathbf { X } } ^ { \top } ) \geq \lambda _ { i } ( \widetilde { \mathbf { C } } ) - \sigma b _ { r } > 0$ for all $i \leq { \widehat { k } } .$ which gives the upper bound, as rank $( \widetilde { \mathbf { X } } ) \le \mathrm { r a n k } ( \mathbf { X } )$ ;and $\lambda _ { i } ( \widetilde { \mathbf { X } } \widetilde { \mathbf { X } } ^ { \top } ) > 2 \sigma b _ { \eta }$ . implies $\lambda _ { i } ( \tilde { \mathbf { C } } ) > \sigma b _ { r }$ , which gives the lower bound. □

## C Proof of Theorem 6

## C.1 Auxiliary Results

We use the following convergence result for the noisy power method by Hardt and Price (2014) (see also Balcan et al., 2016), which implies the convergence bound of Thm. C.4. Throughout this section, $\lambda _ { i } : =$ $\lambda _ { i } ( \mathbf { X X } ^ { \top } )$

Theorem C.1 (Hardt and Price 2014, Cor. 1.1). Let $k \leq b \leq p$ . Let $\mathbf { U } _ { k } \in \mathbb { R } ^ { p \times k }$ be the top-k eigenvectors of a positive semi-definite matrix $\mathbf { K } \in \mathbb { R } ^ { p \times p }$ with eigenvalues $\lambda _ { 1 } \geq \cdot \cdot \cdot \geq \lambda _ { p } \geq 0$ . Suppose the noisy power method is run with block size b (iterates $\mathbf { V } _ { \ell } \in \mathbb { R } ^ { p \times b } )$ , starting from an orthonormal basis of a uniformly random b-dimensional subspace, and that at every iteration the noise matrix $\mathbf { G } _ { \ell } \in \mathbb { R } ^ { p \times b }$ satisfies

$$
5 \| \mathbf { G } _ { \ell } \| _ { 2 } \leq \zeta ( \lambda _ { k } - \lambda _ { k + 1 } ) , \qquad 5 \| \mathbf { U } _ { k } ^ { \top } \mathbf { G } _ { \ell } \| _ { 2 } \leq ( \lambda _ { k } - \lambda _ { k + 1 } ) \frac { \sqrt { b } - \sqrt { k - 1 } } { \tau \sqrt { p } }
$$

for some $\tau > 0$ and $\zeta < 1 / 2$ . Then, with probability at least $1 - \tau ^ { - \Omega ( b + 1 - k ) } - e ^ { - \Omega ( p ) }$ , there exists $L =$ $\begin{array} { r } { O \left( \frac { \lambda _ { k } } { \lambda _ { k } - \lambda _ { k + 1 } } \log \left( \frac { \tau p } { \zeta } \right) \right) } \end{array}$ such that $\left\| ( \mathbf { I } - \mathbf { V } _ { T } \mathbf { V } _ { T } ^ { \top } ) \mathbf { U } _ { k } \right\| _ { 2 } \leq \zeta$ for all $T \geq L$

To analyze the effect of the additive noise of Analyze Gauss, we will need a concentration bound for the spectral norm of symmetric Gaussian matrices.

Lemma C.2 (Thm. II.7, Davidson and Szarek, 2001). Let $\mathbf { E } \in \mathbb { R } ^ { r \times r }$ be a symmetric Gaussian matrix where entries $E _ { i j } \ ( i \leq j )$ are i.i.d. $\mathcal { N } ( 0 , 1 )$ and let $t \geq 0$ . Then, the spectral norm satisfies

$$
\mathbb { P } \left( \| \mathbf { E } \| _ { 2 } \geq 2 { \sqrt { r } } + t \right) \leq 2 \exp \left( - { \frac { t ^ { 2 } } { 4 } } \right) .
$$

We will also need the following concentration bound for converting the bounds of Thm. C.1.

Lemma C.3 (Rudelson and Vershynin, 2010, Eq. (2.3)). Let $\mathbf { A } \in \mathbb { R } ^ { N \times n }$ be a matrix with i.i.d. $\mathcal { N } ( 0 , 1 )$ entries. Then, for every $t \geq 0$

$$
\mathbb { P } \Big ( s _ { \operatorname* { m a x } } ( \mathbf { A } ) \ \leq \ \sqrt { N } + \sqrt { n } + t \Big ) \ \geq \ 1 - 2 e ^ { - t ^ { 2 } / 2 } .
$$

The following result gives a bound for the projection error $\begin{array} { r } { \left\| ( \mathbf { I } - \mathbf { V } _ { i } \mathbf { V } _ { i } ^ { \top } ) \mathbf { U } _ { k } \right\| _ { 2 } , } \end{array}$ where $\mathbf { U } _ { k } \in \mathbb { R } ^ { p \times k }$ gives the principal k eigenvectors. It is an adaptation of (Thm 1.3, Hardt and Price, 2014) to our noise calibration. We will need this result in the overall error bound given in Thm. 6.

Theorem C.4. Let $\mathbf { U } _ { k } \in \mathbb { R } ^ { p \times k }$ be the matrix of leading k eigenvectors of $\mathbf { X X ^ { \top } }$ , and let $\mathbf { V } _ { T }$ be the output of the noisy power method (Algorithm 1) with block size k, a public number of iterations T and i.i.d. noise entries $( \mathbf { Z } _ { t } ) _ { i j } \sim \mathcal { N } ( 0 , C ^ { 4 } \sigma ^ { 2 } )$ , where C is the public bound on $\| \mathbf { x } _ { i } \| _ { 2 }$

(i) The algorithm is ${ \sqrt { T } } / \sigma { - } G D P$ In particular, $i f \sigma = \varepsilon ^ { - 1 } \sqrt { 4 T \log ( 1 / \delta ) }$ for some $\delta \in ( 0 , 1 )$ and $0 < \varepsilon \le$ $2 \log ( 1 / \delta )$ , the algorithm satisfies $( \varepsilon , \delta ) \ – D P$

(ii) Fix a constant $\tau \geq 1$ , let $\nu \in ( 0 , 1 )$ , and let

$$
\bar { \zeta } : = \frac { 1 0 \tau C ^ { 2 } \sigma \sqrt { p } \left( 2 + \sqrt { 2 \log ( 4 T / \nu ) } \right) } { \lambda _ { k } - \lambda _ { k + 1 } } \cdot \frac { \sqrt { k } } { \sqrt { k } - \sqrt { k - 1 } } = { \cal O } \left( \frac { C ^ { 2 } \sigma \sqrt { p } \left( 1 + \sqrt { \log ( T / \nu ) } \right) } { \lambda _ { k } - \lambda _ { k + 1 } } \cdot k \right) .
$$

If T ≥ L for $\begin{array} { r } { L = O \Big ( \frac { \lambda _ { k } } { \lambda _ { k } - \lambda _ { k + 1 } } \log ( \tau p / \bar { \zeta } ) \Big ) } \end{array}$ as in Theorem $C . 1 ,$ then, $w . p .$ at least $1 - \nu - p _ { 0 }$ with p0 = τ−Ω(1) + e−Ω(p),

$$
\begin{array} { r } { \left\| ( { \mathbf { I } } - { \mathbf { V } } _ { T } { \mathbf { V } } _ { T } ^ { \top } ) { \mathbf { U } } _ { k } \right\| _ { 2 } \leq \operatorname* { m i n } \{ 1 , \bar { \zeta } \} . } \end{array}
$$

Proof. (i) Each iteration is a Gaussian mechanism with sensitivity $C ^ { 2 }$ and noise standard deviation $C ^ { 2 } \sigma$ (Section 4), so Lemma 2 gives $\widetilde { \Delta } = \sqrt { T } / \sigma$ , and the $( \varepsilon , \delta )$ -claim follows from Corollary A.5 with $\Delta = C ^ { 2 }$

(ii) We apply Theorem C.1 with block size $b = k$ , so that $\tau ^ { - \Omega ( b + 1 - k ) } = \tau ^ { - \Omega ( 1 ) }$ and the two conditions read

$$
5 \| \mathbf { G } _ { t } \| _ { 2 } \ \leq \ \zeta \left( \lambda _ { k } - \lambda _ { k + 1 } \right) , \qquad 5 \| \mathbf { U } _ { k } ^ { \top } \mathbf { G } _ { t } \| _ { 2 } \ \leq \ \left( \lambda _ { k } - \lambda _ { k + 1 } \right) \frac { \sqrt { k } - \sqrt { k - 1 } } { \tau \sqrt { p } } .\tag{C.1}
$$

At each iteration, $\mathbf { G } _ { t }$ is a $p \times k$ matrix with i.i.d. ${ \mathcal { N } } ( 0 , C ^ { 4 } \sigma ^ { 2 } )$ entries. By the (constant-free) Gaussian concentration bound for the largest singular value (Rudelson and Vershynin, 2010, Eq. (2.3)), applied after factoring out the scale $C ^ { 2 } \sigma$ , for any $s \geq 0$

$$
\begin{array} { r } { \mathbb { P } \Big ( \| \mathbf { G } _ { t } \| _ { 2 } \leq C ^ { 2 } \sigma \big ( \sqrt { p } + \sqrt { k } + s \big ) \Big ) \geq 1 - 2 e ^ { - s ^ { 2 } / 2 } . } \end{array}
$$

Since $\mathbf { U } _ { k } ^ { \top } \mathbf { G } _ { t } \in \mathbb { R } ^ { k \times k }$ is again a Gaussian matrix with i.i.d. ${ \mathcal { N } } ( 0 , C ^ { 4 } \sigma ^ { 2 } )$ entries (by rotational invariance of the Gaussian and the orthonormality of $\mathbf { U } _ { k } )$ , the same bound gives

$$
\begin{array} { r } { \mathbb { P } \Big ( \| \mathbf { U } _ { k } ^ { \top } \mathbf { G } _ { t } \| _ { 2 } \ \leq \ C ^ { 2 } \sigma \big ( 2 \sqrt { k } + s \big ) \Big ) \ \geq \ 1 - 2 e ^ { - s ^ { 2 } / 2 } . } \end{array}
$$

Setting $s = \sqrt { 2 \log ( 4 T / \nu ) }$ and taking a union bound over the $T$ iterations and the two events at each iteration, we obtain that simultaneously for all $t \leq T$ , with probability at least $1 - \nu ,$

$$
\| \mathbf { G } _ { t } \| _ { 2 } \leq C ^ { 2 } \sigma \sqrt { p } \left( 2 + s \right) , \qquad \| \mathbf { U } _ { k } ^ { \top } \mathbf { G } _ { t } \| _ { 2 } \leq C ^ { 2 } \sigma \sqrt { k } \left( 2 + s \right) ,\tag{C.2}
$$

where we used ${ \sqrt { p } } + { \sqrt { k } } + s \leq { \sqrt { p } } ( 2 + s )$ and $2 { \sqrt { k } } + s \leq { \sqrt { k } } ( 2 + s )$ (since $1 \leq k \leq p )$ . We work on this event and on the success event of the initialization in Theorem C.1; a union bound gives the stated probability. Let

$$
\zeta ^ { \prime } : = \frac { 5 C ^ { 2 } \sigma \sqrt { p } \left( 2 + s \right) } { \lambda _ { k } - \lambda _ { k + 1 } } , \qquad \mathrm { s o ~ t h a t } \qquad \bar { \zeta } = 2 \tau \zeta ^ { \prime } \frac { \sqrt { k } } { \sqrt { k } - \sqrt { k - 1 } } \ge 2 \zeta ^ { \prime } .
$$

By (C.2), the first condition of (C.1) holds with $\zeta = \zeta ^ { \prime } .$ Since $\| ( \mathbf { I } - \mathbf { V } _ { T } \mathbf { V } _ { T } ^ { \top } ) \mathbf { U } _ { k } \| _ { 2 } \leq 1$ always holds, it suffices to consider the case $\bar { \zeta } < 1$ , which implies $\zeta ^ { \prime } < 1 / 2$ . In this case also the second condition of (C.1) holds: otherwise, by (C.2),

$$
( \lambda _ { k } - \lambda _ { k + 1 } ) \frac { \sqrt { k } - \sqrt { k - 1 } } { \tau \sqrt { p } } < 5 \| \mathbf { U } _ { k } ^ { \top } \mathbf { G } _ { t } \| _ { 2 } \leq 5 C ^ { 2 } \sigma \sqrt { k } \left( 2 + s \right) ,
$$

i.e., $\tau \zeta ^ { \prime } \sqrt { k } / ( \sqrt { k } - \sqrt { k - 1 } ) > 1$ and hence $\bar { \zeta } > 2$ , a contradiction. Theorem C.1 with $\zeta = \zeta ^ { \prime }$ then gives $\| ( \mathbf { I } - \mathbf { V } _ { T } \mathbf { V } _ { T } ^ { \top } ) \mathbf { U } _ { k } \| _ { 2 } \leq \zeta ^ { \prime } \leq \bar { \zeta }$ for all $T \geq L$ where $\begin{array} { r } { L = { \cal O } \big ( { \frac { \lambda _ { k } } { \lambda _ { k } - \lambda _ { k + 1 } } } \log { \frac { \tau p } { \zeta ^ { \prime } } } \big ) = { \cal O } \big ( { \frac { \lambda _ { k } } { \lambda _ { k } - \lambda _ { k + 1 } } } \log { \frac { \tau p } { \bar { \zeta } } } \big ) } \end{array}$ since $\zeta ^ { \prime } \geq \bar { \zeta } / ( 4 \tau k )$ and $k \leq p$ Finally, $\begin{array} { r } { \frac { \sqrt { k } } { \sqrt { k } - \sqrt { k - 1 } } = \sqrt { k } ( \sqrt { k } + \sqrt { k - 1 } ) \leq 2 k } \end{array}$ gives the O(·)-form of $\bar { \zeta } .$ □

Let $\mathbf { K } = \mathbf { X } \mathbf { X } ^ { \top }$ and let $\mathbf { Q } \in \mathbb { R } ^ { p \times k }$ have orthonormal columns with $\left\| ( \mathbf { I } - \mathbf { Q Q } ^ { \top } ) \mathbf { U } _ { k } \right\| _ { 2 } \leq \zeta$ (as given by Theorem C.4). This directly translates to the projection error (cf. Hardt and Price, 2014, Eq. (3))

$$
\left\| \mathbf { K } - \mathbf { Q Q } ^ { \top } \mathbf { K } \right\| _ { 2 } \leq \lambda _ { k + 1 } + \zeta \lambda _ { 1 } ,\tag{C.3}
$$

since, writing $\mathbf { K } = \mathbf { U } _ { k } \pmb { \Lambda } _ { k } \mathbf { U } _ { k } ^ { \top } + \mathbf { U } _ { \bot } \pmb { \Lambda } _ { \bot } \mathbf { U } _ { \parallel } ^ { \top }$ , the term $( \mathbf { I } - \mathbf { Q Q } ^ { \top } ) \mathbf { U } _ { k } \pmb { \Lambda } _ { k } \mathbf { U } _ { k } ^ { \top }$ has spectral norm at most $\zeta \lambda _ { 1 }$ and $( \mathbf { I } - \mathbf { Q Q } ^ { \top } ) \mathbf { U } _ { \perp } \pmb { \Lambda } _ { \perp } \mathbf { U } _ { \perp } ^ { \top }$ at most $\lambda _ { k + 1 }$

This can be further translated to a bound for the error $\left\| \mathbf { K } - \mathbf { Q Q } ^ { \top } \mathbf { K Q Q } ^ { \top } \right\| _ { 2 }$ using the following auxiliary lemma.

Lemma C.5. Let $\mathbf { A } \in \mathbb { R } ^ { m \times m }$ be symmetric and let $\mathbf { Q } \in \mathbb { R } ^ { m \times r }$ have orthonormal columns, i.e., $\mathbf { Q } ^ { \top } \mathbf { Q } = \mathbf { I } _ { r }$ Then the following inequality holds:

$$
\left\| \mathbf { A } - \mathbf { Q Q } ^ { \top } \mathbf { A Q Q } ^ { \top } \right\| _ { 2 } \leq \sqrt { 2 } \left\| \mathbf { A } - \mathbf { Q Q } ^ { \top } \mathbf { A } \right\| _ { 2 } .
$$

Proof. Define the orthogonal projector $\mathbf { P } : = \mathbf { Q Q } ^ { \top }$ . Then

$$
\mathbf { A } - \mathbf { P A P } = ( \mathbf { I } - \mathbf { P } ) \mathbf { A } + \mathbf { P A } ( \mathbf { I } - \mathbf { P } ) .
$$

The two terms on the right-hand side have orthogonal column spaces, so

$$
\begin{array} { r } { \| \mathbf { A } - \mathbf { P A P } \| _ { 2 } \leq \sqrt { \| ( \mathbf { I } - \mathbf { P } ) \mathbf { A } \| _ { 2 } ^ { 2 } + \| \mathbf { P A } ( \mathbf { I } - \mathbf { P } ) \| _ { 2 } ^ { 2 } } . } \end{array}
$$

By symmetry of A, $\| \mathbf { P A } ( \mathbf { I } - \mathbf { P } ) \| _ { 2 } \ = \ \| ( \mathbf { I } - \mathbf { P } ) \mathbf { A } \mathbf { P } \| _ { 2 } \ \leq \ \| ( \mathbf { I } - \mathbf { P } ) \mathbf { A } \| _ { 2 }$ , so both terms are bounded by $\| \mathbf { A } - \mathbf { P } \mathbf { A } \| _ { 2 } ,$ and the result follows □

With the bound (C.3) and with Lemma C.5, the error analysis for the two phases (computing the basis and the noisy projected matrix) becomes straightforward.

## C.2 The Proof of Theorem 6

Proof. (i) By Theorem C.4(i) with $\sigma = \sigma _ { 1 } / C ^ { 2 }$ , the first stage (whose output $\mathbf { V } _ { k }$ is a post-processing of the noisy iterates) is $( \varepsilon _ { 1 } , \delta _ { 1 } ) – \mathrm { D P }$ Given $\mathbf { V } _ { k } ,$ the second stage is a single Gaussian mechanism with sensitivity $C ^ { 2 }$ (Section 4) and noise standard deviation $\sigma _ { 2 }$ , hence $( \varepsilon _ { 2 } , \delta _ { 2 } ) – \mathrm { D P }$ by Corollary A.5 with $T = 1$ and $\Delta = C ^ { 2 }$ The claim follows from basic (adaptive) composition.

(ii) Substituting $\widetilde { \mathbf { Y } } _ { k } = \mathbf { V } _ { k } ^ { \top } \mathbf { K } \mathbf { V } _ { k } + \mathbf { Z }$ , the error decomposes as

$$
\mathrm { E r r } _ { k } = { \bf K } - { \bf V } _ { k } { \widetilde { \bf Y } } _ { k } { \bf V } _ { k } ^ { \top } = \left( { \bf K } - { \bf V } _ { k } { \bf V } _ { k } ^ { \top } { \bf K } { \bf V } _ { k } { \bf V } _ { k } ^ { \top } \right) - { \bf V } _ { k } { \bf Z } { \bf V } _ { k } ^ { \top } .
$$

Since $\mathbf { V } _ { k }$ has orthonormal columns, $\| \mathbf { V } _ { k } \mathbf { Z } \mathbf { V } _ { k } ^ { \top } \| _ { 2 } = \| \mathbf { Z } \| _ { 2 }$ , and the triangle inequality gives

$$
\| \mathrm { E r r } _ { k } \| _ { 2 } \ \leq \ \| \mathbf { K } - \mathbf { V } _ { k } \mathbf { V } _ { k } ^ { \top } \mathbf { K } \mathbf { V } _ { k } \mathbf { V } _ { k } ^ { \top } \| _ { 2 } \ + \ \| \mathbf { Z } \| _ { 2 } .\tag{C.4}
$$

We bound the two terms separately.

By Lemma C.5 (K is symmetric),

$$
\| \mathbf { K } - \mathbf { V } _ { k } \mathbf { V } _ { k } ^ { \top } \mathbf { K } \mathbf { V } _ { k } \mathbf { V } _ { k } ^ { \top } \| _ { 2 } \leq \sqrt { 2 } \| \mathbf { K } - \mathbf { V } _ { k } \mathbf { V } _ { k } ^ { \top } \mathbf { K } \| _ { 2 } .
$$

Combining this with (C.3) and Theorem C.4(ii) (with $C ^ { 2 } \sigma = \sigma _ { 1 } )$ , we obtain, with probability at least $1 - \nu - p _ { 0 }$ 2

$$
\| \mathbf { K } - \mathbf { V } _ { k } \mathbf { V } _ { k } ^ { \top } \mathbf { K } \mathbf { V } _ { k } \mathbf { V } _ { k } ^ { \top } \| _ { 2 } \leq \sqrt { 2 } \big ( \lambda _ { k + 1 } + \lambda _ { 1 } \zeta \big ) , \qquad \zeta = \operatorname* { m i n } \{ 1 , \bar { \zeta } \} ,\tag{C.5}
$$

with $\bar { \zeta }$ as in Theorem C.4.

Write $\mathbf { Z } \ = \ \sigma _ { 2 } \mathbf { G }$ , where $\textbf { G } \in \mathbb { R } ^ { k \times k }$ is a symmetric Gaussian matrix whose upper-triangular entries (including the diagonal) are i.i.d. $\mathcal { N } ( 0 , 1 )$ . By Lemma C.2, for any $t \geq 0$

$$
\begin{array} { r } { \mathbb { P } \big ( \| \mathbf { G } \| _ { 2 } \leq 2 \sqrt { k } + t \big ) \geq 1 - 2 e ^ { - t ^ { 2 } / 4 } . } \end{array}
$$

Setting $t = 2 \sqrt { \log ( 2 / \rho ) }$ , we obtain, with probability at least $1 - \rho _ { ; }$

$$
\| \mathbf { Z } \| _ { 2 } \leq \sigma _ { 2 } \Big ( 2 \sqrt { k } + 2 \sqrt { \log ( 2 / \rho ) } \Big ) .\tag{C.6}
$$

Substituting (C.5) and (C.6) into (C.4) and applying a union bound over the high-probability events yields the stated inequality with probability at least $1 - \nu - \rho - p _ { 0 }$ □

## D Proofs for Section 6

Write $e _ { j } : = { \theta } _ { j } - { \theta } _ { * }$ . Since $\nabla \mathcal L ( \theta ) = \mathbf C _ { \lambda } ( \theta - \theta _ { * } )$ , (6.1) reads $e _ { j + 1 } = ( { \bf I } - \eta { \bf P } ^ { - 1 } { \bf C } _ { \lambda } ) e _ { j } - \eta { \bf P } ^ { - 1 } \mathcal { E } _ { j }$ . For $\mathbf { G } \succ 0 .$ Courant-Fischer applied to ${ \bf G } ^ { - 1 / 2 } { \bf C } _ { \lambda } { \bf G } ^ { - 1 / 2 } \ ( \mathrm { s e e } , \mathrm { e . g . }$ , Horn and Johnson, 2012, Thm. 4.2.2 and Thm. 4.2.6) shows that the extreme generalized eigenvalues of $( \mathbf { C } _ { \lambda } , \mathbf { G } )$ are the extreme values of $\mathbf { x } ^ { \top } \mathbf { C } _ { \lambda } \mathbf { x } / \mathbf { x } ^ { \top }$ Gx over $\mathbf { x } \neq \mathbf { 0 }$

## D.1 Auxiliary Lemma

Lemma D.1. $I f \mathbf { V } \in \mathbb { R } ^ { p \times k }$ is orthonormal and $\widetilde { \mathbf { Y } } + \lambda \mathbf { I } _ { k } \succ 0$ , then $\widehat { \mathbf G } \succ 0$ and

$$
\begin{array} { r } { \widehat { \mathbf { G } } ^ { - 1 } = \mathbf { V } \big ( \widetilde { \mathbf { Y } } + \lambda \mathbf { I } _ { k } \big ) ^ { - 1 } \mathbf { V } ^ { \top } + \frac { 1 } { \lambda } \big ( \mathbf { I } - \mathbf { V } \mathbf { V } ^ { \top } \big ) . } \end{array}\tag{D.1}
$$

For the oracle preconditioner $\mathbf { G } _ { \star } : = \mathbf { U } _ { k } \pmb { \Lambda } _ { k } \mathbf { U } _ { k } ^ { \top } + \lambda \mathbf { I } \ \big ( \mathbf { V } = \mathbf { U } _ { k } , \mathbf { Z } = \mathbf { 0 } \big )$ , the generalized eigenvalues of $( \mathbf { C } _ { \lambda } , \mathbf { G } _ { \star } )$ are 1 for $i \leq k$ and $( \lambda _ { i } + \lambda ) / \lambda f o r \ : i > k$ . They all lie in $[ 1 , \kappa _ { k } ]$ , and $\begin{array} { r } { \mathrm { t r } ( \mathbf { G } _ { \star } ^ { - 1 } ) = \sum _ { i < k } ( \lambda _ { i } + \lambda ) ^ { - 1 } + ( p - k ) / \lambda } \end{array}$

Proof. Let $\mathbf { O } = \left[ \mathbf { V } , \mathbf { V } _ { \perp } \right]$ be orthogonal. Then $\mathbf { O } ^ { \top } \widehat { \mathbf { G } } \mathbf { O } = \mathrm { d i a g } \big ( \widetilde { \mathbf { Y } } + \lambda \mathbf { I } _ { k } , \lambda \mathbf { I } _ { p - k } \big ) \succ 0$ . Inverting blockwise and using $\mathbf { V } _ { \bot } \mathbf { V } _ { \bot } ^ { \top } = \mathbf { I } - \mathbf { V } \mathbf { V } ^ { \top }$ gives (D.1). For $\mathbf { V } = \mathbf { U } _ { k } , \mathbf { Z } = \mathbf { 0 }$ we have $\widetilde { \mathbf { Y } } = \boldsymbol { \Lambda } _ { k } .$ so in the basis U both $\mathbf { C } _ { \lambda }$ and $\mathbf { G } _ { \star }$ are diagonal. The entries of $\mathbf { C } _ { \lambda }$ are $\lambda _ { i } + \lambda$ . The entries of $\mathbf { G } _ { \ast }$ are $\lambda _ { i } + \lambda$ for $i \leq k$ and λ for $i > k$ The generalized eigenvalues are the ratios: 1 for $i \leq k$ , and $1 + \lambda _ { i } / \lambda \in [ 1 , \kappa _ { k } ]$ for $i > k ,$ since $0 \leq \lambda _ { i } \leq \lambda _ { k + 1 }$ The trace is the sum of the reciprocal diagonal entries of $\mathbf { G } _ { \star }$ □

## D.2 Proof of Lemma 9

Proof. Let $\mathbf { M } : = \mathbf { G } _ { \star } ^ { - 1 / 2 } \widehat { \mathbf { G } } \mathbf { G } _ { \star } ^ { - 1 / 2 }$ . By definition of $\gamma , ( 1 - \gamma ) \mathbf { I } \preccurlyeq \mathbf { M } \preccurlyeq ( 1 + \gamma ) \mathbf { I }$ , and congruence with $\mathbf { G } _ { \star } ^ { 1 / 2 }$ gives the sandwich. If $\gamma < 1$ , then $\widehat { \mathbf G } \succ 0$ , and for $\mathbf { x } \neq \mathbf { 0 }$

$$
\frac { \mathbf { x } ^ { \top } \mathbf { C } _ { \lambda } \mathbf { x } } { \mathbf { x } ^ { \top } \widehat { \mathbf { G } } \mathbf { x } } = \frac { \mathbf { x } ^ { \top } \mathbf { C } _ { \lambda } \mathbf { x } } { \mathbf { x } ^ { \top } \mathbf { G } _ { \lambda } \mathbf { x } } \cdot \frac { \mathbf { x } ^ { \top } \mathbf { G } _ { \star } \mathbf { x } } { \mathbf { x } ^ { \top } \widehat { \mathbf { G } } \mathbf { x } } \in \left[ \frac { 1 } { 1 + \gamma } , \frac { \kappa _ { k } } { 1 - \gamma } \right]
$$

by Lemma D.1 and the sandwich. Inversion is order-reversing on SPD matrices, so $\widehat { \mathbf { G } } ^ { - 1 } \preccurlyeq ( 1 - \gamma ) ^ { - 1 } \mathbf { G } _ { \star } ^ { - 1 }$ Taking traces gives the trace bound.

For the bound on $\gamma ,$ note that $\lambda _ { \operatorname* { m i n } } ( \mathbf { G } _ { \star } ) = \lambda ( \mathrm { a s } \ k < p )$ , SO $\boldsymbol { \gamma } \le \left\| \widehat { \mathbf { G } } - \mathbf { G } _ { \star } \right\| _ { 2 } / \lambda$ Let $\mathbf { I I } : = \mathbf { V } \mathbf { V } ^ { \top }$ and $\boldsymbol { \Pi } _ { \star } : = \mathbf { U } _ { k } \mathbf { U } _ { k } ^ { \top }$ . Then $\mathbf { G } _ { \star } = \mathbf { I I } _ { \star } \mathbf { C I I } _ { \star } + \lambda \mathbf { I }$ and $\mathbf { V } \widetilde { \mathbf { Y } } \mathbf { V } ^ { \top } = \mathbf { \Pi } \mathbf { I } \mathbf { C } \mathbf { I } \mathbf { I } + \mathbf { V } \mathbf { Z } \mathbf { V } ^ { \top }$ , SO

$$
\widehat { \mathbf { G } } - \mathbf { G } _ { \star } = ( \mathbf { I I } - \mathbf { I I } _ { \star } ) \mathbf { C } \mathbf { I I } + \mathbf { I I } _ { \star } \mathbf { C } ( \mathbf { I I } - \mathbf { I I } _ { \star } ) + \mathbf { V Z V } ^ { \top } ,
$$

and hence $\begin{array} { r } { \left\| \widehat { \mathbf { G } } - \mathbf { G } _ { \star } \right\| _ { 2 } \leq 2 \lambda _ { 1 } \left\| \Pi - \Pi _ { \star } \right\| _ { 2 } + \left\| \mathbf { Z } \right\| _ { 2 } } \end{array}$ For orthogonal projectors of equal rank, $\begin{array} { r } { \| \mathbf { I I } - \mathbf { I I } _ { \star } \| _ { 2 } = } \end{array}$ $\| ( \mathbf { I } - \mathbf { I I } ) \pmb { \Pi } _ { \star } \| _ { 2 } = \zeta$ , which is the sine of the largest principal angle (Stewart and Sun, 1990, Thm. I.5.5). The tail bound is Lemma C.2 with $t = 2 \sqrt { \log ( 2 / \rho ) }$ □

## D.3 Proof of Theorem 10

Proof. Condition on P. Let $f _ { j } : = \mathbf { C } _ { \lambda } ^ { 1 / 2 } e _ { j }$ , so that $\| f _ { j } \| _ { 2 } ^ { 2 } = 2 ( \mathcal { L } ( \theta _ { j } ) - \mathcal { L } ( \theta _ { * } ) )$ , and let $\mathbf { S } : = \mathbf { C } _ { \lambda } ^ { 1 / 2 } \mathbf { P } ^ { - 1 } \mathbf { C } _ { \lambda } ^ { 1 / 2 }$ . The matrix S is similar to $\mathbf { P } ^ { - 1 } \mathbf { C } _ { \lambda }$ , so its spectrum lies in $[ a , b ]$ . Since $\eta b \leq 1$ , this gives $\mathbf { 0 } \preccurlyeq \mathbf { I } - \eta \mathbf { S } \preccurlyeq ( 1 - \eta a ) \mathbf { I }$ The error evolves as $f _ { j + 1 } = ( \mathbf { I } - \eta \mathbf { S } ) f _ { j } - \eta \mathbf { C } _ { \lambda } ^ { 1 / 2 } \mathbf { P } ^ { - 1 } \mathcal { E } _ { j }$ . The $\mathcal { E } _ { j }$ are independent, zero-mean and independent of P. With $ { \mathbf { N } } : =  { \mathbf { C } } _ { \lambda } ^ { 1 / 2 }  { \mathbf { P } } ^ { - 2 }  { \mathbf { C } } _ { \lambda } ^ { 1 / 2 } \succcurlyeq 0$ , this gives

$$
\mathbb { E } \left\| f _ { T } \right\| _ { 2 } ^ { 2 } = \left\| \left( \mathbf { I } - \eta \mathbf { S } \right) ^ { T } f _ { 0 } \right\| _ { 2 } ^ { 2 } + \eta ^ { 2 } \sigma _ { g } ^ { 2 } \sum _ { j = 0 } ^ { T - 1 } \mathrm { t r } \big ( ( \mathbf { I } - \eta \mathbf { S } ) ^ { 2 j } \mathbf { N } \big ) .
$$

The first term is at most $( 1 - \eta a ) ^ { 2 T } \parallel f _ { 0 } \parallel _ { 2 } ^ { 2 }$ . For the second term, note that $\eta \mathbf { S } \preccurlyeq \mathbf { I }$ , SO

$$
\sum _ { j \geq 0 } ( \mathbf { I } - \eta \mathbf { S } ) ^ { 2 j } = \left( \eta \mathbf { S } ( 2 \mathbf { I } - \eta \mathbf { S } ) \right) ^ { - 1 } \preccurlyeq ( \eta \mathbf { S } ) ^ { - 1 } .
$$

Since $\mathbf { A } \mapsto \operatorname { t r } ( \mathbf { A N } )$ is monotone for a positive semidefinite N, we have that

$$
\sum _ { j = 0 } ^ { T - 1 } \mathrm { t r } \big ( ( \mathbf { I } - \eta \mathbf { S } ) ^ { 2 j } \mathbf { N } \big ) \leq \eta ^ { - 1 } \mathrm { t r } ( \mathbf { S } ^ { - 1 } \mathbf { N } ) = \eta ^ { - 1 } \mathrm { t r } ( \mathbf { C } _ { \lambda } ^ { - 1 / 2 } \mathbf { P } ^ { - 1 } \mathbf { C } _ { \lambda } ^ { 1 / 2 } ) = \eta ^ { - 1 } \mathrm { t r } ( \mathbf { P } ^ { - 1 } ) .
$$

Dividing by 2 shows the claim.

## D.4 Proof of Corollary 11

Proof. (i) Take $\eta = 1 / b$ in Theorem 10 and write $\kappa : = b / a$ . For DP-GD, $\kappa = \kappa _ { 0 }$ and $\eta \mathrm { t r } ( \mathbf { I } ) = p / ( \lambda _ { 1 } + \lambda ) =$ $N _ { \mathrm { G D } } / \kappa _ { 0 }$ . For , Lemma 9 gives $\begin{array} { r } { a = \frac { 1 } { 1 + \gamma } } \end{array}$ and $\begin{array} { r } { b = \frac { \kappa _ { k } } { 1 - \gamma } } \end{array}$ , sO $\kappa = \widehat { \kappa } _ { k }$ and $\eta \mathrm { t r } ( \widehat { \mathbf { G } } ^ { - 1 } ) \leq \mathrm { t r } ( \mathbf { G } _ { \star } ^ { - 1 } ) / \kappa _ { k } = N _ { k } / \widehat { \kappa } _ { k }$ . In both cases, using $\sigma _ { g } ^ { 2 } = T \Delta _ { g } ^ { 2 } / \mu _ { g } ^ { 2 }$ and $( 1 - 1 / \kappa ) ^ { 2 T } \leq e ^ { - 2 T / \kappa }$

$$
\mathbb { E } \mathcal { L } ( \theta _ { T } ) - \mathcal { L } ( \theta _ { * } ) \le e ^ { - 2 T / \kappa } R _ { 0 } + \frac { T \Delta _ { g } ^ { 2 } N } { 2 \mu _ { g } ^ { 2 } \kappa } .
$$

Let $\beta : = \log \frac { 4 \mu _ { g } ^ { 2 } R _ { 0 } } { \Delta _ { q } ^ { 2 } N } \ge 0$ and $T = \lceil \kappa \beta / 2 \rceil$ . Then $\begin{array} { r } { e ^ { - 2 T / \kappa } R _ { 0 } \leq e ^ { - \beta } R _ { 0 } = \frac { \Delta _ { g } ^ { 2 } N } { 4 \mu _ { g } ^ { 2 } } } \end{array}$ . Also $T \le \kappa \beta / 2 + 1$ , so the second term is at most $\frac { \Delta _ { g } ^ { 2 } N } { 4 \mu _ { q } ^ { 2 } } \big ( \beta + 2 / \kappa \big )$ . Since $\kappa \geq 1$ , the total is at most $\Phi ( N )$

(ii) By Lemma D.1, $\mathrm { t r } ( \mathbf { G } _ { \star } ^ { - 1 } ) \leq p / \lambda = ( 1 + \lambda _ { p } / \lambda ) N _ { \mathrm { G D } }$ . For $\gamma = \lambda _ { p } = 0$ we have $N _ { \mathrm { G D } } = p / \lambda$ and $\begin{array} { r } { N _ { k } = \frac { p } { \lambda } - \sum _ { i \leq k } \bigl ( \frac { 1 } { \lambda } - \frac { 1 } { \lambda _ { i } + \lambda } \bigr ) } \end{array}$ □

## E Extension of the Convergence Analysis to Generalized Linear Models

The analysis of Section 6 was stated for linear regression because that is the case in which the contraction constant can be computed exactly. We now show that the same statements hold for general GLMs under a two-sided curvature condition, with $\kappa _ { k }$ replaced by $\kappa _ { k } / c$ . Lemma D.1 and Lemma 9 concern the preconditioner alone and are unchanged. Theorem 10 is replaced by Theorem E.2 and Corollary E.3.

By the mean value theorem (Section 6), with $t _ { j } : = \theta _ { j } - \theta _ { * }$ , the DP-FC iteration takes the form

$$
\begin{array} { r l } & { t _ { j + 1 } = \mathbf { A } _ { j } t _ { j } + \mathbf { B } \mathcal { E } _ { j } , \qquad \mathbf { B } = - \eta \hat { \mathbf { G } } ^ { - 1 } , } \\ & { \mathbf { A } _ { j } = \mathbf { I } - \eta \hat { \mathbf { G } } ^ { - 1 } \big ( \bar { \mathbf { H } } _ { j } + \lambda \mathbf { I } \big ) , } \\ & { \bar { \mathbf { H } } _ { j } = \frac { 1 } { n } \displaystyle \sum _ { ( x , y ) \in D } \ell ^ { \prime \prime } ( \xi _ { x } , y ) x x ^ { \top } , } \end{array}\tag{E.1}
$$

where $\xi _ { x }$ lies between $x ^ { \top } \theta _ { j }$ and $x ^ { \top } \theta _ { * }$ . Note that $\mathbf { A } _ { j }$ now varies across iterations and successive $\mathbf { A } _ { j }$ do not commute. This is exactly why we bound the norm in the fixed metric $\Vert { \cdot } \Vert _ { \widehat { \mathbf { G } } }$ , with the product bound $\left\| \Pi _ { i < j } \mathbf { A } _ { i } \right\| _ { \widehat { \mathbf { G } } } \leq q ^ { j }$

Definition E.1. We say the loss satisfies the $( c _ { - } , c _ { + } )$ -curvature condition if $\ell ^ { \prime \prime } ( \xi _ { x } , y ) \in [ c _ { - } , c _ { + } ]$ for all $( x , y ) \in D$ and all $\xi _ { x }$ arising in (E.1), for some $0 < c _ { - } \le c _ { + }$ . After rescaling l by $1 / c _ { + }$ we may and do assume $c _ { + } = 1$ and write $c : = c _ { - } / c _ { + } \in ( 0 , 1 ]$ , so that

$$
c { \bf C } \approx \bar { \bf H } _ { j } \preccurlyeq { \bf C } f o r a l l j .\tag{E.2}
$$

Condition (E.2) is the two-sided strengthening of the one-sided assumption $0 \preccurlyeq \bar { \mathbf { H } } _ { j } \preccurlyeq \mathbf { C }$ . It is satisfied for the logistic loss $\ell ^ { \prime \prime } ( t ) = \varrho ( t ) ( 1 - \varrho ( t ) )$ , with $\varrho$ the sigmoid, whenever the logits along the path are bounded, $| \xi _ { x } | \le M$ . In that case $c = 4 \varrho ( M ) ( 1 - \varrho ( M ) )$ . Boundedness of the logits follows from $\| x \| _ { 2 } \leq C$ together with a bound on $\lVert \theta _ { j } \rVert _ { 2 }$ . Since the DP noise is unbounded, such a bound does not follow from the λ-regularization alone; it holds, e.g., for a projected variant of the iteration, and otherwise we take (E.2) along the iterate path as an assumption. The linear regression case is $c = 1$

Theorem E.2. Assume (E.2) and $\gamma < 1$ (Lemma 9). Then every eigenvalue $\mu$ $o f \widehat { \mathbf { G } } ^ { - 1 } ( \bar { \mathbf { H } } _ { j } + \lambda \mathbf { I } )$ is real and satisfies

$$
\frac { c } { 1 + \gamma } \ \leq \ \mu \ \leq \ \frac { \kappa _ { k } } { 1 - \gamma } \qquad f o r \ e v e r y \ j .
$$

Consequently, for $0 < \eta \leq ( 1 - \gamma ) / \kappa _ { k }$ ，

$$
\operatorname* { s u p } _ { j } \| \mathbf { A } _ { j } \| _ { \widehat { \mathbf { G } } } \leq q : = 1 - \frac { \eta c } { 1 + \gamma } ,
$$

which equals $1 - c / \widehat { \kappa } _ { k } f o r \eta = ( 1 - \gamma ) / \kappa _ { k }$ . Setting $c = 1$ recovers the range of Lemma 9.

Corollary E.3. Under the assumptions of Theorem E.2, with $\widehat { \mathbf { G } }$ independent of $( \mathcal { E } _ { j } )$

$$
\mathbb { E } \left. t _ { T } \right. _ { \widehat { \mathbf { G } } } ^ { 2 } \leq q ^ { 2 T } \left. t _ { 0 } \right. _ { \widehat { \mathbf { G } } } ^ { 2 } + \frac { \left( 1 + \gamma \right) \eta \sigma _ { g } ^ { 2 } } { c } \operatorname { t r } ( \widehat { \mathbf { G } } ^ { - 1 } ) ,
$$

and E $\begin{array} { r } { \mathcal L ( \boldsymbol { \theta } _ { T } ) - \mathcal L ( \boldsymbol { \theta } _ { * } ) \leq { \frac { \kappa _ { k } } { 2 ( 1 - \gamma ) } } \mathbb E \left\| \boldsymbol { t } _ { T } \right\| _ { \widehat { \mathbf G } } ^ { 2 } } \end{array}$

Proof of Theorem E.2. As in the proof of Lemma $9 , \widehat { \mathbf { G } } \succ 0$ , and the matrix $\widehat { \mathbf { G } } ^ { - 1 } ( \bar { \mathbf { H } } _ { j } + \lambda \mathbf { I } )$ is similar to the symmetric matrix $\widehat { \mathbf { G } } ^ { - 1 / 2 } ( \bar { \mathbf { H } } _ { j } + \lambda \mathbf { I } ) \widehat { \mathbf { G } } ^ { - 1 / 2 }$ . Its eigenvalues are therefore real and lie between the extreme generalized Rayleigh quotients. By (E.2), $c \mathbf { C } + \lambda \mathbf { I } \preccurlyeq \bar { \mathbf { H } } _ { j } + \lambda \mathbf { I } \preccurlyeq \mathbf { C } _ { \lambda }$ , so it suffices to bound the largest quotient for $\mathbf { C } _ { \lambda }$ from above and the smallest quotient for $c \mathbf { C } + \lambda \mathbf { I }$ from below.

By Lemma 9, max, $\begin{array} { r } { \frac { v ^ { \top } \mathbf C _ { \lambda } v } { v ^ { \top } \widehat { \mathbf G } v } \leq \frac { \kappa _ { k } } { 1 - \gamma } } \end{array}$

Then, consider the oracle preconditioner. In the eigenbasis U, both $c \mathbf { C } + \lambda \mathbf { I }$ and $\mathbf { G }$ are diagonal, so the generalized eigenvalues are

$$
{ \frac { c \lambda _ { i } + \lambda } { \lambda _ { i } + \lambda } } \ ( i \leq k ) , \qquad { \frac { c \lambda _ { i } + \lambda } { \lambda } } \ ( i > k ) .
$$

For $i \leq k , c \lambda _ { i } + \lambda \geq c ( \lambda _ { i } + \lambda )$ because $c \leq 1$ , so the ratio is at least c. For $i > k ,$ the ratio is at least $1 \geq c .$ Hence minv $\begin{array} { r } { \frac { v ^ { \top } ( c \mathbf { C } + \lambda \mathbf { I } ) v } { v ^ { \top } \mathbf { G } _ { \star } v } \geq c . } \end{array}$ Applying $v ^ { \top } \widehat { \mathbf { G } } v \leq ( 1 + \gamma ) v ^ { \top } \mathbf { G } _ { \star } v$ from Lemma 9 gives minv $\begin{array} { r } { \frac { v ^ { \top } ( c \mathbf { C } + \lambda \mathbf { I } ) v } { v ^ { \top } \widehat { \mathbf { G } } v } \geq \frac { c } { 1 + \gamma } , } \end{array}$ The matrix $\widehat { \bf A } _ { j } = \widehat { \bf G } ^ { 1 / 2 } { \bf A } _ { j } \widehat { \bf G } ^ { - 1 / 2 } = { \bf I } - \eta \widehat { \bf G } ^ { - 1 / 2 } ( \bar { \bf H } _ { j } + \lambda { \bf I } ) \widehat { \bf G } ^ { - 1 / 2 }$ is symmetric, so $\begin{array} { r } { \| \mathbf { A } _ { j } \| _ { \widehat { \mathbf { G } } } = \operatorname* { m a x } _ { i } | 1 - \eta \mu _ { i } | . } \end{array}$ For $\eta \le ( 1 - \gamma ) / \kappa _ { k }$ , each $1 - \eta \mu _ { i }$ is in $\textstyle \left[ 0 , 1 - { \frac { \eta c } { 1 + \gamma } } \right]$ □

Proof of Corollary E.3. Condition on $\widehat { \mathbf { G } }$ Since $\bar { \mathbf { H } } _ { j }$ depends only on $\theta _ { j }$ , the noise $\mathcal { E } _ { j }$ is zero-mean and independent of $( t _ { j } , \mathbf { A } _ { j } )$ . Also $\left\| \eta \widehat { \mathbf { G } } ^ { - 1 } \mathbf { \mathcal { E } } _ { j } \right\| _ { \widehat { \mathbf { G } } } ^ { 2 } = \eta ^ { 2 } \mathcal { E } _ { j } ^ { \top } \widehat { \mathbf { G } } ^ { - 1 } \mathcal { E } _ { j }$ . Therefore, by Theorem E.2,

$$
\begin{array} { r } { \mathbb { E } \left\| t _ { j + 1 } \right\| _ { \widehat { \mathbf { G } } } ^ { 2 } \leq q ^ { 2 } \mathbb { E } \left\| t _ { j } \right\| _ { \widehat { \mathbf { G } } } ^ { 2 } + \eta ^ { 2 } \sigma _ { g } ^ { 2 } \mathrm { t r } ( \widehat { \mathbf { G } } ^ { - 1 } ) . } \end{array}
$$

Then, using $1 - q ^ { 2 } \ge 1 - q = \eta c / ( 1 + \gamma )$ gives the first claim. For the second claim, $\ell ^ { \prime \prime } \leq 1$ gives $\nabla ^ { 2 } { \mathcal { L } } \preccurlyeq \mathbf { C } _ { \lambda }$ and $\nabla \mathcal { L } ( \theta _ { * } ) = 0$ . Thus $\begin{array} { r } { \mathcal { L } ( { \boldsymbol { \theta } } ) - \mathcal { L } ( \theta _ { * } ) \leq \frac { 1 } { 2 } \left\| { \boldsymbol { t } } \right\| _ { \mathbf { C } _ { \lambda } } ^ { 2 } \leq \frac { \kappa _ { k } } { 2 ( 1 - \gamma ) } \left\| { \boldsymbol { t } } \right\| _ { \widehat { \mathbf { G } } } ^ { 2 } } \end{array}$ by Lemma 9. □

## F Further Details on Experiments

## F.1 Additional Information on Datasets and Baseline Methods

Datasets. In addition to well-known benchmark image datasets, we consider low-dimensional sensor dataset UCI-HAR and WISDM

MNIST (LeCun et al., 1998). Handwritten digit images of size $2 8 { \times } 2 8 ~ ( p = 7 8 4 )$ , used for the rankestimation illustration in Section 7.1. SVHN (Street View House Numbers) (Netzer et al., 2011). A real-world image dataset consisting of 32×32 RGB images of house numbers collected from Google Street View.

FashionMNIST, CIFAR-10, CIFAR-100. For the downstream machine learning experiments in Section 7.3, we additionally use FashionMNIST (Xiao et al., 2017), with 512-dimensional feature vectors extracted from a pretrained ResNet-34 (He et al., 2016), and CIFAR-10/100 (Krizhevsky, 2009), with 768- dimensional feature vectors extracted from a pretrained ViT-B/16 (Dosovitskiy et al., 2021).

UCI Human Activity Recognition (HAR) (Anguita et al., 2013). A multivariate time-series dataset for human activity recognition. The raw signals are sampled into windows of 2.56 second and from each window, a set of $p = 5 6 1$ time- and frequency-domain features (means, standard deviations, signal energy, entropy, autoregressive coefficients, spectral properties) are extracted, giving 10299 instances.

WISDM (Wireless Sensor Data Mining) (Kwapisz et al., 2011). A benchmark for human activity recognition. The raw data consists of time series of $( x , y , z )$ accelerometer measurements. We segment the raw signals into windows of 128 samples (≈ 6.4 seconds) with 50% overlap. For each window, we extract a set of time- and frequency-domain statistics including means, variances, medians, spectral energies and correlation coefficients across axes. This results in a feature representation of dimension 200 per window. Across all participants, we obtain 17158 windows, giving a data matrix of shape $1 7 1 5 8 \times 2 0 0 .$

Baseline Methods. Among prior methods with $O ( p k )$ communication cost, we focus our quantitative comparison on the federated DP-PCA method of Grammenos et al. (2020). The remaining $O ( p k )$ communication baselines (Guo et al., 2024; Li et al., 2024) are dominated by our approach by construction:

they report only local privacy guarantees and do not benefit from privacy amplification by summation. Furthermore, Guo et al. (2024) uses the centralized DP-PCA method of Dwork et al. (2014) as its core building block, and we show below that we improve upon that method directly even when the latter is given the benefit of summation amplification. Similarly, Li et al. (2024) uses the DP power iteration as its building block but lacks our second phase for accurately approximating the projected covariance matrix. We therefore benchmark against two strong baselines: the centralized $O ( p ^ { 2 } )$ -communication method of Dwork et al. (2014) and the federated $O ( p k )$ -communication method of Grammenos et al. (2020).

## F.2 Comparison to Locally DP Baseline (Method of Grammenos et al. (2020))

We next consider the federated setting and compare against the distributed DP-PCA method of Grammenos et al. (2020), which is built on top of the centralized noisy covariance method of Dwork et al. (2014). In this method, each client locally applies the DP-PCA mechanism to its user-wise covariance matrix, producing a pair $\left( \mathbf { U } _ { i } , \pmb { \Sigma } _ { i } \right)$ with $\mathbf { U } _ { i } \in \mathbb { R } ^ { p \times r }$ orthogonal and $\pmb { \Sigma } _ { i } \in \mathbb { R } ^ { r \times r }$ diagonal. The server then merges these pairs through the procedure described in Algorithm 2 of Grammenos et al. (2020); with N clients, this requires $\approx \log _ { 2 } N$ rounds and $\approx N$ pairwise merges.

Since each client carries out the DP randomization before computing its truncated SVD, the method does not benefit from summation of the client-wise contributions. In contrast, our method offloads noise addition to the clients in such a way that the aggregated noise corresponds to that of a single centrally calibrated Gaussian mechanism applied at the server. Consequently, the per-client noise variance scales as $\sigma ^ { 2 } / N$ , and the privacy-utility trade-off does not depend on the number of clients. Figure 5 verifies this empirically: the relative error of Grammenos et al. (2020) increases as the number of clients grows from 4 to 16.

![](images/5d64a6d306563b2a2aa4e5399314d2291099602268187b7a08032d18f67a4a19.jpg)  
Figure 5: Comparison against the federated DP-PCA method of Grammenos et al. (2020). The privacyutility trade-off of Grammenos et al. (2020) degrades as the number of clients grows.

## F.3 Qualitative Illustration: Private Covariance Heatmaps

To complement the quantitative results, Figure 6 visualizes the WISDM covariance matrix recovered by our method and by the Analyze Gauss baseline (Dwork et al., 2014) at $( \varepsilon , \delta ) = ( 0 . 1 6 , 1 0 ^ { - 5 } )$ and rank $k = 2 0$ alongside the true (non-private) covariance. Our method preserves the block-correlation structure of the original covariance matrix while the baseline produces a heatmap dominated by noise. A similar qualitative comparison for UCI HAR is provided in Figure 7.

![](images/f620383a3b576b7bb8b64396ece277abc32942a16c13c70983b4083e4cd39700.jpg)

![](images/e0e388b2c392b01718e0a6e2eab62fcd1dc0a119d35e8a824af60d83a84c5325.jpg)

![](images/609a8e35eb9ef439bd650a3f1ffb2b8bd5227678de0bfc9015f0655985bd789b.jpg)  
Figure 6: Heatmaps of the WISDM covariance matrix when $\varepsilon = 0 . 1 6$ and $\delta = 1 0 ^ { - 5 }$ Left: the true (nonprivate) covariance. Center: the rank-20 approximation produced by our method. Right: the rank-20 approximation produced by the Analyze Gauss baseline (Dwork et al., 2014).

![](images/740754240133fed19f46e851a761ca2a172f7905bf1c47d1e74ad30c15a4e49e.jpg)

![](images/a19a11e4a35b10f6ed20a73be56f74fdf04ca6f29e1589d2596cd0261b618fb2.jpg)

![](images/385221fc870825238045ce8d350f248bb35ac7426d9dbb655c1d30b7aa6b7baa.jpg)  
Figure 7: The original heatmap of the covariance matrix for UCI HAR dataset features, heatmap obtained using our method with $\varepsilon = 0 . 1 6$ and heatmap obtained using the baseline Analyze Gauss with $\varepsilon = 0 . 1 6$ and rank 20.

## F.4 Additional DP-PCA Comparison: SVHN

Figure 8 shows the comparison of Analyze Gauss approximation and the DP low rank approximation of the covariance $X X ^ { \top }$ when applied to SVHN.

![](images/a0d9a74b9876d3a25805da32b7c6e68c69f249e9c3983499a8892a0c70ec94bd.jpg)  
Figure 8: Comparison of our method against the central DP-PCA method of Dwork et al. (2014) on SVHN $\left( p = 3 0 7 2 \right)$ , with target subspace dimension $k \in \{ 1 0 , 2 0 , 4 0 \}$ . The vertical axis reports the relative Frobenius error of the rank-k covariance approximation, the horizontal axis ε for $\delta = 1 0 ^ { - 5 }$ . The results are means of 5 independent runs.

## G Complete Experimental Results for Section 7

## G.1 Number of Communication Rounds vs. Final Accuracy

Cifar-10 with ViT Features (p = 768): accuracy vs rounds

![](images/95fcdc1fc15c613c253acd7ae257c7c5a524002ab41d93e4327aca0ed3db4f83.jpg)  
Figure 9: CIFAR-10 total communication rounds vs. accuracy.

![](images/d643fa6b1c22246ecf063aaacda7a5118aa4785c61aed22858d36abbc3fde9d3.jpg)  
Figure 10: CIFAR-100 total communication rounds vs. accuracy.

![](images/13959dcd68ef15bad47805ae143c4185881ca4381cc80a7842291c3cf8d8a1ce.jpg)  
Figure 11: FashionMNIST total communication rounds vs. accuracy.

## G.2 Total Transmitted Parameters vs. Final Accuracy

Cifar-10 with ViT Features (p = 768): accuracy vs communication cost

![](images/ae03d938bcbf6c4ea53c71b985ff83855cce996aa57cdf3340cc3970d2b606fb.jpg)  
Figure 12: CIFAR-10 total transmitted parameters vs. accuracy.

![](images/3adc521b7aade8de34ad9cc997dc927de42b4db008fc491e6fd79e3ebc61a502.jpg)  
Figure 13: CIFAR-100 total transmitted parameters vs. accuracy.

FashionMNIST with ResNet features (p = 512): accuracy vs communication cost

![](images/f28638540ec8f52dcdedb11ffa4975c9ad797b094466af4346de47429197d5dc.jpg)  
Figure 14: FashionMNIST total transmitted parameters vs. accuracy.

## H Pseudocodes for Algorithms

Algorithm 1 DP Power Method   
Require: Dataset $S = \{ \mathbf { x } _ { i } \in \mathbb { R } ^ { p } \} _ { i = 1 } ^ { n }$ , noise parameter $\sigma ,$ number of iterations $T$ , block size k   
1: Initialize: Choose $\mathbf { V } _ { 0 } \in \mathbb { R } ^ { p \times k }$ with orthonormal columns (e.g. $\mathbf { V } _ { 0 } = \operatorname { Q R } ( \mathbf { G } )$ for $G _ { i j } \sim \mathcal { N } ( 0 , 1 ) )$   
2: C: public bound on $\| \mathbf { x } _ { i } \| _ { 2 }$ (data clipped to norm $C )$   
3: M $\textstyle \gets \sum _ { i = 1 } ^ { n } \mathbf { x } _ { i } \mathbf { x } _ { i } ^ { \top }$   
4: for $t = 1 , 2 , \dots , T$ do   
5: Sample $\mathbf { Z } _ { t } \in \mathbb { R } ^ { p \times k }$ with entries i.i.d. $\mathcal { N } ( 0 , 1 )$   
6: $\mathbf { V } _ { t } ^ { \prime } \gets \mathbf { M } \mathbf { V } _ { t - 1 } + C ^ { 2 } \sigma \mathbf { Z } _ { t }$   
7: $[ \mathbf { Q } _ { t } , \mathbf { R } _ { t } ] \longleftarrow \mathrm { Q R } ( \mathbf { V } _ { t } ^ { \prime } )$ reduced ${ \mathrm { Q R } } ; \mathbf { Q } _ { t } ^ { \top } \mathbf { Q } _ { t } = \mathbf { I } _ { k }$   
8: $\mathbf { V } _ { t } \gets \mathbf { Q } _ { t }$   
9: end for   
10: return $\mathbf { V } _ { T }$

Algorithm 2 Federated DP Power Method with Server-Side QR-Orthogonalization   
Require: N clients with local datasets $S = \{ \mathbf { X } _ { i } \in \mathbb { R } ^ { p \times n _ { i } } \} _ { i = 1 } ^ { N }$ , privacy $( \varepsilon , \delta )$ , iterations $T ,$ block size k   
1: Server initialization: Choose $\mathbf { V } _ { 0 } ~ \in ~ \mathbb { R } ^ { p \times k }$ with orthonormal columns (e.g. $\mathbf { V } _ { \mathrm { 0 } } = \mathrm { Q R } ( \mathbf { G } )$ $G _ { i j }$ 2   
2   
$\mathcal { N } ( 0 , 1 ) )$   
2: Let $\begin{array} { r } { n = \sum _ { i = 1 } ^ { N } n _ { i } } \end{array}$ Let $C$ be a public bound on the 2-norm of the data samples (enforced by clipping).   
3: for $t = 1 , 2 , \dots , T$ do   
4: Server broadcasts $\mathbf { V } _ { t - 1 }$ to all clients   
5: for each client $m = 1 , \ldots , N$ in parallel do   
6: Compute local moment:   
$\mathbf { M } ^ { ( m ) } \gets \sum _ { j = 1 } ^ { n _ { m } } \mathbf { x } _ { j } ^ { ( m ) } ( \mathbf { x } _ { j } ^ { ( m ) } ) ^ { \top }$   
7: Sample $\mathbf { Z } _ { t } ^ { ( m ) } \in \mathbb { R } ^ { p \times k }$ with entries i.i.d. $\mathcal { N } ( 0 , 1 )$   
8: Form local contribution:   
$\mathbf { Y } _ { t } ^ { ( m ) } \gets \mathbf { M } ^ { ( m ) } \mathbf { V } _ { t - 1 } + C ^ { 2 } \frac { \sigma } { \sqrt { N } } \mathbf { Z } _ { t } ^ { ( m ) }$   
9: Send $\mathbf { Y } _ { t } ^ { ( m ) }$ to server (optionally via secure aggregation)   
10: end for   
11: Server aggregates: $\begin{array} { r } { \mathbf { V } _ { t } ^ { \prime }  \sum _ { m = 1 } ^ { N } \mathbf { Y } _ { t } ^ { ( m ) } } \end{array}$   
12: Server QR step: $[ \mathbf { Q } _ { t } , \mathbf { R } _ { t } ] \longleftarrow \mathrm { Q } \bar { \mathrm { R } } ( \mathbf { V } _ { t } ^ { \prime } )$ , (reduced $\mathrm { Q R } ; \mathbf { Q } _ { t } ^ { \top } \mathbf { Q } _ { t } = \mathbf { I } _ { k } )$   
13: Server sets $\mathbf { V } _ { t } \gets \mathbf { Q } _ { t }$   
14: end for   
15: return $\mathbf { V } _ { T }$

Algorithm 3 Federated Private Estimation of the Covariance based on Gaussian mechanism (Dwork et al..   
2014)   
Require: N clients with local examples $\{ \mathbf { x } _ { i } ^ { ( m ) } \in \mathbb { R } ^ { p } \} _ { i = 1 } ^ { n _ { m } }$ , orthonormal basis $\mathbf { V } _ { k } \in \mathbb { R } ^ { p \times k }$ , privacy $( \varepsilon , \delta )$ , max   
2-norm (or clipping bound) $C > 0$   
1: Server: set $\begin{array} { r } { n \gets \sum _ { m = 1 } ^ { N } n _ { m } } \end{array}$ and broadcast $\mathbf { V } _ { k }$ to all clients   
2: for each client $m = 1 , \ldots , N$ in parallel do   
3: Project:   
${ \bf Y } _ { k } ^ { ( m ) } \gets { \bf V } _ { k } ^ { \top } \Big ( \frac { 1 } { n } \sum _ { i = 1 } ^ { n _ { m } } { \bf x } _ { i } ^ { ( m ) } { \bf x } _ { i } ^ { ( m ) \top } \Big ) { \bf V } _ { k } \quad \in \mathbb { R } ^ { k \times k }$   
4: Draw $\mathbf { Z } ^ { ( m ) } \in \mathbb { R } ^ { k \times k }$ symmetric with i.i.d. $\mathcal { N } ( 0 , 1 )$ entries on the upper triangular part:   
$Z _ { i j } ^ { ( m ) } \stackrel { \mathrm { i . i . d . } } { \sim } N ( 0 , 1 )$ for $i \leq j ,$ $Z _ { j i } ^ { ( m ) } = Z _ { i j } ^ { ( m ) }$ for $i < j .$   
5: Compute DP release:   
$\widetilde { \mathbf { Y } } _ { k } ^ { ( m ) } \gets \mathbf { Y } _ { k } ^ { ( m ) } + \frac { C ^ { 2 } \sigma ^ { \prime } } { n \sqrt { N } } \mathbf { Z } ^ { ( m ) } .$   
6: Upload $\widetilde { \mathbf { Y } } _ { k } ^ { ( m ) }$ to the server   
7: end for   
8: Server aggregation: $\begin{array} { r } { \widetilde { \mathbf { Y } } _ { k } \gets \sum _ { m = 1 } ^ { N } \widetilde { \mathbf { Y } } _ { k } ^ { ( m ) } } \end{array}$   
9: return $( \mathbf { V } _ { k } , \widetilde { \mathbf { Y } } _ { k } )$ (the rank-k private approximation $\mathbf { V } _ { k } \widetilde { \mathbf { Y } } _ { k } \mathbf { V } _ { k } ^ { \top }$ of the covariance is formed locally if   
needed)

Algorithm 4 Private Rank Selection via Random Projection & Gaussian mechanism   
Require: N clients with local vectors $\{ \mathbf { x } _ { i } ^ { ( m ) } \in \mathbb { R } ^ { p } \} _ { i = 1 } ^ { n _ { m } }$ , privacy $( \varepsilon , \delta )$ , projection dimension r, clipping bound   
$\bar { C } > 0 ,$ failure level $\rho \in ( 0 , 1 )$   
1: Server broadcasts a seed and each client forms the (same) projector:   
$\mathbf { G } \in \mathbb { R } ^ { p \times r } , \quad \mathbf { G } _ { i j } \sim { \mathcal { N } } ( 0 , 1 / r )$   
2: for each client $m = 1 , \ldots , N$ in parallel do   
3: Project and clip:   
$\widetilde { \mathbf { x } } _ { i } ^ { ( m ) } \gets \mathrm { c l i p } _ { \bar { C } } ( \mathbf { G } ^ { \top } \mathbf { x } _ { i } ^ { ( m ) } ) \in \mathbb { R } ^ { r }$   
4: Compute the covariance:   
$\mathbf { Y } _ { r } ^ { ( m ) } \gets \sum _ { i = 1 } ^ { n _ { m } } \widetilde { \mathbf { x } } _ { i } ^ { ( m ) } \widetilde { \mathbf { x } } _ { i } ^ { ( m ) \top } \in \mathbb { R } ^ { r \times r }$   
5: Draw client noise $\mathbf { Z } ^ { ( m ) } \in \mathbb { R } ^ { r \times r }$ symmetric with i.i.d. $\mathcal { N } ( 0 , 1 )$ entries on and above the diagonal:   
$Z _ { i j } ^ { ( m ) } \stackrel { \mathrm { i . i . d . } } { \sim } { \mathcal { N } } ( 0 , 1 ) \mathrm { f o r } i \leq j , Z _ { j i } ^ { ( m ) } = Z _ { i j } ^ { ( m ) } \mathrm { f o r } i < j .$   
6: Compute DP release:   
$\widetilde { { \mathbf Y } } _ { r } ^ { ( m ) } \gets { \mathbf Y } _ { r } ^ { ( m ) } + \frac { \sigma } { \sqrt { N } } { \mathbf Z } ^ { ( m ) } ,$   
where $\sigma$ is adjusted by Lemma $^ { 7 . }$   
7: Upload $\widetilde { { \mathbf Y } } _ { r } ^ { ( m ) }$   
8: end for   
9: Aggregation at the server: $\begin{array} { r } { \widetilde { \mathbf { Y } } _ { r }  \sum _ { m = 1 } ^ { N } \widetilde { \mathbf { Y } } _ { r } ^ { ( m ) } } \end{array}$   
10: Select the rank: Set   
$\theta _ { \mathrm { n o i s e } } ~  ~ ( 2 { \sqrt { r } } + t ) \sigma ,$   
where   
$t  2 { \sqrt { \log ( 2 / \rho ) } } .$   
Compute the spectrum:   
$\lambda _ { 1 } ( \widetilde { { \mathbf Y } } _ { r } ) \ge \cdots \ge \lambda _ { r } ( \widetilde { { \mathbf Y } } _ { r } ) .$   
Define   
$\mathcal { T }  \Big \{ i \in \{ 1 , \dots , r \} : \lambda _ { i } ( \widetilde { \mathbf { Y } } _ { r } ) > \theta _ { \mathrm { n o i s e } } \Big \} ,$   
and set   
${ \widehat { k } }  \vert { \mathcal { I } } \vert .$   
43   
11: return The estimated rank $\widehat { k }$