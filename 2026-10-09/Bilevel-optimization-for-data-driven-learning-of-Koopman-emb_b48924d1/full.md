# Bilevel optimization for data-driven learning of Koopman embeddings using kernel-based autoencoders

Joel-Pascal Ntwali N’konzi,<sup>1,</sup> <sup>a)</sup> Feliks Nüske,<sup>2,</sup> <sup>3,</sup> <sup>b)</sup> and Stefan Klus<sup>4,</sup> <sup>c)</sup>

<sup>1)</sup>Maxwell Institute for Mathematical Sciences, The University of Edinburgh & Heriot–Watt University, Edinburgh, United Kingdom

<sup>2)</sup>Max Planck Institute for Dynamics of Complex Technical Systems, Magdeburg, Germany

<sup>3)</sup>LAAS-CNRS, Université de Toulouse, Toulouse, France

<sup>4)</sup>School of Mathematical & Computer Sciences, Heriot–Watt University, Edinburgh, United Kingdom

(Dated: 9 October 2026)

Koopman operator theory provides a linear framework for analyzing nonlinear dynamical systems and has become a major tool for data-driven modeling. A central challenge, however, is that finite-dimensional approximations computed by methods such as extended dynamic mode decomposition (EDMD) require the dictionary to be specified a priori. Recent machine-learning approaches address this limitation by learning the dictionary from data, predominantly using artificial neural network (ANN) autoencoder architectures. Although kernel methods ofer an alternative with greater interpretability and tractability for theoretical analysis, they have received little attention in this setting. We introduce extended dynamic mode decomposition with kernel-based dictionary learning (EDMD-kDL), a kernel-based method for learning finite-dimensional Koopman embeddings directly from data. The method combines ideas from collocation methods and bilevel optimization to simultaneously learn a kernel dictionary and the corresponding Koopman approximation. We evaluate EDMD-kDL against state-of-the-art ANN-based approaches on a range of numerical experiments, including global sea-surface-temperature forecasting and learning directly from video data. Across all tested settings, EDMD-kDL achieves performance comparable to or better than the ANN-based methods. Moreover, in contrast to standard kernel methods, the proposed approach is scalable to large datasets by design since the size of the required kernel matrices depends on the number of collocation points rather than the size of the training dataset.

The Koopman operator approach to dynamics, which provides a way of treating nonlinear dynamical systems with linear techniques, is at the forefront of the exploding field of data-driven modeling and analysis of complex systems. The recent rise in prominence is mostly due to the development of algorithms, such as extended dynamic mode decomposition (EDMD) and its variants, for obtaining finite-dimensional approximations of the Koopman operator from trajectory data. Despite its great success, the EDMD algorithm is limited by the constraint that a suitable subspace in which the operator approximation is carried out has to be determined a priori, which is a challenging task in practice. This has prompted the development of various alternative approaches that aim to identify such subspaces directly from data using machine learning. The vast majority of these methods rely on artificial neural network (ANN)-based autoencoders, where the encoder parametrizes a mapping from the state space to a higher dimensional embedding space, which in turn determines the subspace of interest.

Such mappings are then called Koopman embeddings. Despite the fact that kernel methods can be as expressive as ANNs, and are both more interpretable and amenable to theoretical analysis, there has been no attempt to apply them to this problem. In this paper, we develop a novel kernelbased algorithm for learning finite-dimensional Koopman embeddings from data that we call extended dynamic mode decomposition with kernelbased dictionary learning (EDMD-kDL). We derive the proposed method by leveraging ideas from collocation methods and bilevel optimization, and compare its performance against stateof-the-art ANN-based approaches using both simulated and real-world datasets.

## I. INTRODUCTION

Dynamical systems, whether deterministic or stochastic, are ubiquitous in science and engineering due to their efectiveness in modeling a wide range of natural phenomena. In contrast to the standard geometric approach to dynamics, the Koopman perspective focuses on the temporal evolution of observables, which are scalar functions of the state. The Koopman operator <sup>1–4</sup> is linear, albeit infinite-dimensional, even when the state-space dynamics

are nonlinear.

Despite the framework being almost a century old, its recent gain in prominence is mostly due to the development, over the past two decades, of data-driven learning algorithms that yield finite-dimensional approximations of the Koopman operator or its associated spectral information from simulation or observational data; see, e.g., Refs. 5–10. This includes in particular the extended dynamic mode decomposition (EDMD) algorithm<sup>6,7</sup> and the variational approach to conformation dynamics (VAC)<sup>10,11</sup>. As the name suggests, EDMD is an extension of dynamic mode decomposition (DMD)<sup>5</sup>, an approach that was initially introduced for decomposing fluid flows into coherent patterns, whereas VAC generalizes time-lagged independent component analysis (TICA)<sup>12</sup>, see Refs. 7, 9, 13–17 and references therein for more details. These advances have placed the operator approach to dynamics at the forefront of the exploding field of data-driven modeling and analysis of dynamical systems. Applications include system identification<sup>18–20</sup>, model reduction<sup>9,21,22</sup>, model predictive control<sup>23–27</sup>, and time-series forecasting<sup>28–32</sup>.

The EDMD algorithm works by considering a dictionary of functions defining a so-called Koopman embedding, i.e., a mapping from the state space to a typically high-dimensional space such that the embedded dynamical system is, at least approximately, linear. Then, given training data, the matrix representing the operator is estimated by solving a linear regression problem, and approximate eigenvalues and eigenfunctions of the Koopman operator are obtained from the spectral decomposition of this matrix<sup>6</sup>.

The main limitation of standard EDMD comes from the fact that designing informative Koopman embeddings is in general challenging. The same is true for the standard VAC algorithm. This has prompted the development of various dictionary learning approaches for EDMD/VAC in the literature<sup>8,30,33–38</sup>. At the core, these methods typically consider autoencoder artificial neural networks (ANNs), where the encoder parametrizes a Koopman embedding map and the (possibly fixed) decoder provides a way of reconstructing inputs from the corresponding embedding representations. The network is then trained with a linear embedded dynamics objective in addition to the standard encoder–decoder loss. Diferences among these approaches include the use of hybrid dictionaries<sup>8</sup>, stability promotion<sup>39</sup>, training with a multi-step prediction loss<sup>37</sup>, learning consistent forward and backward linear models in the embedding space<sup>30</sup>, and the focus on stochastic systems<sup>34,35</sup>.

An alternative approach for tackling the dictionary selection problem is kernel EDMD<sup>40,41</sup>, which has been derived in two separate ways: the first strategy is to apply the well-known kernel trick to a dual formulation of the standard EDMD algorithm<sup>40</sup>, whereas the second approach considers an empirical estimation of the Koopman operator defined on a reproducing kernel Hilbert space (RKHS)<sup>41</sup>. The distinguishing feature of this approach is that the embedding map can, in theory, be infinitedimensional since only inner products between embedded vectors – which can be obtained by evaluating a symmetric positive semi-definite kernel – are required. In practice, however, one is constrained to work in the finitedimensional subspace spanned by the kernel centered at the training data points.

The main theoretical underpinning of ANN-based dictionary learning for EDMD is the universal approximation theorem<sup>42</sup>, which guarantees that any multivariate continuous function defined on a hypercube can be uniformly approximated with arbitrary accuracy by an appropriate ANN. However, this property is not specific to ANNs. In fact, the RKHSs associated with so-called universal kernels are dense in the space of continuous functions defined on compact domains with respect to the uniform norm<sup>43,44</sup>. This prompts one to wonder whether kernel methods can be as efective as ANN-based approaches for dictionary learning within EDMD/VAC. In this work, we show that this is indeed the case by developing a novel kernel-based approach for learning finite-dimensional Koopman embeddings from data, with a focus on EDMD for deterministic dynamical systems. To achieve this, we introduce the concept of kernel-based autoencoders by leveraging ideas from collocation methods<sup>45,46</sup> and bilevel optimization. The resulting method, which we call extended dynamic mode decomposition with kernel-based dictionary learning (EDMD-kDL), is less parametrized than ANN-based counterparts and relies on numerically solving a nonlinear optimization problem involving the values taken by the Koopman embedding map at a finite set of collocation points. Furthermore, we compare EDMD-kDL against state-of-the-art ANNbased approaches on a wide range of numerical experiments using both synthetic and real-world data. We find that, across settings, EDMD-kDL either outperforms or has comparable performance to ANN-based methods.

The rest of this paper is organized as follows. In Section II we provide the necessary background material on Koopman learning. This is followed by detailed derivations of the proposed method in Section III. We consider both the case of hybrid dictionaries, where one part of the dictionary is fixed whilst the other is learned from data, and the case of fully learned dictionaries. Moreover, we show how the approach can be easily modified to use a multi-step prediction loss for training, which is not possible for standard kernel EDMD. We then provide results from extensive numerical experiments in Section IV and conclude with a discussion of open problems and ideas for future work in Section V.

## II. BACKGROUND ON KOOPMAN LEARNING

## A. Koopman operator theory

In this work, we consider discrete-time dynamical systems of the form

$$
\begin{array} { r } { \pmb { x } _ { t + 1 } = \pmb { f } ( \pmb { x } _ { t } ) , \quad t \in \mathbb { N } , } \end{array}\tag{1}
$$

where $\pmb { x } _ { t } \in \Omega \subseteq \mathbb { R } ^ { n }$ is the state and $f \colon \Omega  \Omega$ is a nonlinear map. The Koopman operator <sup>1</sup> , defined by

$$
\left[ \mathcal { K } \psi \right] ( \pmb { x } ) = \psi \left( \pmb { f } ( \pmb { x } ) \right) , \quad \forall \pmb { x } \in \Omega ,\tag{2}
$$

determines the time-evolution of observables $\psi ~ \in ~ \mathbb { M }$ which are real- or complex-valued functions of the state. A typical choice is $\bar { { \mathbb { M } } } \ = \ L ^ { 2 } ( \Omega , \mu )$ for some positive measure $\mu ,$ see, e.g., Refs. 4 and 6. For vectorvalued observables $\pmb { \psi } = [ \psi _ { 1 } , \ldots , \psi _ { N } ] ^ { \top } : \Omega  \mathbb { C } ^ { N }$ , the Koopman operator acts element-wise, i.e., $\kappa \psi ( x ) =$ $\left[ { \mathcal { K } } \psi _ { 1 } ( x ) , \ldots , { \mathcal { K } } \psi _ { N } ( x ) \right] ^ { \top }$

The main property of the Koopman operator is its linearity. This is the case even when f is nonlinear. As a consequence, it is amenable to spectral decomposition. In particular, the spectral information of $\kappa$ encodes important properties of the dynamical system such as the dominant time scales, coherent patterns, and metastabil-$\operatorname { i t y } ^ { 4 , 5 , 9 , 4 7 , 4 8 }$ . Eigenfunctions of $\kappa$ are observables $\varphi$ for which the equality

$$
\kappa \varphi = \lambda \varphi
$$

holds for an associated eigenvalue $\lambda \in \mathbb { C } .$ . It follows that subspaces of M spanned by sets of eigenfunctions are invariant under the action of the Koopman operator. This characterization makes it useful to linearly decompose observables in terms of eigenfunctions. In particular, for the so-called full-state observable $\mathbf { 0 } ( { \pmb x } ) = { \pmb x }$ and a set $\{ ( \lambda _ { j } , \varphi _ { j } ) \} _ { j \in J }$ of eigenpairs of $\kappa ,$ , the Koopman mode decomposition<sup>4,47</sup> is defined as

$$
\pmb { g } ( \pmb { x } ) = \sum _ { j \in J } \pmb { w } _ { j } \varphi _ { j } ( \pmb { x } ) ,
$$

where the vectors $w _ { j } \in \mathbb { C } ^ { n }$ are called Koopman modes. Here, the index set J is possibly infinite-dimensional. That is, depending on the system, it might not be possible to write the full-state observable as a linear combination of finitely many eigenfunctions. The relevance of this decomposition stems from the fact that, at least in theory, one can write

$$
\begin{array} { l } { { \displaystyle { \pmb x } _ { t + 1 } = { \pmb g } ( { \pmb x } _ { t + 1 } ) = [ { \boldsymbol \mathcal \chi } { \pmb g } ] ( { \pmb x } _ { t } ) } } \\ { = \displaystyle \sum _ { j \in J } { \pmb w } _ { j } { \boldsymbol K } \varphi _ { j } ( { \pmb x } _ { t } ) = \displaystyle \sum _ { j \in J } \lambda _ { j } { \pmb w } _ { j } \varphi _ { j } ( { \pmb x } _ { t } ) , } \end{array}
$$

or, more generally, for $\ell \geq 1$ time steps:

$$
\pmb { x } _ { t + \ell } = \sum _ { j \in J } \lambda _ { j } ^ { \ell } \pmb { w } _ { j } \varphi _ { j } ( \pmb { x } _ { t } ) .
$$

Despite its attractive linearity, the Koopman operator poses multiple challenges due to its infinite dimensionality such as potentially possessing a continuous spectrum<sup>4,17</sup>. From a data-driven perspective, the main interest is to obtain a finite dimensional approximation of the Koopman operator from trajectory data. Such an approximation can then be used for prediction and to obtain information on the dominant modes and associated timescales of the system. We describe various methods that have been developed for this purpose in what fol lows.

## B. Extended dynamic mode decomposition (EDMD)

The EDMD algorithm<sup>6</sup> is a data-driven approach that provides a finite-dimensional approximation of $\kappa .$ , its eigenvalues, eigenfunctions, and the associated Koopman modes based on data of the form $\left\{ \left( \pmb { x } ^ { ( i ) } , \pmb { y } ^ { ( i ) } \right) \right\} _ { i = 1 } ^ { m }$ , where each pair is such that $\pmb { y } ^ { ( i ) } = \pmb { f } ( \pmb { x } ^ { ( i ) } )$ The dataset is typically arranged into two data matrices

$$
X = \left[ \pmb { x } ^ { ( 1 ) } \ldots \pmb { x } ^ { ( m ) } \right] , \quad Y = \left[ \pmb { y } ^ { ( 1 ) } \ldots \pmb { y } ^ { ( m ) } \right] ,\tag{3}
$$

where $Y$ is the one-step forward version of X. This includes data coming from a single or multiple trajectories.

Let $\mathbb { D } = \{ \psi _ { j } \} _ { j = 1 } ^ { \breve { N } }$ be a dictionary of real-valued observables and define the associated embedding map as $\pmb { x } \mapsto \pmb { \psi } ( \pmb { x } ) = \left[ \psi _ { 1 } ( \pmb { x } ) , \dots , \psi _ { N } ( \pmb { x } ) \right] ^ { \top } \in \mathbb { R } ^ { N }$ with $N \geq n .$ In practice, it is required that the full-state observable is contained in $\mathbb { V } = \operatorname { s p a n } \mathbb { D } \subset \mathbb { M }$ , the linear span of the dictionary. If V is Koopman-invariant, the dynamics in the embedded space is linear, i.e.,

$$
\psi ( { \pmb x } _ { t + 1 } ) = K \psi ( { \pmb x } _ { t } ) , \quad t \in \mathbb { N } ,
$$

for some matrix $K \in \mathbb { R } ^ { N \times N }$ . Hence, the matrix K fully determines the action of the Koopman operator restricted to the subspace V. EDMD then finds an estimate $\widehat { K }$ of K from data by solving the linear regression problem

$$
\widehat { K } = \underset { K \in \mathbb { R } ^ { N \times N } } { \arg \operatorname* { m i n } } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left\| \psi \big ( \pmb { y } ^ { ( i ) } \big ) - K \pmb { \psi } \big ( \pmb { x } ^ { ( i ) } \big ) \right\| _ { 2 } ^ { 2 } .\tag{4}
$$

Defining the transformed data matrices

$$
\begin{array} { r } { \Psi _ { X } = \left[ \psi \big ( \pmb { x } ^ { ( 1 ) } \big ) ~ . . . ~ \psi \big ( \pmb { x } ^ { ( m ) } \big ) \right] , } \\ { \Psi _ { Y } = \left[ \psi \big ( \pmb { y } ^ { ( 1 ) } \big ) ~ . . . ~ \psi \big ( \pmb { y } ^ { ( m ) } \big ) \right] , } \end{array}
$$

the (minimum-norm) solution of (4) is given by

$$
\widehat K = \Psi _ { Y } \Psi _ { X } ^ { \dagger } = \Psi _ { Y } \Psi _ { X } ^ { \top } \left( \Psi _ { X } \Psi _ { X } ^ { \top } \right) ^ { \dagger } ,\tag{5}
$$

where <sup>†</sup> denotes the Moore–Penrose pseudoinverse. Moreover, EDMD provides up to $N$ approximate eigenpairs $\{ ( \lambda _ { j } , \varphi _ { j } ) \} _ { j = } ^ { N }$ of with

$$
\varphi _ { j } ( x ) = { v } _ { j } ^ { \top } \psi ( x ) ,
$$

where $v _ { j }$ is a left eigenvector of $\widehat { K }$ associated with the eigenvalue $\lambda _ { j } .$ . Koopman modes can similarly be approximated, see Ref. 6 for more details. It has been shown that the EDMD matrix $\widehat { K }$ converges to the Galerkin projection of onto $\mathbb { V }$ in the infinite-data limit<sup>6,7</sup>.

## C. Reproducing kernel Hilbert spaces and kernel EDMD

As already mentioned above, the choice of dictionary has a significant impact on the performance of the EDMD algorithm. This is due to the assumption that the dictionary spans, at least approximately, a Koopman-invariant subspace. However, handcrafting such dictionaries in practice has proven to be a challenging task. The problem becomes even more severe in high-dimensional settings, where even seemingly simple dictionaries – such as those constituted of all polynomials up to a fixed degree – explode combinatorially with the dimension of the state space. To alleviate this issue, the authors of Ref. 40 proposed to leverage the well-known kernel trick to perform EDMD with a possibly infinite-dimensional dictionary that is implicitly defined by a chosen kernel. The resulting method is called kernel EDMD. We summarize this approach below using the more general derivation of Ref. 41, but we need to introduce the required concepts first.

Definition II.1 (Positive semi-definite kernel). Let X be a nonempty set. A function $k \colon \mathbb { X } \times \mathbb { X } \to \mathbb { R }$ is a positive semi-definite (PSD) kernel if

1. k is symmetric, i.e., it holds that $k ( x , y ) = k ( y , x )$ for all $x , y \in \mathbb { X } ,$

2. for any finite number of points $x ^ { ( 1 ) } , \ldots , x ^ { ( m ) } \in \mathbb { X }$ the Gram matrix

$$
\boldsymbol { G } = \left[ \begin{array} { c c c } { k \big ( x ^ { ( 1 ) } , x ^ { ( 1 ) } \big ) } & { \ldots } & { k \big ( x ^ { ( 1 ) } , x ^ { ( m ) } \big ) } \\ { \vdots } & { \ddots } & { \vdots } \\ { k \big ( x ^ { ( m ) } , x ^ { ( 1 ) } \big ) } & { \ldots } & { k \big ( x ^ { ( m ) } , x ^ { ( m ) } \big ) } \end{array} \right]
$$

is symmetric $P S D , \ i . e . , \ c ^ { \top } G c \geq 0$ for all $c \in \mathbb { R } ^ { m }$ and for all $m \in \mathbb { N }$

Definition II.2 (Reproducing kernel Hilbert space<sup>49,50</sup>). Let X be a nonempty set and H a Hilbert space of functions $f \colon \mathbb { X } \to \mathbb { R }$ . Then H is called a reproducing kernel Hilbert space (RKHS) with inner product $\langle \cdot , \cdot \rangle$ and induced norm $\| f \| : = { \sqrt { \langle f , f \rangle } }$ if there exists a function k : $\mathbb { X } \times \mathbb { X } \to \mathbb { R }$ , called reproducing kernel, with the following properties:

1. k has the reproducing property

$$
\langle f , k ( x , \cdot ) \rangle = f ( x ) \ f o r \ a l l \ f \in \mathbb { H } ;
$$

in particular,

$$
\langle k ( x , \cdot ) , k ( x ^ { \prime } , \cdot ) \rangle = k ( x , x ^ { \prime } ) .
$$

2. k spans H, $\therefore e . , \mathbb { H } = \overline { { \mathrm { s p a n } \{ k ( x , \cdot ) \mid x \in \mathbb { H } \} } }$ , where A is denotes the completion of the metric space A.

It follows from the Moore–Aronszajn theorem that every PSD kernel k uniquely defines an RKHS for which it is the reproducing kernel; and, conversely, every reproducing kernel is positive semi-definite<sup>51</sup>. Henceforth, we assume that all kernels are PSD. The mapping $x \mapsto k ( x , \cdot )$ is typically referred to as the canonical feature map.

Consider a kernel k and let H be the associated RKHS. Assuming that H is Koopman-invariant, the authors of Ref. 41 showed that the restriction of $\kappa$ onto H, which they called kernel Koopman operator, is given by

$$
\begin{array} { r } { \mathcal { K } _ { k } = \left( \mathcal { C } _ { 0 0 } + \epsilon \mathcal { T } \right) ^ { - 1 } \mathcal { C } _ { 0 1 } , } \end{array}\tag{6}
$$

where ϵ is a positive regularization parameter, $\mathcal { T }$ is the identity operator, $\mathcal { C } _ { 0 0 }$ is the covariance operator between two observables at time $t ,$ and $\mathcal { C } _ { 0 1 }$ is the cross-covariance operator between an observable evaluated at time t and another observable at time t+1; see Ref. 41 for the precise mathematical definitions. Then, for the data matrices X and $Y$ as in (3), it was shown that the empirical estimate is provided by the finite-rank operator

$$
\widehat { \mathcal { K } } _ { k } = \pmb { k } _ { X } ^ { \top } \left( G _ { X X } + m \epsilon I \right) ^ { - 1 } \pmb { k } _ { Y } ,\tag{7}
$$

and approximate eigenfunctions can be obtained by solving the eigenvalue problem

$$
\left( G _ { X X } + m \epsilon I \right) ^ { - 1 } G _ { Y X } \pmb { v } = \lambda \pmb { v } ,\tag{8}
$$

where $\begin{array} { r l r } { G _ { X X } } & { { } = } & { \left( k \big ( \pmb { x } ^ { ( i ) } , \pmb { x } ^ { ( j ) } \big ) \right) _ { i , j = 1 } ^ { m } } \end{array}$ and $\begin{array} { r l } { G _ { Y X } } & { { } = } \end{array}$ $\left( k \left( \pmb { y } ^ { ( i ) } , \pmb { x } ^ { ( j ) } \right) \right) _ { i , j = 1 } ^ { m }$ are kernel matrices, and

$$
\pmb { k } _ { X } = \left[ \begin{array} { c } { k ( \pmb { x } ^ { ( 1 ) } , \cdot ) } \\ { \vdots } \\ { k ( \pmb { x } ^ { ( m ) } , \cdot ) } \end{array} \right] , \quad \pmb { k } _ { Y } = \left[ \begin{array} { c } { k ( \pmb { y } ^ { ( 1 ) } , \cdot ) } \\ { \vdots } \\ { k ( \pmb { y } ^ { ( m ) } , \cdot ) } \end{array} \right] .
$$

## D. Learning Koopman dictionaries with neural networks

Kernel EDMD relies on the assumption that the induced RKHS is Koopman-invariant. However, this might not hold in practice. An alternative approach is to simultaneously learn an approximately Koopman-invariant dictionary and the corresponding EDMD matrix directly from data. This idea was first proposed by the authors of Ref. 8, who considered hybrid dictionaries of the form

$$
\pmb { x } \mapsto \psi _ { \pmb { \theta } } ( \pmb { x } ) = \left[ \pmb { \rho } ( \pmb { x } ) ^ { \top } , \pmb { \phi } _ { \pmb { \theta } } ( \pmb { x } ) ^ { \top } \right] ^ { \top } \in \mathbb { R } ^ { N } ,\tag{9}
$$

where $\rho \colon \Omega  \mathbb { R } ^ { o }$ is a fixed component, typically the full-state observable, and $\phi _ { \theta } \colon \Omega \to  { \mathbb { R } } ^ { p }$ , with $N = o + p ,$ is a learnable complementary part parameterized by an ANN with weights and biases denoted by θ. The EDMD matrix K and the ANN parameters are then optimized si multaneously by minimizing the objective function in (4), possibly with regularization over both K and θ. Specifically, they used an iterative approach where, at each optimization step, first an exact update of K for fixed θ is made as in (5); then θ is updated using stochastic gradient descent for fixed K.

The main motivation for including a fixed component $\rho$ was so that uninformative solutions, such as ψ<sub>θ</sub> being constant and K the identity map, could be avoided. Another approach for avoiding such trivial solutions is to consider an autoencoder ANN $( \psi _ { \boldsymbol { \theta } } , \varphi _ { \omega } )$ , where the encoder ψ parametrizes the EDMD dictionary and the decoder $\varphi _ { \omega }$ maps from the embedding to the state space<sup>36,37</sup>. Then the network parameters $( \theta , \omega )$ and the Koopman matrix K are jointly optimized by minimizing a linear combination of the forward prediction loss in (4) and the encoder-decoder reconstruction loss

$$
\mathcal { L } _ { \mathrm { e n c - d e c } } ( \theta , \omega ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left. \mathbf { x } ^ { ( i ) } - \varphi _ { \omega } \left( \psi _ { \theta } \left( \mathbf { x } ^ { ( i ) } \right) \right) \right. _ { 2 } ^ { 2 } .
$$

Methods of this type are typically referred to as Koopman autoencoders (KAEs). In practice, it has been observed that using multi-step forward prediction instead of the single-step loss in (4) results in improved performance. Furthermore, it was demonstrated in Ref. 30 that additional performance gains can be made by learning matrices for both the forward and backward dynamics and enforcing that the product of the two is close to the identity (forward-backward consistency). The associated models are called consistent Koopman autoencoders (cKAEs).

## III. EDMD WITH KERNEL-BASED DICTIONARY LEARNING

We propose a novel approach, called extended dynamic mode decomposition with kernel-based dictionary learning (EDMD-kDL), for learning finite-dimensional Koopman embeddings of nonlinear dynamics from data. In contrast to standard methods that rely on ANNs, EDMD-kDL optimizes EDMD dictionaries over an RKHS associated with a user-specified kernel. Furthermore, our approach is fundamentally diferent from standard kernel EDMD in that the dimensionality of learned dictionary is independent of the number of training data points. Additionally, EDMD-kDL allows the use of a multi-step prediction objection at training stage, in contrast to kernel EDMD.

## A. Derivation of EDMD-kDL

Let H be the RKHS associated with a kernel k : $\Omega \times$ $\Omega \to \mathbb { R }$ . We are interested in learning a suitable N-valued dictionary map $\boldsymbol { \psi } \in \mathbb { H } ^ { N }$ for the data-driven approximation the Koopman operator. We require that the induced subspace contains, at least approximately, the full-state observable. To this end, we first focus on dictionaries of the form (9) and extend the approach to an autoencoder formulation in Section III B. Furthermore, we consider training with a multi-step prediction objective in Section III C.

Assuming data of the form (3), we formulate the dictionary learning problem as solving

$$
\begin{array} { r } { \{ \underset { \mathcal { F } \in \mathbb { R } ^ { N \times N } } { \operatorname* { m i n } } \ \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \| \psi \big ( \pmb { y } ^ { ( i ) } \big ) - K \psi \big ( \pmb { x } ^ { ( i ) } \big ) \| _ { 2 } ^ { 2 } + \beta \| \phi \| _ { \mathbb { H } ^ { p } } ^ { 2 } } \\ { \mathrm { K } \in \mathbb { R } ^ { N \times N } } \\ { \mathrm { s . t . } \ \psi ( \pmb { x } ) = [ \rho ( \pmb { x } ) ^ { \top } , \phi ( \pmb { x } ) ^ { \top } ] ^ { \top } , } \end{array}\tag{10}
$$

where $\beta ~ > ~ 0$ is a regularization parameter. Unless the kernel k admits a finite-dimensional feature map, (10) involves an optimization problem over an infinitedimensional function space. However, if function values for $\phi$ were available at a finite set of points, we could obtain an equivalent finite-dimensional formulation due the particular RKHS structure of H, specifically the representer theorem<sup>52,53</sup>. To obtain a finite-dimensional problem in the absence of function values and without an explicit parametrization of the RKHS (this could be achieved approximately by truncating the Mercer expansion of the kernel<sup>49</sup>, for example), we consider pseudovalues $\left\{ \pmb { u } ^ { ( j ) } \right\} _ { j = 1 } ^ { m _ { c } } \subset \mathbb { R } ^ { p }$ at a finite set of collocation points $\left\{ z ^ { ( j ) } \right\} _ { j = 1 } ^ { m _ { c } } \subset \mathcal { \bar { \Omega } } ;$ i.e., we require $\phi \big ( z ^ { ( j ) } \big ) = \mathbf { u } ^ { ( j ) }$ for each j and aim to automatically learn the optimal matrix $\mathbf { \bar { \boldsymbol { U } } } = \left\lceil \pmb { u } ^ { ( 1 ) } , \dots , \pmb { u } ^ { ( m _ { c } ) } \right\rceil$ from data. We then reformulate (10) as a bilevel optimization problem:

$$
\begin{array} { r }  \{ \begin{array} { l l } { \displaystyle \widehat { K } , \displaystyle \widehat { U } = \underset { K \in \mathbb { R } ^ { N \times N \times S } } { \arg \operatorname* { m i n } } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \| \widehat { \psi } _ { U } \big ( \pmb { y } ^ { ( i ) } \big ) - K \widehat { \psi } _ { U } \big ( \pmb { x } ^ { ( i ) } \big ) \| _ { 2 } ^ { 2 } + \beta \| \widehat { \phi } _ { U } \| _ { \mathbb { H } ^ { p } } ^ { 2 } \quad } & { \mathrm { ( o u t e r ~ o p t i m i z a t i o n ) } } \\ { \displaystyle \qquad \{ \begin{array} { l l } { \displaystyle \widehat { \phi } _ { U } = \underset { \phi \in \mathbb { R } ^ { p } } { \operatorname* { a r g m i n } } \frac { 1 } { m } \sum _ { c } ^ { m _ { c } } \| \phi \big ( z ^ { ( j ) } \big ) - \pmb { u } ^ { ( j ) } \| _ { 2 } ^ { 2 } + \gamma \| \phi \| _ { \mathbb { H } ^ { p } } ^ { 2 } } & \\ { \displaystyle \widehat { \phi } _ { U } ( \pmb { x } ) = [ \rho ( \pmb { x } ) ^ { \top } , \widehat { \phi } _ { U } ( \pmb { x } ) ^ { \top } ] ^ { \top } , } & { \mathrm { ( i n n e r ~ o p t i m i z a t i o n ) } } \end{array}  } \end{array} \end{array}\tag{11}
$$

for the inner optimization problem, we fix the matrix of pseudo-values $\bar { U }$ and optimize over the complementary dictionary ϕ. We show in what follows that this problem can be solved explicitly and the minimizer is a parametric function of the pseudo-values. This reduces the dictionary learning problem to finding the optimal U. We then jointly solve for the optimal U and the corresponding EDMD matrix K in the outer optimization problem. A similar bilevel optimization framework was used in Ref. 46 for solving nonlinear partial diferential equations with Gaussian processes. We provide more details below.

## 1. Solving the inner optimization problem

We start by observing that the p components in the inner problem are decoupled, i.e., it can be rewritten as

$$
\widehat { \phi } _ { U ; \ell } = \underset { \phi _ { \ell } \in \mathbb { H } } { \arg \operatorname* { m i n } } \frac { 1 } { m _ { c } } \sum _ { j = 1 } ^ { m _ { c } } \left( \phi _ { \ell } \big ( z ^ { ( j ) } \big ) - u _ { \ell } ^ { ( j ) } \right) ^ { 2 } + \gamma \left\| \phi _ { \ell } \right\| _ { \mathbb { H } } ^ { 2 } ,\tag{12}
$$

for $\ell = 1 , \ldots , p ,$ with $\phi _ { \ell }$ and $u _ { \ell } ^ { ( j ) }$ denoting the $\ell ^ { t h }$ components of $\phi$ and $\mathbf { \pmb { u } } ^ { ( j ) }$ , respectively. The key ingredient for solving (12) is the following result.

Theorem III.1 (Representer theorem<sup>49,52,53</sup>). Let $\mathcal { R } \colon [ 0 , \infty )  \mathbb { R }$ denote a strictly monotonically increasing function, X a set, and $\mathcal { L } \colon ( \bar { \mathbb { X } } \times \mathbb { R } ^ { 2 } ) ^ { m }  \bar { \mathbb { R } } \cup \{ \infty \}$ an arbitrary loss function. Then each minimizer $f \in \mathbb { H }$ of the regularized risk

$$
\mathcal { L } \big ( \big ( x ^ { ( i ) } , y ^ { ( i ) } , f \big ( x ^ { ( i ) } \big ) \big ) _ { i = 1 } ^ { m } \big ) + \mathcal { R } \left( \| f \| _ { \mathbb { H } } \right)
$$

admits a representation of the form

$$
f ( x ) = \sum _ { i = 1 } ^ { m } \alpha _ { i } k \bigl ( x ^ { ( i ) } , x \bigr ) .
$$

It then follows from this representer theorem that any solution of (12) must lie in the span of the kernel evaluated at the collocation points. That is, it is suficient to consider functions of the form

$$
\phi _ { \ell } \big ( \cdot \big ) = \sum _ { j = 1 } ^ { m _ { c } } q _ { j \ell } k \big ( z ^ { ( j ) } , \cdot \big ) = \mathbf { q } _ { \ell } ^ { \top } \mathbf { k } _ { Z } \big ( \cdot \big ) ,\tag{13}
$$

where $q _ { \ell } \in \mathbb { R } ^ { m _ { c } }$ is a vector of coeficients. Given this representation, it is easy to show that $\left\| \phi _ { \ell } \right\| _ { \mathbb { H } } ^ { 2 } = \pmb q _ { \ell } ^ { \top } G \pmb q _ { \ell } .$ where $G = G _ { Z Z }$ is the Gram matrix evaluated at the collocation points. Therefore, (12) is equivalent to

$$
\widehat { \pmb q } _ { \ell } = \underset { \pmb q _ { \ell } \in \mathbb { R } ^ { m _ { c } } } { \arg \operatorname* { m i n } } \widehat { \frac { 1 } { m _ { c } } \sum _ { j = 1 } ^ { m _ { c } } \left( \pmb q _ { \ell } ^ { \top } \pmb k _ { Z } \big ( \pmb z ^ { ( j ) } \big ) - \ b u _ { \ell } ^ { ( j ) } \right) ^ { 2 } + \gamma \pmb q _ { \ell } ^ { \top } G \pmb q _ { \ell } } .
$$

The objective function can be written compactly as

$$
\mathcal { I } ( \pmb { q } _ { \ell } ) = \frac { 1 } { m _ { c } } \sqrt { \left[ \begin{array} { c } { \frac { G } { { k _ { Z } ^ { \top } } \left( z ^ { ( 1 ) } \right) } } \\ { \vdots } \\ { \pmb { k } _ { Z } ^ { \top } \left( z ^ { ( m ) } \right) } \end{array} \right] } \ q _ { \ell } - \overbrace { \left[ \begin{array} { c } { u _ { \ell } ^ { ( 1 ) } } \\ { \vdots } \\ { u _ { \ell } ^ { ( m ) } } \end{array} \right] } ^ { u _ { \ell } } \bigg \Vert _ { 2 } ^ { 2 } + \gamma \pmb { q } _ { \ell } ^ { \top } G q _ { \ell } ,
$$

and its gradient with respect to $\pmb q \ell$ is given by

$$
\nabla _ { \pmb { q } _ { \ell } } \mathscr { T } = \frac { 2 } { m _ { c } } G \left( G \pmb { q } _ { \ell } - \pmb { u } _ { \ell } \right) + 2 \gamma G \pmb { q } _ { \ell } .
$$

It follows from the first-order condition $\nabla _ { q _ { \ell } } { \mathcal { I } } \left( { \widehat { \pmb q } } _ { \ell } \right) = { \mathbf 0 }$ that

$$
G \left[ \left( G + \gamma m _ { c } \operatorname { I } \right) \widehat { \pmb q } _ { \ell } - { \pmb u } _ { \ell } \right] = { \bf 0 } ,\tag{14}
$$

which admits a unique solution given by

$$
\widehat { \pmb q } \_ = \left( G + \gamma m _ { c } \mathrm { I } \right) ^ { - 1 } \pmb u _ { \ell } ,\tag{15}
$$

provided that G is invertible, where I denotes the identity matrix of dimension $m _ { c }$ . Note that even when $G$ is singular, the solution to (12) is still unique and given by

$$
\widehat { \phi } _ { U ; \ell } ( \cdot ) = \widehat { \pmb q } _ { \ell } ^ { \top } \pmb k _ { Z } ( \cdot ) ,\tag{16}
$$

despite (14) not having a unique solution.

Remark III.2. It is worth noting that although we have provided detailed derivations for the sake of completeness and to increase accessibility for readers who are not familiar with kernel methods, we could have instead relied on the fact that (12) corresponds to a standard kernel ridge regression problem<sup>54</sup>.

## 2. Solving the outer optimization problem

We now turn our focus to solving the outer optimization problem in (11). To this end, we first calculate the squared RKHS norm of $\widehat { \phi } _ { U }$ as

$$
\begin{array} { r l r } {  { \| \widehat { \boldsymbol { \phi } } _ { U } \| _ { \mathbb { H } ^ { p } } ^ { 2 } = \sum _ { \ell = 1 } ^ { p } \| \widehat { \boldsymbol { \phi } } _ { U ; \ell } \| _ { \mathbb { H } } ^ { 2 } = \sum _ { \ell = 1 } ^ { p } \widehat { q } _ { \ell } ^ { \top } G \widehat { q } _ { \ell } } } \\ & { } & { = \sum _ { \ell = 1 } ^ { p } \boldsymbol { u } _ { \ell } ^ { \top } ( G + \gamma m _ { c } \mathrm { I } ) ^ { - 1 } G ( G + \gamma m _ { c } \mathrm { I } ) ^ { - 1 } \boldsymbol { u } _ { \ell } . } \end{array}\tag{17}
$$

This expression can be written more compactly. To see that, let $G = V \Sigma V ^ { \top }$ be the (compact) singular value decomposition (SVD) of the Gram matrix G. Note that the left and right singular vectors coincide here since $G$ is symmetric and positive semi-definite. Define $L = V \Sigma ^ { 1 / 2 }$ and $Q = L ^ { \top } \left( \dot { G } + \gamma m _ { c } \mathrm { I } \right) ^ { - 1 }$ . Then $L L ^ { \top } = G$ and $Q ^ { \top } Q = \left( G + \gamma m _ { c } \mathrm { I } \right) ^ { - 1 } G \left( G + \gamma m _ { c } \mathrm { I } \right) ^ { - 1 }$ . It follows that (17) is equivalent to

$$
\begin{array} { r } { \left\| \widehat { \phi } _ { U } \right\| _ { \mathbb { H } ^ { p } } ^ { 2 } = \operatorname { t r a c e } \left[ ( Q U ^ { \top } ) ^ { \top } ( Q U ^ { \top } ) \right] = \left\| Q U ^ { \top } \right\| _ { F } ^ { 2 } , } \end{array}
$$

where $\| \cdot \| _ { F }$ denotes the Frobenius norm of a matrix.

Furthermore, we note that the variables U and $K$ in the outer optimization problem are separable in the sense that for fixed $U ,$ we can solve for $K$ exactly as in standard EDMD. Therefore, we can either directly optimize over both U and $K$ jointly or use the variable projection $m e t h o d ^ { 5 5 , 5 6 }$ to obtain

$$
\{ \begin{array} { l } { \displaystyle \widehat { U } = \underset { U \in \mathbb { R } ^ { p \times m _ { c } } } { \arg \operatorname* { m i n } } \frac { 1 } { m } \| \widehat { \Psi } _ { U , Y } - \widehat { K } _ { U } \widehat { \Psi } _ { U , X } \| _ { F } ^ { 2 } + \beta \| Q U ^ { \top } \| _ { F } ^ { 2 } , } \\ { \mathrm { s . t . } \{ \widehat { K } _ { U } = \underset { K \in \mathbb { R } ^ { ( n + p ) \times ( n + p ) } } { \arg \operatorname* { m i n } } \frac { 1 } { m } \| \widehat { \Psi } _ { U , Y } - K \widehat { \Psi } _ { U , X } \| _ { F } ^ { 2 }  } \\ { \displaystyle \qquad = \widehat { \Psi } _ { U , Y } \widehat { \Psi } _ { U , X } ^ { \top } ( \widehat { \Psi } _ { U , X } \widehat { \Psi } _ { U , X } ^ { \top } ) ^ { \dagger } , } \end{array} 
$$

as the final optimization problem over $U .$ . We use the L-BFGS optimizer<sup>57</sup>, a quasi-Newton method, with automatic diferentiation as implemented in the PyTorch library to solve (18). Alternatively, one could calculate an analytic expression for the gradient of the objective function and use standard numerical optimization routines. We summarize the approach in Algorithm 1.

(18)

Algorithm 1: EDMD-kDL algorithm   
Data: kernel $k ,$ data matrices $X , Y \in \mathbb { R } ^ { n \times m }$ , collocation points $Z \in \mathbb { R } ^ { n \times m _ { c } }$ , number of learnable dictionary elements   
$p \geq 1 ,$ , fixed component $\rho \colon \Omega  \mathbb { R } ^ { o }$ , and regularization parameters $\gamma , \beta > 0$   
Result: learned dictionary ${ \widehat { \psi } } ,$ EDMD matrix ${ \widehat { K } } ,$ and approximated Koopman spectral information   
1 Assemble the Gram matrix $\begin{array} { r } { \dot { G } = k ( Z , Z ) ; } \end{array}$   
2 Perform a (compact) SVD of $G ;$ i.e., $G = V \Sigma V ^ { \top }$   
3 $L \gets V \Sigma ^ { 1 / 2 } ;$   
4 $Q \gets L ^ { T } \left( G + \gamma m _ { c } \mathrm { I } \right) ^ { - 1 } ;$   
5 Solve (18) for the optimal $\widehat { U } = \left[ \begin{array} { c } { \widehat { \pmb { u } } _ { 1 } ^ { \top } } \\ { \vdots } \\ { \widehat { \pmb { u } } _ { p } ^ { \top } } \end{array} \right]$ $^ { \prime * }$ using L-BFGS or your preferred optimizer $^ { * / }$   
6 for $\ell = 1 , \ldots , p$ do   
7 $\widehat { \pmb q _ { \ell } } \gets ( G + \gamma m _ { c } \mathbb I ) ^ { - 1 } \widehat { \pmb u _ { \ell } }$ $^ { \prime * }$ learned dictionary elements \*/   
8 $\widehat { \phi } _ { \widehat { U } ; \ell } \gets \widehat { \pmb { q } } _ { \ell } ^ { \top } { \pmb { k } } _ { Z } ( \cdot )$   
9 $\widehat { \phi } _ { \widehat { U } } \gets \left[ \widehat { \phi } _ { \widehat { U } ; 1 } , \dots , \widehat { \phi } _ { \widehat { U } ; p } \right] ^ { \top } ;$   
10 $\widehat { \psi } \gets \left[ \bar { \rho } ^ { \top } , \widehat { \phi } _ { \widehat { U } } ^ { \top } \right] ^ { \top }$ $^ { \prime * }$ assemble full dictionary \*/   
11 $\widehat { K } \gets \bar { \Psi } _ { Y } \widehat { \Psi } _ { X } ^ { \dagger } ;$ /\* EDMD matrix \*/   
12 Compute eigenvalues, eigenfunctions, and modes as in standard EDMD

## B. EDMD-kDL with kernel-based autoencoders

We now extend our derivation to an autoencoder formulation where, instead of including fixed features, we simultaneously learn the full dictionary and a decoder map that is trained to reconstruct the full-state observable. We call this new type of architecture kernel-based autoencoders.

Since one of the prominent applications of EDMD is the decomposition of the full-state observable into a linear combination of (approximate) Koopman eigenfunctions, which requires the full-state observable to be in the linear span of the dictionary, we constrain the decoder to be linear. Specifically, we formulate the new

learning task as solving

$$
\left\{ \begin{array} { l l } { \displaystyle \mathop { \arg \operatorname* { m i n } } _ { \psi \in \mathbb { R } ^ { N } } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left\| \psi \big ( \pmb { y } ^ { ( i ) } \big ) - K \psi \big ( \pmb { x } ^ { ( i ) } \big ) \right\| _ { 2 } ^ { 2 } + \beta \left\| \pmb { \psi } \right\| _ { \mathbb { H } ^ { N } } ^ { 2 } } \\ { K \in \mathbb { R } ^ { N \times N } } \\ { D \in \mathbb { R } ^ { n \times N } } \\ { \mathrm { s . t . } \ \pmb { x } = D \psi ( \pmb { x } ) , \quad \forall \pmb { x } \in \Omega . } \end{array} \right.\tag{19}
$$

Then, following the same strategy as in Section III $\mathrm { A } ,$ we obtain a bilevel reformulation by introducing pseudovalues at a finite set of collocations points. The inner problem, with solution denoted by $\widehat { \psi } _ { U } ,$ is the same as in (11), with the diference that ϕ and $\mathbb { H } ^ { p }$ are now replaced by ψ and $\mathbb { H } ^ { N }$ , respectively. For the outer problem, the objective function is a linear combination of the following loss functions:

i) The standard EDMD loss

$$
\mathcal { L } _ { 1 } ( K , D , U ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left. \widehat { \psi } _ { U } \big ( \pmb { y } ^ { ( i ) } \big ) - K \widehat { \psi } _ { U } \big ( \pmb { x } ^ { ( i ) } \big ) \right. _ { 2 } ^ { 2 } .\tag{20}
$$

ii) The autoencoder reconstruction loss

$$
\mathcal { L } _ { 2 } ( K , D , U ) = \frac { 1 } { m _ { c } } \sum _ { j = 1 } ^ { m _ { c } } \left. z ^ { ( j ) } - D \widehat { \psi } _ { U } \big ( z ^ { ( j ) } \big ) \right. _ { 2 } ^ { 2 } ,\tag{21}
$$

which imposes the soft encoder–decoder constraint $\pmb { x } = D \pmb { \psi } ( \pmb { x } )$ at the collocation points $\left\{ z ^ { ( j ) } \right\} _ { j = 1 } ^ { m _ { c } }$ . Note that, in principle, we could choose a diferent set of points. For simplicity, however, we choose this.

iii) The RKHS regularization

$$
\mathcal { L } _ { 3 } ( K , D , U ) = \left\| \widehat { \psi } _ { U } \right\| _ { \mathbb { H } ^ { N } } ^ { 2 } = \left\| Q U ^ { \top } \right\| _ { F } ^ { 2 } .\tag{22}
$$

It is clear that if U is fixed, we can solve the optimization problems for obtaining K and D separately. Therefore, as in the previous section, we can either directly optimize over K, D, and U simultaneously, or use variable projection to obtain

$$
\widehat { U } = \underset { U \in \mathbb { R } ^ { N \times m _ { c } } } { \arg \operatorname* { m i n } } \sum _ { i = 1 } ^ { 3 } \beta _ { i } \mathcal { L } _ { i } \big ( \widehat { K } _ { U } , \widehat { D } _ { U } , U \big ) ,\tag{23}
$$

where the $\beta _ { i } \mathrm { { ^ { s } } }$ are positive weights. Here, $\widehat { K } _ { U }$ is as in (18) and $\widehat { D } _ { U }$ is given by

$$
\begin{array} { r l r } {  { \widehat { D } _ { U } = \arg \operatorname* { m i n } _ { D \in \mathbb { R } ^ { n \times N } } \frac { 1 } { m _ { c } } \sum _ { j = 1 } ^ { m _ { c } } \| z ^ { ( j ) } - D \widehat { \psi } _ { U } ( z ^ { ( j ) } ) \| _ { 2 } ^ { 2 } } } \\ & { } & { = Z \widehat { \Psi } _ { U , Z } ^ { \top } ( \widehat { \Psi } _ { U , Z } \widehat { \Psi } _ { U , Z } ^ { \top } ) ^ { \dagger } , } \end{array}\tag{24}
$$

with $Z = \left[ z ^ { ( 1 ) } , \dots , z ^ { ( m _ { c } ) } \right]$

## C. Learning with a multi-step prediction objective

Thus far, we have optimized EDMD dictionaries with a single-step prediction objective. However, it has been observed that training ANN-based Koopman autoencoders using multi-step prediction leads to higher accuracy and robustness<sup>30,37</sup>. We expect a similar pattern for the proposed framework and therefore extend it to this setting. Considering the kernel-based autoencoder formulation, we now assume we have access to training data of the form $\left\{ \left( \pmb { x } ^ { ( \ell ) } , \pmb { Y } ^ { ( \ell ) } \right) \right\} _ { \ell = 1 } ^ { m }$ , where

$$
Y ^ { ( \ell ) } = \left[ \pmb { y } _ { 1 } ^ { ( \ell ) } , \pmb { y } _ { 2 } ^ { ( \ell ) } , \dots , \pmb { y } _ { T } ^ { ( \ell ) } \right] ,
$$

with ${ \pmb y } _ { t } ^ { ( \ell ) } = { \pmb f } ^ { t } \big ( { \pmb x } ^ { ( \ell ) } \big )$ , for $t = 1 , \ldots , T .$ . Here, $T \geq 1$ is the number of forward prediction steps. Then, using pseudovalues at a set of collocation points to obtain a bilevel

formulation again, the final optimization problem can be written as

$$
\widehat K , \widehat D , \widehat U = \underset { K , D , U } { \arg \operatorname* { m i n } } \sum _ { i = 1 } ^ { 3 } \beta _ { i } \mathcal L _ { i } ( K , D , U ) ,
$$

where

$$
\mathcal { L } _ { 1 } = \frac { 1 } { m } \sum _ { \ell = 1 } ^ { m } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left. \widehat { \psi } _ { U } \big ( \pmb { y } _ { t } ^ { ( \ell ) } \big ) - K ^ { t } \widehat { \psi } _ { U } \big ( \pmb { x } ^ { ( \ell ) } \big ) \right. _ { 2 } ^ { 2 } ,
$$

and $\mathcal { L } _ { 2 }$ and $\mathcal { L } _ { 3 }$ are as in (21) and (22), respectively. Additionally, it is possible to use variable projection for the decoder matrix D as in Section III B, but not for the Koopman matrix K.

## IV. NUMERICAL EXPERIMENTS

In this section, we illustrate the performance of EDMD-kDL on a wide range of benchmark problems. We consider synthetic data from the model of an undamped nonlinear pendulum and the simulation of a fluid flow past a cylinder exhibiting vortex shedding. Addi tionally, we consider real-world datasets. We test the performance of EDMD-kDL on the task of forecasting global sea-surface temperature (SST) and predicting the next frame of a video of a pendulum. We compare the proposed kernel-based methods against state-of-the art ANN-based alternatives and EDMD with random Fourier features (EDMD-RFF)<sup>58,59</sup>, which is an approximation of the standard kernel EDMD method. In all experiments, collocation points for EDMD-kDL are generated using a Sobol sequence<sup>60</sup> and we use a Gaussian RBF kernel with bandwidth σ. In addition, we use the oficial implementations of the ANN-based approaches proposed in Ref. 30, publicly available at github.com/erichson/koopmanAE.

## A. Undamped nonlinear pendulum

The dynamics of a frictionless pendulum with no forcing is modeled by the two-dimensional Hamiltonian system

$$
\left\{ \begin{array} { l l } { \displaystyle \dot { x } _ { 1 } = x _ { 2 } , } \\ { \displaystyle \dot { x } _ { 2 } = - \frac { g } { \ell } \sin x _ { 1 } , } \end{array} \right.
$$

where $x _ { 1 }$ (in radians) is the angular position, $x _ { 2 } \left( m / s \right)$ is the velocity, $g = \dot { 9 . 8 1 } ( m / s ^ { 2 } )$ is the gravitational constant, and $\ell = 1 ( m )$ is the string’s length. We assume throughout that $x _ { 2 } ( 0 ) = 0$ . It follows that the degree of nonlinearity of the dynamics is determined by the value of the initial angular position $x _ { 1 } ( 0 )$

Following Ref. 30, we generate 1600 data points from a single trajectory simulated with time step $\Delta t = 0 . 1$ using the exact solution given in terms of elliptical functions and consider two scenarios: an almost-linear regime with $x _ { 1 } ( 0 ) = 0 . 8$ and a truly nonlinear regime with $x _ { 1 } ( 0 ) =$ 2.4. We use data from the first 600 time steps for training and the remaining 1000 steps for testing.

Throughout all of the experiments, we set the embedding dimension to $N = 6$ and the number of forward steps to $T = 8$ for EDMD-kDL and Koopman autoencoders (KAE and cKAE). For the consistent KAE approach, the number of backward steps is also set to 8. The bandwidth of the kernel is set to $\sigma = 0 . 7 5$ for EDMD-kDL and EDMD-RFF. Furthermore, we use this simple model to investigate the impact of noise on the performance of each method.

## 1. Clean data setting

We start with the noise-free data scenario. Figure 1 shows the phase portraits and prediction errors over the testing horizon. For each method, predictions are generated autoregressively by simulating the learned linear Koopman models, using the first testing point as initial condition, and then mapping to the state space using the learned decoders. We observe that for this simple problem, EDMD-RFF achieves the lowest prediction error, suggesting that the RFF dictionary – which consists of sinusoidal functions – is well-suited for the problem. However, among the dictionary learning frameworks, EDMD-kDL consistently outperforms the neural network approaches. Furthermore, despite cKAE performing clearly better than standard KAE in accordance with observations made in Ref. 30, they both fail to properly capture the underlying phase portrait of the system in the truly nonlinear regime (Figures 1c & 1d).

## 2. Noisy data setting

To investigate the robustness of the proposed method with respect to noise, we corrupt the pendulum data with mean-free Gaussian noise. To account for varying scales between the two sets of initial conditions, the noise scale $\sigma _ { \mathrm { n o i s e } }$ is set according to

$$
\sigma _ { \mathrm { n o i s e } } = \mathrm { N R } \times \frac { \| X _ { \mathrm { t r a j } } \| _ { F } } { \sqrt { n \times M } } ,
$$

where $\mathrm { N R } \in ( 0 , 1 )$ denotes the noise ratio and $X _ { \mathrm { t r a j } } \in$ R<sup>n×M</sup> denotes the complete trajectory data with $M =$ 1600. We experiment with noise ratios going from 1% to 10%, specifically $\mathrm { N R } \in \{ 0 . 0 1 , 0 . 0 5 , 0 . 1 \}$ . For the sake of fair comparison, we keep all hyperparameters fixed to the same values as in the clean data case for all methods.

a. The almost-linear regime. Figure 2 shows the phase portraits and prediction errors over time for each of the four methods. We observe that in the low-noise setting $( \mathrm { N R } ~ = ~ 1 \% )$ EDMD-kDL performs marginally worse compared to EDMD-RFF as was the case in the clean data scenario. Interestingly, however, the relative prediction error incurred by EDMD-RFF grows substantially faster than that of EDMD-kDL when the noise level is high (second and third rows). To see why this happens despite the phase portraits suggesting the opposite, we zoom in at the final quarter of the predicted trajectories. Figures 4a & 4c show that despite properly preserving the amplitude, EDMD-RFF incurs a substantial phase drift from the ground truth solution, which explains the observed high error levels. In contrast, the model learned by EDMD-kDL produces a small error in the amplitude, but this is compensated by an accurate preservation of the phase compared to EDMD-RFF, which then results in lower errors overall. Furthermore, across noise levels, EDMD-kDL performs substantially better than KAE and cKAE in terms of both the prediction error and and the phase portrait preservation.

b. The truly nonlinear regime. Figure 3 shows the phase portraits and prediction errors over time for the four methods across the three noise levels. Additionally, the right panel of Figure 4 shows the corresponding trajectories over the final quarter of the prediction horizon. The observations are similar to the almost-linear setting: EDMD-RFF best preserves the amplitude of the system. However, when the noise level is considerable, it sufers from a phase drift that results in high prediction errors. In contrast, EDMD-kDL properly balances the preservation of phase and amplitude and hence outperforms all the other methods in terms of prediction errors when the noise level is substantial. Moreover, Figure 4 shows that models learned by the ANN-based approaches produce predictions that are highly irregular even when they are trained with forward-backward consistency.

## B. Kármán vortex shedding

As a second classical benchmark problem, we consider fluid flow past a cylinder in two dimensions and choose the Reynolds number $\mathrm { R e } = 1 0 0$ . We discretize the domain using a regular grid of size $1 8 0 \times 5 2 0$ and generate $m = 5 0 0$ snapshots. We then reshape the snapshots into 93600-dimensional column vectors and assemble them into a single matrix S with columns ordered sequentially in time. In order to reduce the dimensionality, we perform an SVD of the mean-subtracted snapshot matrix; i.e., ${ \bar { \cal S } } = U \Sigma V ^ { \top }$ , and project the data onto the dominant $r = 1 9$ principal components, retaining 95% of the cumulative energy. Here, $\bar { \bar { S } }$ denotes the matrix obtained by subtracting the column-wise mean of S from each of its columns. We use the first 400 snapshots for training and the remaining 100 snapshots for testing. In the previous example, we used the auto-encoder formulation for EDMD-kDL. We now choose the full-state observable as a fixed dictionary component instead and learn a complementary part of dimension $p = 2 r$ . On the other hand, we find that KAE and cKAE perform best with a lower embedding dimension and hence set it to $N = 6$ after some manual hyperparameter tuning.

(a) Predicted phase portraits, x<sub>1</sub>(0) = 0.8.  
![](images/372a2433b2444f84e4018b0ba661d1cb5400645bb295139c970e5129e2284139.jpg)

(c) Predicted phase portraits, x<sub>1</sub>(0) = 2.4.  
![](images/6861d17e4c4f6873ca066ee8d6fd89acfefd8fced2323aa23184ba3ab4658740.jpg)

(b) Relative prediction error, x<sub>1</sub>(0) = 0.8.  
![](images/fbe08af2a714e977e84afc9f618d9c7606cf6a4bd9ee08832785ecadb618d2c1.jpg)

(d) Relative prediction error, x<sub>1</sub>(0) = 2.4.  
![](images/78fe38ad5719b14c31a55ac42cbb7f9277030e8a75d00cd78e003ea71aae7091.jpg)  
FIG. 1: Nonlinear pendulum in the clean data setting. Phase portraits and relative prediction errors over 1000 time steps of testing using learned Koopman models. The top and bottom rows correspond to the almost-linear and truly nonlinear regimes, respectively.

Figure 5 shows the obtained relative prediction errors over the testing horizon. EDMD-RFF is not included in the plot since it incurred much higher errors, in contrast to the preceding example. The corresponding predicted snapshots are shown in Figure 6. We observe that all the methods except EDMD-RFF predict the dynamics of the system reasonably well. Of these, KAE performs the worst overall, whereas EDMD-kDL has the lowest error for long-term prediction.

## C. Learning from video data

Thus far, we have only considered experiments based on synthetic data. To illustrate the performance of the proposed method on real-world data, we consider the problem of predicting a moving pendulum directly from video data. The raw dataset, obtained from ScienceOnline: The Pendulum and Galileo, consists of a 20.92 seconds video recorded at a 25 frames per second rate on a $7 2 0 \times 5 7 6$ pixel grid. The authors of Ref. 41 used this dataset to illustrate the data-driven approximation of Koopman eigenfunctions using kernel EDMD. As a preprocessing step, we convert the video from RGB to grayscale with normalized values between zero and one and subtract the median frame to remove the back ground. This results in a dataset of size $7 2 0 \times 5 7 6 \times 5 2 3$ which we reshape into a single 414720 523 snapshot matrix. We use the first 418 frames (80% of the data) for training and the remaining 105 frames (20% of the data) for testing. As in the previous example, we apply SVD to the training data and retain only 90% of the total cumulative singular energy, hence reducing the dimensionality to r = 14.

For EDMD-kDL, we use the autoencoder formulation.

We choose the kernel bandwidth to $\sigma = 5$ and the dictionary size $N = 3 r$ . The embedding dimension is set to 8 for KAE and cKAE after manual hyperparameter tuning. For all methods, we use multi-step prediction for training with $T = 1 2$

Figure 7 shows the predictions obtained by each method. We include the reconstructed test data from the SVD projection (the second row) to highlight the error induced by dimensionality reduction, which is independent of the method used to model the temporal evolution in the reduced space. KAE and cKAE drasti cally fail to properly learn the dynamics of the system. In contrast, EDMD-kDL produces sensible predictions over the entire test horizon. In some cases, the position of the pendulum object is blurred but can still be correctly identified. Moreover, we observe no increase in performance for KAE and cKAE by increasing the embedding dimension.

## D. Global sea-surface temperature forecasting

As a final example, we consider the task of forecast ing sea-surface temperature (SST). The dataset contains 1400 weekly snapshots of global SST at $1 ^ { \circ } \times 1 ^ { \circ }$ spatial resolution on a 180  360 grid over the period from 1992 to 2019. Of the 64800 grid points, there are exactly 44 219 valid SST locations. This reanalysis dataset was produced by the National Oceanic and Atmospheric Administration (NOAA); the methodology is described in Ref. 61. We obtain the data from Ref. 62 and use the first 1000 snapshots for training and the remaining 400 for testing. Since SST values are on diferent scales across the globe, we first apply min-max scaling at each location so that all the values are contained in [0, 1]. Moreover, we remove 720 rows with constant values over the training period. We then reduce the dimensionality from 43 499 to r = 19 by applying SVD and retaining 90% of the cumulative singular energy.

(a) Predicted phase portraits, NR = 1%.  
![](images/0e453f194d28c96d2ba38893b5087f014b49866019cb1c56ac95d0763de22891.jpg)  
(c) Predicted phase portraits, NR = 5%.

(b) Relative prediction error, NR = 1%.  
![](images/28b2f2f8cf4a62c88df564b7a2a3137fb815b0cc36a7be57c7992537429518f6.jpg)  
(d) Relative prediction error, NR = 5%.

![](images/84c3c2e5101d128b3779b6e7baf8c5673b8432ae366cf315165f7920bbafbee7.jpg)  
(e) Predicted phase portraits, NR = 10%.

![](images/6243d4a85c14379d007f33c5b28ce8299e1f20a2d32c426c8704d0b4597a405b.jpg)

![](images/bc00c130fd20733925bba924eeb6ccd1c75005bdbc18136d9bcabcc51b876a53.jpg)

(f) Relative prediction error, NR = 10%.  
![](images/6fbb932e7f4448aca2fbc15e50a635d515eab226de50b9feaf6f12a8cae9d3cc.jpg)  
FIG. 2: Pendulum with measurement noise in the almost-linear regime. Across noise levels, EDMD-kDL outperforms the two Koopman autoencoder methods, which produce non-smooth predictions. On the other hand, EDMD-RFF sufers from phase drift (see the left column of Figure 4) which leads to worse performance in terms of prediction error compared to EDMD-kDL when the amount of noise is large.

For EDMD-kDL, we set the kernel bandwidth to σ = 10 and the dictionary size to $\textit { N } = \ 2 r$ with a learned decoder. The latent dimension is set to 8 for KAE and cKAE after manual tuning. We further compare against SINDy-SHRED<sup>63</sup>, a recently proposed architecture that combines sparse nonlinear system identification<sup>64</sup> with shallow recurrent decoders<sup>65</sup>. We use the oficial SINDy-SHRED implementation available at github.com/pyshred-dev/pyshred with the same settings as in Ref. 63, i.e., 250 sensors with a lag time of 52 weeks and a total forecasting period of 318 weeks, corresponding to just over 6 years.

Figure 8 shows predictions obtained by diferent meth ods for a set of randomly sampled SST locations over the forecast period. EDMD-kDL and SINDy-SHRED produce similar forecasts across the board. The predicted trajectories appear like smoothed versions of the ground truth trajectories. Overall, the forecasts obtained from KAE and cKAE are of lesser quality.

To obtain a more global picture, we unscale the predictions and calculate relative prediction errors based on full snapshots over time. We observe that for the first 100 weeks, EDMD-kDL, KAE, and cKAE have comparably low error levels, roughly below 5%. For this initial phase, SINDy-SHRED has the worst performance overall, see Figure 9. In contrast, for longer term prediction, EDMDkDL is the only method that maintains a low error level over the entire forecast horizon. The maximum relative error over the 318-week period is 6.7% for EDMD-kDL, followed by 11.8% for SINDy-SHRED, 17.5% for cKAE, and 20.9% for KAE. Furthermore, we calculate the mean absolute error across SST locations and over time for each method as a summarized performance measure. We obtain 0.51<sup>◦</sup>C for EDMD-kDL, 0.67<sup>◦</sup>C for SINDy-SHRED, 0.83<sup>◦</sup>C for KAE, and 0.88<sup>◦</sup>C for cKAE. This shows that, across the entire prediction horizon, SINDy-SHRED performs better overall compared to KAE and cKAE. Finally, we provide representative predicted spatiotemporal SST maps and the corresponding absolute prediction error maps in Figure 10a and Figure 10b, respectively, for illustration purposes.

## V. DISCUSSION

In this work, we proposed EDMD-kDL, a novel approach for the data-driven approximation of the Koopman operator that combines the standard EDMD algorithm<sup>6</sup> with kernel-based autoencoders introduced herein for dictionary learning. EDMD-kDL bridges between standard kernel-based methods for the data-driven spectral decomposition of the Koopman operator<sup>40,41</sup> and ANN-based autoencoders for learning Koopman embeddings from data<sup>8,30,33–37</sup>. The proposed method is formulated as solving a joint optimization problem over an embedding map in a reproducing kernel Hilbert space and the associated Koopman matrix such that the embedded dynamical system is (approximately) linear. The approach also allows us to learn a linear decoder that reconstructs state space vectors from their embedding space representations. We tackle the resulting functionspace optimization problem by introducing pseudo-values for the embedding map at a finite set of collocation points, and reformulating the problem using bilevel optimization. Then, by leveraging the particular structure of RKHSs, notably the representer theorem<sup>52,53</sup>, we obtain a finite-dimensional optimization problem over the pseudo-values, the Koopman matrix, and the decoder matrix, which we solve using standard numerical methods.

(a) Predicted phase portraits, NR = 1%.  
![](images/3510cc03c9f99be64cb9ed9cce7bd424ad362a571ae86f5f5bec2f0b548b65fb.jpg)  
(c) Predicted phase portraits, NR = 5%.

(b) Relative prediction error, NR = 1%.  
![](images/13547fa6f6f5605f8299358e4edc41174e16fc1cebfb1300791da87505be915b.jpg)

![](images/6fadd4c6109083af2a37125e02e80926a58244d10542c93ba1d03839cc8a6fa9.jpg)  
(e) Predicted phase portraits, NR = 10%.

(d) Relative prediction error, NR = 5%.  
![](images/6c94436d83699d5f9a55adb851faeac75a868ab11a6400220d9219434dadd3c1.jpg)

![](images/215dea92d61dba5a4c0adc2110f52048276f86bb866780ced3e65dc2dc35e18e.jpg)

(f) Relative prediction error, NR = 10%.  
![](images/5010594e093cd25af2aa1da396121c0a3066c43ed67eb66270a04d2309e60c28.jpg)  
FIG. 3: Pendulum with measurement noise in the truly nonlinear regime. Across noise levels, EDMD-kDL outperforms the two Koopman autoencoder methods, which produce non-smooth predictions. On the other hand, EDMD-RFF sufers from phase drift (see the right column of Figure 4), which leads to worse performance in terms of prediction error compared to EDMD-kDL when the amount of noise is substantial.

(a) x<sub>1</sub>(0) = 0.8 and NR = 5%.  
![](images/d969742b6498e7d45101fbc73cb0febbd5aac12681decc97f078dfebe189f64e.jpg)  
(c) x<sub>1</sub>(0) = 0.8 and NR = 10%.

(b) x<sub>1</sub>(0) = 2.4 and NR = 5%.  
![](images/456fc674745e81ff9a1d19e0993e277b05a220ace059c4c7496a3cb70d71a44c.jpg)

![](images/219355554e93aa08303136a48d1843cf53c1a13d74dbd3c5ed7c774834e9237d.jpg)

(d) x<sub>1</sub>(0) = 2.4 and NR = 10%.  
![](images/a67ddde526f1bc493ea4c9724657449653c5109d52f373b01d15c625679a1329.jpg)  
FIG. 4: Pendulum with measurement noise. Comparison of predicted trajectories obtained by EDMD-kDL and three baseline methods over the final quarter of the prediction horizon. EDMD-RFF sufers from phase drift when the noise level is substantial whereas KAE and cKAE produce non-smooth predictions.

![](images/e1a35310f3f2074995602b8851903d6cfb20c441b79241227b71591cf146879f.jpg)  
FIG. 5: Kármán vortex shedding. Relative prediction errors over the testing horizon measured in the SVD-reduced space.

We considered four diferent examples to illustrate the performance of EDMD-kDL in practice: the twodimensional nonlinear pendulum model with no friction, a simulated fluid flow past a cylinder exhibiting vortex shedding, a video of a pendulum, and global SST. For each experiment, we compare the performance of EDMDkDL against state-of-the-art ANN-based counterparts on the task of predicting the evolution of the system using learned Koopman models. Across all settings, we find that EDMD-kDL performs comparably or better than the ANN-based approaches. This includes the recently proposed SINDy-SHRED method<sup>63</sup> for the global SST forecasting example. Notably, the proposed method is able to accurately predict the dynamics of a pendulum directly from video data, a setting in which the ANNbased approaches drastically failed. Similar observations have been made in the context of operator learning<sup>66</sup>.

It is worth noting that, although we have focused on deterministic systems in the current work, the proposed approach is directly applicable to stochastic systems. Alternatively, the optimization paradigm can be modified to use the variational formulation, see, e.g., Refs. 10, 11, 67, and 68. In this context, it would be of interest to compare EDMD-kDL against VAMPnets<sup>34</sup> and time-lagged autoencoders<sup>35</sup>.

Furthermore, we envision multiple potential avenues for future research. Firstly, elucidating the convergence properties and finite-data error bounds for the proposed method – potentially by combining ideas from the convergence analysis of standard (kernel) EDMD methods<sup>69–71</sup> and the error analysis of kernel collocation methods for solving nonlinear parametric PDEs<sup>46,72</sup> – would be of great theoretical value. Secondly, our current approach for dealing with high-dimensional systems is to first reduce dimensionality using principal component analysis and then apply EDMD-kDL in the reduced space. Extending the proposed method to work directly with high dimensional data, for example by jointly learning a nonlinear projection map akin to kernel principal component analysis<sup>73</sup>, is a problem that we would like to investigate in the future. Finally, from a computational standpoint, we observed that starting with a first-order optimizer before switching to L-BFGS when training EDMD-kDL models can substantially reduce the computational cost. We hope to investigate this, and more broadly strategies for eficient training, in the future.

## ACKNOWLEDGMENTS

J.-P. was supported by the EPSRC Centre for Doctoral Training in Mathematical Modelling, Analysis and Computation (MAC-MIGS) funded by the UK Engineering and Physical Sciences Research Council (grant EP/S023291/1), Heriot–Watt University and The University of Edinburgh.

## DATA AVAILABILITY STATEMENT

The data that support the findings of this study are available from the corresponding author upon reasonable request.

## REFERENCES

<sup>1</sup>B. O. Koopman, “Hamiltonian systems and transformations in Hilbert space,” Proceedings of the National Academy of Sciences 17, 315 (1931).

<sup>2</sup>B. O. Koopman and J. v. Neumann, “Dynamical systems of continuous spectra,” Proceedings of the National Academy of Sciences 18, 255–263 (1932).

<sup>3</sup>A. Lasota and M. C. Mackey, Chaos, fractals, and noise: Stochastic aspects of dynamics, 2nd ed., Applied Mathematical Sciences, Vol. 97 (Springer, New York, 1994).

<sup>4</sup>I. Mezić, “Spectral properties of dynamical systems, model reduction and decompositions,” Nonlinear Dynamics 41, 309–325 (2005).

<sup>5</sup>P. J. Schmid, “Dynamic mode decomposition of numerical and experimental data,” Journal of Fluid Mechanics 656, 5–28 (2010).

(a) t = 0  
![](images/b89d2106f19ed7499169e1137faf5ab0f6a196249d459687a25131ac19b4c2fe.jpg)  
FIG. 6: Kármán vortex shedding. Full-space predictions obtained by various Koopman methods. “Ground truth” refers to data obtained by projecting the original test data onto the retained principal components and reconstructing.

![](images/298444987eb911ad8bd680362dff71b17df341e413897f8bf60da742f0158cc4.jpg)  
FIG. 7: Pendulum video data prediction. Predictions obtained by various Koopman methods in the pixel space. “Ground truth” refers to data obtained by projecting the original test data onto the retained SVD modes and reconstructing.

<sup>6</sup>M. O. Williams, I. G. Kevrekidis, and C. W. Rowley, “A datadriven approximation of the Koopman operator: Extending dynamic mode decomposition,” Journal of Nonlinear Science 25, 1307–1346 (2015).

<sup>7</sup>S. Klus, P. Koltai, and C. Schütte, “On the numerical approximation of the Perron–Frobenius and Koopman operator,” Journal of Computational Dynamics 3, 51–79 (2016).

<sup>8</sup>Q. Li, F. Dietrich, E. M. Bollt, and I. G. Kevrekidis, “Extended dynamic mode decomposition with dictionary learning: A datadriven adaptive spectral decomposition of the Koopman opera-

tor,” Chaos: An Interdisciplinary Journal of Nonlinear Science 27, 103111 (2017).

<sup>9</sup>S. Klus, F. Nüske, P. Koltai, H. Wu, I. Kevrekidis, C. Schütte, and F. Noé, “Data-driven model reduction and transfer operator approximation,” Journal of Nonlinear Science 28, 985–1010 (2018).

<sup>10</sup>F. Noé and F. Nüske, “A variational approach to modeling slow processes in stochastic dynamical systems,” Multiscale Modeling & Simulation 11, 635–655 (2013).

![](images/55616b1b7bf253faa0fa9b2c696d46d26b07f15a5fd7c5bbe8d55c295d6db0ad.jpg)  
FIG. 8: Sea surface temperature forecasting. Scaled predictions at randomly sampled SST locations.

![](images/0fb85ab68def22d19edc9f0f1fbf334de75c2ca199117d9780f84f13d5de3450.jpg)  
FIG. 9: Sea surface temperature forecasting. Relative prediction errors over time.

<sup>11</sup>F. Nüske, B. G. Keller, G. Pérez-Hernández, A. S. J. S. Mey, and F. Noé, “Variational approach to molecular kinetics,” Journal of Chemical Theory and Computation 10, 1739–1752 (2014).

<sup>12</sup>L. Molgedey and H. G. Schuster, “Separation of a mixture of independent signals using time delayed correlations,” Physical Review Letters 72, 3634–3637 (1994).

<sup>13</sup>I. Mezić, “Analysis of fluid flows via spectral properties of the Koopman operator,” Annual Review of Fluid Mechanics 45, 357– 378 (2013).

<sup>14</sup>J. H. Tu, C. W. Rowley, D. M. Luchtenburg, S. L. Brunton, and J. N. Kutz, “On dynamic mode decomposition: Theory and applications,” Journal of Computational Dynamics 1 (2014).

<sup>15</sup>S. E. Otto and C. W. Rowley, “Koopman operators for estimation and control of dynamical systems,” Annual Review of Control, Robotics, and Autonomous Systems 4, 59–87 (2021).

<sup>16</sup>P. J. Schmid, “Dynamic mode decomposition and its variants,” Annual Review of Fluid Mechanics 54, 225–254 (2022).

<sup>17</sup>M. J. Colbrook, “The multiverse of dynamic mode decomposition algorithms,” in Handbook of numerical analysis, Vol. 25 (Elsevier, 2024) pp. 127–230.

<sup>18</sup>A. Mauroy and J. Goncalves, “Linear identification of nonlinear systems: A lifting technique based on the Koopman operator,” in 2016 IEEE 55th Conference on Decision and Control (CDC) (2016) pp. 6500–6505.

<sup>19</sup>D. Bruder, C. D. Remy, and R. Vasudevan, “Nonlinear system identification of soft robot dynamics using Koopman operator theory,” in 2019 International Conference on Robotics and Automation (ICRA) (IEEE, 2019) pp. 6244–6250.

<sup>20</sup>S. Klus, F. Nüske, S. Peitz, J.-H. Niemann, C. Clementi, and C. Schütte, “Data-driven approximation of the Koopman generator: Model reduction, system identification, and control,” Physica D: Nonlinear Phenomena 406, 132416 (2020).

<sup>21</sup>S. Peitz and S. Klus, “Koopman operator-based model reduction for switched-system control of PDEs,” Automatica 106, 184–191 (2019).

<sup>22</sup>V. Nateghi and F. Nüske, “Kinetically consistent coarse graining using kernel-based extended dynamic mode decomposition,” Journal of Chemical Theory and Computation 21, 7236–7248 (2025).

<sup>23</sup>M. Korda and I. Mezić, “Linear predictors for nonlinear dynamical systems: Koopman operator meets model predictive control,” Automatica 93, 149–160 (2018).

<sup>24</sup>A. Surana, “Koopman operator based observer synthesis for control-afine nonlinear systems,” in 2016 IEEE 55th Conference on Decision and Control (CDC) (IEEE, 2016) pp. 6492–6499.

<sup>25</sup>D. Goswami and D. A. Paley, “Global bilinearization and controllability of control-afine nonlinear systems: A Koopman spectral approach,” in 2017 IEEE 56th Annual Conference on Decision and Control (CDC) (IEEE, 2017) pp. 6107–6112.

<sup>26</sup>H. Arbabi, M. Korda, and I. Mezić, “A data-driven Koopman model predictive control framework for nonlinear partial diferential equations,” in 2018 IEEE Conference on Decision and Control (CDC) (IEEE, 2018) pp. 6409–6414.

<sup>27</sup>P. Bevanda, S. Sosnowski, and S. Hirche, “Koopman operator dynamical models: Learning, analysis and control,” Annual Reviews in Control 52, 197–212 (2021).

<sup>28</sup>J. Hua, F. Noorian, D. Moss, P. H. W. Leong, and G. H. Gunaratne, “High-dimensional time series prediction using kernel-

![](images/f8c065823be72c79cf743e2708a8f6aae0dad5fdc2fb3e4753b46a6776f56b61.jpg)  
(a) Predictions vs true SST values (<sup>◦</sup>C).

![](images/3f5bade234973814b7e6058523c397dfdec7937c81f737e6dc532c504ceb34a5.jpg)  
(b) Absolute prediction errors ( C).  
FIG. 10: Sea surface temperature forecasting. (a) Predicted global SST maps at specified time steps. (b) Associated absolute errors.

based Koopman mode regression,” Nonlinear Dynamics 90, 1785–1806 (2017).

<sup>29</sup>D. Giannakis, “Data-driven spectral decomposition and forecasting of ergodic dynamical systems,” Applied and Computational Harmonic Analysis 47, 338–396 (2019).

<sup>30</sup>O. Azencot, N. B. Erichson, V. Lin, and M. Mahoney, “Forecasting sequential data using consistent Koopman autoencoders,” in International Conference on Machine Learning (PMLR, 2020) pp. 475–485.

<sup>31</sup>H. Lange, S. L. Brunton, and J. N. Kutz, “From Fourier to Koopman: Spectral methods for long-term time series prediction,” Journal of Machine Learning Research 22, 1–38 (2021).

<sup>32</sup>P. Bevanda, M. Beier, A. Lederer, S. Sosnowski, E. Hüllermeier, and S. Hirche, “Koopman kernel regression,” Advances in Neural Information Processing Systems 36, 16207–16221 (2023).

<sup>33</sup>N. Takeishi, Y. Kawahara, and T. Yairi, “Learning Koopman invariant subspaces for dynamic mode decomposition,” Advances in neural information processing systems 30 (2017).

<sup>34</sup>A. Mardt, L. Pasquali, H. Wu, and F. Noé, “VAMPnets for deep learning of molecular kinetics,” Nature Communications 9 (2018), 10.1038/s41467-017-02388-1.

<sup>35</sup>C. Wehmeyer and F. Noé, “Time-lagged autoencoders: Deep learning of slow collective variables for molecular kinetics,” The Journal of chemical physics 148 (2018).

<sup>36</sup>E. Yeung, S. Kundu, and N. Hodas, “Learning deep neural network representations for Koopman operators of nonlinear dynamical systems,” in 2019 American Control Conference (ACC) (IEEE, 2019) pp. 4832–4839.

<sup>37</sup>S. E. Otto and C. W. Rowley, “Linearly-recurrent autoencoder networks for learning dynamics,” SIAM Journal on Applied Dynamical Systems 18, 558–593 (2019).

<sup>38</sup>M. Tabish, B. Leimkuhler, and S. Klus, “How deep is your network? Deep vs. shallow learning of transfer operators,” (2025), arXiv:2509.19930.

<sup>39</sup>N. B. Erichson, M. Muehlebach, and M. W. Mahoney, “Physicsinformed autoencoders for Lyapunov-stable fluid flow prediction,” arXiv preprint arXiv:1905.10866 (2019).

<sup>40</sup>M. O. Williams, C. W. Rowley, and I. G. Kevrekidis, “A kernelbased method for data-driven Koopman spectral analysis,” Journal of Computational Dynamics 2, 247–265 (2015).

<sup>41</sup>S. Klus, I. Schuster, and K. Muandet, “Eigendecompositions of transfer operators in reproducing kernel Hilbert spaces,” Journal of Nonlinear Science (2019), 10.1007/s00332-019-09574-z.

<sup>42</sup>G. Cybenko, “Approximation by superpositions of a sigmoidal function,” Mathematics of control, signals and systems 2, 303–

314 (1989).

<sup>43</sup>C. A. Micchelli, Y. Xu, and H. Zhang, “Universal kernels.” Journal of Machine Learning Research 7 (2006).

<sup>44</sup>B. K. Sriperumbudur, K. Fukumizu, and G. R. Lanckriet, “Universality, characteristic kernels and RKHS embedding of measures.” Journal of Machine Learning Research 12 (2011).

<sup>45</sup>R. Schaback and H. Wendland, “Kernel techniques: from machine learning to meshless methods,” Acta numerica 15, 543–639 (2006).

<sup>46</sup>Y. Chen, B. Hosseini, H. Owhadi, and A. M. Stuart, “Solving and learning nonlinear PDEs with Gaussian processes,” Journal of Computational Physics 447, 110668 (2021).

<sup>47</sup>C. W. Rowley, I. Mezić, S. Bagheri, P. Schlatter, and D. S. Henningson, “Spectral analysis of nonlinear flows,” Journal of Fluid Mechanics 641, 115–127 (2009).

<sup>48</sup>M. Budišić, R. Mohr, and I. Mezić, “Applied Koopmanism,” Chaos: An Interdisciplinary Journal of Nonlinear Science 22 (2012), 10.1063/1.4772195.

<sup>49</sup>B. Schölkopf and A. J. Smola, Learning with Kernels: Support Vector Machines, Regularization, Optimization and Beyond (MIT press, Cambridge, USA, 2001).

<sup>50</sup>I. Steinwart and A. Christmann, Support Vector Machines, 1st ed. (Springer, New York, 2008).

<sup>51</sup>N. Aronszajn, “Theory of reproducing kernels,” Transactions of the American Mathematical Society 68, 337–404 (1950).

<sup>52</sup>G. Kimeldorf and G. Wahba, “Some results on Tchebychefian spline functions,” Journal of Mathematical Analysis and Applications 33, 82–95 (1971).

<sup>53</sup>B. Schölkopf, R. Herbrich, and A. J. Smola, “A generalized representer theorem,” in International conference on computational learning theory (Springer, 2001) pp. 416–426.

<sup>54</sup>A. E. Hoerl and R. W. Kennard, “Ridge regression: Biased estimation for nonorthogonal problems,” Technometrics 12, 55–67 (1970).

<sup>55</sup>G. H. Golub and V. Pereyra, “The diferentiation of pseudoinverses and nonlinear least squares problems whose variables separate,” SIAM Journal on numerical analysis 10, 413–432 (1973).

<sup>56</sup>G. Golub and V. Pereyra, “Separable nonlinear least squares: the variable projection method and its applications,” Inverse problems 19, R1–R26 (2003).

<sup>57</sup>D. C. Liu and J. Nocedal, “On the limited memory BFGS method for large scale optimization,” Mathematical programming 45, 503–528 (1989).

<sup>58</sup>A. M. DeGennaro and N. M. Urban, “Scalable extended dynamic mode decomposition using random kernel approximation,” SIAM Journal on Scientific Computing 41, A1482–A1499 (2019).

<sup>59</sup>F. Nüske and S. Klus, “Eficient approximation of molecular kinetics using random Fourier features,” The Journal of Chemical Physics 159 (2023).

<sup>60</sup>I. M. Sobol, “Distribution of points in a cube and approximate evaluation of integrals,” USSR Computational Mathematics and Mathematical Physics 7, 86–112 (1967).

<sup>61</sup>R. W. Reynolds, N. A. Rayner, T. M. Smith, D. C. Stokes, and W. Wang, “An improved in situ and satellite SST analysis for climate,” Journal of climate 15, 1609–1625 (2002).

<sup>62</sup>L. M. Gao, J. Williams, and N. Kutz, “Sparse identification of nonlinear dynamics and Koopman operators with shallow recurrent decoder networks [datasets],” (2025).

<sup>63</sup>M. L. Gao, J. P. Williams, and J. N. Kutz, “Sparse identification of nonlinear dynamics and Koopman operators with shallow recurrent decoder networks,” Proceedings of the National Academy of Sciences 123, e2508144123 (2026).

<sup>64</sup>S. L. Brunton, J. L. Proctor, and J. N. Kutz, “Discovering governing equations from data by sparse identification of nonlinear dynamical systems,” Proceedings of the National Academy of Sciences 113, 3932–3937 (2016).

<sup>65</sup>J. P. Williams, O. Zahn, and J. N. Kutz, “Sensing with shallow recurrent decoder networks,” Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 480, 20240054 (2024).

<sup>66</sup>P. Batlle, M. Darcy, B. Hosseini, and H. Owhadi, “Kernel methods are competitive for operator learning,” Journal of Computa-

tional Physics 496, 112549 (2024).

<sup>67</sup>H. Wu and F. Noé, “Variational approach for learning Markov processes from time series data,” Journal of Nonlinear Science 30, 33–66 (2020).

<sup>68</sup>C. Schütte, S. Klus, and C. Hartmann, “Overcoming the timescale barrier in molecular dynamics: Transfer operators, variational principles and machine learning,” Acta Numerica 32, 517–673 (2023).

<sup>69</sup>M. Korda and I. Mezić, “On convergence of extended dynamic mode decomposition to the Koopman operator,” Journal of Nonlinear Science 28, 687–710 (2018).

<sup>70</sup>F. M. Philipp, M. Schaller, K. Worthmann, S. Peitz, and F. Nüske, “Error analysis of kernel EDMD for prediction and control in the Koopman framework,” Journal of Nonlinear Science 35, 92 (2025).

<sup>71</sup>F. Köhne, F. M. Philipp, M. Schaller, A. Schiela, and K. Worthmann, “L<sup>∞</sup>-error bounds for approximations of the Koopman operator by kernel extended dynamic mode decomposition,” SIAM journal on applied dynamical systems 24, 501–529 (2025).

<sup>72</sup>P. Batlle, Y. Chen, B. Hosseini, H. Owhadi, and A. M. Stuart, “Error analysis of kernel/GP methods for nonlinear and parametric PDEs,” Journal of Computational Physics 520, 113488 (2025).

<sup>73</sup>B. Schölkopf, A. Smola, and K.-R. Müller, “Nonlinear component analysis as a kernel eigenvalue problem,” Neural computation 10, 1299–1319 (1998).