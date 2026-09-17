# Gradient Descent with Stochastic Subspaces via Persistence of Memory

Subhro Ghosh<sup>∗1</sup>, Clement Z.Q. Ng<sup>†1</sup>, Pierre-Louis Poirion<sup>‡2</sup>, and Akiko Takeda<sup>§2,3</sup>

<sup>1</sup>Department of Mathematics, National University of Singapore, Singapore <sup>2</sup>Center for Advanced Intelligence Project, RIKEN, Tokyo, Japan <sup>3</sup>Department of Mathematical Informatics, The University of Tokyo, Tokyo, Japan

## Abstract

Stochastic subspace methods have gained popularity as gradient descent based techniques for large scale optimisation problems, especially in distributed settings. In this paper, we introduce the technique of "persistence of memory" to greatly extend and improve the random subspace methods. To this end, we leverage a vector that is only weakly correlated with the gradient in order to provide a guiding structure to the generative process of the random subspace along which the descent is going to take place. This guidance vector may be fixed for a large number of iterations, only to be refreshed at wide intervals (on whose size we can provide guarantees in terms of problem parameters). In important machine learning settings, such as optimisation problems embodying sparsity or a minibatch structure, we show that the guidance vector can be obtained in an efective and computationally inexpensive manner by leveraging the structured properties of the problem. En route, we establish to our knowledge the first theoretical analysis of classical SSD methods for sparse functions. In a local neighbourhood of the optimum, we demonstrate an alignment phenomenon of our gradient estimates with a low-lying eigenvector of the Hessian, allowing a once-for-all computation of the guidance vector which renders the method computationally favourable even in scenarios with unstructured objectives.

## Contents

1 Introduction 4   
1.1 Motivation . 4   
1.2 Persistence of Memory. 5   
1.2.1 Structural Overview of Our Method 5   
1.2.2 Iteration Complexity . 7   
1.2.3 Fast Generation of Weakly Correlated Vectors for Machine Learning Applications 7   
1.2.4 Computational Cost 8   
1.2.5 Parallelisation and Hardware Eficiency 9   
1.2.6 Interpolation Between Random Subspace and Full Gradient Descent 10   
1.2.7 Local Regime and Fast Alignment of Gradient Vectors 10   
1.3 Application Domains for Stochastic Subspaces 11   
1.3.1 Memory Constrained Optimisation 11   
1.3.2 Communication-Constrained Optimisation 11   
1.3.3 Long-Horizon and Nested Optimisation 12   
1.3.4 Black-Box and Non-Diferentiable Optimisation 12   
1.3.5 Optimiser-State Memory . 12   
1.4 Related Works 13   
1.5 Structure and Notation of the Paper 14   
2 Main Results 14   
2.1 Preliminaries 14   
2.2 Algorithm . 15   
2.2.1 Discussion on Algorithm . 15   
2.3 Guarantees for Iteration and Computational Complexities 16   
2.4 Local Regimes and Gradient Alignment Phenomena 18   
2.4.1 Strong Alignment in Local Regimes 19   
2.4.2 Application Scenarios in which Local Regime is Natural 20   
3 Applications to Machine Learning 22   
3.1 Structured Optimisation and Fast Generation of Guidance Vectors 22   
3.2 Sparse Functions . 23   
3.2.1 Structure of the Problem and Motivations 23   
3.2.2 Fast Guidance Vectors in Sparse Settings 24   
3.2.3 Analysis of Classical SSD in Sparse Settings 26   
3.3 Finite Sum Structure with Minibatch Gradient Descent 26   
3.3.1 Structure of The Problem and Motivations 27   
3.3.2 Analysis for Convergence Iteration 28   
4 Proof Ideas 29   
4.1 General Algorithm 29   
4.2 Mini-batch Setting 30   
4.3 Local Regime . 31   
5 Numerical Experiments 33   
6 Acknowledgements 34   
A Mathematical Tools Needed 39   
B Proof for General Variant 40   
B.1 Proof for Iteration Complexity (Theorem 1) 40   
B.2 Proof for Computational Cost (Theorem 2) 47   
C Proof for Sparse Variant 47   
C.1 Proof for Refresh Cost in Sparse Case (Proposition 2) 47   
C.2 Proof for Computational Cost in Sparse Case (Theorem 3) 48   
C.3 Analysis of Classical SSD in Sparse Setting (Theorem 6) 49   
D Proof for Minibatch Variant 53   
D.1 Proof for Alignment in Minibatch Setting (Proposition 3) 53   
D.2 Proof for Iteration Complexity in Minibatch Setting (Theorem 7) 59   
D.3 Proof for Computational Cost in Minibatch Setting (Theorem 4) 63   
E Proof for the Local Regime (Theorem 5) 63   
E.1 Setting and Algorithm . 63   
E.2 Build-up to Main Theorem 64   
E.3 Analysis for Persistent Alignment 71

## 1 Introduction

In recent years, there are many literature being developed around the study of high-dimensional unconstrained optimization. In simple terms, such a problem can be formulated as

$$
\operatorname* { m i n } _ { x \in \mathbb { R } ^ { n } } f ( x ) .\tag{1}
$$

for a function $f : \mathbb { R } ^ { n } \mapsto \mathbb { R }$ . When the dimension n is suficiently large, as is the case in modern machine learning applications, classical gradient methods often become prohibitively expensive due to the need to compute full gradients at every iteration step.

To address this issue, subspace optimization methods have been developed, which are, in turn generalisations of the more basic coordinate descent approach. In order to save computational costs and bypass the high dimensional constraints $( x \in \mathbb { R } ^ { n } )$ , randomised variants of these methods have gained popularity; a key example of this being the so-called Stochastic Subspace Descent (SSD) method. For details on the substantial literature on subspace descent, randomised and otherwise, we refer the reader to [28] and the references, therein as a partial list.

However, standard SSD methods generally incur a high cost in terms of iteration complexity, which is in part a consequence of the random noise inherent in their setup. In this work, we propose to enhance the state of the art in this problem by unveiling structured stochastic subspace generation techniques that vastly improve the iteration complexity while retaining computational tractability in a wide range of structured machine learning problems.

## 1.1 Motivation

In Stochastic Subspace Descent (SSD), at each update step, the gradients are computed along a randomly chosen low-dimensional subspace. In other words, instead of computing the gradient $\nabla f ( x )$ , SSD calculates the projection of the gradient vector onto a random subspace of dimension d. In formulaic terms, the k-th update step in SSD looks like

$$
\begin{array} { r } { x _ { k } = x _ { k - 1 } - \alpha P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) , } \end{array}
$$

where $P _ { k } \in \mathbb { R } ^ { n \times d }$ is a tall matrix with orthogonal columns and $d \ll n$

In general, this subspace is chosen uniformly at random; for instance, the specific device used by the initial work [28] is that of Haar-distributed matrices $P _ { k }$ . This algorithm is particularly useful for high-dimensional problems, and the main advantage of this method is that instead of calculating the full gradient, it only calculates a projection of the gradient onto a lower dimension. Indeed in some cases, computing the full gradient of the function might be memory heavy, in such cases we only need to compute a small number of directional derivatives (using d random directions) rather than n of them (think of the gradient as n directional derivatives in the coordinate directions).

In fact, this idea generalises coordinate descent - by allowing projections onto non-coordinate subspaces, and can be particularly efective in cases where the function exhibits low intrinsic dimensionality or when many of the gradient components are negligible. Other situations where subspace methods can be considered are discussed in the later subsections.

Denoting the cost of 1 computation of directional derivative as $\xi ,$ the standard gradient descent has a iteration complexity of $O ( \epsilon ^ { - 2 } )$ and the cost per iteration is $O ( n \xi )$ (n number of directional derivatives), for a total computational cost of $O ( n \xi \epsilon ^ { - 2 } )$ . In comparison, SSD has a decreased per iteration cost of $O ( d \xi )$ as compared to $O ( n \xi )$ , but at the same time sufers from the curse of dimensionality in its iteration complexity $( O ( \epsilon ^ { - 2 } n / d )$ instead). While each iteration is computationally cheaper, the total complexity will be similar to the classical methods to achieve the same degree of accuracy. Recognising this trade-of, recent research has focused on enhancing the eficiency of each subspace, by integrating techniques that "remember" past useful information. For instance, enhanced variants of SSD - such as those incorporating elements from stochastic variance reduction (SVRG) and trust-region frameworks - demonstrate how leveraging historical gradient information can efectively reduce noise and improve stability [9, 14, 28].

## 1.2 Persistence of Memory.

In this paper, we introduce the technique of "persistence of memory" to greatly extend and improve the random subspace methods. To wit, we leverage a vector that is only weakly correlated with the gradient in order to provide a guiding structure to the generative process of the random subspace along which the descent is going to take place. This guidance vector (or alignment vector) may be fixed for a large number of iterations, only to be refreshed at wide intervals (on whose size we can provide guarantees in terms of problem parameters). In view of the persistence of the guidance vector throughout large cycles of the optimisation procedure, we call this approach persistence of memory. In fact, in significant setups such as the local regime described in Section 2.4, the guidance vector may be fixed once and for all without any refresh being required. Thus, reinforcing the concept of persistence of structural memory in the random subspaces throughout the descent to the optimum.

In important machine learning setups, such as optimisation problems embodying sparsity or a minibatch structure, the guidance vector can be obtained in a computationally inexpensive manner by leveraging the structured properties of the problem (refer to the applications in Section 3 for examples). In unstructured scenarios, the guidance vector may be obtained from black-box gradient generation methods typically hypothesised in the optimisation literature; in fact, our results in the local regime show that our approach enjoys advantages even in such setups.

Because the randomness is still retained in a major way in this approach, the method also enjoys the benefits due to randomness as the original method for a majority of the steps.

## 1.2.1 Structural Overview of Our Method

In this section, we provide a brief structural description of our method, in suficient detail, so as to provide an overall discussion of our contributions. For a more in-depth description of the algorithm, we refer the reader to Sections 2, and Section 3 for estimating the guidance vector in structured setups.

To understand our approach, it is beneficial to first recall the classical SSD algorithm. This consists of computing, at every iteration, the gradient of the objective function only along a randomly chosen subspace of dimension d. In practice, this can be accomplished, e.g. by picking at the k-th iteration (the projection $P _ { k }$ on to) a uniformly random subspace of dimension $d ,$ usually denominated by its columns (in an orthogonal matrix representation). Then, the projection of the gradient onto the random subspace can be easily computed in terms of the directional derivatives of the objective function along these columns. Typically, d is taken to be low (so as to keep the cost per iteration small), but high enough to ensure that a reasonable representation of the gradient can still be obtained. By the famous Johnson-Lindenstrauss Lemma (and its derivatives), this can already be achieved when d is of the order log n, which is a ballpark scale for us to keep in mind throughout this paper. Of course, $d$ can also be taken to be somewhat larger, such as a small power of $n ,$ depending on the computational capacity available and other problem considerations.

If the ambient dimension is $n ,$ then this costs $d / n$ , a fraction in terms computational load compared to a full gradient computation. However, the randomness inherent in the SSD algorithm makes the iterative procedure take much longer to converge, more precisely, $O ( n / d \cdot \epsilon ^ { - 2 } )$ steps (where ϵ is the accuracy threshold for terminating the algorithm). Thus, what is gained in terms of computational cost per iteration is essentially lost in a vastly increased number of iterations, leaving the overall computational cost more or less unchanged.

Our method consists in a more nuanced random subspace generation mechanism that is sensitive to the problem at hand, thereby mitigating the problem of a large iteration complexity. At the same time, this achieves an overall computational advantage in structural settings that are of fundamental importance in machine learning applications. For the purposes of discussion later, we define the notion of alignment between two vectors u and v to be $\frac { \bar { \langle } u , v \rangle ^ { 2 } } { \| u \| _ { 2 } ^ { 2 } \| v \| _ { 2 } ^ { 2 } }$ . The alignment therefore ranges between 0 and 1, with alignment 1 indicating the vectors identifying the same straight line.

The cornerstone of our approach is a guidance vector, denoted by $v _ { k }$ at the k-th step in the gradient decent algorithm. We require that this guidance vector is weakly correlated with the true gradient $\nabla f ( x _ { k } )$ at the k-th iterate $x _ { k }$ , in the sense that $\langle \hat { v } _ { k } , \nabla f ( x _ { k } ) \rangle ^ { 2 } \geq \delta \| \nabla f ( x _ { k } ) \| _ { 2 } ^ { 2 }$ , where $\delta > 0$ is an alignment parameter and $\hat { v } _ { k }$ is the normalised vector of $v _ { k }$ . Instead of generating a uniformly random subspace, we now generate a random subspace that is conditioned to contain the guidance vector $v _ { k }$ . In practice, this may be achieved simply by concatenating $v _ { k }$ to a uniformly random subspace of dimension d that is orthogonal to $v _ { k }$ . The random subspace generated will naturally carry much more information about the true gradient (which is the direction that we would ideally want to descend along).

Of course, obtaining such a guidance vector $v _ { k }$ might incur a cost, especially if it is necessary to do it at every step. We therefore recommend generating the guidance vector only occasionally, a phenomenon that we call a refresh. At any refresh step, the guidance vector will have an initial alignment $\gamma _ { 0 }$ with the gradient, i.e. $\langle v _ { \mathrm { r e f r e s h } } , \nabla f ( x _ { \mathrm { r e f r e s h } } ) \rangle ^ { 2 } = \gamma _ { 0 } \| \nabla f ( x _ { \mathrm { r e f r e s h } } ) \| _ { 2 } ^ { 2 } .$ , which typically will much larger than the alignment threshold δ. This vector v<sub>refresh</sub> is going to be used as the guidance vector in the many subsequent steps of the descent $( \mathrm { i . e . , ~ } v _ { k } = v _ { \mathrm { r e f r e s h } }$ until the next refresh step). The point is that, because the evolution of the argument $x _ { k }$ in gradient descent is gradual, the v<sub>refresh</sub> will still have reasonably good alignment with the true gradient for many steps; in other words, we have persistence of memory of $v _ { \mathrm { r e f r e s h } }$ in the actual gradient.

However, as the iteration proceeds, the alignment between the guidance vector and the true gradient worsens, starting from the initial alignment $\gamma _ { 0 }$ but decaying towards the alignment threshold δ. After a certain number of steps, the memory fades and we perform another refresh of the guidance vector (before the alignment falls below δ). We continue in this vein until we reach the desired termination condition of our descent algorithm, which is typically enunciated in terms $\| \nabla f ( x ) \| _ { 2 } \leq \epsilon$

We provide a quantitative and principled structure to the idea outlined above, with explicit guarantees on the iteration complexity and the number of iterations between two successive refresh steps, purely in terms of the parameters of the problem introduced above. For a detailed technical statement of the results, we refer to Theorem 1 and Corollary 1 respectively; for an overview of the results and discussion on their eficacy, we refer to the subsequent Sections 2 and 3.

## 1.2.2 Iteration Complexity

In Theorem 1, we demonstrate that the iteration complexity of our method is

$$
\frac { \alpha ^ { - 1 } \left( 1 - \frac { \alpha L } { 2 } \right) ^ { - 1 } \left( f ( x _ { 0 } ) - f ^ { * } \right) } { \frac { 4 \alpha L ( 1 - \alpha L ) } { \left( 1 - 2 \alpha L \right) \log \left( \gamma _ { 0 } / \tilde { \delta } \right) } \left( 1 - \frac { d } { n - 1 } \right) \gamma _ { 0 } + \frac { d } { n - 1 } - \tilde { t } _ { 0 } } \epsilon ^ { - 2 }\tag{2}
$$

where α is the step size, L is the smoothness constant of the objective function $f , \ x _ { 0 }$ is the initialisation, $f ^ { * }$ the true minimum, ϵ is the accuracy threshold for termination, $\gamma _ { 0 }$ the starting alignment at every refresh, $\begin{array} { r } { \delta = \tilde { \delta } - \sqrt { \frac { d } { n - 1 } } } \end{array}$ the alignment lower bound that holds throughout the process and $\tilde { t } _ { 0 }$ a function of the problem parameters which roughly equals ${ \frac { 1 } { 2 } } { \sqrt { \frac { d } { n } } } .$

To illustrate our method clearly, we discuss the ballpark regime where we set the initial and threshold alignment parameters, namely γ<sub>0</sub> and δ, to be $\Theta ( 1 )$ . Given that the step-size α is typically (a small) constant times $1 / L$ and $d \ll n ,$ we may then deduce from (2) that the iteration complexity is $O ( \epsilon ^ { - 2 } )$ . This, in particular, is a very substantial improvement from the classical SSD algorithm, which entails an iteration complexity of $O ( n / d )$ . With a common choice of d being $\Theta ( \log n )$ , this accords us a speed-up almost by a factor of n (up to log terms), which is very substantial in the high dimensional, large scale applications that these methods are designed for.

## 1.2.3 Fast Generation of Weakly Correlated Vectors for Machine Learning Applications

Functions with structure are of a wider interest to us and the wider community. In some problem settings (as described below), we show that by leveraging specific properties of the function, the computational cost of computing the guidance vector at each refresh step can be relatively cheap by sacrificing slightly on the initial alignment parameter. This trade-of can give us an overall decreased computational complexity.

Sparse Functions. The first class of such functions we test the algorithm on is when the function is sparse (i.e. the function depends only on a fixed subset of coordinates). Such functions are natural in machine learning, where some features might be highly correlated, resulting in redundant features and an intrinsic low dimensionality of the problem. We utilise well-established algorithms in the compressed sensing literature to construct the guidance vector in this setting. Suppose there exists a matrix Ψ satisfying the Restricted Isometry Property (RIP)

$$
( 1 - \delta _ { s } ) \| z \| ^ { 2 } \leq \| \Psi z \| ^ { 2 } \leq ( 1 + \delta _ { s } ) \| z \| ^ { 2 }
$$

for some constant $\delta _ { s }$ and for all s−sparse vectors $z \in \mathbb { R } ^ { n }$ , then many of these algorithms can find a vector that is close to the true sparse vector in $\ell _ { 2 }$ norm, namely

$$
\| y _ { T } - y ^ { * } \| \leq \rho \| y ^ { * } \| ,
$$

where $y _ { T }$ is the output of the algorithm after T iterations, and $y ^ { * }$ is the true sparse vector. In particular, we analyse the Iterative Hard Thresholding (IHT) algorithm [7], which has that $\rho$ decreases exponentially in T. This means that the cost to compute $v _ { 0 } = y _ { T } , O ( s \log ( n ) )$ directional derivatives coupled with the cost of the IHT algorithm whcih scales with $s \log ( n ) \log ( 1 / \rho )$ , is substantially lower as compared to $O ( n )$ many directional derivatives for the full gradient. For $\rho = \Theta ( 1 )$ , the logarithmic factor with $\rho$ is essentially order 1. This leads to an overall decrease in the total computational cost compared to when the problem has no structure to $O ( s \log ( n / s ) \epsilon ^ { - 2 } \xi )$

En route our investigations, we also provide an analysis of classical SSD methods in the sparse setting; see Section 3.2.3. To our knowledge, this is the first theoretical analysis of vanilla SSD in such a setup.

Additive Structure. Another class of functions that we analyse is functions that are a sum of functions, that is, $\begin{array} { r } { f ( x ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } f _ { i } ( x ) } \end{array}$ . Functions in this form are common in machine learning, such as evaluating the loss on the data set. Gradient descent is usually performed (in practice) using mini-batch as the direction, especially when $M \gg n$ . We show in Section 3.3.2 that the algorithm can also be applied to mini-batch gradient descent, with a slight tweak. For this scenario, one possible oracle of the alignment vector is to choose a mini-batch gradient of a larger batch size, namely let $B ^ { \prime } = \{ i _ { 1 } , \cdots , i _ { m ^ { \prime } } \}$ , where each $i ^ { \prime } s$ are iid sampled from [M], then set

$$
v _ { 0 } = \frac { 1 } { m ^ { \prime } } \sum _ { i \in B ^ { \prime } } \nabla f _ { i } ( x _ { 0 } ) ,
$$

where the batch size $m ^ { \prime }$ can potentially be diferent from m. Under the bounded variance assumption and a specific rate of updating the guidance vector, we show that the iteration complexity scales as $O \left( \epsilon ^ { - 4 \left( \frac { 1 + \beta } { 1 - \beta } \right) } \right)$ , for some $\beta \in ( 0 , 1 )$

In both cases, the structure of the problem allows us to compute a guidance vector in the refresh steps at a cost that is much cheaper than the full gradient. At the same time, we also enjoy the benefits of the original SSD method, where the dimension per step we consider is vastly reduced.

We believe that the idea and concepts of persistence of memory can be fruitfully applied to wider range of stochastic optimisation.

## 1.2.4 Computational Cost

The computational overhead in our approach can be divided into two principal components – one due to the random gradient sketch at every iteration, and the other due to the generation of the guidance vector at the refresh steps.

The generation of the random subspace at every step is done as usual in an SSD-type approach, essentially via d independent Gaussian random vectors (followed by certain orthogonal projections to get the desired structure of the subspace). We discuss the details of the generation of random subspace in Section 2.2; here, we only observe that the cost of each random subspace generation is $O ( n d ^ { 2 } )$ , which is a similar order as the case of the classical SSD method.

Once the random subspace is generated at a particular iteration, represented usually as an orthogonal basis, the gradient sketch along this subspace can be computed simply in terms of the directional derivatives along the rows of this matrix. If the computational cost of each directional derivative is $\xi ,$ this yields a total cost of $O ( n d ^ { 2 } + d \xi )$ for each SSD step.

From an algorithmic point of view, it might be even more inexpensive to use independent Gaussian random vectors orthogonal to the guidance vector in order to construct the random projection (as opposed to fully orthogonal columns). For a quick estimate, generation of d such vectors is possible at $O ( n d )$ cost (as opposed to $O ( n d ^ { 2 } )$ for fully orthogonal vectors). In fact, independent Gaussian random vectors enjoy approximate orthogonality properties which would already sufice for our method to give efective results. However, in this paper we focus mostly on the orthogonal vectors setup for greater conceptual clarity and ease of presentation, leaving the analogous theoretical development of the independent Gaussian vectors alternative for another occasion. For the remaining part of the paper, we assume the cost of generating this matrix to be $O ( n d )$ , which only difers from that of the orthogonalised version by a factor of $d ,$ which we typically take to be orders smaller than n.

This brings us to the computational cost of the refresh steps. Applications in machine learning and statistics typically involve optimisation objectives that are highly structured. In this work, we investigate some of the most common structures widely present in such objectives – namely, sparsity and mini-batch structure. We show that well-aligned guidance vectors in these settings can be computed in a fast and relatively inexpensive manner (in comparison to full gradient computations); see the discussion in Section 3 for details. In the sparse setting with sparsity parameter s, this leads to an overall computational cost of $O ( s \log ( n ) \xi \epsilon ^ { - 2 } )$ (c.f. Theorem 3). In a mini-batch setting, this leads to a total cost of $O \left( \epsilon ^ { - 4 \frac { 1 + \beta } { 1 - \beta } } ( n d + m d \xi ) + \epsilon ^ { \frac { - 4 } { 1 - \beta } } m ^ { \prime } n \xi \right)$ , where m<sup>′</sup> is refresh batch size and may be taken as

$$
m ^ { \prime } > \frac { \gamma _ { 0 } \sigma ^ { 2 } } { ( 1 - \gamma _ { 0 } ) \Vert \nabla f ( x ) \Vert ^ { 2 } } ,
$$

where $\sigma ^ { 2 }$ is the variance of the stochastic gradient, m is the mini-batch size in each iteration and $\gamma _ { 0 }$ is the desired initial alignment. (For more details, we refer the reader to Section 3.3.2.)

Specifically in the sparse setting, the overall computational cost may be seen to be smaller than standard SSD, which is $O ( n \xi \epsilon ^ { - 2 } )$ . For a completely unstructured objective function, perhaps the only reliable way to compute a guidance vector would be compute a full gradient at the refresh steps. This cost may be estimated as $O ( n \xi )$ , which assumes the full gradient is computed via n directional derivatives (c.f. Theorem 2). It may be noted that even in this “worst case scenario” for our approach, the overall cost $O ( n \xi \epsilon ^ { - 2 } )$ is comparable to the standard SSD, and in turn to full space gradient descent.

## 1.2.5 Parallelisation and Hardware Eficiency

An important aspect of our approach is that it yields itself seamlessly to parallelisation techniques, which is a particularly salutary property in view of modern GPU-based computational architectures that are particularly strong in parallel computation. In particular, computing the gradient sketches at each (guided) SSD step consists of d directional derivatives that can be computed independently. In conjunction with the fact that d is typically small (e.g. O(log n)), this leads to practically feasible parallelisation – the entire sketch may fit comfortably within the parallel capacity of the accelerator unit used. Although the total arithmetic work associated with d directional derivatives need not be equal to that of a single derivative evaluation, their wall-clock cost can be substantially reduced by parallel execution.

Low-dimensional sketches may also ofer advantages from the perspective of the memory hierarchy. When the data and intermediate quantities required for the computation fit within the cache or fast on-device memory of the processor (say VRAM of the GPU), repeated transfers to slower levels of the memory hierarchy (say storage devices) can be reduced. Since data movement is an important cost on modern accelerator architectures, this cache locality provides a complementary hardware-level motivation for keeping the optimisation computation low-dimensional.

In the case of the refresh steps, the computation of the guidance vector involves similar number of evaluations of independent directional derivatives. The number of such directional derivatives is larger than $d ,$ but as already noted, much smaller than the ambient dimension n in structured settings. As such, the computation of guidance vectors in the refresh steps can also be easily parallelised, which leads to overall beneficial parallelisability properties of our approach.

## 1.2.6 Interpolation Between Random Subspace and Full Gradient Descent

Our method based on persistence of memory can be fruitfully viewed as an interpolation between classical SSD and full space gradient descent. A natural interpolation parameter is given by the initial alignment parameter $\gamma _ { 0 }$ and the . If $\gamma _ { 0 } = d / n$ (in the case where the guidance vector is a projection onto a uniform d−dimensional subspace), there is no concept of guidance to the random subspace and we are back to the setting of classical SSD. On the other hand, if $\gamma _ { 0 } = \delta = 1$ , we are essentially operating with full, exact gradients at each iteration, and we are in the regime of full-space gradient descent.

Our approach focusses, in spirit, on the setting of $\gamma _ { 0 }$ and δ being bounded away from both 0 and 1. The proximity to 0 and 1 of these parameters may be though of as modulating how close the method is to classical SSD and full-space GD respectively. In the intermediate regime of these parameters, which is where we work, we can profit from the beneficial aspects of both these standard methods. In structured settings such as those with sparsity and mini-batches, this is possible even at a computational advantage.

## 1.2.7 Local Regime and Fast Alignment of Gradient Vectors

In gradient descent for strongly convex functions, it is reasonably well-understood that if we start from a local neighbourhood of the optimum (in other words, a so-called warm start), then the normalised gradient $\nabla f ( x _ { k } ) / \| \nabla f ( x _ { k } ) \|$ converges to the smallest eigenvector of the Hessian of the objective at the true optimum. The essential reason for this is that, in such local regimes, the evolution of the gradient can be efectively captured by a power iteration involving this Hessian(see, eg, [37] for related discussions).

The persistence of memory approach to SSD does not lend itself to such a simple geometric recursion; indeed, the presence of the guidance vector significantly complicates the iterative behaviour even in the local regime. Leveraging a detailed and delicate analysis that traces the gradient descent dynamics in the local regime with a modified scaling and step size, we can demonstrate that the alignment between the normalised gradient and the minimum eigenvector of the Hessian still holds true. In this endeavour, we demonstrate that once we are in the local regime (ie, have a warm start), it sufices to fix the guidance vector once and for all. In other words, no refresh steps are necessary for the persistence of memory method to be efective in such a setting. We would like to point out that, in contrast, the classical SSD method does not appear to have such alignment properties in local regimes, which once again illustrates the benefits of our persistence of memory approach to SSD.

This makes our method in the setting of warm starts to be particularly attractive even for objective functions without any structure. Indeed, even in large dimensional optimisation problems, it is quite reasonable to justify the computation of the full gradient once and for all – this will be made at the beginning of descent in the local regime, and will be repeatedly used in all iterations thereafter. For functions with structure, further computational savings may be obtained (bypassing the full gradient computation) in the presence of a salutary spectral properties of the Hessian at the optimum (such as a low-lying spectral gap) or sparsity in the function. The alignment holds with probability at least $1 - e ^ { - c d \tau ^ { 2 } }$ , where $\tau \lesssim \lambda _ { 2 } - \lambda _ { 1 }$ . When the bottom gap $\lambda _ { 2 } - \lambda _ { 1 }$ is $\Theta ( 1 )$ : for instance, when one feature has low variance while the remaining dimensions are well spread, as is common in representation learning, where the covariance of learned features often has one direction that is nearly degenerate while the rest of the representation remains well-conditioned. In such cases, τ need not decay with dimension, and the event still holds with high probability. When the function is sparse (i.e. s grows as a small power of n such as $s = n ^ { \iota } )$ , a cost of $O ( s \log ( n ) \times ( n \log ( n ) + \xi ) )$ , as compared to $O ( s \log ( n ) \times ( n + \xi ) )$ to obtain a suficiently good initial alignment $\gamma _ { 0 } = \Theta ( 1 )$ . These considerations are discussed in detail in Section 2.4.

## 1.3 Application Domains for Stochastic Subspaces

The motivation for stochastic subspace methods is not simply the idea that the computation of a small number of directional derivatives can be cheaper than computing a full gradient. Indeed, reverse mode automatic diferentiation (AD) has revolutionised gradient computations of diferentiable function, by providing a principled way to evaluate the full gradient with an additional cost that is only a small constant multiple of the cost of evaluating the function [4, 19]. Nevertheless, there are important settings where the arithmetic cost is not the principal bottleneck. In large-scale optimisation, the dominant cost may instead arise from memory, communication, diferentiation through long computational trajectories, optimiser state or the absence of a diferentiable gradient oracle. These settings provide natural applications for stochastic subspace methods.

## 1.3.1 Memory Constrained Optimisation

The computational eficiency of reverse mode AD comes with a potentially significant memory requirement. Intermediate quantities from the forward computation must be stored or recomputed during the backward sweep (c. f. the chain rule), implying that memory usage can become prohibitive for large or deeply nested computational graphs [4,19]. In contrast, directional derivatives $D _ { u } f ( x ) =$ $\langle \nabla f ( x ) , u \rangle$ can be propagated using forward mode AD without storing the complete reverse-mode tape. Consequently, when the subspace dimension d is small, SSD trades the memory requirements of reverse diferentiation for a small collection of forward directional computations. This distinction is particularly relevant in the training of large-models: MeZO demonstrated inference-level memory requirements (compared to training-level memory requirements) for fine-tuning of language-model through forward only computations [32], while the more recent SubZero method explicitly uses random low-dimensional perturbation subspaces to improve zeroth-order LLM fine-tuning [53]. Thus, even where full gradients are computationally eficient, subspace methods can enable optimisation when reverse-mode memory is the limiting resource.

## 1.3.2 Communication-Constrained Optimisation

In distributed and federated optimisation, communication can be substantially more expensive than local computations. A standard first-order method may require a worker to communicate vectors (gradient or update direction) in $\mathbb { R } ^ { n }$ in every round, with n ranging from millions to billions of parameters in modern models. Suppose communicating parties instead share a subspace $P \in \mathbb { R } ^ { n \times d }$ such as through common pseudorandom seeds, only the d projections need to be transmitted, reducing the communication cost from $O ( n )$ to $O ( d )$

FedKSeed has demonstrated the potential scale of this reduction, via its ability to perform federated full-parameter tuning of billion-parameter LLMs using random seeds and scalar directional information with less than 18KB bandwith in communication in their experiments [40]. More recently, Ferret showed that the same principle can be coupled with local optimisation, by projecting local updates onto a low-dimensional random space before communication [44]. This suggests that for $d \ll n$ , substantial communication savings can compensate for the reduced information contained in each subspace update.

## 1.3.3 Long-Horizon and Nested Optimisation

Subspace methods are also attractive when diferentiation must pass through a long inner computation, such as in bilevel optimisation, hyperparameter optimisation and meta-learning. Given an outer objective

$$
\Phi ( \lambda ) = L _ { \mathrm { o u t } } ( x _ { T } ( \lambda ) ) ,
$$

reverse AD through the $T$ inner iterations may require storing, check pointing or reconstructing the optimisation trajectory $\{ x _ { i } ( \lambda ) \} _ { i = 1 } ^ { T } \ [ 1 6 ]$ . Rather than maintaining the complete sensitivity $\textstyle { \frac { \partial x _ { t } } { \partial \lambda } }$ a subspace method can propagate only directional sensitivities $\textstyle { \frac { \partial x _ { t } } { \partial \lambda } } u _ { j }$ for $j \in [ d ]$ . When d is small, these quantities can be computed sequentially through inner computations, providing a memoryeficient alternative to forming the full gradient on the hyperparameter. This is particularly relevant when both the number of outer variables and the length of the inner optimisation trajectory are large.

## 1.3.4 Black-Box and Non-Diferentiable Optimisation

In many applications, the full gradient is not merely expensive, but potentially unavailable; for instance, in simulation-based objectives, discrete performance metrics, physical experiments and propreitary machine learning models accessible only through inference interfaces. In these cases, directional finite diferences

$$
D _ { u } f ( x ) \approx { \frac { f ( x + h u ) - f ( x - h u ) } { 2 h } }
$$

provide a natural mechanism for obtaining optimisation information within a low-dimensional subspace. Recent black-box prompt-tuning methods provide insight into this regime. $\mathrm { E . g . }$ , ZOT performs zeroth-order prompt optimisation using only inference access [54], while ZIP uses a lowdimensional representation to reduce both the dimension dependence and query cost of zeroth-order prompt tuning [39]. In such settings, SSD is not competing against an inexpensive full-gradient oracle; rather it provides an optimisation mechanism when such an oracle does not even exist.

## 1.3.5 Optimiser-State Memory

Large-scale optimisation can also be limited by the auxiliary states needed to be maintained by adaptive optimisers. Methods like Adam store first and second moment estimates with dimension comparable to the parameter vector, introducing an additional $O ( n )$ memory requirement. If useful updates are concentrated in a d-dimensional subspace, these statistics may indeed be maintained in a compressed representation. The recently proposed method GaLore demonstrates this principle by projecting layer-wise gradient into low-rank spaces to reduce optimiser state memory, while retaining full-parameter training [55]. Although GaLore still computes the full backpropagated gradient, it illustrates the broader potential of subspace representations. An SSD method that directly computes only the required directional information, could in principle reduce both gradient related and optimiser-state memory.

Taken together, these examples illustrate that the principal advantage of stochastic subspace methods need not merely be a reduction in floating-point arithmetic. Instead, a low-dimensional optimisation interface can reduce the need to materialise, store, communicate or even access fulldimensional derivative information. The resulting challenge is to then retain these computational advantages while selecting subspaces that contain suficiently rich information about descent directions, which is precisely the motivation for the guided subspace constructions considered in this work.

## 1.4 Related Works

Coordinate and Random Subspace First-Order Methods. Coordinate descent replaces a full gradient step by updates along individual coordinates or blocks; its randomised complexity and sampling rules are developed by [34, 41], and the survey of [52]. Greedy Gauss–Southwell selection can improve the rate when the extra selection cost is justified [35]. Beyond coordinate subspaces, [28] study a stochastic low-dimensional subspace method for settings in which gradients are not directly available, and [29] analyse zeroth-order optimisation with orthogonal random directions. These methods resample directions; they do not analyse reuse of a single historically informative direction.

Probabilistic Subspace Models. Probabilistic-model analysis replaces the deterministic model accuracy by a conditional high-probability condition [10]. In a non-convex random subspace framework, [9] obtain high-probability $O ( \epsilon ^ { - 2 } )$ iteration complexity for safeguarded trust-region or quadratic regularisation schemes under a probabilistic subspace gradient condition. Related derivative-free trust-region analysis in random subspaces is given by [14]. These results are important comparators for the present alignment assumption, but their safeguards and oracle models difer from a fixed-step projected gradient update.

Gradient Sketches and Variance Reduction. Another randomised first-order method SEGA builds a variance-reduced gradient estimate by accumulating random linear measurements of the gradient over time [21]. Its state is a reconstructed estimator, whereas the present method proposes to retain a single guidance direction and complete it by a fresh random orthogonal subspace. Randomised forward-mode gradient estimators based on directional derivatives are studied by [45]; standard automatic-diferentiation cost and memory distinctions are reviewed by [4, 19]. Recent work also studies accelerated projected-gradient oracles [36] and adaptive low-rank subspaces for memory-eficient model training [11, 30].

Relation to Variance-Reduction Methods There is also a conceptual connection between our guided subspace construction and control-variate techniques in stochastic optimisation. Classical variance-reduced methods, such as SVRG and SAGA, exploit auxiliary gradient information that is correlated with a stochastic gradient estimator to reduce its variance while preserving the desired expectation [12, 25]. Although our mechanism is diferent, where the guidance vector modifies the geometry of the sampled subspace rather than correcting a stochastic estimator, the underlying principle conceptually has a related flavour. Namely, auxiliary information correlated with the true gradient is used to improve the quality of the stochastic search direction. Exploring such potential connections is a natural direction for future work.

## 1.5 Structure and Notation of the Paper

We define the initial point as $x _ { 0 }$ , and the solution space lies in $\mathbb { R } ^ { n }$ . The lower dimension of the projection is denoted by d (or in our case, d+1 as we consider a d-dimensional subspace concatenated to a fixed vector $v _ { k } )$ . The alignment vector is denoted by $v _ { k }$ , whereby $\hat { v } _ { k }$ denotes the normalised vector $v _ { k } / \vert \vert v _ { k } \vert \vert$ . When necessary, we will denote the normalised form of a vector u by uˆ.

We use L as the Lipschitz constant of the derivative of the function $f ,$ and α denotes the step-size (which may contain a subscript denoting the iteration when applicable). δ denotes the lower bound on the alignment we have and $\gamma _ { k }$ represents a lower bound on the current alignment in iteration k. In the sections involving computational complexity, ξ is assumed to be the cost of computing 1 directional derivative, and $\nu ,$ the cost of updating the alignment vector.

We present the paper as follows. In Section 2, we describe the main algorithm and motivation behind it, followed by the main results on the computational cost of the algorithm under structured problems commonly found in machine learning tasks. We then take a look at how we only require to compute a well aligned vector "once and for all" under a local regime. In Section 3, we take a look at various oracles to find such a guidance vector and show that with a simple tweak to the algorithm, it can also be applied to functions of a finite sum structure. In Section 4, we briefly describe the proof techniques involved in showing the results in the previous 2 sections, which is then followed by some numerical demonstrations of the algorithm in Section 5.

## 2 Main Results

## 2.1 Preliminaries

We are working under the standard assumption that the function is L-smooth, which is the same as saying that the gradient of the function is $L { \mathrm { - L i p s c h i t z } }$ . Here, we provide the definition of such functions.

Definition 1 (L-Smoothness). A function $f : \mathbb { R } ^ { n }  \mathbb { R }$ has L-Lipschitz gradient $\mathrm { i f ~ } \exists L > 0$ such that $\forall x , y \in \mathbb { R } ^ { n }$ , we have

$$
\| \nabla f ( x ) - \nabla f ( y ) \| \leq L \| x - y \| .\tag{3}
$$

As a consequence, for any $x , y \in \mathbb { R } ^ { n }$ , we have the following inequality:

$$
f ( y ) \leq f ( x ) + \nabla { f ( x ) } ^ { \top } ( y - x ) + { \frac { L } { 2 } } \| x - y \| ^ { 2 } .
$$

In particular, letting $x = x _ { k - 1 }$ and $y = x _ { k }$ gives us

$$
f ( x _ { k } ) \leq f ( x _ { k - 1 } ) + \langle \nabla f ( x _ { k - 1 } ) , x _ { k } - x _ { k - 1 } \rangle + { \frac { L } { 2 } } \| x _ { k } - x _ { k - 1 } \| ^ { 2 }\tag{4}
$$

which will be an inequality used frequently in the later proofs.

## 2.2 Algorithm

We will now provide the conditions under which our method converges to a point $x _ { N }$ such that min $\mathsf { i } _ { k \in [ N ] } \mathbb { E } \left[ \| \nabla f ( x _ { k } ) \| ^ { 2 } \right] < \epsilon ^ { 2 }$ . Let $f : \mathbb { R } ^ { n }  \mathbb { R }$ be a L−smooth function. At each iteration, the point is updated as

$$
x _ { k } = x _ { k - 1 } - \alpha P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } )\tag{5}
$$

where $P _ { k - 1 } \in \mathbb { R } ^ { n \times ( d + 1 ) } \ ( d \ll n )$ is a matrix of the form

$$
P _ { k - 1 } = \left( \hat { v } _ { k - 1 } \quad \tilde { P } _ { k - 1 } \right) ,\tag{6}
$$

and $\tilde { P } _ { k - 1 }$ is a random $n \times d$ matrix on $v _ { k - 1 } ^ { \perp }$ such that

$$
\tilde { P } _ { k - 1 } ^ { \top } \tilde { P } _ { k - 1 } = I _ { d } \qquad \& \qquad \mathbb { E } \left[ \tilde { P } _ { k - 1 } \tilde { P } _ { k - 1 } ^ { \top } \right] = \frac { d } { n - 1 } \left( I _ { n } - \hat { v } _ { k - 1 } \hat { v } _ { k - 1 } ^ { \top } \right) .\tag{7}
$$

```latex
Algorithm 1 SSD with Persistence of Memory (SSDPM)
1: Inputs: $\alpha , d , \delta , \gamma _ { 0 }$ ▷ step size, subspace rank, alignment threshold, initial alignment
2: Initialize: $x _ { 0 }$ ▷ arbitrary initialization
3: $j  0$
4: $\begin{array} { r } { \tilde { \delta } \gets \delta + \sqrt { \frac { d } { n - 1 } } } \end{array}$
5: $\begin{array} { r } { r \gets \frac { ( 1 - \dot { 2 } \alpha L ) } { 4 \alpha L ( 1 - \alpha L ) } \log \left( \gamma _ { 0 } / \tilde { \delta } \right) } \end{array}$ ▷ Theoretical upper bound on re-uses for alignment vector
6: for $k = 1 , 2 , \dots$ do
7: if j = 0 then
8: Generate a new $v _ { k - 1 }$ as a guiding descent direction
9: $\hat { v } _ { k - 1 }  v _ { k - 1 } / \| v _ { k - }$ <sub>−1</sub>∥
10: else
11: $v _ { k - 1 }  v _ { k - 2 }$
12: end if
13: Generate $\tilde { P } _ { k - 1 } \in \mathbb { R } ^ { n \times d }$ orthogonal to $v _ { k - 1 }$
14: $P _ { k - 1 }  ( \hat { v } _ { k - 1 } \quad \tilde { P } _ { k - 1 } )$
15: $x _ { k }  x _ { k - 1 } - \alpha P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } )$
16: $j \gets ( j + 1 )$ mod r
17: end for
```

The key diference between our algorithm and the original SSD algorithm is in the generation of the guidance vector $v _ { k }$ at each step. With this algorithm, we will then present the main theorem arising from this algorithm. The initial alignment $\gamma _ { 0 }$ depends on the way the guidance vector is generated, which we will see 2 examples in the later sections.

## 2.2.1 Discussion on Algorithm

In the original SSD algorithm proposed by [28], the iterates are updated in a random projected direction of the gradient at that point. A d−dimensional subspace is randomly chosen and the gradient will be projected onto this subspace, whereby the iterates will then move in accordance to this direction. In comparison, our algorithm appends this random subspace with an additional alignment vector indicated with $v _ { k }$ , which has a correlation with the gradient of at least $\delta$ in expectation. More concretely, we have $\mathbb { E } [ \langle \hat { v } _ { k } , \nabla f ( x _ { k } ) \rangle ^ { 2 } ] \ge \delta \| \nabla f ( x _ { k } ) \| ^ { 2 }$ , where $\hat { v } _ { k }$ represents the normalised version of the vector $v _ { k }$ . Doing so ensures that the projected gradient is of size $( \delta +$ $\textstyle { \frac { d } { n - 1 } } ) \| \nabla f ( x _ { k } ) \| ^ { 2 }$ as compared to $\begin{array} { r l r } {  { \frac { d } { n } \| \nabla f ( x _ { k } ) \| ^ { 2 } } } \end{array}$ in the case of the original algorithm by [28]. At the same time, to ensure that the matrix remains a projection, the random part of the projection will be sampled randomly from $\left( I - \hat { v } _ { k } \hat { v } _ { k } \right)$ instead. We denote this random matrix as $\tilde { P } _ { k }$ , and the projection matrix we consider will be

$$
P _ { k } P _ { k } ^ { \top } = \hat { v } _ { k } \hat { v } _ { k } ^ { \top } + \tilde { P } _ { k } \tilde { P } _ { k } ^ { \top } , \qquad P _ { k } = \left( \hat { v } _ { k } \quad \tilde { P } _ { k } \right) .\tag{8}
$$

Next, we will re-use this alignment vector for r steps, until the correlation with the gradient falls below $\delta .$ In practice, this is a variable which can be tuned by the practitioner, but our analysis in Corollary 1 (upper bound on reuse) gives us a theoretical upper bound on r given the starting correlation (the correlation of the alignment vector with the gradient at the iteration where the alignment vector is first used). In the most vanilla case, we can simply choose $v _ { k } = \nabla f ( x _ { k } )$ , and the initial alignment will be 1.

We would also like to note that the computation of the projected gradient can be much cheaper than the gradient, by first computing the d directional derivatives of the gradient with respect to $P _ { k }$ . i.e. $( P _ { k } ^ { \top } \nabla f ( x _ { k } ) ) _ { j } = ( P _ { k } ) _ { \cdot , j } ^ { \top } \nabla f ( x _ { k } )$ , and then doing a matrix vector multiplication with the matrix $P _ { k }$

## 2.3 Guarantees for Iteration and Computational Complexities

Our main contribution is that the algorithm enjoys an improved iteration complexity as compared to the original SSD algorithm. Given an oracle that produces an alignment vector $v _ { 0 }$ satisfying

$$
\mathbb { E } \left[ \langle \hat { v } _ { 0 } , \nabla f ( x _ { 0 } ) \rangle ^ { 2 } \mid x _ { 0 } \right] \geq \gamma _ { 0 } \Vert \nabla f ( x _ { 0 } ) \Vert ^ { 2 }
$$

for the initial alignment, while also satisfying

$$
\begin{array} { r } { \mathbb { E } \left[ \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } \mid x _ { 0 } \right] \geq \gamma _ { k - 1 } \mathbb { E } \left[ \Vert \nabla f ( x _ { k - 1 } ) \Vert ^ { 2 } \mid x _ { 0 } \right] } \end{array}\tag{9}
$$

for consequent steps with $\gamma _ { k - 1 } \geq \delta$ (in other words, conditional on the iterate where the alignment vector is updated, consequent alignments are lower bounded by $\gamma _ { k - 1 } )$ , we obtain rates comparable to the standard GD algorithm which does not have a factor of $\textstyle { \frac { n } { d } }$ like the SSD algorithm. More precisely, we have the following theorem.

Theorem 1 (Iteration Complexity). Let the alignment vector v<sub>0</sub> satisfy $\mathbb { E } \left[ \langle \hat { v } _ { 0 } , \nabla f ( x _ { 0 } ) \rangle ^ { 2 } \mid x _ { 0 } \right] \ge$ $\gamma _ { 0 } \| \nabla f ( x _ { 0 } ) \| ^ { 2 }$ with $\gamma _ { 0 } > \delta + \sqrt { d / n }$ when updated. Then the number of iterations to obtain a solution satisfying min $_ { i } \mathbb { E } \left[ \left. \nabla f ( x _ { i } ) \right. ^ { 2 } \right] < \epsilon ^ { 2 }$ is at least

$$
N \ge \frac { \alpha ^ { - 1 } \left( 1 - \frac { \alpha L } { 2 } \right) ^ { - 1 } \left( f ( x _ { 0 } ) - f ^ { * } \right) } { \frac { 4 \alpha L ( 1 - \alpha L ) } { ( 1 - 2 \alpha L ) \log ( \gamma _ { 0 } / \tilde { \delta } ) } \left( 1 - \frac { d } { n - 1 } \right) \gamma _ { 0 } + \frac { d } { n - 1 } - \tilde { t } _ { 0 } } \epsilon ^ { - 2 } ,\tag{10}
$$

where $\begin{array} { r } { \tilde { t } _ { 0 } = t _ { 0 } \left( 1 - \frac { d } { n - 1 } \right) \sqrt { \frac { d } { n - 1 } } \left( 1 - \frac { 4 \alpha L \left( 1 - \alpha L \right) } { \left( 1 - 2 \alpha L \right) \log \left( \gamma _ { 0 } / \tilde { \delta } \right) } \right) } \end{array}$ and $\begin{array} { r } { t _ { 0 } \in ( 4 / 9 , 1 / 2 + O ( d / n ) ) . \ \tilde { \delta } = \delta + \sqrt { \frac { d } { n - 1 } } } \end{array}$ and α is the step size to be chosen satisfying

$$
\frac { 1 } { L } \cdot \frac { K } { 2 + K + \sqrt { 4 + K ^ { 2 } } } < \alpha \leq \frac { 1 } { 2 L } , \quad K = \log \left( \frac { \gamma _ { 0 } } { \tilde { \delta } } \right) \cdot \frac { t _ { 0 } \left( 1 - \frac { d } { n - 1 } \right) \sqrt { \frac { d } { n - 1 } } - \frac { d } { n - 1 } } { \left( 1 - \frac { d } { n - 1 } \right) \left( \gamma _ { 0 } + t _ { 0 } \sqrt { \frac { d } { n - 1 } } \right) } .\tag{11}
$$

Remark 1. The lower bound on the step size is natural, considering that the numerator has a factor $\alpha ^ { - 1 }$ . If step size is taken to be too small, say order $d / n$ , then efectively we get Kozak’s rate. Simplifying expression (11), we have

$$
\alpha > \alpha ^ { - } \approx \frac { K } { 4 L } \approx \frac { \log ( \gamma _ { 0 } / \tilde { \delta } ) } { 9 L \gamma _ { 0 } } \sqrt { \frac { d } { n - 1 } } ,
$$

which is much larger than the step size of $d / 2 n L$ used in Kozak’s algorithm.

Observe that in the denominator, apart from a $d / n$ factor, we have an additional term approximately $\frac { 4 \alpha L ( 1 - \alpha L ) \gamma _ { 0 } } { ( 1 - 2 \alpha L ) \log ( \gamma _ { 0 } / \tilde { \delta } ) }$ . When $d \ll n ,$ , the original SSD sufers from the curse of dimensionality due to the $d / n$ factor. In comparison, by asserting $\gamma _ { 0 } \ge \delta = \Theta ( 1 )$ and having α not too small (which can be realised by the upper bound), the denominator is of order $\Theta ( 1 )$ , simplifying the iteration complexity above to $O ( \epsilon ^ { - 2 } )$ . This is comparable to the standard gradient descent algorithm. Additionally, δ controls the extent of how much randomness we want to incorporate into the algorithm; when $\delta  { \frac { d } { n } } ;$ , we recover the original SSD algorithm, where $v _ { k }$ can be thought of the projection of the gradient onto a d−dimensional subspace in $\mathbb { R } ^ { n }$ . On the other hand, when $\delta  1$ , we recover the standard gradient descent, where at each iteration, the projected gradient carries the full gradient information $v _ { k } = \nabla f ( x _ { k } )$ . More specifically, if $\nabla f ( x _ { 0 } ) = v _ { 0 }$ , then $\gamma _ { 0 } = 1$ . On the other hand, if v<sub>0</sub> is a random vector in $\mathbb { R } ^ { n }$ , then we have $\gamma _ { 0 } = 1 / n$ . More discussions of how to choose v<sub>0</sub> under specific structure of f will be covered in Section 3.

Next, we present the overall computational complexity of the algorithm in the general setting, where there is no additional known structure of the objective function beyond L-smoothness.

Theorem 2 (Computational Complexity). Let $\xi$ be the cost for computing a directional derivative and ν be the cost of a refresh. Then, the computational cost for computing N steps denoted by Equation 10 is of the order

$$
O \left( \epsilon ^ { - 2 } \gamma _ { 0 } ^ { - 1 } \left[ d \log \left( \frac { \gamma _ { 0 } } { \delta } \right) ( \xi + n ) + \nu \right] \right) .\tag{12}
$$

In particular, if we do not have a good way to obtain a refresh, and simply assume that we use n directional derivatives to compute the gradient in the elementary coordinate space, we regardless retrieve the cost of the standard GD algorithm. If $\xi = O ( n )$ , we can simplify Equation 12 to $O ( \epsilon ^ { - 2 } n \xi )$ . We note that the focus of our algorithm is in the special cases where the function has an underlying structure which allows for a guidance vector to be obtained much cheaper. We first present their computational complexities here, before going into details in Section 3.

Suppose the objective function is s-sparse, meaning that it only depends on s of the coordinates of the input space, we have an improved complexity contributed by the reduced cost of a refresh step in comparison to computing a full gradient.

Theorem 3 (Sparse Computational Complexity). Let the objective function be s-sparse and the cost to compute a directional derivative to be $\xi .$ . The expected computational cost for the algorithm to produce min<sub>k</sub> E $\left[ \left. \nabla f ( x _ { k } ) \right. ^ { 2 } \right] \leq \epsilon ^ { 2 }$ is given by

$$
O \left( \frac { \epsilon ^ { - 2 } } { \gamma _ { 0 } } \left[ \left( d \log ( \gamma _ { 0 } / \delta ) + k ^ { \prime } \log \left( \frac { 1 } { 1 - \gamma _ { 0 } } \right) \right) \times n + \left( d \log ( \gamma _ { 0 } / \delta ) + k ^ { \prime } \right) \times \xi \right] \right)\tag{13}
$$

where $k ^ { \prime } \gtrsim s \log ( n / s )$ and $\gamma _ { 0 }$ is the initial alignment desired.

Remark 2. In the scheme where $d = O ( \log ( n ) )$ and $\gamma _ { 0 } = \Theta ( 1 )$ , the first term for each of the factors for $n , \xi$ is dominated by the $k ^ { \prime }$ term, giving us a complexity of $O ( \epsilon ^ { - 2 } s \log ( n ) ( n + \xi ) )$ . Assuming that $\xi = O ( n )$ minimally, since the function is n−dimensional, and computationally it requires at least $O ( n )$ operations on the input space, then the complexity is $O ( s \log ( n ) \xi \epsilon ^ { - 2 } )$ , scaling in the order of the sparsity s instead of the full dimension n.

For an objective function with a finite sum structure, we can perform a mini-batch variant of our proposed algorithm to obtain the following computational complexity.

Theorem 4 (Mini-batch Computational Complexity). Let ν be the cost of updating the alignment vector and ξ be the cost of computing a directional derivative for one function $f _ { i }$ . Then, the cost of the algorithm to achieve a point min<sub>k</sub> $\mathbb { E } \left[ \left. \nabla f ( x _ { k } ) \right. ^ { 2 } \right] < \epsilon ^ { 2 }$ is given by

$$
O \left( \epsilon ^ { - 4 \frac { 1 + \beta } { 1 - \beta } } ( n d + m d \xi ) + \epsilon ^ { \frac { - 4 } { 1 - \beta } } \nu \right) ,\tag{14}
$$

where m is the batch size of the algorithm and $\beta \in ( 0 , 1 )$ fixed. $I f \nu = O ( m ^ { \prime } n \xi )$ (cost of computing n directional derivatives for m<sup>′</sup> functions), we have

$$
O \left( \epsilon ^ { - 4 { \frac { 1 + \beta } { 1 - \beta } } } ( n d + m d \xi ) + \epsilon ^ { { \frac { - 4 } { 1 - \beta } } } m ^ { \prime } n \xi \right) .\tag{15}
$$

## 2.4 Local Regimes and Gradient Alignment Phenomena

The main bottleneck of this algorithm is the fact that we have to update the alignment vector after a certain number of steps. However, it turns out that under special circumstances, we forgo the need to update this vector and it is still suficiently well-aligned to the gradient.

It is well-understood (Theorem 10.1.3 in [37]) that in gradient descent for strongly convex functions, if we start from a point where the gradient has non-zero correlation with the smallest eigenvector $u _ { 1 }$ of the Hessian at the optimal solution, and if this point is suficiently close to the optimal solution (in the local regime), then we have that the normalised gradient $\nabla f ( x _ { k } ) / \| \nabla f ( x _ { k } ) \|$ converges to $u _ { 1 }$ . We will show that our algorithm also enjoys a similar result in a finite horizon. Define the finite horizon

$$
K _ { \epsilon } = \operatorname* { m i n } \{ k \mid \| \nabla f ( x _ { k } ) \| \leq \epsilon \} .\tag{16}
$$

We will also assume additional properties of the function and the alignment vector, namely the function $f \in C ^ { 3 }$ and a weakly correlated alignment vector exists at every step with high probability, which increases to 1 as the correlation increases to 1.

Assumption 1. There exists an alignment vector with alignment $\delta$ exists with high probability. More formally, for all $k \leq K _ { \epsilon }$ , denote the event

$$
{ \mathcal { K } } _ { k } ( \delta ) : = \left\{ \langle { \hat { v } } _ { k } , \nabla f ( x _ { k } ) \rangle ^ { 2 } \geq \delta \| \nabla f ( x _ { k } ) \| ^ { 2 } \right\}\tag{17}
$$

and assume

$$
\begin{array} { r } { \mathbb { P } \left[ \mathcal { K } _ { k } ( \delta ) \mid \nabla f ( x _ { k } ) \right] \geq 1 - p _ { v } ( \delta ) , } \end{array}\tag{18}
$$

where $p _ { v } ( \delta )$ is an exponentially decreasing function of $\delta .$

## 2.4.1 Strong Alignment in Local Regimes

For this section, we will assume that the function is strongly convex and that the Hessian at the optimal point, denoted by H, has eigenvalues and eigenvectors $\{ ( \lambda _ { i } , u _ { i } ) \} _ { i = 1 } ^ { n }$ , where $\lambda _ { 1 } < \lambda _ { 2 } < \cdots <$ $\lambda _ { n }$ . Under this condition, we show that for any given threshold $\delta _ { 0 }$ , if $\langle \nabla f ( x _ { 0 } ) , u _ { 1 } \rangle \neq 0$ , then there exists a $k _ { 1 }$ iterate such that $\langle u _ { 1 } , \nabla f ( x _ { k _ { 1 } } ) \rangle ^ { 2 } \geq \bar { \delta } _ { 0 } \| \nabla f ( x _ { k _ { 1 } } ) \| ^ { 2 }$ . What this means is that if the initial gradient is not orthogonal to $u _ { 1 }$ , eventually we have that the gradient of the function is strongly aligned to this vector.

This strong alignment in the $k _ { 1 } ^ { t h }$ step allows us to freeze the alignment vector. We show that even after this step, the gradient $\nabla f ( x _ { k } )$ remains strongly aligned with $u _ { 1 }$ , coupled with the fact that $v _ { k } = v _ { k _ { 1 } }$ is also strongly aligned with $u _ { 1 }$ , we naturally get that $\nabla f ( x _ { k } )$ is also strongly aligned with $u _ { 1 }$ . The next proposition gives us a lower bound on this alignment.

Proposition 1. Let $u , v ,$ w be unit vectors in $\mathbb { R } ^ { n }$ such that

$$
\langle u , v \rangle ^ { 2 } \geq \delta _ { 1 } , \qquad \langle v , w \rangle ^ { 2 } \geq \delta _ { 2 } .
$$

$I f \delta _ { 1 } + \delta _ { 2 } \geq 1$ , then

$$
\begin{array} { r } { \langle u , w \rangle ^ { 2 } \geq \left( \sqrt { \delta _ { 1 } \delta _ { 2 } } - \sqrt { ( 1 - \delta _ { 1 } ) ( 1 - \delta _ { 2 } ) } \right) ^ { 2 } . } \end{array}\tag{19}
$$

Note that the condition that $\delta _ { 1 } + \delta _ { 2 } \geq 1$ is natural, as we would expect that if the sum of the $2$ angles is larger than 1, efectively in the worse case, u, v will be orthogonal. Nonetheless strong alignment between $u _ { 1 } , \nabla f ( x _ { k } )$ and $u _ { 1 } , v _ { k _ { 1 } }$ means that we no longer have to refresh the alignment vector after the $k _ { 1 } ^ { t h }$ step, efectively making the algorithm as cheap as the original Kozak’s algorithm, while enjoying the faster convergence due to the alignment vector.

Following that, we state the main theorem below.

Theorem 5. Let $f \in C ^ { 3 }$ and strongly convex with parameter $\mu$ and $x ^ { * }$ be the unique minimiser. Suppose the Hessian of f near $x ^ { * } \colon H = \nabla ^ { 2 } f ( x ^ { * } )$ , is locally Lipschitz around radius $r _ { 0 } > 0$ with simple eigenvalues $\lambda _ { 1 } < \cdots < \lambda _ { n }$ . Let the initial gap satisfy $\begin{array} { r } { f ( x _ { 0 } ) - f ^ { * } \leq \frac { 1 } { 2 } \mu r _ { 0 } ^ { 2 } } \end{array}$ , where $f ^ { * }$ is function value at $x ^ { * }$ . Consider the finite horizon $K _ { \epsilon } = \operatorname* { m i n } \{ k \mid \| \nabla f ( x _ { k } ) \| \leq \epsilon \}$ and suppose Assumption 1 holds in this horizon, with δ satisfying

$$
\delta \gtrsim 1 - \operatorname* { m i n } \left( \omega \frac { \mu _ { 1 } } { \mu _ { 2 } ( 1 + \eta ) } \frac { \lambda _ { 2 } - \lambda _ { 1 } } { \lambda _ { 1 } } , \frac { \lambda _ { 2 } - \lambda _ { 1 } } { ( 1 + \eta ) \omega ^ { - 1 } \frac { \mu _ { 2 } } { \mu _ { 1 } } \lambda _ { 1 } + ( 1 - \omega ^ { 2 } ) ^ { - 1 / 2 } \lambda _ { n } \mu _ { 1 } } \right) ^ { 2 } ,
$$

where $\omega > 0$ is the initial alignment $| \langle u _ { 1 } , \nabla f ( x _ { 0 } ) \rangle | = \omega \| \nabla f ( x _ { 0 } ) \|$ and $\mu _ { i } = 1 - \alpha \lambda _ { i }$ . Then with Algorithm (1) with $\alpha \in ( 0 , 1 / \lambda _ { n } )$ , given $\delta _ { 0 } \in ( 1 / 2 , 1 ]$ , there exists an iteration $k _ { 1 } < K _ { \mathrm { { \ell } } }$ such that

$$
\begin{array} { r } { \mathbb { P } \left[ \langle \nabla f ( x _ { k _ { 1 } } ) , u _ { 1 } \rangle ^ { 2 } \geq \delta _ { 0 } \| \nabla f ( x _ { k _ { 1 } } ) \| ^ { 2 } \right] \geq 1 - e ^ { - \frac { 1 } { 8 } k _ { 1 } \left( 1 - p _ { d e c a y } ( \hat { \tau } , \delta , d ) \right) } , } \end{array}\tag{20}
$$

where $p _ { d e c a y } ( \hat { \tau } , \delta , d ) = e ^ { - c d \hat { \tau } ^ { 2 } } + p _ { v } ( \delta )$ , and $\hat { \tau } \in ( 0 , 1 )$ . Furthermore, for $k > k _ { 1 }$ , we fix the alignment vector

$$
v _ { k } = v _ { k _ { 1 } } = \nabla f ( x _ { k _ { 1 } } )
$$

and scale the random part of the matrix $P _ { k }$ by a factor of $\sqrt { \frac { n - 1 } { d } }$ , i.e. $\tilde { P } _ { k }  \sqrt { \frac { n - 1 } { d } } \tilde { P } _ { k }$ . Then with probability at least

$$
\left( 1 - \sum _ { i = 1 } ^ { n } e ^ { - c d \tau _ { i } ^ { 2 } } - e ^ { - c d \tau ^ { 2 } } \right) ^ { K _ { \epsilon } - k _ { 1 } } ,
$$

for $k \in ( k _ { 1 } , K _ { \epsilon } )$ , we still have $\begin{array} { r } { \frac { | u _ { i } ^ { \top } \nabla f ( x _ { k } ) | } { | u _ { 1 } ^ { \top } \nabla f ( x _ { k } ) | } = O ( \lambda _ { i } \tau _ { i } ) } \end{array}$ , where

$$
\begin{array} { r l } & { \tau _ { 1 } \lesssim \displaystyle \frac { 1 - \alpha \lambda _ { 1 } } { 1 - \alpha \lambda _ { 2 } } \frac { \lambda _ { 2 } - \lambda _ { 1 } } { \lambda _ { 1 } } \sqrt { \delta _ { 0 } } } \\ & { \tau _ { i } \lesssim \displaystyle \frac { \lambda _ { i } - \lambda _ { 1 } } { \lambda _ { i } } \sqrt { 1 - \delta _ { 0 } } } \\ & { \alpha _ { k _ { 1 } } = \displaystyle \frac { d } { L } \frac { 1 - 4 \tau \delta _ { 0 } ( 1 - \delta _ { 0 } ) } { ( n - 1 ) ( - \tau + 4 \delta _ { 0 } ( 1 - \delta _ { 0 } ) ( 1 + \tau ) ) + d ( 2 \delta _ { 0 } - 1 ) ^ { 2 } } , } \end{array}
$$

with some $\tau \in \mathsf { \Gamma } ( 0 , 1 )$ and $\alpha _ { k _ { 1 } }$ being the step size under the rescaled random matrix to ensure exponential decay of the gradient. The $\lesssim$ hides some constant and exponentially decaying term under the local regime, and the constant $c > 0$ is a universal independent of all other variables. The projected dimension d is also required to fulfill

$$
d \geq \frac { 1 } { c } \operatorname* { m a x } \left\{ \log ( n + 1 ) \cdot \operatorname* { m a x } \left\{ \frac { 1 } { \delta _ { 0 } } \left( \frac { \lambda _ { 1 } } { \lambda _ { 2 } - \lambda _ { 1 } } \frac { 1 - \alpha \lambda _ { 2 } } { 1 - \alpha \lambda _ { 1 } } \right) ^ { 2 } , \frac { \operatorname* { m a x } _ { 2 \leq i \leq n } \left( \frac { \lambda _ { i } } { \lambda _ { i } - \lambda _ { 1 } } \right) ^ { 2 } } { ( 1 - \delta _ { 0 } ) } , \frac { 1 } { \tau ^ { 2 } } \right\} , \frac { 1 } { \bar { \tau } ^ { 2 } } \log \left( \frac { 1 } { 1 - p _ { v } ( \delta ) } \right) \right\} .\tag{21}
$$

Remark 3. This means that we can control the alignment of the gradient with the non leading eigenvectors to be arbitrarily small, by controlling the closeness of $\delta _ { 0 } ~  ~ 1$ . In other words, the gradient remains weakly correlated with the leading eigenvector, and consequently, is weakly correlated with the frozen alignment vector.

Remark 4. Although the step size $\begin{array} { r } { \alpha _ { k _ { 1 } } \approx \frac { d } { n L \delta _ { 0 } ( 1 - \delta _ { 0 } ) } } \end{array}$ is larger than the step size when $k < k _ { 1 }$ , it can be arbitrarily close to it by taking δ<sub>0</sub> → 1. This means that our step size does not sufer the same issue as in the Kozak’s SSD algorithm, which is of order $\textstyle { \frac { d } { n } }$

Remark 5. Under certain scenarios (which we will discuss in the following section), the bottom eigengap $\lambda _ { 2 } - \lambda _ { 1 } = \Theta ( 1 )$ This means that the bounds on $\tau _ { i }$ solely depend on the alignment parameter $\delta _ { 0 }$ . In fact, since $\lambda _ { i } > \lambda _ { 2 }$ for $i > 2 , \tau _ { i }$ for the corresponding $i ^ { \prime } s$ is also allowed to be larger, making the success probability larger. Suppose $\begin{array} { r } { d = \frac { ( \log ( n ) ) ^ { \frac { 1 } { m } } } { c \operatorname* { m i n } _ { i } \tau _ { i } ^ { 2 } } } \end{array}$ for $m > 1$ , then d need not be too large to achieve a probability of $( 1 - n ^ { 1 - m } ) ^ { K _ { \epsilon } - k _ { 1 } }$

## 2.4.2 Application Scenarios in which Local Regime is Natural

We will take a look at some applications where the structure of the problem leads to advantage in the local regime.

The Order of the Bottom Spectral Gap. For a matrix H, define the bottom gap as

$$
\delta ( H ) = \lambda _ { 2 } ( H ) - \lambda _ { 1 } ( H ) .
$$

In problems where a single feature direction is unusually weak (nearly degenerate) while the remaining directions are uniformly well-conditioned (variance of order one), we have $\delta ( H ) = \Theta ( 1 )$ . Spectral models consisting of isolated eigenvalues separated from a bulk have been studied extensively in random matrix theory and high dimensional statistics, such as through spiked covariance models and finite rank perturbations [1].

Such a spectral structure can be motivated by several settings in machine learning and statistics. For example, in linear regression with highly correlated inputs, a particular combination of nearly redundant features may have small but non zero variance, while the remaining directions retain $O ( 1 )$ variance. More generally, how the non uniformity of the spectrum of the covariance afects overparameterised linear and kernel regression has been studied in benign and tempered overfitting [3, 33, 47]. Related weak directions can also arise in latent variable models, such as VAEs, where posterior collapse happens when one or more latent variables become uninformative. This has been related to the geometry of the objective function and the covariance structure of the data [31]. Although these examples do not immediately imply an isolated bottom eigenvalue of the Hessian, it motivates the considering of spectral models containing a distinguished weak direction.

A rather straightforward example to look at is ridge regression models, which naturally lead to positive definite matrices of the form

$$
H = \frac { 1 } { m } X ^ { \top } X + \lambda I , \qquad X = [ x _ { 1 } , x _ { 2 } , \cdot \cdot \cdot x _ { m } ] ^ { \top } \in \mathbb { R } ^ { m \times n } .
$$

With $\begin{array} { r } { S = \frac { 1 } { m } X ^ { \top } X } \end{array}$ , we get

$$
\delta ( H ) = \lambda _ { 2 } ( S ) - \lambda _ { 1 } ( S ) = \delta ( S ) .
$$

Essentially, this means that the regularisation does not change the bottom-gap, despite making the associated function strongly convex. In practice, X might be our data set which is sampled from some population with covariance $\ b { \Sigma } \in \mathbb { R } ^ { n \times n }$ . Using the spiked model from random matrices as a concrete example, let the rows $x _ { i } \in \mathbb { R } ^ { n }$ of X be sampled independently from

$$
\begin{array} { r } { x _ { i } \sim N ( 0 , \Sigma ) , \Sigma = \sigma ^ { 2 } I _ { n } - c \cdot u u ^ { \top } , } \end{array}
$$

where u is a fixed unit vector, $\sigma ^ { 2 } > c$ and $c > 0$ are fixed constants. Then $\lambda _ { 1 } ( \Sigma ) = \sigma ^ { 2 } - c$ and $\lambda _ { i } = \sigma ^ { 2 }$ for $i \geq 2$ , implying that the population bottom-gap is $c ~ ( \mathrm { i . e . } ~ \delta ( \Sigma ) = c )$ . In this case, a suficient condition on the bottom gap for Σ ensures that the sample covariance also has a gap of the same order. Moreover, by Weyl’s inequality

$$
\delta ( S ) = \lambda _ { 2 } ( S ) - \lambda _ { 1 } ( S ) \geq \lambda _ { 2 } ( \Sigma ) - \left\| S - \Sigma \right\| _ { o p } - \lambda _ { 1 } ( \Sigma ) - \left\| S - \Sigma \right\| _ { o p } = \delta ( \Sigma ) - 2 \left\| S - \Sigma \right\| _ { o p }
$$

When S concentrates around $\Sigma \ ( \mathrm { i . e . } \ \| S - \Sigma \| _ { o p } = o ( c ) )$ , we get

$$
\delta ( H ) = \delta ( S ) = c + o ( 1 ) ,
$$

implying that the bottom-gap for the problem is asymptotically c. [27] showed that for iid Gaussian rows with covariance $\Sigma .$

$$
\| S - \Sigma \| _ { o p } \lesssim \| \Sigma \| _ { o p } \left( \sqrt { \frac { r _ { \mathrm { e f f } } } { m } } + \frac { r _ { \mathrm { e f f } } } { m } \right) , \qquad r _ { \mathrm { e f f } } = \frac { \mathrm { t r } ( \Sigma ) } { \| \Sigma \| _ { o p } } .
$$

Equivalently, if $\begin{array} { r } { \left. \Sigma \right. _ { o p } \left( \sqrt { \frac { r _ { \mathrm { e f f } } } { m } } + \frac { r _ { \mathrm { e f f } } } { m } \right) \ll c , } \end{array}$ , then $\delta ( H ) \approx c = \Theta ( 1 )$ . The same idea can be extended to generalised linear models of the form

$$
H = \frac { 1 } { m } A ^ { \top } D ^ { * } A + \lambda I .
$$

Fast Estimation of Alignment Vector in Sparse Functions. Here, we show that in structured scenarios, a suficiently good "once for all" estimation of the gradient could be computed relatively cheaply. In some sparse setups where the sparsity s grows as a factor of n. i.e. $s = n ^ { \iota }$ for some small $\iota > 0$ . As we will see later in Section 3.2, we have the alignment between the estimation and the vector to be

$$
\gamma _ { 0 } = \left( 1 - C e ^ { - c k ^ { \prime } } \right) ( 1 - \rho ^ { 2 } ) ,
$$

where $k ^ { \prime }$ is the number of rows of the sensing matrix of the order $k ^ { \prime } = O ( s \log ( n / s ) ) , \rho$ is the error incurred by the reconstruction algorithm and $C , c > 0$ are universal constants. Suppose we want an alignment $\gamma _ { 0 } = ( 1 - C e ^ { - c k ^ { \prime } } ) ( 1 - \Theta ( d / n ) )$ ). Equivalently, $\rho = \sqrt { \Theta ( d / n ) }$ , then the number of iterations required is

$$
T = \log \left( 1 / \rho \right) = \frac { 1 } { 2 } \log \left( \frac { 1 } { \Theta ( d / n ) } \right) = \Theta ( \log ( n / d ) ) .
$$

Coupled with the cost of each iteration in the recovery algorithm, we have that the cost to estimate such a vector is $O ( T \times k ^ { \prime } n + k ^ { \prime } \xi ) = O ( s \log ( n / s ) \times ( n \log ( n / d ) + \xi ) )$

## 3 Applications to Machine Learning

## 3.1 Structured Optimisation and Fast Generation of Guidance Vectors

In machine learning, the optimisation problems arising naturally out of inferential problems (or otherwise) typically have structural properties that make them amenable to algorithmic solutions, despite their typically high ambient dimensionality and other superficial complexities. A canonical example of this is accorded by the notion of sparsity, whereby a function depends only on a (relatively small) subset of the ambient coordinates, and its generalisation to the so-called manifold hypothesis, which posits that real world data may be envisaged to come from some low-dimensional manifold (whose specifics would in general be unknown to the practitioner). The natural goal, which has been achieved to a significant degree of success in the machine learning and statistics literature, is to leverage such structural properties to the efect that the complexity of algorithms scale with the intrinsic dimensionality (as opposed to the ambient dimensionality, which is typically much larger).

In this work, we bring this philosophy to bear on SSD approaches to optimisation, focussing on two foundational structures nearly ubiquitous in optimisation for ML – sparsity and a minibatch structure. We demonstrate that these structures can be efectively leveraged for fast and inexpensive generation of the guidance vector, which is a key component of our persistence of memory based approach to sparse SSD. This makes our method computationally attractive, with the additional advantage that our guidance vector generation mechanisms are also very strongly parallelisable, which is an additional benefit with regard to modern GPU-based or distributed computing architectures.

En route, we also establish to our knowledge the first theoretical analysis of classical SSD methods in the setting of sparse functions, which could be of independent interest.

## 3.2 Sparse Functions

## 3.2.1 Structure of the Problem and Motivations

A common structure exploited in machine learning objectives is when the function itself depends on only a small number of directions in its input space, even though it is nominally defined on a high-dimensional domain.

We say $f : \mathbb { R } ^ { n } \to \mathbb { R }$ is sparse (or has low-dimensional structure, which can also be called "functions with low dimensionality" [51]) if there exists an orthogonal projection matrix Π on a $\mathbb { R } ^ { s }$ subspace $( s \leq n )$ such that for all $x \in \mathbb { R } ^ { n }$

$$
f ( x ) = f ( \Pi x ) .
$$

Let $R \in \mathbb { R } ^ { s \times n }$ be a matrix formed from an orthogonal basis of $\Pi \mathbb { R } ^ { n }$ . We define $g : \mathbb { R } ^ { s } \mapsto \mathbb { R }$

$$
\begin{array} { r } { g ( y ) = f ( R ^ { \top } y ) . } \end{array}
$$

Since $f ( x ) = f ( \Pi x )$ , we have that

$$
g ( R x ) = f ( x ) .\tag{22}
$$

That is, f only varies along the s-dimensional subspace spanned by the columns of $R ,$ and is constant along all directions orthogonal to it. This structure is often called a multi-index model in the statistics literature, with the columns of R referred to as the indices or relevant directions. Some examples include:

1. Single-Index Models $( s ~ = ~ 1 ) \colon \ : f ( x ) \ : = \ : g ( r ^ { \top } x )$ for a single direction $r ~ \in ~ \mathbb { R } ^ { n }$ , such as generalised linear models $\begin{array} { r } { f ( x ) = \sigma ( r ^ { \top } x ) } \end{array}$ for a link function $\sigma ,$ or the activation of a single neuron; see e.g. [22] for classical estimation theory in this setting.

2. Finite-Index Models $( s > 1 { \mathrm { ~ f i x e d } } )$ : These are extensions of the single-index model, closely related to projection pursuit [13], where for fixed $s > 1$ , we have index vectors $\{ w _ { 1 } , \cdot \cdot \cdot , w _ { s } \} \subset$ $\mathbb { R } ^ { n }$ and $f$ is a function on the indices $w _ { j } ^ { \top } x .$ i.e. $f ( \boldsymbol { x } ) = g ( w _ { 1 } ^ { \top } \boldsymbol { x } , \cdot \cdot \cdot , w _ { s } ^ { \top } \boldsymbol { x } )$ . Neural networks with one hidden layer are an example of a finite-index model: given weights $v _ { j } \in \mathbb { R } ^ { n } , a _ { j } \in \mathbb { R }$ and biases $b _ { j } \in \mathbb { R }$ , a one-hidden-layer neural network can be expressed as

$$
\boldsymbol { y } = \sum _ { j = 1 } ^ { s } a _ { j } \sigma ( \boldsymbol { v } _ { j } ^ { \top } \boldsymbol { x } + b _ { j } ) ,
$$

where σ is the activation function. Then, the indices are simply $z _ { j } = v _ { j } ^ { \top } x + b _ { j }$ and the function is given by $\begin{array} { r } { g ( z _ { 1 } , \cdot \cdot \cdot , z _ { s } ) = \sum _ { j = 1 } ^ { s } a _ { j } \sigma ( z _ { j } ) } \end{array}$

3. Axis-Aligned Sparsity: When R is restricted to a subset $S ~ \subset ~ \{ 1 , \ldots , n \}$ of $| S | = s$ coordinate directions (i.e. columns of the identity), this recovers the more familiar notion of coordinate sparsity, where f depends on only s of its n input coordinates, as exploited by sparse regression methods such as the lasso [46].

4. High-Dimensional Regression and Variable Selection. In genomics and biomedical statistics, one often wishes to predict a phenotype or clinical outcome from a feature vector $x \in \mathbb { R } ^ { n }$ where n (e.g. the number of measured genes, SNPs, or biomarkers) vastly exceeds the number of available samples. Domain knowledge typically suggests that only a small number s of features, or linear combinations thereof, are causally relevant, motivating models of exactly the form (22) – both for statistical identifiability with limited samples and for interpretability of the resulting model [15].

Beyond these settings where sparsity is an explicit modelling assumption, such functions are frequently encountered in many applications. For instance, the loss functions of neural networks often have low rank Hessians [20, 38, 43]. This phenomenon is also prevalent in other areas such as hyper-parameter optimization for neural networks [5], heuristic algorithms for combinatoria optimization problems [24], complex engineering and physical simulation problems as in climate modeling [26], and policy search [17].

The sparse structure induces a restriction on the gradient of f. By the chain rule,

$$
\nabla f ( x ) = R ^ { \top } \nabla g ( R x ) ,
$$

so $\nabla f ( x )$ always lies in the m-dimensional row space of $R ,$ regardless of x. This is the key structural fact: although $f$ is defined on $\mathbb { R } ^ { n }$ , its entire first-order behavior is confined to a m-dimensional subspace. In particular, if one knew R in advance, optimising f would reduce to optimising the s-dimensional function $g \mathrm { ~ - ~ } \mathrm { a }$ dramatic reduction in complexity when $s \ll n$

In practice, of course, R is unknown and must be estimated alongside g, and much of the algorithmic interest in this setting lies precisely in how to identify the relevant subspace eficiently; for instance using zeroth- or first-order queries whose number scales with s rather than n.

## 3.2.2 Fast Guidance Vectors in Sparse Settings

Here, we look at how to obtain a guidance vector much cheaper than computing the full gradient when the sparsity of the problem is axis-aligned (i.e. the function only depends on a number of fixed coordinates, and R is made up of coordinate vectors). The key to obtaining such a vector hinges on a result from compressed sensing, which allows us to measure a sparse vector up to an error relative to the vector norm, with cost that scales in the order of the $O ( s \log ( n / s ) ( n + \xi ) )$ , in comparison to the $O ( n \xi )$ cost for computing n−directional derivatives. (When $\xi = { \cal O } ( n )$ , we have that the former scales much better in terms of n than the latter.)

Given a sparse signal (vector) $y ^ { * }$ , compressed sensing aims to recover this vector from linear measurements $x = \Psi y ^ { \ast } + e$ , where the matrices Ψ satisfy the restricted isometry property (RIP). A matrix $\Psi \in \mathbb { R } ^ { k ^ { \prime } \times n }$ is said to satisfy the RIP with constant $\delta _ { s }$ if for every s−sparse vector $z \in \mathbb { R } ^ { n }$

$$
\begin{array} { r } { ( 1 - \delta _ { s } ) \| z \| ^ { 2 } \leq \| \Psi z \| ^ { 2 } \leq ( 1 + \delta _ { s } ) \| z \| ^ { 2 } . } \end{array}\tag{23}
$$

Some examples include the Gaussian, subsampled Hadamard and partial Fourier matrices. For example, if Ψ is a Gaussian matrix, with entries $\Psi _ { i j } \sim N ( 0 , 1 / k ^ { \prime } )$ . For $k ^ { \prime } \gtrsim s \log ( n / s ) \eta ^ { - 2 }$ , we have $\mathbb { P } \left[ \delta _ { s } \le \eta \right] \ge 1 - C \exp ( - c k ^ { \prime } )$ , where $C , c$ are universal constants [2].

Many compressed sensing recovery algorithms (CoSaMP, IHT, $\mathrm { H T P , \ldots ) }$ produce $y _ { T }$ with

$$
\| y _ { T } - y ^ { * } \| \leq \rho \| y ^ { * } \| ,\tag{24}
$$

after T iterations at a cost of $O ( \nu ( \rho ) )$ , when $e = 0 , y ^ { * }$ is exactly s−sparse, provided that Ψ satisfies a RIP condition of the relevant order with probability $\geq p$ . The small error means that these 2

vectors have lower bounded correlation. To see this, we observe that $y _ { T }$ lies in a ball of radius $\rho \| y ^ { * } \|$ around the vector $y ^ { * }$ . The largest possible angle between $y _ { T }$ and $y ^ { * }$ occurs when $y _ { T }$ is tangent to the ball around $y ^ { * }$ . Denoting the angle between these 2 vectors as $\theta ,$ we find that

$$
\sin \theta \leq { \frac { \rho \| y ^ { * } \| } { \| y ^ { * } \| } } = \rho .
$$

Thus,

$$
\frac { \left. y _ { T } , y ^ { * } \right. ^ { 2 } } { \left\| y _ { T } \right\| ^ { 2 } \left\| y ^ { * } \right\| ^ { 2 } } = \cos ^ { 2 } \theta \geq 1 - \rho ^ { 2 } ,
$$

and correspondingly

$$
\begin{array} { r } { \langle \hat { y _ { T } } , y ^ { * } \rangle ^ { 2 } \geq ( 1 - \rho ^ { 2 } ) \Vert y ^ { * } \Vert ^ { 2 } . } \end{array}\tag{25}
$$

Applying this to our algorithm, we let $y ^ { * } = \nabla f ( x _ { 0 } ) , x = \Psi \nabla f ( x _ { 0 } )$ and $y _ { 0 } = 0$ . Then, with $v _ { 0 } = y _ { T }$ we will obtain

$$
\langle \hat { v } _ { 0 } , \nabla f ( x _ { 0 } ) \rangle ^ { 2 } \geq ( 1 - \rho ^ { 2 } ) \| \nabla f ( x _ { 0 } ) \| ^ { 2 } .
$$

Of course, this only occurs conditioned on the event $\mathcal { A }$ that the algorithm satisfies Equation (24) (which is normally only dependent on the choice of Ψ used and independent of $x _ { 0 } )$ . In the complement event, we can use 0 as a lower bound to obtain,

$$
\begin{array} { r } { \mathbb { E } \left[ \langle \hat { v } _ { 0 } , \nabla f ( x _ { 0 } ) \rangle ^ { 2 } \right] \geq \mathbb { E } \left[ \langle \hat { v } _ { 0 } , \nabla f ( x _ { 0 } ) \rangle ^ { 2 } 1 _ { A } \right] \geq ( 1 - \rho ^ { 2 } ) \mathbb { E } \left[ \Vert \nabla f ( x _ { 0 } ) \Vert ^ { 2 } 1 _ { A } \right] = \gamma _ { 0 } \mathbb { E } \left[ \Vert \nabla f ( x _ { 0 } ) \Vert ^ { 2 } \right] , } \end{array}\tag{26}
$$

where $\gamma _ { 0 } = ( 1 - \rho ^ { 2 } ) \times p$ , with $\mathbb { P } [ \mathcal { A } ] \geq p$

Iterative Hard Thresholding (IHT). The algorithm is as follows: starting with $y _ { 0 } = 0$ and $x = \Psi y ^ { * }$ , the iterates $y _ { t }$ are updated by

$$
y _ { t + 1 } = H _ { s } \left[ y _ { t } + \mu \Psi ^ { \top } ( x - \Psi y _ { t } ) \right] ,\tag{27}
$$

where $\begin{array} { r } { \mu = \frac { 1 } { 1 + \delta _ { s } } } \end{array}$ and $H _ { s }$ is the thresholding function that sets all but the top s elements to 0. By Corollary 1 of [7], if Ψ has $\mathrm { R I P }$ with $\delta _ { 3 s } < 1 / 1 5$ , then $y _ { t }$ satisfies

$$
\| y _ { t } - y ^ { * } \| \leq 2 ^ { - t } \| y ^ { * } \| .\tag{28}
$$

This means that $\begin{array} { r } { T = \left\lceil \frac { \log ( 1 / \rho ) } { \log ( 2 ) } \right\rceil } \end{array}$ to get Equation (24). The follow proposition follows for the cost of a single refresh:

Proposition 2. Let f be a s−sparse function and let ξ be the cost of computing a directional derivative. For an initial alignment $\gamma _ { 0 } ~ \in ~ ( 0 , 1 )$ , the IHT algorithm with Gaussian matrix of appropriate variance requires a computational cost of

$$
O \left( \log \left( \frac { 1 - C e ^ { - c k ^ { \prime } } } { 1 - C e ^ { - c k ^ { \prime } } - \gamma _ { 0 } } \right) k ^ { \prime } n + k ^ { \prime } \xi \right)\tag{29}
$$

where $k ^ { \prime } \gtrsim s \log ( n / s )$ and $C , c > 0$ are constants independent of $n , s$

Remark 6 (Structured Random Matrices). While Gaussian Ψ is convenient for analysis, it requires $O ( k ^ { \prime } n )$ storage and $O ( k ^ { \prime } n )$ time per matrix-vector product. Structured alternatives allow both to be reduced substantially, at the cost of a worse (but still logarithmic) dependence on n in the sample complexity $k ^ { \prime }$

A subsampled Fourier (or Hadamard) matrix formed by selecting $k ^ { \prime }$ rows uniformly at random from the $n \times n$ discrete Fourier (Walsh-Hadamard) transform and rescaled by $1 / \sqrt { k ^ { \prime } }$ , satisfies RIP of order s with constant η with high probability provided

$$
k ^ { \prime } \gtrsim s \log ^ { 2 } ( s ) \log ( n ) \eta ^ { - 2 } ,\tag{30}
$$

the current best known bound, due to [23] (improving on the earlier s $\log ^ { 4 } ( n )$ -type bounds of [42]) . This is known to be close to optimal; [6] showed that $k ^ { \prime } = \Omega ( s \log s \log ( n / s ) )$ rows are necessary for subsampled Hadamard matrices to satisfy RIP at all. Crucially, both Ψz and $\Psi ^ { \top }$ w can be computed via the FFT in $O ( n \log n )$ time which is independent of $k ^ { \prime } ,$ rather than the $O ( k ^ { \prime } n )$ required for a dense Gaussian $\Psi$ . This improves the per-iteration IHT cost in Proposition 2 from $O ( k ^ { \prime } n )$ to O(n log n) whenever $k ^ { \prime } = \omega ( \log n )$ (the typical regime, since $k ^ { \prime } \gtrsim s \log ( n / s ) )$ .

## 3.2.3 Analysis of Classical SSD in Sparse Settings

Here we analyse the classical SSD method (i.e., [28]) under the assumption that the function f is sparse in some unknown basis (i.e. the function has a low intrinsic dimension and not necessarily limited to the axis-aligned sparsity discussed above). More precisely, we do not have any constraints on R other than it is an s−dimensional orthogonal matrix in $\mathbb { R } ^ { n }$ . Leveraging Equations $^ { 2 2 , }$ we can actually prove that if $f$ has a low intrinsic dimension $s ,$ then the original SSD method only sufers from a $\frac { s } { d }$ multiplicative factor, with respect to the number of iterations, instead of the $\frac { n } { d }$ multiplicative factor.

More precisely we have the following theorem:

Theorem 6. Assume that f has an intrinsic dimension of $s < n$ . Let $\Pi , R , g$ be defined as above. Let $d , \alpha$ satisfy

$$
\operatorname* { m a x } \biggl \{ 1 , 2 \log \biggl ( \frac { 2 n ^ { 2 } } { 9 s } \biggr ) \biggr \} \leq d \leq \frac { s } { 1 6 } , \qquad \alpha = \frac { n } { 1 8 s L } .
$$

Then to obtain min $\begin{array} { r } { 1 \le k \le N \mathbb { E } \bigl [ \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \bigr ] \le \epsilon ^ { 2 } } \end{array}$ , we require

$$
N \geq \frac { 3 6 L s } { d \epsilon ^ { 2 } } \big ( f ( x _ { 0 } ) - f ^ { * } \big ) ,\tag{31}
$$

where $f ^ { * }$ is the optimal solution.

## 3.3 Finite Sum Structure with Minibatch Gradient Descent

Many machine learning tasks are formulated within the framework of empirical risk minimisation (ERM) [48, 49]. Suppose data $( x , y )$ are drawn from some distribution $\mathcal { D }$ and the aim is to find parameters θ minimising the population risk

$$
R ( \theta ) = \mathbb { E } _ { ( x , y ) \sim \mathcal { D } } \left[ \ell ( \theta ; x , y ) \right] ,
$$

where $\ell ( \theta ; x _ { i } , y _ { i } )$ is the loss incurred by parameters $\theta$ on the single example $( x , y ) \mathrm { - \ e . g }$ . squared loss $\begin{array} { r } { \ell = \frac { 1 } { 2 } ( f _ { \theta } ( x ) - y ) ^ { 2 } } \end{array}$ for regression, or cross-entropy loss for classification. Since $\mathcal { D }$ is unknown, $R ( \theta )$

cannot be computed or optimised directly. Instead, in practice, given a dataset of M i.i.d. samples $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { M } \sim \mathcal { D }$ , the standard approach is to minimise the empirical risk

$$
f ( \theta ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \ell ( \theta ; x _ { i } , y _ { i } ) ,
$$

which serves as an unbiased estimator for the loss $R ( \theta )$ . This substitution of an intractable expectation by a finite sum over observed data is precisely what gives f its finite-sum structure, and it is this structure that the mini-batch exploits [8].

Because f is the sum of per-example losses, the gradient is also an average:

$$
\nabla f ( \theta ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \nabla _ { \theta } \ell ( \theta ; x _ { i } , y _ { i } ) .
$$

This linearity is what enables mini-batching possible: the gradient of an average is the average of the gradients. Full-batch gradient descent requires computing gradients over the entire dataset at every iteration, making each update prohibitively expensive when the dataset contains millions of training examples, as is common in modern machine learning applications [8,18]. This computational burden is further compounded by the high dimensionality of contemporary neural networks, which may contain billions of trainable parameters in large language models [11, 30]. Consequently, practical learning algorithms typically employ stochastic or mini-batch gradients, since gradients computed from small subsets of the data often provide suficiently informative descent directions while substantially reducing the computational cost per iteration [8].

Concretely, instead of using all M points, mini-batch gradient descent uniformly samples m points from [M] (we will consider sampling with replacement, but similar results hold for sampling without replacement), and computes the gradient estimator

$$
g _ { B } ( \theta ) = \frac { 1 } { m } \sum _ { i \in B } \nabla _ { \theta } \ell ( \theta ; x _ { i } , y _ { i } ) .
$$

The key property of this gradient estimator is that it is unbiased:

$$
\mathbb { E } _ { B } \left[ g _ { B } ( \boldsymbol { \theta } ) \right] = \mathbb { E } _ { B } \left[ \frac { 1 } { m } \sum _ { i \in B } \nabla _ { \boldsymbol { \theta } } \boldsymbol { \ell } ( \boldsymbol { \theta } ; x _ { i } , y _ { i } ) \right] = \nabla f ( \boldsymbol { \theta } ) ,
$$

which is what allows the method to still converge (in expectation) despite each step being "noisy" - the noise averages out over iterations. In this section, we show that our algorithm can also be applied in the mini-batch setting.

## 3.3.1 Structure of The Problem and Motivations

We will consider functions of the form

$$
f ( x ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } f _ { i } ( x ) ,\tag{32}
$$

where the size M may potentially be larger than the dimension of the problem n. Here, we will have an additional assumption on the variance of the mini-batch gradient, which is a common assumption when dealing with mini-batch gradient descent [8].

Assumption 2. (Variance-bound on mini-batch gradient): For a batch $B _ { m }$ iid sampled from [M] with $| B _ { m } | = m$ , we have E $\begin{array} { r } { \left[ \left\| \frac { 1 } { m } \sum _ { i \in B _ { m } } \nabla f _ { i } ( x ) - \nabla f ( x ) \right\| ^ { 2 } \right] \leq \sigma _ { m } ^ { 2 } } \end{array}$

If we assume the that this holds for any $m = 1$ with $\sigma _ { 1 } ^ { 2 }$ , then through simple algebra, we can set $\sigma _ { m } ^ { 2 } = \sigma _ { 1 } ^ { 2 } / m$ . This means that the variance has an order of $m ^ { - 1 }$ , where m is the batch size.

Refresh Schedule. In the convergence of the standard SGD, the existence of the variance term requires the use of a vanishing step-size to control the impact of this factor. Similarly, our algorithm will employ a decaying step size to control the variance, which at the same time allows us to use an increasing number of steps before refresh. To this end, we consider the refresh schedule $r _ { i + 1 } - r _ { i } = i ^ { \beta }$ where $r _ { i }$ is the iteration where the the alignment vector is updated for the $i ^ { t h }$ time, and $\beta > 0$ is a parameter that we can choose depending on the problem. There can be other refresh schedule which could be considered, but our analysis will mainly cover this example and the eficacy of other schedules could be verified in a similar manner. More details on the analysis of the refresh schedule is in Section 3.3.2.

Mini-batch Variant Pseudocode. Here, we present the mini-batch variant of the proposed algorithm.

```latex
Algorithm 2 Subspace SGD with Persistence of Memory (SSGDPM)
1: Inputs: $d , \delta , m , \beta$ ▷ subspace rank, alignment threshold, batch size, refresh parameter
2: Initialize: x<sub>0</sub> ▷ arbitrary initialisation
3: $i \gets 0$
4: $r _ { i } \gets 0$
5: for $\mathrm { k } = 1 , 2 , \ldots$ do
6: if $i = 0$ or $k - r _ { i } > i ^ { \beta }$ then
7: Generate a new $v _ { k } .$ −1
8: $i \gets i + 1$
9: $r _ { i } \gets k$
10: else
11: $v _ { k - 1 }  v _ { k - 2 }$
12: end if
13: Generate $\tilde { P } _ { k - 1 }$ orthogonal to $v _ { k - 1 }$
14: $\hat { v } _ { k - 1 }  v _ { k - 1 } / \| v _ { k - 1 } \|$
15: $P _ { k - 1 }  ( \hat { v } _ { k - 1 } \quad \tilde { P } _ { k - 1 } )$
16: Sample $B _ { k - 1 }$ iid from [M] with $| B _ { k - 1 } | = m$
17: $\begin{array} { r } { x _ { k } \gets x _ { k - 1 } - \alpha _ { k - 1 } \frac { 1 } { m } \sum _ { i \in B _ { k - 1 } } P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f _ { i } ( x _ { k - 1 } ) } \end{array}$
18: end for
```

## 3.3.2 Analysis for Convergence Iteration

Here, we show that under bounded variance assumption, the convergence rate of the algorithm can be arbitrarily close to the standard SGD rate. Before that, we will take a look at 2 interesting properties of under this setup. Firstly, for a desired initial alignment $\gamma _ { 0 }$ , suppose we consider the

same oracle for the alignment vector, a batch size $m ^ { \prime }$ satisfying

$$
m ^ { \prime } \geq \frac { \gamma _ { 0 } \sigma _ { 1 } ^ { 2 } } { ( 1 - \gamma _ { 0 } ) \| \nabla f ( x ) \| ^ { 2 } }\tag{33}
$$

is required. Secondly, the alignment between consecutive gradients will incur an additional variance error, which is described below.

Proposition 3. Assuming that $\begin{array} { r } { f ( x ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } f _ { i } ( x ) } \end{array}$ where each $f _ { i }$ is L-smooth, then Algorithm 2 with $\beta \in ( 0 , 1 )$ has that E $\left[ \langle v _ { k } , \nabla f ( x _ { k } ) \rangle ^ { 2 } \right] \ \geq \ \delta \mathbb { E } \left[ \| \nabla f ( x _ { k } ) \| ^ { 2 } \right] - \eta _ { k } \sigma _ { m } ^ { 2 }$ for all $k \geq 1$ under the step-size schedule of $\begin{array} { r } { \alpha _ { k } = \frac { 1 } { L \sqrt { k + 1 } } } \end{array}$ and $\eta _ { k } = \sqrt { k } - \sqrt { r _ { i } }$ , where $r _ { i }$ is the iteration of the $i ^ { t h }$ update to the alignment vector.

With these in place, we have the convergence rate of the mini-batch variant of the proposed algorithm under minimal assumptions.

Theorem 7. Suppose the objective function is (32) and Assumption (2) holds. Suppose $r _ { i + 1 } - r _ { i } = i ^ { \beta }$ for some $\beta \in ( 0 , 1 )$ , and let $k _ { 0 } > 0$ satisfy $k _ { 0 } = r _ { i ^ { * } }$ with $\begin{array} { r } { i ^ { * } \ge \left( \frac { 5 } { \log \left( \gamma _ { 0 } / 2 \delta \right) } \right) ^ { 2 / ( 1 - \beta ) } } \end{array}$ . Let τ denote the index of the minimally obtained gradient, i.e. $\tau = \arg \operatorname* { m i n } _ { t \in [ N ] } \mathbb { E } [ \| \nabla f ( x _ { t } ) \| ^ { 2 } ]$ . Then to achieve an ϵ gradient, i.e. E $\left[ \lVert \nabla f ( x _ { \tau } ) \rVert ^ { 2 } \right] < \epsilon ^ { 2 }$ , we will require

$$
\epsilon ^ { - 2 } \left( 1 - \left( \frac { k _ { 0 } } { N } \right) ^ { \frac { \beta } { 1 + \beta } } \right) + \frac { \sqrt { k _ { 0 } } } { N ^ { \frac { \beta } { 1 + \beta } } } \lesssim N ^ { \frac { 1 - \beta } { 2 ( 1 + \beta ) } }\tag{34}
$$

iterations. Equivalently, we have $N = O \left( \epsilon ^ { - 4 \left( \frac { 1 + \beta } { 1 - \beta } \right) } \right)$

Remark 7. Although this analysis assumes that the refresh schedule follows a specific structure, in general, this can also be fixed like in the vanilla algorithm, or even have another formulation by itself. The analysis for this would be similar to the schedule we are assuming here, which is discussed in Appendix D.

## 4 Proof Ideas

In this section, we will briefly describe the techniques and tools required to show the results in Section 2 and 3. For a more detailed analysis, we refer the reader to the appendix.

## 4.1 General Algorithm

In the simple variant of our algorithm, our aim is to obtain an $\Theta ( 1 )$ dependence on the dimension of the projection, as compared the SSD algorithm. From the L−smoothness property of the function, we get

$$
f ( x _ { k } ) - f ( x _ { k - 1 } ) \leq - \alpha \Big \| P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) \Big \| ^ { 2 } + \frac { \alpha ^ { 2 } L } { 2 } \Big \| P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) \Big \| ^ { 2 } .\tag{35}
$$

In expectation, the projected gradient is lower bounded by $( ( 1 - d / ( n - 1 ) ) \gamma _ { k - 1 } + d / ( n - 1 ) ) \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 }$ by the construction of our algorithm. The rest of the proof would then follow by adding steps 1 to N and rearranging the terms. We note that the SSD algorithm does not have the $\gamma _ { k - 1 } > \delta$ term in

this expression, which would result in an overall dependence of $n / d$ in the final expression for the iteration complexity.

In the next part, we show that given an initial alignment of $\gamma _ { 0 }$ , the alignment vector remains weakly correlated with the next gradient. This is possible as gradient updates are local and we ask how quickly the correlation with the moving gradient decays. We first rewrite

$$
\langle \nabla f ( x _ { k } ) , \hat { v } _ { k } \rangle = \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle + \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k } ) - \nabla f ( x _ { k - 1 } ) \rangle ,
$$

with $v _ { k } = v _ { k - 1 }$ . Squaring this and using Cauchy-Schwartz Inequality on the second term yields

$$
\begin{array} { r } { \langle \hat { v } _ { k } , \nabla f ( x _ { k } ) \rangle ^ { 2 } \geq \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } - 2 \| \nabla f ( x _ { k } ) - \nabla f ( x _ { k - 1 } ) \| \vert \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle \vert . } \end{array}
$$

The first term follows by the induction hypothesis. The second term is controlled using the L−smoothness of the property of the gradient, which means we have to bound the term $\left\| P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f ( x _ { k } ) \right\|$ This is done through careful analysis of the projection $P _ { k - 1 } P _ { k - 1 } ^ { \top }$ with conditional arguments. To translate back to the correct norm (i.e. $\| \nabla f ( x _ { k } ) \|$ instead of $\| \nabla f ( x _ { k - 1 } ) \|$ , we need an upper bound for $\| \nabla f ( x _ { k } ) \|$ in terms of the norm of the previous gradient, givingus a formula for $\gamma _ { k }$ in terms of $\gamma _ { k - 1 }$ . This recurrence relation then allows us to find the maximum number of steps before the alignment drops below $\delta .$ Because gradient updates are local in terms of the step size, we are expected to have the maximum number of steps before a refresh is required to also depend on the step size, as shown in Corollary 1.

The last part of this section is dedicated to finding a tight upper bound for the sum of $\gamma _ { k }$ Although we can simply lower bound the individual alignments by δ to get rδ for r steps of re-using the same alignment vector, doing so will not give us a sharp bound in terms of the dependence on δ. We explicitly study the recurrence relation of $\gamma _ { k }$ to obtain a sharper lower bound on the sum. Instead of $N \times \delta .$ we see that it is $\begin{array} { r } { N \times \left( \frac { d } { n - 1 } + \frac { 4 \alpha L } { \log ( \gamma _ { 0 } / \delta ) } \gamma _ { 0 } \right) } \end{array}$

All bounds in the proof are derived by first conditioning on the previous iteration, to get a descent inequality for the current step, before taking full expectations.

The computational complexity can then be obtained from the iteration complexity by noting that the cost per iteration during refresh is $n d + \nu .$ , while the cost per iteration during the non-refresh steps is $n d + d \xi$

## 4.2 Mini-batch Setting

Here, we will discuss the general idea to prove convergence for the mini-batch variant of the algorithm, and refer the readers to the more detailed analysis in the respective Appendix D. The general steps to take are rather consistent with the general algorithm, as described in Section 4.1. The only diference here is that the descent direction is $P _ { k - 1 } P _ { k - 1 } \nabla g _ { k - 1 }$ instead of $P _ { k - 1 } P _ { k - 1 } \nabla f ( x _ { k - 1 } )$ This means that we require to control the term $\| P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla g _ { k - 1 } \|$ in terms of $c | | \nabla f ( x _ { k - 1 } ) | |$ and the noise.

Through direct computation, we will obtain that

$$
\begin{array} { r } { \mathbb { E } \left[ \langle \hat { v } _ { k } , \nabla f ( x _ { k } ) \rangle ^ { 2 } \right] \geq \gamma _ { k } \mathbb { E } \left[ \| \nabla f ( x _ { k } ) \| ^ { 2 } \right] - \eta _ { k } \sigma _ { m } ^ { 2 } , } \end{array}
$$

where the variance term $\sigma _ { m }$ will be added directly to the descent equation (35). Then, all that remains is to analyse the recurrence relation of $\gamma _ { k }$ like before. This time however, because of the decaying step size, $\gamma _ { k }$ decays at an increasingly slower rate as $k$ increases, meaning we are allowed

to use the same alignment vector for even more steps, motivating the use of $r _ { i + 1 } - r _ { i } = i ^ { \beta }$ for some $\beta \in ( 0 , 1 )$ . The recurrence is then solved with this assumption to find that it sufices to have

$$
i ^ { ( \beta - 1 ) / 2 } \lesssim \frac { \log ( \gamma _ { 0 } / 2 \delta ) } { 5 } ,
$$

for every refresh step $r _ { i } .$

Finally, equation (35) is summed from $k \ = \ 1$ to $N .$ , with careful analysis of the variance contributions $\eta _ { k } \sigma _ { m } ^ { 2 }$ and also the number of updates to the alignment vector needed in N steps, i.e. s such that $r _ { s } = N$ , which turns out to be $O ( N ^ { 1 / ( 1 + \beta ) } )$ ).

## 4.3 Local Regime

The proof establishes that the gradient iterates $g _ { k } = \nabla f ( x _ { k } )$ aligns progressively with the leading eigenvector $u _ { 1 }$ of the Hessian $H = \nabla ^ { 2 } f ( x ^ { * } )$ . We measure alignment through the quantity

$$
t _ { k } : = \frac { \| \mathrm { P r o j } _ { u _ { 1 } ^ { \perp } } g _ { k } \| } { \| \mathrm { P r o j } _ { u _ { 1 } } g _ { k } \| } ,
$$

which is the tangent of the angle between $g _ { k }$ and $u _ { 1 }$ . Showing $t _ { k } \to 0$ is equivalent to showing that $g _ { k }$ converges in direction to $u _ { 1 }$ . The argument proceeds in two phases.

Intuition and Key Recursion. The starting point is a gradient recursion obtained by Taylorexpanding the gradient of $f$ around $x ^ { * }$ and substituting the projected-gradient update (5). A short calculation gives

$$
g _ { k + 1 } \ = \ M g _ { k } \ + \ \alpha H E _ { k } \ + \ r _ { k } , M \ = \ I - \alpha H ,\tag{36}
$$

where $E _ { k } = ( I - P _ { k } P _ { k } ^ { \top } ) g _ { k }$ is the error due to the random projection of the algorithm and $r _ { k } =$ $r ( e _ { k + 1 } ) - r ( e _ { k } )$ collects the Taylor remainders from the Hessian Lipschitz condition. Because $\| r _ { k } \| =$ $O ( \| g _ { k } \| ^ { 2 } )$ , the remainder is second order in the gradient norm and becomes negligible as the iterates approach $x ^ { * }$ . The directional behaviour of $g _ { k }$ is therefore governed by the first two terms of (36).

Phase 1: Gradient Alignment via a Perturbed Power Iteration. Observe that in the absence of projection errors $( E _ { k } = 0 )$ and Taylor error terms, the recursion (36) reduces to $g _ { k + 1 } =$ $M g _ { k }$ . By assumption, $M = I - \alpha H$ has eigenvalues $\mu _ { i } = 1 - \alpha \lambda _ { i }$ satisfying $\mu _ { 1 } > \mu _ { 2 } > \dots > \mu _ { n } > 0$ and iterating $g _ { k + 1 } = M g _ { k }$ is precisely a power iteration on $M$ . The component of $g _ { k }$ along u<sub>1</sub> stays relatively constant, while every orthogonal component decays at a strictly smaller rate $\frac { \mu _ { i } } { \mu _ { 1 } }$ Consequently $t _ { k }$ contracts at a rate $\begin{array} { r } { \frac { \mu _ { 2 } } { \mu _ { 1 } } < 1 } \end{array}$ , and $g _ { k }$ aligns to $u _ { 1 }$ geometrically fast.

In the presence of the projection error $\alpha H E _ { k }$ , one has to bound the numerator and denominator of $t _ { k + 1 }$ separately. We utilise the alignment assumption $\langle \hat { v } _ { k } , g _ { k } \rangle ^ { 2 } \geq \delta \| g _ { k } \| ^ { 2 }$ to show that this error is of order $\sqrt { 1 - \delta }$ . Specifically, we have that

• Upper bound on the numerator. Because M and $\mathrm { P r o j } _ { u _ { 1 } ^ { \perp } }$ share the same eigenbasis, they commute and

$$
\begin{array} { r } { \| \mathrm { P r o j } _ { u _ { 1 } ^ { \perp } } M g _ { k } \| \le \mu _ { 2 } \| \mathrm { P r o j } _ { u _ { 1 } ^ { \perp } } g _ { k } \| . } \end{array}
$$

The projection-error term $\alpha \| \mathrm { P r o j } _ { u _ { 1 } ^ { \perp } } H E _ { k } \|$ contributes an additive noise of order $\sqrt { 1 - \delta } \| g _ { k } \|$ ， controlled by the alignment parameter $\delta .$ . We can bound the contribution from H by $\lambda _ { n }$ which is constant in terms of the free parameter $\delta .$

• Lower bound on the denominator. The leading contribution is $\mu _ { 1 } \| \mathrm { P r o j } _ { u _ { 1 } } g _ { k } \|$ , reduced by a perturbation of the same order $\sqrt { 1 - \delta } \| g _ { k } \|$ . The noise in the denominator is similar controlled as in the numerator case by utilising the alignment property we assumed.

The residual terms are controlled by using a pre-established exponential decay estimate for $\| g _ { k } \|$ whose own probability guarantee will show in the final bound. These two bounds can then be combined and expressing $\| g _ { k } \|$ in terms of $t _ { k }$ via $\| g _ { k } \| = \| \mathrm { P r o j } _ { u _ { 1 } } g _ { k } \| \sqrt { 1 + t _ { k } ^ { 2 } }$ yields a scalar recursion of the form

$$
t _ { k + 1 } \lesssim \frac { { \tilde { \mu } _ { 2 } } ^ { ( k ) } } { \mu _ { 1 } } t _ { k } + C ( \delta ) ,
$$

where $C ( \delta ) \to 0$ as $\delta  1$ and ${ \tilde { \mu } _ { 2 } } ^ { ( k ) }$ is a perturbation of $\mu _ { 2 }$ which also depends on $\delta .$ We then find the condition on $\delta$ such that we ensure that the leading factor is $< 1$ so that the above is a perturbed contraction. The standard fixed-point iteration argument shows that $t _ { k }$ converges to a neighbourhood of zero with radius $O \big ( C ( \delta ) / ( 1 - \operatorname* { m a x } _ { k } \tilde { \mu } _ { 2 } ^ { ( k ) } / \mu _ { 1 } ) \big )$ . By choosing $\delta$ suficiently close to 1, in other words, by requiring the adaptive direction $\hat { v } _ { k }$ to be suficiently well aligned with $g _ { k }$ one can make this neighbourhood arbitrarily small. In particular, there exists a finite iterate $k _ { 1 } < K _ { \varepsilon }$ at which $t _ { k _ { 1 } } = O ( 1 - \delta )$ , or equivalently,

$$
\begin{array} { r } { \langle u _ { 1 } , g _ { k _ { 1 } } \rangle ^ { 2 } \ge \delta _ { 0 } ^ { 2 } \| g _ { k _ { 1 } } \| ^ { 2 } , } \end{array}
$$

for a parameter $\delta _ { 0 }$ that can be made arbitrarily small by increasing δ. All concentration events required for this phase hold simultaneously with probability at least $1 - k _ { 0 } p _ { d e c a y } ( \hat { \tau } , \delta , d )$ by a union bound, which we can improve to $1 - e ^ { - \frac { 1 } { 8 } k _ { 1 } \left( 1 - p _ { d e c a y } \left( \hat { \tau } , \delta , d \right) \right) }$ by an application of Bernstein Inequality. Next, we set $g _ { k _ { 1 } } = v _ { k _ { 1 } }$ and for phase 2, have that for $k \geq k _ { 1 }$

$$
\begin{array} { r } { \langle u _ { 1 } , \hat { v } _ { k } \rangle ^ { 2 } \geq \delta _ { 0 } ^ { 2 } . } \end{array}
$$

Phase 2: Frozen Direction and Continued Alignment. Having shown that $g _ { k _ { 1 } }$ (and hence the alignment vector $v _ { k _ { 1 } } : = g _ { k _ { 1 } }$ used from this point on) is δ -aligned with $u _ { 1 }$ , we now ask whether the alignment persists once $v _ { k }$ is frozen at $v _ { k _ { 1 } }$ for $k > k _ { 1 }$ , rather than re-estimated at every few steps. This is the regime relevant to the "no-refresh" claim: the guiding direction is fixed, so any further error induced by the algorithm comes from the fresh random matrix $\tilde { P } _ { k }$ alone.

Unlike Phase 1, there is no standing alignment hypothesis to fall back on here, so the numerator and denominator bounds on the ratio $t _ { i } ^ { ( k ) } : = | u _ { i } ^ { \top } g _ { k } | / | u _ { 1 } ^ { \top } g _ { k } |$ (for each $i \geq 2 )$ must be derived directly from a Johnson-Linderstrauss-type concentration bound. With probability $1 - 2 e ^ { - c d \tau _ { i } ^ { 2 } }$ , the random projection $\tilde { P } _ { k }$ approximately preserves the inner products $\left. \tilde { P } _ { k } ^ { \top } u _ { i } , \tilde { P } _ { k } ^ { \top } g _ { k } \right.$ up to a slack $\tau _ { i }$ relative to their projections onto $v _ { k _ { 1 } } ^ { \perp }$ . More importantly, the same result holds for the re-scaled random matrix $\hat { P } _ { k }$ which we use instead in this phase. Feeding this into the same expansion in (36) yields, for every $i \geq 2$ , a scalar recursion of the same shape as in Phase 1,

$$
t _ { i } ^ { k + 1 } \lesssim \frac { \tilde { \mu } _ { i } ^ { ( k ) } } { \mu _ { 1 } } t _ { i } ^ { ( k ) } + \tilde { \epsilon } _ { i } ^ { ( k ) } ,
$$

where now the slack term $\tilde { \epsilon } _ { i } ^ { ( k ) }$ is driven by the chosen concentration parameter $\tau _ { i }$ (in place of $\sqrt { 1 - \delta }$ as in Phase 1) together with the same negligible exponentially-decaying remainder. As before, requiring $\tilde { \mu } _ { i } ^ { ( k ) } < \mu _ { 1 }$ gives a contraction condition, which translates into an explicit upper bound on $\tau _ { i } ,$ , of order $( \lambda _ { i } - \lambda _ { 1 } ) / \lambda _ { i } \cdot \sqrt { 1 - \delta _ { 0 } }$ . Eigen-gaps that are small relative to $\lambda _ { 1 }$ force tighter concentration, i.e. a higher-probability (larger d) requirement.

One-subtlety remains, because of the re-scaling of the random matrix $\tilde { P } _ { k }  \hat { P } _ { k }$ , the efective projection operator onto the random subspace $\hat { P } _ { k } \hat { P } _ { k } ^ { \top }$ has a slightly worse operator-norm, of order $( n - 1 ) / d$ instead of 1. This inflates the remainder term by the same factor, and mildly perturbs the step-size required to ensure that we still have a exponential rate of decay, but does not change the qualitative picture that the dominant contractions are still goverened by the spectral gaps $\lambda _ { i } - \lambda _ { 1 }$ One only needs $k _ { 1 } = O ( \log ( n / d ) )$ steps of "burn-in" before this extra factor becomes immaterial.

Running the same perturbed-fixed-point argument componentwise, each $t _ { i } ^ { ( k ) }$ is shown to be nonincreasing (given the desired condition on $\tau _ { i } )$ , so the aggregate quantity $\textstyle \sum _ { i \geq 2 } ( t _ { i } ^ { ( k ) } ) ^ { 2 }$ or equivalently $\| \mathrm { P r o j } _ { u _ { 1 } ^ { \perp } } g _ { k } \| / | u _ { 1 } ^ { \top } g _ { k } |$ stays controlled for the enture horizon $k \in ( k _ { 1 } , K _ { \epsilon } )$

Finally, combining the events across all $i \in [ n ]$ and all $k \in ( k _ { 1 } , K _ { \epsilon } )$ via a union bound (using a successive conditioning argument for each consecutive event, together with the Phase 1 probability for reaching the initial alignment $\delta _ { 0 } )$ , we obtain the stated overall success probability in (140). Translating the resulting bound on $\tau _ { i }$ back into a bound on $\left\| \operatorname { P r o j } _ { u _ { 1 } ^ { \perp } } g _ { k } \right\| / | u _ { 1 } ^ { \top } g _ { k } |$ gives the relationship between the target terminal correlation floor $\delta _ { 1 }$ and the required initial threshold $\delta _ { 0 }$ . A tighter final alignment guarantee requires pushing $\delta _ { 0 }$ closer to 1, i.e. a longer or more tightly concentrated Phase 1.

## 5 Numerical Experiments

In this section, we demonstrate our algorithms numerically for both the sparse and mini-batch variant. These experiments were run on an HPC cluster with 12 CPU cores and 96GB ram, with 2,000 concurrent directional derivatives computed at each iteration. For a Haar-distributed matrix with the lower dimension $d \ll n$ (such as $d = \log ( n ) )$ , it is orthogonal to any fixed vector with high probability, which allowed us to forgo the need to orthogonalise the matrix to lie in $v _ { k } ^ { \perp }$ in the experiments. Each experiment is run with 5 seeds, with the min-max error bars plotted.

Sparse Setting. To illustrate the theoretical improvements of the computational complexity under the sparse setting, we conducted numerical experiments on a Rosenbrock function. A general Rosenbrock function has the form

$$
f ( \boldsymbol { x } ) = \sum _ { i = 1 } ^ { M } a _ { i } ( x _ { i } - b _ { i } ) ^ { 2 } + c _ { i } ( x _ { i + 1 } - d _ { i } x _ { i } ^ { 2 } ) ^ { 2 } ,\tag{37}
$$

where $a _ { i } , b _ { i } , c _ { i } , d _ { i }$ are constants which define the structure of the function, and the objective is to minimise this function. To sparsify this function, we simply add a prefactor $m _ { i } \in \{ 0 , 1 \}$ in front of each summand. In this experiment, we choose $M = 5 0 , 0 0 0$ with a sparsity factor of 250. i.e. only 125 $m _ { i } ^ { \prime } s$ are 1 (which are guaranteed to be non-consecutive).

We run the standard gradient descent algorithm, Kozak’s original SSD algorithm, and the 2 variants of our proposed algorithm on this test function. The step sizes for all algorithms are taken to be the same at 0.01. For our algorithm where the alignment vector is estimated using IHT, we choose to run the algorithm with an upper bound on the sparsity, $s \leq s ^ { * }$ . Using this $s ^ { * }$ , we then select $k ^ { \prime } = s ^ { * } \log ( n )$ as an estimate on the number of rows of the random matrix to draw to compute x in Equation (27) and for the subsequent IHT steps. We test 2 values of $s ^ { * }$ satisfying $p : = k ^ { \prime } / n = 0 . 1 , 0 . 2$

For the value of d in both Kozak’s SSD algorithm and our algorithm, we opted to use $d = 1 0$ Numerically, the convergence rates did not show much diference when d varied within a small range around 10, hence we only showed it with $d = 1 0$ . All results are plotting with $f ( x )$ against time s.

Our proposed algorithm used 10GB of memory for the one with $p = 0 . 2$ and 5.4GB for $p = 0 . 1$ GD used 56.4GB and SSD used 0.6GB.

Minibatch Setting. For the minibatch setting, we consider a regularised multi-layer neural network (MLP) on the mnist dataset. We consider a function of the form

$$
f ( x ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \ell ( f _ { \theta } ( x _ { i } ) - y _ { i } ) + \lambda \| \theta \| _ { 2 } ^ { 2 } ,\tag{38}
$$

where the cross-entropy loss is used as the loss function, and the regularising coeficient is taken to be 0.005. The architecture we choose is a 3-layer MLP, where the 2 hidden layers are dimensions 64 each. The output layer is a 10 dimension vector which is then given into the loss function through the cross-entropy loss. The weights of the network are initialised with the He initialisation.

We benchmark our algorithm against the vanilla SGD and Kozak’s SSD, where we run a variant of Kozak’s SSD by projecting the minibatch gradient instead of the full gradient at each iterate. The batch sizes m in both the kozak’s algorithm and SGD were taken to be 128, while this same batch size was used to update the new alignment vector in our proposed algorithm, with the consequent iterations done with batch sizes 16. To keep the noise from the gradient estimator small, the batch $B _ { k }$ was taken to be $B _ { k } \subset B _ { r _ { i } } ^ { \prime }$ for $r _ { i } \le k < r _ { i + 1 }$ , and the step size follows $\begin{array} { r } { \alpha _ { k } = \frac { \alpha _ { 0 } } { \sqrt { r _ { i } } } } \end{array}$ . The step sizes for the minibatch variant of Kozak’s method and SGD uses $\begin{array} { r } { \alpha _ { k } = \frac { \alpha _ { 0 } } { \sqrt { k } } } \end{array}$ . Since Kozak’s algorithm has a much higher iteration complexity, the step size decays to 0 much faster than the other algorithms, hence we decided to work with $\alpha _ { 0 } ~ = ~ 1 0$ for the minibatch variant of Kozak’s algorithm while $\alpha _ { 0 } = 0 . 1$ for our proposed algorithm and SGD. All random subspaces are taken with $d = 1 0$

Our algorithm and SGD utilised 72GB of memory (because the refresh size used in our algorithm is the same as the batch size used in SGD), while the minibatch variant of Kozak’s algorithm used 4GB. Although our algorithm used much more memory as compared to Kozak’s algorithm, the number of iterations where such a capacity is needed is much less than the total iterations; it is only needed during the steps where the alignment vector is updated.

The results of both experiments can be observed in the figures below.

## 6 Acknowledgements

The authors would like to thank Alexandre d’Aspremont for illuminating discussions. AT was supported by the JSPS Grant-in-Aid for Scientific Research(B) JP23K28041. SG was supported in part by the NUS Dean’s Chair Associate Professorship E-146-00-0037-01 and the Singapore MOE grants A-8002014-00-00 and A-8003802-00-00.

![](images/c1558520e254960f45937ad78f31d293041bd7d72da1ebc688f8d51e2a0c098c.jpg)

![](images/b8e628ba79d6ccfc773da6198f7fa27260bbb3783c40265bd69e471237942701.jpg)  
Figure 1: The left figure is the minibatch setting with mnist dataset, while the right figure is the sparse setting with the Rosenbrock function.

## References

[1] J. Baik, G. Ben Arous, and S. Péché. Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices. The Annals of Probability, 33(5):1643–1697, 2005.

[2] R. Baraniuk, M. Davenport, R. DeVore, and M. Wakin. A simple proof of the restricted isometry property for random matrices. Constructive approximation, 28(3):253–263, 2008.

[3] P. L. Bartlett, P. M. Long, G. Lugosi, and A. Tsigler. Benign overfitting in linear regression. Proceedings of the National Academy of Sciences, 117(48):30063–30070, 2020.

[4] A. G. Baydin, B. A. Pearlmutter, A. A. Radul, and J. M. Siskind. Automatic diferentiation in machine learning: a survey. Journal of machine learning research, 18(153):1–43, 2018.

[5] J. Bergstra and Y. Bengio. Random search for hyper-parameter optimization. Journal of machine learning research, 13(2), 2012.

[6] J. Blasiok, P. Lopatto, K. Luh, J. Marcinek, and S. Rao. An improved lower bound for sparse reconstruction from subsampled hadamard matrices. In 2019 ieee 60th annual symposium on foundations of computer science (focs), pages 1564–1567. IEEE, 2019.

[7] T. Blumensath and M. E. Davies. Iterative hard thresholding for compressed sensing. Applied and computational harmonic analysis, 27(3):265–274, 2009.

[8] L. Bottou, F. E. Curtis, and J. Nocedal. Optimization methods for large-scale machine learning. SIAM review, 60(2):223–311, 2018.

[9] C. Cartis, J. Fowkes, and Z. Shao. Randomised subspace methods for non-convex optimization, with applications to nonlinear least-squares. arXiv preprint arXiv:2211.09873, 2022.

[10] C. Cartis and K. Scheinberg. Global convergence rate analysis of unconstrained optimization methods based on probabilistic models. Mathematical Programming, 169(2):337–375, 2018.

[11] Y. Chen, Y. Zhang, Y. Liu, K. Yuan, and Z. Wen. A memory eficient randomized subspace optimization method for training large language models. arXiv preprint arXiv:2502.07222, 2025.

[12] A. Defazio, F. Bach, and S. Lacoste-Julien. Saga: A fast incremental gradient method with support for non-strongly convex composite objectives. Advances in neural information processing systems, 27, 2014.

[13] P. Diaconis and M. Shahshahani. On nonlinear functions of linear combinations. SIAM Journal on Scientific and Statistical Computing, 5(1):175–191, 1984.

[14] K. J. Dzahini and S. M. Wild. Stochastic trust-region algorithm in random subspaces with convergence and expected complexity analyses. SIAM Journal on Optimization, 34(3):2671– 2699, 2024.

[15] J. Fan and J. Lv. A selective overview of variable selection in high dimensional feature space. Statistica Sinica, 20(1):101, 2010.

[16] L. Franceschi, M. Donini, P. Frasconi, and M. Pontil. Forward and reverse gradient-based hyperparameter optimization. In International conference on machine learning, pages 1165– 1173. PMLR, 2017.

[17] L. P. Fröhlich, E. D. Klenske, C. G. Daniel, and M. N. Zeilinger. Bayesian optimization for policy search in high-dimensional systems via automatic domain selection. In 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 757–764. IEEE, 2019.

[18] I. Goodfellow, Y. Bengio, A. Courville, and Y. Bengio. Deep learning, volume 1. MIT press Cambridge, 2016.

[19] A. Griewank and A. Walther. Evaluating derivatives: principles and techniques of algorithmic diferentiation. SIAM, 2008.

[20] G. Gur-Ari, D. A. Roberts, and E. Dyer. Gradient descent happens in a tiny subspace. arXiv preprint arXiv:1812.04754, 2018.

[21] F. Hanzely, K. Mishchenko, and P. Richtárik. Sega: Variance reduction via gradient sketching. Advances in Neural Information Processing Systems, 31, 2018.

[22] W. Härdle, P. Hall, and H. Ichimura. Optimal smoothing in single-index models. The annals of Statistics, pages 157–178, 1993.

[23] I. Haviv and O. Regev. The restricted isometry property of subsampled fourier matrices. In Geometric aspects of functional analysis: israel seminar (gafa) 2014–2016, pages 163–179. Springer, 2017.

[24] F. Hutter, H. Hoos, and K. Leyton-Brown. An eficient approach for assessing hyperparameter importance. In International conference on machine learning, pages 754–762. PMLR, 2014.

[25] R. Johnson and T. Zhang. Accelerating stochastic gradient descent using predictive variance reduction. Advances in neural information processing systems, 26, 2013.

[26] C. G. Knight, S. H. Knight, N. Massey, T. Aina, C. Christensen, D. J. Frame, J. A. Kettleborough, A. Martin, S. Pascoe, B. Sanderson, et al. Association of parameter, software, and hardware variation with large-scale behavior across 57,000 climate models. Proceedings of the National Academy of Sciences, 104(30):12259–12264, 2007.

[27] V. Koltchinskii and K. Lounici. Concentration inequalities and moment bounds for sample covariance operators. Bernoulli, pages 110–133, 2017.

[28] D. Kozak, S. Becker, A. Doostan, and L. Tenorio. A stochastic subspace approach to gradientfree optimization in high dimensions: D. kozak et al. Computational Optimization and Applications, 79(2):339–368, 2021.

[29] D. Kozak, C. Molinari, L. Rosasco, L. Tenorio, and S. Villa. Zeroth-order optimization with orthogonal random directions: D. kozak et al. Mathematical Programming, 199(1):1179–1219, 2023.

[30] K. Liang, B. Liu, L. Chen, and Q. Liu. Memory-eficient llm training with online subspace descent. Advances in Neural Information Processing Systems, 37:64412–64432, 2024.

[31] J. Lucas, G. Tucker, R. B. Grosse, and M. Norouzi. Don’t blame the elbo! a linear vae perspective on posterior collapse. Advances in neural information processing systems, 32, 2019.

[32] S. Malladi, T. Gao, E. Nichani, A. Damian, J. D. Lee, D. Chen, and S. Arora. Fine-tuning language models with just forward passes. Advances in Neural Information Processing Systems, 36:53038–53075, 2023.

[33] N. Mallinar, J. Simon, A. Abedsoltan, P. Pandit, M. Belkin, and P. Nakkiran. Benign, tempered, or catastrophic: Toward a refined taxonomy of overfitting. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 1182–1195. Curran Associates, Inc., 2022.

[34] Y. Nesterov. Eficiency of coordinate descent methods on huge-scale optimization problems. SIAM Journal on Optimization, 22(2):341–362, 2012.

[35] J. Nutini, M. Schmidt, I. Laradji, M. Friedlander, and H. Koepke. Coordinate descent converges faster with the gauss-southwell rule than random selection. In International Conference on Machine Learning, pages 1632–1641. PMLR, 2015.

[36] G. Omiya, P.-L. Poirion, and A. Takeda. Randomized subspace nesterov accelerated gradient. arXiv preprint arXiv:2605.00740, 2026.

[37] J. M. Ortega and W. C. Rheinboldt. Iterative solution of nonlinear equations in several variables. SIAM, 2000.

[38] V. Papyan. The full spectrum of deepnet hessians at scale: Dynamics with sgd training and sample size. arXiv preprint arXiv:1811.07062, 2018.

[39] S. Park, J. Jeong, Y. Kim, J. Lee, and N. Lee. Zip: An eficient zeroth-order prompt tuning for black-box vision-language models. In International Conference on Learning Representations, volume 2025, pages 62988–63021, 2025.

[40] Z. Qin, D. Chen, B. Qian, B. Ding, Y. Li, and S. Deng. Federated full-parameter tuning of billion-sized language models with communication cost under 18 kilobytes. arXiv preprint arXiv:2312.06353, 2023.

[41] Z. Qu and P. Richtárik. Coordinate descent with arbitrary sampling i: Algorithms and complexity. Optimization Methods and Software, 31(5):829–857, 2016.

[42] M. Rudelson and R. Vershynin. On sparse reconstruction from fourier and gaussian measurements. Communications on Pure and Applied Mathematics: A Journal Issued by the Courant Institute of Mathematical Sciences, 61(8):1025–1045, 2008.

[43] L. Sagun, U. Evci, V. U. Guney, Y. Dauphin, and L. Bottou. Empirical analysis of the hessian of over-parametrized neural networks. arXiv preprint arXiv:1706.04454, 2017.

[44] Y. Shu, W. Hu, S.-K. Ng, B. K. H. Low, and F. R. Yu. Ferret: Federated full-parameter tuning at scale for large language models. arXiv preprint arXiv:2409.06277, 2024.

[45] K. Shukla and Y. Shin. Randomized forward mode of automatic diferentiation for optimization algorithms. arXiv preprint arXiv:2310.14168, 2023.

[46] R. Tibshirani. Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society Series B: Statistical Methodology, 58(1):267–288, 1996.

[47] A. Tsigler and P. L. Bartlett. Benign overfitting in ridge regression. Journal of Machine Learning Research, 24(123):1–76, 2023.

[48] V. Vapnik. Principles of risk minimization for learning theory. Advances in neural information processing systems, 4, 1991.

[49] V. N. Vapnik. Statistical learning theory {Adaptive and learning systems for signal processing, communications, and control}. Wiley and Sons, 1998.

[50] R. Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science, volume 47 of Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge, 2018.

[51] Z. Wang, F. Hutter, M. Zoghi, D. Matheson, and N. De Feitas. Bayesian optimization in a billion dimensions via random embeddings. Journal of Artificial Intelligence Research, 55:361– 387, 2016.

[52] S. J. Wright. Coordinate descent algorithms. Mathematical programming, 151(1):3–34, 2015.

[53] Z. Yu, P. Zhou, S. Wang, J. Li, M. Tian, and H. Huang. Zeroth-order fine-tuning of llms in random subspaces. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4475–4485. IEEE, 2025.

[54] H. Zhan, C. Chen, T. Ding, Z. Li, and R. Sun. Unlocking black-box prompt tuning eficiency via zeroth-order optimization. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 14825–14838, 2024.

[55] J. Zhao, Z. Zhang, B. Chen, Z. Wang, A. Anandkumar, and Y. Tian. Galore: Memory-eficient llm training by gradient low-rank projection. arXiv preprint arXiv:2403.03507, 2024.

## A Mathematical Tools Needed

We study several properties of the random matrix $\tilde { P } _ { k }$ which will be of use in the later proofs.

Proposition $\textbf { 4 } ( \mathbf { \Sigma } [ 5 0 ]$ Lemma 5.3.2). Let $Q$ be a projection from $\mathbb { R } ^ { n }$ to a random d-dimensional subspace uniformly distributed in $G _ { n , d }$ . For a fixed vector $z \in \mathbb { R } ^ { n }$ , we have

$$
\mathbb { E } \left[ \left. Q z \right. ^ { 2 } \right] = \frac { d } { n } \Vert z \Vert ^ { 2 } ,\tag{39}
$$

$$
( 1 - \tau ) \frac { d } { n } \| z \| ^ { 2 } \leq \| Q z \| ^ { 2 } \leq ( 1 + \tau ) \frac { d } { n } \| z \| ^ { 2 }\tag{40}
$$

with probability at least $1 - 2 \exp \left( - c \tau ^ { 2 } d \right)$

Lemma 1. Let $u , v \in \mathbb { R } ^ { n }$ be fixed vectors and let $P \in \mathbb { R } ^ { n \times d }$ be Haar distributed on the subspace orthogonal to $u _ { 1 } , i . e . , P$ has orthonormal columns chosen uniformly from $u _ { 1 } ^ { \perp }$ . Then

$$
\mathbb { P } \Big [ \big | \langle P ^ { \top } u , P ^ { \top } v \rangle - \frac { d } { n - 1 } \langle u _ { \bot } , v _ { \bot } \rangle \big | \leq \tau \frac { d } { n - 1 } \| u _ { \bot } \| \| v _ { \bot } \| \Big ] \geq 1 - 2 \exp ( - c \tau ^ { 2 } d ) ,\tag{41}
$$

where $u _ { \bot } = ( I - u _ { 1 } u _ { 1 } ^ { \top } ) \ d t$ u and $\boldsymbol { v } _ { \bot } = ( I - u _ { 1 } \boldsymbol { u } _ { 1 } ^ { \top } ) \boldsymbol { v }$ are the components in $u _ { 1 } ^ { \perp }$ .

Proof. Let $\Pi = I - u _ { 1 } u _ { 1 } ^ { \top }$ denote the projection onto $u _ { 1 } ^ { \perp }$ . Then P is Haar distributed on Π, i.e., its columns form an orthonormal basis of a uniformly random d-dimensional subspace of Π.

Define $u \perp =$ Πu and $v _ { \perp } = \Pi v$ . Then

$$
\langle P ^ { \top } u , P ^ { \top } v \rangle = \langle P ^ { \top } u _ { \bot } , P ^ { \top } v _ { \bot } \rangle .
$$

Now, P restricted to $u _ { 1 } ^ { \perp }$ is equivalent to saying $P P ^ { \top }$ is a uniform random projection in $\mathbb { R } ^ { n - 1 }$ That is, if we identify $u _ { 1 } ^ { \perp } \simeq \mathbb { R } ^ { n - 1 }$ , then $P P ^ { \top }$ is uniform in $G _ { n - 1 , d }$ . Applying Lemma 5.3.2 in $\mathbb { R } ^ { n }$ −1 and noting that $\| P P ^ { \top } \bar { u } \| = \| P ^ { \top } u \|$ , we find that

$$
\mathbb { E } [ \| P ^ { \top } u _ { \bot } \| ^ { 2 } ] = \frac { d } { n - 1 } \| u _ { \bot } \| ^ { 2 } , \quad \mathbb { E } [ \| P ^ { \top } v _ { \bot } \| ^ { 2 } ] = \frac { d } { n - 1 } \| v _ { \bot } \| ^ { 2 } ,
$$

and the standard concentration inequality yields

$$
\| P ^ { \top } u _ { \bot } \| ^ { 2 } \approx ( 1 \pm \tau ) \frac { d } { n - 1 } \| u _ { \bot } \| ^ { 2 } \quad \mathrm { w i t h ~ p r o b a b i l i t y ~ a t ~ l e a s t ~ } 1 - 2 \exp ( - c \tau ^ { 2 } d ) ,
$$

and similarly for $v _ { \perp }$ . Next, using the polarization identity

$$
\langle P ^ { \top } u _ { \bot } , P ^ { \top } v _ { \bot } \rangle = \frac { 1 } { 4 } \Big ( \| P ^ { \top } ( u _ { \bot } + v _ { \bot } ) \| ^ { 2 } - \| P ^ { \top } ( u _ { \bot } - v _ { \bot } ) \| ^ { 2 } \Big ) ,
$$

we obtain

$$
\begin{array} { r l } & { \langle P ^ { \top } \tilde { u } _ { \perp } , P ^ { \top } \tilde { v } _ { \perp } \rangle \leq \displaystyle \frac { 1 } { 4 } \frac { d } { n - 1 } \Big ( ( 1 + \tau ) \| \tilde { u } _ { \perp } + \tilde { v } _ { \perp } \| ^ { 2 } - ( 1 - \tau ) \| \tilde { u } _ { \perp } - \tilde { v } _ { \perp } \| ^ { 2 } \Big ) } \\ & { \qquad = \displaystyle \frac { 1 } { 4 } \frac { d } { n - 1 } \Big ( \| \tilde { u } _ { \perp } + \tilde { v } _ { \perp } \| ^ { 2 } - \| \tilde { u } _ { \perp } - \tilde { v } _ { \perp } \| ^ { 2 } + \tau \big ( \| \tilde { u } _ { \perp } + \tilde { v } _ { \perp } \| ^ { 2 } + \| \tilde { u } _ { \perp } - \tilde { v } _ { \perp } ^ { 2 } \| \big ) \Big ) } \\ & { \qquad = \displaystyle \frac { d } { n - 1 } \langle \tilde { u } _ { \perp } , \tilde { v } _ { \perp } \rangle + \frac { \tau } { 2 } \frac { d } { n - 1 } ( \| \tilde { u } _ { \perp } \| ^ { 2 } + \| \tilde { v } _ { \perp } \| ^ { 2 } ) } \\ & { \qquad = \displaystyle \frac { d } { n - 1 } \langle \tilde { u } _ { \perp } , \tilde { v } _ { \perp } \rangle + \tau \frac { d } { n - 1 } , } \end{array}
$$

where $\tilde { u } = u / \lVert u \rVert$ . Consequently, this implies that

$$
\langle P ^ { \top } u _ { \bot } , P ^ { \top } v _ { \bot } \rangle \leq \frac { d } { n - 1 } \left( \langle u _ { \bot } , v _ { \bot } \rangle + \tau \| u _ { \bot } \| \| v _ { \bot } \| \right)\tag{42}
$$

in the same event. The lower bound is similarly obtained, completing the proof.

## B Proof for General Variant

In this section, we will provide the details to the proof for the results in the general algorithm, namely the iteration complexity in Theorem 1 and the time complexity in Theorem 2. The main bulk of the argument is done in a consecutive conditioning manner, so we will first define the σ−algebras

$$
\mathcal { F } _ { k } = \sigma ( P _ { 0 } , P _ { 1 } , \cdot \cdot \cdot , P _ { k - 1 } ) .\tag{43}
$$

## B.1 Proof for Iteration Complexity (Theorem 1)

To solve for Equation (10), we will employ a 3-stage approach.

1. Obtain a lower bound on the value of $\gamma _ { k }$ given the previous $\gamma _ { k - 1 }$

2. Find an upper bound on k such that $\gamma _ { k } > \delta$ is still satisfied.

3. Sum the values of $\gamma _ { k }$ from 1 to N and consequently find a lower bound on N which suficiently gives us $\mathbb { E } \left[ \lVert \nabla f ( x _ { N } ) \rVert ^ { 2 } \right] < \epsilon ^ { 2 }$

Obtain a Recurrence Relation on $\gamma _ { k }$ . To find the recurrence relationship between $\gamma _ { k }$ and $\gamma _ { k - 1 }$ , we first prove the following lemma that tells us how much the norm of the gradients changes after 1 iteration. WLOG, we will assume that the refresh step is at $x _ { 0 } .$ , which is the initial point. Since the algorithm is sequential, we can always condition on the corresponding refresh steps for the iterations after it.

Lemma 2. Suppose f is L-smooth, then in expectation, the squared norm of the gradient of f at consecutive steps can be controlled by the inequality

$$
\begin{array} { r } { \mathbb { E } \left[ \| \nabla f ( x _ { k } ) \| ^ { 2 } \right] \leq ( 1 + \alpha L ) ^ { 2 } \mathbb { E } \left[ \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \right] . } \end{array}\tag{44}
$$

Proof. By the Lipschitz continuity of the gradient, we have from Equation 3

$$
\| \nabla f ( x _ { k } ) \| - \| \nabla f ( x _ { k - 1 } ) \| \leq L \| x _ { k } - x _ { k - 1 } \| .
$$

Taking conditional expectation over $\mathcal { F } _ { k - 1 }$ and substituting the update step 5, we can get

$$
\begin{array} { r } { \mathbb { E } \left[ \Vert \nabla f ( x _ { k } ) \Vert \mid \mathcal { F } _ { k - 1 } \right] \leq \Vert \nabla f ( x _ { k - 1 } ) \Vert + \alpha L \mathbb { E } \left[ \left. P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) \right. \mid \mathcal { F } _ { k - 1 } \right] . } \end{array}\tag{45}
$$

We now require a bound on the projected gradient. Observe that by the orthogonality of the columns of the matrix $P _ { k - 1 }$ , we have

$$
\begin{array} { r l } { \| P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) \| ^ { 2 } = \nabla f ( x _ { k - 1 } ) ^ { \top } P _ { k - 1 } P _ { k - 1 } ^ { \top } P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) } & { } \\ { = \nabla f ( x _ { k - 1 } ) ^ { \top } P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) } & { } \\ { = \| P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) \| ^ { 2 } . } \end{array}
$$

Taking conditional expectation over $\mathcal { F } _ { k - 1 }$ , we have

$$
\begin{array} { r l } {  { \mathbb { E } [ \| P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \mid \mathcal { F } _ { k - 1 } ] =  \nabla f ( x _ { k - 1 } ) , \mathbb { E } [ P _ { k - 1 } P _ { k - 1 } ^ { \top } ] \nabla f ( x _ { k - 1 } )  } } \\ & { =  \nabla f ( x _ { k - 1 } ) , ( ( 1 - \frac { d } { n - 1 } ) \hat { v } _ { k - 1 } \hat { v } _ { k - 1 } ^ { \top } + \frac { d } { n - 1 } I _ { n } ) \nabla f ( x _ { k - 1 } )  } \\ & { = ( 1 - \frac { d } { n - 1 } ) \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } + \frac { d } { n - 1 } \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } . \qquad ( 4 6 ) } \end{array}
$$

By Cauchy-Schwartz and Jensen’s Inequality, we have

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left. P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) \right. \mid \mathcal { F } _ { k - 1 } \right] \leq \left( \mathbb { E } \left[ \left. P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) \right. ^ { 2 } \mid \mathcal { F } _ { k - 1 } \right] \right) ^ { 1 / 2 } } \\ & { \qquad \leq \left( \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \right) ^ { 1 / 2 } } \\ & { \qquad = \| \nabla f ( x _ { k - 1 } ) \| . } \end{array}
$$

Substituting this back into (45), we can get

$$
\begin{array} { r l } & { \mathbb { E } [ \| \nabla f ( x _ { k } ) \| ^ { 2 } \mid \mathcal { F } _ { k - 1 } ] \leq \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } + 2 \alpha L \| \nabla f ( x _ { k - 1 } ) \| \mathbb { E } \left[ \| P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) \| \mid \mathcal { F } _ { k - 1 } \right] } \\ & { \qquad + \alpha ^ { 2 } L ^ { 2 } \mathbb { E } \left[ \| P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \mid \mathcal { F } _ { k - 1 } \right] } \\ & { \qquad \leq ( 1 + \alpha L ) ^ { 2 } \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } , } \end{array}
$$

and the inequality follows by taking expectation on both sides conditional on the σ−algebra of the refresh step and using the tower property of conditional expectation. □

This gives us a control on how much the gradient norm changes in 1 update step. The change can be controlled arbitrarily small using the step size, as these changes are local. With this, we can obtain the recurrence relationship between consecutive $\gamma \mathrm { { s } }$ as shown by the next lemma.

Lemma 3. Suppose $\gamma _ { i }$ satisfies $\mathbb { E } [ \langle \nabla f ( x _ { i } ) , \hat { v } _ { i } \rangle ^ { 2 } ] \geq \gamma _ { i } \mathbb { E } [ \| \nabla f ( x _ { i } ) \| ^ { 2 } ] f o r i = k { - } 1$ , then with $v _ { k } = v _ { k - 1 }$ the same inequality is satisfied $f o r \ i = k$ with

$$
\gamma _ { k } = \frac { \left( 1 - 2 \alpha L \sqrt { \left( 1 - \frac { d } { n - 1 } \right) } \right) \gamma _ { k - 1 } - 2 \alpha L \sqrt { \frac { d } { n - 1 } } } { \left( 1 + \alpha L \right) ^ { 2 } } .\tag{47}
$$

Note that this value can be gauaranteed to be positive with control of the step size α, which will be explained in the next section.

Proof. By rewriting $\nabla f ( x _ { k } ) = \nabla f ( x _ { k - 1 } ) + \left( \nabla f ( x _ { k } ) - \nabla f ( x _ { k - 1 } ) \right)$ , we have

$$
\begin{array} { r l } & { \langle \hat { v } _ { k } , \nabla f ( x _ { k } ) \rangle ^ { 2 } = \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } + 2 \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k } ) - \nabla f ( x _ { k - 1 } ) \rangle \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle } \\ & { \qquad + \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k } ) - \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } } \\ & { \qquad \geq \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } + 2 \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k } ) - \nabla f ( x _ { k - 1 } ) \rangle \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle } \\ & { \qquad \geq \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } - 2 \| \nabla f ( x _ { k } ) - \nabla f ( x _ { k - 1 } ) \| \vert \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle , } \end{array}
$$

where the last line is by Cauchy-Schwartz Inequality. Using the assumption that the function is L-smooth, we have

$$
\langle \hat { v } _ { k } , \nabla f ( x _ { k } ) \rangle ^ { 2 } \geq \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } - 2 L \| x _ { k } - x _ { k - 1 } \| | \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle |\tag{48}
$$

and

$$
\begin{array} { r } { \mathbb { E } [ \langle \hat { v } _ { k } , \nabla f ( x _ { k } ) \rangle ^ { 2 } \mid \mathcal { F } _ { k - 1 } ] \ge \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } - 2 L \lvert \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle \rvert \times \mathbb { E } [ \Vert x _ { k } - x _ { k - 1 } \Vert \mid \mathcal { F } _ { k - 1 } ] . } \end{array}
$$

To upper-bound the last term, we first use Cauchy-Schwartz and then Equation (46) to get

$$
\begin{array} { r l } & { \mathbb { E } [ \| x _ { k } - x _ { k - 1 } \| \ | \ \mathcal { F } _ { k - 1 } ] = \alpha \mathbb { E } \left[ \| P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) \| \ | \ \mathcal { F } _ { k - 1 } \right] } \\ & { \qquad \leq \alpha \mathbb { E } \left[ \| P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \ | \ \mathcal { F } _ { k - 1 } \right] ^ { 1 / 2 } } \\ & { \qquad \leq \alpha \left( \left( 1 - \displaystyle \frac { d } { n - 1 } \right) \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } + \displaystyle \frac { d } { n - 1 } \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \right) ^ { 1 / 2 } . } \end{array}
$$

Using the inequality ${ \sqrt { a + b } } \leq { \sqrt { a } } + { \sqrt { b } }$ , we have

$$
\mathbb { E } [ \left. x _ { k } - x _ { k - 1 } \right. \mid \mathcal { F } _ { k - 1 } ] \leq \alpha \left[ \sqrt { \left( 1 - \frac { d } { n - 1 } \right) } \vert \langle \nabla \hat { v } _ { k - 1 } , f ( x _ { k - 1 } ) \rangle \vert + \sqrt { \frac { d } { n - 1 } } \| \nabla f ( x _ { k - 1 } ) \| \right] .\tag{49}
$$

Substituting this back into the equation above, we have

$$
\begin{array} { r l } & { \mathbb { E } [ \langle \hat { \nu } _ { k } , \nabla f ( x _ { k } ) \rangle ^ { 2 } \left| \mathcal { F } _ { k - 1 } \right| \geq \langle \hat { \nu } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } - 2 \alpha L [ \langle \hat { \nu } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ] \times } \\ & { \qquad \left[ \sqrt { \left( 1 - \frac { d } { n - 1 } \right) } \left| \langle \hat { \nu } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle \right| + \sqrt { \frac { d } { n - 1 } } \| \nabla f ( x _ { k - 1 } ) \| \right] } \\ & { \qquad = \left( 1 - 2 \alpha L \sqrt { \left( 1 - \frac { d } { n - 1 } \right) } \right) \langle \hat { \nu } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } } \\ & { \qquad - 2 \alpha L \left[ \sqrt { \frac { d } { n - 1 } } \| \nabla f ( x _ { k - 1 } ) \| \right] \left| \langle \hat { \nu } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle \right| } \\ & { \qquad \geq \left( 1 - 2 \alpha L \sqrt { \left( 1 - \frac { d } { n - 1 } \right) } \right) \langle \hat { \nu } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } - 2 \alpha L \sqrt { \frac { d } { n - 1 } } \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } . } \end{array}
$$

Taking expectation and using the induction hypothesis on $\langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 }$ , we have

$$
\mathbb { E } [ \langle \hat { v } _ { k } , \nabla f ( x _ { k } ) \rangle ^ { 2 } ] \geq \left( \left( 1 - 2 \alpha L \sqrt { \left( 1 - \frac { d } { n - 1 } \right) } \right) \gamma _ { k - 1 } - 2 \alpha L \sqrt { \frac { d } { n - 1 } } \right) \mathbb { E } [ \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } ] .
$$

Finally, we use Lemma (2) to translate the gradient norm on the RHS back to $\| \nabla f ( x _ { k } ) \|$ to get

$$
\gamma _ { k } = \frac { \left( 1 - 2 \alpha L \sqrt { \left( 1 - \frac { d } { n - 1 } \right) } \right) \gamma _ { k - 1 } - 2 \alpha L \sqrt { \frac { d } { n - 1 } } } { \left( 1 + \alpha L \right) ^ { 2 } } .\tag{50}
$$

Finding an Upper Bound for Number of Reuse. As a direct consequence, we can find the number of steps $r$ in which we the same alignment vector $v _ { 0 }$ still retains $\delta$ amount of correlation with the gradient. i.e., $r = \operatorname* { m a x } \{ k \in \mathbb { N } \mid \gamma _ { k } > \delta \}$ , where $\gamma _ { k } = \mathbb { E } [ \langle \hat { v } _ { k } , \nabla f ( x _ { k } ) \rangle ^ { 2 } ] / \mathbb { E } [ \| \nabla f ( x _ { k } ) \| ^ { 2 } ]$

Corollary 1. Suppose $\gamma _ { 0 }$ is the initial alignment parameter. Then, the number of steps $r$ in which we can reuse the same alignment vector v<sub>0</sub> is upper bounded by

$$
r \leq \frac { \log { \left( \frac { \gamma _ { 0 } } { \delta + \sqrt { \frac { d } { n - 1 } } } \right) } } { \log { \left( \frac { ( 1 + \alpha L ) ^ { 2 } } { 1 - 2 \alpha L } \right) } } ,\tag{51}
$$

where $\alpha < 1 / 2 L$

Proof. From Lemma (3), we have the recurrence relation

$$
\gamma _ { k } = \frac { \left( 1 - 2 \alpha L \sqrt { \left( 1 - \frac { d } { n - 1 } \right) } \right) \gamma _ { k - 1 } - 2 \alpha L \sqrt { \frac { d } { n - 1 } } } { \left( 1 + \alpha L \right) ^ { 2 } } .
$$

Solving the recurrence by induction, we obtain

$$
\begin{array} { r l } & { \gamma _ { \mathbb { R } } = \left( \frac { 1 - 2 \alpha L \sqrt { 1 - \frac { d } { n - 1 } } } { ( 1 + \alpha L ) ^ { 2 } } \right) ^ { k } \gamma _ { 0 0 } = - \frac { 2 \alpha L \sqrt { \frac { d } { n + 1 } } - 2 \alpha L \sqrt { \frac { d } { n - 1 } } \left( \frac { 1 - 2 \alpha L \sqrt { 1 - \frac { d } { n - 1 } } } { ( 1 + \alpha L ) ^ { 2 } } \right) ^ { k } } { ( 1 + \alpha L ) ^ { 2 } - \left( 1 - 2 \alpha L \sqrt { 1 - \frac { d } { n - 1 } } \right) } } \\ & { \ge \left( \frac { 1 - 2 \alpha L \sqrt { 1 - \frac { d } { n - 1 } } } { ( 1 + \alpha L ) ^ { 2 } } \right) ^ { k } \gamma _ { 0 0 } - \frac { 2 \alpha L \sqrt { \frac { d } { n - 1 } } } { ( 1 + \alpha L ) ^ { 2 } - \left( 1 - 2 \alpha L \sqrt { 1 - \frac { d } { n - 1 } } \right) } } \\ & { \ge \left( \frac { 1 - 2 \alpha L \sqrt { 1 - \frac { d } { n - 1 } } } { ( 1 + \alpha L ) ^ { 2 } } \right) ^ { k } \gamma _ { 0 0 } - \frac { 2 } { 2 + \alpha L \sqrt { \frac { d } { n - 1 } } } } \\ & { \ge \left( \frac { 1 - 2 \alpha L } { ( 1 + \alpha L ) ^ { 2 } } \right) ^ { k } \gamma _ { 0 0 } - \sqrt { \frac { d } { n - 1 } } . } \end{array}
$$

Setting this to be at least $\delta ,$ we have

$$
\left( \frac { 1 - 2 \alpha L } { ( 1 + \alpha L ) ^ { 2 } } \right) ^ { r } \geq \frac { \delta + \sqrt { \frac { d } { n - 1 } } } { \gamma _ { 0 } } .
$$

Taking logarithm on both sides, we have

$$
r \leq \frac { \log { \left( \frac { \gamma _ { 0 } } { \delta + \sqrt { \frac { d } { n - 1 } } } \right) } } { \log { \left( \frac { ( 1 + \alpha L ) ^ { 2 } } { 1 - 2 \alpha L } \right) } } .\tag{52}
$$

Remark 8. For the denominator of the form above, using log $( 1 + x ) \leq x$ for $x > - 1$ and $- \log ( 1 -$ $\begin{array} { r } { y ) \le \frac { y } { 1 - y } } \end{array}$ for $y \in [ 0 , 1 )$ , we have

$$
\log \left( \frac { ( 1 + \alpha L ) ^ { 2 } } { 1 - 2 \alpha L } \right) = 2 \log ( 1 + \alpha L ) - \log ( 1 - 2 \alpha L ) \leq 2 \alpha L + \frac { 2 \alpha L } { 1 - 2 \alpha L } = \frac { 4 \alpha L ( 1 - \alpha L ) } { 1 - 2 \alpha L } .
$$

So, $\begin{array} { r } { r \leq \frac { 1 - 2 \alpha L } { 4 \alpha L ( 1 - \alpha L ) } \log \left( \frac { \gamma _ { 0 } } { \delta + \sqrt { \frac { d } { n - 1 } } } \right) } \end{array}$ is suficient.

Summing $\gamma _ { k }$ and Combining Everything. Now, we can finally prove the main iteration complexity result in Theorem 1.

Proof. By L-smoothness of $f ,$ we have from Equation (4) that

$$
f ( x _ { k } ) - f ( x _ { k - 1 } ) \leq - \alpha \| P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) \| ^ { 2 } + \frac { \alpha ^ { 2 } L } { 2 } \| P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) \| ^ { 2 } ,
$$

where we used the update step as defined in (5). Taking conditional expectation using the σ-algebras defined previously in (43), we have

$$
\begin{array} { r l } & { \mathbb { E } [ f ( x _ { k - 1 } ) - f ( x _ { k } ) \ \big \vert \ \mathcal { F } _ { k - 1 } ] \ge \alpha \mathbb { E } \left[ \| P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \ \vert \ \mathcal { F } _ { k - 1 } \right] - \frac { \alpha ^ { 2 } L } { 2 } \mathbb { E } \left[ \| P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \ \vert \mathcal { F } _ { k - 1 } \right] } \\ & { \qquad = \alpha \left( 1 - \frac { \alpha L } { 2 } \right) \mathbb { E } \left[ \| P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \ \vert \ \mathcal { F } _ { k - 1 } \right] } \\ & { \qquad = \alpha \left( 1 - \frac { \alpha L } { 2 } \right) \left[ \left( 1 - \frac { d } { n - 1 } \right) \mathbb { E } \left[ \langle \bar { \nu } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } \ \big \vert \mathcal { F } _ { k - 1 } \right] + \frac { d } { n - 1 } \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \right] . } \end{array}
$$

Assuming we choose an $\alpha < { \frac { 1 } { 2 L } }$ , we have $\alpha \left( 1 - \textstyle { \frac { \alpha L } { 2 } } \right) ~ > ~ 0$ . Next, taking expectation and with Equation (9), we can get

$$
\mathbb { E } [ f ( x _ { k - 1 } ) - f ( x _ { k } ) ] \geq \alpha \left( 1 - { \frac { \alpha L } { 2 } } \right) \left( \left( 1 - { \frac { d } { n - 1 } } \right) \gamma _ { k - 1 } + { \frac { d } { n - 1 } } \right) \mathbb { E } \left[ \Vert \nabla f ( x _ { k - 1 } ) \Vert ^ { 2 } \right] .
$$

Taking sum over the N steps, we get

$$
\alpha \left( 1 - \frac { \alpha L } { 2 } \right) \sum _ { i = 1 } ^ { N } \left( \left( 1 - \frac { d } { n - 1 } \right) \gamma _ { i - 1 } + \frac { d } { n - 1 } \right) \mathbb { E } \left[ \Vert \nabla f ( x _ { i - 1 } ) \Vert ^ { 2 } \right] \leq \mathbb { E } [ f ( x _ { 0 } ) - f ( x _ { N - 1 } ) ] \leq f ( x _ { 0 } ) - f ^ { * } ,\tag{53}
$$

since $f ^ { * } \leq f ( x )$ . Rewriting the inequality, we have

$$
\operatorname* { m i n } _ { i } \mathbb { E } \left[ \left\| \nabla f ( x _ { i } ) \right\| ^ { 2 } \right] \leq \frac { \alpha ^ { - 1 } \left( 1 - \frac { \alpha L } { 2 } \right) ^ { - 1 } \left( f ( x _ { 0 } ) - f ^ { * } \right) } { \sum _ { i = 1 } ^ { N } \left( \left( 1 - \frac { d } { n - 1 } \right) \gamma _ { i - 1 } + \frac { d } { n - 1 } \right) } .\tag{54}
$$

Now, all that is left is to simplify the LHS term. From Lemma (3), we have

$$
\begin{array} { l } { \gamma _ { i } = \frac { \displaystyle \left( 1 - 2 \alpha L \sqrt { 1 - \frac { d } { n - 1 } } \right) \gamma _ { i - 1 } - 2 \alpha L \sqrt { \frac { d } { n - 1 } } } { \displaystyle \left( 1 + \alpha L \right) ^ { 2 } } } \\ { = a \gamma _ { i - 1 } - b } \\ { = a ^ { i } \gamma _ { 0 } - b \left( \frac { 1 - a ^ { i } } { 1 - a } \right) , } \end{array}
$$

where

$$
\begin{array} { l } { a = \displaystyle \frac { 1 - 2 \alpha L \sqrt { 1 - \frac { d } { n - 1 } } } { \left( 1 + \alpha L \right) ^ { 2 } } } \\ { b = \displaystyle \frac { 2 \alpha L \sqrt { \frac { d } { n - 1 } } } { \left( 1 + \alpha L \right) ^ { 2 } } . } \end{array}
$$

Suppose at the $k ^ { \mathrm { t h } }$ refresh, the initial alignment is $\gamma _ { 0 } ^ { ( k ) }$ and the alignment threshold is $\delta _ { k }$ . Denote the number of steps before the next refresh as $r _ { k }$ , i.e. $\begin{array} { r } { r _ { k } < \frac { 1 - 2 \alpha L } { 4 \alpha L ( 1 - \alpha L ) } \log \left( \frac { \gamma _ { 0 } ^ { ( k ) } } { \delta _ { k } + \sqrt { \frac { d } { n - 1 } } } \right) } \end{array}$ by Corollary (1). For simplicity, assume we have κ number of refreshes, i.e. $\begin{array} { r } { N = \sum _ { i = 1 } ^ { \kappa } r _ { i } } \end{array}$ . Then, the sum over $\gamma _ { i }$ can be simplified to be

$$
\begin{array} { r l } { \displaystyle \sum _ { k = 0 } ^ { N - 1 } \gamma _ { k } = \sum _ { k = 1 } ^ { N } \gamma _ { k } ^ { ( k - 1 ) } } & { } \\ & { = 0 \quad \quad \operatorname* { m a x } _ { k = 1 } ^ { N - 1 } \sum _ { i = 0 } ^ { N - 1 } \left( a ^ { i } \gamma _ { 0 } ^ { ( k ) } - b \left( \frac { 1 - a ^ { i } } { 1 - a } \right) \right) } \\ & { = \displaystyle \sum _ { k = 1 } ^ { N } \left( \gamma _ { 0 } ^ { ( k ) } \sum _ { i = 0 } ^ { N - 1 } a ^ { i } - b \sum _ { i = 1 } ^ { n - 1 } \left( \frac { 1 - a ^ { i } } { 1 - a } \right) \right) } \\ & { = \displaystyle \sum _ { k = 1 } ^ { N } \left[ \gamma _ { 0 } ^ { ( k ) } \frac { 1 - a ^ { i } } { 1 - a } \frac { b } { 1 - a } \left( \frac { 1 - a ^ { i } } { 1 - a } \right) \right) } \\ & { = \displaystyle \sum _ { k = 1 } ^ { N } \left[ \gamma _ { 0 } ^ { ( k ) } \frac { 1 - a ^ { i } \mathrm { e } ^ { k } } { 1 - a } - \frac { b } { 1 - a } \left( r _ { k } - 1 - \frac { a ( 1 - a ^ { i } \mathrm { e } ^ { k - 1 } ) } { 1 - a } \right) \right] } \\ & { = \displaystyle \frac { 1 } { 1 - a } \sum _ { k = 1 } ^ { N } \left[ \gamma _ { 0 } ^ { ( k ) } ( 1 - a ^ { i } \gamma _ { k } ) - b \left( r _ { k } - 1 - \frac { a } { 1 - a } + \frac { a ^ { i } \mathrm { e } ^ { k } } { 1 - a } \right) \right] . } \end{array}
$$

Since $b > 0$ and $\gamma _ { 0 } ^ { ( k ) } > 0 ;$ , we have $\gamma _ { 0 } ^ { ( k ) } ( 1 - a ^ { r _ { k } } ) > \gamma _ { 0 } ^ { ( k ) } ( 1 - a ) \mathrm { ~ a n d ~ } - a ^ { r _ { k } } b > - a b ,$ , giving us

$$
\begin{array} { l } { \displaystyle \sum _ { i = 0 } ^ { N - 1 } \gamma _ { i } \geq \frac { 1 } { 1 - a } \sum _ { k = 1 } ^ { \kappa } \left[ \gamma _ { 0 } ^ { ( k ) } ( 1 - a ) - b \left( r _ { k } - 1 - \frac { a } { 1 - a } + \frac { a } { 1 - a } \right) \right] } \\ { \displaystyle \quad = \sum _ { k = 1 } ^ { \kappa } \left[ \gamma _ { 0 } ^ { ( k ) } - b \left( \frac { r _ { k } - 1 } { 1 - a } \right) \right] } \\ { \displaystyle \quad = \sum _ { k = 1 } ^ { \kappa } \gamma _ { 0 } ^ { ( k ) } - \frac { b } { 1 - a } \left( \sum _ { k = 1 } ^ { \kappa } ( r _ { k } - 1 ) \right) } \\ { \displaystyle \quad = \sum _ { k = 1 } ^ { \kappa } \gamma _ { 0 } ^ { ( k ) } - \frac { b } { 1 - a } ( N - \kappa ) . } \end{array}
$$

For the 2nd term, we can simplify it to be

$$
\begin{array} { r l } & { \frac { b } { 1 - a } = \left( \frac { 2 \alpha L \sqrt { \frac { d } { n - 1 } } } { ( 1 + \alpha L ) ^ { 2 } } \right) \left( 1 - \frac { 1 - 2 \alpha L \sqrt { 1 - \frac { d } { n - 1 } } } { ( 1 + \alpha L ) ^ { 2 } } \right) ^ { - 1 } } \\ & { \qquad = \frac { 2 \alpha L } { 1 + 2 \alpha L + \alpha ^ { 2 } L ^ { 2 } - \left( 1 - 2 \alpha L \sqrt { 1 - \frac { d } { n - 1 } } \right) } \sqrt { \frac { d } { n - 1 } } } \\ & { \qquad = \frac { 1 } { \frac { \alpha L } { 2 } + \left( 1 + \sqrt { 1 - \frac { d } { n - 1 } } \right) } \sqrt { \frac { d } { n - 1 } } } \\ & { \qquad = t _ { 0 } \sqrt { \frac { d } { n - 1 } } , } \end{array}
$$

where

$$
t _ { 0 } = { \frac { 1 } { { \frac { \alpha L } { 2 } } + \left( 1 + { \sqrt { 1 - { \frac { d } { n - 1 } } } } \right) } } \in \left( { \frac { 4 } { 9 } } , { \frac { 1 } { 2 } } + O \left( { \frac { d } { n } } \right) \right) .\tag{55}
$$

Consequently, we have

$$
\sum _ { i = 0 } ^ { N - 1 } \left( \left( 1 - \frac { d } { n - 1 } \right) \gamma _ { i } + \frac { d } { n - 1 } \right) \geq \left( 1 - \frac { d } { n - 1 } \right) \sum _ { k = 1 } ^ { \kappa } \gamma _ { 0 } ^ { ( k ) } - t _ { 0 } \left( 1 - \frac { d } { n - 1 } \right) \sqrt { \frac { d } { n - 1 } } ( N - \kappa ) + N \frac { d } { n - 1 } .
$$

To find $\kappa ,$ we can sum the $r _ { i } ^ { \prime } s$ to get

$$
N = \sum _ { i = i } ^ { \kappa } r _ { i } \le \frac { 1 - 2 \alpha L } { 4 \alpha L ( 1 - \alpha L ) } \sum _ { i = 1 } ^ { \kappa } \log \left( \frac { \gamma _ { 0 } ^ { ( i ) } } { \delta _ { i } + \sqrt { \frac { d } { n - 1 } } } \right) .
$$

Equivalently, we have

$$
\frac { 4 \alpha L ( 1 - \alpha L ) } { 1 - 2 \alpha L } N \le \log \left( \prod _ { i = 1 } ^ { \kappa } \frac { \gamma _ { 0 } ^ { ( i ) } } { \delta _ { i } + \sqrt { \frac { d } { n - 1 } } } \right) = \sum _ { i = 1 } ^ { \kappa } \log \left( \frac { \gamma _ { 0 } ^ { ( i ) } } { \delta _ { i } + \sqrt { \frac { d } { n - 1 } } } \right) .
$$

Since we have $\delta _ { k } = \delta$ and $\gamma _ { 0 } ^ { ( k ) } = \gamma _ { 0 }$ , we have

$$
\kappa \ge \frac { 4 \alpha L ( 1 - \alpha L ) N } { ( 1 - 2 \alpha L ) \log ( \gamma _ { 0 } / \tilde { \delta } ) } ,
$$

where $\begin{array} { r } { \tilde { \delta } = \delta + \sqrt { \frac { d } { n - 1 } } } \end{array}$ . For α satisfying

$$
\alpha \geq \frac { 1 } { L } \cdot \frac { K } { 2 + K + \sqrt { 4 + K ^ { 2 } } } , \quad K = \log \left( \frac { \gamma _ { 0 } } { \tilde { \delta } } \right) \cdot \frac { t _ { 0 } \left( 1 - \frac { d } { n - 1 } \right) \sqrt { \frac { d } { n - 1 } } - \frac { d } { n - 1 } } { \left( 1 - \frac { d } { n - 1 } \right) \left( \gamma _ { 0 } + t _ { 0 } \sqrt { \frac { d } { n - 1 } } \right) } ,
$$

we have

$$
\frac { d } { n - 1 } + \frac { 4 \alpha L ( 1 - \alpha L ) } { ( 1 - 2 \alpha L ) \log ( \gamma _ { 0 } / \tilde { \delta } ) } \left( 1 - \frac { d } { n - 1 } \right) \gamma _ { 0 } - \tilde { t } _ { 0 } > 0 ,
$$

where $\begin{array} { r } { \tilde { t } _ { 0 } = t _ { 0 } \left( 1 - \frac { d } { n - 1 } \right) \sqrt { \frac { d } { n - 1 } } \left( 1 - \frac { 4 \alpha L \left( 1 - \alpha L \right) } { \left( 1 - 2 \alpha L \right) \log \left( \gamma _ { 0 } / \tilde { \delta } \right) } \right) } \end{array}$ . Substituting this back into the equation above, we have

$$
N \operatorname* { m i n } _ { k \in \left[ N \right] } \mathbb { E } \left[ \left\| \nabla f ( x _ { k } ) \right\| ^ { 2 } \right] \leq \frac { \alpha ^ { - 1 } \left( 1 - \frac { \alpha L } { 2 } \right) ^ { - 1 } \left( f ( x _ { 0 } ) - f ^ { * } \right) } { \frac { d } { n - 1 } + \frac { 4 \alpha L ( 1 - \alpha L ) } { ( 1 - 2 \alpha L ) \log ( \gamma _ { 0 } / \tilde { \delta } ) } \left( 1 - \frac { d } { n - 1 } \right) \gamma _ { 0 } - \tilde { t } _ { 0 } } ,\tag{56}
$$

It then sufices to have

$$
N \ge \frac { \alpha ^ { - 1 } \left( 1 - \frac { \alpha L } { 2 } \right) ^ { - 1 } \left( f ( x _ { 0 } ) - f ^ { * } \right) } { \frac { 4 \alpha L ( 1 - \alpha L ) } { ( 1 - 2 \alpha L ) \log ( \gamma _ { 0 } / \tilde { \delta } ) } \left( 1 - \frac { d } { n - 1 } \right) \gamma _ { 0 } + \frac { d } { n - 1 } - \tilde { t } _ { 0 } } \epsilon ^ { - 2 }\tag{57}
$$

to obtain min $\mathsf { l } _ { k \in [ N ] } \mathbb { E } \left[ \| \nabla f ( x _ { k } ) \| ^ { 2 } \right] < \epsilon ^ { 2 } .$

## B.2 Proof for Computational Cost (Theorem 2)

To compute the total cost of the algorithm, we let r be the number of steps before we refresh the correlation vector $v _ { k } = \nabla f ( x _ { k } )$ . For each step where $v _ { k } = v _ { k - 1 }$ , the computational cost is $O ( n d + d \xi )$ , which is the cost of computing d directional derivatives and the cost of the matrix-vector multiplication. For $v _ { k } = \nabla f ( x _ { k } )$ , the cost is $O ( n d + \nu )$ instead (the matrix-vector multiplication still costs the same).

The total computational cost can then be computed as

$$
\begin{array} { r l } & { \underbrace { N \times ( n d + d \xi ) } _ { v _ { k } = v _ { k - 1 } } + \underbrace { \frac { N } { \gamma } \times ( n d + \nu ) } _ { v _ { k } = \nabla f ( \kappa ) } } \\ & { = \frac { \alpha ^ { - 1 } \big ( 1 - \frac { \alpha L } { n - 1 } \big ) ^ { - 1 } \epsilon ^ { - 2 } } { \big ( 1 - \frac { d } { n - 1 } \big ) \gamma _ { 0 } / \gamma + \frac { \alpha ^ { - 1 } } { n - 1 } - t _ { 0 } \big ( 1 - \frac { \alpha L } { n - 1 } \big ) \sqrt { \frac { d } { n - 1 } } \big ( 1 - 1 / r \big ) } \big ( f ( x _ { 0 } ) - f ^ { * } \big ) \bigg [ ( d \xi + n d ) + \frac { 1 } { r } \big ( n d + \nu \big ) \bigg ] } \\ & { = \frac { \alpha ^ { - 1 } \big ( 1 - \frac { \alpha L } { n - 1 } \big ) ^ { - 1 } \epsilon ^ { - 2 } } { \big ( 1 - \frac { d } { n - 1 } \big ) \gamma _ { 0 } + r \frac { \alpha } { n - 1 } - t _ { 0 } \big ( 1 - \frac { d } { n - 1 } \big ) \sqrt { \frac { d } { n - 1 } } \big ( r - 1 \big ) } \big ( f ( x _ { 0 } ) - f ^ { * } \big ) \big [ r ( d \xi + n d ) + ( n d + \nu ) \big ] } \\ & { = O \big ( \epsilon ^ { - 2 } \gamma _ { 0 } [ r d \big ( n + \xi ) + \nu ] \big ) } \\ & { = O \left( \epsilon ^ { - 2 } \gamma _ { 0 } ^ { - 1 } \left[ d \log \left( \frac { \gamma _ { 0 } } { \delta } \right) \big ( \xi + n \big ) + \nu \right] \right) . } \end{array}
$$

## C Proof for Sparse Variant

This section wil first show derivations for the computational complexity of our proposed algorithm in the sparse setting, followed by the analysis of the classical SSD algorithm in the same setting.

## C.1 Proof for Refresh Cost in Sparse Case (Proposition 2)

In this section, we will analyse the computational cost of the IHT algorithm applied to our problem. Recall that the algorithm is

$$
y _ { t + 1 } = H _ { s } \left[ y _ { t } + \mu \Psi ^ { \top } ( z - \Psi y _ { t } ) \right] ,
$$

where $z = \Psi \nabla f ( x ) , \Psi \in \mathbb { R } ^ { k ^ { \prime } \times n } , y _ { 0 } \in \mathbb { R } ^ { n }$ and $H _ { s }$ is the hard thresholding function. The initial cost to compute z is $O ( k ^ { \prime } \xi )$ , while each iteration is simply a matrix vector multiplication costing $O ( k ^ { \prime } n )$ The function $H _ { s }$ at most requires $O ( n \log ( n ) )$ with sorting (which could be improved to $O ( n )$ with partitioning, but the cost is dominated by $O ( k ^ { \prime } n )$ either way as $k ^ { \prime } \gtrsim s \log ( n / s ) )$ . This means that 1 iteration of IHT has a complexity of $O ( k ^ { \prime } n + n \log ( n ) ) = O ( k ^ { \prime } \xi )$ . Consequently, $T$ iterations has a complexity of $O ( T k ^ { \prime } n )$ .

For a fixed initial alignment $\gamma _ { 0 } , \gamma _ { 0 } = ( 1 - \rho ^ { 2 } ) \times p$ as shown in Equation (26), where $\mathbb { P } [ \mathcal { A } ] \geq p$ for an appropriately defined event A. By Corollary 1 of $[ 7 ] , \mathcal { A } = \{ \delta _ { 3 s } \leq \frac { 1 } { 1 5 } \}$ , and with Gaussian matrices $\Psi \in \mathbb { R } ^ { k ^ { \prime } \times n }$ where $k ^ { \prime } \gtrsim n \log ( n / s )$ and $\Psi _ { i j } \sim N ( 0 , 1 / k ^ { \prime } )$ , we have by [2] that $\mathbb { P } [ A ] \geq 1 - C e ^ { - c k ^ { \prime } }$ where $C , c > 0$ constants. So, we have

$$
\rho ( \gamma _ { 0 } ) = \sqrt { 1 - \frac { \gamma _ { 0 } } { 1 - C e ^ { - c k ^ { \prime } } } } = \sqrt { \frac { 1 - C e ^ { - c k ^ { \prime } } - \gamma _ { 0 } } { 1 - C e ^ { - c k ^ { \prime } } } } ,
$$

coupled with $T = \lceil \log ( 1 / p ) / \log ( 2 ) \rceil$ allows us to simplify the cost to

$$
\nu ( \rho ( \gamma _ { 0 } ) ) + O ( k ^ { \prime } \xi ) = O \left( \log \left( \frac { 1 - C e ^ { - c k ^ { \prime } } } { 1 - C e ^ { - c k ^ { \prime } } - \gamma _ { 0 } } \right) k ^ { \prime } n + k ^ { \prime } \xi \right) ,
$$

Note that when $C e ^ { - c k ^ { \prime } } \ll 1$ , we have the logarithm term to be approximately $- \log ( 1 - \gamma _ { 0 } )$ , which will not be too large whenever $\gamma _ { 0 }$ is not too close to 1.

## C.2 Proof for Computational Cost in Sparse Case (Theorem 3)

To compute the computational cost of the algorithm in the case where the function is linearly sparse, we do similar calculations as in the proof for Theorem 2. Let $r$ be the number of steps before $v _ { k } = y _ { T } .$ , where $y _ { T }$ is the estimation of $\nabla f ( x _ { k } )$ obtained from $T$ iterations of IHT. For each step where $v _ { k } = v _ { k - 1 }$ , the computational cost is $O ( n d + d \xi )$ . In the update step, the computational cost is $O \left( v ( \gamma _ { 0 } ) k ^ { \prime } n + k ^ { \prime } \xi + n d + d \xi \right)$ , where $\begin{array} { r } { v ( \gamma _ { 0 } ) = \log \left( \frac { 1 - C e ^ { - c k ^ { \prime } } } { 1 - C e ^ { - c k ^ { \prime } } - \gamma _ { 0 } } \right) } \end{array}$ . The total computational cost will be given as

$$
\begin{array} { l } { \displaystyle \underbrace { \left( N - \frac { N } { r } \right) \times ( n d + d \xi ) } _ { \displaystyle v _ { k } = v _ { k - 1 } } + \underbrace { \frac { N } { \gamma } \times \left( v ( \gamma _ { 0 } ) k ^ { \prime } n + k ^ { \prime } \xi + n d + d \xi \right) } _ { \displaystyle v _ { k } = \nabla f ( x _ { k } ) } } \\ { = N \left( n d + d \xi + \frac { v ( \gamma _ { 0 } ) k ^ { \prime } n + k ^ { \prime } \xi } { r } \right) } \\ { = \displaystyle \frac { \alpha ^ { - 1 } \left( 1 - \frac { \alpha L } { 2 } \right) ^ { - 1 } \epsilon ^ { - 2 } ( f ( x _ { 0 } ) - f ^ { * } ) } { \left( 1 - \frac { d } { n - 1 } \right) \gamma _ { 0 } + r \frac { d } { n - 1 } - t _ { 0 } \left( 1 - \frac { d } { n - 1 } \right) \sqrt { \frac { d } { n - 1 } } \left( r - 1 \right) } \left( r ( n d + d \xi ) + v ( \gamma _ { 0 } ) k ^ { \prime } n + k ^ { \prime } \xi \right) . } \end{array}
$$

Suppose we assume $\gamma _ { 0 } / r \gg d / ( n - 1 )$ , then we have

$$
\begin{array} { r l } { \mathrm { C o s t } \ \approx \ \frac { \alpha ^ { - 1 } ( 1 - \alpha L / 2 ) ^ { - 1 } \epsilon ^ { - 2 } } { \gamma _ { 0 } + ( 1 - t _ { 0 } ) r d / ( n - 1 ) } ( f ( x _ { 0 } ) - f ^ { * } ) \left[ r ( n d + d \xi ) + v ( \gamma _ { 0 } ) k ^ { \prime } n + k ^ { \prime } \xi \right] } & { } \\ { \quad = O \left( \epsilon ^ { - 2 } \gamma _ { 0 } ^ { - 1 } \left[ r ( n d + d \xi ) + v ( \gamma _ { 0 } ) k ^ { \prime } n + k ^ { \prime } \xi \right] \right) } & { } \\ { \quad = O \left( \epsilon ^ { - 2 } \gamma _ { 0 } ^ { - 1 } \left[ ( r d + v ( \gamma _ { 0 } ) k ^ { \prime } ) n + ( r d + k ^ { \prime } ) \xi \right] \right) } & { } \\ { \quad = O \left( \epsilon ^ { - 2 } \gamma _ { 0 } ^ { - 1 } \left[ ( d \log ( \gamma _ { 0 } / \delta ) + v ( \gamma _ { 0 } ) k ^ { \prime } ) \times n + ( d \log ( \gamma _ { 0 } / \delta ) + k ^ { \prime } ) \times \xi \right] \right) , } \end{array}
$$

where $r = O ( \log ( \gamma _ { 0 } / \delta ) )$ as in Corollary (1). Substituting $v ( \gamma _ { 0 } )$ back in and assuming $C e ^ { - c k ^ { \prime } } \ll 1$ we get

$$
O \left( \frac { \epsilon ^ { - 2 } } { \gamma _ { 0 } } \left[ \left( d \log ( \gamma _ { 0 } / \delta ) + k ^ { \prime } \log \left( \frac { 1 } { 1 - \gamma _ { 0 } } \right) \right) \times n + \left( d \log ( \gamma _ { 0 } / \delta ) + k ^ { \prime } \right) \times \xi \right] \right) ,\tag{58}
$$

where $k ^ { \prime } \gtrsim s \log ( n / s )$

## C.3 Analysis of Classical SSD in Sparse Setting (Theorem 6)

Theorem 8. Assume that f has intrinsic dimension $s < n ,$ so that

$$
\begin{array} { r } { f ( x ) = g ( R x ) , \qquad \Pi = R ^ { \top } R , } \end{array}
$$

where $R \in \mathbb { R } ^ { s \times n }$ has orthonormal rows. Let

$$
\begin{array} { r } { x _ { k } = x _ { k } - \alpha P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) , } \end{array}
$$

where $P _ { k - 1 }$ are i.i.d. random matrices uniformly distributed on $\operatorname { S t } ( n , d )$ , and define $y _ { k } : = R x _ { k }$ Assume that $g$ is L-smooth. Suppose that

$$
\operatorname* { m a x } \biggl \{ 1 , 2 \log \biggl ( \frac { 2 n ^ { 2 } } { 9 s } \biggr ) \biggr \} \leq d \leq \frac { s } { 1 6 } , \qquad \alpha = \frac { n } { 1 8 s L } ,\tag{59}
$$

then

$$
\operatorname* { m i n } _ { 1 \leq k \leq N } \mathbb { E } \big [ \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \big ] \leq \frac { 3 6 L s } { N d } \big ( f ( x _ { 0 } ) - f ^ { * } \big ) .\tag{60}
$$

Consequently, if

$$
N \geq \frac { 3 6 L s } { d \epsilon ^ { 2 } } \big ( f ( x _ { 0 } ) - f ^ { * } \big ) ,
$$

then

$$
\operatorname* { m i n } _ { 1 \leq k \leq N } \mathbb { E } \big [ \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \big ] \leq \epsilon ^ { 2 } .
$$

Proof. Since $f ( x ) = g ( R x )$ , the chain rule gives

$$
\nabla f ( x ) = R ^ { \top } \nabla g ( R x ) .\tag{61}
$$

Hence

$$
\| \nabla f ( x _ { k } ) \| = \| \nabla g ( y _ { k } ) \| , \qquad f ( x _ { k } ) = g ( y _ { k } ) .
$$

Let

$$
u _ { k } : = \nabla g ( y _ { k } ) , \qquad B _ { k } : = R P _ { k } ( R P _ { k } ) ^ { \top } = R P _ { k } P _ { k } ^ { \top } R ^ { \top } \in \mathbb { R } ^ { s \times s } .
$$

Then

$$
y _ { k } = R x _ { k } = R x _ { k - 1 } - \alpha R P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla f ( x _ { k - 1 } ) = y _ { k - 1 } - \alpha B _ { k - 1 } u _ { k - 1 } ,
$$

where we used (61). By L-smoothness of $g _ { \mathrm { { ; } } }$

$$
g ( y _ { k } ) - g ( y _ { k - 1 } ) \leq - \alpha u _ { k - 1 } ^ { \top } B _ { k - 1 } u _ { k - 1 } + \frac { \alpha ^ { 2 } L } { 2 } \| B _ { k - 1 } u _ { k - 1 } \| ^ { 2 } .\tag{62}
$$

We now estimate the two terms on the right-hand side in conditional expectation. Let

$$
\mathcal F _ { k } : = \sigma ( P _ { 0 } , \ldots , P _ { k - 1 } ) .
$$

Since $P _ { k - 1 }$ is independent of $\mathcal { F } _ { k - 1 }$ , conditioning on $\mathcal { F } _ { k - 1 }$ allows us to regard $u _ { k - 1 }$ as fixed.

Step 1: The Linear Term. Since $P _ { k - 1 }$ is Haar-distributed on $\operatorname { S t } ( n , d )$ , rotational invariance gives

$$
\mathbb { E } \left[ P _ { k - 1 } P _ { k - 1 } ^ { \top } \right] = \frac { d } { n } I _ { n } .
$$

Therefore

$$
\mathbb { E } \left[ B _ { k - 1 } ~ | ~ \mathcal { F } _ { k - 1 } \right] = R \mathbb { E } \left[ P _ { k - 1 } P _ { k - 1 } ^ { \top } \right] ~ R ^ { \top } = \frac { d } { n } R R ^ { \top } = \frac { d } { n } I _ { s } ,
$$

and thus

$$
\mathbb { E } \Big [ u _ { k - 1 } ^ { \top } B _ { k - 1 } u _ { k - 1 } \mid \mathcal { F } _ { k - 1 } \Big ] = u _ { k - 1 } ^ { \top } \mathbb { E } \left[ B _ { k - 1 } \mid \mathcal { F } _ { k - 1 } \right] u _ { k - 1 } = \frac { d } { n } \| u _ { k - 1 } \| ^ { 2 } .\tag{63}
$$

Step 2: The Quadratic Term. Write

$$
P _ { k - 1 } = G _ { k - 1 } \left( G _ { k - 1 } ^ { \top } G _ { k - 1 } \right) ^ { - 1 / 2 } ,
$$

where $G _ { k } \in \mathbb { R } ^ { n \times d }$ is a standard Gaussian matrix. Then

$$
B _ { k - 1 } = R G _ { k - 1 } \left( G _ { k - 1 } ^ { \top } G _ { k - 1 } \right) ^ { - 1 } G _ { k - 1 } ^ { \top } R ^ { \top } .
$$

Since R has orthonormal rows, $R G _ { k - 1 } ~ \in ~ \mathbb { R } ^ { s \times d }$ is also a standard Gaussian matrix. A direct computation gives

$$
u _ { k - 1 } ^ { \top } B _ { k - 1 } u _ { k - 1 } = { \Big \| } ( G _ { k - 1 } ^ { \top } G _ { k - 1 } ) ^ { - 1 / 2 } ( R G _ { k - 1 } ) ^ { \top } u _ { k - 1 } { \Big \| } ^ { 2 } .
$$

Moreover,

$$
\begin{array} { r l } & { \| B _ { k - 1 } u _ { k - 1 } \| ^ { 2 } = \Big \| R G _ { k - 1 } \big ( G _ { k - 1 } ^ { \top } G _ { k - 1 } \big ) ^ { - 1 } \big ( R G _ { k - 1 } \big ) ^ { \top } u _ { k - 1 } \Big \| ^ { 2 } } \\ & { \qquad \leq \lambda _ { \operatorname* { m a x } } \Big ( \big ( G _ { k - 1 } ^ { \top } G _ { k - 1 } \big ) ^ { - 1 / 2 } \big ( R G _ { k - 1 } \big ) ^ { \top } \big ( R G _ { k - 1 } \big ) \big ( G _ { k - 1 } ^ { \top } G _ { k - 1 } \big ) ^ { - 1 / 2 } \Big ) \times } \\ & { \qquad \Big \| \big ( G _ { k - 1 } ^ { \top } G _ { k - 1 } \big ) ^ { - 1 / 2 } \big ( R G _ { k - 1 } \big ) ^ { \top } u _ { k - 1 } \Big \| ^ { 2 } . } \end{array}\tag{64}
$$

Now fix $t = { \sqrt { d } }$ , and define

$$
\beta : = \frac { ( \sqrt { s } + 2 \sqrt { d } ) ^ { 2 } } { ( \sqrt { n } - 2 \sqrt { d } ) ^ { 2 } } .\tag{65}
$$

Let

$$
\mathcal { E } _ { k - 1 } : = \left\{ \lambda _ { \operatorname* { m a x } } \Bigl ( ( G _ { k - 1 } ^ { \top } G _ { k - 1 } ) ^ { - 1 / 2 } ( R G _ { k - 1 } ) ^ { \top } ( R G _ { k - 1 } ) ( G _ { k - 1 } ^ { \top } G _ { k - 1 } ) ^ { - 1 / 2 } \Bigr ) \leq \beta \right\} ,
$$

and

$$
p _ { k - 1 } : = \mathbb { P } \left[ \mathcal { E } _ { k - 1 } ^ { c } \mid \mathcal { F } _ { k - 1 } \right] .
$$

On $\mathcal { E } _ { k - 1 }$ , (64) yields

$$
\begin{array} { r } { \| B _ { k - 1 } u _ { k - 1 } \| ^ { 2 } \leq \beta u _ { k - 1 } ^ { \top } B _ { k - 1 } u _ { k - 1 } . } \end{array}
$$

On $\mathcal { E } _ { k - 1 } ^ { c } ,$ we only need to use that $B _ { k - 1 }$ is a positive contraction. Indeed, for every $v \in \mathbb { R } ^ { s }$

$$
\begin{array} { r } { \boldsymbol { v } ^ { \top } \boldsymbol { B } _ { k - 1 } \boldsymbol { v } = \| \boldsymbol { P } _ { k - 1 } ^ { \top } \boldsymbol { R } ^ { \top } \boldsymbol { v } \| ^ { 2 } \leq \| \boldsymbol { R } ^ { \top } \boldsymbol { v } \| ^ { 2 } = \boldsymbol { v } ^ { \top } \boldsymbol { R } \boldsymbol { R } ^ { \top } \boldsymbol { v } = \| \boldsymbol { v } \| ^ { 2 } , } \end{array}
$$

because $P _ { k - 1 } P _ { k - 1 } ^ { \top } \preceq I _ { n }$ and $R R ^ { \top } = I _ { s }$ . Therefore

$$
0 \preceq B _ { k - 1 } \preceq I _ { s } , \qquad \mathrm { h e n c e } \qquad \| B _ { k - 1 } u _ { k - 1 } \| \leq \| u _ { k - 1 } \| .
$$

Splitting according to $\mathcal { E } _ { k }$ , we obtain

$$
\begin{array} { r l } & { \mathbb { E } \left[ \Vert B _ { k - 1 } u _ { k - 1 } \Vert ^ { 2 } \mid \mathcal { F } _ { k - 1 } \right] = \mathbb { E } \left[ \Vert B _ { k - 1 } u _ { k - 1 } \Vert ^ { 2 } \mathbf { 1 } _ { \mathcal { E } _ { k - 1 } } \mid \mathcal { F } _ { k - 1 } \right] + \mathbb { E } \left[ \Vert B _ { k - 1 } u _ { k - 1 } \Vert ^ { 2 } \mathbf { 1 } _ { \mathcal { E } _ { k - 1 } ^ { c } } \mid \mathcal { F } _ { k - 1 } \right] } \\ & { \qquad \leq \beta \mathbb { E } \left[ u _ { k - 1 } ^ { \top } B _ { k - 1 } u _ { k - 1 } \mid \mathcal { F } _ { k - 1 } \right] + p _ { k - 1 } \Vert u _ { k - 1 } \Vert ^ { 2 } } \\ & { \qquad = \left( \beta \frac { d } { n } + p _ { k - 1 } \right) \Vert u _ { k - 1 } \Vert ^ { 2 } , } \end{array}\tag{66}
$$

where we used (63).

Step 3: Bound on $p _ { k }$ . Since

$$
\lambda _ { \operatorname* { m a x } } \Bigl ( ( G _ { k - 1 } ^ { \top } G _ { k - 1 } ) ^ { - 1 / 2 } ( R G _ { k - 1 } ) ^ { \top } ( R G _ { k - 1 } ) ( G _ { k - 1 } ^ { \top } G _ { k - 1 } ) ^ { - 1 / 2 } \Bigr ) \le \frac { \lambda _ { \operatorname* { m a x } } \bigl ( ( R G _ { k - 1 } ) ^ { \top } ( R G _ { k - 1 } ) \bigr ) } { \lambda _ { \operatorname* { m i n } } ( G _ { k - 1 } ^ { \top } G _ { k - 1 } ) } ,
$$

the event $\mathcal { E } _ { k - 1 } ^ { c }$ is contained in

$$
\Big \{ \sigma _ { \operatorname* { m a x } } ( R G _ { k - 1 } ) > \sqrt { s } + 2 \sqrt { d } \Big \} \cup \Big \{ \sigma _ { \operatorname* { m i n } } ( G _ { k - 1 } ) < \sqrt { n } - 2 \sqrt { d } \Big \} .
$$

By the standard Gaussian singular value bounds,

$$
\begin{array} { r } { \mathbb { P } \Big ( \sigma _ { \operatorname* { m a x } } \big ( R G _ { k - 1 } \big ) > \sqrt { s } + 2 \sqrt { d } \Big ) \leq e ^ { - d / 2 } , } \end{array}
$$

and

$$
\begin{array} { r } { \mathbb { P } \Big ( \sigma _ { \operatorname* { m i n } } \big ( G _ { k - 1 } \big ) < \sqrt { n } - 2 \sqrt { d } \Big ) \leq e ^ { - d / 2 } , } \end{array}
$$

where the second estimate is valid because $2 { \sqrt { d } } < { \sqrt { n } }$ by (59). Therefore, by a union bound,

$$
p _ { k - 1 } = \mathbb { P } \left[ \mathcal { E } _ { k - 1 } ^ { c } \mid \mathcal { F } _ { k - 1 } \right] = \mathbb { P } \left[ \mathcal { E } _ { k - 1 } ^ { c } \right] \leq 2 e ^ { - d / 2 } .\tag{67}
$$

By the lower bound on d in (59),

$$
2 e ^ { - d / 2 } \leq \frac { 9 s } { n ^ { 2 } } .
$$

Since $d \geq 1$ , this implies that

$$
p _ { k - 1 } \leq { \frac { 9 s d } { n ^ { 2 } } } .\tag{68}
$$

Step 4: Simplify $\beta .$ . From $d \leq s / 1 6$ , we have $2 \sqrt { d } \leq \sqrt { s } / 2$ , so

$$
{ \sqrt { s } } + 2 { \sqrt { d } } \leq { \frac { 3 } { 2 } } { \sqrt { s } } .
$$

From $d \leq n / 1 6$ , we have $2 \sqrt { d } \leq \sqrt { n } / 2$ , so

$$
{ \sqrt { n } } - 2 { \sqrt { d } } \geq { \frac { 1 } { 2 } } { \sqrt { n } } .
$$

Therefore

$$
\beta = { \frac { ( { \sqrt { s } } + 2 { \sqrt { d } } ) ^ { 2 } } { ( { \sqrt { n } } - 2 { \sqrt { d } } ) ^ { 2 } } } \leq { \frac { ( { \frac { 3 } { 2 } } { \sqrt { s } } ) ^ { 2 } } { ( { \frac { 1 } { 2 } } { \sqrt { n } } ) ^ { 2 } } } = 9 { \frac { s } { n } } .\tag{69}
$$

Combining (68) and (69), we get

$$
\beta \frac { d } { n } + p _ { k - 1 } \leq 9 \frac { s d } { n ^ { 2 } } + 9 \frac { s d } { n ^ { 2 } } = 1 8 \frac { s d } { n ^ { 2 } } .\tag{70}
$$

Step 5: One-step Descent in Expectation. Taking conditional expectation in (62) and using (63) and (66), we obtain

$$
\begin{array} { l } { \displaystyle \mathbb { E } \big [ g ( y _ { k - 1 } ) - g ( y _ { k } ) \mid \mathcal { F } _ { k - 1 } \big ] \geq \alpha \frac { d } { n } \| u _ { k - 1 } \| ^ { 2 } - \frac { \alpha ^ { 2 } L } { 2 } \left( \beta \frac { d } { n } + p _ { k - 1 } \right) \| u _ { k - 1 } \| ^ { 2 } } \\ { \geq \left( \alpha \frac { d } { n } - \frac { \alpha ^ { 2 } L } { 2 } \cdot 1 8 \frac { s d } { n ^ { 2 } } \right) \| u _ { k - 1 } \| ^ { 2 } . } \end{array}
$$

With the choice $\begin{array} { r } { \alpha = \frac { n } { 1 8 s L } } \end{array}$ , this becomes

$$
\begin{array} { r l } {  { \mathbb { E } [ g ( y _ { k - 1 } ) - g ( y _ { k } ) \ | \ \mathcal { F } _ { k - 1 } ] \ge ( \frac { d } { 1 8 s L } - \frac { L } { 2 } \cdot \frac { n ^ { 2 } } { ( 1 8 s L ) ^ { 2 } } \cdot 1 8 \frac { s d } { n ^ { 2 } } ) \| u _ { k - 1 } \| ^ { 2 } } } \\ & { = ( \frac { d } { 1 8 s L } - \frac { d } { 3 6 s L } ) \| u _ { k - 1 } \| ^ { 2 } } \\ & { = \frac { d } { 3 6 s L } \| u _ { k - 1 } \| ^ { 2 } . } \end{array}
$$

Using $\| u _ { k - 1 } \| = \| \nabla f ( x _ { k - 1 } ) \|$ , we obtain

$$
\mathbb { E } [ f ( x _ { k - 1 } ) - f ( x _ { k } ) \mid { \mathcal { F } } _ { k - 1 } ] \geq { \frac { d } { 3 6 s L } } \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } .\tag{71}
$$

Taking expectation again yields

$$
\mathbb { E } [ f ( x _ { k - 1 } ) - f ( x _ { k } ) ] \ge \frac { d } { 3 6 s L } \mathbb { E } \big [ \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \big ] .
$$

Step 6: Telescoping. Summing over $k = 0 , \ldots , N - 1$ , we obtain

$$
\frac { d } { 3 6 s L } \sum _ { k = 1 } ^ { N } \mathbb { E } \big [ \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \big ] \leq \sum _ { k = 1 } ^ { N } \mathbb { E } \big [ f ( x _ { k - 1 } ) - f ( x _ { k } ) \big ] = f ( x _ { 0 } ) - \mathbb { E } \big [ f ( x _ { N } ) \big ] \leq f ( x _ { 0 } ) - f ^ { * } .
$$

Therefore

$$
\frac { 1 } { N } \sum _ { k = 1 } ^ { N } \mathbb { E } \big [ \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \big ] \leq \frac { 3 6 L s } { N d } \big ( f ( x _ { 0 } ) - f ^ { * } \big ) .
$$

Finally,

$$
\operatorname* { m i n } _ { 1 \leq k \leq N } \mathbb { E } \big [ \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \big ] \leq \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \mathbb { E } \big [ \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \big ] ,
$$

which proves (60).

## D Proof for Minibatch Variant

This section is dedicated to proving both the iteration and computational complexity of the minibatch variant of the algorithm. The proof is split into 2 parts: first obtain an expression for the lower bound of the alignment in expectation, then subsequently simplifying the iteration convergence proof with a tweak to the proof used in Section B.

## D.1 Proof for Alignment in Minibatch Setting (Proposition 3)

The persistence of the alignment tells us that the same guidance vector can still be used in the subsequent gradient descent step. To show this, we split the proof into 2 main parts.

1. Find a recurrence for $\begin{array} { r } { \mathbb { E } [ \langle \hat { v } _ { k } , \nabla f ( x _ { k } ) \rangle ^ { 2 } ] \ge \gamma _ { k } \mathbb { E } [ \| \nabla f ( x _ { k } ) \| ^ { 2 } ] - \eta _ { k } \sigma ^ { 2 } } \end{array}$ , where $\eta _ { k }$ can even potentially be 0.

2. Show that $\gamma _ { k } > \delta$ is satisfied under the assumption $r _ { i + 1 } - r _ { i } = i ^ { \beta }$

Part 1a: Obtain a recursion with respect to the previous gradient. Define the σ-algebras $\mathcal { F } _ { k } = \sigma ( \nabla g _ { 0 } , \widetilde { P } _ { 0 } , \nabla g _ { 1 } , \widetilde { P } _ { 1 } , \cdot \cdot \cdot , \nabla g _ { k - 1 } , \widetilde { P } _ { k - 1 } )$ and $\mathcal { G } _ { k } = \sigma ( \mathcal { F } _ { k } , \nabla g _ { k } )$ . Then, by re-writing $\nabla f ( x _ { k } ) =$ $\nabla f ( x _ { k } ) - \nabla f ( x _ { k - 1 } ) + \nabla f ( x _ { k - 1 } )$ like before, we can get the following lower bound:

$$
\langle \hat { v } _ { k } , \nabla f ( x _ { k } ) \rangle ^ { 2 } \geq \langle \hat { v } _ { k } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } - 2 L | \langle \hat { v } _ { k } , \nabla f ( x _ { k - 1 } ) \rangle | \| x _ { k } - x _ { k - 1 } \| .
$$

For the second term on the RHS, the descent direction is the projected gradient of the gradient estimate, instead of the projected gradient of the true gradient:

$$
x _ { k } - x _ { k - 1 } = - \alpha _ { k - 1 } P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla g _ { k - 1 } .
$$

Instead of taking conditional expectation under $\mathcal { F } _ { k - 1 }$ , we first take it under $\mathcal { G } _ { k - 1 }$ to get

$$
\begin{array} { r } { \mathbb { E } \left[ \langle \hat { v } _ { k } , \nabla f ( x _ { k } ) \rangle ^ { 2 } \mid \mathcal { G } _ { k - 1 } \right] \geq \langle \hat { v } _ { k } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } - 2 \alpha _ { k - 1 } L \left| \langle \hat { v } _ { k } , \nabla f ( x _ { k - 1 } ) \rangle \right| \mathbb { E } \left[ \left\| P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla g _ { k - 1 } \right\| \mid \mathcal { G } _ { k - 1 } \right] , } \end{array}
$$

where the expectation term on the RHS can be bounded similarly as before to obtain

$$
\mathbb { E } \left[ \Vert P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla g _ { k - 1 } \Vert \mid { \mathcal G } _ { k - 1 } \right] \leq \sqrt { \left( 1 - \frac { d } { n - 1 } \right) } \left. \langle \hat { v } _ { k - 1 } , \nabla g _ { k - 1 } \rangle \right. + \sqrt { \frac { d } { n - 1 } } \Vert \nabla g _ { k - 1 } \Vert ,
$$

finally giving us

$$
\begin{array} { r l } & { \mathbb { E } \left[ \langle \hat { v } _ { k } , \nabla f ( x _ { k } ) \rangle ^ { 2 } \mid { \mathcal G } _ { k - 1 } \right] \geq \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } - 2 \alpha _ { k - 1 } L | \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle | \times } \\ & { \qquad \left( \sqrt { \left( 1 - \displaystyle \frac { d } { n - 1 } \right) } | \langle \hat { v } _ { k - 1 } , \nabla g _ { k - 1 } \rangle | + \sqrt { \displaystyle \frac { d } { n - 1 } } \| \nabla g _ { k - 1 } \| \right) . } \end{array}
$$

Compared to earlier, where we only had $\nabla f ( x _ { k - 1 } )$ throughout, here we have $\nabla g _ { k - 1 }$ , so we have to work around diferently. Using the identity $2 a b \leq a ^ { 2 } + b ^ { 2 }$ , we can bound the second term by

$$
| \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle | | \langle \hat { v } _ { k - 1 } , \nabla g _ { k - 1 } \rangle | \leq \frac { 1 } { 2 } \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } + \frac { 1 } { 2 } \langle \hat { v } _ { k - 1 } , \nabla g _ { k - 1 } \rangle ^ { 2 } .
$$

As for the third term, taking expectation under $\mathcal { F } _ { k - 1 }$ , we can obtain

$$
\begin{array} { r l } { \mathbb { E } \left[ | \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle | \left. \lVert \nabla g _ { k - 1 } \right. \right. | \mathcal { F } _ { k - 1 } ] \leq \mathbb { E } \left[ \left. \nabla g _ { k - 1 } \right. \right. | \mathcal { F } _ { k - 1 } ] \cdot \left. \nabla f ( x _ { k - 1 } ) \right. } & { } \\ & { \qquad \leq \left. \nabla f ( x _ { k - 1 } ) \right. \sqrt { \sigma _ { m } ^ { 2 } + \left. \nabla f ( x _ { k - 1 } ) \right. ^ { 2 } } } \\ & { \qquad \leq \displaystyle \frac { 1 } { 2 } \lVert \nabla f ( x _ { k - 1 } ) \rVert ^ { 2 } + \frac { 1 } { 2 } \left( \sigma _ { m } ^ { 2 } + \left. \nabla f ( x _ { k - 1 } ) \right. ^ { 2 } \right) } \\ & { \qquad = \left. \nabla f ( x _ { k - 1 } ) \right. ^ { 2 } + \displaystyle \frac { 1 } { 2 } \sigma _ { m } ^ { 2 } , } \end{array}
$$

where we used the same inequality as before and the upper bound of the variance of the mini-batch. Substituting everything back into the conditional expectation of the correlation, we obtain the lower bound

$$
\begin{array} { r l } & { \mathbb { E } \left[ \langle \hat { v } _ { k } , \nabla f ( x _ { k } ) \rangle ^ { 2 } \mid \mathcal { G } _ { k - 1 } \right] \geq \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } - 2 \alpha _ { k - 1 } L \sqrt { \cfrac { d } { n - 1 } } \left( \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } + \frac { 1 } { 2 } \sigma _ { m } ^ { 2 } \right) } \\ & { \qquad - \alpha _ { k - 1 } L \sqrt { 1 - \cfrac { d } { n - 1 } } \Big ( \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } + \mathbb { E } [ \langle \hat { v } _ { k - 1 } , \nabla g _ { k - 1 } \rangle ^ { 2 } \mid \mathcal { F } _ { k - 1 } ] \Big ) } \\ & { \qquad = \left( 1 - \alpha _ { k - 1 } L \sqrt { 1 - \cfrac { d } { n - 1 } } \right) \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } - \alpha _ { k - 1 } L \sqrt { \cfrac { d } { n - 1 } } \sigma _ { m } ^ { 2 } } \\ & { \qquad - \alpha _ { k - 1 } L \sqrt { 1 - \cfrac { d } { n - 1 } } \mathbb { E } \left[ \langle \hat { v } _ { k - 1 } , \nabla g _ { k - 1 } \rangle ^ { 2 } \mid \mathcal { F } _ { k - 1 } \right] - 2 \alpha _ { k - 1 } L \sqrt { \cfrac { d } { n - 1 } } \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } . } \end{array}
$$

For the second term, we can similarly express $\nabla g _ { k - 1 } = \nabla g _ { k - 1 } - \nabla f ( x _ { k - 1 } ) + \nabla f ( x _ { k - 1 } )$ to get

$$
\begin{array} { r } { \mathbb { E } \left[ \langle \hat { v } _ { k - 1 } , \nabla g _ { k - 1 } \rangle ^ { 2 } \mid \mathcal { F } _ { k - 1 } \right] = \mathbb { E } \left[ \left( \langle \hat { v } _ { k - 1 } , \nabla g _ { k - 1 } - \nabla f ( x _ { k - 1 } ) \rangle + \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle \right) ^ { 2 } \mid \mathcal { F } _ { k - 1 } \right] } \\ { = \mathbb { E } \left[ \langle \hat { v } _ { k - 1 } , \nabla g _ { k - 1 } - \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } + \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } \mid \mathcal { F } _ { k - 1 } \right] } \end{array}
$$

where the cross term cancels out due to $\nabla f ( x _ { k - 1 } ) = \mathbb { E } [ \nabla g _ { k - 1 } ]$ . We can then bound the first term with Cauchy-Schwartz and the variance assumption to get

$$
\begin{array} { r } { \mathbb { E } \left[ \langle \hat { v } _ { k - 1 } , \nabla g _ { k - 1 } \rangle ^ { 2 } \mid \mathcal { F } _ { k - 1 } \right] \leq \sigma _ { m } ^ { 2 } + \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } . } \end{array}
$$

Substituting this bound and simplifying everything, we obtain

$$
\begin{array} { r l } { \mathbb { E } \left[ \langle \hat { v } _ { k } , \nabla f ( x _ { k } ) \rangle ^ { 2 } \mid \mathcal { G } _ { k - 1 } \right] \geq \left( 1 - 2 \alpha _ { k - 1 } L \sqrt { 1 - \cfrac { d } { n - 1 } } \right) \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } } & { } \\ { - 2 \alpha _ { k - 1 } L \sqrt { \cfrac { d } { n - 1 } } \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } - \alpha _ { k - 1 } L \left( \sqrt { \cfrac { d } { n - 1 } } + \sqrt { 1 - \cfrac { d } { n - 1 } } \right) \sigma _ { m } ^ { 2 } } & { } \\ { \geq \left( 1 - 2 \alpha _ { k - 1 } L \sqrt { 1 - \cfrac { d } { n - 1 } } \right) \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } } & { } \\ { - 2 \alpha _ { k - 1 } L \sqrt { \cfrac { d } { n - 1 } } \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } - 2 \alpha _ { k - 1 } L \sigma _ { m } ^ { 2 } . } \end{array}
$$

Using the assumption that $\begin{array} { r } { \mathbb { E } \left[ \langle \hat { v } _ { k - 1 } , \nabla f ( x _ { k - 1 } ) \rangle ^ { 2 } \right] \geq \gamma _ { k - 1 } \mathbb { E } [ \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } ] - \eta _ { k - 1 } \sigma _ { m } ^ { 2 } } \end{array}$ , we can get

$$
\begin{array} { r l } & { \mathbb { E } [ \langle \hat { v } _ { k } , \nabla f ( x _ { k } ) \rangle ^ { 2 } ] \geq \left[ \left( 1 - 2 \alpha _ { k - 1 } L \sqrt { 1 - \frac { d } { n - 1 } } \right) \gamma _ { k - 1 } - 2 \alpha _ { k - 1 } L \sqrt { \frac { d } { n - 1 } } \right] \mathbb { E } [ \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } ] } \\ & { \qquad - \left[ \left( 1 - 2 \alpha _ { k - 1 } L \sqrt { 1 - \frac { d } { n - 1 } } \right) \eta _ { k - 1 } + 2 \alpha _ { k - 1 } L \right] \sigma _ { m } ^ { 2 } . } \end{array}
$$

Part 1b: Obtain a relationship between norms of consecutive gradient. This is done similarly to the vanilla case and the technique is the same; we just have to note that there is a variance bound on the mini-batch gradient instead of directly working with the variance of the true gradient. Using the decomposition of $\nabla f ( x _ { k } ) = \nabla f ( x _ { k } ) - \nabla f ( x _ { k - 1 } ) + \nabla f ( x _ { k - 1 } )$ , we get

$$
\| \nabla f ( x _ { k } ) \| \leq \| \nabla f ( x _ { k } ) - \nabla f ( x _ { k - 1 } ) \| + \| \nabla f ( x _ { k - 1 } ) \| \leq L \| x _ { k } - x _ { k - 1 } \| + \| \nabla f ( x _ { k - 1 } ) \| .
$$

For the first term, we have

$$
\begin{array} { r l } & { \mathbb { E } \left[ \Vert x _ { k } - x _ { k - 1 } \Vert ^ { 2 } \middle | \mathcal { F } _ { k - 1 } \right] = \alpha _ { k - 1 } ^ { 2 } \mathbb { E } \left[ \Vert P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla g _ { k - 1 } \Vert ^ { 2 } \middle | \mathcal { F } _ { k - 1 } \right] } \\ & { \qquad \leq \alpha _ { k - 1 } ^ { 2 } \mathbb { E } \left[ \Vert \nabla g _ { k - 1 } \Vert ^ { 2 } \middle | \mathcal { F } _ { k - 1 } \right] } \\ & { \qquad \leq \alpha _ { k - 1 } ^ { 2 } \left( \Vert \nabla f ( x _ { k - 1 } ) \Vert ^ { 2 } + \sigma _ { m } ^ { 2 } \right) . } \end{array}
$$

Consequently, we can get

$$
\begin{array} { r } { \mathbb { E } \left[ \| x _ { k } - x _ { k - 1 } \| | \mathcal F _ { k - 1 } \right] \leq \alpha _ { k - 1 } \left( \| \nabla f ( x _ { k - 1 } ) \| + \sigma _ { m } \right) . } \end{array}
$$

Then, we have

$$
\begin{array} { r l } & { \mathbb { E } \left[ \Vert \nabla f ( x _ { k } ) \Vert ^ { 2 } \mid \mathcal { F } _ { k - 1 } \right] \leq \Vert \nabla f ( x _ { k - 1 } ) \Vert ^ { 2 } + 2 L \Vert \nabla f ( x _ { k - 1 } ) \Vert \mathbb { E } \left[ \Vert x _ { k } - x _ { k - 1 } \Vert ~ \middle | ~ \mathcal { F } _ { k - 1 } \right] } \\ & { \qquad + L ^ { 2 } \mathbb { E } \left[ \Vert x _ { k } - x _ { k - 1 } \Vert ^ { 2 } ~ \middle | ~ \mathcal { F } _ { k - 1 } \right] } \\ & { \qquad \leq \Vert \nabla f ( x _ { k - 1 } ) \Vert ^ { 2 } + 2 \alpha _ { k - 1 } L \Vert \nabla f ( x _ { k - 1 } ) \Vert ( \sigma _ { m } + \Vert \nabla f ( x _ { k - 1 } ) \Vert ) } \\ & { \qquad + \alpha _ { k - 1 } ^ { 2 } L ^ { 2 } ( \sigma _ { m } ^ { 2 } + \Vert \nabla f ( x _ { k - 1 } ) \Vert ^ { 2 } ) } \\ & { \qquad = ( 1 + \alpha _ { k - 1 } L ) ^ { 2 } \Vert \nabla f ( x _ { k - 1 } ) \Vert ^ { 2 } + 2 \alpha _ { k - 1 } L \Vert \nabla f ( x _ { k - 1 } ) \Vert \sigma _ { m } + \alpha _ { k - 1 } ^ { 2 } L ^ { 2 } \sigma _ { m } ^ { 2 } . } \end{array}
$$

The cross-term can once again be bounded by $2 \| \nabla f ( x _ { k - 1 } ) \| \sigma _ { m } \leq \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } + \sigma _ { m } ^ { 2 }$ to get

$$
\begin{array} { r l } & { \mathbb { E } \left[ \Vert \nabla f ( x _ { k } ) \Vert ^ { 2 } \mid \mathcal { F } _ { k - 1 } \right] \leq ( 1 + \alpha _ { k - 1 } L ) ^ { 2 } \Vert \nabla f ( x _ { k - 1 } ) \Vert ^ { 2 } + \alpha _ { k - 1 } L ( \Vert \nabla f ( x _ { k - 1 } ) \Vert ^ { 2 } + \sigma _ { m } ^ { 2 } ) + \alpha _ { k - 1 } ^ { 2 } L ^ { 2 } \sigma _ { m } ^ { 2 } } \\ & { \qquad = \left[ ( 1 + \alpha _ { k - 1 } L ) ^ { 2 } + \alpha _ { k - 1 } L \right] \Vert \nabla f ( x _ { k - 1 } ) \Vert ^ { 2 } + \left[ \alpha _ { k - 1 } L + \alpha _ { k - 1 } ^ { 2 } L ^ { 2 } \right] \sigma _ { m } ^ { 2 } . } \end{array}
$$

Rewriting this inequality, we have

$$
\mathbb { E } \left[ \Vert \nabla f ( x _ { k - 1 } ) \Vert ^ { 2 } \right] \geq \frac { 1 } { ( 1 + \alpha _ { k - 1 } L ) ^ { 2 } + \alpha _ { k - 1 } L } \mathbb { E } \left[ \Vert \nabla f ( x _ { k } ) \Vert ^ { 2 } \right] - \frac { ( 1 + \alpha _ { k - 1 } L ) \alpha _ { k - 1 } L } { ( 1 + \alpha _ { k - 1 } L ) ^ { 2 } + \alpha _ { k - 1 } L } \sigma _ { m } ^ { 2 } .\tag{72}
$$

With this bound on the norms of consecutive gradients, we obtain the expression

$$
\begin{array} { r l } & { \mathbb { E } [ \langle \hat { v } _ { k } , \nabla f ( x _ { k } ) \rangle ^ { 2 } ] = \frac { \Big ( 1 - 2 \alpha _ { k - 1 } L \sqrt { 1 - \frac { d } { n - 1 } } \Big ) \gamma _ { k - 1 } - 2 \alpha _ { k - 1 } L \sqrt { \frac { d } { n - 1 } } } { ( 1 + \alpha _ { k - 1 } L ) ^ { 2 } + \alpha _ { k - 1 } L } \mathbb { E } [ \| \nabla f ( x _ { k } ) \| ^ { 2 } ] } \\ & { \qquad - \left[ 2 \alpha _ { k - 1 } L + \frac { \left[ \left( 1 - 2 \alpha _ { k - 1 } L \sqrt { 1 - \frac { d } { n - 1 } } \right) \gamma _ { k - 1 } - 2 \alpha _ { k - 1 } L \sqrt { \frac { d } { n - 1 } } \right] [ 1 + \alpha _ { k - 1 } L ] \alpha _ { k - 1 } L } { ( 1 + \alpha _ { k - 1 } L ) ^ { 2 } + \alpha _ { k - 1 } L } \right. } \\ & { \qquad \left. + \left( 1 - 2 \alpha _ { k - 1 } L \sqrt { 1 - \frac { d } { n - 1 } } \right) \eta _ { k - 1 } \right] \sigma _ { m } ^ { 2 } } \\ & { = \gamma _ { k } \mathbb { E } [ \| \nabla f ( x _ { k } ) \| ^ { 2 } ] - \eta _ { k } \sigma _ { m } ^ { 2 } , } \end{array}
$$

where we have the recurrence relation on $\gamma _ { k }$ and $\eta _ { k }$ as

$$
\begin{array} { l } { \gamma _ { k } = \frac { \left( 1 - 2 \alpha _ { k - 1 } L \sqrt { 1 - \frac { d } { n - 1 } } \right) \gamma _ { k - 1 } - 2 \alpha _ { k - 1 } L \sqrt { \frac { d } { n - 1 } } } { ( 1 + \alpha _ { k - 1 } L ) ^ { 2 } + \alpha _ { k - 1 } L } } \\ { \eta _ { k } = 2 \alpha _ { k - 1 } L + \frac { \left[ \left( 1 - 2 \alpha _ { k - 1 } L \sqrt { 1 - \frac { d } { n - 1 } } \right) \gamma _ { k - 1 } - 2 \alpha _ { k - 1 } L \sqrt { \frac { d } { n - 1 } } \right] [ 1 + \alpha _ { k - 1 } L ] \alpha _ { k - 1 } L } { \left( 1 + \alpha _ { k - 1 } L \right) ^ { 2 } + \alpha _ { k - 1 } L } } \\ { + \left( 1 - 2 \alpha _ { k - 1 } L \sqrt { 1 - \frac { d } { n - 1 } } \right) \eta _ { k - 1 } . } \end{array}\tag{73}
$$

(74)

Part 2: Solving the recurrence under step schedule assumption. First, we would want to find out $\gamma _ { 0 } , \eta _ { 0 } , \mathrm { i . e }$ . the value of $\gamma , \eta$ at the step where we update the alignment vector $v _ { 0 }$ . Given that have $\begin{array} { r } { v _ { 0 } = \frac { 1 } { m ^ { \prime } } \sum _ { i \in B ^ { \prime } } \nabla f _ { i } ( x _ { 0 } ) } \end{array}$ for a mini-batch $B ^ { \prime }$ , where we choose $| B ^ { \prime } | = m ^ { \prime } > m$ , we can obtain

$$
\begin{array} { r l } & { \mathbb { E } \left[ \langle v _ { 0 } , \nabla f ( x _ { 0 } ) \rangle \right] ^ { 2 } = \mathbb { E } \left[ \langle \hat { v } _ { 0 } , \nabla f ( x _ { 0 } ) \rangle \| v _ { 0 } \| \right] ^ { 2 } } \\ & { \quad \quad \quad \quad \leq \mathbb { E } \left[ \langle \hat { v } _ { 0 } , \nabla f ( x _ { 0 } ) \rangle ^ { 2 } \right] \mathbb { E } \left[ \| v _ { 0 } \| ^ { 2 } \right] } \\ & { \quad \quad \quad \leq \mathbb { E } \left[ \langle \hat { v } _ { 0 } , \nabla f ( x _ { 0 } ) \rangle ^ { 2 } \right] \left( \sigma _ { m ^ { \prime } } ^ { 2 } + \| \nabla f ( x _ { 0 } ) \| ^ { 2 } \right) . } \end{array}
$$

On the other hand, since $v _ { 0 }$ is an unbiased estimator of the gradient, we also have

$$
\mathbb { E } \left[ \langle v _ { 0 } , \nabla f ( x _ { 0 } ) \rangle \right] ^ { 2 } = \| \nabla f ( x _ { 0 } ) \| ^ { 4 } .
$$

Combining these 2, we get

$$
\mathbb { E } \left[ \langle \hat { v } _ { 0 } , \nabla f ( x _ { 0 } ) \rangle ^ { 2 } \right] \geq \frac { \| \nabla f ( x _ { 0 } ) \| ^ { 4 } } { \sigma _ { m ^ { \prime } } ^ { 2 } + \| \nabla f ( x _ { 0 } ) \| ^ { 2 } } = \left( 1 - \frac { \sigma _ { m ^ { \prime } } ^ { 2 } } { \sigma _ { m ^ { \prime } } ^ { 2 } + \| \nabla f ( x _ { 0 } ) \| ^ { 2 } } \right) \| \nabla f ( x _ { 0 } ) \| ^ { 2 } ,
$$

giving us

$$
\gamma _ { 0 } = 1 - \frac { \sigma _ { m ^ { \prime } } ^ { 2 } } { \sigma _ { m ^ { \prime } } ^ { 2 } + \| \nabla f ( x _ { 0 } ) \| ^ { 2 } }\tag{75}
$$

$$
\eta _ { 0 } = 0 .\tag{76}
$$

Assuming $r _ { i } < k < r _ { i + 1 }$ , where $r _ { i }$ is the $i ^ { t h }$ refresh step, we can use the recurence relation of $\gamma _ { k }$ to obtain

$$
\begin{array} { l } { \gamma _ { k } \geq \frac { \displaystyle \left( 1 - 2 \alpha _ { k - 1 } L \right) \gamma _ { k - 1 } - 2 \alpha _ { k - 1 } L \sqrt { \frac { d } { n - 1 } } } { \displaystyle \left( 1 + \alpha _ { k - 1 } L \right) ^ { 2 } + \alpha _ { k - 1 } L } } \\ { \geq \gamma _ { r _ { i } } \prod _ { i = r _ { i } } ^ { k - 1 } a _ { i } - \displaystyle \sum _ { i = r _ { i } } ^ { k - 1 } b _ { i } \prod _ { j = i + 1 } ^ { k - 1 } a _ { j } , } \end{array}
$$

where the coeficients $a _ { j }$ and $b _ { j }$ can be simplified to

$$
\begin{array} { r l } & { a _ { j } = \cfrac { 1 - 2 \alpha _ { j - 1 } L } { ( 1 + \alpha _ { j - 1 } L ) ^ { 2 } + \alpha _ { j - 1 } L } = \cfrac { 1 - \frac { 2 } { \sqrt { j } } } { \left( 1 + \frac { 1 } { \sqrt { j } } \right) ^ { 2 } + \frac { 1 } { \sqrt { j } } } = \cfrac { ( \sqrt { j } - 2 ) ( \sqrt { j } ) } { j + 1 + 3 \sqrt { j } } } \\ & { b _ { j } = \cfrac { 2 \alpha _ { j - 1 } L \sqrt { \frac { d } { n - 1 } } } { ( 1 + \alpha _ { j - 1 } L ) ^ { 2 } + \alpha _ { j - 1 } L } = \cfrac { 2 \sqrt { j } } { j + 1 + 3 \sqrt { j } } \sqrt { \frac { d } { n - 1 } } , } \end{array}
$$

under the assumption that the step-size is $\begin{array} { r } { \alpha _ { k } \ = \ \frac { 1 } { L \sqrt { k + 1 } } } \end{array}$ . Equivalently, we can obtain that the correlation right before the next update has to satisfy the equation

$$
\gamma _ { r _ { i + 1 } - 1 } \geq \gamma _ { r _ { i } } \prod _ { k = r _ { i } } ^ { r _ { i + 1 } - 2 } a _ { k } - \sum _ { k = r _ { i } } ^ { r _ { i + 1 } - 2 } b _ { k } \prod _ { j = k + 1 } ^ { r _ { i + 1 } - 2 } a _ { j } > \delta .\tag{77}
$$

to ensure that it is still at least $\delta .$ To make calculations slightly easier, we will simplify the expressions for $a _ { j }$ and $b _ { j }$ to get

$$
\begin{array} { l } { \displaystyle a _ { j } = \frac { \left( \sqrt j - 2 \right) \left( \sqrt j \right) } { j + 1 + 3 \sqrt j } = \frac { j - 2 \sqrt j } { j + 1 + 3 \sqrt j } = 1 - \frac { 5 \sqrt j + 1 } { j + 1 + 3 \sqrt j } } \\ { \displaystyle \quad \geq 1 - \frac { 5 \sqrt j } { j + 3 \sqrt j } = 1 - \frac { 5 } { 3 + \sqrt j } , } \\ { \displaystyle b _ { j } = \frac { 2 \sqrt j } { j + 1 + 3 \sqrt j } \sqrt { \frac { d } { n - 1 } } \leq \frac { 2 } { \sqrt j } \sqrt { \frac { d } { n - 1 } } . } \end{array}
$$

With these simplified expressions, we can simplify the expression in $7 7$ to get

$$
\begin{array} { r l } & { \gamma _ { r _ { i + 1 } } \geq \gamma _ { r _ { i } } \displaystyle \prod _ { k = r _ { i } + 1 } ^ { r _ { i + 1 } - 2 } a _ { k } - \displaystyle \sum _ { k = r _ { i } } ^ { r _ { i + 1 } - 2 } b _ { k } \displaystyle \prod _ { j = k + 1 } ^ { r _ { i + 1 } - 2 } a _ { j } } \\ & { \qquad \geq \gamma _ { r _ { i } } a _ { r _ { i } + 1 } ^ { r _ { i + 1 } - r _ { i } - 1 } - \displaystyle \sum _ { k = r _ { i } } ^ { r _ { i + 1 } - 2 } b _ { k } } \\ & { \qquad \geq \gamma _ { r _ { i } } \left( 1 - \displaystyle \frac { 5 } { 3 + \sqrt { r _ { i } + 1 } } \right) ^ { r _ { i + 1 } - r _ { i } - 1 } - 2 \sqrt { \displaystyle \frac { d } { n - 1 } } \sum _ { k = r _ { i } } ^ { r _ { i + 1 } - 2 } \displaystyle \frac { 1 } { \sqrt { j } } . } \end{array}
$$

For the second term, we use the approximation $\begin{array} { r } { \sum _ { i = m } ^ { n } { \frac { 1 } { \sqrt { i } } } \leq 2 ( \sqrt { n } - \sqrt { m } ) + m ^ { - 1 / 2 } } \end{array}$ to get that $\begin{array} { r } { \sum _ { k = r _ { i } } ^ { r _ { i + 1 } - 2 } \frac { 1 } { \sqrt { j } } \leq 2 ( \sqrt { r _ { i + 1 } } - \sqrt { r _ { i } } ) + ( r _ { i } ) ^ { - 1 / 2 } } \end{array}$ for suficiently large $r _ { i }$ . Letting $r _ { i + 1 } - r _ { i } = \epsilon _ { i }$ and assuming $\epsilon _ { i } \ll r _ { i }$ , we can expand the square root by Taylor’s formula to get the approximation

$$
{ \sqrt { r _ { i } + \epsilon _ { i } } } - { \sqrt { r _ { i } } } = { \sqrt { r _ { i } } } \left( { \sqrt { 1 + { \frac { \epsilon _ { i } } { r _ { i } } } } } - 1 \right) = { \sqrt { r _ { i } } } \left( { \frac { \epsilon _ { i } } { 2 r _ { i } } } + o \left( { \frac { \epsilon _ { i } } { r _ { i } } } \right) \right) .\tag{78}
$$

Consequently, we obtain the bound

$$
\begin{array} { l } { \gamma _ { r _ { i + 1 } } \geq \gamma _ { r _ { i } } \left( 1 - \displaystyle \frac { 5 } { 3 + \sqrt { r _ { i } + 1 } } \right) ^ { r _ { i + 1 } - r _ { i } } - 2 \sqrt { \frac { d } { n - 1 } } \left( \displaystyle \frac { r _ { i + 1 } - r _ { i } } { \sqrt { r _ { i } } } + o \left( \displaystyle \frac { \sqrt { r _ { i + 1 } } - \sqrt { r _ { i } } } { \sqrt { r _ { i } } } \right) \right) } \\ { \geq \gamma _ { r _ { i } } \left( 1 - \displaystyle \frac { 5 } { 3 + \sqrt { r _ { i } } } \right) ^ { r _ { i + 1 } - r _ { i } } - 4 \sqrt { \frac { d } { n - 1 } } \left( \displaystyle \frac { r _ { i + 1 } - r _ { i } } { \sqrt { r _ { i } } } \right) } \end{array}
$$

for suficiently large $r _ { i }$ . Using the assumption that $r _ { i + 1 } - r _ { i } = i ^ { \beta }$ , we can find that the order of $r _ { i }$ is approximately bounded by

$$
r _ { i } = \sum _ { k = 1 } ^ { i } ( r _ { k } - r _ { k - 1 } ) = \sum _ { k = 1 } ^ { i } k ^ { \beta } \geq \int _ { 0 } ^ { i } k ^ { \beta } d k = { \frac { i ^ { \beta + 1 } } { \beta + 1 } } .\tag{79}
$$

and

$$
\sqrt { r _ { i } } \ge \sqrt { \frac { i ^ { \beta + 1 } } { \beta + 1 } } \ge \frac { i ^ { ( \beta + 1 ) / 2 } } { \sqrt { \beta + 1 } }
$$

for suficiently large i. We can now simplify the bound to get

$$
\begin{array} { c } { \gamma _ { r _ { i + 1 } } \geq \gamma _ { r _ { i } } \left( 1 - \frac { 5 \sqrt { \beta + 1 } } { i ^ { ( \beta + 1 ) / 2 } } \right) ^ { i ^ { \beta } } - 4 \sqrt { \frac { d } { n - 1 } } \left( \sqrt { \beta + 1 } \frac { i ^ { \beta } } { i ^ { ( \beta + 1 ) / 2 } } \right) } \\ { = \gamma _ { r _ { i } } \left( 1 - \frac { 5 \sqrt { \beta + 1 } } { i ^ { ( \beta + 1 ) / 2 } } \right) ^ { i ^ { \beta } } - 4 \sqrt { \frac { d ( \beta + 1 ) } { n - 1 } } \left( i ^ { ( \beta - 1 ) / 2 } \right) . } \end{array}
$$

Setting the RHS to be $> \delta$ and simplifying the inequality, we obtain

$$
\gamma _ { r _ { i } } \left( 1 - \frac { 5 \sqrt { \beta + 1 } } { i ^ { ( \beta + 1 ) / 2 } } \right) ^ { i ^ { \beta } } > \delta + 4 \sqrt { \frac { d ( \beta + 1 ) } { n - 1 } } i ^ { ( \beta - 1 ) / 2 } .
$$

Taking logarithm on both sides and moving the terms around, we get

$$
i ^ { \beta } < \frac { \log \left( \frac { \delta + 4 \sqrt { \frac { d ( \beta + 1 ) } { n - 1 } } i ^ { ( \beta - 1 ) / 2 } } { \gamma _ { r _ { i } } } \right) } { \log \left( 1 - \frac { 5 \sqrt { \beta + 1 } } { i ^ { ( \beta + 1 ) / 2 } } \right) } = \frac { \log \left( \frac { \gamma _ { r _ { i } } } { \delta + 4 \sqrt { \frac { d ( \beta + 1 ) } { n - 1 } } i ^ { ( \beta - 1 ) / 2 } } \right) } { \log \left( 1 + \frac { 5 \sqrt { \beta + 1 } } { i ^ { ( \beta + 1 ) / 2 } - 5 \sqrt { \beta + 1 } } \right) } .
$$

For suficiently large i, we can bound the numerator by $\log ( \gamma _ { r _ { i } } / 2 \delta )$ while the denominator can be

expanded using Taylor’s expansion to get

$$
\begin{array} { c } { { i ^ { \beta } \leq \frac { \log \big ( \frac { \gamma _ { r } } { 2 \delta } \big ) } { \frac { 5 \sqrt { \beta + 1 } } { i ^ { ( \beta + 1 ) / 2 } - 5 \sqrt { \beta + 1 } } \big ( 1 + o ( 1 ) \big ) } } } \\ { { \leq \frac { \log \big ( \frac { \gamma _ { r } } { 2 \delta } \big ) } { 5 \sqrt { \beta + 1 } \big ( 1 + o ( 1 ) \big ) } i ^ { ( \beta + 1 ) / 2 } } } \\ { { \lesssim \frac { \log \big ( \frac { \gamma _ { r } } { 2 \delta } \big ) } { 5 \sqrt { \beta + 1 } } i ^ { ( \beta + 1 ) / 2 } . } } \end{array}
$$

Consequently, we have an upper-bound on the equation satisfied by $\beta \colon$

$$
i ^ { ( \beta - 1 ) / 2 } \lesssim \frac { 1 } { 5 } \log ( \gamma _ { r _ { i } } / 2 \delta ) ,\tag{80}
$$

where $\begin{array} { r } { \gamma _ { r _ { i } } = 1 - \frac { \sigma _ { m ^ { \prime } } ^ { 2 } } { \sigma _ { m ^ { \prime } } ^ { 2 } + \| \nabla f ( x _ { r _ { i } } ) \| ^ { 2 } } } \end{array}$ . For suficiently large i and large $\| \nabla f ( x _ { r _ { i } } ) \|$ (where we are still far from convergence), this inequality is eventually satisfied. We simply translate the starting iteration with $k _ { 0 }$ satisfying the inequality, with $\begin{array} { r } { \alpha _ { k } = \frac { 1 } { \sqrt { k + k _ { 0 } } } } \end{array}$ , where

$$
k _ { 0 } = r _ { i ^ { * } } . \qquad i ^ { * } \geq \left( \frac { 5 } { \log ( 1 / 2 \delta ) } \right) ^ { 2 / ( 1 - \beta ) } .\tag{81}
$$

## D.2 Proof for Iteration Complexity in Minibatch Setting (Theorem 7)

The proof for the convergence is similar to the vanilla case, just that now we have the mini-batch gradient and the variance term in the lower-bound for the correlation. By L-smoothness, we have

$$
f ( x _ { k } ) - f ( x _ { k - 1 } ) = - \alpha _ { k - 1 } \langle \nabla f ( x _ { k - 1 } ) , P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla g _ { k - 1 } \rangle + \frac { \alpha _ { k - 1 } ^ { 2 } L } { 2 } \| P _ { k - 1 } P _ { k - 1 } ^ { \top } \nabla g _ { k - 1 } \| ^ { 2 } .
$$

Using the same definitions for $\mathcal { F } _ { k }$ and $\mathcal { G } _ { k }$ , we have

$$
\begin{array} { r l } { \mathbb { E } [ f ( x _ { k } ) - f ( x _ { k - 1 } ) \big | \mathcal { G } _ { k } } & { \mathsf { 1 } = - \alpha _ { k } \mathsf { 1 }  \nabla f ( x _ { k - 1 } ) , [ ( 1 - \frac { d } { n - 1 } ) \hat { v } _ { k - 1 } \hat { v } _ { k - 1 } ^ { \top } + \frac { d } { n - 1 } \frac { 1 } { n - 1 } ] \nabla g _ { k - 1 }  } \\ & { \qquad + \frac { \alpha _ { k - 1 } ^ { 2 } L } { 2 }  \nabla g _ { k - 1 } , [ ( 1 - \frac { d } { n - 1 } ) \hat { v } _ { k - 1 } \hat { v } _ { k - 1 } ^ { \top } + \frac { d } { n - 1 } T ] \nabla g _ { k - 1 }  } \\ & { = - \alpha _ { k - 1 } ( 1 - \frac { d } { n - 1 } )  \nabla f ( x _ { k - 1 } ) , \hat { v } _ { k - 1 } \hat { v } _ { k - 1 } , \nabla g _ { k - 1 }  - \alpha _ { k - 1 } \frac { d } { n - 1 } \frac { 1 } { | \nabla f ( x _ { k - 1 } ) , \nabla g _ { k - 1 } \rangle } } \\ & { \qquad + \frac { \alpha _ { k - 1 } ^ { 2 } L } { 2 } ( 1 - \frac { d } { n - 1 } )  \nabla g _ { k - 1 } , \hat { v } _ { k - 1 }  ^ { 2 } + \frac { \alpha _ { k - 1 } ^ { 2 } L } { 2 } \frac { d } { n - 1 } \| \nabla g _ { k - 1 } \| ^ { 2 } } \\ & { \leq - \alpha _ { k - 1 } ( 1 - \frac { d } { n - 1 } )  \nabla f ( x _ { k - 1 } ) , \hat { v } _ { k - 1 } \rangle \langle \hat { v } _ { k - 1 } , \nabla g _ { k - 1 }  } \\ &  \qquad - \alpha _ { k - 1 } \frac { d } { n - 1 }  \nabla f ( x _ { k - 1 } ) , \nabla g _  k -  \end{array}
$$

Taking expectation with respect to $\mathcal { F } _ { k - 1 }$ and using the tower property of conditional expectation, we obtain

$$
\begin{array} { r l } & { \mathbb { E } \left[ f ( x _ { k } ) - f ( x _ { k - 1 } ) \middle \vert \mathcal { F } _ { k - 1 } \right] = \mathbb { E } \left[ \mathbb { E } \left[ f ( x _ { k } ) - f ( x _ { k - 1 } ) \middle \vert \mathcal { G } _ { k - 1 } \right] \middle \vert \mathcal { F } _ { k - 1 } \right] } \\ & { \qquad \leq - \alpha _ { k - 1 } \left( 1 - \frac { d } { n - 1 } \right) \langle \nabla f ( x _ { k - 1 } ) , \hat { v } _ { k - 1 } \rangle ^ { 2 } - \alpha _ { k - 1 } \frac { d } { n - 1 } \Vert \nabla f ( x _ { k - 1 } ) \Vert ^ { 2 } } \\ & { \qquad + \frac { \alpha _ { k - 1 } ^ { 2 } L } { 2 } \mathbb { E } \left[ \Vert \nabla g _ { k - 1 } \Vert ^ { 2 } \middle \vert \mathcal { F } _ { k - 1 } \right] , } \end{array}
$$

since $\nabla g _ { k - 1 }$ is an unbiased estimator of the gradient. i.e. $\mathbb { E } [ \nabla g _ { k - 1 } ] = \nabla f ( x _ { k - 1 } )$ . Equivalently, we have

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left[ f ( x _ { k } ) - f ( x _ { k - 1 } ) \right] \leq - \alpha _ { k - 1 } \left( 1 - \frac { d } { n - 1 } \right) \mathbb { E } \left[ \langle \nabla f ( x _ { k - 1 } ) , \hat { v } _ { k - 1 } \rangle ^ { 2 } \right] - \alpha _ { k - 1 } \frac { d } { n - 1 } \mathbb { E } \left[ \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \right] } \\ { \displaystyle \qquad + \frac { \alpha _ { k - 1 } ^ { 2 } L } { 2 } \mathbb { E } \left[ \| \nabla g _ { k - 1 } \| ^ { 2 } \right] . } \end{array}
$$

Using the assumption on the variance of the mini-batch gradient and the lower bound of the correlation, we obtain

$$
\begin{array} { r l } & { \mathbb { E } \left[ f ( x _ { k } ) - f ( x _ { k - 1 } ) \right] \leq - \alpha _ { k - 1 } \left( 1 - \displaystyle \frac { d } { n - 1 } \right) \left( \gamma _ { k - 1 } \mathbb { E } \left[ \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \right] - \eta _ { k - 1 } \sigma _ { m } ^ { 2 } \right) } \\ & { \qquad - \alpha _ { k - 1 } \displaystyle \frac { d } { n - 1 } \mathbb { E } \left[ \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \right] + \displaystyle \frac { \alpha _ { k - 1 } ^ { 2 } L } { 2 } \left( \sigma _ { m } ^ { 2 } + \mathbb { E } \left[ \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \right] \right) } \\ & { \qquad = - \left( \alpha _ { k - 1 } \left( 1 - \displaystyle \frac { d } { n - 1 } \right) \gamma _ { k - 1 } + \alpha _ { k - 1 } \displaystyle \frac { d } { n - 1 } - \displaystyle \frac { \alpha _ { k - 1 } ^ { 2 } L } { 2 } \right) \mathbb { E } \left[ \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \right] } \\ & { \qquad + \left( \alpha _ { k - 1 } \left( 1 - \displaystyle \frac { d } { n - 1 } \right) \eta _ { k - 1 } + \displaystyle \frac { \alpha _ { k - 1 } ^ { 2 } L } { 2 } \right) \sigma _ { m } ^ { 2 } . } \end{array}
$$

By assumption, we have $\gamma _ { k - 1 } > \delta _ { }$ , allowing us to simplify the above to

$$
\begin{array} { r l } {  { \mathbb { E } [ f ( x _ { k } ) - f ( x _ { k - 1 } ) ] \le - ( \alpha _ { k - 1 } ( 1 - \frac { d } { n - 1 } ) \delta + \alpha _ { k - 1 } \frac { d } { n - 1 } - \frac { \alpha _ { k - 1 } ^ { 2 } L } { 2 } ) \mathbb { E } [ \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } ] } } \\ & { \quad \quad \quad + ( \alpha _ { k - 1 } ( 1 - \frac { d } { n - 1 } ) \eta _ { k - 1 } + \frac { \alpha _ { k - 1 } ^ { 2 } L } { 2 } ) \sigma _ { m } ^ { 2 } . } \end{array}
$$

for $\alpha _ { k - 1 }$ such that $\begin{array} { r } { \alpha _ { k - 1 } \left( 1 - \frac { d } { n - 1 } \right) \delta + \alpha _ { k - 1 } \frac { d } { n - 1 } - \frac { \alpha _ { k - 1 } ^ { 2 } L } { 2 } > 0 } \end{array}$ , which can be satisfied for $\alpha _ { k - 1 } =$ $\begin{array} { r } { \frac { 1 } { L \sqrt { k } } \in \left( 0 , \frac { 2 } { L } \left( \left( 1 - \frac { d } { n - 1 } \right) \delta + \frac { d } { n - 1 } \right) \right) } \end{array}$ . Taking sum up to N steps, we have

$$
\begin{array} { r l } & { \mathbb { E } \left[ f ( x _ { N } ) - f ( x _ { 0 } ) \right] \leq - \displaystyle \sum _ { k = 1 } ^ { N } \left( \alpha _ { k - 1 } \left( 1 - \frac { d } { n - 1 } \right) \delta + \alpha _ { k - 1 } \frac { d } { n - 1 } - \frac { \alpha _ { k - 1 } ^ { 2 } L } { 2 } \right) \mathbb { E } \left[ \| \nabla f ( x _ { k - 1 } ) \| ^ { 2 } \right] } \\ & { \qquad + \displaystyle \sum _ { k = 1 } ^ { N } \left( \alpha _ { k - 1 } \left( 1 - \frac { d } { n - 1 } \right) \eta _ { k - 1 } + \frac { \alpha _ { k - 1 } ^ { 2 } L } { 2 } \right) \sigma _ { m } ^ { 2 } . } \end{array}
$$

Rewriting the inequality, we get

$$
\begin{array} { r l } {  { \mathbb { E } [ \| \nabla f ( x _ { \tau } ) \| ^ { 2 } ] \sum _ { k = 1 } ^ { N } ( \alpha _ { k - 1 } ( 1 - \frac { d } { n - 1 } ) \delta + \alpha _ { k - 1 } \frac { d } { n - 1 } - \frac { \alpha _ { k - 1 } ^ { 2 } L } { 2 } ) } } \\ & { \leq \displaystyle \sum _ { k = 1 } ^ { N } ( \alpha _ { k - 1 } ( 1 - \frac { d } { n - 1 } ) \eta _ { k - 1 } + \frac { \alpha _ { k - 1 } ^ { 2 } L } { 2 } ) \sigma _ { m } ^ { 2 } + ( f ( x _ { 0 } ) - f * ) , } \end{array}
$$

where τ = arg min $\phantom { } _ { k } \mathbb { E } [ \| \nabla f ( x _ { k } ) \| ]$ is the index that gives the minimum gradient. All that remains is to check the order of $\begin{array} { r } { \sum _ { k = 1 } ^ { N } \alpha _ { k - 1 } \eta _ { k - 1 } } \end{array}$ Similar to the lower bound for $r _ { s }$ which we obtained previously, we can use the fact that $x ^ { \beta }$ is an increasing function to obtain an upper bound for $\boldsymbol { r } _ { s } ,$ giving us

$$
r _ { s } \leq \frac { ( s + 1 ) ^ { \beta + 1 } - 1 } { \beta + 1 } .\tag{83}
$$

Setting the upper-bound to be equal to $N ,$ we find that $s$ is equal to

$$
{ \frac { ( s + 1 ) ^ { \beta + 1 } - 1 } { \beta + 1 } } = N \Longrightarrow s = [ ( \beta + 1 ) N + 1 ] ^ { \frac { 1 } { \beta + 1 } } - 1 .\tag{84}
$$

For $\eta _ { k }$ where $r _ { i } < k < r _ { i + 1 }$ , we have from Equation (74)

$$
\eta _ { k } \leq 3 \alpha _ { k - 1 } L + \eta _ { k - 1 } ,
$$

which gives us the upper bound

$$
\eta _ { k } \leq 3 L \sum _ { j = r _ { i } } ^ { k - 1 } \alpha _ { j } + \eta _ { r _ { i } } .
$$

Since we have $\eta _ { r _ { i } } = 0$ from Equation (76), we find that $\begin{array} { r } { \eta _ { k } \leq 3 \sum _ { j = r _ { i } } ^ { k - 1 } ( j + 1 ) ^ { - 1 / 2 } \leq 6 ( \sqrt { k } - \sqrt { r _ { i } + 1 } ) + } \end{array}$ $3 r _ { i } ^ { - 1 / 2 }$ . Consequently, we have

$$
\sum _ { k = r _ { i } } ^ { r _ { i + 1 } } \eta _ { k } \leq 6 \left[ \sum _ { k = r _ { i } + 1 } ^ { r _ { i + 1 } } ( \sqrt { k } - \sqrt { r _ { i } + 1 } ) \right] + 3 \left( \frac { r _ { i + 1 } - r _ { i } } { \sqrt { r _ { i } } } \right) .
$$

For the first sum, we have

$$
\sum _ { k = r _ { i } + 1 } ^ { r _ { i + 1 } } \left( \sqrt { k } - \sqrt { r _ { i } + 1 } \right) \leq \left( r _ { i + 1 } - r _ { i } \right) \left( \sqrt { r _ { i + 1 } } - \sqrt { r _ { i } } \right) = \left( r _ { i + 1 } - r _ { i } \right) \left( \sqrt { r _ { i } + \epsilon _ { i } } - \sqrt { r _ { i } } \right) .
$$

Using the same approximation for the second component as before, $\begin{array} { r } { \sqrt { r _ { i } + \epsilon _ { i } } - \sqrt { r _ { i } } = \sqrt { r _ { i } } \left( \frac { \epsilon _ { i } } { 2 r _ { i } } + o \left( \frac { \epsilon _ { i } } { r _ { i } } \right) \right) \le } \end{array}$ $\frac { \epsilon _ { i } } { \sqrt { r _ { i } } }$ for suficiently large $i ,$ we have

$$
\sum _ { k = r _ { i } } ^ { r _ { i + 1 } } \eta _ { k } \leq 6 \sum _ { k = r _ { i } + 1 } ^ { r _ { i + 1 } } \left( \sqrt { k - 1 } - \sqrt { r _ { i } } \right) + 3 \frac { r _ { i + 1 } - r _ { i } } { \sqrt { r _ { i } } } \leq \frac { 6 \epsilon _ { i } ^ { 2 } + 3 \epsilon _ { i } } { \sqrt { r _ { i } } } .
$$

Consequently, the variance contribution is approximately

$$
\sum _ { k = k _ { 0 } } ^ { N } \eta _ { k } \alpha _ { k } = \sum _ { i = i ^ { * } } ^ { s } \sum _ { j = r _ { i } } ^ { r _ { i + 1 } - 1 } \alpha _ { j } \eta _ { j } \le \sum _ { i = i ^ { * } } ^ { s } \alpha _ { r _ { i } } \sum _ { j = r _ { i } } ^ { r _ { i + 1 } - 1 } \eta _ { j } \le \sum _ { i = i ^ { * } } ^ { s } \left( \frac { 1 } { L \sqrt { r _ { i } + 1 } } \right) \frac { 6 \epsilon _ { i } ^ { 2 } + 3 \epsilon _ { i } } { \sqrt { r _ { i } } } \le \frac { 1 } { L } \sum _ { i = i ^ { * } } ^ { s } \frac { 6 \epsilon _ { i } ^ { 2 } + 3 \epsilon _ { i } } { r _ { i } } .\tag{85}
$$

Once again, using the fact that $\epsilon _ { i } = i ^ { \beta }$ and a lower bound $\begin{array} { r } { r _ { i } \ge \frac { i ^ { \beta + 1 } } { \beta + 1 } } \end{array}$ , the above has an upper bound of

$$
\sum _ { k = k _ { 0 } } ^ { N } \eta _ { k } \alpha _ { k } \leq \frac { \beta + 1 } { L } \sum _ { i = 1 } ^ { s } \frac { 6 i ^ { 2 \beta } + 3 i ^ { \beta } } { i ^ { \beta + 1 } } = \frac { \beta + 1 } { L } \sum _ { i = i ^ { * } } ^ { s } \left( 6 i ^ { \beta - 1 } + 3 i ^ { - 1 } \right) .
$$

The individual sums in the equation can be similarly bounded by integrals, giving us

$$
\sum _ { i = i ^ { * } } ^ { s } i ^ { \beta - 1 } \leq \frac { s ^ { \beta } - ( i ^ { * } - 1 ) ^ { \beta } } { \beta }
$$

$$
\sum _ { i = i ^ { * } } ^ { s } i ^ { - 1 } \leq \ln ( s ) - \ln ( i ^ { * } - 1 ) .
$$

Following that, we have

$$
\begin{array} { r l } { \displaystyle \sum _ { k = 1 } ^ { N } \alpha _ { k } \eta _ { k } \le \frac { \beta + 1 } { L } \left( \frac { 6 } { \beta } \left( s ^ { \beta } - ( i ^ { * } - 1 ) ^ { \beta } \right) + 3 \ln \frac { s } { i ^ { * } - 1 } \right) } & { } \\ { \displaystyle \lesssim \frac { \beta + 1 } { L } \left( \frac { 6 } { \beta } ( \beta + 1 ) ^ { \frac { \beta } { 1 + \beta } } \left( N ^ { \frac { \beta } { 1 + \beta } } - k _ { 0 } ^ { \frac { \beta } { 1 + \beta } } \right) + \frac { 3 } { \beta + 1 } \ln \left( \frac { N } { k _ { 0 } } \right) \right) } & { } \\ { \displaystyle } & { = \frac { 1 } { L } \left( 6 \beta ^ { - 1 } ( \beta + 1 ) ^ { \frac { \beta - 1 } { 1 + \beta } } \left( N ^ { \frac { \beta } { 1 + \beta } } - k _ { 0 } ^ { \frac { \beta } { 1 + \beta } } \right) + 3 \ln ( ( \beta + 1 ) N / k _ { 0 } ) \right) . } \end{array}
$$

where we used the approximation $s \approx [ ( \beta + 1 ) N ] ^ { \frac { 1 } { \beta + 1 } }$ derived from Equation (84). For the factor with the gradient norm, we have

$$
\begin{array} { l } { \displaystyle \sum _ { k = k _ { 0 } } ^ { N } \left( \alpha _ { k - 1 } \left( 1 - \frac { d } { n - 1 } \right) \delta + \alpha _ { k - 1 } \frac { d } { n - 1 } - \frac { \alpha _ { k - 1 } ^ { 2 } L } { 2 } \right) = \displaystyle \sum _ { k = k _ { 0 } } ^ { N } \frac { \left( 1 - \frac { d } { n - 1 } \right) \delta + \frac { d } { n - 1 } - \frac { 1 } { 2 \sqrt { k } } } { L \sqrt { k } } } \\ { \displaystyle \approx \frac { 1 } { L } ( \sqrt { N } - \sqrt { k _ { 0 } } ) . } \end{array}
$$

This means we get

$$
\begin{array} { r } { \mathbb { E } \left[ \| \nabla f ( x _ { \tau } ) \| ^ { 2 } \right] ( \sqrt { N } - \sqrt { k _ { 0 } } ) \lesssim \beta ^ { - 1 } ( \beta + 1 ) ^ { \frac { \beta - 1 } { 1 + \beta } } \left( N ^ { \frac { \beta } { 1 + \beta } } - k _ { 0 } ^ { \frac { \beta } { 1 + \beta } } \right) \sigma _ { m } ^ { 2 } + ( f ( x _ { 0 } ) - f * ) } \\ { \mathbb { E } \left[ \| \nabla f ( x _ { \tau } ) \| ^ { 2 } \right] \lesssim \frac { \beta ^ { - 1 } ( \beta + 1 ) ^ { \frac { \beta - 1 } { 1 + \beta } } \left( N ^ { \frac { \beta } { 1 + \beta } } - k _ { 0 } ^ { \frac { \beta } { 1 + \beta } } \right) \sigma _ { m } ^ { 2 } + \left( f ( x _ { 0 } ) - f * \right) } { \sqrt { N } - \sqrt { k _ { 0 } } } , } \end{array}
$$

where the leading term is $\frac { N ^ { \frac { \beta } { 1 + \beta } } - k _ { 0 } ^ { \frac { \beta } { 1 + \beta } } } { \sqrt { N } - \sqrt { k _ { 0 } } }$ . It sufices that this term is $\leq \epsilon ^ { 2 }$ to ensure that the LHS is $\leq \epsilon ^ { 2 }$ , giving us the condition

$$
\frac { N ^ { \frac { \beta } { 1 + \beta } } - k _ { 0 } ^ { \frac { \beta } { 1 + \beta } } } { \sqrt { N } - \sqrt { k _ { 0 } } } \le \epsilon ^ { 2 } \iff \epsilon ^ { - 2 } \left( 1 - \left( \frac { k _ { 0 } } { N } \right) ^ { \frac { \beta } { 1 + \beta } } \right) + \frac { \sqrt { k _ { 0 } } } { N ^ { \frac { \beta } { 1 + \beta } } } \le N ^ { \frac { 1 - \beta } { 2 ( 1 + \beta ) } } ,\tag{86}
$$

which simplifies to

$$
N \ge \left[ \epsilon ^ { - 2 } \left( 1 - \left( \frac { k _ { 0 } } { N } \right) ^ { \frac { \beta } { 1 + \beta } } \right) + \frac { \sqrt { k _ { 0 } } } { N ^ { \frac { \beta } { 1 + \beta } } } \right] ^ { \frac { 2 ( 1 + \beta ) } { 1 - \beta } } \gtrsim O \left( \epsilon ^ { - 4 \frac { 1 + \beta } { 1 - \beta } } \right) ,\tag{87}
$$

where we hide some constant and smaller order terms in $\gtrsim$ notation.

## D.3 Proof for Computational Cost in Minibatch Setting (Theorem 4)

From Equation (84), we have the number of times we need to update the alignment vector to be $s = N ^ { \frac { 1 } { \beta + 1 } }$ for a choice of $\beta \in ( 0 , 1 )$ . Suppose the cost of computing an alignment vector is $O ( \nu )$ and the cost of a directional derivative is $O ( \xi )$ . During the update step, the cost is $O ( \nu + n d )$ while during a normal iteration it is $O ( n d + m d \xi )$ . This gives us

$$
\begin{array} { r l } & { N \times ( n d + m d \xi ) + s \times \nu = N \times ( n d + m d \xi ) + N ^ { \frac { 1 } { \beta + 1 } } \times \nu } \\ & { \qquad = N ^ { \frac { 1 } { \beta + 1 } } \left( N ^ { \frac { \beta } { \beta + 1 } } ( n d + m d \xi ) + \nu \right) } \\ & { \qquad = O \left( \epsilon ^ { \frac { - 4 } { 1 - \beta } } \left( \epsilon ^ { \frac { - 4 \beta } { 1 - \beta } } ( n d + m d \xi ) + \nu \right) \right) . } \end{array}
$$

## E Proof for the Local Regime (Theorem 5)

This section is dedicated to show, in detail, the analysis for the persistence of memory in the local regime. We will start of with defining the local regime setup, then show some lemmas which hold in the local regime. Finally, the 2-stage proof will be presented in the last part.

## E.1 Setting and Algorithm

Let $f : \mathbb { R } ^ { n }  \mathbb { R }$ be $C ^ { 3 }$ and let $x ^ { * }$ be its unique minimizer. Define

$$
H = \nabla ^ { 2 } f ( x ^ { * } ) , \qquad \nabla f ( x ^ { * } ) = 0 .\tag{88}
$$

By assumption of L-Lipschtiz and µ-strongly convex, we have $\forall x \in \mathbb { R } ^ { n }$ 2

$$
\mu I \preceq \nabla ^ { 2 } f ( x ) \preceq L I .\tag{89}
$$

We also assume that the Hessian is locally Lipschitz near the unique optimal minimiser.

Assumption 3 (Locally Lipschitz Hessian). Assume $\exists L _ { H } , r _ { 0 } > 0$ such that for all $x , y$ with $\lVert x -$ $x ^ { * } \| , \| y - x ^ { * } \| \leq r _ { 0 }$ 2

$$
\| \nabla ^ { 2 } f ( x ) - \nabla ^ { 2 } f ( y ) \| \leq L _ { H } \| x - y \| .\tag{90}
$$

We define the notations below.

$$
\begin{array} { r } { e _ { k } : = x _ { k } - x ^ { * } , \qquad g _ { k } : = \nabla f ( x _ { k } ) , \qquad H : = \nabla ^ { 2 } f ( x ^ { * } ) . } \end{array}
$$

Assume H has distinct eigenvalues

$$
0 < \lambda _ { 1 } < \lambda _ { 2 } < \cdots < \lambda _ { n } ,
$$

with orthonormal eigenvectors $( u _ { i } ) _ { i = 1 } ^ { n }$ , so $H u _ { i } = \lambda _ { i } u _ { i }$ . Define

$$
M : = I - \alpha H , \mu _ { i } : = 1 - \alpha \lambda _ { i } ,
$$

so that

$$
0 < \mu _ { n } < \cdot \cdot \cdot < \mu _ { 2 } < \mu _ { 1 } < 1 .\tag{91}
$$

Define the projection error

$$
\begin{array} { r } { E _ { k } : = ( I - P _ { k } P _ { k } ^ { \top } ) g _ { k } . } \end{array}\tag{92}
$$

This term will determine how far our iterates diverge from the standard gradient descent. The main bulk of the proof is dedicated to controlling the deviations induced by this term.

For a fixed tolerance $\varepsilon > 0$ , let

$$
K _ { \varepsilon } : = \operatorname* { m i n } \{ k \geq 0 : \| g _ { k } \| \leq \varepsilon \} .
$$

All the events below will be enforced only for $k < K _ { \varepsilon }$ ; this allows a union bound over finitely many steps.

## E.2 Build-up to Main Theorem

We will introduce various lemmas that allow us to control the errors of the algorithm through Taylor’s Expansion. We first show that when suficiently close to the optimal point, the gradient is close to $H e ,$ where $e = x - x ^ { * }$

Lemma 4 (Local gradient expansion). There exist constants $C _ { 1 } ~ > ~ 0$ such that for all e with $\| e \| \le r _ { 0 }$

$$
\nabla f ( x ^ { * } + e ) = H e + r ( e ) ,\tag{93}
$$

where the remainder satisfies

$$
\| r ( e ) \| \leq C _ { 1 } \| e \| ^ { 2 } .\tag{94}
$$

One may take $C _ { 1 } = L _ { H } / 2$

Proof. For e such that $\| e \| < r _ { 0 }$ , the fundamental theorem of calculus gives

$$
\nabla f ( x ^ { * } + e ) - \nabla f ( x ^ { * } ) = \int _ { 0 } ^ { 1 } \nabla ^ { 2 } f ( x ^ { * } + t e ) e d t .
$$

Since $\nabla f ( x ^ { * } ) = 0$ , add and subtract He on both sides to get

$$
\nabla f ( x ^ { * } + e ) = H e + \int _ { 0 } ^ { 1 } \left( \nabla ^ { 2 } f ( x ^ { * } + t e ) - H \right) e d t .
$$

Define $r ( e )$ as the integral term. By (90), we have

$$
\begin{array} { l } { \displaystyle \| r ( e ) \| \leq \int _ { 0 } ^ { 1 } \| \nabla ^ { 2 } f ( x ^ { * } + t e ) - H \| \| e \| d t } \\ { \displaystyle \leq \int _ { 0 } ^ { 1 } L _ { H } t \| e \| \| e \| d t } \\ { \displaystyle = \frac { L _ { H } } { 2 } \| e \| ^ { 2 } . } \end{array}
$$

The lemma above allows us to characterise the relationship between consecutive gradients in terms of an iteration with some additional error terms.

Denote

$$
g _ { k } = \nabla f ( x _ { k } ) ,
$$

then the follow gives us the main recursion satisfied by subsequent gradients.

Lemma 5 (Modified gradient recursion). Assume $\| e _ { k } \| \le r _ { 0 }$ and $\| e _ { k + 1 } \| \le r _ { 0 }$ . Then there exists $C _ { 2 } > 0$ such that

$$
g _ { k + 1 } = M g _ { k } + \alpha H E _ { k } + r _ { k } ,\tag{95}
$$

where

$$
r _ { k } = r ( e _ { k + 1 } ) - r ( e _ { k } )\tag{96}
$$

satisfies $\| r _ { k } \| \le C _ { 2 } \| g _ { k } \| ^ { 2 }$ . One valid choice for $C _ { 2 } = 5 C _ { 1 } / \mu ^ { 2 }$

Proof. By Lemma 4, at $x _ { k } = x ^ { * } + e _ { k }$ , we find that

$$
g _ { k } = H e _ { k } + r ( e _ { k } ) .
$$

From the update (5) and $E _ { k } = ( I - P _ { k } P _ { k } ^ { \top } ) g _ { k }$ ，

$$
\begin{array} { r l } & { e _ { k + 1 } = e _ { k } + ( x _ { k + 1 } - x _ { k } ) } \\ & { \qquad = e _ { k } - \alpha P _ { k } P _ { k } ^ { \top } g _ { k } } \\ & { \qquad = e _ { k } - \alpha g _ { k } + \alpha E _ { k } . } \end{array}
$$

Substituting the first equation into the above equation, we get

$$
\boldsymbol { e } _ { k + 1 } = ( \boldsymbol { I } - \alpha \boldsymbol { H } ) \boldsymbol { e } _ { k } - \alpha \boldsymbol { r } ( \boldsymbol { e } _ { k } ) + \alpha \boldsymbol { E } _ { k } .\tag{97}
$$

Apply Lemma 4 at $x _ { k + 1 } = x ^ { * } + e _ { k + 1 }$

$$
g _ { k + 1 } = H e _ { k + 1 } + r ( e _ { k + 1 } ) .
$$

Substitute (97) into the RHS to get

$$
g _ { k + 1 } = H ( I - \alpha H ) e _ { k } - \alpha H r ( e _ { k } ) + \alpha H E _ { k } + r ( e _ { k + 1 } ) .\tag{98}
$$

On the other hand,

$$
\begin{array} { r l } & { M g _ { k } = ( I - \alpha H ) ( H e _ { k } + r ( e _ { k } ) ) } \\ & { \qquad = ( I - \alpha H ) H e _ { k } + ( I - \alpha H ) r ( e _ { k } ) } \\ & { \qquad = H ( I - \alpha H ) e _ { k } + ( I - \alpha H ) r ( e _ { k } ) , } \end{array}
$$

since $H ( I - \alpha H ) = ( I - \alpha H ) H$ . Then, combining both equations, we get

$$
\begin{array} { c } { g _ { k + 1 } = \left( M g _ { k } - r ( e _ { k } ) \right) + \alpha H E _ { k } + r ( e _ { k + 1 } ) } \\ { = M g _ { k } + \alpha H E _ { k } + r _ { k } . } \end{array}
$$

For the size of $r _ { k }$ , by Lemma 4 and the triangle inequality,

$$
\| r _ { k } \| \leq \| r ( e _ { k + 1 } ) \| + \| r ( e _ { k } ) \| \leq C _ { 1 } \| e _ { k + 1 } \| ^ { 2 } + C _ { 1 } \| e _ { k } \| ^ { 2 } .
$$

From the update $e _ { k + 1 } = e _ { k } - \alpha P _ { k } P _ { k } ^ { \top } g _ { k }$ and $\| P _ { k } P _ { k } ^ { \top } \| _ { \mathrm { o p } } = 1$

$$
\lVert e _ { k + 1 } \rVert \leq \lVert e _ { k } \rVert + \alpha \lVert g _ { k } \rVert .
$$

By strong convexity and $\nabla f ( x ^ { * } ) = 0$ , we have $\mu \| e _ { k } \| \leq \| g _ { k } \|$ , hence

$$
\| e _ { k } \| \leq { \frac { 1 } { \mu } } \| g _ { k } \| .
$$

Also $\| g _ { k } \| \le L \| e _ { k } \|$ by Lipschitz continuity of the gradient, so we get $\| e _ { k + 1 } \| \le ( 1 + \alpha L ) \| e _ { k } \| \le 2 \| e _ { k } \|$ as $\alpha < 1 / L$ . Therefore

$$
\| r _ { k } \| \leq C _ { 1 } ( 4 \| e _ { k } \| ^ { 2 } + \| e _ { k } \| ^ { 2 } ) = 5 C _ { 1 } \| e _ { k } \| ^ { 2 } \leq \frac { 5 C _ { 1 } } { \mu ^ { 2 } } \| g _ { k } \| ^ { 2 } .
$$

Before we begin, we will first pay of the debt we had above by proving Proposition 1.

Proof for Proposition 1. Let

$$
\alpha = \langle u , v \rangle , \quad \beta = \langle v , w \rangle , \quad x = \langle u , w \rangle .
$$

Since $u , v , w$ are unit vectors, the Gram matrix

$$
G = { \binom { u ^ { \top } } { v ^ { \top } } } \left( u v w \right) = { \binom { 1 } { x } } \left( { \begin{array} { l l l } { \alpha } & { x } \\ { v ^ { \top } } \\ { w ^ { \top } } \end{array} } \right)
$$

is positive semidefinite. Hence det $\left( G \right) \geq 0$ , which gives

$$
1 + 2 \alpha \beta x - \alpha ^ { 2 } - \beta ^ { 2 } - x ^ { 2 } \ge 0 .
$$

Rewriting,

$$
x ^ { 2 } - 2 \alpha \beta x + ( \alpha ^ { 2 } + \beta ^ { 2 } - 1 ) \le 0 .
$$

Viewing this as a quadratic inequality in $x ,$ the roots are

$$
x = \alpha \beta \pm { \sqrt { ( 1 - \alpha ^ { 2 } ) ( 1 - \beta ^ { 2 } ) } } ,
$$

which means

$$
| x | \geq | \alpha \beta | - \sqrt { ( 1 - \alpha ^ { 2 } ) ( 1 - \beta ^ { 2 } ) } .
$$

Using $\alpha ^ { 2 } \geq \delta _ { 1 }$ and $\beta ^ { 2 } \geq \delta _ { 2 }$ (which minimizes the lower bound), we get

$$
| x | \geq \sqrt { \delta _ { 1 } \delta _ { 2 } } - \sqrt { ( 1 - \delta _ { 1 } ) ( 1 - \delta _ { 2 } ) } .
$$

If $\delta _ { 1 } + \delta _ { 2 } \geq 1$ , then

$$
\sqrt { \delta _ { 1 } \delta _ { 2 } } \geq \sqrt { ( 1 - \delta _ { 1 } ) ( 1 - \delta _ { 2 } ) } ,
$$

so the right-hand side is non-negative and we may square both sides to obtain the desired result.

Next, for the convergence to hold true, we require that the gradient decays exponentially, which we show below.

Lemma 6 (Exponential Decay of Gradient with High Probability). The gradient of the iterates decrease at a rate of

$$
\| g _ { k } \| \le C \rho _ { d e c a y } ^ { k }\tag{99}
$$

for some constant $C > 0$ with probability at least $1 - e ^ { - \frac { 1 } { 2 } \sigma ^ { 2 } k \left( 1 - p _ { d e c a y } ( \tau , \delta , d ) \right) }$ , where

$$
p _ { d e c a y } ( \tau , \delta , d ) = 2 \exp { ( - c d \tau ^ { 2 } ) } + p _ { v } ( \delta )\tag{100}
$$

$$
\rho _ { d e c a y } = \left[ 1 - 2 \mu \alpha \left( 1 - \frac { \alpha L } { 2 } \right) \left( \delta + \frac { d } { n - 1 } ( 1 + \tau ) \right) \right] ^ { ( 1 - \sigma ) \left( 1 - p _ { d e c a y } ( \tau , \delta , d ) \right) } ,\tag{101}
$$

for some $\sigma \in ( 0 , 1 )$ and $\alpha \in ( 0 , 2 / L )$

Proof. From the same workings for the convergence, we have that conditioned on the past,

$$
f ( x _ { k - 1 } ) - f ( x _ { k } ) \geq \alpha \left( 1 - { \frac { \alpha L } { 2 } } \right) \left. P _ { k - 1 } ^ { \top } g _ { k - 1 } \right. ^ { 2 } ,
$$

which implies that $f ( x _ { k - 1 } ) \leq f ( x )$ necessarily. From the same equation, we also get

$$
f ( x _ { k - 1 } ) - f ( x _ { k } ) \geq \alpha \left( 1 - \frac { \alpha L } { 2 } \right) \left( \delta + \frac { d } { n - 1 } ( 1 + \tau ) \right) \left. g _ { k - 1 } \right. ^ { 2 }
$$

with probability at least $1 - 2 \exp { ( - c d \tau ^ { 2 } ) } - p _ { v } ( \delta ) : = 1 - p _ { d e c a y } ( \tau , \delta , d )$ by Lemma 1. Let $Y _ { k - 1 }$ be the indicator event of the inequality above, then we have that $\mathbb { E } [ Y _ { k - 1 } ] \ge 1 - p _ { d e c a y } ( \tau , \delta , d )$ . Furthermore, since the step size is chosen such that the RHS is positive, we have

$$
Y _ { k - 1 } \| g _ { k - 1 } \| ^ { 2 } \leq \frac { f ( x _ { k - 1 } ) - f ( x _ { k } ) } { \alpha \left( 1 - \frac { \alpha L } { 2 } \right) \left( \delta + \frac { d } { n - 1 } ( 1 + \tau ) \right) } .
$$

Hence, we get

$$
\left( \operatorname* { m i n } _ { i \in [ 0 , k - 1 ] } \lVert g _ { i } \rVert ^ { 2 } \right) \frac { 1 } { k } \sum _ { i = 0 } ^ { k - 1 } Y _ { i } \leq \frac { f ( x _ { 0 } ) - f ( x _ { k } ) } { k \alpha \left( 1 - \frac { \alpha L } { 2 } \right) \left( \delta + \frac { d } { n - 1 } ( 1 + \tau ) \right) } .
$$

We have by a Chernof bound (see [50]), that for all $\sigma \in ( 0 , 1 )$

$$
\mathbb { P } \left[ \sum _ { i = 1 } ^ { k - 1 } Y _ { i } \ge ( 1 - \sigma ) ( 1 - p _ { d e c a y } ( \tau , \delta , d ) ) k \right] \ge 1 - \exp \left( - \frac { \sigma ^ { 2 } } { 2 } ( 1 - p _ { d e c a y } ( \tau , \delta , d ) ) k \right) .\tag{102}
$$

Substituting the PL-inequality, we obtain

$$
\begin{array} { r l } { f ( x _ { k } ) - f ( x ^ { * } ) = f ( x _ { k } ) - f ( x _ { k - 1 } ) + f ( x _ { k - 1 } ) - f ( x ^ { * } ) } \\ { \ } & { \leq - \alpha \left( 1 - \displaystyle \frac { \alpha L } { 2 } \right) \left( \delta + \displaystyle \frac { d } { n - 1 } ( 1 + \tau ) \right) \| g _ { k - 1 } \| ^ { 2 } Y _ { k - 1 } + \left( f ( x _ { k - 1 } ) - f ( x ^ { * } ) \right) } \\ { \ } & { \leq \left[ 1 - 2 \mu \alpha \left( 1 - \displaystyle \frac { \alpha L } { 2 } \right) \left( \delta + \displaystyle \frac { d } { n - 1 } ( 1 + \tau ) \right) Y _ { k - 1 } \right] \left( f ( x _ { k - 1 } ) - f ( x ^ { * } ) \right) } \\ { \ } & { = \left[ 1 - 2 \mu \alpha \left( 1 - \displaystyle \frac { \alpha L } { 2 } \right) \left( \delta + \displaystyle \frac { d } { n - 1 } ( 1 + \tau ) \right) \right] ^ { Y _ { k - 1 } } \left( f ( x _ { k - 1 } ) - f ( x ^ { * } ) \right) . } \end{array}
$$

Consequently, we obtain

$$
f ( x _ { k } ) - f ( x ^ { * } ) \leq \left[ 1 - 2 \mu \alpha \left( 1 - \frac { \alpha L } { 2 } \right) \left( \delta + \frac { d } { n - 1 } ( 1 + \tau ) \right) \right] ^ { \sum _ { i = 0 } ^ { k - 1 } Y _ { i } } ( f ( x _ { 0 } ) - f ( x ^ { * } ) ) .\tag{103}
$$

Using Equation (102), we have that

$$
f ( x _ { k } ) - f ( x ^ { * } ) \leq \left[ 1 - 2 \mu \alpha \left( 1 - \frac { \alpha L } { 2 } \right) \left( \delta + \frac { d } { n - 1 } ( 1 + \tau ) \right) \right] ^ { ( 1 - \sigma ) ( 1 - p _ { d e c a y } ( \tau , \delta , d ) ) k } ( f ( x _ { 0 } ) - f ( x ^ { * } ) ) .\tag{104}
$$

Let

$$
\rho = \left[ 1 - 2 \mu \alpha \left( 1 - \frac { \alpha L } { 2 } \right) \left( \delta + \frac { d } { n - 1 } ( 1 + \tau ) \right) \right] ^ { ( 1 - \sigma ) ( 1 - p _ { d e c a y } ( \tau , \delta , d ) ) } ,
$$

then we have

$$
f ( x _ { k } ) - f ( x ^ { * } ) \leq \rho ^ { k } ( f ( x _ { 0 } ) - f ( x ^ { * } ) ) .
$$

From L-smoothness of the function, we have

$$
\| g _ { k } \| ^ { 2 } \leq 2 L ( f ( x _ { k } ) - f ^ { * } ) \leq 2 L ( f ( x _ { 0 } ) - f ( x ^ { * } ) ) \rho ^ { k } ,
$$

giving the desired equation with constant

$$
C = { \sqrt { 2 L ( f ( x _ { 0 } ) - f ( x ^ { * } ) ) } }
$$

holding with probability at least $1 - e ^ { - \frac { 1 } { 2 } \sigma ^ { 2 } k \left( 1 - p _ { d e c a y } \left( \tau , \delta , d \right) \right) }$

Remark 9. For simplicity, we can let $\sigma = 1 / 2$ such that the result holds with

$$
\rho _ { d e c a y } = \left[ 1 - 2 \mu \alpha \left( 1 - \frac { \alpha L } { 2 } \right) \left( \delta + \frac { d } { n - 1 } ( 1 + \tau ) \right) \right] ^ { \frac { 1 } { 2 } \left( 1 - p _ { d e c a y } ( \tau , \delta , d ) \right) }
$$

with probability at least $1 - e ^ { - \frac { 1 } { 8 } k \left( 1 - p _ { d e c a y } ( \tau , \delta , d ) \right) }$

Rescaling Matrix for Phase 2. For phase 2 of the argument, after $k _ { 1 }$ has been fixed, we will re-scale the random part of the matrix by a factor of $\sqrt { \frac { n - 1 } { d } }$ . Under this setup, we will obtain a similar decay rate (with a slightly diferent step size) as in the original matrix. Consider the rescaled version of the random matrix $\tilde { P } _ { k }$ by a factor of $\sqrt { \frac { n - 1 } { d } }$ , which we denote by $\begin{array} { r } { \hat { P } _ { k } = \sqrt { \frac { n - 1 } { d } } } \end{array}$ . Then, it satisfies the basic properties

$$
\hat { P } _ { k } ^ { \top } \hat { P } _ { k } = \frac { n - 1 } { d } I , \qquad \mathbb { E } \left[ \hat { P } _ { k } \hat { P } _ { k } ^ { \top } \right] = I - \hat { v } _ { k } \hat { v } _ { k } ^ { \top } .\tag{105}
$$

Then, the matrix $P _ { k } = \left( \hat { v } _ { k } \quad \hat { P } _ { k } \right)$ has the properties

$$
P _ { k } ^ { \top } P _ { k } = \left( { 1 \atop 0 } \quad { \frac { 0 } { \frac { n - 1 } { d } } } I _ { d } \right) , \qquad \mathbb { E } [ P _ { k } P _ { k } ^ { \top } ] = I _ { n } .\tag{106}
$$

Corollary 2. As a corollary of Lemma (1), we have that

$$
\begin{array} { r } { \mathbb { P } \left[ \left| \left. \hat { P } ^ { \top } u , \hat { P } ^ { \top } v \right. - \langle u _ { \bot } , v _ { \bot } \rangle \right| \leq \tau \| u _ { \bot } \| \| v _ { \bot } \| \right] \geq 1 - 2 \exp \left( - c d \tau ^ { 2 } \right) . } \end{array}\tag{107}
$$

Proof. Notice that

$$
\begin{array} { r l } & { \Big | \Big \langle \hat { P } ^ { \top } u , \hat { P } ^ { \top } v \Big \rangle - \langle u _ { \bot } , v _ { \bot } \rangle \Big | = \displaystyle \frac { n - 1 } { d } \Big | \Big \langle \tilde { P } ^ { \top } u , \tilde { P } ^ { \top } v \Big \rangle - \frac { d } { n - 1 } \langle u _ { \bot } , v _ { \bot } \rangle \Big | } \\ & { \quad \quad \leq \displaystyle \frac { n - 1 } { d } \left( \tau \frac { d } { n - 1 } \| u _ { \bot } \| \| v _ { \bot } \| \right) } \\ & { \quad \quad = \tau \| u _ { \bot } \| \| v _ { \bot } \| . } \end{array}
$$

□

Lemma 7 (Decay Rate Under Rescaled $P _ { k } )$ . Suppose $P _ { k } = \left( \hat { v } _ { k } \quad \hat { P } _ { k } \right)$ , we have

$$
f ( x _ { k + 1 } ) - f ( x _ { k } ) \leq \left[ - \alpha \tau + \frac { \alpha ^ { 2 } L } { 2 } \left( 1 - \frac { n - 1 } { d } ( 1 + \tau ) \right) \right] \langle \hat { v } _ { k } , g _ { k } \rangle ^ { 2 } + \left[ - \alpha ( 1 - \tau ) + \frac { \alpha ^ { 2 } L } { 2 } \frac { n - 1 } { d } \right] \| g _ { k } \| ^ { 2 }
$$

with probability at least $1 - 2 e ^ { - c d \tau ^ { 2 } }$ . Consequently, if $\left. \hat { v } _ { k } , g _ { k } \right. ^ { 2 } \geq \delta \Vert g _ { k } \Vert ^ { 2 }$ , we have

$$
f ( x _ { k + 1 } ) - f ( x _ { k } ) \leq \left( \left[ - \alpha \tau + \frac { \alpha ^ { 2 } L } { 2 } \left( 1 - \frac { n - 1 } { d } ( 1 + \tau ) \right) \right] \delta + \left[ - \alpha ( 1 - \tau ) + \frac { \alpha ^ { 2 } L } { 2 } \frac { n - 1 } { d } \right] \right) \| g _ { k } \| ^ { 2 } .\tag{108}
$$

Proof. By L-smoothness, we have

$$
f ( x _ { k + 1 } ) - f ( x _ { k } ) \leq - \alpha \bigg \| P _ { k } ^ { \top } g _ { k } \bigg \| ^ { 2 } + \frac { \alpha ^ { 2 } L } { 2 } \bigg \| P _ { k } P _ { k } ^ { \top } g _ { k } \bigg \| ^ { 2 } .\tag{109}
$$

Using Corollary 2, we have

$$
\begin{array} { r l } & { \left| \left| P _ { k } ^ { \top } g _ { k } \right| \right| ^ { 2 } = \langle \hat { v } _ { k } , g _ { k } \rangle ^ { 2 } + \left\| \tilde { P } _ { k } ^ { \top } g _ { k } \right\| ^ { 2 } } \\ & { \qquad \geq \langle \hat { v } _ { k } , g _ { k } \rangle ^ { 2 } + ( 1 - \tau ) \Big \| \mathrm { P r o j } _ { \hat { v } _ { k } ^ { \perp } } g _ { k } \Big \| ^ { 2 } } \\ & { \qquad = \langle \hat { v } _ { k } , g _ { k } \rangle ^ { 2 } + ( 1 - \tau ) \left( \| g _ { k } \| ^ { 2 } - \langle \hat { v } _ { k } , g _ { k } \rangle ^ { 2 } \right) } \\ & { \qquad = \tau \langle \hat { v } _ { k } , g _ { k } \rangle ^ { 2 } + ( 1 - \tau ) \| g _ { k } \| ^ { 2 } . } \end{array}
$$

with probability $1 - 2 \exp { ( - c d \tau ^ { 2 } ) }$ . For the second term, we find that

$$
\begin{array} { r l } { g _ { k } ^ { \top } P _ { k } P _ { k } ^ { \top } P _ { k } P _ { k } ^ { \top } g _ { k } = g _ { k } ^ { \top } \left( \hat { v } _ { k } \hat { v } _ { k } ^ { \top } + \frac { n - 1 } { d } \tilde { P } _ { k } \tilde { P } _ { k } ^ { \top } \right) g _ { k } } & { } \\ { } & { ~ = \left. \hat { v } _ { k } , g _ { k } \right. ^ { 2 } + \frac { n - 1 } { d } \left\| \tilde { P } _ { k } ^ { \top } g _ { k } \right\| ^ { 2 } } \\ { } & { ~ \leq \left. \hat { v } _ { k } , g _ { k } \right. ^ { 2 } + \frac { n - 1 } { d } ( 1 + \tau ) \left\| \mathrm { P r o j } _ { \hat { v } _ { k } ^ { 1 } } g _ { k } \right\| ^ { 2 } } \\ { } & { ~ = \left. \hat { v } _ { k } , g _ { k } \right. ^ { 2 } + \frac { n - 1 } { d } ( 1 + \tau ) \left( \left\| g _ { k } \right\| ^ { 2 } - \left. \hat { v } _ { k } , g _ { k } \right. ^ { 2 } \right) } \\ { } & { ~ = \frac { n - 1 } { d } \| g _ { k } \| ^ { 2 } + \left( 1 - \frac { n - 1 } { d } ( 1 + \tau ) \right) \left. \hat { v } _ { k } , g _ { k } \right. ^ { 2 } } \end{array}
$$

under the same event. Substituting both back into the equation above, we obtain

$$
\begin{array} { r l } { f ( x _ { k + 1 } ) - f ( x _ { k } ) \le - \alpha \tau \langle \hat { v } _ { k } , g _ { k } \rangle ^ { 2 } - \alpha ( 1 - \tau ) \| g _ { k } \| ^ { 2 } } & { } \\ { + \displaystyle \frac { \alpha ^ { 2 } L } { 2 } \left( \frac { n - 1 } { d } \| g _ { k } \| ^ { 2 } + \left( 1 - \frac { n - 1 } { d } ( 1 + \tau ) \right) \langle \hat { v } _ { k } , g _ { k } \rangle ^ { 2 } \right) } & { } \\ { = \displaystyle \left[ - \alpha \tau + \frac { \alpha ^ { 2 } L } { 2 } \left( 1 - \frac { n - 1 } { d } ( 1 + \tau ) \right) \right] \langle \hat { v } _ { k } , g _ { k } \rangle ^ { 2 } } & { } \\ { + \left[ - \alpha ( 1 - \tau ) + \frac { \alpha ^ { 2 } L } { 2 } \frac { n - 1 } { d } \right] \| g _ { k } \| ^ { 2 } . } & { } \end{array}\tag{110}
$$

Remark 10. From Equations (104) and (110), it is clear that with suitable step sizes in either phases, $f ( x _ { k + 1 } ) \leq f ( x _ { k } )$ . (The step sizes will be chosen to satisfy this, as is the case for all gradient-descent based methods.) This means that $f ( x _ { k } ) \leq f ( x _ { 0 } )$ , and by strong-convexity,

$$
\frac { \mu } { 2 } \| x _ { k } - x ^ { * } \| \leq f ( x _ { k } ) - f ^ { * } \leq f ( x _ { 0 } ) - f ^ { * } \leq \frac { \mu r _ { 0 } ^ { 2 } } { 2 } .\tag{111}
$$

So, $\left\| \boldsymbol { e } _ { k } \right\| = \left\| \boldsymbol { x } _ { k } - \boldsymbol { x } ^ { * } \right\| \leq r _ { 0 }$ and the taylor approximations hold for all iterates.

The main idea of the proof is that the iteration is a perturbed contraction, meaning that the main part contracts with some additional error added to it. This means that the iterates converge to a small neighbourhood around 0 of size determined by the perturbation. By ensuring the perturbation is small at the start, we can ensure that the iterates converge to a very small value. The lemma below proves the convergence of the perturbed fixed point iteration.

Lemma 8 (Perturbed Fixed Point). Let $\mu _ { 2 } < \mu _ { 1 }$ and $\epsilon > 0$ be a small term. Then the iteration

$$
t _ { k + 1 } \leq \frac { \mu _ { 2 } } { \mu _ { 1 } } t _ { k } + \epsilon\tag{112}
$$

satisfies

$$
\operatorname* { l i m } \operatorname* { s u p } t _ { k } = O ( \epsilon ) .\tag{113}
$$

Proof. Consider the recurrence

$$
s _ { k + 1 } = \frac { \mu _ { 2 } } { \mu _ { 1 } } s _ { k } + \epsilon ,
$$

which has a solution

$$
s _ { k } = \left( \frac { \mu _ { 2 } } { \mu _ { 1 } } \right) ^ { k } t _ { 0 } + \epsilon \sum _ { i = 0 } ^ { k - 1 } \left( \frac { \mu _ { 2 } } { \mu _ { 1 } } \right) ^ { i } = \left( \frac { \mu _ { 2 } } { \mu _ { 1 } } \right) ^ { k } t _ { 0 } + \frac { \epsilon } { 1 - \frac { \mu _ { 2 } } { \mu _ { 1 } } } \left[ 1 - \left( \frac { \mu _ { 2 } } { \mu _ { 1 } } \right) ^ { k } \right] .
$$

Taking lim sup, we have

$$
\operatorname* { l i m } _ { k } \operatorname* { s u p } _ { s _ { k } } { s _ { k } } = \epsilon \frac { \mu _ { 1 } } { \mu _ { 1 } - \mu _ { 2 } } .
$$

Since $t _ { k } < s _ { k }$ by induction, we have lim $\operatorname* { s u p } _ { k } t _ { k } \leq c \epsilon$ , where $\begin{array} { r } { c = \frac { \mu _ { 1 } } { \mu _ { 1 } - \mu _ { 2 } } } \end{array}$

Lastly, the analysis will be done through a sequential conditioning argument. Each step will be independent of the past events and we will obtain a conditional recurrence. The lemma below will then allow us to apply this result to obtain a final probability of the events of interest.

Lemma 9. Consider the sequence of σ−algebras ${ \mathcal { F } } _ { 1 } ~ \subset ~ \cdots ~ \subset ~ { \mathcal { F } } _ { k }$ and the sequence of events $A _ { 1 } , \cdots , A _ { k }$ which are measurable in their respective σ-algebra. i.e. $\mathbb { E } [ 1 _ { A _ { i } } ~ \mid ~ \mathcal { F } _ { i } ] ~ = ~ \mathbb { P } [ A _ { i } ]$ . If ∀k, $\mathbb { P } [ A _ { k } \ | \ \mathcal { F } _ { k - 1 } ] \ge 1 - p _ { k }$ , then

$$
\mathbb { P } \left[ \bigcap _ { i = 1 } ^ { k } A _ { i } \right] \geq ( 1 - p _ { k } ) \times \mathbb { P } \left[ \bigcap _ { i = 1 } ^ { k - 1 } A _ { i } \right] \geq \prod _ { i = 1 } ^ { k } ( 1 - p _ { i } ) .\tag{114}
$$

Proof. Since $\mathbb { P } [ A ] = \mathbb { E } [ 1 _ { A } ]$ , we have

$$
\begin{array} { r } { \mathbb { E } \left[ \mathbf { 1 } _ { \bigcap _ { i = 1 } ^ { k } A _ { i } } \right] = \mathbb { E } \left[ \mathbb { E } \left[ \mathbf { 1 } _ { \bigcap _ { i = 1 } ^ { k - 1 } A _ { i } } \mathbf { 1 } _ { A _ { k } } ~ \Big | ~ \mathcal { F } _ { k - 1 } \right] \right] } \\ { = \mathbb { E } \left[ \mathbf { 1 } _ { \bigcap _ { i = 1 } ^ { k - 1 } A _ { i } } \mathbb { E } \left[ \mathbf { 1 } _ { A _ { k } } ~ \Big | ~ \mathcal { F } _ { k - 1 } \right] \right] } \\ { \geq ( 1 - p _ { k } ) \times \mathbb { E } \left[ \mathbf { 1 } _ { \bigcap _ { i = 1 } ^ { k - 1 } A _ { i } } \right] } \end{array}
$$

and the result follows.

## E.3 Analysis for Persistent Alignment

Proof of Theorem 5. The proof will mainly proceed in 2 steps.

1. We will first show that there is a $k _ { 1 } \in ( 0 , K _ { \epsilon } )$ such that $\langle g _ { k _ { 1 } } , u _ { 1 } \rangle ^ { 2 } \geq \delta _ { 0 } \| g _ { k _ { 1 } } \| ^ { 2 }$ . In other words, we find that eventually the gradient of the function will be close to the direction $u _ { 1 }$

2. In the next part, we will use the above result to show that even with a fixed $v _ { k } = v _ { k _ { 1 } }$ for all $k > k _ { 1 } , g _ { k }$ still remains closely correlated to the direction $u _ { 1 }$ . This is done by controlling the error $E _ { k }$ of successive gradient descent steps due to the random projections, by the closeness of the vectors between $u _ { 1 }$ and $v _ { k _ { 1 } } = g _ { k _ { 1 } }$

In the heart of this analysis, we will consider the events which we desire and obtain a one-step recursion and the probability of this one step, conditioned on the past. This allows us to only consider the randomness introduced in this step, which is simply $\tilde { P } _ { k }$

Phase 1: Gradient eventually enters into cone around $u _ { 1 }$ . Define the following variable

$$
t _ { k } = { \frac { \left\| \operatorname* { P r o j } _ { u _ { 1 } ^ { \perp } } g _ { k } \right\| } { \left\| \operatorname* { P r o j } _ { u _ { 1 } } g _ { k } \right\| } } .\tag{115}
$$

Notice that $t _ { k }$ is essentially the tangent of the angle between $g _ { k }$ and $u _ { 1 }$ . To find an upper bound of $t _ { k + 1 }$ , we can first find an upper bound for the numerator and a lower bound for the denominator.

Part 1a: Upper bound for Numerator. Using Lemma (5), we have

$$
\Big \| \mathrm { P r o j } _ { u _ { 1 } ^ { \bot } } g _ { k + 1 } \Big \| \leq \Big \| \mathrm { P r o j } _ { u _ { 1 } ^ { \bot } } M g _ { k } \Big \| + \alpha \Big \| \mathrm { P r o j } _ { u _ { 1 } ^ { \bot } } H E _ { k } \Big \| + \Big \| \mathrm { P r o j } _ { u _ { 1 } ^ { \bot } } r _ { k } \Big \| ,
$$

where H is the Hessian at the optimal point and $M = I - \alpha H$ . For the first term, we use the fact that M and $\mathrm { P r o j } _ { u _ { 1 } }$ commute to get

$$
\begin{array} { r } { \left\| \operatorname { P r o j } _ { u _ { 1 } ^ { \perp } } M g _ { k } \right\| = \left\| M \operatorname { P r o j } _ { u _ { 1 } ^ { \perp } } g _ { k } \right\| = \left\| M \operatorname { P r o j } _ { u _ { 1 } ^ { \perp } } ^ { 2 } g _ { k } \right\| \leq \left\| M \operatorname { P r o j } _ { u _ { 1 } ^ { \perp } } \left\| \left\| \operatorname { P r o j } _ { u _ { 1 } ^ { \perp } } g _ { k } \right\| = \mu _ { 2 } \right\| \operatorname { P r o j } _ { u _ { 1 } ^ { \perp } } g _ { k } \right\| . } \end{array}
$$

For the second term, we can simply bound it by

$$
\begin{array} { r } { \Big \| \mathrm { P r o j } _ { u _ { 1 } ^ { \perp } } H E _ { k } \Big \| \leq \lambda _ { n } \Big \| ( I - \tilde { P } _ { k } \tilde { P } _ { k } ^ { \top } ) \mathrm { P r o j } _ { \hat { v } _ { k } ^ { \perp } } g _ { k } \Big \| , } \end{array}\tag{116}
$$

where $\boldsymbol { E _ { k } } = ( I - \tilde { P } _ { k } \tilde { P } _ { k } ^ { \top } - \hat { v } _ { k } \hat { v } _ { k } ^ { \top } ) g _ { k } = ( I - \tilde { P } _ { k } \tilde { P } _ { k } ^ { \top } ) ( I - \hat { v } _ { k } \hat { v } _ { k } ^ { \top } ) g _ { k }$ . Using the fact that $I - \tilde { P } _ { k } \tilde { P } _ { k } ^ { \top }$ is a projection and the alignment assumption, we have

$$
\left\| \operatorname { P r o j } _ { \hat { v } _ { k } ^ { \perp } } g _ { k } \right\| = \sqrt { \left( I - \hat { v } _ { k } \hat { v } _ { k } ^ { \top } \right) g _ { k } } \leq \sqrt { 1 - \delta } \| g _ { k } \| .\tag{117}
$$

For the third term, we use Lemma (5) to get

$$
\left\| \operatorname { P r o j } _ { u _ { 1 } ^ { \perp } } r _ { k } \right\| \leq C _ { 2 } \| g _ { k } \| ^ { 2 } .
$$

Combining everything, the numerator can be bounded by

$$
\begin{array} { r } { \Big \| \mathrm { P r o j } _ { u _ { 1 } ^ { \perp } } g _ { k + 1 } \Big \| \leq \mu _ { 2 } \Big \| \mathrm { P r o j } _ { u _ { 1 } ^ { \perp } } g _ { k } \Big \| + \alpha \lambda _ { n } \sqrt { 1 - \delta } \| g _ { k } \| + C _ { 2 } \| g _ { k } \| ^ { 2 } . } \end{array}
$$

Part 1b: Lower bound for Denominator. For the denominator (the projection onto $u _ { 1 } )$ , we have

$$
\begin{array} { r l } & { \left\| \mathrm { P r o j } _ { u _ { 1 } } g _ { k + 1 } \right\| = \left\| \mathrm { P r o j } _ { u _ { 1 } } \left[ M g _ { k } + \alpha H E _ { k } + r _ { k } \right] \right\| } \\ & { \qquad = \Big | \mu _ { 1 } u _ { 1 } ^ { \top } g _ { k } + \alpha \lambda _ { 1 } u _ { 1 } ^ { \top } E _ { k } + u _ { 1 } ^ { \top } r _ { k } \Big | } \\ & { \qquad \geq \mu _ { 1 } \left\| \mathrm { P r o j } _ { u _ { 1 } } g _ { k } \right\| - \alpha \lambda _ { 1 } | u _ { 1 } ^ { \top } E _ { k } | - \| r _ { k } \| . } \end{array}
$$

The second and third term will use the same bounds as in the numerator, giving us

$$
\begin{array} { r } { \left\| \mathrm { P r o j } _ { u _ { 1 } } g _ { k + 1 } \right\| \geq \mu _ { 1 } \left\| \mathrm { P r o j } _ { u _ { 1 } } g _ { k } \right\| - \alpha \lambda _ { 1 } \sqrt { 1 - \delta } \| g _ { k } \| - C _ { 2 } \| g _ { k } \| ^ { 2 } . } \end{array}\tag{118}
$$

Part 1c: Combining both bounds. Taking both bounds, we obtain

$$
\begin{array} { r } { t _ { k + 1 } \leq \frac { \mu _ { 2 } \left\| \mathrm { P r o j } _ { u _ { 1 } ^ { \perp } } g _ { k } \right\| + \alpha \lambda _ { n } \sqrt { 1 - \delta } \| g _ { k } \| + C _ { 2 } \| g _ { k } \| ^ { 2 } } { \mu _ { 1 } \left\| \mathrm { P r o j } _ { u _ { 1 } } g _ { k } \right\| - \alpha \lambda _ { 1 } \sqrt { 1 - \delta } \| g _ { k } \| - C _ { 2 } \| g _ { k } \| ^ { 2 } } } \\ { = \frac { \mu _ { 2 } t _ { k } + \alpha \lambda _ { n } \sqrt { 1 - \delta } \frac { \| g _ { k } \| } { \left\| \mathrm { P r o j } _ { u _ { 1 } } g _ { k } \right\| } + C _ { 2 } \frac { \| g _ { k } \| ^ { 2 } } { \left\| \mathrm { P r o j } _ { u _ { 1 } } g _ { k } \right\| } } { \mu _ { 1 } - \alpha \lambda _ { 1 } \sqrt { 1 - \delta } \frac { \| g _ { k } \| } { \left\| \mathrm { P r o j } _ { u _ { 1 } } g _ { k } \right\| } - C _ { 2 } \frac { \| g _ { k } \| ^ { 2 } } { \left\| \mathrm { P r o j } _ { u _ { 1 } } g _ { k } \right\| } } . } \end{array}
$$

Define $\epsilon _ { \delta } = \sqrt { 1 - \delta }$ and $\begin{array} { r } { \epsilon _ { k } = \frac { \left\| g _ { k } \right\| } { \left\| \operatorname* { P r o j } _ { u _ { 1 } } g _ { k } \right\| } } \end{array}$ . We have by Lemma 6 that the gradient norm decays exponentially with factor $\rho _ { d e c a y }$ (omitting the variables for simplicity). Since we are in the local regime, we can assume that $\| g _ { k } \| \le \rho _ { d e c a y } ^ { k _ { 0 } + k }$ for some fixed $k _ { 0 }$ where the iteration of this convergence begins. Then, we can express the recursion of $t _ { k + 1 }$ as

$$
t _ { k + 1 } \leq \frac { \mu _ { 2 } t _ { k } + \alpha \lambda _ { n } \epsilon _ { \delta } \epsilon _ { k } + C _ { 2 } \epsilon _ { k } \rho _ { d e c a y } ^ { k _ { 0 } + k } } { \mu _ { 1 } - \alpha \lambda _ { 1 } \epsilon _ { \delta } \epsilon _ { k } - C _ { 2 } \epsilon _ { k } \rho _ { d e c a y } ^ { k + k _ { 0 } } } .\tag{119}
$$

Part 1d: Finding the Conditions For a Perturbed Fixed Point. Since $\epsilon _ { \delta }$ can be controlled arbitrarily small by having $\delta \gg 0$ and $\rho ^ { k _ { 0 } }$ is exponentially small, we can use Taylor’s expansion for $\begin{array} { r } { \frac { 1 } { 1 - x } = 1 + x ( 1 + o ( 1 ) ) } \end{array}$ to get

$$
\begin{array} { r l } & { t _ { k + 1 } \leq \frac { \mu _ { 2 } t _ { k } + \alpha \lambda _ { n } \epsilon _ { \delta } \epsilon _ { k } + C _ { 2 } \epsilon _ { k } \rho _ { d e c a y } ^ { k + k _ { 0 } } } { \mu _ { 1 } \left( 1 - \alpha \lambda _ { 1 } \mu _ { 1 } ^ { - 1 } \epsilon _ { \delta } \epsilon _ { k } - C _ { 2 } \mu _ { 1 } ^ { - 1 } \epsilon _ { k } \rho _ { d e c a y } ^ { k + k _ { 0 } } \right) } } \\ & { = \frac { \mu _ { 2 } t _ { k } + \alpha \lambda _ { n } \epsilon _ { \delta } \epsilon _ { k } + C _ { 2 } \epsilon _ { k } \rho _ { d e c a y } ^ { k + k _ { 0 } } } { \mu _ { 1 } } \left( 1 + \left( \alpha \lambda _ { 1 } \mu _ { 1 } ^ { - 1 } \epsilon _ { \delta } \epsilon _ { k } + C _ { 2 } \mu _ { 1 } ^ { - 1 } \epsilon _ { k } \rho _ { d e c a y } ^ { k + k _ { 0 } } \right) ( 1 + o ( 1 ) ) \right) } \\ & { = \frac { \tilde { \mu } _ { 2 } ^ { ( k ) } } { \mu _ { 1 } } t _ { k } + \tilde { \epsilon } _ { k } , } \end{array}
$$

where

$$
\begin{array} { l } { { \tilde { \mu } _ { 2 } ^ { ( k ) } = \mu _ { 2 } \left( 1 + \left( \alpha \lambda _ { 1 } \mu _ { 1 } ^ { - 1 } \epsilon _ { \delta } \epsilon _ { k } + C _ { 2 } \mu _ { 1 } ^ { - 1 } \epsilon _ { k } \rho _ { d e c a y } ^ { k + k _ { 0 } } \right) \left( 1 + o ( 1 ) \right) \right) } } \\ { { \tilde { \epsilon } _ { k } = \mu _ { 1 } ^ { - 1 } \left( \alpha \lambda _ { n } \epsilon _ { \delta } \epsilon _ { k } + C _ { 2 } \epsilon _ { k } \rho _ { d e c a y } ^ { k + k _ { 0 } } \right) \left( 1 + \left( \alpha \lambda _ { 1 } \mu _ { 1 } ^ { - 1 } \epsilon _ { \delta } \epsilon _ { k } + C _ { 2 } \mu _ { 1 } ^ { - 1 } \epsilon _ { k } \rho _ { d e c a y } ^ { k + k _ { 0 } } \right) \left( 1 + o ( 1 ) \right) \right) } } \end{array}
$$

By assumption, we have $| \langle u _ { 1 } , g _ { k _ { 0 } } \rangle | = \omega \| g _ { k _ { 0 } } \|$ for some $\omega > 0$ , which is equivalent to $\epsilon _ { 0 } = \omega ^ { - 1 }$ Consequently, we have $\left\| \operatorname* { P r o j } _ { u _ { 1 } ^ { \perp } } g _ { k } \right\| = \sqrt { 1 - \omega ^ { 2 } } \| g _ { k } \|$ . This tells us that

$$
t _ { 0 } = { \frac { \sqrt { 1 - \omega ^ { 2 } } } { \omega } } .\tag{120}
$$

Consider the following decomposition of expressing $\| g _ { k } \|$ in terms of $\left\| \operatorname { P r o j } _ { u _ { 1 } ^ { \perp } } g _ { k } \right\|$

$$
\left\| g _ { k } \right\| = { \sqrt { \left\| \operatorname* { P r o j } _ { u _ { 1 } } g _ { k } \right\| ^ { 2 } + \left\| \operatorname* { P r o j } _ { u _ { 1 } ^ { \perp } } g _ { k } \right\| ^ { 2 } } } = \left\| \operatorname* { P r o j } _ { u _ { 1 } } g _ { k } \right\| { \sqrt { 1 + t _ { k } ^ { 2 } } } .\tag{121}
$$

We find that

$$
\epsilon _ { k } = \sqrt { 1 + t _ { k } ^ { 2 } } \implies \epsilon _ { k + 1 } = \sqrt { 1 + t _ { k + 1 } ^ { 2 } } \leq \sqrt { 1 + \left( \frac { \tilde { \mu } _ { 2 } ^ { ( k ) } } { \mu _ { 1 } } t _ { k } + \tilde { \epsilon } _ { k } \right) ^ { 2 } } .\tag{122}
$$

Then, an upper bound for $\epsilon _ { k + 1 }$ yields

$$
\epsilon _ { k + 1 } \leq \sqrt { \frac { 1 + \biggl ( \frac { \tilde { \mu } _ { 2 } ^ { ( k ) } } { \mu _ { 1 } } t _ { k } + \tilde { \epsilon } _ { k } \biggr ) ^ { 2 } } { 1 + t _ { k } ^ { 2 } } } \epsilon _ { k } .
$$

To show that it does not grow, we just have to show that the factor in front of $\epsilon _ { k } \ \mathrm { i s } < 1$ . Equivalently, we have

$$
\begin{array} { r l } & { \sqrt { \frac { 1 + \left( \frac { \tilde { \mu } _ { 2 } ^ { ( k ) } } { \mu _ { 1 } } t _ { k } + \tilde { \epsilon } _ { k } \right) ^ { 2 } } { 1 + t _ { k } ^ { 2 } } } < 1 } \\ & { \Longleftrightarrow \frac { \tilde { \mu } _ { 2 } ^ { ( k ) } } { \mu _ { 1 } } t _ { k } + \tilde { \epsilon } _ { k } < t _ { k } } \\ & { \Longleftrightarrow \mu _ { 1 } \tilde { \epsilon } _ { k } < \left( \mu _ { 1 } - \tilde { \mu } _ { 2 } ^ { ( k ) } \right) t _ { k } . } \end{array}
$$

So, as long as $\begin{array} { r } { \tilde { \epsilon } _ { 0 } < \left( 1 - \frac { \tilde { \mu } _ { 2 } ^ { ( 0 ) } } { \mu _ { 1 } } \right) t _ { 0 } = \left( 1 - \frac { \tilde { \mu } _ { 2 } ^ { ( 0 ) } } { \mu _ { 1 } } \right) \frac { \sqrt { 1 - \omega ^ { 2 } } } { \omega } } \end{array}$ , we can ensure that $\epsilon _ { 1 } < \epsilon _ { 0 } = \omega ^ { - 1 }$ , which also implies that $\tilde { \epsilon } _ { 1 } < \tilde { \epsilon } _ { 0 } . \mathrm { ~ S o }$ , we require that

$$
\begin{array} { r l } & { ~ \mu _ { 1 } ^ { - 1 } \left( \alpha \lambda _ { n } \epsilon _ { \delta } \epsilon _ { 0 } + C _ { 2 } \epsilon _ { 0 } \rho _ { d e c a y } ^ { k _ { 0 } } \right) \left( 1 + \left( \alpha \lambda _ { 1 } \mu _ { 1 } ^ { - 1 } \epsilon _ { \delta } \epsilon _ { 0 } + C _ { 2 } \mu _ { 1 } ^ { - 1 } \epsilon _ { 0 } \rho _ { d e c a y } ^ { k _ { 0 } } \right) ( 1 + o ( 1 ) ) \right) } \\ & { < \left( 1 - \frac { \mu _ { 2 } \left( 1 + \left( \alpha \lambda _ { 1 } \mu _ { 1 } ^ { - 1 } \epsilon _ { \delta } \epsilon _ { 0 } + C _ { 2 } \mu _ { 1 } ^ { - 1 } \epsilon _ { 0 } \rho _ { d e c a y } ^ { k _ { 0 } } \right) ( 1 + o ( 1 ) ) \right) } { \mu _ { 1 } } \right) \left( \frac { \sqrt { 1 - \omega ^ { 2 } } } { \omega } \right) . } \end{array}
$$

To simplify the terms further, we have by definition that $| o ( 1 ) | < \eta$ for some $\eta \in ( 0 , 1 )$ . Then, a suficient condition for the above inequality is

$$
\left[ \frac { \sqrt { 1 - \omega ^ { 2 } } } { \omega } \mu _ { 2 } + \alpha \lambda _ { n } \epsilon _ { \delta } \epsilon _ { 0 } + C _ { 2 } \epsilon _ { 0 } \rho _ { d e c a y } ^ { k _ { 0 } } \right] \left[ 1 + \left( \alpha \lambda _ { 1 } \mu _ { 1 } ^ { - 1 } \epsilon _ { \delta } \epsilon _ { 0 } + C _ { 2 } \mu _ { 1 } ^ { - 1 } \epsilon _ { 0 } \rho _ { d e c a y } ^ { k _ { 0 } } \right) \left( 1 + \eta \right) \right] < \frac { \sqrt { 1 - \omega ^ { 2 } } } { \omega } \mu _ { 1 } .
$$

Expanding and expressing the equation in terms of $\epsilon _ { \delta }$ , we get

$$
a \epsilon _ { \delta } ^ { 2 } + b \epsilon _ { \delta } + c < 0 ,\tag{123}
$$

where

(124)

$$
\begin{array} { r l } & { \begin{array} { r l } & { \alpha _ { 1 } = ( 1 + \gamma ) ( \alpha \lambda \eta _ { 1 } ^ { \beta } \ c _ { \nu \alpha } ) ( \alpha \mu _ { 1 } \lambda _ { \alpha \tau \beta } ) } \\ & { = ( 1 + \gamma ) \alpha ^ { 2 } \lambda \lambda \lambda _ { \alpha \tau } ^ { \beta } } \\ & { b = \alpha \mu _ { 3 } \lambda _ { \alpha \tau } ^ { \beta } [ 1 + C _ { 2 , \eta _ { 1 } } ^ { 2 } \ \mathbf { 1 } _ { \eta \eta \alpha } ^ { \beta } \frac { 1 } { \rho \lambda _ { \alpha \tau } ^ { \beta } } ] + \eta [ \cfrac { \sqrt { 1 - \omega ^ { 2 } } } { \omega } \mu _ { 2 } + C _ { 2 } \eta _ { 0 } \lambda _ { \alpha \tau } ^ { \beta } \frac { 1 } { \rho \lambda _ { \alpha \tau } } ] \alpha \lambda _ { 1 } \mu _ { 1 } ^ { \beta } \ \mathfrak { r } _ { \alpha } ( 1 + \eta ) } \\ & { = \alpha \mu _ { \alpha } [ ( 1 + \eta ) \displaystyle \frac { \sqrt { 1 - \omega ^ { 2 } } } { \mu } \mu _ { 1 } ( \cfrac { \sqrt { 1 - \omega ^ { 2 } } } { \omega } ) ^ { 2 } + C _ { 2 } \eta _ { 0 } \lambda _ { \alpha \tau } ^ { \beta } \frac { 1 } { \rho \lambda _ { \alpha \tau } ^ { \beta } } ) + \lambda _ { \eta } ( \mu _ { 1 } + C _ { 2 } \eta \eta _ { 0 } ^ { \beta } \frac { 1 } { \rho \lambda _ { \alpha \tau } \eta } ) \frac { \sqrt { 1 - \omega ^ { 2 } } } { \omega } \mu _ { 1 } } \end{array} } \\ &  \begin{array} { r l } &  c = ( \cfrac { \sqrt { 1 - \omega ^ { 2 } } } { \omega } \mu _ { 2 } + C _ { 2 } \eta \eta _ { 0 } ^ { \beta } \frac { 1 } { \omega } ) ( 1 + C _ { 2 } \mu _ { 1 } ^ { - 1 } \eta _ { 0 } ^  \ \end{array} \end{array}\tag{125}
$$

(126)

Since $a > 0$ , to obtain an admissible solution $0 \le \epsilon _ { \delta } < \epsilon _ { \delta } ^ { * }$ , we require that $c < 0$ . Under this condition, coupled with $a > 0 .$ , it sufices that $\epsilon _ { \delta } ^ { * } = - \frac { c } { b }$

$$
\epsilon _ { \delta } ^ { * } = \frac { \alpha \frac { \sqrt { 1 - \omega ^ { 2 } } } { \omega } ( \lambda _ { 2 } - \lambda _ { 1 } ) - C _ { 2 } \epsilon _ { 0 } \rho _ { d e c a y } ^ { k _ { 0 } } } { \alpha \epsilon _ { 0 } } \left( 1 + ( 1 + \eta ) \frac { \sqrt { 1 - \omega ^ { 2 } } \frac { \mu _ { 2 } } { \mu _ { 1 } } } { \omega } + ( 1 + \eta ) C _ { 2 } \mu _ { 1 } ^ { - 1 } \epsilon _ { 0 } \rho _ { d e c a y } ^ { k _ { 0 } } \right)  { \alpha \epsilon _ { 0 } } .
$$

Assuming that $\rho _ { d e c a y } ^ { k _ { 0 } }$ is suficiently small in the local regime, we have

$$
\epsilon _ { \delta } ^ { * } \lesssim \frac { \frac { \sqrt { 1 - \omega ^ { 2 } } } { \omega } ( \lambda _ { 2 } - \lambda _ { 1 } ) } { \epsilon _ { 0 } \left[ ( 1 + \eta ) \frac { \sqrt { 1 - \omega ^ { 2 } } } { \omega } \frac { \mu _ { 2 } } { \mu _ { 1 } } \lambda _ { 1 } + \lambda _ { n } \mu _ { 1 } \right] } < \frac { \lambda _ { 2 } - \lambda _ { 1 } } { ( 1 + \eta ) \omega ^ { - 1 } \frac { \mu _ { 2 } } { \mu _ { 1 } } \lambda _ { 1 } + ( 1 - \omega ^ { 2 } ) ^ { - 1 / 2 } \lambda _ { n } \mu _ { 1 } } ,\tag{127}
$$

where we used $\omega ^ { - 1 } = \epsilon _ { 0 } \geq 1$ . Since $\epsilon _ { \delta } ^ { * } = \sqrt { 1 - \delta ^ { * } }$ , we equivalently have

$$
\delta ^ { * } \gtrsim 1 - \left( \frac { \lambda _ { 2 } - \lambda _ { 1 } } { ( 1 + \eta ) \omega ^ { - 1 } \frac { \mu _ { 2 } } { \mu _ { 1 } } \lambda _ { 1 } + ( 1 - \omega ^ { 2 } ) ^ { - 1 / 2 } \lambda _ { n } \mu _ { 1 } } \right) ^ { 2 } .\tag{128}
$$

Now, we want to ensure that this is indeed a contraction, to do so, we require $\mu _ { 2 } ^ { ( 0 ) } < \mu _ { 1 }$

$$
\begin{array} { c } { { 1 + \left( \alpha \lambda _ { 1 } \mu _ { 1 } ^ { - 1 } \epsilon _ { \delta } \epsilon _ { 0 } + C _ { 2 } \mu _ { 1 } ^ { - 1 } \epsilon _ { 0 } \rho _ { d e c a y } ^ { k _ { 0 } } \right) ( 1 + \eta ) < { \displaystyle \frac { \mu _ { 1 } } { \mu _ { 2 } } } } } \\ { { \alpha \lambda _ { 1 } \mu _ { 1 } ^ { - 1 } \epsilon _ { \delta } \epsilon _ { 0 } + C _ { 2 } \mu _ { 1 } ^ { - 1 } \epsilon _ { 0 } \rho _ { d e c a y } ^ { k _ { 0 } } < { \displaystyle \frac { \mu _ { 1 } - \mu _ { 2 } } { \mu _ { 2 } ( 1 + \eta ) } } } } \\ { { \lambda _ { 1 } \mu _ { 1 } ^ { - 1 } \epsilon _ { \delta } \epsilon _ { 0 } < { \displaystyle \frac { \lambda _ { 2 } - \lambda _ { 1 } } { \mu _ { 2 } ( 1 + \eta ) } } - C _ { 2 } \alpha ^ { - 1 } \mu _ { 1 } ^ { - 1 } \epsilon _ { 0 } \rho _ { d e c a y } ^ { k _ { 0 } } } } \\ { { \epsilon _ { \delta } < \epsilon _ { 0 } ^ { - 1 } { \displaystyle \frac { \mu _ { 1 } } { \mu _ { 2 } ( 1 + \eta ) } } { \displaystyle \frac { \lambda _ { 2 } - \lambda _ { 1 } } { \lambda _ { 1 } } } - C _ { 2 } { \displaystyle \frac { \rho _ { d e c a y } ^ { k _ { 0 } } } { \alpha \lambda _ { 1 } } } . } } \end{array}
$$

Equivalently, we have

$$
\epsilon _ { \delta } ^ { * } \lesssim \frac { \omega \mu _ { 1 } } { \mu _ { 2 } ( 1 + \eta ) } \frac { \lambda _ { 2 } - \lambda _ { 1 } } { \lambda _ { 1 } } \implies \delta ^ { * } \gtrsim 1 - \left( \omega \frac { \mu _ { 1 } } { \mu _ { 2 } ( 1 + \eta ) } \frac { \lambda _ { 2 } - \lambda _ { 1 } } { \lambda _ { 1 } } \right) ^ { 2 } .\tag{129}
$$

For the inductive step, for $\epsilon _ { 2 }$ to not expand, we require that $\mu _ { 1 } \tilde { \epsilon } _ { 1 } < \left( \mu _ { 1 } - \tilde { \mu } _ { 2 } ^ { ( 1 ) } \right) t _ { 1 }$ . Since $\tilde { \epsilon } _ { 1 } < \tilde { \epsilon } _ { 0 }$ and $\mu _ { 2 } ^ { ( 1 ) } < \mu _ { 2 } ^ { ( 0 ) }$ , we require $\mu _ { 1 } \tilde { \epsilon } _ { 0 } < \left( \mu _ { 1 } - \tilde { \mu } _ { 2 } ^ { ( 0 ) } \right) t _ { 1 }$ . Note that in Lemma (8), we have that lim sup $t _ { k } <$ $\frac { \mu _ { 1 } } { \mu _ { 1 } - \mu _ { 2 } ^ { ( 0 ) } } \tilde { \epsilon } _ { 0 }$ . So either we have reached the stable point $\left( \mathrm { i . e . } ~ O ( \tilde { \epsilon } _ { 0 } ) \right)$ or $t _ { k }$ is still larger than this radius. If the former, we are done, if the latter, we satisfy the inequality above implying $\epsilon _ { 2 } < \epsilon _ { 1 }$ . Hence, we have that either we have reached the neighbourhood around $\tilde { \epsilon } _ { 0 }$ or $\epsilon _ { k }$ will be non-increasing which implies the factors $\mu _ { 2 } ^ { ( k ) } , \tilde { \epsilon } _ { k }$ are non-increasing.

Then, for

$$
\delta \gtrsim 1 - \operatorname* { m i n } \left( \omega \frac { \mu _ { 1 } } { \mu _ { 2 } ( 1 + \eta ) } \frac { \lambda _ { 2 } - \lambda _ { 1 } } { \lambda _ { 1 } } , \frac { \lambda _ { 2 } - \lambda _ { 1 } } { ( 1 + \eta ) \omega ^ { - 1 } \frac { \mu _ { 2 } } { \mu _ { 1 } } \lambda _ { 1 } + ( 1 - \omega ^ { 2 } ) ^ { - 1 / 2 } \lambda _ { n } \mu _ { 1 } } \right) ^ { 2 } ,
$$

we have that $\begin{array} { r l r } { t _ { k + 1 } } & { { } \le } & { \frac { \tilde { \mu } _ { 2 } ^ { ( 0 ) } } { \mu _ { 1 } } t _ { k } + \tilde { \epsilon } _ { 0 } } \end{array}$ is a perturbed contracting, which according to Lemma (8), lim sup $t _ { k } = O ( \epsilon _ { \delta } ) < C \dot { \epsilon } _ { \delta }$ for some fixed constant $C .$ . The contraction forces the iterates to stay within a $C \epsilon _ { \delta }$ radius around 0. Then fix any target $\tilde { \delta } _ { 0 }$ , let $\begin{array} { r } { \epsilon _ { \delta } = \frac { 1 } { C } \sqrt { \frac { 1 - \tilde { \delta } _ { 0 } } { \tilde { \delta } _ { 0 } } } } \end{array}$ , there exists $k _ { 1 }$ such that $t _ { k _ { 1 } } < C \epsilon _ { \delta }$ (we assume $k _ { 1 } < K _ { \epsilon } ,$ if not then we have already converged). Since $t _ { k _ { 1 } }$ is the tangent of the angle between $u _ { 1 }$ and $g _ { k _ { 1 } }$ , we have by the formula $\begin{array} { r } { \cos ^ { 2 } \theta = \frac { 1 } { 1 + \tan ^ { 2 } \theta } } \end{array}$ that cos $\dot { \ell } _ { k _ { 1 } } \geq \tilde { \delta } _ { 0 }$ which is equivalent to $\left. u _ { 1 } , g _ { k _ { 1 } } \right. ^ { 2 } \ge \tilde { \delta } _ { 0 } \| g _ { k _ { 1 } } \| ^ { 2 }$ . For simplicity, we let $\langle u _ { 1 } , g _ { k _ { 1 } } \rangle ^ { 2 } = \delta _ { 0 } \| g _ { k _ { 1 } } \| ^ { 2 }$ where $\delta _ { 0 } \ge \tilde { \delta } _ { 0 }$ .

Phase 2: Freezing $v _ { k }$ still promises continued alignment with $u _ { 1 }$ . For this phase, consider $k \geq k _ { 1 }$ . Consider the expansion

$$
g _ { k + 1 } = \left( I - \alpha H P _ { k } P _ { k } ^ { \top } \right) g _ { k } + r _ { k } .\tag{130}
$$

Part 2a: Upper Bounding Denominator. Expanding $P _ { k } P _ { k } ^ { \top } = \hat { v } _ { k _ { 1 } } \hat { v } _ { k _ { 1 } } ^ { \top } + \tilde { P } _ { k } \tilde { P } _ { k } ^ { \top }$ , we find that for $i \neq 1$

$$
\begin{array} { r l } & { u _ { i } ^ { \top } g _ { k + 1 } = u _ { i } ^ { \top } \left( I - \alpha H \left( \hat { v } _ { k _ { 1 } } \hat { v } _ { k _ { 1 } } ^ { \top } + \tilde { P } _ { k } \tilde { P } _ { k } ^ { \top } \right) \right) g _ { k } + u _ { i } ^ { \top } r _ { k } } \\ & { \qquad = u _ { i } ^ { \top } g _ { k } - \alpha \lambda _ { i } \Big \langle \tilde { P } _ { k } ^ { \top } u _ { i } , \tilde { P } _ { k } ^ { \top } g _ { k } \Big \rangle - \alpha \lambda _ { i } u _ { i } ^ { \top } \hat { v } _ { k _ { 1 } } \hat { v } _ { k _ { 1 } } ^ { \top } g _ { k } + u _ { i } ^ { \top } r _ { k } . } \end{array}
$$

For the second term, we have

$$
\begin{array} { r l } & { \left. \tilde { P } _ { k } ^ { \top } u _ { i } , \tilde { P } _ { k } ^ { \top } g _ { k } \right. \geq \left. \left( I - \hat { v } _ { k _ { 1 } } \hat { v } _ { k _ { 1 } } ^ { \top } \right) u _ { i } , \left( I - \hat { v } _ { k _ { 1 } } \hat { v } _ { k _ { 1 } } ^ { \top } \right) g _ { k } \right. - \tau \Big \| \left( I - \hat { v } _ { k _ { 1 } } \hat { v } _ { k _ { 1 } } ^ { \top } \right) u _ { i } \Big \| \Big \| \left( I - \hat { v } _ { k _ { 1 } } \hat { v } _ { k _ { 1 } } ^ { \top } \right) g _ { k } \Big \| } \\ & { \qquad = u _ { i } ^ { \top } \left( I - \hat { v } _ { k _ { 1 } } \hat { v } _ { k _ { 1 } } ^ { \top } \right) g _ { k } - \tau \Big \| \mathrm { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \bot } } g _ { k } \Big \| \Big \| \mathrm { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \bot } } u _ { i } \Big \| } \\ & { \qquad = u _ { i } ^ { \top } g _ { k } - u _ { i } ^ { \top } \hat { v } _ { k _ { 1 } } \hat { v } _ { k _ { 1 } } ^ { \top } g _ { k } - \tau \Big \| \mathrm { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \bot } } g _ { k } \Big \| \Big \| \mathrm { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \bot } } u _ { i } \Big \| } \end{array}
$$

with probability $1 - \exp { \left( - c d \tau ^ { 2 } \right) }$ . Substituting back into the original equation, we obtain an upper bound

$$
\begin{array} { r l } & { u _ { i } ^ { \top } g _ { k + 1 } \leq u _ { i } ^ { \top } g _ { k } - \alpha \lambda _ { i } \left( u _ { i } ^ { \top } g _ { k } - u _ { i } ^ { \top } \hat { v } _ { k _ { 1 } } \hat { v } _ { k _ { 1 } } ^ { \top } g _ { k } - \tau \Big | \Big | \mathrm { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \bot } } g _ { k } \Big | \Big | \Big | \mathrm { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \bot } } u _ { i } \Big | \right) } \\ & { \qquad - \alpha \lambda _ { i } u _ { i } ^ { \top } \hat { v } _ { k _ { 1 } } \hat { v } _ { k _ { 1 } } ^ { \top } g _ { k } + u _ { i } ^ { \top } r _ { k } } \\ & { = ( 1 - \alpha \lambda _ { i } ) \langle u _ { i } , g _ { k } \rangle + \alpha \lambda _ { i } \tau \Big | \Big | \mathrm { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \bot } } g _ { k } \Big | \Big | \Big | \Big | \mathrm { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \bot } } u _ { i } \Big | \Big | + u _ { i } ^ { \top } r _ { k } . } \end{array}\tag{131}
$$

Taking absolute value, we obtain

$$
\left| u _ { i } ^ { \top } g _ { k + 1 } \right| \leq ( 1 - \alpha \lambda _ { i } ) | \langle u _ { i } , g _ { k } \rangle | + \alpha \lambda _ { i } \tau \left( \left\| \mathrm { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \bot } } g _ { k } \right\| \left\| \mathrm { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \bot } } u _ { i } \right\| \right) + \left\| r _ { k } \right\|\tag{132}
$$

Part 2b: Lower Bounding Denominator. On the other hand, for the projection onto $u _ { 1 }$ , we have

$$
\begin{array} { r } { u _ { 1 } ^ { \top } g _ { k + 1 } = u _ { 1 } ^ { \top } g _ { k } - \alpha \lambda _ { 1 } \Big \langle \tilde { P } _ { k } ^ { \top } u _ { 1 } , \tilde { P } _ { k } ^ { \top } g _ { k } \Big \rangle - \alpha \lambda _ { 1 } u _ { 1 } ^ { \top } \hat { v } _ { k _ { 1 } } \hat { v } _ { k _ { 1 } } ^ { \top } g _ { k } + u _ { 1 } ^ { \top } r _ { k } . } \end{array}
$$

Similarly, by JLL, the second term can be upper bounded by

$$
\begin{array} { r } { \left. \tilde { P } _ { k } ^ { \top } u _ { 1 } , \tilde { P } _ { k } ^ { \top } g _ { k } \right. \leq u _ { 1 } ^ { \top } \left( I - \hat { v } _ { k _ { 1 } } \hat { v } _ { k _ { 1 } } ^ { \top } \right) g _ { k } + \tau \left\| \mathrm { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \bot } g _ { k } } \right\| \left\| \mathrm { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \bot } u _ { 1 } } \right\| } \\ { = u _ { 1 } ^ { \top } g _ { k } - u _ { i } ^ { \top } \hat { v } _ { k _ { 1 } } \hat { v } _ { k _ { 1 } } ^ { \top } g _ { k } + \tau \left\| \mathrm { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \bot } g _ { k } } \right\| \left\| \mathrm { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \bot } u _ { 1 } } \right\| } \end{array}
$$

with probability $1 - \exp \left( - c d \tau ^ { 2 } \right)$ . This give us

$$
\begin{array} { r l } & { u _ { 1 } ^ { \top } g _ { k + 1 } \geq u _ { 1 } ^ { \top } g _ { k } - \alpha \lambda _ { 1 } \left( u _ { 1 } ^ { \top } g _ { k } - u _ { 1 } ^ { \top } \hat { v } _ { k _ { 1 } } \hat { v } _ { k _ { 1 } } ^ { \top } g _ { k } + \tau \Big \| \mathrm { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \bot } } g _ { k } \Big \| \Big \| \mathrm { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \bot } } u _ { 1 } \Big \| \right) } \\ & { \qquad - \alpha \lambda _ { 1 } u _ { 1 } ^ { \top } \hat { v } _ { k _ { 1 } } \hat { v } _ { k _ { 1 } } ^ { \top } g _ { k } + u _ { 1 } ^ { \top } r _ { k } } \\ & { = ( 1 - \alpha \lambda _ { 1 } ) \langle u _ { 1 } , g _ { k } \rangle - \alpha \lambda _ { 1 } \tau \Big \| \mathrm { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \bot } } g _ { k } \Big \| \Big \| \mathrm { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \bot } } u _ { 1 } \Big \| + u _ { 1 } ^ { \top } r _ { k } . } \end{array}
$$

Using $\left\| \operatorname* { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \perp } } u _ { 1 } \right\| \le \sqrt { 1 - \delta _ { 0 } } = \epsilon _ { \delta _ { 0 } }$ and triangle inequality, the absolute value of the correlation yields a lower bound

$$
\left| u _ { 1 } ^ { \top } g _ { k + 1 } \right| \geq ( 1 - \alpha \lambda _ { 1 } ) \Big | u _ { 1 } ^ { \top } g _ { k } \Big | - \alpha \lambda _ { 1 } { \tau } \epsilon _ { \delta _ { 0 } } \Big \| \mathrm { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \perp } } g _ { k } \Big \| - \| r _ { k } \| .\tag{133}
$$

Part 2c: Combining Both Bounds. If we define $\begin{array} { r } { t _ { i } ^ { ( k ) } = \frac { | \langle u _ { i } , g _ { k } \rangle | } { | \langle u _ { 1 } , g _ { k } \rangle | } } \end{array}$ , then we have the recursion

$$
t _ { i } ^ { ( k + 1 ) } \leq \frac { ( 1 - \alpha \lambda _ { i } ) | \langle u _ { i } , g _ { k } \rangle | + \alpha \lambda _ { i } \tau _ { i } } { ( 1 - \alpha \lambda _ { 1 } ) \big | u _ { 1 } ^ { \top } g _ { k } \big | - \alpha \lambda _ { 1 } \tau _ { 1 } \epsilon _ { \delta _ { 0 } } \big | \big | \operatorname { P r o j } _ { \hat { v } _ { k _ { 1 } } ^ { \perp } } g _ { k } \big | \big | - \big \| r _ { k } \big \| } , \qquad \forall i \geq 2 ,\tag{134}
$$

which has a similar expression as in phase 1, holding with probability at least $\begin{array} { r } { 1 - \sum _ { i } \exp { ( - c d \tau _ { i } ^ { 2 } ) } } \end{array}$ Simplifying the equation above, we get

$$
t _ { i } ^ { ( k + 1 ) } \leq \frac { \mu _ { i } t _ { i } ^ { ( k ) } + \alpha \lambda _ { i } \tau _ { i } \epsilon _ { k } + C _ { 2 } ^ { \prime } \epsilon _ { k } \| g _ { k } \| } { \mu _ { 1 } - \alpha \lambda _ { 1 } \tau _ { 1 } \epsilon _ { \delta _ { 0 } } \epsilon _ { k } - C _ { 2 } ^ { \prime } \epsilon _ { k } \| g _ { k } \| } ,\tag{135}
$$

where $\begin{array} { r } { \epsilon _ { k } = \frac { \| g _ { k } \| } { | u _ { 1 } ^ { \top } g _ { k } | } } \end{array}$ and $\begin{array} { r } { C _ { 2 } ^ { \prime } = \left( \frac { n - 1 } { d } \right) ^ { 2 } C _ { 2 } } \end{array}$ . In Lemma 5, we used the fact that $\| P _ { k } P _ { k } ^ { \top } \| _ { o p } = 1$ , however in phase 2, we have that this is $\textstyle { \frac { n - 1 } { d } }$ instead. Despite the case, the main contributor is still the exponentially decaying term, and it would only require $O ( \log ( n / d ) )$ steps for the extra factor to be negligible. The iteration starts from $k = k _ { 1 }$ , where from phase 1, we have $\epsilon _ { k _ { 1 } } = \delta _ { 0 } ^ { - 1 / 2 }$ . Doing the same trick as in phase 1, where we used the Taylor expansion for $\scriptstyle { \frac { 1 } { 1 - x } }$ , we obtain

$$
t _ { i } ^ { ( k + 1 ) } \leq \frac { \mu _ { i } t _ { i } ^ { ( k ) } + \alpha \lambda _ { i } \tau _ { i } \epsilon _ { k } + C _ { 2 } ^ { \prime } \epsilon _ { k } \| g _ { k } \| } { \mu _ { 1 } } \left[ 1 + \left( 1 + \eta \right) \left( \alpha \mu _ { 1 } ^ { - 1 } \lambda _ { 1 } \tau _ { 1 } \epsilon _ { \partial } \epsilon _ { k } + C _ { 2 } ^ { \prime } \mu _ { 1 } ^ { - 1 } \epsilon _ { k } \| g _ { k } \| \right) \right] ,
$$

for some $\eta \in ( 0 , 1 )$ . With a small perturbation to Lemma (6), we find that for $k > k _ { 1 }$ , we have $\left\| g _ { k } \right\| ^ { 2 } \leq C \tilde { \rho } ^ { k - \bar { k _ { 1 } } } \rho _ { d e c a y } ^ { k _ { 1 } } .$ where $\tilde { \rho }$ is the rate of decay in phase 2 with the rescaled matrix. Rewriting the equation yields

$$
\begin{array} { r l } & { t _ { i } ^ { ( k + 1 ) } \le \frac { \mu _ { i } t _ { i } ^ { ( k ) } + \alpha \lambda _ { i } \tau _ { i } \epsilon _ { k } + C _ { 2 } ^ { \prime } \epsilon _ { k } \bar { \rho } ^ { k - k _ { 1 } } \rho _ { d e c a y } ^ { k _ { 1 } } } { \mu _ { 1 } } \left[ 1 + ( 1 + \eta ) \left( \alpha \mu _ { 1 } ^ { - 1 } \lambda _ { 1 } \tau _ { 1 } \epsilon _ { \delta _ { 0 } } \epsilon _ { k } + C _ { 2 } ^ { \prime } \mu _ { 1 } ^ { - 1 } \epsilon _ { k } \tilde { \rho } ^ { k - k _ { 1 } } \rho _ { d e c a y } ^ { k _ { 1 } } \right) \right] } \\ & { \quad \quad \quad = \frac { \tilde { \mu } _ { i } ^ { ( k ) } } { \mu _ { 1 } } t _ { i } ^ { ( k ) } + \tilde { \epsilon } _ { i } ^ { ( k ) } , } \end{array}
$$

where

$$
\begin{array} { r l } & { \tilde { \mu } _ { i } ^ { ( k ) } = \mu _ { i } \left[ 1 + ( 1 + \eta ) \left( \alpha \mu _ { 1 } ^ { - 1 } \lambda _ { 1 } \tau _ { 1 } \epsilon _ { \delta _ { 0 } } \epsilon _ { k } + C _ { 2 } ^ { \prime } \mu _ { 1 } ^ { - 1 } \epsilon _ { k } \tilde { \rho } ^ { k - k _ { 1 } } \rho _ { d e c a y } ^ { k _ { 1 } } \right) \right] } \\ & { \tilde { \epsilon } _ { i } ^ { ( k ) } = \mu _ { 1 } ^ { - 1 } \left( \alpha \lambda _ { i } \tau _ { i } \epsilon _ { k } + C _ { 2 } ^ { \prime } \epsilon _ { k } \tilde { \rho } ^ { k - k _ { 1 } } \rho _ { d e c a y } ^ { k _ { 1 } } \right) \left[ 1 + ( 1 + \eta ) \left( \alpha \mu _ { 1 } ^ { - 1 } \lambda _ { 1 } \tau _ { 1 } \epsilon _ { \delta _ { 0 } } \epsilon _ { k } + C _ { 2 } ^ { \prime } \mu _ { 1 } ^ { - 1 } \epsilon _ { k } \tilde { \rho } ^ { k - k _ { 1 } } \rho _ { d e c a y } ^ { k _ { 1 } } \right) \right] . } \end{array}
$$

Part 2d: Finding the Condition for Perturbed Fixed Point. First, for the factor to be a contraction, we require $\tilde { \mu } _ { i } ^ { ( k ) } < \mu _ { 1 }$

$$
\begin{array} { r l } & { 1 + ( 1 + \eta ) \left( \alpha \mu _ { 1 } ^ { - 1 } \lambda _ { 1 } \tau _ { 1 } \epsilon _ { \delta _ { 0 } } \epsilon _ { k } + C _ { 2 } ^ { \prime } \mu _ { 1 } ^ { - 1 } \epsilon _ { k } \tilde { \rho } ^ { k - k _ { 1 } } \rho _ { d e c a y } ^ { k _ { 1 } } \right) < \frac { \mu _ { 1 } } { \mu _ { i } } } \\ & { \qquad \alpha \mu _ { 1 } ^ { - 1 } \lambda _ { 1 } \tau _ { 1 } \epsilon _ { \delta _ { 0 } } \epsilon _ { k } + C _ { 2 } ^ { \prime } \mu _ { 1 } ^ { - 1 } \epsilon _ { k } \tilde { \rho } ^ { k - k _ { 1 } } \rho _ { d e c a y } ^ { k _ { 1 } } < \alpha \frac { \lambda _ { i } - \lambda _ { 1 } } { \mu _ { i } } } \\ & { \qquad \quad \lambda _ { 1 } \tau _ { 1 } \epsilon _ { \delta _ { 0 } } \epsilon _ { k } < \frac { \mu _ { 1 } } { \mu _ { i } } ( \lambda _ { i } - \lambda _ { 1 } ) - C _ { 2 } ^ { \prime } \alpha ^ { - 1 } \epsilon _ { k } \tilde { \rho } ^ { k - k _ { 1 } } \rho _ { d e c a y } ^ { k _ { 1 } } } \\ & { \qquad \quad \tau _ { 1 } \epsilon _ { \delta _ { 0 } } < \frac { \mu _ { 1 } } { \mu _ { i } } \frac { \lambda _ { i } - \lambda _ { 1 } } { \lambda _ { 1 } } \epsilon _ { k } ^ { - 1 } - C _ { 2 } ^ { \prime } \lambda _ { 1 } ^ { - 1 } \alpha ^ { - 1 } \tilde { \rho } ^ { k - k _ { 1 } } \rho _ { d e c a y } ^ { k _ { 1 } } . } \end{array}
$$

For $k = k _ { 1 }$ , it sufices that

$$
\tau _ { 1 } \epsilon _ { \delta _ { 0 } } < \frac { \mu _ { 1 } } { \mu _ { 2 } } \frac { \lambda _ { 2 } - \lambda _ { 1 } } { \lambda _ { 1 } } \sqrt { \delta _ { 0 } } - C _ { 2 } ^ { \prime } \lambda _ { 1 } ^ { - 1 } \alpha ^ { - 1 } \rho _ { d e c a y } ^ { k _ { 1 } } ,\tag{136}
$$

where the minimum is taken over all $i \geq 2$ and satisfied when $i \ = \ 2$ . Under this $\tau _ { 1 }$ condition, the initial inequality at the top is satisfied for all $i \geq 2$ . To control $\epsilon _ { k }$ , we will consider a slightly diferent decomposition of $\| g _ { k } \|$ as before.

$$
\| g _ { k } \| = \sqrt { \sum _ { i = 1 } ^ { n } \bigl | u _ { i } ^ { \top } g _ { k } \bigr | ^ { 2 } } = \Bigl | u _ { 1 } ^ { \top } g _ { k } \Bigr | \sqrt { 1 + \sum _ { i \geq 2 } \left( t _ { i } ^ { ( k ) } \right) ^ { 2 } } .\tag{137}
$$

Consequently, we have

$$
\epsilon _ { k } = \sqrt { 1 + \sum _ { i \geq 2 } \left( t _ { i } ^ { ( k ) } \right) ^ { 2 } } \implies \epsilon _ { k + 1 } \leq \epsilon _ { k } \sqrt { \frac { 1 + \sum _ { i \geq 2 } \left( \frac { \tilde { \mu } _ { i } ^ { ( k ) } } { \mu _ { 1 } } t _ { i } ^ { ( k ) } + \tilde { \epsilon } _ { i } ^ { ( k ) } \right) ^ { 2 } } { 1 + \sum _ { i \geq 2 } \left( t _ { i } ^ { ( k ) } \right) ^ { 2 } } } .
$$

For $\epsilon _ { k }$ to be non-increasing in k, the factor has to be $\leq 1$ . Equivalently, we require

$$
\sum _ { i \geq 2 } \bigg ( \frac { \tilde { \mu } _ { i } ^ { ( k ) } } { \mu _ { 1 } } t _ { i } ^ { ( k ) } + \tilde { \epsilon } _ { i } ^ { ( k ) } \bigg ) ^ { 2 } \leq \sum _ { i \geq 2 } \Big ( t _ { i } ^ { ( k ) } \Big ) ^ { 2 } ,
$$

and it sufices that

$$
\frac { \tilde { \mu } _ { i } ^ { ( k ) } } { \mu _ { 1 } } t _ { i } ^ { ( k ) } + \tilde { \epsilon } _ { i } ^ { ( k ) } \le t _ { i } ^ { ( k ) } \qquad \forall i \ge 2 .\tag{138}
$$

Consider $k = k _ { 1 }$ , we require

$$
\begin{array} { r l } & { \quad \left( \alpha \lambda _ { i } \tau _ { i } \epsilon _ { k _ { 1 } } + C _ { 2 } ^ { \prime } \epsilon _ { k _ { 1 } } \rho _ { d e c a y } ^ { k _ { 1 } } \right) \left[ 1 + ( 1 + \eta ) \left( \alpha \mu _ { 1 } ^ { - 1 } \lambda _ { 1 } \tau _ { 1 } \epsilon _ { \delta _ { 0 } } \epsilon _ { k _ { 1 } } + C _ { 2 } ^ { \prime } \mu _ { 1 } ^ { - 1 } \epsilon _ { k _ { 1 } } \rho _ { d e c a y } ^ { k _ { 1 } } \right) \right] } \\ & { \leq \left( \mu _ { 1 } - \mu _ { i } \left[ 1 + ( 1 + \eta ) \left( \alpha \mu _ { 1 } ^ { - 1 } \lambda _ { 1 } \tau _ { 1 } \epsilon _ { \delta _ { 0 } } \epsilon _ { k _ { 1 } } + C _ { 2 } ^ { \prime } \mu _ { 1 } ^ { - 1 } \epsilon _ { k _ { 1 } } \rho _ { d e c a y } ^ { k _ { 1 } } \right) \right] \right) t _ { i } ^ { ( k _ { 1 } ) } . } \end{array}
$$

If $\begin{array} { r } { t _ { i } ^ { ( k _ { 1 } ) } < \frac { \mu _ { 1 } } { \mu _ { 1 } - \tilde { \mu } _ { i } ^ { ( k _ { 1 } ) } } \tilde { \epsilon } _ { i } ^ { ( k _ { 1 } ) } } \end{array}$ , we are done as that is indeed the best result we can obtain for the convergence.

In the worst case, we have $\begin{array} { r } { t _ { i } ^ { ( k _ { 1 } ) } = \sqrt { \frac { 1 - \delta _ { 0 } } { \delta _ { 0 } } } } \end{array}$ and $\begin{array} { r } { \epsilon _ { k _ { 1 } } = \frac { 1 } { \sqrt { \delta _ { 0 } } } } \end{array}$ . For this, we simplify the above inequality by multiplying both sides with $\sqrt { \delta _ { 0 } }$ to get

$$
\begin{array} { r l r } & { } & { \left( \alpha \lambda _ { i } \tau _ { i } + C _ { 2 } ^ { \prime } \rho _ { d e c a y } ^ { k _ { 1 } } \right) \Big [ 1 + ( 1 + \eta ) \left( \alpha \mu _ { 1 } ^ { - 1 } \lambda _ { 1 } \tau _ { 1 } \epsilon _ { \delta _ { 0 } } \epsilon _ { k _ { 1 } } + C _ { 2 } ^ { \prime } \mu _ { 1 } ^ { - 1 } \epsilon _ { k _ { 1 } } \rho _ { d e c a y } ^ { k _ { 1 } } \right) \Big ] } \\ & { } & { \quad \le \left( \mu _ { 1 } - \mu _ { i } \left[ 1 + ( 1 + \eta ) \left( \alpha \mu _ { 1 } ^ { - 1 } \lambda _ { 1 } \tau _ { 1 } \epsilon _ { \delta _ { 0 } } \epsilon _ { k _ { 1 } } + C _ { 2 } ^ { \prime } \mu _ { 1 } ^ { - 1 } \epsilon _ { k _ { 1 } } \rho _ { d e c a y } ^ { k _ { 1 } } \right) \right] \right) \sqrt { 1 - \delta _ { 0 } } . } \end{array}
$$

Combining the second term on the RHS with the LHS, we obtain

$$
\left( \sqrt { 1 - \delta _ { 0 } } \mu _ { i } + \alpha \lambda _ { i } \tau _ { i } + C _ { 2 } ^ { \prime } \rho _ { d e c a y } ^ { k _ { 1 } } \right) \left[ 1 + \left( 1 + \eta \right) \left( \alpha \mu _ { 1 } ^ { - 1 } \lambda _ { 1 } \tau _ { 1 } \epsilon _ { \delta _ { 0 } } \epsilon _ { k _ { 1 } } + C _ { 2 } ^ { \prime } \mu _ { 1 } ^ { - 1 } \epsilon _ { k _ { 1 } } \rho _ { d e c a y } ^ { k _ { 1 } } \right) \right] \leq \mu _ { 1 } \sqrt { 1 - \delta _ { 0 } } .
$$

Considering this as a function of $\tau _ { i }$ , we get a linear function

$$
a \tau _ { i } + b \leq 0 ,
$$

where

$$
\begin{array} { r l } & { a = \alpha \lambda _ { i } \left[ 1 + ( 1 + \eta ) \left( \alpha \mu _ { 1 } ^ { - 1 } \lambda _ { 1 } \pi \kappa _ { \delta _ { 0 } } \epsilon _ { k _ { 1 } } + C _ { 2 } ^ { \prime } \mu _ { 1 } ^ { - 1 } \epsilon _ { k _ { 1 } } \rho _ { d e c a y } ^ { k _ { 1 } } \right) \right] } \\ & { b = \left( \sqrt { 1 - \delta _ { 0 } } \mu _ { i } + C _ { 2 } ^ { \prime } \rho _ { d e c a y } ^ { k _ { 1 } } \right) \left[ 1 + ( 1 + \eta ) \left( \alpha \mu _ { 1 } ^ { - 1 } \lambda _ { 1 } \tau _ { 1 } \epsilon _ { \delta _ { 0 } } \epsilon _ { k _ { 1 } } + C _ { 2 } ^ { \prime } \mu _ { 1 } ^ { - 1 } \epsilon _ { k _ { 1 } } \rho _ { d e c a y } ^ { k _ { 1 } } \right) \right] - \mu _ { 1 } \sqrt { 1 - \delta _ { 0 } } . } \end{array}
$$

Since $a > 0$ this is equivalent to $\begin{array} { r } { \tau _ { i } < \frac { b } { a } } \end{array}$

$$
\lambda _ { i } \tau _ { i } < \frac { \mu _ { 1 } \sqrt { 1 - \delta _ { 0 } } - \left( \sqrt { 1 - \delta _ { 0 } } \mu _ { i } + C _ { 2 } ^ { \prime } \rho _ { d c e a y } ^ { k _ { 1 } } \right) \left[ 1 + \left( 1 + \eta \right) \left( \alpha \mu _ { 1 } ^ { - 1 } \lambda _ { 1 } \tau _ { 1 } \epsilon _ { \delta } \epsilon _ { k _ { 1 } } + C _ { 2 } ^ { \prime } \mu _ { 1 } ^ { - 1 } \epsilon _ { k _ { 1 } } \rho _ { d c e a y } ^ { k _ { 1 } } \right) \right] } { \alpha \left[ 1 + \left( 1 + \eta \right) \left( \alpha \mu _ { 1 } ^ { - 1 } \lambda _ { 1 } \tau _ { 1 } \epsilon _ { \delta _ { 0 } } \epsilon _ { k _ { 1 } } + C _ { 2 } ^ { \prime } \mu _ { 1 } ^ { - 1 } \epsilon _ { k _ { 1 } } \rho _ { d e c e a y } ^ { k _ { 1 } } \right) \right] }
$$

Similar to Phase 1, we assume the contribution by $\rho _ { d e c a y } ^ { k _ { 1 } }$ is small in the local regime and the second term in the denominator on the RHS is controlled small by $\epsilon _ { \delta _ { 0 } }$ in comparison to 1, giving us

$$
\tau _ { i } \lesssim \frac { ( 1 - \tilde { \eta } ) \mu _ { 1 } - \mu _ { i } } { \alpha \lambda _ { i } } \sqrt { 1 - \delta _ { 0 } } \lesssim \frac { \lambda _ { i } - \lambda _ { 1 } } { \lambda _ { i } } \sqrt { 1 - \delta _ { 0 } } ,
$$

where $0 < \tilde { \eta } \ll 1$

Part 2e: Exponential Decay in the Inductive Step. Under these conditions for $\tau _ { 1 } , \cdots , \tau _ { n } .$ we have that

$$
\epsilon _ { k _ { 1 } + 1 } < \epsilon _ { k _ { 1 } } \implies { \langle u _ { 1 } , g _ { k _ { 1 } + 1 } \rangle } ^ { 2 } \geq \delta _ { 0 } \| g _ { k _ { 1 } + 1 } \| ^ { 2 } .
$$

By Proposition (1), we have $\langle v _ { k _ { 1 } } , g _ { k _ { 1 } + 1 } \rangle ^ { 2 } \geq ( 2 \delta _ { 0 } - 1 ) ^ { 2 }$ . Then, Lemma (7) with $\delta = ( 2 \delta _ { 0 } - 1 ) ^ { 2 }$ , we obtain

$$
\begin{array} { r l } & { f ( x _ { k _ { 1 } + 2 } ) - f ( x _ { k _ { 1 } + 1 } ) \leq \left\{ \left[ - \alpha \tau + \frac { \alpha ^ { 2 } L } { 2 } \left( 1 - \frac { n - 1 } { d } ( 1 + \tau ) \right) \right] ( 2 \delta _ { 0 } - 1 ) ^ { 2 } \right. } \\ & { \qquad \quad \left. + \left[ - \alpha ( 1 - \tau ) + \frac { \alpha ^ { 2 } L } { 2 } \frac { n - 1 } { d } \right] \right\} \| g _ { k _ { 1 } + 1 } \| ^ { 2 } . } \end{array}
$$

The minimum of the polynomial $a x ^ { 2 } - b x$ where $a , b > 0$ is attained at $\begin{array} { r } { x ^ { * } = \frac { b } { 2 a } } \end{array}$ , meaning

$$
\begin{array} { l } { \displaystyle \alpha ^ { * } = \frac { d } { L } \left( \frac { \tau ( 2 \delta _ { 0 } - 1 ) ^ { 2 } + ( 1 - \tau ) } { ( n - 1 ) - ( 2 \delta _ { 0 } - 1 ) ^ { 2 } ( ( n - 1 ) ( 1 + \tau ) - d ) } \right) } \\ { \displaystyle = \frac { d } { L } \left( \frac { 1 - 4 \tau \delta _ { 0 } ( 1 - \delta _ { 0 } ) } { ( n - 1 ) ( 1 - ( 2 \delta _ { 0 } - 1 ) ^ { 2 } ( 1 + \tau ) ) + d ( 2 \delta _ { 0 } - 1 ) ^ { 2 } ) } \right) } \\ { \displaystyle = \frac { d } { L } \left( \frac { 1 - 4 \tau \delta _ { 0 } ( 1 - \delta _ { 0 } ) } { ( n - 1 ) ( - \tau + 4 \delta _ { 0 } ( 1 - \delta _ { 0 } ) ( 1 + \tau ) ) + d ( 2 \delta _ { 0 } - 1 ) ^ { 2 } ) } \right) . } \end{array}\tag{139}
$$

Substituting this back, and noting that $\begin{array} { r } { a x ^ { * 2 } - b x ^ { * } = - \frac { b ^ { 2 } } { 4 a } = - \frac { b } { 2 } x ^ { * } } \end{array}$ , we find that

$$
\begin{array} { l } { \displaystyle f ( x _ { k _ { 1 } + 2 } ) - f ( x _ { k _ { 1 } + 1 } ) \leq - \frac 1 2 ( 1 - 4 \tau \delta _ { 0 } ( 1 - \delta _ { 0 } ) ^ { 2 } ) \alpha ^ { * } \| g _ { k _ { 1 } + 1 } \| ^ { 2 } } \\ { \displaystyle = - \frac { d } { L } \left( \frac { [ 1 - 4 \tau \delta _ { 0 } ( 1 - \delta _ { 0 } ) ] ^ { 2 } } { ( n - 1 ) ( - \tau + 4 \delta _ { 0 } ( 1 - \delta _ { 0 } ) ( 1 + \tau ) ) + d ( 2 \delta _ { 0 } - 1 ) ^ { 2 } ) } \right) \| g _ { k _ { 1 } + 1 } \| ^ { 2 } . } \end{array}
$$

Consequently, we have

$$
\begin{array} { r l } & { f ( x _ { k _ { 1 } + 2 } ) - f ( x ^ { * } ) \leq f ( x _ { k _ { 1 } + 1 } ) - f ( x * ) - \hat { \rho } \| g _ { k _ { 1 } + 1 } \| ^ { 2 } } \\ & { \qquad \leq ( 1 - 2 \mu \hat { \rho } ) \big ( f ( x _ { k _ { 1 } + 1 } ) - f ( x * ) \big ) } \\ & { \qquad \leq \tilde { \rho } ^ { 4 } \rho _ { d e c a y } ^ { 2 k _ { 1 } } , } \end{array}
$$

with probability at least $1 - e ^ { - c d \tau ^ { 2 } }$ , conditioned on the iterate $k _ { 1 } + 1$ . We indicate the decay in 1 step with $\tilde { \rho } ^ { 2 }$ , since the gradient term is also squared. Then, $\tilde { \rho }$ satisfies

$$
\tilde { \rho } ^ { 2 } = ( 1 - 2 \mu \hat { \rho } ) = 1 - 2 \mu \frac { d } { L } \left( \frac { [ 1 - 4 \tau \delta _ { 0 } ( 1 - \delta _ { 0 } ) ] ^ { 2 } } { ( n - 1 ) ( - \tau + 4 \delta _ { 0 } ( 1 - \delta _ { 0 } ) ( 1 + \tau ) ) + d ( 2 \delta _ { 0 } - 1 ) ^ { 2 } ) } \right) < 1 .
$$

Note that for $f ( x _ { k _ { 1 } + 1 } ) - f ( x _ { k _ { 1 } } )$ , since $g _ { k _ { 1 } } ~ = ~ v _ { k _ { 1 } }$ , we have that the polynomial has a larger root, which means we can use the same step size as for $k _ { 1 } + 1$ step. Therefore, the decay factor will be the same (we use $\tilde { \rho }$ to denote $\tilde { \rho } _ { k _ { 2 } }$ above since we find that this factor is independent of the index). Now, we have shown that given the past, the next step still enjoys a decay with a slightly diferent factor as compared to Phase 1. Nonetheless, the inductive step will still hold and we have $t _ { i } ^ { ( k ) } \leq O ( \tilde { \epsilon } _ { i } ^ { ( k ) } )$ ).

Part 3: Concluding the Proof. Consider the events as follow.

$$
\begin{array} { r l } & { \quad \mathcal { A } ( \tau ) : = \{ \| g _ { k } \| \leq C \rho _ { d e c a y } ^ { k } \quad \forall k \in [ k _ { 0 } , k _ { 1 } ] \} } \\ & { \quad B _ { i } ^ { ( k ) } ( \tau _ { i } ) : = \{ |  \hat { P } _ { k } ^ { \top } u _ { i } , \hat { P } _ { k } ^ { \top } g _ { k }  -  \mathrm { P r o j } _ { v _ { k _ { 1 } } } u _ { 1 } , \mathrm { P r o j } _ { v _ { k _ { 1 } } ^ { \bot } } g _ { k }  |  } \\ & { \qquad \leq { \tau _ { i } } \| \mathrm { P r o j } _ { v _ { k _ { 1 } } ^ { \bot } u _ { 1 } } \| \| \mathrm { P r o j } _ { v _ { k _ { 1 } } ^ { \bot } g _ { k } } \| \} , \quad k \in ( k _ { 1 } , K _ { \epsilon } ) , i \in [ n ] } \\ & { \quad { \mathcal { C } } _ { k } ( \tau ) : = \{ \| \hat { P } _ { k } ^ { \top } g _ { k } \| ^ { 2 } \approx ( 1 \pm \tau ) \| \mathrm { P r o j } _ { \hat { v } _ { k _ { 1 } } g _ { k } } \| ^ { 2 } \} , \quad k \in ( k _ { 1 } , K _ { \epsilon } ) . } \end{array}
$$

where

$$
\begin{array} { c } { { \mathbb { P } [ A _ { 1 } ( \tau ) ] \geq 1 - e ^ { - \frac { 1 } { 8 } k _ { 1 } \left( 1 - p _ { d e c a y } ( \tau , \delta , n , d ) \right) } } } \\ { { \mathbb { P } \left[ \mathcal { B } _ { i } ^ { ( k ) } ( \tau _ { i } ) \right] \geq 1 - 2 e ^ { - c d \tau _ { i } ^ { 2 } } } } \\ { { \mathbb { P } \left[ \mathcal { C } _ { k } ( \tau ) \right] \geq 1 - 2 e ^ { - c d \tau ^ { 2 } } . } } \end{array}
$$

Then, with $\mathcal { F } _ { k } = \sigma ( \tilde { P } _ { 0 } , \cdots , \tilde { P } _ { k - 1 } )$ , we know that $\boldsymbol { \mathcal { A } } ( \tau ) \in \mathcal { F } _ { k _ { 1 } }$ , similarly for $B _ { i } ^ { ( k ) } ( \tau _ { i } ) , \mathcal { C } _ { k } ( \tau ) \in \mathcal { F } _ { k }$ . Let $B _ { k } = \mathcal C _ { k } \cap \left( \bigcap _ { i \in [ n ] } B _ { i } ^ { ( k ) } \right)$ , we have $\boldsymbol { B } _ { k }$ holds with probability at least $\textstyle 1 - 2 \sum _ { i = 1 } ^ { n } e ^ { - c d \tau _ { i } ^ { 2 } } - 2 e ^ { - c d \tau ^ { 2 } }$ Using Lemma 9, we have with probability at least

$$
\left( 1 - 2 \sum _ { i \in [ n ] } e ^ { - c d \tau _ { i } ^ { 2 } } - 2 e ^ { - c d \tau ^ { 2 } } \right) ^ { K _ { \epsilon } - k _ { 1 } } \times \left( 1 - e ^ { - { \frac { 1 } { 8 } } k _ { 1 } \left( 1 - p _ { d e c a y } ( \tau , \delta , n , d ) \right) } \right)\tag{140}
$$

that the projections onto every other directions $\left| u _ { i } ^ { \top } g _ { k } \right| / \left| u _ { 1 } ^ { \top } g _ { k } \right| = O ( ( \lambda _ { i } - \lambda _ { 1 } ) \sqrt { 1 - \delta _ { 0 } } )$ to be much smaller than the projection onto the dominant direction $u _ { 1 }$ □