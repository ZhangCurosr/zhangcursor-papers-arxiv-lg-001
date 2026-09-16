# Near-Optimal Nonconvex Matrix Completion

Jian-Feng Cai<sup>∗</sup> Xiliang Lu<sup>†</sup> Juntao You<sup>‡</sup>

## Abstract

We study nonconvex methods for matrix completion, the problem of recovering a lowrank matrix from a subset of its entries. Convex methods achieve sample complexity linear in the matrix dimension and the rank, up to logarithmic factors, whereas global guarantees for commonly used nonconvex methods require a higher polynomial dependence on the rank. We close this gap by analyzing Riemannian gradient descent (RGD) and Riemannian Gauss–Newton (RGN) methods. For an n n matrix of rank r with incoherence parameter $\mu$ and condition number κ, the two methods achieve exact recovery with high probability from O(µnr log n log(nκ)) and O(µnr log n log(2µrκ)) observations, respectively. The methods use a multiscale residual initialization, while the analysis simultaneously controls the spectral error and incoherence. The resulting RGD iterates converge linearly, whereas RGN eventually converges Q-quadratically.

Keywords. nonconvex matrix completion, sample complexity, Riemannian gradient descent, Riemannian Gauss–Newton method, multiscale initialization, leave-one-out analysis

## 1 Introduction

Matrix completion seeks to recover a low-rank matrix from a subset of its entries and arises in a broad range of problems in machine learning and data analysis, including collaborative filtering [18], dimensionality reduction and clustering [9], model reduction and system identifi cation [20], and sensor network localization [9]. Given an unknown matrix $\ b X _ { \star } \in \mathbb { R } ^ { n \times n }$ of rank r and an observed index set Ω, a natural formulation is

$$
\operatorname* { m i n } _ { \pmb { X } \in \mathbb { R } ^ { n \times n } } \operatorname { r a n k } ( \pmb { X } ) \quad \mathrm { s u b j e c t ~ t o } \quad \mathcal { P } _ { \Omega } ( \pmb { X } ) = \mathcal { P } _ { \Omega } ( \pmb { X } _ { \star } ) ,\tag{1}
$$

where $\mathcal { P } _ { \Omega }$ retains the entries indexed by Ω.

Since rank minimization is computationally intractable in general, the seminal work of Cand\`es and Recht [6] studied the convex relaxation obtained by replacing the rank with the nuclear norm. Cand\`es and Tao [7] proved exact recovery from $O ( \mu _ { \mathrm { s } } ^ { 2 } n r \log ^ { 6 } n )$ randomly observed entries under the strong incoherence condition. Chen [8] later showed that $O ( \mu n r \log ^ { 2 } n )$ observations sufice under standard incoherence. This bound is optimal in its dependence on $\mu ,$ $n ,$ and r, up to logarithmic factors. Nevertheless, nuclear norm minimization can be computationally expensive. To reduce this computational cost, a number of eficient nonconvex methods have been developed that exploit the rank constraint directly. Their global recovery guarantees, however, generally require a higher polynomial dependence on r; see Table 1. A natural question is whether eficient nonconvex methods for matrix completion can attain a sample complexity linear in n and r, up to logarithmic factors.

We address this question by establishing near-optimal sample complexity bounds for RGD and RGN equipped with a multiscale residual initialization. Under standard incoherence condi tion, we establish global recovery guarantees for RGD and RGN, with RGD converging linearly and RGN eventually converging Q-quadratically. With high probability, the two methods recover $X _ { \star }$ with sample complexities

$$
O ( \mu n r \log n \log ( n \kappa ) ) \qquad \mathrm { a n d } \qquad O ( \mu n r \log n \log ( 2 \mu r \kappa ) ) ,
$$

respectively, where $\kappa = \sigma _ { 1 } ( X _ { \star } ) / \sigma _ { r } ( X _ { \star } )$ is the condition number.

Related work. A common nonconvex approach to matrix completion is to factorize $\boldsymbol { X } =$ $\mathbf { \delta } _ { L R } ^ { L R ^ { \top } }$ and optimize over the factors, as in OptSpace [18], alternating minimization [13, 17], and gradient-based methods [3, 23, 29]. For incoherent positive semidefinite matrices, gradient descent [21] converges linearly without explicit regularization while maintaining incoherence along the iterates, and scaled projected gradient descent [4, 22, 25, 32] further removes the dependence of the convergence rate on κ. Projected-gradient and hard-thresholding methods [2, 15, 16, 24] instead work directly with the matrix variable, but require a rank-r approximation after each gradient step. Riemannian methods [28, 31] avoid this large-scale truncation by restricting the search direction to the tangent space of the fixed-rank manifold. Second-order methods have also been studied, including MatrixIRLS [19] and Gauss–Newton methods [27, 33, 34]. Despite these developments, the known global recovery guarantees generally retain a higher polynomial dependence on the rank; for example, Riemannian gradient descent [31] requires $O ( \mu \kappa ^ { 6 } n r ^ { 2 } \log ^ { 2 } n )$ observations under certain conditions. In contrast, for Gaussian matrix sensing, Riemannian gradient descent [5] attains the optimal $O ( n r )$ sample complexity. For matrix completion, however, the sampling operator does not satisfy a uniform restricted isometry over low-rank matrices, and the iterates must be shown to remain incoherent. This makes the anal ysis more challenging.

Another issue is the dependence on the condition number in initialization. The ordinary spectral estimator, widely used in nonconvex methods, can incur an additional factor $\kappa ^ { 2 }$ in the sampling requirement (see Theorem 3.4 in this work). Stagewise methods such as SoftDeflate [14] and the projected-gradient method [16] reduce or remove this dependence, but the recovery guarantees still have a higher polynomial dependence on $^ { r } \cdot$ Our multiscale initialization instead controls the spectral, row, column, and entrywise errors simultaneously, which avoids an additional polynomial dependence on r in the sampling requirement.

Our contributions and proof strategy. Our main contribution is to establish near-optimal global sample complexity bounds for eficient nonconvex RGD and RGN methods in matrix completion. As discussed above, two sources of additional sample complexity arise in the usual analysis: the conversion from spectral to Frobenius error can introduce an additional factor $r ,$ while ordinary spectral initialization can incur a factor $\kappa ^ { 2 }$

To control the loss in rank $r ,$ we keep track of the spectral, row, column, and entrywise errors simultaneously. For $\ b { Z } \in \mathbb { R } ^ { n \times n }$ , define

$$
\left\| Z \right\| _ { 2 , \infty } : = \operatorname* { m a x } _ { i } \left\| e _ { i } ^ { T } Z \right\| _ { 2 } , \qquad \left\| Z ^ { T } \right\| _ { 2 , \infty } : = \operatorname* { m a x } _ { j } \left\| Z e _ { j } \right\| _ { 2 } , \qquad \left\| Z \right\| _ { \infty } : = \operatorname* { m a x } _ { i , j } \left| ( Z ) _ { i j } \right| ,
$$

and introduce the sharp-norm as

$$
\left\| Z \right\| _ { \sharp } : = \operatorname* { m a x } \left\{ \left\| Z \right\| _ { \mathrm { o p } } , \frac { 1 } { 2 } \sqrt { \frac { n } { \mu r } } \left\| Z \right\| _ { 2 , \infty } , \frac { 1 } { 2 } \sqrt { \frac { n } { \mu r } } \left\| Z ^ { T } \right\| _ { 2 , \infty } , \frac { n } { 4 \mu r } \left\| Z \right\| _ { \infty } \right\} .\tag{2}
$$

The multiscale initialization maintains this sharp-norm control and reaches the required local regions without introducing an additional polynomial dependence on r in the sampling requirement. For RGD, the eventual conversion to the Frobenius norm afects only the number of initialization steps; for RGN, the sharp-norm control is continued through the initial iterations before the analysis enters the Frobenius regime.

Table 1: Complexity comparison for matrix completion under the stated recovery guarantees. All results assume incoherence, while Factorized GD [21] also assumes PSD and $\kappa = O ( 1 )$ ; Riemannian GD [31] additionally assumes spikiness.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Iterationcomplexity</td><td rowspan=1 colspan=1>Cost periteration</td><td rowspan=1 colspan=1>Samplecomplexity</td><td rowspan=1 colspan=1>Total computationalcomplexity</td></tr><tr><td rowspan=1 colspan=1>Nuclear normminimization[7, 8, 10]</td><td rowspan=1 colspan=1>Polynomialtime; solverdependent</td><td rowspan=1 colspan=1>Solver dependent</td><td rowspan=1 colspan=1> $O ( \mu n r \log ( 2 \mu r ) \log n )$ </td><td rowspan=1 colspan=1>Solver dependent</td></tr><tr><td rowspan=1 colspan=1>FactorizedGD[21]</td><td rowspan=1 colspan=1> $O \left( \kappa ^ { 2 } \log \frac { 1 } { \varepsilon } \right)$ </td><td rowspan=1 colspan=1> $O ( | \Omega | r + n r )$ </td><td rowspan=1 colspan=1> $O ( \mu ^ { 3 } n r ^ { 3 } \log ^ { 3 } n )$ </td><td rowspan=1 colspan=1> $O \left( \mu ^ { 3 } \kappa ^ { 2 } n r ^ { 4 } \log ^ { 3 } n \log \frac { 1 } { \varepsilon } \right)$ </td></tr><tr><td rowspan=1 colspan=1>SVP/PGD[10, 15, 30]</td><td rowspan=1 colspan=1> $\begin{array} { r } { O \left( \log \frac { 1 } { \varepsilon } \right) } \end{array}$ </td><td rowspan=1 colspan=1> $O ( n ^ { 3 } )$ </td><td rowspan=1 colspan=1> $O ( \mu ^ { 2 } \kappa ^ { 4 } n r ^ { 2 } \log n )$ </td><td rowspan=1 colspan=1> $O \left( n ^ { 3 } \log { \frac { 1 } { \varepsilon } } \right)$ </td></tr><tr><td rowspan=1 colspan=1>ScaledPGD[25]</td><td rowspan=1 colspan=1> $\begin{array} { r } { O \left( \log \frac { 1 } { \varepsilon } \right) } \end{array}$ </td><td rowspan=1 colspan=1> $O ( | \Omega | r + n r ^ { 2 } )$ </td><td rowspan=1 colspan=1> $O \big ( \mu \kappa ^ { 2 } n r ^ { 2 } ( \mu \kappa ^ { 2 } \vee \log n ) \big )$ </td><td rowspan=1 colspan=1> $\begin{array} { r } { O \big ( \mu \kappa ^ { 2 } n r ^ { 3 } ( \mu \kappa ^ { 2 } \vee \log n ) \log \frac { 1 } { \varepsilon } \big ) } \end{array}$ </td></tr><tr><td rowspan=1 colspan=1>RiemannianGD[31]</td><td rowspan=1 colspan=1> $\begin{array} { r } { O \left( \log \frac { 1 } { \varepsilon } \right) } \end{array}$ </td><td rowspan=1 colspan=1> $O ( | \Omega | r + n r ^ { 2 } )$ </td><td rowspan=1 colspan=1> $O \big ( \operatorname* { m a x } \{ \mu _ { 0 } , \mu _ { 1 } ^ { 2 } \} \kappa ^ { 6 } n r ^ { 2 } \log ^ { 2 } n \big )$ </td><td rowspan=1 colspan=1> $O \big ( \operatorname* { m a x } \{ \mu _ { 0 } , \mu _ { 1 } ^ { 2 } \} \kappa ^ { 6 } n r ^ { 3 } \log ^ { 2 } n \log \frac { 1 } { \varepsilon } \big )$ </td></tr><tr><td rowspan=1 colspan=1>RGD(this paper)</td><td rowspan=1 colspan=1> $O \left( \log { \frac { 1 } { \varepsilon } } \right)$ </td><td rowspan=1 colspan=1> $O ( | \Omega | r + n r ^ { 2 } )$ </td><td rowspan=1 colspan=1> $O ( \mu n r \log n \log ( n \kappa ) )$ </td><td rowspan=1 colspan=1> $\begin{array} { r } { O \left( \mu n r ^ { 2 } \log ^ { 2 } ( n \kappa ) \left( \log n + \log \frac { 1 } { \varepsilon } \right) \right) } \end{array}$ </td></tr><tr><td rowspan=1 colspan=1>RGN(this paper)</td><td rowspan=1 colspan=1> $\begin{array} { r } { O \left( \log \log \log \frac { 1 } { \varepsilon } \right) } \end{array}$ </td><td rowspan=1 colspan=1> $O \Big ( J _ { k } \big ( | \widehat \Omega | r + n r ^ { 2 } \big ) \Big )$ </td><td rowspan=1 colspan=1> $O ( \mu n r \log n \log ( 2 \mu r \kappa ) )$ </td><td rowspan=1 colspan=1> $\begin{array} { r } { O \left( \mu J n r ^ { 2 } \log ^ { 2 } n \log ( 2 \mu r \kappa ) \log \frac { 1 } { \varepsilon } \right) } \end{array}$ </td></tr></table>

To control the loss in $\kappa ^ { 2 }$ , we use a multiscale residual initialization in place of the ordinary spectral estimator. Successive residual reconstructions reduce the sharp-norm error by a fixed factor at the sampling level $p \gtrsim \mu r$ log $n / n ,$ and we establish both the sampling and computational complexities of this procedure. We further show that the $\mu \kappa ^ { 2 } r / n$ sampling scale of the ordinary spectral estimator is necessary over an explicit family of incoherent matrices.

The proof combines these initialization estimates with the local convergence analysis. For RGD, the initialization reaches the required Frobenius neighborhood, where a uniform tangentspace sampling estimate yields linear convergence. For RGN, a finite leave-one-out argument propagates the sharp-norm control through the initial iterations; once the iterates enter a suficiently small Frobenius neighborhood, a local deterministic argument yields quadratic convergence.

Organization and notation. Section 2 introduces the observation model, the Riemannian algorithms, and the multiscale initialization. Section 3 states the global recovery guarantees and the lower bound for ordinary spectral initialization. Section 4 presents the proof framework, including the local convergence and initialization results. Their proofs, together with the proofs of the global theorems, are given in Sections 5–7. Sections 8 and 9 present the numerical experiments and concluding remarks. The appendices collect the supporting probabilistic and geometric estimates, the proof of the spectral lower bound, the sharp-norm estimates for spectral reconstruction, the leave-one-out analysis, and the implementation and complexity analysis.

Throughout the paper, bold lowercase letters denote vectors and bold uppercase letters denote matrices, while scalars are written in ordinary type. The vector $e _ { i }$ denotes the ith standard basis vector. For a vector $x , \ \| x \| _ { 2 }$ denotes the Euclidean norm. For a matrix $\boldsymbol { z }$ $\| Z \| _ { \mathrm { o p } }$ and $\Vert Z \Vert _ { \mathrm { F } }$ denote the operator norm and Frobenius norm respectively. We write $\sigma _ { i } ( Z )$ for the i-th largest singular value of $Z ,$ and $\operatorname { r a n k } ( Z )$ and $\mathrm { r a n g e } ( Z )$ for its rank and column space. The symbols I and  denote the identity matrix and identity operator, respectively. We use

O( ) for bounds up to an absolute numerical constant independent of the problem parameters.

## 2 Problem Formulation and Riemannian Algorithms

We first formulate the matrix completion problem and then describe the Riemannian algorithms considered in this paper.

## 2.1 Problem setup

Suppose that $\pmb { X } _ { \star } \in \mathbb { R } ^ { n \times n } , n \geq 2$ is an unknown matrix of rank r, where $1 \leq r < n$ . Assume we observe its entries independently according to the Bernoulli sampling model:

$$
Y _ { i j } = \left\{ \begin{array} { l l } { ( X _ { \star } ) _ { i j } , } & { \mathrm { w i t h ~ p r o b a b i l i t y } ~ p , } \\ { * , } & { \mathrm { w i t h ~ p r o b a b i l i t y } ~ 1 - p , } \end{array} \right. \quad 1 \leq i , j \leq n ,
$$

where $0 < p \le 1$ . Let $\Omega : = \{ ( i , j ) : Y _ { i j } \neq * \}$ denote the set of observed indices, also denoted by $\Omega \sim \mathrm { B e r n o u l l i } ( p )$ . The associated sampling operator $\mathcal { P } _ { \Omega }$ is defined by

$$
( { \mathcal { P } } _ { \Omega } ( Z ) ) _ { i j } = \left\{ Z _ { i j } , \begin{array} { l l } { ( i , j ) \in \Omega , } \\ { 0 , } \end{array} \right.
$$

The matrix completion problem is to recover $X _ { \star }$ from the observed entries $\mathcal { P } _ { \Omega } ( X _ { \star } )$ . Let

$$
X _ { \star } = U _ { \star } \Sigma _ { \star } V _ { \star } ^ { T } , \qquad \Sigma _ { \star } = \mathrm { d i a g } ( \sigma _ { 1 } , \dots , \sigma _ { r } ) , \qquad \sigma _ { 1 } \geq \cdot \cdot \cdot \geq \sigma _ { r } > 0 ,
$$

be a compact singular value decomposition of $X _ { \star }$ . We assume that $X _ { \star }$ satisfies the following standard incoherence condition.

Assumption 2.1 (Incoherence [6–8]). For some $1 \leq \mu \leq n / r$ , it holds

$$
\operatorname* { m a x } \left\{ \operatorname* { m a x } _ { i } \left\| \pmb { U } _ { \star } ^ { T } \pmb { e } _ { i } \right\| _ { 2 } ^ { 2 } , \operatorname* { m a x } _ { j } \left\| \pmb { V } _ { \star } ^ { T } \pmb { e } _ { j } \right\| _ { 2 } ^ { 2 } \right\} \leq \frac { \mu r } { n } .
$$

The standard incoherence condition was introduced by Cand\`es and Recht [6] for low-rank matrix completion. It requires the left and right singular spaces of X<sub>⋆</sub> to be weakly correlated with the canonical basis, preventing the matrix from being concentrated on only a few entries. For the completion problem, we consider the following nonconvex formulation:

$$
\operatorname* { m i n } _ { \pmb { X } \in \mathcal { M } _ { r } } f _ { \Omega } ( \pmb { X } ) , \qquad f _ { \Omega } ( \pmb { X } ) : = \frac { 1 } { 2 p } \left\| \mathcal { P } _ { \Omega } ( \pmb { X } - \pmb { X } _ { \star } ) \right\| _ { \mathrm { F } } ^ { 2 } ,\tag{3}
$$

where

$$
\mathcal { M } _ { r } : = \{ X \in \mathbb { R } ^ { n \times n } : \operatorname { r a n k } ( X ) = r \}
$$

is the manifold of rank-r matrices.

## 2.2 Riemannian gradient descent

We first present the RGD method for (3). The algorithm updates the current iterate along the negative gradient direction in the tangent space of the fixed-rank manifold and then retracts the tangent update back onto the manifold. We first recall the geometry of the fixed low-rank manifold. Let $\pmb { X } = \pmb { U } \pmb { \Sigma } \pmb { V } ^ { T } \in \mathcal { M } _ { r }$ be a compact singular value decomposition. The tangent space of $\mathcal { M } _ { r }$ at X is [28]

$$
T _ { \mathbf {  { X } } } \mathbf {  { \mathcal { M } } } _ { r } = \left\{ U Z _ { 1 } ^ { T } + Z _ { 2 } V ^ { T } : Z _ { 1 } , Z _ { 2 } \in \mathbb { R } ^ { n \times r } \right\} ,
$$

and the orthogonal projection onto $T _ { X } { \mathcal { M } } _ { r }$ is

$$
\mathcal { P } _ { T _ { X } } ( Z ) = U U ^ { T } Z + Z V V ^ { T } - U U ^ { T } Z V V ^ { T } .\tag{4}
$$

Since $\nabla f _ { \Omega } ( { \pmb X } ) = p ^ { - 1 } \mathcal { P } _ { \Omega } ( { \pmb X } - { \pmb X } _ { \star } )$ , the Riemannian gradient of $f _ { \Omega }$ at X is

$$
\mathrm { g r a d } f _ { \Omega } ( { \pmb X } ) = \mathcal { P } _ { T _ { \pmb X } } \nabla f _ { \Omega } ( { \pmb X } ) = p ^ { - 1 } \mathcal { P } _ { T _ { \pmb X } } \mathcal { P } _ { \Omega } ( { \pmb X } - { \pmb X } _ { \star } ) .
$$

A tangent update does not in general belong to $\mathcal { M } _ { r }$ . We use the orthographic retraction [1] to map it back onto the fixed-rank manifold. For $\pmb { \xi } \in T _ { \pmb { X } } \mathcal { M } _ { r }$ , define

$$
\operatorname { R e t r } _ { \boldsymbol { X } } ( \boldsymbol { \xi } ) : = ( \boldsymbol { X } + \boldsymbol { \xi } ) \boldsymbol { V } \big [ \boldsymbol { U } ^ { T } ( \boldsymbol { X } + \boldsymbol { \xi } ) \boldsymbol { V } \big ] ^ { - 1 } \boldsymbol { U } ^ { T } ( \boldsymbol { X } + \boldsymbol { \xi } ) ,\tag{5}
$$

whenever $U ^ { T } ( \pmb { X } + \pmb { \xi } ) \pmb { V }$ is nonsingular. Let $\xi _ { k } \ = \ -$ grad $f _ { \Omega } ( \boldsymbol { X } _ { k } )$ . Since $\langle \nabla f _ { \Omega } ( \mathbf { { X } } _ { k } ) , \pmb { \xi } _ { k } \rangle \ =$ $\| \pmb { \xi } _ { k } \| _ { \mathrm { F } } ^ { 2 }$ , exact line search along this tangent direction [31] gives

$$
\alpha _ { k } = \arg \operatorname* { m i n } _ { \alpha \in \mathbb { R } } f _ { \Omega } ( X _ { k } + \alpha \pmb { \xi } _ { k } ) = \frac { \| \pmb { \xi } _ { k } \| _ { \mathrm { F } } ^ { 2 } } { p ^ { - 1 } \| \mathcal { P } _ { \Omega } ( \pmb { \xi } _ { k } ) \| _ { \mathrm { F } } ^ { 2 } } ,\tag{6}
$$

whenever $\xi _ { k } \neq 0$ . The RGD update is then

$$
X _ { k + 1 } = \operatorname { R e t r } _ { X _ { k } } ( \alpha _ { k } \pmb { \xi } _ { k } ) ,
$$

as summarized in Algorithm 1. If $\pmb { \xi } _ { k } = \mathbf { 0 }$ , the algorithm terminates.

Algorithm 1 Riemannian gradient descent (RGD)   
Input: $p , \mathcal { P } _ { \Omega } ( \mathbf { { X } } _ { \star } )$ , and $\pmb { X _ { 0 } } \in \mathcal { M } _ { r }$   
1: for $k = 0 , 1 , 2 , \ldots$ do   
2: $\pmb { \xi } _ { k } \gets - p ^ { - 1 } \mathcal { P } _ { T _ { \pmb { X } _ { k } } } \left( \mathcal { P } _ { \Omega } ( \pmb { X } _ { k } ) - \mathcal { P } _ { \Omega } ( \pmb { X } _ { \star } ) \right)$   
3: $\alpha _ { k }  \Vert \pmb { \xi } _ { k } \Vert _ { \mathrm { F } } ^ { 2 } / \bigl ( p ^ { - 1 } \Vert \mathscr { P } _ { \Omega } ( \pmb { \xi } _ { k } ) \Vert _ { \mathrm { F } } ^ { 2 } \bigr ) \mathrm { i f } \ \pmb { \xi } _ { k } \neq \mathbf { 0 } ;$ otherwise, stop.   
4: $X _ { k + 1 }  \operatorname { R e t r } _ { X _ { k } } ( \alpha _ { k } \pmb { \xi } _ { k } ) .$   
5: end for

The tangent gradient can be evaluated using sparse matrix–factor products in $O ( | \boldsymbol { \Omega } | \boldsymbol { r } + n \boldsymbol { r } ^ { 2 } )$ operations. The exact line-search stepsize can be evaluated in the same order by computing $\| \pmb { \xi } _ { k } \| _ { \mathrm { F } }$ and the entries of $\xi _ { k }$ on Ω. The orthographic retraction can be computed from low-rank factors using two thin QR factorizations and an $r \times r$ singular value decomposition in $O ( n r ^ { 2 } + r ^ { 3 } )$ operations [1]. Thus, since $r < n .$ , each RGD iteration costs $O ( | \Omega | r + n r ^ { 2 } )$ operations. The implementation details are given in Section F.

## 2.3 Riemannian Gauss–Newton

Riemannian Gauss–Newton first computes a search direction in the tangent space and then retracts the tangent update back onto $\mathcal { M } _ { r }$ . Let $\widehat { \Omega } \subset \Omega$ denote the subset of observations used for the RGN iterations. At $\pmb { X } \in \mathcal { M } _ { r }$ , since

$$
D \operatorname { R e t r } _ { X } ( 0 ) [ \pmb { \xi } ] = \pmb { \xi } , \qquad \pmb { \xi } \in T _ { X } \mathcal { M } _ { r } ,
$$

linearizing the sampled residual along the retraction gives the Gauss–Newton subproblem

$$
\operatorname* { m i n } _ { \pmb { \xi } \in T _ { \pmb { X } } \mathcal { M } _ { r } } \frac { 1 } { 2 } \left\| \mathcal { P } _ { \widehat { \Omega } } ( \pmb { X } - \pmb { X } _ { \star } + \pmb { \xi } ) \right\| _ { \mathrm { F } } ^ { 2 } .\tag{7}
$$

Its first-order optimality condition is the tangent normal equation

$$
\mathcal { P } _ { T _ { \mathbf { X } } } \mathcal { P } _ { \widehat { \Omega } } \mathcal { P } _ { T _ { \mathbf { X } } } \pmb { \xi } = \mathcal { P } _ { T _ { \mathbf { X } } } \mathcal { P } _ { \widehat { \Omega } } ( \pmb { X } _ { \star } - \pmb { X } ) .\tag{8}
$$

When the sampled tangent normal operator is positive definite on $T _ {  { \mathbf { X } } }  { \mathcal { M } } _ { r } , \ ( 7 )$ has a unique minimizer. We compute this direction by applying the conjugate gradient method to (8) on $T _ { X } . M _ { r }$ . At the k-th nonterminal RGN iteration, let $J _ { k } \geq 1$ denote the number of CG iterations used to solve the normal equation. CG is started from zero and run to the exact solution. If the right-hand side is zero, the algorithm terminates; the sequence is then continued by its final iterate for the convergence statements. In exact case, the finite-termination property of CG [11, Theorems 2.3.2 and 3.1.1] gives

$$
J _ { k } \leq \dim ( T _ { X _ { k } } { \mathcal { M } } _ { r } ) = r ( 2 n - r ) .
$$

The resulting iteration is summarized in Algorithm 2.

Algorithm 2 Riemannian Gauss–Newton (RGN)   
Input: $\overline { { \mathcal { P } _ { \widehat { \Omega } } ( \pmb { X } _ { \star } ) } }$ and $\pmb { X _ { 0 } } \in \mathcal { M } _ { r }$   
1: for $k = 0 , \bar { 1 } , 2 , . . .$ . do   
2: Apply $J _ { k }$ CG iterations, starting from zero, to solve the following normal equation for   
$\xi _ { k } \mathrm { : }$   
$\mathcal { P } _ { T _ { \mathbf { X } _ { k } } } \mathcal { P } _ { \widehat { \Omega } } \mathcal { P } _ { T _ { \mathbf { X } _ { k } } } \pmb { \xi } = \mathcal { P } _ { T _ { \mathbf { X } _ { k } } } \mathcal { P } _ { \widehat { \Omega } } ( \mathbf { X } _ { \star } - \mathbf { X } _ { k } ) .$   
3: $X _ { k + 1 }  \operatorname { R e t r } _ { X _ { k } } ( \pmb { \xi } _ { k } )$   
4: end for

We also consider a regularized variant of RGN. At the k-th iteration, define

$$
\lambda _ { k } = { \frac { 1 } { \sigma _ { r } ( X _ { k } ) } } \left\| \operatorname { \mathcal { P } } _ { T _ { X _ { k } } } { \mathcal { P } } _ { { \widehat { \Omega } } } ( X _ { \star } - X _ { k } ) \right\| _ { \mathrm { F } }\tag{9}
$$

and add $\lambda _ { k } \pmb { \xi }$ to the left-hand side of the normal equation (8). At every nonterminal iteration, $\lambda _ { k } > 0$ , so the regularized normal equation is positive definite on $T _ { { X _ { k } } } { \mathcal { M } } _ { r }$ . Using the same CG and termination conventions as above gives Algorithm 3.

Algorithm 3 Regularized Riemannian Gauss–Newton   
Input: $\overline { { \mathcal { P } _ { \widehat { \Omega } } ( \pmb { X } _ { \star } ) } }$ and $\pmb { X _ { 0 } } \in \mathcal { M } _ { r }$   
1: for $k = 0 , 1 , 2 , \ldots$ do   
2: $\lambda _ { k }  \frac { 1 } { \sigma _ { r } ( X _ { k } ) } \| \mathcal { P } _ { T _ { X _ { k } } } \mathcal { P } _ { \widehat { \Omega } } ( X _ { \star } - X _ { k } ) \| _ { \mathrm { F } } .$   
3: Apply $J _ { k }$ CG iterations, starting from zero, to solve the following normal equation for   
$\xi _ { k } \mathrm { : }$   
$\mathcal { P } _ { T _ { \mathbf { X } _ { k } } } \mathcal { P } _ { \widehat { \Omega } } \mathcal { P } _ { T _ { \mathbf { X } _ { k } } } \xi + \lambda _ { k } \xi = \mathcal { P } _ { T _ { \mathbf { X } _ { k } } } \mathcal { P } _ { \widehat { \Omega } } ( { \mathbf { X } _ { \star } } - { \mathbf { X } _ { k } } ) .$   
4: $X _ { k + 1 } $ Retr ${ \bf \sigma } _ { X _ { k } } ( \pmb { \xi } _ { k } )$   
5: end for

The normal equations in Algorithms 2 and 3 can be solved matrix-free without forming the sampled tangent normal matrix. Each normal-operator application costs $O ( | \widehat { \Omega } | r + n r ^ { 2 } )$ operabtions, and the regularization term in Algorithm 3 does not change this order. The orthographic retraction costs ${ \dot { O } } ( n r ^ { 2 } + r ^ { 3 } )$ operations. Hence, since $r < n$ and $J _ { k } \geq 1$ , the k-th iteration of either method costs

$$
O \Big ( J _ { k } \big ( | \widehat \Omega | r + n r ^ { 2 } \big ) \Big )
$$

operations. The matrix-free implementation and detailed complexity analysis are given in Section F.

## 2.4 Multiscale initialization

The local convergence results for both RGD and RGN require an initial point suficiently close to $X _ { \star }$ . By Theorem 3.4, the standard spectral estimator can require a sampling probability of order $\mu \kappa ^ { 2 } r / n$ to reach a constant sharp-norm neighborhood. We instead use a multiscale residual spectral initialization, related to residual spectral updates in singular value projection and iterative hard thresholding [2, 15, 24] and to stagewise constructions for matrix completion [14, 16].

At each scale, the current approximation is corrected by the observed residual and then truncated at a decreasing spectral level. Let $\Omega ^ { ( \ell ) } \subset \Omega$ denote the observations used at scale $\ell ,$ with sampling probability $q .$ . Starting from ${ \cal Z } _ { 0 } \ = \ { \bf 0 }$ , set $\tau _ { 0 } : = 2 q ^ { - 1 / 2 } \| \mathcal { P } _ { \Omega ^ { ( 0 ) } } ( \mathbf { X } _ { \star } ) \| _ { \mathrm { F } }$ and $\tau _ { \ell } : = 4 ^ { - \ell } \tau _ { 0 }$ . At iteration $\ell ,$ the residual correction satisfies

$$
Z _ { \ell } + q ^ { - 1 } \mathcal { P } _ { \Omega ^ { ( \ell + 1 ) } } ( X _ { \star } - Z _ { \ell } ) = X _ { \star } + \big ( q ^ { - 1 } \mathcal { P } _ { \Omega ^ { ( \ell + 1 ) } } - \mathcal { Z } \big ) ( X _ { \star } - Z _ { \ell } ) .
$$

Thus, the sampling perturbation acts on the current error $X _ { \star } - Z _ { \ell }$ . Accordingly, we use the following update:

$$
{ \pmb Z } _ { \ell + 1 } = \mathcal T _ { \tau _ { \ell } } \big ( { \pmb Z } _ { \ell } + q ^ { - 1 } \mathcal P _ { \Omega ^ { ( \ell + 1 ) } } \big ( { \pmb X } _ { \star } - { \pmb Z } _ { \ell } \big ) \big ) , \qquad 0 \le \ell < K ,\tag{10}
$$

where $\mathcal { T } _ { \tau _ { \ell } }$ denotes the spectral truncation operator defined below. For a singular value decomposition $\begin{array} { r } { \mathbf { \tilde { Y } } = \sum _ { j } \sigma _ { j } \pmb { u } _ { j } \pmb { v } _ { j } ^ { T } } \end{array}$ , first define the hard spectral thresholding operator

$$
\mathcal { H } _ { \ge \lambda } ( Y ) : = \sum _ { \sigma _ { j } \ge \lambda } \sigma _ { j } \pmb { u } _ { j } \pmb { v } _ { j } ^ { T } .
$$

For $\tau > 0 , \mathcal { T } _ { \tau } ( Y )$ is computed as follows: Starting from $Q _ { 0 } = \operatorname { q f } ( G )$ , where $G \in \mathbb { R } ^ { n \times r }$ has independent standard Gaussian entries, compute

$$
\pmb { Q } _ { t + 1 } = \mathrm { q f } \left( \pmb { Y } ( \pmb { Y } ^ { T } \pmb { Q } _ { t } ) + \frac { \tau ^ { 2 } } { 4 0 9 6 } \pmb { Q } _ { t } \right) , \qquad 0 \le t < \left\lceil 1 2 \log n \right\rceil ,\tag{11}
$$

where $\mathrm { q f }$ denotes the orthogonal factor in a thin QR factorization. The term $\frac { \tau ^ { 2 } } { 4 0 9 6 } Q _ { t }$ preserves the eigenspaces of $\mathbf { \nabla } _ { \mathbf { Y } \mathbf { Y } } T$ and keeps the block iteration well defined. Writing $Q$ for the final factor, define

$$
\mathcal { T } _ { \tau } ( Y ) : = Q \mathcal { H } _ { \geq \tau / 8 } ( Q ^ { T } Y ) .\tag{12}
$$

Thus each reconstruction uses matrix–factor products and a compressed singular value decomposition. Its sharp-norm approximation property is established in Theorem 4.3, and the computational cost of one reconstruction is as follows.

Proposition 2.2. Suppose that rank $( Z ) \leq r$ and $| \Lambda | = m$ . Then

$$
\mathcal { T } _ { \tau } \big ( Z + q ^ { - 1 } \mathcal { P } _ { \Lambda } ( X _ { \star } - Z ) \big )
$$

can be computed in matrix-free form using $O \big ( ( m r + n r ^ { 2 } ) \log n \big )$ operations. Evaluating the observed residual costs $O ( m r )$ additional operations.

Proof. The proof is given in Subsection F.1.

As for the topping rule in the above initialization stage, we adopt the following residual test. For the rank-r iterates, we use the observed residual

$$
R _ { \ell } : = q ^ { - 1 / 2 } \| \mathcal { P } _ { \Omega ^ { ( \ell + 1 ) } } ( X _ { \star } - \pmb { Z } _ { \ell } ) \| _ { \mathrm { F } } ,
$$

and stop the initialization at $R _ { \ell } { } ^ { \cdot } \mathrm { s }$ first increase or after K reconstructions. The output is denoted by $\boldsymbol { z } _ { \widehat { K } }$ . The procedure is summarized in Algorithm 4.

Algorithm 4 Multiscale initialization   
Input: rank $r ,$ sampling probability q, maximum number of reconstructions K, and   
$\{ { \mathcal { P } } _ { \Omega ^ { ( \ell ) } } ( X _ { \star } ) \} _ { \ell = 0 } ^ { K } .$   
1: $Z _ { 0 } \gets \mathbf { 0 } , \tau _ { 0 } \gets 2 q ^ { - 1 / 2 } \left. \mathscr { P } _ { \Omega ^ { ( 0 ) } } ( \mathbf { X } _ { \star } ) \right. _ { \mathrm { F } } , j \gets \emptyset$   
2: for $\ell = 0 , \dots , K - 1$ do   
3: ${ \pmb Z } _ { \ell + 1 }  T _ { \tau _ { \ell } } \big ( { \pmb Z } _ { \ell } + q ^ { - 1 } { \mathcal { P } } _ { \Omega ^ { ( \ell + 1 ) } } \big ( { \pmb X } _ { \star } - { \pmb Z } _ { \ell } \big ) \big )$   
4: $\tau _ { \ell + 1 } \gets \tau _ { \ell } / 4 .$   
5: If $\ell + 1 < K$ , set $R _ { \ell + 1 }  q ^ { - 1 / 2 } \| \mathcal { P } _ { \Omega ^ { ( \ell + 2 ) } } ( X _ { \star } - \mathbf { Z } _ { \ell + 1 } ) \| _ { \mathrm { F } } .$   
6: Stop and return $Z _ { j }$ if $\ell + 1 < K , j \neq \emptyset .$ , rank $\textstyle ( Z _ { \ell + 1 } ) = r$ , and $R _ { \ell + 1 } > R _ { j }$   
7: Set $j  \ell + 1 \mathrm { ~ i f ~ } \ell + 1 < K$ and rank $\begin{array} { r } { ( Z _ { \ell + 1 } ) = r . } \end{array}$   
8: end for   
9: return $Z _ { K }$

We now specify the observation sets used in the analysis. Set

$$
B : = \left\{ K + 1 , \mathrm { \quad f o r ~ R G D } , \mathrm { \qquad ~ } q : = 1 - ( 1 - p ) ^ { 1 / B } . \right.
$$

The two choices of the upper bound K are specified in Section 3. For each $( i , j ) \in \Omega$ , independently draw $b _ { i j } \in \{ 0 , 1 \} ^ { B } \backslash \{ \mathbf { 0 } \}$ from the product Bernoulli(q) distribution conditioned on being nonzero, and set $\Omega ^ { ( \ell ) } : = \{ ( i , j ) \in \Omega : ( b _ { i j } ) _ { \ell } = 1 \}$ for $0 \leq \ell \leq B - 1$ . Under the unconditional law, $\Omega ^ { ( 0 ) } , \ldots , \Omega ^ { ( B - 1 ) }$ are mutually independent Bernoull $\mathrm { i } ( q )$ subsets with $\begin{array} { r } { \Omega = \bigcup _ { \ell = 0 } ^ { B - 1 } \Omega ^ { ( \ell ) } } \end{array}$ , where the subsets may not be disjoint. The first $K + 1$ components are used for initialization. For RGN, the last component $\widehat \Omega : = \Omega ^ { ( K + 1 ) }$ is reserved for the subsequent iterations.

Remark 2.3. The above decomposition is used only in the analysis. In numerical implementation, the same observation set Ω is used throughout the algorithms, with q replaced by $p .$

## 3 Main Results

In this section, we state the global recovery guarantees for RGD and RGN, and then give a lower bound for ordinary spectral initialization.

## 3.1 Global convergence of RGD

For RGD, let $X _ { 0 }$ be the output of Algorithm 4 with the upper bound

$$
K = \left\lceil 5 + \log _ { 4 } ( \kappa \sqrt { n r } ) \right\rceil .
$$

Starting from $X _ { 0 }$ , let $\{ X _ { k } \} _ { k \ge 0 }$ be the iterates of Algorithm 1 on Ω. The following theorem establishes linear convergence from this initialization.

Theorem 3.1 (Global convergence of RGD). Suppose that Assumption 2.1 holds and $\Omega \sim$ Bernoulli(p). Let $X _ { 0 }$ be the output of Algorithm 4 with K specified above, and let $\{ X _ { k } \} _ { k \ge 0 }$ be generated by Algorithm 1. If

$$
p \geq C _ { 1 } \frac { \mu r \log n \log ( n \kappa ) } { n } ,
$$

where $C _ { 1 } > 0$ is a suficiently large absolute constant, then, with probability at least $1 - n ^ { - 1 0 }$ the initialization and all subsequent iterates are well defined, and

$$
\| X _ { k } - X _ { \star } \| _ { \mathrm { F } } \leq \left( \frac 3 8 \right) ^ { k } \| X _ { 0 } - X _ { \star } \| _ { \mathrm { F } } , \qquad k \geq 0 .
$$

Proof. The proof is deferred to Subsection 6.5.

Thus O(µnr log n log(nκ)) observations sufice for exact recovery with high probability. After Algorithm 4, relative Frobenius accuracy ε is attained within $O ( \log ( 1 / \varepsilon ) )$ RGD iterations. The complete initialization and computational costs are given in Theorem F.1.

## 3.2 Global convergence of RGN

For RGN, set

$$
{ \cal K } = \Big \lceil 6 + \log _ { 4 } ( { \mu r ^ { 3 / 2 } { \kappa } } ) \Big \rceil , \qquad q = 1 - ( 1 - p ) ^ { 1 / ( { K } + 2 ) } , \qquad \bar { \cal K } : = \lceil \log _ { 2 } \log _ { 2 } ( 4 n ) \rceil .
$$

We first run Algorithm 4 with this upper bound K using the RGN subsets specified in Subsection 2.4, and then run Algorithm 2 on $\widehat { \Omega } .$ The following theorem establishes global bconvergence and eventual quadratic convergence of the resulting RGN iterates.

Theorem 3.2 (Global convergence of RGN). Suppose that Assumption 2.1 holds and $\Omega \sim$ Bernoulli $( p )$ . Let $X _ { 0 }$ denote the output of Algorithm $\it 4$ with K specified above, and let $\{ X _ { k } \} _ { k \ge 0 }$ be generated by Algorithm $\mathcal { Q }$ on $\widehat { \Omega }$ . If

$$
p \geq C _ { 2 } \frac { \mu r \log n \log ( 2 \mu r \kappa ) } { n } ,
$$

where $C _ { 2 } > 0$ is a suficiently large absolute constant, then, with probability at least $1 - n ^ { - 1 0 }$ the initialization and all subsequent iterates are well defined. Moreover,

$$
\| X _ { k } - X _ { \star } \| _ { \sharp } \leq \frac { \sigma _ { r } ( X _ { \star } ) } { 2 8 0 \mu r } \left( \frac { 7 } { 2 5 } \right) ^ { 2 ^ { k } } , \qquad 0 \leq k \leq \bar { K } ,
$$

and

$$
\Vert \mathbf { X } _ { k + 1 } - \mathbf { X } _ { \star } \Vert _ { \mathrm { F } } \leq 4 \frac { \Vert \mathbf { X } _ { k } - \mathbf { X } _ { \star } \Vert _ { \mathrm { F } } ^ { 2 } } { \sqrt { q } \sigma _ { r } ( X _ { \star } ) } , \qquad k \geq \bar { K } .
$$

Proof. The proof is deferred to Subsection 7.5.

The RGN can also be regularized and admits a similar global recovery guarantee with the eventual quadratic rate.

Corollary 3.3. Under the conditions of Theorem 3.2, let $\{ X _ { k } \} _ { 0 \leq k \leq \bar { K } }$ be the RGN iterates in Theorem 3.2. Starting from $X _ { \bar { K } }$ , replace Algorithm ${ \mathcal { Q } } \ b y$ Algorithm 3 on $\widehat \Omega$ . Then, with probability at least $1 - n ^ { - 1 0 }$ , all subsequent iterates are well defined and satisfy

$$
\Vert \mathbf { X } _ { k + 1 } - \mathbf { X } _ { \star } \Vert _ { \mathrm { F } } \leq \bar { C } _ { 2 } \frac { \Vert \mathbf { X } _ { k } - \mathbf { X } _ { \star } \Vert _ { \mathrm { F } } ^ { 2 } } { \sqrt { q } \sigma _ { r } ( X _ { \star } ) } , \qquad k \geq \bar { K } ,
$$

where $\bar { C } _ { 2 } > 0$ is an absolute constant.

Proof. The proof is deferred to Subsection 7.6.

Thus O(µnr log n $\log ( 2 \mu r \kappa ) )$ observations sufice for both RGN and its regularized variant to converge to $X _ { \star }$ with high probability, with quadratic convergence after finitely many iterations. Relative Frobenius accuracy ε is attained within $O ( \log \log ( 1 / \varepsilon ) )$ subsequent RGN iterations. The computational cost additionally depends on the CG iteration counts $J _ { k } ;$ see Theorem F.1.

## 3.3 Lower bound for ordinary spectral initialization

We next explain why the multiscale initialization is needed in place of the ordinary spectral estimator. The latter may require an additional factor $\kappa ^ { 2 }$ in the sampling probability to reach the sharp-norm neighborhood required by the local convergence results. For $\ b { Y } \in \mathbb { R } ^ { n \times n }$ , let

$$
\mathcal { H } _ { r } ( \pmb { Y } ) \in \arg \operatorname* { m i n } _ { \operatorname { r a n k } ( \pmb { Z } ) \leq r } \| \pmb { Y } - \pmb { Z } \| _ { \mathrm { F } }
$$

denote a fixed best rank-r approximation. The following theorem shows that a sampling probability of order $\mu \kappa ^ { 2 } r / n$ is necessary over an explicit family of incoherent matrices, with the initialization error measured in the sharp norm (2).

Theorem 3.4. Let $r , s \ge 2$ be integers with $r s \ \leq \ n$ , set $\mu = n / ( r s )$ , and let $1 \leq \kappa \leq \sqrt { s }$ There exist an absolute constant $C _ { 3 } > 0$ and a rank-r matrix $X _ { \star }$ satisfying Assumption 2.1 with coherence parameter $\mu$ and condition number κ such that, for every

$$
0 < q < C _ { 3 } \frac { \mu \kappa ^ { 2 } r } { n } ,
$$

the spectral estimate

$$
\begin{array} { r } { \pmb { X } _ { 0 } = \mathcal { H } _ { r } \left( \ b { q } ^ { - 1 } \mathcal { P } _ { \Lambda } ( \pmb { X } _ { \star } ) \right) , \qquad \Lambda \sim \mathrm { B e r n o u l l i } ( \ b { q } ) , } \end{array}
$$

satisfies

$$
\operatorname* { P r } \left\{ \| X _ { 0 } - X _ { \star } \| _ { \sharp } \geq \frac 1 2 \sigma _ { r } ( X _ { \star } ) \right\} \geq \frac 1 2 .
$$

Proof. The proof is deferred to Section A.

The theorem identifies the $\mu \kappa ^ { 2 } r / n$ sampling scale for the ordinary spectral estimator over this family. In Algorithm 4, reconstruction is applied to successive residuals whose sharp error decreases geometrically.

## 4 Proof Framework: Local Convergence and Initialization

The global results are obtained by combining local convergence of the fixed-sample iterations with the multiscale initialization. We first state the local results for RGD and RGN and then show that Algorithm 4 reaches their respective hypotheses.

## 4.1 Local convergence

The two local results use diferent error controls:

$$
\Vert \boldsymbol { X } - \boldsymbol { X } _ { \star } \Vert _ { \mathrm { F } } \leq \frac { 1 } { 1 2 8 } \sqrt { p } \sigma _ { r } ( \boldsymbol { X } _ { \star } ) ,
$$

$$
\mathrm { f o r ~ R G D } ,\tag{13a}
$$

$$
\Vert \ b { X } - \ b { X } _ { \star } \Vert _ { \sharp } \leq \frac { \sigma _ { r } ( \ b { X } _ { \star } ) } { 1 0 0 0 \mu r } ,
$$

$$
\mathrm { f o r ~ R G N . }\tag{13b}
$$

Thus RGD is controlled in a Frobenius neighborhood whose radius depends on the sampling probability, whereas RGN requires the stronger sharp-norm control.

A common geometric identity underlying both analyses is the exactness of the graph retraction for the population tangent correction. If $\begin{array} { r } { \| \pmb { X } - \pmb { X } _ { \star } \| _ { \sharp } \le \frac 1 4 \sigma _ { r } ( \pmb { X } _ { \star } ) } \end{array}$ , then a later established Lemma 5.3 shows that the graph core is invertible and

$$
\operatorname { R e t r } _ { X } ( { \mathcal { P } } _ { T _ { X } } ( X _ { \star } - X ) ) = X _ { \star } .\tag{14}
$$

Hence the local analysis reduces to controlling the error introduced by sampling. For a nonterminal RGD step, let $\pmb { \xi } = \mathcal { P } _ { T _ { \pmb { X } } } p ^ { - 1 } \mathcal { P } _ { \Omega } ( \pmb { X } ,$ <sub>⋆</sub> X) and let α be the stepsize in (6). Then

$$
\alpha \pmb { \xi } - \mathcal { P } _ { T _ { X } } ( X _ { \star } - X ) = ( \alpha - 1 ) \mathcal { P } _ { T _ { X } } ( X _ { \star } - X ) + \alpha \mathcal { P } _ { T _ { X } } ( p ^ { - 1 } \mathcal { P } _ { \Omega } - \mathbb { Z } ) ( X _ { \star } - X ) .
$$

The tangent sampling isometry controls both terms, which are first order in the error. For RGN, if ξ denotes the tangent correction, then

$$
\begin{array} { r } { \mathcal { P } _ { T _ { \mathbf { X } } } q ^ { - 1 } \mathcal { P } _ { \widehat { \Omega } } \mathcal { P } _ { T _ { \mathbf { X } } } \left( \xi - \mathcal { P } _ { T _ { \mathbf { X } } } ( X _ { \star } - \mathbf { X } ) \right) = \mathcal { P } _ { T _ { \mathbf { X } } } q ^ { - 1 } \mathcal { P } _ { \widehat { \Omega } } \mathcal { P } _ { T _ { \mathbf { X } } ^ { \bot } } ( X _ { \star } - \mathbf { X } ) . } \end{array}\tag{15}
$$

The right-hand side is driven by the normal component, which is quadratic in the error. Indeed, under the local sharp-norm condition, a later established Lemma 5.2 gives

$$
\left. \mathcal { P } _ { T _ { X } ^ { \perp } } ( X _ { \star } - X ) \right. _ { \sharp } \leq \frac { 1 6 } { 3 } \frac { \Vert X - X _ { \star } \Vert _ { \sharp } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .\tag{16}
$$

The two mechanisms are summarized in Figure 1.

![](images/f7cbacd4de128f44b99ff2aca48fefac4dc15e7b4f41d2bb74fd86593319309e.jpg)  
Figure 1: Local mechanisms for RGD and RGN. For RGD, the tangent sampling isometry controls the stepsize and the first-order tangent perturbation. For RGN, the sampled correction is driven by the quadratic normal component. Finite leave-one-out control brings the iterates into a Frobenius neighborhood where quadratic convergence follows deterministically.

For RGD, the high-probability event is uniform over the entire Frobenius neighborhood in (13a). Consequently, the initial point need not be independent of the observation set in the analysis.

Theorem 4.1 (Local convergence of RGD). Suppose that Assumption 2.1 holds and $\Omega \sim$ Bernoulli(p). If $p \geq C _ { 4 } \frac { \mu r \log n } { n }$ , where $C _ { 4 } > 0$ is a suficiently large absolute constant, then, with probability at least $1 { \stackrel {  } { - } } n ^ { - 1 0 } / 4 8$ , the following holds simultaneously for every $\pmb { X } _ { 0 } \in \mathcal { M } _ { r }$ satisfying

$$
\| \mathbf { \cal { X } } _ { 0 } - \mathbf { \cal { X } } _ { \star } \| _ { \mathrm { F } } \leq \frac { 1 } { 1 2 8 } \sqrt { p } \sigma _ { r } ( \mathbf { \cal { X } } _ { \star } ) :
$$

Algorithm 1 on Ω is well defined, and its iterates satisfy

$$
\| X _ { k } - X _ { \star } \| _ { \mathrm { F } } \leq \left( \frac 3 8 \right) ^ { k } \| X _ { 0 } - X _ { \star } \| _ { \mathrm { F } } , \qquad k \geq 0 .
$$

Proof. The proof is deferred to Subsection 5.3.

For RGN, the initial point need be independent of $\widehat \Omega$ in the analysis. This independence bpermits the leave-one-out argument used to control the initial RGN iterates, after which the quadratic Frobenius recursion applies.

Theorem 4.2 (Local convergence of RGN). Suppose that Assumption 2.1 holds and $\widehat { \Omega } \sim$ Bernoulli(q). Let $\pmb { X } _ { 0 } \in \mathcal { M } _ { r }$ be independent of $\widehat \Omega$ and satisfy $\begin{array} { r } { \| \mathbf { \phi } \mathbf { { \cal X } } _ { 0 } - { \mathbf { \cal X } } _ { \star } \| _ { \sharp } \ \le \ \frac { \sigma _ { r } ( \mathbf { { \cal X } } _ { \star } ) } { 1 0 0 0 \mu r } } \end{array}$ b. Set $\bar { K } : = \ \lceil \log _ { 2 } \log _ { 2 } ( 4 n ) \rceil$ . If $q ~ \geq ~ C _ { 4 } \frac { \mu r \log { n } } { n }$ b, then, conditional on $X _ { 0 }$ , with probability at least $1 - { n ^ { - 1 0 } } / { 1 2 } \ o v e r \widehat \Omega$ , the iterates of Algorithm 2 are well defined and satisfy, simultaneously for all $0 \le k \le \bar { K }$

$$
\| \mathbf { } \mathbf { } X _ { k } - \mathbf { } X _ { \star } \| _ { \sharp } \leq \frac { \sigma _ { r } ( \mathbf { } X _ { \star } ) } { 2 8 0 \mu r } \left( \frac { 7 } { 2 5 } \right) ^ { 2 ^ { k } } .
$$

Moreover, for every $k \geq \bar { K }$ 2

$$
\Vert \mathbf { X } _ { k + 1 } - \mathbf { X } _ { \star } \Vert _ { \mathrm { F } } \leq 4 \frac { \Vert \mathbf { X } _ { k } - \mathbf { X } _ { \star } \Vert _ { \mathrm { F } } ^ { 2 } } { \sqrt { q } \sigma _ { r } ( \mathbf { X } _ { \star } ) } .
$$

Proof. The proof is deferred to Subsection 7.4.

## 4.2 Multiscale initialization

We now show that Algorithm 4 reaches the two local conditions above. For a fixed approximation Z and an independent observation subset Λ, the residual correction satisfies

$$
\pmb { Z } + q ^ { - 1 } \mathcal { P } _ { \Lambda } ( \pmb { X } _ { \star } - \pmb { Z } ) = \pmb { X } _ { \star } + ( q ^ { - 1 } \mathcal { P } _ { \Lambda } - \mathbb { Z } ) ( \pmb { X } _ { \star } - \pmb { Z } ) .\tag{17}
$$

Thus its sampling perturbation is determined by the current error, rather than by the largest singular value of $X _ { \star }$ . The first result bounds one spectral reconstruction.

Theorem 4.3. Suppose that Assumption 2.1 holds. Let Z be fixed with rank $( Z ) \ \leq \ r$ and $\| Z - X _ { \star } \| _ { \sharp } \leq \tau$ , where $\tau > 0$ . Let $\Lambda \sim$ Bernoulli(q) be independent of the Gaussian matrix used in $\mathcal { T } _ { \tau } . ~ H f$

$$
q \geq C _ { 5 } { \frac { \mu r \log n } { n } } ,
$$

where $C _ { 5 } > 0$ is a suficiently large absolute constant, then

$$
{ \pmb Z } ^ { + } = \mathcal T _ { \tau } \big ( { \pmb Z } + q ^ { - 1 } \mathcal P _ { \Lambda } ( { \pmb X } _ { \star } - { \pmb Z } ) \big )
$$

satisfies, with probability at least $1 - n ^ { - 1 2 } / 2 4$

$$
\operatorname { r a n k } ( Z ^ { + } ) \leq r , \qquad \left\| Z ^ { + } - X _ { \star } \right\| _ { \sharp } \leq \frac { 1 } { 4 } \tau .\tag{18}
$$

Proof. The proof is deferred to Subsection 6.2.

The proof controls the sampling perturbation and its products with the true singular spaces before estimating the reconstructed matrix. The finite computation in (11) attains the required accuracy without a gap between adjacent singular values. The supporting estimates are proved in Subsection B.2 and section D.

Theorem 4.4 (Multiscale initialization). Suppose that Assumption 2.1 holds and $1 \leq K \leq n$ Let $\{ Z _ { \ell } \} _ { \ell = 0 } ^ { K }$ denote the complete sequence in (10), and let $\boldsymbol { z } _ { \widehat { K } }$ be the output of Algorithm 4. If $q \geq C _ { 5 } \mu r \log n / n$ , then, with probability at least $1 - n ^ { - 1 0 } / 3$ , Algorithm 4 is well defined, ${ \widehat { K } } \leq K$ and

$$
\begin{array} { r } { \| \pmb { X } _ { \star } \| _ { \mathrm { F } } \leq \tau _ { 0 } \leq 3 \| \pmb { X } _ { \star } \| _ { \mathrm { F } } , \qquad \mathrm { r a n k } ( \pmb { Z } _ { \ell } ) \leq r , \qquad \| \pmb { Z } _ { \ell } - \pmb { X } _ { \star } \| _ { \sharp } \leq 4 ^ { - \ell } \tau _ { 0 } , \quad 0 \leq \ell \leq K . } \end{array}\tag{19}
$$

In particular, $K = \lceil 5 + \log _ { 4 } ( \kappa \sqrt { n r } ) \rceil$ gives a rank-r output satisfying

$$
\big \| Z _ { \widehat { K } } - X _ { \star } \big \| _ { \mathrm { F } } \leq \frac { 1 } { 1 2 8 } \sqrt { q } \sigma _ { r } ( X _ { \star } ) ,\tag{20}
$$

whereas $K = \left\lceil 6 + \log _ { 4 } ( \mu r ^ { 3 / 2 } \kappa ) \right\rceil$ gives a rank-r output satisfying

$$
\left\| Z _ { \widehat { K } } - X _ { \star } \right\| _ { \sharp } \leq \frac { \sigma _ { r } ( X _ { \star } ) } { 1 0 0 0 \mu r } .\tag{21}
$$

Proof. The proof is deferred to Subsection 6.4.

The two choices of K allow at most $O ( \log ( n \kappa ) )$ and $O ( \log ( 2 \mu r \kappa ) )$ reconstructions, respec tively. For RGD, $q \leq p$ and (20) imply the hypothesis of Theorem 4.1. For RGN, (21) gives the sharp-norm hypothesis of Theorem 4.2, while the initialization, including the stopping decision, is independent of the reserved set $\widehat { \Omega }$ . The following proposition justifies the residual test, which bshows a residual increase cannot cause a return before the corresponding local convergence condition is satisfied.

Proposition 4.5. Under the assumptions of Theorem $4 { \cdot } 4 ,$ , with probability at least $1 - n ^ { - 1 0 } / 6$ if Algorithm $\it 4$ returns at the residual test, then its output satisfies

$$
\left\| Z _ { \widehat { K } } - X _ { \star } \right\| _ { \mathrm { F } } \leq \frac { \sigma _ { r } ( X _ { \star } ) } { 1 0 2 4 n } .\tag{22}
$$

Proof. The proof is $\mathrm { g i }$ ven in Subsection 6.3.

## 5 Local Convergence of RGD

We first establish the geometric estimates used in the convergence and initialization arguments. We then prove a sampling estimate that holds uniformly over a Frobenius neighborhood of $X _ { \star }$ and use it to prove Theorem 4.1. The initialization results are proved in Section 6.

## 5.1 Deterministic geometry

We begin with bounds for the singular subspaces and the normal component of the error, followed by perturbation bounds for the graph retraction. The proofs are given in Section C.

For $\pmb { X } = \pmb { U } \pmb { \Sigma } \pmb { V } ^ { T } \in \mathcal { M } _ { r }$ , define

$$
G _ { X } : = U ^ { T } X _ { \star } V , \qquad L _ { X } : = ( I - U U ^ { T } ) X _ { \star } V , \qquad R _ { X } : = ( I - V V ^ { T } ) X _ { \star } ^ { T } U .\tag{23}
$$

We also write

$$
t _ { X } : = { \mathcal { P } } _ { T _ { X } } ( X _ { \star } - X ) , \qquad N _ { X } : = { \mathcal { P } } _ { T _ { X } ^ { \perp } } ( X _ { \star } - X )\tag{24}
$$

for the tangent and normal components of the error. Here $G _ { X }$ is the representation of $X _ { \star }$ in the current singular bases, while $\pmb { L } _ { X }$ and $\scriptstyle { R _ { X } }$ describe its components outside the current singular subspaces. The following lemma bounds the incoherence of these bases and the variation of the associated projectors. Its operator-norm estimates follow the argument in [31, Lemma 4.1], while the rowwise estimates use the sharp norm.

Lemma 5.1. Suppose that Assumption 2.1 holds. Let $\pmb { X } = \pmb { U } \pmb { S } \pmb { V } ^ { T } \in \mathcal { M } _ { \pmb { \imath } }$ be a compact $o r \mathrm { - }$ thonormal factorization, and set $e _ { \sharp } : = \lVert X - X _ { \star } \rVert _ { \sharp } . \ I f e _ { \sharp } \leq \sigma _ { r } ( X _ { \star } ) / 8$ , then

$$
\operatorname* { m a x } \left. \| U \| _ { 2 , \infty } , \| V \| _ { 2 , \infty } \right. \leq 2 \sqrt { \frac { \mu r } { n } } .
$$

The singular-space projectors satisfy the rowwise bounds

$$
\operatorname* { m a x } \Big \{ \big \| \pmb { U } \pmb { U } ^ { T } - \pmb { U } _ { \star } \pmb { U } _ { \star } ^ { T } \big \| _ { 2 , \infty } , \big \| \pmb { V } \pmb { V } ^ { T } - \pmb { V } _ { \star } \pmb { V } _ { \star } ^ { T } \big \| _ { 2 , \infty } \Big \} \leq \frac { 3 2 } { 7 } \sqrt { \frac { \mu r } { n } } \frac { e _ { \sharp } } { \sigma _ { r } ( X _ { \star } ) } .
$$

Moreover,

$$
\Vert \mathcal { P } _ { T _ { \mathbf { X } } } - \mathcal { P } _ { T _ { \mathbf { X _ { \star } } } } \Vert _ { \mathrm { F  F } } \leq \frac { 1 6 } { 7 } \frac { e _ { \sharp } } { \sigma _ { r } ( X _ { \star } ) } .
$$

Proof. The proof is deferred to Subsection C.2.1.

The next lemma gives a sharp-norm counterpart of the quadratic normal-component estimate in [31, Lemma 4.1], together with the graph-factor representation used below.

Lemma 5.2. Suppose that Assumption 2.1 holds. Let $\pmb { X } = \pmb { U } \pmb { S } \pmb { V } ^ { T } \in \mathcal { M } _ { \tau }$ be a compact orthonormal factorization, and set $e _ { \sharp } : = \lVert \boldsymbol { X } - \boldsymbol { X } _ { \star } \rVert _ { \sharp }$ . If $e _ { \sharp } \leq \sigma _ { r } ( \pmb { X } _ { \star } ) / 8$ , then $G _ { X }$ is invertible, and the graph factors in (23) satisfy

$$
\begin{array} { r } { N _ { X } = L _ { X } G _ { X } ^ { - 1 } R _ { X } ^ { T } , } \end{array}\tag{25}
$$

and

$$
\left\| G _ { X } ^ { - 1 } \right\| _ { \mathrm { o p } } \leq \frac { 4 } { 3 \sigma _ { r } ( X _ { \star } ) } , \quad \operatorname* { m a x } \left\{ \left\| L _ { X } \right\| _ { \mathrm { o p } } , \left\| R _ { X } \right\| _ { \mathrm { o p } } \right\} \leq e _ { \sharp } ,
$$

$$
\operatorname* { m a x } \left\{ \left\| L \boldsymbol { x } \right\| _ { 2 , \infty } , \left\| R \boldsymbol { x } \right\| _ { 2 , \infty } \right\} \leq 4 \sqrt { \frac { \mu r } { n } } \boldsymbol { e } _ { \sharp } .\tag{26}
$$

Moreover, the normal component satisfies

$$
\| N _ { X } \| _ { \sharp } \leq \frac { 1 6 } { 3 } \frac { e _ { \sharp } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .
$$

Its row, column, and entrywise norms satisfy

$$
\operatorname* { m a x } \Big \{ \| N _ { X } \| _ { 2 , \infty } , \big \| N _ { X } ^ { T } \big \| _ { 2 , \infty } \Big \} \leq \frac { 1 6 } { 3 } \sqrt { \frac { \mu r } { n } } \frac { e _ { \sharp } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } , \qquad \| N _ { X } \| _ { \infty } \leq \frac { 6 4 \mu r } { 3 n } \frac { e _ { \sharp } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .
$$

Proof. The proof is deferred to Subsection C.2.2.

The following identity shows that the population tangent correction recovers X<sub>⋆</sub> exactly. It is the local inverse formula for the orthographic retraction [1, Sec. 3.2.3]; its proof verifies the required core invertibility under the stated error bound.

Lemma 5.3. Let $\pmb { X } \in \mathcal { M } _ { r }$ satisfy $\| X - X _ { \star } \| _ { \sharp } \leq \sigma _ { r } ( X _ { \star } ) / 4$ . Then Retr $\mathbf { \mu } _ { X } ( t _ { X } ) = { X } _ { \star }$

Proof. The proof is deferred to Subsection C.2.3.

We next bound the efect of perturbing the population tangent correction.

Lemma 5.4. Suppose that Assumption 2.1 holds. Let $\pmb { X } \in \mathcal { M } _ { r } ,$ set $e _ { \sharp } : = \lVert X - X _ { \star } \rVert _ { \sharp } ,$ and let $\eta \in T _ { X } \mathcal { M } _ { r } . \ I f \ e _ { \sharp } \leq \sigma _ { r } ( X _ { \star } ) / 8$ and $\lVert \pmb { \eta } \rVert _ { \sharp } \leq \sigma _ { r } ( \pmb { X } _ { \star } ) / 1 6$ , then Retr $_ X ( t _ { X } + \eta )$ is well defined and

$$
\left\| \operatorname { R e t r } _ { X } ( t _ { X } + \eta ) - X _ { \star } - \eta \right\| _ { \sharp } \leq 1 3 \frac { e _ { \sharp } \left\| \eta \right\| _ { \sharp } } { \sigma _ { r } ( X _ { \star } ) } + 6 \frac { \left\| \eta \right\| _ { \sharp } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .
$$

Proof. The proof is deferred to Subsection C.2.4.

The following Frobenius estimates will be used for local RGD convergence and for the eventual quadratic convergence of RGN. Part (i) is a variant of [31, Lemma 4.1].

Lemma 5.5. Let $\pmb { X } \in \mathcal { M } _ { r }$ . Then the following statements hold.

(i) Suppose that $\| X - X _ { \star } \| _ { \mathrm { F } } \leq \sigma _ { r } ( X _ { \star } ) / 4$ . Then

$$
\| \mathcal { P } _ { T _ { \mathbf { X } } ^ { \bot } } ( X _ { \star } - X ) \| _ { \mathrm { F } } \leq 2 \frac { \| X - X _ { \star } \| _ { \mathrm { F } } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } , \qquad \| \mathcal { P } _ { T _ { X } } - \mathcal { P } _ { T _ { X _ { \star } } } \| _ { \mathrm { F  F } } \leq 3 \frac { \| X - X _ { \star } \| _ { \mathrm { F } } } { \sigma _ { r } ( X _ { \star } ) } .
$$

(ii) Let $\pmb { \eta } \in T _ { X } \mathcal { M } _ { r }$ . Suppose that $\Vert X - X _ { \star } \Vert _ { \mathrm { F } } \leq \sigma _ { r } ( X _ { \star } ) / 4 0$ and $\| \pmb { \eta } \| _ { \mathrm { F } } \le \sigma _ { r } ( \pmb { X } _ { \star } ) / 3 2 0$ . Then the graph retraction is well defined and its nonlinear remainder satisfies

$$
\Vert \mathrm { R e t r } _ { X } ( t _ { X } + \eta ) - X _ { \star } - \eta \Vert _ { \mathrm { F } } \leq \frac { 1 1 } { 5 } \frac { \Vert X - X _ { \star } \Vert _ { \mathrm { F } } \Vert \eta \Vert _ { \mathrm { F } } } { \sigma _ { r } ( X _ { \star } ) } + \frac { 1 1 } { 1 0 } \frac { \Vert \eta \Vert _ { \mathrm { F } } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .
$$

Proof. The proof is deferred to Subsection C.3.

We next establish the standard sampling isometry at the tangent space of X<sub>⋆</sub> [8, 31] and transfer it uniformly to nearby tangent spaces.

## 5.2 Sampling isometry and uniform transfer

Lemma 5.6. Suppose that Assumption 2.1 holds, and let Λ  Bernoulli $\left( p _ { \Lambda } \right)$ If $p _ { \Lambda } \ge c _ { 1 } \mu r$ log $n / n$ , where $c _ { 1 } > 0$ is a suficiently large absolute constant, then, with probability at least $1 - n ^ { - 1 0 } / 4 8$

$$
\left. \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } p _ { \Lambda } ^ { - 1 } \mathcal { P } _ { \Lambda } \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } - \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } \left. _ { \mathrm { F \right. F } } \leq \frac { 1 } { 1 6 } . \right.
$$

Proof. The proof is deferred to Subsection B.1.

We next transfer the preceding isometry to nearby tangent spaces. A related local tangentspace estimate appears in [31, Lemma 4.2]. The following deterministic lemma holds simultaneously throughout the stated Frobenius neighborhood, so the matrix X may depend on Λ.

Lemma 5.7. Let Λ be an index set and let $0 < p _ { \Lambda } \le 1$ . Suppose that

$$
\| \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } p _ { \Lambda } ^ { - 1 } \mathcal { P } _ { \Lambda } \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } - \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } \| _ { \mathrm { F  F } } \leq \frac { 1 } { 1 6 } .\tag{27}
$$

Then, simultaneously for all $\pmb { X } \in \mathcal { M } _ { r }$ satisfying

$$
\Vert \boldsymbol { X } - \boldsymbol { X } _ { \star } \Vert _ { \mathrm { F } } \leq \frac { 1 } { 4 0 } \sqrt { p _ { \Lambda } } \sigma _ { r } ( \boldsymbol { X } _ { \star } ) ,
$$

and all $\zeta \in T _ { X } \mathcal { M } _ { r }$ , we have

$$
\frac { 2 } { 3 } \left\| \zeta \right\| _ { \mathrm { F } } ^ { 2 } \leq \left. \zeta , p _ { \Lambda } ^ { - 1 } \mathcal { P } _ { \Lambda } \zeta \right. \leq \frac { 5 } { 4 } \left\| \zeta \right\| _ { \mathrm { F } } ^ { 2 } .\tag{28}
$$

If, in addition, $\Vert \boldsymbol { X } - \boldsymbol { X } _ { \star } \Vert _ { \mathrm { F } } \leq \sqrt { p _ { \Lambda } } \sigma _ { r } ( \boldsymbol { X } _ { \star } ) / 1 2 8$ then

$$
\frac 7 8 \left. \zeta \right. _ { \mathrm { F } } ^ { 2 } \le \left. \zeta , p _ { \Lambda } ^ { - 1 } \mathcal { P } _ { \Lambda } \zeta \right. \le \frac 9 8 \left. \zeta \right. _ { \mathrm { F } } ^ { 2 } .\tag{29}
$$

Proof. Fix any $\pmb { X } \in \mathcal { M } _ { r }$ satisfying $\Vert \boldsymbol { X } - \boldsymbol { X } _ { \star } \Vert _ { \mathrm { F } } \leq \sqrt { p _ { \Lambda } } \sigma _ { r } ( \boldsymbol { X } _ { \star } ) / 4 0$ and any $\zeta \in T _ { X } \mathcal { M } _ { r }$ . By homogeneity, it sufices to consider $\| \zeta \| _ { \mathrm { F } } = 1$ . The assumed neighborhood is contained in $\Vert X - X _ { \star } \Vert _ { \mathrm { F } } \leq \sigma _ { r } ( X _ { \star } ) / 4 0$ , and hence Lemma 5.5 applies. Since $\mathcal { P } _ { T _ { \mathbf { X } } } \zeta = \zeta$ , we obtain

$$
\begin{array} { r l } { \left. { \big \| \zeta - \mathcal { P } _ { T _ { X _ { \star } } } \zeta \big \| _ { \mathrm { F } } = \big \| \big ( \mathcal { P } _ { T _ { X } } - \mathcal { P } _ { T _ { X _ { \star } } } \big ) \zeta \big \| _ { \mathrm { F } } } \right.} \\ & { \leq \big \| \mathcal { P } _ { T _ { X } } - \mathcal { P } _ { T _ { X _ { \star } } } \big \| _ { \mathrm { F  F } } } \\ & { \leq 3 \frac { \| X - X _ { \star } \| _ { \mathrm { F } } } { \sigma _ { r } \left( X _ { \star } \right) } \leq \frac { 3 } { 4 0 } \sqrt { p _ { \Lambda } } , } \\ & { \big \| \mathcal { P } _ { T _ { X _ { \star } } } \zeta \big \| _ { \mathrm { F } } \geq 1 - \big \| \zeta - \mathcal { P } _ { T _ { X _ { \star } } } \zeta \big \| _ { \mathrm { F } } \geq 1 - \frac { 3 } { 4 0 } \sqrt { p _ { \Lambda } } . } \end{array}\tag{30}
$$

Applying (27) to $\mathcal { P } _ { T _ { X _ { \star } } } \zeta \in T _ { X _ { \star } } \mathcal { M } _ { \iota }$ <sub>r</sub> gives

$$
\sqrt { \frac { 1 5 } { 1 6 } } \left. \mathcal { P } _ { T _ { X _ { \star } } } \xi \right. _ { \mathrm { F } } \leq p _ { \Lambda } ^ { - 1 / 2 } \left. \mathcal { P } _ { \Lambda } \mathcal { P } _ { T _ { X _ { \star } } } \xi \right. _ { \mathrm { F } } \leq \sqrt { \frac { 1 7 } { 1 6 } } \left. \mathcal { P } _ { T _ { X _ { \star } } } \xi \right. _ { \mathrm { F } } .
$$

By (27), (30), the triangle inequality, and $\| \mathcal { P } _ { \Lambda } \| _ { \mathrm { F \to F } } \le 1$ , we have

$$
\begin{array} { r l } {  { p _ { \Lambda } ^ { - 1 / 2 } \| \mathcal { P } _ { \Lambda } \boldsymbol { \zeta } \| _ { \mathrm { F } } \geq p _ { \Lambda } ^ { - 1 / 2 } \| \mathcal { P } _ { \Lambda } \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } \boldsymbol { \zeta } \| _ { \mathrm { F } } - p _ { \Lambda } ^ { - 1 / 2 } \| \mathcal { P } _ { \Lambda } \big ( \boldsymbol { \zeta } - \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } \boldsymbol { \zeta } \big ) \| _ { \mathrm { F } } } \quad } & { } \\ & { \geq \sqrt { \frac { 1 5 } { 1 6 } } \| \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } \boldsymbol { \zeta } \| _ { \mathrm { F } } - p _ { \Lambda } ^ { - 1 / 2 } \| \boldsymbol { \zeta } - \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } \boldsymbol { \zeta } \| _ { \mathrm { F } } } \\ & { \geq \sqrt { \frac { 2 } { 3 } } . } \end{array}
$$

By (27), (30), $\left\| \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } \zeta \right\| _ { \mathrm { F } } \leq 1$ , and the triangle inequality, we also have

$$
\begin{array} { r l } {  { p _ { \Lambda } ^ { - 1 / 2 } \| \mathcal { P } _ { \Lambda } \zeta \| _ { \mathrm { F } } \leq p _ { \Lambda } ^ { - 1 / 2 } \| \mathcal { P } _ { \Lambda } \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } \zeta \| _ { \mathrm { F } } + p _ { \Lambda } ^ { - 1 / 2 } \| \mathcal { P } _ { \Lambda } \big ( \zeta - \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } \zeta \big ) \| _ { \mathrm { F } } } \quad } & { } \\ & { \leq \sqrt { \frac { 1 7 } { 1 6 } } \| \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } \zeta \| _ { \mathrm { F } } + p _ { \Lambda } ^ { - 1 / 2 } \| \zeta - \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } \zeta \| _ { \mathrm { F } } } \\ & { \leq \sqrt { \frac { 5 } { 4 } } . } \end{array}
$$

Finally, since $\mathcal { P } _ { \Lambda }$ is an orthogonal projector, the above inequalities give

$$
\frac { 2 } { 3 } \leq \left. \zeta , p _ { \Lambda } ^ { - 1 } \mathcal { P } _ { \Lambda } \zeta \right. = p _ { \Lambda } ^ { - 1 } \left. \mathcal { P } _ { \Lambda } \zeta \right. _ { \mathrm { F } } ^ { 2 } \leq \frac { 5 } { 4 } .
$$

Rescaling the ζ proves (28). Since X was arbitrary in the Frobenius neighborhood specified in Lemma 5.7, (28) holds simultaneously throughout that neighborhood.

To prove (29), suppose that $\Vert X - X _ { \star } \Vert _ { \mathrm { F } } \leq \sqrt { p _ { \Lambda } } \sigma _ { r } ( X _ { \star } ) / 1 2 8$ and $\| \zeta \| _ { \mathrm { F } } = 1$ . By Lemma 5.5 and orthogonality, we have

$$
\begin{array} { r l r } {  { \big \| \zeta - \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } \zeta \big \| _ { \mathrm { F } } \leq \frac { 3 } { 1 2 8 } \sqrt { p _ { \Lambda } } , } } \\ & { } & { \big \| \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } \zeta \big \| _ { \mathrm { F } } ^ { 2 } = 1 - \big \| \zeta - \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } \zeta \big \| _ { \mathrm { F } } ^ { 2 } \geq 1 - \frac { 9 } { 1 2 8 ^ { 2 } } . } \end{array}
$$

The preceding comparison with the true tangent space now $\mathrm { g i }$ ves

$$
\sqrt { \frac { 7 } { 8 } } \leq p _ { \Lambda } ^ { - 1 / 2 } \left\| \mathcal { P } _ { \Lambda } \xi \right\| _ { \mathrm { F } } \leq \sqrt { \frac { 9 } { 8 } } .
$$

Squaring and rescaling proves (29).

## 5.3 Proof of Theorem 4.1

Proof. For $C _ { 4 } \geq c _ { 1 }$ , Lemma 5.6 implies that, with probability at least $1 - n ^ { - 1 0 } / 4 8$

$$
\Vert \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } p ^ { - 1 } \mathcal { P } _ { \Omega } \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } - \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } \Vert _ { \mathrm { F } \to \mathrm { F } } \leq \frac { 1 } { 1 6 } .
$$

On the event supplied by Lemma 5.6, fix any $\pmb { X } \in \mathcal { M } _ { \tau }$ <sub>r</sub> satisfying

$$
\Vert \boldsymbol { X } - \boldsymbol { X } _ { \star } \Vert _ { \mathrm { F } } \leq \frac { 1 } { 1 2 8 } \sqrt { p } \sigma _ { r } ( \boldsymbol { X } _ { \star } ) ,
$$

and set $E : = X _ { \star } - X$ and $\pmb { \xi } : = \mathcal { P } _ { T _ { \mathbf { X } } } p ^ { - 1 } \mathcal { P } _ { \Omega } \pmb { E }$ . Applying (29) and using the self-adjointness of $\mathcal { P } _ { T _ { \mathbf { X } } } p ^ { - 1 } \mathcal { P } _ { \Omega } \mathcal { P } _ { T _ { \mathbf { X } } }$ , we obtain

$$
\mathcal { P } _ { T _ { X } } p ^ { - 1 } \mathcal { P } _ { \Omega } \mathcal { P } _ { T _ { X } } - \mathcal { P } _ { T _ { X } }  _ { \mathrm { F  F } } \leq \frac { 1 } { 8 } , \qquad p ^ { - 1 / 2 }  \mathcal { P } _ { \Omega } \mathcal { P } _ { T _ { X } }  _ { \mathrm { F  F } } \leq \frac { 3 } { 2 \sqrt { 2 } } .\tag{31}
$$

If ${ \pmb { \xi } } \neq { \bf 0 }$ , then (29) gives

$$
p ^ { - 1 } \left\| \mathcal { P } _ { \Omega } \pmb { \xi } \right\| _ { \mathrm { F } } ^ { 2 } = \left. \pmb { \xi } , p ^ { - 1 } \mathcal { P } _ { \Omega } \pmb { \xi } \right. \geq \frac { 7 } { 8 } \left\| \pmb { \xi } \right\| _ { \mathrm { F } } ^ { 2 } > 0 .
$$

Thus the stepsize in (6) is well defined and satisfies

$$
{ \frac { 8 } { 9 } } \leq \alpha \leq { \frac { 8 } { 7 } } , \qquad | \alpha - 1 | \leq { \frac { 1 } { 7 } } .\tag{32}
$$

For ${ \boldsymbol { \xi } } = \mathbf { 0 }$ , set $\alpha = 1$ in the following estimates, so that (32) still holds. Set $\eta : = \alpha \pmb { \xi } - \pmb { t } _ { X }$ Decomposing E into its tangent and normal components, we have

$$
\eta = ( \alpha - 1 ) t _ { X } + \alpha \big ( \mathcal { P } _ { T _ { X } } p ^ { - 1 } \mathcal { P } _ { \Omega } \mathcal { P } _ { T _ { X } } - \mathcal { P } _ { T _ { X } } \big ) t _ { X } + \alpha \mathcal { P } _ { T _ { X } } p ^ { - 1 } \mathcal { P } _ { \Omega } \mathcal { P } _ { T _ { X } ^ { \perp } } E .\tag{33}
$$

The assumed neighborhood is contained in $\| E \| _ { \mathrm { F } } < \sigma _ { r } ( X _ { \star } ) / 4$ . Hence part (i) of Lemma 5.5 applies, and (31)–(33) give

$$
\begin{array} { r l r } & { } & { \displaystyle \| \pmb { \eta } \| _ { \mathrm { F } } \leq \left( | \alpha - 1 | + \frac { \alpha } { 8 } \right) \| \pmb { E } \| _ { \mathrm { F } } + \frac { 3 \alpha } { 2 \sqrt { 2 p } } \left\| \mathcal { P } _ { T _ { \pmb { X } } ^ { \bot } } \pmb { E } \right\| _ { \mathrm { F } } } \\ & { } & { \displaystyle \leq \frac { 2 } { 7 } \| \pmb { E } \| _ { \mathrm { F } } + \frac { 2 4 } { 7 \sqrt { 2 } } \frac { \| \pmb { E } \| _ { \mathrm { F } } ^ { 2 } } { \sqrt { p } \sigma _ { r } ( \pmb { X } _ { \star } ) } \leq \frac { 1 } { 3 } \| \pmb { E } \| _ { \mathrm { F } } . } \end{array}\tag{34}
$$

In particular, $\Vert E \Vert _ { \mathrm { F } } \ \leq \ \sigma _ { r } ( X _ { \star } ) / 1 2 8 \ < \ \sigma _ { r } ( X _ { \star } ) / 4 0$ and $\Vert \eta \Vert _ { \mathrm { F } } \ \le \ \sigma _ { r } ( X _ { \star } ) / 3 8 4 \ < \ \sigma _ { r } ( X _ { \star } ) / 3 2 0$ Consequently, part (ii) of Lemma 5.5 applies. Since $\alpha \pmb { \xi } = \pmb { t } _ { X } + \pmb { \eta }$ , we obtain

$$
\begin{array} { r l r } {  { \| \mathrm { R e t r } _ { \boldsymbol { X } } ( \alpha \boldsymbol { \xi } ) - \boldsymbol { X } _ { \star } \| _ { \mathrm { F } } \leq \| \eta \| _ { \mathrm { F } } ( 1 + \frac { 1 1 } { 5 } \frac { \| \boldsymbol { E } \| _ { \mathrm { F } } } { \sigma _ { r } ( \boldsymbol { X } _ { \star } ) } + \frac { 1 1 } { 1 0 } \frac { \| \eta \| _ { \mathrm { F } } } { \sigma _ { r } ( \boldsymbol { X } _ { \star } ) } ) } } \\ & { } & { \leq \frac { 1 } { 3 } ( 1 + \frac { 1 1 } { 6 4 0 } + \frac { 1 1 } { 3 8 4 0 } ) \| \boldsymbol { E } \| _ { \mathrm { F } } \leq \frac { 3 } { 8 } \| \boldsymbol { E } \| _ { \mathrm { F } } . } \end{array}\tag{35}
$$

If ${ \boldsymbol { \xi } } = \mathbf { 0 }$ , then Ret $\mathbf { \partial } \cdot \mathbf { \partial } \cdot \mathbf { \partial } \cdot \mathbf { \partial } \cdot$ , so (35) implies $\ b X = \ b X _ { \star }$ . Thus every terminal iterate in the stated neighborhood equals $X _ { \star }$

Since X was arbitrary in the Frobenius ball of Theorem 4.1, (35) holds simultaneously throughout that ball. In particular, the neighborhood is invariant under the RGD update. Therefore, if $X _ { 0 }$ satisfies the hypothesis of the theorem, induction gives

$$
\| \boldsymbol { X } _ { k } - \boldsymbol { X } _ { \star } \| _ { \mathrm { F } } \leq \left( \frac 3 8 \right) ^ { k } \| \boldsymbol { X } _ { 0 } - \boldsymbol { X } _ { \star } \| _ { \mathrm { F } } , \qquad k \geq 0 .
$$

The same induction verifies the hypotheses of Lemma 5.5 and the positivity of every nonterminal line-search denominator, and therefore guarantees that all subsequent iterates are well defined. This proves the theorem. □

## 6 Multiscale Initialization and Global Convergence of RGD

We now prove Theorems 4.3 and 4.4. We first bound the approximate reconstruction error in the sharp norm, then justify the residual stopping rule and verify the two local convergence conditions. Combining these estimates with Theorem 4.1 also proves the global RGD result.

## 6.1 Sharp spectral reconstruction

For a fixed error matrix E and an independent subset Λ Bernoulli(q), the sampling perturbation is $\boldsymbol { W } = ( \boldsymbol { q } ^ { - 1 } \mathcal { P } _ { \boldsymbol { \Lambda } } - \mathcal { T } ) \boldsymbol { E }$ . To control both row and column errors, we use the symmetric dilation

$$
\begin{array} { r } { \mathcal { W } : = \left[ \begin{array} { l l } { \mathbf { 0 } } & { W } \\ { W ^ { T } } & { \mathbf { 0 } } \end{array} \right] , \qquad \mathcal { Q } _ { \star } : = \left[ \begin{array} { l l } { U _ { \star } } & { \mathbf { 0 } } \\ { \mathbf { 0 } } & { V _ { \star } } \end{array} \right] . } \end{array}\tag{36}
$$

The columns of $\mathcal { Q } _ { \star }$ are orthonormal, and Assumption 2.1 gives $\| \mathcal { L } _ { \star } \| _ { 2 , \infty } \leq \sqrt { \mu r / n }$ . The following lemma controls powers of the sampling perturbation on these columns.

Lemma 6.1. Suppose that Assumption 2.1 holds. Let E be fixed with $\| \pmb { \cal { E } } \| _ { \sharp } \le \tau$ , where $\tau > 0$ $I f q \geq c _ { 2 } \mu r$ log $n / n$ , where $c _ { 2 } > 0$ is a suficiently large absolute constant, then, with probability at least $1 - n ^ { - 1 2 } / 4 8$ over Λ,

$$
\left\| \mathcal { W } \right\| _ { \mathrm { o p } } \leq \frac { \tau } { 5 1 2 } , \qquad \left\| \mathcal { W } ^ { j } \mathcal { Q } _ { \star } \right\| _ { 2 , \infty } \leq 2 \sqrt { \frac { \mu r } { n } } \left( \frac { \tau } { 6 4 } \right) ^ { j } , \qquad j \geq 0 .\tag{37}
$$

Proof. The proof is deferred to Subsection B.2.

We next give a deterministic reconstruction estimate. Its hypotheses concern approximate singular factors, rather than an exact truncated SVD. This distinction allows the computation in (11) to stop after a prescribed number of steps.

Lemma 6.2. Suppose that Assumption 2.1 holds. Let $Y = X _ { \star } + W$ , and suppose that (37) holds for some $\tau > 0$ . Let $Z ^ { + }$ have rank at most r. If ${ \pmb Z } ^ { + } \neq { \bf 0 }$ , let $\pmb { Z } ^ { + } = \pmb { \hat { U } } \hat { \pmb { \Sigma } } \hat { \pmb { V } } ^ { T }$ be a compact singular value decomposition and suppose that

$$
\sigma _ { \mathrm { m i n } } ( \widehat { \Sigma } ) \geq \frac { \tau } { 8 } ,
$$

$$
\begin{array} { r } { \pmb { Y } ^ { T } \pmb { \widehat { U } } = \pmb { \widehat { V } } \pmb { \widehat { \Sigma } } , } \end{array}\tag{38a}
$$

$$
\left. \mathbf { V } \widehat { \mathbf { V } } - \widehat { U } \widehat { \mathbf { \Sigma } } \right. _ { \mathrm { o p } } \leq \frac { \tau } { 6 4 } \sqrt { \frac { \mu r } { n } } ,
$$

$$
\left. \mathbf { Y } - \pmb { Z } ^ { + } \right. _ { \mathrm { o p } } \leq \frac { 7 \tau } { 3 2 } .\tag{38b}
$$

$I f z ^ { + } = \mathbf { 0 }$ , suppose instead that $\| \mathbf { Y } \| _ { \mathrm { o p } } \leq 7 \tau / 3 2$ . Then $\lVert Z ^ { + } - X _ { \star } \rVert _ { \sharp } \leq \tau / 4$

Proof. The proof is deferred to Subsection D.1.

The iteration in (11) is a randomized subspace iteration; see, e.g., [12]. The next lemma verifies the approximation conditions for $\mathcal { T } _ { \tau }$ . Only the singular components above the current error scale must be accurately represented. No gap between adjacent singular values is assumed.

Lemma 6.3. Let $\ b { Y } \in \mathbb { R } ^ { n \times n }$ satisfy $\sigma _ { r + 1 } ( Y ) \leq \tau / 5 1 2$ , where $\tau > 0$ . Then $Z ^ { + } = \mathcal { T } _ { \tau } ( Y )$ has rank at most r and, with probability at least $1 - n ^ { - 1 2 } / 4 8$ over its Gaussian initial matrix, satisfies (38). $I f z ^ { + } = { \bf 0 }$ , the conclusion is $\| \mathbf { Y } \| _ { \mathrm { o p } } \leq 7 \tau / 3 2$

Proof. The proof is deferred to Subsection D.2.

## 6.2 Proof of Theorem 4.3

Proof. Set

$$
\pmb { W } = ( q ^ { - 1 } \mathcal { P } _ { \Lambda } - \mathcal { T } ) ( \pmb { X } _ { \star } - \pmb { Z } ) , \qquad \pmb { Y } = \pmb { X } _ { \star } + \pmb { W } .
$$

Choose $C _ { 5 } \geq c _ { 2 }$ . By Lemma 6.1, except with probability $n ^ { - 1 2 } / 4 8$

(37) holds,

$$
\sigma _ { r + 1 } ( { \bf Y } ) \leq \| { \bf W } \| _ { \mathrm { o p } } \leq \frac { \tau } { 5 1 2 } .
$$

Conditional on such a realization of Y , Lemma 6.3 gives, except with probability $n ^ { - 1 2 } / 4 8$

$$
\operatorname { r a n k } ( { \mathcal { T } } _ { \tau } ( { \cal Y } ) ) \leq r , \qquad ( 3 8 ) \ \mathrm { h o l d s } .
$$

Hence Lemma 6.2 yields

$$
\lVert \mathcal { T } _ { \tau } ( \boldsymbol { Y } ) - \boldsymbol { X } _ { \star } \rVert _ { \sharp } \leq \frac { \tau } { 4 } .
$$

The result follows by a union bound.

## 6.3 Proof of Proposition 4.5

Proof. We continue (10) through step K, irrespective of the stopping test. Scalar Bernstein gives $\| X _ { \star } \| _ { \mathrm { F } } \leq \tau _ { 0 } \leq 3 \| X _ { \star } \| _ { \mathrm { F } }$ except with probability $n ^ { - 1 2 } / 4 8$ . At each reconstruction, conditional on the preceding observations and Gaussian matrices, Lemma 6.1 gives (37) except with probability $n ^ { - 1 2 } / 4 8$ . Conditional on the corrected matrix, (86) holds with the same failure bound and implies (38) by the proof of Lemma 6.3. Induction using Lemma 6.2 therefore gives an event, with failure probability at most $( 2 K + 1 ) n ^ { - 1 2 } / 4 8$ , on which (19), the sampling estimates (37), and the Gaussian estimate (86) hold throughout the complete sequence.

Observed residuals. For every nonzero iterate on this event, (83) and the projection onto the true singular spaces give

$$
\operatorname* { m a x } \{ \| U _ { \ell } \| _ { 2 , \infty } , \| V _ { \ell } \| _ { 2 , \infty } \} \leq 2 \sqrt { \frac { \mu r } { n } } ,
$$

where $Z _ { \ell } = U _ { \ell } \Sigma _ { \ell } V _ { \ell } ^ { T }$ is a compact singular value decomposition. Fix an index $\ell < K$ for which these preceding reconstruction estimates hold and rank $\begin{array} { r } { ( Z _ { \ell } ) = r . } \end{array}$ and condition on the observations and Gaussian matrices used to construct $\pmb { Z } _ { \ell }$ . Write $E _ { \ell } = Z _ { \ell } - X ,$ . The identity

$$
\pmb { { E } } _ { \ell } = \pmb { U } _ { \star } \pmb { U } _ { \star } ^ { T } \pmb { E } _ { \ell } + ( \pmb { I } - \pmb { U } _ { \star } \pmb { U } _ { \star } ^ { T } ) \pmb { E } _ { \ell } \pmb { V } _ { \ell } \pmb { V } _ { \ell } ^ { T }
$$

implies

$$
\left\| E _ { \ell } \right\| _ { \infty } \leq 3 \sqrt { \frac { \mu r } { n } } \left\| E _ { \ell } \right\| _ { \mathrm { F } } .\tag{39}
$$

The next observation set $\Omega ^ { ( \ell + 1 ) }$ is independent of $\scriptstyle { E _ { \ell } }$ . For the centered sum defining $R _ { \ell } ^ { 2 } - \Vert \mathbf { \mathbf { E } } _ { \ell } \Vert _ { \mathrm { F } } ^ { 2 }$ ， the variance and summand bounds in Lemma B.1 are at most

$$
\frac { 9 \mu r } { n q } \left\| E _ { \ell } \right\| _ { \mathrm { F } } ^ { 4 } \quad \mathrm { a n d } \quad \frac { 9 \mu r } { n q } \left\| E _ { \ell } \right\| _ { \mathrm { F } } ^ { 2 } ,
$$

respectively. Set $\pmb { W } = ( q ^ { - 1 } \mathcal { P } _ { \Omega ^ { ( \ell + 1 ) } } - \mathcal { T } ) ( - E _ { \ell } )$ and $T _ { \star } = T _ { X _ { \star } } \mathcal { M } _ { \iota }$ . Since

$$
\left. \mathcal { P } _ { T _ { \star } } ( e _ { i } e _ { j } ^ { T } ) \right. _ { \mathrm { F } } ^ { 2 } \leq \frac { 2 \mu r } { n } ,
$$

the corresponding bounds for the vectorization of $\mathcal { P } _ { T _ { \star } } W$ are at most $2 \mu r \| E _ { \ell } \| _ { \mathrm { F } } ^ { 2 } / ( n q )$ and $3 \sqrt { 2 } \mu r \| E _ { \ell } \| _ { \mathrm { F } } / ( n q )$ . Applying Lemma B.1 with $t = 2 4$ log n gives, after increasing $C _ { 5 }$

$$
\frac { 3 } { 4 } \left\| E _ { \ell } \right\| _ { \mathrm { F } } ^ { 2 } \leq R _ { \ell } ^ { 2 } \leq \frac { 5 } { 4 } \left\| E _ { \ell } \right\| _ { \mathrm { F } } ^ { 2 } , \qquad \left\| \mathcal { P } _ { T _ { \star } } W \right\| _ { \mathrm { F } } \leq \frac { 1 } { 1 6 } \left\| E _ { \ell } \right\| _ { \mathrm { F } } ,\tag{40}
$$

except with conditional probability $n ^ { - 1 2 } / 4 8$ . These estimates also hold when $\mathbf { \delta E } _ { \ell } ~ = ~ \mathbf { 0 }$ . A conditional union bound over $\ell < K$ , together with the preceding reconstruction events, gives total failure probability at most

$$
\frac { 3 K + 1 } { 4 8 } n ^ { - 1 2 } < \frac { n ^ { - 1 0 } } { 6 } .
$$

We work on this joint event for the remainder of the proof.

Reconstruction after rank r is attained. Fix a rank-r iterate $\pmb { Z } _ { \ell }$ with $\ell < K$ , and write $\sigma =$ $\sigma _ { r } ( \pmb { X } _ { \star } )$ and $\tau = \tau _ { \ell }$ . The preceding reconstruction retains r singular values above $\tau _ { \ell - 1 } / 8 = \tau / 2$ By Weyl’s inequality and (37),

$$
{ \frac { \tau } { 2 } } \leq \sigma + { \frac { \tau } { 1 2 8 } } , \qquad \mathrm { h e n c e } \qquad \tau \leq 3 \sigma .
$$

For the next corrected matrix $Y = X _ { \star } + W$ , it follows that

$$
\| W \| _ { \mathrm { o p } } \le \frac { \tau } { 5 1 2 } \le \frac { \sigma } { 1 6 } , \qquad \sigma _ { r } ( Y ) \ge \frac { 1 5 \sigma } { 1 6 } .\tag{41}
$$

Let $\widetilde { X } = \mathcal { H } _ { r } ( Y )$ and $\widetilde { E } = \widetilde { X } - X _ { \star }$ . Then $\left\| \widetilde { E } \right\| _ { \mathrm { o p } } \leq 2 \left\| W \right\| _ { \mathrm { o p } }$ . In the left and right singular coordinates of $X _ { \star }$ , write $\boldsymbol { \widetilde { E } } = \left[ \begin{array} { l l } { A } & { B } \\ { C } & { D } \end{array} \right]$ . The matrix $\Sigma _ { \star } + A$ is invertible, and $\operatorname { r a n k } ( { \widetilde { X } } ) = r$ gives $D = C ( \Sigma _ { \star } + A ) ^ { - 1 } B$ e. Consequently,

$$
\left| \left. \mathcal { P } _ { T _ { \star } ^ { \perp } } W , \widetilde { E } \right. \right| \leq \frac { \left\| W \right\| _ { \mathrm { o p } } } { \sigma - 2 \left\| W \right\| _ { \mathrm { o p } } } \left\| B \right\| _ { \mathrm { F } } \left\| C \right\| _ { \mathrm { F } } \leq \frac { \left\| W \right\| _ { \mathrm { o p } } } { 2 ( \sigma - 2 \left\| W \right\| _ { \mathrm { o p } } ) } \left\| \widetilde { E } \right\| _ { \mathrm { F } } ^ { 2 } .
$$

The best-approximation property of $\widetilde { X }$ implies

$$
\left\| { \widetilde { \pmb { E } } } \right\| _ { \mathrm { F } } ^ { 2 } \leq 2 \left. W , { \widetilde { \pmb { E } } } \right. \leq 2 \left\| { \mathcal { P } } _ { T _ { \star } } W \right\| _ { \mathrm { F } } \left\| { \widetilde { \pmb { E } } } \right\| _ { \mathrm { F } } + { \frac { \left\| W \right\| _ { \mathrm { o p } } } { \sigma - 2 \left\| W \right\| _ { \mathrm { o p } } } } \left\| { \widetilde { \pmb { E } } } \right\| _ { \mathrm { F } } ^ { 2 } .
$$

Using (41) and (40), we obtain

$$
\left. \widetilde { X } - X _ { \star } \right. _ { \mathrm { F } } \leq 4 \left. \mathcal { P } _ { T _ { \star } } W \right. _ { \mathrm { F } } \leq \frac { 1 } { 4 } \left. E _ { \ell } \right. _ { \mathrm { F } } .\tag{42}
$$

It remains to account for the finite subspace iteration in $\mathcal { T } _ { \tau }$ . By (41) and $\tau \leq 3 \sigma$ , the space range $( U _ { 1 } )$ in the proof of Lemma 6.3 is precisely the leading r-dimensional left singular space of $\mathbf { Y }$ . Write $\Sigma _ { 1 }$ for its singular values and ${ \pmb { P } } = { \pmb { Q } } { \pmb { Q } } ^ { T }$ for the final subspace projector. Equation (87), with $t = \lceil 1 2 \log n \rceil$ and $n ^ { 3 2 } 3 0 ^ { - t } \leq n ^ { - 7 }$ , yields

$$
\| ( I - P ) U _ { 1 } \| _ { \mathrm { o p } } \leq n ^ { - 7 } , \qquad \| ( I - P ) U _ { 1 } \Sigma _ { 1 } \| _ { \mathrm { o p } } \leq \frac { \tau } { 8 \sqrt { 2 } } n ^ { - 7 } .
$$

In particular,

$$
\sigma _ { r } ( { \pmb Q } ^ { T } { \pmb Y } ) \geq \sqrt { 1 - n ^ { - 1 4 } } \sigma _ { r } ( { \pmb Y } ) > \frac { \tau } { 8 } .
$$

Thus all $r$ singular values are retained and $Z _ { \ell + 1 } = P Y$ has rank r. The equality of the largest principal angles between two r-dimensional spaces also gives

$$
\left\| P ( I - U _ { 1 } U _ { 1 } ^ { T } ) \right\| _ { \mathrm { o p } } = \left\| ( I - P ) U _ { 1 } \right\| _ { \mathrm { o p } } \leq n ^ { - 7 } .
$$

Since $\left. \mathbf { Y } - \widetilde { X } \right. _ { \mathrm { o p } } \leq \tau / 5 1 2$ , we have

$$
\begin{array} { r } { \left\| Z _ { \ell + 1 } - \widetilde { X } \right\| _ { \mathrm { F } } \leq \left\| ( I - P ) \widetilde { X } \right\| _ { \mathrm { F } } + \left\| P ( Y - \widetilde { X } ) \right\| _ { \mathrm { F } } } \\ { \leq \sqrt { r } { n ^ { - 7 } } \tau \left( \frac { 1 } { 8 \sqrt { 2 } } + \frac { 1 } { 5 1 2 } \right) \leq n ^ { - 6 } \tau . } \end{array}
$$

Together with (42), this proves

$$
\| E _ { \ell + 1 } \| _ { \mathrm { F } } \le \frac { 1 } { 4 } \| E _ { \ell } \| _ { \mathrm { F } } + n ^ { - 6 } \tau _ { \ell } .\tag{43}
$$

The same argument shows that every iterate after the first rank-r iterate also has rank $r .$

The stopping test. Therefore, whenever the residual test returns a candidate on the joint event, the compared rank-r iterates are consecutive. Write them as $\scriptstyle { Z _ { \ell } }$ and $\pmb { Z } _ { \ell + 1 }$ , where $\ell { + } 1 < K$ and $R _ { \ell + 1 } > R _ { \ell }$ . By (40) and (43),

$$
\sqrt { \frac { 3 } { 4 } } \left\| \mathbf { \boldsymbol { E } } _ { \ell } \right\| _ { \mathrm { \scriptsize { F } } } < \sqrt { \frac { 5 } { 4 } } \left( \frac { 1 } { 4 } \left\| \mathbf { \boldsymbol { E } } _ { \ell } \right\| _ { \mathrm { \scriptsize { F } } } + n ^ { - 6 } \tau _ { \ell } \right) .
$$

It follows that

$$
\| E _ { \ell } \| _ { \mathrm { F } } < 2 n ^ { - 6 } \tau _ { \ell } \leq 6 n ^ { - 6 } \sigma _ { r } ( X _ { \star } ) \leq \frac { \sigma _ { r } ( X _ { \star } ) } { 1 0 2 4 n } ,
$$

where the sampling condition, with $C _ { 5 }$ suficiently large, implies $n \geq 8$ . Since the algorithm returns $\scriptstyle { Z _ { \ell } }$ , this proves (22). Finally, (2) gives $\| E _ { \ell } \| _ { \sharp } \le n \| E _ { \ell } \| _ { \mathrm { F } } / ( \mu r )$ , and the sampling condition gives $q \geq r / n$ . Thus the returned matrix satisfies both local entrance conditions. This proves the proposition. □

## 6.4 Proof of Theorem 4.4

Proof of Theorem $4 { \cdot } 4 .$ We first consider the complete sequence in (10).

Initial scale. By incoherence,

$$
{ \frac { | ( X _ { \star } ) _ { i j } | ^ { 2 } } { \| X _ { \star } \| _ { \mathrm { F } } ^ { 2 } } } \leq { \frac { \mu ^ { 2 } r ^ { 2 } } { n ^ { 2 } } } , \qquad \sum _ { i , j } { \frac { | ( X _ { \star } ) _ { i j } | ^ { 2 } } { \| X _ { \star } \| _ { \mathrm { F } } ^ { 2 } } } = 1 .
$$

Scalar Bernstein applied to the observed squared entries gives

$$
\operatorname* { P r } \left\{ \left| \frac { \| \mathcal { P } _ { \Omega ^ { ( 0 ) } } ( \pmb { X } _ { \star } ) \| _ { \mathrm { F } } ^ { 2 } } { q \| \pmb { X } _ { \star } \| _ { \mathrm { F } } ^ { 2 } } - 1 \right| > \frac { 1 } { 2 } \right\} \le 2 \exp \left( - \frac { 3 n ^ { 2 } q } { 2 8 \mu ^ { 2 } r ^ { 2 } } \right) \le \frac { n ^ { - 1 2 } } { 4 8 } ,
$$

after increasing $C _ { 5 }$ , since $\mu r \leq n$ . Thus, outside this event,

$$
\| \boldsymbol { X } _ { \star } \| _ { \mathrm { F } } \leq \tau _ { 0 } \leq 3 \| \boldsymbol { X } _ { \star } \| _ { \mathrm { F } } .\tag{44}
$$

In particular, $\tau _ { 0 } > 0$ and $\| \boldsymbol { X } _ { \star } \| _ { \sharp } = \sigma _ { 1 } ( \boldsymbol { X } _ { \star } ) \le \tau _ { 0 }$

Induction step. Let $\mathcal { F } _ { \ell }$ contain the observations in $\Omega ^ { ( 0 ) } , \ldots , \Omega ^ { ( \ell ) }$ and the Gaussian matrices used in the first ℓ reconstructions. Conditional on $\mathcal { F } _ { \ell }$ , the matrix $\scriptstyle { Z _ { \ell } }$ and the scale $\tau _ { \ell }$ are fixed, whereas $\Omega ^ { ( \ell + 1 ) }$ and the next Gaussian matrix are independent. Let $\mathcal { G } _ { 0 }$ be the event in (44), and define

$$
\begin{array} { r } { \mathcal G _ { \ell + 1 } : = \mathcal G _ { \ell } \cap \left\{ \mathrm { r a n k } ( \pmb { Z } _ { \ell + 1 } ) \leq r , \lVert \pmb { Z } _ { \ell + 1 } - \pmb { X } _ { \star } \rVert _ { \sharp } \leq 4 ^ { - ( \ell + 1 ) } \tau _ { 0 } \right\} . } \end{array}
$$

On $\mathcal { G } _ { \ell }$ , the induction hypothesis and Theorem 4.3 imply

$$
\operatorname* { P r } ( { \mathcal G } _ { \ell } \setminus { \mathcal G } _ { \ell + 1 } \mid { \mathcal F } _ { \ell } ) \le \frac { n ^ { - 1 2 } } { 2 4 } .
$$

Taking expectations and proceeding by induction proves (19) through step $K$ on $\mathcal { G } _ { K }$

Probability estimate. Since $K \leq n$ and ${ \mathcal { G } } _ { 0 } \supseteq \cdots \supseteq { \mathcal { G } } _ { K }$ 2

$$
\operatorname* { P r } ( \mathcal G _ { K } ^ { c } ) \le \frac { n ^ { - 1 2 } } { 4 8 } + K \frac { n ^ { - 1 2 } } { 2 4 } = \frac { 2 K + 1 } { 4 8 } n ^ { - 1 2 } < \frac { n ^ { - 1 0 } } { 6 } .\tag{45}
$$

Verification of the local hypotheses at step K. For $K = \lceil 5 + \log _ { 4 } ( \kappa \sqrt { n r } ) \rceil$ , the diference $Z _ { K } - X _ { \star }$ has rank at most $2 r$ . Using (19), $\tau _ { 0 } \leq 3 \sqrt { r } \kappa \sigma _ { r } ( X _ { \star } )$ , and $q \geq r / n$ , we obtain

$$
\| Z _ { K } - { \pmb X } _ { \star } \| _ { \mathrm { F } } \leq \sqrt { 2 r } { \pmb \ 4 } ^ { - K } \tau _ { 0 } \leq \frac { 3 \sqrt { 2 } } { 1 0 2 4 } \sqrt { \frac { r } { n } } \sigma _ { r } ( { \pmb X } _ { \star } ) \leq \frac { \sqrt { q } } { 1 2 8 } \sigma _ { r } ( { \pmb X } _ { \star } ) .
$$

For $K = \lceil 6 + \log _ { 4 } ( \mu r ^ { 3 / 2 } \kappa ) \rceil$ , the same invariant gives

$$
\| Z _ { K } - \mathbf { { \cal X } _ { \star } } \| _ { \sharp } \le 4 ^ { - K } \tau _ { 0 } \le \frac { 3 } { 4 0 9 6 \mu r } \sigma _ { r } ( \mathbf { { \cal X } _ { \star } } ) \le \frac { \sigma _ { r } ( \mathbf { { \cal X } _ { \star } } ) } { 1 0 0 0 \mu r } .
$$

In both cases,

$$
\begin{array} { r } { \| Z _ { K } - { \pmb X } _ { \star } \| _ { \mathrm { o p } } < \sigma _ { r } ( { \pmb X } _ { \star } ) , \qquad \mathrm { r a n k } ( { \pmb Z } _ { K } ) \leq r , } \end{array}
$$

so Weyl’s inequality gives rank $\left( Z _ { K } \right) = r$

Residual stopping. Intersect $\mathcal { G } _ { K }$ with the event in Proposition 4.5. The total failure probability is at most $n ^ { - 1 0 } / 3$ . If no residual increase is detected, then ${ \widehat { K } } = K$ and the preceding bounds apply. Otherwise, (22), (2), and $q \geq r / n$ give

$$
\bigl \| Z _ { \widehat { K } } - X _ { \star } \bigr \| _ { \mathrm { F } } \leq \frac { \sqrt { q } } { 1 2 8 } \sigma _ { r } ( X _ { \star } ) , \qquad \bigl \| Z _ { \widehat { K } } - X _ { \star } \bigr \| _ { \sharp } \leq \frac { \sigma _ { r } ( X _ { \star } ) } { 1 0 0 0 \mu r } .
$$

The residual test returns only an iterate of rank r. This proves (20)–(21) in both cases. □

We can now apply the local RGD theorem to the initialization output. The corresponding RGN argument uses the independence of $\widehat \Omega$ in Subsection 7.5.

## 6.5 Proof of Theorem 3.1

Proof. Choose $C _ { 1 } \geq 2 4 ( C _ { 4 } + C _ { 5 } + 1 )$ . Since

$$
K + 1 \leq 1 2 \log ( n \kappa ) , \qquad p = 1 - ( 1 - q ) ^ { K + 1 } \leq ( K + 1 ) q ,
$$

the sampling condition gives $\begin{array} { r } { q \geq \frac { C _ { 1 } \mu r \log { n } } { 1 2 n } } \end{array}$ . It also implies $K + 1 \leq n$ . Hence Theorem 4.4 gives, with probability at least $1 - n ^ { - 1 0 } / 3 ,$

$$
\| X _ { 0 } - X _ { \star } \| _ { \mathrm { F } } \leq \frac { \sqrt { q } } { 1 2 8 } \sigma _ { r } ( X _ { \star } ) \leq \frac { \sqrt { p } } { 1 2 8 } \sigma _ { r } ( X _ { \star } ) .
$$

The event in Theorem 4.1 holds simultaneously throughout this neighborhood, so no independence between $X _ { 0 }$ and Ω is required. A union bound proves the convergence statement.

## 7 Convergence of RGN

We prove Theorem 4.2 by controlling the initial RGN iterates in the sharp norm and then establishing a quadratic Frobenius recurrence. For the initial iterates, we compare the original sequence with auxiliary sequences obtained by completing one row or column. Once the Frobenius error is suficiently small, Lemma 5.7 gives the required estimate uniformly, without further leave-one-out comparisons.

Throughout this section, $\widehat { \Omega } \sim$ Bernoulli(q) and $\mathcal { R } ^ { 0 } : = \boldsymbol { q } ^ { - 1 } \mathcal { P } _ { \widehat { \Omega } }$ <sub>b</sub> . The initial point $X _ { 0 }$ is independent of $\widehat { \Omega }$ b. Inverses of tangent normal operators are taken on their corresponding tangent bspaces. Whenever Lemma 7.2 is used below, it is applied with $( \Lambda , p _ { \Lambda } ) = ( \widehat { \Omega } , q )$

## 7.1 Dependence and the leave-one-out construction

The dificulty is that $\pmb { X } _ { k } = \pmb { X } _ { k } ( \widehat \Omega )$ , so concentration cannot be applied to $T _ { X _ { k } } { \mathcal { M } } _ { \tau }$ as though it bwere independent of the observation set $\widehat \Omega$ . Following the row- and column-deletion construction bin [21, Sec. 7.2 and Algorithm 5] and [10], define, for each row i and column $j$

$$
\begin{array} { r } { \mathcal { R } ^ { \mathrm { r } , i } ( Z ) : = e _ { i } e _ { i } ^ { T } Z + q ^ { - 1 } \mathcal { P } _ { \widehat { \Omega } } \big ( ( I - e _ { i } e _ { i } ^ { T } ) Z \big ) , } \end{array}
$$

$$
\begin{array} { r } { \mathcal { R } ^ { \mathrm { c } , j } ( Z ) : = Z e _ { j } e _ { j } ^ { T } + q ^ { - 1 } \mathcal { P } _ { \widehat { \Omega } } \big ( Z ( I - e _ { j } e _ { j } ^ { T } ) \big ) , } \end{array}
$$

and set

$$
\mathfrak { A } : = \{ 0 \} \cup \{ ( \mathrm { r } , i ) : 1 \leq i \leq n \} \cup \{ ( \mathrm { c } , j ) : 1 \leq j \leq n \} .
$$

Each completed operator replaces one row or column by its population counterpart. The corresponding auxiliary sequence is independent of the Bernoulli variables in that row or column. These sequences are used only for $0 \le k \le \bar { K }$

The following lemma gives a sampling isometry that holds simultaneously for the original operator and all completed operators.

Lemma 7.1. Suppose that Assumption 2.1 holds. $I f q \geq c _ { 1 } \mu r \log n / n$ , with $c _ { 1 }$ as in Lemma ${ 5 . 6 } ,$ then, with probability at least $1 - n ^ { - 1 0 } / 2 4$

$$
\operatorname* { m a x } _ { \alpha \in \mathfrak { A } }  \mathcal { P } _ { T _ { X _ { \star } } } \mathcal { R } ^ { \alpha } \mathcal { P } _ { T _ { X _ { \star } } } - \mathcal { P } _ { T _ { X _ { \star } } }  _ { \mathrm { F }  \mathrm { F } } \leq \frac { 1 } { 1 6 } .
$$

Proof. The proof is deferred to Subsection E.1.1.

For every $\alpha \in { \mathfrak { A } }$ , set $X _ { 0 } ^ { \alpha } = X _ { 0 }$ . We use the deterministic error bound

$$
\rho _ { k } : = \frac { \sigma _ { r } ( X _ { \star } ) } { 2 8 0 \mu r } \left( \frac { 7 } { 2 5 } \right) ^ { 2 k } , \qquad \rho _ { k + 1 } = 2 8 0 \mu r \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .\tag{46}
$$

To define each auxiliary sequence on every outcome, we use the following stopping rule for $0 \le k < \bar { K }$ . Given $X _ { k } ^ { \alpha }$ , let $t _ { X _ { k } ^ { \alpha } }$ be its tangent error component from (24). If the tangent least-squares problem

$$
\operatorname* { m i n } _ { \pmb { \xi } \in T _ { \pmb { X } _ { k } ^ { \alpha } } \mathcal { M } _ { r } } \frac { 1 } { 2 } \left. \pmb { X } _ { k } ^ { \alpha } + \pmb { \xi } - \pmb { X } _ { \star } , \mathcal { R } ^ { \alpha } ( \pmb { X } _ { k } ^ { \alpha } + \pmb { \xi } - \pmb { X } _ { \star } ) \right.
$$

has a unique minimizer, let $\xi _ { k } ^ { \alpha }$ be that minimizer and set $\eta _ { k } ^ { \alpha } : = \pmb { \xi } _ { k } ^ { \alpha } - \pmb { t } _ { X _ { k } ^ { \alpha } }$ . Otherwise, set

$$
\begin{array} { r } { \pmb { \xi } _ { k } ^ { \alpha } : = \pmb { t } _ { X _ { k } ^ { \alpha } } , \qquad \pmb { \eta } _ { k } ^ { \alpha } : = \mathbf { 0 } . } \end{array}
$$

When the minimizer is unique and the graph retraction is defined, denote the candidate next state by

$$
\widehat { X _ { k + 1 } ^ { \alpha } } : = \operatorname { R e t r } _ { X _ { k } ^ { \alpha } } ( \xi _ { k } ^ { \alpha } ) .
$$

We accept the update only if the minimizer is unique, the graph retraction is defined, and

$$
\begin{array} { r } { \| \pmb { X } _ { k } ^ { \alpha } - \pmb { X } _ { \star } \| _ { \sharp } \leq \rho _ { k } , \qquad \Big \| \widehat { \pmb { X } } _ { k + 1 } ^ { \alpha } - \pmb { X } _ { \star } \Big \| _ { \sharp } \leq \rho _ { k + 1 } , } \end{array}
$$

hold. In this case, set $\textstyle X _ { k + 1 } ^ { \alpha } : = { \widehat { X } } _ { k + 1 } ^ { \alpha }$ . Otherwise, leave $X _ { k } ^ { \alpha }$ unchanged and set the subsequent states equal to $X _ { \star }$ : for every $k + 1 \leq \ell \leq \bar { K }$

$$
X _ { \ell } ^ { \alpha } : = X _ { \star } , \qquad \pmb { \xi } _ { \ell } ^ { \alpha } = \pmb { \eta } _ { \ell } ^ { \alpha } : = { \bf 0 } .
$$

After stopping, we use the fixed compact SVD $\pmb { X } _ { \star } = \pmb { U } _ { \star } \pmb { \Sigma } _ { \star } \pmb { V } _ { \star } ^ { T }$ . Thus the auxiliary sequence is defined on every outcome and remains independent of the variables in its completed row or column.

For these stopped processes, define

$$
d _ { k } ^ { \mathrm { l o o } } : = \operatorname* { m a x } _ { \alpha \in \mathfrak { A } } \left\| \mathbf { { X } } _ { k } ^ { \alpha } - \mathbf { { X } } _ { k } ^ { 0 } \right\| _ { \mathrm { F } } , \qquad h _ { k } ^ { \mathrm { c o r r } } : = \operatorname* { m a x } _ { \alpha \in \mathfrak { A } } \left\| \eta _ { k } ^ { \alpha } - \eta _ { k } ^ { 0 } \right\| _ { \mathrm { F } } .
$$

For $0 \le k \le \bar { K }$ , write $ { \boldsymbol { X } } _ { k } : =  { \boldsymbol { X } } _ { k } ^ { 0 }$ . If the updates at indices $0 , \ldots , k - 1$ are accepted, this sequence agrees with Algorithm 2 through index k. We will prove that all these updates are accepted on the event used in the convergence proof.

## 7.2 Simultaneous sampling events

We first bound the number of observations in each row and column. These bounds will be used to control the sampled tangent operators.

Lemma 7.2. Let $\Lambda \sim$ Bernoulli $\left( { p _ { \Lambda } } \right)$ , and set $\delta _ { i j } : = \mathbf { 1 } _ { \{ ( i , j ) \in \Lambda \} }$ . Then, with probability at least $1 - 2 n e ^ { - p _ { \Lambda } n / 3 }$

$$
\operatorname* { m a x } _ { i } \sum _ { j } \delta _ { i j } \leq 2 p _ { \Lambda } n , \qquad \operatorname* { m a x } _ { j } \sum _ { i } \delta _ { i j } \leq 2 p _ { \Lambda } n .
$$

Proof. For every row or column degree $d \sim$ Binomial $( n , p _ { \Lambda } )$ , the multiplicative Chernof bound [26, Corollary 5.2], with deviation parameter one, gives

$$
\mathrm { P r } \{ d > 2 p _ { \Lambda } n \} \le e ^ { - p _ { \Lambda } n / 3 } .
$$

Therefore, a union bound over the 2n row and column degrees proves the result.

The next lemma gives concentration bounds that hold simultaneously for every completed row, every completed column, and every index $0 \leq k < \bar { K }$

Lemma 7.3. Let $X _ { 0 }$ be the initial point and write $\delta _ { i j } : = \mathbf { 1 } _ { \{ ( i , j ) \in \widehat { \Omega } \} }$ . For the stopped row-i and column-j trajectories, define

$$
\begin{array} { r } { \boldsymbol { z } _ { k } ^ { \mathrm { r } , i } : = \left[ \mathcal { P } _ { T _ { \mathbf { X } _ { k } ^ { \mathrm { r } , i } } ^ { \bot } } ( X _ { \star } - X _ { k } ^ { \mathrm { r } , i } ) - \boldsymbol { \eta } _ { k } ^ { \mathrm { r } , i } \right] ^ { T } \boldsymbol { e } _ { i } , \qquad \boldsymbol { z } _ { k } ^ { \mathrm { c } , j } : = \left[ \mathcal { P } _ { T _ { \mathbf { X } _ { k } ^ { \mathrm { c } , j } } ^ { \bot } } ( X _ { \star } - X _ { k } ^ { \mathrm { c } , j } ) - \boldsymbol { \eta } _ { k } ^ { \mathrm { c } , j } \right] \boldsymbol { e } _ { j } , } \end{array}
$$

and set

$$
( \pmb { w } _ { k } ^ { \mathrm { r } , i } ) _ { a } : = \left( \frac { \delta _ { i a } } { q } - 1 \right) ( \pmb { z } _ { k } ^ { \mathrm { r } , i } ) _ { a } , \qquad ( \pmb { w } _ { k } ^ { \mathrm { c } , j } ) _ { a } : = \left( \frac { \delta _ { a j } } { q } - 1 \right) ( \pmb { z } _ { k } ^ { \mathrm { c } , j } ) _ { a } .
$$

Let $V _ { k } ^ { \mathrm { r } , i }$ and $U _ { k } ^ { \mathrm { c } , j }$ be the right and ${ \it l e f t }$ singular factors selected by the fixed compact-SVD convention. Suppose that Assumption 2.1 holds and $X _ { 0 }$ is independent $o f \widehat \Omega$ and satisfies

$$
\| X _ { 0 } - X _ { \star } \| _ { \sharp } \leq \rho _ { 0 } = \frac { \sigma _ { r } ( X _ { \star } ) } { 1 0 0 0 \mu r } .
$$

Then, conditional on $X _ { 0 }$ , with probability at least $1 - n ^ { - 1 0 } / 4 8$ , the following two inequalities hold simultaneously for all $1 \leq i , j \leq n$ and $0 \leq k < \bar { K }$

$$
2 \sqrt { \frac { \mu r } { n } } \left\| w _ { k } ^ { \mathrm { r } , i } \right\| _ { 2 } + \left\| ( V _ { k } ^ { \mathrm { r } , i } ) ^ { T } w _ { k } ^ { \mathrm { r } , i } \right\| _ { 2 } \leq 8 \sqrt { \frac { 3 \log n } { q } } \left( 4 \sqrt { \frac { \mu r } { n } } \left\| z _ { k } ^ { \mathrm { r } , i } \right\| _ { 2 } + \left\| z _ { k } ^ { \mathrm { r } , i } \right\| _ { \infty } \right)
$$

$$
+  \frac { 6 4 \sqrt { \mu r / n } \log n } { q } \| z _ { k } ^ { \mathrm { r } , i } \| _ { \infty } ,\tag{47}
$$

$$
2 \sqrt { \frac { \mu r } { n } } \left\| w _ { k } ^ { \mathrm { c } , j } \right\| _ { 2 } + \left\| ( U _ { k } ^ { \mathrm { c } , j } ) ^ { T } w _ { k } ^ { \mathrm { c } , j } \right\| _ { 2 } \leq 8 \sqrt { \frac { 3 \log n } { q } } \left( 4 \sqrt { \frac { \mu r } { n } } \left\| z _ { k } ^ { \mathrm { c } , j } \right\| _ { 2 } + \left\| z _ { k } ^ { \mathrm { c } , j } \right\| _ { \infty } \right)
$$

$$
+  \frac { 6 4 \sqrt { \mu r / n } \log n } { q } \| z _ { k } ^ { \mathrm { c } , j } \| _ { \infty } .\tag{48}
$$

Proof. The proof is deferred to Subsection E.2.

## 7.3 Error bounds for the initial RGN iterates

The next lemma bounds the sampled tangent normal operators uniformly over the sharp-norm neighborhood. It also bounds the diference between the sampled and population tangent corrections.

Lemma 7.4. Let $\alpha \in { \mathfrak { A } }$ and $\pmb { X } \in \mathcal { M } _ { r }$ , set $e _ { \sharp } : = \lVert \boldsymbol { X } - \boldsymbol { X } _ { \star } \rVert _ { \sharp } ,$ and let $t _ { X }$ be defined by (24). Suppose that Assumption 2.1 and the conclusions of Lemmas 7.1 and 7.2 hold, and that $e _ { \sharp } \leq$ $\frac { \sigma _ { r } ( X _ { \star } ) } { 1 0 0 0 \mu r }$ . Then

$$
\frac { 9 } { 1 0 } \left\| \zeta \right\| _ { \mathrm { F } } ^ { 2 } \leq \langle \zeta , \mathcal { R } ^ { \alpha } \zeta \rangle \leq \frac { 1 1 } { 1 0 } \left\| \zeta \right\| _ { \mathrm { F } } ^ { 2 } , \qquad \zeta \in T _ { X } \mathcal { M } _ { r } .\tag{49}
$$

Let $\xi _ { X } ^ { \alpha }$ be the resulting unique sampled tangent minimizer and set $\eta _ { X } ^ { \alpha } : = \xi _ { X } ^ { \alpha } - t _ { X }$ . Then

$$
\| \eta _ { X } ^ { \alpha } \| _ { \mathrm { F } } \leq 9 \mu r \frac { e _ { \sharp } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .\tag{50}
$$

Proof. The proof is deferred to Subsection E.3.

To propagate the sharp-norm error bound, we also need coordinatewise control of the corrections. The following lemma relates this control to the diference between the original and completed corrections and then bounds that diference.

Lemma 7.5. Let $0 \leq k < \bar { K }$ . Suppose that Assumption 2.1 and the conclusions of Lemmas 7.1 and 7.2 hold, and that

$$
\operatorname* { m a x } _ { \alpha \in \mathfrak { A } } \| \mathbf { X } _ { k } ^ { \alpha } - \mathbf { X } _ { \star } \| _ { \sharp } \leq \rho _ { k } , \qquad d _ { k } ^ { \mathrm { l o o } } \leq 2 \sqrt { \frac { \mu r } { n } } \rho _ { k } .
$$

Then every correction satisfies

$$
\| \eta _ { k } ^ { \alpha } \| _ { \sharp } \leq 2 \sqrt { \frac { n } { \mu r } } h _ { k } ^ { \mathrm { c o r r } } + 1 9 \mu r \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } , \qquad \alpha \in { \mathfrak { A } } .\tag{51}
$$

If, in addition, $q \geq c _ { 4 } \mu r$ log n/n for a suficiently large absolute constant $c _ { 4 } > 0$ and the conclusion of Lemma 7.3 holds, then

$$
h _ { k } ^ { \mathrm { c o r r } } \leq 1 2 8 \sqrt { \frac { \mu r } { n } } \mu r \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .\tag{52}
$$

Proof. The proof is deferred to Subsection E.5.

We next pass from the tangent corrections to the retracted iterates. By Lemma 5.4, if $\| X - X _ { \star } \| _ { \sharp } \leq \rho _ { k }$ and $\| \pmb { \eta } \| _ { \sharp } \le \sigma _ { r } ( \pmb { X } _ { \star } ) / 1 6$ , then

$$
\Vert \mathrm { R e t r } _ { X } ( \mathcal { P } _ { T _ { X } } ( X _ { \star } - X ) + \eta ) - X _ { \star } \Vert _ { \sharp } \leq \Vert \eta \Vert _ { \sharp } + 1 3 \frac { \rho _ { k } \Vert \eta \Vert _ { \sharp } } { \sigma _ { r } ( X _ { \star } ) } + 6 \frac { \Vert \eta \Vert _ { \sharp } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .\tag{53}
$$

The following lemma bounds the diference between retracted iterates at two nearby matrices. For $\pmb { X } , \pmb { Y } \in \mathcal { M } _ { r }$ , write $d _ { X , Y } : = \| X - Y \| _ { \mathrm { F } }$

Lemma 7.6. Let $\pmb { X } , \pmb { Y } \in \mathcal { M } _ { r }$ , let $\eta _ { X } \in T _ { X } \mathcal { M } _ { r }$ and $\eta _ { Y } \in T _ { Y } \mathcal { M } _ { r }$ , and use $\mathbf { \Delta } t _ { X } , t _ { Y }$ from (24). Set $d _ { \eta } : = \| \pmb { \eta } _ { \pmb { X } } - \pmb { \eta } _ { \pmb { Y } } \| _ { \mathrm { F } }$ . Suppose that

$$
\operatorname* { m a x } \left\{ \| X - X _ { \star } \| _ { \sharp } , \| Y - X _ { \star } \| _ { \sharp } \right\} \leq \rho \leq \frac { \sigma _ { r } ( X _ { \star } ) } { 1 0 0 0 \mu r }
$$

and

$$
\operatorname* { m a x } \left\{ \| \pmb { \eta } \pmb { x } \| _ { \mathrm { F } } , \| \pmb { \eta } _ { Y } \| _ { \mathrm { F } } \right\} \le 9 \mu r \frac { \rho ^ { 2 } } { \sigma _ { r } ( \pmb { X } _ { \star } ) } .
$$

Then Ret $\operatorname { r } _ { X } ( t _ { X } + \eta _ { X } )$ and Ret $\operatorname { r } _ { Y } ( t _ { Y } + \eta _ { Y } )$ are well defined, and

$$
\Vert \mathrm { R e t r } _ { X } ( t _ { X } + \eta _ { X } ) - \mathrm { R e t r } _ { Y } ( t _ { Y } + \eta _ { Y } ) \Vert _ { \mathrm { F } } \leq \frac { 5 } { 4 } d _ { \eta } + 1 9 \frac { \mu r \rho ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) ^ { 2 } } d _ { X , Y } .\tag{54}
$$

Proof. The proof is deferred to Subsection C.4.2.

## 7.4 Proof of Theorem 4.2

Proof. We prove by induction that

$$
\operatorname* { m a x } _ { \alpha \in \mathfrak { A } } \| \mathbf { X } _ { k } ^ { \alpha } - \mathbf { X } _ { \star } \| _ { \sharp } \leq \rho _ { k } , \qquad d _ { k } ^ { \mathrm { l o o } } \leq 2 \sqrt { \frac { \mu r } { n } } \rho _ { k } .\tag{55}
$$

Whenever the candidate retraction is defined, the update rule gives

$$
\widehat { \pmb { X } } _ { k + 1 } ^ { \alpha } - \pmb { X } _ { \star } = \pmb { \eta } _ { k } ^ { \alpha } + \left[ \mathrm { R e t r } _ { { \pmb { X } } _ { k } ^ { \alpha } } ( { \pmb { t } } _ { { \pmb { X } } _ { k } ^ { \alpha } } + { \pmb { \eta } } _ { k } ^ { \alpha } ) - \pmb { X } _ { \star } - { \pmb { \eta } } _ { k } ^ { \alpha } \right] .\tag{56}
$$

For the actual and an auxiliary candidate, subtracting the two updates gives

$$
\widehat { \pmb { X } } _ { k + 1 } ^ { 0 } - \widehat { \pmb { X } } _ { k + 1 } ^ { \alpha } = \eta _ { k } ^ { 0 } - \eta _ { k } ^ { \alpha } + \left[ \widehat { \pmb { X } } _ { k + 1 } ^ { 0 } - \pmb { X } _ { \star } - \eta _ { k } ^ { 0 } - \left( \widehat { \pmb { X } } _ { k + 1 } ^ { \alpha } - \pmb { X } _ { \star } - \eta _ { k } ^ { \alpha } \right) \right] .
$$

Step 1: bounds for the initial RGN iterates. Condition on an arbitrary realization of $X _ { 0 }$ satisfying $\| X _ { 0 } - X _ { \star } \| _ { \sharp } \leq \sigma _ { r } ( X _ { \star } ) / ( 1 0 0 0 \mu r )$ . Since $X _ { 0 }$ is independent of $\widehat { \Omega }$ , this conditioning leaves the law of $\widehat \Omega$ unchanged. Fix $C _ { 4 } \geq c _ { 1 } + c _ { 4 } + 1 2 8$ . By the sampling assumption,

$$
q \geq C _ { 4 } { \frac { \mu r \log n } { n } } \geq ( c _ { 1 } + c _ { 4 } ) { \frac { \mu r \log n } { n } } .
$$

Thus the sampling-rate hypotheses of Lemmas 7.1 and 7.5 are satisfied.

Suppose that the conclusions of Lemmas 7.1 to 7.3 hold simultaneously. At $k = 0$ , all auxiliary trajectories equal $X _ { 0 }$ and $d _ { 0 } ^ { \mathrm { l o o } } = 0 ;$ , so (55) holds at $k = 0 .$ . Suppose inductively that transitions $0 , \ldots , k - 1$ have been accepted for every auxiliary process and that (55) holds at some $k < \bar { K }$ . Since $\rho _ { k } \le \sigma _ { r } ( X _ { \star } ) / ( 1 0 0 0 \mu r )$ , Lemma 7.4 implies that every sampled tangent normal operator at iteration k is invertible. Hence every tangent minimizer is unique, and its correction satisfies

$$
\mathcal { P } _ { T _ { \mathbf { X } _ { k } ^ { \alpha } } } \mathcal { R } ^ { \alpha } \mathcal { P } _ { T _ { \mathbf { X } _ { k } ^ { \alpha } } } \eta _ { k } ^ { \alpha } = \mathcal { P } _ { T _ { \mathbf { X } _ { k } ^ { \alpha } } } \mathcal { R } ^ { \alpha } N _ { X _ { k } ^ { \alpha } } , \qquad \alpha \in \mathfrak { A } .
$$

Combining (51) and (52), and using $\rho _ { k } \le \sigma _ { r } ( X _ { \star } ) / ( 1 0 0 0 \mu r )$ , gives

$$
\| \eta _ { k } ^ { \alpha } \| _ { \sharp } \leq 2 7 5 \mu r \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } \leq \frac { 1 1 } { 4 0 } \rho _ { k } < \frac { 1 } { 1 6 } \sigma _ { r } ( X _ { \star } ) , \qquad \alpha \in \mathfrak { A } .\tag{57}
$$

The bound (55) at index k and (57) verify the hypotheses of Lemma 5.4 with $\pmb { X } = \pmb { X } _ { k } ^ { \alpha }$ and $\eta = \eta _ { k } ^ { \alpha }$ . Hence every candidate graph core is invertible and every candidate retraction is well defined. Applying (53) gives

$$
\Big \| \widehat { \mathbfcal { X } } _ { k + 1 } ^ { \alpha } - \mathbfcal { X } _ { \star } \Big \| _ { \sharp } < 2 8 0 \mu r \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( \mathbfcal { X } _ { \star } ) } = \rho _ { k + 1 } .
$$

Thus transition k is accepted for every $\alpha \in { \mathfrak { A } }$

For the two-base update, Lemma 7.4 and the induction hypothesis give max $\{ \| \pmb { \eta } _ { k } ^ { 0 } \| _ { \mathrm { F } } , \| \pmb { \eta } _ { k } ^ { \alpha } \| _ { \mathrm { F } } \} \le$ $9 \mu r \rho _ { k } ^ { 2 } / \sigma _ { r } ( X _ { \star } )$ , which verifies the correction-size hypothesis of Lemma 7.6. Using (54), (52), and the induction hypothesis, we obtain

$$
\left. \mathbf { X } _ { k + 1 } ^ { 0 } - \mathbf { X } _ { k + 1 } ^ { \alpha } \right. _ { \mathrm { F } } \leq \frac { 5 } { 4 } h _ { k } ^ { \mathrm { c o r r } } + 1 9 \frac { \mu r \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) ^ { 2 } } d _ { k } ^ { \mathrm { l o o } } < 2 \sqrt { \frac { \mu r } { n } } \rho _ { k + 1 } .
$$

Taking the maximum over α proves (55) at index $k + 1$ . Therefore, all updates before $\bar { K }$ are accepted, and

$$
\operatorname* { m a x } _ { \alpha \in \mathfrak { A } } \| \mathbf { X } _ { k } ^ { \alpha } - \mathbf { X } _ { \star } \| _ { \sharp } \leq \rho _ { k } , \qquad d _ { k } ^ { \mathrm { l o o } } \leq 2 \sqrt { \frac { \mu r } { n } } \rho _ { k } , \qquad 0 \leq k \leq \bar { K } .
$$

In particular, $\| X _ { k } - X _ { \star } \| _ { \sharp } \leq \rho _ { k }$ for $0 \leq k \leq \bar { K }$

Step 2: the Frobenius error at $\bar { K }$ . Since the diference of two rank-r matrices has rank at most $2 r , 2 ^ { - 2 ^ { K } } \leq ( 4 n ) ^ { - 1 } , \sqrt { 2 r } \leq 2 \mu r$ , and $q \geq C _ { 4 } \mu r$ log $n / n \geq n ^ { - 2 }$ , we have

$$
\begin{array} { r l } & { \| \pmb { X } _ { \bar { K } } - \pmb { X } _ { \star } \| _ { \mathrm { F } } \leq \sqrt { 2 r } \ \| \pmb { X } _ { \bar { K } } - \pmb { X } _ { \star } \| _ { \sharp } \leq \sqrt { 2 r } \rho _ { \bar { K } } } \\ & { \qquad \leq \displaystyle \frac { \sqrt { 2 r } } { 2 8 0 \mu r } 2 ^ { - 2 ^ { \bar { K } } } \sigma _ { r } ( \pmb { X } _ { \star } ) \leq \frac { 1 } { 4 0 n } \sigma _ { r } ( \pmb { X } _ { \star } ) \leq \frac { 1 } { 4 0 } \sqrt { q } \sigma _ { r } ( \pmb { X } _ { \star } ) . } \end{array}
$$

Since all updates before $\bar { K }$ are accepted, the stopped sequence agrees with Algorithm 2 through index K<sup>¯</sup> . From this point onward, $X _ { k }$ denotes the iterates of that algorithm without the auxiliary stopping rule.

Step 3: quadratic convergence in the Frobenius norm. Since $0 \in { \mathfrak { A } }$ , Lemma 7.1 gives

$$
\left. \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } q ^ { - 1 } \mathcal { P } _ { \widehat { \Omega } } \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } - \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } \left. _ { \mathrm { F \right. F } } \leq \frac { 1 } { 1 6 } . \right.
$$

Fix $k \geq \bar { K }$ and suppose that

$$
\boldsymbol { e } _ { k } : = \| \boldsymbol { X } _ { k } - \boldsymbol { X } _ { \star } \| _ { \mathrm { F } } \leq \frac { 1 } { 4 0 } \sqrt { q } \sigma _ { r } ( \boldsymbol { X } _ { \star } ) .
$$

Whenever the tangent minimizer is unique, set $\eta _ { k } : = \xi _ { k } - t _ { X _ { k } }$ . The update rule gives

$$
{ \pmb X } _ { k + 1 } - { \pmb X } _ { \star } = { \pmb \eta } _ { k } + \left[ \mathrm { R e t r } { \pmb X } _ { k } ( { \pmb t } _ { { \pmb X } _ { k } } + { \pmb \eta } _ { k } ) - { \pmb X } _ { \star } - { \pmb \eta } _ { k } \right] .\tag{58}
$$

By Lemma $5 . 7$ with $( \Lambda , p _ { \Lambda } ) = ( \widehat { \Omega } , q )$ , the sampled tangent normal operator on $T _ { X _ { k } } { \mathcal { M } } _ { \mathrm { 1 } }$ is invertible and

$$
( \mathcal { P } _ { T _ { X _ { k } } q } { } ^ { - 1 } \mathcal { P } _ { \widehat { \Omega } } \mathcal { P } _ { T _ { X _ { k } } } ) ^ { - 1 }  _ { \mathrm { F  F } } \leq \frac { 3 } { 2 } , \qquad q ^ { - 1 / 2 }  \mathcal { P } _ { \widehat { \Omega } } \mathcal { P } _ { T _ { X _ { k } } }  _ { \mathrm { F  F } } \leq \sqrt { \frac { 5 } { 4 } } .\tag{59}
$$

Hence the tangent minimizer $\xi _ { k }$ is unique, and

$$
\mathcal { P } _ { T _ { \mathbf { X } _ { k } } } q ^ { - 1 } \mathcal { P } _ { \widehat { \Omega } } \mathcal { P } _ { T _ { \mathbf { X } _ { k } } } \eta _ { k } = \mathcal { P } _ { T _ { \mathbf { X } _ { k } } } q ^ { - 1 } \mathcal { P } _ { \widehat { \Omega } } N _ { \mathbf { X } _ { k } } .
$$

The self-adjointness of $\mathcal { P } _ { \widehat { \Omega } }$ , (59), part (i) of Lemma 5.5, and $e _ { k } \leq \sqrt { q } \sigma _ { r } ( X _ { \star } ) / 4 0$ give

$$
\begin{array} { r l } { \displaystyle \| \eta _ { k } \| _ { \mathrm { F } } \leq \frac { 3 } { 2 } \left\| \mathcal { P } _ { T _ { X _ { k } } } q ^ { - 1 } \mathcal { P } _ { \widehat { \Omega } } N _ { X _ { k } } \right\| _ { \mathrm { F } } } & { } \\ { \displaystyle \leq \frac { 3 } { 2 } \sqrt { \frac { 5 } { 4 q } } \left\| N _ { X _ { k } } \right\| _ { \mathrm { F } } } & { } \\ { \displaystyle \leq \frac { 3 \sqrt { 5 } } { 2 } \frac { e _ { k } ^ { 2 } } { \sqrt { q } \sigma _ { r } ( X _ { \star } ) } < \frac { 1 } { 3 2 0 } \sigma _ { r } ( X _ { \star } ) . } \end{array}\tag{60}
$$

The same neighborhood gives $e _ { k } ~ \leq ~ \sigma _ { r } ( X _ { \star } ) / 4 0$ . Combining (58), (60), and part (ii) of Lemma 5.5, and using $e _ { k } \leq \sqrt { q } \sigma _ { r } ( X _ { \star } ) / 4 0$ , shows that the retraction is well defined and yields

$$
\begin{array} { r } { \| \mathbf { \boldsymbol { X } } _ { k + 1 } - \mathbf { \boldsymbol { X } } _ { \star } \| _ { \mathrm { F } } \leq \| \pmb { \eta } _ { k } \| _ { \mathrm { F } } \left( 1 + \frac { 1 1 } { 5 } \frac { e _ { k } } { \sigma _ { r } ( \pmb { X } _ { \star } ) } + \frac { 1 1 } { 1 0 } \frac { \| \pmb { \eta } _ { k } \| _ { \mathrm { F } } } { \sigma _ { r } ( \pmb { X } _ { \star } ) } \right) } \\ { \leq 4 \frac { e _ { k } ^ { 2 } } { \sqrt { q } \sigma _ { r } ( \pmb { X } _ { \star } ) } \leq \frac { 1 } { 1 0 } e _ { k } \leq \frac { 1 } { 4 0 } \sqrt { q } \sigma _ { r } ( \pmb { X } _ { \star } ) . } \end{array}
$$

Starting from $k = \bar { K }$ , induction proves the Q-quadratic recurrence for every $k \geq \bar { K }$ . It follows that $X _ { k } \to X _ { \star }$

Conditional on the chosen realization of $X _ { 0 }$ , a union bound for the events in Lemmas 7.1 to $7 . 3 ,$ together with $q \geq C _ { 4 } \mu r \log n / n$ with $C _ { 4 } \geq 1 2 8$ and $n \geq 2$ , gives total failure probability at most

$$
\frac { n ^ { - 1 0 } } { 2 4 } + 2 n e ^ { - q n / 3 } + \frac { n ^ { - 1 0 } } { 4 8 } < \frac { n ^ { - 1 0 } } { 1 2 } .\tag{61}
$$

The argument applies to every realization of $X _ { 0 }$ satisfying $\| \mathbf { \boldsymbol { X } } _ { 0 } - \mathbf { \boldsymbol { X } } _ { \star } \| _ { \sharp } \leq \sigma _ { r } ( \mathbf { \boldsymbol { X } } _ { \star } ) / ( 1 0 0 0 \mu r )$ Therefore, (61) gives the conditional probability asserted in Theorem 4.2 and completes the proof. □

## 7.5 Proof of Theorem 3.2

Proof. Choose $C _ { 2 } \geq 3 2 ( C _ { 4 } + C _ { 5 } + 1 )$ . Since

$$
K + 2 \leq 1 6 \log ( 2 \mu r \kappa ) , \qquad p = 1 - ( 1 - q ) ^ { K + 2 } \leq ( K + 2 ) q ,
$$

the sampling condition gives

$$
q \geq { \frac { C _ { 2 } \mu r \log n } { 1 6 n } } .
$$

It also implies $K + 2 \leq n$ . Thus Theorem 4.4 gives, with failure probability at most $n ^ { - 1 0 } / 3$

$$
\| X _ { 0 } - X _ { \star } \| _ { \sharp } \leq \frac { \sigma _ { r } ( X _ { \star } ) } { 1 0 0 0 \mu r } .
$$

The stopping index and the returned initializer depend only on $\Omega ^ { ( 0 ) } , \ldots , \Omega ^ { ( K ) }$ and the independent Gaussian matrices, and are therefore independent of $\widehat { \Omega }$ . Conditional on a successful initialization, Theorem 4.2 fails with probability at most $n ^ { - 1 0 } / 1 2$ . Hence the union bound gives the convergence statement in Theorem 3.2 fails with probability at most $\textstyle { \frac { n ^ { - 1 0 } } { 3 } } + { \frac { n ^ { - 1 0 } } { 1 2 } } < n ^ { - 1 0 }$ □

## 7.6 Proof of Theorem 3.3

Proof. The proof is based on the event in the proof of Theorem 3.2. By Step 2 in the proof of Theorem 4.2, we have $\begin{array} { r } { \| \pmb { X } _ { \bar { K } } - \pmb { X } _ { \star } \| _ { \mathrm { F } } \leq \frac { \sigma _ { r } ( \pmb { X } _ { \star } ) } { 4 0 n } } \end{array}$ . For $C _ { 2 }$ suficiently large, the sampling condition also gives $\begin{array} { r } { \| \boldsymbol { X } _ { \bar { K } } - \boldsymbol { X } _ { \star } \| _ { \mathrm { F } } \leq \frac { \sqrt { q } } { 1 2 8 } \sigma _ { r } ( \boldsymbol { X } _ { \star } ) } \end{array}$ . Fix $k \geq \bar { K }$ and suppose that

$$
e _ { k } : = \| X _ { k } - X _ { \star } \| _ { \mathrm { F } } \leq \frac { \sqrt { q } } { 1 2 8 } \sigma _ { r } ( X _ { \star } ) .
$$

Let $\widehat { \xi _ { k } }$ denote the ordinary RGN direction at $X _ { k }$ and set

$$
\widehat { \eta } _ { k } : = \widehat { \xi _ { k } } - \pmb { t } _ { X _ { k } } .
$$

The argument in Step 3 of the proof of Theorem 4.2, together with (29), gives

$$
\Vert \widehat { \pmb { \eta } } _ { k } \Vert _ { \mathrm { F } } \leq \frac { 3 \sqrt { 5 } } { 2 } \frac { e _ { k } ^ { 2 } } { \sqrt { q } \sigma _ { r } ( \pmb { X } _ { \star } ) } , \qquad \left. \widehat { \pmb { \xi } } _ { k } \right. _ { \mathrm { F } } \leq \frac { 1 7 } { 1 6 } e _ { k } .
$$

Moreover, by the definition of $\lambda _ { k } ,$ the ordinary normal equation, (29), and Weyl’s inequality,

$$
\frac { \lambda _ { k } } { q } = \frac { \left\| \mathcal { P } _ { T _ { \mathbf { X } _ { k } } } q ^ { - 1 } \mathcal { P } _ { \widehat { \Omega } } \mathcal { P } _ { T _ { \mathbf { X } _ { k } } } \widehat { \pmb { \xi } } _ { k } \right\| _ { \mathrm { F } } } { \sigma _ { r } ( \mathbf { X } _ { k } ) } \leq \frac { 5 } { 4 } \frac { e _ { k } } { \sigma _ { r } ( \mathbf { X } _ { \star } ) } .
$$

Subtracting the ordinary and regularized normal equations gives

$$
\left( \mathcal { P } _ { T _ { \mathbf { X } _ { k } } } q ^ { - 1 } \mathcal { P } _ { \widehat { \Omega } } \mathcal { P } _ { T _ { \mathbf { X } _ { k } } } + \frac { \lambda _ { k } } { q } \mathcal { Z } \right) ( \pmb { \xi } _ { k } - \pmb { \widehat { \xi } } _ { k } ) = - \frac { \lambda _ { k } } { q } \pmb { \widehat { \xi } } _ { k } .
$$

Again by (29), we have

$$
\left. \pmb { \xi } _ { k } - \widehat { \pmb { \xi } } _ { k } \right. _ { \mathrm { F } } \leq \frac { 8 5 } { 5 6 } \frac { e _ { k } ^ { 2 } } { \sigma _ { r } ( { \pmb X } _ { \star } ) } .
$$

Hence, with $\eta _ { k } : = \xi _ { k } - t _ { X _ { k } }$ , we have

$$
\Vert \eta _ { k } \Vert _ { \mathrm { F } } \leq \Vert \widehat { \eta } _ { k } \Vert _ { \mathrm { F } } + \left. \pmb { \xi } _ { k } - \widehat { \xi } _ { k } \right. _ { \mathrm { F } } < 5 \frac { e _ { k } ^ { 2 } } { \sqrt { q } \sigma _ { r } ( X _ { \star } ) } < \frac { \sigma _ { r } ( X _ { \star } ) } { 3 2 0 } .
$$

Part (ii) of Lemma 5.5 therefore applies and yields

$$
\Vert \mathbf { X } _ { k + 1 } - \mathbf { X } _ { \star } \Vert _ { \mathrm { F } } \leq \Vert \eta _ { k } \Vert _ { \mathrm { F } } \left( 1 + \frac { 1 1 } { 5 } \frac { e _ { k } } { \sigma _ { r } ( \mathbf { X } _ { \star } ) } + \frac { 1 1 } { 1 0 } \frac { \Vert \eta _ { k } \Vert _ { \mathrm { F } } } { \sigma _ { r } ( \mathbf { X } _ { \star } ) } \right) \leq \bar { C } _ { 2 } \frac { e _ { k } ^ { 2 } } { \sqrt { q } \sigma _ { r } ( \mathbf { X } _ { \star } ) }
$$

for an absolute constant $\bar { C } _ { 2 } > 0$ . The right-hand side is at most $\sqrt { q } \sigma _ { r } ( X _ { \star } ) / 1 2 8$ for $C _ { 2 }$ suficiently large. Thus the stated neighborhood is invariant, and induction proves the result for every $k \geq \bar { K }$ □

## 8 Numerical Experiments

In this section, we compare factorized gradient descent (FGD), ScaledGD [25], RGD, and RGN on randomly generated matrix completion problems. All computations were performed in MAT-LAB R2024b on 64-bit Windows, with an Intel Core Ultra 5 125H CPU and 16 GB memory. We measure the reconstruction error by

$$
\mathrm { e r r } _ { k } : = \frac { \| X _ { k } - X _ { \star } \| _ { \mathrm { F } } } { \| X _ { \star } \| _ { \mathrm { F } } } .
$$

Convergence and running time. We first compare the convergence and running time under diferent condition numbers. We take $n = 1 0 0 0 , r = 1 0 \mathrm { . }$ and $p = 0 . 2$ . Let $U _ { \star }$ contain the left singular vectors of an $n \times r$ matrix with independent random signs. For $\kappa \in \{ 1 , 1 0 , 1 0 0 \}$ , we set

$$
\begin{array} { r } { \pmb { X } _ { \star } = \pmb { U } _ { \star } \operatorname { d i a g } ( \sigma _ { 1 } , \varrho \cdot \varrho , \sigma _ { r } ) \pmb { U } _ { \star } ^ { T } , } \end{array}
$$

where the nonzero singular values are linearly spaced from 1 to $1 / \kappa$ . Each entry is observed independently with probability $p$ without noise. We use the same $U _ { \star }$ and Ω for all three values of $\kappa ;$ for each $\kappa ,$ the four methods receive the same observations $\mathcal { P } _ { \Omega } ( \mathbf { \cal { X } } _ { \star } )$ . FGD and ScaledGD use the same spectral initialization and the respective updates in [25], with stepsize 0.5 for both methods. RGD and RGN share the initialization in Algorithm 4, with $q = p , \Omega ^ { ( \ell ) } = \Omega$ and $K = 8$ , returning the rank-r candidate with the smallest observed residual. In (11), we additionally stop when

$$
\left( 1 - r ^ { - 1 } \left. Q _ { \ell , t - 1 } ^ { T } \pmb { Q } _ { \ell , t } \right. _ { \mathrm { F } } ^ { 2 } \right) ^ { 1 / 2 } \leq 1 0 ^ { - 3 }
$$

holds at two consecutive iterations with $t \geq 3 ;$ the iteration limit remains 12 log n . RGD and RGN follow Algorithms 1 and $^ { 3 , }$ respectively, with $\widehat { \Omega } = \Omega$ . For RGN, CG is stopped at relative bresidual 0.05 or after 200 iterations. All four methods terminate when

$$
\frac { \| \mathcal { P } _ { \Omega } ( \boldsymbol { X } _ { k } - \boldsymbol { X } _ { \star } ) \| _ { \mathrm { F } } } { \| \mathcal { P } _ { \Omega } ( \boldsymbol { X } _ { \star } ) \| _ { \mathrm { F } } } \leq 1 0 ^ { - 1 4 } ,
$$

in at most 1000 iterations. Figure 2 plots $\mathrm { e r r } _ { k }$ against the iteration count and elapsed time for one realization at each κ. Elapsed time includes initialization and the subsequent iterations.

Empirical recovery rates. We next examine how the number of observations needed for recovery varies with the rank. We set $n = 5 0 0$ and conduct 30 random trials for each pair $r \in \{ 2 , 4 , \ldots , 3 0 \} , m / n \in \{ 4 , 8 , \ldots , 8 4 \}$ . In each trial, let $U _ { \star , r }$ and $V _ { \star , r }$ consist of the first $r$ columns of two independent $n \times 3 0$ Haar orthonormal frames, and form $X _ { \star } = U _ { \star , r } V _ { \star , r } ^ { T }$ . We observe m entries uniformly without replacement. For this experiment, the search directions are those in Algorithms 1 and 2, with $\widehat { \Omega } = \Omega$ . The relative observed residual tolerances are $1 0 ^ { - 1 2 }$ for RGD and $1 0 ^ { - 8 }$ bfor RGN. Recovery is declared successful if $\mathrm { e r r } _ { k } \le 1 0 ^ { - 6 }$ . Figure 3 reports the fraction of successful trials at each $( r , m )$

Both panels exhibit a transition from failure to successful recovery as m increases. These results show an approximately linear dependence on nr over the tested ranks. The solid lines in Figure 3 provide simple reference relations for this observed transition.

## 9 Concluding Remarks

We have established global recovery guarantees for RGD and RGN under standard incoherence and Bernoulli sampling. With high probability, $O ( \mu n r$ log n log(nκ)) and $O ( \mu n r$ log n log(2µrκ)) observations sufice for the two methods, respectively. The sample requirements are linear in n and $^ { r , }$ up to logarithmic factors. RGD converges linearly, while RGN satisfies a doubly exponential sharp-norm error bound during its initial iterations and a quadratic Frobenius recurrence thereafter. Both methods use the multiscale residual initialization with residual stopping and diferent upper bounds on the number of reconstructions for their respective local conditions. The observation set is fixed at the beginning, and the sampling requirement does not depend on the target accuracy.

![](images/4dce69eba90d3afac7501ab34ceb9ef4db28ec4f21172c550378efffef6223ab.jpg)

![](images/395111885787a5466e021d6bccce321b9d99a4e1e4864a64d248790343763b9d.jpg)  
elapsed time (s)

(c) κ = 10  
![](images/371ec38339d73b5cb5bc019f6e57fb4e214124c71378277d81b38fa88035ff15.jpg)

(d) κ = 10  
![](images/e803c1d78de472d6f6b8936ab3d8514c87f2b8c953011a58468435a3cadd5bae.jpg)  
elapsed time (s)

![](images/52e8b867fa79b8d57cb005fdd3e26760fbb5bb00cac172e6de32e9656a16856c.jpg)

(f) κ = 100  
![](images/1fdb2755c4f228768d6e5b58038a3678086e13e4a32bc54e57aca4261d59d921.jpg)  
elapsed time (s)  
Figure 2: Relative reconstruction errors of FGD, ScaledGD, RGD, and RGN versus iteration count (left) and elapsed time in seconds (right), with n = 1000, r = 10, and $p = 0 . 2$ . From top to bottom, κ = 1, 10, 100.

Nevertheless, several extensions merit further study. One is to establish the same guarantees when every initialization step reuses the entire observation set. Another is to analyze RGN with an inexact tangent solve and a computable stopping criterion, so that the convergence rate can be related to the total number of CG iterations.

![](images/a04667661d003c46570f736ccb30fcb5b5299a3c6d1a207375fb90b60269aea3.jpg)

![](images/5c9d5f1b2de1d220ae626b2b93f515e7b5da935c23d6d4398d0e81096d334254.jpg)  
Figure 3: Empirical recovery rates of RGN (left) and RGD (middle) for $n = 5 0 0$ and $\kappa = 1$ over 30 trials at each $( r , m )$ . The horizontal axis is $m / n$ and the vertical axis is r. White denotes success in all trials, and black denotes failure in all trials. The solid reference lines are $m = ( 2 r + 9 ) n$ for RGN and $m = ( 2 . 5 r + 1 0 ) n$ for RGD.

## Declaration on the use of artificial intelligence

Generative AI tools were used during revision to assist with language editing, structural reorganization, bibliographic cross-checking, and consistency checks of notation, formulas, and proofs.

## A Lower bound for ordinary spectral initialization

We prove Theorem 3.4 by constructing a block-diagonal family of incoherent matrices.

Proof of Theorem $\ 3 . 4$ . Take $C _ { 3 } = 2 ^ { - 1 2 }$ . Choose pairwise disjoint sets $I _ { 1 } , \ldots , I _ { r } \subseteq \{ 1 , \ldots , n \}$ with $| I _ { a } | = s ,$ , and set

$$
\pmb { u } _ { a } = \pmb { v } _ { a } = s ^ { - 1 / 2 } \pmb { 1 } _ { I _ { a } } , \qquad \pmb { X } _ { \star } = \kappa \pmb { u } _ { 1 } \pmb { v } _ { 1 } ^ { T } + \sum _ { a = 2 } ^ { r } \pmb { u } _ { a } \pmb { v } _ { a } ^ { T } .
$$

Then

$$
\sigma _ { r } ( \boldsymbol { X } _ { \star } ) = 1 , \qquad \kappa ( \boldsymbol { X } _ { \star } ) = \kappa , \qquad \operatorname* { m a x } _ { i } \left\| e _ { i } ^ { T } U _ { \star } \right\| _ { 2 } ^ { 2 } = \operatorname* { m a x } _ { j } \left\| e _ { j } ^ { T } V _ { \star } \right\| _ { 2 } ^ { 2 } = \frac { 1 } { s } = \frac { \mu r } { n } .
$$

Thus Assumption 2.1 holds. Set $d = q s$ and $Y = q ^ { - 1 } \mathcal { P } _ { \Lambda } ( X _ { \star } )$ . On the active coordinates,

$$
\mathbf { \boldsymbol { Y } } = \operatorname { d i a g } ( \mathbf { \boldsymbol { Y } } _ { 1 } , \dots , \mathbf { \boldsymbol { Y } } _ { r } ) , \qquad \mathbf { \boldsymbol { Y } } _ { 1 } = \frac { \kappa } { d } \mathbf { \boldsymbol { \Delta } } _ { 1 } , \qquad \mathbf { \boldsymbol { Y } } _ { a } = \frac { 1 } { d } \mathbf { \boldsymbol { \Delta } } _ { a } , \quad 2 \leq a \leq r ,
$$

where the $\pmb { \Delta } _ { a } \in \{ 0 , 1 \} ^ { s \times s }$ have independent Bernoulli(q) entries. Since $\mu = n / ( r s )$ and $\kappa ^ { 2 } \leq s$

$$
d = q s < { C _ { 3 } \kappa } ^ { 2 } , \qquad q < { C _ { 3 } \frac { \kappa ^ { 2 } } { s } } \le { C _ { 3 } } < \frac 1 2 .
$$

Suppose first that $\begin{array} { r } { d \leq \frac { 1 } { 4 } \log ( r s ) } \end{array}$ . For each active row,

$$
\pi : = \operatorname* { P r } \{ \mathrm { t h e ~ r o w ~ i s ~ e m p t y ~ o n ~ i t s ~ s u p p o r t } \} = ( 1 - q ) ^ { s } \geq e ^ { - 2 q s } = e ^ { - 2 d } \geq ( r s ) ^ { - 1 / 2 } .
$$

Hence

$$
\operatorname* { P r } \{ \mathrm { a n ~ a c t i v e ~ r o w ~ i s ~ e m p t y } \} = 1 - ( 1 - \pi ) ^ { r s } \geq 1 - e ^ { - r s \pi } \geq 1 - e ^ { - { \sqrt { r s } } } > \frac { 1 } { 2 } .
$$

If $i \in I _ { a }$ is such a row, then

$$
e _ { i } ^ { T } { \mathbf { } } \mathbf { } \mathbf { } Y = \mathbf { 0 } \quad \implies \quad e _ { i } ^ { T } \mathcal { H } _ { r } ( { \mathbf { } } Y ) = \mathbf { 0 } , \qquad \left. e _ { i } ^ { T } { \mathbf { } } { \mathbf { } } X _ { \star } \right. _ { 2 } \ge s ^ { - 1 / 2 } ,
$$

and therefore

$$
\| \mathcal { H } _ { r } ( \pmb { Y } ) - \pmb { X } _ { \star } \| _ { \sharp } \geq \frac { 1 } { 2 } \sqrt { s } \left\| e _ { i } ^ { T } ( \mathcal { H } _ { r } ( \pmb { Y } ) - \pmb { X } _ { \star } ) \right\| _ { 2 } \geq \frac { 1 } { 2 } .
$$

Now suppose that $\begin{array} { r } { d > \frac { 1 } { 4 } \log ( r s ) } \end{array}$ . Let $\mathcal { D }$ be the event that every row and column degree of every $\Delta _ { a }$ is at most 16d. The multiplicative Chernof bound gives

$$
\operatorname* { P r } ( \mathcal { D } ^ { c } ) \leq 2 r s \exp \bigl ( - ( 1 6 \log 1 6 - 1 5 ) d \bigr ) < 2 ( r s ) ^ { 1 - ( 1 6 \log 1 6 - 1 5 ) / 4 } \leq \frac { 1 } { 3 0 } .
$$

With $M = \| \Delta _ { 1 } \| _ { \mathrm { F } } ^ { 2 }$ , we have

$$
\mathbb { E } M = s d , \qquad \mathbb { E } M ^ { 2 } \leq s ^ { 2 } d ^ { 2 } + s d , \qquad s d > \frac { d ^ { 2 } } { C _ { 3 } } > \frac { \log ^ { 2 } ( r s ) } { 1 6 C _ { 3 } } .
$$

Hence the Paley–Zygmund inequality yields

$$
\operatorname* { P r } \{ \mathcal { D } \cap \{ M \geq s d / 4 \} \} \geq \frac { 9 } { 1 6 } \frac { s d } { s d + 1 } - \operatorname* { P r } ( \mathcal { D } ^ { c } ) > \frac { 1 } { 2 } .
$$

On $\mathcal { D } _ { : }$ , the maximum row sum and the maximum column sum of $\Delta _ { a }$ are both at most 16d. Hence

$$
\Vert \Delta _ { a } \Vert _ { \mathrm { o p } } \leq \sqrt { \left( \underset { i } { \operatorname* { m a x } } \sum _ { j } | ( \Delta _ { a } ) _ { i j } | \right) \left( \underset { j } { \operatorname* { m a x } } \sum _ { i } | ( \Delta _ { a } ) _ { i j } | \right) } \leq 1 6 d , \qquad 1 \leq a \leq r ,
$$

and therefore

$$
\begin{array} { r l r l } { \displaystyle \left\| Y _ { 1 } \right\| _ { \mathrm { F } } ^ { 2 } \geq \frac { \kappa ^ { 2 } s } { 4 d } , } & { \qquad } & { \left\| Y _ { 1 } \right\| _ { \mathrm { o p } } \leq 1 6 \kappa , } \\ { \displaystyle \operatorname* { m a x } _ { 2 \leq a \leq r } \left\| Y _ { a } \right\| _ { \mathrm { o p } } \leq 1 6 , } & { } & { \displaystyle \sum _ { j \geq 2 } \sigma _ { j } { ( Y _ { 1 } ) } ^ { 2 } \geq \frac { \kappa ^ { 2 } s } { 4 d } - 2 5 6 \kappa ^ { 2 } > \left( \frac { 1 } { 4 C _ { 3 } } - 2 5 6 \right) s > 2 5 6 ( s - 1 ) . } \end{array}
$$

Thus

$$
\sigma _ { 2 } ( { Y _ { 1 } } ) > 1 6 \geq \operatorname* { m a x } _ { 2 \leq a \leq r } \left. Y _ { a } \right. _ { \mathrm { o p } } .
$$

If ${ \cal Y } _ { 1 }$ has at least r singular values larger than 16, then every best rank-at-most-r approximation of $\mathbf { Y }$ is supported on $I _ { 1 } \times I _ { 1 }$ , and hence

$$
\begin{array} { r } { \| \mathcal { H } _ { r } ( \pmb { Y } ) - \pmb { X } _ { \star } \| _ { \sharp } \geq \big \| \mathcal { P } _ { ( I _ { 2 } \cup \cdots \cup I _ { r } ) \times ( I _ { 2 } \cup \cdots \cup I _ { r } ) } \pmb { X } _ { \star } \big \| _ { \mathrm { o p } } = 1 . } \end{array}
$$

Otherwise, the Eckart–Young–Mirsky theorem and $\sigma _ { 2 } ( Y _ { 1 } ) > 1 6$ imply, with $I = I _ { 2 } \cup \cdots \cup I _ { r }$

$$
\operatorname { r a n k } ( \mathcal { P } _ { I \times I } \mathcal { H } _ { r } ( Y ) ) \leq r - 2 .
$$

Since ${ \mathcal { P } } _ { I \times I } X ,$ <sub>⋆</sub> has $r - 1$ singular values equal to one,

$$
\begin{array} { r } { \| \mathcal { H } _ { r } ( \pmb { Y } ) - \pmb { X } _ { \star } \| _ { \sharp } \geq \| \mathcal { P } _ { I \times I } ( \mathcal { H } _ { r } ( \pmb { Y } ) - \pmb { X } _ { \star } ) \| _ { \mathrm { o p } } \geq 1 . } \end{array}
$$

In both cases,

$$
\operatorname* { P r } \biggr \{ \| \mathcal { H } _ { r } ( \mathbf { Y } ) - X _ { \star } \| _ { \sharp } \geq \frac { 1 } { 2 } \biggr \} > \frac { 1 } { 2 } .
$$

This proves the theorem.

## B Probabilistic Lemmas

We collect the concentration tools used throughout the sampling arguments. We use the following standard forms of the rectangular and self-adjoint matrix Bernstein inequalities; see [26, Theorems 1.6 and 6.1].

Lemma B.1. Let $Z _ { 1 } , \dots , Z _ { N } \in \mathbb { R } ^ { d _ { 1 } }$ ×<sup>d</sup>2 be independent mean-zero random matrices. Suppose that $\| Z _ { a } \| _ { \mathrm { o p } } \leq L$ almost surely, and define

$$
v : = \operatorname* { m a x } \left\{ \left\| \sum _ { a } \mathbb { E } ( Z _ { a } Z _ { a } ^ { T } ) \right\| _ { \mathrm { o p } } , \left\| \sum _ { a } \mathbb { E } ( Z _ { a } ^ { T } Z _ { a } ) \right\| _ { \mathrm { o p } } \right\} .
$$

Then, for every $t > 0$

$$
\operatorname* { P r } \left\{ \left\| \sum _ { a } {  \boldsymbol Z } _ { a } \right\| _ { \mathrm { o p } } > \sqrt { 2 v t } + \frac 2 3 L t \right\} \le ( d _ { 1 } + d _ { 2 } ) e ^ { - t } .
$$

If the $Z _ { a }$ are self-adjoint operators on a d-dimensional Hilbert space, then, for every $u > 0$

$$
\operatorname* { P r } \left\{ \left\| \sum _ { a } { Z _ { a } } \right\| _ { \mathrm { o p } } \ge u \right\} \le 2 d \exp \left( - \frac { u ^ { 2 } } { 2 ( v + L u / 3 ) } \right) .
$$

## B.1 Proof of Lemma 5.6

We next prove the true-tangent sampling isometry used in the local RGD analysis.

Proof of Lemma 5.6. Fix $c _ { 1 } = 2 ^ { 1 5 }$ . Let $\delta _ { i j } : = \mathbf { 1 } _ { \{ ( i , j ) \in \Lambda \} }$ and ${ \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } = { \mathcal { P } } _ { T _ { { \mathbf { } } { \mathbf { } } _ { \star } } } ( { \mathbf { } } e _ { i } e _ { j } ^ { T } )$ , and define $( { \pmb a } \otimes { \pmb a } ) ( { \pmb Z } ) : = \langle { \pmb a } , { \pmb Z } \rangle _ { \mathrm { F } } { \pmb a }$ . Since the matrices $e _ { i } e _ { j } ^ { T }$ form an orthonormal basis of $\mathbb { R } ^ { n \times n }$ , we have

$$
\sum _ { i , j } { \pmb { a } } _ { i j } \otimes { \pmb { a } } _ { i j } = { \pmb { \mathscr { I } } } _ { T _ { { \pmb { X } } _ { \star } } } .
$$

Consequently, we have

$$
\mathcal { P } _ { T _ { X _ { \star } } } p _ { \Lambda } ^ { - 1 } \mathcal { P } _ { \Lambda } \mathcal { P } _ { T _ { X _ { \star } } } - \mathcal { P } _ { T _ { X _ { \star } } } = \sum _ { i , j } \left( \frac { \delta _ { i j } } { p _ { \Lambda } } - 1 \right) ( { \bf a } _ { i j } \otimes { \bf a } _ { i j } ) .
$$

By (4) and the orthogonality of its two summands, we have

$$
\begin{array} { r l } & { \qquad \mathbf { \delta } \mathbf { { \boldsymbol { a } } } _ { i j } = U _ { \star } U _ { \star } ^ { T } e _ { i } e _ { j } ^ { T } + ( I - U _ { \star } U _ { \star } ^ { T } ) e _ { i } e _ { j } ^ { T } V _ { \star } V _ { \star } ^ { T } , } \\ & { \qquad \quad \| \mathbf { \boldsymbol { a } } _ { i j } \| _ { \mathrm { { F } } } ^ { 2 } = \left\| U _ { \star } U _ { \star } ^ { T } e _ { i } \right\| _ { 2 } ^ { 2 } + \left\| ( I - U _ { \star } U _ { \star } ^ { T } ) e _ { i } \right\| _ { 2 } ^ { 2 } \left\| V _ { \star } V _ { \star } ^ { T } e _ { j } \right\| _ { 2 } ^ { 2 } \leq \frac { 2 \mu r } { n } . } \end{array}
$$

Each summand has norm at most $2 \mu r / ( n p _ { \Lambda } )$ . Moreover, $\mathbb { E } ( \delta _ { i j } / p _ { \Lambda } - 1 ) ^ { 2 } = ( 1 - p _ { \Lambda } ) / p _ { \Lambda }$ and $( \mathbf { { a } } \otimes \mathbf { { a } } ) ^ { 2 } = \left\| \mathbf { { a } } \right\| _ { \mathrm { F } } ^ { 2 } ( \mathbf { { a } } \otimes \mathbf { { a } } )$ , so the variance operator satisfies

$$
\frac { 1 - p _ { \Lambda } } { p _ { \Lambda } } \sum _ { i , j } \| { \pmb { a } } _ { i j } \| _ { \mathrm { F } } ^ { 2 } \left( { \pmb { a } } _ { i j } \otimes { \pmb { a } } _ { i j } \right) \preceq \frac { 2 \mu r } { n p _ { \Lambda } } { \cal { T } } _ { { \pmb { T } } _ { \pmb { X } _ { \star } } } .
$$

Since dim $( T _ { X _ { \star } } ) = r ( 2 n - r ) \leq 2 n r , p _ { \Lambda } \geq c _ { 1 } \mu r \log n / n , r \leq n _ { - }$ , and $n \geq 2$ , Lemma B.1 at threshold 1/16 gives

$$
\operatorname* { P r } \{ \| \mathcal P _ { T _ { \mathbf { X } _ { \star } } } p _ { \Lambda } ^ { - 1 } \mathcal P _ { \Lambda } \mathcal P _ { T _ { \mathbf { X } _ { \star } } } - \mathcal P _ { T _ { \mathbf { X } _ { \star } } } \| _ { \mathrm { F  F } } > \frac { 1 } { 1 6 } \} \leq \frac { n ^ { - 1 0 } } { 4 8 } .
$$

This proves the lemma.

## B.2 Proof of Lemma 6.1

We prove the sampling estimates for a fixed error matrix. Independence from the current observation component is supplied by conditioning in Subsection 6.4.

Proof of Lemma 6.1. By (2),

$$
\left\| E \right\| _ { \infty } \leq \frac { 4 \mu r } { n } \tau , \qquad \operatorname* { m a x } \{ \left\| E \right\| _ { 2 , \infty } , \left\| E ^ { T } \right\| _ { 2 , \infty } \} \leq 2 \sqrt { \frac { \mu r } { n } } \tau .
$$

Hence, for $\boldsymbol { W } = ( q ^ { - 1 } \mathcal { P } _ { \boldsymbol { \Lambda } } - \mathcal { T } ) \boldsymbol { E }$

$$
| W _ { i j } | \leq \frac { 4 \mu r \tau } { n q } , \qquad \operatorname* { m a x } \left\{ \operatorname* { m a x } _ { i } \sum _ { j } \mathbb { E } W _ { i j } ^ { 2 } , \operatorname* { m a x } _ { j } \sum _ { i } \mathbb { E } W _ { i j } ^ { 2 } \right\} \leq \frac { 4 \mu r \tau ^ { 2 } } { n q } .\tag{62}
$$

By Lemma B.1,

$$
\mathrm { P r } \Big \{ \| \mathbf { \mathcal { W } } \| _ { \mathrm { o p } } > \frac { \tau } { 5 1 2 } \Big \} \le 4 n \exp \left( - c \frac { n q } { \mu r } \right) .\tag{63}
$$

For $| z | = \tau / 6 4$ , set

$$
\pmb { F } ( z ) = ( z \pmb { I } - \mathcal { W } ) ^ { - 1 } \mathcal { \mathcal { Q } } _ { \star } .
$$

For $i \in [ 2 n ]$ , let $\mathcal { W } ^ { ( i ) }$ be the principal minor obtained by deleting row and column $i ,$ let ${ \pmb w } _ { i }$ be the deleted of-diagonal column, and define

$$
\begin{array} { r } { \pmb { F } ^ { ( i ) } ( z ) = \left\{ \begin{array} { l l } { ( z \pmb { I } - \mathcal { W } ^ { ( i ) } ) ^ { - 1 } ( \mathcal { Q } _ { \star } ) _ { - i , : } , } & { \| \mathcal { W } ^ { ( i ) } \| _ { \mathrm { o p } } \leq \tau / 5 1 2 , } \\ { \mathbf { 0 } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}
$$

Conditional on $\mathcal { W } ^ { ( i ) }$ , (62) and Lemma B.1, applied to the real and imaginary parts, $\mathrm { g i }$ ve

$$
\operatorname* { P r } \biggr \{ \left\| w _ { i } ^ { T } \boldsymbol { F } ^ { ( i ) } ( z ) \right\| _ { 2 } > \frac { \tau } { 1 0 2 4 } \left\| \boldsymbol { F } ^ { ( i ) } ( z ) \right\| _ { 2 , \infty } \bigg | \mathcal { W } ^ { ( i ) } \biggr \} \leq ( 4 r + 1 ) \exp \left( - c \frac { n q } { \mu r } \right) .\tag{64}
$$

Let Γ be a uniform grid on $| z | = \tau / 6 4$ with

$$
| \Gamma | \leq 5 2 { \sqrt { n } } , \qquad \operatorname* { m i n } _ { z _ { 0 } \in \Gamma } | z - z _ { 0 } | \leq { \frac { \tau } { 1 0 2 4 } } { \sqrt { \frac { \mu r } { n } } } .
$$

A union bound in (63)–(64) yields, except with probability $C n ^ { 3 } \exp ( - c n q / ( \mu r ) )$

$$
\| \mathcal { W } \| _ { \mathrm { o p } } \leq \frac { \tau } { 5 1 2 } , \qquad \left\| \boldsymbol { w } _ { i } ^ { T } \boldsymbol { F } ^ { ( i ) } ( \boldsymbol { z } ) \right\| _ { 2 } \leq \frac { \tau } { 1 0 2 4 } \left\| \boldsymbol { F } ^ { ( i ) } ( \boldsymbol { z } ) \right\| _ { 2 , \infty }\tag{65}
$$

simultaneously for $i \in [ 2 n ]$ and $z \in \Gamma$ . Increasing $c _ { 2 }$ makes this failure probability at most $n ^ { - 1 2 } / 4 8$ . Fix a realization in (65). Then

$$
\left. ( z I - \mathcal { W } ^ { ( i ) } ) ^ { - 1 } \right. _ { \mathrm { o p } } \leq \frac { 5 1 2 } { 7 \tau } , \qquad \left. \pmb { w } _ { i } \right. _ { 2 } \leq \frac { \tau } { 5 1 2 } ,
$$

and

$$
\pmb { F } _ { - i , : } ( z ) = \pmb { F } ^ { ( i ) } ( z ) + ( z \pmb { I } - \varkappa ^ { ( i ) } ) ^ { - 1 } \pmb { w } _ { i } \pmb { F } _ { i , : } ( z ) ,
$$

$$
\left\| F ^ { ( i ) } ( z ) \right\| _ { 2 , \infty } \leq \frac { 8 } { 7 } \left\| F ( z ) \right\| _ { 2 , \infty } ,
$$

$$
F _ { i , : } ( z ) = \frac { ( \mathcal { Q } _ { \star } ) _ { i , : } + { \pmb w } _ { i } ^ { T } { \pmb F } ^ { ( i ) } ( z ) } { z - { \pmb w } _ { i } ^ { T } ( z { \pmb I } - \mathcal { W } ^ { ( i ) } ) ^ { - 1 } { \pmb w } _ { i } } .
$$

Moreover,

$$
\left| z - { \pmb w } _ { i } ^ { T } ( z { \pmb I } - { \pmb \mathcal { W } } ^ { ( i ) } ) ^ { - 1 } { \pmb w } _ { i } \right| \geq \frac { 5 5 \tau } { 3 5 8 4 } > \frac { \tau } { 6 6 } .
$$

Using Assumption 2.1 and (65),

$$
\| \ b { F } ( z ) \| _ { 2 , \infty } \leq \frac { 6 6 } { \tau } \left( \sqrt { \frac { \mu r } { n } } + \frac { \tau } { 8 9 6 } \| \ b { F } ( z ) \| _ { 2 , \infty } \right) \leq \frac { 7 2 } { \tau } \sqrt { \frac { \mu r } { n } } , \qquad z \in \Gamma .
$$

For arbitrary $| z | = \tau / 6 4$ , choose $z _ { 0 } \in \Gamma$ as above. The resolvent identity gives

$$
\| F ( z ) - F ( z _ { 0 } ) \| _ { \mathrm { o p } } \leq \biggl ( \frac { 5 1 2 } { 7 \tau } \biggr ) ^ { 2 } | z - z _ { 0 } | \leq \frac { 6 } { \tau } \sqrt { \frac { \mu r } { n } } ,
$$

and hence

$$
\operatorname* { s u p } _ { | z | = \tau / 6 4 } \left\| ( z I - \mathcal { W } ) ^ { - 1 } \mathcal { Q } _ { \star } \right\| _ { 2 , \infty } \leq \frac { 1 2 8 } { \tau } \sqrt { \frac { \mu r } { n } } .\tag{66}
$$

Finally,

$$
\psi ^ { j } \mathcal { Q } _ { \star } = \frac { 1 } { 2 \pi \mathrm { i } } \oint _ { | z | = \tau / 6 4 } z ^ { j } ( z I - \mathcal { W } ) ^ { - 1 } \mathcal { L } _ { \star } d z , \qquad j \ge 0 .
$$

Combining this identity with (66) proves (37).

口

## C Deterministic geometry of fixed-rank matrices

We prove the geometric estimates used in Sections 5 to 7. We first derive the graph-retraction identity and the sharp-norm bounds at one matrix. We then prove the Frobenius estimates and the comparison bounds for two nearby matrices.

## C.1 Tangent coordinates and the graph-retraction identity

Let $\pmb { X } = \pmb { U } \pmb { \Sigma } \pmb { V } ^ { T } \in \mathcal { M } _ { \tau }$ <sub>r</sub> and $\pmb { \xi } \in T _ { \pmb { X } } \mathcal { M } _ { r }$ . Define the orthogonal tangent coordinates

$$
M : = U ^ { T } \xi V , \qquad B : = ( I - U U ^ { T } ) \xi V , \qquad C : = ( I - V V ^ { T } ) \xi ^ { T } U .
$$

Then $\pmb { U } ^ { T } \pmb { B } = 0 , \pmb { V } ^ { T } \pmb { C } = 0$ , and

$$
\pmb { \xi } = \pmb { U } \pmb { M } \pmb { V } ^ { T } + \pmb { B } \pmb { V } ^ { T } + \pmb { U } \pmb { C } ^ { T } .
$$

Moreover, $U ^ { T } ( X + \pmb { \xi } ) V = \pmb { \Sigma } + \pmb { M }$ . Hence, whenever $\Sigma + M$ is nonsingular, (5) gives

$$
\operatorname { R e t r } _ { X } ( \pmb { \xi } ) = \big ( \pmb { U } + \pmb { B } ( \pmb { \Sigma } + \pmb { M } ) ^ { - 1 } \big ) ( \pmb { \Sigma } + \pmb { M } ) \big ( \pmb { V } + \pmb { C } ( \pmb { \Sigma } + \pmb { M } ) ^ { - T } \big ) ^ { T } .
$$

Expanding the product yields

$$
\operatorname { R e t r } _ { X } ( \xi ) = X + \xi + B ( \Sigma + M ) ^ { - 1 } C ^ { T } .\tag{67}
$$

For later two-base comparisons, we also record the invariance under orthogonal changes of basis. For a compact orthonormal factorization $\mathbf { \boldsymbol { X } } = \mathbf { \boldsymbol { U } } \mathbf { \boldsymbol { S } } \mathbf { \boldsymbol { V } } ^ { T }$ , orthogonal changes of basis $U \mapsto U O _ { U }$ and $V \mapsto V O _ { V }$ leave X, $\mathcal { P } _ { T _ { X } }$ , and the graph retraction unchanged when the core and tangent coordinates are transformed accordingly. Thus, when comparing two matrices, we align their left and right bases separately by orthogonal Procrustes transformations and transform their cores at the same time. The estimates below are invariant under these choices; in particular, $\sigma _ { \mathrm { m i n } } ( S ) = \sigma _ { r } ( X )$

## C.2 Sharp-norm estimates

We begin by controlling the current singular spaces and their projectors in the sharp-norm neighborhood.

## C.2.1 Proof of Lemma 5.1

Proof. Let $E : = X _ { \star } - X$ . Weyl’s inequality [5, Supplement, Theorem 2] gives $\sigma _ { r } ( { \pmb X } ) \geq \sigma _ { r } ( { \pmb X } _ { \star } ) -$ $e _ { \sharp }$ . Since $( I - U _ { \star } U _ { \star } ^ { T } ) X _ { \star } = 0$ and $( I - V _ { \star } \dot { V _ { \star } ^ { T } } ) X _ { \star } ^ { T } = 0$ , we have

$$
( I - U _ { \star } U _ { \star } ^ { T } ) U = - ( I - U _ { \star } U _ { \star } ^ { T } ) E V S ^ { - 1 } ,\tag{68a}
$$

$$
( I - V _ { \star } V _ { \star } ^ { T } ) V = - ( I - V _ { \star } V _ { \star } ^ { T } ) E ^ { T } U S ^ { - T } .\tag{68b}
$$

For every i and $j$ , Assumption 2.1 gives

$$
\operatorname* { m a x } \Big \{ \big \| e _ { i } ^ { T } U _ { \star } U _ { \star } ^ { T } U \big \| _ { 2 } , \big \| e _ { j } ^ { T } V _ { \star } V _ { \star } ^ { T } V \big \| _ { 2 } \Big \} \leq \sqrt { \frac { \mu r } { n } } .
$$

Equations (68a) and (68b), together with $e _ { \sharp } \leq \sigma _ { r } ( \pmb { X } _ { \star } ) / 8$ , give

$$
\left\| e _ { i } ^ { T } ( I - U _ { \star } U _ { \star } ^ { T } ) U \right\| _ { 2 } \leq \frac { \left\| e _ { i } ^ { T } E \right\| _ { 2 } + \left\| U _ { \star } ^ { T } e _ { i } \right\| _ { 2 } \left\| U _ { \star } ^ { T } E \right\| _ { \mathrm { o p } } } { \sigma _ { r } ( X _ { \star } ) - e _ { \sharp } } \leq \frac { 3 \sqrt { \mu r / n } e _ { \sharp } } { \sigma _ { r } ( X _ { \star } ) - e _ { \sharp } } ,
$$

$$
\left. e _ { i } ^ { T } U \right. _ { 2 } \leq \sqrt { \frac { \mu r } { n } } \left( 1 + \frac { 3 e _ { \sharp } } { \sigma _ { r } ( X _ { \star } ) - e _ { \sharp } } \right) \leq \frac { 1 0 } { 7 } \sqrt { \frac { \mu r } { n } } ,
$$

$$
\left\| e _ { j } ^ { T } ( I - V _ { \star } V _ { \star } ^ { T } ) V \right\| _ { 2 } \leq \frac { \left\| e _ { j } ^ { T } E ^ { T } \right\| _ { 2 } + \left\| V _ { \star } ^ { T } e _ { j } \right\| _ { 2 } \left\| V _ { \star } ^ { T } E ^ { T } \right\| _ { \mathrm { o p } } } { \sigma _ { r } ( X _ { \star } ) - e _ { \sharp } } \leq \frac { 3 \sqrt { \mu r / n } e _ { \sharp } } { \sigma _ { r } ( X _ { \star } ) - e _ { \sharp } } ,
$$

$$
\left\| e _ { j } ^ { T } V \right\| _ { 2 } \leq \sqrt { \frac { \mu r } { n } } \left( 1 + \frac { 3 e _ { \sharp } } { \sigma _ { r } ( X _ { \star } ) - e _ { \sharp } } \right) \leq \frac { 1 0 } { 7 } \sqrt { \frac { \mu r } { n } } ,
$$

$$
\operatorname* { m a x } \{ \| \pmb { U } \| _ { 2 , \infty } , \| \pmb { V } \| _ { 2 , \infty } \} \leq \frac { 1 0 } { 7 } \sqrt { \frac { \mu r } { n } } < 2 \sqrt { \frac { \mu r } { n } } .
$$

For two orthogonal projectors of the same rank, the operator norm of their diference equals the largest sine of the principal angles. Hence (68a), (68b), and $e _ { \sharp } \leq \sigma _ { r } ( \pmb { X } _ { \star } ) / 8$ give

$$
\left. U U ^ { T } - U _ { \star } U _ { \star } ^ { T } \right. _ { \mathrm { o p } } = \left. ( I - U _ { \star } U _ { \star } ^ { T } ) U \right. _ { \mathrm { o p } } \leq \frac { e _ { \sharp } } { \sigma _ { r } ( X _ { \star } ) - e _ { \sharp } } ,
$$

$$
\big \| { \boldsymbol { V } } { \boldsymbol { V } } ^ { T } - { \boldsymbol { V } } _ { \star } { \boldsymbol { V } } _ { \star } ^ { T } \big \| _ { \mathrm { o p } } = \big \| ( I - { \boldsymbol { V } } _ { \star } { \boldsymbol { V } } _ { \star } ^ { T } ) { \boldsymbol { V } } \big \| _ { \mathrm { o p } } \leq \frac { e _ { \sharp } } { \sigma _ { r } ( { \boldsymbol { X } } _ { \star } ) - e _ { \sharp } } ,\tag{69}
$$

$$
\operatorname* { m a x } \left\{ \left\| U U ^ { T } - U _ { \star } U _ { \star } ^ { T } \right\| _ { \mathrm { o p } } , \left\| V V ^ { T } - V _ { \star } V _ { \star } ^ { T } \right\| _ { \mathrm { o p } } \right\} \leq \frac { 8 } { 7 } \frac { e _ { \sharp } } { \sigma _ { r } ( X _ { \star } ) } .
$$

For the rowwise projector bounds, the two projector identities, Assumption 2.1, $e _ { \sharp } \le \sigma _ { r } ( \pmb { X } _ { \star } ) / 8$ , and max $\left\{ \| U \| _ { 2 , \infty } , \| V \| _ { 2 , \infty } \right\} \le ( 1 0 / 7 ) \sqrt { \mu r / n }$ give

$$
\pmb { U U } ^ { T } - \pmb { U } _ { \star } \pmb { U } _ { \star } ^ { T } = ( \pmb { I } - \pmb { U } _ { \star } \pmb { U } _ { \star } ^ { T } ) \pmb { U } \pmb { U } ^ { T } - \pmb { U } _ { \star } \pmb { U } _ { \star } ^ { T } ( \pmb { I } - \pmb { U } \pmb { U } ^ { T } ) ,
$$

$$
\bigl \| \boldsymbol { U } \boldsymbol { U } ^ { T } - \boldsymbol { U } _ { \star } \boldsymbol { U } _ { \star } ^ { T } \bigr \| _ { 2 , \infty } \leq \frac { 4 \sqrt { \mu r / n } e _ { \sharp } } { \sigma _ { r } ( \boldsymbol { X } _ { \star } ) - e _ { \sharp } } \leq \frac { 3 2 } { 7 } \sqrt { \frac { \mu r } { n } } \frac { e _ { \sharp } } { \sigma _ { r } ( \boldsymbol { X } _ { \star } ) } ,
$$

$$
\begin{array} { r } { V V ^ { T } - V _ { \star } V _ { \star } ^ { T } = ( I - V _ { \star } V _ { \star } ^ { T } ) V V ^ { T } - V _ { \star } V _ { \star } ^ { T } ( I - V V ^ { T } ) , } \end{array}
$$

$$
\bigl \| \boldsymbol { V } \boldsymbol { V } ^ { T } - \boldsymbol { V } _ { \star } \boldsymbol { V } _ { \star } ^ { T } \bigr \| _ { 2 , \infty } \leq \frac { 4 \sqrt { \mu r / n } e _ { \sharp } } { \sigma _ { r } ( \boldsymbol { X } _ { \star } ) - e _ { \sharp } } \leq \frac { 3 2 } { 7 } \sqrt { \frac { \mu r } { n } } \frac { e _ { \sharp } } { \sigma _ { r } ( \boldsymbol { X } _ { \star } ) } ,
$$

$$
\operatorname* { m a x } \Big \{ \left\| { U U } ^ { T } - { U _ { \star } } { U _ { \star } ^ { T } } \right\| _ { 2 , \infty } , \left\| { V V ^ { T } } - { V _ { \star } } { V _ { \star } ^ { T } } \right\| _ { 2 , \infty } \Big \} \leq \frac { 3 2 } { 7 } \sqrt { \frac { \mu r } { n } } \frac { e _ { \sharp } } { \sigma _ { r } ( X _ { \star } ) } .
$$

Finally, the tangent-projector identity and (69) give

$$
\begin{array} { r l } & { \quad ( \mathcal { P } _ { T _ { X } } - \mathcal { P } _ { T _ { X _ { \star } } } ) Z = ( U U ^ { T } - U _ { \star } U _ { \star } ^ { T } ) Z ( I - V V ^ { T } ) + ( I - U _ { \star } U _ { \star } ^ { T } ) Z ( V V ^ { T } - V _ { \star } V _ { \star } ^ { T } ) , } \\ & { \| \mathcal { P } _ { T _ { X } } - \mathcal { P } _ { T _ { X _ { \star } } } \| _ { \mathrm { F  \mathrm { F } } } \leq \frac { 1 6 } { 7 } \frac { e _ { \sharp } } { \sigma _ { r } ( X _ { \star } ) } . } \end{array}
$$

This proves the lemma.

We next control the normal component and the associated graph factors.

## C.2.2 Proof of Lemma 5.2

Proof. Let $E : = X _ { \star } - X$ . Weyl’s inequality and $\| G _ { X } - S \| _ { \mathrm { o p } } \leq e _ { \sharp }$ give

$$
\sigma _ { \operatorname* { m i n } } ( G _ { X } ) \geq \sigma _ { r } ( X _ { \star } ) - 2 e _ { \sharp } \geq \frac { 3 } { 4 } \sigma _ { r } ( X _ { \star } ) , \qquad \big \| G _ { X } ^ { - 1 } \big \| _ { \mathrm { o p } } \leq \frac { 4 } { 3 \sigma _ { r } ( X _ { \star } ) } .
$$

Relative to the orthogonal decompositions generated by $U$ and $V ,$ the upper-left, lower-left, and upper-right blocks of $X _ { \star }$ are $G _ { X } , ~ L _ { X }$ , and $R _ { X } ^ { T }$ , respectively. Since $\mathrm { r a n k } ( X _ { \star } ) = r$ and $G _ { X }$ is invertible, the Schur complement of $G _ { X }$ vanishes. Therefore, the normal component is exactly

$$
N _ { X } = L _ { X } G _ { X } ^ { - 1 } R _ { X } ^ { T } .\tag{70}
$$

Moreover, we have

$$
\operatorname* { m a x } \{ \| L _ { X } \| _ { \mathrm { o p } } , \| R _ { X } \| _ { \mathrm { o p } } \} \le e _ { \sharp } , \qquad \| N _ { X } \| _ { \mathrm { o p } } \le \frac { 4 } { 3 } \frac { e _ { \sharp } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .
$$

For the rowwise graph factors, the definition of the sharp norm and the incoherence bound in Lemma 5.1 give, for every $i ,$ we have

$$
\left\| e _ { i } ^ { T } L _ { X } \right\| _ { 2 } \leq \left\| e _ { i } ^ { T } E \right\| _ { 2 } + \left\| e _ { i } ^ { T } U \right\| _ { 2 } \left\| U ^ { T } E \right\| _ { \mathrm { o p } } \leq 4 { \sqrt { \frac { \mu r } { n } } } e _ { \sharp } ,
$$

$$
\left\| e _ { i } ^ { T } R _ { X } \right\| _ { 2 } \leq \left\| e _ { i } ^ { T } { \pmb { E } } ^ { T } \right\| _ { 2 } + \left\| e _ { i } ^ { T } { \pmb { V } } \right\| _ { 2 } \left\| { \pmb { V } } ^ { T } { \pmb { E } } ^ { T } \right\| _ { \mathrm { o p } } \leq 4 { \sqrt { \frac { \mu r } { n } } } { \pmb { e } } _ { \sharp } ,
$$

$$
\operatorname* { m a x } \{ \| L _ { X } \| _ { 2 , \infty } , \| R _ { X } \| _ { 2 , \infty } \} \leq 4 \sqrt { \frac { \mu r } { n } } e _ { \sharp } .\tag{71}
$$

Applying (71) to the left and right factors in (70) yields

$$
\operatorname* { m a x } \left\{ \| N _ { X } \| _ { 2 , \infty } , \left\| N _ { X } ^ { T } \right\| _ { 2 , \infty } \right\} \leq \frac { 1 6 } { 3 } \sqrt { \frac { \mu r } { n } } \frac { e _ { \sharp } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .
$$

For every $i , j$ , using the rowwise bounds on both graph factors gives

$$
\left| e _ { i } ^ { T } N _ { X } e _ { j } \right| \leq \left\| e _ { i } ^ { T } L _ { X } \right\| _ { 2 } \left\| G _ { X } ^ { - 1 } \right\| _ { \mathrm { o p } } \left\| e _ { j } ^ { T } R _ { X } \right\| _ { 2 } \leq \frac { 6 4 \mu r } { 3 n } \frac { e _ { \sharp } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } ,
$$

$$
\| N _ { X } \| _ { \infty } \leq \frac { 6 4 \mu r } { 3 n } \frac { e _ { \sharp } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .
$$

Applying (2) to the bounds for $\left\| { N _ { X } } \right\| _ { \mathrm { o p } } , \left\| { N _ { X } } \right\| _ { 2 , \infty } , \left\| { N _ { X } ^ { T } } \right\| _ { 2 , \infty }$ , and $\| \boldsymbol { N } _ { \boldsymbol { X } } \| _ { \infty }$ gives

$$
\| N _ { X } \| _ { \sharp } \leq \frac { 1 6 } { 3 } \frac { e _ { \sharp } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } ,
$$

which proves the sharp-norm bound in Lemma 5.2.

We next use the graph factorization to verify exactness of the population tangent correction.

## C.2.3 Proof of Lemma 5.3

Proof. Equation (4) gives the tangent coordinates

$$
M = G _ { X } - \Sigma , \qquad B = L _ { X } , \qquad C = R _ { X } .
$$

Moreover, Weyl’s inequality gives

$$
\sigma _ { \operatorname* { m i n } } ( G _ { X } ) \geq \sigma _ { r } ( X _ { \star } ) - 2 \left\| X - X _ { \star } \right\| _ { \sharp } \geq \frac { 1 } { 2 } \sigma _ { r } ( X _ { \star } ) > 0 ,
$$

so the updated core is invertible. Since $\operatorname { r a n k } ( X _ { \star } ) = r$ , the Schur complement of $G _ { X }$ in the block representation of X<sub>⋆</sub> vanishes. Using $t _ { X } = \mathcal { P } _ { T _ { X } } ( X _ { \star } - X )$ in (67) gives

$$
\begin{array} { r l } & { \mathrm { R e t r } _ { X } ( t _ { X } ) = X + t _ { X } + L _ { X } G _ { X } ^ { - 1 } R _ { X } ^ { T } , } \\ { \mathcal { P } _ { T _ { X } ^ { \perp } } ( X _ { \star } - X ) = L _ { X } G _ { X } ^ { - 1 } R _ { X } ^ { T } , } \\ & { \quad \mathrm { R e t r } _ { X } ( t _ { X } ) = X + \mathcal { P } _ { T _ { X } } ( X _ { \star } - X ) + \mathcal { P } _ { T _ { X } ^ { \perp } } ( X _ { \star } - X ) = X _ { \star } . } \end{array}
$$

This proves the population identity.

For the proofs of Lemmas 5.4 and 5.5, let $\pmb { X } = \pmb { U } \pmb { \Sigma } \pmb { V } ^ { T } \in \mathcal { M } _ { r }$ and $\pmb { \eta } \in T _ { \pmb { X } } \mathcal { M } _ { r }$ , and define

$$
\begin{array} { r } { { M } _ { \eta } : = { U } ^ { T } \eta { V } , \qquad B _ { \eta } : = ( I - { U } { U } ^ { T } ) \eta { V } , \qquad C _ { \eta } : = ( I - { V } { V } ^ { T } ) \eta ^ { T } { U } , \qquad S _ { \eta } : = G _ { X } + { M } _ { \eta } . } \end{array}\tag{72}
$$

These coordinates describe the perturbation of the population tangent correction.

## C.2.4 Proof of Lemma 5.4

Proof. By the definition of the sharp norm and (72), we have

$$
\begin{array} { r l } & { \operatorname* { m a x } \left\{ \left\| M _ { \eta } \right\| _ { \mathrm { o p } } , \left\| B _ { \eta } \right\| _ { \mathrm { o p } } , \left\| C _ { \eta } \right\| _ { \mathrm { o p } } \right\} \leq \left\| \eta \right\| _ { \sharp } , } \\ & { \qquad \operatorname* { m a x } \left\{ \left\| B _ { \eta } \right\| _ { 2 , \infty } , \left\| C _ { \eta } \right\| _ { 2 , \infty } \right\} \leq 4 \sqrt { \displaystyle \frac { \mu r } { n } } \left\| \eta \right\| _ { \sharp } . } \end{array}
$$

By Lemma 5.2, we have $\| G _ { X } ^ { - 1 } \| _ { \mathrm { o p } } \leq 4 / ( 3 \sigma _ { r } ( X _ { \star } ) )$ . Since $\sigma _ { \mathrm { m i n } } ( G _ { X } ) \geq 3 \sigma _ { r } ( X _ { \star } ) / 4$ and $\| \pmb { \eta } \| _ { \sharp } \le$ $\sigma _ { r } ( \pmb { X } _ { \star } ) / 1 6$ , we have

$$
\big \| ( G _ { X } + M _ { \eta } ) ^ { - 1 } \big \| _ { \mathrm { o p } } \leq \frac { 1 6 } { 1 1 \sigma _ { r } ( X _ { \star } ) } .
$$

Since $e _ { \sharp } \le \sigma _ { r } ( X _ { \star } ) / 8 < \sigma _ { r } ( X _ { \star } ) / 4$ , Lemma 5.3 applies. It follows from Lemma 5.3 and (67) that

$$
\begin{array} { r l } & { \quad \mathrm { R e t r } _ { X } ( t _ { X } + \eta ) - X _ { \star } - \eta } \\ & { = B _ { \eta } S _ { \eta } ^ { - 1 } R _ { X } ^ { T } + L _ { X } S _ { \eta } ^ { - 1 } C _ { \eta } ^ { T } + B _ { \eta } S _ { \eta } ^ { - 1 } C _ { \eta } ^ { T } - L _ { X } G _ { X } ^ { - 1 } M _ { \eta } S _ { \eta } ^ { - 1 } R _ { X } ^ { T } . } \end{array}\tag{73}
$$

For $\pmb { F } , \pmb { H } \in \mathbb { R } ^ { n \times r }$ and $\pmb { K } \in \mathbb { R } ^ { r \times r }$ , we have

$$
\begin{array} { r l } & { \left\| { F K H ^ { T } } \right\| _ { \sharp } \leq \left\| { K } \right\| _ { \mathrm { o p } } \operatorname* { m a x } \bigg \{ \left\| { F } \right\| _ { \mathrm { o p } } \left\| { H } \right\| _ { \mathrm { o p } } , \displaystyle \frac { 1 } { 2 } \sqrt { \frac { n } { \mu r } } \left\| { F } \right\| _ { 2 , \infty } \left\| { H } \right\| _ { \mathrm { o p } } , } \\ & { \qquad \displaystyle \frac { 1 } { 2 } \sqrt { \frac { n } { \mu r } } \left\| { F } \right\| _ { \mathrm { o p } } \left\| { H } \right\| _ { 2 , \infty } , \displaystyle \frac { n } { 4 \mu r } \left\| { F } \right\| _ { 2 , \infty } \left\| { H } \right\| _ { 2 , \infty } \bigg \} . } \end{array}\tag{74}
$$

Applying (74) to the four terms in (73), and using (26), $e _ { \sharp } \leq \sigma _ { r } ( \pmb { X } _ { \star } ) / 8$ , and $\lVert \pmb { \eta } \rVert _ { \sharp } \leq \sigma _ { r } ( \pmb { X } _ { \star } ) / 1 6 .$ gives

$$
\left\| \operatorname { R e t r } _ { X } ( t _ { X } + \eta ) - X _ { \star } - \eta \right\| _ { \sharp } \leq 1 3 \frac { e _ { \sharp } \left\| \eta \right\| _ { \sharp } } { \sigma _ { r } ( X _ { \star } ) } + 6 \frac { \left\| \eta \right\| _ { \sharp } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .
$$

This proves the nonlinear remainder bound.

## C.3 Proof of Lemma 5.5

We next prove the Frobenius estimates used in the local RGD argument and in the quadratic RGN continuation.

Proof of Lemma 5.5. We first prove part (i). Let $\pmb { X } = \pmb { U } \pmb { \Sigma } \pmb { V } ^ { T }$ be a compact singular value decomposition, and use the graph factors in (23). Weyl’s inequality and $\| G _ { X } - \Sigma \| _ { \mathrm { o p } } ~ \leq$ $\| X - X _ { \star } \| _ { \mathrm { F } } \ { \mathrm { g i v e } }$

$$
\sigma _ { \operatorname* { m i n } } ( G _ { X } ) \geq \sigma _ { r } ( { X } _ { \star } ) - 2 \left\| \boldsymbol { X } - { X } _ { \star } \right\| _ { \mathrm { F } } \geq \frac { 1 } { 2 } \sigma _ { r } ( { X } _ { \star } ) , \qquad \left\| G _ { X } ^ { - 1 } \right\| _ { \mathrm { o p } } \leq \frac { 2 } { \sigma _ { r } ( { X } _ { \star } ) } .
$$

Since rank $\mathbf { \nabla } ( X _ { \star } ) = r$ and $G _ { X }$ is invertible, the Schur complement of $G _ { X }$ in the block representation of $X _ { \star }$ vanishes. Moreover, $\| { L } _ { X } \| _ { \mathrm { F } } \leq \| { X } - { X } _ { \star } \| _ { \mathrm { F } }$ and $\| \pmb { R } \pmb { X } \| _ { \mathrm { o p } } \le \| \pmb { X } - \pmb { X } _ { \star } \| _ { \mathrm { F } }$ . Therefore, we have

$$
\mathcal { P } _ { T _ { X } ^ { \perp } } \big ( X _ { \star } - X \big ) = L _ { X } G _ { X } ^ { - 1 } R _ { X } ^ { T } , \qquad \left\| \mathcal { P } _ { T _ { X } ^ { \perp } } \big ( X _ { \star } - X \big ) \right\| _ { \mathrm { F } } \leq 2 \frac { \| X - X _ { \star } \| _ { \mathrm { F } } ^ { 2 } } { \sigma _ { r } \left( X _ { \star } \right) } .
$$

For the tangent-space motion, Weyl’s inequality gives $\sigma _ { r } ( { \pmb X } ) \geq \sigma _ { r } ( { \pmb X } _ { \star } ) - \| { \pmb X } - { \pmb X } _ { \star } \| _ { \mathrm { F } }$ . Hence the equal-rank projector identity gives

$$
\big \| \boldsymbol { U } \boldsymbol { U } ^ { T } - \boldsymbol { U } _ { \star } \boldsymbol { U } _ { \star } ^ { T } \big \| _ { \mathrm { o p } } \leq \frac { \| \boldsymbol { X } - \boldsymbol { X } _ { \star } \| _ { \mathrm { F } } } { \sigma _ { r } ( \boldsymbol { X } _ { \star } ) - \| \boldsymbol { X } - \boldsymbol { X } _ { \star } \| _ { \mathrm { F } } } \leq \frac { 4 \| \boldsymbol { X } - \boldsymbol { X } _ { \star } \| _ { \mathrm { F } } } { 3 \sigma _ { r } ( \boldsymbol { X } _ { \star } ) } ,
$$

$$
\bigl \| \boldsymbol { V } \boldsymbol { V } ^ { T } - \boldsymbol { V } _ { \star } \boldsymbol { V } _ { \star } ^ { T } \bigr \| _ { \mathrm { o p } } \leq \frac { \| \boldsymbol { X } - \boldsymbol { X } _ { \star } \| _ { \mathrm { F } } } { \sigma _ { r } ( \boldsymbol { X } _ { \star } ) - \| \boldsymbol { X } - \boldsymbol { X } _ { \star } \| _ { \mathrm { F } } } \leq \frac { 4 \| \boldsymbol { X } - \boldsymbol { X } _ { \star } \| _ { \mathrm { F } } } { 3 \sigma _ { r } ( \boldsymbol { X } _ { \star } ) } .
$$

Using the tangent-projector identity, we consequently obtain

$$
 \mathcal { P } _ { T _ { X } } - \mathcal { P } _ { T _ { X _ { \star } } }  _ { \mathrm { F  F } } \leq \frac { 8  X - X _ { \star }  _ { \mathrm { F } } } { 3 \sigma _ { r } ( X _ { \star } ) } \leq 3 \frac {  X - X _ { \star }  _ { \mathrm { F } } } { \sigma _ { r } ( X _ { \star } ) } .
$$

This proves part (i).

We next prove part (ii). Let $\pmb { X } = \pmb { U } \pmb { \Sigma } \pmb { V } ^ { T }$ be a compact singular value decomposition, and use the graph factors in (23). Weyl’s inequality and the tangent-coordinate bound $\| M _ { \eta } \| _ { \mathrm { o p } } \le$ $\Vert \eta \Vert _ { \mathrm { F } }$ give

$$
\big \| G _ { X } ^ { - 1 } \big \| _ { \mathrm { o p } } \leq \frac { 1 } { \sigma _ { r } ( X _ { \star } ) - 2 \| X - X _ { \star } \| _ { \mathrm { F } } } , \qquad \big \| S _ { \eta } ^ { - 1 } \big \| _ { \mathrm { o p } } \leq \frac { 1 } { \sigma _ { r } ( X _ { \star } ) - 2 \| X - X _ { \star } \| _ { \mathrm { F } } - \| \eta \| _ { \mathrm { F } } } .
$$

In particular, the two cores are invertible under the hypotheses. Since rank $( X _ { \star } ) = r ,$ the Schur   
complement identity gives $\mathcal { P } _ { T _ { \mathbf { x } } ^ { \perp } } ( X _ { \star } - X ) = L _ { X } G _ { X } ^ { - 1 } R _ { X } ^ { T }$ . Applying (67) to $\mathbf { \boldsymbol { t } } _ { X } + \mathbf { \boldsymbol { \eta } }$ and using X   
$S _ { \eta } ^ { - 1 } - G _ { X } ^ { - 1 } = - G _ { X } ^ { - 1 } M _ { \eta } S _ { \eta } ^ { - 1 }$ , we have

$$
\begin{array} { r l } & { \quad \operatorname { R e t r } _ { X } ( t _ { X } + \eta ) - X _ { \star } - \eta } \\ & { = B _ { \eta } S _ { \eta } ^ { - 1 } R _ { X } ^ { T } + L _ { X } S _ { \eta } ^ { - 1 } C _ { \eta } ^ { T } + B _ { \eta } S _ { \eta } ^ { - 1 } C _ { \eta } ^ { T } - L _ { X } G _ { X } ^ { - 1 } M _ { \eta } S _ { \eta } ^ { - 1 } R _ { X } ^ { T } . } \end{array}\tag{75}
$$

Moreover, max $\{ \| { \pmb L } { \pmb X } \| _ { \mathrm { F } } , \| { \pmb R } { \pmb X } \| _ { \mathrm { F } } \} \ \le \ \| { \pmb X } - { \pmb X } _ { \star } \| _ { \mathrm { F } }$ and max $\left\{ \left\| M _ { \eta } \right\| _ { \mathrm { F } } , \left\| B _ { \eta } \right\| _ { \mathrm { F } } , \left\| C _ { \eta } \right\| _ { \mathrm { F } } \right\} \le \left\| \eta \right\| _ { \mathrm { F } }$ Combining these bounds with (75) gives

$$
\begin{array} { r l } { \| \operatorname { R e t r } _ { X } ( t _ { X } + \eta ) - X _ { \star } - \eta \| _ { \mathrm { F } } \leq \frac { 2 \| X - X _ { \star } \| _ { \mathrm { F } } \| \eta \| _ { \mathrm { F } } + \| \eta \| _ { \mathrm { F } } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) - 2 \| X - X _ { \star } \| _ { \mathrm { F } } - \| \eta \| _ { \mathrm { F } } } } & { } \\ { + \frac { \| X - X _ { \star } \| _ { \mathrm { F } } ^ { 2 } \| \eta \| _ { \mathrm { F } } } { ( \sigma _ { r } ( X _ { \star } ) - 2 \| X - X _ { \star } \| _ { \mathrm { F } } ) ( \sigma _ { r } ( X _ { \star } ) - 2 \| X - X _ { \star } \| _ { \mathrm { F } } - \| \eta \| _ { \mathrm { F } } ) } } & { } \\ { \leq \frac { 1 1 } { 5 } \frac { \| X - X _ { \star } \| _ { \mathrm { F } } \| \eta \| _ { \mathrm { F } } } { \sigma _ { r } ( X _ { \star } ) } + \frac { 1 1 } { 1 0 } \frac { \| \eta \| _ { \mathrm { F } } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } . } & { } \end{array}
$$

This proves part (ii).

## C.4 Comparison of two nearby matrices

We first compare aligned singular factors and graph cores at two nearby matrices.

## C.4.1 Variation of the aligned factors

The next lemma compares the singular factors and graph cores of two nearby matrices. We use the projector and Procrustes estimates in [31, Lemmas 4.1 and 4.5], aligning the left and right bases separately.

Lemma C.1. Let $\pmb { X } , \pmb { Y } \in \mathcal { M } _ { r }$ . For $Z \in \{ X , Y \}$ , represent

$$
{ \pmb Z } = { \pmb U } _ { Z } { \pmb S } _ { Z } { \pmb V } _ { Z } ^ { T }
$$

in independently aligned Procrustes gauges so that $U _ { X } ^ { T } U _ { Y } \succeq 0$ and $V _ { X } ^ { T } V _ { Y } \succeq 0$ , and use the graph factors from (23) for each $Z \in \{ X , Y \}$ . Suppose that

$$
\operatorname* { m a x } \left\{ \| X - X _ { \star } \| _ { \sharp } , \| Y - X _ { \star } \| _ { \sharp } \right\} \leq \rho \leq \frac { \sigma _ { r } ( X _ { \star } ) } { 1 0 0 0 \mu r } .
$$

Then

$$
\operatorname* { m a x } \left\{ \left\| U _ { X } - U _ { Y } \right\| _ { \mathrm { F } } , \left\| V _ { X } - V _ { Y } \right\| _ { \mathrm { F } } , \left\| U _ { X } U _ { X } ^ { T } - U _ { Y } U _ { Y } ^ { T } \right\| _ { \mathrm { F } } , \left\| V _ { X } V _ { X } ^ { T } - V _ { Y } V _ { Y } ^ { T } \right\| _ { \mathrm { F } } \right\} \leq \frac { 3 } { 2 } \frac { d _ { X , Y } } { \sigma _ { r } \left( X _ { \star } \right) } .
$$

Moreover,

$$
\operatorname* { m a x } \left\{ \| L _ { X } - L _ { Y } \| _ { \mathrm { F } } , \| R _ { X } - R _ { Y } \| _ { \mathrm { F } } \right\} \leq \left( 1 + 3 \frac \rho { \sigma _ { r } ( X _ { \star } ) } \right) d _ { X , Y } ,
$$

and

$$
\| G _ { X } - G _ { Y } \| _ { \mathrm { F } } \leq 5 d _ { X , \mathbf { Y } } , \qquad \operatorname* { m a x } _ { Z \in \{ X , Y \} } \Big \| G _ { Z } ^ { - 1 } \Big \| _ { \mathrm { o p } } \leq \frac { 1 } { \sigma _ { r } ( X _ { \star } ) - 2 \rho } .
$$

Finally,

$$
\bigl \| G _ { X } ^ { - 1 } - G _ { Y } ^ { - 1 } \bigr \| _ { \mathrm { F } } \leq \frac { 5 d _ { X , Y } } { \bigl ( \sigma _ { r } ( X _ { \star } ) - 2 \rho \bigr ) ^ { 2 } } .
$$

Proof. By Weyl’s inequality, min $\{ \sigma _ { r } ( { \pmb X } ) , \sigma _ { r } ( { \pmb Y } ) \} > 0$ . Since $( I - U _ { X } U _ { X } ^ { T } ) X = \mathbf { 0 } , Y V _ { Y } S _ { Y } ^ { - 1 } =$ $U _ { Y }$ , and the transposed identities hold on the right, the Procrustes estimates in [31, Lemmas 4.1 and 4.5] give

$$
\begin{array} { r l r } { \displaystyle { \| { \boldsymbol U } _ { X } - { \boldsymbol U } _ { Y } \| _ { \mathrm { F } } \le \frac { 3 } { 2 } \frac { d _ { X , Y } } { \sigma _ { r } ( X _ { \star } ) } } , } & { \qquad } & { \displaystyle { \| { \boldsymbol V } _ { X } - { \boldsymbol V } _ { Y } \| _ { \mathrm { F } } \le \frac { 3 } { 2 } \frac { d _ { X , Y } } { \sigma _ { r } ( X _ { \star } ) } } , } \\ { \displaystyle { \| { \boldsymbol U } _ { X } { \boldsymbol U } _ { X } ^ { T } - { \boldsymbol U } _ { Y } { \boldsymbol U } _ { Y } ^ { T } \| _ { \mathrm { F } } \le \frac { 3 } { 2 } \frac { d _ { X , Y } } { \sigma _ { r } ( X _ { \star } ) } } , } & { \qquad } & { \displaystyle { \| { \boldsymbol V } _ { X } { \boldsymbol V } _ { X } ^ { T } - { \boldsymbol V } _ { Y } { \boldsymbol V } _ { Y } ^ { T } \| _ { \mathrm { F } } \le \frac { 3 } { 2 } \frac { d _ { X , Y } } { \sigma _ { r } ( X _ { \star } ) } } . } \end{array}
$$

This proves the factor and projector estimates.

In the Procrustes gauges of the statement, set

$$
C _ { U } : = U _ { X } ^ { T } U _ { Y } \succeq 0 , \qquad C _ { V } : = V _ { X } ^ { T } V _ { Y } \succeq 0 .
$$

Since rank $( X - Y ) \leq 2 r$ and max $\begin{array} { r } { \mathopen { } \mathclose \bgroup \left\{ \| X - X _ { \star } \| _ { \sharp } , \| Y - X _ { \star } \| _ { \sharp } \aftergroup \egroup \right\} \leq \rho , } \end{array}$ we have $d _ { X , Y } \leq 2 { \sqrt { 2 r } } \rho .$ Combining this with $\rho \le \sigma _ { r } ( X _ { \star } ) / ( 1 0 0 0 \mu r )$ and the projector estimates obtained from [31, Lemmas 4.1 and 4.5] shows that $C _ { U }$ and $C _ { V }$ are invertible. Their definitions give

$$
\begin{array} { r } { I - C _ { U } ^ { 2 } = U _ { Y } ^ { T } ( I - U _ { X } U _ { X } ^ { T } ) U _ { Y } , \qquad I - C _ { V } ^ { 2 } = V _ { Y } ^ { T } ( I - V _ { X } V _ { X } ^ { T } ) V _ { Y } . } \end{array}
$$

Consequently, we have

$$
\begin{array} { r l } & { ( I - C _ { U } ) S _ { Y } = ( I + C _ { U } ) ^ { - 1 } U _ { Y } ^ { T } ( I - U _ { X } U _ { X } ^ { T } ) ( Y - X ) V _ { Y } , } \\ & { S _ { Y } ( I - C _ { V } ) = U _ { Y } ^ { T } ( Y - X ) ( I - V _ { X } V _ { X } ^ { T } ) V _ { Y } ( I + C _ { V } ) ^ { - 1 } . } \end{array}
$$

Since $S _ { X } = U _ { X } ^ { T } X V _ { X }$ and ${ \cal S } _ { Y } = U _ { Y } ^ { T } { \cal Y } V _ { Y }$ , substituting these identities gives

$$
\begin{array} { r l } & { { S } _ { X } - { S } _ { Y } = { U } _ { X } ^ { T } ( X - Y ) { V } _ { X } + ( { C } _ { U } - { I } ) { S } _ { Y } { C } _ { V } + { S } _ { Y } ( { C } _ { V } - { I } ) , } \\ & { \| { S } _ { X } - { S } _ { Y } \| _ { \mathrm { F } } \leq 3 d _ { X , Y } . } \end{array}
$$

Using the definitions of the graph factors together with the factor and projector bounds, we have

$$
\Vert L _ { X } - L _ { Y } \Vert _ { \mathrm { F } } \leq d _ { X , Y } + 3 \frac { \rho d _ { X , Y } } { \sigma _ { r } ( X _ { \star } ) } ,
$$

$$
\Vert R _ { X } - R _ { Y } \Vert _ { \mathrm { F } } \leq d _ { X , Y } + 3 { \frac { \rho d _ { X , Y } } { \sigma _ { r } ( X _ { \star } ) } } .
$$

Moreover, we have

$$
\left\| U _ { X } ^ { T } ( X _ { \star } - X ) V _ { X } - U _ { Y } ^ { T } ( X _ { \star } - Y ) V _ { Y } \right\| _ { \mathrm { F } } \leq d _ { X , Y } + 3 \frac { \rho d _ { X , Y } } { \sigma _ { r } ( X _ { \star } ) } .
$$

Combining the preceding estimate with $\| S _ { X } - S _ { Y } \| _ { \mathrm { F } } \leq 3 d _ { X , Y }$ yields

$$
\Vert { G _ { X } - G _ { Y } } \Vert _ { \mathrm { F } } \leq 5 d _ { X , Y } .
$$

For $Z \in \{ X , Y \}$ , the definition of $G _ { Z }$ and Weyl’s inequality give

$$
\sigma _ { \operatorname* { m i n } } ( G _ { Z } ) \geq \sigma _ { r } ( X _ { \star } ) - 2 \rho , \qquad \left\| G _ { Z } ^ { - 1 } \right\| _ { \mathrm { o p } } \leq \frac { 1 } { \sigma _ { r } ( X _ { \star } ) - 2 \rho } .
$$

The resolvent identity then gives

$$
\bigl \| G _ { X } ^ { - 1 } - G _ { Y } ^ { - 1 } \bigr \| _ { \mathrm { F } } \leq \frac { 5 d _ { X , Y } } { \bigl ( \sigma _ { r } ( X _ { \star } ) - 2 \rho \bigr ) ^ { 2 } } .
$$

This completes the proof.

We now apply the factor estimates to the graph-retraction identity. The proof also verifies that both updated cores are invertible, so both retractions are well defined.

## C.4.2 Proof of Lemma 7.6

Proof. We align the singular factors as in Lemma C.1. For $Z \in \{ X , Y \}$ , write

$$
M _ { Z } : = U _ { Z } ^ { T } \eta _ { Z } V _ { Z } , \qquad B _ { Z } : = ( I - U _ { Z } U _ { Z } ^ { T } ) \eta _ { Z } V _ { Z } ,
$$

and

$$
\begin{array} { r } { C _ { Z } : = ( I - V _ { Z } V _ { Z } ^ { T } ) \eta _ { Z } ^ { T } U _ { Z } , \qquad S _ { \eta , Z } : = G _ { Z } + M _ { Z } . } \end{array}
$$

Set

$$
s _ { \eta } : = \operatorname* { m a x } \{ \| \eta _ { X } \| _ { \mathrm { F } } , \| \eta _ { Y } \| _ { \mathrm { F } } \} , \quad \quad \Delta : = d _ { \eta } + 3 \frac { s _ { \eta } d _ { X , Y } } { \sigma _ { r } ( X _ { \star } ) } .
$$

Using the aligned bases, we have

$$
\begin{array} { r l } & { M _ { X } - M _ { Y } = ( U _ { X } - U _ { Y } ) ^ { T } \eta _ { X } V _ { X } + U _ { Y } ^ { T } ( \eta _ { X } - \eta _ { Y } ) V _ { X } + U _ { Y } ^ { T } \eta _ { Y } ( V _ { X } - V _ { Y } ) , } \\ & { \ B _ { X } - B _ { Y } = ( U _ { Y } U _ { Y } ^ { T } - U _ { X } U _ { X } ^ { T } ) \eta _ { X } V _ { X } + ( I - U _ { Y } U _ { Y } ^ { T } ) ( \eta _ { X } - \eta _ { Y } ) V _ { X } } \\ & { \qquad + ( I - U _ { Y } U _ { Y } ^ { T } ) \eta _ { Y } ( V _ { X } - V _ { Y } ) , } \\ & { C _ { X } - C _ { Y } = ( V _ { Y } V _ { Y } ^ { T } - V _ { X } V _ { X } ^ { T } ) \eta _ { X } ^ { T } U _ { X } + ( I - V _ { Y } V _ { Y } ^ { T } ) ( \eta _ { X } - \eta _ { Y } ) ^ { T } U _ { X } } \\ & { \qquad + ( I - V _ { Y } V _ { Y } ^ { T } ) \eta _ { Y } ^ { T } ( U _ { X } - U _ { Y } ) . } \end{array}
$$

Therefore, Lemma C.1 gives

$$
\operatorname* { m a x } \left\{ \left\| M _ { X } - M _ { Y } \right\| _ { \mathrm { F } } , \left\| B _ { X } - B _ { Y } \right\| _ { \mathrm { F } } , \left\| C _ { X } - C _ { Y } \right\| _ { \mathrm { F } } \right\} \leq \Delta .\tag{76}
$$

Moreover, the definitions of the tangent and graph coordinates imply

$$
\operatorname* { m a x } _ { z \in \{ X , Y \} } \operatorname* { m a x } \left\{ \left\| M z \right\| _ { \mathrm { F } } , \left\| B z \right\| _ { \mathrm { F } } , \left\| C z \right\| _ { \mathrm { F } } \right\} \leq s _ { \eta } , \qquad \operatorname* { m a x } _ { z \in \{ X , Y \} } \operatorname* { m a x } \left\{ \left\| L z \right\| _ { \mathrm { o p } } , \left\| R z \right\| _ { \mathrm { o p } } \right\} \leq \rho .\tag{77}
$$

The correction-size hypothesis gives $s _ { \eta } \le 9 \mu r \rho ^ { 2 } / \sigma _ { r } ( X _ { \star } )$ . Furthermore, Weyl’s inequality and (77) give

$$
\sigma _ { \operatorname* { m i n } } ( S _ { \eta , Z } ) \geq \sigma _ { r } ( X _ { \star } ) - 2 \rho - s _ { \eta } > 0 , \qquad \left\| S _ { \eta , Z } ^ { - 1 } \right\| _ { \mathrm { o p } } \leq \frac { 1 } { \sigma _ { r } ( X _ { \star } ) - 2 \rho - s _ { \eta } } , \qquad Z \in \{ X , Y \} .
$$

Using the resolvent identity, Lemma C.1, and (76), we also obtain

$$
\left. S _ { \eta , X } ^ { - 1 } - S _ { \eta , Y } ^ { - 1 } \right. _ { \mathrm { F } } \leq \frac { 5 d _ { X , Y } + \Delta } { \left( \sigma _ { r } ( X _ { \star } ) - 2 \rho - s _ { \eta } \right) ^ { 2 } } .\tag{78}
$$

Since $\rho < \sigma _ { r } ( X _ { \star } ) / 4$ , Lemma 5.3 applies at both base points. Hence (67) gives, for $Z \in$ $\{ X , Y \}$ , we have

$$
\begin{array} { r l } & { \quad \mathrm { R e t r } _ { Z } ( t _ { Z } + \eta _ { Z } ) - X _ { \star } - \eta _ { Z } } \\ & { = B _ { Z } S _ { \eta , Z } ^ { - 1 } R _ { Z } ^ { T } + L _ { Z } S _ { \eta , Z } ^ { - 1 } C _ { Z } ^ { T } + B _ { Z } S _ { \eta , Z } ^ { - 1 } C _ { Z } ^ { T } - L _ { Z } G _ { Z } ^ { - 1 } M _ { Z } S _ { \eta , Z } ^ { - 1 } R _ { Z } ^ { T } . } \end{array}\tag{79}
$$

We compare the four terms in (79). In each product, we use the Frobenius norm for a correction or a diference and the operator norm for the remaining graph factors. For the first term, we have

$$
\begin{array} { r l } & { \quad B _ { X } S _ { \eta , X } ^ { - 1 } R _ { X } ^ { T } - B _ { Y } S _ { \eta , Y } ^ { - 1 } R _ { Y } ^ { T } } \\ & { = ( B _ { X } - B _ { Y } ) S _ { \eta , X } ^ { - 1 } R _ { X } ^ { T } + B _ { Y } ( S _ { \eta , X } ^ { - 1 } - S _ { \eta , Y } ^ { - 1 } ) R _ { X } ^ { T } + B _ { Y } S _ { \eta , Y } ^ { - 1 } ( R _ { X } - R _ { Y } ) ^ { T } . } \end{array}
$$

Interchanging the left and right factors gives the corresponding expansion for $L _ { Z } S _ { \eta , Z } ^ { - 1 } C _ { Z } ^ { T }$ . For the third term, we have

$$
\begin{array} { r l } & { \quad B _ { X } S _ { \eta , X } ^ { - 1 } C _ { X } ^ { T } - B _ { Y } S _ { \eta , Y } ^ { - 1 } C _ { Y } ^ { T } } \\ & { = ( B _ { X } - B _ { Y } ) S _ { \eta , X } ^ { - 1 } C _ { X } ^ { T } + B _ { Y } ( S _ { \eta , X } ^ { - 1 } - S _ { \eta , Y } ^ { - 1 } ) C _ { X } ^ { T } + B _ { Y } S _ { \eta , Y } ^ { - 1 } ( C _ { X } - C _ { Y } ) ^ { T } . } \end{array}
$$

Finally, the five-factor diference has the telescoping expansion

$$
\begin{array} { r l } & { \quad L _ { X } G _ { X } ^ { - 1 } M _ { X } S _ { \eta , X } ^ { - 1 } R _ { X } ^ { T } - L _ { Y } G _ { Y } ^ { - 1 } M _ { Y } S _ { \eta , Y } ^ { - 1 } R _ { Y } ^ { T } } \\ & { = ( L _ { X } - L _ { Y } ) G _ { X } ^ { - 1 } M _ { X } S _ { \eta , X } ^ { - 1 } R _ { X } ^ { T } + L _ { Y } ( G _ { X } ^ { - 1 } - G _ { Y } ^ { - 1 } ) M _ { X } S _ { \eta , X } ^ { - 1 } R _ { X } ^ { T } } \\ & { \qquad + L _ { Y } G _ { Y } ^ { - 1 } ( M _ { X } - M _ { Y } ) S _ { \eta , X } ^ { - 1 } R _ { X } ^ { T } + L _ { Y } G _ { Y } ^ { - 1 } M _ { Y } ( S _ { \eta , X } ^ { - 1 } - S _ { \eta , Y } ^ { - 1 } ) R _ { X } ^ { T } } \\ & { \qquad + L _ { Y } G _ { Y } ^ { - 1 } M _ { Y } S _ { \eta , Y } ^ { - 1 } ( R _ { X } - R _ { Y } ) ^ { T } . } \end{array}
$$

Applying Lemma C.1 and (76)–(78) termwise in these expansions, and then using $\Delta = d _ { \eta } +$ $3 s _ { \eta } d _ { X , Y } / \sigma _ { r } ( X _ { \star } ) , s _ { \eta } \le 9 \mu r \rho ^ { 2 } / \sigma _ { r } ( X _ { \star } )$ , and $\rho \le \sigma _ { r } ( X _ { \star } ) / ( 1 0 0 0 \mu r )$ , yields

$$
\Vert \mathrm { R e t r } _ { X } ( t _ { X } + \eta _ { X } ) - \mathrm { R e t r } _ { Y } ( t _ { Y } + \eta _ { Y } ) - ( \eta _ { X } - \eta _ { Y } ) \Vert _ { \mathrm { F } } \leq \frac { 1 } { 4 } d _ { \eta } + 1 9 \frac { \mu r \rho ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) ^ { 2 } } d _ { X , Y } .
$$

Adding $\| \eta _ { X } - \eta _ { Y } \| _ { \mathrm { F } } = d _ { \eta }$ proves (54).

## D Sharp-norm estimates for spectral initialization

We prove the deterministic reconstruction estimate and the finite approximation used in Section 6. The estimates concern the reconstructed matrix and do not require separation between adjacent singular values.

## D.1 Proof of Lemma 6.2

Proof of Lemma 6.2. Suppose first that ${ \pmb Z } ^ { + } \neq { \bf 0 }$ . With the dilation in (36), set

$$
\widehat { \mathcal { Q } } : = \frac { 1 } { \sqrt { 2 } } \left[ \widehat { \pmb { U } } \quad \widehat { \pmb { U } } \right] , \qquad \mathcal { D } : = \mathrm { d i a g } ( \widehat { \pmb { \Sigma } } , - \widehat { \pmb { \Sigma } } ) , \qquad \pmb { P } : = \mathcal { Q } _ { \star } \mathcal { Q } _ { \star } ^ { T } .
$$

The columns of $\hat { \mathcal { Q } }$ are orthonormal, and $\begin{array} { r } { \left. \mathcal { D } ^ { - 1 } \right. _ { \mathrm { o p } } \leq 8 / \tau } \end{array}$ . The one-sided singular equation in b(38a) and the residual bound in (38b) give

$$
\left( \left[ \begin{array} { c c } { { \bf 0 } } & { X _ { \star } } \\ { X _ { \star } ^ { T } } & { { \bf 0 } } \end{array} \right] + \mathcal { W } \right) \widehat { \mathcal { Q } } = \widehat { \mathcal { Q } } \mathcal { D } + \mathcal { F } , \qquad \| \mathcal { F } \| _ { \mathrm { o p } } \leq \frac { \tau } { 6 4 } \sqrt { \frac { \mu r } { n } } .\tag{80}
$$

The coeficient of the true singular spaces is

$$
\begin{array} { r } { \pmb { C } : = \left[ \begin{array} { l l } { \mathbf { 0 } } & { \pmb { \Sigma } _ { \star } } \\ { \pmb { \Sigma } _ { \star } } & { \mathbf { 0 } } \end{array} \right] \mathcal { Q } _ { \star } ^ { T } \mathcal { \widehat { Q } } \mathcal { D } ^ { - 1 } . } \end{array}
$$

By (80), we have

$$
\mathcal { Q } _ { \star } C = \widehat { \mathcal { Q } } - \mathcal { W } \widehat { \mathcal { Q } } \mathcal { D } ^ { - 1 } + \mathcal { F } \mathcal { D } ^ { - 1 } , \qquad \| C \| _ { \mathrm { o p } } \leq 1 + \frac { 8 } { \tau } \left( \frac { \tau } { 5 1 2 } + \frac { \tau } { 6 4 } \sqrt { \frac { \mu r } { n } } \right) \leq \frac { 5 } { 4 } .
$$

In particular, this estimate does not use the ratio $\sigma _ { 1 } ( X _ { \star } ) / \tau$

Since $\| \mathcal { W } \| _ { \mathrm { o p } } \left\| \mathcal { D } ^ { - 1 } \right\| _ { \mathrm { o p } } \leq 1 / 6 4$ , the same equation has the convergent expansion

$$
\widehat { \mathcal { Q } } = \sum _ { j \geq 0 } \mathcal { W } ^ { j } \mathcal { Q } _ { \star } C \mathcal { D } ^ { - j } - \sum _ { j \geq 0 } \mathcal { W } ^ { j } \mathcal { F } \mathcal { D } ^ { - j - 1 } .\tag{81}
$$

For $j \geq 1$ , (37) implies

$$
\left\| ( I - P ) \mathcal { W } ^ { j } \mathcal { Q } _ { \star } \right\| _ { 2 , \infty } \leq 2 \sqrt { \frac { \mu r } { n } } \left( \frac { \tau } { 6 4 } \right) ^ { j } + \sqrt { \frac { \mu r } { n } } \left\| \mathcal { W } \right\| _ { \mathrm { o p } } ^ { j } \leq 3 \sqrt { \frac { \mu r } { n } } \left( \frac { \tau } { 6 4 } \right) ^ { j } .
$$

Also, $\left\| ( I - P ) \mathcal { W } ^ { j } \mathcal { F } \right\| _ { 2 , \infty } \leq \| \mathcal { W } \| _ { \mathrm { o p } } ^ { j } \| \mathcal { F } \| _ { \mathrm { o p } }$ . Applying these estimates to (81) gives

$$
\left\| ( I - P ) \widehat { \mathcal { Q } } \mathcal { D } \right\| _ { 2 , \infty } \leq \sqrt { \frac { \mu r } { n } } \left( \frac { 1 5 \tau } { 2 5 6 ( 1 - 1 / 8 ) } + \frac { \tau } { 6 4 ( 1 - 1 / 6 4 ) } \right) \leq \frac { 3 \tau } { 3 2 } \sqrt { \frac { \mu r } { n } } ,\tag{82}
$$

$$
\left\| ( I - P ) \widehat { \mathcal { Q } } \right\| _ { 2 , \infty } \leq \frac { 3 } { 4 } \sqrt { \frac { \mu r } { n } } , \qquad \left\| ( I - P ) \widehat { \mathcal { Q } } \mathcal { D } \widehat { \mathcal { Q } } ^ { T } ( I - P ) \right\| _ { \infty } \leq \frac { 9 \tau \mu r } { 1 2 8 n } .\tag{83}
$$

The matrix $\widehat { \mathcal { Q } } \mathcal { D } \widehat { \mathcal { Q } } ^ { T }$ is the symmetric dilation of $Z ^ { + }$ . By (38b), we have

$$
\left. Z ^ { + } - X _ { \star } \right. _ { \mathrm { o p } } \leq \frac { 7 \tau } { 3 2 } + \frac { \tau } { 5 1 2 } = \frac { 1 1 3 \tau } { 5 1 2 } .\tag{84}
$$

Decompose its error on the left by $_ { P }$ and ${ \pmb I } - { \pmb P }$ . Since $\| \mathcal { L } _ { \star } \| _ { 2 , \infty } \leq \sqrt { \mu r / n }$ , (82)–(84) yield

$$
\operatorname* { m a x } \{ \left\| Z ^ { + } - \mathbf { X } _ { \star } \right\| _ { 2 , \infty } , \left\| ( Z ^ { + } - \mathbf { X } _ { \star } ) ^ { T } \right\| _ { 2 , \infty } \} \leq \left( \frac { 1 1 3 } { 5 1 2 } + \frac { 3 } { 3 2 } \right) \tau \sqrt { \frac { \mu r } { n } } = \frac { 1 6 1 \tau } { 5 1 2 } \sqrt { \frac { \mu r } { n } } .
$$

For the entrywise norm, decompose the error on both sides by the same projections. The truespace term is bounded by $( \mu r / n ) \| Z ^ { + } - X _ { \star } \| _ { \mathrm { o p } }$ , the two mixed terms by $3 \tau \mu r / ( 3 2 n )$ each, and the remaining term by (83). Therefore,

$$
\left\| Z ^ { + } - X _ { \star } \right\| _ { \infty } \leq \left( \frac { 1 1 3 } { 5 1 2 } + \frac { 6 } { 3 2 } + \frac { 9 } { 1 2 8 } \right) \frac { \tau \mu r } { n } = \frac { 2 4 5 \tau \mu r } { 5 1 2 n } .
$$

Combining these three bounds in (2) gives $\lVert Z ^ { + } - X _ { \star } \rVert _ { \sharp } \leq \tau / 4$

If ${ \pmb Z } ^ { + } = { \bf 0 }$ , then $\sigma _ { 1 } ( \mathbf { X _ { \star } } ) \leq \| \pmb { Y } \| _ { \mathrm { o p } } + \| \pmb { W } \| _ { \mathrm { o p } } \leq 1 1 3 \tau / 5 1 2$ . Incoherence gives $\| \pmb { X } _ { \star } \| _ { \sharp } = \sigma _ { 1 } ( \pmb { X } _ { \star } )$ so the same conclusion follows. □

## D.2 Proof of Lemma 6.3

Proof of Lemma 6.3. We condition on the fixed matrix $\mathbf { Y } .$ . Let $d _ { 1 } \geq \dots \geq d _ { n } > 0$ be the eigenvalues of $Y Y ^ { T } + \tau ^ { 2 } I / 4 0 9 6$ , with corresponding orthonormal eigenvectors $\mathbf { } { \pmb u } _ { 1 } , \ldots , { \pmb u } _ { n }$ . The hypothesis on $\sigma _ { r + 1 } ( Y )$ gives

$$
d _ { r + 1 } \leq \frac { \tau ^ { 2 } } { 5 1 2 ^ { 2 } } + \frac { \tau ^ { 2 } } { 4 0 9 6 } < \frac { \tau ^ { 2 } } { 4 0 0 0 } .\tag{85}
$$

Let $U _ { 1 }$ contain the eigenvectors whose eigenvalues are at least $\tau ^ { 2 } / 1 2 8$ , and let $D _ { 1 }$ be the diagonal matrix of those eigenvalues. By (85), their number is at most r. Let $U _ { 2 }$ and $D _ { 2 }$ contain the remaining eigenvectors and eigenvalues, so that

$$
{ \pmb Y } { \pmb Y } ^ { T } + \frac { \tau ^ { 2 } } { 4 0 9 6 } { \pmb I } = { \pmb U } _ { 1 } { \pmb D } _ { 1 } { \pmb U } _ { 1 } ^ { T } + { \pmb U } _ { 2 } { \pmb D } _ { 2 } { \pmb U } _ { 2 } ^ { T } , \qquad \| { \pmb D } _ { 2 } \| _ { \mathrm { o p } } \leq \frac { \tau ^ { 2 } } { 1 2 8 } .
$$

These decompositions are used only in the proof, not in the computation of $\tau$

Step 1: the initial subspace. Put $U _ { [ r ] } = [ \pmb { u } _ { 1 } , \ldots , \pmb { u } _ { r } ]$ . In the eigenbasis $[ U _ { [ r ] } , U _ { [ r ] } ^ { \perp } ]$ , write the Gaussian initial matrix as $[ G _ { 1 } ^ { T } , G _ { 2 } ^ { T } ] ^ { T }$ , where $G _ { 1 } \in \mathbb { R } ^ { r \times r }$ . Orthogonal invariance preserves the standard Gaussian distribution. Since $\operatorname* { P r } \{ | g | \leq t \} \leq t$ for $g \sim N ( 0 , 1 )$ ,

$$
\operatorname* { P r } \biggr \{ \operatorname* { m i n } _ { 1 \leq j \leq r } \mathrm { d i s t } \bigl ( ( G _ { 1 } ) _ { : , j } , \mathrm { s p a n } \{ ( G _ { 1 } ) _ { : , k } : k \neq j \} \bigr ) < n ^ { - 2 0 } \biggr \} \leq n ^ { - 1 9 } ,
$$

and therefore

$$
\left. \mathbfcal { G } _ { 1 } ^ { - 1 } \right. _ { \mathrm { o p } } \leq \sqrt { r } n ^ { 2 0 } .
$$

Moreover,

$$
\begin{array} { r } { \mathbb { E } \left\| G _ { 2 } \right\| _ { \mathrm { F } } ^ { 2 } \leq n ^ { 2 } , \qquad \operatorname* { P r } \{ \| G _ { 2 } \| _ { \mathrm { F } } > n ^ { 1 1 } \} \leq n ^ { - 2 0 } . } \end{array}
$$

Hence, with probability at least $1 - 2 n ^ { - 1 9 } \geq 1 - n ^ { - 1 2 } / 4 8$

$$
\begin{array} { r } { \left\| \boldsymbol { G } _ { 2 } \boldsymbol { G } _ { 1 } ^ { - 1 } \right\| _ { \mathrm { o p } } \leq n ^ { 3 2 } . } \end{array}\tag{86}
$$

Step 2: weighted subspace estimates. Fix a realization satisfying (86). Thin QR does not change the column space, so (11) gives

$$
\operatorname { r a n g e } ( Q _ { t } ) = \operatorname { r a n g e } \left( \left( Y Y ^ { T } + { \frac { \tau ^ { 2 } } { 4 0 9 6 } } I \right) ^ { t } G \right) .
$$

Relative to ${ \pmb U } _ { [ r ] } \oplus { \pmb U } _ { [ r ] } ^ { \perp }$ , this space has graph matrix

$$
\mathrm { d i a g } ( d _ { r + 1 } ^ { t } , \ldots , d _ { n } ^ { t } ) G _ { 2 } G _ { 1 } ^ { - 1 } \mathrm { d i a g } ( d _ { 1 } ^ { - t } , \ldots , d _ { r } ^ { - t } ) .
$$

Since ran $\mathsf { y e } ( U _ { 1 } ) \subseteq \mathrm { r a n g e } ( U _ { [ r ] } )$ , (85) and $d _ { j } \ \ge \ \tau ^ { 2 } / 1 2 8$ on range(U<sub>1</sub>) imply, for $t \geq 1$ and $s \in \{ 0 , 1 / 2 , 1 \}$ , we have

$$
\left\| ( I - Q _ { t } Q _ { t } ^ { T } ) U _ { 1 } D _ { 1 } ^ { s } \right\| _ { \mathrm { o p } } \leq \left( { \frac { \tau ^ { 2 } } { 1 2 8 } } \right) ^ { s } n ^ { 3 2 } 3 0 ^ { - t } .\tag{87}
$$

For $t = \lceil 1 2 \log n \rceil$ and $n \geq 2 .$ , we have

$$
n ^ { 3 2 } 3 0 ^ { - t } \leq n ^ { - 7 } \leq { \frac { 1 } { 6 4 } } { \sqrt { \frac { \mu r } { n } } } .
$$

Let $Q$ denote the final factor, and put ${ \pmb { P } } = { \pmb { Q } } { \pmb { Q } } ^ { T }$ and ${ \pmb E } = ( { \pmb I } - { \pmb P } ) { \pmb U } _ { 1 }$ . Then

$$
\| E D _ { 1 } ^ { s } \| _ { \mathrm { o p } } \leq \frac { 1 } { 6 4 } \sqrt { \frac { \mu r } { n } } \left( \frac { \tau ^ { 2 } } { 1 2 8 } \right) ^ { s } , \qquad s \in \{ 0 , 1 / 2 , 1 \} .\tag{88}
$$

If $U _ { 1 }$ has no columns, then $\| \mathbf { \boldsymbol { Y } } \| _ { \mathrm { o p } } < \tau / ( 8 \sqrt { 2 } ) < \tau / 8$ . No singular value is retained and the conclusion is immediate. We henceforth suppose that $U _ { 1 }$ is nonempty.

Step 3: the retained singular factors. An orthonormal basis for $\mathrm { r a n g e } ( P U _ { 1 } )$ is

$$
\widetilde { \pmb { Q } } _ { 1 } = ( \pmb { U } _ { 1 } - \pmb { E } ) ( \pmb { I } - \pmb { E } ^ { T } \pmb { E } ) ^ { - 1 / 2 } , \qquad \left\| ( \pmb { I } - \pmb { E } ^ { T } \pmb { E } ) ^ { - 1 / 2 } \right\| _ { \mathrm { o p } } \le \frac { 4 } { 3 } .
$$

Complete it inside range(Q) by ${ \widetilde { Q } } _ { 2 }$ . Then $\widetilde { Q } _ { 2 } ^ { T } U _ { 1 } = \mathbf { 0 }$ . Since ${ \pmb U } _ { 1 } ^ { T } { \pmb E } = { \pmb E } ^ { T } { \pmb E }$ , direct substitution gives

$$
\begin{array} { r l } & { ( I - P ) Y Y ^ { T } \widetilde { Q } _ { 1 } = \left[ E D _ { 1 } - E D _ { 1 } E ^ { T } E - ( I - P ) U _ { 2 } D _ { 2 } U _ { 2 } ^ { T } E \right] ( I - E ^ { T } E ) ^ { - 1 / 2 } , } \\ & { \qquad \widetilde { Q } _ { 2 } ^ { T } Y Y ^ { T } \widetilde { Q } _ { 1 } = - \widetilde { Q } _ { 2 } ^ { T } U _ { 2 } D _ { 2 } U _ { 2 } ^ { T } E ( I - E ^ { T } E ) ^ { - 1 / 2 } . } \end{array}
$$

Using (88) and $\mu r / n \leq 1$ , these identities imply

$$
\left. ( I - P ) \pmb { Y } \pmb { Y } ^ { T } \widetilde { \pmb { Q } } _ { 1 } \right. _ { \mathrm { o p } } \le \frac { \tau ^ { 2 } } { 2 0 4 8 } \sqrt { \frac { \mu r } { n } } ,\tag{89a}
$$

$$
\Bigl \| \widetilde { Q } _ { 2 } ^ { T } Y Y ^ { T } \widetilde { Q } _ { 1 } \Bigr \| _ { \mathrm { o p } } \le \frac { \tau ^ { 2 } } { 4 0 9 6 } \sqrt { \frac { \mu r } { n } } , \qquad \widetilde { Q } _ { 2 } ^ { T } Y Y ^ { T } \widetilde { Q } _ { 2 } \preceq \frac { \tau ^ { 2 } } { 1 2 8 } I .\tag{89b}
$$

The compressed SVD in (12) gives $\pmb { Y } ^ { T } \pmb { \hat { U } } = \pmb { \hat { V } } \pmb { \hat { \Sigma } }$ and $\sigma _ { \mathrm { m i n } } ( \widehat { \Sigma } ) \geq \tau / 8$ . Its retained left factors are Ritz vectors of $\mathbf { \Delta } _ { Y Y } \mathbf { \Delta } _ { T }$ bin range(Q). Write $\widehat { U } = \widetilde { Q } _ { 1 } Z _ { 1 } + \widetilde { Q } _ { 2 } Z _ { 2 }$ . The lower block of the Ritz equation is

$$
\begin{array} { r } { { \cal Z } _ { 2 } \widehat { \Sigma } ^ { 2 } - ( \widetilde { \pmb { Q } } _ { 2 } ^ { T } { \cal Y } { \pmb { Y } } ^ { T } \widetilde { \pmb { Q } } _ { 2 } ) { \cal Z } _ { 2 } = \widetilde { \pmb { Q } } _ { 2 } ^ { T } { \pmb { Y } } { \pmb { Y } } ^ { T } \widetilde { \pmb { Q } } _ { 1 } { \cal Z } _ { 1 } . } \end{array}
$$

The spectra on the two sides are separated by at least $\tau ^ { 2 } / 1 2 8$ . The integral solution of this Sylvester equation, together with (89b) and $\| Z _ { 1 } \| _ { \mathrm { o p } } \leq 1$ , therefore gives

$$
\| Z _ { 2 } \| _ { \mathrm { o p } } \leq \frac { 1 2 8 } { \tau ^ { 2 } } \left\| \widetilde { Q } _ { 2 } ^ { T } { \cal Y } { \cal Y } ^ { T } \widetilde { Q } _ { 1 } Z _ { 1 } \right\| _ { \mathrm { o p } } \leq \frac { 1 } { 3 2 } \sqrt { \frac { \mu r } { n } } .
$$

Since ${ \widetilde { Q } } _ { 2 }$ is orthogonal to $U _ { 1 }$ , also $\left. \mathbf { { \cal Y } } \mathbf { { \cal Y } } ^ { T } \widetilde { \mathbf { { Q } } } _ { 2 } \right. _ { \mathrm { o p } } \le \tau ^ { 2 } / 1 2 8$ . Combining this with (89a) yields

$$
\Big \Vert ( I - P ) \pmb { Y } \pmb { Y } ^ { T } \pmb { \widehat { U } } \Big \Vert _ { \mathrm { o p } } \leq \frac { \tau ^ { 2 } } { 1 0 2 4 } \sqrt { \frac { \mu r } { n } } .
$$

Moreover, $P Y \widehat { V } = \widehat { U } \widehat { \Sigma }$ . Consequently,

$$
\Big \| Y \widehat { V } - \widehat { U } \widehat { \Sigma } \Big \| _ { \mathrm { o p } } = \Big \| ( I - P ) Y Y ^ { T } \widehat { U } \widehat { \Sigma } ^ { - 1 } \Big \| _ { \mathrm { o p } } \leq \frac { \tau } { 1 2 8 } \sqrt { \frac { \mu r } { n } } \leq \frac { \tau } { 6 4 } \sqrt { \frac { \mu r } { n } } .
$$

If no singular value is retained, this residual estimate is unnecessary.

Step 4: the unreconstructed part. The $s = 1 / 2$ estimate in (88) controls the part of Y in $\mathrm { r a n g e } ( U _ { 1 } )$ , since $\sigma _ { j } ( { \pmb Y } ) \le \sqrt { d _ { j } }$ . On its orthogonal complement the singular values are smaller than $\tau / ( 8 \sqrt { 2 } )$ . Thus

$$
\| ( I - P ) \pmb { Y } \| _ { \mathrm { o p } } \leq \frac { \tau } { 5 1 2 } \sqrt { \frac { \mu r } { n } } + \frac { \tau } { 8 \sqrt { 2 } } < \frac { 3 \tau } { 3 2 } .
$$

The singular values discarded from P Y are smaller than $\tau / 8$ . Therefore,

$$
\left\| Y - Z ^ { + } \right\| _ { \mathrm { o p } } \leq \left\| ( I - P ) Y \right\| _ { \mathrm { o p } } + \left\| P Y - Q \mathcal { H } _ { \geq \tau / 8 } ( Q ^ { T } Y ) \right\| _ { \mathrm { o p } } \leq \frac { 3 \tau } { 3 2 } + \frac { \tau } { 8 } = \frac { 7 \tau } { 3 2 } .
$$

This includes the case ${ \cal Z } ^ { + } \ = \ { \bf 0 }$ . Thus all approximation conclusions hold on (86). Since $Y Y ^ { T } + \tau ^ { 2 } I / 4 0 9 6 \succ \mathbf { 0 }$ and G has full column rank almost surely, every thin QR step is well defined. A fixed orthonormal completion defines the procedure on the remaining null event without changing its operation count. 口

## E Leave-one-out estimates for the initial RGN iterates

We prove the sampling and correction estimates used to control the RGN iterates for $0 \leq k \leq \bar { K }$ The concentration and geometric tools are given in Sections B and C. The passage to the quadratic Frobenius recurrence is proved in Subsection 7.4.

## E.1 Sampling estimates for the completed operators

We use the completed sampling operators ${ \mathcal { R } } ^ { \alpha }$ and the index set A defined in Subsection 7.1. For $\alpha \in { \mathfrak { A } }$ , let ${ \mathcal { A } } ^ { \alpha }$ be the entrywise nonnegative square root of ${ \mathcal { R } } ^ { \alpha }$ , so $( \mathcal { A } ^ { \alpha } ) ^ { * } \mathcal { A } ^ { \alpha } = \mathcal { R } ^ { \alpha }$

We first establish the simultaneous sampling isometries for the actual and completed operators.

## E.1.1 Proof of Lemma 7.1

Proof. Let $\delta _ { i j } \ : = \ \mathbf { 1 } _ { \{ ( i , j ) \in \widehat \Omega \} }$ and ${ \mathbf { a } } _ { i j } ~ = ~ \mathcal { P } _ { T _ { \mathbf { X _ { \star } } } } ( e _ { i } e _ { j } ^ { T } )$ , where $( { \pmb a } \otimes { \pmb a } ) ( { \pmb Z } ) : = \langle { \pmb a } , { \pmb Z } \rangle _ { \mathrm { F } } { \pmb a }$ . By Assumption 2.1, $\| \pmb { a } _ { i j } \| _ { \mathrm { F } } ^ { 2 } \le 2 \mu r / n$ . Since the matrices $e _ { i } e _ { j } ^ { T }$ form an orthonormal basis and $\mathcal { P } _ { T _ { X } }$ is an orthogonal projector, we have

$$
\begin{array} { c } { { \displaystyle \sum _ { i , j } { \pmb a } _ { i j } \otimes { \pmb a } _ { i j } = \mathbb { Z } _ { T _ { \pmb X _ { \star } } } , } } \\ { { \displaystyle ( { \pmb a } _ { i j } \otimes { \pmb a } _ { i j } ) ^ { 2 } = \| { \pmb a } _ { i j } \| _ { \mathrm F } ^ { 2 } ( { \pmb a } _ { i j } \otimes { \pmb a } _ { i j } ) , } } \\ { { \displaystyle \frac { 1 } { q } \sum _ { i , j } \| { \pmb a } _ { i j } \| _ { \mathrm F } ^ { 2 } ( { \pmb a } _ { i j } \otimes { \pmb a } _ { i j } ) \preceq \displaystyle \frac { 2 \mu r } { n q } \mathbb { Z } _ { T _ { \pmb X _ { \star } } } . } } \end{array}
$$

For the actual operator, $\mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } \mathcal { R } ^ { 0 } \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } - \mathcal { P } _ { T _ { \mathbf { X } } }$ is the centered sum of $( \delta _ { i j } / q - 1 ) ( { \pmb a } _ { i j } \otimes { \pmb a } _ { i j } )$ over all coordinates. For $\mathcal { R } ^ { \mathrm { r } , i }$ , the centered sum is restricted to coordinates $( a , b )$ with $a \neq i ;$ for $\mathcal { R } ^ { \mathrm { c } , j }$ it is restricted to coordinates with $b \neq j$

For the actual centered sum, each summand and the variance operator satisfy

$$
\begin{array} { r l } & { \quad \quad \quad \| ( \delta _ { i j } / q - 1 ) ( { a _ { i j } } \otimes { a _ { i j } } ) \| _ { \mathrm { F }  \mathrm { F } } \leq \displaystyle \frac { 2 \mu r } { n q } , } \\ & { \quad \quad \quad \sum _ { i , j } \mathbb { E } [ ( \delta _ { i j } / q - 1 ) ^ { 2 } ( { a _ { i j } } \otimes { a _ { i j } } ) ^ { 2 } ] = \displaystyle \frac { 1 - q } { q } \sum _ { i , j } \| { a _ { i j } } \| _ { \mathrm { F } } ^ { 2 } ( { a _ { i j } } \otimes { a _ { i j } } ) \preceq \frac { 2 \mu r } { n q } \mathcal { T } _ { T _ { X _ { k } } } . } \end{array}
$$

Restricting the sums to a completed row or column preserves these two inequalities. Since $q \geq c _ { 1 } \mu r$ log $n / n$ and $c _ { 1 } = 2 ^ { 1 5 }$ , the Bernstein exponent at threshold $1 / 1 6$ is at least 24 log n. Since dim $( T _ { X _ { \star } } ) \leq$ 2nr and $| \mathfrak { A } | = 2 n + 1$ , Lemma B.1 followed by a union bound gives

$$
\operatorname* { P r } \{ \operatorname* { m a x } _ { \alpha \in \mathfrak { A } } \| \mathcal { P } _ { T _ { X _ { \star } } } \mathcal { R } ^ { \alpha } \mathcal { P } _ { T _ { X _ { \star } } } - \mathcal { P } _ { T _ { X _ { \star } } } \| _ { \mathrm { F  F } } > \frac { 1 } { 1 6 } \} \le 4 n r ( 2 n + 1 ) n ^ { - 2 4 } \le \frac { n ^ { - 1 0 } } { 2 4 } .
$$

□

We next record the weighted product estimate used to transfer these sampling bounds to nearby tangent spaces.

## E.1.2 A weighted product estimate

Lemma E.1. Suppose that the conclusion of Lemma 7.2 holds with $( \Lambda , p _ { \Lambda } ) = ( \widehat { \Omega } , q )$ . Then, for every $\alpha \in { \mathfrak { A } }$ band all matrices F, H with n rows and the same number of columns,

$$
\begin{array} { r } { \left\| \boldsymbol { \mathcal { A } } ^ { \alpha } ( \boldsymbol { F } \boldsymbol { H } ^ { T } ) \right\| _ { \mathrm { F } } \leq \sqrt { 3 n } \operatorname* { m i n } \left\{ \| \boldsymbol { F } \| _ { \mathrm { F } } \| \boldsymbol { H } \| _ { 2 , \infty } , \| \boldsymbol { F } \| _ { 2 , \infty } \| \boldsymbol { H } \| _ { \mathrm { F } } \right\} . } \end{array}
$$

Proof. For the actual operator, weighted row and column sums are the degrees divided by $q ,$ hence at most $2 n$ . Completing one row or column adds at most one to every opposite weighted degree and replaces the completed weighted degree by $n _ { : }$ , so all weighted sums are at most 3n.

If $\pmb { f } _ { i } ^ { T }$ and $h _ { j } ^ { T }$ are the rows of $\pmb { F }$ and $H _ { \mathrm { { i } } }$ and $w _ { i j } ^ { \alpha }$ are the weights of ${ \mathcal { R } } ^ { \alpha }$ , then

$$
\begin{array} { r l } {  { \big \| \boldsymbol { \mathcal { A } } ^ { \alpha } ( \boldsymbol { F } \boldsymbol { H } ^ { T } ) \big \| _ { \mathrm { F } } ^ { 2 } = \displaystyle \sum _ { i , j } w _ { i j } ^ { \alpha } ( f _ { i } ^ { T } h _ { j } ) ^ { 2 } } \qquad } & { } \\ & { \leq \displaystyle \sum _ { i } \| \boldsymbol { f } _ { i } \| _ { 2 } ^ { 2 } \displaystyle \sum _ { j } w _ { i j } ^ { \alpha } \| h _ { j } \| _ { 2 } ^ { 2 } } \\ & { \leq 3 n \| \boldsymbol { F } \| _ { \mathrm { F } } ^ { 2 } \| \boldsymbol { H } \| _ { 2 , \infty } ^ { 2 } . } \end{array}
$$

Interchanging rows and columns gives

$$
\begin{array} { r } { \left\| \boldsymbol { \mathcal { A } } ^ { \alpha } ( \boldsymbol { F } \pmb { H } ^ { T } ) \right\| _ { \mathrm { F } } ^ { 2 } \leq 3 n \left\| \boldsymbol { F } \right\| _ { 2 , \infty } ^ { 2 } \left\| \boldsymbol { H } \right\| _ { \mathrm { F } } ^ { 2 } . } \end{array}
$$

Taking square roots in the two preceding inequalities and minimizing the resulting upper bounds proves the lemma. □

## E.2 Proof of Lemma 7.3

We first prove the conditional row estimate; the corresponding column estimate follows by transposition in the subsequent leave-one-out argument. The Proof of Lemma 7.3 is after the following lemma.

Lemma E.2. Let $\delta _ { 1 } , \ldots , \delta _ { n }$ be independent Bernoulli(p<sub>Λ</sub>) variables, and let $z \in \mathbb { R } ^ { n }$ and $V \in$ $\mathbb { R } ^ { n \times r }$ be deterministic. Set

$$
{ \pmb w } = ( w _ { j } ) _ { j = 1 } ^ { n } , \qquad w _ { j } = \left( \frac { \delta _ { j } } { p _ { \Lambda } } - 1 \right) z _ { j } .
$$

$I f { \bf V } ^ { T } { \bf V } = I$ and $\| V \| _ { 2 , \infty } \leq 2 \sqrt { \mu r / n }$ , then, with probability at least $1 - ( n + r + 2 ) n ^ { - 2 4 }$

$$
2 \sqrt { \frac { \mu r } { n } } \left\| w \right\| _ { 2 } + \left\| V ^ { T } w \right\| _ { 2 } \leq 8 \sqrt { \frac { 3 \log n } { p _ { \Lambda } } } \left( 4 \sqrt { \frac { \mu r } { n } } \left\| z \right\| _ { 2 } + \left\| z \right\| _ { \infty } \right) + \frac { 6 4 \sqrt { \mu r / n } \log n } { p _ { \Lambda } } \left\| z \right\| _ { \infty } .
$$

Proof. For $Z _ { j } : = ( p _ { \Lambda } ^ { - 1 } \delta _ { j } - 1 ) z _ { j } e _ { j }$ , we have $\begin{array} { r } { \| Z _ { j } \| _ { 2 } \le \| z \| _ { \infty } / p _ { \Lambda } , \left\| \sum _ { j } \mathbb { E } ( Z _ { j } Z _ { j } ^ { T } ) \right\| _ { \mathrm { o p } } \le \| z \| _ { \infty } ^ { 2 } / p _ { \Lambda } } \end{array}$ and $\begin{array} { r } { \sum _ { j } \mathbb { E } \| Z _ { j } \| _ { 2 } ^ { 2 } \leq \| z \| _ { 2 } ^ { 2 } / p _ { \Lambda } } \end{array}$ . Thus the variance parameter in Lemma B.1 satisfies $v \leq \| z \| _ { 2 } ^ { 2 } / p _ { \Lambda }$ 2 and applying the lemma with t = 24 log n gives

$$
L \leq \frac { \| z \| _ { \infty } } { p _ { \Lambda } } , \qquad v \leq \frac { \| z \| _ { 2 } ^ { 2 } } { p _ { \Lambda } } , \qquad \| w \| _ { 2 } \leq 4 \sqrt { \frac { 3 \log n } { p _ { \Lambda } } } \| z \| _ { 2 } + \frac { 1 6 \log n } { p _ { \Lambda } } \left\| z \right\| _ { \infty } .
$$

For $\widetilde { Z } _ { j } : = ( p _ { \Lambda } ^ { - 1 } \delta _ { j } - 1 ) z _ { j } V ^ { T } e _ { j }$ , the incoherence bound $\| V \| _ { 2 , \infty } ~ \leq ~ 2 \sqrt { \mu r / n }$ gives $\begin{array} { r } { \left\| \widetilde { Z } _ { j } \right\| _ { 2 } \leq } \end{array}$ $2 \sqrt { \mu r / n } \left. z \right. _ { \infty } / p _ { \Lambda }$ . Moreover,

$$
\left\| \sum _ { j } \mathbb { E } ( \widetilde { Z } _ { j } \widetilde { Z } _ { j } ^ { T } ) \right\| _ { \mathrm { o p } } \leq \| z \| _ { \infty } ^ { 2 } / p _ { \Lambda } \qquad \mathrm { a n d } \qquad \sum _ { j } \mathbb { E } \left\| \widetilde { Z } _ { j } \right\| _ { 2 } ^ { 2 } \leq 4 \mu r \| z \| _ { 2 } ^ { 2 } / ( n p _ { \Lambda } ) .
$$

Thus we may take $v \leq \left\| z \right\| _ { \infty } ^ { 2 } / p _ { \Lambda } + 4 \mu r \left\| z \right\| _ { 2 } ^ { 2 } / ( n p _ { \Lambda } )$ in Lemma B.1. Applying the lemma with $t = 2 4 \log n$ gives

$$
\big \| { \cal V } ^ { T } { \pmb w } \big \| _ { 2 } \leq 4 \sqrt { \frac { 3 \log n } { p _ { \Lambda } } } \left( 2 \sqrt { \frac { \mu r } { n } } \left\| z \right\| _ { 2 } + \left\| z \right\| _ { \infty } \right) + \frac { 3 2 \sqrt { \mu r / n } \log n } { p _ { \Lambda } } \left\| z \right\| _ { \infty } .
$$

The Bernstein bounds for $\lVert \pmb { w } \rVert _ { 2 }$ and $\lVert \boldsymbol { V } ^ { T } \boldsymbol { w } \rVert _ { 2 }$ fail with probability at most $( n + 1 ) n ^ { - 2 4 }$ and $( r + 1 ) n ^ { - 2 4 }$ , respectively. Therefore, by a union bound, both estimates hold with probability at least $1 - ( n + r + 2 ) n ^ { - 2 4 }$ . On the intersection of the two Bernstein events, adding the two estimates and collecting the $\left. \boldsymbol { z } \right. _ { 2 }$ and $\| z \| _ { \infty }$ terms gives

$$
\begin{array} { r l r } {  { 2 \sqrt { \frac { \mu r } { n } } \| { \pmb w } \| _ { 2 } + \| { \pmb V } ^ { T } { \pmb w } \| _ { 2 } \le 1 6 \sqrt { \frac { 3 \mu r \log n } { n p _ { \Lambda } } } \| { \pmb z } \| _ { 2 } + 4 \sqrt { \frac { 3 \log n } { p _ { \Lambda } } } \| { \pmb z } \| _ { \infty } + \frac { 6 4 \sqrt { \mu r / n } \log n } { p _ { \Lambda } } \| { \pmb z } \| _ { \infty } } } \\ & { } & { \le 8 \sqrt { \frac { 3 \log n } { p _ { \Lambda } } } ( 4 \sqrt { \frac { \mu r } { n } } \| { \pmb z } \| _ { 2 } + \| { \pmb z } \| _ { \infty } ) + \frac { 6 4 \sqrt { \mu r / n } \log n } { p _ { \Lambda } } \| { \pmb z } \| _ { \infty } , } \end{array}
$$

which is the claimed inequality.

We use the stopped auxiliary sequences and the quantities $\rho _ { k } , \ d _ { k } ^ { \mathrm { l o o } }$ , and $h _ { k } ^ { \mathrm { c o r r } }$ defined in Subsection 7.1. With the fixed compact-SVD convention in Lemma 7.3, write

$$
\pmb { X } _ { k } ^ { \alpha } = \pmb { U } _ { k } ^ { \alpha } \pmb { S } _ { k } ^ { \alpha } ( \pmb { V } _ { k } ^ { \alpha } ) ^ { T } ,
$$

and set $N _ { k } ^ { \alpha } : = N _ { X _ { k } ^ { \alpha } }$ as in (24). For each comparison, the left and right singular bases are aligned as in Section C.Now we give the proof of Lemma 7.3.

Proof of Lemma 7.3. Fix a row i and condition on $X _ { 0 }$ and the Bernoulli variables outside row i. The stopped row-i trajectory, its residual row, and its right singular factors are then fixed, whereas the row-i Bernoulli variables remain independent.

Before stopping, the initial error bound and the accepted updates give $\left\| { \mathbf { X } } _ { k } ^ { \mathrm { { r } } , i } - { \mathbf { X } } _ { \star } \right\| _ { \sharp } \leq \rho _ { k }$ Therefore, the proof of Lemma 5.1 gives, for every such $k < \bar { K }$

$$
\begin{array} { r } { \left\| { \bf X } _ { k } ^ { \mathrm { r } , i } - { \bf X } _ { \star } \right\| _ { \sharp } \leq \rho _ { k } \leq \rho _ { 0 } \leq \displaystyle \frac { \sigma _ { r } ( { \bf X } _ { \star } ) } { 8 } . } \\ { \left\| { \bf V } _ { k } ^ { \mathrm { r } , i } \right\| _ { 2 , \infty } \leq \displaystyle \frac { 1 0 } { 7 } \sqrt { \displaystyle \frac { \mu r } { n } } < 2 \sqrt { \displaystyle \frac { \mu r } { n } } . } \end{array}
$$

After stopping, the sequence is equal to $X _ { \star }$ , and Assumption 2.1 gives $\| V _ { \star } \| _ { 2 , \infty } \leq \sqrt { \mu r / n } <$ $2 \sqrt { \mu r / n }$

Thus Lemma E.2 applies at every $k < \bar { K }$ . Applying Lemma E.2 to the transposed column-$j$ process, conditional on the variables outside column $j ,$ gives the column inequalities in Lemma 7.3. Taking expectations over the conditioned variables and a union bound over at most $2 n \bar { K }$ pairs gives failure probability at most $n ^ { - 1 0 } / 4 8$ for $\bar { K } \leq n$

The row and column bounds in Lemma 7.3 are obtained before alignment and are invariant under the subsequent left and right orthogonal Procrustes transformations. Consequently, conditional on every admissible realization of $X _ { 0 }$ , all row and column inequalities in the lemma hold simultaneously with failure probability at most $n ^ { - 1 0 } / 4 8$ □

## E.3 Local solvability and correction size

We next use the simultaneous sampling events to prove local invertibility and bound the tangent correction.

Proof of Lemma $\it 7 . 4 .$ Since $\mu r \geq 1$ and $e _ { \sharp } \le \sigma _ { r } ( X _ { \star } ) / ( 1 0 0 0 \mu r )$ , Lemmas 5.1 and E.1 and (4) give

$$
\begin{array} { r l r } {  { \big \| { U U ^ { T } - U _ { \star } U _ { \star } ^ { T } } \big \| _ { 2 , \infty } + \big \| { V V ^ { T } - V _ { \star } V _ { \star } ^ { T } } \big \| _ { 2 , \infty } \le \frac { 6 4 } { 7 } \sqrt { \frac { \mu r } { n } } \frac { e _ { \sharp } } { \sigma _ { r } ( X _ { \star } ) } , } } \\ & { } & { \big \| { A ^ { \alpha } ( \mathcal { P } _ { T _ { X } } - \mathcal { P } _ { T _ { X _ { \star } } } ) } \big \| _ { \mathrm { F  F } } \le \frac { 6 4 \sqrt { 3 } } { 7 } \sqrt { \mu r } \frac { e _ { \sharp } } { \sigma _ { r } ( X _ { \star } ) } , ~ } \\ & { } & { \big \| \mathcal { P } _ { T _ { X } } - \mathcal { P } _ { T _ { X _ { \star } } } \big \| _ { \mathrm { F  F } } \le \frac { 1 6 } { 7 } \frac { e _ { \sharp } } { \sigma _ { r } ( X _ { \star } ) } . ~ } \end{array}\tag{90}
$$

Fix $\zeta \in T _ { X } \mathcal { M } _ { r }$ . By Lemma 7.1, the triangle inequality, and (90), we have

$$
\begin{array} { r l } & { \| \mathcal { A } ^ { \alpha } \zeta \| _ { \mathrm { F } } \geq \left\| \mathcal { A } ^ { \alpha } \mathcal { P } _ { T _ { X _ { \star } } } \zeta \right\| _ { \mathrm { F } } - \left\| \mathcal { A } ^ { \alpha } ( \mathcal { P } _ { T _ { X } } - \mathcal { P } _ { T _ { X _ { \star } } } ) \zeta \right\| _ { \mathrm { F } } } \\ & { \qquad \geq \sqrt { \frac { 1 5 } { 1 6 } } \left\| \mathcal { P } _ { T _ { X _ { \star } } } \zeta \right\| _ { \mathrm { F } } - \frac { 6 4 \sqrt { 3 } } { 7 } \sqrt { \mu r } \frac { e _ { \sharp } } { \sigma _ { r } ( X _ { \star } ) } \left\| \zeta \right\| _ { \mathrm { F } } } \\ & { \qquad \geq \sqrt { \frac { 9 } { 1 0 } } \left\| \zeta \right\| _ { \mathrm { F } } . } \end{array}
$$

Using the upper inequality in Lemma 7.1, the triangle inequality, and (90), we also have

$$
\begin{array} { r l } & { \| \mathcal { A } ^ { \alpha } \zeta \| _ { \mathrm { F } } \leq \left\| \mathcal { A } ^ { \alpha } \mathcal { P } _ { T _ { X _ { \star } } } \zeta \right\| _ { \mathrm { F } } + \left\| \mathcal { A } ^ { \alpha } ( \mathcal { P } _ { T _ { X } } - \mathcal { P } _ { T _ { X _ { \star } } } ) \zeta \right\| _ { \mathrm { F } } } \\ & { \qquad \leq \sqrt { \frac { 1 7 } { 1 6 } } \left\| \mathcal { P } _ { T _ { X _ { \star } } } \zeta \right\| _ { \mathrm { F } } + \frac { 6 4 \sqrt { 3 } } { 7 } \sqrt { \mu r } \frac { e _ { \sharp } } { \sigma _ { r } ( X _ { \star } ) } \left\| \zeta \right\| _ { \mathrm { F } } } \\ & { \qquad \leq \sqrt { \frac { 1 1 } { 1 0 } } \left\| \zeta \right\| _ { \mathrm { F } } . } \end{array}
$$

Here we used

$$
\begin{array} { r } { \left\| \mathcal { P } _ { T _ { X _ { \star } } } \zeta \right\| _ { \mathrm { F } } \geq \| \zeta \| _ { \mathrm { F } } - \left\| ( \mathcal { P } _ { T _ { X } } - \mathcal { P } _ { T _ { X _ { \star } } } ) \zeta \right\| _ { \mathrm { F } } } \end{array}
$$

in the lower bound and $\left\| \mathcal { P } _ { T _ { \mathbf { X } _ { \star } } } \zeta \right\| _ { \mathrm { F } } \leq \| \zeta \| _ { \mathrm { F } }$ in the upper bound. Squaring the inequalities $\| \mathcal { A } ^ { \alpha } \zeta \| _ { \mathrm { F } } \geq \sqrt { 9 / 1 0 } \| \zeta \| _ { \mathrm { F } }$ and $\left\| \mathcal { A } ^ { \alpha } \zeta \right\| _ { \mathrm { F } } \leq \sqrt { 1 1 / 1 0 } \left\| \zeta \right\| _ { \mathrm { F } }$ , and using $( \mathcal { A } ^ { \alpha } ) ^ { * } \mathcal { A } ^ { \alpha } = \mathcal { R } ^ { \alpha }$ , proves (49).

We next prove the correction estimate. The definitions of $t _ { X } , \xi _ { X } ^ { \alpha }$ , and $\eta _ { X } ^ { \alpha }$ give

$$
\mathcal { P } _ { T _ { \mathbf { X } } } \mathcal { R } ^ { \alpha } \mathcal { P } _ { T _ { \mathbf { X } } } \eta _ { X } ^ { \alpha } = \mathcal { P } _ { T _ { \mathbf { X } } } \mathcal { R } ^ { \alpha } N _ { X } .
$$

Write $\pmb { X } = \pmb { U } \pmb { \Sigma } \pmb { V } ^ { T }$ as a compact singular value decomposition. By (49), Lemmas E.1 and 5.2, and Weyl’s inequality, we obtain

$$
\sigma _ { \operatorname* { m i n } } ( G _ { X } ) \geq \sigma _ { r } ( X _ { \star } ) - 2 e _ { \sharp } \geq \frac { 9 } { 1 0 } \sigma _ { r } ( X _ { \star } ) ,
$$

$$
\begin{array} { r l } & { \displaystyle \| \eta _ { X } ^ { \alpha } \| _ { \mathrm { F } } \leq \frac { 1 0 } { 9 } \sqrt { \frac { 1 1 } { 1 0 } } \| A ^ { \alpha } N _ { X } \| _ { \mathrm { F } } } \\ & { \qquad \leq \displaystyle \frac { 1 0 } { 9 } \sqrt { \frac { 1 1 } { 1 0 } } \sqrt { 3 n } \left\| L _ { X } G _ { X } ^ { - 1 } \right\| _ { \mathrm { F } } \| R _ { X } \| _ { 2 , \infty } } \\ & { \qquad \leq 9 \mu r \frac { e _ { \sharp } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } . } \end{array}
$$

Therefore, (50) follows.

## E.4 Comparison of the tangent corrections

We first bound the variation in the sampled normal component between two nearby matrices.

Lemma E.3. Let $\alpha \in { \mathfrak { A } }$ and X, $\pmb { Y } \in \mathcal { M } _ { r }$ , and use $d _ { X , Y }$ from Lemma $C . 1 .$ Suppose that Assumption 2.1 holds and

$$
\operatorname* { m a x } \left\{ \| X - X _ { \star } \| _ { \sharp } , \| Y - X _ { \star } \| _ { \sharp } \right\} \leq \rho \leq \frac { \sigma _ { r } ( X _ { \star } ) } { 1 0 0 0 \mu r } .
$$

If $d _ { X , Y } \leq 2 { \sqrt { \mu r / n } } \rho$ , then, on the joint events in Lemmas 7.1 and 7.2, it holds

$$
\| \mathcal { P } _ { T _ { \mathbf { X } } } \mathcal { R } ^ { \alpha } ( N _ { X } - N _ { Y } ) \| _ { \mathrm { F } } \leq 3 2 \sqrt { \frac { \mu r } { n } } \mu r \frac { \rho ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .
$$

Proof. By the graph-factor representation in Lemma 5.2, we have

$$
\begin{array} { r l } & { N _ { X } - N _ { Y } = ( L _ { X } - L _ { Y } ) G _ { X } ^ { - 1 } R _ { X } ^ { T } + L _ { Y } ( G _ { X } ^ { - 1 } - G _ { Y } ^ { - 1 } ) R _ { X } ^ { T } } \\ & { ~ + L _ { Y } G _ { Y } ^ { - 1 } ( R _ { X } - R _ { Y } ) ^ { T } . } \end{array}\tag{91}
$$

Applying Lemmas C.1, E.1 and 5.2 termwise in (91), and using $\rho \le \sigma _ { r } ( X _ { \star } ) / ( 1 0 0 0 \mu r )$ , gives

$$
\Vert \mathcal { A } ^ { \alpha } ( N _ { X } - N _ { Y } ) \Vert _ { \mathrm { F } } \leq 1 5 \sqrt { \mu r } \frac { \rho d _ { X , Y } } { \sigma _ { r } ( X _ { \star } ) } .
$$

Since $d _ { X , Y } \leq 2 \sqrt { \mu r / n } \rho .$ , it follows that

$$
\Vert { \mathcal { A } } ^ { \alpha } ( N _ { X } - N _ { Y } ) \Vert _ { \mathrm { F } } \leq 3 0 \mu r \sqrt { \frac { \mu r } { n } } \frac { \rho ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .
$$

Finally, (49) and $( \mathcal { A } ^ { \alpha } ) ^ { * } \mathcal { A } ^ { \alpha } = \mathcal { R } ^ { \alpha }$ give

$$
\| \mathcal { P } _ { T _ { X } } \mathcal { R } ^ { \alpha } ( N _ { X } - N _ { Y } ) \| _ { \mathrm { F } } \leq \sqrt { \frac { 1 1 } { 1 0 } } \| \mathcal { A } ^ { \alpha } ( N _ { X } - N _ { Y } ) \| _ { \mathrm { F } } \leq 3 2 \sqrt { \frac { \mu r } { n } } \mu r \frac { \rho ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .
$$

This proves the claim.

We now estimate the tangent transport, which bound the error incurred by projecting a tangent correction onto a nearby tangent space.

Lemma E.4. Let $\alpha \in { \mathfrak { A } } , h \geq 0$ , X, $\pmb { Y } \in \mathcal { M } _ { r }$ , and $Z \in T _ { Y } { \mathcal { M } } _ { r }$ . Suppose that X and Y satisfy the hypotheses of Lemma E.3, that $d _ { X , Y } \leq 2 \sqrt { \mu r / n } \rho ,$ and that

$$
\left\| Z \right\| _ { \mathrm { F } } \leq 9 \mu r \frac { \rho ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } , \qquad \operatorname* { m a x } \left\{ \left\| Z \right\| _ { 2 , \infty } , \left\| Z ^ { T } \right\| _ { 2 , \infty } \right\} \leq 2 h + 3 8 \mu r \sqrt { \frac { \mu r } { n } } \frac { \rho ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .
$$

Then, on the joint events in Lemmas 7.1 and 7.2,

$$
\| \mathcal { P } _ { T _ { X } } \mathcal { R } ^ { \alpha } ( \mathcal { P } _ { T _ { Y } } - \mathcal { P } _ { T _ { X } } ) \pmb { Z } \| _ { \mathrm { F } } \leq \frac { 1 } { 2 } \mu r \sqrt { \frac { \mu r } { n } } \frac { \rho ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } + \frac { 1 } { 4 0 } h .
$$

Proof. Write $Y = U _ { Y } S _ { Y } V _ { Y } ^ { T }$ and use the tangent decomposition

$$
Z = U _ { Y } M V _ { Y } ^ { T } + B V _ { Y } ^ { T } + U _ { Y } C ^ { T } , \qquad U _ { Y } ^ { T } B = 0 , \quad V _ { Y } ^ { T } C = 0 .
$$

Since the three terms are mutually orthogonal, we have

$$
\operatorname* { m a x } \left\{ \left\| M \right\| _ { \mathrm { F } } , \left\| B \right\| _ { \mathrm { F } } , \left\| C \right\| _ { \mathrm { F } } \right\} \leq \left\| Z \right\| _ { \mathrm { F } } .
$$

Moreover, $B = Z V _ { Y } - U _ { Y } M$ and $\begin{array} { r } { { \cal { C } } = { Z ^ { T } } { U _ { Y } } - { V _ { Y } } { M ^ { T } } } \end{array}$ . Therefore, Lemma 5.1 and the assumed bounds on Z give

$$
\operatorname* { m a x } \left\{ \left\| \pmb { { \cal B } } \right\| _ { 2 , \infty } , \left\| \pmb { { \cal C } } \right\| _ { 2 , \infty } \right\} \leq 2 h + 5 6 \mu r \sqrt { \frac { \mu r } { n } } \frac { \rho ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .
$$

Define

$$
\begin{array} { r } { \bigtriangleup _ { U } : = ( I - U _ { X } U _ { X } ^ { T } ) U _ { Y } , \qquad \bigtriangleup _ { V } : = ( I - V _ { X } V _ { X } ^ { T } ) V _ { Y } . } \end{array}
$$

Since we have

$$
\begin{array} { r } { \Delta _ { U } = ( I - U _ { X } U _ { X } ^ { T } ) ( Y - X ) V _ { Y } S _ { Y } ^ { - 1 } , \qquad \Delta _ { V } = ( I - V _ { X } V _ { X } ^ { T } ) ( Y - X ) ^ { T } U _ { Y } S _ { Y } ^ { - T } , } \end{array}
$$

Weyl’s inequality gives

$$
\operatorname* { m a x } \left\{ \| \Delta _ { U } \| _ { \mathrm { F } } , \| \Delta _ { V } \| _ { \mathrm { F } } \right\} \leq \frac { d _ { X , Y } } { \sigma _ { r } ( X _ { \star } ) - \rho } .\tag{92}
$$

For every i, the identity $\pmb { \Delta } _ { U } = ( \pmb { I } - \pmb { U } _ { \pmb { X } } \pmb { U } _ { \pmb { X } } ^ { T } ) ( \pmb { Y } - \pmb { X } ) V _ { \pmb { Y } } \pmb { S } _ { \pmb { Y } } ^ { - 1 }$ gives

$$
\left. e _ { i } ^ { T } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Sigma } \right. _ { 2 } \leq \frac { \left. e _ { i } ^ { T } ( \mathbf { Y } - \mathbf { \Delta } \mathbf { X } ) \right. _ { 2 } + \left. e _ { i } ^ { T } \mathbf { { U } } _ { \mathbf { \mathcal { X } } } \right. _ { 2 } \left. \mathbf { \mathcal { Y } } - \mathbf { \Xi } \mathbf { \Delta } \mathbf { \tilde { X } } \right. _ { \mathrm { o p } } } { \sigma _ { r } ( \mathbf { Y } ) } \leq 9 \sqrt { \frac { \mu r } { n } } \frac { \rho } { \sigma _ { r } ( \mathbf { \mathcal { X } _ { \star } } ) } ,
$$

where the last inequality follows from the definition of the sharp norm, Lemma 5.1, and $\sigma _ { r } ( { \bf Y } ) \geq$ $\sigma _ { r } ( \pmb { X } _ { \star } ) - \rho _ { }$ . Applying the same calculation to $\begin{array} { r } { \pmb { \Delta } _ { V } = ( \pmb { I } - \pmb { V } _ { \pmb { X } } \pmb { V } _ { \pmb { X } } ^ { T } ) ( \pmb { Y } - \pmb { X } ) ^ { T } \pmb { U } _ { \pmb { Y } } \pmb { S } _ { \pmb { Y } } ^ { - T } } \end{array}$ gives

$$
\operatorname* { m a x } \left\{ \left\| \Delta _ { U } \right\| _ { 2 , \infty } , \left\| \Delta _ { V } \right\| _ { 2 , \infty } \right\} \leq 9 \sqrt { \frac { \mu r } { n } } \frac { \rho } { \sigma _ { r } ( X _ { \star } ) } .\tag{93}
$$

Furthermore, since ${ \pmb U } _ { \pmb V } ^ { T } { \pmb B } = 0$ and $V _ { Y } ^ { T } C = 0$ , we have

$$
\begin{array} { r } { \left\| { ( I - U _ { X } U _ { X } ^ { T } ) B } \right\| _ { 2 , \infty } \leq \left\| { B } \right\| _ { 2 , \infty } + \left\| { U _ { X } } \right\| _ { 2 , \infty } \left\| { ( U _ { X } - U _ { Y } ) ^ { T } B } \right\| _ { \mathrm { F } } , } \end{array}
$$

$$
\left\| { ( I - { V _ { X } } { V _ { X } ^ { T } } ) C } \right\| _ { 2 , \infty } \leq \left\| { C } \right\| _ { 2 , \infty } + \left\| { V _ { X } } \right\| _ { 2 , \infty } \left\| { ( V _ { X } - { V _ { Y } } ) ^ { T } C } \right\| _ { \mathrm { F } } .
$$

Combining these inequalities with Lemma C.1, $d _ { X , Y } \leq 2 \sqrt { \mu r / n } \rho$ , and the Frobenius bound on $z$ , we obtain

$$
\operatorname* { m a x } \Big \{ \big \| ( I - U _ { X } U _ { X } ^ { T } ) B \big \| _ { 2 , \infty } , \big \| ( I - V _ { X } V _ { X } ^ { T } ) C \big \| _ { 2 , \infty } \Big \} \leq 2 h + 5 7 \mu r \sqrt { \frac { \mu r } { n } } \frac { \rho ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .\tag{94}
$$

Since $\pmb { Z } \in T _ { \pmb { Y } } \mathcal { M } _ { r }$ , substituting its tangent decomposition into (4) gives

$$
( \mathcal { P } _ { T _ { Y } } - \mathcal { P } _ { T _ { X } } ) \boldsymbol { Z } = \Delta _ { U } M \boldsymbol { \Delta } _ { V } ^ { T } + ( \boldsymbol { I } - U _ { X } U _ { X } ^ { T } ) \boldsymbol { B } \boldsymbol { \Delta } _ { V } ^ { T } + \boldsymbol { \Delta } _ { U } C ^ { T } ( \boldsymbol { I } - V _ { X } V _ { X } ^ { T } ) .\tag{95}
$$

Applying Lemma E.1 to the first term and using (92)–(93), we have

$$
\left. \mathcal { A } ^ { \alpha } ( \Delta _ { U } M \Delta _ { V } ^ { T } ) \right. _ { \mathrm { F } } \leq 9 \sqrt { 3 \mu r } \frac { \rho } { \sigma _ { r } ( X _ { \star } ) } \left. \pmb { \Delta } _ { U } \right. _ { \mathrm { F } } \left. M \right. _ { \mathrm { F } } .
$$

Applying Lemma E.1 to the second and third terms in (95), and then using (92) and (94), we obtain

$$
\begin{array} { r l } & { \quad \left\| \boldsymbol A ^ { \alpha } ( ( \boldsymbol I - \boldsymbol U _ { X } \boldsymbol U _ { X } ^ { T } ) B \boldsymbol \Delta _ { V } ^ { T } ) \right\| _ { \mathrm F } + \left\| \boldsymbol A ^ { \alpha } ( \boldsymbol \Delta _ { U } \boldsymbol C ^ { T } ( \boldsymbol I - \boldsymbol V _ { X } \boldsymbol V _ { X } ^ { T } ) ) \right\| _ { \mathrm F } } \\ & { \leq 2 \sqrt { 3 n } \frac { d _ { X , Y } } { \sigma _ { r } ( X _ { \star } ) - \rho } \left( 2 h + 5 7 \mu r \sqrt { \frac { \mu r } { n } } \frac { \rho ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } \right) . } \end{array}
$$

Substituting $d _ { \pmb { X } , \pmb { Y } } \leq 2 \sqrt { \mu r / n } \rho , \rho \leq \sigma _ { r } ( \pmb { X } _ { \star } ) / ( 1 0 0 0 \mu r ) , \| \pmb { M } \| _ { \mathrm { F } } \leq \| \pmb { Z } \| _ { \mathrm { F } }$ , and $\mu r \leq n$ into the bounds for the three terms in (95) gives

$$
\Vert \mathcal { A } ( \mathcal { P } _ { T _ { Y } } - \mathcal { P } _ { T _ { X } } ) Z \Vert _ { \mathrm { F } } \leq \frac { 2 } { 5 } \mu r \sqrt { \frac { \mu r } { n } } \frac { \rho ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } + \frac { 1 } { 4 5 } h .\tag{96}
$$

Finally, (49), $( \mathcal { A } ^ { \alpha } ) ^ { * } \mathcal { A } ^ { \alpha } = \mathcal { R } ^ { \alpha }$ , and (96) give

$$
\| \mathcal { P } _ { T _ { X } } \mathcal { R } ^ { \alpha } ( \mathcal { P } _ { T _ { Y } } - \mathcal { P } _ { T _ { X } } ) \pmb { Z } \| _ { \mathrm { F } } \leq \sqrt { \frac { 1 1 } { 1 0 } } \| \mathcal { A } ^ { \alpha } ( \mathcal { P } _ { T _ { Y } } - \mathcal { P } _ { T _ { X } } ) \pmb { Z } \| _ { \mathrm { F } } \leq \frac { 1 } { 2 } \mu r \sqrt { \frac { \mu r } { n } } \frac { \rho ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } + \frac { 1 } { 4 0 } h .
$$

This proves the result.

## E.5 Proof of Lemma 7.5

We now combine the normal-motion and tangent-transport estimates to compare the actual and leave-one-out tangent corrections.

Proof of Lemma 7.5. First estimate: coordinatewise bounds. Fix α and a row i. By the definitions of $d _ { k } ^ { \mathrm { l o o } }$ and $h _ { k } ^ { \mathrm { c o r r } }$ and by Lemma C.1 in the aligned gauges, we have

$$
\left. \eta _ { k } ^ { \alpha } - \eta _ { k } ^ { \mathrm { r } , i } \right. _ { \mathrm { F } } \leq 2 h _ { k } ^ { \mathrm { c o r r } } , \ \left. \mathbf { X } _ { k } ^ { \alpha } - \mathbf { X } _ { k } ^ { \mathrm { r } , i } \right. _ { \mathrm { F } } \leq 2 d _ { k } ^ { \mathrm { l o o } } , \ \left. \mathbf { V } _ { k } ^ { \alpha } - \mathbf { V } _ { k } ^ { \mathrm { r } , i } \right. _ { \mathrm { o p } } \leq 3 \frac { d _ { k } ^ { \mathrm { l o o } } } { \sigma _ { r } \left( X _ { \star } \right) } .
$$

Set $\mathbf { \boldsymbol { X } } = \mathbf { \boldsymbol { X } } _ { k } ^ { \mathrm { r } , i } = \mathbf { \boldsymbol { U } } \mathbf { \boldsymbol { S } } \mathbf { \boldsymbol { V } } ^ { T } , \ \mathbf { \boldsymbol { N } } = \mathbf { \boldsymbol { N } } _ { k } ^ { \mathrm { r } , i }$ , and $\eta = \eta _ { k } ^ { \mathrm { r } , i }$ . Since $\mathcal { R } ^ { \mathrm { r } , i }$ is the identity on row i and $N V = 0$ , the correction equation and, for every $\mathbf { \pmb { a } } \in \mathbb { R } ^ { r }$ , its test against $e _ { i } { \pmb a } ^ { T } { \pmb V } ^ { T }$ give

$$
\begin{array} { r } { \mathcal { P } _ { T _ { X } } \mathcal { R } ^ { \mathsf { r } , i } ( N - \eta ) = 0 , \quad e _ { i } a ^ { T } V ^ { T } = U \big [ ( U ^ { T } e _ { i } ) a ^ { T } \big ] V ^ { T } + \big [ ( I - U U ^ { T } ) e _ { i } a ^ { T } \big ] V ^ { T } \in T _ { X } \mathcal { M } _ { \tau } , } \end{array}
$$

$$
\begin{array} { r } { e _ { i } ^ { T } ( N - \pmb { \eta } ) \pmb { V } = 0 , \quad e _ { i } ^ { T } \pmb { \eta } _ { k } ^ { \mathrm { r } , i } \pmb { V } _ { k } ^ { \mathrm { r } , i } = 0 . } \end{array}
$$

Since $e _ { i } ^ { T } \pmb { \eta } _ { k } ^ { \mathrm { r } , i } \pmb { V } _ { k } ^ { \mathrm { r } , i } = 0$ , the ith row of $\eta _ { k } ^ { \mathrm { r } , i }$ contains only its $U C ^ { T }$ component. Hence Lemma 5.1, (50), $\mu r / n \leq 1$ from Assumption 2.1, and $\rho _ { k } / \sigma _ { r } ( X _ { \star } ) \leq 1 / ( 1 0 0 0 \mu r )$ give

$$
{ \left\| { { e } _ { i } ^ { T } } { { \eta } _ { k } ^ { \mathrm { { r } } , i } } \right\| } _ { 2 } \leq 2 \sqrt { \frac { \mu r } { n } } \left\| { { \eta } _ { k } ^ { \mathrm { { r } } , i } } \right\| _ { \mathrm { { F } } } \leq 1 8 \sqrt { \frac { \mu r } { n } } \mu r \frac { \rho _ { k } ^ { 2 } } { { { \sigma } _ { r } } ( X _ { \star } ) } ,
$$

$$
\begin{array} { r l r } {  { \big \| e _ { i } ^ { T } \eta _ { k } ^ { \alpha } V _ { k } ^ { \alpha } \big \| _ { 2 } \leq 2 h _ { k } ^ { \mathrm { c o r r } } + 5 4 \sqrt { \frac { \mu r } { n } } \mu r \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } \frac { d _ { k } ^ { \mathrm { l o o } } } { \sigma _ { r } ( X _ { \star } ) } } } \\ & { } & { \leq 2 h _ { k } ^ { \mathrm { c o r r } } + \frac { 1 } { 8 } \sqrt { \frac { \mu r } { n } } \mu r \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } . } \end{array}\tag{97}
$$

Replacing $( \mathrm { r } , i , U , V )$ by $( \mathrm { c } , j , V , U )$ and transposing gives the column form of (97). Moreover, comparing each row with its row-completed correction, and each column with its columncompleted correction, gives

$$
\operatorname* { m a x } \left\{ \| \eta _ { k } ^ { \alpha } \| _ { 2 , \infty } , \big \| ( \eta _ { k } ^ { \alpha } ) ^ { T } \big \| _ { 2 , \infty } \right\} \leq 2 h _ { k } ^ { \mathrm { c o r r } } + 1 8 \sqrt { \frac { \mu r } { n } } \mu r \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } , \qquad \alpha \in \mathfrak { A } .\tag{98}
$$

For a tangent matrix η at $\pmb { X } = \pmb { U } \pmb { \Sigma } \pmb { V } ^ { T }$ , we have

$$
\pmb { \eta } = ( \pmb { \eta } \pmb { V } ) \pmb { V } ^ { T } + \pmb { U } ( \pmb { \eta } ^ { T } \pmb { U } ) ^ { T } - \pmb { U } ( \pmb { U } ^ { T } \pmb { \eta } \pmb { V } ) \pmb { V } ^ { T } .
$$

Equation $( 9 7 )$ and its transpose, together with the tangent decomposition, Lemma 5.1, and $\| \pmb { \eta } \| _ { \mathrm { F } } \le 9 \mu r \rho _ { k } ^ { 2 } / \sigma _ { r } ( X _ { \star } )$ , give

$$
\Vert \eta \Vert _ { \mathrm { o p } } \leq 9 \mu r \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } , \qquad \Vert \eta \Vert _ { \sharp } \leq 2 \sqrt { \frac { n } { \mu r } } h _ { k } ^ { \mathrm { c o r r } } + 1 9 \mu r \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .
$$

This proves (51).

Second estimate: comparison of the corrections. Fix a row i, and set

$$
X = X _ { k } ^ { 0 } , \qquad Y = X _ { k } ^ { \mathrm { r } , i } , \qquad \eta = \eta _ { k } ^ { 0 } , \qquad \overline { { { \eta } } } = \eta _ { k } ^ { \mathrm { r } , i } , \qquad Z = N _ { Y } - \overline { { { \eta } } } .
$$

The two correction equations are

$$
\mathcal { P } _ { T _ { \mathbf { X } } } \mathcal { R } ^ { 0 } \mathcal { P } _ { T _ { \mathbf { X } } } \eta = \mathcal { P } _ { T _ { \mathbf { X } } } \mathcal { R } ^ { 0 } N _ { X } ,\tag{99a}
$$

$$
\mathcal { P } _ { T _ { \mathbf { Y } } } \mathcal { R } ^ { \mathrm { r } , i } \mathcal { P } _ { T _ { \mathbf { Y } } } \overline { { \eta } } = \mathcal { P } _ { T _ { \mathbf { Y } } } \mathcal { R } ^ { \mathrm { r } , i } N _ { \mathbf { Y } } .\tag{99b}
$$

Since $\pmb { \eta } \in T _ { \pmb { X } }$ and $\overline { { \eta } } \in T _ { Y } , ( 9 9 \mathrm { b } )$ gives $\mathcal { P } _ { T _ { \mathbf { Y } } } \mathcal { R } ^ { \mathrm { r } , i } Z = 0$ . Subtracting (99b) from (99a) and adding and subtracting $N _ { Y }$ and $\mathcal { R } ^ { \mathrm { r } , i }$ give

$$
\begin{array} { r l } & { ( \mathcal { P } _ { T _ { \mathbf { X } } } \mathcal { R } ^ { 0 } \mathcal { P } _ { T _ { \mathbf { X } } } ) ( \eta - \mathcal { P } _ { T _ { \mathbf { X } } } \overline { { \eta } } ) = \underbrace { \mathcal { P } _ { T _ { \mathbf { X } } } \mathcal { R } ^ { 0 } ( N _ { X } - N _ { Y } ) } _ { T _ { 1 } } + \underbrace { \mathcal { P } _ { T _ { \mathbf { X } } } ( \mathcal { R } ^ { 0 } - \mathcal { R } ^ { \mathrm { r } , i } ) Z } _ { T _ { 2 } } } \\ & { \quad \quad \quad \quad \quad + \underbrace { ( \mathcal { P } _ { T _ { X } } - \mathcal { P } _ { T _ { Y } } ) \mathcal { R } ^ { \mathrm { r } , i } } _ { T _ { 3 } } + \underbrace { \mathcal { P } _ { T _ { X } } \mathcal { R } ^ { 0 } ( \mathcal { P } _ { T _ { Y } } - \mathcal { P } _ { T _ { X } } ) \overline { { \eta } } } _ { T _ { 4 } } . } \end{array}\tag{100}
$$

Replacing $( \mathbf { r } , i )$ by $( \mathrm { c } , j )$ and transposing gives the column form of (100). By Lemma $7 . 4 .$ , the inverse of the left-hand operator has norm at most $1 0 / 9$ . We bound $\pmb { T } _ { 1 } , \ldots , \pmb { T } _ { 4 }$ separately, as in [21, Lemma 11].

For $\pmb { T } _ { 1 }$ , Lemma E.3 gives $\| T _ { 1 } \| _ { \mathrm { F } } \leq 3 2 \sqrt { \frac { \mu r } { n } } \mu r \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .$

For $\mathbf { { T } } _ { 2 } ^ { }$ , write the completed row of $z$ as $z _ { i } ^ { T }$ , set $\delta _ { i j } : = \mathbf { 1 } _ { \{ ( i , j ) \in \widehat { \Omega } \} }$ , and write $w _ { j } = ( \delta _ { i j } / q -$ $1 ) ( z _ { i } ) _ { \it 3 }$ . Conditional on $X _ { 0 }$ and the Bernoulli variables outside row $i , \ z _ { i }$ and $V _ { k } ^ { \mathrm { r } , i }$ are fixed,

whereas $\{ \delta _ { i j } \} _ { j = 1 } ^ { n }$ remain independent Bernoulli(q) variables. Hence the conclusion of Lemma 7.3 gives (47) with $z = z _ { i }$ and $V = V _ { k } ^ { \mathrm { r } , i }$

Write $\mathbf { \boldsymbol { X } } _ { k } ^ { \mathrm { r } , i } = \mathbf { \boldsymbol { U } } \boldsymbol { \Sigma } \mathbf { \boldsymbol { V } } ^ { T } , \mathbf { \boldsymbol { N } } = \mathbf { \boldsymbol { N } } _ { k } ^ { \mathrm { r } , i }$ , and $\overline { { \eta } } = \eta _ { k } ^ { \mathrm { r } , i }$ . Testing the correction equation against $e _ { i } { \pmb { a } } ^ { T } V ^ { T } \in \ddot { T _ { { \pmb { X } } _ { k } ^ { \mathrm { r } , i } } } \mathcal { M } _ { \qquad }$ <sub>r</sub> and using $N V = 0$ give

$$
e _ { i } ^ { T } ( N - \overline { { \eta } } ) V = 0 , e _ { i } ^ { T } \overline { { \eta } } V = 0 .
$$

Set $\begin{array} { r } { C _ { \mathrm { o f f } } : = ( I - V V ^ { T } ) \overline { { \eta } } ^ { T } U } \end{array}$ . Using Lemma 5.2 for $N _ { k } ^ { \mathrm { r } , i }$ , the identity $e _ { i } ^ { T } \overline { { \eta } } V = 0$ and Lemma 7.4 for $e _ { i } ^ { T } \overline { { \eta } } .$ , and the transposed form of (97) for $C _ { \mathrm { o f f } }$ , the tangent decomposition of $\overline { { \eta } }$ gives

$$
\left. e _ { i } ^ { T } N _ { k } ^ { \mathrm { r } , i } \right. _ { 2 } \leq \frac { 1 6 } { 3 } \sqrt { \frac { \mu r } { n } } \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } ,
$$

$$
\left. N _ { k } ^ { \mathrm { r } , i } \right. _ { \infty } \leq \frac { 1 6 } { 3 } \frac { 4 \mu r } { n } \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } ,
$$

$$
\left. e _ { i } ^ { T } \overline { { \eta } } \right. _ { 2 } \leq 2 \sqrt { \frac { \mu r } { n } } \left. \overline { { \eta } } \right. _ { \mathrm { F } } \leq 1 8 \sqrt { \frac { \mu r } { n } } \mu r \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } ,
$$

$$
\begin{array} { r l } & { \left\| e _ { j } ^ { T } C _ { \mathrm { o f f } } \right\| _ { 2 } \leq \left\| e _ { j } ^ { T } \overline { { \eta } } ^ { T } U \right\| _ { 2 } + \left\| e _ { j } ^ { T } V \right\| _ { 2 } \left\| U ^ { T } \overline { { \eta } } V \right\| _ { \mathrm { o p } } } \\ & { \qquad \leq 2 h _ { k } ^ { \mathrm { c o r r } } + \displaystyle \frac { 1 } { 8 } \sqrt { \frac { \mu r } { n } } \mu r \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } + 1 8 \sqrt { \frac { \mu r } { n } } \mu r \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } , } \end{array}
$$

$$
\| z _ { i } \| _ { 2 } \leq 2 4 \mu r \sqrt { \frac { \mu r } { n } } \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } ,\tag{101a}
$$

$$
\| z _ { i } \| _ { \infty } \leq \frac { 6 0 ( \mu r ) ^ { 2 } } { n } \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } + 4 \sqrt { \frac { \mu r } { n } } h _ { k } ^ { \mathrm { c o r r } } .\tag{101b}
$$

Substituting (101a) and (101b) into (47), and using $q \geq c _ { 4 } \mu r \log n / n$ with $c _ { 4 } = 2 ^ { 2 0 }$ , gives

$$
2 \sqrt { \frac { \mu r } { n } } \left\| w \right\| _ { 2 } + \left\| ( V _ { k } ^ { \mathrm { r } , i } ) ^ { T } w \right\| _ { 2 } \leq 1 3 \sqrt { \frac { \mu r } { n } } \mu r \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } + \frac { 1 7 } { 5 0 } h _ { k } ^ { \mathrm { c o r r } } .
$$

Equation (4), Lemma C.1, and $\rho _ { k } \le \sigma _ { r } ( X _ { \star } ) / ( 1 0 0 0 \mu r )$ give

$$
\begin{array} { r } { \left\| \mathcal { P } _ { T _ { Y } } ( e _ { i } w ^ { T } ) \right\| _ { \mathrm { F } } ^ { 2 } = \left\| U _ { Y } ^ { T } e _ { i } \right\| _ { 2 } ^ { 2 } \left\| w \right\| _ { 2 } ^ { 2 } + \left\| ( I - U _ { Y } U _ { Y } ^ { T } ) e _ { i } \right\| _ { 2 } ^ { 2 } \left\| V _ { Y } ^ { T } w \right\| _ { 2 } ^ { 2 } , } \end{array}
$$

$$
3 \frac { d _ { k } ^ { \mathrm { l o o } } } { \sigma _ { r } ( \pmb { X } _ { \star } ) } \left\| \pmb { w } \right\| _ { 2 } \leq \frac { 3 \rho _ { k } } { \sigma _ { r } ( \pmb { X } _ { \star } ) } \left( 2 \sqrt { \frac { \mu r } { n } } \left\| \pmb { w } \right\| _ { 2 } + \left\| \pmb { V } _ { \pmb { Y } } ^ { T } \pmb { w } \right\| _ { 2 } \right) ,
$$

$$
\| { \pmb T } _ { 2 } \| _ { \mathrm { F } } \le 1 4 \mu r \sqrt { \frac { \mu r } { n } } \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( { \pmb X } _ { \star } ) } + \frac { 7 } { 2 0 } h _ { k } ^ { \mathrm { c o r r } } .
$$

For $\mathbf { T } _ { 3 } .$ , we use the row-coordinate estimate (97) and its transpose. Together with the tangent decomposition and the Frobenius bound on $\overline { { \eta } } ,$ , they give

$$
\left\| \overline { { \eta } } \right\| _ { \infty } \leq 8 \sqrt { \frac { \mu r } { n } } h _ { k } ^ { \mathrm { c o r r } } + 3 7 \frac { ( \mu r ) ^ { 2 } } { n } \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .
$$

Consequently, (99b), Lemmas C.1, 5.2 and 7.2, $d _ { k } ^ { \mathrm { l o o } } \le 2 \sqrt { \mu r / n } \rho _ { k }$ , and $\rho _ { k } \le \sigma _ { r } ( X _ { \star } ) / ( 1 0 0 0 \mu r )$

give

$$
\left\| Z \right\| _ { \infty } \leq 8 { \sqrt { \frac { \mu r } { n } } } h _ { k } ^ { \mathrm { c o r r } } + 6 4 { \frac { ( \mu r ) ^ { 2 } } { n } } { \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } } ,
$$

$$
\begin{array} { r } { U _ { Y } ^ { T } \mathcal { R } ^ { \mathrm { r } , i } Z = 0 , \quad \mathcal { R } ^ { \mathrm { r } , i } Z V _ { Y } = 0 , } \end{array}
$$

$$
\begin{array} { r } { \left. \mathcal { R } ^ { \mathrm { r } , i } Z \right. _ { \mathrm { o p } } \leq 3 n \left. Z \right. _ { \infty } , } \end{array}
$$

$$
\left. { ( \mathcal { P } _ { T _ { X } } - \mathcal { P } _ { T _ { Y } } ) \mathcal { R } ^ { \mathrm { r } , i } Z } \right. _ { \mathrm { F } } \leq 3 \frac { d _ { k } ^ { \mathrm { l o o } } } { \sigma _ { r } ( X _ { \star } ) } \left. \mathcal { R } ^ { \mathrm { r } , i } Z \right. _ { \mathrm { o p } } ,
$$

$$
\| T _ { 3 } \| _ { \mathrm { F } } \leq \frac { 6 } { 5 } \sqrt { \frac { \mu r } { n } } \mu r \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } + \frac { 3 } { 2 0 } h _ { k } ^ { \mathrm { c o r r } } .
$$

For $\mathbf { T } _ { 4 } .$ , (50) and (98), followed by Lemma E.4, give

$$
\begin{array} { r l r } & { } & { \| \overline { { \eta } } \| _ { \mathrm { F } } \leq 9 \mu r \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } , } \\ & { } & { \operatorname* { m a x } \{ \| \overline { { \eta } } \| _ { 2 , \infty } , \| \overline { { \eta } } ^ { T } \| _ { 2 , \infty } \} \leq 2 h _ { k } ^ { \mathrm { c o r r } } + 3 8 \mu r \sqrt { \frac { \mu r } { n } } \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } , ~ } \\ & { } & { \| T _ { 4 } \| _ { \mathrm { F } } \leq \frac { 1 } { 2 } \sqrt { \frac { \mu r } { n } } \mu r \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } + \frac { 1 } { 4 0 } h _ { k } ^ { \mathrm { c o r r } } . ~ } \end{array}
$$

To compare the two corrections themselves, we also account for the component of $\overline { { \eta } }$ normal to $T _ { X }$ . By Lemmas C.1 and 7.4 and (100), we have

$$
\left\| \eta - \overline { { \eta } } \right\| _ { \mathrm { F } } \leq \frac { 1 0 } { 9 } \sum _ { j = 1 } ^ { 4 } \left\| \pmb { T } _ { j } \right\| _ { \mathrm { F } } + \frac { 3 d _ { k } ^ { \mathrm { l o o } } } { \sigma _ { r } \left( \pmb { X } _ { \star } \right) } \left\| \overline { { \eta } } \right\| _ { \mathrm { F } } .
$$

Substituting the four estimates for $\| \pmb { T } _ { 1 } \| _ { \mathrm { F } } , \dots , \| \pmb { T } _ { 4 } \| _ { \mathrm { F } }$ derived in this proof, using $d _ { k } ^ { \mathrm { l o o } } \leq 2 \sqrt { \mu r / n } \rho _ { k }$ , and applying their transposed counterparts to each completed column yield

$$
h _ { k } ^ { \mathrm { c o r r } } \leq \left( 5 3 + 5 4 \frac { \rho _ { k } } { \sigma _ { r } ( X _ { \star } ) } \right) \sqrt { \frac { \mu r } { n } } \mu r \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } + \frac { 7 } { 1 2 } h _ { k } ^ { \mathrm { c o r r } } .
$$

Since $\rho _ { k } / \sigma _ { r } ( X _ { \star } ) \leq 1 / ( 1 0 0 0 \mu r )$ and $\mu r \geq 1$ , rearranging gives

$$
h _ { k } ^ { \mathrm { c o r r } } \leq 1 2 8 \sqrt { \frac { \mu r } { n } } \mu r \frac { \rho _ { k } ^ { 2 } } { \sigma _ { r } ( X _ { \star } ) } .
$$

This proves (52). Together with (51), this completes the proof.

## F Matrix-free implementation and computational complexity

We describe the matrix-free realization of the RGD and RGN steps and record the resulting complexity. Let $\pmb { X } = \pmb { U } \pmb { S } \pmb { V } ^ { T } \in \mathcal { M } _ { \tau }$ be a compact singular value decomposition. Every $\xi \in \mathbf { \Xi }$ $T _ { X } \mathcal { M } _ { r }$ can be written as

$$
\pmb { \xi } = \pmb { U } \pmb { M } \pmb { V } ^ { T } + \pmb { B } \pmb { V } ^ { T } + \pmb { U } \pmb { C } ^ { T } , \qquad \pmb { U } ^ { T } \pmb { B } = 0 , \quad \pmb { V } ^ { T } \pmb { C } = 0 ;
$$

see [28, Sec. 2.1]. Define $\Phi _ { X } ( M , B , C ) : = U M V ^ { T } + B V ^ { T } + U C ^ { T }$ . Then $\Phi _ { X }$ is an isometry onto $T _ { X } . M _ { r }$ with

$$
\Phi _ { X } ^ { * } ( Z ) = \big ( U ^ { T } Z V , ( I - U U ^ { T } ) Z V , ( I - V V ^ { T } ) Z ^ { T } U \big ) .
$$

For the RGD direction, orthogonality gives

$$
\left. \pmb { \xi } \right. _ { \mathrm { F } } ^ { 2 } = \left. \pmb { M } \right. _ { \mathrm { F } } ^ { 2 } + \left. \pmb { B } \right. _ { \mathrm { F } } ^ { 2 } + \left. \pmb { C } \right. _ { \mathrm { F } } ^ { 2 } .
$$

Thus the numerator in (6) costs $O ( n r )$ operations. Its denominator is obtained by first forming UM, then evaluating $\Phi _ { X } ( M , B , C )$ on Ω and summing the squared entries, which costs $O ( | \Omega | r + n r ^ { 2 } )$ operations. The graph retraction costs $O ( n r ^ { 2 } + r ^ { 3 } )$ operations. Since $r \leq n$ , the exact line search and the graph retraction therefore preserve the $O ( | \Omega | r + n r ^ { 2 } )$ cost per RGD iteration.

For RGN, writing $z = ( M , B , C )$ , the tangent normal equation (8) is equivalent to

$$
\mathcal { H } _ { X } z = - g _ { X } , \qquad \mathcal { H } _ { X } : = \Phi _ { X } ^ { * } q ^ { - 1 } \mathcal { P } _ { \widehat { \Omega } } \Phi _ { X } , \qquad g _ { X } : = \Phi _ { X } ^ { * } q ^ { - 1 } \mathcal { P } _ { \widehat { \Omega } } ( X - X _ { \star } ) .\tag{102}
$$

Thus no $r ( 2 n - r ) \times r ( 2 n - r )$ normal matrix is formed. Both the formation of $\mathbf { \pmb { g } } \mathbf { \pmb { x } }$ and one application of cost $O ( | \widehat { \Omega } | r + n r ^ { 2 } )$ operations, and the graph retraction costs $O ( n r ^ { 2 } + r ^ { 3 } )$ boperations; see [1]. This gives the standard orders for fixed-rank Riemannian matrix completion; see, for example, [31].

By Lemma 7.4, $\mathcal { H } _ { X }$ is positive definite along the RGN iterates covered by Theorem 4.2. Therefore, exact CG applied to (102) is well defined and terminates in at most dim $\left( T _ { X } \mathcal { M } _ { r } \right) =$ $r ( 2 n - r )$ iterations [11]. In particular, $1 \leq J _ { k } \leq r ( 2 n - r )$ whenever an RGN correction is computed, and the kth RGN iteration costs

$$
O \Big ( J _ { k } ( | \widehat \Omega | r + n r ^ { 2 } ) \Big ) .
$$

## F.1 Proof of Proposition 2.2

Proof. Set

$$
\pmb { Y } = \pmb { Z } + q ^ { - 1 } \pmb { \mathcal { P } } _ { \Lambda } ( \pmb { X } _ { \star } - \pmb { Z } ) = \pmb { U } \pmb { \Sigma } \pmb { V } ^ { T } + \pmb { S } ,
$$

where rank $( Z ) \leq r$ and S has at most m nonzero entries. For $Q \in \mathbb { R } ^ { n \times r }$

$$
\pmb { Y } ^ { T } \pmb { Q } = \pmb { V } \pmb { \Sigma } ( \pmb { U } ^ { T } \pmb { Q } ) + \pmb { S } ^ { T } \pmb { Q } , \qquad \pmb { Y } \pmb { Q } = \pmb { U } \pmb { \Sigma } ( \pmb { V } ^ { T } \pmb { Q } ) + \pmb { S } \pmb { Q } .
$$

Thus the two matrix products in each step of (11), together with the thin QR factorization, require $O ( m r + n r ^ { 2 } )$ operations. Since there are 12 log n such steps, the subspace iteration costs $O \big ( ( m r + n r ^ { 2 } ) \log n \big )$ . For the final spectral reconstruction, compute

$$
\begin{array} { r } { { \boldsymbol { Y } } ^ { T } { \boldsymbol { Q } } = \tilde { \boldsymbol { Q } } { \boldsymbol { R } } , \qquad { \boldsymbol { R } } ^ { T } = { \boldsymbol { U } } _ { R } \boldsymbol { \Sigma } _ { R } { \boldsymbol { V } } _ { R } ^ { T } . } \end{array}
$$

Then

$$
\begin{array} { r } { \mathcal { H } _ { \ge \tau / 8 } ( \pmb { Q } ^ { T } \pmb { Y } ) = \pmb { U } _ { R } \mathcal { H } _ { \ge \tau / 8 } ( \pmb { \Sigma } _ { R } ) ( \pmb { \widetilde { Q } } \pmb { V } _ { R } ) ^ { T } , } \end{array}
$$

and hence

$$
\begin{array} { r } { \mathcal { T } _ { \tau } ( \pmb { Y } ) = \pmb { Q } U _ { R } \mathcal { H } _ { \geq \tau / 8 } ( \pmb { \Sigma } _ { R } ) ( \widetilde { \pmb { Q } } V _ { R } ) ^ { T } . } \end{array}
$$

The thin QR factorization and the $r \times r$ singular value decomposition require $O ( m r + n r ^ { 2 } )$ additional operations. Evaluating the entries of Z on Λ and summing the squared residuals costs $O ( m r )$ operations. This proves the result. □

## F.2 Computational complexity of RGD and RGN

Theorem F.1. Let $0 < \varepsilon \le 1 / 4$ . Under the conditions of Theorem 3.1, RGD attains relative Frobenius accuracy ε with complexity

$$
O \left( \mu n r ^ { 2 } \log n \log ( n \kappa ) \left[ \log n + \log { \frac { 1 } { \varepsilon } } \right] \right) .
$$

Under the conditions of Theorem 3.2, RGN attains relative Frobenius accuracy ε with arithmetic complexity

$$
O \left( \mu n r ^ { 2 } \log n \left[ \log n \log ( 2 \mu r \kappa ) + J \log \log \frac { 1 } { \varepsilon } \right] \right) ,
$$

where $J _ { k } \le J \le r ( 2 n - r )$ for all RGN iterations used to attain the prescribed accuracy.

Proof. Initialization. Let $m _ { \ell } = | \Omega ^ { ( \ell ) } |$ and let $B = K + 1$ for RGD and $B = K + 2$ for RGN. By Proposition 2.2, at most K reconstruction steps require

$$
O \left( \left[ r \sum _ { \ell = 1 } ^ { K } m _ { \ell } + K n r ^ { 2 } \right] \log n \right)
$$

operations. The residual tests cost at most $O ( r \sum _ { \ell = 1 } ^ { K } m _ { \ell } )$ operations and have lower order. Since the observation components are independent Bernoulli(q) sets, $\begin{array} { r } { \sum _ { \ell = 0 } ^ { B - 1 } m _ { \ell } = O ( B q n ^ { 2 } ) } \end{array}$ with high probability. Moreover, $B q = O ( p )$ and $q n \gtrsim \mu r$ log n, so $K n r ^ { 2 } = O ( B q n ^ { 2 } r )$ . Therefore the initialization cost is

$$
O ( p n ^ { 2 } r \log n ) .\tag{103}
$$

RGD. By the matrix-free implementation above, one RGD iteration costs $O ( | \boldsymbol { \Omega } | \boldsymbol { r } + n \boldsymbol { r } ^ { 2 } )$ . At the sampling rate of Theorem 3.1, $| \Omega | = O ( \mu n r \log n \log ( n \kappa ) )$ with high probability. Combining the resulting per-iteration cost, the linear convergence in Theorem 3.1, and (103) gives

$$
O \left( \mu n r ^ { 2 } \log n \log ( n \kappa ) \left[ \log n + \log { \frac { 1 } { \varepsilon } } \right] \right) .
$$

RGN. By the matrix-free implementation above, the kth RGN iteration costs $O \Big ( J _ { k } \big ( | \widehat \Omega | r + n r ^ { 2 } \big ) \Big )$ . At the sampling rate of Theorem 3.2, $| \widehat \Omega | \ = \ O ( \mu n r \log n )$ with high b bprobability. Combining this estimate, the convergence rate in Theorem 3.2, and (103) gives

$$
O \bigg ( \mu n r ^ { 2 } \log n \bigg [ \log n \log ( 2 \mu r \kappa ) + J \log \log \frac { 1 } { \varepsilon } \bigg ] \bigg ) .
$$

□

## References

[1] P.-A. Absil and Ivan V. Oseledets. Low-rank retractions: A survey and new results. Computational Optimization and Applications, 62(1):5–29, 2015.

[2] Jefrey D. Blanchard, Jared Tanner, and Ke Wei. CGIHT: Conjugate gradient iterative hard thresholding for compressed sensing and matrix completion. Information and Inference: A Journal of the IMA, 4(4):289–327, 2015.

[3] Hanqin Cai, Jian-Feng Cai, and Juntao You. Structured gradient descent for fast robust low-rank Hankel matrix completion. SIAM Journal on Scientific Computing, 45(3):A1172– A1198, 2023.

[4] HanQin Cai, Longxiu Huang, Xiliang Lu, and Juntao You. Accelerating ill-conditioned Hankel matrix recovery via structured Newton-like descent. Inverse Problems, 41(7):075015, 2025.

[5] Jian-Feng Cai, Tong Wu, and Ruizhe Xia. Fast non-convex matrix sensing with optimal sample complexity. In Proceedings of the Forty-first Conference on Uncertainty in Artificial Intelligence, pages 497–520, 2025.

[6] Emmanuel J. Cand\`es and Benjamin Recht. Exact matrix completion via convex optimization. Foundations of Computational Mathematics, 9(6):717–772, 2009.

[7] Emmanuel J. Cand\`es and Terence Tao. The power of convex relaxation: Near-optimal matrix completion. IEEE Transactions on Information Theory, 56(5):2053–2080, 2010.

[8] Yudong Chen. Incoherence-optimal matrix completion. IEEE Transactions on Information Theory, 61(5):2909–2923, 2015.

[9] Yudong Chen, Srinadh Bhojanapalli, Sujay Sanghavi, and Rachel Ward. Completing any low-rank matrix, provably. Journal of Machine Learning Research, 16(94):2999–3034, 2015.

[10] Lijun Ding and Yudong Chen. Leave-one-out approach for matrix completion: Primal and dual analysis. IEEE Transactions on Information Theory, 66(11):7274–7301, 2020.

[11] Anne Greenbaum. Iterative Methods for Solving Linear Systems, volume 17 of Frontiers in Applied Mathematics. Society for Industrial and Applied Mathematics, Philadelphia, PA, 1997.

[12] Nathan Halko, Per-Gunnar Martinsson, and Joel A. Tropp. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Review, 53(2):217–288, 2011.

[13] Moritz Hardt. Understanding alternating minimization for matrix completion. In 2014 IEEE 55th Annual Symposium on Foundations of Computer Science, pages 651–660, 2014.

[14] Moritz Hardt and Mary Wootters. Fast matrix completion without the condition number. In Proceedings of the 27th Conference on Learning Theory, pages 638–678, 2014.

[15] Prateek Jain, Raghu Meka, and Inderjit S. Dhillon. Guaranteed rank minimization via singular value projection. In Advances in Neural Information Processing Systems, volume 23, pages 937–945, 2010.

[16] Prateek Jain and Praneeth Netrapalli. Fast exact matrix completion with finite samples. In Proceedings of the 28th Conference on Learning Theory, pages 1007–1034, 2015.

[17] Prateek Jain, Praneeth Netrapalli, and Sujay Sanghavi. Low-rank matrix completion using alternating minimization. In Proceedings of the Forty-Fifth Annual ACM Symposium on Theory of Computing, pages 665–674, 2013.

[18] Raghunandan H. Keshavan, Andrea Montanari, and Sewoong Oh. Matrix completion from a few entries. IEEE Transactions on Information Theory, 56(6):2980–2998, 2010.

[19] Christian K¨ummerle and Claudio M. Verdun. A scalable second order method for illconditioned matrix completion from few samples. In Proceedings of the 38th International Conference on Machine Learning, pages 5872–5883, 2021.

[20] Zhang Liu and Lieven Vandenberghe. Interior-point method for nuclear norm approximation with application to system identification. SIAM Journal on Matrix Analysis and Applications, 31(3):1235–1256, 2009.

[21] Cong Ma, Kaizheng Wang, Yuejie Chi, and Yuxin Chen. Implicit regularization in nonconvex statistical estimation: Gradient descent converges linearly for phase retrieval, matrix completion, and blind deconvolution. Foundations of Computational Mathematics, 20(3):451–632, 2020.

[22] Thanh T. Ngo and Yousef Saad. Scaled gradients on Grassmann manifolds for matrix completion. In Advances in Neural Information Processing Systems, volume 25, pages 1412–1420, 2012.

[23] Ruoyu Sun and Zhi-Quan Luo. Guaranteed matrix completion via non-convex factorization. IEEE Transactions on Information Theory, 62(11):6535–6579, 2016.

[24] Jared Tanner and Ke Wei. Normalized iterative hard thresholding for matrix completion. SIAM Journal on Scientific Computing, 35(5):S104–S125, 2013.

[25] Tian Tong, Cong Ma, and Yuejie Chi. Accelerating ill-conditioned low-rank matrix estimation via scaled gradient descent. Journal of Machine Learning Research, 22(150):1–63, 2021.

[26] Joel A. Tropp. User-friendly tail bounds for sums of random matrices. Foundations of Computational Mathematics, 12(4):389–434, 2012.

[27] Eilon Vaknin Laufer and Boaz Nadler. RGNMR: A Gauss–Newton method for robust matrix completion with theoretical guarantees. In Advances in Neural Information Processing Systems, volume 38, pages 66931–66973, 2025.

[28] Bart Vandereycken. Low-rank matrix completion by Riemannian optimization. SIAM Journal on Optimization, 23(2):1214–1236, 2013.

[29] Lingxiao Wang, Xiao Zhang, and Quanquan Gu. A unified computational and statistical framework for nonconvex low-rank matrix estimation. In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, pages 981–990, 2017.

[30] Tianming Wang and Ke Wei. Leave-one-out analysis for nonconvex robust matrix completion with general thresholding functions. Numerical Algorithms, 2026.

[31] Ke Wei, Jian-Feng Cai, Tony F. Chan, and Shingyu Leung. Guarantees of Riemannian optimization for low rank matrix completion. Inverse Problems and Imaging, 14(2):233–265, 2020.

[32] Xingyu Xu, Yandi Shen, Yuejie Chi, and Cong Ma. The power of preconditioning in overparameterized low-rank matrix sensing. In Proceedings of the 40th International Conference on Machine Learning, pages 38611–38654, 2023.

[33] Xiaojing Zhu and Fengyi Yuan. A Riemannian regularized Gauss–Newton method for low-rank matrix completion. AIMS Mathematics, 10(12):28556–28582, 2025.

[34] Pini Zilber and Boaz Nadler. GNMR: A provable one-line algorithm for low rank matrix recovery. SIAM Journal on Mathematics of Data Science, 4(2):909–934, 2022.