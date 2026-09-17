# Provable Guarantees and Eficient Learning of Structural Equation Models with Latent Confounders

Weijian Yu CIS, The University of Melbourne weijian.yu@student.unimelb.edu.au

Jean Honorio CIS, The University of Melbourne jean.honorio@unimelb.edu.au

## Abstract

Causal discovery aims to recover causal relationships from observed data. In various fields, exploring causal relationships among variables remains an important topic, but this task becomes challenging due to the existence of latent confounders. Ignoring such confounders can lead to false associations and incorrect edge directions. In this paper, we study the linear structural equation model with latent confounders. We propose an algorithm that iteratively identifies terminal (observed) nodes and reconstructs the directed acyclic graph of the observed variables. To do this, we recover the precision matrix of the observed variables as a sparse plus low-rank matrix: a sparse matrix captures the conditional dependencies among observed variables, while a low-rank matrix captures the combined influence of a few latent confounders. We establish that for p observed variables, r latent confounders and s edges, our procedure correctly identifies the directed causal relationship among observed variables, for n ≳ max{s log p, rp} samples. Experimental results validate our theoretical contributions.

## 1 Introduction

Causal discovery allows to infer causal relationships from observed data. This is crucial for understanding complex systems such as genetics and finance, especially when control experiments are not feasible, costly or unethical. A frequently adopted simplification is to assume causal suficiency, i.e., there are no hidden confounding factors, and thus any association is fully determined by the other observed variables. Real data rarely meets this requirement. For instance, a psychological questionnaire reflects hidden psychological factors. In vision and language, pixels and symbols are driven by underlying semantic factors. These latent variables can induce interdependencies, masquerading as causal relationships.

Our goal is to discover directed causal relationships between observed variables, in the presence of latent confounders. In this paper, we focus on the parameterization of the precision matrix (encoding undirected causal relationships) and the identification of directed causal relationships among observed variables. The closest to our goal is fast causal inference [15, 3], which does not impose any particular assumptions on latent variables, but outputs less information (i.e., a partial ancestral graph) and can potentially make an exponential number of conditional independence tests. The case of undirected causal relationships between observed variables in the presence of latent confounders was previously studied by [2, 8]. For directed causal relationships, while the fully observed case has been largely studied [5, 18, 11] and we cannot do justice to the large body of work, there is a lack of methods that work in the presence of latent confounders.

In the literature, there are problems that include latent variables in directed causal discovery, but that do not relate to our goal. [14, 6, 10] propose algorithms that can efectively learn directed causal relationships between latent variables and observed variables, as well as directed causal relationships among latent variables. However, their methods do not focus on directed causal relationships between observed variables. Another line work focuses on grouping observed variables that are likely afected by the same latent variable [13, 16]. Finally, a last line of work assumes that such latent-to-observed variable grouping is known [4].

Contributions. We develop a linear structural equation model (SEM) with latent confounders, which can efectively identify the causal relationships between observed variables. In our model, the precision matrix follows a sparse plus low-rank decomposition. We use a regularized maximum likelihood estimation (MLE) approach [2] to estimate the precision matrix (encoding undirected causal relationships). Inspired by [17, 8], under the conditions of Restricted Strong Convexity and Structural Incoherence, we derive error bounds for the regularized MLE. Then we use the precision matrix to identify the directed causal relationships among the observed variables, using an algorithm similar to [5]. We then provide provable recovery guarantees for our approach. More specifically, we show that if the number of samples n fulfills n $\gtrsim \frac { 1 } { \epsilon ^ { 2 } }$ max {s log p, rp} for $p$ observed variables, r latent confounders and s edges, then our method correctly recovers the directed causal edges.

## 2 Preliminaries

In this section, we first define some notations. We then explain our main object of analysis: linear structural equation models (SEMs) with latent confounders. We then present the connection between SEMs and undirected graphical models, which is important for our algorithm and guarantees.

## 2.1 Notations

We use bold lowercase to represent vectors, and bold uppercase to represent matrices. We write $[ \boldsymbol { p } ] : = \{ 1 , \dots , p \}$ We write the set $- i : = [ p ] \backslash \{ i \}$ . For a matrix A, the trace is denotes as $\operatorname { t r } ( \mathbf { A } )$ and vec(A) denotes the vectorization of the matrix. The support of matrix A is denoted as $\mathrm { s u p p } ( \mathbf { A } ) = \{ ( i , j ) \in [ p ] \times [ p ] | \mathbf { A } _ { i , j } \neq 0 \}$ diag(A) extracts the main diagonal of A and Diag(v) is the diagonal matrix with v on its diagonal. We first introduce some notation. For index sets $I , J \subseteq [ { \bar { p } } ] , \mathbf { A } _ { I , J } \in \mathbb { R } ^ { | { \bar { I } } | \times | J | }$ is the sub-matrix of $\mathbf { A } \in \mathbb { R } ^ { p \times p }$ with rows in I and columns in $J ;$ in this context, the symbol • denotes all rows (or columns). For a matrix A, $\| \mathbf { A } \| _ { 1 }$ and $\| \mathbf { A } \| _ { \infty }$ denote its entrywise $\ell _ { 1 }$ and $\ell _ { \infty }$ norms, respectively. $\| \mathbf { A } \| _ { 2 } , \| \mathbf { A } \| _ { F } , \| \mathbf { A } \|$ denote the spectral, Frobenius and nuclear of matrix A, respectively.

A directed graph of p nodes is denoted as $G = ( [ p ] , E )$ with edges $E \subseteq [ p ] \times [ p ]$ , where $( i , j ) \in E$ represents a directed edge from j to i. A directed acyclic graph (DAG) is a directed graph without cycles (i.e., starting from any node and following the direction of the edges, one cannot return to the original node). For node $i , \pi _ { G } ( i )$ denote the sets of parents and $\phi _ { G } ( i )$ denote the sets of children in graph G. We call node i terminal ${ \mathfrak { i } } \phi _ { G } ( i ) = \emptyset$ Let $\mathcal { T } _ { G }$ be a set of topological orderings over [p] in G. $\mathcal { T } _ { G } = \left\{ \tau \in S _ { p } | \tau ( j ) < \tau ( i ) \mathrm { ~ i f ~ } ( i , j ) \in E \right\}$ , where $S _ { p }$ is the set of all possible permutations of [p]. For any topological order $\tau \in { \mathcal { T } } _ { G }$ and any $m \in [ p ]$ , define the sequence of graphs $G [ m , \tau ] = ( V [ m , \tau ] , E [ \bar { m } , \bar { \tau } ] )$ where $G [ m , \tau ]$ is the induced subgraph of G on the first m nodes in the ordering τ, i.e., $\dot { V } [ m , \tau ] = \{ i \in \lbrack \dot { p } ] \mid \tau ( i ) \leq m \}$ and $E [ m , \tau ] = \{ ( i , j ) \in E \mid i \in V [ m , \tau ] , j \in V [ m , \tau ] \}$ Equivalently, $G [ m , \tau ]$ contains exactly the first m nodes under τ and all the edges among them.

## 2.2 A Linear SEM with Latent Confounders

Let $p$ be the number of observed variables and $r < p$ be the number of latent confounders. Let $\mathbf { B } \in \mathbb { R } ^ { p \times p }$ encode the causal efects between observed variables, $\mathbf { C _ { \lambda } } \in \mathbb { R } ^ { p \times r }$ encode the causal efects from latent confounders to observed variables, and $\mathbf { D } \in \mathbb { R } ^ { r \times r }$ encode the causal efects among latent confounders. In other words, the edges in the DAG is defined by the support of the matrices $\mathbf { B } , \mathbf { C }$ and D.

In a linear SEM, the random observed vector $\mathbf { x } \in \mathbb { R } ^ { p }$ and the latent vector $\mathbf { z } \in \mathbb { R } ^ { r }$ can be written as the linear combination $\mathbf { x } = \mathbf { B } \mathbf { x } + \mathbf { C } \mathbf { z } + \varepsilon _ { X }$ and $\mathbf { z } = \mathbf { D } \mathbf { z } + \varepsilon _ { Z }$ , or equivalently,

$$
{ \Bigg [ } \mathbf { x } { \Bigg ] } = { \Bigg [ } { \mathbf { B } } { \mathbf { C } } { \Bigg ] } { \Big [ } { \mathbf { x } } { \Big ] } + { \Big [ } { \pmb { \varepsilon } } _ { X } { \Big ] } ,\tag{1}
$$

where the noise variables fulfill $\mathbb { E } [ \pmb { \varepsilon } _ { X } ] = \mathbf { 0 } , \mathbb { E } [ \pmb { \varepsilon } _ { Z } ] = \mathbf { 0 }$ and E ${ \mathfrak { T } } \left[ { \left[ { \pmb { \varepsilon } } _ { X } \right] \left[ { \pmb { \varepsilon } } _ { X } \right] } ^ { \top } \right] = \mathbf { D i a g } \big ( \{ \sigma _ { i } ^ { 2 } \} _ { i = 1 } ^ { p + r } \big )$ . We denote the SEM over the observed variables as $( G , \mathbf { B } , \{ \sigma _ { i } ^ { 2 } \} _ { i = 1 } ^ { p } )$ where $\mathbf { \bar { \boldsymbol { G } } } = ( [ \boldsymbol { p } ] , E )$ and $E = \mathrm { s u p p } ( \mathbf { B } )$ . We denote the SEM over observed variables and latent confounders as $\left( G _ { f } , \left[ \mathbf { B } \quad \mathbf { C } \right] , \{ \sigma _ { i } ^ { 2 } \} _ { i = 1 } ^ { p + r } \right)$ where $G _ { f } = ( [ p + r ] , E _ { f } )$ and $E _ { f } = \operatorname { s u p p } \left( { \left[ \begin{array} { l l } { \mathbf { B } } & { \mathbf { C } } \\ { \mathbf { 0 } } & { \mathbf { D } } \end{array} \right] } \right)$

We assume we have n samples from the true distribution, but which access to only p observed variables (x). That is, we do not have access to r latent confounders (z). Our goal is to recover B (and its support) by using only observed data. Formally speaking, we receive a data matrix $\mathbf { X } \in \mathbb { R } ^ { n \times p }$ of observed variables, which comes from $\left( G _ { f } , \left[ \mathbf { B } ^ { * } \quad \mathbf { C } ^ { * } \right] , \{ \sigma _ { i } ^ { 2 } \} _ { i = 1 } ^ { p + r } \right)$ , and we want to recover a SEM $( \widehat { G } , \widehat { \mathbf { B } } , \{ \widehat { \sigma } _ { i } ^ { 2 } \} _ { i = 1 } ^ { p } )$ such that $G ^ { * } = { \widehat { G } }$ or equivalently, such that $\mathrm { s u p p } ( \widehat { \mathbf { B } } ) = \mathrm { s u p p } ( \mathbf { B } ^ { * } )$ .

## 2.3 SEMs and Undirected Graphical Models with Latent Confounders

While linear SEMs are directed graphical models, we can also view them as undirected graphical models. We leverage this connection later in Section 3.4 to motivate a new algorithm for learning SEMs from data, with provable guarantees. Define the (full) covariance matrix as follows:

$$
\Sigma _ { f } : = \mathbb { E } \left[ \left[ \mathbf { x } \right] \left[ \mathbf { x } \right] ^ { \top } \right] : = \left[ \begin{array} { l l } { \Sigma } & { \Sigma _ { X , Z } } \\ { \Sigma _ { X , Z } ^ { \top } } & { \Sigma _ { Z , Z } } \end{array} \right] ,\tag{2}
$$

where $\Sigma : = \mathbb { E } \left[ \mathbf { x x } ^ { \top } \right] \in \mathbb { R } ^ { p \times p } , \ \Sigma _ { X , Z } \in \mathbb { R } ^ { p \times r }$ and $\Sigma _ { Z , Z } \in \mathbb { R } ^ { r \times r }$ . Let $\mathbf { J } _ { f }$ be the precision matrix related to the (full) covariance matrix $\Sigma _ { f }$ , i.e.,

$$
\mathbf { J } _ { f } : = \Sigma _ { f } ^ { - 1 } : = \left[ \begin{array} { l l } { \mathbf { S } } & { \mathbf { J } _ { X , Z } } \\ { \mathbf { J } _ { X , Z } ^ { \top } } & { \mathbf { J } _ { Z , Z } } \end{array} \right] ,\tag{3}
$$

where $\mathbf { S } \in \mathbb { R } ^ { p \times p } , \mathbf { J } _ { X , Z } \in \mathbb { R } ^ { p \times r }$ and $\mathbf { J } _ { Z , Z } \in \mathbb { R } ^ { r \times r }$ . Let $\ b { \Omega } \in \mathbb { R } ^ { p \times p }$ be the marginal precision matrix related to $\pmb { \Sigma } = \mathbb { E } \left[ \mathbf { x } \mathbf { x } ^ { \top } \right]$ , i.e., $\pmb { \Omega } : = \pmb { \Sigma } ^ { - 1 }$ . It can be shown that (see Appendix A for the derivations),

$$
\begin{array} { r } { \Omega = \mathbf { S } + \mathbf { L } , \qquad \mathbf { S } = ( \mathbf { I } - \mathbf { B } ) ^ { \top } \mathbf { N } _ { X } ^ { - 1 } ( \mathbf { I } - \mathbf { B } ) , \qquad \mathbf { L } = - \mathbf { J } _ { X , Z } \mathbf { J } _ { Z , Z } ^ { - 1 } \mathbf { J } _ { X , Z } ^ { \top } . } \end{array}\tag{4}
$$

where $\mathbf { N } _ { X } = \mathbf { D i a g } \big ( \{ \sigma _ { i } ^ { 2 } \} _ { i = 1 } ^ { p } \big )$ . First, note that S captures the condition dependencies among observed variables, while L captures the commbined influence of latent confounders. For this reason, we call S the confounder-free precision matrix. Moreover, S is sparse and L is a matrix of rank at most r (since $\mathbf { J } _ { Z , Z } \in \mathbb { R } ^ { r \times r } )$ . Thus, the marginal precision matrix Ω can be written as the sum of a sparse component and a low-rank component.

Remark 2.1. From $e q . ( 4 )$ , one can observe that successful recovery of S from training data, would imply successful recovery of B. This motivates our algorithm and the study of its theoretical guarantees.

Given n samples for the observed variables only, i.e., $\mathbf { X } \in \mathbb { R } ^ { n \times p }$ , we define the observed sample covariance matrix as $\begin{array} { r } { \widehat { \mathbf { \Sigma } } = \frac { \widehat { \mathbf { \Xi } } } { n } \mathbf { X } ^ { \top } \mathbf { X } } \end{array}$ . The regularized MLE for undirected graphical models with latent confounders [2, 8] solves the following optimization problem:

$$
\begin{array} { r l } { \underset { \mathbf { s } , \mathbf { L } } { \mathrm { m i n i m i z e } } } & { { } \mathcal { L } ( \mathbf { S } + \mathbf { L } ; \mathbf { X } ) + \lambda \| \mathbf { S } \| _ { 1 } + \mu \| \mathbf { L } \| _ { * } } \\ { \mathrm { s u b j e c t ~ t o } } & { { } - \mathbf { L } \succeq 0 , \quad \mathbf { S } + \mathbf { L } \succeq 0 , } \end{array}\tag{5}
$$

where $\lambda , \mu > 0$ are regularization constants and $\mathcal { L } ( \Omega ; \mathbf { X } ) : = \langle \widehat { \boldsymbol { \Sigma } } , \Omega \rangle - \log \operatorname* { d e t } ( \Omega )$ is the negative log-likelihood function. In $\mathrm { e q . ( 5 ) }$ the $\ell _ { 1 }$ norm regularizer on S encourages sparsity since the $\ell _ { 1 }$ norm is a convex surrogate for the number of none-zero entries in S. Similarly, the nuclear norm regularizer on L encourages low-rankness since the nuclear norm is a convex surrogate for the rank of L.

## 3 Main Results

In this section, we first discuss the theoretical framework needed for the analysis, which borrows from undirected graphical models. We then provide recovery guarantees for the confounder-free precision matrix. Then, we turn our attention to directed graphical models and present a suficient and necessary condition for identifiability. Armed with those results, we end the section by presenting our algorithms and its provable theoretical guarantees.

## 3.1 Decomposable Regularization for Undirected Graphical Models with Latent Confounders

Our results build on the framework of [9, 17] for estimation with superposition of structurally constrained parameters. This framework was also used in the estimation of sparse and low-rank undirected graphica models in [8]. Next, we discuss the diferent definitions and assumptions relevant this problem.

Decomposable regularizers. We first review some terms in [9]. Let M be the model subspace which captures constraints on the model parameters and $\overline { { \mathcal { M } } } ^ { \perp }$ be the perturbation subspace with perturbations far from the model subspace. $( \mathcal { M } , \overline { { \mathcal { M } } } ^ { \perp } )$ denote a pair of subspaces, where ${ \mathcal { M } } \subseteq { \overline { { { \mathcal { M } } } } }$ . A regularization function $\mathcal { R } ( \cdot )$ is called decomposable for a subspace pair $( \mathcal { M } , \overline { { \mathcal { M } } } ^ { \perp } ) \mathrm { i f } \mathcal { R } ( u + v ) = \mathcal { R } ( u ) + \mathcal { R } ( v )$ , for all $u \in \mathcal { M }$ and $v \in \overline { { \mathcal { M } } } ^ { \perp }$

The $\ell _ { 1 }$ norm is decomposable for sparse matrices. Let $E \subseteq [ p ] \times [ p ]$ be a set of index pairs where the entries of the sparse matrix is non-zero. Let $\mathcal { M } ( E ) = \overline { { \mathcal { M } } } ( E )$ denote the subspace of all sparse matrices in $\mathbb { R } ^ { p \times p }$ supported in the subset of $E .$ Let $E ^ { c }$ be the complement of $E .$ . Note that $\| \mathbf { A } \| _ { 1 } = \| \mathbf { A } _ { E } \| _ { 1 } + \| \mathbf { A } _ { E ^ { c } } \|$ 1 which implies decomposability.

The nuclear norm is decomposable for symmetric positive semi-definite low-rank matrices, as shown in [8]. Let $( \mathcal { M } , \overline { { \mathcal { M } } } ^ { \perp } )$ be a pair of subspaces. Following [9], we define the structural error set at $\Omega ^ { * }$ by

$$
\mathcal { C } ( \mathcal { M } , \overline { { \mathcal { M } } } ^ { \perp } ; \Omega ^ { * } ) : = \Big \{ \Delta \in \mathbb { R } ^ { n \times p } \Big | \mathcal { R } ( \Delta _ { \overline { { \mathcal { M } } } ^ { \perp } } ) \leq 3 \mathcal { R } ( \Delta _ { \overline { { \mathcal { M } } } } ) + 4 \mathcal { R } \big ( \Omega _ { \overline { { \mathcal { M } } } ^ { \perp } } ^ { * } \big ) \Big \} ,
$$

In essence, if the true precision matrix $\Omega ^ { * }$ has only a small component in the orthogonal complement $\overline { { \mathcal { M } } } ^ { \perp }$ then any $\pmb { \Delta }$ from C must have a small projection onto ${ \overline { { \mathcal { M } } } } ^ { \perp }$

Let the true precision matrix $\Omega ^ { * }$ decompose into a sparse component $\mathbf { S } ^ { \ast }$ and a low-rank component $\mathbf { L } ^ { * } , \mathrm { i . e . , } \Omega ^ { * } = \mathbf { S } ^ { * } + \mathbf { L } ^ { * }$ . For the sparse component $\mathbf { S } ^ { \ast }$ , we define a pair of subspaces $\left( \mathcal { M } ( E ) , \overline { { \mathcal { M } } } ^ { \perp } ( E ) \right)$ , and we let $\mathcal { C } ( E ) ~ = ~ \mathcal { C } \big ( \mathcal { M } ( E ) , \mathcal { M } ( E ) ^ { \perp } ; ~ \mathbf { S } ^ { * } \big )$ . For the low-rank component $\mathbf { L } ^ { \ast }$ , we define a pair of subspaces $( \mathcal { M } ( U ) , \overline { { \mathcal { M } } } ^ { \perp } ( U ) )$ , and we let $\mathcal { C } ( U ) \ = \ \mathcal { C } ( \mathcal { M } ( U ) , \mathcal { M } ( U ) ^ { \perp } ; ~ \mathbf { L } ^ { * } )$ . In later analysis, perturbations of $\Omega ^ { * }$ are restricted to the directions in these two sets.

Restricted Strong Convexity (RSC). Given some set ${ \mathcal { C } } _ { : }$ , the loss $\mathcal { L }$ satisfies RSC [9] on $\mathcal { C }$ if there exists a tolerance function $\tau _ { \mathcal { L } }$ and some curvature parameter $\kappa _ { \mathcal { L } } > 0$ such that

$$
\delta \mathcal { L } ( \Delta ; \Omega ^ { * } ) \geq \kappa _ { \mathcal { L } } \| \Delta \| _ { F } ^ { 2 } - \tau _ { \mathcal { L } } ( \Omega ^ { * } ) , \qquad \forall \Delta \in \mathcal { C } .
$$

where $\pmb { \Delta } = \pmb { \Omega } ^ { * } - \pmb { \Omega }$ and $\delta \mathcal { L }$ is the first-order Taylor remainder of the loss $\mathcal { L }$ at $\Omega ^ { * }$ , i.e., $\delta \mathcal { L } ( \Delta ; \Omega ^ { * } ) =$ $\mathcal { L } ( \Omega ^ { * } + \Delta ) - \mathcal { L } ( \Omega ^ { * } ) - \left. \nabla \mathcal { L } ( \Omega ^ { * } ) , \Delta \right.$

Structural Incoherence (SI). To control the interaction between the sparse and the low-rank components, we assume that $\mathcal { L }$ satisfies the SI condition [17]. That is, for all $\Delta _ { S } \in { \mathcal { C } } ( E ) , \ \Delta _ { L } \in { \mathcal { C } } ( U )$

$$
c _ { \mathcal { L } } ( \Delta _ { S } , \Delta _ { L } ; \Omega ^ { * } ) \ \leq \ \frac { \kappa _ { \mathcal { L } } } { 2 } \Big ( \| \Delta _ { S } \| _ { F } ^ { 2 } + \| \Delta _ { L } \| _ { F } ^ { 2 } \Big ) ,
$$

where $\kappa _ { \mathcal { L } }$ is defined as in RSC, and $c _ { \mathcal { L } }$ is the incoherence function, defined as $c _ { \mathcal { L } } ( \Delta _ { S } , \Delta _ { L } ; \Omega ^ { * } ) = | \mathcal { L } ( \Omega ^ { * } +$ $\Delta _ { S } + \Delta _ { L } ) + \mathcal { L } ( \Omega ^ { * } ) - \mathcal { L } ( \Omega ^ { * } + \Delta _ { S } ) - \mathcal { L } ( \Omega ^ { * } + \Delta _ { L } ) |$

Recall that we use the (undirected graphical model) optimization problem in $\mathrm { e q . } ( 5 )$ . Unfortunately, the analysis of [8] does not provide a recovery guarantee for $\mathbf { S } _ { \mathrm { i } }$ , but for $\Omega = \mathbf { S } + \mathbf { L }$ . In this paper, we follow similar assumptions as in [8], but provide a recovery guarantee for S.

The following two assumptions for the Fisher information from [8] allows to show that the problem in $\mathrm { e q . } ( 5 )$ fulfills the RSC and SI conditions. The Fisher information at the true precision matrix $\Omega ^ { * }$ is $\mathcal { F } ^ { * } = \Omega ^ { * - 1 } \otimes \Omega ^ { * - 1 }$ , where $\otimes$ denotes the Kronecker product. The Fisher inner product between matrices $\Delta _ { A }$ and $\Delta _ { B }$ is defined as $\langle \pmb { \Delta } _ { A } , \pmb { \Delta } _ { B } \rangle _ { \mathcal { F } ^ { * } } : = \mathbf { v e c } ( \pmb { \Delta } _ { A } ) ^ { \top } \mathcal { F } ^ { * } \mathbf { v e c } ( \pmb { \Delta } _ { B } ) = \mathrm { t r } \big ( \pmb { \Omega } ^ { * - 1 } \pmb { \Delta } _ { A } \pmb { \Omega } ^ { * - 1 } \pmb { \Delta } _ { B } \big )$ . This inner product induces the Fisher norm [7], formally defined as $\| \pmb { \Delta } \| _ { \mathcal { F } ^ { * } } ^ { 2 } : = \mathbf { v e c } ( \pmb { \Delta } ) ^ { \top } \mathcal { F } ^ { * } \mathbf { v e c } ( \pmb { \Delta } ) = \mathrm { t r } \big ( \pmb { \Omega } ^ { * - 1 } \pmb { \Delta } \pmb { \Omega } ^ { * - 1 } \pmb { \Delta } \big )$

Assumption 3.1 (Restricted Fisher Eigenvalue, Assumption 1 in [8]). There exists a constant $\kappa _ { \mathrm { m i n } } ^ { * } > 0$ such that

$$
\| \boldsymbol { \Delta } \| _ { \mathcal { F } ^ { * } } ^ { 2 } \ \ge \ \kappa _ { \mathrm { m i n } } ^ { * } \| \boldsymbol { \Delta } \| _ { F } ^ { 2 } , \qquad \forall \boldsymbol { \Delta } \in \mathcal { C } ( E ) \cup \mathcal { C } ( U ) .
$$

This RFE condition generalizes the restricted eigenvalue condition for sparsity-promoting linear regression problems [1].

Let $\mathcal { P } _ { E } = \mathcal { P } _ { \overline { { \mathcal { M } } } ( E ) } , \mathcal { P } _ { U } = \mathcal { P } _ { \overline { { \mathcal { M } } } ( U ) } , \mathcal { P } _ { E ^ { \perp } } = \mathcal { P } _ { \overline { { \mathcal { M } } } ( E ) ^ { \perp } } , \mathcal { P } _ { U ^ { \perp } } = \mathcal { P } _ { \overline { { \mathcal { M } } } ( U ) } .$ <sub>⊥</sub> be the projection operator onto the subspaces $\overline { { \mathcal { M } } } ( E ) , \overline { { \mathcal { M } } } ( U ) , \overline { { \mathcal { M } } } ( E ) ^ { \perp } , \overline { { \mathcal { M } } } ( U ) ^ { \perp }$ , respectively. We assume the following conditions for the Fisher information.

Assumption 3.2 (Structural Fisher Incoherence, Assumption 2 in [8]). Given $M > 6$ , and define the subspace pairs $\left( \mathcal { M } ( E ) , \overline { { \mathcal { M } } } ^ { \perp } ( E ) \right)$ and $( \mathcal { M } ( U ) , \overline { { \mathcal { M } } } ^ { \perp } ( U ) )$ . Let $\begin{array} { r } { \Lambda = 2 + 3 \operatorname* { m a x } \biggr \{ \frac { \lambda \sqrt { s } } { \mu \sqrt { r } } , \ \frac { \mu \sqrt { r } } { \lambda \sqrt { s } } \biggr \} } \end{array}$ , where $s = | E |$ is the number of elements in $E$ and $r = { \mathrm { r a n k } } ( U )$ . Given regularization parameters λ and $\mu _ { ; }$ , then the Fisher information ${ \mathcal { F } } ^ { * }$ satisfies:

$$
\operatorname* { m a x } \Big \{ \bar { \sigma } ( \mathcal { P } _ { E } \mathcal { F } ^ { * } \mathcal { P } _ { U } ) , ~ \bar { \sigma } ( \mathcal { P } _ { E ^ { \perp } } \mathcal { F } ^ { * } \mathcal { P } _ { U } ) , ~ \bar { \sigma } ( \mathcal { P } _ { E } \mathcal { F } ^ { * } \mathcal { P } _ { U ^ { \perp } } ) , ~ \bar { \sigma } ( \mathcal { P } _ { E ^ { \perp } } \mathcal { F } ^ { * } \mathcal { P } _ { U ^ { \perp } } ) \Big \} ~ \leq ~ \frac { \kappa _ { \operatorname* { m i n } } ^ { * } } { c _ { 1 } \Lambda ^ { 2 } } ,
$$

where $\begin{array} { r } { c _ { 1 } = \frac { 1 6 M } { M - 6 } } \end{array}$ and $\bar { \sigma } ( \cdot )$ is the maximum singular value.

The next technical result from [8] show that Restricted Fisher Eigenvalue and Structural Fisher Incoherence, together imply the RSC and SI condition. This allows us to use the framework of decomposable regularization for the analysis of $\mathrm { e q . ( 5 ) }$ .

Proposition 3.3 (RFE and SFI imply RSC and SI, Lemma 2 and 3 in [8]). Let $\Omega ^ { * }$ be the true marginal precision matrix and suppose Assumption 3.1 and Assumption 3.2 hold for Ω<sup>∗</sup>, and let $M > 6$ . Restricted Strong Convexity is satisfied with tolerance function $\tau _ { \mathscr { L } } = 0$ and curvature parameter $\kappa _ { \mathcal { L } } ~ = ~ \frac { M - 2 } { 2 ( M - 1 ) } \kappa _ { \mathrm { m i n } } ^ { \star }$ for all $\Delta \in { \mathcal { C } } ( E ) \cup { \mathcal { C } } ( U )$ such that $\| \Delta \| _ { \mathcal { F } ^ { * } } ^ { 2 } \ \le \ \frac { 1 } { 2 M ^ { 2 } }$ . Furthermore, Structural Incoherence is satisfied for all $\Delta _ { S } \in { \mathcal { C } } ( E )$ and $\Delta _ { L } \in { \mathcal { C } } ( U )$ , such that max $\left\{ \| \Delta _ { S } \| _ { \mathcal { F } ^ { \star } } ^ { 2 } , \| \Delta _ { L } \| _ { \mathcal { F } ^ { \star } } ^ { 2 } \right\} \ \leq \ \frac { 1 } { 6 M ^ { 2 } }$

## 3.2 Recovery Guarantees for the Confounder-Free Precision Matrix

Next, we present recovery guarantees for the undirected graphical model. In particular, we show that the confounder-free precision matrix S can be successfully recovered. We want to point out that the results in [8] provided recovery guarantees for $\Omega = \mathbf { S } + \mathbf { L }$ only, making it dificult to disentangle de contributions of S and L in the final error bound. In contrast, we provide an error bound for both the sparse and low-rank components.

Theorem 3.4 (Deterministic bound for S and L). Let $\Omega ^ { * }$ be the true marginal precision matrix and suppose Assumption 3.1 and Assumption 3.2 hold for Ω<sup>∗</sup>. Let $\widehat { \boldsymbol { \Sigma } } = \frac { 1 } { n } \boldsymbol { \mathbf { X } } ^ { \intercal }$ X denote the sample covariance matrix, and $\Sigma ^ { * }$ the true covariance matrix. If the regularization parameters $f u l f l l$

$$
\lambda \geq 2 \| \widehat { \pmb { \Sigma } } - { \pmb { \Sigma } } ^ { * } \| _ { \infty } \quad a n d \quad \mu \geq 2 \| \widehat { \pmb { \Sigma } } - { \pmb { \Sigma } } ^ { * } \| _ { 2 } ,
$$

then the following error bound holds for the estimators $\widehat { \mathbf { S } }$ and $\widehat { \mathbf { L } }$

$$
\| \widehat { \mathbf { S } } - \mathbf { S } ^ { * } \| _ { \infty } + \| \widehat { \mathbf { L } } - \mathbf { L } ^ { * } \| _ { \infty } \leq \frac { 6 } { \kappa _ { \mathcal { L } } } \operatorname* { m a x } \Big \{ \lambda \sqrt { s } , \mu \sqrt { r } \Big \} .
$$

where s is the number of none-zero entries in $\mathbf { S ^ { * } } , r$ is the number of latent confounders, and $\begin{array} { r } { \kappa _ { \mathscr { L } } : = \frac { M - 2 } { 2 ( M - 1 ) } \kappa _ { \operatorname* { m i n } } ^ { * } . } \end{array}$

(Ommitted proofs can be found in Appendix B.)

Remark 3.5. Note that the above guarantee is in the form $\| \widehat { \mathbf { S } } - \mathbf { S } ^ { * } \| _ { \infty } + \| \widehat { \mathbf { L } } - \mathbf { L } ^ { * } \| _ { \infty } \leq \epsilon$ . Since norms are non-negative, we have that $\| \widehat { \mathbf { S } } - \mathbf { S } ^ { * } \| _ { \infty } \leq \epsilon$ and $\| \widehat { \mathbf { L } } - \mathbf { L } ^ { * } \| _ { \infty } \leq \epsilon$

[17] provided a general result for estimation with superposition of structurally constrained parameters. We prove the theorem above by applying the general results of [17] to the specific problem of sparse and low-rank regularization in $\mathrm { e q . ( 5 ) }$

Theorem 3.4 provides a deterministic statement that does not consider the fact that data is random, and thus the sample covariance matrix $\hat { \Sigma }$ is a random variable. Next, we address this issue.

Theorem 3.6 (High-probability bound for S and L). Let $\Omega ^ { * }$ be the true marginal precision matrix and suppose Assumption 3.1 and Assumption 3.2 hold for Ω<sup>∗</sup>. Choose some constants $C _ { 1 } \geq 3 / 2$ and $C _ { 2 } \geq 1$ Assume the number of samples n satisfies $n \geq \operatorname* { m a x } \{ 4 C _ { 1 } ^ { 2 } \log p , ~ C _ { 2 } ^ { 2 } p \}$ . Set the regularization parameters as

$$
\lambda = 1 6 0 C _ { 1 } \bar { \sigma } ^ { * } \sqrt { \frac { \log p } { n } } \qquad a n d \qquad \mu = 1 6 C _ { 2 } \rho ^ { * } \sqrt { \frac { p } { n } } ,
$$

where $\bar { \sigma } ^ { * } = \operatorname* { m a x } _ { i } \Sigma _ { i , i } ^ { * } , \rho ^ { * } = \lVert \Sigma ^ { * } \rVert _ { 2 }$ and $p$ is the number of observed variables. With probability at least $1 - 4 p ^ { - 2 ( C _ { 1 } - 1 ) } - 2 \exp \bigl ( - \frac { C _ { 2 } ^ { 2 } p } { 2 } \bigr )$ , we have

$$
\| \widehat { \mathbf { S } } - \mathbf { S } ^ { \ast } \| _ { \infty } + \| \widehat { \mathbf { L } } - \mathbf { L } ^ { \ast } \| _ { \infty } \leq \operatorname* { m a x } \Bigl \{ c _ { 1 } \sqrt { \frac { s \log p } { n } } , ~ c _ { 2 } \sqrt { \frac { r p } { n } } \Bigr \} ,
$$

where $c _ { 1 } = \frac { 9 6 0 } { \kappa _ { L } } \bar { \sigma } ^ { * } C _ { 1 } , c _ { 2 } = \frac { 9 6 } { \kappa _ { L } } \rho ^ { * } C _ { 2 }$ and $\begin{array} { r } { \kappa _ { \mathcal { L } } = \frac { M - 2 } { 2 ( M - 1 ) } \kappa _ { \mathrm { m i n } } ^ { * } , s } \end{array}$ is the number of none-zero entries in $\mathbf { S } ^ { \ast }$ , and r is the number of latent confounders.

Later in Section 3.4, we motivate an algorithm that removes terminal nodes sequentially, one at a time, while having guarantees of recovering the true topological ordering. At each iteration, our algorithm needs to solve $\mathrm { e q . ( 5 ) }$ . For our algorithm to have good statistical guarantees, we now show that if a (entrywise $\ell _ { \infty }$ or spectral) norm deviation holds for matrices S and L, then it also holds for all principal submatrices of S and L.

Claim 3.7. $I f \lambda \ge 2 \| \widehat { \pmb { \Sigma } } - { \pmb { \Sigma } } ^ { * } \| _ { \infty }$ and $\mu \geq 2 \| \widehat { \pmb { \Sigma } } - \pmb { \Sigma } ^ { * } \| _ { 2 }$ , then for all $I \subset [ p ]$ , we have $\lambda \geq 2 \| \widehat { \pmb { \Sigma } } _ { I , I } - { \pmb { \Sigma } } _ { I , I } ^ { * } \| _ { \infty }$ and $\mu \geq 2 \Vert \widehat { \Sigma } _ { I , I } - \Sigma _ { I , I } ^ { * } \Vert _ { 2 }$

## 3.3 Identifiability for SEMs over the Observed Variables

Now, we turn our attention to directed graphical models, specifically to SEMs. The previous section focused on the recovery of the confounder-free precision matrix S. Here we provide assumptions under which the weight matrix B can be successfully recovered from S.

The next assumption is very relevant since our algorithm sequentially removes terminal nodes. Thus, we require that our RSC and SI conditions hold for all induced subgraphs of G and topological orderings. By Proposition 3.3, RFE and SFI imply the above conditions.

Assumption 3.8 (RFE and SFI for all induced subgraphs). Let $\Omega [ m , \tau ]$ denote the true precision matrix over the observed nodes in $V [ m , \tau ]$ . For all induced subgraphs $G [ m , \tau ]$ $m \in [ p ]$ , and all topological orderings $\tau \in \mathcal { T } _ { G }$ , suppose Assumption 3.1 and Assumption 3.2 hold with constant $\kappa _ { \mathrm { m i n } } ^ { * } > 0$ . That $i s , \kappa _ { \mathrm { m i n } } ^ { * }$ is the smallest constant among all $m \in [ p ]$ , and $\tau \in { \mathcal { T } } _ { G }$ satisfying Assumption 3.1 and Assumption 3.2.

In the context of fully observed data (without latent confounders), the next identifiability condition is not only suficient but also necessary for the identifiability of SEMs. For instance, Lemma 1 in [5] showed that if the identifiability condition does not hold, then there exists exponentially many diferent SEMs that could have produced the given training data.

Assumption 3.9 (Identifiability Condition, Assumption 1 in [5]). Let $( G , \mathbf { B } , \{ \sigma _ { i } ^ { 2 } \} _ { i = 1 } ^ { p } ) )$ be a SEM and let $\mathbf { \bar { S } } = ( \mathbf { I } - \mathbf { B } ) ^ { \top } \mathbf { N } _ { X } ^ { - 1 } ( \mathbf { I } - \mathbf { B } )$ be the related precision matrix, where $\mathbf { N } _ { X } = \mathbf { D i a g } \big ( \{ \sigma _ { i } ^ { 2 } \} _ { i = 1 } ^ { p } \big )$ . For all $( i , j ) \in V [ m , \tau ] \times V [ m , \tau ] , m \in [ p ]$ , and $\tau \in \mathcal { T } _ { G }$ such that $\phi _ { G [ m , \tau ] } ( i ) = \emptyset$ and $\phi _ { G [ m , \tau ] } ( j ) \neq \emptyset$

$$
\frac { 1 } { \sigma _ { i } ^ { 2 } } < \frac { 1 } { \sigma _ { j } ^ { 2 } } + \sum _ { l \in \phi _ { G [ m , \tau ] } ( j ) } \frac { { \bf B } _ { l , j } ^ { 2 } } { \sigma _ { l } ^ { 2 } } .
$$

Given the above assumption, the next technical result from [5] allows us to recover both the true topological order of the graph G as well as the edge weights B, from the confounder-free precision matrix S.

Proposition 3.10 (Recovery of B from S, Proposition 3 and 4 in [5]). Let $( G , \mathbf { B } , \{ \sigma _ { i } ^ { 2 } \} _ { i = 1 } ^ { p } ) )$ be a SEM and let $\mathbf { S } = ( \mathbf { I } - \mathbf { B } ) ^ { \top } \mathbf { N } _ { X } ^ { - 1 } ( \mathbf { I } - \mathbf { B } )$ be the related precision matrix, where $\mathbf { N } _ { X } = \mathbf { D i a g } \big ( \{ \bar { \sigma } _ { i } ^ { 2 } \bar  \} _ { i = 1 } ^ { p } \big )$ . Under Assumption $3 . 9 , \ i$ is a terminal node in G $i f i \in \mathrm { a r g m i n } \mathbf { S } _ { i , i }$ . Moreover, if i is a terminal node in $G ,$ then $\begin{array} { r } { \mathbf { B } _ { i , \bullet } = - \frac { \mathbf { S } _ { i , \bullet } } { \mathbf { S } _ { i , i } } \ a n d \ \sigma _ { i } ^ { 2 } = 1 / \mathbf { S } _ { i , i } } \end{array}$

The following assumption was inspired by Assumption 2 in [5], and is essentially a stricter requirement than Assumption 3.9, that allows to handle the randomness of the finite-sample data.

Assumption 3.11 (Finite Sample Identifiability Condition). Let $( G , \mathbf { B } , \{ \sigma _ { i } ^ { 2 } \} _ { i = 1 } ^ { p } ) )$ be a SEM and let $\mathbf { S } = ( \mathbf { I } - \mathbf { B } ) ^ { \top } \mathbf { N } _ { X } ^ { - 1 } ( \mathbf { I } - \mathbf { B } )$ be the related precision matrix, where $\mathbf { N } _ { X } = \mathbf { D i a g } \big ( \{ \sigma _ { i } ^ { 2 } \} _ { i = 1 } ^ { p } \big )$ . Suppose Assumption 3.8 hold, and let $\mathbf { S } [ ( m , \tau ) ]$ denote the confounder-free precision matrix over the observed nodes in $V [ m , \tau ]$ . Then we assume that:

i) For all $( i , j ) \in V [ m , \tau ] \times V [ m , \tau ]$ , m $\in [ p ]$ , and $\tau \in \mathcal { T } _ { G }$ such that $\phi _ { G [ m , \tau ] } ( i ) = \alpha$ and $\phi _ { G [ m , \tau ] } ( j ) \neq \emptyset$

$$
\frac { 1 } { \sigma _ { i } ^ { 2 } } < \frac { 1 } { \sigma _ { j } ^ { 2 } } + \sum _ { l \in \phi _ { G [ m , \tau ] } ( j ) } \frac { { \bf B } _ { l , j } ^ { 2 } } { \sigma _ { l } ^ { 2 } } - \frac { 1 2 } { \kappa _ { \mathscr { L } } } \operatorname * { m a x } \Bigl \{ \lambda \sqrt { s } , \mu \sqrt { r } \Bigr \} ,
$$

$$
\begin{array} { r } { \mathrm { i i } ) \operatorname* { m i n } \big \{ | ( \mathbf { S } [ m , \tau ] ) _ { i , j } | : ( \mathbf { S } [ m , \tau ] ) _ { i , j } \neq 0 , \ ( i , j ) \in V [ m , \tau ] \times V [ m , \tau ] , \ m \in [ p ] , \ \tau \in \mathcal { T } _ { G } \big \} > \frac { 6 } { \kappa _ { \mathscr { L } } } \operatorname* { m a x } \Big \{ \lambda \sqrt { s } , \ \mu \sqrt { r } \Big \} , } \end{array}
$$

where s is the number of none-zero entries in $\mathbf { S ^ { * } } , r$ is the number of latent confounders, and $\begin{array} { r } { \kappa _ { \mathcal { L } } = \frac { M - 2 } { 2 ( M - 1 ) } \kappa _ { \mathrm { m i n } } ^ { * } } \end{array}$ with $M \ > \ 6$

## 3.4 Algorithm and Recovery Guarantees for SEMs over the Observed Variables

Here we present our algorithm and show its theoretical guarantees. First, we motivate an algorithm that recovers the true topological ordering of $G ,$ by detecting terminal nodes sequentially, one at a time. More specifically, at each iteration we solve the (undirected graphical model) optimization problem in $\mathrm { e q . } ( 5 )$ in order to recover S. By Proposition 3.10, the node i with smallest value in the diagonal $( \mathbf { S } _ { i , i } )$ is a terminal node. We then recover the weights $\mathbf { B } _ { i , \bullet }$ and noise variance $\sigma _ { i } ^ { 2 }$ from S, also by using Proposition 3.10. Algorithm 1 describes this process in detail.

Algorithm 1 Learning a SEM over the observed variables   
1: Input: Data matrix $\mathbf { X } \in \mathbb { R } ^ { n \times p } ,$ , regularization parameters $\lambda > 0$ and $\mu > 0$   
2: $\widehat { \mathbf { B } }  \mathbf { 0 } \in \mathbb { R } ^ { p \times p } ;$ vars $ [ p ] ; \widehat { \Sigma }  \mathbf { X } ^ { \top } \mathbf { X } / n$   
3: for $t = 1 , \dotsc , p - 1$ do   
4: Solve eq. (5) to obtain $\widehat { \mathbf { S } }$ and $\widehat { \mathbf { L } }$ using $\widehat { \boldsymbol { \Sigma } } ; i \gets \mathrm { a r g m i n } \widehat { \mathbf { S } } _ { i , i }$   
5: for each index $j$ and value v in vars do $\widehat { \mathbf { B } } _ { \mathrm { v a r s } [ i ] , v } \gets - \widehat { \mathbf { S } } _ { i , j } / \widehat { \mathbf { S } } _ { i , i }$   
$\begin{array} { r l } { { 6 } ; \ : } & { \ : \widehat { \mathbf { B } } _ { \mathrm { v a r s } [ i ] , \mathrm { v a r s } [ i ] } \ :  \ : 0 ; \ : \widehat { \sigma } _ { \mathrm { v a r s } [ i ] } ^ { 2 } = 1 / \widehat { \mathbf { S } } _ { i , i } ; \mathrm { v a r s } \ :  \ : \mathrm { v a r s } \ : \ : \backslash \{ i \} ; \mathbf { X } \ :  \ : \mathbf { X } _ { \bullet , - i } ; \ : \widehat { \Sigma }  \ : \mathbf { X } ^ { \top } \mathbf { X } / n } \end{array}$   
7: end for   
8: return $( \widehat { G } , \widehat { \mathbf { B } } , \{ \widehat { \sigma } _ { i } ^ { 2 } \} _ { i = 1 } ^ { p } )$ where ${ \widehat { G } } = ( [ p ] , { \widehat { E } } )$ and ${ \widehat E } = \operatorname { s u p p } ( { \widehat { \mathbf { B } } } )$

Next, we show the theoretical guarantees for Algorithm 1. We show that the value of every entry in $\widehat { \bf B }$ is close to those of $\mathbf { B } ^ { \ast }$ . In order to do so, we use the entrywise $\ell _ { \infty }$ norm in our deviation bound. Furthermore, we show that the recovered edges $\mathrm { s u p p } ( \widehat { \mathbf { B } } )$ are the same as the true edges $\mathrm { s u p p } ( \mathbf { B } ^ { * } )$

Theorem 3.12 (Deterministic recovery guarantee for B). Suppose Assumption 3.8 and Assumption 3.11 hold. Let $( G ^ { * } , \mathbf { B } ^ { * } , \{ \sigma _ { i } ^ { * 2 } \} )$ be the true SEM and let $\mathbf { S } ^ { \ast } = ( \mathbf { I } - \mathbf { B } ^ { \ast } ) ^ { \top } \mathbf { N } _ { X } ^ { \ast } ^ { - 1 } ( \mathbf { I } - \mathbf { B } ^ { \ast } )$ be the related precision matrix, where $\mathbf { N } _ { X } ^ { * } = \mathbf { D i a g } \big ( \{ \sigma _ { i } ^ { * 2 } \} _ { i = 1 } ^ { p } \big )$ . If the regularization parameters satisfy $\lambda \geq 2 \| \widehat { \Sigma } - \Sigma ^ { * } \| _ { \infty }$ and $\mu \geq 2 \| \widehat { \pmb { \Sigma } } - \pmb { \Sigma } ^ { * } \| _ { 2 }$ Algorithm 1 returns an estimation $\hat { \bf B }$ such that

$$
\| \widehat { \mathbf B } - \mathbf B ^ { * } \| _ { \infty } \leq c \left( 1 + \| \mathbf B ^ { * } \| _ { \infty } \right) m ,
$$

where $\begin{array} { r } { m = \frac { 6 } { \kappa _ { c } } \operatorname* { m a x } \Bigl \{ \lambda \sqrt { s } , \ \mu \sqrt { r } \Bigr \} , \ s } \end{array}$ is the number of none-zero entries in $\mathbf { S ^ { * } } , \ r$ is the number of latent confounders, $\begin{array} { r } { \kappa _ { \mathcal { L } } = \frac { M - 2 } { 2 ( M - 1 ) } \kappa _ { \operatorname* { m i n } } ^ { * } , M > 6 } \end{array}$ and $c \geq \operatorname* { m a x } _ { i \in [ p ] } \sigma _ { i } ^ { 2 } / \left( 1 - m \sigma _ { i } ^ { 2 } \right) > 0$ . Furthermore, Algorithm 1 correctly recovers the edges of the true SEM the true SEM $( G ^ { * } , \mathbf { B } ^ { * } , \{ \sigma _ { i } ^ { * 2 } \} )$ , i.e., $\mathrm { s u p p } ( \widehat { \mathbf { B } } ) = \mathrm { s u p p } ( \mathbf { B } ^ { * } )$

Theorem 3.12 provides a deterministic statement that does not consider the fact that the training data is random, which implies that the sample covariance matrix $\widehat { \pmb { \Sigma } }$ is also random. In what follows, we address this. The following theorem shows that if the number of samples n fulfills $n \gtrsim \frac { 1 } { \epsilon ^ { 2 } }$ max {s log p, rp}, then Algorithm 1 successfully recovers the weights in B as well as the edges in the graph $G .$

Theorem 3.13 (High-probability recovery guarantee for B). Suppose Assumption 3.8 and Assumption 3.11 hold. Let $( \mathbf { \bar { \it G } } ^ { * } , \mathbf { \bar { B } } ^ { * } , \{ \sigma _ { i } ^ { * 2 } \} )$ be the true SEM and let $\mathbf { S } ^ { \ast } = ( \mathbf { I } - \mathbf { B } ^ { \ast } ) ^ { \top } \mathbf { N } _ { X } ^ { \ast } { } ^ { - 1 } ( \mathbf { I } - \mathbf { B } ^ { \ast } )$ be the related precision matrix, where $\mathbf { N } _ { X } ^ { * } = \bar { \mathbf { D } } \mathbf { i a g } \big ( \{ \sigma _ { i } ^ { * 2 } \} _ { i = 1 } ^ { p } \big )$ . Choose some constants $C _ { 1 } \geq 3 / 2$ and $C _ { 2 } \geq 1$ . Assume the number of samples n satisfies

$$
\begin{array} { r } { n \ge \operatorname* { m a x } \left\{ 4 C _ { 1 } ^ { 2 } \log p , C _ { 2 } ^ { 2 } p , c ^ { 2 } ( 1 + \| \mathbf { B } ^ { * } \| _ { \infty } ) ^ { 2 } / \epsilon ^ { 2 } \operatorname* { m a x } \{ c _ { 1 } ^ { 2 } s \log p , c _ { 2 } ^ { 2 } r p \} \right\} , } \end{array}
$$

Set the regularization parameters as

$$
\lambda = 1 6 0 C _ { 1 } \bar { \sigma } ^ { * } \sqrt { \frac { \log p } { n } } \qquad a n d \qquad \mu = 1 6 C _ { 2 } \rho ^ { * } \sqrt { \frac { p } { n } } ,
$$

where σ¯<sup>∗</sup> = max<sub>i</sub> $\begin{array} { r } { \sum _ { i , i } ^ { * } , \rho ^ { * } = \| \Sigma ^ { * } \| _ { 2 } , } \end{array}$ s is the number of non-zero entries in $\mathbf { S } ^ { \ast } , p$ is the number of observed variables, r is the number of latent confounders, $c > 0$ is defined as in Theorem 3.12, $c _ { 1 } = \frac { 9 6 0 } { \kappa _ { L } } \bar { \sigma } ^ { * } C _ { 1 }$ 2 $c _ { 2 } = \frac { 9 6 } { \kappa _ { L } } \rho ^ { * } C _ { 2 } , \kappa _ { \mathscr { L } } = \frac { M - 2 } { 2 ( M - 1 ) } \kappa _ { \mathrm { m i n } } ^ { * }$ and $M \ > \ 6$ . Algorithm 1 returns an estimation $\widehat { \bf B }$ such that

$$
\| \widehat { \mathbf B } - \mathbf B ^ { * } \| _ { \infty } \le \epsilon ,
$$

with probability at least $1 - 4 p ^ { - 2 ( C _ { 1 } - 1 ) } - 2 \exp \bigl ( - \frac { C _ { 2 } ^ { 2 } p } { 2 } \bigr )$ . Furthermore, Algorithm 1 correctly recovers the edges of the true SEM $( G ^ { * } , \mathbf { B } ^ { * } , \{ \sigma _ { i } ^ { * 2 } \} ) , ~ i . e . , \mathrm { s u p p } ( \widehat { \mathbf { B } } ) = \mathrm { s u p p } ( \mathbf { B } ^ { * } )$ .

## 4 Experimental Validation

In this section, we validate our main theoretical result in Theorem 3.12. Our experiments consider noise $\varepsilon _ { X }$ and $\varepsilon _ { Z }$ from various distributions, such as Gaussian, uniform, a mixture distribution, as well as heavy-tailed distribution. We use a mixture distribution of a Gaussian (50%) and a uniform distribution (50%). For the heavy-tailed distribution, we use a t-Student distribution with 3 degrees of freedom.

We consider $r = 2$ latent confounders and diferent number of observed variables $p \in \{ 1 0 , 1 5 , 2 0 , 2 5 , 3 0 \}$ The topological ordering of the graph is created randomly. We create an edge set E randomly. Every observed variable can be afected by at most 2 observed variables and 1 latent confounder. Every latent confounder can be afected by at most 1 latent confounder. But a latent confounder can afect several observed variables. After we decide that edge $( i , j ) \in E$ , we set $\mathbf { B } _ { i , j } = - 1 \mathrm { ~ o r ~ } \mathbf { B } _ { i , j } = + 1$ with equal probability. Similarly, $\mathbf { C } _ { i , j } = - 1 / 4$ or $\mathbf { C } _ { i , j } = + 1 / 4$ is set with equal probability. Finall $\mathbf { \Delta } _ { \mathfrak { y } } , \mathbf { D } _ { i , j } = - 1 / 4 \ \mathrm { o r } \ \mathbf { D } _ { i , j } = + 1 / 4$ is also set with equal probability.

For every node i, the noise standard deviation $\sigma _ { i }$ was generated uniformly at random from [0.08, 0.12]. We then generate n samples where $n = C \log p$ for $C \in \{ 1 0 , 2 0 , 5 0 , 1 0 0 , 2 0 0 , 5 0 0 \}$

We run our Algorithm 1 with $\lambda = 0 . 0 1 { \sqrt { ( \log p ) / n } } , \mu = 0 . 0 0 5 { \sqrt { ( \log p ) / n } }$ . We used the method of $[ 5 ]$ as a baseline which does not take into account the latent confounders. We also compared to fast causal inference (FCI) [15, 3] which outputs partial ancestral graphs. Unfortunately, other methods do not apply for our problem.

We use the F1 score as the metric, computed between the predicted edges and the true edges. We repeat this produce 10 times and report the average F1 score as well as 95%-confidence error bars.

Figure 1 shows that our method performs better than the baseline method of [5] for various sample sizes and noise types. Moreover, when the number of samples is large enough, i.e., $n = C \log p$ for $C = 5 0 0$ , our method obtains a very good F1 score.

![](images/5a22ef68ddd5f0af45820258d7f193bfa295384e4254ba55bafc49e05980c1ee.jpg)  
(a) Gaussian distribution

![](images/d4a4ed07c565da117e5c3158a5bdaf22715f9fa223e0f4e53c3b771e90ebd26f.jpg)  
(b) Uniform distribution

![](images/eda30207b3f94d0d02af2fe76c98dae61ab794423b1158595ac71c8cfd7d4973.jpg)  
(c) Mixture

![](images/24384287669ad12769ce2847f0607a1d95f3c3da3bd907113c98c4875078a916.jpg)  
(d) Heavy-tailed distribution  
Figure 1: F1 score (predicted edges versus true edges) for various experimental regimes: (a) Gaussian noise, (b) uniformly-distributed noise, (c) noise following a mixture distribution between a Gaussian and a uniform distribution, and (d) t-Student distributed noise. Our method performs better than the baseline method of [5] for various sample sizes and noise types. Moreover, when the number of samples is large enough, i.e., $n = C \log p$ for $C = 5 0 0$ , our method obtains a very good F1 score.

## 5 Concluding Remarks

Our results open several interesting research questions. Although our model provides a theoretical guarantee for recovering the causal edges between observed variables, it does not pay much attention to the causal edges between latent confounders and the causal edges from latent confounders to observed variables. Future work could focus on recovery guarantees for those aspects. In addition, our model is based on a sparse plus low-rank structure. Another research direction is enable diferent structural conditions.

## References

[1] Peter J Bickel, Ya’acov Ritov, and Alexandre B Tsybakov. Simultaneous analysis of lasso and dantzig selector. Annals of Statistics, 2009.

[2] Venkat Chandrasekaran, Pablo A Parrilo, and Alan S Willsky. Latent variable graphical model selection via convex optimization. In Allerton Conference on Communication, Control, and Computing, pages

1610–1613, 2010.

[3] Diego Colombo, Marloes H. Maathuis, Markus Kalisch, and Thomas S. Richardson. Learning highdimensional directed acyclic graphs with latent and selection variables. The Annals of Statistics, 40(1), February 2012.

[4] Ruifei Cui, Perry Groot, Moritz Schauer, and Tom Heskes. Learning the causal structure of copula models with latent variables. In Uncertainty in Artificial Intelligence, pages 188–197, 2018.

[5] Asish Ghoshal and Jean Honorio. Learning linear structural equation models in polynomial time and sample complexity. In Artificial Intelligence and Statistics, pages 1466–1475, 2018.

[6] Biwei Huang, Charles Jia Han Low, Feng Xie, Clark Glymour, and Kun Zhang. Latent hierarchical causal structure discovery with rank constraints. Neural Information Processing Systems, 35:5549–5561, 2022.

[7] Sham Kakade, Ohad Shamir, Karthik Sindharan, and Ambuj Tewari. Learning exponential families in high-dimensions: Strong convexity and sparsity. In Artificial Intelligence and Statistics, pages 381–388, 2010.

[8] Zhaoshi Meng, Brian Eriksson, and Al Hero. Learning latent variable Gaussian graphical models. In International Conference on Machine Learning, pages 1269–1277, 2014.

[9] Sahand N Negahban, Pradeep Ravikumar, Martin J Wainwright, and Bin Yu. A unified framework for high-dimensional analysis of M-estimators with decomposable regularizers. Statistical Science, 2012.

[10] Ignavier Ng, Xinshuai Dong, Haoyue Dai, Biwei Huang, Peter Spirtes, and Kun Zhang. Score-based causal discovery of latent variable causal models. In International Conference on Machine Learning, 2024.

[11] Ignavier Ng, AmirEmad Ghassami, and Kun Zhang. On the role of sparsity and DAG constraints for learning linear DAGs. In Neural Information Processing Systems, volume 33, pages 17943–17954, 2020.

[12] Pradeep Ravikumar, Martin J Wainwright, Garvesh Raskutti, and Bin Yu. High-dimensional covariance estimation by minimizing ℓ<sub>1</sub>-penalized log-determinant divergence. Electronic Journal of Statistics, 2011.

[13] Ricardo Silva, Richard Scheine, Clark Glymour, and Peter Spirtes. Learning the structure of linear latent variable models. Journal of Machine Learning Research, 7(8):191–246, 2006.

[14] Ricardo Silva, Richard Scheines, Clark Glymour, and Peter Spirtes. Learning measurement models for unobserved variables. Uncertainty in Artificial Intelligence, 2003.

[15] Peter Spirtes, Clark N Glymour, and Richard Scheines. Causation, prediction, and search. MIT press, 2000.

[16] Feng Xie, Biwei Huang, Zhengming Chen, Yangbo He, Zhi Geng, and Kun Zhang. Identification of linear non-Gaussian latent hierarchical structure. In International Conference on Machine Learning, volume 162, pages 24370–24387, 2022.

[17] Eunho Yang and Pradeep K Ravikumar. Dirty statistical models. Neural Information Processing Systems, 26, 2013.

[18] Xun Zheng, Bryon Aragam, Pradeep K Ravikumar, and Eric Xing. DAGs with NO TEARS: Continuous optimization for structure learning. In Neural Information Processing Systems, volume 31, 2018.

## A Covariance and Precision Matrix for SEMs with Latent Confounders

From $\mathrm { e q . } ( 1 )$ we have that

$$
\left( \mathbf { I } - \left[ \mathbf { B } \quad \mathbf { C } \right] \right) \left[ \mathbf { x } \right] = \left[ \pmb { \varepsilon } _ { X } \right] ,
$$

and thus

$$
{ \bf \Big [ } \mathbf { x } \Big ] = \Big ( \mathbf { I } - \Big [ \mathbf { B } \mathbf { C } \Big ] \Big ) ^ { - 1 } \Big [ \pmb { \varepsilon } _ { X } \Big ] .
$$

By $\operatorname { e q . } ( 2 )$ we have that

$$
\begin{array} { r l } & { \Sigma _ { f } = \mathbb { E } \left[ \left[ \mathbf { x } \right] \left[ \mathbf { x } \right] ^ { \top } \right] } \\ & { \quad = \left( \mathbf { I } - \left[ \mathbf { B } _ { \mathbf { \Lambda } } \mathbf { C } \right] \right) ^ { - 1 } \mathbb { E } \left[ \left[ \varepsilon _ { Z } \right] \left[ \varepsilon _ { Z } \right] ^ { \top } \right] \left( \mathbf { I } - \left[ \mathbf { B } _ { \mathbf { \Lambda } } \mathbf { C } \right] \right) ^ { - \top } } \\ & { \quad = \left( \mathbf { I } - \left[ \mathbf { B } _ { \mathbf { \Lambda } } \mathbf { C } \right] \right) ^ { - 1 } \left[ \mathbf { N } _ { X } \quad 0 \right] \left( \mathbf { I } - \left[ \mathbf { B } _ { \mathbf { \Lambda } } \mathbf { C } \right] \right) ^ { - \top } , } \end{array}
$$

where $\mathbf { N } _ { X } = \mathbf { D i a g } \big ( \{ \sigma _ { i } ^ { 2 } \} _ { i = 1 } ^ { p } \big )$ and $\mathbf { N } _ { Z } = \mathbf { D i a g } \big ( \{ \sigma _ { i } ^ { 2 } \} _ { i = p + 1 } ^ { p + r } \big )$ . Now, by $\mathrm { { e q . } ( 3 ) }$ we have

$$
\begin{array} { r l } & { \mathbf { J } _ { f } = \pmb { \Sigma } _ { f } ^ { - 1 } } \\ & { \mathrm {  ~ \Lambda ~ } = \left( \mathbf { I } - \left[ \mathbf { B } \begin{array} { c c } { \mathbf { C } } \\ { \mathbf { 0 } } & { \mathbf { D } } \end{array} \right] \right) ^ { \top } \left[ \mathbf { N } _ { X } ^ { - 1 } \begin{array} { c c } { 0 } \\ { 0 } & { \mathbf { N } _ { Z } ^ { - 1 } } \end{array} \right] \left( \mathbf { I } - \left[ \mathbf { B } \begin{array} { c c } { \mathbf { B } } \\ { \mathbf { 0 } } & { \mathbf { D } } \end{array} \right] \right) } \\ & { \mathrm {  ~ \Lambda ~ } = \left[ \begin{array} { c c } { \mathbf { I } - \mathbf { B } } & { - \mathbf { C } } \\ { \mathbf { 0 } } & { \mathbf { I } - \mathbf { D } } \end{array} \right] ^ { \top } \left[ \mathbf { N } _ { X } ^ { - 1 } \begin{array} { c c } { 0 } \\ { 0 } & { \mathbf { N } _ { Z } ^ { - 1 } } \end{array} \right] \left[ \begin{array} { c c } { \mathbf { I } - \mathbf { B } } & { - \mathbf { C } } \\ { \mathbf { 0 } } & { \mathbf { I } - \mathbf { D } } \end{array} \right] } \\ & { \mathrm {  ~ \Lambda ~ } = \left[ \begin{array} { c c } { \left( \mathbf { I } - \mathbf { B } \right) ^ { \top } \mathbf { N } _ { X } ^ { - 1 } ( \mathbf { I } - \mathbf { B } ) } & { - \left( \mathbf { I } - \mathbf { B } \right) ^ { \top } \mathbf { N } _ { X } ^ { - 1 } \mathbf { C } } \\ { - \mathbf { C } ^ { \top } \mathbf { N } _ { X } ^ { - 1 } ( \mathbf { I } - \mathbf { B } ) } & { \mathbf { C } ^ { \top } \mathbf { N } _ { X } ^ { - 1 } \mathbf { C } + \left( \mathbf { I } - \mathbf { D } \right) ^ { \top } \mathbf { N } _ { Z } ^ { - 1 } ( \mathbf { I } - \mathbf { D } ) } \end{array} \right] } \end{array}
$$

From the above and since $\mathbf { J } _ { f } = \left[ \begin{array} { c c } { \mathbf { S } } & { \mathbf { J } _ { X , Z } } \\ { \mathbf { J } _ { X , Z } ^ { \top } } & { \mathbf { J } _ { Z , Z } } \end{array} \right]$ , we conclude that $\mathbf { S } = ( \mathbf { I } - \mathbf { B } ) ^ { \top } \mathbf { N } _ { X } ^ { - 1 } ( \mathbf { I } - \mathbf { B } )$ . Finally, we use $\mathrm { e q . } ( 1 )$ in [8] to obtain the expression for $\pmb { \Omega } = \pmb { \Sigma } ^ { - 1 } = \pmb { \mathrm { S } } + \mathbf { L }$ where $\mathbf { L } = - \mathbf { J } _ { X , Z } \mathbf { J } _ { Z , Z } ^ { - 1 } \mathbf { J } _ { X , Z } ^ { \top }$

## B Proofs

Here we present the proofs for the theorems and lemmas in our main text.

## B.1 Proof of Theorem 3.4

Proof. By Assumption 3.1 and Assumption 3.2 for $\Omega ^ { * }$ , by Proposition 3.3 we have that RSC and SI conditions hold. The proof of Theorem 3.4 refers to the proof of Theorem 1 and Corollary 4 in [17]. They pointed out that under RSC and SI condition, if $\lambda _ { \alpha } \geq 2 \mathcal { R } _ { \alpha } ^ { \ast } ( \nabla _ { \Omega _ { \alpha } } \mathcal { L } ( \Omega ^ { \ast } ; Z _ { 1 } ^ { n } ) )$ ), then

$$
\sum _ { \alpha \in I } \| \widehat { \mathbf { \Delta } } \widehat { \mathbf { \Delta } } _ { \alpha } \| \leq \frac { | I | } { \bar { \kappa } } \left( \frac { 3 } { 2 } \Phi + \sqrt { \bar { \kappa } \tau _ { \mathcal { L } } } \right)
$$

where $\Phi = \operatorname* { m a x } _ { \alpha \in I } \lambda _ { \alpha } \Psi _ { \alpha } ( { \overline { { \mathcal { M } } } } _ { \alpha } )$ and $\mathcal { R } ^ { * }$ is the dual norm of the norm. The dual norm of the entrywise $\ell _ { 1 }$ norm is the entrywise $\ell _ { \infty }$ norm, and the dual norm of the nuclear norm is the spectral norm. Note that

$\nabla \mathcal { L } ( \Omega ; \mathbf { X } ) = \widehat { \pmb { \Sigma } } - \Omega ^ { - 1 } = \widehat { \pmb { \Sigma } } - \pmb { \Sigma } ^ { * }$ . Thus, when we set $\lambda \geq 2 \| \widehat { \Sigma } - \Sigma ^ { * } \| _ { \infty }$ and $\mu \geq 2 \| \widehat { \pmb { \Sigma } } - \pmb { \Sigma } ^ { * } \| _ { 2 }$ , we obtain an error bound.

Next, we calculate the error bound. In our particular problem, $\begin{array} { r } { I = \{ S , L \} , \tau _ { \mathcal { L } } = 0 , \bar { \kappa } = \frac { \kappa _ { \mathcal { L } } } { 2 } } \end{array}$ , and thus we can get $\begin{array} { r l r } { \| \widehat { \boldsymbol { \Delta } } _ { S } \| + \| \widehat { \boldsymbol { \Delta } } _ { L } \| } & { { } \le } & { \frac { 6 } { \kappa _ { \mathcal { L } } } \operatorname* { m a x } \Bigl \{ \lambda \Psi _ { S } \bigl ( \overline { { \mathcal { M } } } _ { S } \bigr ) , \mu \Psi _ { L } \bigl ( \overline { { \mathcal { M } } } _ { L } \bigr ) \Bigr \} } \end{array}$ . Next, we get:

$$
\Psi _ { S } \big ( \overline { { \mathcal { M } } } _ { S } \big ) = \operatorname* { s u p } _ { \Delta \in \overline { { \mathcal { M } } } ( E ) \backslash \{ 0 \} } \frac { \| \Delta \| _ { 1 } } { \| \Delta \| _ { F } } \leq \sqrt { s } ,
$$

$$
\Psi _ { L } \big ( \overline { { \mathcal { M } } } _ { L } \big ) = \operatorname* { s u p } _ { \Delta \in \overline { { \mathcal { M } } } ( U ) \backslash \{ 0 \} } \frac { \| \Delta \| _ { * } } { \| \Delta \| _ { F } } \leq \sqrt { r } .
$$

Since we are using the Frobenius norm in RSC and SI, we can get $\| \widehat { \Delta } _ { S } \| _ { F } + \| \widehat { \Delta } _ { L } \| _ { F } \ \leq \ \frac { 6 } { \kappa _ { \mathcal { L } } } \operatorname* { m a x } \Big \{ \lambda \sqrt { s } , \mu \sqrt { r } \big \}$ Since the entrywise $\ell _ { \infty }$ norm is no greater than the Frobenius norm, we get

$$
\| \widehat { \pmb { \Delta } } _ { S } \| _ { \infty } + \| \widehat { \pmb { \Delta } } _ { L } \| _ { \infty } \leq \frac { 6 } { \kappa _ { \mathcal { L } } } \operatorname* { m a x } \biggr \{ \lambda \sqrt { s } , \mu \sqrt { r } \biggr \} .
$$

## B.2 Proof of Theorem 3.6

Proof. Theorem 3.4 is a deterministic theorem, and the error bound and the regularization parameters λ and µ depend on the randomness in the data. By Assumption 3.1 and Assumption 3.2 for $\Omega ^ { * }$ , by Proposition 3.3 we have that RSC and SI conditions hold. To prove Theorem 3.12, we only need to prove that $\lambda \geq 2 \| \widehat { \Sigma } - \Sigma ^ { * } \| _ { \infty }$ and $\mu \geq 2 \| \widehat { \pmb { \Sigma } } - \pmb { \Sigma } ^ { * } \| _ { 2 }$ holds with high probability.

According to Lemma 5 in [8] or Lemma 1 [12], we can get:

$$
P \Big \{ \Big \| \widehat { \boldsymbol { \Sigma } } - \boldsymbol { \Sigma } ^ { * } \Big \| _ { \infty } \leq \frac { 1 } { 2 } \lambda \Big \} \ \geq \ 1 - 4 p ^ { - 2 ( C _ { 1 } - 1 ) } .
$$

where $C _ { 1 } > 1$ and n fulfills $n \geq 4 C _ { 1 } ^ { 2 }$ log p. According to Lemma 6 in [8] or Lemma 5.4 in [2], we can get:

$$
P \Big \{ \Big \| \widehat { \boldsymbol { \Sigma } } - \boldsymbol { \Sigma } ^ { * } \Big \| _ { 2 } \leq \frac { 1 } { 2 } \mu \Big \} \geq 1 - 2 \exp \left( - \frac { C _ { 2 } ^ { 2 } p } { 2 } \right) .
$$

where $C _ { 2 } \geq 1$ and n should be satisfy $n \geq C _ { 2 } ^ { 2 } p$ . Thus, with probability at least $1 - 4 p ^ { - 2 ( C _ { 1 } - 1 ) } - 2 \exp \bigl ( - \frac { C _ { 2 } ^ { 2 } p } { 2 } \bigr )$ $\lambda \ge 2 \| \widehat { \pmb { \Sigma } } - { \pmb { \Sigma } } ^ { * } \| _ { \infty }$ and $\mu \geq 2 \| \widehat { \pmb { \Sigma } } - \pmb { \Sigma } ^ { * } \| _ { 2 }$ are both true. □

## B.3 Proof of Claim 3.7

Proof. Let $\Delta = \widehat { \Sigma } - \Sigma ^ { * }$ and note that $\Delta$ is symmetric since both $\widehat { \pmb { \Sigma } }$ and $\Sigma ^ { * }$ are symmetric. We need to show that $\lambda \ge 2 \| \pmb { \Delta } \| _ { \infty }$ and $\mu \geq 2 \Vert \Delta \Vert _ { 2 }$ hold for $\Delta ,$ then for all principal submatrices $\Delta _ { I , I }$ where $I \subset [ p ]$ , we have $\lambda \geq 2 \| \Delta _ { I , I } \| _ { \infty }$ and $\mu \geq 2 \Vert \Delta _ { I , I } \Vert _ { 2 }$ . The claim follows straightforwardly by properties of the entrywise $\ell _ { \infty }$ and spectral norms, for symmetric principal submatrices. □

## B.4 Proof of Theorem 3.12

Proof. According to Lemma 3.7, after removing the terminal node, the error bound between the update precision matrix and the true precision matrix is always established, which implies $\| \widehat { \mathbf { S } } - \mathbf { S } ^ { * } \| _ { \infty } ~ \leq$ $\frac { 6 } { \kappa _ { \mathscr { L } } }$ max $\left\{ \lambda { \sqrt { s } } , \mu { \sqrt { r } } \right\}$ . Let $\begin{array} { r } { m = \frac { 6 } { \kappa _ { \mathscr { L } } } \operatorname* { m a x } \biggl \{ \lambda \sqrt { s } , \mu \sqrt { r } \biggr \} , \varepsilon _ { i , j } = { \bf S } _ { i , j } ^ { * } - \widehat { \bf S } _ { i , j } } \end{array}$ , we get $| \varepsilon _ { i , j } | \leq m$ for all i and $j .$ . For

any $i \neq j$

$$
\begin{array} { r l } { \tilde { \mathbf { u } } _ { \perp , - } } & { - \frac { 1 } { 8 } \tilde { \mathbf { u } } _ { \perp , - } - \frac { 1 } { 8 } \tilde { \mathbf { u } } _ { \perp , - } } \\ & { - [ \frac { 3 } { 8 } \tilde { \mathbf { u } } _ { \perp , - } ^ { \prime } \frac { \mathbf { x } _ { \perp , + } } { 8 \tilde { \mathbf { u } } _ { \perp , - } } - \frac { 3 } { 8 } \tilde { \mathbf { u } } _ { \perp , - } ^ { \prime } ] } \\ & { - \frac { 1 } { 8 } \tilde { \mathbf { u } } _ { \perp , - } ^ { \prime } ( \frac { 3 } { 8 } \tilde { \mathbf { u } } _ { \perp , - } ^ { \prime } - \overline { { \mathbf { u } } } _ { \perp , - } ^ { \prime } - \overline { { \mathbf { u } } } _ { \perp , - } ^ { \prime } ) \tilde { \mathbf { u } } _ { \perp , - } } \\ & { - [ \frac { 3 } { 1 8 } \tilde { \mathbf { u } } _ { \perp , - } ^ { \prime } - \overline { { \mathbf { u } } } _ { \perp , - } ^ { \prime } ] \tilde { \mathbf { u } } _ { \perp , - } ^ { \prime } } \\ & { - [ \frac { 3 } { 1 8 } \tilde { \mathbf { u } } _ { \perp , - } ^ { \prime } - \overline { { \mathbf { u } } } _ { \perp , + } ^ { \prime } ] \tilde { \mathbf { u } } _ { \perp , - } ^ { \prime } } \\ & { - [ \frac { 5 } { 8 } \tilde { \mathbf { u } } _ { \perp , - } ^ { \prime } - \overline { { \mathbf { u } } } _ { \perp , - } ^ { \prime } ] } \\ & { - [ \frac { 3 } { 8 } \tilde { \mathbf { u } } _ { \perp , - } ^ { \prime } - \overline { { \mathbf { u } } } _ { \perp , - } ^ { \prime } ] } \\ &  - [ \frac { 3 } { 1 8 } \tilde { \mathbf { u } } _ { \perp , - } ^  \end{array}
$$

In the proof, we use $\mathbf { S } _ { i , i } ^ { * } = 1 / \sigma _ { i } ^ { 2 }$ and $\mathbf { S } _ { i , j } ^ { * } = - \mathbf { B } _ { i , j } ^ { * } / \sigma _ { i } ^ { 2 }$ . In Assumption 3.11, we use $1 / \sigma _ { i } ^ { 2 } > m \ge \varepsilon _ { i , i }$ . Then, every time a terminal node is removed, $\big | \widehat { \mathbf { B } } _ { i , j } - \mathbf { B } _ { i , j } ^ { * } \big | \leq c m \left( 1 + \| \mathbf { B } ^ { * } \| _ { \infty } \right)$ is correct for any $i \neq j$ . Therefore, we can get $\| \widehat { \mathbf B } - \mathbf B ^ { * } \| _ { \infty } \leq c m \left( 1 + \| \mathbf B ^ { * } \| _ { \infty } \right)$

Regarding the correct topological ordering, by Assumption 3.11, we know that for all terminal nodes i and non-terminal nodes $j \colon$

$$
\frac { 1 } { \sigma _ { i } ^ { 2 } } < \frac { 1 } { \sigma _ { j } ^ { 2 } } + \sum _ { l \in \phi _ { G [ m , \tau ] } ( j ) } \frac { { \bf B } _ { l , j } ^ { 2 } } { \sigma _ { l } ^ { 2 } } - \frac { 1 2 } { \kappa _ { \mathscr { L } } } \operatorname * { m a x } \Bigl \{ \lambda \sqrt { s } , \mu \sqrt { r } \Bigr \} .
$$

Thus, we can get:

$$
\frac { 1 } { \sigma _ { i } ^ { 2 } } + \frac { 6 } { \kappa _ { \mathscr { L } } } \operatorname* { m a x } \Bigl \{ \lambda \sqrt { s } , ~ \mu \sqrt { r } \Bigr \} \ < \ \frac { 1 } { \sigma _ { j } ^ { 2 } } \ + \ \sum _ { l \in \phi _ { G [ m , \tau ] } ( j ) } \frac { { \bf B } _ { l , j } ^ { 2 } } { \sigma _ { l } ^ { 2 } } \ - \ \frac { 6 } { \kappa _ { \mathscr L } } \operatorname* { m a x } \Bigl \{ \lambda \sqrt { s } , ~ \mu \sqrt { r } \Bigr \} .
$$

We know that $\begin{array} { r } { \| \widehat { \mathbf { S } } - \mathbf { S } \| _ { \infty } \leq \frac { 6 } { \kappa _ { \mathcal { L } } } \operatorname* { m a x } \Bigl \{ \lambda \sqrt { s } , \ \mu \sqrt { r } \Bigr \} } \end{array}$ and by Assumption 3.11(ii):

$$
\frac { 1 } { \sigma _ { i } ^ { 2 } } + \frac { 6 } { \kappa _ { \mathscr { L } } } \operatorname* { m a x } \Bigl \{ \lambda \sqrt { s } , ~ \mu \sqrt { r } \Bigr \} \geq \widehat { \mathbf { S } } _ { i , i } \quad \mathrm { a n d } \quad \frac { 1 } { \sigma _ { j } ^ { 2 } } - \frac { 6 } { \kappa _ { \mathscr { L } } } \operatorname* { m a x } \Bigl \{ \lambda \sqrt { s } , ~ \mu \sqrt { r } \Bigr \} \leq \widehat { \mathbf { S } } _ { j , j } .
$$

Therefore, we can get for all terminal nodes i and non-terminal nodes $j \colon \widehat { \mathbf { S } } _ { i , i } < \widehat { \mathbf { S } } _ { j , j }$ . Thus, we can still find the terminal node by finding the minimum value of the diagonal entries of Sb.

Regarding the correct edge recovery, by Assumption 3.11(ii), the support recovery of S is correct and since $\mathbf { B } _ { i , \bullet } = - \frac { \mathbf { B } _ { i , \bullet } } { \mathbf { B } _ { i , i } }$ , the support of B is also correct. □

## B.5 Proof of Theorem 3.13

Proof. Theorem 3.12 is a deterministic theorem, and the error bound and the regularization parameters depend on the randomness in the data. To prove Theorem 3.13, we first need to show that $\lambda \geq 2 \| \widehat { \Sigma } - \Sigma ^ { * } \| _ { \infty }$ and $\mu \geq 2 \| \widehat { \pmb { \Sigma } } - \pmb { \Sigma } ^ { * } \| _ { 2 }$ satisfy with high probability. The proof is similar to the proof of Theorem 3.6 and

thus, we will not repeat the probability proof here. Under Theorem 3.6, the number of samples n satisfies $n \geq \operatorname* { m a x } \{ 4 C _ { 1 } ^ { 2 } \log p , ~ C _ { 2 } ^ { 2 } p \}$

We can get the error bound is $\begin{array} { r } { m = \operatorname* { m a x } \Bigl \{ c _ { 1 } \sqrt { \frac { s \log p } { n } } , \ c _ { 2 } \sqrt { \frac { r p } { n } } \Bigr \} } \end{array}$ . In order to guarantee that $\| \widehat { \mathbf { B } } - \mathbf { B } ^ { * } \| _ { \infty } \leq \epsilon .$

$$
c m \left( 1 + \| \mathbf { B } ^ { * } \| _ { \infty } \right) \ \leq \ \epsilon
$$

Thus, we can get

$$
n \ge c ^ { 2 } ( 1 + \| \mathbf { B } ^ { * } \| _ { \infty } ) ^ { 2 } / \epsilon ^ { 2 } \operatorname* { m a x } \{ c _ { 1 } ^ { 2 } s \log p , ~ c _ { 2 } ^ { 2 } r p \}
$$

Combined with the requirements previously needed for n, we obtain

$$
n \ge \operatorname* { m a x } \Bigl \{ 4 C _ { 1 } ^ { 2 } \log p , ~ C _ { 2 } ^ { 2 } p , ~ c ^ { 2 } ( 1 + \| { \bf B } ^ { * } \| _ { \infty } ) ^ { 2 } / \epsilon ^ { 2 } \operatorname* { m a x } \{ c _ { 1 } ^ { 2 } s \log p , ~ c _ { 2 } ^ { 2 } r p \} \Bigr \}
$$

and we prove our claim.