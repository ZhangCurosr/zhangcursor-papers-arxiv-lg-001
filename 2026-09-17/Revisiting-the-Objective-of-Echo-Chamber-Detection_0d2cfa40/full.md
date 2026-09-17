# Revisiting the Objective of Echo Chamber Detection

Abylaikhan Bexeit CIS, The University of Melbourne abylaikhan.bexeit@student.unimelb.edu.au

Shanika Karunasekera CIS, The University of Melbourne karus@unimelb.edu.au

Kushani Perera CIS, The University of Melbourne kushani.perera@unimelb.edu.au

Jean Honorio CIS, The University of Melbourne jean.honorio@unimelb.edu.au

## Abstract

In this paper, we study the detection of an echo chamber in a social network, i.e., the identification of a set of nodes that agree on a topic, while disagreeing with the rest of nodes. We argue that this problem is diferent from other social network analysis problems such as community detection, and from other graph problems such as maximum graph cut and maximum clique. To the best of our knowledge, we are the first to formalize the objective function of echo chamber detection, by using the theory of Fourier transforms of set functions [14]. We propose scalable semidefinite relaxation, solved via an interior point method and sparse linear algebra. Experimentally, our algorithm recovers the ground truth echo chamber better than competing methods on small synthetic experiments. Our algorithm produces echo chambers with better network properties than competing methods on large real-world datasets. To independently validate our proposed objective function, we show that our algorithm finds echo chambers with more agreements with suspended users than competing methods on a small real-world dataset.

## 1 Introduction

Social networks are powerful instruments capable of influencing public opinion, both positively and negatively. Therefore, social network analysis with the intention of detecting their potential threats to the society is vital. Echo chambers are identified as a major cause for the widespread of misinformation, aggressive content and extremist ideologies in social networks. For instance, social media echo chambers have significantly contributed to the promotion of anti-vaccine and flat earth [7, 2] ideologies, resulting in serious negative repercussions to the society. Consequently, echo chamber detection plays an important role in social network analysis.

An echo chamber can be defined as a network of users with the same opinion regarding a given topic whose users frequently reinforce the content that supports their pre-existing opinions while discrediting and excluding dissenting opinions [2]. To detect echo chambers, existing work maps echo chamber detection to a community detection problem and apply community detection algorithms [7, 8, 6, 11]. However, direct application of existing community detection algorithms is problematic due to the specific properties of the echo chamber detection problem.

In echo chamber detection, interactions inside the echo chamber and between the echo chamber and outside should be considered (but not between nodes outside the echo chamber) which is not supported by the community detection problem as well as by classical theoretical computer science problems, such as maximum graph cut and maximum clique [9]. Echo chambers do not have to be cliques (i.e., where everybody interacts with one another). As shown in Figure 1 and discussed in more detail in Section 3.1, maximum graph cut fails to capture the interactions inside the echo chamber, while community detection unnecessarily captures the interactions between nodes outside the echo chamber. To the best of our knowledge, we are the first to formalize the objective function of echo chamber detection, from theoretical principles.

![](images/276f8c91c5f84dd5872e93e06b014727515753bc977c64655bc3f83e8b389114.jpg)

![](images/110e39b0255311d6d95c11e55d95b7c9419232f1702047689e705306004a831c.jpg)

![](images/816497fbaac021a3505eca275679b0c35f910ba8309db6bd89b14faa07b5ba48.jpg)

(a) Original graph  
(b) Echo chamber detection  
![](images/95d1c28dceb3d915bd141dd8b70d10253bdea5d5b2fcc53f9e2b143c7e040801.jpg)

(c) Maximum graph cut  
![](images/901336a90c3793189e59c4db5791a780098e85e1c6138c2e7f65ba2deeaed924.jpg)  
(d) Community detection  
(e) Maximum clique  
Figure 1: (a) Original graph with edges shown as black dashed. $\scriptstyle ( \log , \mathrm { c } , \mathrm { d } , \mathrm { e } )$ Participation of edges in the objective function of echo chamber detection in $\mathrm { e q . } ( 5 )$ , maximum graph cut in $\mathrm { e q . ( 3 ) }$ , community detection in $\mathrm { e q . } ( 4 )$ and maximum clique. Red solid edges are those which are added to the objective function. Blue solid edges are those which are subtracted from the objective function. Black dashed edges are not used in the objective function. In our example, $A = \{ 1 , 2 , 3 \}$ is a candidate set of nodes (e.g., echo chamber) used in the diferent objective functions. The set of nodes outside A is $V \setminus A = \{ 4 , 5 , 6 , 7 \}$ . In echo chamber detection, the edges inside A, and the edges between A and $V \setminus A$ are considered, but the edges inside $V \setminus A$ are not considered. This is not supported by the other objective functions.

Our contribution is four-fold. First, we derive a proper objective function for the problem of echo chamber detection, by using the theory of Fourier transforms of set functions. Our objective function encourages more agreements and fewer disagreements inside the echo chamber, while encouraging more disagreements and fewer agreements to outside the echo chamber. Second, we propose scalable semidefinite relaxation for our proposed problem, solved via an interior point method and sparse linear algebra. Third, our algorithm recovers the ground truth echo chamber better than competing methods on small synthetic experiments. Our algorithm produces echo chambers with better network properties than competing methods on large real-world datasets. Fourth, as an independent validation of our objective function, we show that our algorithm finds echo chambers with more agreements with suspended users than competing methods on a small real-world dataset.

## 2 Preliminaries

In this section, we introduce the main notations and concepts that are used throughout the paper. We denote sets by uppercase letters $( \mathrm { e } . \mathrm { g } . , A )$ . We use lowercase bold letters for vectors $\left( \mathrm { e . g . , { \bf v } } \right)$ and uppercase bold letters for matrices $( \mathrm { e . g . , \mathbf { M } } )$ , and subscripts of non-bold letters to denote respective entries of a vector or matrix $( \mathrm { e . g . , } v _ { i }$ and $m _ { i j } )$ . For a matrix M, its trace is denoted by tr(M) and its determinant is denoted by det(M). A vector of all ones (or all zeros) of size n is denoted as ${ \mathbf { 1 } } _ { n } \ \left( \operatorname { o r } \ \mathbf { 0 } _ { n } \right)$ . We denote by $\mathbf { e } _ { i }$ a vector which has zero entries everywhere except for entry i which contains 1. An identity matrix of size n is denoted as $\mathbf { I } _ { n } . \ \mathbf { M } \succ \mathbf { 0 } _ { n } \mathbf { 0 } _ { n } ^ { \top }$ (or $\mathbf { M } \succeq \mathbf { 0 } _ { n } \mathbf { 0 } _ { n } ^ { \top } )$ denotes matrix M is positive definite (or positive semidefinite). We denote the inner product of two matrices H and M as $\begin{array} { r } { \langle \mathbf { H } , \mathbf { M } \rangle = \sum _ { i j } h _ { i j } m _ { i j } } \end{array}$ . For a vector v, Diag(v) is the matrix containing v on its diagonal, and zero everywhere else. For a matrix M, diag(M) is the vector containing the diagonal entries of M.

The social network is represented as a graph of n nodes. The node set of the graph is $V = \{ 1 , \ldots , n \}$ Edges are represented by a signed adjacency weight matrix $\mathbf { W } \in \mathbb { R } ^ { n \times n }$ . We consider undirected graphs and thus, W is symmetric and $\mathrm { d i a g } ( \mathbf { W } ) = \mathbf { 0 } _ { n }$ . A positive edge weight $w _ { i j } > 0$ represents reinforcing interaction between node i and j. A negative edge weight $w _ { i j } < 0$ represents an antagonistic interaction between node i and j. A zero weight $w _ { i j } = 0$ represents no interaction (i.e., no edge) between node i and $j .$ . In this paper, we focus on detecting one echo chamber. We denote by r the size of the echo chamber to be identified.

Our analysis relies on the theory of Fourier transforms of set functions. The power set $2 ^ { V }$ represents the set of all subsets of V of zero elements, one element, two elements, and so on and so forth, up to n elements. That is, $2 ^ { V } = \{ A \mid A \subseteq V \} = \{ \emptyset , \{ 1 \} , \ldots , \{ n \} , \{ 1 , 2 \} , \ldots , \{ n - 1 , n \} , \ldots , \{ 1 , \ldots , n \} \}$ . For any arbitrary set function $f : 2 ^ { V } \to \mathbb { R }$ , its Fourier coeficients are defined as [14]:

$$
{ \widehat { f } } ( B ) = 2 ^ { - n } \sum _ { A \in 2 ^ { V } } f ( A ) ( - 1 ) ^ { | A \cap B | }\tag{1}
$$

for all $B \in 2 ^ { V }$ . Furthermore, we can reconstruct the original set function f from its Fourier coeficients:

$$
f ( A ) = \sum _ { B \in 2 ^ { V } } { \widehat { f } } ( B ) ( - 1 ) ^ { | A \cap B | } ,\tag{2}
$$

Next, we provide a couple of examples of Fourier coeficients from prior work. The maximum graph cut objective function has the form [9]:

$$
g ( A ) = \sum _ { i \in A , j \notin A } w _ { i j }\tag{3}
$$

and has Fourier coeficients [14]:

$$
\widehat { g } ( B ) = \left\{ \begin{array} { l l } { \frac { 1 } { 2 } \sum _ { i < j } w _ { i j } } & { \mathrm { i f ~ } B = \emptyset , } \\ { - \frac { 1 } { 2 } w _ { i j } } & { \mathrm { i f ~ } B = \{ i , j \} , } \\ { 0 } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

The community detection objective function has the form [1]:

$$
g ( A ) = \sum _ { i , j \in A , i < j } w _ { i j } + \sum _ { \substack { i , j \notin A , i < j } } w _ { i j } - \sum _ { \substack { i \in A , j \notin A } } w _ { i j }\tag{4}
$$

and has Fourier coeficients:

$$
\begin{array} { r } { \widehat { g } ( B ) = \left\{ \begin{array} { l l } { w _ { i j } } & { \mathrm { i f ~ } B = \{ i , j \} , } \\ { 0 } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}
$$

## 3 Main Results

In this section, we derive an objective function for echo chamber detection, by using the theory of Fourier transforms of set functions. We then devise scalable semidefinite relaxation, solved via an interior point method and sparse linear algebra.

## 3.1 Objective Function via Fourier Transform of Set Functions

In echo chamber detection, the goal is to find a set of nodes A such that there are many positive edges (and few negative edges) between nodes in A, while having very few positive edges (and many negative edges) between nodes in A and outside A. This definition encompasses the intuition of echo chambers. More formally, we define the set function $f : 2 ^ { V } \to \mathbb { R }$ to be:

$$
f ( A ) = \sum _ { i , j \in A , i < j } w _ { i j } - \sum _ { i \in A , j \not \in A } w _ { i j }\tag{5}
$$

Assume A is a candidate (echo chamber) set of nodes. Note that when compared to $\mathrm { e q . } ( 5 )$ , the maximum graph cut objective function in $\mathrm { { e q . } ( 3 ) }$ disregards the term $\textstyle \sum _ { i , j \in A , i < j } w _ { i j }$ and thus, maximum graph cut fails to capture the interactions inside the echo chamber A. Also note that when compared to $\mathrm { e q . ( 5 ) }$ , the community detection objective function in $\mathrm { e q . } ( 4 )$ has an extra unnecessary term $\textstyle \sum _ { i , j \not \in A , i < j } w _ { i j }$ which captures the interactions between nodes outside the echo chamber A. Furthermore, both ${ \dot { \operatorname { e q . } } } ( 3 )$ and $\mathrm { e q . } ( 4 )$ are symmetric, in the sense that $g ( A ) = g ( V \setminus A )$ , meaning that they return the same value for an echo chamber $( A )$ or for all the nodes outside the echo chamber $( V \setminus A )$ . The discussion above makes $\mathrm { { e q . } ( 3 ) }$ and $\mathrm { e q . } ( 4 )$ not suitable for echo chamber detection.

In order to devise a scalable algorithm, our initial goal is to figure out whether there exists an equivalent (and eficient) representation of the set function $f : 2 ^ { \bar { V } } \to$ R as a multivariate function ${ \overline { { f } } } : \{ - 1 , + 1 \} ^ { n } \to \mathbb { R }$ In order to do this, we denote by $\mathbf { x } \equiv \mathbf { x } ( A ) \in \{ - 1 , + 1 \} ^ { n }$ a binary n-dimensional vector that encodes set A as follows:

$$
x _ { i } ( A ) = { \left\{ \begin{array} { l l } { - 1 } & { { \mathrm { i f ~ } } i \in A , } \\ { + 1 } & { { \mathrm { i f ~ } } i \notin A . } \end{array} \right. }\tag{6}
$$

Our initial goal is then to find a multivariate function ${ \overline { { f } } } : \{ - 1 , + 1 \} ^ { n } \to \mathbb { R }$ such that $f ( A ) = { \overline { { f } } } ( \mathbf { x } ( A ) )$ for all $A \in 2 ^ { V }$

We highlight that such multivariate function might not necessarily have an eficient representation. As shown in Lemma 1, a multivariate function $f : \{ - 1 , + 1 \} ^ { n } \to \mathbb { R }$ could potentially depend on high-order monomials. Fortunately, we show in Lemma 2 that we only need terms for up to size 2 in the set function $( \mathrm { i . e . , } | B | \leq 2 )$ or equivalently, up to quadratic terms in the multivariate function.

First, we show that any set function has an equivalent multivariate function representation.

Lemma 1. Given an arbitrary set function $f : 2 ^ { V } \to \mathbb { R }$ , the equivalent multivariate function ${ \overline { { f } } } : \{ - 1 , + 1 \} ^ { n } \to$ R such that $f ( A ) = { \overline { { f } } } ( \mathbf { x } ( A ) )$ for all $A \in 2 ^ { V }$ , is given by:

$$
{ \overline { { f } } } ( \mathbf { x } ) = \sum _ { B \in 2 ^ { V } } { \widehat { f } } ( B ) \prod _ { i \in B } x _ { i }
$$

Proof. From $\mathrm { e q . } ( 2 )$ , we have: $\begin{array} { r } { f ( A ) = \sum _ { B \in 2 ^ { V } } \widehat { f } ( B ) ( - 1 ) ^ { | A \cap B | } } \end{array}$ . We can observe that in order obtain the desired result, we need to show that for all $A , B \in 2 ^ { V }$ , we have: $\textstyle \prod _ { i \in B } x _ { i } ( A ) = ( - 1 ) ^ { \left| A \cap B \right| }$ . By $\mathrm { e q . } ( 6 )$ and since $x _ { i } ( A ) \in \{ - 1 , + 1 \}$ , we have:

$$
\prod _ { i \in B } x _ { i } ( A ) = \left( \prod _ { i \in B , i \in A } x _ { i } ( A ) \right) \left( \prod _ { i \in B , i \notin A } x _ { i } ( A ) \right) = \left( \prod _ { i \in A \cap B } ( - 1 ) \right) \left( \prod _ { i \in B \setminus A } 1 \right) = ( - 1 ) ^ { | A \cap B | } ( 1 ) \left( \prod _ { i \in B \setminus A } 1 \right) = ( - 1 ) ^ { | A \cap B | } ( 1 ) \left( \prod _ { i \in B \setminus A } 1 \right) .
$$

which proves our claim.

Next, we derive the Fourier coeficients of our our echo-chamber detection set function $\mathrm { e q . ( 5 ) }$ . We highlight that terms of size greater than 2 in the set function $\left( \mathrm { i . e . , } | B | > 2 \right)$ have zero Fourier coeficient. Thus, from Lemma 1, we only need up to quadratic terms in the multivariate function.

Lemma 2. For our echo-chamber detection set function in $e q . ( 5 )$ , the Fourier coeficients are:

$$
\widehat { f } ( B ) = \left\{ \begin{array} { l l } { - \frac { 1 } { 4 } \sum _ { i < j } w _ { i j } } & { i f B = \emptyset , } \\ { - \frac { 1 } { 4 } \sum _ { j \neq i } w _ { i j } } & { i f B = \{ i \} , } \\ { \frac { 3 } { 4 } w _ { i j } } & { i f B = \{ i , j \} , } \\ { 0 } & { o t h e r w i s e . } \end{array} \right.
$$

(The proof is included in Appendix A.1.)

Proof sketch. We start with the Fourier coeficient definition in $\mathrm { e q . } ( 1 )$ and analyze the size of the set $A \cap B$ for the diferent cases of B and for all A. This reasoning leads to summands of the Fourier coeficient for which $| A \cap B |$ is even and thus $( - 1 ) ^ { | A \cap B | } = 1$ , and summands for which $| A \cap B |$ is odd and thus $( - 1 ) ^ { | A \cap B | } = - 1$ We then simplify the expressions to arrive to our claimed results. □

Given our previous results, we show multivariate function for echo chamber detection. This will later help us take advantage of the machinery of convex relaxations to construct a scalable algorithm.

Theorem 1. For our echo-chamber detection set function in $e q . ( 5 )$ , the equivalent multivariate function ${ \overline { { f } } } : \{ - 1 , + 1 \} ^ { n } \to \mathbb { R }$ such that $f ( A ) = { \overline { { f } } } ( \mathbf { x } ( A ) )$ for all $A \in 2 ^ { V }$ , is given by:

$$
\overline { { f } } ( { \bf x } ) = - \frac { 1 } { 4 } \left( \sum _ { i < j } w _ { i j } \right) - \frac { 1 } { 4 } \sum _ { i } \left( \sum _ { j \ne i } w _ { i j } \right) x _ { i } + \frac { 3 } { 4 } \sum _ { i , j } w _ { i j } x _ { i } x _ { j }
$$

Proof. By Lemmas 1 and 2.

## 3.2 Scalable Semidefinite Relaxation

Armed with an equivalent and eficient representation of the set function as a multivariate function, here we devise a scalable algorithm using a convex relaxation, interior-point methods and sparse linear algebra.

Echo chamber detection aims to find a set of size r what maximizes the objective function in $\mathrm { e q . } ( 5 )$ . Thus, the original (set) combinatorial optimization problem can be expressed as:

$$
{ \begin{array} { r l } & { { \mathrm { m a x i m i z e ~ } } f ( A ) } \\ & { { \mathrm { s u b j e c t ~ t o ~ } } A \in 2 ^ { V } , \quad | A | = r . } \end{array} }
$$

Given our Lemma 1, we can express the above as the following (multivariate) combinatorial problem:

$$
\begin{array} { r l } & { \mathrm { m a x i m i z e ~ } \overline { { f } } ( \mathbf { x } ) } \\ & { \mathrm { s u b j e c t ~ t o ~ } \mathbf { x } \in \{ - 1 , + 1 \} ^ { n } , \quad \mathbf { 1 } _ { n + 1 } ^ { \top } \mathbf { x } = n - 2 r . } \end{array}\tag{7}
$$

Note that since given $\mathrm { e q . } ( 6 )$ , having $| { \cal { A } } | = r$ is equivalent to having r minus ones and $n - r$ ones in vector $\mathbf { x } ,$ and thus the equivalent constraint is $\mathbf { 1 } _ { n + 1 } ^ { \top } \mathbf { x } = n - 2 r$

We now define the Fourier matrix $\widehat { \mathbf { F } } \in \mathbb { R } ^ { ( n + 1 ) \times ( n + 1 ) }$ as follows:

$$
\widehat { \mathbf { F } } = \left[ \begin{array} { c c c c c } { 0 } & { \widehat { f } ( \{ 1 , 2 \} ) } & { \cdots } & { \widehat { f } ( \{ 1 , n \} ) } & { \widehat { f } ( \{ 1 \} ) } \\ { \widehat { f } ( \{ 1 , 2 \} ) } & { 0 } & { \cdots } & { \widehat { f } ( \{ 2 , n \} ) } & { \widehat { f } ( \{ 2 \} ) } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } & { \vdots } \\ { \widehat { f } ( \{ 1 , n \} ) } & { \widehat { f } ( \{ 2 , n \} ) } & { \cdots } & { 0 } & { \widehat { f } ( \{ n \} ) } \\ { \widehat { f } ( \{ 1 \} ) } & { \widehat { f } ( \{ 2 \} ) } & { \cdots } & { \widehat { f } ( \{ n \} ) } & { \widehat { f } ( \emptyset ) } \end{array} \right] ,
$$

The above definition allows to write our result in Theorem 1 as follows:

$$
\overline { { f } } ( \mathbf { x } ) = \left[ \mathbf { x } \right] ^ { \top } \widehat { \mathbf { F } } \left[ \mathbf { 1 } \right]\tag{8}
$$

Instead of working directly with the vector x, we consider its lifted form $\mathbf { X } = \left[ \mathbf { X } \right] \left[ \mathbf { x } \right] ^ { \top }$ , where $\mathbf { X \in }$ $\mathbb { R } ^ { ( n + 1 ) \times ( n + 1 ) }$ is a positive semidefinite matrix of rank 1. In this representation, the objective becomes linear since ${ \bf \left[ { \bf x } _ { 1 } ^ { \bf { x } } \right] } ^ { \top } \hat { \bf F } \left[ { \bf x } _ { 1 } ^ { \bf { x } } \right] = \operatorname { t r } \left( \left[ { \bf x } _ { 1 } ^ { \bf { x } } \right] ^ { \top } \hat { \bf F } \left[ { \bf x } _ { 1 } ^ { \bf { x } } \right] \right) = \operatorname { t r } \left( \hat { \bf F } \left[ { \bf x } _ { 1 } ^ { \bf { x } } \right] \left[ { \bf x } _ { 1 } ^ { \bf { x } } \right] ^ { \top } \right) = \left. \hat { \bf F } , { \bf x } \right.$ . The combinatorial constraint $\mathbf { x } \in \{ - 1 , + 1 \} ^ { n }$ is equivalent to the constraint diag $\mathbf { \partial } ( \mathbf { X } ) = \mathbf { 1 } _ { n + 1 }$ <sub>1</sub> since $x _ { i } ^ { 2 } = 1$ for all i. Let $\mathbf { H } = \left[ \begin{array} { c c } { \mathbf { 0 } _ { n } \mathbf { 0 } _ { n } ^ { \top } } & { \mathbf { 1 } _ { n } } \\ { \mathbf { 1 } _ { n } ^ { \top } } & { 0 } \end{array} \right]$ The constraint $\mathbf { 1 } _ { n + 1 } ^ { \top } \mathbf { x } = n - 2 r$ is equivalent to $\langle \mathbf { H } , \mathbf { X } \rangle \ = \ 2 ( n - 2 r )$ since X can also be written as $\mathbf { X } = { \left\lceil \begin{array} { l l } { \mathbf { x x } ^ { \top } } & { \mathbf { x } } \\ { \mathbf { x } ^ { \top } } & { 1 } \end{array} \right\rceil }$

For $\mathbf { X } \in \mathbb { R } ^ { ( n + 1 ) \times ( n + 1 ) }$ , given our reasoning above and dropping the rank-1 constraint, we can relax the optimization problem $\mathrm { e q . } ( 7 )$ as the following semidefinite program:

$$
\mathrm { m a x i m i z e } \left. \widehat { \mathbf { F } } , \mathbf { X } \right.
$$

$$
\mathrm { s u b j e c t ~ t o ~ } \mathbf { X } \succeq \mathbf { 0 } _ { n + 1 } \mathbf { 0 } _ { n + 1 } ^ { \top } , \quad \mathrm { d i a g } ( \mathbf { X } ) = \mathbf { 1 } _ { n + 1 } , \quad \langle \mathbf { H } , \mathbf { X } \rangle = 2 ( n - 2 r ) ,\tag{9}
$$

We now devise a scalable solver for the above semidefinite relaxation. We follow an interior point method [4], which replaces the inequality constraints $( { \mathrm { i . e . , ~ } } \mathbf { X } \succeq \mathbf { 0 } _ { n + 1 } \mathbf { 0 } _ { n + 1 } ^ { \top }$ in our problem) with a logarithmic barrier function (i.e., log det(X) in our case). That is, for a logarithmic barrier factor $t > 0 ,$ , we have:

$$
\begin{array} { r l } & { \mathrm { m a x i m i z e } \left. \widehat { \mathbf { F } } , \mathbf { X } \right. + \frac { 1 } { t } \log \operatorname* { d e t } ( \mathbf { X } ) } \\ & { \mathrm { s u b j e c t ~ t o ~ } \operatorname { d i a g } ( \mathbf { X } ) = \mathbf { 1 } _ { n + 1 } , \quad \langle \mathbf { H } , \mathbf { X } \rangle = 2 ( n - 2 r ) , } \end{array}\tag{10}
$$

The logarithmic barrier log $\operatorname* { d e t } ( \mathbf { X } )$ ensures that X remains strictly within the interior of the semidefinite cone, i.e., $\mathbf { X } \succ \mathbf { 0 } _ { n } \mathbf { 0 } _ { n } ^ { \top }$ [4], thereby aiding convergence and preventing numerical degeneracy.

We then follow a dual gradient ascent approach that updates the dual variables $( { \mathrm { i . e . , ~ } } \nu \in \mathbb { R } ^ { n + 1 }$ associated with the constraint diag $( \mathbf { X } ) = \mathbf { 1 } _ { n + 1 }$ , and $\lambda \in \mathbb { R }$ associated with the constraint $\langle \mathbf { H } , \mathbf { X } \rangle = 2 ( n - 2 r ) )$ and can recover the primal variable X at any iteration. Algorithm 1 describes our method. (Full derivation is included in Appendix B.1.)

Algorithm 2 Scalable Interior Point Method for   
Solving $\operatorname { E q . } ( 9 )$   
Algorithm 1 Interior Point Method for Solving 1: Input: Sparse Fourier matrix $\widehat { \mathbf { F } } \in \mathbb { R } ^ { ( n + 1 ) \times ( n + 1 ) }$   
Eq.(9) echo chamber size $r ,$ logarithmic barrier factor   
1: Input: Fourier matrix $\widehat { \mathbf { F } } \in \mathbb { R } ^ { ( n + 1 ) \times ( n + 1 ) }$ , echo $t > 0 ,$ number of iterations $L ,$ step size $\eta > 0$   
chamber size r, logarithmic barrier factor $t > 0 ,$ 2: $\nu \gets \mathbf { 0 } _ { n + 1 }$   
3: $\lambda  0$   
number of iterations L, step size $\eta > 0 .$   
4: for $l = 1 , \ldots , L$ do   
2: $\nu \gets \mathbf { 0 } _ { n + 1 }$   
3: $\lambda  0$ 5: $\mathbf { M } \gets - t \widehat { \mathbf { F } } + \mathrm { D i a g } ( \pmb { \nu } ) + \lambda \mathbf { H }$   
4: for $l = 1 , \ldots , L$ do 6: for $i = 1 , \ldots , n + 1$ (easily parallelizable) do   
5: $\mathbf { M } \gets - t \widehat { \mathbf { F } } + \mathrm { D i a g } ( \pmb { \nu } ) + \lambda \mathbf { H }$ 7: y = solve $\left( \mathbf { M } , \mathbf { e } _ { i } \right)$   
6: $\mathbf { X }  \mathbf { M } ^ { - 1 }$ 8: $d _ { i } = \mathbf { y } ^ { \top } \mathbf { M } \mathbf { y }$   
7: 8: $\pmb { \nu }  \pmb { \nu } + \eta ( \mathrm { d i a g } ( \mathbf { X } ) - \mathbf { 1 } _ { n + 1 } )$ $\lambda  \lambda + \eta ( \langle \mathbf { H } , \mathbf { X } \rangle - 2 ( n - 2 r ) )$ 10: 9: end for $\mathbf { z } = \mathrm { s o l v e } \left( \mathbf { M } , \left[ \mathbf { 1 } _ { n } \right] \right)$   
10: 9: end for $\begin{array} { l } { { \displaystyle \left[ { \bf x } \right] } } \\ { { \displaystyle \left[ 1 \right] } } \end{array}$ ← the maximum eigenvector of X 12: 11: $\nu  \nu + \eta \big ( \mathbf { d } - \mathbf { \bar { 1 } } _ { n + 1 } \big )$ $\lambda  \lambda + \eta ( 2 \mathbf { y } ^ { \top } \mathbf M \mathbf { z } - 2 ( n - 2 r ) )$   
11: Output: $\mathbf { x } \in \mathbb { R } ^ { n }$ 13: end for   
14: <sup>x</sup><sub>1</sub> ← the minimum eigenvector of M   
15: Output: $\mathbf { x } \in \mathbb { R } ^ { n }$

When applying Algorithm 1, there is the need to: compute an inverse X of a large matrix M, compute the diagonal of X, compute the inner product ⟨H, X⟩, and compute the maximum eigenvector of X. Matrix Fb is sparse in practice, while H is sparse by definition, which makes M sparse as well. Note that even though M is sparse, its inverse X is usually dense, which we also observe in practice. This presents an issue for datasets with a large number of nodes n. The goal is then to devise an algorithm that does not store X at any point. To do this, we can take advantage of iterative linear equation system solvers, such as the Gauss-Seidel, Jaccobi, Richardson, successive over relaxation, minimal residual, among other methods.

Assume a black-box iterative linear equation system solver. That is, $\mathbf { y } = \operatorname { s o l v e } ( \mathbf { M } , \mathbf { v } )$ returns a solution for My = v or equivalently, $\mathbf { y } = \mathbf { M } ^ { - 1 } \mathbf { v }$ . Algorithm 2 describes our method. (Full derivation is included in Appendix B.2.) For a graph with E edges, the sparse matrix $\widehat { \mathbf F }$ has $O ( n + E )$ nonzero entries, while H has $O ( n )$ nonzero entries, thus, M has $O ( n + E )$ nonzero entries. Since iterative solvers are based on matrix-vector multiplications, their complexity is $O ( n + E )$ for a sparse M. Thus, Algorithm 2 has a computational complexity of $O ( L n ( n + E ) )$ and a space complexity of $O ( n + E )$ . Note that with parallelization, the computational complexity could be improved.

![](images/370ef7d9925bf55edf091371565a58ea17719cdb4f31e791aab79f3e2d8f73e2.jpg)

![](images/339667fba759c79e9720eac6293fb2fe3b7cdb3593bc9577faf7051165e3c48e.jpg)  
Figure 2: Left: F1 score between the recovered echo chamber and the ground truth echo chamber, versus various number of nodes (n), for density $( p = 9 0 \% )$ . Right: F1 score between the recovered echo chamber and the ground truth echo chamber, versus various densities (p), for number of nodes $( n = 2 0 0 )$ . Error bars at 95% confidence level over 30 repetitions. Our method outperforms others on recovering the ground truth echo chamber. Signed Louvain was not included due to their high computational demand.

Table 1: Number of nodes, positive edges and negative edges of real-world datasets used in our experiments.
<table><tr><td>Dataset</td><td>Nodes (n)</td><td>Positive edges</td><td>Negative edges</td></tr><tr><td>Russia-Ukraine</td><td>246</td><td>624</td><td>142</td></tr><tr><td>Facebook</td><td>4,039</td><td>176,468</td><td>0</td></tr><tr><td>Abortion</td><td>7,242</td><td>3,105,862</td><td>132,758</td></tr><tr><td>Obamacare</td><td>8,539</td><td>5,077,974</td><td>273,478</td></tr><tr><td>Twitter</td><td>81,306</td><td>2,684,606</td><td>0</td></tr><tr><td>Google</td><td>107,614</td><td>24,476,570</td><td>0</td></tr></table>

## 4 Experiments

In this section, we show that our method recovers the ground truth echo chamber better than competing methods on small synthetic experiments. We also show that our method produces echo chambers with better network properties than competing methods on large real-world datasets. Finally, as an independent validation, we show that our method finds echo chambers with more agreements with suspended users than competing methods on a small real-world dataset.

For all of our experiments and methods, we set the echo chamber size to be $r = \lceil { \sqrt { n } } \rceil$ , where n is the number of nodes. We use our method in Algorithm 2 with the minimal residual method as the sparse linear equation system solver.

Most of the current research applies existing community detection algorithms to detect echo chambers. We chose several popular community detection algorithms as comparison methods. Louvain [3], Signed Louvain [16], Girvan-Newman [10], and Leiden [15] are the algorithms based on modularity and community structure metrics. Infomap [13] and WalkTrap [12] are the algorithms based on walk metrics. They have been extensively used in community detection and echo chamber detection tasks [7, 8, 2, 6]. Signed Louvain is a Louvain algorithm’s [3] adaptation for signed graphs. For these comparison methods, we first choose the smallest community with at least r nodes, and then retain the r nodes with highest number of neighbors.

Synthetic Data. Here, we perform experiments on small synthetic datasets. We create graphs of n nodes, following an approach similar to that for Erdös–Rényi graphs, that produces graphs with properties resembling those of echo chambers. We use a global parameter $p \in ( 0 . 3 , 1 )$ that controls the edge density.

Table 2: Edge statistics (positive/negative edges inside the echo chamber, and positive/negative edges between the echo chamber and outside the echo chamber) and Cheeger constant, which measures the amount of connectivity inside the echo chamber. For edge statistics, we also include percentages relative to all edges. Bold values represent the best outcome. Our method (Ours) consistently produces the most positive edges and best connectivity inside the echo chamber compared to Infomap (IM), Leiden (Le), Louvain (Lo), Walktrap (WT), Girvan-Newman (GN) and signed Louvain (SLo). SLo was only included for the smallest dataset (Russia-Ukraine), and GN was only included the two smallest datasets (Russia-Ukraine and Facebook), due their high computational demand. WT was not included for the largest dataset (Google) due to its high memory demand.
<table><tr><td>Dataset</td><td>Method</td><td>Pos. edges inside</td><td>Neg. edges inside</td><td>Pos. edges between</td><td>Neg. edges between</td><td>Cheeger constant</td></tr><tr><td rowspan="6">Russia-Ukraine n =246</td><td>Ours</td><td>38 (5.0%)</td><td>0 (0.0%)</td><td>120 (15.7%)</td><td>17 (2.2%)</td><td>1.54</td></tr><tr><td>IM</td><td>23 (3.0%)</td><td>0 (0.0%)</td><td>57 (7.4%)</td><td>11 (1.4%)</td><td>0.63</td></tr><tr><td>Le</td><td>26 (3.4%)</td><td>0 (0.0%)</td><td>70 (9.1%)</td><td>4 (0.5%)</td><td>0.53</td></tr><tr><td>Lo</td><td>23 (3.0%)</td><td>0 (0.0%)</td><td>61 (8.0%)</td><td>3 (0.4%)</td><td>0.68</td></tr><tr><td>WT</td><td>31 (4.0%)</td><td>0 (0.0%)</td><td>134 (17.5%)</td><td>15 (2.0%)</td><td>0.79</td></tr><tr><td>GN</td><td>21 (2.7%)</td><td>0 (0.0%)</td><td>197 (25.7%)</td><td>6 (0.8%)</td><td>4.6e-16</td></tr><tr><td>Facebook</td><td>SLo</td><td>15 (2.0%)</td><td>0 (0.0%)</td><td>8 (1.0%)</td><td>1 (0.1%)</td><td>0.99</td></tr><tr><td rowspan="6">n =4,039</td><td>Ours</td><td>2008 (1.1%)</td><td>0 (0.0%)</td><td>9047 (5.1%)</td><td>0 (0.0%)</td><td>61.00</td></tr><tr><td>IM</td><td>1654 (0.9%)</td><td>0 (0.0%)</td><td>9085 (5.1%)</td><td>0 (0.0%)</td><td>1.28</td></tr><tr><td>Le</td><td>767 (0.4%)</td><td>0 (0.0%)</td><td>2176 (1.2%)</td><td>0 (0.0%)</td><td>1.23</td></tr><tr><td>Lo</td><td>742 (0.42%)</td><td>0 (0.0%)</td><td>2232 (1.26%)</td><td>0 (0.0%)</td><td>0.9</td></tr><tr><td>WT</td><td>1241 (0.7%)</td><td>0 (0.0%)</td><td>3686 (2.1%)</td><td>0 (0.0%)</td><td>5.66</td></tr><tr><td>GN</td><td>935 (0.5%)</td><td>0 (0.0%)</td><td>14004 (7.9%)</td><td>0 (0.0%)</td><td>5e-15</td></tr><tr><td rowspan="5">Abortion n =7,242</td><td>Ours</td><td>3529 (0.11%)</td><td>0 (0.0%)</td><td>109329 (3.37%)</td><td>170 (0.01%)</td><td>287.26</td></tr><tr><td>IM</td><td>3422 (0.11%)</td><td>0 (0.0%)</td><td>113948 (3.52%)</td><td>297 (0.01%)</td><td>169.95</td></tr><tr><td>Le</td><td>2557 (0.08%)</td><td>56 (0.002%)</td><td>41305 (1.28%)</td><td>5483 (0.17%)</td><td>4.1e-15</td></tr><tr><td>Lo</td><td>2675 (0.08%)</td><td>0 (0.0%)</td><td>40225 (1.24%)</td><td>5101 (0.16%)</td><td>42.4</td></tr><tr><td>WT</td><td>3422 (0.11%)</td><td>0 (0.0%)</td><td>113948 (3.52%)</td><td>297 (0.01%)</td><td>169.95</td></tr><tr><td rowspan="5">Obamacare n =8,539</td><td>Ours</td><td>4045 (0.08%)</td><td>0 (0.0%)</td><td>56754 (1.06%)</td><td>10483 (0.20%)</td><td>253.27</td></tr><tr><td>IM</td><td>4079 (0.08%)</td><td>0 (0.0%)</td><td>177317 (3.31%)</td><td>416 (0.01%)</td><td>271.02</td></tr><tr><td>Le</td><td>4069 (0.08%)</td><td>0 (0.0%)</td><td>176669 (3.30%)</td><td>406 (0.01%)</td><td>271.00</td></tr><tr><td>Lo</td><td>3412 (0.06%)</td><td>0 (0.0%)</td><td>128331 (2.40%)</td><td>1454 (0.03%)</td><td>68.4</td></tr><tr><td>WT</td><td>4089 (0.08%)</td><td>0 (0.0%)</td><td>175424 (3.28%)</td><td>383 (0.01%)</td><td>289.40</td></tr><tr><td rowspan="5">Twitter n =81,306</td><td>Ours</td><td>17864 (0.67%)</td><td>0 (0.0%)</td><td>65237 (2.43%)</td><td>0 (0.0%)</td><td>41.28</td></tr><tr><td>IM</td><td>10018 (0.37%)</td><td>0 (0.0%)</td><td>121624 (4.53%)</td><td>0 (0.0%)</td><td>3.00</td></tr><tr><td>Le</td><td>6167 (0.23%)</td><td>0 (0.0%)</td><td>77701 (2.89%)</td><td>0 (0.0%)</td><td>3.50</td></tr><tr><td>Lo</td><td>11025 (0.41%)</td><td>0 (0.0%)</td><td>105505 (3.93%)</td><td>0 (0.0%)</td><td>10.7</td></tr><tr><td>WT</td><td>5739 (0.21%)</td><td>0 (0.0%)</td><td>114174 (4.25%)</td><td>0 (0.0%)</td><td>3.30</td></tr><tr><td rowspan="4">Google n =107,614</td><td>Ours</td><td>47941 (0.20%)</td><td>0 (0.0%)</td><td>1606500 (6.56%)</td><td>0 (0.0%)</td><td>208.70</td></tr><tr><td>IM</td><td>28312 (0.12%)</td><td>0 (0.0%)</td><td>1491175 (6.09%)</td><td>0 (0.0%)</td><td>65.85</td></tr><tr><td>Le</td><td>29088 (0.12%)</td><td>0 (0.0%)</td><td>1510019 (6.17%)</td><td>0 (0.0%)</td><td>70.60</td></tr><tr><td>Lo</td><td>28888 (0.12%)</td><td>0 (0.0%)</td><td>1509087 (6.16%)</td><td>0 (0.0%)</td><td>65.8</td></tr></table>

(Full details are provided in Appendix C.) By using eq.(6) we equivalently defined the ground truth vector $\mathbf { x } ^ { * } = \mathbf { x } ( A ^ { * } ) \in \{ - 1 , + 1 \} ^ { n }$ . For all methods, we compute the F1 score between the recovered vector $\mathbf { x } \in \mathbb { R } ^ { n }$ and the ground truth vector $\mathbf { x } ^ { * }$ . That is, we convert x to a sign vector. Note that $\mathbf { x } ^ { * }$ is already a sign vector. We then compute precision and recall and therefore the F1 score. For our method in Algorithm 2, we used logarithmic barrier factor t = 0.08, L = 100 iterations and step size $\eta = 0 . 1$

Table 3: Analysis of interactions with suspended users through an unpaired t-test in the Russia-Ukraine dataset. ∆SR is the diference between the mean of SuspendedReplies of echo chamber members and the mean of SuspendedReplies of non-echo chamber members. ∆SA follows the same procedure for SuspendedAgrees. We also report their corresponding p-values. Our method (Ours) shows the largest and most statistically significant positive diference as compared to Infomap (IM), Leiden (Le), Louvain (Lo), Walktrap (WT), Girvan-Newman (GN) and signed Louvain (SLo).
<table><tr><td>Method</td><td>∆SR</td><td>p-value</td><td>∆SA</td><td>p-value</td></tr><tr><td>Ours</td><td>2.7054</td><td>0.0012</td><td>2.6484</td><td>0.0009</td></tr><tr><td>IM</td><td>0.7668</td><td>0.2797</td><td>0.5092</td><td>0.4158</td></tr><tr><td>Le</td><td>1.2348</td><td>0.0591</td><td>1.3114</td><td>0.0381</td></tr><tr><td>Lo</td><td>1.6359</td><td>0.0435</td><td>1.6457</td><td>0.0351</td></tr><tr><td>WT</td><td>2.2375</td><td>0.007</td><td>2.1804</td><td>0.0054</td></tr><tr><td>GN</td><td>1.8364</td><td>0.0323</td><td>1.8462</td><td>0.0241</td></tr><tr><td>SLo</td><td>-0.7707</td><td>0.0913</td><td>-0.6272</td><td>0.1463</td></tr></table>

We run 30 repetitions of the above procedure, and report the mean and standard error bars. As observed in Figures 2, our method recovers the ground truth echo chamber better than competing methods on small synthetic experiments.

Real-World Data. In what follows, we perform experiments on real-world datasets. We consider several datasets of various sizes, but favor mostly large datasets. Table 1 shows the statistics regarding number of nodes, positive edges and negative edges. For all methods, we evaluate the recovered echo chamber with various meaningful metrics. We compute the number of positive and negative edges between nodes in the echo chamber. We also compute the number of positive negative edges between nodes in the echo chamber and nodes outside the echo chamber. To measure the amount of overall/global connectivity inside the echo chamber, we use the Cheeger constant of the subgraph formed by the nodes in the echo chamber. While computing this quantity is computationally intractable, one can approximate it by computing the second minimum eigenvalue of the Laplacian of the subgraph [5]. For our method in Algorithm 2, we used logarithmic barrier factor $t = 0 . 0 0 0 1$ , L = 10 iterations and step size $\eta = 0 . 1$

As observed in Table 2, our method produces echo chambers with most positive edges and best connectivity than those of competing methods on large real-world datasets.

Independent Validation. As discussed in Section 1, echo chambers are extensively associated with misinformation, aggressive content and extremist ideas. In social networks, the users who perform such activities tend to be suspended by the administrators. Therefore, it is reasonable to assume that echo chamber members have many interactions and agreements with the suspended users compared to the other users in the network. Fortunately, the Russia-Ukraine dataset contains ground-truth information of which users were suspended. We held this information and was not provided to any of the tested algortithms, including ours. To assess whether an echo chamber detected by any method would likely be so in the real world, we use two measures, the number of replies a given user has with a suspended user (SuspendedReplies) and the number of agree replies a given user has with a suspended user (SuspendedAgrees). Since we have two sets of users (echo chamber and non-echo chamber members), we perform an unpaired t-test. In Table 3, we report the diference between the mean of SuspendedReplies of echo chamber members and the mean of SuspendedReplies of non-echo chamber members. We follow the same procedure for SuspendedAgrees. We also report their corresponding p-values.

As shown in Table 3, our method shows the largest positive SuspendReplies diference. The small p-value indicates that this diference is statistically significant. This means that the echo chamber members detected by our method tend to interact with the suspended users more than the echo chamber members detected by the other algorithms. Similarly, our method shows the largest positive SuspendedAgrees diference with a small p-value. This shows that the echo chamber members detected by our method tend to agree with the suspended users’ opinions more than the echo chamber members detected by the other algorithms. The above results suggests that, compared to the baselines, the echo chambers detected by our method might more likely be actual echo chambers in the real world.

## 5 Concluding Remarks

Our contributions open several questions for future work. While we could use our method recursively for identifying one echo chamber at a time, it would be interesting to have a direct generalization that identifies several echo chambers. While we focused on pairwise interactions between two nodes which led to consider edges and graphs, it would be interesting to analyze the case where several nodes interact together which would lead to hyperedges and hypergraphs.

## References

[1] Emmanuel Abbe. Community detection and stochastic block models. Foundations and Trends in Communications and Information Theory, 14(1-2):1–162, 2018.

[2] Faisal Alatawi, Lu Cheng, Anique Tahir, Mansooreh Karami, Bohan Jiang, Tyler Black, and Huan Liu. A survey on echo chambers on social media: Description, detection and mitigation. arXiv preprint arXiv:2112.05084, 2021.

[3] Vincent D Blondel, Jean-Loup Guillaume, Renaud Lambiotte, and Etienne Lefebvre. Fast unfolding of communities in large networks. Journal of statistical mechanics: theory and experiment, 2008(10):P10008, 2008.

[4] S. Boyd and L. Vandenberghe. Convex Optimization. Cambridge University Press, 2006.

[5] Jef Cheeger. A lower bound for the smallest eigenvalue of the laplacian. In Proceedings of the Princeton conference in honor of Professor S. Bochner, 1969.

[6] Matteo Cinelli, Gianmarco De Francisci Morales, Alessandro Galeazzi, Walter Quattrociocchi, and Michele Starnini. The echo chamber efect on social media. Proceedings of the national academy of sciences, 118(9):e2023301118, 2021.

[7] Alessandro Cossard, Gianmarco De Francisci Morales, Kyriaki Kalimeri, Yelena Mejova, Daniela Paolotti, and Michele Starnini. Falling into the echo chamber: the italian vaccination debate on twitter. In International AAAI conference on web and social media, volume 14, pages 130–140, 2020.

[8] Michela Del Vicario, Fabiana Zollo, Guido Caldarelli, Antonio Scala, and Walter Quattrociocchi. Mapping social dynamics on facebook: The brexit debate. Social Networks, 50:6–16, 2017.

[9] Michael R. Garey and David S. Johnson. Computers and intractability: a guide to the theory of NP-completeness. A series of books in the mathematical sciences. Freeman, 2009.

[10] Mark EJ Newman. Modularity and community structure in networks. Proceedings of the national academy of sciences, 103(23):8577–8582, 2006.

[11] Kushani Perera and Shanika Karunasekera. Quantifying opinion rejection: A method to detect social media echo chambers. In Pacific-Asia Conference on Knowledge Discovery and Data Mining, pages 57–69. Springer, 2024.

[12] Pascal Pons and Matthieu Latapy. Computing communities in large networks using random walks. In International symposium on computer and information sciences, pages 284–293, 2005.

[13] Martin Rosvall, Daniel Axelsson, and Carl T Bergstrom. The map equation. The European Physical Journal Special Topics, 178(1):13–23, 2009.

[14] P. Stobbe and A. Krause. Learning Fourier sparse set functions. International Conference on Artificial Intelligence and Statistics, pages 1125–1133, 2012.

[15] Vincent A Traag, Ludo Waltman, and Nees Jan Van Eck. From louvain to leiden: guaranteeing well-connected communities. Scientific reports, 9(1):1–12, 2019.

[16] Chengyi Xia, Yongping Luo, Li Wang, and Hui-Jia Li. A fast community detection algorithm based on reconstructing signed networks. IEEE Systems Journal, 16(1):614–625, 2021.

## A Detailed Proofs for Section 3.1

In this section, we provide detailed proofs for the lemmas in the main text.

## A.1 Proof of Lemma 2

Proof. For brevity, we will shorten our notation and write A instead of $A \in 2 ^ { V }$ in the sums and sets. Note that in all the proofs, we have $i \neq j$ in general, while in some cases we also have $i < j$ . We analyze four diferent cases.

Case 1. First, we consider the case $B = \varnothing$ . Note that in this case, we have $| A \cap B | = 0$ for all A. Therefore, by eq.(1) we have:

$$
\begin{array} { r l } { \hat { H } ( R ) = 2 ^ { - 1 } \sum _ { j = 1 } ^ { N } f ( \hat { \textbf { z } } _ { j } | \hat { \textbf { z } } _ { j } \leq \hat { \textbf { z } } _ { j } ) } \\ & { - 2 ^ { - 1 } \sum _ { j = 1 } ^ { N } \hat { \textbf { z } } _ { j } \leq \hat { \textbf { z } } _ { j } } \\ & { - 2 ^ { - 1 } \sum _ { j = 1 } ^ { N } ( \displaystyle \sum _ { j = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \exp { - \sum _ { j = 1 } ^ { N } \exp { - \sum _ { j = 1 } ^ { N } \hat { \textbf { z } } _ { j } } } ) } \\ & { - 2 ^ { - 1 } ( \displaystyle \sum _ { j = 1 } ^ { N } \displaystyle \sum _ { j = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \displaystyle \sum _ { j = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \exp { - \sum _ { j = 1 } ^ { N } \hat { \textbf { z } } _ { j } } ) } \\ & { - 2 ^ { - 1 } ( \displaystyle \sum _ { j = 1 } ^ { N } \exp { - \sum _ { j = 1 } ^ { N } \hat { \textbf { z } } _ { j } } ) - \displaystyle \sum _ { j = 1 } ^ { N } \exp { - \sum _ { j = 1 } ^ { N } \hat { \textbf { z } } _ { j } } \exp { - \hat { \textbf { z } } _ { j } } ) } \\ & { - 2 ^ { - 1 } ( \displaystyle \sum _ { j = 1 } ^ { N } \hat { \textbf { z } } _ { j } \leq \hat { \textbf { z } } _ { j } \leq \hat { \textbf { z } } _ { j } ) + \displaystyle \sum _ { j = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \hat { \textbf { z } } _ { j } \leq \hat { \textbf { z } } _ { j } \leq \hat { \textbf { z } } _ { j } ) \Bigg ) } \\ &  = 2 ^ { - 1 } ( \displaystyle \sum _ { j = 1 } ^ { N } g _ { 1 } ) \exp  - \hat  \ \end{array}
$$

where the second to the last line follows since $| \{ A \mid i , j \in A \} | = 2 ^ { n - 2 }$ since we fix elements i and $j$ in A and we can choose any subset of $n - 2$ other elements. Similarly, $\vert \{ A \vert i \in A , j \not \in A \} \vert = 2 ^ { n - 2 }$ since we fix element i in A and $j$ not in A and we can choose any subset of $n - 2$ other elements.

Case 2. Second, we consider the case $B = \{ k \}$ . Note that in this case, we have $| A \cap B | = 0 { \mathrm { ~ i f ~ } } k \notin A$ , and $| A \cap B | = 1 { \mathrm { ~ i f ~ } } k \in A$ . Therefore, by eq.(1) we have:

$$
\begin{array} { l } { \displaystyle \widehat { f } ( B ) = 2 ^ { - n } \sum _ { A } f ( A ) ( - 1 ) ^ { | A \cap B | } } \\ { = 2 ^ { - n } \left( \displaystyle \sum _ { A | k \notin A } f ( A ) - \displaystyle \sum _ { A | k \in A } f ( A ) \right) } \\ { = 2 ^ { - n } \left( \displaystyle \sum _ { A | k \notin A } \left( \displaystyle \sum _ { i , j \in A , i < j } w _ { i j } - \displaystyle \sum _ { i \in A , j \notin A } w _ { i j } \right) - \displaystyle \sum _ { A | k \in A } \left( \displaystyle \sum _ { i , j \in A , i < j } w _ { i j } - \displaystyle \sum _ { i \in A , j \notin A } w _ { i j } \right) \right) } \end{array}
$$

$$
\begin{array} { r l } & { = 2 ^ { - n } \left( \displaystyle \sum _ { i < j } \displaystyle \sum _ { A | i , j \in A , k \not \in A } w _ { i j } - \displaystyle \sum _ { i \neq j } \displaystyle \sum _ { A | i \in A , j , k \not \in A } w _ { i j } - \displaystyle \sum _ { i < j } \displaystyle \sum _ { A | i , j , k \in A } w _ { i j } + \sum _ { i \neq j } \sum _ { A | i , k \in A , j \not \in A } w _ { i j } \right) } \\ & { = 2 ^ { - n } \left( \displaystyle \sum _ { i < j } \left( \displaystyle \sum _ { A | i , j \in A , k \not \in A } w _ { i j } - \displaystyle \sum _ { A | i , j , k \in A } w _ { i j } \right) + \displaystyle \sum _ { i \neq j } \left( \displaystyle \sum _ { A | i , k \in A , j \not \in A } w _ { i j } - \displaystyle \sum _ { A | i \in A , j , k \not \in A } w _ { i j } \right) \right) } \end{array}
$$

Next, we analyze the two inner terms in the above expression. Regarding the first inner term:

$$
\begin{array} { l } { { \displaystyle F _ { 1 } = \sum _ { A | i , j \in A , k \notin A } w _ { i j } - \sum _ { A | i , j , k \in A , k \notin \{ i , j \} } w _ { i j } - \sum _ { A | i , j , k \in A , k \in \{ i , j \} } w _ { i j } } } \\ { { \displaystyle \quad = w _ { i j } | \big \{ A \mid i , j \in A , k \notin A \} | - w _ { i j } | \big \{ A \mid i , j , k \in A , k \notin \{ i , j \} \big \} | - \sum _ { A | i , k \in A } w _ { i k } - \sum _ { A | k , j \in A } w _ { k j } } } \\ { { \displaystyle \quad = - \sum _ { A | i , k \in A } w _ { i k } - \sum _ { A | k , j \in A } w _ { k j } } } \\ { { \displaystyle \quad = - w _ { i k } | \big \{ A \mid i , k \in A \big \} | - w _ { k j } | \big \{ A \mid k , j \in A \} \big \} } } \\ { { \displaystyle \quad = - ( w _ { i k } + w _ { k j } ) ^ { 2 n - 2 } } } \end{array}
$$

where the third to the last line follows since $| \{ A \mid i , j \in A , k \not \in A \} | = | \{ A \mid i , j , k \in A , k \not \in \{ i , j \} \} |$ . The last line follows since $| \{ A \mid i , k \in A \} | = | \{ A \mid k , j \in A \} | = 2 ^ { n - 2 }$ as argued in Case 1. Regarding the second inner term:

$$
\begin{array} { r l } & { F _ { 2 } = \underset { A \mid i , k \in A , j \notin A } { \sum } w _ { i j } - \underset { A \mid i \in A , j , k \notin A } { \sum } w _ { i j } } \\ & { \quad = w _ { i j } \left| \{ A \mid i , k \in A , j \notin A \} \right| - w _ { i j } \left| \{ A \mid i \in A , j , k \notin A \} \right| } \\ & { \quad = 0 } \end{array}
$$

where the last line follows since $| \{ A \mid i , k \in A , j \not \in A \} | = | \{ A \mid i \in A , j , k \not \in A \} |$ . Thus, we have:

$$
\begin{array} { l } { { \displaystyle { \widehat f ( B ) = 2 ^ { - n } ( F _ { 1 } + F _ { 2 } ) } } } \\ { { \displaystyle ~ = 2 ^ { - n } \sum _ { i < j } \left( - ( w _ { i k } + w _ { k j } ) 2 ^ { n - 2 } \right) } } \\ { { \displaystyle ~ = - \frac { 1 } { 4 } \sum _ { j \neq k } w _ { k j } } } \end{array}
$$

Case 3. Third, we consider the case $B = \left\{ \boldsymbol { k } , \boldsymbol { l } \right\}$ . Note that in this case, we have $| A \cap B | = 0 { \mathrm { ~ i f ~ } } k , l \notin A$ $| A \cap B | = 1 { \mathrm { ~ i f ~ } } k \in A , l \not \in A$ , and $| A \cap B | = 2 { \mathrm { ~ i f ~ } } k , l \in A$ . Therefore, by $\mathrm { e q . } ( 1 )$ we have:

$$
\begin{array} { r l } & { \hat { f } ( B ) = 2 ^ { - n } \displaystyle \sum _ { A } f ( A ) ( - 1 ) ^ { | A \cap B | } } \\ & { \quad = 2 ^ { - n } \left( \displaystyle \sum _ { A \mid k , l \notin A \mathrm { \& } \kappa , l \in A } f ( A ) - \displaystyle \sum _ { A \mid \kappa \in A , l \notin A } f ( A ) \right) } \\ & { \quad = 2 ^ { - n } \left( \displaystyle \sum _ { A \mid k , l \notin A \mathrm { \& } \kappa , l \in A } \left( \displaystyle \sum _ { i , j \in A , i \in A } w _ { i j } - \displaystyle \sum _ { i \in A , i \notin A } w _ { i j } \right) - \displaystyle \sum _ { A \mid k \in A , l \notin A } \left( \displaystyle \sum _ { i , j \in A , i \in A } w _ { i j } - \displaystyle \sum _ { i \in A , j \notin A } w _ { i j } \right) \right) } \\ & { \quad = 2 ^ { - n } \left( \displaystyle \sum _ { i \in \mathcal { A } } \displaystyle \sum _ { \alpha : j \in A , k \in A } w _ { i j } - \displaystyle \sum _ { i \in \mathcal { A } } \displaystyle \sum _ { \alpha : j \in A , j \in A } w _ { i j } - \displaystyle \sum _ { i \in \mathcal { A } } \displaystyle \sum _ { \lambda \in \mathcal { A } } w _ { i j } \right) w _ { i j } + \displaystyle \sum _ { i \in \mathcal { A } } w _ { i j } \sum _ { \lambda \in A , i \in A , j \in A } w _ { i j } } \\ & { \quad = 2 ^ { - n } \left( \displaystyle \sum _ { i \in \mathcal { A } } \displaystyle \sum _ { \alpha : j \in A , k \in A } w _ { i j } \cdot \displaystyle \sum _ { i \in \mathcal { A } } \displaystyle \sum _ { \alpha : j \in A , j \in A } w _ { i j } - \displaystyle \sum _ { i \in \mathcal { A } } \displaystyle \sum _ { \lambda \in \mathcal { A } } w _ { i j , k \in A , l \notin A } w _ { i j } + \displaystyle \sum _ { i \in \mathcal { A } } \displaystyle \sum _ { \lambda \in \mathcal { A } } w _ { i j } \cdot \displaystyle \sum _ { i \in \mathcal { A } } w _ { i j } \right) w _ { i j } } \end{array}
$$

$$
= 2 ^ { - n } \left( \sum _ { i < j } \left( \underbrace { \sum _ { A | i , j \in A , k , l \notin A } w _ { i j } - \sum _ { A | i , j , k \in A , l \notin A } w _ { i j } } _ { \mathrm { o r } ~ i , j , k , l \in A } \right) + \underbrace { \sum _ { i \neq j } \left( \sum _ { A | i , k \in A , j , l \notin A } w _ { i j } - \sum _ { A | i \in A , j , k , l \notin A } w _ { i j } \right) } _ { F _ { 1 } } \right)
$$

Next, we analyze the two inner terms in the above expression. Regarding the first inner term:

$$
\begin{array} { r l } { F _ { 1 } = } & { \displaystyle \sum _ { \lambda \atop 4 | y _ { 2 } \leq \lambda _ { 1 } \leq \mu \leq \lambda } w _ { \lambda j } + \displaystyle \sum _ { \lambda \atop 4 | y _ { 2 } \leq \lambda _ { 1 } \leq \mu \leq \lambda } w _ { \lambda j } - \displaystyle \sum _ { \lambda \atop 4 | y _ { 2 } \leq \lambda _ { 1 } \leq \mu \leq \lambda } w _ { \lambda j } } \\ { = } & { \displaystyle \sum _ { \lambda \atop 4 | y _ { 2 } \leq \lambda _ { 1 } \leq \mu \leq \lambda } w _ { \lambda j } + \displaystyle \sum _ { \lambda \atop 4 | y _ { 2 } \leq \lambda _ { 1 } \leq \mu \leq \lambda ( \lambda _ { 1 } \leq \mu ) } w _ { \lambda j } + \displaystyle \sum _ { \lambda \atop 4 | y _ { 2 } \leq \lambda _ { 1 } \leq \mu \leq \lambda } w _ { \lambda j } - \displaystyle \sum _ { \lambda \atop 4 | y _ { 2 } \leq \lambda _ { 1 } \leq \mu \leq \lambda , \mu \leq \lambda } w _ { \lambda j } } \\ { = } & { \displaystyle \sum _ { \lambda \atop 4 | y _ { 2 } \leq \lambda _ { 1 } \leq \mu \leq \lambda } w _ { \lambda j } + \displaystyle \sum _ { \lambda \atop 4 | y _ { 2 } \leq \lambda _ { 1 } \leq \mu \leq \lambda ( \lambda _ { 1 } \leq \mu ) } w _ { \lambda j } + \displaystyle \sum _ { \lambda \neq y _ { 2 } \leq \lambda } w _ { \lambda \lambda } - \displaystyle \sum _ { \lambda \neq y \leq \lambda } w _ { \lambda j } w _ { \lambda j } } \\ { = } & { w _ { \lambda j } [ \displaystyle 4 \lambda \mid j , j \leq \mathscr { A } _ { \lambda } , \mathscr { A } \neq \lambda ] + w _ { \lambda j } [ \displaystyle 4 \lambda \mid \lambda \mid j , j \leq \lambda ] } \\ &  + \displaystyle \sum _ { \lambda \neq \lambda } w _ { \lambda j } - w _ { \lambda j } [ \displaystyle 4 \lambda \mid j , j \leq \lambda , \mathscr { A } \neq \lambda ] [ \displaystyle 4 \lambda \mid j , j \geq \lambda ] [ \displaystyle 6 \lambda , \mathscr { A } ] [ \displaystyle 6 \lambda , \mathscr { A } ] [ \ \end{array}
$$

where the third to the last line follows since $\left| \{ A \mid i , j \in A , k , l \notin A \} \right| + \left| \{ A \mid i , j , k , l \in A , ( i , j ) \neq ( k , l ) \} \right| =$ $| \{ A \mid i , j , k \in A , l \not \in A \}$ |. The last line follows since $| \{ A \mid k , l \in A \} | = 2 ^ { n - 2 }$ as argued in Case 1. Regarding the second inner term:

$$
\begin{array} { l } { { F _ { 2 } = \displaystyle \sum _ { A \downarrow \downarrow \downarrow \times A \leq i \leq n } w _ { i \cdot i \cdot i } - \sum _ { A \uparrow \downarrow \downarrow \times A \leq i \leq n } w _ { i \cdot i \cdot i } - \sum _ { A \uparrow \downarrow \downarrow \times A \leq i \leq n } w _ { i \cdot i \cdot i } } } \\   = \displaystyle \sum _ { A \downarrow \downarrow \times A \leq i \leq n } \sum _ { A \uparrow \downarrow \downarrow \cdot A \uparrow \downarrow \cdot A \uparrow \downarrow \downarrow \downarrow } w _ { i \cdot i \cdot i } + \sum _ { A \downarrow \downarrow \times A \cdot A \downarrow \downarrow \downarrow \downarrow \downarrow \downarrow \downarrow \downarrow \downarrow \downarrow } w _  i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \cdot i \end{array}
$$

where the third to the last line follows from the fact that $| \{ A \mid i , k \in A , j , l \notin A , ( i , j ) \notin \{ ( k , l ) , ( l , k ) \} \} | =$ $| \{ A \mid i \in A , j , k , l \not \in A \} | + | \{ A \mid i , k , l \in A , j \not \in A \}$ |. The last line follows since $| \{ A \mid k \in A , l \not \in A \} | = 2 ^ { n - 2 }$ as argued in Case 1. Thus, we have:

$$
\begin{array} { l } { { \displaystyle { \widehat f ( B ) = 2 ^ { - n } ( F _ { 1 } + F _ { 2 } ) } } } \\ { { \displaystyle ~ = 2 ^ { - n } \sum _ { i < j } \left( w _ { k l } 2 ^ { n - 2 } + ( w _ { k l } + w _ { l k } ) 2 ^ { n - 2 } \right) } } \\ { { \displaystyle ~ = \frac { 3 } { 4 } w _ { k l } } } \end{array}
$$

Case 4. Finally, we consider the case $| B | > 2$ . Note that we can write:

$$
\begin{array} { l } { { \displaystyle \widehat { f } ( B ) = 2 ^ { - n } \sum _ { A } f ( A ) ( - 1 ) ^ { | A \cap B | } } } \\ { { \displaystyle = 2 ^ { - n } \sum _ { A } \left( \sum _ { i , j \in A , i < j } w _ { i j } - \sum _ { i \in A , j \notin A } w _ { i j } \right) ( - 1 ) ^ { | A \cap B | } } } \\ { { \displaystyle = 2 ^ { - n } \left( \sum _ { A } \sum _ { i , j \in A , i < j } w _ { i j } ( - 1 ) ^ { | A \cap B | } - \sum _ { A } \sum _ { i \in A , j \notin A } w _ { i j } ( - 1 ) ^ { | A \cap B | } \right) } } \end{array}
$$

We now argue that $F _ { 1 } = 0$ . Assume we take a set A, and elements $i , j \in A$ . Now choose an element $k \in B \backslash \{ i , j \}$ and $k \not \in A$ . Define $A ^ { \prime } = A \cup \{ k \}$ . Note that the parity of $| A \cap B |$ is diferent from the parity of $| A ^ { \prime } \cap B |$ by construction, i.e., one quantity is odd and the other quantity is even. That is, $( - 1 ) ^ { | \mathsf { \bar { A } } \cap B | } = - ( - 1 ) ^ { | A ^ { \prime } \cap B | }$ Therefore, $F _ { 1 }$ contains two summands $w _ { i j } ( - 1 ) ^ { | A \cap B | } + \bar { w } _ { i j } ( - 1 \bar { ) } ^ { | A ^ { \prime } \cap B | } = 0$ . By following this argument for all summands of $F _ { 1 }$ , we have that $F _ { 1 } = 0$

We then argue that $F _ { 2 } = 0$ . Assume we take a set A, and elements $i \in A , j \notin A$ . Now choose an element $k \in B \setminus \{ i , j \}$ and $k \in A$ . Define $A ^ { \prime } = A \setminus \{ k \}$ . Note that the parity of $| A \cap B |$ is diferent from the parity of $| A ^ { \prime } \cap B |$ by construction, i.e., one quantity is odd and the other quantity is even. That is, $( - 1 ) ^ { \dot { | } A \cap B | } = \dot { - } ( - 1 ) ^ { | A ^ { \prime } \cap \dot { B } | }$ . Therefore, $F _ { 2 }$ contains two summands $w _ { i j } ( - 1 ) ^ { | A \cap B | } + w _ { i j } ( - 1 ) ^ { | A ^ { \prime } \cap B | } = 0$ . By following this argument for all summands of $F _ { 2 }$ , we have that $F _ { 2 } = 0$ . Therefore ${ \widehat { f } } ( B ) = 2 ^ { - n } ( F _ { 1 } + F _ { 2 } ) = 0$ .

## B Additional Details for Section 3.2

In this section, we provide additional details for the algorithm in the main text.

## B.1 Derivation of our Interior Point Method in Algorithm 2

Since $t > 0 , { \mathrm { e q . } } ( 1 0 )$ can be equivalently written as:

$$
\begin{array} { r l } & { \mathrm { m i n i m i z e ~ } - t \left. \widehat { \mathbf { F } } , \mathbf { X } \right. - \log \operatorname* { d e t } ( \mathbf { X } ) } \\ & { \mathrm { s u b j e c t ~ t o ~ } \operatorname { d i a g } ( \mathbf { X } ) = \mathbf { 1 } _ { n + 1 } , } \\ & { \quad \quad \quad \langle \mathbf { H } , \mathbf { X } \rangle = 2 ( n - 2 r ) , } \end{array}
$$

Let $\nu \in \mathbb { R } ^ { n + 1 }$ be the dual variable associated with the constraint $\mathrm { d i a g } ( \mathbf { X } ) = \mathbf { 1 } _ { n + 1 }$ . Let $\lambda \in \mathbb { R }$ be the dual variable associated with the constraint $\langle \mathbf { H } , \mathbf { X } \rangle = 2 ( n - 2 r )$ . We now define the Lagrangian associated with this optimization problem:

$$
\begin{array} { r l } & { L ( \mathbf { X } , \nu , \lambda ) = - t \left. \hat { \mathbf { F } } , \mathbf { X } \right. - \log \operatorname* { d e t } ( \mathbf { X } ) + \left. \nu , \operatorname { d i a g } ( \mathbf { X } ) - \mathbf { 1 } _ { n + 1 } \right. + \lambda ( \langle \mathbf { H } , \mathbf { X } \rangle - 2 ( n - 2 r ) ) } \\ & { \qquad = - t \left. \hat { \mathbf { F } } , \mathbf { X } \right. - \log \operatorname* { d e t } ( \mathbf { X } ) + \left. \mathrm { D i a g } ( \nu ) , \mathbf { X } \right. - \mathbf { 1 } _ { n + 1 } ^ { \top } \nu + \lambda ( \langle \mathbf { H } , \mathbf { X } \rangle - 2 ( n - 2 r ) ) } \\ & { \qquad = \left. - t \hat { \mathbf { F } } + \mathrm { D i a g } ( \nu ) + \lambda \mathbf { H } , \mathbf { X } \right. - \log \operatorname* { d e t } ( \mathbf { X } ) - \mathbf { 1 } _ { n + 1 } ^ { \top } \nu - 2 ( n - 2 r ) \lambda } \end{array}
$$

Taking the gradient with respect to X and equating to zero, leads to:

$$
\frac { \partial L } { \partial \mathbf { X } } = - t \widehat { \mathbf { F } } + \mathrm { D i a g } ( \pmb { \nu } ) + \lambda \mathbf { H } - \mathbf { X } ^ { - 1 } = \mathbf { 0 } _ { n + 1 } \mathbf { 0 } _ { n + 1 } ^ { \top }
$$

Solving the above for X leads to its optimal value with respect to the Lagrangian L, which is:

$$
\mathbf { X } ^ { \mathrm { o p t } } = \left( - t \widehat { \mathbf { F } } + \operatorname { D i a g } ( \pmb { \nu } ) + \lambda \mathbf { H } \right) ^ { - 1 }
$$

We can now compute the objective function of the dual problem as follows:

$$
\begin{array} { r l } & { g ( \nu , \lambda ) } \\ & { = \frac { \operatorname* { m i n } L \left( \mathbf { X } , \nu , \lambda \right) } { \mathbf { x } } } \\ & { = L ( \mathbf { x } ^ { \mathrm { o p } } , \nu , \lambda ) } \\ & { = \left. - t \hat { \mathbf { F } } + \mathrm { D i a g } ( \nu ) + \lambda \mathbf { H } , \mathbf { X } ^ { \mathrm { o u t } } \right. - \log \operatorname* { d e t } ( \mathbf { X } ^ { \mathrm { o p t } } ) - \mathbf { I } _ { n + 1 } ^ { \top } \nu - 2 ( n - 2 r ) \lambda } \\ & { = \left. - t \hat { \mathbf { F } } + \mathrm { D i a g } ( \nu ) + \lambda \mathbf { H } , \left( - t \hat { \mathbf { F } } + \mathrm { D i a g } ( \nu ) + \lambda \mathbf { H } \right) ^ { - 1 } \right. - \log \operatorname* { d e t } \left( \left( - t \hat { \mathbf { F } } + \mathrm { D i a g } ( \nu ) + \lambda \mathbf { H } \right) ^ { - 1 } \right) } \\ & { \quad \quad - \mathbf { I } _ { n + 1 } ^ { \top } \nu - 2 ( n - 2 r ) \lambda } \\ & { = \mathrm { t r } ( \mathbf { I } _ { n + 1 } ) + \log \operatorname* { d e t } \left( - t \hat { \mathbf { F } } + \mathrm { D i a g } ( \nu ) + \lambda \mathbf { H } \right) - \mathbf { I } _ { n + 1 } ^ { \top } \nu - 2 ( n - 2 r ) \lambda } \\ & { = ( n + 1 ) + \log \operatorname* { d e t } \left( - t \hat { \mathbf { F } } + \mathrm { D i a g } ( \nu ) + \lambda \mathbf { H } \right) - \mathbf { I } _ { n + 1 } ^ { \top } \nu - 2 ( n - 2 r ) \lambda } \end{array}
$$

We now proceed with a gradient ascent approach in order to maximize the objective function of the dual problem [4]. For this, we compute the gradient with respect to $\nu ,$ which is:

$$
\begin{array} { c } { \displaystyle \frac { \partial g } { \partial \nu _ { i } } = \left. \mathbf { e } _ { i } \mathbf { e } _ { i } ^ { \top } , \left( - t \widehat { \mathbf { F } } + \mathrm { D i a g } ( \pmb { \nu } ) + \lambda \mathbf { H } \right) ^ { - 1 } \right. - 1 } \\ { = \left. \mathbf { e } _ { i } \mathbf { e } _ { i } ^ { \top } , \mathbf { X } ^ { \mathrm { o p t } } \right. - 1 } \\ { = \mathbf { X } _ { i i } ^ { \mathrm { o p t } } - 1 } \end{array}
$$

Therefore, we have:

$$
\frac { \partial g } { \partial \pmb { \nu } } = \mathrm { d i a g } ( \mathbf { X } ^ { \mathrm { o p t } } ) - \mathbf { 1 } _ { n + 1 }
$$

Then, we compute the derivative with respect to $\lambda ,$ which is:

$$
\begin{array} { l } { \displaystyle { \frac { \partial g } { \partial \lambda } = \left. { \bf H } , \left( - t \widehat { \bf F } + \mathrm { D i a g } ( \pmb { \nu } ) + \lambda { \bf H } \right) ^ { - 1 } \right. - 2 ( n - 2 r ) } } \\ { \displaystyle { \phantom { \frac { \partial g } { \partial \lambda } } } } \\ { \displaystyle { \phantom { \frac { \partial g } { \partial \lambda } = } = \left. { \bf H } , { \bf X } ^ { \mathrm { o p t } } \right. - 2 ( n - 2 r ) } } \end{array}
$$

For a step size $\eta > 0$ , the final dual gradient ascent approach, is given by the update rules: $\textstyle \nu  \nu + \eta { \frac { \partial g } { \partial \nu } }$ and $\begin{array} { r } { \lambda  \lambda + \eta \frac { \partial g } { \partial \lambda } } \end{array}$

## B.2 Derivation of our Scalable Interior Point Method in Algorithm 2

Recall that the logarithmic barrier log det(X) ensures that X remains strictly within the interior of the semidefinite cone, i.e., $\mathbf { X } \succ \mathbf { 0 } _ { n } \mathbf { 0 } _ { n } ^ { \top } \ \left[ 4 \right]$ . Since $\mathbf { X } = \mathbf { M } ^ { - 1 }$ this also implies that $\mathbf { M } \succ \mathbf { 0 } _ { n } \mathbf { 0 } _ { n } ^ { \top }$

First, we reason about the diagonal of X since diag(X) is involved in the gradient ascent update of $\nu .$ Let ${ \bf d } = \mathrm { d i a g } ( { \bf X } ) , \mathrm { i . e . , } d _ { i } = x _ { i i }$ . Note that $x _ { i i }$ can be written as:

$$
\begin{array} { r l } & { d _ { i } = x _ { i i } } \\ & { \quad = { { \bf e } _ { i } ^ { \top } } { \bf X } { \bf e } _ { i } } \\ & { \quad = { { \bf e } _ { i } ^ { \top } } { \bf M } ^ { - 1 } { \bf e } _ { i } } \\ & { \quad = { { \bf e } _ { i } ^ { \top } } { \bf M } ^ { - 1 } { \bf M } { \bf M } ^ { - 1 } { \bf e } _ { i } } \\ & { \quad = { { \bf y } ^ { \top } } { \bf M } { \bf y } } \end{array}
$$

for $\mathbf { y } = \mathbf { M } ^ { - 1 } \mathbf { e } _ { i } .$ . Second, we reason about the inner product ⟨H, X⟩ which is involved in the gradient ascent update of λ. Note that H can be written as:

$$
\mathbf { H } = { \left[ \mathbf { 1 } _ { n } \right] } { \mathbf { e } } _ { n + 1 } ^ { \top } + { \mathbf { e } } _ { n + 1 } { \left[ \mathbf { 1 } _ { n } \right] } ^ { \top }
$$

Given the above, we have:

$$
\begin{array} { r l } & { \langle \mathbf H , { \mathbf X } \rangle = { \mathrm { t r } } ( \mathbf H { \mathbf X } ) } \\ & { \qquad = { \mathrm { t r } } \left( \left( \left[ \begin{array} { l } { \mathbf 1 _ { n } } \\ { 0 } \end{array} \right] { \mathbf e } _ { n + 1 } ^ { \top } + { \mathbf e } _ { n + 1 } \left[ \begin{array} { l } { \mathbf 1 _ { n } } \\ { 0 } \end{array} \right] ^ { \top } \right) { \mathbf M } ^ { - 1 } \right) } \\ & { \qquad = { \mathrm { t r } } \left( \mathbf e _ { n + 1 } ^ { \top } { \mathbf M } ^ { - 1 } \left[ \begin{array} { l } { \mathbf 1 _ { n } } \\ { 0 } \end{array} \right] + \left[ \begin{array} { l } { \mathbf 1 _ { n } } \\ { 0 } \end{array} \right] ^ { \top } { \mathbf M } ^ { - 1 } { \mathbf e } _ { n + 1 } \right) } \\ & { \qquad = 2 { \mathbf e } _ { n + 1 } ^ { \top } { \mathbf M } ^ { - 1 } \left[ \begin{array} { l } { \mathbf 1 _ { n } } \\ { 0 } \end{array} \right] } \\ & { \qquad = 2 { \mathbf e } _ { n + 1 } ^ { \top } { \mathbf M } ^ { - 1 } \mathbf M { \mathbf M } ^ { - 1 } \left[ \begin{array} { l } { \mathbf 1 _ { n } } \\ { 0 } \end{array} \right] } \\ & { \qquad = 2 { \mathbf e } _ { n + 1 } ^ { \top } { \mathbf M } ^ { - 1 } \mathbf M { \mathbf M } ^ { - 1 } \left[ \begin{array} { l } { \mathbf 1 _ { n } } \\ { 0 } \end{array} \right] } \\ & { \qquad = 2 { \mathbf y } { \mathbf M } \mathbf z } \end{array}
$$

where $\mathbf { y } = \mathbf { M } ^ { - 1 } \mathbf { e } _ { n + 1 }$ and $\mathbf { z } = \mathbf { M } ^ { - 1 } \left[ \mathbf { 1 } _ { n } \right]$ . Finally, since $\mathbf { X } = \mathbf { M } ^ { - 1 }$ and since $\mathbf { X } \succ \mathbf { 0 } _ { n } \mathbf { 0 } _ { n } ^ { \top }$ , the maximum eigenvector of X is the minimum eigenvector of M.

## C Additional Experimental Details

## C.1 Synthetic Data: Graph Creation

We create graphs of n nodes, following an approach similar to that for Erdös–Rényi graphs, that produces graphs with properties resembling those of echo chambers. We use a global parameter $p \in ( 0 . 3 , 1 )$ that controls the edge density. First, we choose r nodes uniformly at random from the n nodes, to be the ground truth echo chamber. Then, we create edges as follows. Let $A ^ { * }$ be the ground truth echo chamber with $| A ^ { * } | = r$ nodes. For nodes $i , j \in A ^ { * }$ , we create an edge $( \mathrm { i . e . , } w _ { i j } \ne 0 )$ with probability $p .$ Further, if $w _ { i j } \neq 0$ then $w _ { i j } = 1$ with probability p and $w _ { i j } = - 1$ with probability $1 - p$ . For nodes $i , j \notin \in A ^ { * }$ , we create an edge $( \mathrm { i . e . , } w _ { i j } \ne 0 )$ with probability $1 - p .$ . Further, if $w _ { i j } \neq 0$ , then $w _ { i j } = 1 \mathrm { o r } w _ { i j } = - 1$ with the same probability $( 5 0 \% )$ . For node $i \in A ^ { * }$ and node $j \notin A ^ { * }$ , we create an edge $( \mathrm { i . e . , } w _ { i j } \ne 0 )$ with probability $1 . 2 - p$ . Further, if $w _ { i j } \neq 0$ , then $w _ { i j } = 1$ with probability $1 - p$ and $w _ { i j } = - 1$ with probability $p .$

For instance, for $p = 9 0 \%$ , the edge density inside the echo chamber is approximately $p = 0 . 9$ , and among those edges $p = 9 0 \%$ are positive, resembling the expected interactions inside an echo chamber. The edge density outside the echo chamber is approximately $1 - p = 1 0 \%$ , and among those edges 50% are positive, resembling what we expect on a regular social network. The edge density between the echo chamber and nodes outside, is approximately $1 . 2 - p = 3 0 \%$ , and among those edges $1 - p = 1 0 \%$ are positive, resembling the expected interactions between the echo chamber and nodes outside the echo chamber.