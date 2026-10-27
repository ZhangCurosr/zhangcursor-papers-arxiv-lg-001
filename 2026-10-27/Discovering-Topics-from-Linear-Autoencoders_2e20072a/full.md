# Discovering Topics from Linear Autoencoders

Jutiyakhun Srisoparp Chiang Mai University Chiang Mai, Thailand jutiyakhun\_s@cmu.ac.th

## Abstract

Linear recommenders and topic models are similar in that they implicitly learn from co-occurrences of items. However, there has been little study of how these two model families are related. In this work, we show that, under a latent-factor model of the useritem interactions, the latent topics can be discovered from the weight matrix of a closed-form linear autoencoder. These topics provide a basis for interpreting the recommendations, and can be obtained directly from the trained recommender without additional training. Building on our analysis, we further propose novel item and user normalizations that prevent popular items and very active users from dominating the topics. Our experiments on six public datasets demonstrate that our method achieves significantly higher topic quality than every dedicated topic model and matches or exceeds the best linear autoencoder on all six datasets, while its recommendation performance is competitive with the best tuned recommender.

## CCS Concepts

• Information systems  Recommender systems.

## Keywords

linear autoencoder, EASE, topic model, recommender system, cooccurrence matrix, matrix factorization

## 1 Introduction

A linear recommender and a topic model can both be fitted from the same user-item interactions. Specifically, given a binary user-item matrix $X \in \{ 0 , 1 \} ^ { n \times m }$ for � users and � items, a linear recommender scores item � for user � by $( X B ) _ { u i }$ , where $B \in \mathbb { R } ^ { m \times m }$ is an itemitem weight matrix fitted from � [30, 37]. A topic model instead summarizes the � items by � latent groups, called topics, where each topic is represented by a discrete distribution over the items [5, 11].

There have been several techniques that incorporate topic models in a recommender system [16, 27, 41]. However, these techniques require textual descriptions or reviews of the items as well as a jointly fitted topic model. We ask instead: can the topics be estimated from the weight matrix � of a linear recommender, without item texts and without training a separate topic model? To answer this question, we focus on the family of linear autoencoders [29, 30, 37, 38], which fit � so that �� approximates �, subject to $B _ { i i } = 0$ for every item �, so that no item contributes to its own score. Specifically, we study the linear autoencoders whose solutions are available in closed form. The best known of such autoencoders is EASE [37], whose weight matrix � is computed from the cooccurrence counts of the items. EASE has attracted interest due to

Donlapark Ponnoprat✉   
Chiang Mai University   
Chiang Mai, Thailand   
donlapark.p@cmu.ac.th

its simplicity and strong recommendation performance, giving rise to several variants [29, 31, 38].

Despite their success, no prior work has studied whether these linear autoencoders can recover the latent topics when the interaction data is assumed to follow a topic model. In this paper, we answer this question and use the answer to build a linear autoencoder that provides both recommendations and topics.

Our contributions are summarized as follows.

Topics in linear autoencoders. We show that, when the interactions follow a latent-factor model, the topics can be estimated from the weight matrix of a closed-form linear autoencoder.

Item and user normalizations. We propose Topic-Enabled Autoencoders (TEA), which apply novel item and user normalizations to the item co-occurrence matrix. Unlike previous work [31], our user normalization does not require any hyperparameter tuning.

Empirical results. We demonstrate that TEA achieves significantly higher topic quality than every dedicated topic model and matches or exceeds the best linear autoencoder on all six public datasets, while its recommendation performance is competitive with the best tuned recommender.

Notations. For any matrix $A , A _ { i j }$ denotes the $( i , j )$ -entry of �. For any sets �, � of positive integers, $A _ { I , J }$ denotes the submatrix of � with rows in � and columns in �. Similarly, $A _ { i , J }$ and $A _ { I , i }$ denote the corresponding parts of the �-th row and the �-th column of �. For a square matrix �, we write diag � for the diagonal matrix whose �-th diagonal entry is $A _ { i i } .$ . For a vector $^ { a , }$ <sup>we</sup> <sup>write</sup> <sup>diag</sup>(<sup>�</sup>) <sup>for</sup> the diagonal matrix whose �-th diagonal entry is $a _ { i } .$ . For a (possibly <sup>non-square)</sup> <sup>matrix</sup> <sup>�,</sup> <sup>we</sup> <sup>denote</sup> <sup>by</sup> <sup>col</sup>(<sup>�</sup>) <sup>the</sup> <sup>column</sup> <sup>space</sup> <sup>of</sup> <sup>�</sup> and rank � the rank of �. We write $1 \in \mathbb { R } ^ { m }$ for the all-ones vector.

## 2 Setup

We observe � users interacting with � items as a binary matrix $X \in \{ 0 , 1 \} ^ { n \times m }$ , where $X _ { u i } = 1$ if user � has interacted with item � and $X _ { u i } = 0$ otherwise. Throughout this paper, users are indexed by � and items are indexed by $i , j .$ . We write $x _ { u } \in \{ 0 , 1 \} ^ { m }$ for the �-th row of �, as a column vector. The co-occurrence matrix of the interactions is the following Gram matrix:

$$
G \ = \ X ^ { \top } X \ = \ \sum _ { u = 1 } ^ { n } x _ { u } x _ { u } ^ { \top } , \qquad G _ { i j } \ = \ \sum _ { u = 1 } ^ { n } X _ { u i } X _ { u j } ,
$$

where $G _ { i j }$ is the number of users who interacted with both item � and item �. In particular, the diagonal entry $g _ { i } = G _ { i i }$ is the number of users who interacted with item �, which we refer to as the popularity of item �.

## 2.1 Closed-form linear autoencoders

Given an interaction matrix $X \in \{ 0 , 1 \} ^ { n \times m }$ , an item-based recommender scores the items by an item-item weight matrix $B \in \mathbb { R } ^ { m \times m }$ In regression-based recommenders, the �-th column of � contains the coeficients of a linear regression of the interactions of item � on those of the other items. The score of item � for user � is $\begin{array} { r } { ( X B ) _ { u i } = \sum _ { j } X _ { u j } B _ { j i } } \end{array}$ , that is, the sum of the weights $B _ { j i }$ over the items � in �’s history. In this work, we focus on item-based recom menders whose weight matrix � has a closed form. The best known of these is EASE [37], which solves the following problem, with a constraint diag � = 0 so that no item contributes to its own score:

$$
\widetilde { B } \ = \ \underset { B : \mathrm { d i a g } ( B ) = 0 } { \arg \operatorname* { m i n } } \ \| X - X B \| _ { F } ^ { 2 } + \lambda \| B \| _ { F } ^ { 2 } ,\tag{1}
$$

where $\left\| \cdot \right\| _ { F }$ is the Frobenius norm and $\lambda > 0$ is a regularization parameter. For a square matrix �, let ${ \mathcal { H } } ( A ) = A - \mathrm { d i a g } ( A )$ denote the of-diagonal matrix of �. With the inverse $P : = ( G + \lambda I ) ^ { - 1 }$ of the regularized co-occurrence matrix, the solution of (1) is

$$
\widetilde { B } \ = \ I - P \mathrm { d i a g } ( P ) ^ { - 1 } \ = \ - \mathcal { H } ( P ) \mathrm { d i a g } ( P ) ^ { - 1 } .\tag{2}
$$

Here, the right multiplication by diag $P ) ^ { - 1 }$ multiplies the �-th column of $- { \mathcal { H } } ( P )$ by $P _ { i i } ^ { - 1 }$

Motivated by (2), we consider a broader class of weight matrices for recommender systems.

Definition 1 (GLAE). A generalized linear autoencoder (GLAE) is an item-item weight matrix of the form:

$$
B _ { 0 } \ = \ - \mathcal { H } \big ( ( G _ { 0 } + \Lambda ) ^ { - 1 } \big ) \Sigma ,\tag{3}
$$

where $G _ { 0 } ~ \in ~ \mathbb { R } ^ { m \times m }$ is symmetric and positive semidefinite, and $\Lambda , \Sigma \in \mathbb { R } ^ { m \times m }$ are diagonal matrices with positive diagonal entries.

By (2), EASE is the GLAE (3) with $G _ { 0 } = G , \Lambda = \lambda I ,$ and $\Sigma =$ diag � −<sup>1</sup>.

## 2.2 Latent-factor models of user-item interactions

We now propose a latent-factor model of � whose latent factors define the topics. Since we want the topics to be diverse, the latent factors should not be dominated by popular items or very active users. To this end, the model places item popularity and user activity in per-item and per-user weights, separate from the latent factors. The following definition formalizes this idea.

Definition 2 (wLFM). We say that the interaction matrix � follows the weighted latent factor model (wLFM) if there exist a matrix $\beta \in \mathbb { R } _ { + } ^ { m \times K }$ , user factors $f _ { 1 } , . . . , f _ { n } \in \mathbb { R } _ { + } ^ { K }$ , and scalars $w _ { 1 } ^ { I } , \ldots , w _ { m } ^ { I } > 0$ and $w _ { 1 } ^ { U } , \dots , w _ { n } ^ { U } > 0$ such that, for every user � and item $i , X _ { u i }$ is a random variable with mean

$$
\mathbb { E } [ X _ { u i } \ | \ w _ { i } ^ { I } , w _ { u } ^ { U } , \beta _ { i , : } , f _ { u } ] \ = \ w _ { i } ^ { I } w _ { u } ^ { U } \langle \beta _ { i , : } , f _ { u } \rangle .\tag{4}
$$

In addition, for every user � and any two items � ≠ � $, X _ { u i }$ and $X _ { u j }$ are conditionally independent given $w _ { i } ^ { I } , w _ { j } ^ { I } , w _ { u } ^ { U } , \beta _ { i , : } , \beta _ { j , : }$ , and $f _ { u }$ . We call the �-th row $\beta _ { i , : }$ of $\beta$ the item factor of item � and $f _ { u }$ the user factor of user �.

For brevity, we will omit the conditioning variables for the rest of the paper and write $\mathbb { E } [ X _ { u i } ] ~ = ~ w _ { i } ^ { I } w _ { u } ^ { U } \langle \beta _ { i , : } \overline { { { \mathbf { \Omega } } } } f _ { u } \rangle$ . We note that the condition in (4) with unit weights is satisfied by the factor analysis model [13, 34] and the aspect Bernoulli model [4], both of which have been applied to recommender systems [26, 36].

## 2.3 User and item normalizations

To remove the weights $w _ { i } ^ { I }$ and $w _ { u } ^ { U }$ from the mean (4), we normalize � by the diagonal matrices of these weights:

$$
\widehat X : = D _ { U } ^ { - 1 } X D _ { I } ^ { - 1 } , \qquad \widehat G : = \widehat X ^ { \top } \widehat X = D _ { I } ^ { - 1 } X ^ { \top } D _ { U } ^ { - 2 } X D _ { I } ^ { - 1 } ,\tag{5}
$$

where $D _ { I } = \mathrm { d i a g } ( w _ { 1 } ^ { I } , \dots , w _ { m } ^ { I } )$ and $D _ { U } = \mathrm { d i a g } ( w _ { 1 } ^ { U } , \dots , w _ { n } ^ { U } )$ . It then follows that $\mathbb { E } [ \widehat { X } _ { u i } ] = \langle \beta _ { i , : } , f _ { u } \rangle$ for every user � and item �.

We will focus on the normalized matrix �b for the rest of the paper. First, we show that $\mathbb { E } [ \widehat { G } ]$ is the sum ofa positive semidefinite matrix of rank at most � and a diagonal matrix.

Lemma 1. If � follows the wLFM, the expected normalized cooccurrence matrix decomposes as

$$
\mathbb { E } [ \widehat { G } ] = L + D , L = \beta \Phi \beta ^ { \top } , \Phi = \sum _ { u = 1 } ^ { n } f _ { u } f _ { u } ^ { \top } ,
$$

and � is the diagonal matrix with $\begin{array} { r } { D _ { i i } = \sum _ { u = 1 } ^ { n } \operatorname { V a r } ( \widehat { X } _ { u i } ) } \end{array}$ . In particular, $L \succeq 0$ and $\mathrm { r a n k } ( L ) \leq K$

In view of Lemma 1, we make the following assumption on the expected normalized co-occurrence matrix:

Assumption 1. The normalized co-occurrence matrix $\widehat { G }$ in (5) satisfies $G ^ { \star } : = \mathbb { E } [ \widehat { G } ] = L + D$ , where $L = \beta \Phi \beta ^ { \top } , \beta \in \mathbb { R } _ { + } ^ { m \times K }$ , � is a diagonal matrix with positive diagonal entries, rank $( \beta ) = K$ and $\Phi \succ 0$

The most important components of this assumption are the matrices $\beta$ and �, because our definition of topics will be in terms of these two matrices.

## 2.4 Topic models from latent-factor models

Each topic is defined by one column of $\beta$ and the diagonal matrix �.

Definition 3 (Topic). Let $\beta$ and � be as in Assumption 1. For $k = 1 , \ldots , K ,$ , the �-th topic is the solution of the constrained maximization of the signal-to-noise ratio

$$
\operatorname* { m a x } _ { v \in \mathbb { R } ^ { m } } \left\{ \frac { ( v ^ { \top } \beta _ { : , k } ) ^ { 2 } } { v ^ { \top } D v } : \sum _ { i = 1 } ^ { m } v _ { i } = 1 \right\} .\tag{6}
$$

In other words, the �-th topic is the allocation of the items that maximizes the mean interaction along the �-th latent factor, while ensuring that each user’s behavior is consistent (i.e. has low variance) under this allocation.

The maximization problem (6) has the following closed-form solution.

Proposition 1. Suppose that Assumption 1 holds. Then, for every $k ,$ the maximization problem (6) has the unique solution

$$
t _ { k } = \frac { D ^ { - 1 } \beta _ { : , k } } { 1 ^ { \top } D ^ { - 1 } \beta _ { : , k } } .
$$

Thus, the �-th topic is the �-th column of $D ^ { - 1 } \beta ,$ , normalized so that its entries sum to one. Since the entries of $D ^ { - 1 } \beta$ are nonnegative, $t _ { k }$ is a probability distribution over the items. Moreover,

the order of the topics carries no meaning. This leads to the following notion of a topic model.

Definition 4 (topic model). Let $t _ { 1 } , \ldots , t _ { K }$ be the topics of Proposition 1. The topic model is the equivalence class of the matrix $\mathcal { T } _ { K } = [ t _ { 1 } , . . . , t _ { K } ] \mathbf { \bar { \Pi } } \in \mathbb { R } _ { + } ^ { m \times K }$ under permutation of its columns.

## 2.5 Topic recovery from a GLAE

We now tie everything together and state our goal. Motivated by the decomposition of $\mathbb { E } [ \widehat { G } ]$ in Lemma 1, we focus on the GLAE (3) with $G _ { 0 } = { \widehat { G } }$ . Under Assumption 1, our goal is to recover the topic model of Definition 4 from this GLAE in two steps:

Step 1: Choosing the normalizations. We narrow down our choice of GLAE by (1) choosing the item weights $w _ { i } ^ { I }$ and the user weights $w _ { u } ^ { U }$ , which determine the normalized co-occurrence matrix $\widehat { G }$ in $( 5 ) _ { : }$ , and (2) choosing the regularization matrix Λ in (3). Our choices of these weights and of Λ are in Section 5.

Step 2: Recovering the topics. We analyze whether the topic model $\mathcal { T } _ { K }$ can be recovered from the weight matrix $B _ { 0 }$ of (3) when $G _ { 0 } = { \widehat { G } } _ { : }$ the normalized co-occurrence matrix. In Section 3, we show that the GLAE with $G _ { 0 } = \mathbb { E } [ \widehat { G } ]$ determines the matrix $( D + \Lambda ) ^ { - 1 } \beta$ up to permutation and positive column scaling. We normalize the columns of this matrix to estimate $\mathcal { T } _ { K }$ , where Λ regularizes the inverse of �. In Section 4, we analyze the GLAE with the observed matrix $G _ { 0 } = { \widehat { G } }$ and bound how far its eigenspace is from col $( D +$ $\Lambda ) ^ { - 1 } \beta )$

## 3 Topic recovery from linear autoencoders

We first study the GLAE (3) with $G _ { 0 } = \mathbb { E } [ \widehat { G } ]$ and with a regularization matrix Λ. Throughout this section, we assume that Assumption 1 holds, with $\beta , \Phi ;$ , and � as in that assumption. For brevity, we write:

$$
\begin{array} { r l } & { \beta _ { \Lambda } = ( D + \Lambda ) ^ { - 1 } \beta , \qquad M = \Phi ^ { - 1 } + \beta ^ { \top } \beta _ { \Lambda } , } \\ & { P ^ { \star } = ( G ^ { \star } + \Lambda ) ^ { - 1 } , \qquad S ^ { \star } = \beta _ { \Lambda } M ^ { - 1 } \beta _ { \Lambda } ^ { \top } . } \end{array}\tag{7}
$$

When $\Lambda = 0 , \beta _ { \Lambda } = D ^ { - 1 } \beta ,$ so normalizing the columns of $\beta _ { \Lambda }$ to sum to one gives the matrix $\mathcal { T } _ { K }$ of Definition 4. However, some diagonal entries of $\boldsymbol { D }$ can be close to zero, so that the corresponding entries of $D ^ { - 1 }$ are large. The regularization matrix Λ keeps the diagonal entries of $( D + \Lambda ) ^ { - 1 }$ bounded.

To identify col $\left( { \beta } _ { \Lambda } \right)$ from $G ^ { \star }$ , we make the following assumption, which is the condition of Anderson and Rubin [2, Theorem 5.1] for identification in factor analysis.

Assumption 2. For every item �, there are disjoint sets $I , J \subset$ $\{ 1 , . . . , m \} \setminus \{ i \} \ o f K$ items each such that the submatrices $\beta _ { I , : }$ and $\beta _ { J , ; }$ : are invertible.

Note that Assumption 2 requires $m \ge 2 K + 1$ . The next two results show that, under these assumptions, the column space of $\beta _ { \Lambda }$ can be obtained from the GLAE with $G _ { 0 } = G ^ { \star }$

Lemma 2. Suppose that Assumption 1 holds. Then, $M \succ 0$ and

$$
- { \mathcal { H } } ( P ^ { \star } ) = { \mathcal { H } } ( S ^ { \star } ) .
$$

Moreover, $S ^ { \star }$ is positive semidefinite of rank � with column space $\operatorname { c o l } ( \beta _ { \Lambda } )$

The proof is in Section C. By Lemma $2 , G ^ { \star }$ determines $\operatorname { c o l } ( \beta _ { \Lambda } )$ once the diagonal entries of $S ^ { \star }$ are known. The following proposition shows that the of-diagonal entries of $P ^ { \star }$ determine these diagonal entries:

Proposition 2. Suppose that Assumptions 1 and 2 hold. Then $S ^ { \star }$ is the only matrix ofrank at most � whose of-diagonal matrix is $- { \mathcal { H } } ( P ^ { \star } )$ . Moreover, for every item �, with � and � as in Assumption 2,

$$
\begin{array} { r } { S _ { i i } ^ { \star } = - P _ { i , J } ^ { \star } \left( P _ { I , J } ^ { \star } \right) ^ { - 1 } P _ { I , i } ^ { \star } . } \end{array}\tag{8}
$$

Finally, for disjoint sets $I , J \subset \{ 1 , \ldots , m \}$ of � items each, the submatrix $P _ { I , J } ^ { \star }$ is invertible if and only $i f \beta _ { I , }$ : and $\beta _ { J , }$ : are invertible.

The proof is in Section D. Since $i \notin I \cup J$ and $I \cap J = \emptyset$ , the calculation in (8) uses only of-diagonal entries of $P ^ { \star }$ . As a consequence, if we have access to $G ^ { \star }$ , col $\left( \beta _ { \Lambda } \right)$ can be recovered in two steps. First, for every item �, we choose disjoint sets � and � of � items not containing � such that $P _ { I , J } ^ { \star }$ is invertible. Second, we compute $S _ { i i } ^ { \star }$ using (8), which, combined with the of-diagonal entries of $- { \mathcal { H } } ( { \dot { P ^ { \star } } } )$ , gives $S ^ { \star }$ , whose column space is $\operatorname { c o l } ( \beta _ { \Lambda } )$ by Lemma 2.

To obtain $\beta _ { \Lambda }$ from col $( \beta _ { \Lambda } )$ , we require a condition that determines $\beta _ { \Lambda }$ up to permutation and positive column scaling. For this purpose, we adopt the anchor condition of separable non-negative matrix factorization and of anchor-word topic models [3, 12, 15]:

Definition 5. Consider a matrix $U \in \mathbb { R } _ { + } ^ { m \times K }$ . An item $a _ { k }$ is an anchor of topic � if $U _ { a _ { k } , k } > 0$ and $U _ { a _ { k } , k ^ { \prime } } = 0$ for each $k ^ { \prime } \neq k$ . The matrix � is separable if each topic has an anchor and the � anchors are distinct items.

Assumption 3. The matrix � is separable.

We are now in a position to state the main identification result for $\beta _ { \Lambda }$

Theorem 1 (uniqeness under separability). Suppose that Assumptions 1 and 3 hold. Then $\beta _ { \Lambda }$ is separable. Moreover, every separable non-negative basis of col $\left( \beta _ { \Lambda } \right)$ is equal to $\beta _ { \Lambda }$ up to permutation and positive column scaling.

The proof is in Section E. There are several algorithms for obtaining $\beta _ { \Lambda }$ from col $\left( \beta _ { \Lambda } \right)$ , such as the successive projection algorithm (SPA) [15] and its variant, the pseudo-point successive projection algorithm (ppSPA) [22]. The permutation and the column scaling in Theorem 1 do not afect the topics. Indeed, we can normalize each column of $\beta _ { \Lambda }$ to sum to one to remove the column scaling.

## 4 Topic recovery from the observed matrix

In practice, we compute the GLAE (3) with $G _ { 0 } = { \widehat { G } } _ { : }$ , the observed matrix, instead of $G ^ { \star }$ . The resulting GLAE is $\widehat { B } = \widehat { A } \Sigma$ , where

$$
\widehat { \boldsymbol { P } } = \left( \widehat { \boldsymbol { G } } + \boldsymbol { \Lambda } \right) ^ { - 1 } , \qquad \widehat { \boldsymbol { A } } = - \mathcal { H } ( \widehat { \boldsymbol { P } } ) ,\tag{9}
$$

and $\Sigma$ is a known diagonal matrix. In particular, $\widehat { P }$ is a part of the calculation of ${ \widehat { B } } ,$ with which we will estimate the topic model.

However, the results in Section 3 do not apply here since the of-diagonal entries of $\widehat { A }$ need not be those of a rank-� matrix. In particular, the formula of (8) in Proposition 2 can give diferent diagonal entries for diferent choices of � and �. For this reason, unlike the method described in the previous section, we do not fill in the diagonal entries of $\widehat { A } .$ The following theorem shows that we can instead keep the diagonal entries of $\widehat { A }$ as zero and estimate col $\left( \beta _ { \Lambda } \right)$ by an eigenspace of $\widehat { A }$ itself.

Let $\widehat { V } \in \mathbb { R } ^ { m \times K }$ be a matrix whose columns are orthonormal eigenvectors of $\widehat { A }$ associated with its � largest eigenvalues. We measure how far $\operatorname { c o l } ( \widehat { V } )$ is from col $\left( { \beta _ { \Lambda } } \right)$ by calculating mi $\displaystyle { \mathbf { 1 } } _ { C } \| S ^ { \star } - $ ${ \widehat { V } } C \|$ where the minimum is taken over all $\mathbb { R } ^ { K \times m }$ matrices �, and is the spectral norm.

The following theorem provides a bound for the error of estimation of $\operatorname { c o l } ( \beta _ { \Lambda } )$ using $\operatorname { c o l } ( \widehat { V } )$ :

Theorem 2 (error of the estimated subspace). Suppose that Assumption 1 holds, and let $S ^ { \star }$ be as in $( 7 )$ . Let � be the smallest diagonal entry of $D + \Lambda ,$ , and let � = max<sub>�</sub> $D _ { i i } / G _ { i i } ^ { \star }$ . Then, with $\widehat { V }$ as above,

$$
\operatorname* { m i n } _ { C \in \mathbb { R } ^ { K \times m } } \left\| S ^ { \star } - \widehat { V } C \right\| \leq \frac { 5 \left\| \widehat { G } - G ^ { \star } \right\| } { \nu ^ { 2 } } + \frac { 2 \rho } { \nu } .\tag{10}
$$

The proof is in Section F. The two terms in (10) correspond to the two ways in which $\widehat { A }$ difers from $S ^ { \star }$ . The first term is due to the diference between $\widehat { G }$ and $G ^ { \star }$ . The second term is the price of keeping the diagonal entries of $\widehat { A }$ at zero instead of filling in those of $S ^ { \star }$ (which are not observed). This term arises from the fact that $S ^ { \star }$ is related to the latent factor matrix �, whose diagonal entries are at most a fraction $\rho$ of the diagonal entries of $G ^ { \star } = L + D$ . Hence, this term is small when $D$ accounts for most of each diagonal entry of $G ^ { \star }$ . In view of Lemma 1 $\begin{array} { r } { , L _ { i i } = \sum _ { u } \mathbb { E } [ \widehat { X } _ { u i } ] ^ { 2 } } \end{array}$ and $\begin{array} { r } { D _ { i i } = \sum _ { u } \operatorname { V a r } ( \widehat { X } _ { u i } ) } \end{array}$ so this is the case when the interactions with each item are noisy relative to their means.

## 5 Topic-Enabled Autoencoders (TEA)

We now introduce Topic-Enabled Autoencoders (TEA), which is the GLAE (3) with $G _ { 0 } = { \widehat { G } }$ for a specific choice of the item weights $w _ { i } ^ { I } ,$ , the user weights $w _ { u } ^ { U }$ , and the regularization matrix $\Lambda .$ The algorithm of TEA is summarized in Algorithm 1. Below, we explain each component of the algorithm.

Item normalization. Popular items can lower the diversity of the topics. Indeed, a popular item co-occurs with many other items, so the item can appear among the highest-weight items of many topics. To reduce this efect of popularity, we propose an item normalization by setting the item weights to $w _ { i } ^ { I } = g _ { i } ^ { \alpha }$ in (5), where $\alpha \in$ R is a tuning parameter. A positive � down-weights the entries of popular items relative to those of less popular items, while a negative � up-weights them. For an item with $g _ { i } = 0 ;$ , that is, an item that no user interacted with, we set $w _ { i } ^ { I } = 1$ so that every item weight is positive.

User normalization. Very active users can lower the accuracy of the estimated topics. Specifically, a very active user has many interactions, which add a large variance to ${ \widehat { G } } .$ To reduce the efect of user activity on the topics, we propose a user normalization that chooses the user weights so that the variance of $\widehat { G }$ is small relative to its of-diagonal entries.

In (5), the user weights afect $\widehat { G }$ only through the diagonal matrix $D _ { U } ^ { - 2 }$ . We write $D _ { U } ^ { - 2 } = \mathrm { d i a g } ( q _ { 1 } , . . . , q _ { n } )$ so that the weighted co occurrence matrix is $\begin{array} { r } { \bar { G } = X ^ { \top } D _ { U } ^ { - 2 } X = \sum _ { u } q _ { u } x _ { u } x _ { u } ^ { \top } } \end{array}$ . Let $\begin{array} { r } { d _ { u } = \sum _ { i } X _ { u i } } \end{array}$ be the number of items that user � interacted with, and write $d = ( d _ { 1 } , \ldots , d _ { n } )$ . The variance that user � adds to $\bar { G }$ given � depends on the matrix $A _ { u } = \mathbb { E } [ x _ { u } x _ { u } ^ { \top } \mid d ]$ , whose $( i , j )$ entry is the probability that user � interacted with both items � and �. The following theorem gives the weights $q _ { u }$ that minimize a bound on the variance that the users add to $\bar { G } _ { { \scriptsize ; } }$ , relative to the of-diagonal entries of �<sup>¯</sup>.

Theorem 3 (user weights). Write $d = ( d _ { 1 } , \ldots , d _ { n } )$ and $A _ { u } =$ $\mathbb { E } [ x _ { u } x _ { u } ^ { \top } \mid d ]$ . Let the weights $q _ { u } ~ \ge ~ 0$ depend only on $d ,$ let $Y _ { u } ~ =$ $q _ { u } x _ { u } x _ { u } ^ { \top }$ , and write $\sigma _ { u } = d _ { u } ( d _ { u } - 1 )$ and $v _ { u } = d _ { u } \lVert A _ { u } \rVert$ . Then

$$
\mathbb { E } \big [ \mathbf { 1 } ^ { \top } \mathbf { \mathcal { H } } ( \bar { G } ) \mathbf { 1 } \mid d \big ] = \sum _ { u = 1 } ^ { n } q _ { u } \sigma _ { u } ,\tag{11}
$$

$$
\left\| \sum _ { u = 1 } ^ { n } \operatorname { V a r } ( Y _ { u } \mid d ) \right\| \leq \sum _ { u = 1 } ^ { n } q _ { u } ^ { 2 } v _ { u } .\tag{12}
$$

Moreover, among the weights with $\begin{array} { r } { \sum _ { u } q _ { u } \sigma _ { u } > 0 , } \end{array}$ , the ratio

$$
F ( q ) = \frac { \big ( \sum _ { u } q _ { u } ^ { 2 } v _ { u } \big ) ^ { 1 / 2 } } { \sum _ { u } q _ { u } \sigma _ { u } }
$$

is the smallest ifand only $i f q _ { u } \propto ( d _ { u } - 1 ) / \| A _ { u } \|$ for every user � with $d _ { u } \ge 1$

The proof is in Section G. The optimal weights depend on the matrices $A _ { u } ,$ , which are unknown. We therefore approximate $\Vert A _ { u } \Vert$ by a function of $d _ { u }$ via the following probabilistic perspective. Suppose that the $d _ { u }$ interactions of user � are independent draws from a distribution $\pi _ { u } \in \mathbb { R } _ { + } ^ { m }$ over the items. Then the expected number of $\boldsymbol { u ^ { \prime } s }$ interactions with item � is $d _ { u } \pi _ { u i }$ . Similarly, for $i \neq j ,$ , the expected number of ordered pairs of�’s interactions with items � and � is $d _ { u } ( d _ { u } - 1 ) \pi _ { u i } \pi _ { u j }$ . Hence, when repeated draws of the same item are rare,

$$
A _ { u } \ \approx \ d _ { u } ( d _ { u } - 1 ) { \mathcal { H } } ( \pi _ { u } \pi _ { u } ^ { \top } ) + d _ { u } \mathrm { d i a g } ( \pi _ { u } ) .
$$

By the triangle inequality, the spectral norm of the right-hand side is at most $\begin{array} { r } { \| \mathcal { H } ( \pi _ { u } \pi _ { u } ^ { \top } ) \| d _ { u } ^ { 2 } + ( \operatorname* { m a x } _ { i } \pi _ { u i } ) d _ { u } } \end{array}$ . Here, high values of $\| \mathcal { H } ( \pi _ { u } \pi _ { u } ^ { \top } ) \|$ and max ${ \mathrm { : ~ } } \pi _ { u i }$ indicate that the user’s interactions are concentrated only on a few items. But it is dificult to estimate $\pi _ { u }$ from the user’s binary interactions, so we instead posit the following approximation for $\Vert A _ { u }$ ∥<sup>:</sup>

$$
\Vert A _ { u } \Vert \ \lesssim \ \Vert \mathcal { H } ( \pi _ { u } \pi _ { u } ^ { \top } ) \Vert d _ { u } ^ { 2 } + ( \operatorname* { m a x } _ { i } \pi _ { u i } ) d _ { u } \ \approx \ \widehat { s } d _ { u } ^ { 2 } + \widehat { t } d _ { u } ,
$$

where $\widehat { s }$ and $\widehat { t }$ play the same roles as $\| \mathcal { H } ( \pi _ { u } \pi _ { u } ^ { \top } ) \|$ and max<sub>�</sub> $\pi _ { u i }$ respectively. In particular, we choose � based on the pairs of distinct items: The pair $( i , j )$ accounts for a fraction of about $\pi _ { u i } \pi _ { u j }$ of the $d _ { u } ( d _ { u } - 1 )$ ordered pairs of user $u ,$ so the matrix of these fractions is close to $\mathcal { H } ( \pi _ { u } \pi _ { u } ^ { \top } )$ . Accordingly, we choose <sub>b</sub>� to be the largest eigenvalue of the average of these matrices over the users. Thus, we choose:

$$
\widehat { s } = \lambda _ { 1 } \Big ( \frac { 1 } { n } \sum _ { u : d _ { u } \geq 2 } \frac { \mathcal { H } ( x _ { u } x _ { u } ^ { \top } ) } { d _ { u } ( d _ { u } - 1 ) } \Big ) , \quad \widehat { t } = \operatorname* { m a x } _ { i } g _ { i } / \sum _ { i ^ { \prime } } g _ { i ^ { \prime } } ,
$$

where $\lambda _ { 1 } ( \cdot )$ denotes the largest eigenvalue of the matrix.

With this approximation, the optimal weights of Theorem 3 become $q _ { u } \propto ( d _ { u } - 1 ) / \big ( d _ { u } ( d _ { u } + \widehat { \tau } ) \big )$

$$
q _ { u } \ = \ c \ \frac { d _ { u } - 1 } { d _ { u } \left( d _ { u } + \widehat { \tau } \right) } ,\tag{13}
$$

Algorithm 1 TEA: the weight matrix and the topics   
Require: Interaction matrix $X \in \{ 0 , 1 \} ^ { n \times m }$ , number of topics $K ,$   
item exponent $\alpha ,$ dropout probability $\boldsymbol { p } \in \left[ 0 , 1 \right)$ , regularization   
parameter $\lambda > 0$   
Ensure: Weight matrix $\widehat { B } \in \mathbb { R } ^ { m \times m }$ and topics $\widehat { \beta } \in \mathbb { R } _ { + } ^ { m \times K }$ , an esti  
mate of the topic model $\mathcal { T } _ { K }$   
1: $G  X ^ { \top } X ,$ and $\begin{array} { r } { d _ { u } \gets \sum _ { i } X _ { u i } } \end{array}$ for each user �   
max<sub>�</sub> $\textstyle G _ { i i } { \big / } \sum _ { i ^ { \prime } } G _ { i ^ { \prime } i ^ { \prime } }$   
2: $\widehat { \tau } \gets$   
$\begin{array} { r } { \lambda _ { 1 } \bigg ( \frac { 1 } { n } \sum _ { u : d _ { u } \geq 2 } \frac { x _ { u } x _ { u } ^ { \top } - \mathrm { d i a g } ( x _ { u } ) } { d _ { u } ( d _ { u } - 1 ) } \bigg ) } \end{array}$   
3: �<sub>�</sub> $ ( d _ { u } - 1 ) \big / ( d _ { u } ( d _ { u } + \widehat { \tau } ) )$ for each user �   
4: $\begin{array} { r } { q _ { u }  q _ { u } / ( \frac { 1 } { n } \sum _ { v } q _ { v } ) } \end{array}$ for each user �   
5: $Q  \operatorname { d i a g } ( q _ { 1 } , . . . , q _ { n } )$   
6: $D _ { I } \gets \mathrm { d i a g } ( \operatorname* { m a x } ( G _ { 1 1 } , 1 ) ^ { \alpha } , \dots ,$ max $\cdot ( G _ { m m } , 1 ) ^ { \alpha } )$   
7: $\widehat { G } \gets D _ { I } ^ { - 1 } X ^ { \top } Q X D _ { I } ^ { - 1 }$   
8: $\begin{array} { r } { \Lambda  \frac { \hat { p } } { 1 - p } \operatorname { d i a g } ( G ) ^ {  } \lambda I } \end{array}$   
9: ${ \widehat { P } } \gets ( { \widehat { G } } + \Lambda ) ^ { - 1 }$ , and $\widehat { B } \gets - \mathcal { H } ( \widehat { P } )$ di $\mathrm { \lg } (  { \widehat { P } } ) ^ { - 1 }$   
10: $\widehat { A } \gets - \mathcal { H } ( \widehat { P } )$   
11: $\widehat { V } \gets$ the � leading eigenvectors of ${ \widehat { A } } ,$ as columns   
12: ${ \widehat { \beta } } \gets \mathrm { p p S P A } ( { \widehat { V } } , K )$   
13: $\widehat { \beta } _ { : , k } \gets \widehat { \beta } _ { : , k } / \| \widehat { \beta } _ { : , k } \| _ { 1 }$ for $k = 1 , \ldots , K$

where ${ \widehat { \tau } } = { \widehat { t } } / { \widehat { s } } ,$ and $c > 0$ is chosen so that $q _ { 1 } , \ldots , q _ { n }$ sum to � to maintain the original scale of ${ \widehat { G } } .$ Since � is computed from $X ,$ the user weights require no tuning.

Matrix regularization. We use the dropout regularization proposed in [29, 38]:

$$
\Lambda \ = \ { \frac { p } { 1 - p } } \ \operatorname { d i a g } ( G ) + \lambda I ,
$$

where $\textstyle p \in [ 0 , 1 )$ is the dropout probability and $\lambda > 0$ is the regular ization parameter. With ${ \widehat { P } } \operatorname { o f } \left( 9 \right)$ , the weight matrix of TEA is given by (2) with � replaced by ${ \widehat { P } } { : }$

$$
{ \widehat { B } } \ = \ - { \mathcal H } ( { \widehat { P } } ) \ \mathrm { d i a g } ( { \widehat { P } } ) ^ { - 1 } .
$$

Topic estimation. To extract the topics, we compute the matrix $\widehat { V }$ of the � leading eigenvectors of the matrix $\widehat { A } = \widehat { \widehat { B } } \mathrm { d i a g } ( \widehat { P } )$ in (9), whose column space estimates col $\left( \beta _ { \Lambda } \right)$ by Theorem $2 .$ To obtain $\widehat { \beta }$ from the rows of ${ \widehat { V } } ,$ we apply the ppSPA algorithm [22], which allows us to obtain $\beta _ { \Lambda }$ from col $\left( \beta _ { \Lambda } \right)$ in Section 3. Finally, to follow the definition of a topic model in Definition $^ { 4 , }$ we normalize each column of $\widehat { \beta }$ to sum to one and use the columns of $\widehat { \beta }$ as the topics.

## 6 Experiments

## 6.1 Recovery on synthetic data

We perform a simulation by drawing $\beta \in \mathbb { R } _ { + } ^ { m \times K }$ for $m = 4 0 0$ items and � = 5 topics: The first � rows of � form the identity matrix, and the remaining rows are drawn independently from the Dirichlet distribution with parameter $( 0 . 3 , \ldots , 0 . 3 )$ . The user factors $f _ { u }$ are drawn independently from the Dirichlet distribution with parameter $2 \gamma ,$ where $\gamma _ { k } \propto \dot { 3 } ^ { - ( k - 1 ) / ( K - 1 ) }$ and $\textstyle \sum _ { k } \gamma _ { k } = 1$ . The interactions are drawn independently as $X _ { u i }$ Bernoulli $\left( 0 . 1 \left. \beta _ { i , : } , f _ { u } \right. \right)$ . We fit EASE with $\lambda = 2 5 0$ on $G = X ^ { \top } X$ for � ranging from 500 to $1 0 ^ { 7 }$ Here, $\widehat { V }$ is computed from the matrix $\widehat { A }$ in (9) with unit weights, and with $\Lambda = \lambda I$ . With the sampled $\beta ,$ we compute $G ^ { \star }$ and $S ^ { \star }$ in Lemma 2 using (7). Since the size of $S ^ { \star }$ changes with �, we evaluate how far $\operatorname { c o l } ( \widehat { V } )$ is from col ${ \bf \beta } _ { \beta _ { \Lambda } } )$ using the relative error:

![](images/a84f102aa930c670f46eea88b06aac63652ae0dcc1dba1cef22526d8b6b966bb.jpg)

![](images/b80e2e5b47dfd5518d8f673e8caad55c168ae1b4cbbb8ab0c8efa8f2ce0a7c98.jpg)  
Figure 1: Subspace recovery on synthetic data. Left: the largest eigenvalues of $\widehat { A }$ at $n \ : = \ : 1 5 0 0 0 0$ . Right: the relative error min<sub>�</sub> $\| S ^ { \star } - \widehat { V } C \| / \| S ^ { \star } \|$ against $n _ { \mathnormal { \star } }$ with the mean and one standard deviation over four seeds (the error bars are too small to be visible). The dotted line is the limit of the relative error with $\widehat { G } = G ^ { \star }$ as $n \to \infty$

$$
\operatorname* { m i n } _ { C \in \mathbb { R } ^ { K \times m } } \lVert \boldsymbol { S } ^ { \star } - \widehat { V } C \rVert / \lVert \boldsymbol { S } ^ { \star } \rVert .\tag{14}
$$

Figure 1 shows the simulation results. The left panel displays the largest eigenvalues of $\widehat { A }$ from one simulation with $n = 1 5 0 0 0 0$ . We observe that the eigenvalues show a visible cut-of after the �-th largest one. This cut-of is consistent with Lemma 2, by which �b difers from the rank-� matrix $S ^ { \star }$ by the diagonal matrix diag �★ and the sampling noise in ${ \widehat { G } } .$

For the recovery of the subspace col $\left( { \beta _ { \Lambda } } \right)$ , the right panel of Figure 1 illustrates that the relative error (14) decreases as � increases. This decrease is consistent with the first term of (10), which arises from the sampling noise in $\widehat { G }$ and is expected to approach zero as $n  \infty .$ In contrast, the second term in (10), which arises from omitting the diagonal entries of $S ^ { \star }$ , is irreducible as � increases. We can see from the plot that the relative error approaches the irreducible error as $n \to \infty$

## 6.2 Benchmark on real datasets

Our goal is to evaluate the recommendation performances and topic qualities of TEA on several public datasets. We detail our experiment below.

Datasets. We evaluate on six datasets: amazon-books, amazon-cds [20], gowalla [8], ml-1m [17], online-retail-ii [7], and steam [23]. For each dataset, we keep the 5000 items with the most interactions and a random sample of at most 5000 users. We split each user’s interactions by time, holding out the items of the last timestamp for testing and those of the second-to-last timestamp for validation, without the items that the user interacted with earlier.

Baselines. We compare TEA against several baselines, including six collaborative-filtering models: WMF [21], LightGCN [18], SLIMElastic [30], ADMMSLIM [39], NCEPLRec [43], and RecVAE [35], and six topic models, LDA [5], ETM [11], ECRTM [44], Mult VAE [25], NMF of � (NMF), and NMF of  � , followed by ppSPA (NMF-ppSPA) [3]. We also compare TEA against five closed-form linear autoencoders: EASE [37], RDLAE [29], DAN [31], and EASE with the random-walk (RW) and symmetric normalizations (Symmetric), which are commonly used for recommendation models [9, 10, 32, 42].

Table 1: NDCG@10, recall@10 (Rec), and topic quality (TQ). Bold NDCG@10, and recall@10 numbers are based on paired two-sided �-tests over the users at 5% level, and bold TQ numbers are based on bootstrap resampling tests at 5% level.
<table><tr><td></td><td colspan="3">books</td><td colspan="3">cds</td><td colspan="3">gowalla</td><td colspan="3">ml-1m</td><td colspan="3">retail</td><td colspan="3">steam</td></tr><tr><td>Model</td><td>NDCG</td><td>Rec</td><td>TQ</td><td>NDCG</td><td>Rec</td><td>TQ</td><td>NDCG</td><td>Rec</td><td>TQ</td><td>NDCG</td><td>Rec</td><td>TQ</td><td>NDCG</td><td>Rec</td><td>TQ</td><td>NDCG</td><td>Rec</td><td>TQ</td></tr><tr><td>WMF</td><td>0.040</td><td>0.077</td><td></td><td>0.046</td><td>0.085</td><td></td><td>0.083</td><td>0.151</td><td></td><td>0.029</td><td>0.051</td><td></td><td>0.043</td><td>0.049</td><td></td><td>0.018</td><td>0.033</td><td></td></tr><tr><td>LightGCN</td><td>0.021</td><td>0.039</td><td></td><td>0.017</td><td>0.032</td><td></td><td>0.052</td><td>0.091</td><td></td><td>0.025</td><td>0.042</td><td></td><td>0.027</td><td>0.030</td><td></td><td>0.027</td><td>0.054</td><td></td></tr><tr><td>SLIMElastic</td><td>0.047</td><td>0.082</td><td></td><td>0.051</td><td>0.092</td><td></td><td>0.094</td><td>0.164</td><td></td><td>0.042</td><td>0.072</td><td></td><td>0.058</td><td>0.064</td><td></td><td>0.028</td><td>0.057</td><td></td></tr><tr><td>ADMMSLIM</td><td>0.046</td><td>0.081</td><td></td><td>0.052</td><td>0.093</td><td></td><td>0.096</td><td>0.168</td><td></td><td>0.041</td><td>0.070</td><td></td><td>0.056</td><td>0.060</td><td></td><td>0.026</td><td>0.051</td><td></td></tr><tr><td>NCEPLRec</td><td>0.040</td><td>0.073</td><td></td><td>0.050</td><td>0.090</td><td></td><td>0.093</td><td>0.167</td><td></td><td>0.042</td><td>0.072</td><td></td><td>0.056</td><td>0.062</td><td></td><td>0.025</td><td>0.048</td><td></td></tr><tr><td>RecVAE</td><td>0.039</td><td>0.072</td><td></td><td>0.046</td><td>0.086</td><td></td><td>0.089</td><td>0.165</td><td></td><td>0.037</td><td>0.063</td><td></td><td>0.046</td><td>0.050</td><td></td><td>0.025</td><td>0.048</td><td></td></tr><tr><td>LDA</td><td>0.018</td><td>0.038</td><td>0.140</td><td>0.014</td><td>0.029</td><td>0.129</td><td>0.073</td><td>0.135</td><td>0.300</td><td>0.030</td><td>0.052</td><td>0.313</td><td>0.031</td><td>0.033</td><td>0.297</td><td>0.023</td><td>0.046</td><td>0.179</td></tr><tr><td>ETM</td><td>0.018</td><td>0.038</td><td>0.137</td><td>0.007</td><td>0.014</td><td>0.024</td><td>0.032</td><td>0.065</td><td>0.004</td><td>0.025</td><td>0.042</td><td>0.060</td><td>0.031</td><td>0.032</td><td>0.115</td><td>0.027</td><td>0.055</td><td>0.046</td></tr><tr><td>ECRTM</td><td>0.020</td><td>0.044</td><td>0.199</td><td>0.033</td><td>0.061</td><td>0.141</td><td>0.043</td><td>0.068</td><td>0.104</td><td>0.015</td><td>0.027</td><td>0.155</td><td>0.020</td><td>0.020</td><td>0.185</td><td>0.007</td><td>0.014</td><td>0.028</td></tr><tr><td>Mult-VAE</td><td>0.030</td><td>0.058</td><td>0.064</td><td>0.033</td><td>0.063</td><td>0.039</td><td>0.092</td><td>0.164</td><td>0.090</td><td>0.025</td><td>0.041</td><td>0.092</td><td>0.030</td><td>0.031</td><td>0.106</td><td>0.025</td><td>0.050</td><td>0.069</td></tr><tr><td>NMF</td><td>0.019</td><td>0.042</td><td>0.169</td><td>0.010</td><td>0.023</td><td>0.285</td><td>0.079</td><td>0.137</td><td>0.322</td><td>0.030</td><td>0.050</td><td>0.287</td><td>0.032</td><td>0.034</td><td>0.273</td><td>0.031</td><td>0.065</td><td>0.139</td></tr><tr><td>NMF-ppSPA</td><td>0.006</td><td>0.012</td><td>0.187</td><td>0.003</td><td>0.008</td><td>0.166</td><td>0.043</td><td>0.077</td><td>0.094</td><td>0.023</td><td>0.037</td><td>0.422</td><td>0.024</td><td>0.026</td><td>0.346</td><td>0.032</td><td>0.066</td><td>0.117</td></tr><tr><td>EASE</td><td>0.045</td><td>0.081</td><td>0.191</td><td>0.051</td><td>0.092</td><td>0.219</td><td>0.093</td><td>0.163</td><td>0.498</td><td>0.043</td><td>0.073</td><td>0.495</td><td>0.059</td><td>0.066</td><td>0.459</td><td>0.029</td><td>0.061</td><td>0.130</td></tr><tr><td>RW</td><td>0.036</td><td>0.065</td><td>0.209</td><td>0.043</td><td>0.080</td><td>0.255</td><td>0.104</td><td>0.184</td><td>0.488</td><td>0.030</td><td>0.051</td><td>0.407</td><td>0.038</td><td>0.043</td><td>0.278</td><td>0.026</td><td>0.049</td><td>0.188</td></tr><tr><td>Symmetric</td><td>0.038</td><td>0.068</td><td>0.330</td><td>0.038</td><td>0.071</td><td>0.272</td><td>0.086</td><td>0.162</td><td>0.483</td><td>0.033</td><td>0.055</td><td>0.479</td><td>0.048</td><td>0.052</td><td>0.312</td><td>0.017</td><td>0.032</td><td>0.231</td></tr><tr><td>DAN</td><td>0.046</td><td>0.085</td><td>0.046</td><td>0.048</td><td>0.090</td><td>0.233</td><td>0.104</td><td>0.184 0.488</td><td></td><td>0.041</td><td>0.067</td><td>0.418</td><td>0.056</td><td>0.061</td><td>0.285</td><td>0.031</td><td>0.062</td><td>0.037</td></tr><tr><td>RDLAE</td><td>0.046</td><td>0.085</td><td>0.068</td><td>0.051</td><td>0.092</td><td>0.180</td><td>0.099</td><td>0.176</td><td>0.379</td><td>0.043</td><td>0.075</td><td>0.446</td><td>0.058</td><td>0.064</td><td>0.420</td><td>0.029</td><td>0.059</td><td>0.128</td></tr><tr><td>TEA</td><td>0.046</td><td>0.083</td><td>0.331</td><td>0.051</td><td></td><td>0.096 0.380</td><td>0.100</td><td>0.176</td><td>0.508</td><td>0.040</td><td>0.068</td><td>0.492</td><td>0.056</td><td>0.062</td><td>0.472</td><td>0.029</td><td>0.058</td><td>0.290</td></tr></table>

Selection of� and ofthe hyperparameters. For hyperparameter selection, we split the users into three sets. For each candidate value of � and of the hyperparameters, we fit one model on the training items of each two of the three sets of users and evaluate the model on the validation items of those users. To select $K ,$ we extract the topics $\widehat { \beta }$ from EASE with � = 250 by the last five steps of Algorithm 1, and use them to predict each user’s validation items. We model each validation item of user � as a draw from the distribution $\begin{array} { r } { \sum _ { k } \theta _ { u k } \bar { \beta } _ { : , k } } \end{array}$ over the items, where the weights $\theta _ { u k }$ are fitted to the training items of� and $\bar { \beta } _ { : , k } = 0 . 9 9 \widehat { \beta } _ { : , k } + 0 . 0 1 1 / m$ so that no item has probability zero. We score each � by the logperplexity over the validation items. We compute this score for every � in 5, . . . , 10, 12, 15, 18, 20, 30, 50, 100, 150, 200, 300 and select the � with the smallest mean log-perplexity. The selected � is 6 on amazon-books, 7 on amazon-cds, 30 on gowalla, 18 on ml-1m, 6 on online-retail-ii, and 5 on steam. We select the hyperparameters of every method on each dataset by NDCG@10 on the validation items. The selected values of the hyperparameters for all models are reported in Section H.

Evaluation. Each model is fitted with its selected hyperparameters on the training items of all users. The fitted models are then used to rank, for each user, all but their training and validation items. We report NDCG@10 and recall@10 of these rankings on the test items. To evaluate the topics $\widehat { \beta }$ of each method, we use the topic quality (TQ) [11], defined as TD TC, where the topic diversity (TD) is the number of distinct items among the top 25 items of the � columns of ${ \widehat { \beta } } ,$ divided by 25�, and the topic coherence (TC) is

$$
\mathrm { T C } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { 4 5 } \sum _ { \{ i , j \} \subset { T _ { k } } } \frac { \log \left( \eta _ { i j } / { \left( \eta _ { i } \eta _ { j } \right) } \right) } { - \log \eta _ { i j } } ,
$$

where $T _ { k }$ is the top ten items of ${ \widehat { \beta } } _ { ; k } , \eta _ { i }$ is the fraction of users whose training items include item �, and $\eta _ { i j }$ is the fraction of users whose training items include both � and � (the summand is 0 if $\eta _ { i j } = 0 )$ Here, each summand is referred to as the normalized pointwise mutual information (NPMI) between item � and � [1, 6]. To compute TQ, we hold out one third of the users, refit each model on the training items of the other two thirds, and then compute TQ on the held-out users.

To compare NDCG@10 and recall@10 between any two methods, we employ paired two-sided �-tests across the test users at the 5% level. Since TQ is not an average over users, we instead compare TQ using paired bootstrap resampling over the held-out users [24].

Quantitative analysis of the results. Table 1 reports the three metrics of all the methods over the six datasets. The table shows that (1) TEA has the highest topic quality, or statistically ties with the highest topic quality, on all six datasets, and is significantly higher than every other method on cds and steam; (2) TEA’s topic quality is significantly higher than that of every topic model on all six datasets; and (3) TEA has significantly higher topic quality than EASE on three datasets and statistically ties with EASE on the other three, which demonstrates that the normalizations and the dropout regularization that TEA adds to EASE improve the topic quality.

Table 2: Example topics recovered by TEA, each with its five highest-weight items. $n ^ { \mathrm { e f f } }$ is the efective number of users who support the topic (defined in Section I). We omit gowalla, whose items are location identifiers without text.
<table><tr><td>Topic</td><td>Top items</td><td>neff</td></tr><tr><td colspan="3">amazon-books: five of the K = 6 topics. Items are books.</td></tr><tr><td>young adult fiction</td><td>Mockingjay (The Hunger Games), Divergent, Catching Fire (Hunger Games, Book Two), The Hunger Games (Book 1), City of Bones (The Mortal Instruments, Book 1)</td><td>22.5</td></tr><tr><td>Women&#x27;s Murder Club series</td><td>4th of July (Women&#x27;s Murder Club), The 6th Target (Women&#x27;s Murder Club), 2nd Chance (Women&#x27;s Murder Club), 7th Heaven (Women&#x27;s Murder Club), 10th Anniversary (Women&#x27;s Murder Club)</td><td>8.2</td></tr><tr><td>psychological thrillers</td><td>Don&#x27;t Look for Me: A Novel, The Maid: A Novel (Molly the Maid Book 1), The Golden Couple: A Novel, The Plot: A Novel, Dark Roads: A Novel</td><td>7.7</td></tr><tr><td>Nameless thriller series</td><td>In the Heart of the Fire (Nameless: Season One Book 1), The Mercy of Snakes (Nameless: Season One Book 5), The Praying Mantis Bride (Nameless: Season One Book 3), Red Rain (Nameless: Season One Book 4), Photographing the Dead (Nameless:</td><td>16.7</td></tr><tr><td>physics workbooks</td><td>Season One Book 2) Physics Concept Questions - Book 1 (Mechanics 1), Physics Concept Questions - Book 3 (Matter), Physics Concept Questions - Book 4 (Heat), Physics Concept Questions - Book 2 (Mechanics 2), Physics Concept Questions - Book 5 (Sound)</td><td>13.4</td></tr><tr><td colspan="3">amazon-cds: five of the K = 7 topics. Items are albums.</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Metallica</td><td>Master of Puppets, Ride the Lightning, Kill &#x27;Em All, ...And Justice For All, Metallica</td><td>32.6 12.7</td></tr><tr><td>KISS</td><td>Rock And Roll Over, Dynasty, Dressed To Kill, Creatures Of The Night, Kiss</td><td>35.0</td></tr><tr><td>classic rock mid-2000s pop</td><td>Revolver, Abbey Road, The Beatles (The White Album), Meddle, Sgt. Pepper&#x27;s Lonely Hearts Club Band Love. Angel. Music. Baby., Breakaway, Autobiography, Speak, Rebirth</td><td>18.7</td></tr><tr><td>hip-hop</td><td>Stillmatic, The College Dropout, The Black Album, Illmatic, The Blueprint</td><td>14.1</td></tr><tr><td colspan="3">m1-1m: five of the K = 18 topics. Items are films.</td></tr><tr><td>Disney animation</td><td>Cinderella, Sleeping Beauty, Dumbo, Peter Pan, Snow White and the Seven Dwarfs</td><td>435.4</td></tr><tr><td>1990s science-fiction action</td><td>Judge Dredd, Johnny Mnemonic, Spawn, Universal Soldier, Timecop</td><td>379.5</td></tr><tr><td>1980s comedy</td><td>Weird Science, Weekend at Bernie&#x27;s, Police Academy, European Vacation, The Money Pit</td><td>290.3</td></tr><tr><td>Clint Eastwood westerns 1990s horror</td><td>High Plains Drifter, The Outlaw Josey Wales, For a Few Dollars More, A Fistful of Dollars, Hang &#x27;em High</td><td>258.3 206.0</td></tr><tr><td></td><td>Scream 2, I Know What You Did Last Summer, Urban Legend, Thinner, Candyman</td><td></td></tr><tr><td colspan="3">online-retail-ii: five of the K = 6 topics. Items are stock lines.</td></tr><tr><td>polka-dot tableware</td><td>Red Retrospot Cup, Blue Polkadot Cup, Blue Polkadot Plate, Red Retrospot Plate, Pink Polkadot Plate</td><td>152.9</td></tr><tr><td>printed bags metal signs</td><td>Jumbo Bag Vintage Doily, Lunch Bag Vintage Doily, Jumbo Bag Vintage Leaf, Lunch Bag Apple Design, Jumbo Bag Alphabet Hand Over The Chocolate Sign, Gin + Tonic Diet Metal Sign, Please One Person Metal Sign, Cook With Wine Metal Sign, No</td><td>196.5</td></tr><tr><td></td><td>Singing Metal Sign</td><td>272.2</td></tr><tr><td>tea sets and retro kitchenware</td><td>Green Regency Teacup And Saucer, Roses Regency Teacup And Saucer, Pink Regency Teacup And Saucer, Regency Tea Plate Green, Regency Tea Plate Roses</td><td>184.8</td></tr><tr><td>birthday cards and gift wrap</td><td>Spaceboy Birthday Card, Elephant Birthday Card, Tea Party Birthday Card, Cowboys And Indians Birthday Card, Spaceboy Gift Wrap</td><td>173.5</td></tr><tr><td colspan="3">steam: the K = 5 topics. Items are games.</td></tr><tr><td>Half-Life games</td><td>Half-Life 2: Episode One, Half-Life: Opposing Force, Half-Life: Blue Shift, Half-Life: Source, Half-Life 2: Lost Coast Trove, Brawlhalla, Garry&#x27;s Mod, Team Fortress 2, BLOCKADE 3D</td><td>19.5</td></tr><tr><td>online multiplayer games</td><td>Hyperdimension Neptunia Re;Birth1, Hyperdimension Neptunia Re;Birth2, Hyperdimension Neptunia Re;Birth3 V Generation,</td><td>69.4</td></tr><tr><td>anime JRPGs</td><td>Hyperdimension Neptunia U: Action Unleashed, Fairy Fencer F</td><td>10.6</td></tr><tr><td>strategy &amp; simulation</td><td>Stellaris, Sid Meier&#x27;s Civilization: Beyond Earth, Crusader Kings II, Project Zomboid, Kerbal Space Program Nihilumbra, The Bridge, Dust: An Elysian Tail, Ballpoint Universe - Infinite, Botanicula</td><td>33.5 9.2</td></tr><tr><td>hand-drawn indie games</td><td></td><td></td></tr></table>

In terms of recommendation performance, linear autoencoders are as accurate as the collaborative filtering baselines: on every dataset, the best linear autoencoder is at most 0.001 below the best baseline in NDCG@10. TEA is also competitive with the best method. In terms of NDCG@10, TEA statistically ties with the best method on books, cds, gowalla, and steam, and is at most 0.004 below the best method on ml-1m and retail. In terms of recall@10, TEA is within 0.01 of the best method on every dataset.

Qualitative analysis ofthe topics. Table 2 shows five topics of TEA for each dataset, except gowalla whose items are location identifiers. These topics range from broad genres to very narrow groups of items. For example, some topics consist of the albums of a single artist, such as KISS on amazon-cds, or the books of a single series, such as the Nameless series on amazon-books. These narrow topics appear because many users interacted with several albums or books of the same artist or series. Indeed, the efective number of users $n ^ { \mathrm { e f f } }$ (defined in Section I) is above ten for both KISS and the Nameless series. Other topics group the items by both genre and period, such as 1980s comedy and 1990s horror on ml-1m and mid-2000s pop on amazon-cds. On online-retail-ii, the topics are product lines that share a common design, such as polka-dot tableware and Regency teacups and saucers, which suggests that the customers tend to buy matching items together.

The efective number of users also helps explain the diference in topic quality between TEA and EASE. On amazon-books, amazon-cds, and steam, where TEA has significantly higher topic quality than EASE, the topics of EASE are supported by far fewer users than those of TEA. Hence, these topics of EASE can be dominated by the co-occurrences of a few very active users. On the other three datasets, the topics of EASE are already supported by many users, which may explain why the topic quality of TEA is not significantly higher than that of EASE on these datasets. This pattern suggests that the user normalization, which down-weights very active users, improves the topics when their co-occurrences come from a few users.

$$
\alpha ^ { \star }
$$

![](images/ca03470eb3a1ea663f5f9accc796e881264adf6743b678b1e4ede7657971ba10.jpg)  
Figure 2: Validation NDCG@10 of TEA against the item exponent �. For each $\alpha ,$ the line is the largest value over the 20 pairs of � and ${ \boldsymbol { p } } ,$ and the dots are the individual pairs.

Sensitivity to the item exponent. To see how the item normalization afects the recommendations, we plot the validation NDCG@10 of TEA against the item normalization parameter � in Figure 2. On all six datasets, the curve attains its maximum at an $\alpha \in \{ 0 , 0 . 1 2 5 , 0 . 2 5 \}$ , that ${ \mathrm { i } } s ,$ when TEA does not reweight the items or slightly down-weights the popular items. As � moves away from this range, the NDCG@10 decreases, by up to 55% at $\alpha = - 0 . 5$ on ml-1m and 36% at $\alpha = 0 . 5$ on steam. Thus, the choice of � matters for the recommendation performance of TEA.

## 7 Related Work

User and item normalization. Random-walk [10, 32] and symmetric [9, 18, 42] normalizations give the item-item matrices with entries $\textstyle \sum _ { u } X _ { u i } X _ { u j } / ( g _ { i } d _ { u } )$ and $\sum _ { u } X _ { u i } X _ { u j } / ( d _ { u } \sqrt { g _ { i } g _ { j } } )$ , respectively. Other normalizations raise the degrees to exponents and include both matrices as special cases [14, 31]. DAN [31] is the closest method to ours, since both perform item and user normalization. However, DAN normalizes $G _ { i j }$ by $g _ { i } ^ { 1 - \alpha } g _ { j } ^ { \alpha }$ , while ours normalizes $G _ { i j }$ by $g _ { i } ^ { \alpha } g _ { j } ^ { \alpha }$ . Moreover, DAN’s user normalization requires tuning a hyperparameter, whereas ours does not require any.

Latent-factor models for recommendations. Given the latent variables of the users and items, the likelihood of the interactions can be a mixture over discrete latent classes [19], a Gaussian distribution [28], or a multinomial distribution whose parameters are the output of a neural network [25]. Other methods fit the interactions by a matrix factorization with a squared loss weighted by implicit feedback, without an explicit generative model [21]. Our model assumes that the conditional mean of each interaction is linear in a �-dimensional latent vector and that the interactions of a user are conditionally independent across items; examples of models that satisfy these assumptions are aspect Bernoulli models [4] and Poisson models [16].

Topic models in recommender systems. Several recommenders fit a topic model to the texts or reviews of the items jointly with a factorization of the interactions [16, 27, 41]. The topics of these methods are distributions over the words of item texts or reviews. In contrast, our topics are distributions over the items, and we estimate the topics from only the user-item interactions.

Topics and structure from recommenders. Post-hoc methods describe each factor of a matrix factorization by a topic model of the item descriptions [33]. Hierarchical models factorize the interactions through a learned hierarchy of user and item clusters [40]. Both approaches need an additional model, either a topic model of the item text or a new hierarchical recommender. In contrast, we extract the topics directly from the weight matrix of the recommender, demonstrating that the recommender can also be a topic model.

## 8 Limitations

Our uniqueness result (Theorem 1) relies on Assumption 3, which requires that every topic has an anchor item, that is, an item that belongs to none of the other topics. Future work could relax this assumption, extend our analysis to more general latent-factor models, explore other user and item normalizations, and extend the recommendation to sequential or online settings.

## 9 Conclusion

We show that, under a latent-factor model of the user-item interactions, the topics of the items can be estimated from the weight matrix of a closed-form linear autoencoder. Thus, such an autoencoder provides both item recommendations and a topic model. In turn, the topics ofer a way to interpret the recommendations. Building on this result, we propose TEA, which adds an item normalization and a tuning-free user normalization to a linear autoencoder. Our experiments demonstrate that TEA achieves significantly higher topic quality than every dedicated topic model and matches or exceeds the best linear autoencoder on all six public datasets, while its recommendation performance is competitive with the best tuned recommender.

## Acknowledgments

This work is supported by Faculty of Science, Chiang Mai University.

## References

[1] Nikolaos Aletras and Mark Stevenson. 2013. Evaluating Topic Coherence Using Distributional Semantics. In Proceedings ofthe 10th International Conference on Computational Semantics (IWCS 2013) – Long Papers, Alexander Koller and Katrin Erk (Eds.). Association for Computational Linguistics, Potsdam, Germany, 13–22.

[2] T. W. Anderson and Herman Rubin. 1956. Statistical Inference in Factor Analysis. In Proceedings of the Third Berkeley Symposium on Mathematical Statistics and Probability, J. Neyman (Ed.), Vol. 5. University of California Press, Berkeley, 111–150.

[3] Sanjeev Arora, Rong Ge, and Ankur Moitra. 2012. Learning Topic Models – Going beyond SVD. In 2012 IEEE 53rd Annual Symposium on Foundations of Computer Science. 1–10.

[4] Ella Bingham, Ata Kabán, and Mikael Fortelius. 2009. The aspect Bernoulli model: multiple causes of presences and absences. Pattern Analysis and Applications 12, 1 (2009), 55–78.

[5] David M. Blei, Andrew Y. Ng, and Michael I. Jordan. 2003. Latent Dirichlet Allocation. J. Mach. Learn. Res. 3 (2003), 993–1022.

[6] Gerlof Bouma. 2009. Normalized (pointwise) mutual information in collocation extraction. In From Form to Meaning: Processing Texts Automatically. Proceedings ofthe Biennial GSCL Conference 2009 (Potsdam, Germany), Christian Chiarcos, Richard Eckart de Castilho, and Manfred Stede (Eds.). Narr, Tübingen, Germany, 31–40.

[7] Daqing Chen, Sai Laing Sain, and Kun Guo. 2012. Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. Journal ofDatabase Marketing & Customer Strategy Management 19, 3 (2012), 197–208.

[8] Eunjoon Cho, Seth A. Myers, and Jure Leskovec. 2011. Friendship and mobility: user movement in location-based social networks. In Proceedings of the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 1082–1090.

[9] Jeongwhan Choi, Seoyoung Hong, Noseong Park, and Sung-Bae Cho. 2023. Blurring-Sharpening Process Models for Collaborative Filtering. In Proceedings ofthe 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (Taipei, Taiwan) (SIGIR ’23). Association for Computing Machinery, New York, NY, USA, 1096–1106.

[10] Colin Cooper, Sang Hyuk Lee, Tomasz Radzik, and Yiannis Siantos. 2014. Ran dom walks in recommender systems: exact computation and simulations. In Proceedings ofthe 23rd International Conference on World Wide Web (Seoul, Korea) (WWW ’14 Companion). Association for Computing Machinery, New York, NY, USA, 811–816.

[11] Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei. 2020. Topic Modeling in Embedding Spaces. Transactions of the Association for Computational Linguistics 8 (2020), 439–453.

[12] David Donoho and Victoria Stodden. 2003. When Does Non-Negative Matrix Factorization Give a Correct Decomposition into Parts?. In Advances in Neural Information Processing Systems, S. Thrun, L. Saul, and B. Schölkopf (Eds.), Vol. 16. MIT Press.

[13] B. S. Everitt. 1984. An Introduction to Latent Variable Models. Springer Netherlands.

[14] Hao-Ming Fu, Patrick Poirson, Kwot Sin Lee, and Chen Wang. 2022. Revisiting Neighborhood-based Link Prediction for Collaborative Filtering. In Companion Proceedings ofthe Web Conference 2022 (Virtual Event, Lyon, France) (WWW’22). Association for Computing Machinery, New York, NY, USA, 1009––1018.

[15] Nicolas Gillis and Stephen A. Vavasis. 2014. Fast and Robust Recursive Algorithms for Separable Nonnegative Matrix Factorization. IEEE Transactions on Pattern Analysis and Machine Intelligence 36, 4 (2014), 698–714.

[16] Prem Gopalan, Laurent Charlin, and David M. Blei. 2014. Content-based recommendations with Poisson factorization. In Advances in Neural Information Processing Systems, Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K. Weinberger (Eds.), Vol. 27. Curran Associates, Inc.

[17] F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Trans. Interact. Intell. Syst. 5, 4, Article 19 (Dec. 2015), 19 pages.

[18] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, YongDong Zhang, and Meng Wang. 2020. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. In Proceedings ofthe 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (Virtual Event, China) (SIGIR ’20). Association for Computing Machinery, New York, NY, USA, 639–648

[19] Thomas Hofmann. 2004. Latent semantic models for collaborative filtering. ACM Trans. Inf. Syst. 22, 1 (Jan. 2004), 89–115.

[20] Yupeng Hou, Jiacheng Li, Xiangjun Fu, Zhankui He, An Yan, Xiusi Chen, and Julian McAuley. 2026. Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders. In Proceedings of the 64th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (Eds.). Association for Computational Linguistics, San Diego, California, United States, 3251–3265

[21] Yifan Hu, Yehuda Koren, and Chris Volinsky. 2008. Collaborative Filtering for Implicit Feedback Datasets. In 2008 Eighth IEEE International Conference on Data Mining. 263–272.

[22] Jiashun Jin, Tracy Ke, Gabriel Moryoussef, Jiajun Tang, and Jingming Wang. 2024. Improved algorithm and bounds for successive projection. In International Conference on Learning Representations, B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (Eds.), Vol. 2024. 27645–27655.

[23] Wang-Cheng Kang and Julian McAuley. 2018. Self-Attentive Sequential Recommendation . In 2018 IEEE International Conference on Data Mining (ICDM). IEEE Computer Society, Los Alamitos, CA, USA, 197–206.

[24] Philipp Koehn. 2004. Statistical Significance Tests for Machine Translation Evaluation. In Proceedings ofthe 2004 Conference on Empirical Methods in Natural Language Processing, Dekang Lin and Dekai Wu (Eds.). Association for Computational Linguistics, Barcelona, Spain, 388–395.

[25] Dawen Liang, Rahul G. Krishnan, Matthew D. Hofman, and Tony Jebara. 2018. Variational Autoencoders for Collaborative Filtering. In Proceedings ofthe 2018 World Wide Web Conference (Lyon, France) (WWW ’18). International World Wide Web Conferences Steering Committee, Republic and Canton of Geneva, CHE, 689–698.

[26] Benjamin M Marlin. 2003. Modeling User Rating Profiles For Collaborative Filtering. In Advances in Neural Information Processing Systems, S. Thrun, L. Saul, and B. Schölkopf (Eds.), Vol. 16. MIT Press.

[27] Julian McAuley and Jure Leskovec. 2013. Hidden factors and hidden topics: understanding rating dimensions with review text. In Proceedings ofthe 7th ACM Conference on Recommender Systems. ACM, 165–172.

[28] Andriy Mnih and Russ R Salakhutdinov. 2007. Probabilistic Matrix Factorization. In Advances in Neural Information Processing Systems, J. Platt, D. Koller, Y. Singer, and S. Roweis (Eds.), Vol. 20. Curran Associates, Inc.

[29] Jaewan Moon, Hye-young Kim, and Jongwuk Lee. 2023. It’s Enough: Relaxing Diagonal Constraints in Linear Autoencoders for Recommendation. In Proceedings ofthe 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (Taipei, Taiwan) (SIGIR ’23). Association for Computing Machinery, New York, NY, USA, 1639–1648.

[30] Xia Ning and George Karypis. 2011. SLIM: Sparse Linear Methods for Top-N Recommender Systems. In 2011 IEEE 11th International Conference on Data Mining. 497–506.

[31] Seongmin Park, Mincheol Yoon, Hye-young Kim, and Jongwuk Lee. 2025. Why is Normalization Necessary for Linear Recommenders?. In Proceedings ofthe 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (Padua, Italy) (SIGIR ’25). Association for Computing Machinery, New York, NY, USA, 2142–2151.

[32] Bibek Paudel, Fabian Christofel, Chris Newell, and Abraham Bernstein. 2016. Updatable, Accurate, Diverse, and Scalable Recommendations for Interactive Applications. ACM Trans. Interact. Intell. Syst. 7, 1, Article 1 (Dec. 2016), 34 pages.

[33] Marco Rossetti, Fabio Stella, and Markus Zanker. 2013. Towards Explaining Latent Factors with Topic Models in Collaborative Recommender Systems. In 2013 24th International Workshop on Database and Expert Systems Applications. 162–167.

[34] Sam Roweis and Zoubin Ghahramani. 1999. A Unifying Review of Linear Gaussian Models. Neural Computation 11, 2 (1999), 305–345.

[35] Ilya Shenbin, Anton Alekseev, Elena Tutubalina, Valentin Malykh, and Sergey I. Nikolenko. 2020. RecVAE: A New Variational Autoencoder for Top-N Recommendations with Implicit Feedback. In Proceedings ofthe 13th International Conference on Web Search and Data Mining (Houston, TX, USA) (WSDM ’20). Association for Computing Machinery, New York, NY, USA, 528–536.

[36] Nathan Srebro. 2004. Learning with Matrix Factorizations. Ph. D. Dissertation. Massachusetts Institute of Technology.

[37] Harald Steck. 2019. Embarrassingly Shallow Autoencoders for Sparse Data. In The World Wide Web Conference (San Francisco, CA, USA) (WWW ’19). Association for Computing Machinery, New York, NY, USA, 3251––3257.

[38] Harald Steck. 2020. Autoencoders that don't overfit towards the Identity. In Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.), Vol. 33. Curran Associates, Inc., 19598– 19608.

[39] Harald Steck, Maria Dimakopoulou, Nickolai Riabov, and Tony Jebara. 2020. ADMM SLIM: Sparse Recommendations for Many Users. In Proceedings ofthe 13th International Conference on Web Search and Data Mining (Houston, TX, USA) (WSDM ’20). Association for Computing Machinery, New York, NY, USA, 555–563.

[40] Kai Sugahara and Kazushi Okamoto. 2024. Hierarchical matrix factorization for interpretable collaborative filtering. Pattern Recognition Letters 180 (2024), 99–106.

[41] Chong Wang and David M. Blei. 2011. Collaborative topic modeling for recom mending scientific articles. In Proceedings ofthe 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 448–456.

[42] Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua. 2019. Neural Graph Collaborative Filtering. In Proceedings ofthe 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (Paris, France) (SIGIR’19). Association for Computing Machinery, New York, NY, USA, 165–174.

[43] Ga Wu, Maksims Volkovs, Chee Loong Soon, Scott Sanner, and Himanshu Rai. 2019. Noise Contrastive Estimation for One-Class Collaborative Filtering. In Proceedings ofthe 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (Paris, France) (SIGIR’19). Association for Computing Machinery, New York, NY, USA, 135–144.

[44] Xiaobao Wu, Xinshuai Dong, Thong Thanh Nguyen, and Anh Tuan Luu. 2023. Efective Neural Topic Modeling with Embedding Clustering Regularization. In

Proceedings of the 40th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 202), Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (Eds.). PMLR, 37335–37357.

## A Proof of Lemma 1

For any user � and items $i \neq j ,$ the entries $\widehat { X } _ { u i }$ and $\widehat { X } _ { u j }$ are independent. Moreover, $\mathbb { E } [ \widehat { X } _ { u i } ] = \langle \beta _ { i , : } , f _ { u } \rangle$ and $\mathbb { E } [ \widehat { X } _ { u i } ^ { 2 } ] = \operatorname { V a r } ( \widehat { X } _ { u i } ) + ( \mathbb { E } \widehat { X } _ { u i } ) ^ { 2 }$ Summing over � therefore gives:

$$
\begin{array} { r l } & { \mathbb { E } [ \widehat { G } ] _ { i j } = \displaystyle \sum _ { u } \mathbb { E } [ \widehat { X } _ { u i } ] \mathbb { E } [ \widehat { X } _ { u j } ] = ( \beta \Phi \beta ^ { \top } ) _ { i j } \quad ( i \neq j ) , } \\ & { \mathbb { E } [ \widehat { G } ] _ { i i } = \displaystyle \sum _ { u } \big ( \mathrm { V a r } ( \widehat { X } _ { u i } ) + ( \mathbb { E } \widehat { X } _ { u i } ) ^ { 2 } \big ) = ( \beta \Phi \beta ^ { \top } ) _ { i i } + D _ { i i } . } \end{array}
$$

Since $\begin{array} { r } { \Phi = \sum _ { u } f _ { u } f _ { u } ^ { \top } } \end{array}$ , we have $\Phi \succeq 0 .$ , and therefore $L \succeq 0$ with rank $\begin{array} { r } { ( L ) \leq \operatorname { r a n k } ( \Phi ) \leq K . } \end{array}$ □

## B Proof of Proposition 1

Fix �. For any $v \in \mathbb { R } ^ { m }$ , the Cauchy–Schwarz inequality gives:

$$
\begin{array} { r l } & { ( \boldsymbol { v } ^ { \top } \beta _ { : , \boldsymbol { k } } ) ^ { 2 } = \big ( ( D ^ { 1 / 2 } \boldsymbol { v } ) ^ { \top } ( D ^ { - 1 / 2 } \beta _ { : , \boldsymbol { k } } ) \big ) ^ { 2 } } \\ & { \qquad \leq ( \boldsymbol { v } ^ { \top } D \boldsymbol { v } ) ( \beta _ { : , \boldsymbol { k } } ^ { \top } D ^ { - 1 } \beta _ { : , \boldsymbol { k } } ) . } \end{array}
$$

Thus, $( v ^ { \top } \beta _ { : , k } ) ^ { 2 } / ( v ^ { \top } D v ) \leq \beta _ { : , k } ^ { \top } D ^ { - 1 } \beta _ { : , k }$ for every nonzero �. Equality holds if and only $\mathrm { i f } D ^ { 1 / 2 } v$ is proportional to $D ^ { - 1 / 2 } \beta _ { : , k }$ , that ${ \mathrm { i } } s ,$ $v \propto D ^ { - 1 } \beta _ { : , k } .$ . Since rank $( \beta ) = K$ , the column $\beta _ { : , k } \ge 0$ is nonzero. Since � is a diagonal matrix with positive diagonal entries, we have $1 ^ { \top } D ^ { - 1 } \beta _ { : , k } > 0$ . The constraint $\textstyle \sum _ { i } v _ { i } = 1$ therefore fixes the proportionality constant and gives the unique maximizer $t _ { k } . ~ \sqsupset$

## C Proof of Lemma 2

Write $D _ { \Lambda } = D + \Lambda$ . By Assumption 1, � is a diagonal matrix with positive diagonal entries. Since the diagonal entries of Λ are also positive, $D _ { \Lambda } \succ 0$ . Since $\Phi \succ 0$ and $\beta ^ { \top } \beta _ { \Lambda } = \beta ^ { \top } D _ { \Lambda } ^ { - 1 } \beta \succeq 0 ,$ we have $M \succ 0 .$ . Consequently, the Woodbury identity applied to ${ \cal G } ^ { \star } + \Lambda =$ $D _ { \Lambda } + \beta \Phi \beta ^ { \top }$ gives:

$$
P ^ { \star } = D _ { \Lambda } ^ { - 1 } - S ^ { \star } .\tag{15}
$$

Since $M \succ 0$ , we can write $S ^ { \star } = \Gamma \Gamma ^ { \top }$ with $\Gamma = \beta _ { \Lambda } M ^ { - 1 / 2 }$ , so $S ^ { \star } \succeq$ 0. Since $M ^ { - 1 / 2 }$ is invertible, rank �★ = rank $\left( \beta _ { \Lambda } \right)$ and col $( S ^ { \star } ) =$ c $\operatorname { \mathrm { { o l } } } ( \beta _ { \Lambda } )$ . Since rank $( \beta ) = K$ and $D _ { \Lambda } ^ { - 1 }$ is invertible, rank $( \beta _ { \Lambda } ) = K$ Finally, $D _ { \Lambda } ^ { - 1 }$ is a diagonal matrix, $\overset { \vartriangle } { \boldsymbol { s } _ { 0 } } - \mathcal { H } ( \boldsymbol { P } ^ { \star } ) = \mathcal { H } ( \boldsymbol { S } ^ { \star } )$ □

## D Proof of Proposition 2

As in the proof of Lemma 2, write $S ^ { \star } = \Gamma \Gamma ^ { \top }$ with $\Gamma = \beta _ { \Lambda } M ^ { - 1 / 2 }$ . By Lemma 2, $P _ { a b } ^ { \star } = - S _ { a b } ^ { \star }$ for all items $a \neq b$

We first show that, for any disjoint sets $I , J \subset \{ 1 , \ldots , m \}$ of � items each, $P _ { I , J } ^ { \star }$ is invertible if and only if $\beta _ { I , : }$ and $\beta _ { J , ; }$ <sub>:</sub> are invertible. The proof of Lemma 2 shows that $D _ { \Lambda }$ is a diagonal matrix with positive diagonal entries and that $M \succ 0$ . Consequently, the $K \times K$ matrices $( \bar { D _ { \Lambda } ^ { - 1 } } ) _ { I , I }$ and $M ^ { - 1 / 2 }$ are invertible. Since $\dot { D } _ { \Lambda } ^ { - 1 }$ is a diagonal matrix, $\Gamma _ { I ; : } = ( D _ { \Lambda } ^ { - 1 } ) _ { I , I } \beta _ { I , : } M ^ { - 1 / 2 }$ . Hence, the matrix $\Gamma _ { I , : }$ is invertible if and only if $\beta _ { I , : }$ is invertible. The same holds for $\Gamma _ { J , : }$ and $\beta _ { J , : }$ Moreover, $S _ { I , J } ^ { \star } = \Gamma _ { I , : } \Gamma _ { J , : } ^ { \top }$ , so det $S _ { I , J } ^ { \star } = \operatorname* { d e t } \Gamma _ { I , : }$ det $\Gamma _ { J , : }$ . Since $I \cap J = \emptyset ;$ every pair $( a , b ) \in I \times J$ has $a \neq b .$ . Hence, $P _ { I , J } ^ { \star } = - S _ { I , J } ^ { \star }$ . Therefore, $P _ { I , J } ^ { \star }$ is invertible if and only if ${ \bf \nabla } \beta _ { I , : } $ and $\beta _ { J , : }$ are invertible.

Next, we prove (8). Fix an item $i ,$ and let � and � be as in ${ \mathrm { A } } s \cdot$ sumption 2. Since $\beta _ { I , : }$ <sub>:</sub> and $\beta _ { J , : }$ <sub>:</sub> are invertible, the matrices $\Gamma _ { I , : } , \Gamma _ { J , : }$ and $S _ { I , J } ^ { \star } = - P _ { I , J } ^ { \star }$ are invertible. Since $S _ { i , J } ^ { \star } = \Gamma _ { i , : } \Gamma _ { J , : } ^ { \top } , S _ { I , J } ^ { \star } = \Gamma _ { I , : } \Gamma _ { J , : } ^ { \top }$ , and $S _ { I , i } ^ { \star } = \bar { \Gamma } _ { I , : } \Gamma _ { i , : } ^ { \top }$ , we obtain:

$$
S _ { i , J } ^ { \star } ( S _ { I , J } ^ { \star } ) ^ { - 1 } S _ { I , i } ^ { \star } = \Gamma _ { i , : } \Gamma _ { J , : } ^ { \top } \big ( \Gamma _ { J , : } ^ { \top } \big ) ^ { - 1 } \Gamma _ { I , : } ^ { - 1 } \Gamma _ { I , : } \Gamma _ { i , : } ^ { \top } = \Gamma _ { i , : } \Gamma _ { i , : } ^ { \top } = S _ { i i } ^ { \star } .\tag{16}
$$

Since � $\notin I \cup J$ and $I \cap J = \emptyset ;$ , every pair ofindices $( a , b )$ in $\{ i \} \times J , I \times J ,$ or $I \times \{ i \}$ has � ${ \mathrm { : } } \neq b .$ . Hence, $S _ { i , J } ^ { \star } = - P _ { i , J } ^ { \star } , S _ { I , J } ^ { \star } = - P _ { I , J } ^ { \star }$ , and $S _ { I , i } ^ { \star } = - P _ { I , i } ^ { \star } .$ Substituting these three equations into $S _ { i i } ^ { \star } = S _ { i , J } ^ { \star } ( S _ { I , J } ^ { \star } ) ^ { - 1 } S _ { I , i } ^ { \star }$ gives:

$$
S _ { i i } ^ { \star } = - P _ { i , J } ^ { \star } ( P _ { I , J } ^ { \star } ) ^ { - 1 } P _ { I , i } ^ { \star } ,
$$

which is (8).

It remains to prove the uniqueness of $S ^ { \star }$ . Let $S ^ { \prime } \in \mathbb { R } ^ { m \times m }$ have rank at most � and satisfy $\mathcal { H } ( S ^ { \prime } ) = - \mathcal { H } ( P ^ { \star } ) = \mathcal { H } ( S ^ { \star } )$ , where the second equality holds by Lemma 2. Fix an item �, and let � and � be as in Assumption 2. Consider the submatrix $S _ { \{ i \} \cup I , \{ i \} \cup J } ^ { \prime } ,$ whose entries are $S _ { a b } ^ { \prime }$ with $a \in \{ i \} \cup I$ and $b \in \{ i \} \cup J .$ Since ${ \dot { i } } \not \in { \dot { I } } \cup J$ and $I \cap J = \emptyset ;$ , the only index in both � � and $\{ i \} \cup J$ is �. Hence, $a \neq b$ for every such pair $( a , b )$ other than �, � , so $S _ { a b } ^ { \prime } = S _ { a b } ^ { \star }$ because $\mathcal { H } ( S ^ { \prime } ) = \mathcal { H } ( S ^ { \star } )$ . Consequently, with � listed first among the rows and among the columns of this submatrix,

$$
S _ { \{ i \} \cup I , \{ i \} \cup J } ^ { \prime } = \left( S _ { i i } ^ { \prime } \quad S _ { i , J } ^ { \star } \right) .
$$

Since � and � have � items each and $i \notin I \cup J ,$ this submatrix is $\left( K + 1 \right) \times \left( K + 1 \right)$ . Since the rank of a submatrix is at most the rank of the matrix, this submatrix has rank at most rank $( S ^ { \prime } ) \leq K$ and is therefore singular. Moreover, $S _ { I , J } ^ { \star }$ is invertible, since $\beta _ { I , \cdot }$ <sub>:</sub> and $\beta _ { J , }$ <sub>:</sub> are invertible. Consequently, the determinant formula for the Schur complement gives:

$$
0 = \operatorname* { d e t } S _ { \{ i \} \cup I , \{ i \} \cup J } ^ { \prime } = \operatorname* { d e t } ( S _ { I , J } ^ { \star } ) \left( S _ { i i } ^ { \prime } - S _ { i , J } ^ { \star } ( S _ { I , J } ^ { \star } ) ^ { - 1 } S _ { I , i } ^ { \star } \right) .
$$

Since det $( S _ { I , J } ^ { \star } ) \neq 0 ,$ , this equation and (16) give:

$$
S _ { i i } ^ { \prime } = S _ { i , J } ^ { \star } ( S _ { I , J } ^ { \star } ) ^ { - 1 } S _ { I , i } ^ { \star } = S _ { i i } ^ { \star } .
$$

Since the item � is arbitrary, diag ${ \mathfrak { f } } ( S ^ { \prime } ) = \mathrm { d i a g } ( S ^ { \star } )$ . Combined with $\mathcal { H } ( S ^ { \prime } ) = \mathcal { H } ( S ^ { \star } )$ , this gives $S ^ { \prime } = S ^ { \star }$ . Finally, by Lemma $2 , S ^ { \star }$ has rank � and $\mathcal { H } ( S ^ { \star } ) = - \mathcal { H } ( P ^ { \star } )$ . Hence, $S ^ { \star }$ is the only matrix of rank at most � whose of-diagonal matrix $\operatorname { i s } - { \mathcal { H } } ( P ^ { \star } )$ □

## E Proof of Theorem 1

We prove Theorem 1. The notation is that of Section 3, with $D _ { \Lambda } =$ $D + \Lambda$

Proof of Theorem 1. Since rank $( \beta ) = K$ and $D _ { \Lambda } ^ { - 1 }$ is invertible, $\beta _ { \Lambda }$ has rank �. Let the columns of $O \in \mathbb { R } ^ { m \times K }$ form an orthonormal basis of col $\left( \beta _ { \Lambda } \right)$ . Since $O O ^ { \intercal }$ is the orthogonal projection onto <sup>col</sup>(<sup>�</sup>Λ)<sup>,</sup> <sup>we</sup> <sup>have</sup> $\beta _ { \Lambda } = O O ^ { \top } \beta _ { \Lambda } = O N$ with $\overset { \cdot } { N } : = \overset { \cdot } { O } ^ { \top } \beta _ { \Lambda } \in \mathbb { R } ^ { K \times K }$ Since $K = \mathrm { r a n k } ( \beta _ { \Lambda } ) = \mathrm { r a n k } ( O N ) \leq \mathrm { r a n k } ( N )$ , the matrix � is invertible. Write $o _ { i } ^ { \top }$ and $r _ { i } ^ { \top }$ for the �-th rows of � and $\beta _ { \Lambda }$ . Since $\beta _ { \Lambda } = O N$ , we have $r _ { i } ^ { \top } = o _ { i } ^ { \top } N$ , that is, $o _ { i } ^ { \top } = r _ { i } ^ { \top } N ^ { - 1 }$ . Let $c _ { k } ^ { \top }$ be the �-th row of $N ^ { - 1 }$ . Consequently, the vectors $c _ { 1 } , \ldots , c _ { K }$ are linearly independent.

We claim that $\beta _ { \Lambda }$ is separable. Since the entries of both $\beta$ and $D _ { \Lambda } ^ { - 1 }$ are non-negative, the entries of $\beta _ { \Lambda }$ are non-negative. Moreover, each row of $\beta _ { \Lambda }$ has zero entries in the same positions as the same row of �. Consequently, every anchor of � is an anchor of $\beta _ { \Lambda }$ . Since the � anchors of � are distinct items, $\beta _ { \Lambda }$ is separable.

Now, we show that every separable non-negative basis ofcol $\left( \beta _ { \Lambda } \right)$ is $\beta _ { \Lambda }$ up to permutation and positive column scaling. Since $r _ { i } \geq 0 _ { : }$ each $\begin{array} { r } { \boldsymbol { o } _ { i } = \sum _ { k } ( \boldsymbol { r } _ { i } ) _ { k } \boldsymbol { c } _ { k } } \end{array}$ lies in the cone $C : = { \mathrm { c o n e } } \{ c _ { 1 } , \dots , c _ { K } \}$ , that ${ \mathrm { i } } s ,$ the set of non-negative linear combinations of $c _ { 1 } , \ldots , c _ { k } .$ By the definition of $o _ { i } ,$ it is immediate that cone $\left\{ o _ { 1 } , \ldots , o _ { m } \right\} \subseteq C .$ Conversely, since $\beta _ { \Lambda }$ is separable, each column � of $\beta _ { \Lambda }$ has an anchor $a _ { k } , s _ { 0 } r _ { a _ { k } } = \theta _ { k } e _ { k }$ for some $\theta _ { k } > 0 ,$ where $e _ { k }$ is the �-th standard basis vector of $\mathbb { R } ^ { K }$ . Hence, $c _ { k } ~ = ~ o _ { a _ { k } } / \theta _ { k }$ for every �. Therefore, cone $\left\{ o _ { 1 } , . . . , o _ { m } \right\} \supseteq C$ , and so $\mathrm { c o n e } \{ o _ { 1 } , . . . , o _ { m } \} = C$

Now let $W ^ { \prime } \in \mathbb { R } _ { + } ^ { m \times K }$ be a separable matrix whose columns form a basis of col $\left( \beta _ { \Lambda } \right)$ . As for $\beta _ { \Lambda }$ , we have $W ^ { \prime } : = O N ^ { \prime }$ for some invertible matrix $N ^ { \prime }$ . Let $\boldsymbol { c } _ { k } ^ { \prime \intercal }$ be the �-th row of $N ^ { \prime - 1 }$ . Since $W ^ { \prime }$ is non-negative and separable, the same argument as for $\beta _ { \Lambda }$ gives:

$$
\mathrm { c o n e } \{ c _ { 1 } ^ { \prime } , . . . , c _ { K } ^ { \prime } \} = \mathrm { c o n e } \{ o _ { 1 } , . . . , o _ { m } \} = C .
$$

Consequently, $\{ c _ { 1 } , \ldots , c _ { K } \}$ and $\{ c _ { 1 } ^ { \prime } , \ldots , c _ { K } ^ { \prime } \}$ are linearly indepen dent sets that generate the same cone . Hence, these two (ordered) sets of vectors difer by a permutation and positive scaling. Since the rows of $N ^ { - 1 }$ and $\dot { N ^ { \prime } } ^ { - 1 }$ are these two sets of vectors, there are a permutation matrix Π and a diagonal matrix Υ with positive diagonal entries such that $\begin{array} { r } { { N } ^ { \prime - 1 } = \mathbf { \bar { \Upsilon } } { \mit \mathrm { T } } \boldsymbol { \Pi } ^ { \intercal } { N } ^ { - 1 } } \end{array}$ . Since $\boldsymbol { \Pi } ^ { - 1 } = \boldsymbol { \Pi } ^ { \top }$ , we have

$$
\begin{array} { r } { W ^ { \prime } = O N ^ { \prime } = O N \Pi \Upsilon ^ { - 1 } = \beta _ { \Lambda } \Pi \Upsilon ^ { - 1 } , } \end{array}
$$

as claimed.

□

## F Proof of Theorem 2

We use the notation of Sections 3 and 4, with $D _ { \Lambda } = D + \Lambda$ . Write:

$$
\Psi = { \cal P } ^ { \star } - \widehat { \cal P } , \qquad \delta = \| \widehat { G } - G ^ { \star } \| , \qquad \Delta = \| \widehat { A } - { S } ^ { \star } \| ,
$$

and let $\lambda _ { j } ( \cdot )$ denote the �-th largest eigenvalue. By Lemma 2 and $( 1 5 ) , P ^ { \star } = D _ { \Lambda } ^ { - 1 } - S ^ { \star } , S ^ { \star } \succeq 0 ,$ , and $\operatorname { r a n k } ( S ^ { \star } ) = K$ . Since $\widehat { P } = P ^ { \star } - \Psi$ is linear, and $\mathcal { H } ( D _ { \Lambda } ^ { - 1 } ) = 0 , ( 9 )$ gives:

$$
\widehat { A } = - \mathcal { H } ( P ^ { \star } ) + \mathcal { H } ( \Psi ) = S ^ { \star } - \mathrm { d i a g } ( S ^ { \star } ) + \mathcal { H } ( \Psi ) .\tag{17}
$$

Moreover, since $G ^ { \star } + \Lambda = L + D _ { \Lambda } \succeq D _ { \Lambda } \succeq \nu I .$ the matrix $P ^ { \star } = ( G ^ { \star } +$ $\Lambda ) ^ { - 1 }$ is positive definite and $\lVert P ^ { \star } \rVert \leq 1 / \nu$ . Hence, $0 \preceq S ^ { \star } \preceq D _ { \Lambda } ^ { - 1 }$ and $\| S ^ { \star } \| \leq 1 / \nu$

We claim that

$$
\operatorname* { m i n } _ { C \in \mathbb { R } ^ { K \times m } } \lVert S ^ { \star } - \widehat { V } C \rVert \leq 2 \Delta .\tag{18}
$$

Taking $C = \widehat { V } ^ { \top } S ^ { \star }$ $\begin{array} { r } { \mathrm { g i v e s ~ m i n } _ { C } \| S ^ { \star } - \widehat { V } C \| \leq \| ( I - \widehat { V } \widehat { V } ^ { \top } ) S ^ { \star } \| } \end{array}$ . Since the columns of $\widehat { V }$ are orthonormal eigenvectors of $\widehat { A }$ associated with the eigenvalues $\lambda _ { 1 } ( \widehat { A } ) , \ldots , \lambda _ { K } ( \widehat { A } )$ , we have $\| ( I - { \widehat { V } } { \widehat { V } } ^ { \top } ) { \widehat { A } } \| =$ max $_ { \cdot j > K } | \lambda _ { j } ( \widehat { A } ) |$ . Since $S ^ { \star } \succeq 0$ has rank ${ \cal K } , \lambda _ { j } ( S ^ { \star } ) = 0$ for every $j > K ,$ so Weyl’s inequality gives $| \lambda _ { j } ( \widehat { A } ) | \leq \Delta$ for every $j > K ,$ Consequently, since $\| I - \widehat { V } \widehat { V } ^ { \top } \| \leq 1$ , the triangle inequality gives:

$$
\begin{array} { r } { \| ( I - \widehat { V } \widehat { V } ^ { \top } ) S ^ { \star } \| \leq \| ( I - \widehat { V } \widehat { V } ^ { \top } ) \widehat { A } \| + \| ( I - \widehat { V } \widehat { V } ^ { \top } ) ( \widehat { A } - S ^ { \star } ) \| \leq 2 \Delta . } \end{array}
$$

Moreover, by ${ ( 1 7 ) } , \widehat { A } - S ^ { \star } = \mathcal { H } ( \Psi ) - \mathrm { d i a g } ( S ^ { \star } )$ . Since the diagonal entries of $S ^ { \star } \succeq 0$ are non-negative, the triangle inequality gives $\begin{array} { r } { \Delta \leq \| \mathcal { H } ( \Psi ) \| + \operatorname* { m a x } _ { i } S _ { i i } ^ { \star } } \end{array}$

Now, we show that max ${ _ i S _ { i i } ^ { \star } } \le \rho / \nu$ . Fix an item �. Since $G ^ { \star } { + } \Lambda \succ 0 .$ the Cauchy–Schwarz inequality gives $P _ { i i } ^ { \star } \left( G _ { i i } ^ { \star } + \Lambda _ { i i } \right) \ge 1$ . Since

$G _ { i i } ^ { \star } + \Lambda _ { i i } - ( D _ { \Lambda } ) _ { i i } = L _ { i i } , ( D _ { \Lambda } ) _ { i i } \geq \nu ,$ and $\Lambda _ { i i } > 0 , ( 1 5 )$ gives:

$$
\begin{array} { l } { { S _ { i i } ^ { \star } = \displaystyle \frac { 1 } { ( D _ { \Lambda } ) _ { i i } } - P _ { i i } ^ { \star } \le \displaystyle \frac { 1 } { ( D _ { \Lambda } ) _ { i i } } - \displaystyle \frac { 1 } { G _ { i i } ^ { \star } + \Lambda _ { i i } } } } \\ { { { } } } \\ { { \displaystyle \quad = \displaystyle \frac { L _ { i i } } { ( D _ { \Lambda } ) _ { i i } ( G _ { i i } ^ { \star } + \Lambda _ { i i } ) } \le \displaystyle \frac { L _ { i i } } { \nu G _ { i i } ^ { \star } } \le \displaystyle \frac { \rho } { \nu } . } } \end{array}\tag{19}
$$

Finally, we prove (10) in two regimes of �:

If $\delta \leq \nu / 5 ,$ we first bound $\| { \mathcal { H } } ( \Psi ) \|$ . Since $\mathcal { H } ( \Psi ) = \Psi -$ diag Ψ and max<sub>�</sub> $| \Psi _ { i i } | \leq | | \Psi | |$ , we have $\| \mathcal { H } ( \Psi ) \| \leq 2 \| \Psi \|$ By Weyl’s inequality, $\lambda _ { m } ( { \widehat { G } } + \Lambda ) \geq \lambda _ { m } ( G ^ { \star } + \Lambda ) - \delta \geq \nu - \delta \geq$ $4 \nu / 5 ,$ , so $\lVert \widehat { P } \rVert \leq 5 / ( 4 \nu )$ . Since $\Psi = P ^ { \star } \big ( ( \widehat { G } + \Lambda ) - ( G ^ { \star } +$ $\Lambda ) ) { \widehat { P } } = P ^ { \star } ( { \widehat { G } } - G ^ { \star } ) { \widehat { P } }$ , we obtain $\begin{array} { r } { \| \mathcal { H } ( \Psi ) \| \leq 2 \| P ^ { \star } \| \delta \| \widehat { P } \| \leq } \end{array}$ $5 \delta / ( 2 \nu ^ { 2 } )$ . Combining this bound with $\begin{array} { r } { ( 1 8 ) , \Delta \leq \| \mathcal { H } ( \Psi ) \| + } \end{array}$ max<sub>�</sub> $S _ { i i } ^ { \star }$ , and (19) gives:

$$
\operatorname* { m i n } _ { C } \lVert S ^ { \star } - \widehat { V } C \rVert \leq 2 \lVert \mathcal { H } ( \Psi ) \rVert + 2 \operatorname* { m a x } _ { i } S _ { i i } ^ { \star } \leq \frac { 5 \delta } { \nu ^ { 2 } } + \frac { 2 \rho } { \nu } .
$$

If $\delta > \nu / 5 ,$ then $5 \delta / \nu ^ { 2 } > 1 / \nu$ . Taking $C = 0$ and using $\rho \geq 0 ;$ we obtain:

$$
\operatorname* { m i n } _ { C } \lVert S ^ { \star } - \widehat { V } C \rVert \leq \lVert S ^ { \star } \rVert \leq \frac { 1 } { \nu } < \frac { 5 \delta } { \nu ^ { 2 } } \leq \frac { 5 \delta } { \nu ^ { 2 } } + \frac { 2 \rho } { \nu } .
$$

Hence, (10) holds in both regimes.

## G Proof of Theorem 3

We first prove (11). Since $\mathbf { 1 } ^ { \top } x _ { u } x _ { u } ^ { \top } \mathbf { 1 } = d _ { u } ^ { 2 }$ and the diagonal entries of $x _ { u } x _ { u } ^ { \top }$ sum to $d _ { u }$ , we have $\mathbf { 1 } ^ { \top } \mathbf { \mathcal { H } } ( x _ { u } x _ { u } ^ { \top } ) \mathbf { 1 } ~ = ~ d _ { u } ( d _ { u } - 1 )$ . Since $\begin{array} { r } { \bar { G } = \sum _ { u } q _ { u } x _ { u } x _ { u } ^ { \top } } \end{array}$ and the weights $q _ { u }$ depend only on �, summing over the users gives (11).

Next, we prove (12). Here, $\operatorname { V a r } ( Y \mid d ) = \operatorname { \mathbb { E } } [ Y ^ { 2 } \mid d ] - \operatorname { \mathbb { E } } [ Y \mid d ] ^ { 2 } =$ $\mathbb { E } \big [ ( Y - \mathbb { E } [ Y \mid d ] ) ^ { 2 } \mid d \big ] \succeq 0$ for a symmetric random matrix $Y .$ . Since $x _ { u }$ is binary, $( x _ { u } x _ { u } ^ { \top } ) ^ { 2 } { \stackrel { - } { = } } ( x _ { u } ^ { \top } x _ { u } ) x _ { u } x _ { u } ^ { \top } = d _ { u } x _ { u } x _ { u } ^ { \top }$ . Hence,

$$
\mathrm { V a r } ( Y _ { u } \mid d ) = q _ { u } ^ { 2 } \bigl ( d _ { u } A _ { u } - A _ { u } ^ { 2 } \bigr ) \preceq q _ { u } ^ { 2 } d _ { u } A _ { u } ,
$$

where we used $A _ { u } ^ { 2 } \succeq 0 .$ . Since $0 \preceq \operatorname { V a r } ( Y _ { u } \mid d ) \preceq q _ { u } ^ { 2 } d _ { u } A _ { u }$ , we have $\| \mathrm { V a r } ( Y _ { u } \mid d ) \| \le q _ { u } ^ { 2 } d _ { u } \| A _ { u } \|$ for every user �. Hence, the triangle inequality gives:

$$
\Big \| \sum _ { u } \operatorname { V a r } ( Y _ { u } \mid d ) \Big \| \leq \sum _ { u } q _ { u } ^ { 2 } d _ { u } \| A _ { u } \| = \sum _ { u } q _ { u } ^ { 2 } v _ { u } ,
$$

which is (12).

Finally, we find the weights that minimize $F ( q )$ . A user with $d _ { u } = 0$ has $\sigma _ { u } = v _ { u } = 0$ and does not afect $F ( q )$ . Every other user has $v _ { u } > 0 ,$ , since $A _ { u } \succeq 0$ and t $\mathrm { r } ( A _ { u } ) = d _ { u } > 0$ . Over these users, the Cauchy–Schwarz inequality gives:

$$
\begin{array} { r l r } {  { \sum _ { u } q _ { u } \sigma _ { u } = \sum _ { u } ( q _ { u } \sqrt { v _ { u } } ) ( \sigma _ { u } / \sqrt { v _ { u } } ) } } \\ & { } & { \leq \Big ( \sum _ { u } q _ { u } ^ { 2 } v _ { u } \Big ) ^ { 1 / 2 } \Big ( \sum _ { u } \sigma _ { u } ^ { 2 } / v _ { u } \Big ) ^ { 1 / 2 } . } \end{array}
$$

Since $\begin{array} { r } { \sum _ { u } q _ { u } \sigma _ { u } > 0 , } \end{array}$ , this gives $\begin{array} { r } { F ( q ) \ge ( \sum _ { u } \sigma _ { u } ^ { 2 } / v _ { u } ) ^ { - 1 / 2 } } \end{array}$ , with equality if and only if $q _ { u } \sqrt { v _ { u } }$ is proportional to $\sigma _ { u } / \sqrt { v _ { u } }$ , that is, $q _ { u } \propto \sigma _ { u } / v _ { u } =$ $( d _ { u } - 1 ) / \lVert A _ { u } \rVert$ □

Table 3: Selected hyperparameters of each method on each dataset.
<table><tr><td>Model</td><td>Hyperparameter</td><td>books</td><td>cds</td><td>gowalla</td><td>ml-1m</td><td>retail</td><td>steam</td></tr><tr><td>WMF</td><td>number of factors</td><td>128</td><td>128</td><td>64</td><td>128</td><td>128</td><td>64</td></tr><tr><td></td><td>regularization</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td>LightGCN</td><td>learning rate</td><td>0.001</td><td>0.01</td><td>0.01</td><td>0.001</td><td>0.001</td><td>0.001</td></tr><tr><td></td><td>number of layers</td><td>2</td><td>2</td><td>2</td><td>2</td><td>3</td><td>3</td></tr><tr><td>SLIMElastic</td><td>regularization</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td></td><td>L1 ratio</td><td>0.01</td><td>0.01</td><td>0.001</td><td>0.01</td><td>0.01</td><td>0.001</td></tr><tr><td>ADMMSLIM</td><td>L1 regularization</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>5</td><td>1</td></tr><tr><td></td><td>L2 regularization</td><td>50</td><td>200</td><td>500</td><td>500</td><td>500</td><td>500</td></tr><tr><td>NCEPLRec</td><td>regularization</td><td>1000</td><td>1000</td><td>10000</td><td>10000</td><td>10000</td><td>10000</td></tr><tr><td></td><td>rank</td><td>400</td><td>400</td><td>400</td><td>400</td><td>400</td><td>400</td></tr><tr><td>RecVAE</td><td>latent dimension</td><td>200</td><td>200</td><td>200</td><td>200</td><td>200</td><td>200</td></tr><tr><td></td><td>KL scale</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.0025</td><td>0.0025</td><td>0.005</td></tr><tr><td>LDA</td><td>number of iterations</td><td>20</td><td>100</td><td>20</td><td>100</td><td>20</td><td>20</td></tr><tr><td></td><td>prior on topic proportions</td><td>0.5</td><td>0.5</td><td>0.05</td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td>ETM</td><td>prior on topics</td><td>1/K</td><td>1/K</td><td>0.1</td><td>0.1</td><td>1/K</td><td>1/K</td></tr><tr><td></td><td>learning rate</td><td>0.01</td><td>0.005</td><td>0.01</td><td>0.005</td><td>0.005</td><td>0.002</td></tr><tr><td></td><td>hidden size</td><td>800</td><td>400</td><td>800</td><td>800</td><td>800</td><td>400</td></tr><tr><td></td><td>dropout probability</td><td>0.5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.5</td></tr><tr><td></td><td>KL weight</td><td>0.2</td><td>0.5</td><td>0.1</td><td>0.1</td><td>0.5</td><td>0.5</td></tr><tr><td>ECRTM</td><td>learning rate</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td></tr><tr><td></td><td>temperature</td><td>0.2</td><td>0.2</td><td>0.1</td><td>0.2</td><td>0.1</td><td>0.1</td></tr><tr><td></td><td>ECR weight</td><td>20</td><td>20</td><td>20</td><td>50</td><td>200</td><td>200</td></tr><tr><td></td><td>KL weight</td><td>0.2</td><td>0.1</td><td>0.2</td><td>0.5</td><td>0.2</td><td>0.5</td></tr><tr><td>Mult-VAE</td><td>learning rate</td><td>0.003</td><td>0.003</td><td>0.003</td><td>0.003</td><td>0.003</td><td>0.003</td></tr><tr><td></td><td>KL weight</td><td>0.2</td><td>0.1</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td></td><td>dropout probability</td><td>0.5</td><td>0.5</td><td>0</td><td>0.5</td><td>0</td><td>0</td></tr><tr><td>NMF</td><td>regularization</td><td>0</td><td>0</td><td>0.01</td><td>0.01</td><td>0.1</td><td>0.01</td></tr><tr><td></td><td>L1 ratio</td><td>0</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td></tr><tr><td>NMF-ppSPA</td><td>initialization</td><td>nndsvda</td><td>nndsvd</td><td>nndsvd</td><td>nndsvd</td><td>nndsvd</td><td>nndsvd</td></tr><tr><td></td><td>regularization</td><td>0.01</td><td>0.01</td><td>0.1</td><td>0</td><td>0</td><td>0.1</td></tr><tr><td></td><td>L1 ratio</td><td>0.5</td><td>0.5</td><td>1</td><td>0</td><td>0</td><td>0</td></tr><tr><td></td><td>initialization</td><td>nndsvda</td><td>nndsvda</td><td>nndsvd</td><td>nndsvd</td><td>nndsvd</td><td>nndsvda</td></tr><tr><td>EASE</td><td>λ</td><td>50</td><td>50</td><td>100</td><td>1000</td><td>500</td><td>1000</td></tr><tr><td>RW</td><td>λ</td><td>250</td><td>1000</td><td>50</td><td>50</td><td>50</td><td>500</td></tr><tr><td>Symmetric</td><td>λ</td><td>1000</td><td>100</td><td>1000</td><td>50</td><td>50</td><td>100</td></tr><tr><td>DAN</td><td>λ</td><td>1000</td><td>250</td><td>50</td><td>50</td><td>50</td><td>500</td></tr><tr><td></td><td>α</td><td>0.25</td><td>0.25</td><td>0</td><td>0.5</td><td>0.5</td><td>0.25</td></tr><tr><td>RDLAE</td><td> $\beta$  (the user exponent)</td><td>0.2</td><td>0.5</td><td>1</td><td>0</td><td>0</td><td>0.2</td></tr><tr><td></td><td>λ</td><td>50</td><td>50</td><td>100</td><td>250</td><td>500</td><td>500</td></tr><tr><td></td><td>pǐs</td><td>0.5</td><td>0.3</td><td>0.9</td><td>0.3</td><td>0.1</td><td>0.7</td></tr><tr><td></td><td></td><td>0.5</td><td>0.1</td><td>0.1</td><td>0.5</td><td>0.25</td><td>0.1</td></tr><tr><td>TEA</td><td>α</td><td>0.25</td><td>0.125</td><td>0</td><td>0</td><td>0.125</td><td>0.125</td></tr><tr><td></td><td></td><td>0.7</td><td>0.7</td><td>0.7</td><td>0.3</td><td>0</td><td>0.7</td></tr><tr><td></td><td>pλ</td><td>250</td><td>250</td><td>50</td><td>250</td><td>50</td><td>250</td></tr></table>

## H Hyperparameters

We select the hyperparameters of every method on each dataset by NDCG@10 on the validation items. Mult-VAE [25], ETM [11], and ECRTM [44] also keep the training epoch with the highest validation NDCG@10. Table 3 reports the selected hyperparameters of every method. For the linear autoencoders, � is the regularization parameter, � is the item exponent, and � is the dropout probability. For RDLAE, � is the bound on the diagonal entries of �.

## I Selection of the example topics

Among all the topics, we choose to display in Table 2 the ones with both high coherence and a large efective number of users, defined below. Specifically, let $\widehat { \beta }$ be the topics of TEA fitted on the training items of all users, let $T _ { k }$ be the top ten items of $\widehat { \beta } _ { : , k : }$ , and let $\mathrm { T C } _ { k }$ be the average NPMI over the pairs in $T _ { k }$ , with $\eta _ { i }$ and $\eta _ { i j }$ taken over all users. For each user �, let $\omega _ { u k }$ be the fraction of the co-occurrences of pairs of items in $T _ { k } ,$ , counted over the training items of all users, that come from user �. The efective number ofusers of topic � is the reciprocal Simpson index $\begin{array} { r } { n _ { k } ^ { \mathrm { e f f } } = ( \sum _ { u } \omega _ { u k } ^ { 2 } ) ^ { - 1 } } \end{array}$ , which equals 1 when a single user provides all the co-occurrences and equals � when � users provide equal shares. We display the five topics with the largest values of the following score:

$$
\mathrm { T C } _ { k } \cdot \frac { \log n _ { k } ^ { \mathrm { e f f } } } { \log n } .
$$

Here, the factor log $n _ { k } ^ { \mathrm { e f f } }$ log � is 0 when a single user provides all the co-occurrences and 1 when all � users provide equal shares.