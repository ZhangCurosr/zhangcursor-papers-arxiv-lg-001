# Alternatives to Shallow Autoencoders for Collaborative Filtering

Mario Mallea<sup>[0009−0005−8295−088X]</sup>, Àngela Nebot<sup>[0000−0002−4621−8262]</sup>, and Francisco Mugica<sup>[0000−0003−2843−0427]</sup>

Universitat Politècnica de Catalunya, Soft Computing Research Group at the Intelligent Data Science and Artificial Intelligence Research Center, Barcelona, Spain mario.carlos.mallea@upc.edu

Abstract. Collaborative filtering (CF) is a cornerstone of recommender systems and plays a relevant role in many modern applications. CF uses user-item interaction data to discover future preferences. Although deep learning models have shown promise in CF, Embarrassingly Shallow Autoencoders for Sparse Data (EASE) has gained attention for its outstanding ranking accuracy provided by its closed-form solution. EASE relies primarily on relationships among items to fit a full-rank high-dimensional linear mapping. We hypothesize that this design limits its capacity to capture similar but not equivalent fine-grained user relationships, consequently limiting its recommendation accuracy. This paper introduces an alternative formulation based on EASE that takes advantage of useruser information. Furthermore, we propose a hybrid model that combines both user-user and item-item adjacency distributions.

Our experiments reveal that the proposed models outperform EASE on well-known recommendation benchmarks, highlighting the significance of including the user alternative in shallow autoencoder studies for CF.

Keywords: Autoencoder · recommender systems · collaborative filtering.

## 1 Introduction

In our digital age, filtering valuable information is a crucial challenge, as it helps users navigate the overwhelming amount of available content. Recommender systems play an important role in addressing this issue by identifying and prioritizing items that align with user preferences. Collaborative filtering (CF) [21] methods are particularly noteworthy, as they focus on learning patterns and relationships from sparse historical user-item interactions itself. These interactions are usually ratings, clicks, check-ins, or purchases. The main objective of these types of systems is to provide personalized experiences to users [9, 15, 19].

In recent years, eficient machine learning methods have gained significant attention due to their outstanding results through innovative approaches such as infinitely wide random CNN [1], broad learning systems [2, 10], and highdimensional or full-rank shallow linear approximations [27]. These methods often employ a ridge regression formulation, which provides a closed-form solution, typically using the mathematical concept of pseudo-inverse [2,8]. The closed form simplifies model deployment and eliminates the need for extensive and iterative optimization, making such methods especially well suited for fast and efective CF tasks. One of the most representative methods in this domain is EASE [26]. EASE demonstrates exceptional efectiveness through a closed-form solution for a high-dimensional linear mapping. The performance of EASE relies on the Gram matrix or item-item adjacency distribution derived from the number of co-users consuming items. However, when datasets contain more users than items, more granular insights can be extracted by shifting to similar but not equivalent useruser relationships based on the number of co-liked items among users. Consequently, the resolution or dimensionality of the full-rank approximation becomes a critical factor in determining the most suitable approach, particularly because the learned linear mapping has the dimensionality of the total number of items or users. Indeed, we hypothesize that depending on the distribution of users and items, the EASE design could limit its ability to capture user relationships. For instance, we could have users with completely diferent behaviors, but their consumed items could still be linked through other users. Thus, the user Gram matrix reflecting the user similarities will mark zero interaction between them, whereas the item Gram matrix reflecting the item similarities might identify niches in terms of concepts, but this item co-occurrence will mask the lack of user interaction. As a result, in certain scenarios, a user-user approach will distinguish user communities that an item-item approach may fail to diferentiate.

This research addresses the limitations of EASE by introducing alternative solutions that leverage both user-user and item-item information. Specifically, we propose a user-centric variant of EASE (u-EASE) and explore a novel hybrid approach that combines both perspectives (u-EASE-i). Despite the straightforward nature and fast training of the closed-form solutions of u-EASE and u-EASEi, they can improve EASE and even achieve state-of-the-art recommendation results.

Our evidence is provided in terms of computational eficiency and performance in short and large rankings. We compared our contributions with wellknown competitive models, including the most important deep, graph, variational, and nonlinear neural networks, as well as other recent shallow autoencoder approaches. The findings from our experiments indicate that the proposed models surpass EASE in performance on established recommendation benchmarks, for instance on Yelp2018 [28], and exhibit state-of-the-art outcomes on ML-1M [5]. This underscores the significance of integrating the user alternative within the framework of shallow autoencoders employed in collaborative filtering literature.

The remainder of this paper is organized as follows. In Section II, we review related work on shallow autoencoders in the context of collaborative filtering. Section III provides the mathematical preliminaries necessary to understand EASE and its closed-form solution. In Section IV, we introduce a generalization of EASE and our proposed models, u-EASE and u-EASE-i, and we describe their connections. Section V describes the experimental setup, including the datasets, evaluation metrics, and baseline comparisons. In Section VI, we present and analyze our results, highlighting the advantages of our approaches over EASE and their comparison with other recent recommenders. Finally, in Section VII, we conclude the study and present potential directions for future research.

## 2 Related Work

Collaborative filtering has been studied through techniques such as matrix factorization [19], graphs [6, 22, 28], deep neural networks [7, 14] and autoencoders [8,20,26]. Recently, lightweight and eficient methods with closed-form solutions have garnered attention due to their potential to reduce computational overhead while achieving competitive or even state-of-the-art ranking accuracy through simple linear but high-dimensional approximations of the implicit feedback matrix [25].

EASE [26] introduced a full-rank autoencoder with a closed-form solution based on ridge regression, which requires tuning only one hyper parameter. EASE learns to reconstruct the interactions among users and items by learning a linear mapping among items, in particular, these trainable weights are defined by the Gram matrix or adjacency matrix among items.

Recent works study EASE properties in diferent contexts. [13] analyses theoretical insights of the relationships between linear autoencoders and matrix factorization. Also, [12] includes additional side information or metadata about items. The interpretability of EASE from the point of view of graphs was explored in [23], and the personality bias concerns behind their recommendations [16]. In addition, [24] studies the scalability of EASE and develops methods to parallelize it. Finally, its application to the discovery of sub communities is discussed in [4].

In terms of the recommender model, ∞-AE [20] employs the neural tangent kernel in a kernelized ridge regression to form a closed-form solution. Theoretically, this is equivalent to training an infinitely wide autoencoder for an infinite number of stochastic gradient descent steps. Unlike EASE, this model relies on the construction from the user-user perspective. SVD-AE [8] is an extension of EASE, which adopts a fast randomized truncated singular value decomposition as a low-rank inductive bias. Following this approach, they enhance the noise robustness, computation time, and accuracy, particularly in short rankings, as a consequence of preserving the top singular vectors and values.

On the other hand, graph convolutional networks have gained popularity for their ability to learn representations that capture complex relationships within the graph structure. These methods represent users and items as nodes in a graph, with edges representing interactions. Importantly, Graph filter based collaborative filtering (GF-CF) [22] developed a general graph convolution-based framework for CF, particularly they combined a linear filter and ideal low-pass filters, resulting a closed-form solution, that utilizes the item-item Gram matrix or adjacency of the graph and its top singular vectors version.

## 3 Preliminaries

We assume implicit feedback data represented by a sparse matrix $R \in \mathbb { R } ^ { | U | \times | I | }$ 2 where |U| and |I| are the number of users and items, respectively. $R _ { u , i } = 1$ indicates a user-item interaction for all $i \in I _ { u } ^ { + }$ (the set of consumed items by a user).

The EASE model uses an item-item weight matrix $W \in \mathbb { R } ^ { | I | \times | I | }$ to predict scores for user u and item i as: $R _ { u , \cdot } W _ { \cdot , i }$ . To avoid a trivial solution, the diagonal of $W$ is constrained to zero. The weights W are learned by minimizing the regularized Frobenius norm $( \| Z \| _ { F } ^ { 2 } = \mathrm { t r } \big ( Z ^ { t } Z \big ) \big )$ of the approximation:

$$
\operatorname* { m i n } _ { W } \quad \| R - R W \| _ { F } ^ { 2 } + \lambda _ { i } \| W \| _ { F } ^ { 2 }\tag{1}
$$

$$
\begin{array} { r l } { \mathrm { s . t . } } & { { } D i a g ( W ) = 0 } \end{array}\tag{2}
$$

We introduce the sufix i or u to denote an element which corresponds with learnable weights in the space of items or users. $\lambda _ { i }$ is the regularization parameter.

The minimum of (1) is given by:

$$
W _ { i } = ( G _ { i } + \lambda _ { i } I ) ^ { - 1 } G _ { i } ,\tag{3}
$$

where the item-item Gram matrix is $G _ { i } = R ^ { t } R$ . However, considering the constraint (2), to find the minimum, we define $\gamma$ as the vector of Lagrangian multipliers representing the zero diagonal constrain. The new minimum is given by:

$$
W = ( G _ { i } + \lambda _ { i } I ) ^ { - 1 } ( G _ { i } - D i a g M a t ( \gamma ) ) .\tag{4}
$$

We still have to find $\gamma .$ . Let $P _ { i } = ( G _ { i } + \lambda _ { i } I ) ^ { - 1 }$ , and we can rewrite the solution as:

$$
W = P _ { i } ( P _ { i } ^ { - 1 } - \lambda _ { l } I - D i a g M a t ( \gamma ) )\tag{5}
$$

$$
W = I - P _ { i } \ D i a g M a t ( \hat { \gamma } )\tag{6}
$$

where $\hat { \gamma } = \lambda _ { i } { \bf 1 } + \gamma$ , with 1 denoting a vector of ones. Then $\hat { \gamma }$ is determined using the following:

$$
0 = D i a g ( W ) = { \bf 1 } - D i a g ( P _ { i } ) \odot \hat { \gamma } ,\tag{7}
$$

where $\odot$ denotes the product element wise.

Finally, the optimal closed form solution for EASE is given by:

$$
W = I - P _ { i } D i a g M a t \left( { \bf 1 } \oslash D i a g ( P _ { i } ) \right) ,\tag{8}
$$

where $\oslash$ denotes the element wise division.

## 4 Proposal

We formulated an optimization problem to obtain an approximation based on both item-item and user-user perspectives. This approach preserves the oneshoot learning property of the closed-form solutions, while it makes both alternatives compatible. Fig. 1 illustrates the idea of the approximation from the item’s side (EASE), user’s side (u-EASE), and both (u-EASE-i). Although the approach appears simple, some mathematical details are needed to preserve the good features of EASE and well-define all terms in the alternatives.

![](images/e1d7104c10121ec51c28518f333a11d5940d08aabab60e1a7fef1fd4545351a0.jpg)  
Fig. 1: Illustration of the model idea. EASE reconstructs the feedback matrix from the side of the items. We proposed an approximation from the user side with u-EASE. u-EASE-i combines both.

We are going to introduce a generalization of (3), then our approaches and EASE can be seen as special cases. Given matrices $Y \in \mathbb { R } ^ { n \times m } , A \in \mathbb { R } ^ { n \times d }$ and $B \in \mathbb { R } ^ { d \times m }$ . We defined the following optimization problem:

$$
\operatorname* { m i n } _ { W } \| Y - A W B \| _ { F } ^ { 2 } .
$$

To obtain the minimum W, we diferentiate the norm (denoted f):

$$
\frac { \partial } { \partial W } f ( W ) = - 2 A ^ { t } Y B ^ { t } + 2 A ^ { t } A W B B ^ { t } .
$$

Setting the derivative to zero, we find the closed form for W:

$$
W = ( A ^ { t } A ) ^ { - 1 } A ^ { t } Y B ^ { t } ( B B ^ { t } ) ^ { - 1 } .\tag{9}
$$

In fact, mathematically, this is a special case of a more general theorem of generalized inverses, known as the pseudo inverse of Moore-Penrose [18].

## 4.1 u-EASE

Note that we recover EASE by doing $Y = A = R ,$ and $B = I ,$ and considering the extra regularization of $D i a g ( W ) = 0 $ . Equivalently, we can perform the symmetric version from the left instead of the right, that is, $Y = B = R $ $A = I .$ and the regularization $D i a g ( W ) = 0 $ . In other words, we introduced a novel user-centric variant of the EASE model, termed u-EASE, which symmetrically mirrors the item-centric approach. This allows us to exploit user-user relationships similar to how EASE leverages item-item interactions. Therefore, the complete formulation is:

$$
\operatorname* { m i n } _ { W } \quad \| R - W R \| _ { F } ^ { 2 } + \lambda _ { u } \| W \| _ { F } ^ { 2 }\tag{10}
$$

$$
\mathrm { s . t . } \quad \mathrm { d i a g } ( W ) = 0 ,\tag{11}
$$

where $\lambda _ { u }$ denotes the regularization parameter. Equivalent to EASE, the closedform solution without considering the constraint, for $W \in \mathbb { R } ^ { | U | \times | \dot { U } | }$ is derived using the user-user Gram matrix $G _ { u } = R R ^ { t }$ :

$$
W _ { u } = ( G _ { u } + \lambda _ { u } I ) ^ { - 1 } G _ { u } .\tag{12}
$$

To obtain the complete solution, we must incorporate the Lagrangian multipliers $\gamma .$ Let $P _ { u } = ( G _ { u } + \lambda _ { r } I ) ^ { - 1 }$ . Following the same steps as EASE, we can find the final closed form:

$$
W = ( G _ { u } - D i a g M a t ( \gamma ) ) P _ { u }\tag{13}
$$

$$
W = ( P _ { u } ^ { - 1 } - \lambda _ { u } I - D i a g M a t ( \gamma ) ) P _ { u }\tag{14}
$$

$$
W = I - D i a g M a t ( \hat { \gamma } ) \ P _ { u }\tag{15}
$$

$$
W = I - D i a g M a t \left( { \bf 1 } \oslash D i a g ( P _ { u } ) \right) \ P _ { u }\tag{16}
$$

The last step is due to the commutative property of the element-wise product, so the computation of $\hat { \gamma }$ is equivalent to step (7). Fig. 1 shows the diference between EASE and u-EASE, where we can visualize the efect of the resolution. Indeed, based on the amount of users and items of the problem, the decision of the alternative defines the number of parameters to learn, because we are going to fit the best linear mapping in the space of such dimensionality. Therefore, we could expect an impact on the model’s accuracy, because the corresponding Gram matrix represents more grained relationships. Preliminary, we should take EASE if we take more items and u-EASE otherwise. However, experimentally, we found that u-EASE can enhance EASE even with fewer users than items.

## 4.2 u-EASE-i

We propose a combined optimization framework that jointly minimizes reconstruction error for both user-user and item-item interaction matrices. Considering the generalization, if we take $Y = R ^ { t } R R ^ { t }$ , and $A = B = R ^ { t }$ , then we are directly learning to reconstruct the matrix R. The optimization problem is as follows:

$$
\operatorname* { m i n } _ { W } \| R ^ { t } R R ^ { t } - R ^ { t } W R ^ { t } \| _ { F } ^ { 2 } + R e g ( W ) ,\tag{17}
$$

$$
R e g ( W ) = \lambda _ { i } \| R ^ { t } W \| _ { F } ^ { 2 } + \lambda _ { u } \| W R ^ { t } \| _ { F } ^ { 2 } + \lambda _ { u } \lambda _ { i } \| W \| _ { F } ^ { 2 } .\tag{18}
$$

where the regularization terms are mathematically necessary to achieve the factorization. See the details in Appendix 8. The closed form for $W \in \mathbb { R } ^ { | U | \times | I | }$ is the following simple expression:

$$
W _ { u , i } = W _ { u } R W _ { i } = P _ { u } G _ { u } R G _ { i } P _ { i }\tag{19}
$$

This allows our model to capture a richer representation of user preferences and item relationships. Fig. 1 shows the combination of the learned weights $W _ { i } , W _ { u } ,$ and $W _ { u , i }$ . We observed similar results with other possible alternatives such as $R \approx R W R$ or $R R ^ { t } R \approx R W R$ . To end, note that Equations 3, 12, and 19 provide deterministic closed-form solutions to the recommendation problem.

## 4.3 Computational Cost

Training EASE, u-EASE, or u-EASE-i requires the expensive computation of the Gram matrix, however, as the closed-form solution depends on the Gram matrices, this computation can be performed on a big-data pre-processing system prior to the training phase [26]. The computational complexity of the training is determined by the inversion of the matrix, which is $O ( X ^ { 3 } )$ using a basic approach, where X will be |U| or |I|. Thus, depending on the number of users |U| or items |I|, EASE or u-EASE can be prioritized. On the other hand, u-EASE-i needs both Gram matrices, thus its computational complexity will be defined by the possible dominant term.

In terms of scalability or memory concerns, future work could extend the eficient techniques already applied to scale EASE in our proposals, such as using lightweight methods to estimate a truncated singular value decomposition [8] or using a sparse approximate inversion framework [24].

## 5 Experimental setting

## 5.1 Data

We evaluated our approaches in three datasets with more than 1 M interactions, their statistics are presented in Tab. 1. Gowalla [3] is a location-based social networking, where users share their location by checking-in. Yelp-2018 connects users with local businesses [28]. ML-1M [5] comes from a website that helps users find movies.

## 5.2 Models used for evaluation

We compared the performance of our proposed methods against some of the most relevant and recent recommendation models. The most related, EASE [26], and some other recent closed-form approaches, such as GF-CF [22],∞−AE [20] and SVD-AE [8]. Additionally, we consider four important models: MF-BPR [19] is a matrix factorization model that leverages the Bayesian Personalized Ranking loss. NeuMF [7] combines MF (Matrix Factorization) embeddings of users and items with a nonlinear deep neural network. LightGCN [6] is a simplified graph convolutional network, which is designed to retain only the essential neighborhood aggregation component while removing feature transformation and nonlinear activation. This linear propagation framework computes the final embeddings as a weighted sum across layers, significantly improving eficiency and recommendation performance. MultVAE [14] is a variational autoencoder model, which uses a multinomial likelihood to enhance the modeling capacity beyond linear factor models.

Table 1: Statistics of datasets. We call the resolution of the problem to $\rho =$ $| U | / | I |$
<table><tr><td>Dataset</td><td>#User #Item #Interaction Density ρ</td></tr><tr><td>Gowalla 29,858 40,981</td><td>1,027,370 0.0008 0.73</td></tr><tr><td>Yelp2018 31,66838,048</td><td>1,561,406 0.0013 0.83</td></tr><tr><td>ML-1M 6,0403,706</td><td>1,000,209 0.04471.63</td></tr></table>

For a fair comparison, we used the same train, validation, and test splits [6,20,22] (80%/10%/10%). Consequently, we adopted a standardized evaluation procedure used for all those methods. Specifically, we followed the implementation of SVD-AE [8]. Our code is publicly available: https://github.com/ MariodotR/Alternatives\_EASE.git.

## 5.3 Metrics

We utilized the following metrics to evaluate the model’s ability to recommend relevant items and position them highly in the ranked list. Following [8], we evaluated the performance in short and large rankings, taking the top K = 10 and K = 100.

The hit rate or recall [17] measures the proportion of relevant items that are successfully recommended within the top ranked list.

$$
H R @ K = \frac { 1 } { | U | } \sum _ { u \in U } \frac { | I _ { u } ^ { + } \cap \hat { R } _ { u } | } { m i n ( | I _ { u } ^ { + } | , K ) } ,
$$

where $\hat { R } _ { u }$ denotes the set of items recommended to the user.

– The normalized discounted cumulative gain [17] evaluates the ranking quality by considering the positions of relevant items in the list.

$$
n D C G @ K = \frac { 1 } { | U | } \sum _ { u \in U } \frac { D C G _ { u } } { I D C G _ { u } }
$$

$$
D C G _ { u } = \sum _ { i = 1 } ^ { K } \frac { \mathbb { 1 } ( \hat { R } _ { u , i } \in I _ { u } ^ { + } ) } { l o g _ { 2 } ( i + 1 ) } ,
$$

where IDCG represents the best possible DCG at K per user, and the indicator function 1 is one if and only if the recommended item was consumed by the user.

– The propensity-scored precision [11] measures the precision of the recommendations considering a propensity (or popularity) correction term $\phi$ (see the details in [20]).

$$
P S P @ K = \frac { 1 } { | U | } \sum _ { u \in U } \frac { u P S P _ { u } } { m P S P _ { u } } ,
$$

$$
u P S P _ { u } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \frac { \mathbb { 1 } ( \hat { R } _ { u , i } \in I _ { u } ^ { + } ) } { \phi ( \hat { R } _ { u , i } ) } ,
$$

where $m P S P$ represents the best possible $u P S P$ at K per user.

## 6 Results and discussion

In u-EASE, the only hyper parameter to tune is the regularization parameter $\lambda _ { u } .$ , similarly to EASE, we searched for a large regularization to avoid the trivial solution, we tuned it using a large range grid search $[ 1 0 ^ { 1 } , 1 0 ^ { 2 } , 1 0 ^ { 3 } , 1 0 ^ { 4 } , 1 0 ^ { 5 } , 1 0 ^ { 6 } ]$ in the validation phase. Fig. 2 shows the results, where we found $1 0 ^ { 3 }$ as the optimal regularization.

In u-EASE-i, we defined two regularization hyper parameters to tune $\lambda _ { i }$ and $\lambda _ { u }$ . We searched both in the grid search $[ 1 0 ^ { 1 } , \bar { 1 0 ^ { 2 } } , \bar { 1 0 ^ { 3 } } , 1 0 ^ { 4 } ]$ , exploring sixteen options. Fig. 3 shows the results, the best $( \lambda _ { i } , \lambda _ { u } )$ were $( \mathrm { \bar { 1 } 0 ^ { 2 } , 1 0 ^ { 2 } } ) , ( 1 0 ^ { 3 } , 1 0 ^ { 1 } )$ , and $( 1 0 ^ { 2 } , 1 0 ^ { 3 } )$ for Gowalla, Yelp2018, and ML-1M, respectively. We observed robust results, where $( 1 0 ^ { 1 } , 1 0 ^ { 3 } )$ is usually competitive,in addition to an automatic correction behavior, for instance, when $\lambda _ { i }$ is larger, $\lambda _ { u }$ compensates with lower values, and vice versa.

Tables 2,3, and 4 present the results for the Gowalla, Yelp2018, and ML-1M datasets, respectively. Our alternative formulations, u-EASE and u-EASE-i, demonstrated improvements over EASE on Yelp2018 and ML-1M. Specifically, on Yelp2018, u-EASE and u-EASE-i achieved a percentage improvement of 2.5% and 3.7% in terms of nDCG@100, respectively. On the ML-1M dataset, the enhancements were more pronounced, with u-EASE and u-EASE-i showing improvements of 6.0% and 6.3% in terms of nDCG@100, respectively. On ML-1M, where we have a positive resolution $\rho ,$ we achieved state-of-the-art results in short and large rankings, achieving even better results than other recent improvements over EASE, such as ∞-EASE and SVD-AE. Our alternatives also improve EASE’s PSP@10, demonstrating that these enhancements do not overfit popularity bias. These results highlight the efectiveness of our proposed models in improving the ranking accuracy of the classic approach by simply selecting simple alternatives.

On Yelp2018 and Gowalla the best model is diferent based on the size of the ranking, for large rankings GF-CF is the best, while for short rankings is

SVD-AE. Despite this, on Yelp2018 our proposals perform better than EASE for long rankings, and particularly with u-EASE-i in terms of HR@10. Besides, we achieved better metrics than MF-BPR, NeuMF, MultVAE, LightGCN (except by HR@100), and ∞-AE (except by PSP@10). On Gowalla, u-EASE still performs better than NeuMF, MultVAE, ∞-AE (except by PSP@10).

In general, the relative performance gains seem to correlate with the resolution $\rho ,$ suggesting that user-user information becomes more impactful when the number of users is larger than the number of items. Although the results on Yelp2018 show that $\rho > 1$ is not exclusively necessary. Furthermore, in that scenario, by incorporating both user-user and item-item information, u-EASE-i achieves a richer representation of relationships within the data, leading to more accurate recommendations.

Finally, computational eficiency is an important concern in real-world applications. Therefore, we provide evidence to support this concern. We empirically measured the total wall clock time, including pre-processing (Gram matrix), its inversion, and evaluation. The Tab 5 shows the comparison among the alternative approaches. The machine was equipped with 16 VCPUs and 8 GB of system memory per vCPU. All alternatives took a few minutes to complete the entire experiment, a great feature considering the well-known timescale required to train deep learning methods [8]. We observed that the u-EASE alternative is faster than EASE when $\rho < 1$ , since we are considering a lightweight Gram matrix. Therefore, the performance drop in a data condition, such as Gowalla could be accepted, considering limited resources or the better computational eficiency of u-EASE. On the other hand, u-EASE-i consumed more computational resources, since it needs both Gram matrices. In particular, our state-of-the-art results on ML-1M were achieved in seconds.

![](images/e603f5a70f2ef7de7794ebb826bc62d2f3ad4f00b0ad956dfee879a3b0e7d541.jpg)

![](images/0eb91844dff10c88c636adeb948226f4e5f4f8b076904007b7e04ab9b227f631.jpg)

![](images/774f806ffdccd826a00d7ede779b199698ab944910a570ae79a8ead7487fd766.jpg)  
Fig. 2: Regularization’s results for the u-EASE model in the validation set.

## 7 Conclusion and future work

EASE has gained significant attention in the recommender systems community due to its simplicity, eficiency, and strong ranking accuracy. Those properties have been transferred to other important methods based on graph neural networks, singular value decomposition, or neural tangent kernel. However, other alternative high dimensional shallow linear autoencoders have not been explored.

![](images/667378275e09d7fcb08890743f13a8599e3b991f3565cd5191040e8999d0fe1b.jpg)

![](images/d78f394614e27d7fd183c7714205d0c724e74d41a335f7cda6e4db323f0a7499.jpg)

![](images/dec858f4dc2b146443792f1fe753c9efd69ff3736bc412abccb210f67ed50156.jpg)  
Fig. 3: Regularization’s results for the u-EASE-i model in the validation set.

Table 2: Gowalla Results
<table><tr><td>Method</td><td colspan="4">HR@10 HR@100 NDCG@10 NDCG@100 PSP@10</td></tr><tr><td>MF-BPR</td><td>12.08</td><td>32.84</td><td>12.09</td><td>18.44</td><td>1.92</td></tr><tr><td>NeuMF</td><td>8.87</td><td>27.25</td><td>8.27</td><td>13.96</td><td>1.37</td></tr><tr><td>LightGCN</td><td>14.00</td><td>37.40</td><td>13.50</td><td>21.04</td><td>2.26</td></tr><tr><td>GF-CF</td><td>14.08</td><td>38.84</td><td>13.71</td><td>21.25</td><td>2.47</td></tr><tr><td>MultVAE</td><td>11.88</td><td>33.56</td><td>11.30</td><td>18.11</td><td>2.09</td></tr><tr><td>∞-AE</td><td>11.77</td><td>34.20</td><td>10.84</td><td>17.97</td><td>2.02</td></tr><tr><td>SVD-AE</td><td>14.40</td><td>37.34</td><td>13.94</td><td>21.15</td><td>2.48</td></tr><tr><td>EASE</td><td>13.67</td><td>35.74</td><td>13.15</td><td>20.08</td><td>2.31</td></tr><tr><td>u-EASE</td><td>12.10</td><td>34.22</td><td>11.39</td><td>18.33</td><td>1.95</td></tr><tr><td>u-EASE-i</td><td>11.47</td><td>34.15</td><td>10.56</td><td>17.80</td><td>2.04</td></tr></table>

In this research, we introduced two alternative approaches to EASE by leveraging user-user information, instead of the original item-item based method. Through comprehensive experiments on three benchmark datasets, we showed that our proposed methods can outperform EASE, particularly but not limited to datasets with a higher user-to-item ratio, notably achieving state-of-the-art results in ML-1M. Our proposals ofer rapid deployment and streamlined, one shoot learning. Our research indicates that incorporating user-user relationships may improve the accuracy of collaborative filtering recommendations, particularly when dealing with a larger number of users than items.

Future work could explore the formulation of the dual-side factorization idea behind our proposal in a randomly generated low-dimensional embedding for users and items. In addition, we note that graph regularization is possible in all alternatives. We planned to investigate the impact of data characteristics such as sparsity or resolution over the decision of the alternatives, and develop robust strategies for selecting the most appropriate model. Finally, transferring methods of interpretability, local communities, or scalability created for EASE are relevant next steps.

Table 3: Yelp2018 Results
<table><tr><td>Method</td><td colspan="4">HR@10 HR@100 NDCG@10 NDCG@100 PSP@10</td></tr><tr><td>MF-BPR</td><td>3.86</td><td>16.97 3.70</td><td>8.50</td><td>0.34</td></tr><tr><td>NeuMF</td><td>3.57</td><td>14.39 3.08</td><td>7.16</td><td>0.27</td></tr><tr><td>LightGCN</td><td>4.32</td><td>19.01 4.19</td><td>9.33</td><td>0.39</td></tr><tr><td>GF-CF</td><td>4.87</td><td>20.86 4.66</td><td>10.53</td><td>0.44</td></tr><tr><td>MultVAE</td><td>4.31</td><td>18.75 4.10</td><td>9.37</td><td>0.43</td></tr><tr><td>∞-AE</td><td>4.62</td><td>18.33</td><td>4.48 9.02</td><td>0.43</td></tr><tr><td>SVD-AE</td><td>4.90</td><td>19.79</td><td>4.74 10.22</td><td>0.45</td></tr><tr><td>EASE</td><td>4.65</td><td>17.74</td><td>4.55 9.37</td><td>0.42</td></tr><tr><td>u-EASE</td><td>4.63</td><td>18.55</td><td>4.50 9.60</td><td>0.40</td></tr><tr><td>u-EASE-i</td><td>4.68</td><td>18.78</td><td>4.56 9.72</td><td>0.40</td></tr></table>

Table 4: ML-1M Results
<table><tr><td>Method</td><td colspan="4">HR@10 HR@100 NDCG@10 NDCG@100 PSP@10</td></tr><tr><td>MF-BPR</td><td>28.49</td><td>57.21</td><td>29.84 39.47</td><td></td><td>2.89</td></tr><tr><td>NeuMF</td><td>27.95</td><td>54.24</td><td>29.36</td><td>37.98</td><td>2.75</td></tr><tr><td>LightGCN</td><td>29.07</td><td>57.62</td><td>30.33</td><td>39.95</td><td>3.17</td></tr><tr><td>GF-CF</td><td>28.60</td><td>59.10</td><td>30.32</td><td>42.00</td><td>3.13</td></tr><tr><td>MultVAE</td><td>27.86</td><td>57.67</td><td>28.44</td><td>39.34</td><td>3.16</td></tr><tr><td>∞-AE</td><td>31.15</td><td>60.75</td><td>32.27</td><td>42.54</td><td>3.22</td></tr><tr><td>SVD-AE</td><td>31.79</td><td>59.33</td><td>33.55</td><td>42.57</td><td>3.22</td></tr><tr><td>EASE</td><td>30.43</td><td>57.74</td><td>31.90</td><td>40.95</td><td>3.16</td></tr><tr><td>u-EASE</td><td>32.34</td><td>60.79</td><td>33.68</td><td>43.41</td><td>3.37</td></tr><tr><td>u-EASE-i</td><td>32.33</td><td>60.97</td><td>33.71</td><td>43.51</td><td>3.35</td></tr></table>

Acknowledgments. Supported by the ’Siemens Energy AI Chair: Energy Sustainability for a Decarbonized Society 5.0’ (TSI-100930-2023-5), funded by the Secretary of State for Digitalization and Artificial Intelligence through the ENIA 2022 Chairs call, and co-funded by the European Union-Next Generation EU.

## References

1. Arora, S., Du, S.S., Li, Z., Salakhutdinov, R., Wang, R., Yu, D.: Harnessing the power of infinitely wide deep nets on small-data tasks (2019), https://arxiv.org/ abs/1910.01663

2. Chen, C.L.P., Liu, Z.: Broad learning system: An efective and eficient incremental learning system without the need for deep architecture. IEEE Transactions on Neural Networks and Learning Systems 29(1), 10–24 (2018). https://doi.org/ 10.1109/TNNLS.2017.2716952

3. Cho, E., Myers, S.A., Leskovec, J.: Friendship and mobility: user movement in location-based social networks. In: Proceedings of the 17th ACM SIGKDD in-

Table 5: Computation time in minutes.
<table><tr><td></td><td colspan="2">Dataset EASE u-EASE u-EASE-i</td></tr><tr><td>Gowalla</td><td>11.3</td><td>6.8 24.5</td></tr><tr><td>Yelp2018</td><td>9.9</td><td>8.0 23.7</td></tr><tr><td>ML-1M</td><td>0.1</td><td>0.3 0.4</td></tr></table>

ternational conference on Knowledge discovery and data mining. pp. 1082–1090 (2011)

4. Choi, M., Jeong, Y., Lee, J., Lee, J.: Local collaborative autoencoders. In: Proceedings of the 14th ACM international conference on web search and data mining. pp. 734–742 (2021)

5. Harper, F.M., Konstan, J.A.: The movielens datasets: History and context. ACM Trans. Interact. Intell. Syst. 5(4) (Dec 2015). https://doi.org/10.1145/2827872, https://doi.org/10.1145/2827872

6. He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., Wang, M.: Lightgcn: Simplifying and powering graph convolution network for recommendation. In: Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. p. 639–648. SIGIR ’20, Association for Computing Machinery, New York, NY, USA (2020). https://doi.org/10.1145/3397271.3401063, https://doi.org/10.1145/3397271.3401063

7. He, X., Liao, L., Zhang, H., Nie, L., Hu, X., Chua, T.S.: Neural collaborative filtering. In: Proceedings of the 26th International Conference on World Wide Web. p. 173–182. WWW ’17, International World Wide Web Conferences Steering Committee, Republic and Canton of Geneva, CHE (2017). https://doi.org/10. 1145/3038912.3052569, https://doi.org/10.1145/3038912.3052569

8. Hong, S., Choi, J., Lee, Y.C., Kumar, S., Park, N.: Svd-ae: Simple autoencoders for collaborative filtering. ArXiv abs/2405.04746 (2024), https://api. semanticscholar.org/CorpusID:269626235

9. Hu, Y., Koren, Y., Volinsky, C.: Collaborative filtering for implicit feedback datasets. 2008 Eighth IEEE International Conference on Data Mining pp. 263– 272 (2008), https://api.semanticscholar.org/CorpusID:10537313

10. Huang, L., Guan, C.R., Huang, Z.W., Gao, Y., Kuang, Y., Wang, C.D., Chen, C.L.P.: Broad recommender system: An eficient nonlinear collaborative filtering approach (2024), https://arxiv.org/abs/2204.11602

11. Jain, H., Prabhu, Y., Varma, M.: Extreme multi-label loss functions for recommendation, tagging, ranking & other missing label applications. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. p. 935–944. KDD ’16, Association for Computing Machinery, New York, NY, USA (2016). https://doi.org/10.1145/2939672.2939756, https://doi.org/10.1145/2939672.2939756

12. Jeunen, O., Van Balen, J., Goethals, B.: Closed-form models for collaborative filtering with side-information. In: Proceedings of the 14th ACM Conference on Recommender Systems. p. 651–656. RecSys ’20, Association for Computing Machinery, New York, NY, USA (2020). https://doi.org/10.1145/3383313.3418480, https://doi.org/10.1145/3383313.3418480

13. Jin, R., Li, D., Gao, J., Liu, Z., Chen, L.C., Zhou, Y.: Towards a better understanding of linear models for recommendation. Proceedings of the 27th ACM

SIGKDD Conference on Knowledge Discovery & Data Mining (2021), https: //api.semanticscholar.org/CorpusID:235212063

14. Liang, D., Krishnan, R.G., Hofman, M.D., Jebara, T.: Variational autoencoders for collaborative filtering. In: Proceedings of the 2018 World Wide Web Conference. p. 689–698. WWW ’18, International World Wide Web Conferences Steering Committee, Republic and Canton of Geneva, CHE (2018). https://doi.org/10. 1145/3178876.3186150, https://doi.org/10.1145/3178876.3186150

15. Mallea, M., Ñanculef, R., Parra, D.: Adversarial pairwise multimodal recommendation. In: 2024 International Joint Conference on Neural Networks (IJCNN). pp. 1–10 (2024). https://doi.org/10.1109/IJCNN60899.2024.10650977

16. Melchiorre, A.B., Zangerle, E., Schedl, M.: Personality bias of music recommendation algorithms. In: Proceedings of the 14th ACM Conference on Recommender Systems. p. 533–538. RecSys ’20, Association for Computing Machinery, New York, NY, USA (2020). https://doi.org/10.1145/3383313.3412223, https://doi.org/10.1145/3383313.3412223

17. Parra, D., Sahebi, S.: Recommender systems: Sources of knowledge and evaluation metrics. In: Advanced Techniques in Web Intelligence-2: Web User Browsing Behaviour and Preference Analysis, pp. 149–175. Springer (2013)

18. Penrose, R.: On best approximate solutions of linear matrix equations. Mathematical Proceedings of the Cambridge Philosophical Society 52, 17 – 19 (1956), https://api.semanticscholar.org/CorpusID:122260851

19. Rendle, S., Freudenthaler, C., Gantner, Z., Schmidt-Thieme, L.: Bpr: Bayesian personalized ranking from implicit feedback. In: Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence. p. 452–461. UAI ’09, AUAI Press, Arlington, Virginia, USA (2009)

20. Sachdeva, N., Dhaliwal, M.P., Wu, C.J., McAuley, J.: Infinite recommendation networks: a data-centric approach. In: Proceedings of the 36th International Conference on Neural Information Processing Systems. NIPS ’22, Curran Associates Inc., Red Hook, NY, USA (2024)

21. Schafer, J.B., Frankowski, D., Herlocker, J.L., Sen, S.: Collaborative filtering recommender systems. In: The Adaptive Web (2007), https://api. semanticscholar.org/CorpusID:14231524

22. Shen, Y., Wu, Y., Zhang, Y., Shan, C., Zhang, J., Letaief, B.K., Li, D.: How powerful is graph convolution for recommendation? In: Proceedings of the 30th ACM International Conference on Information & Knowledge Management. p. 1619–1629. CIKM ’21, Association for Computing Machinery, New York, NY, USA (2021). https://doi.org/10.1145/3459637.3482264, https://doi.org/10. 1145/3459637.3482264

23. Spišák, M., Bartyzal, R., Hoskovec, A., Peška, L.: On interpretability of linear autoencoders. In: Proceedings of the 18th ACM Conference on Recommender Systems. pp. 975–980 (2024)

24. Spišák, M., Bartyzal, R., Hoskovec, A., Peska, L., Tůma, M.: Scalable approximate nonsymmetric autoencoder for collaborative filtering. In: Proceedings of the 17th ACM Conference on Recommender Systems. pp. 763–770 (2023)

25. Steck, H.: Collaborative filtering via high-dimensional regression. arXiv preprint arXiv:1904.13033 (2019)

26. Steck, H.: Embarrassingly shallow autoencoders for sparse data. In: The World Wide Web Conference. p. 3251–3257. WWW ’19, Association for Computing Machinery, New York, NY, USA (2019). https://doi.org/10.1145/3308558. 3313710, https://doi.org/10.1145/3308558.3313710

27. Steck, H.: Autoencoders that don’t overfit towards the identity. Advances in Neural Information Processing Systems 33, 19598–19608 (2020)

28. Wang, X., He, X., Wang, M., Feng, F., Chua, T.S.: Neural graph collaborative filtering. In: Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. p. 165–174. SIGIR’19, Association for Computing Machinery, New York, NY, USA (2019). https://doi.org/ 10.1145/3331184.3331267, https://doi.org/10.1145/3331184.3331267

## 8 Appendix: The closed-form solution for u-EASE-i

In this section, we present the complete derivation of the closed-form solution (19) for the optimization problem (17) related to the u-EASE-i model. Firstly, we diferentiate each term:

\- First term: Using the generalized solution (9):

$$
\frac { \partial } { \partial W } \| R ^ { t } R R ^ { t } - R ^ { t } W R ^ { t } \| _ { F } ^ { 2 } = - 2 R ( R ^ { t } R R ^ { t } ) R + 2 ( R R ^ { t } ) W ( R ^ { t } R )
$$

\- Second term:

$$
\lambda _ { i } \frac { \partial } { \partial W } \| R ^ { t } W \| _ { F } ^ { 2 } = 2 \lambda _ { i } R R ^ { t } W .
$$

\- Third term:

$$
\lambda _ { u } \frac { \partial } { \partial W } \| W R ^ { t } \| _ { F } ^ { 2 } = 2 \lambda _ { u } W R ^ { t } R .
$$

\- Fourth term:

$$
\lambda _ { u } \lambda _ { i } \frac { \partial } { \partial W } \| W \| _ { F } ^ { 2 } = 2 \lambda _ { u } \lambda _ { i } W .
$$

The Gram matrices $G _ { i }$ and $G _ { u }$ are used to simplify the notation. Secondly, gathering all the derivatives and setting them equal to zero:

$$
\begin{array} { c } { { - 2 G _ { u } R G _ { i } + 2 G _ { u } W G _ { i } + 2 \lambda _ { i } G _ { u } W + 2 \lambda _ { u } W G _ { i } + 2 \lambda _ { u } \lambda _ { i } W = 0 } } \\ { { G _ { u } W G _ { i } + \lambda _ { i } G _ { u } W + \lambda _ { u } W G _ { i } + \lambda _ { u } \lambda _ { i } W = G _ { u } R G _ { i } } } \\ { { ( G _ { u } + \lambda _ { u } I ) W ( G _ { i } + \lambda _ { i } I ) = G _ { u } R G _ { i } } } \end{array}
$$

After simplifying and factorizing the terms from left and right sides, the solution is:

$$
W = ( G _ { u } + \lambda _ { u } I ) ^ { - 1 } G _ { u } R G _ { i } ( G _ { i } + \lambda _ { i } I ) ^ { - 1 } .
$$

Finally, unconstrained solutions for EASE (3) and u-EASE (12) are used to recover the final closed form for u-EASE-i presented in (19).